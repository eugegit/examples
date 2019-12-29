#!/usr/bin/python3

# plot optimal bids/asks for a price realization
# "Eugene Morozov"<Eugene ~at~ HiEugene.com>

from util import *
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

v = 1
start_time = preamble()

epoch_sec = 1 # int(time.time())
print("seeding rand to ", epoch_sec)
np.random.seed(epoch_sec)

T = 1 # [day]
dt = 0.005 # 0.005 ~ 2 min
sigma = 0.05 # daily volatility of 5%
s0 = 100.0
k = 40
gamma = 0.1

n = int(T / dt)
A = 1.0 / dt

# mid-price is updated by a random increment +-sigma*sqrt(dt)
# dx = sigma*dW
# delta_x = sigma*sqrt(delta_t)*N(0,1) = N(0,t*sigma**2)
s = sigma*np.sqrt(dt)*np.random.normal(size=n)
s[0] += s0
s = np.add.accumulate(s)
if v:
  plt.ion()
  fig = plt.figure()
  ax = fig.gca()
  ax.plot(s, c='b', label="mid")
  plt.grid(True)
  plt.title("optimal market-maker quotes")
  plt.tight_layout()
  plt.autoscale(enable=True, axis='x', tight=True)
  ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
  ax.legend(loc="best", ncol=1, scatterpoints=1, numpoints=1)
  plt.show()

  figQ = plt.figure()
  axQ = figQ.gca()
  axQ.set_title("inventory")
  plt.grid(True)
  plt.show()

  figP = plt.figure()
  axP = figP.gca()
  axP.set_title("PnL")
  plt.grid(True)
  plt.show()

# delta_a = np.linspace(0, 0.20, int(0.20 / 0.001))
# prob = A*np.exp(-k*delta_a)*dt
# fig = plt.figure()
# ax = fig.gca()
# ax.plot(delta_a, prob)
# plt.grid(True)
# plt.show()

delta_a = 0.01
delta_b = 0.01
Q = 0 # inventory
X = 1000.0 # cash
q = 10
sigma2 = sigma**2
Q_prev = Q
PNL_prev = X + Q*s0
for t in range(n):
  # with probability lambda_a*dt, the inventory decreases by 1 and the wealth increases by s+delta_a
  lambda_a = A*np.exp(-k*delta_a)
  lambda_b = A*np.exp(-k*delta_b)
  prob_a = lambda_a*dt
  prob_b = lambda_b*dt
  assert(prob_a <= 1.0 and prob_b <= 1.0)
  if np.random.random() < prob_a:
    Q -= q
    X += (s[t] + delta_a)*q
  if np.random.random() < prob_b:
    Q += q
    X -= (s[t] - delta_b)*q

  # calc new delta_a, delta_b
  delta_a = np.log(1 + gamma/k) / gamma + 0.5*(1 - 2*Q)*gamma*sigma2*(n-t)*dt
  delta_b = np.log(1 + gamma/k) / gamma + 0.5*(1 + 2*Q)*gamma*sigma2*(n-t)*dt
  check = True
  if delta_a < 0:
    delta_a = 0
    check = False
  if delta_b < 0:
    delta_b = 0
    check = False
  Delta = delta_a + delta_b
  if check: assert(abs(Delta - (2/gamma*np.log(1 + gamma/k) + gamma*sigma2*(n-t)*dt)) < eps)
  r = s[t] + (delta_a - delta_b) / 2.0
  print("[%3d/%d] s-r = %.5f" % (t, n, s[t]-r))
  if check: assert(abs(r - (s[t] - Q*gamma*sigma2*(n-t)*dt)) < eps)
  PNL = X + Q*s[t]

  # plot a,b,r; Q; PNL
  if v:
    if t == 0:
      ax.plot([t, t], [r, r], c='c', label=r"$r^*$")
      ax.plot([t, t], [r+delta_a, r+delta_a], c='g', label=r"$\delta_a$")
      ax.plot([t, t], [r-delta_b, r-delta_b], c='r', label=r"$\delta_b$")
      ax.legend(loc="best", ncol=1, scatterpoints=1, numpoints=1)
    else:
      ax.plot([t-1, t], [r_prev, r], c='c', label=r"$r^*$")
      ax.plot([t-1, t], [r_prev+delta_a_prev, r+delta_a], c='g', label=r"$\delta_a$")
      ax.plot([t-1, t], [r_prev-delta_b_prev, r-delta_b], c='r', label=r"$\delta_b$")
    axQ.plot([t-1, t], [Q_prev, Q], c='b')
    axP.plot([t-1, t], [PNL_prev, PNL], c='b')
  r_prev = r
  delta_a_prev = delta_a
  delta_b_prev = delta_b
  Q_prev = Q
  PNL_prev = PNL

postscript(start_time)
