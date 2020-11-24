# "Eugene Morozov"<Eugene ~at~ HiEugene.com>

import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg
from ddeint import ddeint
import time
import math
from pos_matrix_approx import sqrtm, makePositiveDefinite, isPositiveDefinite

eps = np.finfo(float).eps # or 7.0/3 — 4.0/3 — 1
print("eps =", eps)
np.set_printoptions(precision=6, suppress=True, linewidth=200)
epoch_sec = int(time.time())
print("seeding rand to", epoch_sec)
np.random.seed(epoch_sec)

# Mackey-Glass chaotic series ###
derivatives = lambda x,t,tau,beta,gamma,n: beta*x(t-tau) / (1 + pow(x(t-tau), n)) - gamma*x(t)

tau   = 17
beta  = 0.2
gamma = 0.1
nn    = 10
hist = lambda t: 1.2 if abs(t) < eps else 0.0

delta_t = 0.5
t_start = 0
t_end   = 200
t_num = int((t_end - t_start) / delta_t)
tt = np.linspace(t_start, t_end, t_num)
yy = ddeint(derivatives, hist, tt, fargs=(tau, beta, gamma, nn))

# plt.figure(1)
# plt.plot(tt, yy, lw=2)
# plt.xlabel('t')
# plt.ylabel('y')
# plt.grid(True)
# plt.figure(2)
# plt.plot(yy[tau:len(yy)], yy[0:len(yy)-tau], lw=2)
# plt.xlabel('y(t)')
# plt.ylabel('y(t-tau)')
# plt.grid(True)
# plt.show()

def f(f_hist, tau, beta, gamma, n, delta_t, sqrtQ):
  if len(f_hist) < tau:
    if len(f_hist) == 0:
      x = 1.2
    else:
      x = (beta*f_hist[0] / (1 + pow(f_hist[0], n)) - gamma*f_hist[-1])*delta_t + f_hist[-1]
  else:
    x = (beta*f_hist[-tau] / (1 + pow(f_hist[-tau], n)) - gamma*f_hist[-1])*delta_t + f_hist[-1]
  x += sqrtQ.dot(np.random.normal(0, 1, size=(sqrtQ.shape[0], 1)))
  return x
#################################

def h(X, sqrtR):
  Y = X[0,:] + sqrtR.dot(np.random.normal(0, 1, size=(sqrtR.shape[0], X.shape[1])))
  return Y

plt.ion()
fig, ax = plt.subplots()
# plt.axis([-3.1, 3.1, -2.1, 2.1])
plt.grid(True)

# augmented state = (x, q, v)^T
n = 3
kappa = 3 - n
W = np.empty([2*n+1])
W[0] = kappa / (n+kappa)
for i in range(1,2*n+1):
  W[i] = 1.0 / (2*(n+kappa))

q = 0.02 # process noise
r = 0.02 # observation noise or measurement error
Q = q*np.eye(1,1)
R = r*np.eye(1,1)
sqrtQ = sqrtm(Q)
sqrtR = sqrtm(R)

def sigma_points(x, P, n, kappa):
  X = np.empty([n, 2*n+1])
  L = sqrtm((n+kappa)*P)
  for i in range(2*n+1):
    if i == 0:
      X[:,[i]] = x[:]
    elif 0 < i <= n:
      X[:,[i]] = x[:] + L[:,[i-1]]
    else:
      X[:,[i]] = x[:] - L[:,[i-1-n]]
  return X

def predict(X, sqrtQ):
  X_pred = np.empty([n, 2*n+1])
  for i in range(n):
    for j in range(2*n+1):
      predict.f_hist[i][j].append(X[i,j])
      X_pred[i,j] = f(predict.f_hist[i][j], tau, beta, gamma, n, delta_t, sqrtQ)
  return X_pred
predict.f_hist = [[[] for col in range(2*n+1)] for row in range(n)]

def predicted_mean(X):
  x_mean = np.zeros([n, 1])
  for i in range(2*n+1):
    x_mean += W[i]*X[:,[i]]
  return x_mean

def predicted_cov(X, x_mean):
  P = np.zeros([n, n])
  for i in range(2*n+1):
    delta = X[:,[i]] - x_mean
    P += W[i]*delta*delta.T
  P_ = makePositiveDefinite(P, semi=True, v=0)
  return P_

def predicted_observation(X, sqrtR):
  Y = h(X, sqrtR)
  return Y

def predicted_obs_mean(Y):
  y_mean = np.zeros([Y.shape[0], 1])
  for i in range(2*n+1):
    y_mean += W[i]*Y[:,[i]]
  return y_mean

def update_var(Y_pri, y_mean_pri):
  P_post = np.zeros([Y_pri.shape[0], Y_pri.shape[0]])
  for i in range(2*n+1):
    delta = Y_pri[:,[i]] - y_mean_pri
    P_post += W[i]*delta*delta.T
  return P_post

def update_cov(X_pri, x_mean_pri, Y_pri, y_mean_pri):
  P_post = np.zeros([X_pri.shape[0], Y_pri.shape[0]])
  for i in range(2*n+1):
    deltaX = X_pri[:,[i]] - x_mean_pri
    deltaY = Y_pri[:,[i]] - y_mean_pri
    P_post += W[i]*deltaX*deltaY.T
  return P_post

def kalman_gain(P_post_yy, P_post_xy):
  G = P_post_xy * scipy.linalg.pinv(P_post_yy)
  print("G =\n", G)
  return G

def observe(x, sqrtR):
  y = x + sqrtR.dot(np.random.normal(0, 1, size=(sqrtR.shape[0], 1)))
  return y

def update_mean(x_mean_pri, G, y, y_mean_pri):
  x_mean_post = x_mean_pri + G.dot(y - y_mean_pri)
  return x_mean_post

def update_post_cov(P_pri, G, P_post_yy):
  P_post = P_pri - G.dot(P_post_yy).dot(G.T)
  P_post_ = makePositiveDefinite(P_post, semi=False, v=0)
  return P_post_

# initial conditions
t = tt[0]
x = yy[0]
x_pri = np.array([[0, 0, 0]]).T
x_post = x_pri
x_mean_post = x_post
P_post = np.zeros((n, n))
P_post[0,0] = max(q, r) * 100
P_post[1,1] = q
P_post[2,2] = r
y = observe(x, sqrtR)

px, = ax.plot(t, x, 'rx', ms=4.0, label='x')
px_pri, = ax.plot(t, x_pri[0], 'go')
px_pri_cir = plt.Circle((t, x_pri[0]), math.sqrt(q), color='g', alpha=0.1); ax.add_artist(px_pri_cir)
px_post, = ax.plot(t, x_post[0], 'bo')
px_post_cir = plt.Circle((t, x_post[0]), math.sqrt(P_post[0,0]), color='b', alpha=0.1); ax.add_artist(px_post_cir)
py, = ax.plot(t, y, 'ro', ms=4.0, label='y')
py_cir = plt.Circle((t, y), math.sqrt(r), color='r', alpha=0.1); ax.add_artist(py_cir)
px_prev = px
px_pri_prev = px_pri
px_pri_cir_prev = px_pri_cir
px_post_prev = px_post
px_post_cir_prev = px_post_cir
fig.canvas.flush_events(); time.sleep(0.01)
t_prev = tt[0]
x_pri_prev = x_pri[0]
y_prev = y[0,0]
x_post_prev = x_post[0]
py_cir_prev = py_cir
X_prev = np.tile(x_mean_post, 2*n+1)

for k in range(1,len(tt)):
  t = tt[k]
  x = yy[k]
  print("t=%f, x=%f" % (t, x))

  X = sigma_points(x_mean_post, P_post, n, kappa)
  X_pri = predict(X, sqrtQ)
  x_mean_pri = predicted_mean(X_pri)
  P_pri = predicted_cov(X_pri, x_mean_pri)
  Y_pri = predicted_observation(X_pri, sqrtR)
  y_mean_pri = predicted_obs_mean(Y_pri)

  P_post_yy = update_var(Y_pri, y_mean_pri)
  P_post_xy = update_cov(X_pri, x_mean_pri, Y_pri, y_mean_pri)
  G = kalman_gain(P_post_yy, P_post_xy)
  y = observe(x, sqrtR)
  x_mean_post = update_mean(x_mean_pri, G, y, y_mean_pri)
  P_post = update_post_cov(P_pri, G, P_post_yy)

  px, = ax.plot(t, x, 'rx', ms=7.0, label='x')
  pX = [ax.plot(t, X[0,i], '.', color=(0.5, 0.5, 0.5), ms=4.0) for i in range(X.shape[1])]
  pX_lines = [ax.plot([t_prev, t], [X_prev[0,i], X[0,i]], lw=1.0, color=(0.5, 0.5, 0.5)) for i in range(X.shape[1])]

  px_pri, = ax.plot(t, x_mean_pri[0], 'go', ms=4.0, label='x_pri')
  ax.plot([t_prev, t], [x_pri_prev, x_mean_pri[0]], lw=0.3, color='g')
  px_pri_cir = plt.Circle((t, x_mean_pri[0]), math.sqrt(P_pri[0,0]), color='g', alpha=0.1); ax.add_artist(px_pri_cir)

  py, = ax.plot(t, y, 'ro', ms=4.0, label='y')
  ax.plot([t_prev, t], [y_prev, y[0,0]], lw=0.3, color='r')
  py_cir = plt.Circle((t, y), math.sqrt(r), color='r', alpha=0.1); ax.add_artist(py_cir)

  px_post, = ax.plot(t, x_mean_post[0], 'bo', ms=4.0, label='x_post')
  ax.plot([t_prev, t], [x_post_prev, x_mean_post[0]], lw=0.3, color='b')
  px_post_cir = plt.Circle((t, x_mean_post[0]), math.sqrt(P_post[0,0]), color='b', alpha=0.1); ax.add_artist(px_post_cir)

  plt.legend(handles=[px, px_pri, py, px_post], loc="best", ncol=1, numpoints=1)
  if k%20 == 0:
    fig.canvas.flush_events(); time.sleep(0.01)
  px_pri_cir_prev.remove()
  py_cir_prev.remove()
  px_post_cir_prev.remove()
  [pX_lines[i].pop(0).remove() for i in range(len(pX_lines))]

  t_prev = t
  y_prev = y[0,0]
  x_pri_prev = x_mean_pri[0]
  x_post_prev = x_mean_post[0]
  X_prev = X
  px_pri_cir_prev = px_pri_cir
  py_cir_prev = py_cir
  px_post_cir_prev = px_post_cir

plt.ioff()
plt.show()
print("end.")
