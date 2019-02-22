#!/usr/bin/python3

# Unscented Particle Filter for Bond Mid and DTM estimation
# "Eugene Morozov"<Eugene ~at~ HiEugene.com>

from enum import Enum
from numbers import Number
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from util import *
from pos_matrix_approx import *
from probabilities import *
import upf_bonds_ukf
from upf_bonds_ukf import UKF
import util

v = 0
start_time = preamble()

epoch_sec = 1 # int(time.time())
print("seeding rand to ", epoch_sec)
np.random.seed(epoch_sec)

d = 3 # number of bonds
# Sigma = np.matrix([[ 9.167089,  9.092542, -3.450822],
#                    [ 9.092542,  9.974566, -2.721507],
#                    [-3.450822, -2.721507,  4.519941]])
Sigma =np.matrix([[  4.689318,   5.14435 ,  -2.589705],
                  [  5.14435 ,  12.219445,  -2.516943],
                  [ -2.589705,  -2.516943,   2.501255]])
assert(isPositiveDefinite(Sigma, v=1))
std = np.sqrt(np.diag(Sigma))
Corr = Sigma / np.outer(std, std) # element-wise division
# A = np.array([36.04, 1.58, 1.00]) # diagonal of A
A = np.array([1.11, 1.66, 1.63])
# psi0 = np.array([0.30, 0.60, 0.44])
psi0 = np.array([0.51, 0.54, 0.41])
# V = np.array([15.52, 1.81, 1.42]) # diagonal of V
V = np.array([1.25, 1.80, 2.01])
dt = 1
VM = np.eye(3,3)*V
G = VM.dot(VM.T) # dtm variance
for i in range(G.shape[0]):
  for j in range(G.shape[0]):
    G[i,j] = G[i,j] / (A[i]+A[j]) * (1 - np.exp(-dt*(A[i]+A[j])))

df = pd.read_csv("bonds.csv", header=None, names=["x1","x2","x3","x4","x5","x6","x7","x8"])
t_start = 300
t_end   = 500
t_num = t_end - t_start
a_x = np.zeros([t_num, d]) # if 0 then unobserved
a_x[:,0] = df["x1"].values[t_start:t_end]
a_x[:,1] = df["x2"].values[t_start:t_end]
a_x[:,2] = df["x3"].values[t_start:t_end]
a_y = np.zeros([t_num, d]) # if 0 then unobserved
a_y[:,0] = df["x4"].values[t_start:t_end]
a_y[:,1] = df["x5"].values[t_start:t_end]
a_y[:,2] = df["x6"].values[t_start:t_end]
trade_type = np.zeros([t_num, 2], dtype=int) # (bond index, trade type)
trade_type[:,0] = df["x7"].values[t_start:t_end]
trade_type[:,1] = df["x8"].values[t_start:t_end]
fig = 2*[None]
ax = 2*[None]
if v >= 1:
  plt.ion()
  plt.rcParams['axes.grid'] = True
  for i in range(len(fig)):
    fig[i], ax[i] = plt.subplots(d,1)
    fig[i].tight_layout()
    for j in range(d):
      ax[i][j].autoscale(enable=True, axis='both', tight=True)
  colors = ['b', 'g', 'r', 'c', 'm', 'y', (0.5, 0.5, 0.5), (0.25, 0.25, 0.25), (0.75, 0.75, 0.75)]
  ax[0][0].text(0.96, 0.6, "1 - cl.buy\n2 - cl.sell\n3 - t.a.buy\n4 - t.a.sell\n5 - d2d", transform=ax[0][0].transAxes)
  win_pos_x = 0 # 1680
  win_pos_y = 0 # 330
  win_width = 1900 # 1200
  win_height = 1000 # 700
  for i in range(len(fig)): move_plot(fig[i], win_pos_x, win_pos_y, win_width, win_height)
  plt.show()

  a_y_prev = np.zeros([d,2])
  for k in range(t_num):
    if k > 0:
      for i in range(d):
        ax[0][i].scatter(k, a_x[k, i], c='r', marker='x', s=40, linewidth=1)
        ax[0][i].plot([k-1, k], [a_x[k-1, i], a_x[k, i]], '--', c='r')

    idx = trade_type[k,0]
    if a_y_prev[idx,1] == 0:
      a_y_prev[idx,0] = k; a_y_prev[idx,1] = a_y[k,idx]
    ax[0][idx].scatter(k, a_y[k,idx], c='r', marker='$%s$' % trade_type[k,1], s=80, linewidth=1)
    ax[0][idx].plot([a_y_prev[idx,0], k], [a_y_prev[idx,1], a_y[k,idx]], c='r')

    a_y_prev[idx,0] = k; a_y_prev[idx,1] = a_y[k,idx]
    if k%20 == 0:
      for i in range(len(fig)): fig[i].canvas.flush_events(); time.sleep(0.01)

def f(x, sqrtQ):
  assert(x.shape==(n,1) or len(x.shape)==1 and x.shape[0]==n)
  rnd = sqrtQ.dot(np.random.normal(0, 1, size=(sqrtQ.shape[0],1)))
  x_new = np.copy(x)
  x_new[0] = x[0] + rnd[0]
  x_new[1] = x[1] + rnd[1]
  x_new[2] = x[2] + rnd[2]
  x_new[3] = next_z(x[3], A[0], V[0], rnd[3])
  x_new[4] = next_z(x[4], A[1], V[1], rnd[4])
  x_new[5] = next_z(x[5], A[2], V[2], rnd[5])
  x_new[3] = max(min(x_new[3], np.log(10.0/psi0[0])), np.log(0.01/psi0[0]))
  x_new[4] = max(min(x_new[4], np.log(10.0/psi0[1])), np.log(0.01/psi0[1]))
  x_new[5] = max(min(x_new[5], np.log(10.0/psi0[2])), np.log(0.01/psi0[2]))
  return x_new

def h(X, sqrtR, trade_type):
  assert(X.shape == (n, 2*n+1) or X.shape == (n, N))
  Y = np.zeros([m, X.shape[1]])
  for j in range(X.shape[1]):
    mid1 = X[0,j]
    mid2 = X[1,j]
    mid3 = X[2,j]
    psi1 = dtm(X[3,j], psi0[0])
    psi2 = dtm(X[4,j], psi0[1])
    psi3 = dtm(X[5,j], psi0[2])
    if trade_type[1] == TradeType.CLIENT_BUY.value:
      y1 = mid1 + psi1 + sqrtR[0,0]*np.random.normal(0, 1, size=(1,1))
      y2 = mid2 + psi2 + sqrtR[1,1]*np.random.normal(0, 1, size=(1,1))
      y3 = mid3 + psi3 + sqrtR[2,2]*np.random.normal(0, 1, size=(1,1))
    elif trade_type[1] == TradeType.CLIENT_SELL.value:
      y1 = mid1 - psi1 + sqrtR[0,0]*np.random.normal(0, 1, size=(1,1))
      y2 = mid2 - psi2 + sqrtR[1,1]*np.random.normal(0, 1, size=(1,1))
      y3 = mid3 - psi3 + sqrtR[2,2]*np.random.normal(0, 1, size=(1,1))
    elif trade_type[1] == TradeType.TRADED_AWAY_BUY.value:
      y1 = mid1 + psi1 + sqrtR[0,0]*scipy.stats.truncnorm(a=psi1, b=np.inf).rvs(1)
      y2 = mid2 + psi2 + sqrtR[1,1]*scipy.stats.truncnorm(a=psi2, b=np.inf).rvs(1)
      y3 = mid3 + psi3 + sqrtR[2,2]*scipy.stats.truncnorm(a=psi3, b=np.inf).rvs(1)
    elif trade_type[1] == TradeType.TRADED_AWAY_SELL.value:
      y1 = mid1 - psi1 + sqrtR[0,0]*scipy.stats.truncnorm(a=-np.inf, b=-psi1).rvs(1)
      y2 = mid2 - psi2 + sqrtR[1,1]*scipy.stats.truncnorm(a=-np.inf, b=-psi2).rvs(1)
      y3 = mid3 - psi3 + sqrtR[2,2]*scipy.stats.truncnorm(a=-np.inf, b=-psi3).rvs(1)
    else: # TradeType.D2D
      y1 = mid1 + sqrtR[0,0]*scipy.stats.truncnorm(a=-psi1, b=psi1).rvs(1)
      y2 = mid2 + sqrtR[1,1]*scipy.stats.truncnorm(a=-psi2, b=psi2).rvs(1)
      y3 = mid3 + sqrtR[2,2]*scipy.stats.truncnorm(a=-psi3, b=psi3).rvs(1)
    Y[0,j] = y1
    Y[1,j] = y2
    Y[2,j] = y3
  return Y

def next_z(z, a, vv, r):
  assert(type(z)==float or type(z)==np.float64)
  assert(type(a)==float or type(a)==np.float64)
  assert(type(vv)==float or type(vv)==np.float64)
  dt = 1
  # r = vv * np.sqrt((1-np.exp(-2*a*dt)) / (2*a)) * np.random.normal(0, 1, size=(1,1))
  z_new = np.exp(-a*dt)*z + r
  return z_new

def dtm(z, psi0):
  assert(type(z)==float or type(z)==np.float64)
  assert(type(psi0)==float or type(psi0)==np.float64)
  psi = psi0 * np.exp(z)
  psi = max(min(psi, 10.0), 0.01)
  return psi

def get_Xs(X):
  assert(X.shape==(N,n) or X.shape==(2*n+1,n))
  x = np.zeros([X.shape[0],2*d])
  for i in range(X.shape[0]):
    x[i,:] = get_xs(X[i,:])
  return x

def get_xs(X):
  assert(len(X.shape)==1 and X.shape[0]==n)
  x = np.zeros(2*d)
  psi1 = dtm(X[3], psi0[0])
  psi2 = dtm(X[4], psi0[1])
  psi3 = dtm(X[5], psi0[2])
  x[0] = X[0] + psi1
  x[1] = X[1] + psi2
  x[2] = X[2] + psi3
  x[3] = X[0] - psi1
  x[4] = X[1] - psi2
  x[5] = X[2] - psi3
  return x

N = 2000 # particles
n =   18 # augmented state dimension
m =    3 # observation dimension
upf_bonds_ukf.init(v, N, n, m, t_num, f, h, d, get_Xs, get_xs, psi0)

q = 0.1 # process noise
r = 0.1 # observation noise or measurement error
Q = scipy.linalg.block_diag(Sigma, G)
R = r*np.eye(m,m)
sqrtQ = sqrtm(Q)
sqrtR = sqrtm(R)

# initial conditions
x01 = x02 = x03 = 100 # arbitrary
z01 = z02 = z03 = 0.5 # arbitrary
a_x_pri  = np.zeros([n,t_num])
a_x_post = np.zeros([n,t_num])
x_mean_post1 = np.empty([N, n])
P_post1 = np.empty([N, n, n])
for i in range(N):
  rnd = sqrtQ.dot(np.random.normal(0, 1, size=(sqrtQ.shape[0],1)))
  x_mean_post1[i,:] = np.array([[x01+rnd[0], x02+rnd[1], x03+rnd[2], z01+rnd[3], z02+rnd[4], z03+rnd[5],
                                 0,0,0, 0,0,0, 0,0,0, 0,0,0]])
  P_post1[i,:] = scipy.linalg.block_diag(Q, q*np.eye(6,6), R, R)
ukf1 = UKF(x_mean_post1, fig[0], ax[0])
ukf2 = UKF(x_mean_post1, fig[1], ax[1])
x_mean_post_prev1 = np.copy(x_mean_post1)
P_post_prev1 = np.copy(P_post1)
a_x_pri[:,0] = col_mean(x_mean_post1).flatten()
a_x_post[:,0] = col_mean(x_mean_post1).flatten()

if v >= 1:
  ba_pri = get_xs(a_x_pri[:,0])
  ba_post = get_xs(a_x_post[:,0])
  for i in range(d):
    ax[0][i].scatter(0, ba_pri[i], c='g', marker='v', linewidths=1, label="x_pri bid" if i==1 else "")
    ax[0][i].scatter(0, ba_pri[i+d], c='g', marker='^', linewidths=1, label="x_pri ask" if i==1 else "")
    ax[0][i].scatter(0, ba_post[i], c='b', marker='v', linewidths=1, label="x_post bid" if i==1 else "")
    ax[0][i].scatter(0, ba_post[i+d], c='b', marker='^', linewidths=1, label="x_post ask" if i==1 else "")
  ax[0][1].legend(loc="upper right", scatterpoints=1, numpoints=1)

w = 1.0 / N * np.ones(N) # initial weights

for k in range(1, t_num):
  x_mean_post1, P_post1 = ukf1.ukf(a_y[k,:], trade_type[k,:], x_mean_post_prev1, P_post_prev1, sqrtR, sqrtQ, Corr, k)
  xhat1 = sample_normal(x_mean_post1, P_post1, N, n)
  likelihood  = calc_likelihood_upf_bonds(a_y[k, trade_type[k,0]], trade_type[k,:], xhat1, h, R, sqrtR, N, n, m)
  trans_prior = calc_trans_prior(xhat1, x_mean_post_prev1, N, n, m)
  proposal    = calc_proposal(xhat1, x_mean_post1, P_post1, N, n)
  w = w * likelihood * trans_prior / proposal
  w = w / np.sum(w)
  if np.isnan(w).any():
    print("warn: resetting w")
    w = 1.0 / N * np.ones(N)
  if abs(np.sum(w) - 1) > 1e-5:
    print("k=%d, sum != 1 !!! sum(w)=%.3f" % (k, np.sum(w)))

  if sum(w > 0.8) == 1:
    print("k=%d, sample degeneracy: max(w)=%.2f" % (k, max(w)))
  ess = 1.0 / np.sum(w**2)
  if ess < N/2:
    print("k=%d, ess=%.3f < %d, resampling" % (k, ess, N/2))
    idx2              = residual_resample(np.arange(N), w, N)
    x_mean_post_prev2 = np.copy(x_mean_post_prev1[idx2,:])
    P_post_prev2      = np.copy(P_post_prev1[idx2,:])
    xhat2             = np.copy(xhat1[idx2,:])
    P_post2           = np.copy(P_post1[idx2,:])
    w = 1.0 / N * np.ones(N)
  else:
    print("k=%d, ess=%.3f >= %d, not resampling" % (k, ess, N/2))
    x_mean_post_prev2 = np.copy(x_mean_post_prev1)
    P_post_prev2      = np.copy(P_post_prev1)
    xhat2             = np.copy(xhat1)
    P_post2           = np.copy(P_post1)

  x_mean_post2, P_tag2 = ukf2.ukf(a_y[k,:], trade_type[k,:], x_mean_post_prev2, P_post_prev2, sqrtR, sqrtQ, Corr, k)
  xtag2        = sample_normal(x_mean_post2, P_tag2, N, n)
  lik_tag      = calc_likelihood_upf_bonds(a_y[k, trade_type[k,0]], trade_type[k,:], xtag2, h, R, sqrtR, N, n, m)
  prior_tag    = calc_trans_prior(xtag2, x_mean_post_prev2, N, n, m)
  proposal_tag = calc_proposal(xtag2, x_mean_post2, P_tag2, N, n)

  lik2      = calc_likelihood_upf_bonds(a_y[k, trade_type[k,0]], trade_type[k,:], xhat2, h, R, sqrtR, N, n, m)
  prior2    = calc_trans_prior(xhat2, x_mean_post_prev2, N, n, m)
  proposal2 = calc_proposal(xhat2, x_mean_post2, P_post2, N, n)

  ratio = (lik_tag * prior_tag * proposal2) / (lik2 * prior2 * proposal_tag)
  if np.isnan(ratio).any():
    print("warn: resetting ratio to 0")
    ratio = np.zeros(N)
  accepted = 0
  rejected = 0
  for i in range(N):
    acceptance = min(1, ratio[i])
    u = np.random.uniform()
    if u <= acceptance:
      xhat2[i,:]   = xtag2[i,:]
      P_post2[i,:] = P_tag2[i,:]
      accepted += 1
    else:
      xhat2[i,:]   = xhat2[i,:]
      P_post2[i,:] = P_post2[i,:]
      rejected += 1
  print("accepted=%d (%.0f%%), rejected=%d" % (accepted, accepted*100.0/N, rejected))

  a_x_post[:,k] = np.sum(np.tile(w, (n,1)).T * xhat2, axis=0)

  a_x_pri[:,k] = col_mean(ukf2.x_mean_pri)

  x_mean_post_prev1 = np.copy(x_mean_post1)
  P_post_prev1      = np.copy(P_post1)

  if v >= 1:
    if v >= 2:
      ukf1.clean_iter_plot()
      ukf2.clean_iter_plot()
    # if k > 3 and k % 5 == 0:
    #   for jj in range(d):
    #     ax[0][jj].cla()
    #     margin = 40.0
    #     ax[0][jj].set_xlim(k-1, k+5)
    #     ax[0][jj].set_ylim(a_x[k,jj]-margin, a_x[k,jj]+margin)
    ba = get_xs(a_x_pri[:,k])
    ba_prev = get_xs(a_x_pri[:,k-1])
    for jj in range(d):
      ax[0][jj].scatter(k, ba[jj], c='g', marker='v', linewidths=1, label="x_pri bid")
      ax[0][jj].scatter(k, ba[jj+3], c='g', marker='^', linewidths=1, label="x_pri ask")
      ax[0][jj].plot([k-1, k], [ba_prev[jj], ba[jj]], c='g')
      ax[0][jj].plot([k-1, k], [ba_prev[jj+3], ba[jj+3]], c='g')

    ba = get_xs(a_x_post[:,k])
    ba_prev = get_xs(a_x_post[:,k-1])
    for jj in range(d):
      ax[0][jj].scatter(k, ba[jj], c='b', marker='v', linewidths=1, label="x_post bid")
      ax[0][jj].scatter(k, ba[jj+3], c='b', marker='^', linewidths=1, label="x_post ask")
      ax[0][jj].plot([k-1, k], [ba_prev[jj], ba[jj]], c='b')
      ax[0][jj].plot([k-1, k], [ba_prev[jj+3], ba[jj+3]], c='b')
    fig[0].canvas.flush_events(); fig[1].canvas.flush_events(); time.sleep(0.01)

# stats
x_x_post = a_x[10:t_num,0] - a_x_post[0,10:t_num]
post_err = np.sqrt(np.einsum('i,i->', x_x_post, x_x_post))
print("q=%f, r=%f, post_err=%.1f" % (q, r, post_err))

file = open(__file__+".a_x_post.csv", 'w')
[file.write("%s\n" % (','.join(str(x) for x in a_x_post[:,j]))) for j in range(t_num)]
file.close()

postscript(start_time)
