# "Eugene Morozov"<Eugene ~at~ HiEugene.com>

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from util import get_color
import time

def plotX(X, mu, M, N, K, r, ax):
  for i in range(M):
    if plotX.px[i]: plotX.px[i].remove()
    if N == 2:
      plotX.px[i] = ax.scatter(X[i,0], X[i,1], color=get_color(r[i]), marker="o")
    else:
      plotX.px[i] = ax.scatter(X[i,0], X[i,1], X[i,2], color=get_color(r[i]), marker="o")
  for i in range(K):
    if plotX.pmu[i]: plotX.pmu[i].remove()
    if N == 2:
      plotX.pmu[i] = ax.scatter(mu[i,0], mu[i,1], color=get_color(i+K), marker="*", linewidths=5)
    else:
      plotX.pmu[i] = ax.scatter(mu[i,0], mu[i,1], mu[i,2], color=get_color(i+K), marker="*", linewidths=5)

def calc_dist(x, y):
  d = ((x - y)**2).sum() # sqrt omitted
  return d

def K_means(X, K, v, title_str):
  M, N = X.shape # M samples of dimension N
  r = np.zeros(M, dtype="int") # cluster id for each point

  mu_new = np.empty([K,N])
  for i in range(K):
    mu_new[i,:] = X[i*M//K,:]
  mu = np.zeros([K,N])

  if v and (N==2 or N==3):
    plotX.px = [None]*M
    plotX.pmu = [None]*K
    fig = plt.figure()
    if N == 2:
      ax = fig.gca()
    else:
      ax = fig.gca(projection='3d')
    plt.ion()
    plt.grid(True)
    plt.show()

  # while np.abs(mu_new - mu).sum() > 1e-10: # or track when the assignments no longer change
  while calc_dist(mu_new, mu) > 1e-10: # or track when the assignments no longer change
    if v and (N==2 or N==3):
      plotX(X, mu_new, M, N, K, r, ax)
      # ax.set_title(f"diff = {np.abs(mu_new - mu).sum()}")
      ax.set_title(f"diff = {calc_dist(mu_new, mu)}")
      fig.canvas.flush_events(); time.sleep(0.5)
    mu = mu_new.copy()
    for i in range(M):
      min_dist = 1e10
      r[i] = 0
      for k in range(K):
        dist = calc_dist(X[i,:], mu[k,:])
        if dist < min_dist:
          min_dist = dist
          r[i] = k

    for k in range(K):
      mu_new[k] = X[r==k,:].sum(axis=0) / X[r==k,:].shape[0]
  if v and (N==2 or N==3):
    ax.set_title(title_str)
    fig.canvas.flush_events(); time.sleep(0.5)
  return r
