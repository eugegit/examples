# generate Gaussian mixture
# "Eugene Morozov"<Eugene ~at~ HiEugene.com>

from util import *
from probabilities import *
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from iris import get_iris, get_iris_full

v = 1
epoch_sec = 1 # int(time.time())
start_time = preamble(epoch_sec)

def gen_gaussian_mixture():
  trueMu = np.array([[2.25, 2.25],
                     [3.0 , 3.0 ],
                     [4.0 , 2.0 ]])
  trueSigma = np.array([[[ 0.1  ,  0.075],
                         [ 0.075,  0.1  ]],
                        [[ 0.1  , -0.06 ],
                         [-0.06 ,  0.1  ]],
                        [[ 0.1  ,  0.0  ],
                         [ 0.0  ,  0.1  ]]])
  tK = trueMu.shape[0]
  N_points = np.array([100, 100, 100])
  N_ranges = np.zeros([tK+1], dtype=int)
  for k in range(tK):
    N_ranges[k+1] = N_ranges[k] + N_points[k]
  X = np.empty(shape=(0, 0))
  for k in range(tK):
    print(f"mu{k} = {trueMu[k,:]}; Sigma{k} =\n", trueSigma[k,:])
    if not isPositiveDefinite(trueSigma[k,:], v=1):
      raise Exception("couldn't get a positive definite correlation matrix")
    Xk = np.random.multivariate_normal(trueMu[k,:], trueSigma[k,:], size=N_points[k])
    if X.size == 0:
      X = Xk
    else:
      X = np.concatenate((X, Xk))
  return X, tK, trueMu, N_ranges

X, tK, trueMu, N_ranges = gen_gaussian_mixture()
# X, tK, trueMu, N_ranges = get_iris()
# X, tK, trueMu, N_ranges = get_iris_full()
N = X.shape[0]
D = X.shape[1]
print("trueMu =\n", trueMu)

def log_likelihood(pi, X, mu, Sigma):
  K = pi.shape[0]
  N = X.shape[0]
  assert(X.shape[1]==mu.shape[1] and mu.shape[0]==K and mu.shape[0]==Sigma.shape[0] and mu.shape[1]==Sigma.shape[1])
  ll = 0.0
  for n in range(N):
    lik = 0.0
    for k in range(K):
      lik += pi[k] * normal_prob(X[n].reshape(D,1), mu[k].reshape(D,1), Sigma[k])
    ll += np.log(lik)
  assert(ll.shape == (1,1))
  return ll[0,0]

def find_closest(mu, trueMu):
  min_dist = float("inf")
  min_index = 0
  for k in range(trueMu.shape[0]):
    dist2 = (trueMu[k,0] - mu[0]) * (trueMu[k,0] - mu[0]) + (trueMu[k,1] - mu[1]) * (trueMu[k,1] - mu[1])
    if dist2 < min_dist:
      min_dist = dist2
      min_index = k
  return min_index

def plot_fig2(fig2, ax2, mu, trueMu, gamma, X):
  ax2.cla()
  fig2.canvas.flush_events()

  tColors = np.zeros([K + tK, 3])
  for k in range(tColors.shape[0]):
    tColors[k, :] = get_color(k)
  next_color_index = tK
  already_mapped = np.zeros([tK], dtype=int)
  # map colors for mu to closest from trueMu
  colors = np.zeros([K, 3])
  for k in range(K):
    index = find_closest(mu[k], trueMu)
    if not already_mapped[index]:
      colors[k] = tColors[index]
      already_mapped[index] = 1
    else: # new color (not seen on fig.1)
      colors[k] = tColors[next_color_index]
      next_color_index += 1

  for n in range(N):
    c = 0
    for k in range(K):
      c += gamma[n, k] * colors[k]
    ax2.scatter(X[n, 0], X[n, 1], color=c, marker="o")
  plt.grid(True)
  ax2.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
  fig2.canvas.flush_events(); # time.sleep(0.1)

if v:
  fig = plt.figure()
  ax = fig.gca()
  plt.ion()
  for k in range(tK):
    ax.scatter(X[N_ranges[k]:N_ranges[k+1], 0], X[N_ranges[k]:N_ranges[k+1], 1], color=get_color(k), marker="o", label=f"source $X_{k}$")
  plt.grid(True)
  ax.legend(loc="best", ncol=1, scatterpoints=1, numpoints=1)
  ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
  plt.show()

  fig2 = plt.figure()
  ax2 = fig2.gca()
  plt.show()

K = 3
mu = np.random.rand(K, D)
for i in range(D):
  mu[:,i] = mu[:,i] * (max(X[:,i]) - min(X[:,i])) + min(X[:,i])
Sigma = np.zeros([K, D, D])
for j in range(K): Sigma[j] = np.eye(D)
pi = np.zeros(K)
for j in range(K): pi[j] = 1.0 / K
ll = float("inf")
ll_new = log_likelihood(pi, X, mu, Sigma)
gamma = np.zeros([N, K])
pmu = [None]*K
pSigma_ell = [None]*K
for i in range(100):
  ll_delta = abs(ll - ll_new)
  ll = ll_new
  if v:
    ax.set_title(f"i = {i}; log likelihood = {ll:.2f}; delta = {ll_delta:f}")
    [pmu[j].pop(0).remove() for j in range(K) if pmu[j]]
    [pSigma_ell[j].remove() for j in range(K) if pSigma_ell[j]]
    fig.canvas.flush_events(); # time.sleep(0.5)
    pmu = [ax.plot(mu[j,0], mu[j,1], color='brown', marker=f"${j+1}$") for j in range(K)]
    for j in range(K):
      pSigma_ell[j] = confidence_ellipse(mu[j], Sigma[j], ax, n_std=3, alpha=0.2, facecolor='pink', edgecolor='purple', zorder=0)
      ax.add_artist(pSigma_ell[j])
    fig.canvas.flush_events(); # time.sleep(0.1)
  print(f"i = {i}; log likelihood = {ll:.2f}; delta = {ll_delta:f}")
  print("mu =\n", mu)

  # E step
  for n in range(N):
    denom = 0.0
    for j in range(K):
      denom += pi[j] * normal_prob(X[n].reshape(D,1), mu[j].reshape(D,1), Sigma[j])
    for k in range(K):
      gamma[n, k] = pi[k] * normal_prob(X[n].reshape(D,1), mu[k].reshape(D,1), Sigma[k]) / denom

  # M step
  for k in range(K):
    muk = np.zeros([1, D])
    Nk = 0.0
    for n in range(N):
      muk += gamma[n, k] * X[n]
      Nk += gamma[n, k]
    mu[k] = muk / Nk
    Sigmak = np.zeros([D, D])
    for n in range(N):
      Sigmak += gamma[n, k] * (X[n] - mu[k]).reshape(D, 1).dot((X[n] - mu[k]).reshape(1, D))
    Sigma[k] = Sigmak / Nk
    pi[k] = Nk / N

  if v:
    plot_fig2(fig2, ax2, mu, trueMu, gamma, X)

  ll_new = log_likelihood(pi, X, mu, Sigma)
  if abs(ll - ll_new) < 1e-5:
    print(f"log likelihood change = {abs(ll - ll_new)} after {i} iterations")
    break
else:
  print(f"max iterations {i+1} reached; log likelihood change = {abs(ll - ll_new):f}")

if v:
  plot_fig2(fig2, ax2, mu, trueMu, gamma, X)
print("trueMu =\n", trueMu)
