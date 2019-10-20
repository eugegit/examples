# mean-variance portfolio selection
#
# "Eugene Morozov"<Eugene ~at~ HiEugene.com>

import scipy.stats
import matplotlib.pyplot as plt
from util import *
from pos_matrix_approx import *

v = 1
start_time = preamble()

epoch_sec = 1 # int(time.time())
print("seeding rand to ", epoch_sec)
np.random.seed(epoch_sec)

m = 3 # stocks + 1 bond
# time: from 0 to T-1
T = 100

Sigma = np.matrix([[ 1.0,  0.8 ,  0.6 , -0.6 ],
                   [ 0.8,  1.0 ,  0.48, -0.48],
                   [ 0.6,  0.48,  1.0 , -0.36],
                   [-0.6, -0.48, -0.36,  1.0 ]])
print("Sigma =\n", Sigma)
if not isPositiveDefinite(Sigma, v=v):
  raise Exception("couldn't get a positive definite correlation matrix")
Z = np.random.multivariate_normal(mean=np.zeros(m+1), cov=Sigma, size=T) / 2.0
Z[0,0] = 5.0
r = np.add.accumulate(Z[:,0])
assert(np.all(r > 0))
if v > 0:
  plt.ion()
  fig = plt.figure();  ax = fig.gca();  plt.grid(True)
  win_pos_x = 1780
  win_pos_y = 0
  win_width = 777
  win_height = 800
  move_plot(fig, win_pos_x, win_pos_y, win_width, win_height)
  ax.plot(range(len(r)), r, label="r")
  plt.tight_layout();  plt.autoscale(enable=True, axis='x', tight=True)

b = np.empty([T,m])
for i in range(m):
  Z[0,i+1] = r[0] + 10 - 2*Sigma[0,i+1] # np.random.rand() * 3 + 2
  b[:,i] = np.add.accumulate(Z[:,i+1])
  assert(np.all(b[:,i] > r))
  if v > 0:
    ax.plot(range(len(b[:,i])), b[:,i], label=r"$b_{%d}$"%(i+1))
    plt.legend(loc="best", ncol=1, scatterpoints=1) # "upper left"
corr12 = np.corrcoef(r, b[:,0])
corr13 = np.corrcoef(r, b[:,1])
corr14 = np.corrcoef(r, b[:,2])
print("r-b: corr12 = %.2f, corr13 = %.2f, corr14 = %.2f" % (corr12[0,1], corr13[0,1], corr14[0,1]))
corr23 = np.corrcoef(b[:,0], b[:,1])
corr24 = np.corrcoef(b[:,0], b[:,2])
corr34 = np.corrcoef(b[:,1], b[:,2])
print("b-b: corr23 = %.2f, corr24 = %.2f, corr34 = %.2f" % (corr23[0,1], corr24[0,1], corr34[0,1]))

B = np.empty([T,m])
for i in range(m):
  B[:,i] = b[:,i] - r
if v > 0:
  fig = plt.figure();  ax = fig.gca();  plt.grid(True)
  ax.plot(range(T), B)
  plt.tight_layout();  plt.autoscale(enable=True, axis='x', tight=True)
  plt.legend([r"$B_{1}=b_{1}—r$", r"$B_{2}=b_{2}-r$", r"$B_{3}=b_{3}—r$"], loc="best", ncol=1, scatterpoints=1)

Sigma = np.matrix([[ 1.0 ,  0.8, -0.48],
                   [ 0.8 ,  1.0, -0.6 ],
                   [-0.48, -0.6,  1.0 ]])
print("Sigma =\n", Sigma)
if not isPositiveDefinite(Sigma, v=v):
  raise Exception("couldn't get a positive definite correlation matrix")

def get_cov_matrix1(Sigma):
  sigma = np.zeros([T,m,m])
  L = scipy.linalg.cholesky(Sigma, lower=True)
  assert(np.all(np.imag(L) == 0))
  assert(np.allclose(Sigma, L.dot(L.T)))
  sigma[0,:,:] = Sigma
  if not isPositiveDefinite(sigma[0,:,:], v=v):
    raise Exception("couldn't get a positive definite correlation matrix")
  for t in range(1,T):
    d = 0.1*np.random.normal(0, 1, size=(2*m,1))
    L[1,0] += d[0]
    L[2,0] += d[1]
    L[2,1] += d[2]
    L[0,0] += d[3]
    L[1,1] += d[4]
    L[2,2] += d[5]
    # The eigenvalues of a triangular matrix are the entries on its main diagonal.
    if L[0,0] < 0.5:
      L[0,0] = 0.5
    if L[1,1] < 0.5:
      L[1,1] = 0.5
    if L[2,2] < 0.5:
      L[2,2] = 0.5
    sigma[t,:,:] = L.dot(L.T)
    if not isPositiveDefinite(sigma[t,:,:], v=v):
      raise Exception("couldn't get a positive definite correlation matrix")

    w, V = scipy.linalg.eigh(sigma[t,:,:]) # sigma is symmetrical
    if t == 1: print("sigma eigenvalues:")
    print(w)
    W = np.asmatrix(np.diag(w))
    V = np.asmatrix(V)
    assert(scipy.allclose(sigma[t,:,:], V.dot(W).dot(V.I), eps))
  return sigma

def get_cov_matrix2(Sigma):
  # see upf_bonds_calib.py
  pass

# from historical data
def get_cov_matrix3(Sigma):
  win = 20
  Z = np.random.multivariate_normal(mean=np.zeros(m), cov=Sigma, size=T+win) / 4.0
  # Z = np.random.randn(T+win, m) / 4.0
  x = np.empty([T+win, m])
  for i in range(m):
    Z[0,i] = i + 5.0
    x[:,i] = np.add.accumulate(Z[:,i])
  corr01 = np.corrcoef(x[:,0], x[:,1])
  corr02 = np.corrcoef(x[:,0], x[:,2])
  corr12 = np.corrcoef(x[:,1], x[:,2])
  print("x: corr01 = %.2f, corr02 = %.2f, corr12 = %.2f" % (corr01[0,1], corr02[0,1], corr12[0,1]))
  if v > 0:
    fig = plt.figure();  ax = fig.gca();  plt.grid(True)
    for i in range(m):  ax.plot(x[:,i])
    plt.legend([r"$x_{1}$", r"$x_{2}$", r"$x_{3}$"], loc="best", ncol=1, scatterpoints=1)
  sigma = np.zeros([T,m,m])
  for t in range(T):
    sigma[t,:,:] = np.corrcoef(x[t:t+win,:].T)
    # sigma[t] = np.cov(x[t:t+win,:].T)
    sigma[t,:,:] = sigma[t,:,:] / 1.0
    for i in range(m):  sigma[t,i,i] = 7
    if not isPositiveDefinite(sigma[t,:,:], v=v):
      raise Exception("couldn't get a positive definite correlation matrix")
    # kappa = np.linalg.cond(sigma[t,:,:])
    # sigma2 = sigma[t,:,:].dot(sigma[t,:,:].T)
    # kappa2 = np.linalg.cond(sigma2)
    # print("condition number = %5.0f; %5.0f" % (kappa, kappa2))
  return sigma

sigma = get_cov_matrix3(Sigma)
if v > 0:
  fig = plt.figure();  ax = fig.gca();  plt.grid(True)
  ax.plot(sigma[:,0,0])
  ax.plot(sigma[:,0,1])
  ax.plot(sigma[:,0,2])
  ax.plot(sigma[:,1,0])
  ax.plot(sigma[:,1,1])
  ax.plot(sigma[:,1,2])
  ax.plot(sigma[:,2,0])
  ax.plot(sigma[:,2,1])
  ax.plot(sigma[:,2,2])
  plt.legend([r"$\sigma_{00}$", r"$\sigma_{01}$", r"$\sigma_{02}$",
              r"$\sigma_{10}$", r"$\sigma_{11}$", r"$\sigma_{12}$",
              r"$\sigma_{20}$", r"$\sigma_{21}$", r"$\sigma_{22}$"], loc="best", ncol=1, scatterpoints=1)

rho = np.zeros([T])
for t in range(T):
  sigma2 = sigma[t,:,:].dot(sigma[t,:,:].T)
  kappa = np.linalg.cond(sigma2)
  print("t = %2d,       condition number = %5.0f" % (t, kappa))
  w, V = scipy.linalg.eigh(sigma2)
  kappa2 = w[m-1] / w[0] # sqrt by wikipedia
  assert(scipy.allclose(kappa, kappa2, 1.0))

  s1 = scipy.linalg.inv(sigma2)
  assert(scipy.allclose(sigma2.dot(s1), np.eye(m), eps))
  s1a = scipy.linalg.solve(sigma2, np.eye(m))
  assert(scipy.allclose(s1, s1a, eps))

  # preconditioning
  P = np.diag(np.diag(sigma2))
  P1 = np.diag(1.0 / np.diag(sigma2))
  s2 = sigma2.dot(P1)
  kappa3 = np.linalg.cond(s2)
  print("condition number for sigma2*P1 = %5.0f" % kappa3)
  y = scipy.linalg.solve(s2, np.eye(m))
  x = scipy.linalg.solve(P, y)
  assert(scipy.allclose(s1, x, 1e-5))

  rho[t] = B[t,:].reshape(1,m).dot(s1).dot(B[t,:].reshape(m,1))

if v > 0:
  fig = plt.figure();  ax = fig.gca();  plt.grid(True)
  ax.plot(rho)
  plt.legend([r"$\rho$"], loc="best", ncol=1, scatterpoints=1)
  fig = plt.figure();  ax = fig.gca();  plt.grid(True)
  ax.plot(r - rho)
  plt.legend([r"$r-\rho$"], loc="best", ncol=1, scatterpoints=1)

dt = 0.01
x0 = 1000

Rho = 0
for t in range(T):
  Rho += rho[t]*dt
R = 0
for t in range(T):
  R += r[t]*dt
RRho = 0
for t in range(T):
  RRho += (r[t] - rho[t])*dt
RRRho = 0
for t in range(T):
  RRRho += (2*r[t] - rho[t])*dt
print(f"Rho = {Rho:,.01f}; R = {R:,.01f}; RRho = {RRho:,.01f}; RRRho = {RRRho:,.01f}")
alpha = np.exp(RRho)
beta = 1 - np.exp(-Rho)
delta = np.exp(RRRho)
print(f"alpha = {alpha:,.01f}; beta = {beta:,.01f}; delta = {delta:,.01f}")

mus = np.concatenate((np.arange(5.5e-6, 1e-5, 1e-7), np.arange(1e-5, 1e-4, 1e-6), np.arange(1e-4, 1e-3, 1e-4), np.arange(1e-3, 1e-2, 1e-3), np.arange(1e-2, 1e-1, 1e-2), np.arange(1e-1, 1, 1e-1),
                      np.arange(1, 10, 1), np.arange(10, 100, 10), np.arange(1e2, 1e3, 1e2), np.arange(1e3, 1e4, 1e3), np.arange(1e4, 1e5, 1e4)))
ef = np.zeros([len(mus), 2])
lmbda_a = np.zeros([len(mus)])
gamma_a = np.zeros([len(mus)])
V_a     = np.zeros([len(mus)])
for i in range(len(mus)):
  mu = mus[i]
  lmbda_star = Rho + 2*mu*x0*R
  gamma_star = lmbda_star / (2*mu)
  print(f"mu = {mu:g}; lmbda_star = {lmbda_star:,.03f}; gamma_star = {gamma_star:,.01f}")
  lmbda_a[i] = lmbda_star
  gamma_a[i] = gamma_star
  V_a[i] = beta*(1-beta)*gamma_star**2 - (2*beta*beta*alpha*x0)/(1-beta) * gamma_star + (alpha**2 + (beta*delta-alpha**2)/(1-beta))*x0**2

  # calc (223)
  Ex = np.zeros([T])
  Ex[0] = x0
  print(f"t = 0; Ex(t) = {Ex[0]:,.01f}")
  for t in range(1,T):
    rr = 0
    for s in range(t,T):
      rr += r[s]*dt
    Ex[t] = Ex[t-1] + ((r[t]-rho[t])*Ex[t-1] + gamma_star*np.exp(-rr)*rho[t])*dt
    print(f"t = {t:d}; Ex(t) = {Ex[t]:,.01f}")
  if v > 1 or mu == 0.5:
    fig = plt.figure();  ax = fig.gca();  plt.grid(True)
    ax.plot(Ex)
    plt.legend([r"$\mathbb{E}[x(t)]$"], loc="best", ncol=1, scatterpoints=1)
    plt.title(r"$\mu=$%.01f" % mu)

  # verify E[x(T)] according to (225)
  ExT = alpha*x0 + beta*gamma_star
  diff = abs(Ex[T-1] - ExT) / Ex[T-1]
  print(f"E[x(T)] = {Ex[T-1]:,.01f}; ExT = {ExT:,.01f}; diff = {diff*100:.01f}%")
  assert(diff < 0.02)

  # Var as in (229)
  ExstarT = Ex[T-1]
  VxstarT = np.exp(-Rho) / (1 - np.exp(-Rho)) * (ExstarT - x0*np.exp(R))**2
  ef[i] = [ExstarT, VxstarT]
if v > 0:
  fig = plt.figure();  ax = fig.gca();  plt.grid(True)
  ax.plot(mus, lmbda_a, "o-")
  ax.set(xlabel=r"$\mu$", ylabel=r"$\lambda^{*}$")
  fig = plt.figure();  ax = fig.gca();  plt.grid(True)
  ax.plot(mus, gamma_a, "o-")
  ax.set(xlabel=r"$\mu$", ylabel=r"$\gamma^{*}$")
  fig = plt.figure();  ax = fig.gca();  plt.grid(True)
  ax.plot(mus, V_a, "o-")
  ax.set(xlabel=r"$\mu$", ylabel=r"$\mathbb{V}\:[x^{*}(T)]$")
  fig = plt.figure();  ax = fig.gca();  plt.grid(True)
  ax.plot(gamma_a, V_a, "o-")
  ax.set(xlabel=r"$\gamma^{*}$", ylabel=r"$\mathbb{V}\:[X^{*}(T)]$")
  # plt.show()
if v >= 0:
  fig = plt.figure();  ax = fig.gca();  plt.grid(True)
  ax.plot(ef[:,0], ef[:,1], 'o-')
  # ax.loglog(ef[:,0], ef[:,1], 'o-')
  ax.set(xlabel=r"$\mathbb{E}\:[X^{*}(T)]$", ylabel=r"$\mathbb{V}\:[x^{*}(T)]$", title="Efficient Frontier")
  for i in range(len(mus)):
    mu = mus[i]
    x,y = ef[i]
    label = r"$\mu=$"+"{:g}".format(mu)
    if i%10 == 0:
      plt.annotate(label, (x,y),
                   textcoords = "offset points", # how to position the text
                   xytext = (20,5), # distance from text to points (x,y)
                   ha = 'center') # horizontal alignment can be left, right or center
  # plt.xticks(np.arange(0,10,1))
  # plt.yticks(np.arange(0,7,0.5))
  plt.show()

postscript(start_time)
exit(0)
