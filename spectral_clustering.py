# "Eugene Morozov"<Eugene ~at~ HiEugene.com>

from probabilities import *
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import Delaunay
from k_means import K_means
from iris import get_iris, get_iris_full
from hearts import get_hearts
from mines_vs_rocks import get_data2

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
  # trueSigma *= 10.0
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
# X, tK, trueMu, N_ranges = get_hearts()
# X, tK, trueMu, N_ranges = get_data2()

M, N = X.shape # M samples of dimension N

def get_eigenvector(sigma):
  A = np.empty([M, M])
  for i in range(M):
    for j in range(M):
      # A[i,j] = np.exp(-((X[i,0] - X[j,0])**2 + (X[i,1] - X[j,1])**2) / (2.0*sigma**2))
      # acc = 0; [acc := acc + (X[i,k] - X[j,k])**2 for k in range(N)]
      # A[i,j] = np.exp(-acc / (2.0*sigma**2))
      A[i,j] = np.exp(-(((X[i,:] - X[j,:])**2).sum()) / (2.0*sigma**2))
  for i in range(M):
    A[i,i] = 0

  D = np.zeros([M, M])
  for i in range(M):
    for j in range(M):
      D[i,i] += A[i,j]
  D2 = np.diag(A.dot(np.ones(M)))
  if not np.allclose(D, D2, eps):
    print("D - D2 =\n", D - D2)

  D12 = sqrtm(D)
  D12 = scipy.linalg.inv(D12)

  L = D12.dot(A).dot(D12)

  w, V = scipy.linalg.eigh(L) # for a real symmetric matrix
  W = np.asmatrix(np.diag(w))
  V = np.asmatrix(V)
  # if v: print("eigenvalues =\n", w)
  # if v: print("eigenvectors =\n", V)
  if not np.allclose(L, V.dot(W).dot(V.I), eps):
    print("L != V*W*V.I; L - V*W*V.I =\n", L - V.dot(W).dot(V.I))
  if not np.allclose(V.T, V.I, eps):
    print("V.T != V.I; V.T - V.I =\n", V.T - V.I)
  return w

def test_gap():
  if v >= 2:
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    XX = np.arange(-5, 5, 0.25)
    YY = np.arange(-5, 5, 0.25)
    XX, YY = np.meshgrid(XX, YY)
    R = np.sqrt(XX**2 + YY**2)
    ZZ = np.sin(R)
    surf = ax.plot_surface(XX, YY, ZZ, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    ax.set_zlim(-1.01, 1.01)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()
    del XX, YY, ZZ, R

  if v:
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    plt.ion()
    plt.grid(True)
    ax.set_xlabel("M")
    ax.set_ylabel(f"$\sigma$")
    ax.set_zlabel("gap")
    # ax.set_xlim3d(-2, 2)
    # ax.set_ylim3d(-2, 2)
    # ax.set_zlim3d(-2, 2)
    plt.tight_layout()
    plt.show()
    # for i in range(len(my_color)): ax.scatter(i, i, i, color=my_color[i])

  sigmas = [0.1, 0.5, 0.75, 1.0, 1.25, 1.5, 2, 3, 5, 7.5, 10]
  g = np.zeros([M,len(sigmas)])
  for si in range(len(sigmas)):
    print(f"sigma={sigmas[si]:.01f}; {si}/{len(sigmas)}")
    w = get_eigenvector(sigmas[si])
    for i in range(M-1, 0, -1):
      # d = w[i] / w[i-1]
      # d = abs((w[i] - w[i-1]) / w[i])
      d = abs(w[i] - w[i-1])
      # print(f"{i}: {d:.03f}")
      g[i,si] = d

      if v and i>=200:
        max_g = 1.0
        ax.scatter(i, sigmas[si], g[i,si], marker=".", color=my_color[int(abs(g[i,si])/max_g*len(my_color))])

    if v and si%10==1: fig.canvas.flush_events(); time.sleep(0.01)

  if v:
    fig = plt.figure()
    ax = fig.gca()
    plt.ion()
    plt.grid(True)
    ax.set_xlabel("M")
    ax.set_ylabel("gap")
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.show()
    pp = [None]*4
    for si in range(len(sigmas)):
      ppp = pp.pop(0)
      if ppp: ppp.pop(0).remove()
      pp.append( ax.plot(g[200:,si], 'o-', label=f"$\sigma={sigmas[si]:.02f}$") )
      ax.legend()
      fig.canvas.flush_events(); time.sleep(2)
  return

def get_scale():
  A = np.empty([M, M])
  for i in range(M):
    for j in range(M):
      A[i,j] = np.sqrt(((X[i,:] - X[j,:])**2).sum())
  for i in range(M):
    A[i,i] = 0

  B = np.zeros(int(0.5*(M+1)**2))
  k = 0
  for i in range(M):
    for j in range(i, M):
      B[k] = A[i,j]
      k += 1
  B = np.sort(B)
  B = B[B>0]

  if v:
    fig = plt.figure()
    ax = fig.gca()
    plt.ion()
    plt.grid(True)
    ax.plot(B)
    ax.set_title("sorted distances")
    plt.tight_layout()
    plt.show()

  if v:
    fig = plt.figure()
    ax = fig.gca()
    plt.ion()
    plt.grid(True)
    ax.hist(B)
    ax.set_title("histogram of sorted distances")
    plt.tight_layout()
    plt.show()

  intra_cluster_dist = B[int(0.1*len(B))]
  inter_cluster_dist = B[int(0.9*len(B))]
  print(f"intra_cluster_dist={intra_cluster_dist:.03f}, inter_cluster_dist={inter_cluster_dist:.03f}")
  sigma = (inter_cluster_dist + intra_cluster_dist) / 2.0
  print(f"sigma={sigma:.03f}")
  w = get_eigenvector(sigma)

  g = np.zeros(M)
  for i in range(M-1, 0, -1):
    # d = w[i] / w[i-1]
    # d = abs((w[i] - w[i-1]) / w[i])
    d = abs(w[i] - w[i-1])
    # d = abs(w[M-1] - w[i]); g[0] = 1
    # d = np.log(abs(w[i] - w[i-1]))
    g[i] = d

  mean = g[0:-1].mean()
  std  = g[0:-1].std()
  # K = 0
  # for i in range(M-1, 0, -1):
  #   if g[i] >= mean + 3*std:
  #     K += 1
  #   else:
  #     break
  for i in range(M):
    if g[i] >= mean + 3*std:
      break
  K = M - i
  print(f"K={K}")

  if v:
    fig = plt.figure()
    ax = fig.gca()
    plt.ion()
    plt.grid(True)
    plt.show()
    ax.plot(g, 'o-', label="eigenvalue gap")
    ax.set_ylabel("gap")
    ax.legend(loc="best", ncol=1, scatterpoints=1, numpoints=1)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    ax.set_title(f"$\sigma={sigma:.02f}$, K={K}, trueK={tK}")
    ax.hlines(mean, 0, M, colors='r')
    ax.hlines(mean+std, 0, M, colors='r')
    ax.hlines(mean+2*std, 0, M, colors='r')
    ax.hlines(mean+3*std, 0, M, colors='r')

  return sigma, K

# test_gap()
sigma, K = get_scale()

if v and (N==2 or N==3):
  fig = plt.figure()
  if N == 2: ax = fig.gca()
  elif N == 3: ax = fig.gca(projection='3d')
  plt.ion()
  for k in range(tK):
    if N == 2: ax.scatter(X[N_ranges[k]:N_ranges[k+1], 0], X[N_ranges[k]:N_ranges[k+1], 1], color=get_color(k), marker="o", label=f"source $X_{k}$")
    elif N == 3: ax.scatter(X[N_ranges[k]:N_ranges[k+1], 0], X[N_ranges[k]:N_ranges[k+1], 1], X[N_ranges[k]:N_ranges[k+1], 2], color=get_color(k), marker="o", label=f"source $X_{k}$")
  plt.grid(True)
  ax.legend(loc="best", ncol=1, scatterpoints=1, numpoints=1)
  ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
  plt.show()

r = K_means(X, K=tK, v=1, title_str="K-means of original data")

A = np.empty([M, M])
for i in range(M):
  for j in range(M):
    A[i,j] = np.exp(-(((X[i,:] - X[j,:])**2).sum()) / (2.0*sigma**2))
for i in range(M):
  A[i,i] = 0

D = np.zeros([M, M])
for i in range(M):
  D[i,i] = A[i,:].sum()
D2 = np.diag(A.dot(np.ones(M)))
if not np.allclose(D, D2, eps):
  print("D - D2 =\n", D - D2)

D12 = sqrtm(D)
D12 = scipy.linalg.inv(D12)

L = D12.dot(A).dot(D12)

# w, V = scipy.linalg.eig(L)
w, V = scipy.linalg.eigh(L)
W = np.asmatrix(np.diag(w))
V = np.asmatrix(V)
if v: print("eigenvalues =\n", w)
if v: print("eigenvectors =\n", V)
if not np.allclose(L, V.dot(W).dot(V.I), eps):
  print("L != V*W*V.I; L - V*W*V.I =\n", L - V.dot(W).dot(V.I))
if not np.allclose(V.T, V.I, eps):
  print("V.T != V.I; V.T - V.I =\n", V.T - V.I)

mean = w[0:-1].mean()
std  = w[0:-1].std()
K2 = 0
for i in range(M-1, 0, -1):
  if w[i] >= mean + 3*std:
    K2 += 1
  else:
    break
if v:
  fig = plt.figure()
  ax = fig.gca()
  ax.plot(w, "bo", label="eigenvalues")
  ax.hlines(mean, 0, M, colors='r')
  ax.hlines(mean+std, 0, M, colors='r')
  ax.hlines(mean+2*std, 0, M, colors='r')
  ax.hlines(mean+3*std, 0, M, colors='r')
  plt.grid(True)
  ax.legend(loc="best", ncol=1, scatterpoints=1, numpoints=1)
  ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
  ax.set_title(f"K2={K2}; trueK={tK}")
  plt.show()

# step 3
# TODO: eigenvectors are to be orthogonal to each other in case of repeated eigenvalues
Y1 = np.empty([M, K])
for j in range(K):
  Y1[:,j] = V[:,M-1-j].flatten()

# step 4
Y2 = np.empty([M, K])
for i in range(M):
  s = np.sqrt((Y1[i,:]**2).sum())
  for j in range(K):
    Y2[i,j] = Y1[i,j] / s

# step 5
r = K_means(Y2, K, v, "K-means of normalized largest eigenvectors of Laplacian")

# step 6
if v and (N==2 or N==3):
  fig = plt.figure()
  if N == 2: ax = fig.gca()
  elif N == 3: ax = fig.gca(projection='3d')
  plt.grid(True)
  ax.set_title("spectral clustering")
  for i in range(M):
    if N == 2: ax.scatter(X[i,0], X[i,1], color=get_color(r[i]), marker="o")
    elif N == 3: ax.scatter(X[i,0], X[i,1], X[i,2], color=get_color(r[i]), marker="o")
  plt.show()

if N == 2:
  tri = Delaunay(X)
  if v:
    fig3 = plt.figure()
    ax3 = fig3.gca()
    ax3.triplot(X[:,0], X[:,1], tri.simplices)
    ax3.plot(X[:,0], X[:,1], 'o')
    plt.title("Delaunay(X)")
    plt.show()

  # adjacency matrix of Delaunay graph
  A = np.zeros([M, M], dtype=bool)
  for i in range(tri.simplices.shape[0]):
    A[tri.simplices[i,0], tri.simplices[i,1]] = True
    A[tri.simplices[i,1], tri.simplices[i,2]] = True
    A[tri.simplices[i,2], tri.simplices[i,0]] = True
  A = np.logical_or(A, A.T).astype(int)
  assert(np.min(A)==0 and np.max(A)==1 and np.trace(A)==0)

  D = np.diag(A.sum(axis=0))
  L = D - A
  w, V = scipy.linalg.eigh(L, eigvals=(0,2))
  if v:
    fig4 = plt.figure()
    ax4 = fig4.gca()
    for i in range(M):
      for j in range(i+1, M):
        if A[i, j] == 1:
          ax4.plot((V[i,1], V[j,1]), (V[i,2], V[j,2]), color='blue', marker='o', markerfacecolor='green')
    plt.title(r"based on $2^{nd}$ and $3^{rd}$ smallest eigenvalues of Laplacian matrix")
    plt.show()

  # the walk matrix of the graph
  W = A.dot(scipy.linalg.inv(D))
  w, V = scipy.linalg.eigh(W.T, eigvals=(M-3,M-1))
  if v:
    fig5 = plt.figure()
    ax5 = fig5.gca()
    for i in range(M):
      for j in range(i+1, M):
        if A[i, j] == 1:
          ax5.plot((V[i,0], V[j,0]), (V[i,1], V[j,1]), color='blue', marker='o', markerfacecolor='green')
    plt.title(r"based on $2^{nd}$ and $3^{rd}$ largest eigenvalues of walk matrix")
    plt.show()

modules = []
objects = []
with open(__file__+".session.txt", 'w') as file:
  for key in dir():
    val = globals()[key]
    if type(val).__name__ == "module" and not key.startswith('_') and not "built-in" in repr(val) and not \
            (vars(val)['__file__'].startswith("/usr/") or vars(val)['__file__'].startswith("C:\\Users") or
             vars(val)['__file__'].startswith("C:\\Program Files") or vars(val)['__file__'].__contains__("venv")):
      modules.append(key)
    elif type(val).__name__ not in ("module", "function", "type", "TextIOWrapper") and not key.startswith('_'):
      if isinstance(val, Number) or isinstance(val, np.ndarray) or isinstance(val, str) or isinstance(val, list) or isinstance(val, dict):
        value = str(val)
        if len(value) > 20:
          file.write("%s =\n%s\n\n" % (key, value))
        else:
          file.write("%s = %s\n\n" % (key, value))
      else:
        if type(val).__name__ not in ["DataFrame", "ABCMeta"]:
          objects.append(key)
  for key in modules:
    val = globals()[key]
    dump_vars(file, "\n\n----- %s -----\n\n" % key, val.get_globals(), "----- end of %s -----\n" % key)
  for key in objects:
    val = globals()[key]
    if val: dump_vars(file, "\n\n----- %s -----\n\n" % key, val.__dict__.items(), "----- end of %s -----\n" % key)
if v:
  fig.savefig(__file__+".png", bbox_inches="tight")
  # fig2.savefig(__file__+"_2.png", bbox_inches="tight")
  plt.ioff()

postscript(start_time)
