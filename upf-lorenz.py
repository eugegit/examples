# Unscented Particle Filter
# "Eugene Morozov"<Eugene ~at~ HiEugene.com>

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy
from scipy.integrate import odeint
from pos_matrix_approx import *
from util import *
from probabilities import *

v = 0
start_time = preamble()

epoch_sec = 1 # int(time.time())
print("seeding rand to ", epoch_sec)
np.random.seed(epoch_sec)

### Lorenz attractor ###
rho   = 28.0
sigma = 10.0
beta  = 8.0 / 3.0

def derivatives(state, t):
  x, y, z = state # unpack the state vector
  return sigma*(y-x), x*(rho-z)-y, x*y-beta*z

x0 = [1.0, 1.0, 1.0]
delta_t = 0.05
t_start = 0
t_end = 40
t_num = int((t_end - t_start) / delta_t)
tt = np.linspace(t_start, t_end, t_num)
a_x = odeint(derivatives, x0, tt).T

if v >= 3:
  fig = plt.figure()
  ax = fig.gca(projection='3d')
  ax.plot(a_x[0,:], a_x[1,:], a_x[2,:], "-o")
  ax.set_xlabel("x")
  ax.set_ylabel("y")
  ax.set_zlabel("z")
  plt.show()

def f(x, sqrtQ):
  assert(x.shape==(n,1) or len(x.shape)==1 and x.shape[0]==n)
  # non-additive noise affecting coefficients
  rnd = sqrtQ.dot(np.random.normal(0, 1, size=(3,1)))
  rho_new   = rho   + rnd[0]
  sigma_new = sigma + rnd[1]
  beta_new  = beta  + rnd[2]
  rnd = sqrtQ.dot(np.random.normal(0, 1, size=(sqrtQ.shape[0],1)))
  x_new = np.copy(x)
  x1 = x[0] + rnd[0]
  x2 = x[1] + rnd[1]
  x3 = x[2] + rnd[2]
  x_new[0] = x1 + sigma_new*(x2-x1)*delta_t
  x_new[1] = x2 + (x1*(rho_new-x3) - x2)*delta_t
  x_new[2] = x3 + (x1*x2 - beta_new*x3)*delta_t
  return x_new

def f_all(x, sqrtQ):
  assert(x.shape[0] == n)
  # # additive noise
  # rnd = sqrtQ.dot(np.random.normal(0, 1, size=(sqrtQ.shape[0],1)))
  # x_new = np.copy(x)
  # for i in range(2*n+1):
  #   x1    = x[0,i]
  #   x2    = x[1,i]
  #   x3    = x[2,i]
  #   rho   = x[3,i]
  #   sigma = x[4,i]
  #   beta  = x[5,i]
  #   x1_new = x1 + sigma*(x2-x1)*delta_t
  #   x2_new = x2 + (x1*(rho-x3) - x2)*delta_t
  #   x3_new = x3 + (x1*x2 - beta*x3)*delta_t
  #   x_new[0,i] = x1_new + rnd[0]
  #   x_new[1,i] = x2_new + rnd[1]
  #   x_new[2,i] = x3_new + rnd[2]
  # return x_new

  # # additive full noise
  # rnd = sqrtQ.dot(np.random.normal(0, 1, size=(sqrtQ.shape[0],2*n+1)))
  # x_new = np.copy(x)
  # for i in range(2*n+1):
  #   x1    = x[0,i]
  #   x2    = x[1,i]
  #   x3    = x[2,i]
  #   # rho   = x[3,i]
  #   # sigma = x[4,i]
  #   # beta  = x[5,i]
  #   x1_new = x1 + sigma*(x2-x1)*delta_t
  #   x2_new = x2 + (x1*(rho-x3) - x2)*delta_t
  #   x3_new = x3 + (x1*x2 - beta*x3)*delta_t
  #   x_new[0,i] = x1_new + rnd[0,i]
  #   x_new[1,i] = x2_new + rnd[1,i]
  #   x_new[2,i] = x3_new + rnd[2,i]
  # return x_new

  # # non-additive noise
  # rnd = sqrtQ.dot(np.random.normal(0, 1, size=(sqrtQ.shape[0],1)))
  # x1 = x[0,:] + rnd[0]
  # x2 = x[1,:] + rnd[1]
  # x3 = x[2,:] + rnd[2]
  # x1_new = x1 + sigma*(x2-x1)*delta_t
  # x2_new = x2 + (x1*(rho-x3) - x2)*delta_t
  # x3_new = x3 + (x1*x2 - beta*x3)*delta_t
  # x_new = np.copy(x)
  # x_new[0,:] = x1_new
  # x_new[1,:] = x2_new
  # x_new[2,:] = x3_new
  # return x_new

  # non-additive noise affecting coefficients
  x_new = np.copy(x)
  for i in range(x.shape[1]):
    x_new[:,i] = f(x[:,i], sqrtQ)
  return x_new

  # # no noise
  # x1 = x[0,:]
  # x2 = x[1,:]
  # x3 = x[2,:]
  # x1_new = x1 + sigma*(x2-x1)*delta_t
  # x2_new = x2 + (x1*(rho-x3) - x2)*delta_t
  # x3_new = x3 + (x1*x2 - beta*x3)*delta_t
  # x_new = np.copy(x)
  # x_new[0,:] = x1_new
  # x_new[1,:] = x2_new
  # x_new[2,:] = x3_new
  # return x_new
########################

def h(X, sqrtR):
  # x[0:2,:] is actually x[0:1,:]
  Y = np.copy(X[0:m,:]) + sqrtR.dot(np.random.normal(0, 1, size=(sqrtR.shape[0], X.shape[1]))) # with noise
  return Y

if v >= 1:
  plt.ion()
  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')

  win_pos_x = 0 # 1680
  win_pos_y = 0 # 400
  win_width = 1300 # 1200
  win_height = 800 # 700
  move_plot(fig, win_pos_x, win_pos_y, win_width, win_height)
  plt.grid(True)
  # ax.set_xlim3d(-2, 3)
  # ax.set_ylim3d(-2, 3)
  # ax.set_zlim3d(-2, 3)
  ax.set_xlabel('X axis')
  ax.set_ylabel('Y axis')
  ax.set_zlabel('Z axis')

# augmented state: (x1, x2, x3, rho, sigma, beta, q1, q2, q3, v1, v2, v3)^T
# observation:     (y1, y2, y3)^T
N = 200 # particles
n =  12 # augmented state dimension
m =   3 # observation dimension
kappa = 1e7  # choose kappa >= 0 to guarantee positive semidefiniteness of the covariance matrix
alpha = 1e-3 # 0 <= alpha <= 1, usually 1e-3
beta1 = 2.0  # beta >= 0; for a Gaussian prior the optimal choice is beta = 2
lmbda = alpha*alpha*(n+kappa)-n
W = np.empty([2*n+1])
W[0] = lmbda / (n+lmbda) # for mean
W0c = lmbda / (n+lmbda) + 1 - alpha*alpha + beta1 # for covariance
for i in range(1, 2*n+1):
  W[i] = 1.0 / (2*(n+lmbda))

q = 0.1 # process noise
r = 0.1 # observation noise or measurement error
Q = q*np.eye(m,m)
R = r*np.eye(m,m)
sqrtQ = sqrtm(Q)
sqrtR = sqrtm(R)

def sigma_points(x, P, n, lmbda):
  X = np.empty([n, 2*n+1])
  L = sqrtmEx((n+lmbda)*P)
  for i in range(2*n+1):
    if i == 0:
      X[:,[i]] = x[:]
    elif 0 < i <= n:
      X[:,[i]] = x[:] + L[:,[i-1]]
    else:
      X[:,[i]] = x[:] - L[:,[i-1-n]]
  return X

def predict(X, sqrtQ):
  X_pred = f_all(X, sqrtQ)
  return X_pred

def predicted_mean(X):
  x_mean = np.zeros([n, 1])
  for i in range(2*n+1):
    x_mean += W[i]*X[:,[i]]
  return x_mean

def predicted_cov(X, x_mean):
  P = np.zeros([n, n])
  for i in range(2*n+1):
    delta = X[:,[i]] - x_mean
    if i == 0:
      P += W0c*delta.dot(delta.T)
    else:
      P += W[i]*delta.dot(delta.T)
  # P_ = makePositiveDefinite(P, semi=True, v=0)
  if not isPositiveDefinite(P, semi=False, msg="predicted_cov", v=1):
    pass
  return P

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
    if i == 0:
      P_post += W0c*delta.dot(delta.T)
    else:
      P_post += W[i]*delta.dot(delta.T)
  if not isPositiveDefinite(P_post, semi=False, msg="update_var", v=1):
    pass
  return P_post

def update_cov(X_pri, x_mean_pri, Y_pri, y_mean_pri):
  P_post = np.zeros([X_pri.shape[0], Y_pri.shape[0]])
  for i in range(2*n+1):
    deltaX = X_pri[:,[i]] - x_mean_pri
    deltaY = Y_pri[:,[i]] - y_mean_pri
    if i == 0:
      P_post += W0c*deltaX.dot(deltaY.T)
    else:
      P_post += W[i]*deltaX.dot(deltaY.T)
  return P_post

def kalman_gain(P_post_yy, P_post_xy):
  G = P_post_xy.dot(scipy.linalg.pinv(P_post_yy))
  return G

def observe(x, sqrtR):
  y = h(x, sqrtR)
  return y

def update_mean(x_mean_pri, G, y, y_mean_pri):
  x_mean_post = x_mean_pri + G.dot(y - y_mean_pri)
  return x_mean_post

def update_post_cov(P_pri, G, P_post_yy):
  P_post = P_pri - G.dot(P_post_yy).dot(G.T)
  # P_post_ = makePositiveDefinite(P_post, semi=False, v=0)
  if not isPositiveDefinite(P_post, semi=False, msg="update_post_cov", v=1):
    pass
  return P_post



# initial conditions
x01 = x02 = x03 = 0 # arbitrary
a_x_pri  = np.zeros([n, t_num])
a_y      = np.zeros([m, t_num])
a_x_post = np.zeros([n, t_num])
a_y[:,0] = observe(a_x[:,0].reshape((m,1)), sqrtR).flatten()
x_mean_post1 = np.empty([N, n])
P_post1 = np.empty([N, n, n])
for i in range(N):
  rnd = sqrtQ.dot(np.random.normal(0, 1, size=(sqrtQ.shape[0],1)))
  x_mean_post1[i,:] = np.array([[x01+rnd[0], x02+rnd[1], x03+rnd[2], rho, sigma, beta, 0,0,0, 0,0,0]])
  P_post1[i,:] = scipy.linalg.block_diag(20*Q, 1*Q, Q, R)
x_mean_pri = np.copy(x_mean_post1)
P_pri = np.copy(P_post1)
a_x_post[:,0] = col_mean(x_mean_post1).flatten()
a_x_pri[:,0] = col_mean(x_mean_pri).flatten()
x_mean_pri_prev = np.copy(x_mean_pri)
x_mean_post_prev1 = np.copy(x_mean_post1)
P_post_prev1 = np.copy(P_post1)

px_mean_pri         = N*[None]
px_mean_pri_lines   = N*[None]
px_mean_pri_prev    = N*[None]
px_mean_post1       = N*[None]
px_mean_post1_lines = N*[None]
px_mean_post_prev1  = N*[None]
if v >= 1:
  ax.scatter(a_x[0,0], a_x[1,0], a_x[2,0], c='r', marker='x', linewidths=1, label="x")
  ax.scatter(a_x_pri[0,0], a_x_pri[1,0], a_x_pri[2,0], c='g', marker='o', linewidths=1, label="x_pri")
  ax.scatter(a_y[0,0], a_y[1,0], a_y[2,0], c='r', marker='o', linewidths=1, label="y")
  ax.scatter(a_x_post[0,0], a_x_post[1,0], a_x_post[2,0], c='b', marker='o', linewidths=1, label="x_post")
  if v >= 2:
    for j in range(N): px_mean_pri_prev[j] = ax.scatter(x_mean_pri[j,0], x_mean_pri[j,1], x_mean_pri[j,2], c=lgreen, marker='o', linewidths=1, label="x_mean_pri" if j==0 else "")
    for j in range(N): px_mean_post_prev1[j] = ax.scatter(x_mean_post1[j,0], x_mean_post1[j,1], x_mean_post1[j,2], c=lblue, marker='o', linewidths=1, label="x_mean_post1" if j==0 else "")
  plt.legend(loc="upper left", ncol=1, scatterpoints=1, numpoints=1)
  fig.canvas.flush_events(); time.sleep(0.01)

# X_prev = np.tile(x_mean_post1, 2*n+1) doesn't work
# np.broadcast_to(x_mean_post1, (N, n, 2*n+1)) doesn't work
X_prev = np.empty([N, n, 2*n+1])
for j in range(2*n+1): X_prev[:,:,j] = x_mean_post1

w = 1.0 / N * np.ones(N) # initial weights

def ukf(y, x_mean_post, P_post):
  x_mean_post_new = np.empty([N, n])
  P_post_new = np.empty([N, n, n])
  for i in range(N):
    X = sigma_points(x_mean_post[i,:].reshape((n,1)), P_post[i,:], n, lmbda)
    if v >= 2: pX = [ax.scatter(X[0,j], X[1,j], X[2,j], marker='.', c=grey, linewidths=1) for j in range(X.shape[1])]
    if v >= 2: pX_lines1 = [ax.plot([X_prev[i,0,j], X[0,j]], [X_prev[i,1,j], X[1,j]], zs=[X_prev[i,2,j], X[2,j]], c=grey) for j in range(X.shape[1])]

    X_pri = predict(X, sqrtQ)
    if v >= 2: pX_lines2 = [ax.plot([X[0,j], X_pri[0,j]], [X[1,j], X_pri[1,j]], zs=[X[2,j], X_pri[2,j]], c=aqua) for j in range(X.shape[1])]

    x_mean_pri[i,:] = predicted_mean(X_pri).flatten()
    if v >= 2: px_mean_pri[i] = ax.scatter(x_mean_pri[i,0], x_mean_pri[i,1], x_mean_pri[i,2], c=lgreen, marker='o', linewidths=1, label="x_mean_pri")
    if v >= 2: px_mean_pri_lines[i] = ax.plot([x_mean_pri_prev[i,0], x_mean_pri[i,0]], [x_mean_pri_prev[i,1], x_mean_pri[i,1]], zs=[x_mean_pri_prev[i,2], x_mean_pri[i,2]], c=lgreen)

    P_pri[i,:] = predicted_cov(X_pri, x_mean_pri[i,:].reshape((n,1)))
    Y_pri = predicted_observation(X_pri, sqrtR)
    y_mean_pri = predicted_obs_mean(Y_pri)
    P_post_yy = update_var(Y_pri, y_mean_pri)
    P_post_xy = update_cov(X_pri, x_mean_pri[i,:].reshape((n,1)), Y_pri, y_mean_pri)
    G = kalman_gain(P_post_yy, P_post_xy)

    x_mean_post_new[i,:] = update_mean(x_mean_pri[i,:].reshape((n,1)), G, a_y[:,k].reshape((m,1)), y_mean_pri).flatten() # [:,0]
    P_post_new[i,:] = update_post_cov(P_pri[i,:], G, P_post_yy)
    if v >= 2: px_mean_post1[i] = ax.scatter(x_mean_post_new[i,0], x_mean_post_new[i,1], x_mean_post_new[i,2], c=lblue, marker='o', label="x_mean_post1")
    if v >= 2: px_mean_post1_lines[i] = ax.plot([x_mean_post_prev1[i,0], x_mean_post_new[i,0]], [x_mean_post_prev1[i,1], x_mean_post_new[i,1]], zs=[x_mean_post_prev1[i,2], x_mean_post_new[i,2]], c=lblue)

    if v >= 2:
      fig.canvas.flush_events(); time.sleep(0.01)
      [pX[j].remove() for j in range(len(pX))]
      [pX_lines1[j].pop(0).remove() for j in range(len(pX_lines1))]
      [pX_lines2[j].pop(0).remove() for j in range(len(pX_lines2))]
    X_prev[i,:,:] = X[:,:]
  return x_mean_post_new, P_post_new

for k in range(1, len(tt)):
  x = a_x[:,k].reshape((m,1))
  if v >= 1: ax.scatter(x[0], x[1], x[2], c='r', marker='x', linewidth=1, label="x")

  a_y[:,k] = observe(x, sqrtR).flatten()
  if v >= 1: ax.scatter(a_y[0,k], a_y[1,k], a_y[2,k], c='r', marker='o', linewidth=1, label="y")
  if v >= 1: ax.plot([a_y[0,k-1], a_y[0,k]], [a_y[1,k-1], a_y[1,k]], zs=[a_y[2,k-1], a_y[2,k]], c='r')

  # 2.1 importance sampling step
  # 2.1.a proposal distribution
  x_mean_post1, P_post1 = ukf(a_y[:,k], x_mean_post_prev1, P_post_prev1)

  xhat1 = sample_normal(x_mean_post1, P_post1, N, n) # xhat1[N, n]
  # our chain is now (a_x_post[:,k-1], xhat1)

  # 2.1.b evaluate the importance weights
  likelihood  = calc_likelihood(a_y[:,k].reshape(m,1), xhat1, h, R, sqrtR, N, n, m) # likelihood[N]
  trans_prior = calc_trans_prior(xhat1, x_mean_post_prev1, N, n, m)
  proposal    = calc_proposal(xhat1, x_mean_post1, P_post1, N, n)
  w = w * likelihood * trans_prior / proposal
  w = w / np.sum(w)
  if abs(np.sum(w) - 1) > 1e-5:
    print("k=%d, sum != 1 !!! sum(w)=%.3f" % (k, np.sum(w)))

  if sum(w > 0.8) == 1:
    print("k=%d, sample degeneracy: max(w)=%.2f" % (k, max(w)))
  ess = 1.0 / np.sum(w*w)
  if ess < N/2:
    print("k=%d, ess=%.3f < %d, resampling" % (k, ess, N/2))
    # 2.2 resample
    idx2 = residual_resample(np.arange(N), w, N, n)
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

  # 2.3 MCMC step (Metropolis-Hastings)
  x_mean_post2, P_tag2 = ukf(a_y[:,k], x_mean_post_prev2, P_post_prev2)
  xtag2 = sample_normal(x_mean_post2, P_tag2, N, n) # get candidate samples
  lik_tag      = calc_likelihood(a_y[:,k].reshape(m,1), xtag2, h, R, sqrtR, N, n, m)
  prior_tag    = calc_trans_prior(xtag2, x_mean_post_prev2, N, n, m)
  proposal_tag = calc_proposal(xtag2, x_mean_post2, P_tag2, N, n)

  lik2      = calc_likelihood(a_y[:,k].reshape(m,1), xhat2, h, R, sqrtR, N, n, m)
  prior2    = calc_trans_prior(xhat2, x_mean_post_prev2, N, n, m)
  proposal2 = calc_proposal(xhat2, x_mean_post2, P_post2, N, n)

  ratio = (lik_tag * prior_tag * proposal2) / (lik2 * prior2 * proposal_tag)
  accepted = 0
  rejected = 0
  for i in range(N):
    acceptance = min(1, ratio[i])
    u = np.random.uniform()
    if u <= acceptance:
      xhat2[i,:] = xtag2[i,:]
      P_post2[i,:] = P_tag2[i,:]
      accepted += 1
    else:
      xhat2[i,:] = xhat2[i,:]
      P_post2[i,:] = P_post2[i,:]
      rejected += 1
  print("accepted=%d (%.0f%%), rejected=%d" % (accepted, accepted*100.0/N, rejected))

  # 2.4 calc estimate
  a_x_post[:,k] = np.sum(np.tile(w, (n,1)).T * xhat2, axis=0)

  a_x_pri[:,k] = col_mean(x_mean_pri) # only for drawing
  # a_x_post[:,k] = col_mean(x_mean_post1) # just temporarily

  x_x_pri = a_x[0:m,k] - a_x_pri[0:m,k]
  pri_dist = np.sqrt(np.einsum('i,i->', x_x_pri, x_x_pri))
  pri_dist2 = np.sqrt(np.sum((a_x[0:m,k] - a_x_pri[0:m,k])**2))
  assert(abs(pri_dist - pri_dist2) < 1e-10)
  x_y = a_x[0:m,k] - a_y[:,k]
  y_dist = np.sqrt(np.einsum('i,i->', x_y, x_y))
  x_x_post = a_x[0:m,k] - a_x_post[0:m,k]
  post_dist = np.sqrt(np.einsum('i,i->', x_x_post, x_x_post))
  print("pri_dist=%.1f, y_dist=%.1f, post_dist=%.1f" % (pri_dist, y_dist, post_dist))

  x_mean_pri_prev   = np.copy(x_mean_pri)
  x_mean_post_prev1 = np.copy(x_mean_post1)
  P_post_prev1      = np.copy(P_post1)

  if v >= 1:
    if k > 3 and k % 15 == 0:
      plt.cla()
      margin = 4.0
      ax.set_xlim3d(a_x[0,k]-margin, a_x[0,k]+margin)
      ax.set_ylim3d(a_x[1,k]-margin, a_x[1,k]+margin)
      ax.set_zlim3d(a_x[2,k]-margin, a_x[2,k]+margin)
    ax.scatter(a_x_pri[0,k], a_x_pri[1,k], a_x_pri[2,k], c='g', marker='o', linewidth=1, label="x_pri")
    ax.scatter(a_x_post[0,k], a_x_post[1,k], a_x_post[2,k], c='b', marker='o', linewidth=1, label="x_post")
    ax.plot([a_x_pri[0,k-1], a_x_pri[0,k]], [a_x_pri[1,k-1], a_x_pri[1,k]], zs=[a_x_pri[2,k-1], a_x_pri[2,k]], c='g')
    ax.plot([a_x_post[0,k-1], a_x_post[0,k]], [a_x_post[1,k-1], a_x_post[1,k]], zs=[a_x_post[2,k-1], a_x_post[2,k]], c='b')
    if v >= 2:
      [px_mean_pri_prev[j].remove() for j in range(N)]
      [px_mean_post_prev1[j].remove() for j in range(N)]
      [px_mean_pri_lines[j].pop(0).remove() for j in range(N)]
      [px_mean_post1_lines[j].pop(0).remove() for j in range(N)]
      for j in range(N): px_mean_pri_prev[j] = px_mean_pri[j]
      for j in range(N): px_mean_post_prev1[j] = px_mean_post1[j]
    fig.canvas.flush_events(); time.sleep(0.01)
  if k%200 == 0:
    print("k = %d / %d" % (k, t_num))
    print("rho=%.3f (%.3f), sigma=%.3f (%.3f), beta=%.3f (%.3f)" % (a_x_post[3,k],rho, a_x_post[4,k],sigma, a_x_post[5,k],beta))

# stats
x_x_pri = a_x[0:m,:] - a_x_pri[0:m,:]
pri_err = np.sum(np.sqrt(np.einsum('ij,ij->j', x_x_pri, x_x_pri)))
pri_err2 = np.sum(np.sqrt(np.sum((a_x[0:m,:] - a_x_pri[0:m,:])**2, axis=0)))
assert(abs(pri_err - pri_err2) < 1e-10)
x_y = a_x[0:m,:] - a_y[:,:]
y_err = np.sum(np.sqrt(np.einsum('ij,ij->j', x_y, x_y)))
x_x_post = a_x[0:m,:] - a_x_post[0:m,:]
post_err = np.sum(np.sqrt(np.einsum('ij,ij->j', x_x_post, x_x_post)))
print("pri_err=%.1f, y_err=%.1f, post_err=%.1f" % (pri_err, y_err, post_err))

with open(__file__+".csv", 'w') as file:
  for j in range(t_num):
    file.write("%f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f\n" % (a_x_pri[0,j],a_x_pri[1,j],a_x_pri[2,j],
               a_x[0,j],a_x[1,j],a_x[2,j], a_y[0,j],a_y[1,j],a_y[2,j], a_x_post[0,j],a_x_post[1,j],a_x_post[2,j]))
with open(__file__+".session.txt", 'w') as file:
  for key in dir():
    if type(globals()[key]).__name__ not in ("function", "module", "type", "TextIOWrapper") and not key.startswith('_'):
      val = str(globals()[key])
      if len(val) > 20:
        file.write("%s =\n%s\n\n" % (key, val))
      else:
        file.write("%s = %s\n\n" % (key, val))
if v >= 1:
  plt.savefig(__file__+".png", bbox_inches="tight")
  plt.ioff()
  plt.show()
postscript(start_time)
