# using scaled unscented transformation
# "Eugene Morozov"<Eugene ~at~ HiEugene.com>

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
import scipy
from scipy.integrate import odeint
import time
from pos_matrix_approx import sqrtm, sqrtmEx, makePositiveDefinite, isPositiveDefinite
from util import move_plot

v = 0
print("start:", time.strftime("%Y-%m-%d %H:%M:%S"))

eps = np.finfo(float).eps # or 7.0/3 - 4.0/3 - 1
print("eps =", eps)

np.set_printoptions(precision=6, suppress=True, linewidth=200)
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
delta_t = 0.001
t_start = 0
t_end = 40
t_num = int((t_end - t_start) / delta_t)
tt = np.linspace(t_start, t_end, t_num)
xx = odeint(derivatives, x0, tt).T
a_x_pri  = np.asmatrix(np.zeros([3,t_num]))
a_x      = np.asmatrix(np.zeros([3,t_num]))
a_y      = np.asmatrix(np.zeros([3,t_num]))
a_x_post = np.asmatrix(np.zeros([3,t_num]))

if v >= 2:
  fig = plt.figure()
  ax = fig.gca(projection='3d')
  ax.plot(xx[0,:], xx[1,:], xx[2,:], "-o")
  ax.set_xlabel("x")
  ax.set_ylabel("y")
  ax.set_zlabel("z")
  plt.show()

def f(x, sqrtQ):
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
  rnd = sqrtQ.dot(np.random.normal(0, 1, size=(3,1)))
  rho_new   = rho   + rnd[0]
  sigma_new = sigma + rnd[1]
  beta_new  = beta  + rnd[2]
  rnd = sqrtQ.dot(np.random.normal(0, 1, size=(sqrtQ.shape[0],2*n+1)))
  x_new = np.copy(x)
  for i in range(2*n+1):
    x1 = x[0,i] + rnd[0,i]
    x2 = x[1,i] + rnd[1,i]
    x3 = x[2,i] + rnd[2,i]
    x1_new = x1 + sigma_new*(x2-x1)*delta_t
    x2_new = x2 + (x1*(rho_new-x3) - x2)*delta_t
    x3_new = x3 + (x1*x2 - beta_new*x3)*delta_t
    x_new[0,i] = x1_new
    x_new[1,i] = x2_new
    x_new[2,i] = x3_new
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
  Y = np.copy(X[0:3,:]) + sqrtR.dot(np.random.normal(0, 1, size=(sqrtR.shape[0], X.shape[1]))) # with noise
  return Y

if v >= 1:
  plt.ion()
  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')

  win_pos_x = 0 # 1680
  win_pos_y = 0 # 400
  win_width = 1200
  win_height = 700
  move_plot(fig, win_pos_x, win_pos_y, win_width, win_height)
  plt.grid(True)
  ax.set_xlim3d(-2, 2)
  ax.set_ylim3d(-2, 2)
  ax.set_zlim3d(-2, 2)
  ax.set_xlabel('X axis')
  ax.set_ylabel('Y axis')
  ax.set_zlabel('Z axis')

# augmented state: (x1, x2, x3, rho, sigma, beta, q1, q2, q3, v1, v2, v3)^T
# observation:     (y1, y2, y3)^T
n = 12
kappa = 1e7  # choose kappa >= 0 to guarantee positive semidefiniteness of the covariance matrix
alpha = 1e-3 # 0 <= alpha <= 1, usually 1e-3
beta1 = 2.0  # beta >= 0; for a Gaussian prior the optimal choice is beta = 2
lmbda = alpha*alpha*(n+kappa)-n
W = np.empty([2*n+1])
W[0] = lmbda / (n+lmbda) # for mean
W0c = lmbda / (n+lmbda) + 1 - alpha*alpha + beta1 # for covariance
for i in range(1,2*n+1):
  W[i] = 1.0 / (2*(n+lmbda))

q = 0.01 # process noise
r = 0.005 # observation noise or measurement error
Q = q*np.eye(3,3)
R = r*np.eye(3,3)
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
  X_pred = f(X, sqrtQ)
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
x = np.matrix([[0], [0], [0], # arbitrary
               [rho], [sigma], [beta],
               [0], [0], [0],
               [0], [0], [0]])
x_mean_post = x
P_post = scipy.linalg.block_diag(20*Q, 1*Q, Q, R)
x_mean_pri = x_mean_post
P_pri = P_post
xx0 = np.matrix(xx[:,0]).T
y = observe(xx0, sqrtR)
a_x_pri[:,0]  = x_mean_pri[0:3,0]
a_x[:,0]      = xx0
a_y[:,0]      = y
a_x_post[:,0] = x_mean_post[0:3,0]

if v >= 1:
  px_pri = ax.scatter(x_mean_pri[0], x_mean_pri[1], x_mean_pri[2], c='g', marker='o', linewidths=1, label='x_pri')
  # https://github.com/minillinim/ellipsoid/blob/master/ellipsoid.py
  # https://stackoverflow.com/questions/7819498/plotting-ellipsoid-with-matplotlib
  px = ax.scatter(xx[0,0], xx[1,0], xx[2,0], c='r', marker='x', linewidths=1, label='x')
  py = ax.scatter(y[0], y[1], y[2], c='r', marker='o', linewidths=1, label='y')
  px_post = ax.scatter(x_mean_post[0], x_mean_post[1], x_mean_post[2], c='b', marker='o', linewidths=1, label='x_post')
  plt.legend(loc="upper left", ncol=1, scatterpoints=1)
  fig.canvas.flush_events(); time.sleep(0.01)

X_prev = np.tile(x_mean_post, 2*n+1)
x_mean_pri_prev = x_mean_pri
y_prev = y
x_mean_post_prev = x_mean_post
if v >= 2: pX_prev = []

for k in range(1, len(tt)):
  X = sigma_points(x_mean_post, P_post, n, lmbda)
  if v >= 2: pX = [ax.scatter(X[0,i], X[1,i], X[2,i], marker='.', c=(0.5, 0.5, 0.5), linewidths=1) for i in range(X.shape[1])]
  if v >= 2: pX_lines1 = [ax.plot([X_prev[0,i], X[0,i]], [X_prev[1,i], X[1,i]], zs=[X_prev[2,i], X[2,i]], c=(0.5, 0.5, 0.5)) for i in range(X.shape[1])]

  X_pri = predict(X, sqrtQ)
  if v >= 2: pX_lines2 = [ax.plot([X[0,i], X_pri[0,i]], [X[1,i], X_pri[1,i]], zs=[X[2,i], X_pri[2,i]], c=(0.75, 0.25, 0.75)) for i in range(X.shape[1])]

  x_mean_pri = predicted_mean(X_pri)
  if v >= 1: px_pri = ax.scatter(x_mean_pri[0,0], x_mean_pri[1,0], x_mean_pri[2,0], c='g', marker='o', linewidths=1, label='x_pri')
  if v >= 1: ax.plot([x_mean_pri_prev[0], x_mean_pri[0]], [x_mean_pri_prev[1], x_mean_pri[1]], zs=[x_mean_pri_prev[2], x_mean_pri[2]], c='g')

  P_pri = predicted_cov(X_pri, x_mean_pri)
  Y_pri = predicted_observation(X_pri, sqrtR)
  y_mean_pri = predicted_obs_mean(Y_pri)
  P_post_yy = update_var(Y_pri, y_mean_pri)
  P_post_xy = update_cov(X_pri, x_mean_pri, Y_pri, y_mean_pri)
  G = kalman_gain(P_post_yy, P_post_xy)

  x = np.matrix(xx[:,k]).T
  if v >= 1: px = ax.scatter(x[0], x[1], x[2], c='r', marker='x', label='x')

  y = observe(x, sqrtR)
  if v >= 1: py = ax.scatter(y[0], y[1], y[2], c='r', marker='o', label='y')
  if v >= 1: ax.plot([y_prev[0,0], y[0,0]], [y_prev[1,0], y[1,0]], zs=[y_prev[2,0], y[2,0]], c='r')

  x_mean_post = update_mean(x_mean_pri, G, y, y_mean_pri)
  P_post = update_post_cov(P_pri, G, P_post_yy)
  if v >= 1: px_post = ax.scatter(x_mean_post[0,0], x_mean_post[1,0], x_mean_post[2,0], c='b', marker='o', label='x_post')
  if v >= 1: ax.plot([x_mean_post_prev[0,0], x_mean_post[0,0]], [x_mean_post_prev[1,0], x_mean_post[1,0]], zs=[x_mean_post_prev[2,0], x_mean_post[2,0]], c='b')

  a_x_pri[:,k]  = x_mean_pri[0:3]
  a_x[:,k]      = x
  a_y[:,k]      = y
  a_x_post[:,k] = x_mean_post[0:3]

  if k%20 == 0:
    print("k = %d / %d" % (k, len(tt)))
    print("G =\n", G)
    print("rho=%.3f (%.3f), sigma=%.3f (%.3f), beta=%.3f (%.3f)" % (x_mean_post[3],rho, x_mean_post[4],sigma, x_mean_post[5],beta))
    if v >= 1: fig.canvas.flush_events(); time.sleep(0.01)

  # if k%50 == 0:
  #   plt.cla()
  #   ax.set_xlim3d(x[0]-5, x[0]+5)
  #   ax.set_ylim3d(x[1]-5, x[1]+5)
  #   ax.set_zlim3d(x[2]-5, x[2]+5)

  if v >= 2:
    fig.canvas.flush_events(); time.sleep(0.1)
    [pX_prev[i].remove() for i in range(len(pX_prev))]
    [pX_lines1[i].pop(0).remove() for i in range(len(pX_lines1))]
    [pX_lines2[i].pop(0).remove() for i in range(len(pX_lines2))]

  X_prev = X
  x_mean_pri_prev = x_mean_pri
  y_prev = y
  x_mean_post_prev = x_mean_post
  if v >= 2:
    pX_prev = pX

  if k%50 == 0:
    if v >= 2: pX_prev = []
    if v >= 1:
      plt.cla()
      ax.set_xlim3d(x[0]-5, x[0]+5)
      ax.set_ylim3d(x[1]-5, x[1]+5)
      ax.set_zlim3d(x[2]-5, x[2]+5)
  # if k%500 == 0:
  #   v = 2

with open(__file__+".csv", 'w') as file:
  for i in range(t_num):
    file.write("%f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f\n" % (a_x_pri[0,i],a_x_pri[1,i],a_x_pri[2,i],
               a_x[0,i],a_x[1,i],a_x[2,i], a_y[0,i],a_y[1,i],a_y[2,i], a_x_post[0,i],a_x_post[1,i],a_x_post[2,i]))
if v >= 1:
  plt.savefig(__file__+".png", bbox_inches="tight")
  plt.ioff()
  plt.show()
print(time.strftime("%Y-%m-%d %H:%M:%S"))
print("end.")
