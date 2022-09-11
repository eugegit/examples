# linear Kalman filter
# "Eugene Morozov"<Eugene ~at~ HiEugene.com>

import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.linalg
import math
import time
from util import is_positive_semi_definite

dt = 1
# transition matrix
F = np.matrix([[1, 0, dt, 0, dt*dt/2.0,         0],
               [0, 1, 0, dt,         0, dt*dt/2.0],
               [0, 0, 1, 0,         dt,         0],
               [0, 0, 0, 1,          0,        dt],
               [0, 0, 0, 0,          1,         0],
               [0, 0, 0, 0,          0,         1]])
q = 0.001 # process noise
Q = q*np.eye(F.shape[0])
# measurement matrix
H = np.matrix([[1, 0, 0, 0, 0, 0],
               [0, 1, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0]])
I = np.eye(H.shape[0])
r = 0.005 # observation noise / measurement error
R = r*I

F1 = scipy.linalg.pinv(H.T.dot(np.linalg.inv(R).dot(H))) # inverse of Fisher information matrix

np.set_printoptions(precision=8, suppress=True)

def calc_hare(t):
  a = 2.0
  b = 1.0
  x1 = a*math.cos(t)
  x2 = b*math.sin(t)
  return np.array([[x1, x2, 0,0, 0,0]]).T

def calc_observation(x):
  stddev = math.sqrt(r)
  y = x + np.array([[np.random.normal(0, stddev), np.random.normal(0, stddev), 0,0, 0,0]]).T
  return y

def calc_hound_predict(x):
  x_pri = F.dot(x) # + B*u
  # a priori covariance matrix
  calc_hound_predict.P_minus = F.dot(calc_hound_update.P).dot(F.T) + Q
  return x_pri
calc_hound_predict.P_minus = None

def calc_hound_update(x_pri, y):
  # innovation or measurement residual
  y_tilde = y - H.dot(x_pri)
  # innovation covariance
  S = H.dot(calc_hound_predict.P_minus).dot(H.T) + R
  # optimal Kalman gain
  K = calc_hound_predict.P_minus.dot(H.T).dot(scipy.linalg.pinv(S))
  # print("K =\n", K)
  x_post = x_pri + K.dot(y_tilde)

  # updated (a posteriori) estimate covariance; only correct for the optima] gain
  calc_hound_update.P = (I - K.dot(H)).dot(calc_hound_predict.P_minus)
  is_positive_semi_definite(calc_hound_update.P, "covariance is not positive semidefinite: ")

  # # making covariance positive semidefinite
  # B = (calc_hound_update.P + calc_hound_update.P.T) / 2
  # w, V = scipy.linalg.eig(B)
  # if np.any(np.real(w) < 0):
  #   print(np.real(w))
  # W = np.asmatrix(np.diag(w))
  # V = np.asmatrix(V)
  # W[W < 0] = 0
  # A_ = V*W*V.T
  # calc_hound_update.P = np.real(A_)

  # Joseph form
  # IKH = I - K.dot(H)
  # calc_hound_update.P = IKH.dot(calc_hound_predict.P_minus).dot(IKH.T) + K.dot(R).dot(K.T)

  # Cramer-Rao inequality for a Gaussian noise: P >= (H^T * R^—1 * H)^-1
  # The matrix inequality A >= B is understood to mean that the matrix A—B is positive semidefinite
  is_positive_semi_definite(calc_hound_update.P - F1, "Cramer—Rao inequality failed: ")

  # print("t =",t, "\nx_pri =\n", x_pri, "\nP_minus =\n",calc_hound_predict.P_minus, "\ny =\n",y, "\nK =\n",K, "\nx_post =\n",x_post, "\nP =\n",calc_hound_update.P)
  # measurement post—fit residual
  y_tilde_post = y - H.dot(x_post)
  print("measurement post-fit residual = (%.3f, %.3f)" % (y_tilde_post[0,0], y_tilde_post[1,0]))
  # x_hat == x_post
  # x_hat_minus == x_pri
  # error x_tilde = x_pri — x_hat
  # estimate y_hat == C*x_pri
  # innovation y_ti1de = y — y_hat
  return x_post
calc_hound_update.P = R # = E[(x0-E[x0])*(x0—E[x0]).T]

plt.ion()
fig, ax = plt.subplots()
plt.axis([-3.1, 3.1, -2.1, 2.1])
plt.grid(True)

# initial conditions at t = O
t = 0
x = calc_hare(t)
y = calc_observation(x)
x_post = np.array([[0, 0,     # x1, x2
                    0, 0,     # v1, v2
                    0, 0]]).T # a1, a2
x_pri = np.copy(x_post)
px, = ax.plot(x[0,0], x[1,0], 'rx')
py, = ax.plot(y[0,0], y[1,0], 'ro')
py_cir = plt.Circle((y[0,0], y[1,0]), math.sqrt(r), color='r', alpha=0.1); ax.add_artist(py_cir)
px_pri, = ax.plot(x_pri[0,0], x_pri[1,0], 'go')
px_pri_cir = plt.Circle((x_pri[0,0], x_pri[1, 0]), math.sqrt(q), color='g', alpha=0.1); ax.add_artist(px_pri_cir)
px_post, = ax.plot(x_post[0,0], x_post[1,0], 'bo')
px_post_cir = plt.Circle((x_post[0,0], x_post[1,0]), math.sqrt(r), color='b', alpha=0.1); ax.add_artist(px_post_cir)
px_prev = px
py_prev = py
py_cir_prev = py_cir
px_pri_prev = px_pri
px_pri_cir_prev = px_pri_cir
px_post_prev = px_post
px_post_cir_prev = px_post_cir
fig.canvas.flush_events(); time.sleep(0.3)
t_pre = t
x_pre = np.copy(x)
y_pre = np.copy(y)
x_pri_pre = np.copy(x_post)
x_post_pre = np.copy(x_post)

for t in np.arange(0.1, 2*math.pi-0.2, 0.1):
  x = calc_hare(t)
  y = calc_observation(x)
  x_pri = calc_hound_predict(x_post)
  x_post = calc_hound_update(x_pri, y)

  px, = ax.plot(x[0,0], x[1,0], 'rx', ms=7.0, label='x')
  ax.plot([x_pre[0,0], x[0,0]], [x_pre[1,0], x[1,0]], '--', lw=1, color='r')
  fig.canvas.flush_events(); time.sleep(0.1)
  px_pri, = ax.plot(x_pri[0,0], x_pri[1,0], 'go', ms=4.0, label='x_pri')
  ax.plot([x_pri_pre[0,0], x_pri[0,0]], [x_pri_pre[1,0], x_pri[1,0]], lw=1, color='g')
  # TODO: draw ellipse off the covariance matrix
  px_pri_cir = plt.Circle((x_pri[0,0], x_pri[1,0]), math.sqrt(np.trace(calc_hound_predict.P_minus.T*calc_hound_predict.P_minus)), color='g', alpha=0.1); ax.add_artist(px_pri_cir)
  fig.canvas.flush_events(); time.sleep(0.1)
  py, = ax.plot(y[0,0], y[1,0], 'ro', ms=7.0, label='y')
  ax.plot([y_pre[0,0], y[0,0]], [y_pre[1,0], y[1,0]], lw=1, color='r')
  py_cir = plt.Circle((y[0,0], y[1,0]), math.sqrt(r), color='r', alpha=0.1); ax.add_artist(py_cir)
  fig.canvas.flush_events(); time.sleep(0.1)
  px_post, = ax.plot(x_post[0,0], x_post[1,0], 'bo', ms=4.0, label='x_post')
  ax.plot([x_post_pre[0,0], x_post[0,0]], [x_post_pre[1,0], x_post[1,0]], lw=1, color='b')
  px_post_cir = plt.Circle((x_post[0,0], x_post[1,0]), math.sqrt(np.trace(calc_hound_update.P.T*calc_hound_update.P)), color='b', alpha=0.1); ax.add_artist(px_post_cir)
  fig.canvas.flush_events(); time.sleep(0.1)
  plt.legend(handles=[px, py, px_pri, px_post])
  #px_prev.remove()
  #py_prev.remove()
  py_cir_prev.remove()
  #px_pri_prev.remove()
  px_pri_cir_prev.remove()
  #px_post_prev.remove()
  px_post_cir_prev.remove()
  px_prev = px
  py_prev = py
  py_cir_prev = py_cir
  px_pri_prev = px_pri
  px_pri_cir_prev = px_pri_cir
  px_post_prev = px_post
  px_post_cir_prev = px_post_cir
  fig.canvas.flush_events(); time.sleep(0.2)
  t_pre = t
  x_pre = np.copy(x)
  y_pre = np.copy(y)
  x_pri_pre = np.copy(x_pri)
  x_post_pre = np.copy(x_post)
plt.ioff()
plt.show()
print("end.")
