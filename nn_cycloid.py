"""
Multilayer perceptron for cycloid and Lorenz attractor.

"Eugene Morozov"<Eugene ~at~ HiEugene.com>
"""

import matplotlib.pyplot as plt
import torch.nn
import torch.nn as nn
import torch.optim as optim
import scipy.linalg
from scipy.integrate import odeint
from util import *

v = 1
epoch_sec = 1 # int(time.time())
start_time, device = preamble(epoch_sec)

rnd = 0.4
do_dropout = False
dropout_prob_zero = 0.5
do_batch_normalization = False
do_regularization = False
L2_lambda = 0.01
do_early_stopping = False
early_stop_patience = 2
info_step = 50
max_epoch = 75
batch_size = 20
train_data_batches = 80

if v:
  plt.ion()
  fig_grad = plt.figure(); ax_grad = fig_grad.gca()
  plt.show(); plt.tight_layout()
  move_plot(fig_grad, 0, 0, 1000, 500); fig_grad.canvas.flush_events()
  i_grad = 0
  def fig_grad_init():
    global i_grad
    ax_grad.cla()
    ax_grad.grid()
    ax_grad.set_title(f"grad norm-2")
    i_grad = 0

  fig_w, ax_w = plt.subplots(3, 4)
  fig_w.show(); plt.tight_layout()
  move_plot(fig_w, 1000, 0, 1600, 1000); fig_w.canvas.flush_events()
  def fig_w_init():
    pass

  fig_loss = plt.figure(); ax_loss = fig_loss.gca()
  plt.show(); plt.tight_layout()
  move_plot(fig_loss, 0, 500, 1000, 800); fig_loss.canvas.flush_events()
  i_loss = 0
  def fig_loss_init(lr):
    global i_loss
    ax_loss.cla()
    ax_loss.grid()
    ax_loss.set_title(f"loss; lr={lr}")
    i_loss = 0

  fig_ver = plt.figure()
  ax_ver = fig_ver.gca()
  # ax_ver = fig_ver.add_subplot(projection="3d")
  fig_ver.show(); plt.tight_layout()
  move_plot(fig_ver, 600, 200, 1000, 800); fig_ver.canvas.flush_events()
  def fig_ver_init():
    ax_ver.cla()
    ax_ver.grid()
    ax_ver.set_title(f"cycloid; rnd={rnd}")
    # ax_ver.set_title(f"Lorenz attractor; rnd={rnd}")

  fig_lr = plt.figure(); ax_lr = fig_lr.gca()
  fig_lr.show() #; plt.tight_layout()
  move_plot(fig_lr, 1200, 200, 1000, 800); fig_lr.canvas.flush_events()
  def fig_lr_init():
    ax_lr.cla()
    ax_lr.grid()
    ax_lr.set_xlabel("learning rate")
    ax_lr.set_ylabel("loss")
  fig_lr_init()

def cycloid(t, rnd, r=0.5):
  x = r*(t-np.sin(t))
  y = r*(1-np.cos(t))
  if rnd > 0:
    noise = np.random.uniform(low=-rnd, high=rnd+1e-8, size=len(x))
    y += noise
  return x, y

def cycloid_normalize_x(x):
  # already in [0, 1]
  return x

### Lorenz attractor ###
Lorenz_rho   = 28.0
Lorenz_sigma = 10.0
Lorenz_beta  = 8.0 / 3.0

def Lorenz_derivatives(state, t):
  x, y, z = state # unpack the state vector
  return Lorenz_sigma*(y-x), x*(Lorenz_rho-z)-y, x*y-Lorenz_beta*z

def Lorenz(t, rnd, x0=np.random.uniform(low=-10, high=10+1e-8, size=3)):
  xx = odeint(Lorenz_derivatives, x0, t)
  xx /= 20
  if rnd > 0:
    noise = np.random.uniform(low=-rnd, high=rnd+1e-8, size=xx.shape[0])
    xx[:,2] += noise
  return xx[:,[0,1]], xx[:,2]

def Lorenz_normalize_x(x):
  return x

if v and False:
  t_num = 600
  delta_t = 0.05
  t_start = 100*np.random.rand()
  t_end = t_start + t_num*delta_t
  t = np.linspace(t_start, t_end, t_num)
  x, y = Lorenz(t, rnd=0)
  x = Lorenz_normalize_x(x)
  fig = plt.figure()
  ax = fig.add_subplot(projection="3d")
  ax.plot(x[:,0], x[:,1], y, "-o")
  ax.set_xlabel("x")
  ax.set_ylabel("y")
  ax.set_zlabel("z")
  plt.show()

class MLP_NN(nn.Module):
  def __init__(self):
    super().__init__()
    self.l1 = nn.Linear(in_features=1, out_features=64, bias=True) # 2 for Lorenz
    self.l2 = nn.Linear(64, 32)
    if do_dropout:
      self.l2a = nn.Dropout(p=dropout_prob_zero)
    if do_batch_normalization:
      self.l2a = nn.BatchNorm1d(32) # batch normalization (especially for convolutional networks and networks with sigmoidal nonlinearities) (allows dropout to be omitted)
    self.l3 = nn.Linear(32, 1)
    # self.leakyReLU = nn.LeakyReLU(0.01)
    # self.softmax = nn.Softmax(dim=1)

  def forward(self, x):
    x = torch.flatten(x, 1) # flatten all dimensions except batch (which is 1st dimension of the tensor)
    x = torch.relu(self.l1(x))
    # x = self.leakyReLU(self.l1(x))
    # x = self.softmax(self.l1(x))
    x = self.l2(x)
    if do_dropout:
      x = self.l2a(x)
    if do_batch_normalization:
      x = self.l2a(x) # we add the BN transform immediately before the nonlinearity
    x = torch.relu(x)
    # x = self.leakyReLU(x)
    # x = self.softmax(x)
    x = self.l3(x)
    return x

# verify gradient by using complex numbers trick
def verify_grad():
  # note: not all methods in torch 1.10 are supported for complex numbers
  try:
    m = 1
    epsilon = 1e-20
    # true mse = mean(abs(h_calc-h_true).^2); % where h is m x 1 matrix of complex numbers
    def loss_function(a, b):
      # if using only torch functions then fine (otherwise need a custom class)
      loss = torch.mean((a-b)**2)
      return loss
    # loss_function = nn.L1Loss()

    t = np.random.uniform(low=0, high=2*np.pi+1e-8, size=m*1)
    x, y = cycloid(t, rnd=rnd)
    x = cycloid_normalize_x(x)
    xc = np.zeros(m, dtype=np.csingle)
    xc.real = x
    yc = np.zeros(m, dtype=np.csingle)
    yc.real = y
    inputs, targets = torch.from_numpy(xc), torch.from_numpy(yc)
    inputs = torch.reshape(inputs, (m,1))
    targets = torch.reshape(targets, (m,1))

    l1 = nn.Linear(1, 64, dtype=torch.complex64)
    l1.weight.data.fill_(0.01)
    l1.bias.data.fill_(0.01)
    l2 = nn.Linear(64, 32, dtype=torch.complex64)
    l2.weight.data.fill_(0.01)
    l2.bias.data.fill_(0.01)
    l3 = nn.Linear(32, 1, dtype=torch.complex64)
    l3.weight.data.fill_(0.01)
    l3.bias.data.fill_(0.01)

    l1.weight.data[0] += 0 + 1j*epsilon

    out = torch.tanh(l1(inputs))
    out = torch.tanh(l2(out))
    out = l3(out)
    loss = loss_function(out, targets)

    deriv = loss.data.numpy().imag / epsilon

    loss.backward()
    auto_deriv = l1.weight.grad.data.numpy()[0].real[0]
    if np.abs(deriv - auto_deriv) > epsilon:
      warn(f"derivatives differ: delta = {np.abs(deriv - auto_deriv)}")
  except Exception as e:
    warn(f"exception: {e}")

def print_weight(model, x):
  if v:
    for i in range(3):
      for j in range(4):
        ax_w[i][j].cla()
        ax_w[i][j].grid()
    ax_w[0][0].hist(model.l1.weight.data.cpu().numpy().flatten()); ax_w[0][1].hist(model.l1.bias.data.cpu().numpy().flatten())
    ax_w[1][0].hist(model.l2.weight.data.cpu().numpy().flatten()); ax_w[1][1].hist(model.l2.bias.data.cpu().numpy().flatten())
    ax_w[2][0].hist(model.l3.weight.data.cpu().numpy().flatten()); ax_w[2][1].hist(model.l3.bias.data.cpu().numpy().flatten())
    x = model.l1(x)
    ax_w[0][2].hist(x.data.cpu().numpy().flatten())
    x = torch.relu(x)
    ax_w[0][3].hist(x.data.cpu().numpy().flatten())
    x = model.l2(x)
    ax_w[1][2].hist(x.data.cpu().numpy().flatten())
    x = torch.relu(x)
    ax_w[1][3].hist(x.data.cpu().numpy().flatten())
    x = model.l3(x)
    ax_w[2][2].hist(x.data.cpu().numpy().flatten())
    for i in range(3):
      if i == 0:   suffix = "st"
      elif i == 1: suffix = "nd"
      elif i == 2: suffix = "rd"
      else:        suffix = "th"
      ax_w[i][0].set_title(f"{i+1}{suffix} layer weights")
      ax_w[i][1].set_title(f"{i+1}{suffix} layer biases")
      ax_w[i][2].set_title(f"{i+1}{suffix} layer preactivation")
      ax_w[i][3].set_title(f"{i+1}{suffix} layer activation")

def print_grad(model):
  l1_weight_norm = torch.sqrt(torch.sum(model.l1.weight.grad.mul(model.l1.weight.grad))).data.cpu().numpy()
  l1_bias_norm   = torch.sqrt(torch.sum(model.l1.bias.grad.mul(model.l1.bias.grad))).data.cpu().numpy()
  l2_weight_norm = torch.sqrt(torch.sum(model.l2.weight.grad.mul(model.l2.weight.grad))).data.cpu().numpy()
  l2_bias_norm   = torch.sqrt(torch.sum(model.l2.bias.grad.mul(model.l2.bias.grad))).data.cpu().numpy()
  l3_weight_norm = torch.sqrt(torch.sum(model.l3.weight.grad.mul(model.l3.weight.grad))).data.cpu().numpy()
  l3_bias_norm   = torch.sqrt(torch.sum(model.l3.bias.grad.mul(model.l3.bias.grad))).data.cpu().numpy()
  print(f"l1 grad weight abs max = {model.l1.weight.grad.abs().max().float():0.3f}, 0 # = {torch.le(model.l1.weight.grad.abs(), 1e-10).sum().int():4d} ({int(100*torch.le(model.l1.weight.grad.abs(), 1e-10).sum().int() / (model.l1.weight.grad.shape[0]*model.l1.weight.grad.shape[1])):2d}%), l1.weight 2-norm = {l1_weight_norm:0.3f}")
  print(f"l1 grad bias   abs max = {model.l1.bias.grad.abs().max().float():0.3f}, 0 # = {torch.le(model.l1.bias.grad.abs(), 1e-10).sum().int():4d} ({int(100*torch.le(model.l1.bias.grad.abs(), 1e-10).sum().int() / model.l1.bias.grad.shape[0]):2d}%), l1.bias   2-norm = {l1_bias_norm:0.3f}")
  print(f"l2 grad weight abs max = {model.l2.weight.grad.abs().max().float():0.3f}, 0 # = {torch.le(model.l2.weight.grad.abs(), 1e-10).sum().int():4d} ({int(100*torch.le(model.l2.weight.grad.abs(), 1e-10).sum().int() / (model.l2.weight.grad.shape[0]*model.l2.weight.grad.shape[1])):2d}%), l2.weight 2-norm = {l2_weight_norm:0.3f}")
  print(f"l2 grad bias   abs max = {model.l2.bias.grad.abs().max().float():0.3f}, 0 # = {torch.le(model.l2.bias.grad.abs(), 1e-10).sum().int():4d} ({int(100*torch.le(model.l2.bias.grad.abs(), 1e-10).sum().int() / model.l2.bias.grad.shape[0]):2d}%), l2.bias   2-norm = {l2_bias_norm:0.3f}")
  print(f"l3 grad weight abs max = {model.l3.weight.grad.abs().max().float():0.3f}, 0 # = {torch.le(model.l3.weight.grad.abs(), 1e-10).sum().int():4d} ({int(100*torch.le(model.l3.weight.grad.abs(), 1e-10).sum().int() / (model.l3.weight.grad.shape[0]*model.l3.weight.grad.shape[1])):2d}%), l3.weight 2-norm = {l3_weight_norm:0.3f}")
  print(f"l3 grad bias   abs max = {model.l3.bias.grad.abs().max().float():0.3f}, 0 # = {torch.le(model.l3.bias.grad.abs(), 1e-10).sum().int():4d} ({int(100*torch.le(model.l3.bias.grad.abs(), 1e-10).sum().int() / model.l3.bias.grad.shape[0]):2d}%), l3.bias   2-norm = {l3_bias_norm:0.3f}")
  # gradient/parameter_value should ~= 1% over a minibatch
  a = torch.abs(model.l1.weight.grad / model.l1.weight).data.cpu().numpy().flatten()
  a1 = (a > 0.01).sum()
  a10 = (a > 0.1).sum()
  if a10 > 0:
    warn(f"L1 grad > 1%: {a1}, > 10%: {a10} out of {len(a)}")
  else:
    print(f"L1 grad > 1%: {a1}, > 10%: {a10} out of {len(a)}")
  a = torch.abs(model.l2.weight.grad / model.l2.weight).data.cpu().numpy().flatten()
  a1 = (a > 0.01).sum()
  a10 = (a > 0.1).sum()
  if a10 > 0:
    warn(f"L2 grad > 1%: {a1}, > 10%: {a10} out of {len(a)}")
  else:
    print(f"L2 grad > 1%: {a1}, > 10%: {a10} out of {len(a)}")
  a = torch.abs(model.l3.weight.grad / model.l3.weight).data.cpu().numpy().flatten()
  a1 = (a > 0.01).sum()
  a10 = (a > 0.1).sum()
  if a10 > 0:
    warn(f"L3 grad > 1%: {a1}, > 10%: {a10} out of {len(a)}")
  else:
    print(f"L3 grad > 1%: {a1}, > 10%: {a10} out of {len(a)}")
  if v:
    # A test that can rule out local minima as the problem is plotting the norm of the gradient over time.
    global i_grad, l1_weight_norm_prev, l1_bias_norm_prev, l2_weight_norm_prev, l2_bias_norm_prev, l3_weight_norm_prev, l3_bias_norm_prev
    if i_grad == 0:
      ax_grad.plot(i_grad, l1_weight_norm, color="blue",  linestyle='-', label="l1 weight norm") # marker='*'
      ax_grad.plot(i_grad, l1_bias_norm,   color=lblue,   linestyle='-', label="l1 bias norm")
      ax_grad.plot(i_grad, l2_weight_norm, color="green", linestyle='-', label="l2 weight norm")
      ax_grad.plot(i_grad, l2_bias_norm,   color=lgreen,  linestyle='-', label="l2 bias norm")
      ax_grad.plot(i_grad, l3_weight_norm, color="red",   linestyle='-', label="l3 weight norm")
      ax_grad.plot(i_grad, l3_bias_norm,   color=lred,    linestyle='-', label="l3 bias norm")
      ax_grad.legend(loc="best", ncol=1, scatterpoints=1, numpoints=1)
    else:
      ax_grad.plot([i_grad-1, i_grad], [l1_weight_norm_prev, l1_weight_norm], color="blue",  linestyle='-', label="l1 weight norm") # marker='*'
      ax_grad.plot([i_grad-1, i_grad], [l1_bias_norm_prev,   l1_bias_norm],   color=lblue,   linestyle='-', label="l1 bias norm")
      ax_grad.plot([i_grad-1, i_grad], [l2_weight_norm_prev, l2_weight_norm], color="green", linestyle='-', label="l2 weight norm")
      ax_grad.plot([i_grad-1, i_grad], [l2_bias_norm_prev,   l2_bias_norm],   color=lgreen,  linestyle='-', label="l2 bias norm")
      ax_grad.plot([i_grad-1, i_grad], [l3_weight_norm_prev, l3_weight_norm], color="red",   linestyle='-', label="l3 weight norm")
      ax_grad.plot([i_grad-1, i_grad], [l3_bias_norm_prev,   l3_bias_norm],   color=lred,    linestyle='-', label="l3 bias norm")
    l1_weight_norm_prev = l1_weight_norm
    l1_bias_norm_prev   = l1_bias_norm
    l2_weight_norm_prev = l2_weight_norm
    l2_bias_norm_prev   = l2_bias_norm
    l3_weight_norm_prev = l3_weight_norm
    l3_bias_norm_prev   = l3_bias_norm
    i_grad += 1
    fig_grad.canvas.flush_events()

def check_hessian(model, loss_function):
  if do_batch_normalization:
    m = 20
  else:
    m = 1
  t = np.random.uniform(low=0, high=2*np.pi+1e-8, size=m)
  t = t.astype(np.float32)
  x, y = cycloid(t, rnd=rnd)
  x = cycloid_normalize_x(x)

  # t_num = m
  # delta_t = 0.05
  # t_start = 100*np.random.rand()
  # t_end = t_start + t_num*delta_t
  # t = np.linspace(t_start, t_end, t_num)
  # t = t.astype(np.float32)
  # x, y = Lorenz(t, rnd=rnd)
  # x = Lorenz_normalize_x(x)
  # x = x.astype(np.float32)
  # y = y.astype(np.float32)

  inputs, targets = torch.from_numpy(x), torch.from_numpy(y)
  inputs = torch.reshape(inputs, (m,1)) # (m,2) for Lorenz
  targets = torch.reshape(targets, (m,1))
  inputs, targets = inputs.to(device), targets.to(device)
  outputs = model(inputs)
  h = torch.autograd.functional.hessian(loss_function, (outputs, targets))
  print(f"Hessian = {h}")
  hm = np.array([[h[0][0].data.cpu().numpy()[0][0][0][0], h[0][1].data.cpu().numpy()[0][0][0][0]], # [0][0][0][0] is for 1 sample
                 [h[1][0].data.cpu().numpy()[0][0][0][0], h[1][1].data.cpu().numpy()[0][0][0][0]]])
  w, _ = scipy.linalg.eig(hm)
  if (w[0].real < -1e-8 and w[1].real > 1e-8) or (w[0].real > 1e-8 and w[1].real < -1e-8):
    warn(f"At a saddle point, the Hessian matrix has both positive and negative eigenvalues: {w}")

def MLP_train(model, loss_function, optimizer):
  start = time.time()
  running_loss_prev = 0.0
  i_loss_valid_prev = 0
  if do_early_stopping:
    validation_loss_prev = 1e8
    early_stop_trigger_times = 0

  def calc():
    nonlocal loss

    outputs = model(inputs)
    loss = loss_function(outputs, targets)

    if do_regularization:
      L2_reg = torch.tensor(0.0).to(device)
      for param in model.parameters():
        L2_reg += torch.pow(param,2).sum()/2
      loss += L2_lambda * L2_reg

    optimizer.zero_grad()
    loss.backward()
    return loss

  for epoch in range(1,max_epoch+1):
    hours, minutes, seconds = sec2hms((max_epoch - epoch) * (time.time() - start) / epoch)
    print(f"starting epoch {epoch}; ETA = {hours:02d}:{minutes:02d}:{seconds:02d}")
    running_loss = 0.0
    train_data_size = train_data_batches*batch_size
    print(f"train_data_size = {train_data_size:,d}")
    for i in range(train_data_size//batch_size):
      t = np.random.uniform(low=0, high=2*np.pi+1e-8, size=batch_size*1)
      t = t.astype(np.float32)
      x, y = cycloid(t, rnd=rnd)
      x = cycloid_normalize_x(x)

      # t_num = batch_size
      # delta_t = 0.05
      # t_start = 100*np.random.rand()
      # t_end = t_start + t_num*delta_t
      # t = np.linspace(t_start, t_end, t_num)
      # t = t.astype(np.float32)
      # x, y = Lorenz(t, rnd=rnd)
      # x = Lorenz_normalize_x(x)
      # x = x.astype(np.float32)
      # y = y.astype(np.float32)

      inputs, targets = torch.from_numpy(x), torch.from_numpy(y)
      inputs = torch.reshape(inputs, (batch_size,1)) # 2 for Lorenz
      targets = torch.reshape(targets, (batch_size,1))
      inputs, targets = inputs.to(device), targets.to(device)

      if optimizer.__repr__().startswith("LBFGS"):
        optimizer.step(calc)
      else:
        loss = calc()
        optimizer.step()

      running_loss += loss.item()
      if i % info_step == info_step-1:
        running_loss = running_loss / info_step / batch_size
        print(f"loss after mini-batch {i+1:5d}: {running_loss:.05f}")
        if v:
          global i_loss
          if i_loss == 0:
            ax_loss.plot(i_loss, running_loss, color="blue", linestyle='-', label="training loss")
          else:
            ax_loss.plot([i_loss-1, i_loss], [running_loss_prev, running_loss], color="blue", linestyle='-') # , label="training loss")
          running_loss_prev = running_loss
          i_loss += 1
          fig_loss.canvas.flush_events()
        running_loss = 0.0
        print_grad(model)
        print_weight(model, inputs)
    if v:
      ax_loss.annotate(f"epoch={epoch}", xy=(i_loss, running_loss_prev), rotation=60)

    if do_early_stopping or v:
      validation_loss, _, _, _, _, _ = MLP_validate(model, loss_function)
      if v:
        if i_loss_valid_prev == 0:
          ax_loss.plot(i_loss, validation_loss, color="red", linestyle='-', label="validation loss")
          ax_loss.legend(loc="best", ncol=1, scatterpoints=1, numpoints=1)
        else:
          ax_loss.plot([i_loss_valid_prev, i_loss], [validation_loss_prev, validation_loss], color="red", linestyle='-', label="validation loss")
        i_loss_valid_prev = i_loss
      if do_early_stopping:
        if validation_loss > validation_loss_prev:
          # or save model to file when < validation_loss_best
          early_stop_trigger_times += 1
          if early_stop_trigger_times >= early_stop_patience:
            print(f"early stopping triggered at epoch={epoch}")
            break
        else:
          early_stop_trigger_times = 0
      validation_loss_prev = validation_loss

  print("finished training")
  check_hessian(model, loss_function)

def report():
  pass

def MLP_validate(model, loss_function):
  t = np.arange(0, 2*np.pi+0.1, 0.1)
  t = t.astype(np.float32)
  xs, ys = cycloid(t, rnd=rnd)
  xs = cycloid_normalize_x(xs)

  # t_num = 300
  # delta_t = 0.05
  # t_start = 100*np.random.rand()
  # t_end = t_start + t_num*delta_t
  # t = np.linspace(t_start, t_end, t_num)
  # t = t.astype(np.float32)
  # x0=np.random.uniform(low=-10, high=10+1e-8, size=3)
  # x, y = Lorenz(t, rnd=rnd, x0=x0)
  # x = Lorenz_normalize_x(x)
  # xs = x.astype(np.float32)
  # ys = y.astype(np.float32)

  length = t.shape[0]
  print(f"t_validate length = {length}")
  with torch.no_grad():
    inputs, targets = torch.from_numpy(xs), torch.from_numpy(ys)
    inputs = torch.reshape(inputs, (length,1)) # 2 for Lorenz
    targets = torch.reshape(targets, (length,1))
    inputs, targets = inputs.to(device), targets.to(device)
    outputs = model(inputs)
    sample_loss = loss_function(outputs, targets)
    sample_loss /= length
    print(f"validation samples loss = {sample_loss:.05f}")
    zs = outputs.data.cpu().numpy()
    # zs = zs.flatten() # for Lorenz, no need for cycloid
    x0 = None
  return sample_loss.data.cpu().numpy().flatten()[0], t, xs, ys, zs, x0

def MLP_verify(model, loss_function):
  sample_loss, t, xs, ys, zs, x0 = MLP_validate(model, loss_function)
  _, ss = cycloid(t, rnd=0)
  # _, ss = Lorenz(t, rnd=0, x0=x0)
  with torch.no_grad():
    zst, sst = torch.from_numpy(zs), torch.from_numpy(ss)
    zst = torch.reshape(zst, (zs.shape[0],1))
    sst = torch.reshape(sst, (ss.shape[0],1))
    true_loss = loss_function(zst, sst)
    # true_loss = true_loss.item() / ss.shape[0]
  print(f"verification true  loss = {true_loss:.05f}")
  if v:
    ax_ver.plot(xs, ys, "b-*", label="samples")
    ax_ver.plot(xs, zs, "r-o", label="predicted")
    ax_ver.plot(xs, ss, "g-x", label="truth")
    # ax_ver.plot(xs[:,0], xs[:,1], ys, "b-*", label="samples")
    # ax_ver.plot(xs[:,0], xs[:,1], zs, "r-o", label="predicted")
    # ax_ver.plot(xs[:,0], xs[:,1], ss, "g-x", label="truth")
    ax_ver.legend()
  report()
  return sample_loss, true_loss

def MLP():
  # tune the learning rate
  # lrs = [0.0001, 0.0005, 0.0010, 0.0025, 0.0050, 0.0075, 0.0100, 0.0500] # SGD; lr=0.001
  # lrs = [0.0001, 0.0002, 0.0006, 0.0006, 0.0008, 0.0010, 0.0015, 0.0020, 0.0025, 0.0050, 0.0075, 0.0100, 0.0500] # Adam; lr=0.001
  lrs = [0.001]
  lr_prev = lrs[0]
  sample_loss_prev, true_loss_prev = None, None

  model = MLP_NN().to(device)
  if v: print(model); # assert(all(p.is_cuda for p in model.parameters()))
  loss_function = nn.MSELoss()
  # loss_function = nn.L1Loss()
  verify_grad()

  for lr in lrs:
    print(f"learning rate = {lr}")
    if v:
      fig_grad_init()
      fig_w_init()
      fig_loss_init(lr)

    # optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9) # lr=0.001, momentum=0.9 # weight_decay is L2 regularization
    optimizer = optim.Adam(params=model.parameters(), lr=lr)
    # optimizer = optim.LBFGS(params=model.parameters(), lr=lr, line_search_fn="strong_wolfe")
    MLP_train(model, loss_function, optimizer)
    if v: fig_ver_init()
    sample_loss, true_loss = MLP_verify(model, loss_function)

    if v:
      if not sample_loss_prev: sample_loss_prev = sample_loss
      if not true_loss_prev: true_loss_prev = true_loss
      ax_lr.plot([lr_prev, lr], [sample_loss_prev, sample_loss], color="red",  linestyle='-', label="validation sample loss")
      ax_lr.plot([lr_prev, lr], [true_loss_prev, true_loss], color="green",  linestyle='-', label="validation truth loss")
      if lr == lrs[0]: ax_lr.legend(loc="best", ncol=1, scatterpoints=1, numpoints=1)
      lr_prev = lr
      sample_loss_prev = sample_loss
      true_loss_prev = true_loss
      fig_lr.canvas.flush_events()
    print()

MLP()

postscript(start_time)
