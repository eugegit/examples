"""
Recurrent Neural Network.

"Eugene Morozov"<Eugene ~at~ HiEugene.com>
"""

import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from util import *

v = 1
rand_seed = int(time.time())
start_time, device = preamble(rand_seed)

rnd = 1.0
epochs = 100
num_layers = 4
hidden_size = 16
batch_size = 40
info_step = 20
dropout = 0.2

def get_data(batch_size, start=0.0, n_batches=500):
  step = 0.1
  for i in range(n_batches):
    t = np.array([start+i*step for i in range(batch_size+1)]) # np.arange() is buggy
    start += batch_size*step
    data = np.sin(t)
    data = data.astype(np.float32)
    z = np.copy(data[1:])
    if rnd > 0:
      noise = np.random.uniform(low=-rnd, high=rnd+1e-8, size=len(data)).astype(np.float32)
      data += noise
    x = np.copy(data[:-1])
    y = np.copy(data[1:])
    yield x, y, z, start

class MyRNN(nn.Module):
  def __init__(self, input_size, hidden_size, num_layers, output_size, dropout):
    super(MyRNN, self).__init__()
    self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, nonlinearity="tanh",
                      bias=True, batch_first=False, dropout=dropout, bidirectional=False)
    self.fc1 = nn.Linear(hidden_size, output_size)

  def forward(self, data, last_hidden):
    x, hidden = self.rnn(data, last_hidden)
    x = self.fc1(x)
    return x, hidden

model = MyRNN(input_size=1, hidden_size=hidden_size, num_layers=num_layers, output_size=1, dropout=dropout).to(device)
print(f"model: {model}")
print(f"rnn weights: {model.rnn._all_weights}")
print(f"linear weights: {model.fc1.weight}")
print([p.shape for p in model.parameters()])
loss_function = nn.MSELoss()
optimizer = optim.Adam(params=model.parameters(), lr=0.01)

start = time.time()
start_i = 0
scheduler = StepLR(optimizer, step_size=10, gamma=0.7, verbose=True)
for epoch in range(1,epochs+1):
  hours, minutes, seconds = sec2hms((epochs - epoch) * (time.time() - start) / epoch)
  print(f"starting epoch {epoch}; ETA = {hours:02d}:{minutes:02d}:{seconds:02d}")
  running_loss = 0.0
  hidden = torch.randn(num_layers, batch_size, hidden_size).to(device)
  for i, (x, y, _, start_i) in enumerate(get_data(batch_size)):
    inputs, targets = torch.from_numpy(x), torch.from_numpy(y)
    inputs = torch.reshape(inputs, (1,batch_size,1))
    targets = torch.reshape(targets, (1,batch_size,1))
    inputs, targets = inputs.to(device), targets.to(device)

    outputs, hidden = model(inputs, hidden)
    loss = loss_function(outputs, targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    hidden.detach_()

    running_loss += loss.item()
    if i % info_step == info_step-1:
      running_loss = running_loss / info_step / batch_size
      print(f"loss after mini-batch {i:5d}: {running_loss:.05f}")
      running_loss = 0.0
  scheduler.step()
print("finished training")

# batch_size = 1
samples  = []
y_trues  = []
predicts = []
with torch.no_grad():
  # hidden = hidden[:,-1,:].reshape(num_layers,batch_size,hidden_size).contiguous() # last in batch
  for i, (x, y, y_true, _) in enumerate(get_data(batch_size, start=start_i, n_batches=25)): # continuity in time and "hidden" state
    inputs = torch.from_numpy(x)
    inputs = torch.reshape(inputs, (1,batch_size,1))
    inputs = inputs.to(device)
    outputs, hidden = model(inputs, hidden)
    samples.extend(y)
    y_trues.extend(y_true)
    predicts.extend(outputs.data.cpu().numpy().flatten())
plt.ion()
fig = plt.figure()
ax = fig.gca()
fig.show(); plt.tight_layout()
ax.grid()
ax.set_title(f"sin; rnd={rnd}")
ax.plot(samples, "b-*", label="samples")
ax.plot(predicts, "r-o", label="predicted")
ax.plot(y_trues, "g-x", label="truth")
ax.legend()

postscript(start_time)
