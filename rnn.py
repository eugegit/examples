"""
Recurrent Neural Network.
Long Short-Term Memory RNN.

"Eugene Morozov"<Eugene ~at~ HiEugene.com>
"""

import matplotlib.pyplot as plt
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from util import *

v = 1
rand_seed = int(time.time())
start_time, device = preamble(rand_seed)

rnd = 1.0
epochs = 25
num_layers = 2
hidden_size = 8
batch_size = 200
info_step = 500
dropout = 0.4

# def get_data(batch_size, start=0.0, n_batches=500):
#   step = 0.1
#   for i in range(n_batches):
#     t = np.array([start+i*step for i in range(batch_size+1)]) # np.arange() is buggy
#     start += batch_size*step
#     data = np.sin(t)
#     data = data.astype(np.float32)
#     z = np.copy(data[1:])
#     if rnd > 0:
#       noise = np.random.uniform(low=-rnd, high=rnd+1e-8, size=len(data)).astype(np.float32)
#       data += noise
#     x = np.copy(data[:-1])
#     y = np.copy(data[1:])
#     yield x, y, z, start

fname = "K:\\proj\\TestP\\data\\household_power_consumption.txt"
skip_columns = ["Date", "Time", "Global_reactive_power", "Voltage", "Global_intensity", "Sub_metering_1", "Sub_metering_2", "Sub_metering_3"]
df = pd.read_csv(fname, sep=';', low_memory=False)
df.dropna(axis=0, how="any", thresh=None, subset=None, inplace=True) # axis: 0 means row, 1 means column
columns = list(df.columns.values)
for col in skip_columns:
  print(f"skipping: {col}:\n{df[col].describe()}\nmin={df[col].min()}, max={df[col].max()}\n")
  columns.remove(col)
print("columns: ", columns)
n = len(columns)
m = int(0.9 * df[columns[0]].size // batch_size * batch_size)
data_train = np.zeros([m+1, n])
m2 = int((df[columns[0]].size - m) // batch_size * batch_size)
if df[columns[0]].size <= m+m2:
  m2 -= batch_size
data_verify = np.zeros([m2+1, n])
for j in range(n):
  data_train[:, j] = df[columns[j]].values[0:m+1].astype(np.float64)
  data_verify[:, j] = df[columns[j]].values[m:m+m2+1].astype(np.float64)
del(df)
if v:
  plt.ion()
  fig = plt.figure()
  for j in range(n):
    ax = plt.subplot(n, 1, j+1) # (nrows, ncols, index)
    ax.plot(data_train[:, j], label=columns[j], c=get_color(j))
    ax.legend(loc="upper right")
    ax.grid(True)
  plt.show()

def get_train_data(batch_size, start_ind=0):
  m = data_train.shape[0] - 1
  for i in range(start_ind, m, batch_size):
    data = data_train[i:i+batch_size+1]
    data = data.astype(np.float32)
    x = np.copy(data[:-1])
    y = np.copy(data[1:])
    start_ind += batch_size
    yield x, y, start_ind

def get_verify_data(batch_size, start_ind=0):
  m = data_verify.shape[0] - 1
  for i in range(start_ind, m, batch_size):
    data = data_verify[i:i+batch_size+1]
    data = data.astype(np.float32)
    x = np.copy(data[:-1])
    y = np.copy(data[1:])
    start_ind += batch_size
    yield x, y, start_ind

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

class MyLSTM(nn.Module):
  def __init__(self, input_size, hidden_size, num_layers, output_size, dropout):
    super(MyLSTM, self).__init__()
    self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                       bias=True, batch_first=False, dropout=dropout, bidirectional=False)
    self.fc1 = nn.Linear(hidden_size, output_size)

  def forward(self, data, last_hidden, last_cell):
    x, (hidden, cell) = self.rnn(data, (last_hidden, last_cell))
    x = self.fc1(x)
    return x, (hidden, cell)

# model = MyRNN(input_size=1, hidden_size=hidden_size, num_layers=num_layers, output_size=1, dropout=dropout).to(device)
model = MyLSTM(input_size=1, hidden_size=hidden_size, num_layers=num_layers, output_size=1, dropout=dropout).to(device)
print(f"model: {model}")
print(f"rnn weights: {model.rnn._all_weights}")
print(f"linear weights: {model.fc1.weight}")
print([p.shape for p in model.parameters()])
loss_function = nn.MSELoss()
optimizer = optim.Adam(params=model.parameters(), lr=0.01)

start = time.time()
start_i = 0
scheduler = StepLR(optimizer, step_size=3, gamma=0.5, verbose=True)
for epoch in range(1,epochs+1):
  hours, minutes, seconds = sec2hms((epochs - epoch) * (time.time() - start) / epoch)
  print(f"starting epoch {epoch}; ETA = {hours:02d}:{minutes:02d}:{seconds:02d}")
  running_loss = 0.0
  hidden = torch.randn(num_layers, batch_size, hidden_size).to(device)
  cell   = torch.randn(num_layers, batch_size, hidden_size).to(device)
  # for i, (x, y, _, start_i) in enumerate(get_data(batch_size)):
  for i, (x, y, start_i) in enumerate(get_train_data(batch_size)):
    inputs, targets = torch.from_numpy(x), torch.from_numpy(y)
    inputs = torch.reshape(inputs, (1,batch_size,1))
    targets = torch.reshape(targets, (1,batch_size,1))
    inputs, targets = inputs.to(device), targets.to(device)

    # outputs, hidden = model(inputs, hidden)
    outputs, (hidden, cell) = model(inputs, hidden, cell)
    loss = loss_function(outputs, targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    hidden.detach_()
    cell.detach_()

    running_loss += loss.item()
    if i % info_step == info_step-1:
      running_loss = running_loss / info_step / batch_size
      print(f"loss after mini-batch {i:5d}: {running_loss:.05f}; {start_i:,d} of {m:,d}")
      running_loss = 0.0
  scheduler.step()
print("finished training")

# batch_size = 1
samples  = []
y_trues  = []
predicts = []
with torch.no_grad():
  # hidden = hidden[:,-1,:].reshape(num_layers,batch_size,hidden_size).contiguous() # last in batch
  # for i, (x, y, y_true, _) in enumerate(get_data(batch_size, start=start_i, n_batches=25)): # continuity in time and "hidden" state
  for i, (x, y, _) in enumerate(get_verify_data(batch_size)): # continuity in time and "hidden" state
    inputs = torch.from_numpy(x)
    inputs = torch.reshape(inputs, (1,batch_size,1))
    inputs = inputs.to(device)
    # outputs, hidden = model(inputs, hidden)
    outputs, (hidden, cell) = model(inputs, hidden, cell)
    samples.extend(y.flatten())
    # y_trues.extend(y_true)
    predicts.extend(outputs.data.cpu().numpy().flatten())
plt.ion()
fig = plt.figure()
ax = fig.gca()
fig.show(); plt.tight_layout()
ax.grid()
# ax.set_title(f"sin; rnd={rnd}")
ax.set_title(f"{columns[0]}")
ax.plot(samples, "b-*", label="samples")
ax.plot(predicts, "r-o", label="predicted")
# ax.plot(y_trues, "g-x", label="truth")
ax.legend()

postscript(start_time)
