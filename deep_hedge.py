import numpy as np
import matplotlib.pyplot as plt
import sys
# sys.path.append("")
from pfhedge.instruments import (BrownianStock, HestonStock, EuropeanOption)
from pfhedge.nn import (Hedger, MultiLayerPerceptron, BlackScholes, WhalleyWilmott)
from util import *

def print_as_comment(obj):
  print("\n".join(f"# {line}" for line in str(obj).splitlines()))

v = 1
epoch_sec = 1 # int(time.time())
start_time, device = preamble(epoch_sec)
flag = 1

S0 = 50.0
theta0 = 0.04
K = 50.0

# prepare instruments
# stock = BrownianStock(cost=1e-4, device=device)
stock = HestonStock(cost=1e-4, dt=0.001, device=device)
derivative = EuropeanOption(stock, strike=K)
print(">>> stock")
print_as_comment(stock)
print(">>> derivative")
print_as_comment(derivative)

# Black-Scholes
model = BlackScholes(derivative)
hedger_bs = Hedger(model, model.inputs()) # default: HedgeLoss = EntropicRiskMeasure(); can be: ExpectedShortfall(HedgeLoss) or OCE
print(">>> hedger_bs")
print_as_comment(hedger_bs)
flag = "number of shares (option's delta); Black-Scholes model"
price_bs = hedger_bs.price(derivative, init_state=(S0, theta0))
print(">>> price_bs")
print_as_comment(price_bs)

if v:
  spot = stock.spot.detach().cpu().numpy()
  var = stock.variance.detach().cpu().numpy()
  vol = stock.volatility.detach().cpu().numpy()

  plt.ion()
  fig_spot = plt.figure()
  ax_spot = fig_spot.gca()
  ax_spot.grid()
  ax_spot.set_title("3 Heston stocks")
  ax_spot.plot(spot[0:3,:].T)

  fig_vol = plt.figure()
  ax_vol = fig_vol.gca()
  ax_vol.grid()
  ax_vol.set_title("their volatility")
  ax_vol.plot(vol[0:3,:].T)
  plt.show()

# Whalley-Wilmott
model = WhalleyWilmott(derivative)
hedger_ww = Hedger(model, model.inputs())
print(">>> hedger_ww")
print_as_comment(hedger_ww)
flag = "number of shares (option's delta); Whalley-Wilmott model"
price_ww = hedger_ww.price(derivative, init_state=(S0, theta0))
print(">>> price_ww")
print_as_comment(price_ww)

# fit and price with deep hedger
model = MultiLayerPerceptron().to(device)
hedger = Hedger(model, ["log_moneyness", "expiry_time", "volatility", "prev_hedge"])
print(">>> hedger")
print_as_comment(hedger)
flag = 0
hedger.fit(derivative, n_epochs=400, n_paths=1000, n_times=1, init_state=(S0, theta0))
flag = "number of shares (option's delta); MLP model"
price = hedger.price(derivative, init_state=(S0, theta0))
print(">>> price")
print_as_comment(price)

postscript(start_time)
