import numpy as np
import random
import gym
from stable_baselines3 import SAC
from stable_baselines3.common.logger import configure
from stable_baselines3.common import results_plotter
from stable_baselines3.common.results_plotter import ts2xy, load_results
from stable_baselines3.common.monitor import Monitor
from gym import spaces
import matplotlib.pyplot as plt

v = 1
log_dir = "log"
logger = configure(log_dir, ["stdout", "csv", "tensorboard"])

# define a highly customizable market making environment (simulator)
class MarketMakingEnv(gym.Env):
  def __init__(self, T=100, dt=1, gamma=0.01,
               baselineintensities = [0.55, 0.55, 2.6, 2.6, 1.5, 1.5, 1, 1],
               excitationmatrix = [[1.5, 0.4, 3.0, 0.5, 0.0, 2.0, 1.0, 0.0],
                                   [0.4, 1.5, 0.5, 3.0, 2.0, 0.0, 0.0, 1.0],
                                   [0.4, 1.2, 0.2, 0.0, 1.5, 0.5, 0.4, 0.4],
                                   [1.2, 0.4, 0.0, 0.2, 0.5, 1.5, 0.4, 0.4],
                                   [0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                   [0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0],
                                   [1.0, 0.0, 2.0, 0.0, 0.0, 1.0, 3.0, 0.0],
                                   [0.0, 1.0, 0.0, 2.0, 1.0, 0.0, 0.0, 3.0]],
               inventoryconstraint = 3, delta = 0.01, spread = 0.1, askprice = 100.05,
               bidprice = 99.95, s = 100.00, w1 = 0.27, w2=0.25, expdispar = 0.08, mtfee = 0.002, mmfee = 0, decayfactor=10, n1 = 0.45145, n2 = 0.495021633365654, n3 = 0, n4 = 0.5958326386388073, Z = 0.25):
    """
    T - terminal time, dt - timestep, gamma - risk aversion parameter, baseline intensities, excitation matrix, inventory constraint,
    delta - tick size, spread - initial spread, askprice - initial ask-price, bidprice - initial bidprice, s - initial price
    w1 - probability of market maker's market order being aggressive, w2 - probability of market maker's limit order cancellation being aggressive
    expdispar - scale of the exponential distribution used for modeling jumps, mtfee - market taker fees, mmfee - market maker fees
    decay - beta decay factor (assumed to be fixed), n1 - parameter used for normalization of the spread state space variable
    n2 - parameter used for normalization of the spread state space variable, n3 - parameter used for normalization of the trend state space variable
    Z - the probability of execution of the market maker’s outstanding limit order standing at the best bid/ask price
    """
    super(MarketMakingEnv, self).__init__()
    self.initial_parameters = (spread, askprice, bidprice, s, baselineintensities)
    self.t = 0 # initial time
    self.s = s # initial midprice value
    self.olds = s # initial old midprice value
    self.T = T # trading period length (terminal time)
    self.dt = dt # time step length
    self.q = 0 # initial inventory level
    self.oldq = 0 # initial old inventory level
    self.x = 0 # initial cash
    self.oldx = 0 # initial old cash
    self.w1 = w1 # probability of market maker's market order being aggressive
    self.w2 = w2 # probability of market maker's limit order cancellation being aggressive
    self.n1 = n1 # normalization of the spread state space variable
    self.n2 = n2 # normalization of the spread state space variable
    self.n3 = n3 # normalization of the trend state space variable
    self.n4 = n4 # normalization of the trend state space variable
    self.mtfee = mtfee
    self.mmfee = mmfee
    self.Z = Z
    self.expdispar = expdispar # exponential distribution parameter
    self.gamma = gamma # risk aversion parameter (part of penalty in the reward function)
    self.decayfactor = decayfactor
    self.excitationmatrix = excitationmatrix
    self.intensities = np.array(baselineintensities)
    self.defaultintensities = self.intensities # baseline intensities
    self.done = False # done indicator (indicates whether the episode has finished)
    self.inventoryconstraint = inventoryconstraint # inventory constraint
    self.delta = delta # tick size
    self.spread = spread # initial spread
    self.limitscounter = 0 # counts the number of executions of the market maker's limit orders in an episode
    self.marketscounter = 0 # counts the number of executions of the market maker's market orders in an episode
    self.askprice = askprice # initial ask price
    self.bidprice = bidprice # initial bid price
    self.ask_order = np.inf # initial ask order
    self.bid_order = -np.inf # initial bid order
    self.action_space = spaces.Box(np.array([-1, -1]), np.array([1,1]),dtype=np.float32) # action space
    self.observation_space = spaces.Box(np.array([-1, -5, -5]), np.array([1, 5, 5]),dtype=np.float32) # observation space

  def step(self, action):
    """Part 1: Canceling the existing (old) market maker's limit orders, if there are any. If such canceling
    is aggressive, intensities are updated as well as the bid-, ask-, mid-price and the spread."""
    if (self.ask_order == self.askprice) and random.random() < self.w2: # aggressive limit sell cancellation
      jumpsize = np.round(np.random.exponential(self.expdispar)+0.01, 2) # shifted exponential distribution
      self.askprice = np.round(self.askprice + jumpsize, 2) # new ask-price
      self.s = np.round((self.askprice+self.bidprice)/2, 3) # new mid-price
      self.spread = np.round(self.askprice-self.bidprice, 2) # new spread
      self.intensities += np.array(self.excitationmatrix[5]) # new intensities
    if (self.bid_order == self.bidprice) and random.random() < self.w2: # aggressive limit buy cancellation
      jumpsize = np.round(np.random.exponential(self.expdispar)+0.01, 2) # shifted exponential distribution
      self.bidprice = np.round(self.bidprice - jumpsize, 2) # new ask-price
      self.s = np.round((self.askprice+self.bidprice)/2, 3) # new ask-price
      self.spread = np.round(self.askprice-self.bidprice, 2) # new spread
      self.intensities += np.array(self.excitationmatrix[4]) # new intensities
    self.ask_order, self.bid_order = np.inf, -np.inf # cancel all existing orders
    self.oldx, self.olds, self.oldq = self.x, self.s, self.q # save old cash, price and inventory
    self.reward = 0 # initialize reward
    assert(self.askprice > self.bidprice) # sanity checking

    """Part 2: Setting new limit orders and checking inventory constraints"""
    self.ask_order, self.bid_order = np.round(self.askprice + action[0], 2), np.round(self.bidprice - action[1], 2) # setting new limit orders
    if self.q <= -self.inventoryconstraint: # if the side is saturated ignore the ask order
      self.ask_order = np.inf
    if self.q >= self.inventoryconstraint:  # if the side is saturated ignore the bid order
      self.bid_order = -np.inf
    if self.ask_order <= self.bid_order: # ignore nonsensical limit orders
      self.ask_order, self.bid_order = np.inf, -np.inf

    """Part 3: Effects of setting the ask limit order."""
    if self.ask_order <= self.bidprice: # treating this case as a market sell
      if random.random() < 1-self.w1: # non-aggressive market sell
        if self.q > -self.inventoryconstraint: # checking the inventory constraint
          self.q -= 1
          self.marketscounter += 1
          self.x += (1-self.mtfee)*self.bidprice
          self.intensities += np.array(self.excitationmatrix[7])
        self.ask_order = np.inf # canceling the ask order
        self.bid_order = -np.inf # canceling the bid order
      else: # aggressive market sell
        if self.q > -self.inventoryconstraint: # checking the inventory constraint
          jumpsize = np.round(np.random.exponential(self.expdispar)+0.01, 2)
          self.q -= 1
          self.marketscounter += 1
          self.x += (1-self.mtfee)*(self.bidprice-jumpsize/2) # simplifying assumption
          self.intensities += np.array(self.excitationmatrix[1])
        self.ask_order = np.inf # canceling the ask order
        self.bid_order = -np.inf # canceling the bid order
        self.bidprice = np.round(self.bidprice - jumpsize, 2) # new bid-price
        self.s = np.round((self.askprice+self.bidprice)/2, 3) # new mid-price
        self.spread = np.round(self.askprice-self.bidprice, 2) # new spread
    elif self.ask_order < self.askprice: # aggressive ask limit order
      self.askprice = np.round(self.ask_order, 2) # new ask-price
      self.s = np.round((self.askprice+self.bidprice)/2, 3) # new mid-price
      self.spread = np.round(self.askprice-self.bidprice, 2) # new spread
      self.intensities += np.array(self.excitationmatrix[3])
    else:
      pass

    """Part 4: Effects of setting the bid limit order."""
    if self.bid_order >= self.askprice: # treating this case as a market buy
      if random.random() < 1-self.w1: # non-aggressive market buy
        if self.q < self.inventoryconstraint: # checking the inventory constraint
          self.q += 1
          self.marketscounter += 1
          self.x -= (1+self.mtfee)*self.askprice
          self.intensities += np.array(self.excitationmatrix[6])
        self.ask_order = np.inf # canceling the ask order
        self.bid_order = -np.inf # canceling the bid order
      else: # aggressive market buy
        if self.q < self.inventoryconstraint: # checking the inventory constraint
          jumpsize = np.round(np.random.exponential(self.expdispar)+0.01, 2)
          self.q += 1
          self.marketscounter += 1
          self.x -= (1+self.mtfee)*(self.askprice+jumpsize/2) # simplifying assumption
          self.intensities += np.array(self.excitationmatrix[0])
        self.ask_order = np.inf # canceling the ask order
        self.bid_order = -np.inf # canceling the bid order
        self.askprice = np.round(self.askprice + jumpsize, 2) # new ask-price
        self.s = np.round((self.askprice+self.bidprice)/2, 3) # new mid-price
        self.spread = np.round(self.askprice-self.bidprice, 2) # new spread
    elif self.bid_order > self.bidprice: # aggressive bid limit order
      self.bidprice = np.round(self.bid_order, 2) # new bid-price
      self.s = np.round((self.askprice+self.bidprice)/2, 3) # new mid-price
      self.spread = np.round(self.askprice-self.bidprice, 2) # new spread
      self.intensities += np.array(self.excitationmatrix[2])
    else:
      pass

    self.reward += self.x + self.q * self.s - self.oldx - self.oldq * self.olds # first part of the reward (directly due to the action, no time has passed)
    next_t_change = self.t + self.dt # time of the next time step
    while self.t < next_t_change: # as long as the end of the timestep is not reached
      time_to_next_event = np.random.exponential(1/sum(self.intensities)) # exponentially distributed time to the next event
      event_type = random.choices([1,2,3,4,5,6,7,8], cum_weights=np.cumsum(self.intensities))[0] # the type of the next event
      if self.t + time_to_next_event < next_t_change: # if the event time is within the timestep
        self.oldx, self.olds, self.oldq = self.x, self.s, self.q # saving the old cash, price and inventory
        self.t += time_to_next_event # increase the current time
        oldintensities = self.intensities # saving the old intensities
        self.intensities = self.defaultintensities*(1-np.exp(-self.decayfactor*time_to_next_event)) + self.intensities*np.exp(-self.decayfactor*time_to_next_event) # new intensities
        assert(self.askprice > self.bidprice) # sanity check
        if random.random() >= sum(self.intensities) / sum(oldintensities): # discard the event in this case
          event_type = 0
        if event_type == 1: # event type 1 => aggressive market buy
          jumpsize = np.round(np.random.exponential(self.expdispar)+0.01, 2) # jump size
          self.askprice = np.round(self.askprice+jumpsize, 2) # new ask-price
          self.s = np.round((self.askprice+self.bidprice)/2, 3) # new mid-price
          self.spread = np.round(self.askprice-self.bidprice, 2) # new spread
          if self.ask_order < self.askprice: # in this case the ask order must have gotten executed
            prob = 1
          elif self.ask_order == self.askprice: # in this case we assume the probability of 25%
            prob = 0.25
          else: # else it does not get executed
            prob = 0
          if self.ask_order < np.inf and self.q > -self.inventoryconstraint and random.random() < prob: # conditions needed for the execution to take place
            self.q -= 1
            self.limitscounter += 1
            self.x += self.ask_order - self.mmfee*self.ask_order
            self.ask_order = np.inf # ask order is now executed and does not exist anymore
          if self.ask_order < self.askprice:
            self.ask_order = np.inf # the order is canceled (would be executed but can't due to the inventory constraint)
          self.intensities += np.array(self.excitationmatrix[0])
        elif event_type == 2: # event type 2 => aggressive market sell
          jumpsize = np.round(np.random.exponential(self.expdispar)+0.01, 2) # jump size
          self.bidprice = np.round(self.bidprice - jumpsize, 2) # new bid-price
          self.s = np.round((self.askprice+self.bidprice)/2, 3) # new mid-price
          self.spread = np.round(self.askprice-self.bidprice, 2) # new spread
          if self.bid_order > self.bidprice: # in this case the bid order must have gotten executed
            prob = 1
          elif self.bid_order == self.bidprice: # in this case we assume the probability of 25%
            prob = 0.25
          else: # else it does not get executed
            prob = 0
          if self.bid_order > -np.inf and self.q < self.inventoryconstraint and random.random() < prob: # conditions needed for the execution to take place
            self.q += 1
            self.limitscounter += 1
            self.x -= self.bid_order + self.mmfee*self.bid_order # + 0.006*env.s
            self.bid_order = -np.inf # bid order is now executed and does not exist anymore
          if self.bid_order > self.bidprice:
            self.bid_order = -np.inf # the order is canceled (would be executed but can't due to the inventory constraint)
          self.intensities += np.array(self.excitationmatrix[1])
        elif event_type == 3 and (self.askprice-self.bidprice >= np.round(2*self.delta, 2)): # event type 3 => aggressive limit buy
          # only happens if the spread is larger than or equals two ticks (otherwise obviously can not happen)
          jumpsize = np.inf
          while jumpsize >= np.round(self.askprice - self.bidprice, 2):
            jumpsize = np.round(np.random.exponential(self.expdispar)+0.01, 2) # this is in essence a truncated exponential, since there now exists an upper limit to the jump size
          self.bidprice = np.round(self.bidprice + jumpsize, 2) # new bid-price
          self.s = np.round((self.askprice+self.bidprice)/2, 3) # new mid-price
          self.spread = np.round(self.askprice-self.bidprice, 2) # new spread
          self.intensities += np.array(self.excitationmatrix[2]) # new intensities
        elif event_type == 4 and (self.askprice-self.bidprice >= np.round(2*self.delta, 2)): # event type 4 => aggressive limit sell
          jumpsize = np.inf
          while jumpsize >= np.round(self.askprice - self.bidprice, 2):
            jumpsize = np.round(np.random.exponential(self.expdispar)+0.01, 2) # this is in essence a truncated exponential, since there now exists an upper limit to the jump size
          self.askprice = np.round(self.askprice - jumpsize, 2) # new ask-price
          self.s = np.round((self.askprice+self.bidprice)/2, 3) # new mid-price
          self.spread = np.round(self.askprice-self.bidprice, 2) # new spread
          self.intensities += np.array(self.excitationmatrix[3]) # new intensities
        elif event_type == 5 and self.bidprice>=np.round(self.bid_order+self.delta, 2): # event type 5 => aggressive limit buy cancellation
          jumpsize = np.inf
          if self.bid_order > -np.inf: # if the market maker's bid order still exists, it provides an upper bound to the jump size
            while jumpsize > np.round(self.bidprice-self.bid_order, 2):
              jumpsize = np.round(np.random.exponential(self.expdispar)+0.01, 2) # generating jump size
          else: # in this case there is no upper bound to the jump size
            jumpsize = np.round(np.random.exponential(self.expdispar)+0.01, 2) # generating jump size
          self.bidprice = np.round(self.bidprice - jumpsize, 2) # new bid-price
          self.s = np.round((self.askprice+self.bidprice)/2, 3) # new mid-price
          self.spread = np.round(self.askprice-self.bidprice, 2) # new spread
          self.intensities += np.array(self.excitationmatrix[4]) # new intensities
        elif event_type == 6 and self.askprice <= np.round(self.ask_order-self.delta, 2):  # event type 6 => aggressive limit sell cancellation
          jumpsize = np.inf
          if self.ask_order < np.inf: # if the market maker's ask order still exists, it provides an upper bound to the jump size
            while jumpsize > np.round(self.ask_order-self.askprice, 2):
              jumpsize = np.round(np.random.exponential(self.expdispar)+0.01, 2) # generating jump size
          else: # in this case there is no upper bound to the jump size
            jumpsize = np.round(np.random.exponential(self.expdispar)+0.01, 2) # generating jump size
          self.askprice = np.round(self.askprice + jumpsize, 2) # new ask-price
          self.s = np.round((self.askprice+self.bidprice)/2, 3) # new mid-price
          self.spread = np.round(self.askprice-self.bidprice, 2) # new spread
          self.intensities += np.array(self.excitationmatrix[5])  # new intensities
        elif event_type == 7: # event type 7 => non-aggressive market buy
          if self.ask_order == self.askprice and self.q > -self.inventoryconstraint and random.random()<self.Z: # checking conditions
            self.q -= 1
            self.limitscounter += 1
            self.x += self.ask_order - self.mmfee*self.ask_order
            self.ask_order = np.inf
          self.intensities += np.array(self.excitationmatrix[6]) # new intensities
        elif event_type == 8: # event type 8 => non-aggressive market sell
          if self.bid_order == self.bidprice and self.q < self.inventoryconstraint and random.random()<self.Z: # checking conditions
            self.q += 1
            self.limitscounter += 1
            self.x -= self.bid_order + self.mmfee*self.bid_order
            self.bid_order = -np.inf
          self.intensities += np.array(self.excitationmatrix[7]) # new intensities
        else:
          pass
        self.reward += self.x + self.q*self.s - self.oldx - self.oldq*self.olds - self.gamma*time_to_next_event*abs(self.oldq) # the reward function as explained in the paper
      else: # if the event time exceeds the end of the time step
        time_to_next_event = next_t_change - self.t
        self.t += time_to_next_event # setting time to beginning of the next time-step
        self.intensities = self.defaultintensities*(1-np.exp(-self.decayfactor*time_to_next_event)) + self.intensities*np.exp(-self.decayfactor*time_to_next_event) #updating intensities
        self.reward -= self.gamma*time_to_next_event*abs(self.q) # reward function
    self.state = np.array([self.q/self.inventoryconstraint, (self.spread-self.n1)/self.n2, (self.intensities[0]+self.intensities[-2]-self.intensities[1]-self.intensities[-1]-self.n3)/self.n4]) # state space formulation
    if self.t >= self.T: # if the end of the episode is reached
      self.done = True
    return self.state, self.reward, self.done, {}

  def reset(self):
    self.t = 0 # initial time
    self.s = self.initial_parameters[3] # initial midprice value
    self.olds = self.initial_parameters[3] # initial old midprice value
    self.q = 0 # initial inventory level
    self.oldq = 0 # initial old inventory level
    self.x = 0 # initial cash
    self.oldx = 0 # initial old cash
    self.intensities = np.array(self.initial_parameters[4])
    self.defaultintensities = self.intensities # baseline intensities
    self.done = False # done indicator (indicates whether the episode has finished)
    self.limitscounter = 0  # counts the number of executions of the market maker's limit orders in an episode
    self.marketscounter = 0 # counts the number of executions of the market maker's market orders in an episode
    self.spread = self.initial_parameters[0] # initial spread
    self.askprice = self.initial_parameters[1] # initial ask price
    self.bidprice = self.initial_parameters[2] # initial bid price
    self.ask_order = np.inf # initial ask order
    self.bid_order = -np.inf # initial bid order
    self.state = np.array([self.q/self.inventoryconstraint, (self.spread-self.n1)/self.n2, (self.intensities[0]+self.intensities[-2]-self.intensities[1]-self.intensities[-1]-self.n3)/self.n4]) # state space (normalized)
    return self.state

  def render(self):
    print(self.state)

# determine normalization parameters
def return_normalization_parameters(T=10_000, n1=0, n2=1, n3=0, n4=1):
  env = MarketMakingEnv(T=T, n1=n1, n2=n2, n3=n3, n4=n4)
  spreads = []
  alphas = []
  for i in range(1):
    observation = env.reset()
    while True:
      spreads.append(env.state[1])
      alphas.append(env.state[2])
      action = [np.random.uniform(0,1), np.random.uniform(0,1)]
      observation, reward, done, info = env.step(action)
      if done:
        break
  env.close()
  return np.mean(spreads), np.std(spreads), np.mean(alphas), np.std(alphas)

n1, n2, n3, n4 = return_normalization_parameters(T=10_000)
env = MarketMakingEnv(n1=n1, n2=n2, n3=n3, n4=n4)
env = Monitor(env, log_dir)

model = SAC("MlpPolicy", env, verbose=1, gamma=1, batch_size=512, learning_rate=0.0003)
model.set_env(env)
model.set_logger(logger)

# model training
model.learn(1_000_000)

if  v:
  plt.ion()
  fig_pnl = plt.figure()
  ax_pnl = fig_pnl.gca()
  ax_pnl.grid(True)
  ax_pnl.set_xlabel("time")
  ax_pnl.set_ylabel("PnL")
  plt.show()

# test the trained agent
def tester():
  rreturns3 = []
  pvalues3 = []
  qqs3 = []
  qqsw = []
  ppvs = []
  allqs = []
  ssss = []
  for i in range(1_000):
    obs = env.reset()
    returns3 = []
    qs3 = []
    pvs= []
    while True:
      action, _ = model.predict(obs, deterministic=True)
      obs, reward, done, info = env.step(action)
      returns3.append(reward)
      pvs.append(env.x+env.q*env.s)
      qs3.append(env.q)
      if done:
        pvalues3.append(env.x+env.q*env.s)
        # print("episode end")
        break
    qqs3.append(qs3[-1])
    allqs.append(qs3)
    ppvs.append(pvs)
    qqsw.append(np.mean(abs(np.array(qs3))))
    rreturns3.append(np.sum(returns3))
    ssss.append(env.s)
    if  v:
      if i%200 == 0:
        ax_pnl.plot(np.cumsum(returns3))
        fig_pnl.canvas.flush_events()
  return rreturns3, pvalues3, qqs3, qqsw, ppvs, allqs, ssss, env.marketscounter, env.limitscounter

rreturns, pvalues, qqs, qqsw, ppvs, allqs, ssss, marketcounters, limitcounters = tester()
print(f"rreturns={rreturns}, pvalues={pvalues}, qqs={qqs}, qqsw={qqsw}, ppvs={ppvs}, allqs={allqs}, ssss={ssss}, marketcounters={marketcounters}, limitcounters={limitcounters}")
results_plotter.plot_results([log_dir], 100_000, results_plotter.X_TIMESTEPS, "SAC market making")

def moving_average(values, window):
  weights = np.repeat(1.0, window) / window
  return np.convolve(values, weights, "valid")

def plot_results(log_folder, title="Learning Curve"):
  x, y = ts2xy(load_results(log_folder), "timesteps")
  y = moving_average(y, window=50)
  # truncate x
  x = x[len(x) - len(y):]

  fig = plt.figure(title)
  plt.plot(x, y)
  plt.xlabel("Number of Timesteps")
  plt.ylabel("Rewards")
  plt.title(title + " Smoothed")
  plt.show()
  plt.grid()

plot_results(log_dir)
print("end.")
