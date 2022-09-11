"""
Car rental example.
"""

from scipy.stats import poisson
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns
from util import *

verbose = 1
rand_seed = int(time.time())
start_time, device = preamble(rand_seed)

max_cars = 20
max_moved = 5
lambda_request1 = 3
lambda_request2 = 4
lambda_return1  = 3
lambda_return2  = 2
gamma = 0.9
theta = 1e-4

# if n-lambda > this value, then the probability of getting n is truncated to 0
poisson_upper_dist = 8
poisson_cache = dict()

def poisson_probability(n, lmbda):
  assert(lmbda < 10)
  key = n * 10 + lmbda
  if key not in poisson_cache:
    poisson_cache[key] = poisson.pmf(n, lmbda)
  return poisson_cache[key]

# 1. init
# state: [# of cars in first location, # of cars in second location]
# all possible actions
# positive if moving cars from first location to second location, negative if moving cars from second location to first location
A = np.arange(-max_moved, max_moved+1)
V = 10*np.random.rand(max_cars+1, max_cars+1)
pi = np.random.randint(low=-max_moved, high=max_moved+1, size=(max_cars+1, max_cars+1), dtype=int)

def expected_return2(s, a, V):
  v = 0.0 # total return
  v -= 2.0 * abs(a) # cost for moving cars

  # moving cars
  init_num_of_cars1 = min(s[0] - a, max_cars)
  init_num_of_cars2 = min(s[1] + a, max_cars)

  # n - number of rented cars
  # m - number of returned cars
  # s_prime = s - a - n + m for each n and m
  # since events "renting 3 cars at location 1" and "returning 2 cars at location 2", etc. are independent, we multipy their probabilities
  for n1 in range(min(init_num_of_cars1, lambda_request1+poisson_upper_dist)+1):
    for n2 in range(min(init_num_of_cars2, lambda_request2+poisson_upper_dist)+1):
      # (partial) probability for current combination of rental requests
      prob_rent = poisson_probability(n1, lambda_request1) * poisson_probability(n2, lambda_request2)
      # get credits for renting
      reward = 10.0 * (n1 + n2)
      num_of_cars1 = init_num_of_cars1 - n1
      num_of_cars2 = init_num_of_cars2 - n2

      for m1 in range(min(max_cars-init_num_of_cars1, lambda_return1+poisson_upper_dist)+1):
        for m2 in range(min(max_cars-init_num_of_cars2, lambda_return2+poisson_upper_dist)+1):
          prob_return = poisson_probability(m1, lambda_return1) * poisson_probability(m2, lambda_return2)
          s_prime1 = num_of_cars1 + m1
          s_prime2 = num_of_cars2 + m2
          prob = prob_return * prob_rent
          v += prob * (reward + gamma * V[s_prime1, s_prime2])
  return v

# the code seems to inflate returns via V(s_prime) by counting states when # of rented cars > # of available cars
def expected_return(s, a, V):
  poisson_upper_bound = 11
  # initialize total return
  returns = 0.0
  # cost for moving cars
  returns -= 2.0 * abs(a)

  # moving cars
  num_of_cars_first_loc_init = min(s[0] - a, max_cars)
  num_of_cars_second_loc_init = min(s[1] + a, max_cars)

  # go through all possible rental requests
  for rental_request_first_loc in range(poisson_upper_bound):
    for rental_request_second_loc in range(poisson_upper_bound):
      # probability for current combination of rental requests
      prob = poisson_probability(rental_request_first_loc, lambda_request1) * poisson_probability(rental_request_second_loc, lambda_request2)

      num_of_cars_first_loc = num_of_cars_first_loc_init
      num_of_cars_second_loc = num_of_cars_second_loc_init

      # valid rental requests should be less than actual # of cars
      valid_rental_first_loc = min(num_of_cars_first_loc, rental_request_first_loc)
      valid_rental_second_loc = min(num_of_cars_second_loc, rental_request_second_loc)

      # get credits for renting
      reward = 10.0 * (valid_rental_first_loc + valid_rental_second_loc)
      num_of_cars_first_loc -= valid_rental_first_loc
      num_of_cars_second_loc -= valid_rental_second_loc

      for returned_cars_first_loc in range(poisson_upper_bound):
        for returned_cars_second_loc in range(poisson_upper_bound):
          prob_return = poisson_probability(returned_cars_first_loc, lambda_return1) * poisson_probability(returned_cars_second_loc, lambda_return2)
          num_of_cars_first_loc_ = min(num_of_cars_first_loc + returned_cars_first_loc, max_cars)
          num_of_cars_second_loc_ = min(num_of_cars_second_loc + returned_cars_second_loc, max_cars)
          prob_ = prob_return * prob
          returns += prob_ * (reward + gamma * V[num_of_cars_first_loc_, num_of_cars_second_loc_])
  return returns

if  verbose:
  plt.ion()
  _, axes = plt.subplots(2, 4, figsize=(40, 20))
  plt.subplots_adjust(wspace=0.1, hspace=0.2)
  axes = axes.flatten()
  plt.show()

iterations = 0
while True:
  print(f"iteration: {iterations}")
  if verbose:
    fig = sns.heatmap(np.flipud(pi), cmap="YlGnBu", ax=axes[iterations])
    fig.set_ylabel("# cars at 1st location", fontsize=10)
    fig.set_yticks(list(reversed(range(max_cars+1))))
    fig.set_xlabel("# cars at 2nd location", fontsize=10)
    fig.set_title(f"policy {iterations}", fontsize=20)

  # policy evaluation
  delta = 1e8
  while delta >= theta:
    delta = 0
    for s1 in range(max_cars+1):
      for s2 in range(max_cars+1):
        v = V[s1,s2]
        v_new = expected_return([s1,s2], pi[s1,s2], V)
        V[s1,s2] = v_new
        delta = max(delta, abs(v - v_new))
    print(f"max value change {delta:.05f}")

  # policy improvement
  policy_stable = True
  for s1 in range(max_cars+1):
    for s2 in range(max_cars+1):
      a_old = pi[s1,s2]
      action_returns = []
      for a in A:
        if (0 <= a <= s1) or (-s2 <= a <= 0):
          action_returns.append(expected_return([s1, s2], a, V))
        else:
          action_returns.append(-np.inf)
      a_new = A[np.argmax(action_returns)]
      pi[s1, s2] = a_new
      if a_old != a_new:
        policy_stable = False
  print(f"policy stable {policy_stable}")
  if policy_stable:
    break
  iterations += 1

v_star = V
pi_star = pi
print(f"# of iterations = {iterations+1}; v_star = {v_star};\npi_star = {pi_star}")
if verbose:
  fig = sns.heatmap(np.flipud(pi), cmap="YlGnBu", ax=axes[-2])
  fig.set_ylabel("# cars at 1st location", fontsize=10)
  fig.set_yticks(list(reversed(range(max_cars+1))))
  fig.set_xlabel("# cars at 2nd location", fontsize=10)
  fig.set_title(f"$\pi^*$", fontsize=20)

  fig = sns.heatmap(np.flipud(V), cmap="YlGnBu", ax=axes[-1])
  fig.set_ylabel("# cars at 1st location", fontsize=10)
  fig.set_yticks(list(reversed(range(max_cars+1))))
  fig.set_xlabel("# cars at 2nd location", fontsize=10)
  fig.set_title(f"$v^*$", fontsize=20)
  plt.savefig("car_rental.png")
  plt.close()

  fig = plt.figure()
  ax = fig.gca()
  fig = sns.heatmap(np.flipud(pi), cmap="YlGnBu", ax=ax)
  fig.set_ylabel("# cars at 1st location")
  fig.set_yticks(list(reversed(range(max_cars+1))))
  fig.set_xlabel("# cars at 2nd location")
  fig.set_title(f"$\pi^*$")

  fig = plt.figure()
  ax = fig.add_subplot(projection="3d")
  x = np.arange(V.shape[0])
  y = np.arange(V.shape[1])
  xx, yy = np.meshgrid(x, y)
  surf = ax.plot_surface(xx, yy, V, cmap=cm.coolwarm, linewidth=0, antialiased=False)
  ax.set_ylabel("# cars at 1st location")
  ax.set_xlabel("# cars at 2nd location")
  ax.set_title(f"$v^*$")
  fig.show()
postscript(start_time)
