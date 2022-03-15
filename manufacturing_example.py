from cProfile import label
import numpy as np
import matplotlib.pyplot as plt
import nashpy as nash
import itertools
import cvxpy as cp

from compute_solns import *

# Set up game

state_names = ["s1"]
risk_levels = np.linspace(0,100,100)
states = []
for name in state_names:
    for risk in risk_levels:
        states.append((name, risk))

human_actions = [1,5,10]
robot_actions = ["S","F"]

gamma = 1
T = 10

def project_risk(r):
    return risk_levels[np.argmin([np.abs(r-l) for l in risk_levels])]

def transition(state, uH, uR, dr, check=False):
    s = state[0]
    if check and state[1] + dr < 0:
        print("RIP: Risking more than you've won")
    r = risk_levels[np.argmin([np.abs(state[1]+dr-l) for l in risk_levels])]
    return (s,r)
        
def reward(state, uH, uR):
    rR = (2 if uR=="S" else 0)*uH
    rH = (10-uH)
    social = rR + rH
    return (social,social,social)


game = {
    "states":states, "risk_levels":risk_levels,
    "human_actions":human_actions, "robot_actions":robot_actions,
    "transition":transition, "reward":reward,
    "gamma":gamma, "T":T
}

# Compute solutions
solns = compute_ex_post(game)

# print(solns["adv_val"][0])

# Simulation parameters
s0 = ("s1",project_risk(1))

seed = None
n_rollouts_max = 30

sim = lambda opp_type: simulate(s0, game, solns, opp_type, seed)
comp_stats = lambda n, opp_type: compute_stats(n, s0, game, solns, opp_type, seed)

opp_types = ["coop", "adv", "random"]

# Simulate against different agents
for opp in opp_types:
    sim(opp)

print("Computing statistics against different agents")
# Compute statistics for different agents
res = {opp:[] for opp in opp_types}
for opp in opp_types:
    for n in range(1,n_rollouts_max):
        stats = comp_stats(n,opp)
        delta = stats["average_reward"]-stats["baseline_reward"]
        res[opp].append(delta)

print("Plotting results")
# Plot statistics
plt.figure()
plt.title("Average reward - baseline for n rollouts")
x_vals = range(1,n_rollouts_max)
for opp in opp_types:
    res_opp = res[opp]
    plt.plot(x_vals, res_opp, label=opp)
plt.plot(x_vals, [-s0[1] for _ in x_vals], label="Initial risk capital")
plt.xlabel("Number of rollouts")
plt.ylabel("Average reward - baseline")
plt.legend()
plt.show()

# Issues to talk about: projecting onto risk levels when they become negative
# Memoization