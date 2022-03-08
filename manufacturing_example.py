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

sim = lambda opp_type: simulate(s0, game, solns, opp_type)

# Simulate against cooperative agent
sim("coop")

# Simulate against adversarial agent
sim("adv")

# Simulate against random agent
sim("random")

# # Simulate against random agent
sim("random")
# # Simulate against random agent
sim("random")
# # Simulate against random agent
# sim("random")
# # Simulate against random agent
# sim("random")