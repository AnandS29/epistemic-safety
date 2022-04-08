from cProfile import label
import numpy as np
import matplotlib.pyplot as plt
import nashpy as nash
import itertools
import cvxpy as cp
import pickle

from compute_solns import *

# Set up game

game_types = ["RHRHU", "HRHU", "RHU", "HU", "U_pos", "U_neg", "RU", "HRU", "RHRU", "NULL"]
abs_util = [0.1,0.2,0.5]
state_names = []
for g in game_types:
    for u in abs_util:
        state_names.append((g,u))

round = 2
risk_levels = np.linspace(0,5,51)
states = []
for name in state_names:
    for risk in risk_levels:
        states.append((name, risk))

human_actions = []
for r in abs_util:
    for a in [1,2]:
        human_actions.append((a,r))

robot_actions = [1,2]

gamma = 1
T = 10

# def project_risk(r):
#     # r = np.round(r,round)
#     return np.max([l for l in risk_levels if l <= r ]) # TODO : Change projection to take worst case

def transition(state, uH, uR, dr, check=False, initial="NULL"):
    (g,u) = state[0]
    g_next = g
    u_next = u
    if g == "RHRHU":
        if uR == 1:
            g_next = "HRHU"
        else:
            g_next = "NULL"
    elif g == "HRHU":
        if uH[0] == 1:
            g_next = "RHU"
        else:
            g_next = "NULL"
        u_next = uH[1]
    elif g == "RHU":
        if uR == 1:
            g_next = "HU"
        else:
            g_next = "NULL"
    elif g == "HU":
        if uH[0] == 1:
            g_next = "U_pos"
        else:
            g_next = "U_neg"
    elif g == "U_pos":
        g_next = "NULL"
    elif g == "NULL":
        g_next = initial
    elif g == "RHRU":
        if uR == 1:
            g_next = "HRU"
        else:
            g_next = "NULL"
    elif g == "HRU":
        if uH[0] == 1:
            g_next = "RU"
        else:
            g_next = "NULL"
        u_next = uH[1]
    elif g == "RU":
        if uR == 1:
            g_next = "U_pos"
        else:
            g_next = "U_neg"
    elif g == "U_neg":
        g_next = "NULL"

    if g in ["HRU","HU","HRHU"]:
        u_next = uH[1]

    if check and state[1] + dr < 0:
        print("RIP: Risking more than you've won")

    r = proj_risk(state[1]+dr, risk_levels) # risk_levels[np.argmin([np.abs(state[1]+dr-l) for l in risk_levels])]
    return ((g_next,u_next),r)
        
def reward(state, uH, uR):
    (g,u) = state[0]
    if g == "U_pos":
        social = u
    elif g == "U_neg":
        social = -u
    else:
        social = 0
    return (social,social,social)

def get_game(return_state):
    game = {
        "state_names":state_names, "states":states, "risk_levels":risk_levels,
        "human_actions":human_actions, "robot_actions":robot_actions,
        "transition":(lambda state, uH, uR, dr : transition(state, uH, uR, dr, check=False, initial=return_state)), 
        "reward":reward,
        "gamma":gamma, "T":T,
        "misc": {"round":round}
    }

    return game

def run_exp(init_state, init_alpha, init_risk):
    return_state = init_state # "HU"
    initial = (init_state, init_alpha) # ("HU", 0.1)
    risk_tol = init_risk # 1.0

    game = get_game(return_state)

    fname = "general_example_"+str(return_state)+"_"+str(initial)

    print("Checking if solutions exist...")
    try:
        print("Loading solutions...")
        with open(fname+'.pickle', 'rb') as handle:
            solns = pickle.load(handle)
    except:
        # Compute solutions
        print("No solutions, so now computing solutions...")
        solns = compute_ex_post(game)
        print("Saving solutions")
        with open(fname+'.pickle', 'wb') as handle:
            pickle.dump(solns, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("Done!")

# print(solns["adv_val"][0])

# Simulation parameters
def run_sim(init_state, init_alpha, init_risk):
    return_state = init_state # "HU"
    initial = (init_state, init_alpha) # ("HU", 0.1)
    risk_tol = init_risk # 1.0

    game = get_game(return_state)

    fname = "general_example_"+str(return_state)+"_"+str(initial)
    with open(fname+".pickle", 'rb') as handle:
        solns = pickle.load(handle)

    s0 = (initial,proj_risk(risk_tol,risk_levels))

    seed = None
    n_rollouts_max = 20

    sim = lambda opp_type: simulate(s0, game, solns, opp_type, seed)
    comp_stats = lambda n, opp_type: compute_stats(n, s0, game, solns, opp_type, seed)

    opp_types = ["coop", "adv", "random"]

    # Simulate against different agents
    print("Simulating against different agents")
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
    plt.plot(x_vals, [-s0[1] for _ in x_vals], label="Initial risk capital", alpha=0.7)
    plt.xlabel("Number of rollouts")
    plt.ylabel("Average reward - baseline")
    plt.legend()
    plt.show()

# Issues to talk about: projecting onto risk levels when they become negative
# Memoization