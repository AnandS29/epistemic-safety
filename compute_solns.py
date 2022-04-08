from cmath import isclose
import re
import numpy as np
import matplotlib.pyplot as plt
import nashpy as nash
import itertools
import cvxpy as cp
import gurobipy
import math

def compute_baseline(game):
    state_names, states,human_actions,robot_actions,transition,reward,gamma,T = game["state_names"],game["states"],game["human_actions"],game["robot_actions"],game["transition"],game["reward"],game["gamma"],game["T"]
    v_funs_adv = {i:{s:None for s in states} for i in range(T+1)}
    for s in state_names:
        v_funs_adv[T][s] = 0

    t = T-1
    m,n = len(human_actions), len(robot_actions)
    while t >= 0:
        print(t, end=", ")
        for s in state_names:
            A = np.zeros(shape=(m,n))
            for i in range(len(human_actions)):
                for j in range(len(robot_actions)):
                    uH = human_actions[i]
                    uR = robot_actions[j]
                    # print("before: ",s, uH, uR)
                    s_next = transition((s,0),uH,uR,0)[0]
                    # print("after: ",s_next)
                    qH = reward((s,0), uH, uR)[0] + gamma*v_funs_adv[t+1][s_next]
                    A[i,j] = qH
            # rps = nash.Game(A)
            # (s1,s2) = list(rps.support_enumeration())[0]
            # v_funs_adv[t][s] = rps[s1,s2][0]

            v = cp.Variable()
            a = cp.Variable((m,1))

            constraints = [a >= 0, cp.sum(a)==1]
            for j in range(n):
                constraints.append(v <= a.T @ A[:,j])

            obj = cp.Maximize(v)
            prob = cp.Problem(obj,constraints)
            v_funs_adv[t][s] = prob.solve() # solver=cp.GUROBI
        t -= 1

    return v_funs_adv
# Try on iterated RPS <- risk capital should go negative
def compute_ex_post(game):
    # Unpack
    states,human_actions,robot_actions,transition,reward,gamma,T = game["states"],game["human_actions"],game["robot_actions"],game["transition"],game["reward"],game["gamma"],game["T"]

    # Initialize value functions and optimal control storage dictionaries
    v_funs = {i:{s:None for s in states} for i in range(T+1)}
    v_funs_adv_p = {i:{s:None for s in states} for i in range(T+1)}
    uHs = {i:{s:None for s in states} for i in range(T)}
    uRs_coop = {i:{s:None for s in states} for i in range(T)}
    uRs_adv = {i:{s:None for s in states} for i in range(T)}

    for s in states:
        v_funs[T][s] = 0
        v_funs_adv_p[T][s] = 0

    # Compute baseline
    v_funs_adv = compute_baseline(game)

    # Compute ex post solution
    t = T-1
    while t >= 0:
        for s in states:
            m,n = len(human_actions), len(robot_actions)
            A_S = np.zeros(shape=(m,n))
            A_H = np.zeros(shape=(m,n))

            

            for i in range(m):
                for j in range(n):
                    uH = human_actions[i]
                    uR = robot_actions[j]

                    dr = v_funs_adv[t+1][transition(s,uH,uR,0)[0]] + reward(s, uH, uR)[0] - v_funs_adv[t][s[0]]
                    # if (s[1]+dr) < 0 and not isclose(0,s[1]+dr): print(s, uH, uR, v_funs_adv[t][s[0]], transition(s,uH,uR,0)[0], v_funs_adv[t+1][transition(s,uH,uR,0)[0]])
                    if (s[1]+dr) < 0 and not (np.abs(s[1]+dr) <= 1e-6):
                        A_S[i,j] = -1e5
                        A_H[i,j] = -1e5
                    else:
                        s_next = transition(s,uH,uR,dr)
                        q_coop = reward(s, uH, uR)[0] + gamma*v_funs[t+1][s_next]
                        A_S[i,j] = q_coop

                        q_adv = reward(s, uH, uR)[1] + gamma*v_funs_adv_p[t+1][s_next]
                        A_H[i,j] = q_adv

            vals = []
            p_Hs = []
            for j in range(n):
                p_H = cp.Variable((m,1))
                constraints = []
                constraints.extend([p_H>=0, cp.sum(p_H)==1])
                for j in range(n):
                    constraints.append(p_H.T @ A_H[:,j] >= v_funs_adv[t][s[0]] - s[1])
                objective = cp.Maximize(p_H.T @ A_S[:,j])
                prob = cp.Problem(objective, constraints)
                val_j = prob.solve()
                vals.append(val_j)
                p_Hs.append(p_H.value)

            opt_val = np.max(vals)
            opt_p_H = p_Hs[np.argmax(vals)]
            v_funs[t][s] = opt_val
            # print(t,s,opt_val, vals, A_S)
            v_funs_adv_p[t][s] = np.min(opt_p_H.T @ A_H)
            
            uR_adv = np.zeros((n,1))
            uR_coop = np.zeros((n,1))

            uR_adv[np.argmin(opt_p_H.T @ A_H),0] = 1
            uR_coop[np.argmax(vals),0] = 1

            uRs_adv[t][s] = uR_adv
            uRs_coop[t][s] = uR_coop
            uHs[t][s] = opt_p_H
        t = t - 1

    solns = {
        "baseline_val":v_funs_adv, "ex_post_val":v_funs, "adv_val":v_funs_adv_p,
        "r_adv_actions":uRs_adv, "r_coop_actions":uRs_coop, "h_actions":uHs
    }

    return solns

def proj_risk(r,risk_levels,round=2):
    # factor = 10**round
    # r = math.floor(factor*r)/factor
    # if r < 0: print(r)
    if np.abs(r) <= 1e-6: r = 0
    # print(r)
    return np.max([l for l in risk_levels if l <= r ])

def simulate(s0, game, solns, opponent_type, seed=None, output=True):
    # Set random seed
    np.random.seed(seed)

    # Unpack
    states,human_actions,robot_actions,transition,reward,gamma,T,risk_levels, misc = game["states"],game["human_actions"],game["robot_actions"],game["transition"],game["reward"],game["gamma"],game["T"],game["risk_levels"], game["misc"]
    v_funs_adv, uRs_adv, uRs_coop, uHs = solns["baseline_val"], solns["r_adv_actions"], solns["r_coop_actions"], solns["h_actions"]

    # Construct human policy
    def pol_H(t,s):
        prob = uHs[t][s]
        prob[prob<0] = 0
        prob /= np.sum(prob)
        res = np.random.choice(range(len(human_actions)),p=prob.reshape((len(human_actions))))
        return human_actions[res]
    # Construct robot policy according to type
    pol_R = None
    if opponent_type == "adv":
        pol_R = lambda t,s: robot_actions[np.argmax(uRs_adv[t][s])]
    elif opponent_type == "coop":
        pol_R = lambda t,s: robot_actions[np.argmax(uRs_coop[t][s])]
    elif opponent_type == "random":
        pol_R = lambda t,s: robot_actions[np.random.choice(range(len(robot_actions)))]
    else:
        raise Exception("Opponent type not available, select from adv, coop, random.")

    proj = lambda r: proj_risk(r,risk_levels,round=misc["round"])

    # Simulate policy
    if output: print("Testing with " + opponent_type + " agent")
    s = s0
    if output: print(s)
    total_r = 0
    for t in range(T):
        uH_strat = uHs[t][s]
        uH = pol_H(t,s)
        uR = pol_R(t,s)
        uR_adv = robot_actions[np.argmax(uRs_adv[t][s])]
        curr_r = reward(s, uH, uR)[0]
        total_r += curr_r

        # Calculate dr in expectation over my strategy
        dr = -v_funs_adv[t][s[0]]
        for i in range(len(human_actions)):
            h = human_actions[i]
            dr += uH_strat[i,0]*(reward(s, h, uR)[0] + gamma*v_funs_adv[t+1][transition(s,h,uR,0)[0]])

        s_next = transition(s,uH,uR,dr)

        # Calculate amount risked in expectation over strategy against an adversary
        rsk = -v_funs_adv[t][s[0]] + s[1]
        for i in range(len(human_actions)):
            h = human_actions[i]
            rsk += uH_strat[i,0]*(reward(s, h, uR_adv)[0] + gamma*v_funs_adv[t+1][transition(s,h,uR_adv,0)[0]])

        if output: print("state=",s_next,", uH=", uH, ", strat=", uH_strat.T, ", uR=", uR,", dr=", dr, ", r+dr=", s[1]+dr, ", constraint=",rsk)
        s = s_next

    # if output: print("total_reward=",total_r)
    if output: print()

    res = {
        "total_reward":total_r, "baseline_reward":v_funs_adv[0][s0], "initial_risk":s0[1]
    }

    return res

def compute_stats(n, s0, game, solns, opponent_type, seed=None):
    np.random.seed(seed)
    average_total_reward = 0
    ress = []
    baseline_reward = solns["baseline_val"][0][s0[0]]

    # Run n simulations
    for _ in range(n):
        res = simulate(s0, game, solns, opponent_type, seed=None, output=False)
        average_total_reward += res["total_reward"]/n
        ress.append(res)

    # Compute statistics
    stats = {"average_reward":average_total_reward, "baseline_reward":baseline_reward}

    return stats

