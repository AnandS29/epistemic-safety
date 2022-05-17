from cmath import isclose
import functools
import re
import numpy as np
import matplotlib.pyplot as plt
import nashpy as nash
import itertools
import cvxpy as cp
import gurobipy
import math
from joblib import Parallel, delayed
import time
import pickle
# from multiprocessing import Pool
from multiprocessing.pool import ThreadPool as Pool
import os

def process(s,t,states,human_actions,robot_actions,transition,reward,gamma,T,v_funs,v_funs_adv_p,v_funs_adv):
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
                A_S[i,j] = -1e10 # very not optimal
                A_H[i,j] = -1e10 # not feasible
                # print("neg: ", s, uH, uR)
            else:
                s_next = transition(s,uH,uR,dr)
                q_coop = reward(s, uH, uR)[0] + gamma*v_funs[t+1][s_next]
                A_S[i,j] = q_coop

                q_adv = reward(s, uH, uR)[1] + gamma*v_funs_adv_p[t+1][s_next]
                A_H[i,j] = q_adv

    vals = []
    p_Hs = []
    # print(A_S)
    for j in range(n):
        p_H = cp.Variable((m,1))
        constraints = []
        constraints.extend([p_H>=0, cp.sum(p_H)==1])
        for k in range(n):
            constraints.append(p_H.T @ A_H[:,k] >= v_funs_adv[t][s[0]] - s[1])
        objective = cp.Maximize(p_H.T @ A_S[:,j])
        prob = cp.Problem(objective, constraints)
        val_j = prob.solve(solver=cp.GUROBI)
        vals.append(val_j)
        p_Hs.append(p_H.value)

    if False and t==1 and s==("s1",0.0):
        print(A_H)
        print(A_S)
        print(vals)
        print([(p.T, p.T @ A_H[:,0] - v_funs_adv[t][s[0]] + s[1], p.T @ A_H[:,1] - v_funs_adv[t][s[0]] + s[1])  for p in p_Hs])
        print("constraint:",v_funs_adv[t][s[0]] - s[1])

    try:
        opt_val = np.max(vals)
        opt_p_H = p_Hs[np.argmax(vals)]

        uR_adv = np.zeros((n,1))
        uR_coop = np.zeros((n,1))
    
        uR_adv[np.argmin(opt_p_H.T @ A_H),0] = 1
    except:
        print(A_H)
        print(A_S)
        print(v_funs_adv[t][s[0]] - s[1])
        print(s, t, s[0], s[1])
        print(vals)
        p_test = np.zeros((n,1))
        p_test[0,0] = 1
        print([p_test.T @ A_H[:,k] for k in range(n)])
        raise ValueError("Error")
    uR_coop[np.argmax(vals),0] = 1

    return [opt_val, np.min(opt_p_H.T @ A_H), uR_adv, uR_coop, opt_p_H]

def get_adv_pol(game, uHs, type="adv", gift_policy=False):
    state_names, states,human_actions,robot_actions,transition,reward,gamma,T = game["state_names"],game["states"],game["human_actions"],game["robot_actions"],game["transition"],game["reward"],game["gamma"],game["T"]
    if gift_policy:
        state_names = states
    v_funs = {i:{s:None for s in state_names} for i in range(T+1)}
    uRs_adv = {i:{s:None for s in state_names} for i in range(T)}


    for s in state_names:
        v_funs[T][s] = 0
    
    t = T-1
    m,n = len(human_actions), len(robot_actions)
    while t >= 0:
        # print(t, end=", ")
        for s in state_names:
            A = np.zeros(shape=(m,n))
            for i in range(len(human_actions)):
                for j in range(len(robot_actions)):
                    uH = human_actions[i]
                    uR = robot_actions[j]
                    # print("before: ",s, uH, uR)
                    if gift_policy:
                        s_next = transition(s,uH,uR,0)
                        qH = reward(s, uH, uR)[0] + gamma*v_funs[t+1][s_next]
                    else:
                        s_next = transition((s,0),uH,uR,0,is_gift=False)[0]
                        qH = reward((s,0), uH, uR)[0] + gamma*v_funs[t+1][s_next]
                    # print("after: ",s_next)
                    A[i,j] = qH

            uH = uHs[t][s]

            uR = np.zeros(shape=(n,1))
            if type=="adv":
                i_opt = np.argmin(A.T @ uH)
            elif type=="coop":
                i_opt = np.argmax(A.T @ uH)
            else:
                raise ValueError("Invalid type")

            uR[i_opt] = 1
            uRs_adv[t][s] = uR

            v_funs[t][s] = (uH.T @ A @ uR)[0,0]

        t -= 1
    return uRs_adv

def get_coop_pol(game, uHs, gift_policy=False):
    return get_adv_pol(game, uHs, type="coop", gift_policy=gift_policy)

def compute_baseline(game):
    state_names, states,human_actions,robot_actions,transition,reward,gamma,T = game["state_names"],game["states"],game["human_actions"],game["robot_actions"],game["transition"],game["reward"],game["gamma"],game["T"]
    v_funs_adv = {i:{s:None for s in state_names} for i in range(T+1)}
    uHs = {i:{s:None for s in state_names} for i in range(T)}
    uRs = {i:{s:None for s in state_names} for i in range(T)}

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
                    s_next = transition((s,0),uH,uR,0,is_gift=False)[0]
                    # print("after: ",s_next)
                    qH = reward((s,0), uH, uR)[0] + gamma*v_funs_adv[t+1][s_next]
                    A[i,j] = qH

            v = cp.Variable()
            a = cp.Variable((m,1))

            constraints = [a >= 0, cp.sum(a)==1]
            for j in range(n):
                constraints.append(v <= a.T @ A[:,j])

            obj = cp.Maximize(v)
            prob = cp.Problem(obj,constraints)
            v_funs_adv[t][s] = prob.solve() # solver=cp.GUROBI

            uH = a.value
            uHs[t][s] = uH

            uR = np.zeros(shape=(n,1))
            i_opt = np.argmin(A.T @ uH)
            uR[i_opt] = 1
            uRs[t][s] = uR
        t -= 1

    uRs_adv = get_adv_pol(game, uHs)
    uRs_coop = get_coop_pol(game, uHs)

    # print("In func:",v_funs_adv[0][(1,2,1,1)])

    return v_funs_adv, uHs, uRs_adv, uRs_coop

def compute_maximax(game):
    state_names, states,human_actions,robot_actions,transition,reward,gamma,T = game["state_names"],game["states"],game["human_actions"],game["robot_actions"],game["transition"],game["reward"],game["gamma"],game["T"]
    v_funs = {i:{s:None for s in state_names} for i in range(T+1)}
    uHs = {i:{s:None for s in state_names} for i in range(T)}
    uRs = {i:{s:None for s in state_names} for i in range(T)}

    for s in state_names:
        v_funs[T][s] = 0

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
                    s_next = transition((s,0),uH,uR,0,is_gift=False)[0]
                    # print("after: ",s_next)
                    qH = reward((s,0), uH, uR)[0] + gamma*v_funs[t+1][s_next]
                    A[i,j] = qH

            uH = np.zeros(shape=(m,1))
            j_opt = np.argmax(np.max(A,axis=1))
            uH[j_opt] = 1
            uHs[t][s] = uH

            uR = np.zeros(shape=(n,1))
            i_opt = np.argmax(np.max(A,axis=0))
            uR[i_opt] = 1
            uRs[t][s] = uR

            v_funs[t][s] = np.max(A)
        t -= 1

    uRs_adv = get_adv_pol(game, uHs)
    uRs_coop = get_coop_pol(game, uHs)

    return v_funs, uHs, uRs_coop, uRs_adv

# Try on iterated RPS <- risk capital should go negative
def compute_ex_post(game, parallel=False, save=None):
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

    print("Computing baseline...")
    # Compute baseline
    v_funs_adv, uHs_adv, uRs_adv, uRs_adv_coop = compute_baseline(game)
    print("Done with baseline!")

    # print("In ex post:",v_funs_adv[0][(1,2,1,1)])

    # raise Exception("Stop")

    # Compute maximax
    print("Computing maximax...")
    v_funs_maximax, uHs_maximax, uRs_maximax, uRs_maximax_adv  = compute_maximax(game)
    print("Done with maximax!")

    print("Computing ex post solution...")
    # Compute ex post solution
    t = T-1
    while t >= 0:
        start = time.time()
        f = lambda s: process(s,t,states,human_actions,robot_actions,transition,reward,gamma,T,v_funs,v_funs_adv_p,v_funs_adv)
        if parallel:    
            # results = Parallel(n_jobs=10)(delayed(process)(state) for state in states)
            pool = Pool(os.cpu_count())
            results = pool.map(f, states)
            pool.close()
        else:
            results = [f(state) for state in states]

        for i in range(len(results)):
            s = states[i]
            v_funs[t][s] = results[i][0]
            v_funs_adv_p[t][s] = results[i][1]
            uRs_adv[t][s] = results[i][2]
            uRs_coop[t][s] = results[i][3]
            uHs[t][s] = results[i][4]

        if save is not None:
            print("Saving results at time "+str(t)+" ...")
            solns_t = {
                "states": states,
                "v_funs": v_funs[t],
                "v_funs_adv_p": v_funs_adv_p[t],
                "uHs": uHs[t],
                "uRs_adv": uRs_adv[t],
                "uRs_coop": uRs_coop[t],
                "uRs_maximax": uRs_maximax[t],
                "uRs_maximax_adv": uRs_maximax_adv[t],
                "uHs_maximax": uHs_maximax[t],
                "uHs_adv": uHs_adv[t],
                "uRs_adv_coop": uRs_adv_coop[t],
                "uRs_adv_coop": uRs_adv_coop[t],
                "v_funs_maximax": v_funs_maximax[t],
                "v_funs_adv": v_funs_adv[t]
            }
            with open("experiments/"+save+'_'+str(t)+'.pickle', 'wb') as handle:
                pickle.dump(solns_t, handle, protocol=pickle.HIGHEST_PROTOCOL)

        end = time.time()
        print(t, "for",end-start,"s", end=", ")
        t = t - 1
    print("Done with ex post!")

    solns = {
        "baseline_val":v_funs_adv, "ex_post_val":v_funs, "adv_val":v_funs_adv_p, "maximax_val":v_funs_maximax,
        "r_adv_actions":get_adv_pol(game, uHs, gift_policy=True), "r_coop_actions":get_coop_pol(game, uHs, gift_policy=True), "h_actions":uHs,
        "h_base_actions":uHs_adv, "r_base_actions":uRs_adv, "r_base_coop_actions":uRs_adv_coop,
        "h_maximax_actions":uHs_maximax, "r_maximax_actions":uRs_maximax, "h_maximax_adv_actions":uRs_maximax_adv
    }

    return solns

def proj_risk(r,risk_levels,round=2,is_gift=True):
    # factor = 10**round
    # r = math.floor(factor*r)/factor
    if np.abs(r) <= 1e-5: r = 0
    if r < 0: 
        # print(r)
        if is_gift: 
            print(r)
            raise Exception("Risk should be positive!")
        return np.min(risk_levels)
    # print(r)
    return np.max([l for l in risk_levels if l <= r ])

def simulate(s0, game, solns, player_types, seed=None, verbose=False):
    # Unpack
    state_names,states,human_actions,robot_actions,transition,reward,gamma,T = game["state_names"],game["states"],game["human_actions"],game["robot_actions"],game["transition"],game["reward"],game["gamma"],game["T"]

    # Get human's mixed strategy
    uHs = None
    if player_types[0] == "gift":
        uHs = solns["h_actions"]
    elif player_types[0] == "maximax":
        uHs = solns["h_maximax_actions"]
    elif player_types[0] == "baseline":
        uHs = solns["h_base_actions"]
    else:
        raise Exception("Human type not available, select from gift, maximax, baseline.")

    # Construct human policy
    def pol_H(t,s):
        prob = uHs[t][s]
        prob[prob<0] = 0
        prob /= np.sum(prob)
        # print(uHs[t][s].T,prob.T)
        res = np.random.choice(range(len(human_actions)),p=prob.reshape((len(human_actions))))
        return human_actions[res]

    # Construct robot policy according to type
    pol_R = None
    if player_types[1] == "adv":
        if player_types[0] == "gift":
            uRs = solns["r_adv_actions"]
        else:
            uRs = get_adv_pol(game, uHs)
        pol_R = lambda t,s: robot_actions[np.argmax(uRs[t][s])]
    elif player_types[1] == "coop":
        if player_types[0] == "gift":
            uRs = solns["r_coop_actions"]
        else:
            uRs = get_coop_pol(game, uHs)
        pol_R = lambda t,s: robot_actions[np.argmax(uRs[t][s])]
    elif player_types[1] == "random":
        pol_R = lambda t,s: robot_actions[np.random.choice(range(len(robot_actions)))]
    else:
        raise Exception("Opponent type not available, select from adv, coop, random.")

    # Simulate
    v_funs_adv, uRs_adv = solns["baseline_val"], solns["r_adv_actions"]
    # verbose = True
    if verbose: print(player_types)
    if player_types[0] == "gift":
        # Simulate policy
        # print("Testing with " + str(player_types) + " agent")
        s = s0
        if verbose: print(s)
        total_r = 0
        for t in range(T):
            # print(uHs[t].keys())
            uH_strat = uHs[t][s]
            uH = pol_H(t,s)
            uR = pol_R(t,s)
            uR_adv = robot_actions[np.argmax(uRs_adv[t][s])]
            curr_r = reward(s, uH, uR)[0]
            total_r += curr_r

            # Calculate dr in expectation over my strategy
            dr = -v_funs_adv[t][s[0]]
            print(dr)
            for i in range(len(human_actions)):
                h = human_actions[i]
                val = reward(s, h, uR)[0] + gamma*v_funs_adv[t+1][transition(s,h,uR,0,is_gift=False)[0]]
                print(val, h, uH_strat[i,0])
                dr += uH_strat[i,0]*(val)

            s_next = transition(s,uH,uR,dr)

            # Calculate amount risked in expectation over strategy against an adversary
            rsk = -v_funs_adv[t][s[0]] + s[1]
            for i in range(len(human_actions)):
                h = human_actions[i]
                rsk += uH_strat[i,0]*(reward(s, h, uR_adv)[0] + gamma*v_funs_adv[t+1][transition(s,h,uR_adv,0,is_gift=False)[0]])

            if verbose: print("state=",s_next,", uH=", uH, ", strat=", uH_strat.T, ", uR=", uR,", dr=", dr, ", r+dr=", s[1]+dr, ", constraint=",rsk, ", v_adv=", v_funs_adv[t][s[0]])
            # for i in range(len(human_actions)):
            #     h = human_actions[i]
            #     q_h = reward(s, h, uR_adv)[0] + gamma*v_funs_adv[t+1][transition(s,h,uR_adv,0)[0]]
            #     print(str(h),"=", q_h+s[1], end=", ")
            if verbose: print("RHS=", v_funs_adv[t][s[0]])
            if verbose: print()
            s = s_next
    else:
        # Simulate policy
        # print("Testing with " + str(player_types) + " agent")
        s = s0[0]
        if verbose: print(s)
        total_r = 0
        for t in range(T):
            uH_strat = uHs[t][s]
            uH = pol_H(t,s)
            uR = pol_R(t,s)
            curr_r = reward((s,0), uH, uR)
            if verbose: print(curr_r)
            total_r += curr_r[0]

            s_next = transition((s,0),uH,uR,0,is_gift=False)[0]

            if verbose: print("state=",s_next,", uH=", uH, ", strat=", uH_strat.T, ", uR=", uR, "total_r=", total_r)
            s = s_next

    if verbose: print()

    res = {
        "total_reward":total_r, "baseline_reward":v_funs_adv[0][s0[0]]
    }
    
    if player_types[0] == "gift":
        res["initial_risk"] = s0[1]

    return res

def simulate_old(s0, game, solns, opponent_type, seed=None, output=True):
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


    if output: print("total_reward=",total_r)
    # if output: print()

    # res = {
    #     "total_reward":total_r, "baseline_reward":v_funs_adv[0][s0], "initial_risk":s0[1]
    # }

    # return res

def compute_stats(n, s0, game, solns, player_types, seed=None):
    np.random.seed(seed)
    average_total_reward = 0
    ress = []
    baseline_reward = solns["baseline_val"][0][s0[0]]
    rewards = []

    # Run n simulations
    for _ in range(n):
        res = simulate(s0, game, solns, player_types, seed=None, verbose=False)
        rewards.append(res["total_reward"])
        ress.append(res)

    # Compute statistics
    stats = {"average_reward":np.average(rewards),"std_reward":np.std(rewards), "baseline_reward":baseline_reward, "rewards":rewards}

    return stats

