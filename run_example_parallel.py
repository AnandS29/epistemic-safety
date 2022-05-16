from cProfile import label
import numpy as np
import matplotlib.pyplot as plt
import itertools
import cvxpy as cp
import pickle
import manygrid
import time

from compute_solns import *
from grid_examples import *

import argparse


# examples = [(ex1, "manygrid_example_a_10",1,10), 
#             (ex2, "manygrid_example_b_10",1,10),
#             (ex1, "manygrid_example_c_10",1,10)]

examples = [
    (delay_empowerment, "delay_empowerment_parallel", 1, 30),
    (increasing_empowerment, "increasing_empowerment", 1, 30),
    (spend_and_reduce, "spend_and_reduce", 1, 25),
    (twice_empowered, "twice_empowered", 1, 30),
    (gain, "gain", 1, 10),
    (gain_1, "gain_1", 1, 20),
    (gain_2, "gain_2", 1, 20),
    (gain_3, "gain_3", 1, 30),
    (reduce, "reduce", 1, 15),
    (reduce_1, "reduce_1", 1, 20),
    (reduce_2, "reduce_2", 1, 20),
    (reduce_3, "reduce_3", 1, 25),
    (spend, "spend", 1, 10),
    (spend_1, "spend_1", 1, 15),
    (spend_2, "spend_2", 1, 20),
    (spend_3, "spend_3", 1, 20),
    (goods_robot_disempowers_robot_regularized, "goods_robot_disempowers_robot_regularized",1,10),
    (goods_robot_disempowers_robot, "goods_robot_disempowers_robot",1,10),
    (goods_robot_disempowers_robots_ability_to_disempower, "goods_robot_disempowers_robots_ability_to_disempower",1,15),
    (goods_robot_empowers_human, "goods_robot_empowers_human",1,10),
    (goods_robot_reward, "goods_robot_reward",1,5),
    (bads_robot_punish, "bads_robot_punish",1,5),
    (bads_robot_empowers_robot_ability_to_disempower, "bads_robot_empowers_robot_ability_to_disempower",1,15),
    (bads_robot_empowers_robot, "bads_robot_empowers_robot",1,10),
    (bads_robot_disempowers_human, "bads_robot_disempowers_human",1,10),
]

# for (ex, name, gamma, T) in examples:
#     print(name)
#     ex()

parser = argparse.ArgumentParser()
parser.add_argument('--exp', '-e', type=int, required=True)
args = parser.parse_args()

if not(args.exp >= 0 and args.exp < len(examples)):
    raise ValueError("Invalid experiment number")

# Choose example
# Grid, name, gamma, T
example = examples[args.exp]

# Set up game
grid = example[0]()
gamma, T = example[2], example[3]
game = manygrid.get_game(grid,gamma,T)
initial_state = grid.get_state()

# Run experiments
name = example[1]
start = time.time()
manygrid.run_exp(grid, game, True, T=T, name=name)
end = time.time()
print("Total Time for experiment:", end-start)

print("Running simulation...")
start = time.time()
res = manygrid.run_sim_single(initial_state, grid, 5, T=T, name=name)
with open("sim_results/"+name+'.pickle', 'wb') as handle:
    pickle.dump(res, handle, protocol=pickle.HIGHEST_PROTOCOL)
end = time.time()
print("Total time for simulation:", end-start)