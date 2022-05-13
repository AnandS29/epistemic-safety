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
    (disempower_current, "disempower_current",1,15),
]

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
manygrid.run_exp(grid, game, False, T=T, name=name)
end = time.time()
print("Total Time:", end-start)