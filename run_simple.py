from cProfile import label
import numpy as np
import matplotlib.pyplot as plt
import itertools
import cvxpy as cp
import pickle
import manygrid

from compute_solns import *

## Helper functions

def make_grid():
    rows, cols = 5, 5
    grid = manygrid.ManyGrid(rows=rows, cols=cols, proj_risk=proj_risk_l, risk_levels=risk_levels)
    grid.addPlayer("H",1,2)
    grid.addPlayer("R",1,1)

    for i in [0,rows-1]:    
        for j in range(cols):
            grid.addWall(i,j)
    for i in range(rows):    
        for j in [0,cols-1]:
            grid.addWall(i,j)

    grid.addSquare(str(0.1),3,1,0.1)

    return grid

############################################################################################

# Set up game

risk_levels = np.linspace(0,2,21)
proj_risk_l = lambda r: proj_risk(r,risk_levels)
rows, cols = 6, 7
grid = make_grid()

state_names, states = grid.get_all_states()

human_actions = grid.actions
robot_actions = grid.actions

gamma = 1
T = 20
parallel = True

transition = grid.transition
reward = grid.reward

state_names, states = grid.get_all_states()
print(len(states))

name = "simple_example_20"
manygrid.run_exp(grid,parallel=parallel,T=T, name=name)

# 8mins, 78 mins, 35 mins, 70 mins
initial_eps = 0.5
grid.set_epsilon(initial_eps)
state = grid.get_state()

res = manygrid.run_sim_single(state, grid, 20, T=T, name=name)

ress = list(zip(*res.items()))
x,z = ress[0],ress[1]
x = [str(i) for i in x]
y = [i[0] for i in z]
c  = [i[1] for i in z]
plt.figure(figsize=(20,5))
plt.bar(x,y)
plt.errorbar(x, y, yerr=c, fmt="o", color="r")
plt.show()