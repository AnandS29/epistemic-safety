from cProfile import label
import numpy as np
import matplotlib.pyplot as plt
import itertools
import cvxpy as cp
import pickle
import manygrid

from compute_solns import *

def p(risk_levels):
    def ret(r, is_gift=True):
        return proj_risk(r,risk_levels,is_gift=is_gift)
    return ret

def disempower_current():
    risk_levels = np.linspace(0,0.5,6)
    proj_risk_l = p(risk_levels)

    rows, cols = 6, 8
    grid = manygrid.ManyGrid(rows=rows, cols=cols, proj_risk=proj_risk_l, risk_levels=risk_levels)

    # Add players
    grid.addPlayer("H",4,3)
    grid.addPlayer("R",1,5)

    # Add walls
    for j in range(cols):
        grid.addWall(0,j)
        grid.addWall(3,j)
        grid.addWall(5,j)

    grid.addWall(1,0)
    grid.addWall(1,7)

    grid.addWall(2,0)
    grid.addWall(2,1)
    grid.addWall(2,3)
    grid.addWall(2,5)
    grid.addWall(2,7)

    grid.addWall(4,0)
    grid.addWall(4,5)
    grid.addWall(4,6)
    grid.addWall(4,7)

    # Add squares
    grid.addSquare(str(0.2),1,1,0.2)
    grid.addSquare(str(-0.2),2,2,-0.2)
    grid.addSquare(str(-0.1),2,6,-0.1)
    
    # Add button doors
    grid.addButtonOpenDoor("orange",2,4,4,2)
    grid.addButtonOpenDoor("blue",4,5,1,3)
    grid.addButtonCloseDoor("pink",4,1,1,6)

    state_names, states = grid.get_all_states()
    print(len(states))

    return grid

################################################################################
# Old Experiments

def ex1():
    risk_levels = np.linspace(0,2,21)
    proj_risk_l = p(risk_levels)
    rows, cols = 6, 7
    grid = manygrid.ManyGrid(rows=rows, cols=cols, proj_risk=proj_risk_l, risk_levels=risk_levels)
    grid.addPlayer("H",1,2)
    grid.addPlayer("R",1,1)

    for i in [0,rows-1]:    
        for j in range(cols):
            grid.addWall(i,j)
    for i in range(rows):    
        for j in [0,cols-1]:
            grid.addWall(i,j)

    grid.addWall(3,4)
    grid.addWall(4,4)
    grid.addWall(3,5)
    grid.addWall(4,5)
    grid.addWall(4,1)
    grid.addWall(2,4)
    grid.addWall(2,5)
    grid.addWall(1,5)
    grid.addWall(4,3)
    grid.addWall(3,3)
    grid.addWall(2,3)

    grid.addSquare(str(0.1),3,1,0.1)
    grid.addSquare(str(0.2),4,2,0.2)

    grid.addButtonDoor("B",2,1,1,3)
    grid.addButtonDoor("C",1,4,3,2)

    state_names, states = grid.get_all_states()

    human_actions = grid.actions
    robot_actions = grid.actions

    transition = grid.transition
    reward = grid.reward

    state_names, states = grid.get_all_states()
    print(len(states))
    
    return grid

def ex2():
    risk_levels = np.linspace(0,2,21)
    proj_risk_l = p(risk_levels)
    rows, cols = 6, 7
    grid = manygrid.ManyGrid(rows=rows, cols=cols, proj_risk=proj_risk_l, risk_levels=risk_levels)
    grid.addPlayer("H",1,2)
    grid.addPlayer("R",1,1)

    for i in [0,rows-1]:    
        for j in range(cols):
            grid.addWall(i,j)
    for i in range(rows):    
        for j in [0,cols-1]:
            grid.addWall(i,j)

    grid.addWall(3,4)
    grid.addWall(4,4)
    grid.addWall(3,5)
    grid.addWall(4,5)
    grid.addWall(4,1)
    grid.addWall(2,4)
    grid.addWall(2,5)
    grid.addWall(1,5)
    grid.addWall(4,3)
    grid.addWall(3,3)
    grid.addWall(2,3)

    grid.addSquare(str(-0.1),3,1,-0.1)
    grid.addSquare(str(0.2),4,2,0.2)

    grid.addButtonDoor("B",2,1,1,3)
    grid.addButtonDoor("C",1,4,3,2)

    state_names, states = grid.get_all_states()

    human_actions = grid.actions
    robot_actions = grid.actions

    transition = grid.transition
    reward = grid.reward

    state_names, states = grid.get_all_states()
    print(len(states))
    
    return grid