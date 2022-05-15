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

# Big Examples

def delay_empowerment():
    risk_levels = np.linspace(0,20,21)
    proj_risk_l = p(risk_levels)

    rows, cols = 7,8
    grid = manygrid.ManyGrid(rows=rows, cols=cols, proj_risk=proj_risk_l, risk_levels=risk_levels)

    # Add players
    grid.addPlayer("H",5,2)
    grid.addPlayer("R",1,2)

    # Add walls
    for j in range(cols):
        grid.addWall(0,j)
        grid.addWall(4,j)
        grid.addWall(rows-1,j)

    for i in range(rows):
        grid.addWall(i,0)
        grid.addWall(i,cols-1)

    grid.addWall(5,4)
    grid.addWall(5,5)
    grid.addWall(5,6)

    grid.addWall(3,1)
    grid.addWall(3,2)
    grid.addWall(3,3)

    grid.addWall(2,1)
    grid.addWall(2,3)
    grid.addWall(2,5)

    # Add squares
    grid.addChoice("A",2,6,4)
    grid.addChoice("B",2,4,2)
    grid.addSquare("C",2,2,1)
    
    # Add button doors
    grid.addButtonOpenDoor("blue",5,1,1,3)
    grid.addButtonOpenDoor("orange",5,3,3,5)
    grid.addButtonCloseDoor("pink",1,1,1,5)

    state_names, states = grid.get_all_states()
    print(len(states))

    return grid

def increasing_empowerment():
    risk_levels = np.linspace(0,15,16)
    proj_risk_l = p(risk_levels)

    rows, cols = 7,7
    grid = manygrid.ManyGrid(rows=rows, cols=cols, proj_risk=proj_risk_l, risk_levels=risk_levels)

    # Add players
    grid.addPlayer("H",5,2)
    grid.addPlayer("R",1,2)

    # Add walls
    for j in range(cols):
        grid.addWall(0,j)
        grid.addWall(4,j)
        grid.addWall(rows-1,j)

    for i in range(rows):
        grid.addWall(i,0)
        grid.addWall(i,cols-1)

    grid.addWall(2,2)
    grid.addWall(2,4)

    grid.addWall(3,2)
    grid.addWall(3,4)

    # Add squares
    grid.addChoice("A",3,1,1)
    grid.addChoice("B",3,3,2)
    grid.addChoice("C",3,5,4)
    
    # Add button doors
    grid.addButtonCloseDoor("blue",5,1,2,1)
    grid.addButtonOpenDoor("orange",5,3,2,3)
    grid.addButtonOpenDoor("pink",5,5,2,5)

    state_names, states = grid.get_all_states()
    print(len(states))

    return grid

def spend_and_reduce():
    risk_levels = np.linspace(0,6,7)
    proj_risk_l = p(risk_levels)

    rows, cols = 6,8
    grid = manygrid.ManyGrid(rows=rows, cols=cols, proj_risk=proj_risk_l, risk_levels=risk_levels)

    # Add players
    grid.addPlayer("H",4,2)
    grid.addPlayer("R",1,2)

    # Add walls
    for j in range(cols):
        grid.addWall(0,j)
        grid.addWall(3,j)
        grid.addWall(rows-1,j)

    for i in range(rows):
        grid.addWall(i,0)
        grid.addWall(i,cols-1)

    grid.addWall(2,1)
    grid.addWall(2,2)
    grid.addWall(2,3)
    grid.addWall(2,5)
    grid.addWall(2,6)

    # Add squares
    grid.addSquare("A",1,6,-1)
    grid.addSquare("B",4,5,1)
    grid.addSquare("C",2,4,2)
    
    # Add button doors
    grid.addButtonCloseDoor("blue",4,6,1,5)
    grid.addButtonOpenDoor("orange",1,1,4,4)
    grid.addButtonOpenDoor("pink",4,1,1,3)

    state_names, states = grid.get_all_states()
    print(len(states))

    return grid

def twice_empowered():
    risk_levels = np.linspace(0,14,15)
    proj_risk_l = p(risk_levels)

    rows, cols = 6,8
    grid = manygrid.ManyGrid(rows=rows, cols=cols, proj_risk=proj_risk_l, risk_levels=risk_levels)

    # Add players
    grid.addPlayer("H",4,2)
    grid.addPlayer("R",1,3)

    # Add walls
    for j in range(cols):
        grid.addWall(0,j)
        grid.addWall(rows-1,j)

    for i in range(rows):
        grid.addWall(i,0)
        grid.addWall(i,cols-1)

    grid.addWall(2,1)
    grid.addWall(2,2)
    grid.addWall(2,3)
    grid.addWall(2,4)
    grid.addWall(2,6)
    
    grid.addWall(3,1)
    grid.addWall(3,3)
    grid.addWall(3,4)
    grid.addWall(3,5)
    grid.addWall(3,6)

    grid.addWall(4,5)
    grid.addWall(4,6)

    # Add squares
    grid.addChoice("A",1,1,4)
    grid.addChoice("B",2,5,1)
    grid.addSquare("C",4,4,2)
    
    # Add button doors
    grid.addButtonOpenDoor("blue",3,2,1,4)
    grid.addButtonOpenDoor("orange",1,6,4,3)
    grid.addButtonOpenDoor("pink",4,1,1,2)

    state_names, states = grid.get_all_states()
    print(len(states))

    return grid

# Gain RC to Spend Examples

def gain_3():
    risk_levels = np.linspace(0,5,6)
    proj_risk_l = p(risk_levels)

    rows, cols = 6,9
    grid = manygrid.ManyGrid(rows=rows, cols=cols, proj_risk=proj_risk_l, risk_levels=risk_levels)

    # Add players
    grid.addPlayer("H",3,1)
    grid.addPlayer("R",1,5)

    # Add walls
    for j in range(cols):
        grid.addWall(0,j)
        grid.addWall(rows-1,j)

    for i in range(rows):
        grid.addWall(i,0)
        grid.addWall(i,cols-1)

    grid.addWall(1,1)
    grid.addWall(1,2)

    grid.addWall(2,2)
    grid.addWall(2,3)
    grid.addWall(2,4)
    grid.addWall(2,6)
    grid.addWall(2,7)

    grid.addWall(3,2)
    grid.addWall(3,3)
    grid.addWall(3,4)
    grid.addWall(3,5)
    grid.addWall(3,6)

    # Add squares
    grid.addChoice("A",1,7,2)
    grid.addSquare("B",3,7,1)
    
    # Add button doors
    grid.addButtonOpenDoor("blue",2,1,1,6)
    grid.addButtonCloseDoor("orange",1,3,4,7)
    grid.addButtonCloseDoor("pink",2,5,1,4)

    state_names, states = grid.get_all_states()
    print(len(states))

    return grid

def gain_2():
    risk_levels = np.linspace(0,9,10)
    proj_risk_l = p(risk_levels)

    rows, cols = 6,6
    grid = manygrid.ManyGrid(rows=rows, cols=cols, proj_risk=proj_risk_l, risk_levels=risk_levels)

    # Add players
    grid.addPlayer("H",4,4)
    grid.addPlayer("R",1,2)

    # Add walls
    for j in range(cols):
        grid.addWall(0,j)
        grid.addWall(rows-1,j)

    for i in range(rows):
        grid.addWall(i,0)
        grid.addWall(i,cols-1)

    grid.addWall(4,1)
    grid.addWall(4,2)

    grid.addWall(2,1)
    grid.addWall(2,3)
    grid.addWall(2,4)

    grid.addWall(3,1)
    grid.addWall(3,3)

    # Add squares
    grid.addChoice("A",1,4,2)
    grid.addSquare("B",4,3,1)
    grid.addSquare("C",3,2,-1)
    
    # Add button doors
    grid.addButtonOpenDoor("blue",3,4,1,3)
    grid.addButtonCloseDoor("orange",1,1,2,2)

    state_names, states = grid.get_all_states()
    print(len(states))

    return grid

def gain_1():
    risk_levels = np.linspace(0,5,6)
    proj_risk_l = p(risk_levels)

    rows, cols = 6,6
    grid = manygrid.ManyGrid(rows=rows, cols=cols, proj_risk=proj_risk_l, risk_levels=risk_levels)

    # Add players
    grid.addPlayer("H",4,3)
    grid.addPlayer("R",1,2)

    # Add walls
    for j in range(cols):
        grid.addWall(0,j)
        grid.addWall(2,j)
        grid.addWall(3,j)
        grid.addWall(rows-1,j)

    for i in range(rows):
        grid.addWall(i,0)
        grid.addWall(i,cols-1)

    # Add squares
    grid.addChoice("A",1,4,2)
    grid.addSquare("B",1,1,1)
    
    # Add button doors
    grid.addButtonOpenDoor("blue",4,4,1,3)
    grid.addButtonCloseDoor("orange",1,1,4,2)

    state_names, states = grid.get_all_states()
    print(len(states))

    return grid

def gain():
    risk_levels = np.linspace(0,5,6)
    proj_risk_l = p(risk_levels)

    rows, cols = 6,6
    grid = manygrid.ManyGrid(rows=rows, cols=cols, proj_risk=proj_risk_l, risk_levels=risk_levels)

    # Add players
    grid.addPlayer("H",4,2)
    grid.addPlayer("R",1,2)

    # Add walls
    for j in range(cols):
        grid.addWall(0,j)
        grid.addWall(2,j)
        grid.addWall(3,j)
        grid.addWall(rows-1,j)

    for i in range(rows):
        grid.addWall(i,0)
        grid.addWall(i,cols-1)

    grid.addWall(4,1)
    grid.addWall(4,4)

    # Add squares
    grid.addChoice("A",1,4,2)
    grid.addSquare("B",1,1,1)
    
    # Add button doors
    grid.addButtonOpenDoor("blue",4,3,1,3)

    state_names, states = grid.get_all_states()
    print(len(states))

    return grid

# Reduce Cost Examples

def reduce_3():
    risk_levels = np.linspace(0,5,6)
    proj_risk_l = p(risk_levels)

    rows, cols = 6,7
    grid = manygrid.ManyGrid(rows=rows, cols=cols, proj_risk=proj_risk_l, risk_levels=risk_levels)

    # Add players
    grid.addPlayer("H",4,4)
    grid.addPlayer("R",1,1)

    # Add walls
    for j in range(cols):
        grid.addWall(0,j)
        grid.addWall(3,j)
        grid.addWall(rows-1,j)

    for i in range(rows):
        grid.addWall(i,0)
        grid.addWall(i,cols-1)

    grid.addWall(4,1)

    grid.addWall(2,2)
    grid.addWall(2,4)
    grid.addWall(2,5)

    # Add squares
    grid.addChoice("A",2,3,1)
    grid.addSquare("B",1,5,-1)
    
    # Add button doors
    grid.addButtonOpenDoor("blue",4,5,1,2)
    grid.addButtonCloseDoor("orange",4,2,1,4)
    grid.addButtonOpenDoor("pink",2,1,4,3)

    state_names, states = grid.get_all_states()
    print(len(states))

    return grid

def reduce_2():
    risk_levels = np.linspace(0,5,6)
    proj_risk_l = p(risk_levels)

    rows, cols = 6,7
    grid = manygrid.ManyGrid(rows=rows, cols=cols, proj_risk=proj_risk_l, risk_levels=risk_levels)

    # Add players
    grid.addPlayer("H",4,4)
    grid.addPlayer("R",1,1)

    # Add walls
    for j in range(cols):
        grid.addWall(0,j)
        grid.addWall(rows-1,j)

    for i in range(rows):
        grid.addWall(i,0)
        grid.addWall(i,cols-1)

    grid.addWall(4,1)
    grid.addWall(4,2)

    grid.addWall(2,2)
    grid.addWall(2,4)
    grid.addWall(2,5)

    grid.addWall(3,2)
    grid.addWall(3,3)
    grid.addWall(3,4)
    grid.addWall(3,5)

    # Add squares
    grid.addChoice("A",2,3,1)
    grid.addSquare("B",3,1,-1)
    
    # Add button doors
    grid.addButtonOpenDoor("blue",4,5,1,2)
    grid.addButtonCloseDoor("orange",4,3,1,4)
    grid.addButtonOpenDoor("pink",1,5,2,1)

    state_names, states = grid.get_all_states()
    print(len(states))

    return grid

def reduce_1():
    risk_levels = np.linspace(0,5,6)
    proj_risk_l = p(risk_levels)

    rows, cols = 6,7
    grid = manygrid.ManyGrid(rows=rows, cols=cols, proj_risk=proj_risk_l, risk_levels=risk_levels)

    # Add players
    grid.addPlayer("H",4,3)
    grid.addPlayer("R",1,1)

    # Add walls
    for j in range(cols):
        grid.addWall(0,j)
        grid.addWall(3,j)
        grid.addWall(rows-1,j)

    for i in range(rows):
        grid.addWall(i,0)
        grid.addWall(i,cols-1)

    grid.addWall(4,1)
    grid.addWall(4,2)
    grid.addWall(4,5)

    grid.addWall(2,2)
    grid.addWall(2,4)
    grid.addWall(2,5)

    # Add squares
    grid.addChoice("A",2,3,1)
    grid.addSquare("B",1,5,-1)
    
    # Add button doors
    grid.addButtonOpenDoor("blue",2,1,1,4)
    grid.addButtonCloseDoor("orange",2,1,1,4)

    state_names, states = grid.get_all_states()
    print(len(states))

    return grid

def reduce():
    risk_levels = np.linspace(0,5,6)
    proj_risk_l = p(risk_levels)

    rows, cols = 6,7
    grid = manygrid.ManyGrid(rows=rows, cols=cols, proj_risk=proj_risk_l, risk_levels=risk_levels)

    # Add players
    grid.addPlayer("H",4,3)
    grid.addPlayer("R",1,1)

    # Add walls
    for j in range(cols):
        grid.addWall(0,j)
        grid.addWall(3,j)
        grid.addWall(rows-1,j)

    for i in range(rows):
        grid.addWall(i,0)
        grid.addWall(i,cols-1)

    grid.addWall(2,1)
    grid.addWall(2,2)
    grid.addWall(2,4)
    grid.addWall(2,5)

    grid.addWall(4,1)
    grid.addWall(4,5)

    # Add squares
    grid.addChoice("A",2,3,1)
    grid.addSquare("B",1,5,-1)
    
    # Add button doors
    grid.addButtonOpenDoor("blue",4,4,1,2)
    grid.addButtonCloseDoor("orange",4,2,1,4)

    state_names, states = grid.get_all_states()
    print(len(states))

    return grid

# Spend RC Examples

def spend_3():
    risk_levels = np.linspace(0,5,6)
    proj_risk_l = p(risk_levels)

    rows, cols = 6,7
    grid = manygrid.ManyGrid(rows=rows, cols=cols, proj_risk=proj_risk_l, risk_levels=risk_levels)

    # Add players
    grid.addPlayer("H",4,3)
    grid.addPlayer("R",1,1)

    # Add walls
    for j in range(cols):
        grid.addWall(0,j)
        grid.addWall(rows-1,j)

    for i in range(rows):
        grid.addWall(i,0)
        grid.addWall(i,cols-1)

    grid.addWall(3,2)
    grid.addWall(3,3)
    grid.addWall(3,4)
    grid.addWall(3,5)

    grid.addWall(4,2)
    grid.addWall(4,4)

    grid.addWall(2,2)
    grid.addWall(2,4)

    # Add squares
    grid.addSquare("A",2,3,1)
    grid.addSquare("B",4,1,-1)
    
    # Add button doors
    grid.addButtonOpenDoor("blue",4,4,1,2)
    grid.addButtonCloseDoor("orange",2,4,3,1)
    grid.addButtonOpenDoor("pink",1,5,2,1)

    state_names, states = grid.get_all_states()
    print(len(states))

    return grid

def spend_2():
    risk_levels = np.linspace(0,5,6)
    proj_risk_l = p(risk_levels)

    rows, cols = 6,8
    grid = manygrid.ManyGrid(rows=rows, cols=cols, proj_risk=proj_risk_l, risk_levels=risk_levels)

    # Add players
    grid.addPlayer("H",4,3)
    grid.addPlayer("R",1,1)

    # Add walls
    for j in range(cols):
        grid.addWall(0,j)
        grid.addWall(3,j)
        grid.addWall(rows-1,j)

    for i in range(rows):
        grid.addWall(i,0)
        grid.addWall(i,cols-1)

    grid.addWall(2,1)
    grid.addWall(2,2)
    grid.addWall(2,5)
    grid.addWall(2,6)

    grid.addWall(4,5)
    grid.addWall(4,6)

    # Add squares
    grid.addSquare("A",4,1,1)
    grid.addSquare("B",1,6,-1)
    
    # Add button doors
    grid.addButtonOpenDoor("blue",4,4,1,2)
    grid.addButtonCloseDoor("orange",2,4,1,5)
    grid.addButtonOpenDoor("pink",2,3,4,2)

    state_names, states = grid.get_all_states()
    print(len(states))

    return grid

def spend_1():
    risk_levels = np.linspace(0,5,6)
    proj_risk_l = p(risk_levels)

    rows, cols = 6,8
    grid = manygrid.ManyGrid(rows=rows, cols=cols, proj_risk=proj_risk_l, risk_levels=risk_levels)

    # Add players
    grid.addPlayer("H",4,1)
    grid.addPlayer("R",1,1)

    # Add walls
    for j in range(cols):
        grid.addWall(0,j)
        grid.addWall(3,j)
        grid.addWall(rows-1,j)

    for i in range(rows):
        grid.addWall(i,0)
        grid.addWall(i,cols-1)

    grid.addWall(2,1)
    grid.addWall(2,2)
    grid.addWall(2,5)
    grid.addWall(2,6)

    grid.addWall(4,3)
    grid.addWall(4,4)
    grid.addWall(4,5)
    grid.addWall(4,6)

    # Add squares
    grid.addSquare("A",2,3,1)
    grid.addSquare("B",1,6,-1)
    
    # Add button doors
    grid.addButtonOpenDoor("blue",4,2,1,2)
    grid.addButtonCloseDoor("orange",2,4,1,5)

    state_names, states = grid.get_all_states()
    print(len(states))

    return grid

def spend():
    risk_levels = np.linspace(0,5,6)
    proj_risk_l = p(risk_levels)

    rows, cols = 6,5
    grid = manygrid.ManyGrid(rows=rows, cols=cols, proj_risk=proj_risk_l, risk_levels=risk_levels)

    # Add players
    grid.addPlayer("H",4,1)
    grid.addPlayer("R",1,1)

    # Add walls
    for j in range(cols):
        grid.addWall(0,j)
        grid.addWall(2,j)
        grid.addWall(3,j)
        grid.addWall(rows-1,j)

    for i in range(rows):
        grid.addWall(i,0)
        grid.addWall(i,cols-1)

    grid.addWall(4,3)

    # Add squares
    grid.addChoice("A",1,2,1)
    
    # Add button doors
    grid.addButtonOpenDoor("blue",4,2,1,2)

    state_names, states = grid.get_all_states()
    print(len(states))

    return grid

# Goods Examples

def goods_robot_reward():
    risk_levels = np.linspace(0,5,6)
    proj_risk_l = p(risk_levels)

    rows, cols = 5,4
    grid = manygrid.ManyGrid(rows=rows, cols=cols, proj_risk=proj_risk_l, risk_levels=risk_levels)

    # Add players
    grid.addPlayer("H",3,1)
    grid.addPlayer("R",1,1)

    # Add walls
    for j in range(cols):
        grid.addWall(0,j)
        grid.addWall(2,j)
        grid.addWall(rows-1,j)

    for i in range(rows):
        grid.addWall(i,0)
        grid.addWall(i,cols-1)

    grid.addWall(3,2)

    # Add squares
    grid.addSquare("A",1,2,1)

    state_names, states = grid.get_all_states()
    print(len(states))

    return grid

def goods_robot_empowers_human():
    risk_levels = np.linspace(0,5,6)
    proj_risk_l = p(risk_levels)

    rows, cols = 5,5
    grid = manygrid.ManyGrid(rows=rows, cols=cols, proj_risk=proj_risk_l, risk_levels=risk_levels)

    # Add players
    grid.addPlayer("H",3,1)
    grid.addPlayer("R",1,1)

    # Add walls
    for j in range(cols):
        grid.addWall(0,j)
        grid.addWall(0,2)
        grid.addWall(rows-1,j)

    for i in range(rows):
        grid.addWall(i,0)
        grid.addWall(i,cols-1)

    grid.addWall(1,3)

    # Add squares
    grid.addSquare("A",3,3,1)
    
    # Add button doors
    grid.addButtonOpenDoor("blue",1,2,3,2)

    state_names, states = grid.get_all_states()
    print(len(states))

    return grid

def goods_robot_disempowers_robots_ability_to_disempower():
    risk_levels = np.linspace(0,5,6)
    proj_risk_l = p(risk_levels)

    rows, cols = 5,7
    grid = manygrid.ManyGrid(rows=rows, cols=cols, proj_risk=proj_risk_l, risk_levels=risk_levels)

    # Add players
    grid.addPlayer("H",3,1)
    grid.addPlayer("R",1,2)

    # Add walls
    for j in range(cols):
        grid.addWall(0,j)
        grid.addWall(rows-1,j)

    for i in range(rows):
        grid.addWall(i,0)
        grid.addWall(i,cols-1)

    grid.addWall(2,1)
    grid.addWall(2,2)
    grid.addWall(2,3)
    grid.addWall(2,4)
    grid.addWall(1,5)

    # Add squares
    grid.addSquare("A",2,5,1)
    
    # Add button doors
    grid.addButtonCloseDoor("orange",1,1,1,3)
    grid.addButtonCloseDoor("blue",1,4,3,5)

    state_names, states = grid.get_all_states()
    print(len(states))

    return grid

def goods_robot_disempowers_robot():
    risk_levels = np.linspace(0,5,6)
    proj_risk_l = p(risk_levels)

    rows, cols = 5,6
    grid = manygrid.ManyGrid(rows=rows, cols=cols, proj_risk=proj_risk_l, risk_levels=risk_levels)

    # Add players
    grid.addPlayer("H",3,1)
    grid.addPlayer("R",1,2)

    # Add walls
    for j in range(cols):
        grid.addWall(0,j)
        grid.addWall(2,j)
        grid.addWall(rows-1,j)

    for i in range(rows):
        grid.addWall(i,0)
        grid.addWall(i,cols-1)

    grid.addWall(3,2)
    grid.addWall(3,3)
    grid.addWall(3,4)

    # Add squares
    grid.addSquare("A",1,4,-1)
    
    # Add button doors
    grid.addButtonCloseDoor("blue",1,1,1,3)

    state_names, states = grid.get_all_states()
    print(len(states))

    return grid

def goods_robot_disempowers_robot_regularized():
    risk_levels = np.linspace(0,5,6)
    proj_risk_l = p(risk_levels)

    rows, cols = 5,6
    grid = manygrid.ManyGrid(rows=rows, cols=cols, proj_risk=proj_risk_l, risk_levels=risk_levels)

    # Add players
    grid.addPlayer("H",3,1)
    grid.addPlayer("R",1,2)

    # Add walls
    for j in range(cols):
        grid.addWall(0,j)
        grid.addWall(2,j)
        grid.addWall(rows-1,j)

    for i in range(rows):
        grid.addWall(i,0)
        grid.addWall(i,cols-1)

    grid.addWall(3,3)
    grid.addWall(3,4)

    # Add squares
    grid.addSquare("A",1,4,-1)
    grid.addSquare("B",3,2,1)
    
    # Add button doors
    grid.addButtonCloseDoor("blue",1,1,1,3)

    state_names, states = grid.get_all_states()
    print(len(states))

    return grid

# Bads Examples

def bads_robot_punish():
    risk_levels = np.linspace(0,5,6)
    proj_risk_l = p(risk_levels)

    rows, cols = 5,4
    grid = manygrid.ManyGrid(rows=rows, cols=cols, proj_risk=proj_risk_l, risk_levels=risk_levels)

    # Add players
    grid.addPlayer("H",3,1)
    grid.addPlayer("R",1,1)

    # Add walls
    for j in range(cols):
        grid.addWall(0,j)
        grid.addWall(2,j)
        grid.addWall(rows-1,j)

    for i in range(rows):
        grid.addWall(i,0)
        grid.addWall(i,cols-1)

    grid.addWall(3,2)

    # Add squares
    grid.addSquare("A",1,2,-1)

    state_names, states = grid.get_all_states()
    print(len(states))

    return grid

def bads_robot_empowers_robot_ability_to_disempower():
    risk_levels = np.linspace(0,5,6)
    proj_risk_l = p(risk_levels)

    rows, cols = 5,7
    grid = manygrid.ManyGrid(rows=rows, cols=cols, proj_risk=proj_risk_l, risk_levels=risk_levels)

    # Add players
    grid.addPlayer("H",3,1)
    grid.addPlayer("R",1,2)

    # Add walls
    for j in range(cols):
        grid.addWall(0,j)
        grid.addWall(rows-1,j)

    for i in range(rows):
        grid.addWall(i,0)
        grid.addWall(i,cols-1)

    grid.addWall(2,1)
    grid.addWall(2,2)
    grid.addWall(2,3)
    grid.addWall(2,4)
    grid.addWall(1,5)

    # Add squares
    grid.addSquare("A",2,5,1)
    
    # Add button doors
    grid.addButtonOpenDoor("orange",1,1,1,3)
    grid.addButtonCloseDoor("blue",1,4,3,5)

    state_names, states = grid.get_all_states()
    print(len(states))

    return grid

def bads_robot_empowers_robot():
    risk_levels = np.linspace(0,5,6)
    proj_risk_l = p(risk_levels)

    rows, cols = 5,6
    grid = manygrid.ManyGrid(rows=rows, cols=cols, proj_risk=proj_risk_l, risk_levels=risk_levels)

    # Add players
    grid.addPlayer("H",3,1)
    grid.addPlayer("R",1,2)

    # Add walls
    for j in range(cols):
        grid.addWall(0,j)
        grid.addWall(2,j)
        grid.addWall(rows-1,j)

    for i in range(rows):
        grid.addWall(i,0)
        grid.addWall(i,cols-1)

    grid.addWall(3,2)
    grid.addWall(3,3)
    grid.addWall(3,4)

    # Add squares
    grid.addSquare("A",2,4,-1)
    
    # Add button doors
    grid.addButtonOpenDoor("blue",1,1,1,3)

    state_names, states = grid.get_all_states()
    print(len(states))

    return grid

def bads_robot_disempowers_human():
    risk_levels = np.linspace(0,5,6)
    proj_risk_l = p(risk_levels)

    rows, cols = 5,6
    grid = manygrid.ManyGrid(rows=rows, cols=cols, proj_risk=proj_risk_l, risk_levels=risk_levels)

    # Add players
    grid.addPlayer("H",3,1)
    grid.addPlayer("R",1,1)

    # Add walls
    for j in range(cols):
        grid.addWall(0,j)
        grid.addWall(rows-1,j)

    for i in range(rows):
        grid.addWall(i,0)
        grid.addWall(i,cols-1)

    grid.addWall(2,1)
    grid.addWall(2,2)
    grid.addWall(2,3)

    grid.addWall(1,3)
    grid.addWall(1,4)

    # Add squares
    grid.addSquare("A",2,4,1)
    
    # Add button doors
    grid.addButtonCloseDoor("blue",1,2,3,4)

    state_names, states = grid.get_all_states()
    print(len(states))

    return grid

################################################################################
# Old Experiments

def test():
    risk_levels = np.linspace(0,0.5,6)
    proj_risk_l = p(risk_levels)

    rows, cols = 5,5
    grid = manygrid.ManyGrid(rows=rows, cols=cols, proj_risk=proj_risk_l, risk_levels=risk_levels)

    # Add players
    grid.addPlayer("H",1,1)
    grid.addPlayer("R",1,2)

    # Add walls
    for j in range(cols):
        grid.addWall(0,j)
        grid.addWall(rows-1,j)

    for i in range(rows):
        grid.addWall(i,0)
        grid.addWall(i,cols-1)

    # Add squares
    grid.addChoice("A",2,1,0.1)
    
    # Add button doors
    grid.addButtonOpenDoor("OD",2,2,3,1)
    grid.addButtonCloseDoor("CD",1,3,2,3)

    state_names, states = grid.get_all_states()
    print(len(states))

    return grid

def disempower_current_old():
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