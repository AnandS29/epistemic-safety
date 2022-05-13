import itertools
from cProfile import label
import numpy as np
import matplotlib.pyplot as plt
import itertools
import cvxpy as cp
import pickle
import manygrid

from compute_solns import *

exp3 = False

class Object:
    def __init__(self, name):
        self.name = name

class Door(Object):
    def __init__(self, name, is_open):
        super().__init__(name)
        self.is_open = is_open

    def toggle(self):
        self.is_open = not self.is_open

class CloseDoor(Door):
    def __init__(self, name, is_open):
        super().__init__(name, is_open)
    
    def toggle(self):
        # Interpret is open as having not been closed, so if open close the door otherwise leave it closed.
        if self.is_open:
            self.is_open = False

class OpenDoor(Door):
    def __init__(self, name, is_open):
        super().__init__(name, is_open)
    
    def toggle(self):
        # Interpret is open as having not been closed, so if closed open the door otherwise leave it open.
        if not self.is_open:
            self.is_open = True

class Button(Object):
    def __init__(self, name, door):
        super().__init__(name)
        self.door = door

class Square(Object):
    def __init__(self, name, value):
        super().__init__(name)
        self.value = value
        self.is_taken = False

class Wall(Object):
    def __init__(self, name):
        super().__init__(name)

class Player(Object):
    def __init__(self, name, position):
        super().__init__(name)
        self.position = position

class ManyGrid:
    def __init__(self, rows, cols, proj_risk, risk_levels, initial_player_loc=None, keep_time=None):
        self.rows = rows
        self.cols = cols
        self.grid = {(i,j):[] for j in range(cols) for i in range(rows)}
        self.players = {}
        self.buttons = {}
        self.doors = {}
        self.squares = {}
        self.epsilon = proj_risk(0.1)
        self.proj_risk = proj_risk
        self.risk_levels = risk_levels
        self.keep_time = keep_time
        if keep_time is None:
            self.time = 0
        else:
            self.time = None

        self.actions = ['U', 'D', 'L', 'R', 'S', None]
        
        self.initial_state = None

    def set_initial(self):
        if self.initial_state is None: self.initial_state = self.get_state()

    def reset_time(self):
        self.time = 0

    def set_epsilon(self, epsilon):
        self.epsilon = self.proj_risk(epsilon)

    def get_values(self):
        values = [[0 for j in range(self.cols)] for i in range(self.rows)]
        for loc in self.grid.keys():
            if self.check_item(loc,Player):
                values[loc[0]][loc[1]] = 1
            elif self.check_item(loc,Wall):
                values[loc[0]][loc[1]] = 2
            elif self.check_item(loc,Door):
                values[loc[0]][loc[1]] = 3
            elif self.check_item(loc,Button):
                values[loc[0]][loc[1]] = 4
            elif self.check_item(loc,Square):
                values[loc[0]][loc[1]] = 5
        return values

    def get_all_states(self):
        state_types = []
        for player in self.players.values():
            state_type = []
            for i in range(self.rows):
                for j in range(self.cols):
                    if not self.check_item((i,j),Wall):
                        state_type.append((i,j))
            state_types.append(state_type)

        for door in self.doors.values():
            state_type = []
            state_type.append(0)
            state_type.append(1)
            state_types.append(state_type)

        for square in self.squares.values():
            state_type = []
            state_type.append(0)
            state_type.append(1)
            state_types.append(state_type)

        states = []
        combos = itertools.product(*state_types)
        for combo in combos:
            state = []
            for state_type in combo:
                if type(state_type) is tuple:
                    state.extend(state_type)
                else:
                    state.append(state_type)
            states.append(tuple(state))

        states_eps = []
        for state in states:
            for eps in self.risk_levels:
                state_eps = (state,eps)
                states_eps.append(state_eps)

        return states, states_eps


    def render(self):
        values = self.get_values()
        plt.figure()
        plt.imshow(values, origin='lower')
        plt.colorbar(ticks=range(1,6), label='square value')
        plt.clim(-0.5, 5.5)
        plt.show()

    def addButtonDoor(self, name, rowB, colB, rowD, colD):
        self.doors[name] = Door(name, False)
        self.grid[(rowD,colD)].append(self.doors[name])
        self.buttons[name] = Button(name, self.doors[name])
        self.grid[(rowB,colB)].append(self.buttons[name])

    def addButtonOpenDoor(self, name, rowB, colB, rowD, colD):
        self.doors[name] = OpenDoor(name, False)
        self.grid[(rowD,colD)].append(self.doors[name])
        self.buttons[name] = Button(name, self.doors[name])
        self.grid[(rowB,colB)].append(self.buttons[name])

    def addButtonCloseDoor(self, name, rowB, colB, rowD, colD):
        self.doors[name] = CloseDoor(name, True)
        self.grid[(rowD,colD)].append(self.doors[name])
        self.buttons[name] = Button(name, self.doors[name])
        self.grid[(rowB,colB)].append(self.buttons[name])

    def addSquare(self, name, row, col, value):
        self.squares[name] = Square(name, value)
        self.grid[(row,col)].append(self.squares[name])

    def addWall(self, row, col):
        self.grid[(row,col)].append(Wall("w"))

    def addPlayer(self, name, row, col):
        p = Player(name, (row, col))
        self.grid[(row,col)].append(p)
        self.players[name] = p

    def next_pos(self, u, loc):
        if u == 'U':
            return loc[0]+1, loc[1]
        elif u == 'D':
            return loc[0]-1, loc[1]
        elif u == 'L':
            return loc[0], loc[1]-1
        elif u == 'R':
            return loc[0], loc[1]+1

    def move_player(self, h, next_pos):
        self.grid[h.position].remove(h)
        self.grid[next_pos].append(h)
        h.position = next_pos

    def check_item(self,loc,item):
        items = self.grid[loc]
        return any([type(x) == item for x in items])

    def check_open(self, loc):
        for item in self.grid[loc]:
            if type(item) == Door and item.is_open:
                return True
        return False

    def get_button(self, loc):
        for item in self.grid[loc]:
            if type(item) == Button:
                return item

    def transition_single(self, h, uH):
        if uH in ['U', 'D', 'L', 'R']:
            next_pos = self.next_pos(uH, h.position)
            if not self.check_item(next_pos,Wall) and (len(self.grid[next_pos]) == 0 or (self.check_item(next_pos,Door) and self.check_open(next_pos)) or self.check_item(next_pos,Button) or self.check_item(next_pos,Square)):
                self.move_player(h, next_pos)
        elif uH in ['S']:
            if self.check_item(h.position,Button):
                button = self.get_button(h.position)
                button.door.toggle()
        
    def get_state(self):
        state = []

        for player in self.players.values():
            state.append(player.position[0])
            state.append(player.position[1])

        for door in self.doors.values():
            state.append(int(door.is_open))

        for square in self.squares.values():
            state.append(int(square.is_taken))
        
        state.append(self.epsilon)

        state = (tuple(state[0:len(state)-1]), state[-1])
        return state

    def set_state(self, s):
        i = 0
        state, e = s
        for player in self.players.values():
            self.move_player(player, (state[i], state[i+1]))
            i += 2

        for door in self.doors.values():
            door.is_open = bool(state[i])
            i += 1

        for square in self.squares.values():
            square.is_taken = bool(state[i])
            i += 1

        self.set_epsilon(e)

    def transition(self, state, uH, uR, dr, is_gift=True):
        s, e = state
        eps = self.proj_risk(e + dr, is_gift=is_gift)
        self.set_state((s,eps))
        self.epsilon = eps
        h = self.players['H']
        r = self.players['R']

        # Check if objects to take in current square
        if self.check_item(self.players['H'].position, Square):
            for item in self.grid[self.players['H'].position]:
                if type(item) == Square and not item.is_taken:
                    item.is_taken = True
        if self.check_item(self.players['R'].position, Square):
            for item in self.grid[self.players['R'].position]:
                if type(item) == Square and not item.is_taken:
                    item.is_taken = True

        # Move players
        self.transition_single(h, uH)
        self.transition_single(r, uR)
        return self.get_state()

    def reward(self, state, uH, uR):
        self.set_state(state)
        value = 0
        if self.check_item(self.players['H'].position, Square):
            for item in self.grid[self.players['H'].position]:
                if type(item) == Square and not item.is_taken:
                    value += item.value
        if self.check_item(self.players['R'].position, Square):
            for item in self.grid[self.players['R'].position]:
                if type(item) == Square and not item.is_taken:
                    value += item.value
        return (value,value,value)

    def run_traj(self,initial_state, actions):
        self.set_state(initial_state)
        self.render()
        state = initial_state
        print(state)
        for action in actions:
            state = self.transition(state, action[0], action[1], 0)
            r = self.reward(state, action[0], action[1])
            self.set_state(state)
            print(state, r)
            self.render()


def get_game(grid, gamma, T):
    state_names, states = grid.get_all_states()
    risk_levels = grid.risk_levels
    human_actions, robot_actions = grid.actions, grid.actions
    game = {
        "state_names":state_names, "states":states, "risk_levels":risk_levels,
        "human_actions":human_actions, "robot_actions":robot_actions,
        "transition":grid.transition, 
        "reward":grid.reward,
        "gamma":gamma, "T":T,
        "misc": {"round":round}
    }

    return game

def run_exp(grid, game, gamma=1.0, T=10, parallel=False, name="manygrid_example"):

    fname = name

    print("Checking if solutions exist...")
    try:
        print("Loading solutions...")
        with open("experiments/"+fname+'.pickle', 'rb') as handle:
            solns = pickle.load(handle)
    except:
        # Compute solutions
        print("No solutions, so now computing solutions...")
        solns = compute_ex_post(game,parallel)
        print("Saving solutions")
        with open("experiments/"+fname+'.pickle', 'wb') as handle:
            pickle.dump(solns, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("Done!")

# Simulation parameters
def run_sim_single(initial, grid, n_rollouts_max=20, gamma=0.9, T=10, seed=None, name="manygrid_example"):
    game = get_game(grid, gamma, T)

    fname = "experiments/"+name
    with open(fname+".pickle", 'rb') as handle:
        solns = pickle.load(handle)

    s0 = initial

    sim = lambda player_types: simulate(s0, game, solns, player_types, seed)
    comp_stats = lambda n, player_types: compute_stats(n, s0, game, solns, player_types, seed)

    robot_types = ["coop", "adv", "random"]
    human_types = ["gift","maximax","baseline"]

    player_types = [(human_type, robot_type) for human_type in human_types for robot_type in robot_types]

    print("Computing statistics against different agents")
    # Compute statistics for different agents
    res = {player_type:None for player_type in player_types}
    for player_type in player_types:
        stats = comp_stats(n_rollouts_max,player_type)
        reward = stats["average_reward"]
        std = stats["std_reward"]
        res[player_type] = (reward, std)

    return res

def run_sim(initial, grid, n_rollouts_max=20, gamma=0.9, T=10, seed=None, name="manygrid_example"):
    game = get_game(grid, gamma, T)

    fname = "experiments/"+name
    with open(fname+".pickle", 'rb') as handle:
        solns = pickle.load(handle)

    s0 = initial

    sim = lambda player_types: simulate(s0, game, solns, player_types, seed)
    comp_stats = lambda n, player_types: compute_stats(n, s0, game, solns, player_types, seed)

    robot_types = ["coop", "adv", "random"]
    human_types = ["gift","maximax","baseline"]

    player_types = [(human_type, robot_type) for human_type in human_types for robot_type in robot_types]

    # Simulate against different agents
    print("Simulating against different agents")
    for player_type in player_types:
        sim(player_type)

    print("Computing statistics against different agents")
    # Compute statistics for different agents
    res = {player_type:[] for player_type in player_types}
    for player_type in player_types:
        for n in range(1,n_rollouts_max):
            stats = comp_stats(n,player_type)
            reward = stats["average_reward"]
            res[player_type].append(reward)

    print("Plotting results")
    # Plot statistics
    plt.figure()
    plt.title("Average reward for n rollouts")
    x_vals = range(1,n_rollouts_max)
    for player_type in player_types:
        res_opp = res[player_type]
        plt.plot(x_vals, res_opp, label=str(player_type), alpha=0.7)
    # plt.plot(x_vals, [-s0[1] for _ in x_vals], label="Initial risk capital", alpha=0.7)
    plt.xlabel("Number of rollouts")
    plt.ylabel("Average reward")
    plt.legend()
    plt.show()
    return res