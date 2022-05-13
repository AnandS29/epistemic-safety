import gym
import time
from gym.envs.registration import register


r_type = "adv"

if r_type == "adv":
    register(
        id='multigrid-adv',
        entry_point='gym_multigrid.envs:CoopGameAdv',
    )
    env = gym.make('multigrid-adv')
elif r_type == "coop":
    register(
        id='multigrid-coop',
        entry_point='gym_multigrid.envs:CoopGameCoop',
    )
    env = gym.make('multigrid-coop')
else:
    raise ValueError("r_type must be either 'adv' or 'coop'")

_ = env.reset()

nb_agents = len(env.agents)

while True:
    env.render(mode='human', highlight=True)
    time.sleep(0.1)

    ac = [env.action_space.sample() for _ in range(nb_agents)]

    obs, _, done, _ = env.step(ac)

    if done:
        break