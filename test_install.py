import minerl 
import gym

"""
This script demonstrates how to use the MineRL BASALT environment.
This also helps test if the environment is set up correctly and can be rendered.
"""

env = gym.make('MineRLBasaltFindCave-v0')

obs = env.reset()
done = False
while not done:
    # Take a random action
    action = env.action_space.sample()
    # In BASALT environments, sending ESC action will end the episode
    # Lets not do that
    action["ESC"] = 0
    obs, reward, done, _ = env.step(action)
    print(obs['pov'].shape)
    env.render()