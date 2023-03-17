import gym
import random
import numpy as np
import creversi.gym_reversi
from gym.spaces import *
env = gym.make('Reversi-v0').unwrapped
env.reset()

def print_spaces(label,space):

    print(label,space)

    if isinstance(space, Box):
       print('    最小値: ', space.low)
       print('    最大値: ', space.high)
    if isinstance(space, Discrete):
       print('    最小値: ', 0)
       print('    最大値: ', space.n-1)

print_spaces('状態空間: ', env.observation_space)
print_spaces('行動空間: ', env.action_space)