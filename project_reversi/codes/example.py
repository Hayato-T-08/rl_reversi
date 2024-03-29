import gym
import random
import numpy as np
import gym_reversi
env = gym.make('Reversi8x8-v0')
env.reset()
for i_episode in range(20):
    observation = env.reset()
    for t in range(100):
        enables = env.possible_actions
        # if nothing to do ,select pass
        if len(enables)==0:
            action = env.board_size**2 + 1
        # random select (update learning method here)
        else:
            action = random.choice(enables)
        observation, reward, done, info = env.step(action)
        print(observation)
        print('===================')
        env.render(mode='human')
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            black_score = len(np.where(env.state[0,:,:]==1)[0])
            print(black_score)
            break