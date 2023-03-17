import sys
import sklearn
import tensorflow as tf
from tensorflow import keras

import numpy as np
import os
np.random.seed(42)
tf.random.set_seed(42)

import matplotlib.pyplot as plt
import gym
import gym_reversi


env=gym.make('Reversi8x8-v0')
env.reset()
input_shape=(3,8,8)
n_outputs=66
model=keras.models.Sequential([
    keras.layers.Conv2D(32, kernel_size=(2, 2), activation='relu', input_shape=input_shape),
    keras.layers.Conv2D(64, kernel_size=(2, 2), activation='relu'),
    keras.layers.Flatten(),
    keras.layers.Dense(n_outputs, activation='softmax')
])

model.load_weights('reversi-AI.hdf5')

def epsilon_greedy_policy(state, epsilon=0):
    if np.random.rand() < epsilon:
        return np.random.randint(n_outputs)
    else:
        Q_values = model.predict(state[np.newaxis])
        return np.argmax(Q_values[0])
env.seed(42)
state = env.reset()

frames = []

all_reward=0
for i in range(100):
    env.reset()
    while(True):
        action = epsilon_greedy_policy(state)
        state, reward, done, info = env.step(action)
        all_reward+=reward
        if done:
            break
        env.render(mode="human")


print(all_reward)
