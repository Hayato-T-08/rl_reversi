import sys
import sklearn
import tensorflow as tf
from tensorflow import keras

import numpy as np
import os
import random
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

def random_policy(state, player_color=1):
    possible_places = env.get_possible_actions(state, player_color)
    # No places left
    if len(possible_places) == 0:
        return 8**2 + 1
    a=epsilon_greedy_policy(state=state,epsilon=0)
    return possible_places[a]

def epsilon_greedy_policy(state, epsilon=0):
    possible_places = env.get_possible_actions(state, 1)
    if len(possible_places)==0:
        return 8**2+1
    
    if np.random.rand() < epsilon:
        return np.random.randint(0,len(possible_places))
    else:
        Q_values = model.predict(state[np.newaxis])
        print(Q_values)

        max_q=Q_values[0][possible_places[0]]
        idx=0
        i=0
        for item in possible_places:
            if(max_q<Q_values[0][item]):
                max_q=Q_values[0][item]
                print(Q_values[0][item])
                idx=i
            i+=1
        print(possible_places)
        print(max_q)
        print(idx)
        return possible_places[idx]


from collections import deque

replay_memory = deque(maxlen=2000)

def sample_experiences(batch_size):
    indices = np.random.randint(len(replay_memory), size=batch_size)
    batch = [replay_memory[index] for index in indices]
    states, actions, rewards, next_states, dones = [
        np.array([experience[field_index] for experience in batch])
        for field_index in range(5)]
    return states, actions, rewards, next_states, dones

def play_one_step(env, state, epsilon):
    action = epsilon_greedy_policy(state,epsilon)
    next_state, reward, done, info = env.step(action)
    replay_memory.append((state, action, reward, next_state, done))
    return next_state, reward, done, info

batch_size = 32
discount_rate = 0.99
optimizer = keras.optimizers.Adam(learning_rate=1e-2)
loss_fn = keras.losses.mean_squared_error

def training_step(batch_size):
    experiences = sample_experiences(batch_size)
    states, actions, rewards, next_states, dones = experiences
    next_Q_values = model.predict(next_states)
    max_next_Q_values = np.max(next_Q_values, axis=1)
    #print(max_next_Q_values)
    target_Q_values = (rewards +
                       (1 - dones) * discount_rate * max_next_Q_values)
    target_Q_values = target_Q_values.reshape(-1, 1)
    mask = tf.one_hot(actions, n_outputs)
    with tf.GradientTape() as tape:
        all_Q_values = model(states)
        Q_values = tf.reduce_sum(all_Q_values * mask, axis=1, keepdims=True)
        loss = tf.reduce_mean(loss_fn(target_Q_values, Q_values))
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

env.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

rewards = [] 
best_score = 0

for episode in range(60):
    obs = env.reset()    
    for step in range(200):
        epsilon = max(1 - episode / 500, 0.01)
        obs, reward, done, info = play_one_step(env, obs, epsilon)
        if done:
            break
    rewards.append(step) # Not shown in the book
    if step >= best_score: # Not shown
        best_weights = model.get_weights() # Not shown
        best_score = step # Not shown
    print("\rEpisode: {}, Steps: {}, eps: {:.3f}".format(episode, step + 1, epsilon), end="") # Not shown
    if episode > 50:
        training_step(batch_size)

model.set_weights(best_weights)
model.save_weights(f'reversi_AI.hdf5',overwrite=True)

plt.figure(figsize=(8, 4))
plt.plot(rewards)
plt.xlabel("Episode", fontsize=14)
plt.ylabel("Sum of rewards", fontsize=14)
plt.show()

env.seed(42)
state = env.reset()

frames = []

for step in range(200):
    action = epsilon_greedy_policy(state)
    state, reward, done, info = env.step(action)
    if done:
        break
    img = env.render(mode="human")

    
