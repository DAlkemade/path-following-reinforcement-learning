from __future__ import absolute_import, division, print_function, unicode_literals

import random
from collections import deque, namedtuple

import gym
import tensorflow as tf
import numpy as np
# noinspection PyUnresolvedReferences
import gym_path

NUM_RUNS = 100
MAX_STEPS_IN_RUN = 1000
MEMORY_SIZE = 3000
EPSILON = .1
GAMMA = .99
BATCH = 15

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(20,)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(2)
])


class Memory(object):
    """Store robot runs.

    Uses a deque with a fixed length, meaning that it only keeps the last N entries, i.e. it will forget old experiences.
    Adapted from
    https://github.com/shakedzy/notebooks/blob/master/q_learning_and_dqn/Q%20Learning%20and%20Deep%20Q%20Network.ipynb
    """

    def __init__(self, buffer_size):
        self.memory = deque(maxlen=buffer_size)

    def append(self, item):
        self.memory.append(item)

    def sample(self, size):
        return random.sample(self.memory, size)

Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])


loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])

env = gym.make("PathFollower-v0")

memory = Memory(MEMORY_SIZE)

for i_episode in range(NUM_RUNS):
    observation = env.reset()
    cumulative_reward = 0
    for t in range(1, MAX_STEPS_IN_RUN+1):
        env.render()
        if random.random() < EPSILON:
            action = env.action_space.sample()
        else:
            prediction = model(np.squeeze(observation))
            action = np.argmax(prediction)
        prev_observation = observation
        observation, reward, done, info = env.step(action)
        memory.append(Experience(prev_observation, action, reward, observation, done))
        if t % BATCH == 0:
            batch = memory.sample(BATCH)
            states, actions, rewards, next_states, dones = zip(*batch)
            next_predictions = np.max(model(next_states))
            total_rewards_discounted = rewards + GAMMA * next_predictions
            total_rewards_discounted_include_done = np.where(dones, rewards, total_rewards_discounted)
            model.fit(states, total_rewards_discounted_include_done, epochs=1)

        if done:
            print("Episode finished after {} timesteps".format(t + 1))
            break
env.close()


# model.evaluate(x_test, y_test, verbose=2)
#
# probability_model = tf.keras.Sequential([
#     model,
#     tf.keras.layers.Softmax()
# ])
#
# probability_model(x_test[:5])
