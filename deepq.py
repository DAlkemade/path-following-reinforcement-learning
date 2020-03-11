from __future__ import absolute_import, division, print_function, unicode_literals

import random
from collections import deque, namedtuple

import gym
# noinspection PyUnresolvedReferences
import gym_path
import numpy as np
import tensorflow as tf

NUM_RUNS = 1000
MAX_STEPS_IN_RUN = 10000
MEMORY_SIZE = 3000
EPSILON = .1
GAMMA = .99
BATCH = 15

# tf.keras.backend.set_floatx('float64')

discrete_vs = np.linspace(0, 1, 3)
discrete_ws = np.linspace(-np.pi, np.pi, 5)  # make sure it's uneven, so that 0. is in there
discrete_actions = []
for v in discrete_vs:
    for w in discrete_ws:
        discrete_actions.append(np.array([v, w]))


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

env = gym.make("PathFollower-v0")

num_states = len(env.observation_space.sample())
hidden_units = [200, 200]
num_actions = len(discrete_actions)

memory = Memory(MEMORY_SIZE)


class DQN(object):
    def __init__(self, observation_space_size, action_space_size):
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=(observation_space_size,)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(action_space_size)
        ])
        self.optimizer = tf.optimizers.Adam(0.01)

    def predict(self, states):
        return self.model(states)

    # @tf.function
    def train(self, batch):
        states, actions, rewards, next_states, dones = zip(*batch)
        rewards = np.array(rewards, dtype=np.float32)
        dones = np.array(dones, dtype=np.float32)
        next_states = np.reshape(next_states, (BATCH, num_states))
        states = np.reshape(states, (BATCH, num_states))
        next_predictions = self.model(next_states)
        next_predictions_max = np.max(next_predictions, axis=1)
        total_rewards_discounted = rewards + GAMMA * next_predictions_max
        total_rewards_discounted_include_done = np.where(dones, rewards, total_rewards_discounted)
        states = tf.convert_to_tensor(states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.int64)
        total_rewards_discounted_include_done = tf.convert_to_tensor(total_rewards_discounted_include_done,
                                                                     dtype=tf.float32)
        self._optimize_model(states, actions, total_rewards_discounted_include_done)

    @tf.function
    def _optimize_model(self, states, actions, final_rewards):
        with tf.GradientTape() as tape:
            selected_action_values = tf.math.reduce_sum(
                self.model(states) * tf.one_hot(actions, len(discrete_actions), dtype=np.float32),
                axis=1)
            loss = tf.math.reduce_sum(tf.square(final_rewards - selected_action_values))
        variables = self.model.trainable_variables
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))


train_network = DQN(num_states, num_actions)

for i_episode in range(NUM_RUNS):
    observation = env.reset()
    cumulative_reward = 0
    for t in range(1, MAX_STEPS_IN_RUN + 1):
        env.render()
        # TODO action should be onehot (or actually maybe not, think about it)
        if random.random() < EPSILON:
            action_index = random.randint(0, len(discrete_actions) - 1)
        else:
            prediction = train_network.predict(np.reshape(observation, (1, num_states)))
            action_index = np.argmax(prediction)

        action = discrete_actions[action_index]
        prev_observation = observation
        observation, reward, done, info = env.step(action)
        memory.append(Experience(prev_observation, action_index, reward, observation, done))
        if t % BATCH == 0:
            # TODO if choosing to use a larger batch size, do this in a @tf.function
            batch_train = memory.sample(BATCH)
            train_network.train(batch_train)
            # model.fit(states, total_rewards_discounted_include_done, epochs=1)

        if done:
            print("Episode finished after {} timesteps".format(t + 1))
            break
env.close()
