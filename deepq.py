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

mnist = tf.keras.datasets.mnist

discrete_vs = np.linspace(0, 1, 3)
discrete_ws = np.linspace(-np.pi, np.pi, 5) # make sure it's uneven, so that 0. is in there
discrete_actions = []
for v in discrete_vs:
    for w in discrete_ws:
        discrete_actions.append(np.array([v, w]))

# model = tf.keras.models.Sequential([
#     tf.keras.layers.Flatten(input_shape=(20,)),
#     tf.keras.layers.Dense(128, activation='relu'),
#     tf.keras.layers.Dropout(0.2),
#     tf.keras.layers.Dense(len(discrete_actions))
# ])
class MyModel(tf.keras.Model):
    def __init__(self, num_states, hidden_units, num_actions):
        super(MyModel, self).__init__()
        self.input_layer = tf.keras.layers.InputLayer(input_shape=(num_states,))
        self.hidden_layers = []
        for i in hidden_units:
            self.hidden_layers.append(tf.keras.layers.Dense(
                i, activation='tanh', kernel_initializer='RandomNormal'))
        self.output_layer = tf.keras.layers.Dense(
            num_actions, activation='linear', kernel_initializer='RandomNormal')

    @tf.function
    def call(self, inputs):
        z = self.input_layer(inputs)
        for layer in self.hidden_layers:
            z = layer(z)
        output = self.output_layer(z)
        return output



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

optimizer = tf.optimizers.Adam(0.01)
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

# loss_fn = tf.keras.losses.MeanSquaredError()
# model.compile(optimizer='adam',
#               # loss=loss_fn,
#               metrics=['accuracy'])
env = gym.make("PathFollower-v0")

num_states = len(env.observation_space.sample())
hidden_units = [200, 200]
num_actions = len(discrete_actions)
model = MyModel(num_states, hidden_units, num_actions)

memory = Memory(MEMORY_SIZE)

for i_episode in range(NUM_RUNS):
    observation = env.reset()
    cumulative_reward = 0
    for t in range(1, MAX_STEPS_IN_RUN + 1):
        env.render()
        # TODO action should be onehot (or actually maybe not, think about it)
        if random.random() < EPSILON:
            action_index = random.randint(0, len(discrete_actions) - 1)
        else:
            prediction = model(np.reshape(observation, (1, num_states)))
            action_index = np.argmax(prediction)

        action = discrete_actions[action_index]
        prev_observation = observation
        observation, reward, done, info = env.step(action)
        memory.append(Experience(prev_observation, action_index, reward, observation, done))
        if t % BATCH == 0:
            # TODO if choosing to use a larger batch size, do this in a @tf.function
            batch = memory.sample(BATCH)
            states, actions, rewards, next_states, dones = zip(*batch)
            rewards = np.array(rewards)
            dones = np.array(dones)
            next_states = np.reshape(next_states, (BATCH, num_states))
            states = np.reshape(states, (BATCH, num_states))
            next_predictions = model(next_states)
            next_predictions_max = np.max(next_predictions, axis=1)
            total_rewards_discounted = rewards + GAMMA * next_predictions_max
            total_rewards_discounted_include_done = np.where(dones, rewards, total_rewards_discounted)

            with tf.GradientTape() as tape:
                selected_action_values = tf.math.reduce_sum(model(states) * tf.one_hot(actions, len(discrete_actions)), axis=1)
                loss = tf.math.reduce_sum(tf.square(total_rewards_discounted_include_done - selected_action_values))
            variables = model.trainable_variables
            gradients = tape.gradient(loss, variables)
            optimizer.apply_gradients(zip(gradients, variables))

            # model.fit(states, total_rewards_discounted_include_done, epochs=1)

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
