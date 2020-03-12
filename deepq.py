import random
from collections import deque, namedtuple

import gym
# noinspection PyUnresolvedReferences
import gym_path
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

NUM_RUNS = 100
MAX_STEPS_IN_RUN = 10000
MEMORY_SIZE = 10000
EPSILON = .1
GAMMA = .99
BATCH = 15
COPY_STEP = 25


# tf.keras.backend.set_floatx('float64')


class Memory(object):
    """Store robot runs.

    Uses a deque with a fixed length, meaning that it only keeps the last N entries, i.e. it will forget old experiences.
    Adapted from
    https://github.com/shakedzy/notebooks/blob/master/q_learning_and_dqn/Q%20Learning%20and%20Deep%20Q%20Network.ipynb
    """

    def __init__(self, buffer_size):
        self.memory = deque(maxlen=buffer_size)

    @property
    def size(self):
        return len(self.memory)

    def append(self, item):
        """Append item to memory."""
        self.memory.append(item)

    def sample(self, size):
        """Sample elements from memory."""
        return random.sample(self.memory, size)

    def all_entries(self):
        """Return all entries, floored at the nearest 1000.

        This is useful to increase the batch size less often, so
        that the tensorflow graph does not have to be recreated every time for a slightly higher batch size.
        """
        batch_size = self.size - (self.size % 1000) if self.size >= 1000 else self.size
        return list(self.sample(batch_size))


Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])


class DQN(object):
    def __init__(self, observation_space_size, action_space_size):
        self.action_space_size = action_space_size
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=(observation_space_size,)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(self.action_space_size)
        ])
        self.optimizer = tf.optimizers.Adam(0.01)

    def predict(self, states):
        return self.model(states)

    def train(self, batch, target_model):
        batch_size = len(batch)
        num_states = len(batch[0].state)
        states, actions, rewards, next_states, dones = zip(*batch)
        rewards = np.array(rewards, dtype=np.float32)
        dones = np.array(dones, dtype=np.float32)
        next_states = np.reshape(next_states, (batch_size, num_states))
        states = np.reshape(states, (batch_size, num_states))
        next_predictions = target_model.predict(next_states)
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
                self.model(states) * tf.one_hot(actions, self.action_space_size, dtype=np.float32),
                axis=1)
            loss = tf.math.reduce_sum(tf.square(final_rewards - selected_action_values))
        variables = self.model.trainable_variables
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))

    def copy_weights(self, train_model):
        """Copy weights from training model to this (Target) model.

        Adapted from https://github.com/VXU1230/reinforcement_learning/blob/master/dqn/cart_pole.py
        """
        variables1 = self.model.trainable_variables
        variables2 = train_model.model.trainable_variables
        for v1, v2 in zip(variables1, variables2):
            v1.assign(v2.numpy())


class Experiment():
    def __init__(self, env_name: str, discrete_actions: list):
        self.discrete_actions = discrete_actions
        self.env = gym.make(env_name)

        self.memory = Memory(MEMORY_SIZE)
        self.train_network = DQN(self.num_states, self.num_actions)
        self.target_network = DQN(self.num_states, self.num_actions)
        self.rewards = []

    @property
    def num_states(self):
        return len(self.env.observation_space.sample())

    @property
    def num_actions(self):
        return len(self.discrete_actions)

    def _reset(self):
        self.rewards = []

    def run(self):
        self._reset()
        try:
            for i_episode in range(NUM_RUNS):
                observation = self.env.reset()
                cumulative_reward = 0.
                for t in range(1, MAX_STEPS_IN_RUN + 1):
                    self.env.render()
                    if random.random() < EPSILON:
                        action_index = random.randint(0, len(self.discrete_actions) - 1)
                    else:
                        prediction = self.target_network.predict(np.reshape(observation, (1, self.num_states)))
                        action_index = np.argmax(prediction)

                    action = self.discrete_actions[action_index]
                    prev_observation = observation
                    observation, reward, done, info = self.env.step(action)
                    cumulative_reward += reward
                    self.memory.append(Experience(prev_observation, action_index, reward, observation, done))
                    if t % BATCH == 0:
                        batch_train = self.memory.all_entries()
                        self.train_network.train(batch_train, self.target_network)

                    if t % COPY_STEP:
                        self.target_network.copy_weights(self.train_network)

                    if done:
                        print(f"Episode {i_episode} finished after {t + 1} timesteps")
                        break

                self.rewards.append(cumulative_reward)
        except KeyboardInterrupt:
            pass
        self.env.close()

    def analyze(self):
        plt.plot(range(len(self.rewards)), self.rewards)
        plt.xlabel('Games number')
        plt.ylabel('Reward')
        plt.yscale('symlog')
        plt.show()


def main():
    discrete_vs = np.linspace(0, 1, 3)
    discrete_ws = np.linspace(-np.pi, np.pi, 5)  # make sure it's uneven, so that 0. is in there
    discrete_actions = []
    for v in discrete_vs:
        for w in discrete_ws:
            discrete_actions.append(np.array([v, w]))
    experiment = Experiment("PathFollower-DifferentPaths-v0", discrete_actions)
    experiment.run()
    experiment.analyze()


if __name__ == "__main__":
    main()
