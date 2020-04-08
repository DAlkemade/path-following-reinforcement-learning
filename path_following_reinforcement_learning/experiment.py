import random

import gym
# noinspection PyUnresolvedReferences
import gym_path
import numpy as np
from matplotlib import pyplot as plt

from path_following_reinforcement_learning.deep_q_network import DQN
from path_following_reinforcement_learning.memory import Memory, Experience
from gym import logger


class Experiment():
    def __init__(self, env_name: str, discrete_actions: list, num_runs: int, batch: int, memory_size: int,
                 max_steps_in_run: int, epsilon: float, copy_step: int, gamma: float):
        self.copy_step = copy_step
        self.epsilon = epsilon
        self.max_steps_in_run = max_steps_in_run
        self.num_runs = num_runs
        self.train_step = batch
        self.discrete_actions = discrete_actions
        self.env = gym.make(env_name)

        self.memory = Memory(memory_size)
        self.train_network = DQN(self.num_states, self.num_actions, gamma)
        self.target_network = DQN(self.num_states, self.num_actions, gamma)
        self.rewards = []
        self.actions = []
        self.run_started = False

    @property
    def num_states(self):
        return len(self.env.observation_space.sample())

    @property
    def num_actions(self):
        return len(self.discrete_actions)

    def run(self):
        if self.run_started:
            logger.WARN('You should not run a single experiment twice!!')
        self.run_started = True
        try:
            for i_episode in range(self.num_runs):
                observation = self.env.reset()
                cumulative_reward = 0.
                for t in range(1, self.max_steps_in_run + 1):
                    self.env.render()
                    if random.random() < self.epsilon:
                        action_index = random.randint(0, len(self.discrete_actions) - 1)
                    else:
                        prediction = self.target_network.predict(np.reshape(observation, (1, self.num_states)))
                        action_index = np.argmax(prediction)

                    action = self.discrete_actions[action_index]
                    self.actions.append(action)
                    prev_observation = observation
                    observation, reward, done, info = self.env.step(action)
                    cumulative_reward += reward
                    self.memory.append(Experience(prev_observation, action_index, reward, observation, done))
                    if t % self.train_step == 0:
                        batch_train = self.memory.all_entries()
                        self.train_network.train(batch_train, self.target_network)

                    if t % self.copy_step:
                        self.target_network.copy_weights(self.train_network)

                    if done:
                        print(f"Episode {i_episode} finished after {t + 1} timesteps")
                        break

                self.rewards.append(cumulative_reward)
        except KeyboardInterrupt:
            pass
        self.env.close()

    def plot_rewards(self):
        plt.plot(range(len(self.rewards)), self.rewards)
        plt.xlabel('Games number')
        plt.ylabel('Reward')
        plt.yscale('symlog')
        plt.show()

    def plot_actions(self):
        if len(self.actions[0]) > 1:
            raise UserWarning(f"Cannot plot {len(self.actions[0])}d actions")
        plt.plot(range(len(self.actions)), self.actions)
        plt.xlabel('Simulation step')
        plt.ylabel('Action')
        plt.yscale('symlog')
        plt.show()

