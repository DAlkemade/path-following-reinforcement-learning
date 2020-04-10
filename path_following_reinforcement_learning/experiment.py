import random

import gym
# noinspection PyUnresolvedReferences
import gym_path
import numpy as np
import tqdm
import pandas as pd
from gym import logger
from matplotlib import pyplot as plt

from path_following_reinforcement_learning.deep_q_network import DQN
from path_following_reinforcement_learning.memory import Memory, Experience

class DQNParameters:
    def __init__(self, gamma: float, num_layers: int):
        self.num_layers = num_layers
        self.gamma = gamma


class Experiment():
    def __init__(self, env_name: str, discrete_actions: list, num_runs: int, train_step: int, memory_size: int,
                 max_steps_in_run: int, epsilon: float, copy_step: int, dqn_config: DQNParameters, test_env_name: str = None):
        self.test_env_name = test_env_name
        self.copy_step = copy_step
        self.epsilon = epsilon
        self.max_steps_in_run = max_steps_in_run
        self.num_runs = num_runs
        self.train_step = train_step
        self.discrete_actions = discrete_actions
        self.env = gym.make(env_name)

        self.memory = Memory(memory_size)
        self.train_network = DQN(self.num_states, self.num_actions, dqn_config.gamma, dqn_config.num_layers)
        self.target_network = DQN(self.num_states, self.num_actions, dqn_config.gamma, dqn_config.num_layers)
        self.rewards_train = []
        self.actions = []
        self.run_started = False

    @property
    def num_states(self):
        return len(self.env.observation_space.sample())

    @property
    def num_actions(self):
        return len(self.discrete_actions)

    def train(self, render=True):
        if self.run_started:
            logger.WARN('You should not run a single experiment twice!!')
        self.run_started = True
        cum_count = 0
        try:
            for i_episode in tqdm.tqdm(range(self.num_runs)):
                observation = self.env.reset()
                cumulative_reward = 0.
                for t in range(1, self.max_steps_in_run + 1):
                    cum_count += 1
                    if render:
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
                    if cum_count % self.train_step == 0:
                        # batch_train = self.memory.all_entries()
                        batch_train = self.memory.sample(self.train_step)
                        self.train_network.train(batch_train, self.target_network)

                    if cum_count % self.copy_step:
                        self.target_network.copy_weights(self.train_network)

                    if done:
                        logger.debug(f"Episode {i_episode} finished after {t + 1} timesteps")
                        break

                self.rewards_train.append(cumulative_reward)
        except KeyboardInterrupt:
            pass
        self.env.close()

    def test(self, render=True):
        if self.test_env_name is None:
            raise ValueError('test environment not set, please do so before calling this method.')
        test_env = gym.make(self.test_env_name)
        rewards = list()
        try:
            num_test_paths = len(test_env.paths)
        except AttributeError:
            raise ValueError("Provided test environment should have a paths attribute")
        for i_episode in tqdm.tqdm(range(num_test_paths)):
            observation = test_env.reset()
            cumulative_reward = 0.
            for t in range(1, self.max_steps_in_run + 1):
                if render:
                    test_env.render()

                prediction = self.target_network.predict(np.reshape(observation, (1, self.num_states)))
                action_index = np.argmax(prediction)

                action = self.discrete_actions[action_index]
                observation, reward, done, info = test_env.step(action)
                cumulative_reward += reward

                if done:
                    logger.debug(f"Episode {i_episode} finished after {t + 1} timesteps")
                    break

            rewards.append(cumulative_reward)

        test_env.close()
        return rewards

    def plot_rewards(self):
        plot_rewards([self.rewards_train])

    def plot_actions(self):
        if len(self.actions[0]) > 1:
            raise UserWarning(f"Cannot plot {len(self.actions[0])}d actions")
        plt.plot(range(len(self.actions)), self.actions)
        plt.xlabel('Simulation step')
        plt.ylabel('Action')
        plt.yscale('symlog')
        plt.show()


def plot_rewards(reward_lists: list, legend_entries: list = None, tag=''):
    """Plot rewards

    @param reward_lists: **list of lists** of rewards
    """
    add_legend = legend_entries is not None
    if not add_legend:
        legend_entries = [None] * len(reward_lists)
    assert len(reward_lists) == len(legend_entries)

    for i, rewards in enumerate(reward_lists):
        plt.plot(range(len(rewards)), rewards, label=legend_entries[i])

    plt.xlabel('Games number')
    plt.ylabel(f'Reward {tag}')
    plt.yscale('symlog')
    if add_legend:
        plt.legend()
    plt.show()


def compare_experiments(experiments: dict):
    """Deep Q network for differential robot control.

    Learn to control the robot in the PathFollower environment where the actions are the forward and rotational
    velocity.
    """
    logger.info('Train new experiments')
    for name, experiment in experiments.items():
        experiment.train(render=False)

    rewards = [experiment.rewards_train for experiment in experiments.values()]
    names = list(experiments.keys())
    plot_rewards(rewards, names, tag='Training')

    smooth_rewards = []
    for reward in rewards:
        smooth_rewards.append(list(pd.Series(reward).rolling(100).mean()))
    plot_rewards(smooth_rewards, names, tag='Training rolling mean')

    test_rewards = list()
    mean_test_rewards = list()
    for name, experiment in experiments.items():
        reward = experiment.test(render=True)
        test_rewards.append(reward)
        mean_test_rewards.append(np.mean(reward))

    plot_rewards(test_rewards, names, tag='Test')

    print(list(zip(names, mean_test_rewards)))

    smooth_rewards_test = []
    for reward in test_rewards:
        smooth_rewards_test.append(list(pd.Series(reward).rolling(5).mean()))
    plot_rewards(smooth_rewards_test, names, tag='Test rolling mean')

