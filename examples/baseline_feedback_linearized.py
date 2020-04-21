import gym
# noinspection PyUnresolvedReferences
import gym_path
import numpy as np
import tqdm
from gym import logger
from gym_path.envs import PathFeedbackLinearizedTestSuite

from examples.deepq_controller_feedback_linearized import create_discrete_actions_epsilon_kp

env: PathFeedbackLinearizedTestSuite = gym.make("PathFollower-FeedbackLinearizedTestSuite-v0")

num_states = len(env.observation_space.sample())

num_test_paths = len(env.paths)
actions = create_discrete_actions_epsilon_kp()
action = actions[8]
print(action)
rewards = []
for i_episode in tqdm.tqdm(range(num_test_paths)):
    observation = env.reset()
    reward_cum = 0
    for t in range(300):
        # env.render()

        # action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        reward_cum += reward
        if done:
            logger.debug("Episode finished after {} timesteps".format(t + 1))
            break
    rewards.append(reward_cum)
env.close()
print(f'mean reward: {np.mean(rewards)}')
