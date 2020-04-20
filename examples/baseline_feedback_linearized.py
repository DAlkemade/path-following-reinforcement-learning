import gym
# noinspection PyUnresolvedReferences
import gym_path
import numpy as np
from gym import logger
from gym_path.envs import PathFeedbackLinearizedTestSuite

from examples.deepq_controller_feedback_linearized import create_discrete_actions_epsilon_kp

env: PathFeedbackLinearizedTestSuite = gym.make("PathFollower-FeedbackLinearizedTestSuite-v0")

num_states = len(env.observation_space.sample())

num_test_paths = len(env.paths)
actions = create_discrete_actions_epsilon_kp()
action = actions[8]
print(action)
for i_episode in range(num_test_paths):
    observation = env.reset()
    for t in range(300):
        env.render()
        path_points = np.reshape(observation, (int(num_states / 2), 2))

        # action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            logger.debug("Episode finished after {} timesteps".format(t + 1))
            break
env.close()
