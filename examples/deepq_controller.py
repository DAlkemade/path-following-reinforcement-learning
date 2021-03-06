# noinspection PyUnresolvedReferences
import gym_path
import numpy as np

from path_following_reinforcement_learning.config import Config
from path_following_reinforcement_learning.experiment import Experiment

NUM_RUNS = 1000


def main():
    """Deep Q network for differential robot control.

    Learn to control the robot in the PathFollower environment where the actions are the forward and rotational
    velocity.
    """
    config = Config()
    discrete_actions = create_discrete_u_w()
    experiment = Experiment("PathFollower-DifferentPaths-v0", discrete_actions, NUM_RUNS, config)
    experiment.train(render=False)
    experiment.plot_rewards()


def create_discrete_u_w():
    discrete_vs = np.linspace(0, 1, 5)
    discrete_ws = np.linspace(-1., 1., 5)  # make sure it's uneven, so that 0. is in there
    discrete_actions = []
    for v in discrete_vs:
        for w in discrete_ws:
            discrete_actions.append(np.array([v, w]))
    return discrete_actions


if __name__ == "__main__":
    main()
