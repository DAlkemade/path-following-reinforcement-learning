# noinspection PyUnresolvedReferences
import gym_path
import numpy as np

from path_following_reinforcement_learning.experiment import Experiment

NUM_RUNS = 100
EPSILON = .1
GAMMA = .99
BATCH = 15
COPY_STEP = 25
MAX_STEPS_IN_RUN = 10000
MEMORY_SIZE = 10000


def main():
    """Deep Q network for differential robot control.

    Learn to control the robot in the PathFollower environment where the actions are the forward and rotational
    velocity.
    """
    discrete_vs = np.linspace(0, 1, 5)
    discrete_ws = np.linspace(-1., 1., 5)  # make sure it's uneven, so that 0. is in there
    discrete_actions = []
    for v in discrete_vs:
        for w in discrete_ws:
            discrete_actions.append(np.array([v, w]))
    experiment = Experiment("PathFollower-DifferentPaths-v0", discrete_actions, NUM_RUNS, BATCH, MEMORY_SIZE,
                            MAX_STEPS_IN_RUN, EPSILON, COPY_STEP, GAMMA)
    experiment.run()
    experiment.plot_rewards()


if __name__ == "__main__":
    main()
