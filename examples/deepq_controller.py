# noinspection PyUnresolvedReferences
import gym_path
import numpy as np

from path_following_reinforcement_learning.experiment import Experiment

NUM_RUNS = 3000
EPSILON = .1
GAMMA = .99
TRAIN_STEP = 100
COPY_STEP = 200
MAX_STEPS_IN_RUN = 1000
MEMORY_SIZE = 10000
NUM_LAYERS = 1


def main():
    """Deep Q network for differential robot control.

    Learn to control the robot in the PathFollower environment where the actions are the forward and rotational
    velocity.
    """
    discrete_actions = create_discrete_u_w()
    experiment = Experiment("PathFollower-DifferentPaths-v0", discrete_actions, NUM_RUNS, TRAIN_STEP, MEMORY_SIZE,
                            MAX_STEPS_IN_RUN, EPSILON, COPY_STEP, GAMMA, NUM_LAYERS)
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
