# noinspection PyUnresolvedReferences
import gym_path
import numpy as np

from path_following_reinforcement_learning.experiment import Experiment, plot_rewards

NUM_RUNS = 3
EPSILON = .1
GAMMA = .99
TRAIN_STEP = 15
COPY_STEP = 25
MAX_STEPS_IN_RUN = 10000
MEMORY_SIZE = 10000


def compare_experiments(experiments: dict):
    """Deep Q network for differential robot control.

    Learn to control the robot in the PathFollower environment where the actions are the forward and rotational
    velocity.
    """
    for name, experiment in experiments.items():
        experiment.run()

    rewards = [experiment.rewards for experiment in experiments.values()]
    names = list(experiments.keys())
    plot_rewards(rewards, names)


def create_discrete_u_w():
    discrete_vs = np.linspace(0, 1, 5)
    discrete_ws = np.linspace(-1., 1., 5)  # make sure it's uneven, so that 0. is in there
    discrete_actions = []
    for v in discrete_vs:
        for w in discrete_ws:
            discrete_actions.append(np.array([v, w]))
    return discrete_actions


def main():
    discrete_actions = create_discrete_u_w()
    experiments = dict()

    experiments['Discount factor=0.9'] = Experiment("PathFollower-DifferentPaths-v0", discrete_actions, NUM_RUNS,
                                                    TRAIN_STEP, MEMORY_SIZE,
                                                    MAX_STEPS_IN_RUN, EPSILON, COPY_STEP, .9)
    experiments['Discount factor=0.99'] = Experiment("PathFollower-DifferentPaths-v0", discrete_actions, NUM_RUNS,
                                                     TRAIN_STEP, MEMORY_SIZE,
                                                     MAX_STEPS_IN_RUN, EPSILON, COPY_STEP, .99)
    compare_experiments(experiments)


if __name__ == "__main__":
    main()
