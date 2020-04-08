# noinspection PyUnresolvedReferences
import gym_path

from examples.deepq_controller import create_discrete_u_w
from path_following_reinforcement_learning.experiment import Experiment, compare_experiments

NUM_RUNS = 1000
EPSILON = .1
GAMMA = .99
TRAIN_STEP = 15
COPY_STEP = 25
MAX_STEPS_IN_RUN = 10000
MEMORY_SIZE = 10000
NUM_LAYERS = 1


def main():
    discrete_actions = create_discrete_u_w()
    experiments = dict()

    experiments['1 layer'] = Experiment("PathFollower-DifferentPaths-v0", discrete_actions, NUM_RUNS,
                                                    TRAIN_STEP, MEMORY_SIZE,
                                                    MAX_STEPS_IN_RUN, EPSILON, COPY_STEP, GAMMA, 2)
    experiments['2 layers'] = Experiment("PathFollower-DifferentPaths-v0", discrete_actions, NUM_RUNS,
                                                     TRAIN_STEP, MEMORY_SIZE,
                                                     MAX_STEPS_IN_RUN, EPSILON, COPY_STEP, GAMMA, 1)
    compare_experiments(experiments, "PathFollowerTestSuite-v0")


if __name__ == "__main__":
    main()
