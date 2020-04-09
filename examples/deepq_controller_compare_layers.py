# noinspection PyUnresolvedReferences
import gym_path

from examples.deepq_controller import create_discrete_u_w
from path_following_reinforcement_learning.experiment import Experiment, compare_experiments

NUM_RUNS = 9000
EPSILON = .1
GAMMA = .99
TRAIN_STEP = 20
COPY_STEP = 50
MAX_STEPS_IN_RUN = 1000
MEMORY_SIZE = 10000


def main():
    discrete_actions = create_discrete_u_w()
    experiments = dict()

    experiments['1 layer A'] = Experiment("PathFollower-DifferentPaths-v0", discrete_actions, NUM_RUNS,
                                        TRAIN_STEP, MEMORY_SIZE,
                                        MAX_STEPS_IN_RUN, EPSILON, COPY_STEP, GAMMA, 1)
    experiments['1 layer B'] = Experiment("PathFollower-DifferentPaths-v0", discrete_actions, NUM_RUNS,
                                          TRAIN_STEP, MEMORY_SIZE,
                                          MAX_STEPS_IN_RUN, EPSILON, COPY_STEP, GAMMA, 1)
    experiments['1 layer C'] = Experiment("PathFollower-DifferentPaths-v0", discrete_actions, NUM_RUNS,
                                          TRAIN_STEP, MEMORY_SIZE,
                                          MAX_STEPS_IN_RUN, EPSILON, COPY_STEP, GAMMA, 1)
    experiments['2 layers A'] = Experiment("PathFollower-DifferentPaths-v0", discrete_actions, NUM_RUNS,
                                         TRAIN_STEP, MEMORY_SIZE,
                                         MAX_STEPS_IN_RUN, EPSILON, COPY_STEP, GAMMA, 2)
    experiments['2 layers B'] = Experiment("PathFollower-DifferentPaths-v0", discrete_actions, NUM_RUNS,
                                         TRAIN_STEP, MEMORY_SIZE,
                                         MAX_STEPS_IN_RUN, EPSILON, COPY_STEP, GAMMA, 2)
    experiments['2 layers C'] = Experiment("PathFollower-DifferentPaths-v0", discrete_actions, NUM_RUNS,
                                         TRAIN_STEP, MEMORY_SIZE,
                                         MAX_STEPS_IN_RUN, EPSILON, COPY_STEP, GAMMA, 2)
    compare_experiments(experiments, "PathFollowerTestSuite-v0")


if __name__ == "__main__":
    main()
