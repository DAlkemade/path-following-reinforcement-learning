# noinspection PyUnresolvedReferences
import gym_path

from examples.deepq_controller import create_discrete_u_w
from path_following_reinforcement_learning.config import Config
from path_following_reinforcement_learning.experiment import Experiment, compare_experiments

NUM_RUNS = 1000


def main():
    discrete_actions = create_discrete_u_w()
    experiments = dict()

    config2 = Config()
    config2.num_layers = 2

    experiments['2 layers A'] = Experiment("PathFollower-DifferentPaths-v0", discrete_actions, NUM_RUNS,
                                           config2,
                                           test_env_name="PathFollowerTestSuite-v0")
    experiments['2 layers B'] = Experiment("PathFollower-DifferentPaths-v0", discrete_actions, NUM_RUNS,
                                           config2,
                                           test_env_name="PathFollowerTestSuite-v0")
    experiments['2 layers C'] = Experiment("PathFollower-DifferentPaths-v0", discrete_actions, NUM_RUNS,
                                           config2,
                                           test_env_name="PathFollowerTestSuite-v0")
    compare_experiments(experiments, full_memory=False)


if __name__ == "__main__":
    main()
