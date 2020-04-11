# noinspection PyUnresolvedReferences
import gym_path

from examples.deepq_controller import create_discrete_u_w
from examples.deepq_controller_feedback_linearized import create_discrete_actions_epsilon_kp
from path_following_reinforcement_learning.config import Config
from path_following_reinforcement_learning.experiment import Experiment, compare_experiments

NUM_RUNS = 1000


def create_extension_experiment(config: Config, discrete_actions, env: str, test_env: str):
    return Experiment(env, discrete_actions, NUM_RUNS,
                      config, test_env_name=test_env)


def main():
    discrete_actions_regular = create_discrete_u_w()
    discrete_actions_extension = create_discrete_actions_epsilon_kp()
    experiments = dict()
    config = Config()

    experiments['Regular A'] = create_extension_experiment(config, discrete_actions_regular,
                                                           "PathFollower-DifferentPaths-v0", "PathFollowerTestSuite-v0")
    experiments['Regular B'] = create_extension_experiment(config, discrete_actions_regular,
                                                           "PathFollower-DifferentPaths-v0", "PathFollowerTestSuite-v0")
    experiments['Regular C'] = create_extension_experiment(config, discrete_actions_regular,
                                                           "PathFollower-DifferentPaths-v0", "PathFollowerTestSuite-v0")
    experiments['Extension A'] = create_extension_experiment(config, discrete_actions_extension,
                                                             "PathFollower-FeedbackLinearized-v0",
                                                             "PathFollower-FeedbackLinearizedTestSuite-v0")
    experiments['Extension B'] = create_extension_experiment(config, discrete_actions_extension,
                                                             "PathFollower-FeedbackLinearized-v0",
                                                             "PathFollower-FeedbackLinearizedTestSuite-v0")
    experiments['Extension C'] = create_extension_experiment(config, discrete_actions_extension,
                                                             "PathFollower-FeedbackLinearized-v0",
                                                             "PathFollower-FeedbackLinearizedTestSuite-v0")

    compare_experiments(experiments)


if __name__ == "__main__":
    main()
