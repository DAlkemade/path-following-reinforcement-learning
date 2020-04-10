# noinspection PyUnresolvedReferences
import gym_path

from examples.deepq_controller import create_discrete_u_w
from examples.hyperparameters import Config
from path_following_reinforcement_learning.experiment import Experiment, compare_experiments, DQNParameters

NUM_RUNS = 1000


def main():
    discrete_actions = create_discrete_u_w()
    config = Config()
    experiments = dict()

    dqn_config1 = DQNParameters(.9, config.num_layers)
    dqn_config2 = DQNParameters(.99, config.num_layers)
    experiments['Discount factor=0.9'] = Experiment("PathFollower-DifferentPaths-v0", discrete_actions, NUM_RUNS, config.batch_size, config.memory_size,
                            config.max_steps_in_run, config.epsilon, config.copy_step, dqn_config1)
    experiments['Discount factor=0.99'] = Experiment("PathFollower-DifferentPaths-v0", discrete_actions, NUM_RUNS, config.batch_size, config.memory_size,
                            config.max_steps_in_run, config.epsilon, config.copy_step, dqn_config2)
    compare_experiments(experiments, "PathFollowerTestSuite-v0")


if __name__ == "__main__":
    main()
