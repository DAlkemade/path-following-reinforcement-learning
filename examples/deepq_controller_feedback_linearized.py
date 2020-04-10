import numpy as np

from examples.hyperparameters import Config
from path_following_reinforcement_learning.experiment import Experiment, DQNParameters

NUM_RUNS = 1000


def create_discrete_actions_epsilon_kp():
    discrete_epsilons = np.linspace(0.00001, .5, 5, dtype=np.float32)
    discrete_kps = np.linspace(0.0, 1., 3, dtype=np.float32)
    discrete_actions = []
    for v in discrete_epsilons:
        for w in discrete_kps:
            discrete_actions.append(np.array([v, w]))
    return discrete_actions

def main():
    """Deep Q network for differential robot control.

    Learn to control the robot in the PathFollower environment where the actions are the forward and rotational
    velocity.
    """
    # discrete_actions = list(np.linspace(0.00001, 1., 10, dtype=np.float32))
    # discrete_actions = [np.array([action,]) for action in discrete_actions]

    discrete_actions = create_discrete_actions_epsilon_kp()
    config = Config()
    dqn_config = DQNParameters(config.gamma, config.num_layers)
    experiment = Experiment("PathFollower-FeedbackLinearized-v0", discrete_actions, NUM_RUNS, config.batch_size, config.memory_size,
                            config.max_steps_in_run, config.epsilon, config.copy_step, dqn_config)
    experiment.train()
    experiment.plot_rewards()
    experiment.plot_actions()


if __name__ == "__main__":
    main()
