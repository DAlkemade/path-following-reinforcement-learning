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
    discrete_actions = list(np.linspace(0.00001, 1., 10, dtype=np.float32))
    discrete_actions = [np.array([action,]) for action in discrete_actions]
    experiment = Experiment("PathFollower-FeedbackLinearized-v0", discrete_actions, NUM_RUNS, BATCH, MEMORY_SIZE,
                            MAX_STEPS_IN_RUN, EPSILON, COPY_STEP, GAMMA)
    experiment.run()
    experiment.plot_rewards()
    experiment.plot_actions()


if __name__ == "__main__":
    main()