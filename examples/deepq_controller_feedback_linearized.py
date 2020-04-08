import numpy as np

from path_following_reinforcement_learning.experiment import Experiment

NUM_RUNS = 1000
EPSILON = .1
GAMMA = .99
BATCH = 15
COPY_STEP = 25
MAX_STEPS_IN_RUN = 1000
MEMORY_SIZE = 10000
NUM_LAYERS = 2


def main():
    """Deep Q network for differential robot control.

    Learn to control the robot in the PathFollower environment where the actions are the forward and rotational
    velocity.
    """
    # discrete_actions = list(np.linspace(0.00001, 1., 10, dtype=np.float32))
    # discrete_actions = [np.array([action,]) for action in discrete_actions]

    discrete_epsilons = np.linspace(0.00001, .5, 5, dtype=np.float32)
    discrete_kps = np.linspace(0.0, 1., 3, dtype=np.float32)
    discrete_actions = []
    for v in discrete_epsilons:
        for w in discrete_kps:
            discrete_actions.append(np.array([v, w]))

    experiment = Experiment("PathFollower-FeedbackLinearized-v0", discrete_actions, NUM_RUNS, BATCH, MEMORY_SIZE,
                            MAX_STEPS_IN_RUN, EPSILON, COPY_STEP, GAMMA, NUM_LAYERS)
    experiment.train()
    experiment.plot_rewards()
    experiment.plot_actions()


if __name__ == "__main__":
    main()
