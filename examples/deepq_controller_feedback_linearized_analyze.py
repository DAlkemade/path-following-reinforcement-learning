import numpy as np

from examples.deepq_controller_feedback_linearized import create_discrete_actions_epsilon_kp
from path_following_reinforcement_learning.config import Config
from path_following_reinforcement_learning.experiment import Experiment
import tensorflow as tf

NUM_RUNS = 1000


def main():
    """Deep Q network for differential robot control.

    Learn to control the robot in the PathFollower environment where the actions are the forward and rotational
    velocity.
    """
    # discrete_actions = list(np.linspace(0.00001, 1., 10, dtype=np.float32))
    # discrete_actions = [np.array([action,]) for action in discrete_actions]

    discrete_actions = create_discrete_actions_epsilon_kp()
    config = Config()
    experiment = Experiment("PathFollower-FeedbackLinearized-v0", discrete_actions, NUM_RUNS, config, test_env_name="PathFollower-FeedbackLinearizedTestSuite-v0")
    model = tf.keras.models.load_model('model_Extension A.h5')
    experiment.target_network.model = model
    experiment.test(render=True, print_actions=True)
    experiment.plot_rewards()
    experiment.plot_actions()


if __name__ == "__main__":
    main()
