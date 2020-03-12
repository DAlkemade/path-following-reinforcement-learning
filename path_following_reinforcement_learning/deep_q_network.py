import numpy as np
import tensorflow as tf


class DQN(object):
    def __init__(self, observation_space_size, action_space_size, gamma: float):
        self.gamma = gamma
        self.action_space_size = action_space_size
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=(observation_space_size,)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(self.action_space_size)
        ])
        self.optimizer = tf.optimizers.Adam(0.01)

    def predict(self, states):
        return self.model(states)

    def train(self, batch, target_model):
        batch_size = len(batch)
        num_states = len(batch[0].state)
        states, actions, rewards, next_states, dones = zip(*batch)
        rewards = np.array(rewards, dtype=np.float32)
        dones = np.array(dones, dtype=np.float32)
        next_states = np.reshape(next_states, (batch_size, num_states))
        states = np.reshape(states, (batch_size, num_states))
        next_predictions = target_model.predict(next_states)
        next_predictions_max = np.max(next_predictions, axis=1)
        total_rewards_discounted = rewards + self.gamma * next_predictions_max
        total_rewards_discounted_include_done = np.where(dones, rewards, total_rewards_discounted)
        states = tf.convert_to_tensor(states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.int64)
        total_rewards_discounted_include_done = tf.convert_to_tensor(total_rewards_discounted_include_done,
                                                                     dtype=tf.float32)
        self._optimize_model(states, actions, total_rewards_discounted_include_done)

    @tf.function
    def _optimize_model(self, states, actions, final_rewards):
        with tf.GradientTape() as tape:
            selected_action_values = tf.math.reduce_sum(
                self.model(states) * tf.one_hot(actions, self.action_space_size, dtype=np.float32),
                axis=1)
            loss = tf.math.reduce_sum(tf.square(final_rewards - selected_action_values))
        variables = self.model.trainable_variables
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))

    def copy_weights(self, train_model):
        """Copy weights from training model to this (Target) model.

        Adapted from https://github.com/VXU1230/reinforcement_learning/blob/master/dqn/cart_pole.py
        """
        variables1 = self.model.trainable_variables
        variables2 = train_model.model.trainable_variables
        for v1, v2 in zip(variables1, variables2):
            v1.assign(v2.numpy())
