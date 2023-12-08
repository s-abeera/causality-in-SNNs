import numpy as np
import tensorflow as tf
import keras
import tensorflow_probability as tfp

class Controller(tf.keras.Model):
	"""
	Controller class for the ACRE model.

	Args:
		actions (int): Number of possible actions.
		rnn_units (int): Number of units in the RNN layer.
		decay (list, optional): List of decay values for exponential convolution. Defaults to [1.].
	"""

	def __init__(self, actions, rnn_units, decay=[1.]):
		super(Controller, self).__init__()

		self.policy_layer = tf.keras.layers.Dense(actions)
		self.baseline_layer = tf.keras.layers.Dense(1, activation='sigmoid')
		self.decay = decay

	def exp_convolve(self, tensor, decay, initializer=None):
		"""
		Exponential convolution operation.

		Args:
			tensor (tf.Tensor): Input tensor.
			decay (float): Decay value for exponential convolution.
			initializer (tf.Tensor, optional): Initializer tensor. Defaults to None.

		Returns:
			tf.Tensor: Filtered tensor after exponential convolution.
		"""

		assert tensor.dtype in [tf.float16, tf.float32, tf.float64]

		if initializer is None:
			initializer = tf.zeros_like(tensor)

		filtered_tensor = tf.scan(lambda a, x: a * decay + (1 - decay) * x, tensor, initializer=initializer)

		return filtered_tensor

	def sample_policy(self, policy):
		"""
		Sample a new action based on the policy distribution.

		Args:
			policy (tf.Tensor): Policy distribution tensor.

		Returns:
			tf.Tensor: New action sampled from the policy distribution.
		"""

		new_action = tf.random.categorical(policy, num_samples=1)
		new_action = tf.squeeze(new_action, 1)
		return new_action

	def call(self, inputs):
		policy = 0
		baseline = 0

		for decay in self.decay:
			policy_logits = self.policy_layer(inputs)
			baseline = self.baseline_layer(inputs)

			policy = policy_logits

			# exp_convolve broken, commenting out for now
			# policy += self.exp_convolve(policy_logits, decay)
			# baseline += self.exp_convolve(i_baseline, decay)

		new_action = self.sample_policy(policy_logits)

		return policy, new_action, baseline