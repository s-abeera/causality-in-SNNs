import numpy as np
import tensorflow as tf
import keras
import tensorflow_probability as tfp

class Controller(tf.keras.Model):
	def __init__(self, actions, rnn_units, decay=[1.]):
		super(Controller, self).__init__()
		
		#self.inputs = tf.keras.Input(shape=(rnn_units,))
		self.policy_layer	= tf.keras.layers.Dense(actions)
		self.baseline_layer = tf.keras.layers.Dense(1)
		self.decay = decay


	def exp_convolve(self, tensor, decay, initializer=None):

		assert tensor.dtype in [tf.float16, tf.float32, tf.float64]

		if initializer is None:
			initializer = tf.zeros_like(tensor)
		
		filtered_tensor = tf.scan(lambda a, x: a * decay + (1-decay) * x, tensor, initializer=initializer)
		
		return filtered_tensor
		
	def sample_policy(self, policy):
		
		new_action = tf.random.categorical(policy, num_samples=1)
		new_action = tf.squeeze(new_action, 1)
		return new_action
	
	def call(self, inputs):
		
		policy = 0
		baseline = 0
		
		#input = 
		for decay in self.decay:
			
			policy_logits = self.policy_layer(inputs)
			baseline = self.baseline_layer(inputs)
			
			policy = policy_logits
			
			# exp_convolve broken, commenting out for now
			#policy += self.exp_convolve(policy_logits, decay)			
			#baseline += self.exp_convolve(i_baseline, decay)
		
		new_action = self.sample_policy(policy_logits)
		
		return policy, new_action, baseline