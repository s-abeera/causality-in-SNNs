import os
from dataset import ACREDataset

from SpikingCNN import SpikingCNN
from abeera_SpikingRNN import CustomALIF, spike_function
from Controller import Controller

import tensorflow as tf 
import losses as ls
import sonnet as snt

# Initialisation

n_kernel_1 = 8
n_kernel_2 = 8
n_stride_1 = 4
n_stride_2 = 4
n_filters_2 = 32

n_w_1 = (128 - n_kernel_1) // n_stride_1 + 2
n_w_2 = (n_w_1 - n_kernel_2) // n_stride_2 + 1
n_inputs = n_w_2 * n_w_2 * n_filters_2

batch_size = 5
n_time = 10
rnn_units = 100
learning_rate = 1e-4
n_actions = 3
# # 

scnn = SpikingCNN()

scnn_state = scnn.zero_state(batch_size, tf.float32)
cnn_optimizer = tf.keras.optimizers.Adam(learning_rate)

core = CustomALIF(n_in = n_inputs, n_rec = rnn_units)
core_state = core.zero_state(batch_size, tf.float32)
rnn_optimizer = tf.keras.optimizers.Adam(learning_rate)

controller = Controller(n_actions, rnn_units)

dataloader = ACREDataset(5, 'train', data_path = 'ACRE-IID-release/')
n_batches = dataloader.n_batches

cce = tf.keras.losses.CategoricalCrossentropy(from_logits=False)

n_context = 6
n_query   = 4

total_loss = 0
total_loss_for_cnn = 0

for n_x in range(n_batches):
	
	with tf.GradientTape(persistent=True) as tape:

		# Forward Pass
		images, labels = dataloader.get_next_batch()

		images = images[:,0] ; labels = labels[:,0]

		# Conv2d weights and SCNN weights are both 'cnn_variables'
		images = snt.Conv2D(16, 8, stride=4)(images)		

		scnn_output, scnn_state = scnn(images, scnn_state)
		core_output, core_state = core(scnn_output[2], core_state)

		logits, action, baseline = controller(core_output[0])

		value_targets = cce(labels, tf.cast(action, tf.float32))

		advantages = value_targets - baseline

		loss,loss_for_cnn = ls.calc_losses(logits, action, 
										   advantages, value_targets, 
										   baseline, scnn_output, core_output)

		total_loss += loss
		total_loss_for_cnn += loss_for_cnn

		rnn_params = core.variable_list		
		cnn_params = [*scnn.trainable_variables, *controller.trainable_variables]

	# Calculate Gradients
	rnn_grads = tape.gradient(total_loss, rnn_params)
	cnn_grads = tape.gradient(total_loss_for_cnn, cnn_params)

	# Update Weights
	rnn_optimizer.apply_gradients(zip(rnn_grads, rnn_params))
	cnn_optimizer.apply_gradients(zip(cnn_grads, cnn_params))

	print(n_x)
	print(total_loss)
	print(total_loss_for_cnn)
	

	
	
	
	
	
	
	
	