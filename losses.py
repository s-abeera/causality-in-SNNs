import tensorflow as tf


def compute_baseline_loss(advantages):
	return .5 * tf.reduce_sum(tf.square(advantages), 1)


def compute_entropy_loss(logits):
	policy = tf.nn.softmax(logits)
	log_policy = tf.nn.log_softmax(logits)

	entropy = -policy * log_policy
	negentropy = tf.reduce_sum(-entropy, 1)

	return negentropy

def compute_policy_gradient_loss(logits, actions, advantages):
	cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
		labels=actions, logits=logits)
	advantages = tf.stop_gradient(advantages)
	policy_gradient_loss_per_timestep = cross_entropy * advantages
	return tf.reduce_sum(policy_gradient_loss_per_timestep, 1)

def compute_reg_loss(scnn_output, core_output):
	"""
	Compute the regularization loss for a given scnn_output and core_output.

	Args:
		scnn_output (Tensor): The output of the CNN component.
		core_output (Tensor): The output of the RNN component.

	Returns:
		Tensor: The regularization loss.

	Raises:
		None
	"""

	# RNN Componenets
	thr = 1.0
	rate_cost = 50.
	voltage_cost_rnn = 0.0001
	voltage_cost_cnn = 0.5
	beta = 0.1
	thr_scnn = 0.1
	voltage_reg_method = 'avg_all'

	rnn_v   = core_output[1][...,0]
	rnn_thr = thr + beta * core_output[1][..., 1]
	rnn_pos = tf.nn.relu(rnn_v - rnn_thr)
	rnn_neg = tf.nn.relu(-rnn_v - rnn_thr)
	voltage_reg_rnn = tf.reduce_sum(tf.reduce_mean(tf.square(rnn_pos), 1))
	voltage_reg_rnn += tf.reduce_sum(tf.reduce_mean(tf.square(rnn_neg), 1))
	rnn_rate = tf.reduce_mean(core_output[0], (0, 1))
	rnn_mean_rate = tf.reduce_mean(rnn_rate)
	rate_loss = tf.reduce_sum(tf.square(rnn_rate - .02)) * 1.

	# CNN Componenets
	conv1_z = scnn_output[0]
	conv2_z = scnn_output[2]
	lin_z = tf.zeros_like(scnn_output[1]) # likely wrong
	conv_1_rate = tf.reduce_mean(conv1_z, (0, 1))
	conv_2_rate = tf.reduce_mean(conv2_z, (0, 1))
	linear_rate = tf.reduce_mean(lin_z, (0, 1))
	mean_conv_1_rate = tf.reduce_mean(conv_1_rate)
	mean_conv_2_rate = tf.reduce_mean(conv_2_rate)
	mean_linear_rate = tf.reduce_mean(linear_rate)

	conv1_v = scnn_output[1]
	conv2_v = scnn_output[2]
	conv_pos = tf.nn.relu(conv1_v - thr_scnn)
	conv_neg = tf.nn.relu(-conv1_v - thr_scnn)

	if voltage_reg_method == 'avg_all':
		voltage_reg = tf.reduce_sum(tf.square(tf.reduce_mean(conv_pos, (0, 1))))
		voltage_reg += tf.reduce_sum(tf.square(tf.reduce_mean(conv_neg, (0, 1))))
	elif voltage_reg_method == 'avg_time':
		voltage_reg = tf.reduce_sum(tf.reduce_mean(tf.square(tf.reduce_mean(conv_pos, 1)), 0))
		voltage_reg += tf.reduce_sum(tf.reduce_mean(tf.square(tf.reduce_mean(conv_neg, 1)), 0))
	conv_pos = tf.nn.relu(conv2_v - thr_scnn)
	conv_neg = tf.nn.relu(-conv2_v - thr_scnn)
	if voltage_reg_method == 'avg_all':
		voltage_reg += tf.reduce_sum(tf.square(tf.reduce_mean(conv_pos, (0, 1))))
		voltage_reg += tf.reduce_sum(tf.square(tf.reduce_mean(conv_neg, (0, 1))))
	elif voltage_reg_method == 'avg_time':
		voltage_reg += tf.reduce_sum(tf.reduce_mean(tf.square(tf.reduce_mean(conv_pos, 1)), 0))
		voltage_reg += tf.reduce_sum(tf.reduce_mean(tf.square(tf.reduce_mean(conv_neg, 1)), 0))

	reg_loss = rate_loss * rate_cost
	reg_loss += voltage_cost_rnn * voltage_reg_rnn
	reg_loss += voltage_cost_cnn * voltage_reg

	return reg_loss
	
	
def calc_losses(logits, action, advantages, value_targets, baseline, scnn_output, core_output):
	
	reg_factor_cnn = 1e8
	
	# Calculate Loss		
	pg_loss		 = compute_policy_gradient_loss(logits, action, advantages)
	value_loss	 = compute_baseline_loss(advantages)
	entropy_loss = compute_entropy_loss(logits)

	loss_per_timestep = pg_loss + value_loss + entropy_loss
	total_loss = tf.reduce_sum(loss_per_timestep)	

	reg_loss = compute_reg_loss(scnn_output, core_output)	

	total_loss += reg_loss
	total_loss_for_cnn = tf.reduce_sum(pg_loss + value_loss) / reg_factor_cnn + reg_loss	
	
	return total_loss, total_loss_for_cnn