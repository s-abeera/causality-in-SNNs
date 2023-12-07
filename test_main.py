import tensorflow as tf
import os
from dataloader import ACREDataset
from datetime import datetime
#from test_main_no_log import scnn, core, controller, cce, scnn_state, core_state
from SpikingCNN import SpikingCNN
from SpikingRNN import CustomALIF, spike_function
from Controller import Controller


import tensorflow as tf 
import losses as ls
import sonnet as snt
import numpy as np


# Set up TensorBoard writer
current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
test_log_dir = 'logstest/' + current_time + '/test'
test_summary_writer = tf.summary.create_file_writer(test_log_dir)


# Initialisation
n_kernel_1 = 8
n_kernel_2 = 8
n_stride_1 = 4
n_stride_2 = 4
n_filters_2 = 32

n_w_1 = (128 - n_kernel_1) // n_stride_1 + 2
n_w_2 = (n_w_1 - n_kernel_2) // n_stride_2 + 1
n_inputs = n_w_2 * n_w_2 * n_filters_2

n_time = 10
rnn_units = 100
learning_rate = 1e-4
n_actions = 3
epsilon = 1.
batch_size = 100

n_context = 6
n_query   = 4

test_dataloader = ACREDataset(batch_size, 'test')
n_test_batches = test_dataloader.n_batches

scnn = SpikingCNN()
# scnn_checkpoint_path = "scnn/cp-{epoch:04d}.ckpt"
# scnn_checkpoint_dir = os.path.dirname(scnn_checkpoint_path)
scnn_state = scnn.zero_state(batch_size, tf.float32)
scnn_checkpoint_path = "model_weights/scnn/scnn.ckpt"
scnn_checkpoint_dir = os.path.dirname(scnn_checkpoint_path)

core = CustomALIF(n_in = n_inputs, n_rec = rnn_units)
core_state = core.zero_state(batch_size, tf.float32)
rnn_checkpoint_path = "model_weights/rnn/rnn.ckpt"
rnn_checkpoint_dir = os.path.dirname(rnn_checkpoint_path)

controller = Controller(n_actions, rnn_units)
controller_checkpoint_path = "model_weights/controller/controller.ckpt" #look into why it isnt saving correctly
controller_checkpoint_dir = os.path.dirname(controller_checkpoint_path)

scnn.load_weights(scnn_checkpoint_path).expect_partial()
core.load_weights(rnn_checkpoint_path).expect_partial()
controller.load_weights(controller_checkpoint_path).expect_partial()

cce = tf.keras.losses.CategoricalCrossentropy()

test_losses = []  # To store the losses during testing
query_accuracy = []  # To store the query accuracy during testing
problem_accuracy = []  # To store the problem accuracy during testing

for n_x in range(n_test_batches):  # Assuming you have defined the number of test batches as n_test_batches

    # Get the next batch from the test data loader
    #if there is an error, skip the batch
    try:
        images, labels = test_dataloader.get_next_batch()
    except:
        continue
    #images, labels = test_dataloader.get_next_batch()

    # Forward Pass
    images = snt.BatchApply(snt.Conv2D(16, 8, stride=4))(images)
    context_images, query_images = tf.split(images, [n_context, n_query], 1)

    total_loss = 0
    total_loss_for_cnn = 0

    for t in range(n_context):
        scnn_output, scnn_state = scnn(context_images[:, t], scnn_state)
        core_output, core_state = core(scnn_output[2], core_state)

    context_state_cnn = scnn_state
    context_state_rnn = core_state

    batch_accuracy = []

    for t in range(n_query):
        scnn_output, scnn_state = scnn(query_images[:, t], context_state_cnn)
        core_output, core_state = core(scnn_output[2], context_state_rnn)

        logits, action, baseline = controller(core_output[0])

        value_targets = cce(tf.one_hot(labels[:,t],depth=n_actions), tf.one_hot(action, depth=n_actions))
        advantages = value_targets - baseline

        loss, loss_for_cnn = ls.calc_losses(logits, action,
                                            advantages, value_targets,
                                            baseline, scnn_output, core_output)

        total_loss += (loss * (1. / batch_size))
        total_loss_for_cnn += loss_for_cnn

        baseline_loss = ls.compute_baseline_loss(advantages)
        entropy_loss = ls.compute_entropy_loss(logits)
        pg_loss = ls.compute_policy_gradient_loss(logits, action, advantages)
        reg_loss = ls.compute_reg_loss(scnn_output, core_output)

        for i in range(batch_size):
            if action[i] == labels[:, t][i]:
                batch_accuracy.append(1)
            else:
                batch_accuracy.append(0)

        # if action == labels[:, t]:
        #     batch_accuracy.append(1)
        # else:
        #     batch_accuracy.append(0)

    query_accuracy = query_accuracy + batch_accuracy
    batch_accuracy = np.array(batch_accuracy).reshape(n_query, batch_size) 
    accuracy = np.sum(batch_accuracy, axis = 0)
    accuracy = np.where(accuracy == 4, 1, 0)
    problem_accuracy.append(accuracy)

    #log the losses to tensorboard
    with test_summary_writer.as_default():
        tf.summary.scalar('total_loss', total_loss, step=n_x)
        tf.summary.scalar('total_loss_for_cnn', total_loss_for_cnn, step=n_x)
        #tf.summary.scalar('baseline_loss', baseline_loss, step=n_x)
        #tf.summary.scalar('entropy_loss', entropy_loss, step=n_x)
        #tf.summary.scalar('pg_loss', pg_loss, step=n_x)
        #tf.summary.scalar('reg_loss', reg_loss, step=n_x)

    # Store the losses for this batch in the test_losses list
    test_losses.append(total_loss.numpy())
    print('Batch: ', n_x, 'Test Loss: ', total_loss.numpy(), 'Test Loss for CNN: ', total_loss_for_cnn.numpy())

# Calculate the average test loss over all batches
average_test_loss = np.mean(test_losses)
print('Average Test Loss: ', average_test_loss)
print('Query Accuracy: ', np.mean(query_accuracy))
print('Problem Accuracy: ', np.mean(problem_accuracy))