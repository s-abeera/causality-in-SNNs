import tensorflow as tf
import os
from dataloader import ACREDataset

from test_main_no_log import scnn, core, controller, cce, scnn_state, core_state

import tensorflow as tf 
import losses as ls
import sonnet as snt
import numpy as np

batch_size = 5
n_time = 10

n_context = 6
n_query   = 4

test_dataloader = ACREDataset(5, 'test')
n_test_batches = test_dataloader.n_batches


test_losses = []  # To store the losses during testing
query_accuracy = []  # To store the query accuracy during testing
problem_accuracy = []  # To store the problem accuracy during testing

for n_x in range(n_test_batches):  # Assuming you have defined the number of test batches as n_test_batches

    # Get the next batch from the test data loader
    images, labels = test_dataloader.get_next_batch()

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

    for t in range(n_query):
        scnn_output, scnn_state = scnn(query_images[:, t], context_state_cnn)
        core_output, core_state = core(scnn_output[2], context_state_rnn)

        logits, action, baseline = controller(core_output[0])

        value_targets = cce(labels[:,t], tf.cast(action, tf.float32))
        
        advantages = value_targets - baseline

        loss, loss_for_cnn = ls.calc_losses(logits, action,
                                            advantages, value_targets,
                                            baseline, scnn_output, core_output)

        total_loss += loss
        total_loss_for_cnn += loss_for_cnn

        batch_accuracy = []
        if action == labels[:, t]:
            batch_accuracy.append(1)
        else:
            batch_accuracy.append(0)

    query_accuracy.append(batch_accuracy)
    batch_accuracy = np.array(batch_accuracy).reshape(4, 5)
    accuracy = np.sum(accuracy, axis = 1)
    accuracy = np.where(accuracy == 4, 1, 0)
    problem_accuracy.append(accuracy)


    # Store the losses for this batch in the test_losses list
    test_losses.append(total_loss.numpy())
    print('Batch: ', n_x, 'Test Loss: ', total_loss.numpy(), 'Test Loss for CNN: ', total_loss_for_cnn.numpy())

# Calculate the average test loss over all batches
average_test_loss = np.mean(test_losses)
print('Average Test Loss: ', average_test_loss)
print('Query Accuracy: ', np.mean(query_accuracy))
print('Problem Accuracy: ', np.mean(problem_accuracy))