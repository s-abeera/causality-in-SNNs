import os
from dataset import ACREDataset

from SpikingCNN import SpikingCNN
from abeera_SpikingRNN import CustomALIF, spike_function
from Controller import Controller

import tensorflow as tf 
import losses as ls
import sonnet as snt

n_kernel_1 = 8
n_kernel_2 = 8
n_stride_1 = 4
n_stride_2 = 4
n_filters_2 = 32

n_w_1 = (128 - n_kernel_1) // n_stride_1 + 2
n_w_2 = (n_w_1 - n_kernel_2) // n_stride_2 + 1
n_inputs = n_w_2 * n_w_2 * n_filters_2

rnn_units = 100

n_actions = 3

scnn = SpikingCNN()
# scnn_checkpoint_path = "scnn/cp-{epoch:04d}.ckpt"
# scnn_checkpoint_dir = os.path.dirname(scnn_checkpoint_path)
scnn_checkpoint_path = "scnn/scnn.ckpt"
scnn_checkpoint_dir = os.path.dirname(scnn_checkpoint_path)

core = CustomALIF(n_in = n_inputs, n_rec = rnn_units)
rnn_checkpoint_path = "rnn/rnn.ckpt"
rnn_checkpoint_dir = os.path.dirname(rnn_checkpoint_path)

controller = Controller(n_actions, rnn_units)
controller_checkpoint_path = "controller/controller.ckpt" #ook into why it isnt saving correctly
controller_checkpoint_dir = os.path.dirname(controller_checkpoint_path)


scnn.load_weights(scnn_checkpoint_path).expect_partial()
core.load_weights(rnn_checkpoint_path).expect_partial()
controller.load_weights(controller_checkpoint_path).expect_partial()

#


