
"""This is a tensorflow tutorial that I have adapted for the CNN part of our project!"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

tf.logging.set_verbosity(tf.logging.INFO)


def cnn_model_fn(features, labels, mode):
  print("printing to mark the start of an iteration")
  """Model function for CNN."""
  # Input Layer
  # Reshape X to 4-D tensor: [batch_size, width, height, channels]
  # images are 28x28 pixels, and have one color channel
  input_layer = tf.reshape(features["x"], [-1, 480, 640, 3])

  # Convolutional Layer #1
  # Computes 32 features using a 5x5 filter with ReLU activation.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 28, 28, 1]
  # Output Tensor Shape: [batch_size, 28, 28, 8]
  conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=2,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)

  # Pooling Layer #1
  # First max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, 480, 640, 8]
  # Output Tensor Shape: [batch_size, 240, 320, 8]
  pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[4, 4], strides=4)

  # Convolutional Layer #2
  # Computes 64 features using a 5x5 filter.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 240, 320, 8]
  # Output Tensor Shape: [batch_size, 240, 320, 16]
  conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=4,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)

  # Pooling Layer #2
  # Second max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, 240, 320, 16]
  # Output Tensor Shape: [batch_size, 120, 160, 16]
  pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

  # Flatten tensor into a batch of vectors
  # Input Tensor Shape: [batch_size, 120, 160, 16]
  # Output Tensor Shape: [batch_size, 120 * 160 * 16]
  pool2_flat = tf.reshape(pool2, [-1, 60 * 80 * 4])

  # Dense Layer
  # Densely connected layer with 1024 neurons