
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