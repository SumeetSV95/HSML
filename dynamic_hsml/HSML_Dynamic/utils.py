""" Utility functions. """
import numpy as np
import os
import random
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()      # TF1-style graph
tf.disable_v2_behavior() 
# --- Begin TF2 compat shim (replaces tensorflow.contrib.layers) ---
import tensorflow as tf
from tensorflow.keras import layers as _k

class _TFCompatLayers:
    # minimal drop-in replacements for common contrib.layers symbols used in HSML
    @staticmethod
    def flatten(inputs, scope=None):
        # contrib: flatten(x)
        return _k.Flatten(name=scope)(inputs)

    @staticmethod
    def fully_connected(inputs, num_outputs, activation_fn=tf.nn.relu,
                        weights_initializer=None, biases_initializer=tf.zeros_initializer(),
                        scope=None, **_):
        # contrib: fully_connected(x, num_outputs, activation_fn=..., scope=...)
        act = None if activation_fn in (None, tf.identity) else activation_fn
        return _k.Dense(num_outputs, activation=act, name=scope,
                        kernel_initializer=weights_initializer,
                        bias_initializer=biases_initializer)(inputs)

    @staticmethod
    def batch_norm(inputs, is_training=True, decay=0.999, epsilon=1e-3,
                   center=True, scale=True, scope=None, **_):
        # contrib: batch_norm(x, is_training=..., scope=...)
        # decay≈momentum mapping: momentum = 1 - decay
        momentum = 1.0 - decay
        return _k.BatchNormalization(momentum=momentum, epsilon=epsilon,
                                     center=center, scale=scale, name=scope)(inputs, training=is_training)

    @staticmethod
    def layer_norm(inputs, center=True, scale=True, scope=None, **_):
        # contrib: layer_norm(x, scope=...)
        return _k.LayerNormalization(center=center, scale=scale, name=scope)(inputs)

# alias to previous name used around the codebase
tf_layers = _TFCompatLayers
flatten         = tf_layers.flatten
fully_connected = tf_layers.fully_connected
batch_norm      = tf_layers.batch_norm
layer_norm      = tf_layers.layer_norm
# --- End TF2 compat shim ---


from tensorflow.python.platform import flags

FLAGS = flags.FLAGS

## Image helper
def get_images(paths, labels, nb_samples=None, shuffle=True):
    if nb_samples is not None:
        sampler = lambda x: random.sample(x, nb_samples)
    else:
        sampler = lambda x: x
    images = [(i, os.path.join(path, image)) \
        for i, path in zip(labels, paths) \
        for image in sampler(os.listdir(path))]
    if shuffle:
        random.shuffle(images)
    return images

## Network helpers
def conv_block(inp, cweight, bweight, reuse, scope, activation=tf.nn.relu, max_pool_pad='VALID', residual=False):
    """ Perform, conv, batch norm, nonlinearity, and max pool """
    stride, no_stride = [1,2,2,1], [1,1,1,1]

    if FLAGS.max_pool:
        conv_output = tf.nn.conv2d(inp, cweight, no_stride, 'SAME') + bweight
    else:
        conv_output = tf.nn.conv2d(inp, cweight, stride, 'SAME') + bweight
    normed = normalize(conv_output, activation, reuse, scope)
    if FLAGS.max_pool:
        normed = tf.nn.max_pool(normed, stride, stride, max_pool_pad)
    return normed

def normalize(inp, activation, reuse, scope):
    if FLAGS.norm == 'batch_norm':
        return tf_layers.batch_norm(inp, activation_fn=activation, reuse=reuse, scope=scope)
    elif FLAGS.norm == 'layer_norm':
        return tf_layers.layer_norm(inp, activation_fn=activation, reuse=reuse, scope=scope)
    elif FLAGS.norm == 'None':
        if activation is not None:
            return activation(inp)
        else:
            return inp

## Loss functions
def mse(pred, label):
    pred = tf.reshape(pred, [-1])
    label = tf.reshape(label, [-1])
    return tf.reduce_mean(tf.square(pred-label))

def xent(pred, label):
    return tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=label) / FLAGS.update_batch_size
