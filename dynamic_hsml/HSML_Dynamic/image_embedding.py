import tensorflow.compat.v1 as tf
tf.disable_eager_execution()      # TF1-style graph
tf.disable_v2_behavior() 
from tensorflow.python.platform import flags
from utils import tf_layers, flatten
import sys

FLAGS = flags.FLAGS


class ImageEmbedding(object):
    def __init__(self, hidden_num, channels, img_size, conv_initializer):
        """
        Correctly accepts and stores the parameters passed from maml.py.
        """
        self.dim_hidden = hidden_num
        self.channels = channels
        self.conv_initializer = conv_initializer
        self.img_size = img_size

    def __call__(self, images, reuse=False):
        """
        This is the forward pass of the network.
        """
        with tf.variable_scope('image_embedding', reuse=reuse):
            # conv1
            with tf.variable_scope('conv1') as scope:
                # Use the stored initializer and channels
                conv1_weight = tf.get_variable(name='weight', shape=[5, 5, self.channels, self.dim_hidden],
                                               initializer=self.conv_initializer)
                conv1_biases = tf.get_variable(name='biases', shape=[self.dim_hidden], initializer=tf.constant_initializer(0.0))
                conv1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(images, conv1_weight, [1, 1, 1, 1], padding='SAME'),
                                                  conv1_biases), name='conv1_dense')

            # pool1 & norm1
            pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                                   padding='SAME', name='pool1')
            norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                              name='norm1')

            # conv2
            with tf.variable_scope('conv2') as scope:
                conv2_weight = tf.get_variable(name='weight', shape=[5, 5, self.dim_hidden, self.dim_hidden],
                                               initializer=self.conv_initializer)
                conv2_biases = tf.get_variable(name='biases', shape=[self.dim_hidden], initializer=tf.constant_initializer(0.1))
                conv2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(norm1, conv2_weight, [1, 1, 1, 1], padding='SAME'),
                                                  conv2_biases), name='conv2_dense')

            # norm2 & pool2
            norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                              name='norm2')
            pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1],
                                   strides=[1, 2, 2, 1], padding='SAME', name='pool2')

            image_reshape = flatten(pool2)
            
            dim = image_reshape.get_shape().as_list()[1]

            # local3 (dense layer)
            with tf.variable_scope('local3') as scope:
                local3_weight = tf.get_variable(name='weight', shape=[dim, 384],
                                                initializer=tf.truncated_normal_initializer(stddev=0.04))
                local3_biases = tf.get_variable(name='biases', shape=[384], initializer=tf.constant_initializer(0.1))
                local3 = tf.nn.relu(tf.matmul(image_reshape, local3_weight) + local3_biases, name='local3_dense')

            # local4 (output layer)
            with tf.variable_scope('local4') as scope:
                local4_weight = tf.get_variable(name='weight', shape=[384, 64],
                                                initializer=tf.truncated_normal_initializer(stddev=0.04))
                local4_biases = tf.get_variable(name='biases', shape=[64], initializer=tf.constant_initializer(0.1))
                # --- START OF FIX ---
                # The input to this MatMul should be 'local3', not 'image_reshape'.
                local4 = tf.nn.relu(tf.matmul(local3, local4_weight) + local4_biases, name='local4_dense')
                # --- END OF FIX ---
            
            return local4
        
class MLPImageEmbedding(object):
    """
    Lightweight MNIST backbone:
      784 -> 100 -> 100  (ReLU)
    Returns a 100-D feature vector per image.
    """
    def __init__(self, hidden_num, channels, img_size, conv_initializer):
        # keep the same signature as the conv class so you can swap without touching callers
        self.img_size = img_size  # unused, but kept for API parity

    def __call__(self, images, reuse=False):
        with tf.variable_scope('image_embedding', reuse=reuse):
            # Accept [B, 28,28,1] or [B,784]; always flatten to 784
            x = tf.reshape(images, [-1, 784], name='flatten')

            with tf.variable_scope('fc1'):
                W1 = tf.get_variable('weight', shape=[784, 100],
                                     initializer=tf.glorot_uniform_initializer())
                b1 = tf.get_variable('biases', shape=[100],
                                     initializer=tf.zeros_initializer())
                h1 = tf.nn.relu(tf.nn.xw_plus_b(x, W1, b1), name='relu1')

            with tf.variable_scope('fc2'):
                W2 = tf.get_variable('weight', shape=[100, 100],
                                     initializer=tf.glorot_uniform_initializer())
                b2 = tf.get_variable('biases', shape=[100],
                                     initializer=tf.zeros_initializer())
                h2 = tf.nn.relu(tf.nn.xw_plus_b(h1, W2, b2), name='relu2')

            # 100-D features (downstream classifier in maml.py should take 100->10)
            return tf.identity(h2, name='features')

    # The old model method is no longer needed.
    # def model(self, images):
    #     ...
