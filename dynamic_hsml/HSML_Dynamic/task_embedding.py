import tensorflow.compat.v1 as tf
tf.disable_eager_execution()      # TF1-style graph
tf.disable_v2_behavior() 
from tensorflow.python.ops.rnn_cell import LSTMCell, GRUCell

from tensorflow.python.platform import flags


FLAGS = flags.FLAGS

class LSTMAutoencoder(object):
    def __init__(self, hidden_num, cell=None, reverse=True, decode_without_input=False):
        if cell is None:
            self._enc_cell = GRUCell(hidden_num, name='encoder_cell')
            self._dec_cell = GRUCell(hidden_num, name='decoder_cell')
        else:
            self._enc_cell = cell
            self._dec_cell = cell
        self.reverse = reverse
        self.decode_without_input = decode_without_input
        self.hidden_num = hidden_num

        if FLAGS.datasource in ['sinusoid', 'mixture']:
            # ... set elem_num for regression ...
            self.elem_num_init = 2
            self.elem_num=20

        # --- START OF FIX ---
        # The initialization order in main.py causes FLAGS.num_classes to be incorrect (5 instead of 10)
        # when this class is created. We will handle mnist_rotations as a special case.
        elif FLAGS.datasource in ['mnist_rotations', 'mnist_permutations']:
            self.elem_num = 74 # 64 (image_emb) + 10 (label_emb)
        elif FLAGS.datasource in ['miniimagenet', 'omniglot', 'multidataset', 'nonstationary', 'vggflowers_stationary']:
            # This will correctly use 64 + 5 = 69 for 5-way classification datasets.
            self.elem_num = FLAGS.num_classes + 64
        else:
            # Fallback for other datasources; match classification geometry by default
            self.elem_num = FLAGS.num_classes + 64
        # --- END OF FIX ---

        self.dec_weight = tf.Variable(tf.truncated_normal([self.hidden_num,
                                                           self.elem_num], dtype=tf.float32), name='dec_weight')
        self.dec_bias = tf.Variable(tf.constant(0.1, shape=[self.elem_num],
                                                dtype=tf.float32), name='dec_bias')

    def model(self, inputs):
        self.inputs = inputs
        self.batch_num = inputs.get_shape().as_list()[0]
        self.step_num = inputs.get_shape().as_list()[1]
        self.elem_num = inputs.get_shape().as_list()[2]
        # inputs = tf.unstack(inputs, axis=1) # THIS IS THE PROBLEMATIC LINE TO REMOVE

        with tf.variable_scope('lstm_autoencoder', reuse=tf.AUTO_REUSE):
            # CHANGE THIS: Pass the 'inputs' tensor directly to dynamic_rnn
            (self.z_codes, self.enc_state) = tf.nn.dynamic_rnn(
                self._enc_cell, inputs, dtype=tf.float32)
            dec_outs = self.decoder(self.z_codes)
            loss = self.get_loss(dec_outs, inputs)

        return self.emb_all, loss

    def decoder(self, z_codes):
        with tf.variable_scope('decoder') as vs:

            if self.decode_without_input:
                # FIX 1: Use self.step_num and self.inputs
                dec_inputs = [tf.zeros(tf.shape(self.inputs[0]), dtype=tf.float32) for _ in range(self.step_num)]
                (dec_outputs, dec_state) = tf.contrib.rnn.static_rnn(self._dec_cell, dec_inputs,
                                                                     initial_state=self.enc_state,
                                                                     dtype=tf.float32)
                if self.reverse:
                    dec_outputs = dec_outputs[::-1]
                dec_output_ = tf.transpose(tf.stack(dec_outputs), [1, 0, 2])
                dec_weight_ = tf.tile(tf.expand_dims(self.dec_weight, 0), [self.batch_num, 1, 1])
                output_ = tf.matmul(dec_weight_, dec_output_) + self.dec_bias
            else:
                dec_state = self.enc_state
                # CHANGE THIS LINE
                dec_input_ = tf.zeros([self.batch_num, self.elem_num], dtype=tf.float32)

                dec_outputs = []
                # FIX 2: Use self.step_num
                for step in range(self.step_num):
                    if step > 0:
                        vs.reuse_variables()
                    (dec_input_, dec_state) = \
                        self._dec_cell(dec_input_, dec_state)
                    dec_input_ = tf.matmul(dec_input_, self.dec_weight) + self.dec_bias
                    dec_outputs.append(dec_input_)
                if self.reverse:
                    dec_outputs = dec_outputs[::-1]
                output_ = tf.transpose(tf.stack(dec_outputs), [1, 0, 2])

        return output_

    def get_loss(self, dec_outs, inputs):
        # FIX 1: The 'inputs' tensor is already in the correct [batch, steps, features] format.
        # The stack and transpose are no longer needed.
        loss = tf.reduce_mean(tf.square(inputs - dec_outs))

        # FIX 2: Assign the embedding to the instance attribute 'self.emb_all'.
        self.emb_all = tf.reduce_mean(self.z_codes, axis=1)

        return loss

class MeanAutoencoder(object):
    def __init__(self, hidden_num):
        self.hidden_num = hidden_num
        if FLAGS.datasource in ['sinusoid', 'mixture']:
            self.elem_num = 2
        elif FLAGS.datasource in ['miniimagenet', 'omniglot', 'multidataset', 'nonstationary', 'vggflowers_stationary']:
            self.elem_num = 69
        elif FLAGS.datasource in ['mnist_rotations', 'mnist_permutations']:
            # --- START OF FIX ---
            # The original value was hardcoded to 69, assuming 5-way classification (64+5).
            # For 10-way MNIST, it should be 64 (image_emb) + 10 (label_emb) = 74.
            self.elem_num = 74
            # --- END OF FIX ---
        else:
            self.elem_num = FLAGS.num_classes + 64

    def model(self, inputs):
        with tf.variable_scope('encoder', reuse=tf.AUTO_REUSE):
            enc_dense1 = tf.layers.dense(inputs, units=self.hidden_num_mid, activation=tf.nn.relu, name='encoder_dense1')
            enc_dense2 = tf.layers.dense(enc_dense1, units=self.hidden_num, activation=tf.nn.relu, name='encoder_dense2')

        with tf.variable_scope('decoder', reuse=tf.AUTO_REUSE):
            dec_dense1= tf.layers.dense(enc_dense2, units=self.hidden_num_mid, activation=tf.nn.relu, name='decoder_dense1')
            dec_dense2 = tf.layers.dense(dec_dense1, units=self.elem_num, activation=None,
                                         name='decoder_dense2')

        emb_pool = tf.reduce_mean(enc_dense2, axis=0, keepdims=True)
        with tf.variable_scope('last_fc', reuse=tf.AUTO_REUSE):
            self.emb_all = tf.layers.dense(emb_pool, units=self.hidden_num, activation=tf.nn.relu, name='mean_pool')
        self.loss = 0.5*tf.reduce_mean(tf.square(inputs-dec_dense2))

        return self.emb_all, self.loss
