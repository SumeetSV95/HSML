""" Code for the MAML algorithm and network definitions. """
from __future__ import print_function
import math
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()      # TF1-style graph
tf.disable_v2_behavior() 
from lstm_tree import TreeLSTM
from image_embedding import ImageEmbedding
import sys
from tensorflow.keras import mixed_precision
tf.compat.v1.disable_eager_execution()  # you already run v1-style sessions
tf.config.optimizer.set_jit(True)       # XLA JIT


try:
    import special_grads
except KeyError as e:
    print('WARN: Cannot define MaxPoolGrad, likely already defined for this version of tensorflow: %s' % e,
          file=sys.stderr)

from tensorflow.python.platform import flags
from utils import mse, xent, conv_block, normalize
from task_embedding import LSTMAutoencoder, MeanAutoencoder

FLAGS = flags.FLAGS

def conv_block_for_nonstationary(inp, cweight, bweight, reuse, scope):
    """
    A standard conv-bn-relu-pool block that matches the PyTorch NonStationaryEncoder.
    Convolution -> BatchNorm -> ReLU -> MaxPool
    """
    # Convolution
    conv_output = tf.nn.conv2d(inp, cweight, [1, 1, 1, 1], 'SAME') + bweight
    # BatchNorm and ReLU
    normed = normalize(conv_output, activation=tf.nn.relu, reuse=reuse, scope=scope)
    # Max Pooling
    pooled = tf.nn.max_pool(normed, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    return pooled


def versa_conv_block_tf(inp, cweight, bweight, reuse, scope, dropout_rate):
    conv_output = tf.nn.conv2d(inp, cweight, [1, 1, 1, 1], 'SAME') + bweight
    normed = normalize(conv_output, activation=tf.nn.relu, reuse=reuse, scope=scope)
    rate = max(0.0, min(1.0, float(dropout_rate)))
    if rate > 0.0:
        normed = tf.nn.dropout(normed, keep_prob=1.0 - rate)
    pooled = tf.nn.max_pool(normed, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    return pooled

class MAML:
    def __init__(self, sess, dim_input=1, dim_output=1, test_num_updates=5):
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.update_lr = FLAGS.update_lr
        self.meta_lr = tf.placeholder_with_default(FLAGS.meta_lr, ())
        self.classification = False
        self.cluster_layer_0 = FLAGS.cluster_layer_0
        self.cluster_layer_1 = FLAGS.cluster_layer_1
        self.test_num_updates = test_num_updates
        self.sess = sess
        if FLAGS.task_embedding_type == 'rnn':
            self.lstmae = LSTMAutoencoder(hidden_num=FLAGS.hidden_dim)
        elif FLAGS.task_embedding_type == 'mean':
            self.lstmae = MeanAutoencoder(hidden_num=FLAGS.hidden_dim)
        if FLAGS.datasource in ['sinusoid', 'mixture']:
            self.dim_hidden = [40, 40]
            self.loss_func = mse
            self.forward = self.forward_fc
            self.construct_weights = self.construct_fc_weights
        elif FLAGS.datasource in ['omniglot', 'miniimagenet', 'multidataset', 'mnist_rotations', 'mnist_permutations', 'nonstationary', 'vggflowers_stationary']:
            self.loss_func = xent
            self.classification = True
            if FLAGS.conv:
                self.dim_hidden = FLAGS.num_filters
                # --- START OF CHANGE ---
                if FLAGS.datasource == 'nonstationary':
                    self.forward = self.forward_conv_nonstationary
                elif FLAGS.datasource == 'vggflowers_stationary':
                    self.forward = self.forward_conv_stationary
                else:
                    self.forward = self.forward_conv
                # --- END OF CHANGE ---
                self.construct_weights = self.construct_conv_weights
            else:
                if FLAGS.datasource in ['mnist_rotations', 'mnist_permutations']:
                    self.dim_hidden = [100, 100]            # 784 -> 100 -> 100
                else:
                    self.dim_hidden = [256, 128, 64, 64]
                self.forward = self.forward_fc
                self.construct_weights = self.construct_fc_weights
            if FLAGS.datasource in ['miniimagenet', 'multidataset', 'nonstationary', 'vggflowers_stationary']:
                self.channels = 3
            else:
                self.channels = 1
            self.img_size = int(np.sqrt(self.dim_input / self.channels))
            self.image_embed = ImageEmbedding(hidden_num=FLAGS.task_embedding_num_filters, channels=self.channels,
                                              img_size=self.img_size,
                                              conv_initializer=tf.truncated_normal_initializer(stddev=0.04))
            self.stationary_dropout = getattr(FLAGS, 'stationary_dropout', 0.0)
        else:
            raise ValueError('Unrecognized data source.')
    @staticmethod
    def _accuracy_from_logits(logits, labels):
        """Works for int labels or one-hot labels."""
        pred = tf.argmax(logits, axis=1, output_type=tf.int32)
        if len(labels.shape) > 1:
            true = tf.argmax(labels, axis=1, output_type=tf.int32)
        else:
            true = tf.cast(labels, tf.int32)
        return tf.reduce_mean(tf.cast(tf.equal(pred, true), tf.float32))

    def construct_model(self, input_tensors=None, prefix='metatrain_'):
        if input_tensors is None:
            if FLAGS.datasource in ['sinusoid', 'mixture']:
                self.inputa = tf.placeholder(tf.float32, shape=(FLAGS.meta_batch_size, FLAGS.update_batch_size, 1))
                self.inputb = tf.placeholder(tf.float32,
                                             shape=(FLAGS.meta_batch_size, FLAGS.update_batch_size_eval, 1))
                self.labela = tf.placeholder(tf.float32, shape=(FLAGS.meta_batch_size, FLAGS.update_batch_size, 1))
                self.labelb = tf.placeholder(tf.float32,
                                             shape=(FLAGS.meta_batch_size, FLAGS.update_batch_size_eval, 1))
            else:
                self.inputa = tf.placeholder(tf.float32, shape=(FLAGS.meta_batch_size, FLAGS.update_batch_size, self.dim_input))
                self.inputb = tf.placeholder(tf.float32, shape=(FLAGS.meta_batch_size, FLAGS.update_batch_size_eval, self.dim_input))
                self.labela = tf.placeholder(tf.float32, shape=(FLAGS.meta_batch_size, FLAGS.update_batch_size))
                self.labelb = tf.placeholder(tf.float32, shape=(FLAGS.meta_batch_size, FLAGS.update_batch_size_eval))
        else:
            self.inputa = input_tensors['inputa']
            self.inputb = input_tensors['inputb']
            self.labela = input_tensors['labela']
            self.labelb = input_tensors['labelb']

        with tf.variable_scope('tree', reuse=tf.AUTO_REUSE):
            self.tree = TreeLSTM(input_dim=FLAGS.hidden_dim, tree_hidden_dim=FLAGS.hidden_dim,
                                 cluster_layer_0=self.cluster_layer_0, cluster_layer_1=self.cluster_layer_1)

        with tf.variable_scope('model', reuse=tf.AUTO_REUSE) as training_scope:
            if 'weights' in dir(self):
                training_scope.reuse_variables()
                weights = self.weights
            else:
                self.weights = weights = self.construct_weights()

            lossesa, outputas, lossesb, outputbs, emb_loss = [], [], [], [], []
            accuraciesa, accuraciesb = [], []
            num_updates = max(self.test_num_updates, FLAGS.num_updates)
            outputbs = [[]] * num_updates
            lossesb = [[]] * num_updates
            accuraciesb = [[]] * num_updates

            def task_metalearn(inp, reuse=True):
                inputa, inputb, labela, labelb = inp

                if self.classification:
                    labela = tf.one_hot(tf.cast(labela, dtype=tf.int32), self.dim_output)
                    labelb = tf.one_hot(tf.cast(labelb, dtype=tf.int32), self.dim_output)

                if FLAGS.datasource in ['sinusoid', 'mixture']:
                    input_task_emb = tf.concat((inputa, labela), axis=-1)
                elif FLAGS.datasource in ['miniimagenet', 'omniglot', 'multidataset', 'mnist_rotations', 'mnist_permutations', 'nonstationary', 'vggflowers_stationary']:
                    if FLAGS.fix_embedding_sample != -1:
                        input_task_emb = self.image_embed.model(tf.reshape(inputa[:FLAGS.fix_embedding_sample],
                                                                           [-1, self.img_size, self.img_size,
                                                                            self.channels]))
                        one_hot_labela = tf.squeeze(
                            tf.one_hot(tf.to_int32(labela[:FLAGS.fix_embedding_sample]), depth=1, axis=-1))
                    else:
                        reshaped_images = tf.reshape(inputa, [-1, self.img_size, self.img_size, self.channels])
                        image_embeddings = self.image_embed(reshaped_images, reuse=reuse)
                        input_task_emb = tf.concat((image_embeddings, labela), axis=-1)

                input_task_emb = tf.expand_dims(input_task_emb, axis=0)
                input_task_emb = tf.reshape(input_task_emb, [1, FLAGS.update_batch_size, self.lstmae.elem_num])

                task_embed_vec, task_emb_loss = self.lstmae.model(input_task_emb)

                _, meta_knowledge_h = self.tree.model(task_embed_vec)

                task_enhanced_emb_vec = tf.concat([task_embed_vec, meta_knowledge_h], axis=1)

                with tf.variable_scope('task_specific_mapping', reuse=tf.AUTO_REUSE):
                    if FLAGS.emb_type == 'sigmoid':
                        eta = []
                        for key in weights.keys():
                            weight_size = np.prod(weights[key].get_shape().as_list())
                            eta.append(tf.reshape(
                                tf.layers.dense(task_enhanced_emb_vec, weight_size, activation=tf.nn.sigmoid,
                                                name='eta_{}'.format(key)), tf.shape(weights[key])))
                        eta = dict(zip(weights.keys(), eta))

                        task_weights = dict(zip(weights.keys(), [weights[key] * eta[key] for key in weights.keys()]))

                task_outputbs, task_lossesb = [], []

                if self.classification:
                    task_accuraciesb = []

                task_outputa = self.forward(inputa, task_weights, reuse=reuse)
                task_lossa = self.loss_func(task_outputa, labela)

                grads = tf.gradients(task_lossa, list(task_weights.values()))
                if FLAGS.stop_grad:
                    grads = [tf.stop_gradient(grad) for grad in grads]
                gradients = dict(zip(task_weights.keys(), grads))
                fast_weights = dict(
                    zip(task_weights.keys(),
                        [task_weights[key] - self.update_lr * gradients[key] for key in task_weights.keys()]))
                output = self.forward(inputb, fast_weights, reuse=True)
                task_outputbs.append(output)
                task_lossesb.append(self.loss_func(output, labelb))
                for j in range(num_updates - 1):
                    loss = self.loss_func(self.forward(inputa, fast_weights, reuse=True), labela)
                    grads = tf.gradients(loss, list(fast_weights.values()))
                    if FLAGS.stop_grad:
                        grads = [tf.stop_gradient(grad) for grad in grads]
                    gradients = dict(zip(fast_weights.keys(), grads))
                    fast_weights = dict(zip(fast_weights.keys(),
                                            [fast_weights[key] - self.update_lr * gradients[key] for key in
                                             fast_weights.keys()]))
                    output = self.forward(inputb, fast_weights, reuse=True)
                    task_outputbs.append(output)
                    task_lossesb.append(self.loss_func(output, labelb))

                task_output = [task_emb_loss, task_outputa, task_outputbs, task_lossa, task_lossesb]

                if self.classification:
                    task_accuracya = self._accuracy_from_logits(task_outputa, labela)

                    for j in range(num_updates):
                        task_accuraciesb.append(
                            self._accuracy_from_logits(task_outputbs[j], labelb))
                    task_output.extend([task_accuracya, task_accuraciesb])

                return task_output

            if FLAGS.norm != 'None':
                unused = task_metalearn((self.inputa[0], self.inputb[0], self.labela[0], self.labelb[0]), False)

            out_dtype = [tf.float32, tf.float32, [tf.float32] * num_updates, tf.float32, [tf.float32] * num_updates]
            if self.classification:
                out_dtype.extend([tf.float32, [tf.float32] * num_updates])
            result = tf.map_fn(task_metalearn, elems=(self.inputa, self.inputb, self.labela, self.labelb),
                               dtype=out_dtype, parallel_iterations=FLAGS.meta_batch_size)
            if self.classification:
                emb_loss, outputas, outputbs, lossesa, lossesb, accuraciesa, accuraciesb = result
            else:
                emb_loss, outputas, outputbs, lossesa, lossesb = result

        if 'train' in prefix:
            self.total_loss1 = total_loss1 = tf.reduce_sum(lossesa) / tf.to_float(FLAGS.meta_batch_size)
            self.total_losses2 = total_losses2 = [tf.reduce_sum(lossesb[j]) / tf.to_float(FLAGS.meta_batch_size) for j
                                                  in range(num_updates)]
            self.total_embed_loss = total_embed_loss = tf.reduce_sum(emb_loss) / tf.to_float(FLAGS.meta_batch_size)
            self.outputas, self.outputbs = outputas, outputbs
            if self.classification:
                self.total_accuracy1 = total_accuracy1 = tf.reduce_sum(accuraciesa) / tf.to_float(FLAGS.meta_batch_size)
                self.total_accuracies2 = total_accuracies2 = [
                    tf.reduce_sum(accuraciesb[j]) / tf.to_float(FLAGS.meta_batch_size) for j in range(num_updates)]
            self.pretrain_op = tf.train.AdamOptimizer(self.meta_lr).minimize(total_loss1)

            if FLAGS.metatrain_iterations > 0:
                optimizer = tf.train.AdamOptimizer(self.meta_lr)
                self.gvs = gvs = optimizer.compute_gradients(
                    self.total_losses2[FLAGS.num_updates - 1] + FLAGS.emb_loss_weight * self.total_embed_loss)
                if FLAGS.task_embedding_type == 'mean':
                    gvs = [(tf.clip_by_value(grad, -3, 3), var) for grad, var in gvs]
                if FLAGS.datasource == 'miniimagenet':
                    gvs = [(tf.clip_by_value(grad, -10, 10), var) for grad, var in gvs]
                self.metatrain_op = optimizer.apply_gradients(gvs)
        else:
            self.metaval_total_loss1 = total_loss1 = tf.reduce_sum(lossesa) / tf.to_float(FLAGS.meta_batch_size)
            self.metaval_total_losses2 = total_losses2 = [tf.reduce_sum(lossesb[j]) / tf.to_float(FLAGS.meta_batch_size)
                                                          for j in range(num_updates)]
            if self.classification:
                self.metaval_total_accuracy1 = total_accuracy1 = tf.reduce_sum(accuraciesa) / tf.to_float(
                    FLAGS.meta_batch_size)
                self.metaval_total_accuracies2 = total_accuracies2 = [
                    tf.reduce_sum(accuraciesb[j]) / tf.to_float(FLAGS.meta_batch_size) for j in range(num_updates)]

        tf.summary.scalar(prefix + 'Pre-update loss', total_loss1)
        if self.classification:
            tf.summary.scalar(prefix + 'Pre-update accuracy', total_accuracy1)

        for j in range(num_updates):
            tf.summary.scalar(prefix + 'Post-update loss, step ' + str(j + 1), total_losses2[j])
            if self.classification:
                tf.summary.scalar(prefix + 'Post-update accuracy, step ' + str(j + 1), total_accuracies2[j])

    def construct_fc_weights(self):
        weights = {}
        weights['w1'] = tf.Variable(tf.truncated_normal([self.dim_input, self.dim_hidden[0]], stddev=0.01))
        weights['b1'] = tf.Variable(tf.zeros([self.dim_hidden[0]]))
        for i in range(1, len(self.dim_hidden)):
            weights['w' + str(i + 1)] = tf.Variable(
                tf.truncated_normal([self.dim_hidden[i - 1], self.dim_hidden[i]], stddev=0.01))
            weights['b' + str(i + 1)] = tf.Variable(tf.zeros([self.dim_hidden[i]]))
        weights['w' + str(len(self.dim_hidden) + 1)] = tf.Variable(
            tf.truncated_normal([self.dim_hidden[-1], self.dim_output], stddev=0.01))
        weights['b' + str(len(self.dim_hidden) + 1)] = tf.Variable(tf.zeros([self.dim_output]))
        return weights

    def forward_fc(self, inp, weights, reuse=False):
        hidden = normalize(tf.matmul(inp, weights['w1']) + weights['b1'], activation=tf.nn.relu, reuse=reuse, scope='0')
        for i in range(1, len(self.dim_hidden)):
            hidden = normalize(tf.matmul(hidden, weights['w' + str(i + 1)]) + weights['b' + str(i + 1)],
                               activation=tf.nn.relu, reuse=reuse, scope=str(i + 1))
        return tf.matmul(hidden, weights['w' + str(len(self.dim_hidden) + 1)]) + weights[
            'b' + str(len(self.dim_hidden) + 1)]

    def forward_conv_nonstationary(self, inp, weights, reuse=False, scope=''):
        """
        A dedicated 4-layer ConvNet feature extractor for the nonstationary setting.
        This version now uses the correct conv-then-pool block structure.
        """
        # Nonstationary data is 28x28 with 3 channels.
        image_size = 28
        inp = tf.reshape(inp, [-1, image_size, image_size, 3])

        # Block 1
        hidden1 = conv_block_for_nonstationary(inp, weights['conv1'], weights['b1'], reuse, scope + '0')
        # Block 2
        hidden2 = conv_block_for_nonstationary(hidden1, weights['conv2'], weights['b2'], reuse, scope + '1')
        # Block 3
        hidden3 = conv_block_for_nonstationary(hidden2, weights['conv3'], weights['b3'], reuse, scope + '2')
        # Block 4
        hidden4 = conv_block_for_nonstationary(hidden3, weights['conv4'], weights['b4'], reuse, scope + '3')

        # Flatten the output of the last conv block for the classifier
        # For a 28x28 input, the output of 4 pooling layers will be 1x1.
        # The shape will be [batch_size, 1, 1, num_filters].
        flat_output = tf.reshape(hidden4, [-1, np.prod([int(dim) for dim in hidden4.get_shape()[1:]])])

        # Final classification layer
        return tf.matmul(flat_output, weights['w5']) + weights['b5']

    def forward_conv_stationary(self, inp, weights, reuse=False, scope=''):
        image_size = self.img_size
        inp = tf.reshape(inp, [-1, image_size, image_size, self.channels])

        hidden1 = versa_conv_block_tf(inp, weights['conv1'], weights['b1'], reuse, scope + '0', self.stationary_dropout)
        hidden2 = versa_conv_block_tf(hidden1, weights['conv2'], weights['b2'], reuse, scope + '1', self.stationary_dropout)
        hidden3 = versa_conv_block_tf(hidden2, weights['conv3'], weights['b3'], reuse, scope + '2', self.stationary_dropout)
        hidden4 = versa_conv_block_tf(hidden3, weights['conv4'], weights['b4'], reuse, scope + '3', self.stationary_dropout)

        flat_output = tf.reshape(hidden4, [-1, np.prod([int(dim) for dim in hidden4.get_shape()[1:]])])
        return tf.matmul(flat_output, weights['w5']) + weights['b5']

    def construct_conv_weights(self):
        weights = {}

        dtype = tf.float32
        conv_initializer = tf.keras.initializers.glorot_uniform()
        fc_initializer = tf.keras.initializers.glorot_uniform()
        k = 3

        weights['conv1'] = tf.get_variable('conv1', [k, k, self.channels, self.dim_hidden],
                                           initializer=conv_initializer, dtype=dtype)
        weights['b1'] = tf.Variable(tf.zeros([self.dim_hidden]))
        weights['conv2'] = tf.get_variable('conv2', [k, k, self.dim_hidden, self.dim_hidden],
                                           initializer=conv_initializer, dtype=dtype)
        weights['b2'] = tf.Variable(tf.zeros([self.dim_hidden]))
        weights['conv3'] = tf.get_variable('conv3', [k, k, self.dim_hidden, self.dim_hidden],
                                           initializer=conv_initializer, dtype=dtype)
        weights['b3'] = tf.Variable(tf.zeros([self.dim_hidden]))
        weights['conv4'] = tf.get_variable('conv4', [k, k, self.dim_hidden, self.dim_hidden],
                                           initializer=conv_initializer, dtype=dtype)
        weights['b4'] = tf.Variable(tf.zeros([self.dim_hidden]))
        
        # --- START OF FIX ---
        # This calculation must match the actual pooling behavior.
        # The original `conv_block` and the new `conv_block_for_nonstationary` both
        # effectively use 'VALID' pooling which results in integer division.
        size = self.img_size
        for _ in range(4):
            # Use integer division to match 'VALID' pooling behavior.
            size = size // 2
        # --- END OF FIX ---
            
        flat_dim = self.dim_hidden * size * size
        weights['w5'] = tf.get_variable('w5', [flat_dim, self.dim_output],
                                        initializer=fc_initializer)
        weights['b5'] = tf.Variable(tf.zeros([self.dim_output]), name='b5')
        return weights

    def forward_conv(self, inp, weights, reuse=False, scope=''):
        # reuse is for the normalization parameters.
        channels = self.channels
        inp = tf.reshape(inp, [-1, self.img_size, self.img_size, channels])

        hidden1 = conv_block(inp, weights['conv1'], weights['b1'], reuse, scope + '0')
        hidden2 = conv_block(hidden1, weights['conv2'], weights['b2'], reuse, scope + '1')
        hidden3 = conv_block(hidden2, weights['conv3'], weights['b3'], reuse, scope + '2')
        hidden4 = conv_block(hidden3, weights['conv4'], weights['b4'], reuse, scope + '3')
        if FLAGS.datasource in ['miniimagenet', 'multidataset', 'nonstationary']:
            # last hidden layer is 6x6x64-ish, reshape to a vector
            hidden4 = tf.reshape(hidden4, [-1, np.prod([int(dim) for dim in hidden4.get_shape()[1:]])])
        else:
            hidden4 = tf.reduce_mean(hidden4, [1, 2])

        return tf.matmul(hidden4, weights['w5']) + weights['b5']

    def get_params(self):
        tvars = tf.trainable_variables()
        return tvars
