""" Code for loading data. """
import numpy as np
import os
import random
import tensorflow as tf
import ipdb
import pickle
import os

import torch

try:
    from non_stationary_datasets import get_dataset_registry
except ImportError:
    get_dataset_registry = None  # optional dependency; checked at runtime

from tensorflow.python.platform import flags
from utils import get_images
import matplotlib.pyplot as plt

FLAGS = flags.FLAGS


class DataGenerator(object):
    """
    Data Generator capable of generating batches of sinusoid or Omniglot data.
    A "class" is considered a class of omniglot digits or a particular sinusoid function.
    """

    def __init__(self, num_samples_per_class, batch_size, config={}):
        """
        Args:
            num_samples_per_class: num samples to generate per class in one batch
            batch_size: size of meta batch size (e.g. number of functions)
        """
        self.batch_size = batch_size
        self.num_samples_per_class = num_samples_per_class
        self.num_classes = 1  # by default 1 (only relevant for classification problems)
        self.recurring_to_orig = {}
        self.logical_task_ids = []

        if FLAGS.datasource == 'sinusoid':
            self.generate = self.generate_sinusoid_batch
            # self.amp_range = config.get('amp_range', [0.1, 5.0])
            # self.phase_range = config.get('phase_range', [0, np.pi])
            self.amp_range = config.get('amp_range', [0.1, 5.0])
            self.freq_range = config.get('freq_range', [0.8, 1.2])
            self.phase_range = config.get('phase_range', [0, np.pi])
            self.input_range = config.get('input_range', [-5.0, 5.0])
            self.dim_input = 1
            self.dim_output = 1

        elif FLAGS.datasource == 'mixture':
            if FLAGS.online_training:
                self.generate = self.online_generate_mixture_batch
            else:
                self.generate = self.generate_mixture_batch
            self.dim_input = 1
            self.dim_output = 1
            self.input_range = config.get('input_range', [-5.0, 5.0])

        elif 'omniglot' in FLAGS.datasource:
            self.num_classes = config.get('num_classes', FLAGS.num_classes)
            self.img_size = config.get('img_size', (28, 28))
            self.dim_input = np.prod(self.img_size)
            self.dim_output = self.num_classes
            # data that is pre-resized using PIL with lanczos filter
            data_folder = config.get('data_folder', '{}/omniglot_resized'.format(FLAGS.datadir))

            character_folders = [os.path.join(data_folder, family, character) \
                                 for family in os.listdir(data_folder) \
                                 if os.path.isdir(os.path.join(data_folder, family)) \
                                 for character in os.listdir(os.path.join(data_folder, family))]
            random.seed(1)
            random.shuffle(character_folders)
            if FLAGS.no_val:
                num_val = 0
            else:
                num_val = 100
            num_train = config.get('num_train', 1200) - num_val
            self.metatrain_character_folders = character_folders[:num_train]
            if FLAGS.test_set:
                self.metaval_character_folders = character_folders[num_train + num_val:]
            else:
                self.metaval_character_folders = character_folders[num_train:num_train + num_val]
            self.rotations = config.get('rotations', [0, 90, 180, 270])
        elif FLAGS.datasource == 'miniimagenet':
            self.num_classes = config.get('num_classes', FLAGS.num_classes)
            self.img_size = config.get('img_size', (84, 84))
            self.dim_input = np.prod(self.img_size) * 3
            self.dim_output = self.num_classes
            metatrain_folder = config.get('metatrain_folder', '{}/miniImagenet/train'.format(FLAGS.datadir))
            if FLAGS.test_set:
                metaval_folder = config.get('metaval_folder', '{}/miniImagenet/test'.format(FLAGS.datadir))
            else:
                metaval_folder = config.get('metaval_folder', '{}/miniImagenet/val'.format(FLAGS.datadir))

            metatrain_folders = [os.path.join(metatrain_folder, label) \
                                 for label in os.listdir(metatrain_folder) \
                                 if os.path.isdir(os.path.join(metatrain_folder, label)) \
                                 ]
            metaval_folders = [os.path.join(metaval_folder, label) \
                               for label in os.listdir(metaval_folder) \
                               if os.path.isdir(os.path.join(metaval_folder, label)) \
                               ]
            self.metatrain_character_folders = metatrain_folders
            self.metaval_character_folders = metaval_folders
            self.rotations = config.get('rotations', [0])

        elif FLAGS.datasource == 'multidataset':
            self.num_classes = config.get('num_classes', FLAGS.num_classes)
            self.img_size = config.get('img_size', (84, 84))
            self.dim_input = np.prod(self.img_size) * 3
            self.dim_output = self.num_classes
            self.multidataset = ['CUB_Bird', 'DTD_Texture', 'FGVC_Aircraft', 'FGVCx_Fungi']
            metatrain_folders, metaval_folders = [], []
            for eachdataset in self.multidataset:
                metatrain_folders.append(
                    [os.path.join('{0}/meta-dataset/{1}/train'.format(FLAGS.datadir, eachdataset), label) \
                     for label in os.listdir('{0}/meta-dataset/{1}/train'.format(FLAGS.datadir, eachdataset)) \
                     if
                     os.path.isdir(os.path.join('{0}/meta-dataset/{1}/train'.format(FLAGS.datadir, eachdataset), label)) \
                     ])
                if FLAGS.test_set:
                    metaval_folders.append(
                        [os.path.join('{0}/meta-dataset/{1}/test'.format(FLAGS.datadir, eachdataset), label) \
                         for label in os.listdir('{0}/meta-dataset/{1}/test'.format(FLAGS.datadir, eachdataset)) \
                         if os.path.isdir(
                            os.path.join('{0}/meta-dataset/{1}/test'.format(FLAGS.datadir, eachdataset), label)) \
                         ])
                else:
                    metaval_folders.append(
                        [os.path.join('{0}/meta-dataset/{1}/val'.format(FLAGS.datadir, eachdataset), label) \
                         for label in os.listdir('{0}/meta-dataset/{1}/val'.format(FLAGS.datadir, eachdataset)) \
                         if os.path.isdir(
                            os.path.join('{0}/meta-dataset/{1}/val'.format(FLAGS.datadir, eachdataset), label)) \
                         ])
            self.metatrain_character_folders = metatrain_folders
            self.metaval_character_folders = metaval_folders
            self.rotations = config.get('rotations', [0])
        elif FLAGS.datasource == 'mnist_rotations':
            self.mnist_generator = MNIST_Rotations_DataGenerator(FLAGS.datadir)
            self.generate = self.mnist_generator.generate_task_batch
            self.num_classes = self.mnist_generator.num_classes
            self.dim_input = self.mnist_generator.dim_input
            self.dim_output = self.mnist_generator.dim_output
            # --- START OF FIX ---
            # Populate the main DataGenerator's attributes with the data loaded by the internal generator.
            self.dataset_train = self.mnist_generator.train_data
            self.dataset_val   = getattr(self.mnist_generator, 'val_data', [])
            self.dataset_test  = getattr(self.mnist_generator, 'test_data', [])
            self.num_total_tasks = len(self.dataset_train)
            self.recurring_to_orig = getattr(self.mnist_generator, 'recurring_to_orig', {})
            self.logical_task_ids = getattr(self.mnist_generator, 'logical_task_ids', list(range(self.num_total_tasks)))
            # --- END OF FIX ---
        elif FLAGS.datasource == 'mnist_permutations':
            self.mnist_generator = MNIST_Permutations_DataGenerator(FLAGS.datadir)
            self.generate = self.mnist_generator.generate_task_batch
            self.num_classes = self.mnist_generator.num_classes
            self.dim_input = self.mnist_generator.dim_input
            self.dim_output = self.mnist_generator.dim_output
            self.dataset_train = self.mnist_generator.train_data
            self.dataset_val   = getattr(self.mnist_generator, 'val_data', [])
            self.dataset_test  = getattr(self.mnist_generator, 'test_data', [])
            self.num_total_tasks = len(self.dataset_train)
            self.recurring_to_orig = getattr(self.mnist_generator, 'recurring_to_orig', {})
            self.logical_task_ids = getattr(self.mnist_generator, 'logical_task_ids', list(range(self.num_total_tasks)))
        elif FLAGS.datasource in ['nonstationary', 'vggflowers_stationary']:
            if get_dataset_registry is None:
                raise ImportError('non_stationary_datasets.py not available; required for nonstationary setting.')

            data_root = getattr(FLAGS, 'nonstat_data_root', '') or os.path.join(FLAGS.datadir, 'non_stationary')
            train_registry, dataset_order = get_dataset_registry(data_root, split='train')
            test_registry, _ = get_dataset_registry(data_root, split=getattr(FLAGS, 'nonstat_test_split', 'test'))

            if len(train_registry) == 0:
                raise ValueError('No datasets found for nonstationary benchmark. Check --nonstat_data_root path.')

            self.nonstat_train_registry = train_registry
            self.nonstat_test_registry = test_registry
            self.nonstat_order = dataset_order

            n_way = getattr(FLAGS, 'nonstat_n_way', FLAGS.num_classes)
            k_shot = getattr(FLAGS, 'nonstat_k_shot', FLAGS.update_batch_size)
            q_query = getattr(FLAGS, 'nonstat_q_query', FLAGS.update_batch_size_eval)

            self.n_way = n_way
            self.k_shot = k_shot
            self.q_query = q_query

            if FLAGS.datasource == 'vggflowers_stationary':
                target = getattr(FLAGS, 'stationary_dataset', 'vggflowers')
                if target not in train_registry:
                    raise ValueError('Stationary dataset {} not found in registry.'.format(target))
                self.stationary_dataset = target
                self.nonstat_order = [target]

            sample_dataset = self.nonstat_train_registry[self.nonstat_order[0]]
            sample_x = sample_dataset.xs[0]
            if isinstance(sample_x, torch.Tensor):
                sample_x = sample_x.cpu().numpy()
            self.channels = sample_x.shape[0]
            self.img_size = sample_x.shape[1:]
            self.dim_input = int(np.prod(self.img_size) * self.channels)
            self.num_classes = n_way
            self.dim_output = n_way

            # ensure HSML placeholders align with episode geometry
            FLAGS.num_classes = n_way
            FLAGS.update_batch_size = n_way * k_shot
            FLAGS.update_batch_size_eval = n_way * q_query

            self.flatten_cache = {}

        else:
            raise ValueError('Unrecognized data source')

        if not self.logical_task_ids and hasattr(self, 'dataset_train') and self.dataset_train:
            self.logical_task_ids = list(range(len(self.dataset_train)))

    def make_data_tensor(self, train=True):
        if train:
            folders = self.metatrain_character_folders
            # number of tasks, not number of meta-iterations. (divide by metabatch size to measure)
            num_total_batches = 200000
        else:
            folders = self.metaval_character_folders
            num_total_batches = 600

        # make list of files
        print('Generating filenames')
        all_filenames = []
        for _ in range(num_total_batches):
            sampled_character_folders = random.sample(folders, self.num_classes)
            random.shuffle(sampled_character_folders)
            labels_and_images = get_images(sampled_character_folders, range(self.num_classes),
                                           nb_samples=self.num_samples_per_class, shuffle=False)
            # make sure the above isn't randomized order
            labels = [li[0] for li in labels_and_images]
            filenames = [li[1] for li in labels_and_images]
            all_filenames.extend(filenames)

        # make queue for tensorflow to read from
        filename_queue = tf.train.string_input_producer(tf.convert_to_tensor(all_filenames), shuffle=False)
        print('Generating image processing ops')
        image_reader = tf.WholeFileReader()
        _, image_file = image_reader.read(filename_queue)
        if FLAGS.datasource == 'miniimagenet':
            image = tf.image.decode_jpeg(image_file, channels=3)
            image.set_shape((self.img_size[0], self.img_size[1], 3))
            image = tf.reshape(image, [self.dim_input])
            image = tf.cast(image, tf.float32) / 255.0
        else:
            image = tf.image.decode_png(image_file)
            image.set_shape((self.img_size[0], self.img_size[1], 1))
            image = tf.reshape(image, [self.dim_input])
            image = tf.cast(image, tf.float32) / 255.0
            image = 1.0 - image  # invert
        num_preprocess_threads = 1  # TODO - enable this to be set to >1
        min_queue_examples = 256
        examples_per_batch = self.num_classes * self.num_samples_per_class
        batch_image_size = self.batch_size * examples_per_batch
        print('Batching images')
        images = tf.train.batch(
            [image],
            batch_size=batch_image_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_image_size,
        )
        all_image_batches, all_label_batches = [], []
        print('Manipulating image data to be right shape')
        for i in range(self.batch_size):
            image_batch = images[i * examples_per_batch:(i + 1) * examples_per_batch]

            if FLAGS.datasource == 'omniglot':
                # omniglot augments the dataset by rotating digits to create new classes
                # get rotation per class (e.g. 0,1,2,0,0 if there are 5 classes)
                rotations = tf.multinomial(tf.log([[1., 1., 1., 1.]]), self.num_classes)
            label_batch = tf.convert_to_tensor(labels)
            new_list, new_label_list = [], []
            for k in range(self.num_samples_per_class):
                class_idxs = tf.range(0, self.num_classes)
                class_idxs = tf.random_shuffle(class_idxs)

                true_idxs = class_idxs * self.num_samples_per_class + k

                new_list.append(tf.gather(image_batch, true_idxs))
                if FLAGS.datasource == 'omniglot':  # and FLAGS.train:
                    new_list[-1] = tf.stack([tf.reshape(tf.image.rot90(
                        tf.reshape(new_list[-1][ind], [self.img_size[0], self.img_size[1], 1]),
                        k=tf.cast(rotations[0, class_idxs[ind]], tf.int32)), (self.dim_input,))
                        for ind in range(self.num_classes)])
                new_label_list.append(tf.gather(label_batch, true_idxs))
            new_list = tf.concat(new_list, 0)  # has shape [self.num_classes*self.num_samples_per_class, self.dim_input]
            new_label_list = tf.concat(new_label_list, 0)
            all_image_batches.append(new_list)
            all_label_batches.append(new_label_list)
        all_image_batches = tf.stack(all_image_batches)
        all_label_batches = tf.stack(all_label_batches)
        all_label_batches = tf.one_hot(all_label_batches, self.num_classes)
        return all_image_batches, all_label_batches

    def make_data_tensor_multidataset(self, sel_num, train=True):
        print('when cluster increase, now the selection number is {}'.format(sel_num))
        if train:
            folders = self.metatrain_character_folders
            # number of tasks, not number of meta-iterations. (divide by metabatch size to measure)
            num_total_batches = 200000
        else:
            folders = self.metaval_character_folders
            num_total_batches = FLAGS.num_test_task
        # make list of files
        print('Generating filenames')
        all_filenames = []
        # if FLAGS.train == False:
        #     np.random.seed(4)
        for image_itr in range(num_total_batches):
            if FLAGS.online_training:
                if sel_num==2:
                    if image_itr < 60000:
                        sel=np.random.randint(2)
                    elif image_itr >= 60000:
                        sel=np.random.randint(3)
                elif sel_num==3:
                    if image_itr < 40000:
                        sel=np.random.randint(3)
                    elif image_itr >= 40000:
                        sel = np.random.randint(4)
                else:
                    sel=np.random.randint(4)
            else:
                sel = np.random.randint(4)
            if FLAGS.train == False and FLAGS.test_dataset != -1:
                sel = FLAGS.test_dataset
            sampled_character_folders = random.sample(folders[sel], self.num_classes)
            random.shuffle(sampled_character_folders)
            labels_and_images = get_images(sampled_character_folders, range(self.num_classes),
                                           nb_samples=self.num_samples_per_class, shuffle=False)
            # make sure the above isn't randomized order
            labels = [li[0] for li in labels_and_images]
            filenames = [li[1] for li in labels_and_images]
            all_filenames.extend(filenames)

        # make queue for tensorflow to read from
        filename_queue = tf.train.string_input_producer(tf.convert_to_tensor(all_filenames), shuffle=False)
        print('Generating image processing ops')
        image_reader = tf.WholeFileReader()
        _, image_file = image_reader.read(filename_queue)
        if FLAGS.datasource in ['miniimagenet', 'multidataset']:
            image = tf.image.decode_jpeg(image_file, channels=3)
            image.set_shape((self.img_size[0], self.img_size[1], 3))
            image = tf.reshape(image, [self.dim_input])
            image = tf.cast(image, tf.float32) / 255.0
        else:
            image = tf.image.decode_png(image_file)
            image.set_shape((self.img_size[0], self.img_size[1], 1))
            image = tf.reshape(image, [self.dim_input])
            image = tf.cast(image, tf.float32) / 255.0
            image = 1.0 - image  # invert
        num_preprocess_threads = 1  # TODO - enable this to be set to >1
        min_queue_examples = 256
        examples_per_batch = self.num_classes * self.num_samples_per_class
        batch_image_size = self.batch_size * examples_per_batch
        print('Batching images')
        images = tf.train.batch(
            [image],
            batch_size=batch_image_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_image_size,
        )
        all_image_batches, all_label_batches = [], []
        print('Manipulating image data to be right shape')
        for i in range(self.batch_size):
            image_batch = images[i * examples_per_batch:(i + 1) * examples_per_batch]
            label_batch = tf.convert_to_tensor(labels)
            new_list, new_label_list = [], []
            for k in range(self.num_samples_per_class):
                class_idxs = tf.range(0, self.num_classes)
                class_idxs = tf.random_shuffle(class_idxs)
                true_idxs = class_idxs * self.num_samples_per_class + k
                new_list.append(tf.gather(image_batch, true_idxs))
                new_label_list.append(tf.gather(label_batch, true_idxs))
            new_list = tf.concat(new_list, 0)  # has shape [self.num_classes*self.num_samples_per_class, self.dim_input]
            new_label_list = tf.concat(new_label_list, 0)
            all_image_batches.append(new_list)
            all_label_batches.append(new_label_list)
        all_image_batches = tf.stack(all_image_batches)
        all_label_batches = tf.stack(all_label_batches)
        all_label_batches = tf.one_hot(all_label_batches, self.num_classes)
        return all_image_batches, all_label_batches

    def generate_sinusoid_batch(self, train=True, input_idx=None, itr=0):
        """
        Generates a batch of sinusoid tasks for the continual online setting.
        The complexity of the tasks changes based on the training iteration.
        """
        # Define different "regimes" of tasks
        # Regime 0: Simple sine waves (stable frequency)
        amp_0 = np.random.uniform(0.1, 5.0, [self.batch_size])
        phase_0 = np.random.uniform(0., np.pi, [self.batch_size])
        freq_0 = np.random.uniform(0.8, 1.2, [self.batch_size])

        # Regime 1: Higher frequency sine waves
        amp_1 = np.random.uniform(0.1, 5.0, [self.batch_size])
        phase_1 = np.random.uniform(0., np.pi, [self.batch_size])
        freq_1 = np.random.uniform(2.8, 3.2, [self.batch_size]) # New, harder frequency range

        init_inputs = np.zeros([self.batch_size, self.num_samples_per_class, self.dim_input])
        outputs = np.zeros([self.batch_size, self.num_samples_per_class, self.dim_output])
        sel_set = np.zeros(self.batch_size)

        for func in range(self.batch_size):
            init_inputs[func] = np.random.uniform(self.input_range[0], self.input_range[1],
                                                  size=(self.num_samples_per_class, self.dim_input))

            if FLAGS.train:
                # --- This is the continual learning logic ---
                if itr < 25000:
                    # For the first 25k iterations, only show simple sine waves
                    sel = 0
                else:
                    # After 25k iterations, introduce the new, harder sine waves.
                    # Importantly, we still sample the old tasks as well.
                    sel = np.random.randint(2)
                # --- End of logic ---
            else: # For testing, always sample from all seen tasks
                sel = np.random.randint(2)

            if sel == 0:
                outputs[func] = amp_0[func] * np.sin(freq_0[func] * init_inputs[func] + phase_0[func])
            elif sel == 1:
                outputs[func] = amp_1[func] * np.sin(freq_1[func] * init_inputs[func] + phase_1[func])
            
            sel_set[func] = sel

        # The function needs to return the same number of items as the mixture generator
        funcs_params = {'amp0': amp_0, 'phase0': phase_0, 'freq0': freq_0, 'amp1': amp_1, 'phase1': phase_1, 'freq1': freq_1}
        return init_inputs, outputs, funcs_params, sel_set


    def online_generate_mixture_batch(self, itr, train=True, input_idx=None, DRAW_PLOTS=True):
        dim_input = self.dim_input
        dim_output = self.dim_output
        batch_size = self.batch_size
        num_samples_per_class = self.num_samples_per_class

        # sin
        amp = np.random.uniform(0.1, 5.0, size=self.batch_size)
        phase = np.random.uniform(0., 2 * np.pi, size=batch_size)
        freq = np.random.uniform(0.8, 1.2, size=batch_size)

        # linear
        A = np.random.uniform(-3.0, 3.0, size=batch_size)
        b = np.random.uniform(-3.0, 3.0, size=batch_size)

        # quadratic
        A_q = np.random.uniform(-0.2, 0.2, size=batch_size)
        c_q = np.random.uniform(-2.0, 2.0, size=batch_size)
        b_q = np.random.uniform(-3.0, 3.0, size=batch_size)

        # cubic
        A_c = np.random.uniform(-0.1, 0.1, size=batch_size)
        b_c = np.random.uniform(-0.2, 0.2, size=batch_size)
        c_c = np.random.uniform(-2.0, 2.0, size=batch_size)
        d_c = np.random.uniform(-3.0, 3.0, size=batch_size)

        sel_set = np.zeros(batch_size)

        init_inputs = np.zeros([batch_size, num_samples_per_class, dim_input])
        outputs = np.zeros([batch_size, num_samples_per_class, dim_output])

        for func in range(batch_size):
            init_inputs[func] = np.random.uniform(self.input_range[0], self.input_range[1],
                                                  size=(num_samples_per_class, dim_input))

            if FLAGS.train:
                if itr < 15000:
                    sel = np.random.randint(2)
                    if sel == 0:
                        outputs[func] = amp[func] * np.sin(freq[func] * init_inputs[func]) + phase[func]
                    elif sel == 1:
                        outputs[func] = A[func] * init_inputs[func] + b[func]
                elif itr >= 15000 and itr < 30000:
                    sel = np.random.randint(3)
                    if sel == 0:
                        outputs[func] = amp[func] * np.sin(freq[func] * init_inputs[func]) + phase[func]
                    elif sel == 1:
                        outputs[func] = A[func] * init_inputs[func] + b[func]
                    elif sel == 2:
                        outputs[func] = A_q[func] * np.square(init_inputs[func] - c_q[func]) + b_q[func]
                elif itr>=30000:
                    sel = np.random.randint(4)
                    if sel == 0:
                        outputs[func] = amp[func] * np.sin(freq[func] * init_inputs[func]) + phase[func]
                    elif sel == 1:
                        outputs[func] = A[func] * init_inputs[func] + b[func]
                    elif sel == 2:
                        outputs[func] = A_q[func] * np.square(init_inputs[func] - c_q[func]) + b_q[func]
                    elif sel == 3:
                        outputs[func] = A_c[func] * np.power(init_inputs[func], np.tile([3], init_inputs[func].shape)) + \
                                        b_c[
                                            func] * np.square(init_inputs[func]) + c_c[func] * init_inputs[func] + d_c[func]
            else:
                sel = np.random.randint(4)
                if sel == 0:
                    outputs[func] = amp[func] * np.sin(freq[func] * init_inputs[func]) + phase[func]
                elif sel == 1:
                    outputs[func] = A[func] * init_inputs[func] + b[func]
                elif sel == 2:
                    outputs[func] = A_q[func] * np.square(init_inputs[func] - c_q[func]) + b_q[func]
                elif sel == 3:
                    outputs[func] = A_c[func] * np.power(init_inputs[func], np.tile([3], init_inputs[func].shape)) + \
                                    b_c[
                                        func] * np.square(init_inputs[func]) + c_c[func] * init_inputs[func] + d_c[func]
            sel_set[func] = sel
        funcs_params = {'amp': amp, 'phase': phase, 'freq': freq, 'A': A, 'b': b, 'A_q': A_q, 'c_q': c_q, 'b_q': b_q,
                        'A_c': A_c, 'b_c': b_c, 'c_c': c_c, 'd_c': d_c}
        return init_inputs, outputs, funcs_params, sel_set

    # -------- Nonstationary helpers (episode sampling) --------
    def _sample_episode_nonstationary(self, dataset, n_way=None, k_shot=None, q_query=None, return_global=False):
        if n_way is None:
            n_way = self.n_way
        if k_shot is None:
            k_shot = self.k_shot
        if q_query is None:
            q_query = self.q_query

        Sx, Sy, Qx, Qy, Sg, Qg = dataset.sample_n_way_k_shot(n_way, k_shot, q_query)
        if isinstance(Sx, torch.Tensor):
            Sx = Sx.cpu().numpy()
            Sy = Sy.cpu().numpy()
            Qx = Qx.cpu().numpy()
            Qy = Qy.cpu().numpy()
            Sg = Sg.cpu().numpy()
            Qg = Qg.cpu().numpy()

        K = n_way * k_shot
        Q = n_way * q_query
        Sx = Sx.reshape(K, -1).astype(np.float32)
        Qx = Qx.reshape(Q, -1).astype(np.float32)
        Sy = Sy.reshape(K).astype(np.int32)
        Qy = Qy.reshape(Q).astype(np.int32)
        Sg = Sg.reshape(K).astype(np.int32)
        Qg = Qg.reshape(Q).astype(np.int32)
        if return_global:
            return Sx, Sy, Qx, Qy, Sg, Qg
        return Sx, Sy, Qx, Qy

    def sample_meta_batch_nonstationary(self, dataset_name, meta_batch_size, n_way=None, k_shot=None, q_query=None):
        dataset = self.nonstat_train_registry[dataset_name]
        if n_way is None:
            n_way = self.n_way
        if k_shot is None:
            k_shot = self.k_shot
        if q_query is None:
            q_query = self.q_query

        K = n_way * k_shot
        Q = n_way * q_query
        inputa = np.zeros((meta_batch_size, K, self.dim_input), dtype=np.float32)
        labela = np.zeros((meta_batch_size, K), dtype=np.int32)
        inputb = np.zeros((meta_batch_size, Q, self.dim_input), dtype=np.float32)
        labelb = np.zeros((meta_batch_size, Q), dtype=np.int32)

        for i in range(meta_batch_size):
            Sx, Sy, Qx, Qy = self._sample_episode_nonstationary(dataset, n_way, k_shot, q_query)
            inputa[i] = Sx
            labela[i] = Sy
            inputb[i] = Qx
            labelb[i] = Qy

        return inputa, labela, inputb, labelb

    def sample_episode_nonstationary(self, dataset_name, n_way=None, k_shot=None, q_query=None, return_global=False):
        dataset = self.nonstat_train_registry[dataset_name]
        return self._sample_episode_nonstationary(dataset, n_way, k_shot, q_query, return_global=return_global)

    def sample_eval_episode_nonstationary(self, dataset_name, n_way=None, k_shot=None, q_query=None):
        dataset = self.nonstat_test_registry[dataset_name]
        return self._sample_episode_nonstationary(dataset, n_way, k_shot, q_query)

    def get_continuous_stream(self):
        """
        Flattens all task data into a single continuous stream where tasks appear sequentially.
        Handles mnist_rotations and other datasources correctly.
        """
        if FLAGS.datasource in ['mnist_rotations', 'mnist_permutations']:
            # Each entry is a tuple: (images, labels)
            all_images = np.concatenate([self.dataset_train[i][0] for i in range(self.num_total_tasks)])
            all_labels = np.concatenate([self.dataset_train[i][1] for i in range(self.num_total_tasks)])
        else:
            # Each entry is a dict: {'images': ..., 'labels': ...}
            all_images = np.concatenate([self.dataset_train[i]['images'] for i in range(self.num_total_tasks)])
            all_labels = np.concatenate([self.dataset_train[i]['labels'] for i in range(self.num_total_tasks)])
        return all_images, all_labels


# ADD THIS NEW CLASS AT THE VERY END OF THE FILE.
class MNIST_Rotations_DataGenerator(object):
    """
    Loads full Rotated MNIST data and creates sequential batches.
    This is only used when --datasource=mnist_rotations.
    """
    def __init__(self, datadir):
        self.num_classes = 10
        self.dim_input = 28*28
        self.dim_output = self.num_classes
        self.batch_offsets = {}

        file_path = os.path.join(datadir, "rotated_mnist", "mnist_all_rotation_normalized_train_valid.pkl")
        try:
            with open(file_path, "rb") as f:
                payload = pickle.load(f)
            if isinstance(payload, tuple):
                if len(payload) == 3:
                    train_raw, val_raw, test_raw = payload
                elif len(payload) == 2:
                    train_raw, val_raw = payload
                    test_raw = val_raw
                else:
                    train_raw = payload[0]
                    val_raw = payload[1] if len(payload) > 1 else []
                    test_raw = payload[2] if len(payload) > 2 else []
            else:
                train_raw, val_raw, test_raw = payload, [], []
            print(f"Loaded RotMNIST splits: train={len(train_raw)} val={len(val_raw)} test={len(test_raw)} tasks.")
        except FileNotFoundError:
            print(f"ERROR: data file not found at {file_path}")
            raise

        def _as_numpy(pair):
            x, y = pair
            if hasattr(x, "numpy"):
                x = x.numpy()
            if hasattr(y, "numpy"):
                y = y.numpy()
            return np.asarray(x), np.asarray(y)

        desired_order = [18, 1, 19, 8, 10, 17, 6, 13, 4, 2, 5, 14, 9, 7, 16, 11, 3, 0, 15, 12]
        if len(train_raw) == len(desired_order):
            order = desired_order
            train_raw = [train_raw[i] for i in order]
            if len(val_raw) == len(order):
                val_raw = [val_raw[i] for i in order]
            if len(test_raw) == len(order):
                test_raw = [test_raw[i] for i in order]
        else:
            order = list(range(len(train_raw)))
            print(f"Warning: got {len(train_raw)} tasks, expected {len(desired_order)}; keeping file order.")

        self.task_order = list(order)

        train_map = {orig: _as_numpy(train_raw[idx]) for idx, orig in enumerate(order)}
        if val_raw:
            if len(val_raw) == len(order):
                val_map = {orig: _as_numpy(val_raw[idx]) for idx, orig in enumerate(order)}
            else:
                val_map = {idx: _as_numpy(val_raw[idx]) for idx in range(len(val_raw))}
        else:
            val_map = {}
        if test_raw:
            if len(test_raw) == len(order):
                test_map = {orig: _as_numpy(test_raw[idx]) for idx, orig in enumerate(order)}
            else:
                test_map = {idx: _as_numpy(test_raw[idx]) for idx in range(len(test_raw))}
        else:
            test_map = {}

        self.recurring_to_orig = {}
        self.logical_task_ids = list(order)

        if getattr(FLAGS, 'recurring_rot_mnist', False):
            recurring_order = [8, 28, 9, 29, 12, 32, 13, 33, 16, 36,
                               17, 37, 0, 20, 1, 21, 4, 24, 5, 25]
            base_even = [orig for orig in sorted(train_map.keys()) if orig % 2 == 0 and orig <= 18]
            if len(base_even) != 10:
                raise ValueError(f"Recurring RotMNIST expects 10 even base tasks, got {len(base_even)} (ids={base_even})")

            recurring_train = {}
            mapping = {}
            for orig in base_even:
                images, labels = train_map[orig]
                total = images.shape[0]
                if total % 2 != 0:
                    raise ValueError(f"Task {orig} has {total} samples; recurring split requires an even count.")
                mid = total // 2
                halves = (
                    (orig * 2 + 0, images[:mid].copy(), labels[:mid].copy()),
                    (orig * 2 + 1, images[mid:].copy(), labels[mid:].copy()),
                )
                for tid, xs, ys in halves:
                    recurring_train[tid] = (np.asarray(xs), np.asarray(ys))
                    mapping[tid] = orig

            missing = [tid for tid in recurring_order if tid not in recurring_train]
            if missing:
                raise ValueError(f"Recurring order references missing task ids: {missing}")

            def _remap_eval(source_map):
                if not source_map:
                    return []
                return [source_map[mapping[tid]] for tid in recurring_order]

            self.train_data = [recurring_train[tid] for tid in recurring_order]
            self.val_data = _remap_eval(val_map)
            self.test_data = _remap_eval(test_map)
            self.task_order = list(recurring_order)
            self.logical_task_ids = list(recurring_order)
            self.recurring_to_orig = mapping

            preview = [(tid, mapping[tid], recurring_train[tid][0].shape[0]) for tid in recurring_order[:5]]
            print(f"[RECURRING] Enabled: expanded {len(base_even)} base tasks into {len(self.train_data)} logical tasks.")
            print(f"[RECURRING] Order: {recurring_order}")
            print(f"[RECURRING] First splits (tid→orig→count): {preview}")
        else:
            self.train_data = [train_map[orig] for orig in order]
            self.val_data = [val_map[orig] for orig in order] if val_map else []
            self.test_data = [test_map[orig] for orig in order] if test_map else []

        self.num_total_tasks = len(self.train_data)


    def generate_task_batch(self, task_idx, train=True):
        """ Provides a sequential batch of data for a specific task from the full dataset. """
        task_data = self.train_data[task_idx] if train else self.val_data[task_idx]
        images, labels = task_data
        images = images.reshape([-1, self.dim_input])
        labels = labels.reshape([-1])

        if task_idx not in self.batch_offsets:
            self.batch_offsets[task_idx] = 0

        # Define batch size for support and query sets based on MAML setup
        batch_size = FLAGS.update_batch_size * self.num_classes
        
        start_idx = self.batch_offsets[task_idx]
        support_end_idx = start_idx + batch_size
        query_end_idx = support_end_idx + batch_size

        # If we reach the end of the data for this task, loop back to the beginning
        if query_end_idx > images.shape[0]:
            start_idx = 0
            support_end_idx = batch_size
            query_end_idx = 2 * batch_size
        
        self.batch_offsets[task_idx] = query_end_idx

        # Extract sequential batches for support (inputa) and query (inputb)
        inputa = images[start_idx:support_end_idx, :]
        labela = labels[start_idx:support_end_idx]
        inputb = images[support_end_idx:query_end_idx, :]
        labelb = labels[support_end_idx:query_end_idx]

        return inputa, labela, inputb, labelb


class MNIST_Permutations_DataGenerator(object):
    """
    Loads Permuted MNIST data (La-MAML mnist_permutations.pt) and creates sequential batches.
    This is only used when --datasource=mnist_permutations.
    """
    def __init__(self, datadir):
        self.num_classes = 10
        self.dim_input = 28 * 28
        self.dim_output = self.num_classes
        self.batch_offsets = {}

        file_path = os.path.join(datadir, "mnist_permutations.pt")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"PermMNIST data file not found at {file_path}")

        try:
            import torch
        except Exception as exc:
            raise ImportError("torch is required to load mnist_permutations.pt") from exc

        train_raw, test_raw = torch.load(file_path, map_location="cpu")
        val_raw = test_raw
        print(f"Loaded PermMNIST splits: train={len(train_raw)} test={len(test_raw)} tasks.")

        def _as_numpy(entry):
            _tag, x, y = entry
            if hasattr(x, "numpy"):
                x = x.numpy()
            if hasattr(y, "numpy"):
                y = y.numpy()
            return np.asarray(x), np.asarray(y)

        order = list(range(len(train_raw)))
        self.task_order = list(order)
        self.logical_task_ids = list(order)
        self.recurring_to_orig = {}

        train_map = {idx: _as_numpy(train_raw[idx]) for idx in order}
        test_map = {idx: _as_numpy(test_raw[idx]) for idx in order}
        val_map = {idx: _as_numpy(val_raw[idx]) for idx in order}

        if getattr(FLAGS, 'recurring_perm_mnist', False):
            base_even = [orig for orig in sorted(train_map.keys()) if orig % 2 == 0 and orig <= 18]
            if len(base_even) != 10:
                raise ValueError(f"Recurring PermMNIST expects 10 even base tasks, got {len(base_even)} (ids={base_even})")

            recurring_train = {}
            mapping = {}
            for orig in base_even:
                images, labels = train_map[orig]
                total = images.shape[0]
                if total % 2 != 0:
                    raise ValueError(f"Task {orig} has {total} samples; recurring split requires an even count.")
                mid = total // 2
                halves = (
                    (orig * 2 + 0, images[:mid].copy(), labels[:mid].copy()),
                    (orig * 2 + 1, images[mid:].copy(), labels[mid:].copy()),
                )
                for tid, xs, ys in halves:
                    recurring_train[tid] = (np.asarray(xs), np.asarray(ys))
                    mapping[tid] = orig

            recurring_order = [8, 28, 9, 29, 12, 32, 13, 33, 16, 36,
                               17, 37, 0, 20, 1, 21, 4, 24, 5, 25]
            missing = [tid for tid in recurring_order if tid not in recurring_train]
            if missing:
                raise ValueError(f"Recurring PermMNIST order references missing task ids: {missing}")

            def _remap_eval(source_map):
                if not source_map:
                    return []
                return [source_map[mapping[tid]] for tid in recurring_order]

            self.train_data = [recurring_train[tid] for tid in recurring_order]
            self.val_data = _remap_eval(val_map)
            self.test_data = _remap_eval(test_map)
            self.task_order = list(recurring_order)
            self.logical_task_ids = list(recurring_order)
            self.recurring_to_orig = mapping

            preview = [(tid, mapping[tid], recurring_train[tid][0].shape[0]) for tid in recurring_order[:5]]
            print(f"[RECURRING] Enabled: expanded {len(base_even)} base tasks into {len(self.train_data)} logical tasks.")
            print(f"[RECURRING] Order: {recurring_order}")
            print(f"[RECURRING] First splits (tid→orig→count): {preview}")
        else:
            self.train_data = [train_map[idx] for idx in order]
            self.val_data = [val_map[idx] for idx in order]
            self.test_data = [test_map[idx] for idx in order]

        self.num_total_tasks = len(self.train_data)

    def generate_task_batch(self, task_idx, train=True):
        """Provides a sequential batch of data for a specific task from the full dataset."""
        task_data = self.train_data[task_idx] if train else self.val_data[task_idx]
        images, labels = task_data
        images = images.reshape([-1, self.dim_input])
        labels = labels.reshape([-1])

        if task_idx not in self.batch_offsets:
            self.batch_offsets[task_idx] = 0

        batch_size = FLAGS.update_batch_size * self.num_classes
        start_idx = self.batch_offsets[task_idx]
        support_end_idx = start_idx + batch_size
        query_end_idx = support_end_idx + batch_size

        # If we exceed, wrap around (deterministic sequential stream)
        if query_end_idx > images.shape[0]:
            start_idx = 0
            support_end_idx = batch_size
            query_end_idx = support_end_idx + batch_size
        self.batch_offsets[task_idx] = query_end_idx

        inputa = images[start_idx:support_end_idx]
        labela = labels[start_idx:support_end_idx]
        inputb = images[support_end_idx:query_end_idx]
        labelb = labels[support_end_idx:query_end_idx]
        return inputa, labela, inputb, labelb

    def load_data(self, datadir, datasource, train=True):
        """
        Loads data from the specified directory and datasource.
        Supports loading from the full Rotated MNIST dataset, Omniglot, and miniImagenet.
        """
        if 'omniglot' in datasource:
            return self.load_omniglot(datadir, train)
        elif datasource == 'miniimagenet':
            return self.load_miniimagenet(datadir, train)
        elif datasource == 'multidataset':
            return self.load_multidataset(datadir, train)
        elif datasource == 'sinusoid' or datasource == 'mixture':
            return self.load_sinusoid_mixture(datadir, train)
        else:
            raise ValueError(f"Datasource {datasource} not recognized.")

    def load_omniglot(self, datadir, train=True):
        # Similar to the original implementation
        pass

    def load_miniimagenet(self, datadir, train=True):
        # Similar to the original implementation
        pass

    def load_multidataset(self, datadir, train=True):
        # Similar to the original implementation
        pass

    def load_sinusoid_mixture(self, datadir, train=True):
        # Similar to the original implementation
        pass

    def setup_full_data(self, datadir, datasource):
        """
        Sets up the full dataset for training and validation.
        This is used when the --full_data flag is set.
        """
        self.datasource = datasource
        self.datadir = datadir

        self.dataset_train = self.load_data(self.datadir, self.datasource, train=True)
        self.dataset_val = self.load_data(self.datadir, self.datasource, train=False)
        self.num_total_tasks = len(self.dataset_train)

    def get_continuous_stream(self):
        """
        Flattens all task data into a single continuous stream where tasks appear sequentially.
        """
        all_images = np.concatenate([self.dataset_train[i]['images'] for i in range(self.num_total_tasks)])
        all_labels = np.concatenate([self.dataset_train[i]['labels'] for i in range(self.num_total_tasks)])
        return all_images, all_labels

    def add_task(self, buffer):
        task_idx = np.random.randint(0, self.num_total_tasks)
        task_data = self.dataset_train[task_idx]
        buffer.add(task_data['images'], task_data['labels'])
