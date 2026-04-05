""" Code for loading data. """
import numpy as np
import os
import random
import tensorflow as tf
import ipdb
import pickle
import torch # ADDED for dynamic loading
import matplotlib.pyplot as plt

from tensorflow.python.platform import flags
from utils import get_images


FLAGS = flags.FLAGS


class DataGenerator(object):
    def __init__(self, num_samples_per_class, batch_size, config={}):
        """
        Args:
            num_samples_per_class: num samples to generate per class in one batch
            batch_size: size of meta batch size (e.g. number of functions)
        """
        self.batch_size = batch_size
        self.num_samples_per_class = num_samples_per_class
        self.num_classes = 1  # by default 1 (only relevant for classification problems)

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

        elif 'mnist' in FLAGS.datasource:
            # MODIFIED BLOCK FOR MNIST
            self.num_classes = 10 # 10 digits
            self.dim_input = 784 # 28*28
            self.dim_output = self.num_classes
            self.img_size = config.get('img_size', (28, 28))
            # Load the raw PyTorch data directly for dynamic sampling
            self._load_mnist_pt_data()

        elif FLAGS.datasource == 'mixture':
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

        elif FLAGS.datasource == 'multidataset_leave_one_out':
            self.num_classes = config.get('num_classes', FLAGS.num_classes)
            self.img_size = config.get('img_size', (84, 84))
            self.dim_input = np.prod(self.img_size) * 3
            self.dim_output = self.num_classes
            self.multidataset = ['CUB_Bird', 'DTD_Texture', 'FGVC_Aircraft', 'FGVCx_Fungi']
            metatrain_folders, metaval_folders = [], []
            for idx_data, eachdataset in enumerate(self.multidataset):
                if idx_data == FLAGS.leave_one_out_id:
                    continue
                metatrain_folders.append(
                    [os.path.join('{0}/meta-dataset-leave-one-out/{1}/train'.format(FLAGS.datadir, eachdataset), label) \
                     for label in
                     os.listdir('{0}/meta-dataset-leave-one-out/{1}/train'.format(FLAGS.datadir, eachdataset)) \
                     if
                     os.path.isdir(
                         os.path.join('{0}/meta-dataset-leave-one-out/{1}/train'.format(FLAGS.datadir, eachdataset),
                                      label)) \
                     ])
            if FLAGS.test_set:
                metaval_folders = [os.path.join('{0}/meta-dataset-leave-one-out/{1}/train'.format(FLAGS.datadir,
                                                                                                  self.multidataset[
                                                                                                      FLAGS.leave_one_out_id]),
                                                label) \
                                   for label in os.listdir(
                        '{0}/meta-dataset-leave-one-out/{1}/train'.format(FLAGS.datadir,
                                                                          self.multidataset[FLAGS.leave_one_out_id])) \
                                   if os.path.isdir(
                        os.path.join('{0}/meta-dataset-leave-one-out/{1}/train'.format(FLAGS.datadir, self.multidataset[
                            FLAGS.leave_one_out_id]), label)) \
                                   ]
            else:
                metaval_folders = [os.path.join('{0}/meta-dataset-leave-one-out/{1}/val'.format(FLAGS.datadir,
                                                                                                self.multidataset[
                                                                                                    FLAGS.leave_one_out_id]),
                                                label) \
                                   for label in os.listdir(
                        '{0}/meta-dataset-leave-one-out/{1}/val'.format(FLAGS.datadir,
                                                                        self.multidataset[FLAGS.leave_one_out_id])) \
                                   if os.path.isdir(
                        os.path.join('{0}/meta-dataset-leave-one-out/{1}/val'.format(FLAGS.datadir,
                                                                                     self.multidataset[
                                                                                         FLAGS.leave_one_out_id]),
                                     label)) \
                                   ]
            self.metatrain_character_folders = metatrain_folders
            self.metaval_character_folders = metaval_folders
            self.rotations = config.get('rotations', [0])

        else:
            raise ValueError('Unrecognized data source')

    def _load_mnist_pt_data(self):
        """Loads the original MNIST .pt file into memory and converts to NumPy."""
        # REPLACED FUNCTION: This now loads the .pt file directly
        data_folder = os.path.join(FLAGS.datadir, 'data', 'rotated_mnist')
        pt_path = os.path.join(data_folder, 'data', 'mnist_rotations.pt')
        print(f"Loading original MNIST data from {pt_path} for dynamic sampling...")
        train_dataset_pt, test_dataset_pt = torch.load(pt_path)
        
        # Convert PyTorch Tensors to NumPy arrays immediately after loading.
        # This correctly preserves the specific angles from the file.
        self.train_tasks = [(angle, x.numpy(), y.numpy()) for angle, x, y in train_dataset_pt]
        self.val_tasks = [(angle, x.numpy(), y.numpy()) for angle, x, y in test_dataset_pt]
        print(f"Loaded {len(self.train_tasks)} training tasks and {len(self.val_tasks)} validation tasks.")
        train_angles = [round(task[0], 2) for task in self.train_tasks]
        print(f"Using the following {len(train_angles)} training angles from file: {train_angles}")


    def make_data_tensor_mnist(self, train=True):
        """
        Creates a data tensor for MNIST by DYNAMICALLY sampling from the full data pools.
        """
        # REPLACED FUNCTION: This now performs dynamic sampling
        if train:
            tasks = self.train_tasks
            k_shot = FLAGS.update_batch_size
        else:
            tasks = self.val_tasks
            k_shot = FLAGS.update_batch_size
        
        k_query = 15 # A standard query size for evaluation

        s_x_list, s_y_list, q_x_list, q_y_list = [], [], [], []

        for _ in range(self.batch_size):
            # 1. Sample a random task (which includes the correct angle)
            angle, all_x, all_y = random.choice(tasks)
            
            support_x, query_x = [], []
            support_y, query_y = [], []

            # 2. For each class, sample a FRESH support and query set
            for cls in range(self.num_classes):
                class_indices = np.where(all_y == cls)[0]
                sampled_indices = np.random.choice(class_indices, size=k_shot + k_query, replace=False)
                
                support_x.extend(all_x[sampled_indices[:k_shot]])
                support_y.extend(all_y[sampled_indices[:k_shot]])
                query_x.extend(all_x[sampled_indices[k_shot:]])
                query_y.extend(all_y[sampled_indices[k_shot:]])
            
            s_x_list.append(np.array(support_x))
            s_y_list.append(np.array(support_y))
            q_x_list.append(np.array(query_x))
            q_y_list.append(np.array(query_y))

        # Convert to tensors
        inputa = tf.convert_to_tensor(np.array(s_x_list), dtype=tf.float32)
        labela = tf.convert_to_tensor(np.array(s_y_list), dtype=tf.float32)
        inputb = tf.convert_to_tensor(np.array(q_x_list), dtype=tf.float32)
        labelb = tf.convert_to_tensor(np.array(q_y_list), dtype=tf.float32)

        # Reshape labels to be one-hot
        labela = tf.one_hot(tf.cast(labela, dtype=tf.int32), self.num_classes)
        labelb = tf.one_hot(tf.cast(labelb, dtype=tf.int32), self.num_classes)

        return inputa, labela, inputb, labelb

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
            image = 1.0 - image
        num_preprocess_threads = 1
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

    def make_data_tensor_multidataset(self, train=True):
        if train:
            folders = self.metatrain_character_folders
            # number of tasks, not number of meta-iterations. (divide by metabatch size to measure)
            if FLAGS.update_batch_size == 10:
                num_total_batches = 140000
            else:
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

    def make_data_tensor_multidataset_leave_one_out(self, train=True):
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
        for image_itr in range(num_total_batches):
            if train:
                sel = np.random.randint(3)
                sampled_character_folders = random.sample(folders[sel], self.num_classes)
            else:
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
        if FLAGS.datasource in ['miniimagenet', 'multidataset', 'multidataset_leave_one_out']:
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

        num_preprocess_threads = 1
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

    def generate_sinusoid_batch(self, train=True, input_idx=None):
        # Note train arg is not used (but it is used for omniglot method.
        # input_idx is used during qualitative testing --the number of examples used for the grad update
        amp = np.random.uniform(self.amp_range[0], self.amp_range[1], [self.batch_size])
        freq = np.random.uniform(self.freq_range[0], self.freq_range[1], [self.batch_size])
        phase = np.random.uniform(self.phase_range[0], self.phase_range[1], [self.batch_size])
        outputs = np.zeros([self.batch_size, self.num_samples_per_class, self.dim_output])
        init_inputs = np.zeros([self.batch_size, self.num_samples_per_class, self.dim_input])
        for func in range(self.batch_size):
            init_inputs[func] = np.random.uniform(self.input_range[0], self.input_range[1],
                                                  [self.num_samples_per_class, 1])
            if input_idx is not None:
                init_inputs[:, input_idx:, 0] = np.linspace(self.input_range[0], self.input_range[1],
                                                            num=self.num_samples_per_class - input_idx, retstep=False)
            outputs[func] = amp[func] * np.sin(freq[func] * init_inputs[func] - phase[func])
        return init_inputs, outputs, amp, phase

    def generate_mixture_batch(self, train=True, input_idx=None, DRAW_PLOTS=True):
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
        b_q = np.random.uniform(-2.0, 2.0, size=batch_size)
        c_q = np.random.uniform(-3.0, 3.0, size=batch_size)

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
            sel = np.random.randint(4)
            if FLAGS.train == False and FLAGS.test_dataset != -1:
                sel = FLAGS.test_dataset
            if sel == 0:
                outputs[func] = amp[func] * np.sin(freq[func] * init_inputs[func]) + phase[func]
            elif sel == 1:
                outputs[func] = A[func] * init_inputs[func] + b[func]
            elif sel == 2:
                outputs[func] = A_q[func] * np.square(init_inputs[func]) + b_q[func] * init_inputs[func] + c_q[func]
            elif sel == 3:
                outputs[func] = A_c[func] * np.power(init_inputs[func], np.tile([3], init_inputs[func].shape)) + b_c[
                    func] * np.square(init_inputs[func]) + c_c[func] * init_inputs[func] + d_c[func]
            sel_set[func] = sel
        funcs_params = {'amp': amp, 'phase': phase, 'freq': freq, 'A': A, 'b': b, 'A_q': A_q, 'c_q': c_q, 'b_q': b_q,
                        'A_c': A_c, 'b_c': b_c, 'c_c': c_c, 'd_c': d_c}
        return init_inputs, outputs, funcs_params, sel_set
