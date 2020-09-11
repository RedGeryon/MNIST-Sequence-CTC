import tensorflow as tf
import numpy as np
from numpy.random import randint
import matplotlib.pyplot as plt


class DataGenerator(object):

    def __init__(self,
                 min_seq_len=6,
                 max_seq_len=6,
                 batch_size=64):

        self.min_seq_len = min_seq_len
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size

        # Load data
        (self.x_train, self.y_train), (self.x_test, self.y_test) = tf.keras.datasets.mnist.load_data()
        # Normalize
        self.x_train, self.x_test = self.x_train[..., np.newaxis]/255.0, self.x_test[..., np.newaxis]/255.0

        print(f"Loaded {len(self.x_train)} train images, and {len(self.x_test)} test images")
        print(f"Train array of shape {self.x_train.shape}")
        print(f"batch_size set at {self.batch_size}, with digit len from {self.min_seq_len} to {self.max_seq_len}")

    def create_batch(self,
                     data,
                     data_labels,
                     min_overlap=0,
                     max_overlap=18):
        print(f"\nInitializing generator of batch size {self.batch_size}\n")

        # Generator will create images for each batch
        while True:
            # Initialize image canvas for entire batch
            batch_data = np.zeros((self.batch_size, 28, 28 * self.max_seq_len, 1))
            # Length of each sequence (number of digits of each image) of the batch
            rand_seq_len = randint(self.min_seq_len, self.max_seq_len + 1, size= (self.batch_size))
            # Sample indices of images equal to sequence length to compose each training sequence
            rand_idx = [randint(0, len(data), size=seq_len) for seq_len in rand_seq_len]
            # Indices for 'adding' individual images into sequence; left-shift a random number of pixels
            slice_idx = [[i*28 - randint(min_overlap, max_overlap + 1) for i in range(seq_len)] for seq_len in rand_seq_len]
            # Turn into left-side, right-side tuple for slicing
            slice_idx = [[(i, i+ 28) if i > 0 else (0, 28) for i in seq] for seq in slice_idx]

            # Zip up image reference and slicing indices to concat to batch_data
            for img_num, (imgs, indices) in enumerate(zip(rand_idx, slice_idx)):
                for i, (s1, s2) in zip(imgs, indices):
                    batch_data[img_num, :, s1:s2, :] += data[i]

            # Clip max value to 1
            batch_data = np.clip(batch_data, 0, 1)
            # Transpose from (batch_size, height, width, channel) to (batch_size, width, height, channel)
            batch_data = batch_data.transpose((0,2,1,3))

            # Fill labels to consistent size (max_seq_len)
            labels = np.empty((self.batch_size, self.max_seq_len), dtype="int")
            # Fill in actual label
            for b, seq in enumerate(rand_idx):
                l = [data_labels[i] for i in seq]
                labels[b, 0:len(l)] = l

            yield batch_data, labels

    def tensor_batch(self,
                     data,
                     data_labels,
                     min_overlap=0,
                     max_overlap=18):

        # Lambda function to help match inputs and output with our network layer names
        data_to_dict = lambda image, label: {'image': image, 'label': label}

        dataset = tf.data.Dataset.from_generator(
            self.create_batch,
            args=[data, data_labels, min_overlap, max_overlap],
            output_types=(tf.float64, tf.int64),
            output_shapes=([self.batch_size, self.max_seq_len * 28, 28, 1], [self.batch_size, self.max_seq_len])
        ).map(data_to_dict)

        return dataset

    @staticmethod
    def plot_images(batch, labels):
        # Take 20 random samples and plot them
        fig, ax = plt.subplots(10, 2, figsize=(12, 12))
        fig.tight_layout(pad=0.3, rect=[0, 0, 0.9, 0.9])
        for i in range(20):
            rand_idx = randint(0, len(batch))
            img = batch[rand_idx].transpose((1, 0, 2))
            ax[i%10, i//10].imshow(img, cmap='gray')
            canvas_label = ''.join(map(str, labels[rand_idx]))
            ax[i%10, i//10].set_title(canvas_label)
            ax[i%10, i//10].tick_params(labelbottom=False)

        plt.show()