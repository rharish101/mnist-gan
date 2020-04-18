"""Data loading utilities for the MNIST GAN."""
import os

import numpy as np
import tensorflow as tf


def _load_idx(idx):
    """Load an IDX file object opened in 'rb' mode."""
    idx.read(2)  # Skip the zero bytes

    # Get the dtype of the tensor
    dtype_byte = int(idx.read(1).hex(), 16)
    if dtype_byte == 8:
        dtype = np.uint8
    elif dtype_byte == 9:
        dtype = np.int8
    elif dtype_byte == 11:
        dtype = np.int16
    elif dtype_byte == 12:
        dtype = np.int32
    elif dtype_byte == 13:
        dtype = np.float32
    elif dtype_byte == 14:
        dtype = np.float64

    # Get the tensor's dimensions
    num_dims = int(idx.read(1).hex(), 16)
    shape = []
    for i in range(num_dims):
        dim_len = int(idx.read(4).hex(), 16)
        shape.append(dim_len)

    # Row major form
    total_length = np.prod(shape)
    dtype_size = dtype().nbytes
    image = np.empty(total_length, dtype=dtype)
    for i in range(total_length):
        image[i] = int(idx.read(dtype_size).hex(), 16)

    # Original form
    return np.reshape(image, shape, order="C")


def load_dataset(mnist_path):
    """Load the MNIST IDX image files and return numpy arrays.

    The images are grayscale uint8 images, of shape (images, width, height).
    The labels are integers from 0 to 9, of shape (labels,).

    Args:
        mnist_path (str): Path to the MNIST dataset

    Returns:
        `np.ndarray`: A 4D array of all uint8 28x28 grayscale MNIST images
        `np.ndarray`: A 1D array of all uint8 MNIST labels

    """
    # Expand "~"
    mnist_path = os.path.expanduser(mnist_path)
    dataset = {"train": {}, "test": {}}

    # Prefixes and infixes for generating filenames
    prefixes = {"train": "train", "test": "t10k"}
    infixes = {"images": "3", "labels": "1"}
    for mode in prefixes:
        for data in infixes:
            filename = f"{prefixes[mode]}-{data}-idx{infixes[data]}-ubyte"
            data_path = os.path.join(mnist_path, filename)

            print(f"\rLoading {mode} {data}...", end="")
            if os.path.exists(data_path):  # decompressed dataset
                with open(data_path, "rb") as idx:
                    dataset[mode][data] = _load_idx(idx)
            else:
                raise FileNotFoundError(
                    f'MNIST dataset file "{data_path}" not found.'
                )
    print("\rLoaded MNIST dataset successfully")

    images = np.concatenate(
        [dataset["train"]["images"], dataset["test"]["images"]], axis=0
    )
    labels = np.concatenate(
        [dataset["train"]["labels"], dataset["test"]["labels"]], axis=0
    )
    return images, labels


@tf.function
def preprocess(img, lbl):
    """Preprocess a raw MNIST image and its label.

    This converts a 2D 28x28 tensor in the range [0, 255] into a float32 3D
    64x64x1 tensor in the range [-1, 1]. This also casts the label into int64.
    """
    img = tf.expand_dims(img, -1)
    # This resizes and converts to float32 in the range [0, 255]
    img = tf.image.resize(img, (64, 64))
    # Scale from [0, 255] to [-1, 1]
    img = (img / 255) * 2 - 1
    lbl = tf.cast(lbl, tf.int64)
    return img, lbl


def get_mnist_dataset(mnist_path, batch_size):
    """Get a dataset object for the MNIST dataset.

    Args:
        mnist_path (str): Path to the MNIST dataset
        batch_size (int): The batch size

    Returns:
        `tensorflow.data.Dataset`: The dataset object

    """
    mnist_images, mnist_labels = load_dataset(mnist_path)
    dataset = tf.data.Dataset.from_tensor_slices((mnist_images, mnist_labels))
    dataset = dataset.map(preprocess).shuffle(10000).batch(batch_size)
    return dataset
