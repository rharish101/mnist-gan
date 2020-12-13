"""Data loading utilities for the GAN."""
import itertools
import os
from typing import BinaryIO, Dict, Final, List, Tuple

import numpy as np
import tensorflow as tf
from tensorflow import Tensor
from tensorflow.data import Dataset

NUM_CLS: Final = 10  # number of classes in the dataset
IMG_SHAPE: Final = (64, 64, 1)  # (HWC) images will be resized to this

# Map the size as found in IDX files to their respective numpy dtypes
SIZE_TO_DTYPE: Final = {
    8: np.uint8,
    9: np.int8,
    11: np.int16,
    12: np.int32,
    13: np.float32,
    14: np.float64,
}


def _load_idx(idx: BinaryIO) -> np.ndarray:
    """Load an IDX file object opened in 'rb' mode.

    The IDX specification is available at: http://yann.lecun.com/exdb/mnist/
    """
    idx.seek(2, 0)  # skip the first zero bytes

    # Get the dtype of the tensor
    dtype_size = int.from_bytes(idx.read(1), "big")
    dtype = SIZE_TO_DTYPE[dtype_size]

    # Get the tensor's dimensions
    num_dims = int.from_bytes(idx.read(1), "big")
    shape: List[int] = []
    for i in range(num_dims):
        dim_len = int.from_bytes(idx.read(4), "big")
        shape.append(dim_len)

    # Row major form
    total_length = np.prod(shape)
    dtype_size = dtype().nbytes
    image = np.empty(total_length, dtype=dtype)
    for i in range(total_length):
        image[i] = int.from_bytes(idx.read(dtype_size), "big")

    # Original form
    return np.reshape(image, shape, order="C")


def load_dataset(mnist_path: str) -> Dict[str, Dict[str, np.ndarray]]:
    """Load the MNIST IDX image files and return numpy arrays.

    The images are grayscale uint8 images, of shape (images, width, height).
    The labels are integers from 0 to 9, of shape (labels,).

    The returned dataset is a dict with the structure:
        "train":
            "images": A 4D array of uint8 28x28 grayscale training images
            "labels": A 1D array of uint8 training labels
        "test":
            "images": A 4D array of uint8 28x28 grayscale test images
            "labels": A 1D array of uint8 test labels

    Args:
        mnist_path: Path to the MNIST dataset

    Returns:
        The dict of the dataset
    """
    # Expand "~"
    mnist_path = os.path.expanduser(mnist_path)
    dataset: Dict[str, Dict[str, np.ndarray]] = {"train": {}, "test": {}}

    # Prefixes and infixes for generating filenames
    prefixes: Dict[str, str] = {"train": "train", "test": "t10k"}
    infixes: Dict[str, str] = {"images": "3", "labels": "1"}

    for mode, data in itertools.product(prefixes, infixes):
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

    print("\rLoaded dataset successfully")

    return dataset


@tf.function
def preprocess(img: Tensor, lbl: Tensor) -> Tuple[Tensor, Tensor]:
    """Preprocess a raw image and its label.

    This converts a 2D (width, height) tensor in the range [0, 255] into a
    float32 3D (width, height, channels) tensor in the range [-1, 1], after
    resizing. This also casts the label into int64.
    """
    img = tf.expand_dims(img, -1)
    # This resizes and converts to float32 in the range [0, 255]
    img = tf.image.resize(img, IMG_SHAPE[:2])
    # Scale from [0, 255] to [-1, 1]
    img = (img / 255) * 2 - 1
    lbl = tf.cast(lbl, tf.int64)
    return img, lbl


def get_dataset(data_path: str, batch_size: int) -> Tuple[Dataset, Dataset]:
    """Get training and test dataset objects for the dataset.

    Args:
        data_path: Path to the dataset
        batch_size: The batch size

    Returns:
        The training dataset object
        The test dataset object
    """
    dataset = load_dataset(data_path)

    train_dataset = Dataset.from_tensor_slices(
        (dataset["train"]["images"], dataset["train"]["labels"])
    )
    train_dataset = (
        train_dataset.map(preprocess).shuffle(10000).batch(batch_size)
    )

    test_dataset = Dataset.from_tensor_slices(
        (dataset["test"]["images"], dataset["test"]["labels"])
    )
    test_dataset = test_dataset.map(preprocess).batch(batch_size)
    return train_dataset, test_dataset
