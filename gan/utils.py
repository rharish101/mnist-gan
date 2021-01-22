"""Utilities for the GAN."""
import itertools
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Union

import tensorflow as tf
import yaml
from tensorflow import Tensor
from tensorflow.distribute import Strategy
from tensorflow.python.distribute.values import PerReplica

DistTensor = Union[PerReplica, Tensor]


@tf.function
def sqrtm(tensor: Tensor) -> Tensor:
    """Return the matrix sqrt of a tensor along the last two dimensions.

    This computes the square root of a positive definite matrix, with support
    for batching. This is modified from the following implementation:
    https://github.com/pytorch/pytorch/issues/25481#issuecomment-576493693
    """
    s, _, v = tf.linalg.svd(tensor)

    # Remove singular values which are smaller than a threshold, for numerical
    # stability (?). This is done by truncating components common across the
    # batch that are below a threshold, and zeroing out the other components
    # below the threshold.
    eps = tf.keras.backend.epsilon()
    threshold = tf.math.reduce_max(s, -1, keepdims=True) * s.shape[-1] * eps
    good = s > threshold
    components = tf.math.reduce_sum(tf.cast(good, tf.int64), -1)
    common = tf.math.reduce_max(components)
    unbalanced = common != tf.math.reduce_min(components)

    if common < s.shape[-1]:
        s = s[..., :common]
        v = v[..., :common]
        if unbalanced:
            good = good[..., :common]

    if unbalanced:
        s = tf.where(good, s, 0)

    # Compose the square root matrix
    v_dims = len(v.shape)
    v_t = tf.transpose(v, list(range(v_dims - 2)) + [v_dims - 1, v_dims - 2])
    return (v * tf.expand_dims(tf.math.sqrt(s), -2)) @ v_t


@tf.function
def get_grid(img: Tensor) -> Tensor:
    """Convert a batch of float images from [-1, 1] to a uint8 image grid.

    The returned image will contain a batch size of 1.
    """
    img = img / 2 + 0.5  # [-1, 1] to [0, 1]
    img = tf.image.convert_image_dtype(img, tf.uint8)

    batch_size = img.shape[0]
    if batch_size >= 16:
        grid = img[:16]
        grid_shape = (4, 4)
    elif batch_size >= 9:
        grid = img[:9]
        grid_shape = (3, 3)
    elif batch_size >= 4:
        grid = img[:4]
        grid_shape = (2, 2)
    else:
        return img[:1]  # retain the batch axis

    # Reshape to (grid_height, grid_width, img_height, img_width, channels)
    grid = tf.reshape(grid, grid_shape + img.shape[1:])
    # Permute axes to (grid_height, grid_width, img_width, img_height,
    # channels) for convenient combination of widths.
    grid = tf.transpose(grid, perm=[0, 1, 3, 2, 4])
    # Combine grid width and image width
    grid = tf.reshape(grid, (grid_shape[0], -1, img.shape[1], img.shape[3]))
    # Permute axes to (grid_height, img_height, combined_width, channels) for
    # convenient combination of heights.
    grid = tf.transpose(grid, perm=[0, 2, 1, 3])
    # Combine grid height and image height, with a batch axis
    grid = tf.reshape(grid, (1, -1, grid.shape[2], img.shape[3]))

    return grid


def setup_dirs(
    dirs: List[Path],
    config: Dict[str, Any],
    file_name: str,
    dirs_to_tstamp: List[Path] = [],
) -> List[Path]:
    """Create the required directories and dump the config there.

    This supports creating timestamped directories in requested directories. It
    creates a timestamped directory in each of those, and returns them.

    Args:
        dirs: The directories to be setup
        config: The config that is to be dumped
        file_name: The file name for the config
        dirs_to_tstamp: The directories for timestamping

    Returns:
        The list of created timestamped directories, if any
    """
    tstamped_dirs = []
    for directory in dirs_to_tstamp:
        time_stamp = datetime.now().isoformat()
        new_dir = directory / time_stamp
        tstamped_dirs.append(new_dir)

    # Save hyperparams in both log and save directories
    for directory in itertools.chain(dirs, tstamped_dirs):
        if not directory.exists():
            directory.mkdir(parents=True)
        with open(directory / file_name, "w") as conf:
            yaml.dump(config, conf)

    return tstamped_dirs


def iterator_product(*args: Iterable) -> Iterable:
    """Return the cartesian product of given iterators.

    This works similar to `itertools.product`, but without duplicating
    stuff in memory. This notably happens when using TensorFlow datasets.
    """
    if len(args) == 0:
        raise ValueError("At least one iterator must be given")

    for i in args[0]:
        if len(args) == 1:
            yield i
            continue

        for j in iterator_product(*args[1:]):
            if len(args) > 2:
                yield (i, *j)
            else:
                yield i, j


def reduce_concat(strategy: Strategy, dist_tensor: DistTensor) -> Tensor:
    """Reduce a distributed tensor by batch-axis concatenation.

    Args:
        strategy: The multi-device training strategy
        dist_tensor: The (possibly) replica-distributed tensor

    Returns:
        The reduced tensor
    """
    # If there is only one or zero GPU available, then the values are
    # proper tensors, so no need to concatenate any values.
    if strategy.num_replicas_in_sync > 1:
        return tf.concat(dist_tensor.values, axis=0)
    else:
        return dist_tensor
