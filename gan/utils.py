# SPDX-FileCopyrightText: 2019 Harish Rajagopal <harish.rajagopals@gmail.com>
#
# SPDX-License-Identifier: MIT

"""Utilities for the GAN."""
import itertools
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Union

import tensorflow as tf
import toml
from tensorflow import Tensor
from tensorflow.python.distribute.values import PerReplica

DistTensor = Union[PerReplica, Tensor]


@dataclass(frozen=True)
class Config:
    """Class to hold hyper-parameter configs.

    Attributes:
        gan_batch_size: The global batch size for the GAN
        cls_batch_size: The global batch size for the classifier
        noise_dims: The dimensions for the inputs to the generator
        gen_lr: The learning rate for the generator's optimizer
        crit_lr: The learning rate for the critic's optimizer
        cls_lr: The learning rate for the classifier's optimizer
        decay_rate: The rate of exponential learning rate decay
        decay_steps: The base steps for exponential learning rate decay
        gp_weight: Weights for the critic's gradient penalty
        gen_weight_decay: The decay for L2 regularization in the generator
        crit_weight_decay: The decay for L2 regularization in the critic
        cls_weight_decay: The decay for L2 regularization in the classifier
        crit_steps: The number of critic steps per generator step
        gan_epochs: Number of epochs to train the GAN
        cls_epochs: Number of epochs to train the classifier
        mixed_precision: Whether to use mixed-precision for training
        power_iter: The number of iterations of the power method
    """

    gan_batch_size: int = 128
    cls_batch_size: int = 128
    noise_dims: int = 100
    gen_lr: float = 1e-4
    crit_lr: float = 1e-4
    cls_lr: float = 1e-4
    decay_rate: float = 0.8
    decay_steps: int = 3000
    gp_weight: float = 10.0
    gen_weight_decay: float = 2.5e-5
    crit_weight_decay: float = 2.5e-5
    cls_weight_decay: float = 2.5e-5
    crit_steps: int = 1
    gan_epochs: int = 40
    cls_epochs: int = 25
    mixed_precision: bool = False
    power_iter: int = 1


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
    config: Config,
    file_name: str,
    dirs_to_tstamp: List[Path] = [],
) -> List[Path]:
    """Create the required directories and dump the config there.

    This supports creating timestamped directories in requested directories. It
    creates a timestamped directory in each of those, and returns them.

    Args:
        dirs: The directories to be setup
        config: The hyper-param config that is to be dumped
        file_name: The file name for the config
        dirs_to_tstamp: The directories for timestamping

    Returns:
        The list of created timestamped directories, if any
    """
    curr_date = datetime.now().astimezone()

    tstamped_dirs = []
    for directory in dirs_to_tstamp:
        time_stamp = curr_date.isoformat()
        new_dir = directory / time_stamp
        tstamped_dirs.append(new_dir)

    to_dump = {**vars(config), "date": curr_date}

    # Save hyperparams in both log and save directories
    for directory in itertools.chain(dirs, tstamped_dirs):
        if not directory.exists():
            directory.mkdir(parents=True)
        with open(directory / file_name, "w") as conf:
            toml.dump(to_dump, conf)

    return tstamped_dirs


def reduce_concat(
    strategy: tf.distribute.Strategy, dist_tensor: DistTensor
) -> Tensor:
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


def load_config(config_path: Optional[Path]) -> Config:
    """Load the hyper-param config at the given path.

    If the path doesn't exist, then an empty dict is returned.
    """
    if config_path is not None and config_path.exists():
        with open(config_path, "r") as f:
            args = toml.load(f)
    else:
        args = {}
    return Config(**args)
