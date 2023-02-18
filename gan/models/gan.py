# SPDX-FileCopyrightText: 2019 Harish Rajagopal <harish.rajagopals@gmail.com>
#
# SPDX-License-Identifier: MIT

"""Generator and critic network models."""
from typing import Tuple, Union

import tensorflow as tf
from tensorflow import Tensor, TensorShape
from tensorflow.keras import Input, Model
from tensorflow.keras.constraints import Constraint
from tensorflow.keras.layers import (
    BatchNormalization,
    Concatenate,
    Conv2D,
    Conv2DTranspose,
    Embedding,
    Layer,
    LayerNormalization,
    LeakyReLU,
    ReLU,
    Reshape,
)
from tensorflow.keras.regularizers import l2

from ..data import IMG_SHAPE, NUM_CLS
from ..utils import Config

# The type of an input shape spec for a Keras layer
Shape = Union[Tuple[int, ...], TensorShape]


class SpectralNorm(Constraint):
    """Constraint the weights to have unit spectral norm.

    Attributes:
        v: The variable for the power method
        iterations: The number of iterations of the power method
    """

    def __init__(self, dim: int, iterations: int = 1):
        """Initialize the variable for the power method.

        Args:
            dim: The size of the last dimension of the weights (along which
                spectral normalization will be applied)
            iterations: The number of iterations of the power method
        """
        self.v = tf.Variable(tf.random.normal((dim, 1)), trainable=False)
        self.iterations = iterations

    def __call__(self, weights: Tensor) -> Tensor:
        """Normalize the weights by its spectral norm."""
        # For Conv kernels, the last layer will be the output channels.
        # Therefore, (H, W, C_in, C_out) -> (H * W * C_in, C_out)
        w = tf.reshape(weights, (-1, weights.shape[-1]))
        v = self.v

        for _ in range(self.iterations):
            u = tf.linalg.normalize(w @ v)[0]
            v = tf.linalg.normalize(tf.transpose(w) @ u)[0]

        self.v.assign(tf.cast(v, self.v.dtype))
        spectral_norm = tf.transpose(u) @ w @ v
        return weights / spectral_norm

    def get_config(self):
        """Get the config for this constraint."""
        return {"dimensions": self.v.shape[0], "iterations": self.iterations}


class Conditioning(Layer):
    """Class for the conditioning layer.

    This layer conditions with the class labels. An embedding layer is used for
    the class labels without passing through a dense layer as the no. of
    classes is smaller than typical embedding sizes.

    Attributes:
        embed: The core embedding layer
        config: The hyper-param config
    """

    def __init__(self, config: Config):
        """Store the hyper-param config."""
        super().__init__()
        self.config = config

    def build(self, input_shape: Tuple[Shape, Shape]) -> None:
        """Initialize the Conv2DTranspose and Embedding layers."""
        tensor_shape, _ = input_shape
        flat_dim = tensor_shape[1] * tensor_shape[2] * tensor_shape[3]
        self.embed = Embedding(
            NUM_CLS,
            flat_dim,
            input_length=1,
            embeddings_regularizer=l2(self.config.crit_weight_decay),
            embeddings_constraint=SpectralNorm(
                flat_dim, iterations=self.config.power_iter
            ),
        )

    def call(self, inputs: Tuple[Tensor, Tensor]) -> Tensor:
        """Condition the tensor from the label vectors.

        Args:
            inputs: A tuple of the following tensors:
                * The tensor to be conditioned
                * The labels

        Returns:
            The conditioned tensor
        """
        tensor, labels = inputs
        embeddings = self.embed(labels)
        labels_cond = tf.reshape(embeddings, [-1, *tensor.shape[1:]])
        return tensor + labels_cond


def get_generator(config: Config) -> Model:
    """Return the generator model.

    Args:
        config: The hyper-param config

    Returns:
        The generator model
    """
    noise = Input(shape=[config.noise_dims])
    labels = Input(shape=[])

    cond = Embedding(
        NUM_CLS,
        64,
        input_length=1,
        embeddings_regularizer=l2(config.gen_weight_decay),
    )(labels)
    x = Concatenate(axis=-1)([noise, cond])
    x = Reshape([1, 1, x.shape[-1]])(x)

    def conv_t_block(
        inputs: Tensor, filters: int, first: bool = False, last: bool = False
    ) -> Tensor:
        x = Conv2DTranspose(
            filters=filters,
            kernel_size=4,
            strides=1 if first else 2,
            padding="valid" if first else "same",
            activation="tanh" if last else None,
            use_bias=True if last else False,
            kernel_regularizer=l2(config.gen_weight_decay),
            dtype="float32" if last else None,
        )(inputs)
        if not last:
            x = BatchNormalization()(x)
            x = ReLU()(x)
        return x

    x = conv_t_block(x, 512, first=True)
    x = conv_t_block(x, 256)
    x = conv_t_block(x, 128)
    x = conv_t_block(x, 64)
    outputs = conv_t_block(x, IMG_SHAPE[-1], last=True)

    return Model(inputs=[noise, labels], outputs=outputs)


def get_critic(config: Config) -> Model:
    """Return the critic model.

    Args:
        config: The hyper-param config

    Returns:
        The critic model
    """
    inputs = Input(shape=IMG_SHAPE)
    labels = Input(shape=[])

    def conv_block(inputs: Tensor, filters: int, norm: bool = True) -> Tensor:
        x = Conv2D(
            filters,
            kernel_size=4,
            strides=2,
            padding="same",
            use_bias=False,
            kernel_regularizer=l2(config.crit_weight_decay),
            kernel_constraint=SpectralNorm(filters),
        )(inputs)
        x = Conditioning(config)((x, labels))
        if norm:
            x = LayerNormalization()(x)
        x = LeakyReLU(alpha=0.2)(x)
        return x

    x = conv_block(inputs, 64, norm=False)
    x = conv_block(x, 128)
    x = conv_block(x, 256)
    x = conv_block(x, 512)

    outputs = Conv2D(
        filters=1,
        kernel_size=4,
        strides=1,
        padding="valid",
        use_bias=True,
        kernel_regularizer=l2(config.crit_weight_decay),
        kernel_constraint=SpectralNorm(dim=1),
        dtype="float32",
    )(x)

    return Model(inputs=[inputs, labels], outputs=outputs)
