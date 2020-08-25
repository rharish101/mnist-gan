"""Generator and critic network models."""
from typing import Tuple, Type, Union

import tensorflow as tf
from tensorflow import Tensor, TensorShape, Variable
from tensorflow.keras import Input, Model
from tensorflow.keras.initializers import RandomNormal
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

# The type of an input shape spec for a Keras layer
Shape = Union[Tuple[int, ...], TensorShape]


def spectralize(layer: Type[Layer]) -> Type[Layer]:  # noqa: D202
    """Return a class with spectral normalization over the kernel.

    This will require that the given layer should have a "kernel" attribute.
    """

    class SpectralLayer(layer):  # type: ignore
        def build(self, input_shape: Shape) -> None:
            """Initialize the vector for the power iteration method."""
            super().build(input_shape)
            # For Conv kernels, the last layer will be the output channels.
            # Therefore, (H, W, C_in, C_out) -> (H * W * C_in, C_out)
            dim = self.kernel.shape[:-1].num_elements()
            self.u = self.add_weight(
                name="u",
                shape=(dim, 1),
                dtype=self.kernel.dtype,
                initializer=RandomNormal,
                trainable=False,
            )

        @tf.function
        def _spectral_norm(
            self, weights: Variable, training: bool = False
        ) -> Tensor:
            # For Conv kernels, the last layer will be the output channels.
            # Therefore, (H, W, C_in, C_out) -> (H * W * C_in, C_out)
            w = tf.reshape(weights, (-1, weights.shape[-1]))
            v = tf.linalg.normalize(tf.matmul(tf.transpose(w), self.u))[0]
            u = tf.linalg.normalize(tf.matmul(w, v))[0]
            if training:
                self.u.assign(u)

            u = tf.stop_gradient(u)
            v = tf.stop_gradient(v)
            spec_norm = tf.matmul(tf.matmul(tf.transpose(u), w), v)
            return spec_norm

        def call(self, inputs: Tensor, training: bool = False) -> Tensor:
            """Perform spectral normalization before calling the layer."""
            spec_norm = self._spectral_norm(self.kernel, training=training)
            self.kernel.assign(self.kernel / spec_norm)
            return super().call(inputs)

    return SpectralLayer


class Conditioning(Layer):
    """Class for the conditioning layer.

    This layer conditions with the class labels. An embedding layer is used for
    the class labels without passing through a dense layer as the no. of
    classes is smaller than typical embedding sizes.

    Attributes:
        embed: The core embedding layer
        weight_decay: The decay for L2 regularization
    """

    def __init__(self, num_classes: int, weight_decay: float = 0):
        """Store weight decay."""
        super().__init__()
        self.weight_decay = weight_decay
        self.num_classes = num_classes

    def build(self, input_shape: Tuple[Shape, Shape]) -> None:
        """Initialize the Conv2DTranspose and Embedding layers."""
        tensor_shape, _ = input_shape
        flat_dim = tensor_shape[1] * tensor_shape[2] * tensor_shape[3]
        self.embed = Embedding(
            self.num_classes,
            flat_dim,
            input_length=1,
            embeddings_regularizer=l2(self.weight_decay),
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


def get_generator(
    noise_dims: int, num_classes: int, img_ch: int, weight_decay: float = 0
) -> Model:
    """Return the generator model.

    Args:
        noise_dims: The dimensions of the random noise
        num_classes: The number of classes in the dataset
        img_ch: The number of channels needed in the output image
        weight_decay: The decay for L2 regularization

    Returns:
        The generator model
    """
    noise = Input(shape=[noise_dims])
    labels = Input(shape=[])

    cond = Embedding(
        num_classes,
        64,
        input_length=1,
        embeddings_regularizer=l2(weight_decay),
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
            kernel_regularizer=l2(weight_decay),
        )(inputs)
        if not last:
            x = BatchNormalization()(x)
            x = ReLU()(x)
        return x

    x = conv_t_block(x, 512, first=True)
    x = conv_t_block(x, 256)
    x = conv_t_block(x, 128)
    x = conv_t_block(x, 64)
    outputs = conv_t_block(x, 1, last=True)

    return Model(inputs=[noise, labels], outputs=outputs)


def get_critic(
    input_shape: Tuple[int, int, int],
    num_classes: int,
    weight_decay: float = 0,
) -> Model:
    """Return the critic model.

    Args:
        input_shape: The shape of the input images excluding the batch size
        num_classes: The number of classes in the dataset
        weight_decay: The decay for L2 regularization

    Returns:
        The critic model
    """
    SpectralConv2D = spectralize(Conv2D)

    inputs = Input(shape=input_shape)
    labels = Input(shape=[])

    def conv_block(inputs: Tensor, filters: int, norm: bool = True) -> Tensor:
        x = SpectralConv2D(
            filters,
            4,
            strides=2,
            padding="same",
            use_bias=False,
            kernel_regularizer=l2(weight_decay),
            input_shape=input_shape,
        )(inputs)
        x = Conditioning(num_classes, weight_decay)((x, labels))
        if norm:
            x = LayerNormalization()(x)
        x = LeakyReLU(alpha=0.2)(x)
        return x

    x = conv_block(inputs, 64, norm=False)
    x = conv_block(x, 128)
    x = conv_block(x, 256)
    x = conv_block(x, 512)

    outputs = SpectralConv2D(
        filters=1,
        kernel_size=4,
        strides=1,
        padding="valid",
        use_bias=True,
        kernel_regularizer=l2(weight_decay),
    )(x)

    return Model(inputs=[inputs, labels], outputs=outputs)
