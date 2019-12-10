"""Generator, discriminator and encoder network models for MNIST."""
import tensorflow as tf
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


def spectralize(layer):  # noqa: D202
    """Return a class with spectral normalization over the kernel.

    This will require that the given layer should have a "kernel" attribute.
    """

    class SpectralLayer(layer):
        def build(self, input_shape):
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
        def _spectral_norm(self, weights, training=False):
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

        def call(self, inputs, training=False):
            """Perform spectral normalization before calling the layer."""
            spec_norm = self._spectral_norm(self.kernel, training=training)
            self.kernel.assign(self.kernel / spec_norm)
            return super().call(inputs)

    return SpectralLayer


class Conditioning(Layer):
    """Class for the conditioning layer.

    This layer conditions with both the noise and the class labels. An
    embedding layer is used for the class labels without passing through a
    dense layer as the no. of classes is smaller than typical embedding sizes.
    """

    def __init__(self, weight_decay):
        """Store weight decay."""
        super().__init__()
        self.weight_decay = weight_decay

    def build(self, input_shape):
        """Initialize the Conv2DTranspose and Embedding layers."""
        tensor_shape, noise_shape, _ = input_shape

        noise_new_shape = [noise_shape[0], 1, 1, noise_shape[1]]
        self.conv_t = Conv2DTranspose(
            tensor_shape[-1],
            tensor_shape[1:-1],
            strides=1,
            use_bias=False,
            kernel_regularizer=l2(self.weight_decay),
            input_shape=noise_new_shape,
        )

        flat_dim = tensor_shape[1:].num_elements()
        self.embed = Embedding(
            10,
            flat_dim,
            input_length=1,
            embeddings_regularizer=l2(self.weight_decay),
        )

        # Keep reference to the conv kernel for applying spectral norm
        self.conv_t.build(noise_new_shape)
        self.kernel = self.conv_t.kernel

    def call(self, inputs):
        """Condition the tensor from the noise and label vectors.

        Args:
            inputs (tuple): A tuple of two tensors, where the first is the
                tensor to be conditioned, the second is the noise, and the
                third is the labels

        Returns:
            `tf.Tensor`: The conditioned tensor

        """
        tensor, noise, labels = inputs
        reshaped = tf.reshape(noise, [-1, 1, 1, noise.shape[1]])
        noise_cond = self.conv_t(reshaped)
        embeddings = self.embed(labels)
        labels_cond = tf.reshape(
            embeddings, [-1, tensor.shape[1], tensor.shape[2], tensor.shape[3]]
        )
        return tensor + noise_cond + labels_cond


def get_generator(noise_dims, weight_decay=2.5e-5):
    """Return the generator model.

    Args:
        noise_dims (float): The dimensions of the input to the generator
        weight_decay (float): The decay for L2 regularization

    Returns:
        `tf.keras.Model`: The generator model

    """
    noise = Input(shape=[noise_dims])
    labels = Input(shape=[])

    cond = Embedding(
        10, 64, input_length=1, embeddings_regularizer=l2(weight_decay)
    )(labels)
    x = Concatenate(axis=-1)([noise, cond])
    x = Reshape([1, 1, noise_dims + cond.shape[-1]])(x)

    def conv_t_block(inputs, filters, first=False):
        x = Conv2DTranspose(
            filters,
            4,
            strides=1 if first else 2,
            padding="valid" if first else "same",
            use_bias=False,
            kernel_regularizer=l2(weight_decay),
        )(inputs)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        return x

    x = conv_t_block(x, 512, first=True)
    x = conv_t_block(x, 256)
    x = conv_t_block(x, 128)
    x = conv_t_block(x, 64)

    outputs = Conv2DTranspose(
        1,
        4,
        strides=2,
        padding="same",
        activation="tanh",
        use_bias=False,
        kernel_regularizer=l2(weight_decay),
    )(x)

    return Model(inputs=[noise, labels], outputs=outputs)


def get_discriminator(input_shape, noise_dims, weight_decay=2.5e-5):
    """Return the discriminator model.

    Args:
        input_shape (tuple): The shape of the input to the discriminator
            excluding the batch size
        noise_dims (float): The dimensions of the input to the generator
        weight_decay (float): The decay for L2 regularization

    Returns:
        `tf.keras.Model`: The discriminator model

    """
    SpectralConv2D = spectralize(Conv2D)
    SpectralConditioning = spectralize(Conditioning)

    inputs = Input(shape=input_shape)
    noise = Input(shape=[noise_dims])
    labels = Input(shape=[])

    def conv_block(inputs, filters, norm=True):
        x = SpectralConv2D(
            filters,
            4,
            strides=2,
            padding="same",
            use_bias=False,
            kernel_regularizer=l2(weight_decay),
            input_shape=input_shape,
        )(inputs)
        x = SpectralConditioning(weight_decay)((x, noise, labels))
        if norm:
            x = LayerNormalization()(x)
        x = LeakyReLU(alpha=0.2)(x)
        return x

    x = conv_block(inputs, 64, norm=False)
    x = conv_block(x, 128)
    x = conv_block(x, 256)
    x = conv_block(x, 512)

    outputs = SpectralConv2D(
        1,
        4,
        strides=1,
        padding="valid",
        use_bias=False,
        kernel_regularizer=l2(weight_decay),
    )(x)

    return Model(inputs=[inputs, noise, labels], outputs=outputs)


def get_encoder(input_shape, noise_dims, weight_decay=2.5e-5):
    """Return the encoder model.

    Args:
        input_shape (tuple): The shape of the input to the encoder excluding
            the batch size
        noise_dims (float): The dimensions of the input to the generator model
        weight_decay (float): The decay for L2 regularization

    Returns:
        `tf.keras.Model`: The encoder model

    """
    inputs = Input(shape=input_shape)

    def conv_block(inputs, filters, norm=True):
        x = Conv2D(
            filters,
            4,
            strides=2,
            padding="same",
            use_bias=False,
            kernel_regularizer=l2(weight_decay),
            input_shape=input_shape,
        )(inputs)
        if norm:
            x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.2)(x)
        return x

    x = conv_block(inputs, 64, norm=False)
    x = conv_block(x, 128)
    x = conv_block(x, 256)
    x = conv_block(x, 512)

    noise = Conv2D(
        noise_dims,
        4,
        strides=1,
        padding="valid",
        use_bias=False,
        kernel_regularizer=l2(weight_decay),
    )(x)
    noise = Reshape([noise_dims])(noise)

    logits = Conv2D(
        10,
        4,
        strides=1,
        padding="valid",
        use_bias=False,
        kernel_regularizer=l2(weight_decay),
    )(x)
    logits = Reshape([10])(logits)

    return Model(inputs=inputs, outputs=[noise, logits])
