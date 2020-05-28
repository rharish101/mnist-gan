"""Classifier model for FID."""
from typing import Optional, Tuple

from tensorflow import Tensor
from tensorflow.keras import Input, Model, Sequential
from tensorflow.keras.layers import (
    BatchNormalization,
    Conv2D,
    Dense,
    Flatten,
    ReLU,
)
from tensorflow.keras.regularizers import l2


class Classifier(Model):
    """The image classifier model.

    Attributes:
        weight_decay: The decay for L2 regularization
        feature_extract: The feature extraction layers
        final: The final classification layers
    """

    def __init__(
        self,
        input_shape: Tuple[int, int, int],
        num_classes: int,
        weight_decay: float = 2.5e-5,
    ) -> None:
        """Initialize all intermediate layers.

        Args:
            input_shape: The shape of the inputs excluding the batch size
            num_classes: The number of classes for classification
            weight_decay: The decay for L2 regularization
        """
        super().__init__()
        self.weight_decay = weight_decay

        base_layers = [
            Input(shape=input_shape),
            self.conv_block(8),
            self.conv_block(16),
            self.conv_block(32),
            self.conv_block(64),
            Flatten(),
            Dense(128, use_bias=False, kernel_regularizer=l2(weight_decay)),
            BatchNormalization(),
            ReLU(),
        ]

        self.feature_extract = Sequential(base_layers)
        self.final = Dense(num_classes)

    def conv_block(self, out_channels: int) -> Sequential:
        """Return a ResNet block with batch-norm and ReLU."""
        layers = [
            Conv2D(
                out_channels,
                kernel_size=4,
                strides=2,
                use_bias=False,
                padding="same",
                kernel_regularizer=l2(self.weight_decay),
            ),
            BatchNormalization(),
            ReLU(),
        ]
        return Sequential(layers)

    def call(self, inputs: Tensor, training: Optional[bool] = False) -> Tensor:
        """Return the classifier's outputs for the given image."""
        x = self.feature_extract(inputs, training=training)
        outputs = self.final(x, training=training)
        return outputs
