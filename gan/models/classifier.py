"""Classifier model for FID."""
from typing import Optional

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

from ..data import IMG_SHAPE, NUM_CLS
from ..utils import Config


class Classifier(Model):
    """The image classifier model.

    Attributes:
        feature_extract: The feature extraction layers
        final: The final classification layers
        config: The hyper-param config
    """

    def __init__(self, config: Config):
        """Initialize all intermediate layers.

        Args:
            config: The hyper-param config
        """
        super().__init__()
        self.config = config

        base_layers = [
            Input(shape=IMG_SHAPE),
            self.conv_block(8),
            self.conv_block(16),
            self.conv_block(32),
            self.conv_block(64),
            Flatten(),
            Dense(
                128,
                use_bias=False,
                kernel_regularizer=l2(config.cls_weight_decay),
            ),
            BatchNormalization(),
            ReLU(),
        ]

        self.feature_extract = Sequential(base_layers)
        self.final = Dense(NUM_CLS, dtype="float32")

    def conv_block(self, out_channels: int) -> Sequential:
        """Return a convolution block with batch-norm and ReLU."""
        layers = [
            Conv2D(
                out_channels,
                kernel_size=4,
                strides=2,
                use_bias=False,
                padding="same",
                kernel_regularizer=l2(self.config.cls_weight_decay),
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
