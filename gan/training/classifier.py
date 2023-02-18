# SPDX-FileCopyrightText: 2020 Harish Rajagopal <harish.rajagopals@gmail.com>
#
# SPDX-License-Identifier: MIT

"""Class for training a classifier for FID."""
from pathlib import Path
from typing import Final

import tensorflow as tf
from tensorflow.data import Dataset
from tensorflow.keras import Model
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam

from ..utils import Config


class ClassifierTrainer:
    """Class for training the classifier.

    Attributes:
        model: The classifier model being trained
        config: The hyper-param config
    """

    CLS_PATH: Final = "classifier.ckpt"

    def __init__(
        self, model: Model, strategy: tf.distribute.Strategy, config: Config
    ):
        """Store the main model and other required info.

        Args:
            model: The classifier model to be trained
            strategy: The distribution strategy for training the model
            config: The hyper-param config
        """
        with strategy.scope():
            # Takes care of mixed-precision by itself
            model.compile(
                optimizer=Adam(config.cls_lr),
                loss=SparseCategoricalCrossentropy(from_logits=True),
                metrics=["accuracy"],
            )

        self.model = model
        self.config = config

    def train(
        self,
        train_dataset: Dataset,
        val_dataset: Dataset,
        log_dir: Path,
        record_eps: int,
        save_dir: Path,
        save_steps: int,
        log_graph: bool = False,
    ) -> None:
        """Execute the training loops for the classifier.

        Args:
            train_dataset: The training dataset
            val_dataset: The validation dataset
            log_dir: Directory where to log summaries
            record_eps: Epoch interval for recording summaries
            save_dir: Directory where to store model weights
            save_steps: Step interval for saving the model
            log_graph: Whether to log the graph of the model
        """
        # Total no. of batches in the dataset
        dataset_size = train_dataset.cardinality().numpy()

        logger = TensorBoard(
            log_dir=log_dir,
            write_graph=log_graph,
            update_freq=record_eps * dataset_size,
            histogram_freq=record_eps,
            profile_batch=0,
        )

        save_path = save_dir / self.CLS_PATH
        saver = ModelCheckpoint(
            save_path, save_weights_only=True, save_freq=save_steps
        )

        self.model.fit(
            train_dataset,
            epochs=self.config.cls_epochs,
            callbacks=[logger, saver],
            validation_data=val_dataset,
            shuffle=False,  # already done during dataset creation
            validation_freq=record_eps,
        )
        self.model.save_weights(save_path)

    @classmethod
    def load_weights(cls, model: Model, load_dir: Path) -> None:
        """Load the model's weights from disk.

        This replaces the model's weights with the loaded ones, in place.

        Args:
            model: The model whose weights are to be loaded
            load_dir: Directory from where to load model weights
        """
        status = model.load_weights(load_dir / cls.CLS_PATH)
        status.expect_partial()
