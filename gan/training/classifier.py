"""Class for training a classifier for FID."""
import os

import tensorflow as tf
from tensorflow.data import Dataset
from tensorflow.keras import Model
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from typing_extensions import Final


class ClassifierTrainer:
    """Class for training the classifier.

    Attributes:
        model: The classifier model being trained
    """

    CLS_PATH: Final[str] = "classifier.ckpt"

    def __init__(self, model: Model, lr: float):
        """Store the main model and other required info.

        Args:
            model: The classifier model to be trained
            lr: The learning rate for Adam
        """
        model.compile(
            optimizer=Adam(lr),
            loss=SparseCategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"],
        )
        self.model = model

    def train(
        self,
        train_dataset: Dataset,
        val_dataset: Dataset,
        epochs: int,
        log_dir: str,
        record_eps: int,
        save_dir: str,
        save_steps: int,
    ) -> None:
        """Execute the training loops for the classifier.

        Args:
            train_dataset: The training dataset
            val_dataset: The validation dataset
            epochs: Maximum number of epochs to train the classifier
            log_dir: Directory where to log summaries
            record_eps: Epoch interval for recording summaries
            save_dir: Directory where to store model weights
            save_steps: Step interval for saving the model
        """
        # Total no. of batches in the dataset
        dataset_size = tf.data.experimental.cardinality(train_dataset).numpy()

        logger = TensorBoard(
            log_dir=log_dir,
            write_graph=False,
            update_freq=record_eps * dataset_size,
            profile_batch=0,
        )

        save_path = os.path.join(save_dir, self.CLS_PATH)
        saver = ModelCheckpoint(
            save_path, save_weights_only=True, save_freq=save_steps
        )

        self.model.fit(
            train_dataset,
            epochs=epochs,
            callbacks=[logger, saver],
            validation_data=val_dataset,
            shuffle=False,  # already done during dataset creation
            validation_freq=record_eps,
        )
        self.model.save_weights(save_path)

    @classmethod
    def load_weights(cls, model: Model, load_dir: str) -> None:
        """Load the model's weights from disk.

        This replaces the model's weights with the loaded ones, in place.

        Args:
            model: The model whose weights are to be loaded
            load_dir: Directory from where to load model weights
        """
        status = model.load_weights(os.path.join(load_dir, cls.CLS_PATH))
        status.expect_partial()
