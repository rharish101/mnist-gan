"""Class for training the GAN."""
import os
from typing import Dict, Iterable, Tuple, TypeVar

import tensorflow as tf
from tensorflow import Tensor
from tensorflow.data import Dataset
from tensorflow.data.experimental import cardinality
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from tqdm import tqdm
from typing_extensions import Final

from ..evaluation import RunningFID
from ..utils import get_grid, wasserstein_gradient_penalty

_OuterLoopType = TypeVar("_OuterLoopType")
_InnerLoopType = TypeVar("_InnerLoopType")


class GANTrainer:
    """Class to train an MNIST GAN.

    Attributes:
        generator: The generator model being trained
        critic: The critic model being trained
        train_dataset: The dataset of real images and labels for training
        val_dataset: The dataset of real images and labels for validation
        gen_optim: The optimizer for the generator
        crit_optim: The optimizer for the critic
        evaluator: The object that calculates the running FID
        writer: The summary writer to log TensorBoard summaries
        noise_dims: The dimensions for the inputs to the generator
        gp_weight: Weights for the critic's gradient penalty
        save_dir: Directory where to store model weights
    """

    GEN_PATH: Final[str] = "generator.ckpt"
    CRIT_PATH: Final[str] = "critic.ckpt"

    def __init__(
        self,
        generator: Model,
        critic: Model,
        classifier: Model,
        train_dataset: Dataset,
        val_dataset: Dataset,
        noise_dims: int,
        gen_lr: float,
        crit_lr: float,
        gp_weight: float,
        log_dir: str,
        save_dir: str,
    ) -> None:
        """Store main models and info required for training.

        Args:
            generator: The generator model to be trained
            critic: The critic model to be trained
            classifier: The trained classifier model for FID
            train_dataset: The dataset of real images and labels for training
            val_dataset: The dataset of real images and labels for validation
            noise_dims: The dimensions for the inputs to the generator
            gen_lr: The learning rate for the generator's optimizer
            crit_lr: The learning rate for the critic's optimizer
            gp_weight: Weights for the critic's gradient penalty
            log_dir: Directory where to write event logs
            save_dir: Directory where to store model weights
        """
        self.generator = generator
        self.critic = critic

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        self.gen_optim = Adam(gen_lr, 0.5)
        self.crit_optim = Adam(crit_lr, 0.5)

        self.evaluator = RunningFID(classifier)
        self.writer = tf.summary.create_file_writer(log_dir)

        self.noise_dims = noise_dims
        self.gp_weight = gp_weight

        self.save_dir = save_dir

    @tf.function
    def train_step(
        self, real: Tensor, labels: Tensor, train_gen: bool = True
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """Run a single training step.

        The returned dict of losses are used for logging summaries.

        Args:
            real: The input real images
            labels: The corresponding input labels
            train_gen: Whether to train the generator

        Returns:
            The generated images
            dict: The dictionary of losses, as required by `log_summaries`
        """
        noise = tf.random.normal((real.get_shape()[0], self.noise_dims))

        with tf.GradientTape(persistent=True) as tape:
            crit_real_out = self.critic([real, labels], training=True)

            generated = self.generator([noise, labels], training=True)
            crit_fake_out = self.critic([generated, labels], training=True)

            # Tape for calculating gradient-penalty
            with tf.GradientTape() as gp_tape:
                # U[0, 1] random value used for linear interpolation
                gp_rand = tf.random.uniform(())
                gp_inputs = real * gp_rand + generated * (1 - gp_rand)

                # Forces the tape to track the inputs, which is needed for
                # calculating gradients in the gradient penalty.
                gp_tape.watch(gp_inputs)
                crit_gp_out = self.critic([gp_inputs, labels], training=True)

            # Total Wasserstein loss
            wass_loss = tf.reduce_mean(crit_real_out - crit_fake_out)

            gen_reg = sum(self.generator.losses)
            gen_loss = wass_loss + gen_reg

            crit_reg = sum(self.critic.losses)
            grad_pen = wasserstein_gradient_penalty(
                gp_inputs, crit_gp_out, gp_tape
            )
            crit_loss = -wass_loss + crit_reg + self.gp_weight * grad_pen

        crit_grads = tape.gradient(crit_loss, self.critic.trainable_variables)
        self.crit_optim.apply_gradients(
            zip(crit_grads, self.critic.trainable_variables)
        )

        if train_gen:
            gen_grads = tape.gradient(
                gen_loss, self.generator.trainable_variables
            )
            self.gen_optim.apply_gradients(
                zip(gen_grads, self.generator.trainable_variables)
            )

        # Losses required for logging summaries
        losses = {
            "wass": wass_loss,
            "grad_pen": grad_pen,
            "gen_reg": gen_reg,
            "crit_reg": crit_reg,
        }

        # Returned values are used for logging summaries
        return generated, losses

    def _get_fid(self) -> Tensor:
        """Calculate FID over the validation dataset."""
        self.evaluator.reset()

        for real, lbls in self.val_dataset:
            inputs = tf.random.normal((real.shape[0], self.noise_dims))
            generated = self.generator([inputs, lbls])
            self.evaluator.update(real, generated)

        return self.evaluator.get_fid()

    def log_summaries(
        self,
        real: Tensor,
        generated: Tensor,
        losses: Dict[str, Tensor],
        global_step: int,
    ) -> None:
        """Log summaries to disk.

        The dict of losses should have the following key-value pairs:
            wass: The Wasserstein loss
            grad_pen: The Wasserstein gradient penalty
            gen_reg: The L2 regularization loss for the generator
            crit_reg: The L2 regularization loss for the critic

        Args:
            real: The input real images
            generated: The generated images
            losses: The dictionary of losses
            global_step: The current global training step
        """
        with self.writer.as_default():
            with tf.name_scope("losses"):
                tf.summary.scalar(
                    "wasserstein loss", losses["wass"], step=global_step,
                )
                tf.summary.scalar(
                    "gradient penalty", losses["grad_pen"], step=global_step,
                )
                tf.summary.scalar(
                    "generator regularization",
                    losses["gen_reg"],
                    step=global_step,
                )
                tf.summary.scalar(
                    "critic regularization",
                    losses["crit_reg"],
                    step=global_step,
                )

            with tf.name_scope("metrics"):
                fid = self._get_fid()
                tf.summary.scalar("FID", fid, step=global_step)

            # Save generated and real images in a square grid
            with tf.name_scope("image_summary"):
                real_grid = get_grid(real)
                gen_grid = get_grid(generated)
                tf.summary.image("real", real_grid, step=global_step)
                tf.summary.image("generated", gen_grid, step=global_step)

    def save_models(self) -> None:
        """Save the models to disk."""
        for model, file_name in [
            (self.generator, self.GEN_PATH),
            (self.critic, self.CRIT_PATH),
        ]:
            model.save_weights(os.path.join(self.save_dir, file_name))

    @staticmethod
    def _nested_loops(
        outer: Iterable[_OuterLoopType], inner: Iterable[_InnerLoopType]
    ) -> Iterable[Tuple[_OuterLoopType, _InnerLoopType]]:
        """A nested loop generator.

        This works similar to `itertools.product`, but without duplicating
        stuff in memory. This notably happens when using TensorFlow datasets.
        """
        for i in outer:
            for j in inner:
                yield i, j

    def train(
        self, disc_steps: int, epochs: int, record_steps: int, save_steps: int
    ) -> None:
        """Execute the training loops for the GAN.

        Args:
            disc_steps: The number of discriminator steps per generator step
            epochs: Number of epochs to train the GAN
            record_steps: Step interval for recording summaries
            save_steps: Step interval for saving the model
        """
        # Total no. of batches in the training dataset
        total_batches = cardinality(self.train_dataset).numpy()

        # Iterate over dataset in epochs
        data_in_epochs = self._nested_loops(range(epochs), self.train_dataset)

        for global_step, (_, (real, lbls)) in tqdm(
            enumerate(data_in_epochs, 1),
            total=epochs * total_batches,
            desc="Training",
        ):
            # XXX: Run a single step to initialize all optimizer variables.
            # This is a workaround for TensorFlow issue here:
            # https://github.com/tensorflow/tensorflow/issues/27120
            if global_step == 1:
                self.train_step(real, lbls)

            for _ in range(disc_steps - 1):
                self.train_step(real, lbls, train_gen=False)
            gen, losses = self.train_step(real, lbls)

            if global_step % record_steps == 0:
                self.log_summaries(
                    real, gen, losses, global_step,
                )

            if global_step % save_steps == 0:
                self.save_models()

        self.save_models()

    @classmethod
    def load_generator_weights(cls, generator: Model, load_dir: str) -> None:
        """Load the generator weights from disk.

        This replaces the generator's weights with the loaded ones, in place.

        Args:
            generator: The generator model whose weights are to be loaded
            load_dir: Directory from where to load model weights
        """
        generator.load_weights(os.path.join(load_dir, cls.GEN_PATH))
