"""Class for training the GAN."""
import os
from typing import Dict, Tuple

import tensorflow as tf
from tensorflow import Tensor
from tensorflow.data import Dataset
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from tqdm import tqdm
from typing_extensions import Final

from .utils import get_grid, wasserstein_gradient_penalty


class BiGANTrainer:
    """Class to train an MNIST BiGAN."""

    GEN_PATH: Final[str] = "generator.ckpt"
    DISC_PATH: Final[str] = "discriminator.ckpt"
    ENC_PATH: Final[str] = "encoder.ckpt"

    def __init__(
        self,
        generator: Model,
        discriminator: Model,
        encoder: Model,
        noise_dims: int,
        gen_lr: float,
        disc_lr: float,
        enc_lr: float,
        gp_weight: float,
        cl_weight: float,
        log_dir: str,
        save_dir: str,
    ) -> None:
        """Store main models and info required for training.

        Args:
            generator: The generator model to be trained
            discriminator: The discriminator model to be trained
            encoder: The encoder model to be trained
            noise_dims: The dimensions for the inputs to the generator
            gen_lr: The learning rate for the generator's optimizer
            disc_lr: The learning rate for the discriminator's optimizer
            enc_lr: The learning rate for the encoder's optimizer
            gp_weight: Weights for the discriminator's gradient penalty
            cl_weight: Weights for the encoder's classification loss
            log_dir: Directory where to write event logs
            save_dir: Directory where to store model weights
        """
        self.generator = generator
        self.discriminator = discriminator
        self.encoder = encoder

        self.gen_optim = Adam(gen_lr, 0.5)
        self.disc_optim = Adam(disc_lr, 0.5)
        self.enc_optim = Adam(enc_lr, 0.5)

        self.writer = tf.summary.create_file_writer(log_dir)

        self.noise_dims = noise_dims
        self.gp_weight = gp_weight
        self.cl_weight = cl_weight

        self.save_dir = save_dir

    @tf.function
    def train_step(
        self, real: Tensor, labels: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, Dict[str, Tensor]]:
        """Run a single training step.

        The returned dict of losses are used for logging summaries.

        Args:
            real: The input real images
            labels: The corresponding input labels

        Returns:
            The generated images
            The predicted latent vectors for the real images
            The predicted labels for the real images
            dict: The dictionary of losses, as required by `log_summaries`
        """
        noise = tf.random.normal((real.get_shape()[0], self.noise_dims))

        with tf.GradientTape(persistent=True) as tape:
            pred_noise, pred_logits_real = self.encoder(real, training=True)
            pred_labels = tf.argmax(pred_logits_real, axis=-1)
            dis_real_out = self.discriminator(
                [real, pred_noise, pred_labels], training=True
            )

            # Tape for calculating gradient-penalty
            with tf.GradientTape() as gp_tape:
                # By default, the tape only tracks whatever's computed inside.
                # This forces it to track the noise, which is needed for the
                # gradient penalty loss.
                gp_tape.watch(noise)

                generated = self.generator([noise, labels], training=True)
                dis_fake_out = self.discriminator(
                    [generated, noise, labels], training=True
                )

            _, pred_logits_gen = self.encoder(generated, training=True)

            # Classification loss for the generator and encoder
            class_loss = tf.reduce_mean(
                # Loss for real images
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=labels, logits=pred_logits_real
                )
                # Loss for generated images
                + tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=labels, logits=pred_logits_gen
                )
            )

            # Total Wasserstein loss
            wass_loss = tf.reduce_mean(dis_real_out - dis_fake_out)

            gen_reg = sum(self.generator.losses)
            gen_loss = wass_loss + gen_reg + self.cl_weight * class_loss

            disc_reg = sum(self.discriminator.losses)
            grad_pen = wasserstein_gradient_penalty(
                noise, dis_fake_out, gp_tape
            )
            disc_loss = -wass_loss + disc_reg + self.gp_weight * grad_pen

            enc_reg = sum(self.encoder.losses)
            enc_loss = wass_loss + enc_reg + self.cl_weight * class_loss

        disc_grads = tape.gradient(
            disc_loss, self.discriminator.trainable_variables
        )
        self.disc_optim.apply_gradients(
            zip(disc_grads, self.discriminator.trainable_variables)
        )

        gen_grads = tape.gradient(gen_loss, self.generator.trainable_variables)
        self.gen_optim.apply_gradients(
            zip(gen_grads, self.generator.trainable_variables)
        )

        enc_grads = tape.gradient(enc_loss, self.encoder.trainable_variables)
        self.enc_optim.apply_gradients(
            zip(enc_grads, self.encoder.trainable_variables)
        )

        # Losses required for logging summaries
        losses = {
            "wass": wass_loss,
            "class": class_loss,
            "grad_pen": grad_pen,
            "gen_reg": gen_reg,
            "disc_reg": disc_reg,
            "enc_reg": enc_reg,
        }

        # Returned values are used for logging summaries
        return generated, pred_noise, pred_labels, losses

    def log_summaries(
        self,
        real: Tensor,
        generated: Tensor,
        pred_noise: Tensor,
        pred_labels: Tensor,
        losses: Dict[str, Tensor],
        global_step: int,
    ) -> None:
        """Log summaries to disk.

        The dict of losses should have the following key-value pairs:
            wass: The Wasserstein loss
            class: The classification loss
            grad_pen: The Wasserstein gradient penalty
            gen_reg: The L2 regularization loss for the generator
            disc_reg: The L2 regularization loss for the discriminator
            enc_reg: The L2 regularization loss for the encoder

        Args:
            real: The input real images
            generated: The generated images
            pred_noise: The predicted latent vectors for the real images
            pred_labels: The predicted labels for the real images
            losses: The dictionary of losses
            global_step: The current global training step
        """
        with self.writer.as_default():
            with tf.name_scope("losses"):
                tf.summary.scalar(
                    "wasserstein loss", losses["wass"], step=global_step,
                )
                tf.summary.scalar(
                    "classification loss", losses["class"], step=global_step,
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
                    "discriminator regularization",
                    losses["disc_reg"],
                    step=global_step,
                )
                tf.summary.scalar(
                    "encoder regularization",
                    losses["enc_reg"],
                    step=global_step,
                )

            # Save generated and real images in a square grid
            with tf.name_scope("image_summary"):
                real_grid = get_grid(real)
                gen_grid = get_grid(generated)
                recons_grid = get_grid(
                    self.generator([pred_noise, pred_labels])
                )
                tf.summary.image("real", real_grid, step=global_step)
                tf.summary.image("generated", gen_grid, step=global_step)
                tf.summary.image(
                    "reconstructed", recons_grid, step=global_step,
                )

    def save_models(self) -> None:
        """Save the models to disk."""
        self.generator.save_weights(os.path.join(self.save_dir, self.GEN_PATH))
        self.discriminator.save_weights(
            os.path.join(self.save_dir, self.DISC_PATH)
        )
        self.encoder.save_weights(os.path.join(self.save_dir, self.ENC_PATH))

    def train(
        self, dataset: Dataset, epochs: int, record_steps: int, save_steps: int
    ) -> None:
        """Execute the training loops for the BiGAN.

        Args:
            dataset: The dataset of real images
            epochs: Number of epochs to train the GAN
            record_steps: Step interval for recording summaries
            save_steps: Step interval for saving the model
        """
        # Total no. of batches in the dataset
        total_steps = tf.data.experimental.cardinality(dataset).numpy()

        # Global step is used for saving summaries.
        global_step: int = 1

        with tqdm(total=epochs * total_steps, desc="Training") as pbar:
            for _ in range(epochs):
                for real, labels in dataset:
                    (
                        generated,
                        pred_noise,
                        pred_labels,
                        losses,
                    ) = self.train_step(real, labels)

                    if global_step % record_steps == 0:
                        self.log_summaries(
                            real,
                            generated,
                            pred_noise,
                            pred_labels,
                            losses,
                            global_step,
                        )

                    if global_step % save_steps == 0:
                        self.save_models()

                    global_step += 1
                    pbar.update()

        self.save_models()
