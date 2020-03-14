#!/usr/bin/env python3
"""Training a conditional BiGAN for MNIST."""
import os
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from datetime import datetime

import tensorflow as tf
import yaml
from tqdm import tqdm

from data import get_mnist_dataset
from model import get_discriminator, get_encoder, get_generator
from utils import get_grid, wasserstein_gradient_penalty

CONFIG_YAML = "config.yaml"


class BiGANTrainer:
    """Class to train an MNIST BiGAN."""

    GEN_PATH = "generator.ckpt"
    DISC_PATH = "discriminator.ckpt"
    ENC_PATH = "encoder.ckpt"

    def __init__(
        self,
        generator,
        discriminator,
        encoder,
        gen_optim,
        disc_optim,
        enc_optim,
        writer,
        noise_dims,
        gp_weight,
        cl_weight,
        save_dir,
    ):
        """Store main models and info required for training.

        Args:
            generator (`tf.keras.Model`): The generator model to be trained
            discriminator (`tf.keras.Model`): The discriminator model to be
                trained
            encoder (`tf.keras.Model`): The encoder model to be trained
            gen_optim (`tf.train.Optimizer`): The optimizer for training the
                generator
            disc_optim (`tf.train.Optimizer`): The optimizer for training the
                discriminator
            enc_optim (`tf.train.Optimizer`): The optimizer for training the
                encoder
            writer (`tf.summary.SummaryWriter`): The writer for logging
                summaries
            noise_dims (int): The dimensions for the inputs to the generator
            gp_weight (float): Weights for the discriminator's gradient penalty
            cl_weight (float): Weights for the encoder's classification loss
            save_dir (str): Directory where to store model weights

        """
        self.generator = generator
        self.discriminator = discriminator
        self.encoder = encoder

        self.gen_optim = gen_optim
        self.disc_optim = disc_optim
        self.enc_optim = enc_optim

        self.writer = writer

        self.noise_dims = noise_dims
        self.gp_weight = gp_weight
        self.cl_weight = cl_weight

        self.save_dir = save_dir

    @tf.function
    def train_step(self, real, labels):
        """Run a single training step.

        The returned dict of losses are used for logging summaries.

        Args:
            real (`tf.Tensor`): The input real images
            labels (`tf.Tensor`): The corresponding input labels

        Returns:
            `tf.Tensor`: The generated images
            `tf.Tensor`: The predicted latent vectors for the real images
            `tf.Tensor`: The predicted labels for the real images
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
        self, real, generated, pred_noise, pred_labels, losses, global_step
    ):
        """Log summaries to disk.

        The dict of losses should have the following key-value pairs:
            wass: The Wasserstein loss
            class: The classification loss
            grad_pen: The Wasserstein gradient penalty
            gen_reg: The L2 regularization loss for the generator
            disc_reg: The L2 regularization loss for the discriminator
            enc_reg: The L2 regularization loss for the encoder

        Args:
            real (`tf.Tensor`): The input real images
            generated (`tf.Tensor`): The generated images
            pred_noise (`tf.Tensor`): The predicted latent vectors for the real
                images
            pred_labels (`tf.Tensor`): The predicted labels for the real images
            losses (dict): The dictionary of losses
            global_step (int): The current global training step

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

    def save_models(self):
        """Save the models to disk."""
        self.generator.save_weights(os.path.join(self.save_dir, self.GEN_PATH))
        self.discriminator.save_weights(
            os.path.join(self.save_dir, self.DISC_PATH)
        )
        self.encoder.save_weights(os.path.join(self.save_dir, self.ENC_PATH))

    def train(self, dataset, epochs, record_steps, save_steps):
        """Execute the training loops for the BiGAN.

        Args:
            dataset (`tf.data.Dataset`): The dataset of real images
            epochs (int): Number of epochs to train the GAN
            record_steps (int): Step interval for recording summaries
            save_steps (int): Step interval for saving the model

        """
        # Total no. of batches in the dataset
        total_steps = tf.data.experimental.cardinality(dataset).numpy()

        # Global step is used for saving summaries.
        global_step = 1

        with tqdm(total=epochs * total_steps, desc="Training") as pbar:
            for _ in range(epochs):
                for real, labels in dataset:
                    items_to_log = self.train_step(real, labels)

                    if global_step % record_steps == 0:
                        self.log_summaries(
                            real, *items_to_log, global_step,
                        )

                    if global_step % save_steps == 0:
                        self.save_models()

                    global_step += 1
                    pbar.update()

        self.save_models()


def main(args):
    """Run the main program.

    Arguments:
        args (`argparse.Namespace`): The object containing the commandline
            arguments

    """
    dataset = get_mnist_dataset(args.mnist_path, args.batch_size)
    image_shape = dataset.element_spec[0].shape.as_list()[1:]

    generator = get_generator(args.noise_dims, weight_decay=args.weight_decay)
    discriminator = get_discriminator(
        image_shape, args.noise_dims, weight_decay=args.weight_decay,
    )
    encoder = get_encoder(
        image_shape, args.noise_dims, weight_decay=args.weight_decay,
    )

    gen_optim = tf.keras.optimizers.Adam(args.gen_lr, 0.5)
    disc_optim = tf.keras.optimizers.Adam(args.disc_lr, 0.5)
    enc_optim = tf.keras.optimizers.Adam(args.enc_lr, 0.5)

    # Save each run into a directory by its timestamp.
    # Timestamp is in ISO 8601 YYYY-MM-DDThh:mm:ss format.
    time_stamp = datetime.now().isoformat()
    log_dir = os.path.join(args.log_dir, time_stamp)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # Save hyperparams in both log and save directories
    for directory in log_dir, args.save_dir:
        with open(os.path.join(directory, CONFIG_YAML), "w") as conf:
            yaml.dump(vars(args), conf)

    writer = tf.summary.create_file_writer(log_dir)

    trainer = BiGANTrainer(
        generator,
        discriminator,
        encoder,
        gen_optim,
        disc_optim,
        enc_optim,
        writer,
        noise_dims=args.noise_dims,
        gp_weight=args.gp_weight,
        cl_weight=args.cl_weight,
        save_dir=args.save_dir,
    )
    trainer.train(
        dataset,
        epochs=args.epochs,
        record_steps=args.record_steps,
        save_steps=args.save_steps,
    )


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Training an conditional BiGAN for MNIST",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mnist-path",
        type=str,
        default="./datasets/MNIST/",
        help="path to the MNIST dataset",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="the number of images in each batch",
    )
    parser.add_argument(
        "--noise-dims",
        type=int,
        default=100,
        help="dimensions of the generator noise vector",
    )
    parser.add_argument(
        "--gen-lr",
        type=float,
        default=1e-4,
        help="learning rate for generator optimization",
    )
    parser.add_argument(
        "--disc-lr",
        type=float,
        default=1e-4,
        help="learning rate for discriminator optimization",
    )
    parser.add_argument(
        "--enc-lr",
        type=float,
        default=2e-4,
        help="learning rate for encoder optimization",
    )
    parser.add_argument(
        "--gp-weight",
        type=float,
        default=1.0,
        help="weights for the discriminator's gradient penalty",
    )
    parser.add_argument(
        "--cl-weight",
        type=float,
        default=1.0,
        help="weights for the encoder's classification loss",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=2.5e-5,
        help="L2 weight decay rate",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="the number of epochs for training the GAN",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="./checkpoints/",
        help="directory where to save model",
    )
    parser.add_argument(
        "--save-steps",
        type=int,
        default=5000,
        help="the frequency of saving the model",
    )
    parser.add_argument(
        "--record-steps",
        type=int,
        default=100,
        help="the frequency of recording summaries",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="./logs/",
        help="directory where to write event logs",
    )
    main(parser.parse_args())
