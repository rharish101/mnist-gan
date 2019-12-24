#!/usr/bin/env python3
"""Training a conditional BiGAN for MNIST."""
import json
import os
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from datetime import datetime

import tensorflow as tf
from tqdm import tqdm

from data import get_mnist_dataset
from model import get_discriminator, get_encoder, get_generator
from utils import get_grid, wasserstein_gradient_penalty

GEN_PATH = "generator.ckpt"
DISC_PATH = "discriminator.ckpt"
ENC_PATH = "encoder.ckpt"


@tf.function
def _train_step(
    real,
    labels,
    generator,
    discriminator,
    encoder,
    gen_optim,
    disc_optim,
    enc_optim,
    noise_dims,
    gp_weight,
    cl_weight,
):
    """Run a single training step."""
    noise = tf.random.normal((real.get_shape()[0], noise_dims))

    with tf.GradientTape(persistent=True) as tape:
        pred_noise, pred_logits_real = encoder(real, training=True)
        pred_labels = tf.argmax(pred_logits_real, axis=-1)
        dis_real_out = discriminator(
            [real, pred_noise, pred_labels], training=True
        )

        # Tape for calculating gradient-penalty
        with tf.GradientTape() as gp_tape:
            # By default, the tape only tracks whatever's computed inside. This
            # forces it to track the noise, which is needed for the gradient
            # penalty loss.
            gp_tape.watch(noise)

            generated = generator([noise, labels], training=True)
            dis_fake_out = discriminator(
                [generated, noise, labels], training=True
            )

        _, pred_logits_gen = encoder(generated, training=True)

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
        wass_loss = tf.reduce_mean(dis_real_out) - tf.reduce_mean(dis_fake_out)

        gen_reg = sum(generator.losses)
        gen_loss = wass_loss + gen_reg + cl_weight * class_loss

        disc_reg = sum(discriminator.losses)
        grad_pen = wasserstein_gradient_penalty(noise, dis_fake_out, gp_tape)
        disc_loss = -wass_loss + disc_reg + gp_weight * grad_pen

        enc_reg = sum(encoder.losses)
        enc_loss = wass_loss + enc_reg + cl_weight * class_loss

    disc_grads = tape.gradient(disc_loss, discriminator.trainable_variables)
    disc_optim.apply_gradients(
        zip(disc_grads, discriminator.trainable_variables)
    )

    gen_grads = tape.gradient(gen_loss, generator.trainable_variables)
    gen_optim.apply_gradients(zip(gen_grads, generator.trainable_variables))

    enc_grads = tape.gradient(enc_loss, encoder.trainable_variables)
    enc_optim.apply_gradients(zip(enc_grads, encoder.trainable_variables))

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


def train_loop(
    generator,
    discriminator,
    encoder,
    dataset,
    gen_optim,
    disc_optim,
    enc_optim,
    writer,
    noise_dims,
    gp_weight,
    cl_weight,
    epochs,
    record_steps,
    save_steps,
    save_dir,
):
    """Execute the training loops for the BiGAN.

    Args:
        generator (`tf.keras.Model`): The generator model to be trained
        discriminator (`tf.keras.Model`): The discriminator model to be trained
        encoder (`tf.keras.Model`): The encoder model to be trained
        dataset (`tf.data.Dataset`): The dataset of real images
        gen_optim (`tf.train.Optimizer`): The optimizer for training the
            generator
        disc_optim (`tf.train.Optimizer`): The optimizer for training the
            discriminator
        enc_optim (`tf.train.Optimizer`): The optimizer for training the
            encoder
        writer (`tf.summary.SummaryWriter`): The writer for logging summaries
        noise_dims (int): The dimensions for the inputs to the generator
        gp_weight (float): Weights for the discriminator's gradient penalty
        cl_weight (float): Weights for the encoder's classification loss
        epochs (int): Number of epochs to train the GAN
        record_steps (int): Step interval for recording summaries
        save_steps (int): Step interval for saving the model
        save_dir (str): Directory where to store model weights

    """
    # Total no. of batches in the dataset
    total_steps = tf.data.experimental.cardinality(dataset).numpy()

    # Global step is used for saving summaries.
    global_step = 1
    with tqdm(total=epochs * total_steps, desc="Training") as pbar:
        for _ in range(epochs):
            for real, labels in dataset:
                generated, pred_noise, pred_labels, losses = _train_step(
                    real,
                    labels,
                    generator,
                    discriminator,
                    encoder,
                    gen_optim,
                    disc_optim,
                    enc_optim,
                    noise_dims=noise_dims,
                    gp_weight=gp_weight,
                    cl_weight=cl_weight,
                )

                if global_step % record_steps == 0:
                    with writer.as_default():
                        with tf.name_scope("losses"):
                            tf.summary.scalar(
                                "wasserstein loss",
                                losses["wass"],
                                step=global_step,
                            )
                            tf.summary.scalar(
                                "classification loss",
                                losses["class"],
                                step=global_step,
                            )
                            tf.summary.scalar(
                                "gradient penalty",
                                losses["grad_pen"],
                                step=global_step,
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
                                generator([pred_noise, pred_labels])
                            )
                            tf.summary.image(
                                "real", real_grid, step=global_step
                            )
                            tf.summary.image(
                                "generated", gen_grid, step=global_step
                            )
                            tf.summary.image(
                                "reconstructed", recons_grid, step=global_step
                            )

                if global_step % save_steps == 0:
                    generator.save_weights(os.path.join(save_dir, GEN_PATH))
                    discriminator.save_weights(
                        os.path.join(save_dir, DISC_PATH)
                    )
                    encoder.save_weights(os.path.join(save_dir, ENC_PATH))

                global_step += 1
                pbar.update()

    generator.save_weights(os.path.join(save_dir, GEN_PATH))
    discriminator.save_weights(os.path.join(save_dir, DISC_PATH))
    encoder.save_weights(os.path.join(save_dir, ENC_PATH))


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
    # Remove microseconds and convert to ISO 8601 YYYY-MM-DDThh:mm:ss format.
    time_stamp = datetime.now().replace(microsecond=0).isoformat()
    log_dir = os.path.join(args.log_dir, time_stamp)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # Save hyperparams in both log and save directories
    with open(os.path.join(log_dir, "config.json"), "w") as conf:
        json.dump(vars(args), conf)
    with open(os.path.join(args.save_dir, "config.json"), "w") as conf:
        json.dump(vars(args), conf)

    writer = tf.summary.create_file_writer(log_dir)

    train_loop(
        generator,
        discriminator,
        encoder,
        dataset,
        gen_optim,
        disc_optim,
        enc_optim,
        writer,
        noise_dims=args.noise_dims,
        gp_weight=args.gp_weight,
        cl_weight=args.cl_weight,
        epochs=args.epochs,
        record_steps=args.record_steps,
        save_steps=args.save_steps,
        save_dir=args.save_dir,
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
