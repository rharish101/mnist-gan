#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2019 Harish Rajagopal <harish.rajagopals@gmail.com>
#
# SPDX-License-Identifier: MIT

"""Train a GAN."""
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser, Namespace
from pathlib import Path
from typing import Final

import tensorflow as tf
from tensorflow.keras.mixed_precision import set_global_policy

from gan.data import get_dataset
from gan.models import Classifier, get_critic, get_generator
from gan.training import ClassifierTrainer, GANTrainer
from gan.utils import load_config, setup_dirs

CONFIG: Final = "config-gan.toml"


def main(args: Namespace) -> None:
    """Run the main program.

    Arguments:
        args: The object containing the commandline arguments
    """
    config = load_config(args.config)

    strategy = tf.distribute.MirroredStrategy()
    if config.mixed_precision:
        set_global_policy("mixed_float16")

    train_dataset, test_dataset = get_dataset(
        args.data_path, config.gan_batch_size
    )

    with strategy.scope():
        generator = get_generator(config)
        critic = get_critic(config)

        classifier = Classifier(config)
        ClassifierTrainer.load_weights(classifier, args.load_dir)

    # Save each run into a directory by its timestamp
    log_dir = setup_dirs(
        dirs=[args.save_dir],
        dirs_to_tstamp=[args.log_dir],
        config=config,
        file_name=CONFIG,
    )[0]

    trainer = GANTrainer(
        generator,
        critic,
        classifier,
        strategy,
        train_dataset,
        test_dataset,
        config=config,
        log_dir=log_dir,
        save_dir=args.save_dir,
    )
    trainer.train(
        record_steps=args.record_steps,
        log_graph=args.log_graph,
        save_steps=args.save_steps,
    )


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Train a GAN",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default="./datasets/MNIST/",
        help="path to the dataset",
    )
    parser.add_argument(
        "-c",
        "--config",
        type=Path,
        help="Path to a TOML config containing hyper-parameter values",
    )
    parser.add_argument(
        "--load-dir",
        type=Path,
        default="./checkpoints/",
        help="directory where the trained classifier model is saved",
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        default="./checkpoints/",
        help="directory where to save the GAN models",
    )
    parser.add_argument(
        "--save-steps",
        type=int,
        default=5000,
        help="the frequency of saving the model (in steps)",
    )
    parser.add_argument(
        "--record-steps",
        type=int,
        default=500,
        help="the frequency of recording summaries (in steps)",
    )
    parser.add_argument(
        "--log-graph",
        action="store_true",
        help="whether to log the graph of the model",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default="./logs/gan",
        help="directory where to write event logs",
    )
    main(parser.parse_args())
