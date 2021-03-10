#!/usr/bin/env python3
"""Generate images using a trained GAN."""
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser, Namespace
from pathlib import Path

from tensorflow.keras.mixed_precision import set_global_policy

from gan.evaluation import GANEvaluator
from gan.models import get_generator
from gan.training import GANTrainer
from gan.utils import load_config


def main(args: Namespace) -> None:
    """Run the main program.

    Arguments:
        args: The object containing the commandline arguments
    """
    config = load_config(args.config)

    if config.mixed_precision:
        set_global_policy("mixed_float16")

    generator = get_generator(config)
    GANTrainer.load_generator_weights(generator, args.load_dir)

    helper = GANEvaluator(generator, config)
    helper.generate(args.imgs_per_digit, args.output_dir)


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Generate images using a trained GAN",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--load-dir",
        type=Path,
        default="./checkpoints/",
        help="directory where the trained model is saved",
    )
    parser.add_argument(
        "-c",
        "--config",
        type=Path,
        help="Path to a YAML config containing hyper-parameter values",
    )
    parser.add_argument(
        "--imgs-per-digit",
        type=int,
        default=1,
        help="number of images to generate per digit",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default="./outputs/",
        help="where to save the generated images",
    )
    main(parser.parse_args())
