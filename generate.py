#!/usr/bin/env python3
"""Generate images using a trained GAN."""
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser, Namespace
from pathlib import Path

from tensorflow.keras.mixed_precision import set_global_policy

from gan.data import IMG_SHAPE, NUM_CLS
from gan.evaluation import GANEvaluator
from gan.models import get_generator
from gan.training import GANTrainer


def main(args: Namespace) -> None:
    """Run the main program.

    Arguments:
        args: The object containing the commandline arguments
    """
    if args.mixed_precision:
        set_global_policy("mixed_float16")

    generator = get_generator(args.noise_dims, NUM_CLS, IMG_SHAPE[-1])
    GANTrainer.load_generator_weights(generator, Path(args.load_dir))

    helper = GANEvaluator(generator, noise_dims=args.noise_dims)
    helper.generate(
        args.imgs_per_digit, args.batch_size, Path(args.output_dir)
    )


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Generate images using a trained GAN",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--load-dir",
        type=str,
        default="./checkpoints/",
        help="directory where the trained model is saved",
    )
    parser.add_argument(
        "--noise-dims",
        type=int,
        default=100,
        help="dimensions of the generator noise vector",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="the number of images in each batch of generation",
    )
    parser.add_argument(
        "--mixed-precision",
        action="store_true",
        help="train with mixed-precision for higher performance",
    )
    parser.add_argument(
        "--imgs-per-digit",
        type=int,
        default=1,
        help="number of images to generate per digit",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./outputs/",
        help="where to save the generated images",
    )
    main(parser.parse_args())
