#!/usr/bin/env python3
"""Generating MNIST digits using a conditional BiGAN."""
import os
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

from gan.generation import BiGANImgGenHelper
from gan.models import get_generator
from gan.training import BiGANTrainer


def main(args):
    """Run the main program.

    Arguments:
        args (`argparse.Namespace`): The object containing the commandline
            arguments

    """
    generator = get_generator(args.noise_dims)
    generator.load_weights(os.path.join(args.load_dir, BiGANTrainer.GEN_PATH))

    # Create the output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    helper = BiGANImgGenHelper(generator, noise_dims=args.noise_dims)
    helper.generate(imgs_per_digit=args.imgs, output_dir=args.output_dir)


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Generating MNIST digits using a conditional BiGAN",
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
        "--imgs",
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
