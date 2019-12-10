#!/usr/bin/env python3
"""Generating MNIST digits using a conditional BiGAN."""
import os
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

import tensorflow as tf
from tqdm import tqdm

from model import get_generator


def main(args):
    """Run the main program.

    Arguments:
        args (`argparse.Namespace`): The object containing the commandline
            arguments

    """
    generator = get_generator(args.noise_dims)
    generator.load_weights(os.path.join(args.load_dir, "generator.ckpt"))

    # Create the output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    with tqdm(total=10 * args.imgs, desc="Saving") as pbar:
        for digit in range(10):
            for instance in range(args.imgs):
                noise = tf.random.normal([1, args.noise_dims])
                label = tf.convert_to_tensor([digit])  # adding batch-axis of 1
                generated = generator([noise, label])

                # Convert 4D [-1, 1] float32 to 3D [0, 255] uint8
                output = generated / 2 + 0.5
                output = tf.image.convert_image_dtype(output, tf.uint8)
                output = output[0]

                img_str = tf.image.encode_jpeg(output)
                img_name = os.path.join(
                    args.output_dir, f"{digit}_{instance}.jpg"
                )
                with open(img_name, "wb") as img_file:
                    img_file.write(img_str.numpy())

                pbar.update()


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
