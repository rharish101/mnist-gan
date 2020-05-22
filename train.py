#!/usr/bin/env python3
"""Training a conditional BiGAN for MNIST."""
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser, Namespace

from typing_extensions import Final

from gan.data import NUM_CLS, get_mnist_dataset
from gan.models import (
    Classifier,
    get_discriminator,
    get_encoder,
    get_generator,
)
from gan.training import BiGANTrainer, ClassifierTrainer
from gan.utils import setup_dirs

CONFIG: Final[str] = "config-gan.yaml"


def main(args: Namespace) -> None:
    """Run the main program.

    Arguments:
        args: The object containing the commandline arguments
    """
    train_dataset, test_dataset = get_mnist_dataset(
        args.mnist_path, args.batch_size
    )
    image_shape = train_dataset.element_spec[0].shape.as_list()[1:]

    generator = get_generator(args.noise_dims, weight_decay=args.weight_decay)
    discriminator = get_discriminator(
        image_shape, args.noise_dims, weight_decay=args.weight_decay,
    )
    encoder = get_encoder(
        image_shape, args.noise_dims, weight_decay=args.weight_decay,
    )

    classifier = Classifier(image_shape, NUM_CLS)
    ClassifierTrainer.load_weights(classifier, args.load_dir)

    # Save each run into a directory by its timestamp
    log_dir = setup_dirs(
        dirs=[args.save_dir],
        dirs_to_tstamp=[args.log_dir],
        config=vars(args),
        file_name=CONFIG,
    )[0]

    trainer = BiGANTrainer(
        generator,
        discriminator,
        encoder,
        classifier,
        train_dataset,
        test_dataset,
        noise_dims=args.noise_dims,
        gen_lr=args.gen_lr,
        disc_lr=args.disc_lr,
        enc_lr=args.enc_lr,
        gp_weight=args.gp_weight,
        cl_weight=args.cl_weight,
        log_dir=log_dir,
        save_dir=args.save_dir,
    )
    trainer.train(
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
        "--load-dir",
        type=str,
        default="./checkpoints/",
        help="directory where the trained classifier model is saved",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="./checkpoints/",
        help="directory where to save the GAN models",
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
