#!/usr/bin/env python3
"""Training a classifier for FID."""
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser, Namespace

from typing_extensions import Final

from gan.data import NUM_CLS, get_mnist_dataset
from gan.models import Classifier
from gan.training import ClassifierTrainer
from gan.utils import setup_dirs

CONFIG: Final[str] = "config-cls.yaml"


def main(args: Namespace) -> None:
    """Run the main program.

    Arguments:
        args: The object containing the commandline arguments
    """
    train_dataset, test_dataset = get_mnist_dataset(
        args.mnist_path, args.batch_size
    )
    image_shape = train_dataset.element_spec[0].shape.as_list()[1:]

    model = Classifier(image_shape, NUM_CLS, weight_decay=args.weight_decay)

    # Save each run into a directory by its timestamp.
    log_dir = setup_dirs(
        dirs=[args.save_dir],
        dirs_to_tstamp=[args.log_dir],
        config=vars(args),
        file_name=CONFIG,
    )[0]

    trainer = ClassifierTrainer(model, lr=args.lr)
    trainer.train(
        train_dataset,
        test_dataset,
        args.epochs,
        log_dir=log_dir,
        record_eps=args.record_eps,
        save_dir=args.save_dir,
        save_steps=args.save_steps,
    )


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Training a classifier for FID",
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
        "--lr",
        type=float,
        default=1e-4,
        help="learning rate for the optimization",
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
        default=25,
        help="the maximum number of epochs for training the classifier",
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
        help="the frequency of saving the model (in steps)",
    )
    parser.add_argument(
        "--record-eps",
        type=int,
        default=5,
        help="the frequency of recording summaries (in epochs)",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="./logs/",
        help="directory where to write event logs",
    )
    main(parser.parse_args())
