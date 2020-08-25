#!/usr/bin/env python3
"""Train a classifier for FID."""
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser, Namespace

from tensorflow.distribute import MirroredStrategy
from typing_extensions import Final

from gan.data import IMG_SHAPE, NUM_CLS, get_dataset
from gan.models import Classifier
from gan.training import ClassifierTrainer
from gan.utils import setup_dirs

CONFIG: Final = "config-cls.yaml"


def main(args: Namespace) -> None:
    """Run the main program.

    Arguments:
        args: The object containing the commandline arguments
    """
    strategy = MirroredStrategy()

    train_dataset, test_dataset = get_dataset(args.data_path, args.batch_size)

    with strategy.scope():
        model = Classifier(IMG_SHAPE, NUM_CLS, weight_decay=args.weight_decay)

    # Save each run into a directory by its timestamp.
    log_dir = setup_dirs(
        dirs=[args.save_dir],
        dirs_to_tstamp=[args.log_dir],
        config=vars(args),
        file_name=CONFIG,
    )[0]

    trainer = ClassifierTrainer(model, strategy, lr=args.lr)
    trainer.train(
        train_dataset,
        test_dataset,
        epochs=args.epochs,
        log_dir=log_dir,
        record_eps=args.record_eps,
        save_dir=args.save_dir,
        save_steps=args.save_steps,
        log_graph=args.log_graph,
    )


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Train a classifier for FID",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="./datasets/MNIST/",
        help="path to the dataset",
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
        "--log-graph",
        action="store_true",
        help="whether to log the graph of the model",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="./logs/classifier",
        help="directory where to write event logs",
    )
    main(parser.parse_args())
