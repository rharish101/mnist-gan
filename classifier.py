#!/usr/bin/env python3
"""Train a classifier for FID."""
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser, Namespace
from pathlib import Path
from typing import Final

from tensorflow.distribute import MirroredStrategy
from tensorflow.keras.mixed_precision import set_global_policy

from gan.data import get_dataset
from gan.models import Classifier
from gan.training import ClassifierTrainer
from gan.utils import load_config, setup_dirs

CONFIG: Final = "config-cls.yaml"


def main(args: Namespace) -> None:
    """Run the main program.

    Arguments:
        args: The object containing the commandline arguments
    """
    config = load_config(args.config)

    strategy = MirroredStrategy()
    if config.mixed_precision:
        set_global_policy("mixed_float16")

    train_dataset, test_dataset = get_dataset(
        args.data_path, config.cls_batch_size
    )

    with strategy.scope():
        model = Classifier(config)

    # Save each run into a directory by its timestamp.
    log_dir = setup_dirs(
        dirs=[args.save_dir],
        dirs_to_tstamp=[args.log_dir],
        config=config,
        file_name=CONFIG,
    )[0]

    trainer = ClassifierTrainer(model, strategy, config=config)
    trainer.train(
        train_dataset,
        test_dataset,
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
        type=Path,
        default="./datasets/MNIST/",
        help="path to the dataset",
    )
    parser.add_argument(
        "-c",
        "--config",
        type=Path,
        help="Path to a YAML config containing hyper-parameter values",
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
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
        type=Path,
        default="./logs/classifier",
        help="directory where to write event logs",
    )
    main(parser.parse_args())
