#!/usr/bin/env python3
"""Train a GAN."""
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser, Namespace
from typing import Final

from tensorflow.distribute import MirroredStrategy
from tensorflow.keras.mixed_precision import set_global_policy

from gan.data import IMG_SHAPE, NUM_CLS, get_dataset
from gan.models import Classifier, get_critic, get_generator
from gan.training import ClassifierTrainer, GANTrainer
from gan.utils import setup_dirs

CONFIG: Final = "config-gan.yaml"


def main(args: Namespace) -> None:
    """Run the main program.

    Arguments:
        args: The object containing the commandline arguments
    """
    strategy = MirroredStrategy()
    if args.mixed_precision:
        set_global_policy("mixed_float16")

    train_dataset, test_dataset = get_dataset(args.data_path, args.batch_size)

    with strategy.scope():
        generator = get_generator(
            args.noise_dims,
            NUM_CLS,
            IMG_SHAPE[-1],
            weight_decay=args.weight_decay,
        )
        critic = get_critic(IMG_SHAPE, NUM_CLS, weight_decay=args.weight_decay)

        classifier = Classifier(IMG_SHAPE, NUM_CLS)
        ClassifierTrainer.load_weights(classifier, args.load_dir)

    # Save each run into a directory by its timestamp
    log_dir = setup_dirs(
        dirs=[args.save_dir],
        dirs_to_tstamp=[args.log_dir],
        config=vars(args),
        file_name=CONFIG,
    )[0]

    trainer = GANTrainer(
        generator,
        critic,
        classifier,
        strategy,
        train_dataset,
        test_dataset,
        batch_size=args.batch_size,
        crit_steps=args.crit_steps,
        noise_dims=args.noise_dims,
        gen_lr=args.gen_lr,
        crit_lr=args.crit_lr,
        decay_rate=args.decay_rate,
        decay_steps=args.decay_steps,
        gp_weight=args.gp_weight,
        log_dir=log_dir,
        save_dir=args.save_dir,
    )
    trainer.train(
        epochs=args.epochs,
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
        "--crit-lr",
        type=float,
        default=1e-4,
        help="learning rate for critic optimization",
    )
    parser.add_argument(
        "--decay-rate",
        type=float,
        default=0.8,
        help="the rate of exponential learning rate decay",
    )
    parser.add_argument(
        "--decay-steps",
        type=int,
        default=3000,
        help="the base steps for exponential learning rate decay",
    )
    parser.add_argument(
        "--gp-weight",
        type=float,
        default=10.0,
        help="weights for the critic's gradient penalty",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=2.5e-5,
        help="L2 weight decay rate",
    )
    parser.add_argument(
        "--crit-steps",
        type=int,
        default=1,
        help="the number of critic steps per generator step",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=40,
        help="the number of epochs for training the GAN",
    )
    parser.add_argument(
        "--mixed-precision",
        action="store_true",
        help="train with mixed-precision for higher performance",
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
        type=str,
        default="./logs/gan",
        help="directory where to write event logs",
    )
    main(parser.parse_args())
