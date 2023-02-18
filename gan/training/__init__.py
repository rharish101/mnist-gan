# SPDX-FileCopyrightText: 2020 Harish Rajagopal <harish.rajagopals@gmail.com>
#
# SPDX-License-Identifier: MIT

"""Classes for training."""
from .classifier import ClassifierTrainer
from .gan import GANTrainer

__all__ = ["ClassifierTrainer", "GANTrainer"]
