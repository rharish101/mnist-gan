# SPDX-FileCopyrightText: 2020 Harish Rajagopal <harish.rajagopals@gmail.com>
#
# SPDX-License-Identifier: MIT

"""Classes for evaluation."""
from .fid import RunningFID
from .gan import GANEvaluator

__all__ = ["RunningFID", "GANEvaluator"]
