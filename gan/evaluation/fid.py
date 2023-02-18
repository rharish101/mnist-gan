# SPDX-FileCopyrightText: 2020 Harish Rajagopal <harish.rajagopals@gmail.com>
#
# SPDX-License-Identifier: MIT

"""Class for calculating FID."""
from typing import Dict

import tensorflow as tf
from tensorflow import Tensor, Variable
from tensorflow.keras import Model

from ..utils import sqrtm


class RunningFID:
    """Class to calculate a running Frechet Inception Distance (FID).

    This is helpful when FID is to be calculated over more examples than can
    be stored in memory at once. This also provides a classmethod to directly
    compute the FID without keeping running metrics.

    Attributes:
        classifier: The pre-trained classifier model
    """

    def __init__(self, classifier: Model):
        """Initialize the running metrics and store the classifier.

        Args:
            classifier: The pre-trained classifier model
        """
        self.classifier = classifier

        # The initial values will be used for resetting the running metrics
        num_features = classifier.feature_extract.output_shape[-1]
        self._init_mean = tf.zeros([num_features], dtype=tf.float64)
        self._init_cov = tf.zeros(
            [num_features, num_features], dtype=tf.float64
        )
        self._init_num = tf.constant(0, dtype=tf.float64)

        # The running metrics are the mean and covariance for the Gaussian
        # distribution along with the total number of examples.
        self._mean: Dict[str, Variable] = {
            kind: Variable(self._init_mean, trainable=False)
            for kind in ("real", "gen")
        }
        self._cov: Dict[str, Variable] = {
            kind: Variable(self._init_cov, trainable=False)
            for kind in ("real", "gen")
        }
        self._total_num: Dict[str, Variable] = {
            kind: Variable(self._init_num, trainable=False)
            for kind in ("real", "gen")
        }

    @tf.function
    def reset(self) -> None:
        """Reset the running metrics."""
        for kind in "real", "gen":
            self._mean[kind].assign(self._init_mean)
            self._cov[kind].assign(self._init_cov)
            self._total_num[kind].assign(self._init_num)

    def _update_metrics(self, feat: Tensor, kind: str) -> None:
        """Update the values of the running metrics.

        The running metrics here are the mean, the covariance, and the total
        number of examples.

        The stable online mean and covariance are calculated using:
        https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Weighted_batched_version

        Args:
            feat: The batch of 1D features
            kind: The kind of metrics to use. Must be one of: ["real", "gen"]
        """
        # Notation for the following comments:
        # Shapes: B: batch_size, D: dimensions
        # ': transpose
        # @: matrix multiplication
        # sum(X, axis=n): reduce X by summing along the n^th dimension
        # Broadcasting rules are assumed throughout.

        old_mean = self._mean[kind]
        old_cov = self._cov[kind]
        old_num = self._total_num[kind]

        # Reduce any extra batch axes to a single batch axis
        feat = tf.reshape(feat, [-1, feat.shape[-1]])

        # Need this to be float64 to multiply with other float64 tensors
        num = tf.cast(feat.shape[0], tf.float64)
        new_num = old_num + num

        # D = sum(BxD, axis=B)
        feat_sum = tf.math.reduce_sum(feat, axis=0)
        new_mean = old_mean + (feat_sum - num * old_mean) / new_num

        # DxD = (BxD - 1xD)' @ (BxD - 1xD)
        cov_update = tf.transpose(feat - old_mean) @ (feat - new_mean)
        new_cov = old_cov + (cov_update - num * old_cov) / new_num

        self._mean[kind].assign(new_mean)
        self._cov[kind].assign(new_cov)
        self._total_num[kind].assign(new_num)

    @tf.function
    def update(self, real: Tensor, generated: Tensor) -> None:
        """Update the running metrics with the given info.

        Args:
            real: The batch of real images
            generated: The batch of generated images
        """
        features = {
            "real": self.classifier.feature_extract(real),
            "gen": self.classifier.feature_extract(generated),
        }

        for kind in features:
            # Cast to float64 for higher accuracy
            feat = tf.cast(features[kind], tf.float64)
            self._update_metrics(feat, kind)

    @tf.function
    def get_fid(self) -> Tensor:
        """Get the current value of the running FID.

        Returns:
            The FID between the two samples
        """
        dist_1 = tf.math.reduce_sum(
            (self._mean["real"] - self._mean["gen"]) ** 2
        )
        dist_2 = tf.linalg.trace(
            self._cov["real"]
            + self._cov["gen"]
            - 2 * sqrtm(self._cov["real"] @ self._cov["gen"])
        )
        dist = tf.math.sqrt(dist_1 + dist_2)

        # Cast back to the classifier's output dtype from float64
        dist = tf.cast(dist, self.classifier.dtype)

        return dist

    @classmethod
    def fid(cls, real: Tensor, generated: Tensor, classifier: Model) -> Tensor:
        """Return the Frechet Inception Distance (FID) for the two samples.

        This is a classmethod provided in the case where running metrics are
        not desired.

        Args:
            real: The batch of real images
            generated: The batch of generated images
            classifier: The pre-trained classifier model

        Returns:
            The FID between the two samples
        """
        fid_calc = cls(classifier)
        fid_calc.update(real, generated)
        return fid_calc.get_fid()
