"""Class for calculating FID."""
from typing import Dict, Tuple

import tensorflow as tf
from tensorflow import Tensor
from tensorflow.keras import Model

from ..utils import sqrtm


class RunningFID:
    """Class to calculate a running Frechet Inception Distance (FID).

    This is helpful when FID is to be calculated over more examples than can
    be stored in memory at once. This also provides a classmethod to directly
    compute the FID without keeping running metrics.
    """

    def __init__(self, classifier: Model) -> None:
        """Initialize the running metrics and store the classifier.

        Args:
            classifier: The pre-trained classifier model
        """
        self.classifier = classifier
        self.reset()  # used to initialize the running metrics

    def reset(self) -> None:
        """Reset the running metrics."""
        # The running metrics are the mean and covariance for the Gaussian
        # distribution along with the total number of examples.
        self._mean: Dict[str, Tensor] = {
            "real": tf.zeros([], dtype=tf.float64),
            "gen": tf.zeros([], dtype=tf.float64),
        }
        self._cov: Dict[str, Tensor] = {
            "real": tf.zeros([], dtype=tf.float64),
            "gen": tf.zeros([], dtype=tf.float64),
        }
        self._total_num: Dict[str, Tensor] = {
            "real": tf.zeros([], dtype=tf.float64),
            "gen": tf.zeros([], dtype=tf.float64),
        }

    @tf.function
    def _get_updated_metrics(
        self, feat: Tensor, kind: str
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Get the updated values of the running metrics.

        The running metrics here are the mean, the covariance, and the total
        number of examples.

        The stable online mean and covariance are calculated using:
        https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Weighted_batched_version

        Args:
            feat: The batch of 1D features
            kind: The kind of metrics to use. Must be one of: ["real", "gen"]

        Returns:
            The updated running mean
            The updated running covariance
            The updated total number of examples
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

        return new_mean, new_cov, new_num

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
            updates = self._get_updated_metrics(feat, kind)
            self._mean[kind] = updates[0]
            self._cov[kind] = updates[1]
            self._total_num[kind] = updates[2]

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
