"""Class for calculating FID."""
from typing import Tuple

import tensorflow as tf
from tensorflow import Tensor
from tensorflow.keras import Model


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
        # The running metrics are the sufficient statistics for the Gaussian
        # distribution along with the number of examples.
        self._feat_sum = {"real": 0, "gen": 0}
        self._feat_sum_outer = {"real": 0, "gen": 0}
        self._num_eg = {"real": 0, "gen": 0}

    @staticmethod
    @tf.function
    def _get_running_params_gauss(
        feat: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Get the running parameters for the given samples.

        The samples are assumed to be IID Gaussian. Thus, the running
        parameters required to estimate multivariate Gaussians are the
        sufficient statistics and the number of examples.

        Args:
            feat: The batch of 1D features, as a 2D tensor

        Returns:
            The sum (the first sufficient statistic)
            The sum of outer products (the second sufficient statistic)
            The total number of examples
        """
        # Notation for the following comments:
        # Shapes: B1, B2, ...: batch_size dimensions, D: dimensions
        # *: scalar multiplication
        # @: matrix multiplication along last two dimensions
        # sum(X, dim=[n, ...]): reduce X by summing along the n, ... dimensions
        # Broadcasting rules are assumed throughout.

        # D = sum(B1xB2x...xD, dim=[B1, B2, ...])
        feat_sum = tf.math.reduce_sum(feat, range(len(feat.shape) - 1))

        # B1xB2x...xDxD = B1xB2x...xDx1 @ B1xB2x...x1xD
        outer_prod = tf.expand_dims(feat, -1) @ tf.expand_dims(feat, -2)
        # DxD = sum(B1xB2x...xDxD, dim=[B1, B2, ...])
        feat_sum_outer = tf.math.reduce_sum(
            outer_prod, range(len(outer_prod.shape) - 2)
        )

        # scalar = B1 * B2 * ...
        num_eg = tf.math.reduce_prod(feat.shape[:-1])

        return feat_sum, feat_sum_outer, num_eg

    def _get_mean_cov(self, kind: str) -> Tuple[Tensor, Tensor]:
        """Get the mean and covariance from the running metrics.

        Args:
            kind: The kind of metrics to use. Must be one of: ["real", "gen"]

        Returns:
            The mean for the requested running metrics
            The covariance for the requested running metrics
        """
        # Notation for the following comments:
        # Shapes: D: dimensions
        # @: matrix multiplication along last two dimensions
        # +: element-wise addition along all dimensions
        # -: element-wise subtraction along all dimensions
        # /: element-wise division along all dimensions
        # Broadcasting rules are assumed throughout.

        # Epsilon is the fuzz factor to prevent divide-by-zero
        # scalar = scalar + scalar
        normalizer = (
            tf.cast(self._num_eg[kind], tf.float32)
            + tf.keras.backend.epsilon()
        )

        # D = D / scalar
        mean = self._feat_sum[kind] / normalizer

        # Calculate covariance as: Cov[x] = E[xx'] - (E[x])^2
        # DxD = Dx1 @ 1xD
        mean_outer = tf.expand_dims(mean, -1) @ tf.expand_dims(mean, -2)
        # DxD = DxD / scalar
        feat_sum_outer_normalized = self._feat_sum_outer[kind] / normalizer
        # DxD = DxD - DxD
        cov = feat_sum_outer_normalized - mean_outer

        return mean, cov

    def update(self, real: Tensor, generated: Tensor) -> None:
        """Update the running metrics with the given info.

        Args:
            real: The batch of real images
            generated: The batch of generated images
        """
        real_feat = self.classifier.feature_extract(real)
        gen_feat = self.classifier.feature_extract(generated)

        updates = {
            "real": self._get_running_params_gauss(real_feat),
            "gen": self._get_running_params_gauss(gen_feat),
        }

        for kind in "real", "gen":
            self._feat_sum[kind] += updates[kind][0]
            self._feat_sum_outer[kind] += updates[kind][1]
            self._num_eg[kind] += updates[kind][2]

    @tf.function
    def get_fid(self) -> Tensor:
        """Get the current value of the running FID.

        Returns:
            The FID between the two samples
        """
        mean = {}
        cov = {}

        for kind in "real", "gen":
            mean[kind], cov[kind] = self._get_mean_cov(kind)

        dist_1 = tf.norm(mean["real"] - mean["gen"])
        dist_2 = tf.linalg.trace(
            cov["real"]
            + cov["gen"]
            - 2 * tf.linalg.sqrtm(cov["real"] @ cov["gen"])
        )
        dist = dist_1 + dist_2

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
