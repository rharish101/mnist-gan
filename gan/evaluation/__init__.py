"""Classes for evaluation."""
from .fid import RunningFID
from .gan import GANEvaluator

__all__ = ["RunningFID", "GANEvaluator"]
