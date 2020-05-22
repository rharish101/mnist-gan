"""Classes for evaluation."""
from .fid import RunningFID
from .gan import BiGANEvaluator

__all__ = ["RunningFID", "BiGANEvaluator"]
