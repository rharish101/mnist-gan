"""Architectures for all models."""
from .classifier import Classifier
from .gan import get_critic, get_generator

__all__ = ["Classifier", "get_critic", "get_generator"]
