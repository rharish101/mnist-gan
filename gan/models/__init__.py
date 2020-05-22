"""Architectures for all models."""
from .classifier import Classifier
from .gan import get_discriminator, get_encoder, get_generator

__all__ = ["Classifier", "get_discriminator", "get_encoder", "get_generator"]
