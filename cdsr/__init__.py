"""A lightweight, model-agnostic framework for CDSR experiments."""

from cdsr.data.dataset import CanonicalCDSRDataset, build_canonical_corpus
from cdsr.models.base import CDSRModel

__all__ = ["CDSRModel", "CanonicalCDSRDataset", "build_canonical_corpus"]
__version__ = "0.1.0"
