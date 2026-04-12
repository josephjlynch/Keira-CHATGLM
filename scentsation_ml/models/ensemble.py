"""Optional soft ensemble — not used by default in ``train.py`` (see script)."""

import logging
from typing import Any, Dict

import numpy as np

from .base import BaseModel

logger = logging.getLogger(__name__)


def average_proba(models: Dict[str, BaseModel], X: np.ndarray) -> np.ndarray:
    """Average ``predict_proba`` from multiple trained models."""
    stacks = []
    for m in models.values():
        stacks.append(m.predict_proba(X))
    return np.mean(np.stack(stacks, axis=0), axis=0)


class EnsembleClassifier(BaseModel):
    """Placeholder for future soft-voting wrapper."""

    def train(self, X_train, y_train, X_val=None, y_val=None) -> Dict[str, float]:
        """Training is handled externally."""
        raise NotImplementedError("Use average_proba() or train base models directly.")
