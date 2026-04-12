"""Base model interface for Scentsation classifiers."""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import joblib
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

logger = logging.getLogger(__name__)


class BaseModel(ABC):
    """Abstract base class for all Scentsation classifiers."""

    def __init__(self, config: Dict[str, Any]):
        """Store hyperparameters and initialise placeholders."""
        self.config = config
        self.model = None
        self.scaler = None
        self.is_trained = False
        self.model_name = self.__class__.__name__

    @abstractmethod
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """Train the model."""
        ...

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class indices or labels."""
        ...

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Class probabilities if supported."""
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)
        preds = self.predict(X)
        n_classes = len(self.config.get("class_names", [])) or 4
        proba = np.zeros((len(preds), n_classes))
        for i, p in enumerate(preds):
            idx = int(p) if np.issubdtype(type(p), np.integer) or isinstance(p, (int, np.int64)) else 0
            if 0 <= idx < n_classes:
                proba[i, idx] = 1.0
        return proba

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """Return metrics on a held-out set."""
        y_pred = self.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
        precision = precision_score(y_test, y_pred, average="macro", zero_division=0)
        recall = recall_score(y_test, y_pred, average="macro", zero_division=0)
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        return {
            "accuracy": float(acc),
            "f1_macro": float(f1),
            "precision_macro": float(precision),
            "recall_macro": float(recall),
            "confusion_matrix": cm,
            "classification_report": report,
            "y_pred": y_pred,
        }

    def finetune(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """Default: retrain from scratch."""
        return self.train(X_train, y_train, X_val, y_val)

    def save(self, path: str) -> None:
        """Persist model bundle."""
        if not self.is_trained:
            logger.warning("[%s] not trained — skip save", self.model_name)
            return
        joblib.dump({"model": self.model, "scaler": self.scaler, "config": self.config}, path)
        logger.info("[%s] saved to %s", self.model_name, path)

    def load(self, path: str) -> None:
        """Load model bundle."""
        data = joblib.load(path)
        self.model = data["model"]
        self.scaler = data.get("scaler")
        self.config = data.get("config", self.config)
        self.is_trained = True

    def get_feature_importance(self, X_test: np.ndarray, y_test: np.ndarray):
        """Permutation importance when possible."""
        try:
            from sklearn.inspection import permutation_importance

            est = self.model
            result = permutation_importance(
                est, X_test, y_test, n_repeats=5, random_state=42, n_jobs=1
            )
            return result.importances_mean
        except Exception as e:
            logger.warning("Feature importance failed: %s", e)
            return None
