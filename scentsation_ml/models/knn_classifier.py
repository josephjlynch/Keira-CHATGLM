"""KNN classifier with StandardScaler."""

import logging
from typing import Dict, Optional

import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from .base import BaseModel

logger = logging.getLogger(__name__)


class KnnClassifier(BaseModel):
    """KNN + scaler."""

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """Grid search KNN hyperparameters."""
        param_grid = {
            "n_neighbors": self.config.get("n_neighbors", [3, 5]),
            "weights": self.config.get("weights", ["uniform", "distance"]),
            "metric": self.config.get("metric", ["euclidean"]),
        }
        cv_folds = self.config.get("cv_folds", 3)
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_train)
        knn = KNeighborsClassifier(n_jobs=self.config.get("n_jobs", -1))
        search = GridSearchCV(
            knn,
            param_grid,
            cv=cv_folds,
            scoring="f1_macro",
            n_jobs=1,
            verbose=self.config.get("verbose", 0),
            refit=True,
        )
        search.fit(X_scaled, y_train)
        self.model = search.best_estimator_
        self.is_trained = True
        y_pred = self.model.predict(X_scaled)
        logger.info("[KNN] best=%s CV F1=%.4f", search.best_params_, search.best_score_)
        return {
            "best_cv_f1": float(search.best_score_),
            "best_params": search.best_params_,
            "train_accuracy": float(np.mean(y_pred == y_train)),
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict labels."""
        if not self.is_trained:
            raise RuntimeError("KNN not trained")
        return self.model.predict(self.scaler.transform(X))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities."""
        return self.model.predict_proba(self.scaler.transform(X))

    def finetune(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """Refit on new data only."""
        k = min(5, len(X_train))
        self.scaler = StandardScaler()
        self.model = KNeighborsClassifier(n_neighbors=max(1, k), weights="distance")
        Xs = self.scaler.fit_transform(X_train)
        self.model.fit(Xs, y_train)
        self.is_trained = True
        return {"train_accuracy": float(np.mean(self.model.predict(Xs) == y_train))}
