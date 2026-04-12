"""RBF SVM with StandardScaler pipeline."""

import logging
from typing import Dict, Optional

import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .base import BaseModel

logger = logging.getLogger(__name__)


class SvmClassifier(BaseModel):
    """RBF-kernel SVM + scaler, tuned with GridSearchCV."""

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """Fit GridSearchCV on training data."""
        C_values = self.config.get("C_values", [1, 10])
        gamma_values = self.config.get("gamma_values", ["scale"])
        cv_folds = self.config.get("cv_folds", 3)
        n_jobs = self.config.get("n_jobs", -1)

        self.scaler = StandardScaler()
        pipeline = Pipeline(
            [
                ("scaler", self.scaler),
                (
                    "svm",
                    SVC(
                        kernel=self.config.get("kernel", "rbf"),
                        class_weight=self.config.get("class_weight", "balanced"),
                        probability=True,
                        random_state=42,
                    ),
                ),
            ]
        )
        param_grid = {"svm__C": C_values, "svm__gamma": gamma_values}
        search = GridSearchCV(
            pipeline,
            param_grid,
            cv=cv_folds,
            scoring="f1_macro",
            n_jobs=n_jobs,
            verbose=self.config.get("verbose", 0),
            refit=True,
        )
        search.fit(X_train, y_train)
        self.model = search.best_estimator_
        self.scaler = self.model.named_steps["scaler"]
        self.is_trained = True
        y_pred = self.model.predict(X_train)
        logger.info("[SVM] best=%s CV F1=%.4f", search.best_params_, search.best_score_)
        return {
            "best_cv_f1": float(search.best_score_),
            "best_params": search.best_params_,
            "train_accuracy": float(np.mean(y_pred == y_train)),
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict labels."""
        if not self.is_trained:
            raise RuntimeError("SVM not trained")
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities."""
        return self.model.predict_proba(X)
