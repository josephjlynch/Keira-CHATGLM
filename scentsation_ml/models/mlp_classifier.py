"""MLP classifier with StandardScaler."""

import logging
from typing import Dict, Optional

import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from .base import BaseModel

logger = logging.getLogger(__name__)


class MlpClassifier(BaseModel):
    """MLP with grid search and optional warm-start fine-tuning."""

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """Train with GridSearchCV."""
        hidden_options = self.config.get("hidden_layer_sizes", [(128, 64)])
        lr_options = self.config.get("learning_rate_init", [0.001])
        max_iter_options = self.config.get("max_iter", [300])
        cv_folds = self.config.get("cv_folds", 3)
        hidden_tuples = [tuple(h) for h in hidden_options]

        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_train)
        max_iter = max(300, max(max_iter_options))
        mlp = MLPClassifier(
            activation=self.config.get("activation", "relu"),
            solver=self.config.get("solver", "adam"),
            batch_size=self.config.get("batch_size", "auto"),
            early_stopping=self.config.get("early_stopping", True),
            validation_fraction=self.config.get("validation_fraction", 0.1),
            n_iter_no_change=self.config.get("n_iter_no_change", 10),
            random_state=self.config.get("random_state", 42),
            max_iter=max_iter,
            learning_rate="adaptive",
        )
        param_grid = {"hidden_layer_sizes": hidden_tuples, "learning_rate_init": lr_options}
        search = GridSearchCV(
            mlp,
            param_grid,
            cv=cv_folds,
            scoring="f1_macro",
            n_jobs=self.config.get("n_jobs", 1),
            verbose=self.config.get("verbose", 0),
            refit=True,
        )
        search.fit(X_scaled, y_train)
        self.model = search.best_estimator_
        self.is_trained = True
        y_pred = self.model.predict(X_scaled)
        logger.info("[MLP] best=%s CV F1=%.4f", search.best_params_, search.best_score_)
        return {
            "best_cv_f1": float(search.best_score_),
            "best_params": search.best_params_,
            "train_accuracy": float(np.mean(y_pred == y_train)),
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict on scaled features."""
        if not self.is_trained:
            raise RuntimeError("MLP not trained")
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
        """Continue training with warm_start when possible."""
        if not self.is_trained:
            return self.train(X_train, y_train, X_val, y_val)
        X_scaled = self.scaler.transform(X_train)
        self.model.warm_start = True
        self.model.learning_rate_init = self.config.get("finetune_lr", 0.0001)
        extra = self.config.get("finetune_epochs", 100)
        self.model.max_iter = int(getattr(self.model, "n_iter_", 0)) + int(extra)
        self.model.early_stopping = False
        self.model.fit(X_scaled, y_train)
        self.is_trained = True
        return {
            "train_accuracy": float(np.mean(self.model.predict(X_scaled) == y_train)),
            "total_epochs": int(self.model.n_iter_),
        }
