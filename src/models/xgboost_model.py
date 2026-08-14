import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV


class XGBoostModel:
    def __init__(self, scale_pos_weight: float = 3.0):
        self.scale_pos_weight = scale_pos_weight
        self.model = None
        self.feature_names: list = []

    def train(self, X, y) -> None:
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [3, 5],
            'learning_rate': [0.05, 0.1],
        }
        base = XGBClassifier(
            scale_pos_weight=self.scale_pos_weight,
            eval_metric='logloss',
            random_state=42,
        )
        gs = GridSearchCV(base, param_grid, cv=3, scoring='roc_auc', n_jobs=-1)
        gs.fit(X, y)
        self.model = gs.best_estimator_
        self.feature_names = list(X.columns) if hasattr(X, 'columns') else []

    def predict_proba(self, X) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]

    def predict(self, X, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)
