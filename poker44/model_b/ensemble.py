"""Weighted tree ensemble for Model B.

Architecture adapted from the current top Poker44 miner (uid 99,
github.com/tao-miner/hot4-poker-3, MIT) -- a weighted soft-vote of ExtraTrees +
RandomForest + HistGradientBoosting. That miner applies it to action-pattern
features; here it runs on Model B's *geometry / interaction* feature family, so
it lifts accuracy without importing the pattern signal (which would re-correlate
Model B with the primary model and with that miner).
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)


def _make_members(seed: int):
    """The three base learners + their blend weights (top-miner recipe)."""
    return [
        (
            "extratrees",
            0.45,
            ExtraTreesClassifier(
                n_estimators=700,
                max_depth=9,
                class_weight="balanced_subsample",
                n_jobs=-1,
                random_state=seed + 0,
            ),
        ),
        (
            "randomforest",
            0.25,
            RandomForestClassifier(
                n_estimators=700,
                max_depth=9,
                class_weight="balanced_subsample",
                n_jobs=-1,
                random_state=seed + 5,
            ),
        ),
        (
            "histgb",
            0.30,
            HistGradientBoostingClassifier(
                loss="log_loss",
                learning_rate=0.03,
                max_iter=700,
                max_depth=9,
                min_samples_leaf=2,
                l2_regularization=1.0,
                random_state=seed + 11,
            ),
        ),
    ]


class WeightedTreeEnsemble:
    """Weighted soft-vote of ExtraTrees + RandomForest + HistGradientBoosting."""

    def __init__(self, seed: int = 7):
        self.seed = seed
        members = _make_members(seed)
        self.names = [m[0] for m in members]
        self.weights = np.asarray([m[1] for m in members], dtype=float)
        self.weights /= self.weights.sum()
        self.models = [m[2] for m in members]

    def fit(self, X, y):
        for model in self.models:
            model.fit(X, y)
        return self

    def predict_proba(self, X):
        proba = np.zeros(X.shape[0], dtype=float)
        for w, model in zip(self.weights, self.models):
            proba += w * model.predict_proba(X)[:, 1]
        proba = np.clip(proba, 0.0, 1.0)
        return np.column_stack([1.0 - proba, proba])
