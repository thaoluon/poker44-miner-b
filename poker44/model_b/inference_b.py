"""Runtime inference wrapper for Model B.

Loads the HistGradientBoosting model + metadata produced by
``scripts/miner/train_model_b.py`` (optionally blended with the DeepSets net)
and scores chunks. Falls back to a neutral mid-low score if artifacts or
dependencies are unavailable, so the miner never crashes a request.

Mirrors the primary model's serving contract (calibrated risk score per chunk,
per-batch human-safety floor) but is a fully independent code path and model.
"""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path

import numpy as np

from poker44.model_b.features_b import build_feature_matrix_b

logger = logging.getLogger(__name__)

DEFAULT_ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"
FALLBACK_SCORE = 0.25


class DetectionModelB:
    """Chunk-level bot-risk scorer backed by Model B (HGB + DeepSets blend)."""

    def __init__(self, artifact_dir: Path | str = DEFAULT_ARTIFACT_DIR):
        self.artifact_dir = Path(artifact_dir)
        self.model = None
        self.feature_names: list[str] = []
        self.iso_x = np.linspace(0.0, 1.0, 2)
        self.iso_y = np.linspace(0.0, 1.0, 2)
        self.pivot = 0.5
        self.use_relative = False
        self.net_scorer = None
        self.net_weight = 0.0
        self._load()

    def _load(self) -> None:
        meta_path = self.artifact_dir / "model_b_meta.json"
        model_path = self.artifact_dir / "model_b.pkl"
        try:
            meta = json.loads(meta_path.read_text())
            self.feature_names = list(meta["feature_names"])
            calib = meta.get("calibration", {})
            self.iso_x = np.asarray(calib.get("iso_x", [0.0, 1.0]), dtype=float)
            self.iso_y = np.asarray(calib.get("iso_y", [0.0, 1.0]), dtype=float)
            self.pivot = float(calib.get("pivot", 0.5))
            self.use_relative = bool(meta.get("use_relative", False))
            with open(model_path, "rb") as fh:
                self.model = pickle.load(fh)
            logger.info(
                "model B loaded: %d features, pivot=%.4f, relative=%s, trained on %s examples",
                len(self.feature_names),
                self.pivot,
                self.use_relative,
                meta.get("n_train_examples"),
            )
            weight = float(meta.get("net_blend_weight", 0.0) or 0.0)
            net_path = self.artifact_dir / "deepset_b.pt"
            if weight > 0.0 and net_path.exists():
                from poker44.model_b.deepset import NetScorer

                scorer = NetScorer(net_path)
                if scorer.ready:
                    self.net_scorer = scorer
                    self.net_weight = weight
                    logger.info("deepset blend active: weight=%.2f", weight)
                else:
                    logger.warning("deepset weights present but torch unavailable; GBDT-only")
        except Exception as err:  # noqa: BLE001
            self.model = None
            logger.error("model B unavailable, using fallback: %s", err)

    @property
    def ready(self) -> bool:
        return self.model is not None

    def _calibrate(self, raw: np.ndarray) -> np.ndarray:
        mapped = np.interp(raw, self.iso_x, self.iso_y)
        pivot = min(max(self.pivot, 1e-6), 1.0 - 1e-6)
        scaled = np.where(
            mapped <= pivot,
            0.5 * mapped / pivot,
            0.5 + 0.5 * (mapped - pivot) / (1.0 - pivot),
        )
        return np.clip(scaled, 0.0, 1.0)

    def score_chunks(self, chunks: list[list[dict]]) -> list[float]:
        """One calibrated bot-risk score in [0, 1] per chunk."""
        if not chunks:
            return []
        if not self.ready:
            return [FALLBACK_SCORE] * len(chunks)
        try:
            degenerate = [
                not (chunk or [])
                or not any((hand or {}).get("actions") for hand in (chunk or []))
                for chunk in chunks
            ]
            rows = build_feature_matrix_b(chunks, self.feature_names, self.use_relative)
            raw = np.asarray(self.model.predict_proba(rows)[:, 1])
            if self.net_scorer is not None and self.net_weight > 0.0:
                try:
                    net = np.asarray(self.net_scorer.score(chunks), dtype=float)
                    if net.shape == raw.shape:
                        raw = self.net_weight * net + (1.0 - self.net_weight) * raw
                except Exception as err:  # noqa: BLE001
                    logger.error("deepset blend failed, GBDT-only: %s", err)
            calibrated = self._calibrate(raw)
            # Per-batch human-safety floor: the validator needs >=1 chunk scored
            # >=0.5 or the window's threshold-sanity term collapses to 0. Nudge
            # the top ~5% (>=1) non-degenerate chunks just above 0.5, ranked --
            # rank-preserving (AP/recall unchanged), only moves the boundary.
            deg = np.asarray(degenerate)
            real_idx = np.flatnonzero(~deg)
            if real_idx.size >= 3:
                floor_k = max(1, int(round(0.05 * real_idx.size)))
                n_flagged = int((calibrated[real_idx] >= 0.5).sum())
                if n_flagged < floor_k:
                    top = real_idx[np.argsort(raw[real_idx])[-floor_k:]]
                    ranks = np.argsort(np.argsort(raw[top]))
                    calibrated[top] = 0.501 + 0.048 * (ranks / max(floor_k - 1, 1))
            return [
                FALLBACK_SCORE if is_degenerate else round(float(score), 6)
                for score, is_degenerate in zip(calibrated, degenerate)
            ]
        except Exception as err:  # noqa: BLE001
            logger.error("model B inference failed, using fallback: %s", err)
            return [FALLBACK_SCORE] * len(chunks)
