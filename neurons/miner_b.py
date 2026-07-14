"""Poker44 miner backed by Model B (independent geometry/interaction model).

Self-contained: depends only on ``poker44.model_b`` and the subnet base
package -- no dependency on the primary model's ``poker44.model``. Intended to
run on its own hotkey from this repository checkout, so ``repo_url`` /
``repo_commit`` and the manifest digest are genuinely distinct from the primary
miner.

All manifest fields can be overridden per deployment via the ``POKER44_MODEL_*``
environment variables (see poker44/utils/model_manifest.py).
"""

# from __future__ import annotations

import time
from pathlib import Path
from typing import Tuple

import bittensor as bt

from poker44.base.miner import BaseMinerNeuron
from poker44.model_b.inference_b import DetectionModelB
from poker44.model_b.live_log import LiveChunkLoggerB
from poker44.utils.model_manifest import (
    build_local_model_manifest,
    evaluate_manifest_compliance,
    manifest_digest,
)
from poker44.validator.synapse import DetectionSynapse

FALLBACK_SCORE = 0.25


class MinerB(BaseMinerNeuron):
    """Model-B miner: commitment & interaction-geometry detector."""

    def __init__(self, config=None):
        super(MinerB, self).__init__(config=config)
        bt.logging.info("🃏 Poker44 Model-B (geometry) Miner started")
        repo_root = Path(__file__).resolve().parents[1]
        self.detection_model = DetectionModelB()
        self.live_logger = LiveChunkLoggerB()
        bt.logging.info(
            f"Model B ready: {self.detection_model.ready} "
            f"(pivot={self.detection_model.pivot:.4f}, "
            f"features={len(self.detection_model.feature_names)}, "
            f"deepset_weight={self.detection_model.net_weight})"
        )
        self.model_manifest = build_local_model_manifest(
            repo_root=repo_root,
            implementation_files=[
                Path(__file__).resolve(),
                repo_root / "poker44" / "model_b" / "features_b.py",
                repo_root / "poker44" / "model_b" / "inference_b.py",
                repo_root / "poker44" / "model_b" / "deepset.py",
            ],
            defaults={
                "model_name": "poker44-geometry-b",
                "model_version": "1.0.0",
                "framework": "sklearn-histgbdt+torch",
                "license": "MIT",
                "repo_url": "https://github.com/thaoluon/poker44-miner-b",
                "notes": (
                    "HistGradientBoosting over commitment / stack-geometry / "
                    "interaction-response / showdown features, blended with an "
                    "attention-pooled DeepSets net; isotonic-calibrated so the 0.5 "
                    "threshold operates at a low human false-positive rate. "
                    "Independent feature family and learner from poker44-lgbm-behavioral."
                ),
                "open_source": True,
                "inference_mode": "remote",
                "training_data_statement": (
                    "Trained exclusively on the public Poker44 training benchmark "
                    "(api.poker44.net/api/v1/benchmark), all release dates."
                ),
                "training_data_sources": [
                    "https://api.poker44.net/api/v1/benchmark"
                ],
                "private_data_attestation": (
                    "This miner does not train on validator-only evaluation data."
                ),
            },
        )
        self.manifest_compliance = evaluate_manifest_compliance(self.model_manifest)
        self.manifest_digest = manifest_digest(self.model_manifest)
        self._log_manifest_startup(repo_root)
        bt.logging.info(f"Axon created: {self.axon}")

    def _log_manifest_startup(self, repo_root: Path) -> None:
        bt.logging.info("Open-sourced miner manifest standard active for this miner.")
        bt.logging.info(
            f"Miner transparency status: {self.manifest_compliance['status']} "
            f"(missing_fields={self.manifest_compliance['missing_fields']})"
        )
        bt.logging.info(
            f"Manifest summary | model={self.model_manifest.get('model_name', '')} "
            f"version={self.model_manifest.get('model_version', '')} "
            f"repo={self.model_manifest.get('repo_url', '')} "
            f"commit={self.model_manifest.get('repo_commit', '')} "
            f"open_source={self.model_manifest.get('open_source')}"
        )
        bt.logging.info(
            f"Manifest digest={self.manifest_digest} "
            f"inference_mode={self.model_manifest.get('inference_mode', '')}"
        )

    async def forward(self, synapse: DetectionSynapse) -> DetectionSynapse:
        """Assign one Model-B bot-risk score per chunk."""
        chunks = synapse.chunks or []
        if self.detection_model.ready:
            scores = self.detection_model.score_chunks(chunks)
        else:
            scores = [FALLBACK_SCORE] * len(chunks)
        synapse.risk_scores = scores
        synapse.predictions = [s >= 0.5 for s in scores]
        synapse.model_manifest = dict(self.model_manifest)
        self.live_logger.log(chunks, scores)
        bt.logging.info(f"MinerB Predictions: {synapse.predictions}")
        bt.logging.info(
            f"Scored {len(chunks)} chunks "
            f"({'model B' if self.detection_model.ready else 'fallback'})."
        )
        return synapse

    async def blacklist(self, synapse: DetectionSynapse) -> Tuple[bool, str]:
        return self.common_blacklist(synapse)

    async def priority(self, synapse: DetectionSynapse) -> float:
        return self.caller_priority(synapse)


if __name__ == "__main__":
    with MinerB() as miner:
        bt.logging.info("Model-B miner running...")
        while True:
            bt.logging.info(f"Miner UID: {miner.uid} | Incentive: {miner.metagraph.I[miner.uid]}")
            time.sleep(5 * 60)
