"""Rolling log of real validator queries for Model B (self-contained).

Records the (chunk, score) pairs Model B actually receives so the live
benchmark->serve distribution shift can be measured and later adapted to.
Best-effort: must never break scoring. Kept inside ``poker44.model_b`` so the
Model B miner has no dependency on the primary model's package.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_DIR = Path(__file__).resolve().parents[2] / "data" / "live_chunks_b"
MAX_CHUNKS = 4000
TRIM_EVERY = 500


class LiveChunkLoggerB:
    """Append received (chunk, score) pairs to a rolling JSONL corpus."""

    def __init__(self, log_dir: Path | str = DEFAULT_DIR, max_chunks: int = MAX_CHUNKS):
        self.path = Path(log_dir) / "live_chunks.jsonl"
        self.max_chunks = max_chunks
        self._since_trim = 0
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.enabled = True
        except Exception:  # noqa: BLE001
            self.enabled = False

    def log(self, chunks, scores) -> None:
        if not self.enabled or not chunks:
            return
        try:
            now = round(time.time(), 2)
            with self.path.open("a") as fh:
                for chunk, score in zip(chunks, scores):
                    hands = chunk or []
                    fh.write(
                        json.dumps(
                            {
                                "ts": now,
                                "score": round(float(score), 6),
                                "n_hands": len(hands),
                                "hands": hands,
                            },
                            separators=(",", ":"),
                        )
                        + "\n"
                    )
            self._since_trim += len(chunks)
            if self._since_trim >= TRIM_EVERY:
                self._since_trim = 0
                self._trim()
        except Exception as err:  # noqa: BLE001
            logger.debug("live log failed (ignored): %s", err)

    def _trim(self) -> None:
        try:
            with self.path.open("r") as fh:
                lines = fh.readlines()
            if len(lines) <= self.max_chunks:
                return
            keep = lines[-self.max_chunks:]
            tmp = self.path.with_suffix(".tmp")
            with tmp.open("w") as fh:
                fh.writelines(keep)
            os.replace(tmp, self.path)
        except Exception as err:  # noqa: BLE001
            logger.debug("live log trim failed (ignored): %s", err)
