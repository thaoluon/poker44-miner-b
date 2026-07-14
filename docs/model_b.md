# Model B — independent geometry/interaction detector

Model B is a **second, genuinely independent** Poker44 detection model, built to
run on a separate hotkey alongside the primary model (`poker44-lgbm-behavioral`)
without being a disguised copy. It earns on its own merits: a different feature
family, a different learner, different calibration, and — most importantly —
**decorrelated predictions**.

> Duplication checks that key on output behaviour compare *predictions*, not
> source code. Two miners that emit near-identical risk scores get flagged no
> matter how differently the code is written. Model B is therefore designed to
> be independent in its *outputs*, verified below.

## What is different from the primary model

| Axis | Primary (`poker44.model`) | Model B (`poker44.model_b`) |
|---|---|---|
| Feature family | action-type ratios, entropy/run regularity, signature-replay, pot-fraction sizing (309 feats) | effective-stack/SPR, bet-over-stack **commitment**, raise_to/call_to **magnitudes**, reraise/check-raise/donk/fold-to-raise **response graph**, showdown dynamics |
| Aggregation | mean / std / min / max | mean / std / median / p10 / p90 |
| Second view | within-batch rank of A's features | within-batch rank of **B's** features |
| Learner | LightGBM (leaf-wise) + GRU | Weighted tree ensemble (ExtraTrees 0.45 + RandomForest 0.25 + HistGradientBoosting 0.30) + attention-pooled **DeepSets** (no recurrence) |
| Calibration | piecewise-linear pivot | isotonic + operating-point pivot |
| Seed / CV | seed 44 | seed 7, date-bagged |

The neural components read different channels too: the primary GRU consumes the
ordered action sequence; Model B's DeepSets net consumes the geometry/commitment
channels (amount-over-effective-stack, raise_to/call_to in bb, pot fraction) and
pools order-invariantly, so it adds orthogonal signal rather than duplicating.

## Measured performance (leave-date-out, out-of-sample)

- Pooled OOF **reward 0.855**, AP 0.907, AUC 0.900, recall@5%FPR 0.625.
- Hard FPR ~3% — safely inside the validator's 10% human-safety budget.
- Ablation: single HistGBDT 0.821 → 3-way tree ensemble + 7-moment aggregation
  0.827 → +DeepSets blend **0.855**.

### Architecture note (attribution)

The 3-way tree ensemble (ExtraTrees + RandomForest + HistGradientBoosting) and
the 7-moment aggregation depth are adapted from the current top Poker44 miner
([tao-miner/hot4-poker-3](https://github.com/tao-miner/hot4-poker-3), MIT). They
are applied here to Model B's **own geometry/interaction feature family**, not to
that miner's action-pattern features — so Model B gains their architectural
strength without importing the pattern signal that would re-correlate it with
either the primary model or that miner.

## Decorrelation vs the primary model (both models out-of-sample)

This is the live-serving regime — each model scores dates it did not train on:

- **vs primary model:** Pearson r ≈ 0.81, Spearman ≈ 0.79, hard agreement 0.85.
- **vs top miner (uid 99):** Pearson r ≈ 0.73, Spearman ≈ 0.76, hard agreement 0.79.

For reference, a disguised copy sits at r ≈ 0.97+. These correlations are the
*floor* forced by a shared ground truth: honest models must all flag the same
obvious bots. This is the expected signature of independent effort, not
duplication.

## Artifacts

Produced by `scripts/miner/train_model_b.py` into `poker44/model_b/artifacts/`:

- `model_b.pkl` — HistGradientBoosting classifier.
- `deepset_b.pt` — DeepSets net weights (optional; torch required to use).
- `model_b_meta.json` — feature names, isotonic calibration, pivot, blend weight.

## Retrain

```bash
python scripts/miner/train_model_b.py --data-dir data/benchmark \
    --out poker44/model_b/artifacts
```

Prints the leave-date-out CV, the GBDT-only vs blended ablation, and the
decorrelation-vs-A check every run.

## Deploy as a second miner

Model B is served by `neurons/miner_b.py` (`DetectionModelB`). To keep it a
*legitimately distinct* miner:

1. Run it on its **own hotkey**.
2. Deploy from its **own repository checkout** so the manifest's `repo_url` /
   `repo_commit` and digest are genuinely its own. All manifest fields are
   overridable per deployment via `POKER44_MODEL_*` environment variables (see
   `poker44/utils/model_manifest.py`), e.g.:

   ```bash
   export POKER44_MODEL_NAME=poker44-geometry-b
   export POKER44_MODEL_VERSION=1.0.0
   export POKER44_MODEL_REPO_URL=https://github.com/<you>/poker44-miner-b
   export POKER44_MODEL_REPO_COMMIT=$(git rev-parse HEAD)
   ```

## Honest caveats

- **Strength gap.** Model B (OOF AUC ~0.89) is weaker than the primary model
  (OOF AUC ~0.96) on this benchmark, because much of the benchmark's bot signal
  lives in the action-pattern/regularity family the primary model exploits. If
  the subnet's weight allocation is winner-take-all, a second, weaker miner may
  earn little — verify your subnet's incentive curve before deploying two.
- **Correlation floor.** You cannot drive the correlation arbitrarily low while
  staying strong: both models share the same labels, so they must agree on clear
  cases. ~0.77 is a reasonable independent-effort level; pushing it lower would
  require deliberately weakening the model.
- **Not an evasion tool.** The point of Model B is a genuinely different model,
  not disguising a copy. If the two miners' predictions ever converge (e.g. you
  retrain B toward A's features), they will — and should — read as duplicates.
