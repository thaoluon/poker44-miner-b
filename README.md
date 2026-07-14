# Poker44 Model-B Miner

An independent Poker44 bot-detection miner (`poker44-geometry-b`), built to run on
its own hotkey alongside a primary miner without duplicating it. It uses a
distinct feature family and learner, and produces **decorrelated** predictions —
it earns on its own merits.

- **Model:** HistGradientBoosting (level-wise) over commitment / stack-geometry /
  interaction-response / showdown features, blended with an attention-pooled
  DeepSets net; isotonic-calibrated.
- **Measured (leave-date-out, out-of-sample):** reward ≈ 0.839, AUC ≈ 0.89,
  hard-FPR ≈ 3%.
- **Independence:** predictions decorrelate from a typical action-pattern model
  (Pearson ≈ 0.8 out-of-sample), the signature of genuine independent effort.

See [docs/model_b.md](docs/model_b.md) for the full design, ablations, and the
decorrelation methodology.

## Layout

```
neurons/miner_b.py            # the miner neuron (MinerB)
poker44/model_b/              # feature extractor, DeepSets net, inference, live log
poker44/model_b/artifacts/    # trained model + calibration (shipped)
scripts/miner/train_model_b.py# retraining (needs the benchmark data + poker44.score/validator canonicalizer)
poker44/base, utils, validator/synapse.py  # minimal subnet runtime shim
```

> This is a **miner** repository. It intentionally contains no validator
> internals. The subnet infrastructure lives in the upstream
> [poker44/Poker44-subnet](https://github.com/poker44/Poker44-subnet).

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
export POKER44_MODEL_NAME=poker44-geometry-b
export POKER44_MODEL_VERSION=1.0.0
export POKER44_MODEL_REPO_URL=https://github.com/thaoluon/poker44-miner-b
export POKER44_MODEL_REPO_COMMIT=$(git rev-parse HEAD)

python neurons/miner_b.py --netuid <netuid> \
    --wallet.name <coldkey> --wallet.hotkey <hotkey> \
    --subtensor.network <network>
```

The trained artifacts in `poker44/model_b/artifacts/` are loaded automatically;
if they are missing or torch is unavailable the miner degrades gracefully.

## Retrain

`scripts/miner/train_model_b.py` trains against the public Poker44 benchmark. It
imports the subnet's `poker44.score.reward` and `poker44.validator.payload_view`
(the miner-visible canonicalizer), so run it from a full subnet checkout, or add
those two modules, then:

```bash
python scripts/miner/train_model_b.py --data-dir data/benchmark \
    --out poker44/model_b/artifacts
```

## License

MIT — see [LICENSE](LICENSE).
