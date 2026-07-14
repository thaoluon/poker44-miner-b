"""Train Model B (independent, decorrelated) on cached benchmark data.

Model B uses a disjoint feature family (commitment / stack geometry /
interaction response / showdown) and a different learner (sklearn
HistGradientBoosting, level-wise histogram trees) from the primary model.

This trainer:
  1. loads cached chunks through the miner-visible canonicalizer;
  2. runs leave-date-out CV with HistGradientBoosting (seed 7, date-bagged);
  3. calibrates with isotonic regression + an operating-point pivot;
  4. reports the subnet reward AND the correlation of Model B's out-of-fold
     scores against the primary model's scores (the decorrelation check);
  5. saves artifacts to poker44/model_b/artifacts.

Usage:
    python scripts/miner/train_model_b.py [--data-dir data/benchmark] \
        [--out poker44/model_b/artifacts]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from poker44.model_b.features_b import chunk_features_b  # noqa: E402
from poker44.score.scoring import reward  # noqa: E402
from poker44.validator.payload_view import prepare_hand_for_miner  # noqa: E402

try:
    from poker44.model_b.deepset import (  # noqa: E402
        encode_chunk_b,
        train_net,
        predict_net,
        BLEND_NET_WEIGHT,
    )
    import torch  # noqa: F401,E402

    _NET_AVAILABLE = True
except Exception:  # noqa: BLE001
    _NET_AVAILABLE = False
    BLEND_NET_WEIGHT = 0.0

SEED = 7
# Within-batch rank view of the geometry features (magnitude-shift invariant).
# Same idea the primary model uses, but applied to Model B's disjoint base
# features, so it lifts strength without importing the primary model's signal.
USE_RELATIVE_B = True

HGB_PARAMS = dict(
    loss="log_loss",
    learning_rate=0.03,
    max_iter=600,
    max_leaf_nodes=15,
    max_depth=4,
    min_samples_leaf=25,
    l2_regularization=2.0,
    max_features=0.6,
    early_stopping=False,
    random_state=SEED,
)


def to_miner_view(group: list[dict]) -> list[dict]:
    viewed = []
    for hand in group:
        try:
            viewed.append(prepare_hand_for_miner(hand))
        except Exception:  # noqa: BLE001
            viewed.append(hand)
    return viewed


def load_dataset(data_dir: Path, encode: bool = False):
    rows, labels, dates, splits, viewed_chunks, encoded = [], [], [], [], [], []
    for date_dir in sorted(data_dir.iterdir()):
        if not date_dir.is_dir():
            continue
        for path in sorted(date_dir.glob("*.json")):
            if path.name == "manifest.json":
                continue
            payload = json.loads(path.read_text())
            groups = payload.get("chunks") or []
            truth = payload.get("groundTruth") or []
            if len(groups) != len(truth):
                continue
            for group, label in zip(groups, truth):
                viewed = to_miner_view(group)
                rows.append(chunk_features_b(viewed))
                viewed_chunks.append(viewed)
                if encode and _NET_AVAILABLE:
                    encoded.append(encode_chunk_b(viewed))
                labels.append(int(label))
                dates.append(date_dir.name)
                splits.append(payload.get("split") or "train")
    feature_names = sorted({k for row in rows for k in row})
    dates_arr = np.asarray(dates)
    X_abs = np.asarray(
        [[row.get(k, 0.0) for k in feature_names] for row in rows], dtype=float
    )
    if USE_RELATIVE_B:
        # Per-date rank view mirrors the per-query batch the miner sees at serve.
        from poker44.model_b.features_b import batch_relative_matrix_b

        rel = np.zeros_like(X_abs)
        for d in set(dates):
            m = dates_arr == d
            rel[m] = batch_relative_matrix_b(X_abs[m])
        X = np.hstack([X_abs, rel])
    else:
        X = X_abs
    result = (
        X,
        np.asarray(labels),
        np.asarray(dates),
        np.asarray(splits),
        feature_names,
        viewed_chunks,
    )
    if encode:
        return result + (encoded,)
    return result


def net_oof(encoded, y, dates, folds):
    """Leave-date-out out-of-fold DeepSets probabilities."""
    oof = np.full(len(y), np.nan)
    for fold_dates in folds:
        mask = np.isin(dates, fold_dates)
        tr = np.where(~mask)[0]
        rng = np.random.RandomState(SEED)
        rng.shuffle(tr)
        cut = int(len(tr) * 0.85)
        trn, val = tr[:cut], tr[cut:]
        model = train_net(
            [encoded[i] for i in trn], y[trn],
            [encoded[i] for i in val], y[val],
        )
        te = np.where(mask)[0]
        oof[te] = predict_net(model, [encoded[i] for i in te])
    return oof


def make_folds(dates: np.ndarray):
    unique = sorted(set(dates.tolist()))
    big = [d for d in unique if (dates == d).sum() >= 50]
    small = [d for d in unique if d not in big]
    return [[d] for d in big] + ([small] if small else []), unique


def fit(X, y):
    model = HistGradientBoostingClassifier(**HGB_PARAMS)
    model.fit(X, y)
    return model


def calibrate_iso(oof_raw, labels):
    """Isotonic map raw->P(bot), then an operating-point pivot so the validator's
    hard 0.5 threshold lands at the OOF human ~98th percentile (~2-4% FPR)."""
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(oof_raw, labels)
    mapped = iso.predict(oof_raw)
    humans = mapped[labels == 0]
    pivot = float(np.quantile(humans, 0.98)) if humans.size else 0.5
    pivot = min(max(pivot, 0.05), 0.95)
    xs = np.linspace(0.0, 1.0, 512)
    ys = iso.predict(xs)
    return {"iso_x": xs.tolist(), "iso_y": ys.tolist(), "pivot": pivot}


def apply_cal(raw, calib):
    xs = np.asarray(calib["iso_x"])
    ys = np.asarray(calib["iso_y"])
    mapped = np.interp(raw, xs, ys)
    pivot = float(calib.get("pivot", 0.5))
    out = np.where(
        mapped <= pivot,
        0.5 * mapped / max(pivot, 1e-9),
        0.5 + 0.5 * (mapped - pivot) / max(1.0 - pivot, 1e-9),
    )
    return np.clip(out, 0.0, 1.0)


def evaluate(scores, labels, tag):
    rew, detail = reward(scores, labels)
    ap = average_precision_score(labels, scores) if labels.any() else 0.0
    auc = roc_auc_score(labels, scores) if 0 < labels.sum() < len(labels) else 0.0
    print(
        f"  {tag:20s} reward={rew:.4f} AP={ap:.4f} AUC={auc:.4f} "
        f"recall@5%FPR={detail['bot_recall']:.4f} hard_fpr={detail['hard_fpr']:.4f}"
    )
    return rew, ap, auc


def model_a_scores(viewed_chunks):
    """Primary model's calibrated scores on the same chunks (decorrelation ref)."""
    try:
        from poker44.model.inference import DetectionModel

        m = DetectionModel()
        if not m.ready:
            return None
        return np.asarray(m.score_chunks(viewed_chunks), dtype=float)
    except Exception as err:  # noqa: BLE001
        print(f"(model A unavailable for decorrelation check: {err})")
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/benchmark")
    ap.add_argument("--out", default="poker44/model_b/artifacts")
    args = ap.parse_args()

    X, y, dates, splits, feature_names, viewed, encoded = load_dataset(
        Path(args.data_dir), encode=True
    )
    folds, unique_dates = make_folds(dates)
    net_weight = BLEND_NET_WEIGHT if _NET_AVAILABLE else 0.0
    print(
        f"dataset: {len(y)} chunks, {X.shape[1]} features, "
        f"{int(y.sum())} bot / {int((y == 0).sum())} human, {len(unique_dates)} dates | "
        f"deepset blend={'on' if net_weight else 'off'} (w={net_weight})"
    )

    # Leave-date-out OOF (GBDT).
    oof_gbdt = np.full(len(y), np.nan)
    fold_masks = []
    for fold_dates in folds:
        mask = np.isin(dates, fold_dates)
        fold_masks.append((fold_dates, mask))
        model = fit(X[~mask], y[~mask])
        oof_gbdt[mask] = model.predict_proba(X[mask])[:, 1]
    assert not np.isnan(oof_gbdt).any()

    # Blend with out-of-fold DeepSets scores (orthogonal geometry signal).
    if net_weight:
        print("training DeepSets net per fold (out-of-fold)...")
        oof_net = net_oof(encoded, y, dates, folds)
        gbdt_only = apply_cal(oof_gbdt, calibrate_iso(oof_gbdt, y))
        evaluate(gbdt_only, y, "GBDT-only (oof)")
        oof = net_weight * oof_net + (1.0 - net_weight) * oof_gbdt
    else:
        oof = oof_gbdt

    calib = calibrate_iso(oof, y)
    oof_cal = apply_cal(oof, calib)
    print(f"\ncalibration pivot (isotonic OOF): {calib['pivot']:.4f}")

    print("\nleave-date-out CV:")
    for fold_dates, mask in fold_masks:
        tag = fold_dates[0] if len(fold_dates) == 1 else f"small x{len(fold_dates)}"
        evaluate(oof_cal[mask], y[mask], tag)
    print("\npooled out-of-fold:")
    evaluate(oof_cal, y, "ALL (oof)")

    # --- Decorrelation check vs the primary model -----------------------------
    a_scores = model_a_scores(viewed)
    if a_scores is not None and a_scores.shape == oof_cal.shape:
        pear = float(np.corrcoef(oof_cal, a_scores)[0, 1])
        from scipy.stats import spearmanr

        spear = float(spearmanr(oof_cal, a_scores).correlation)
        # Agreement on hard 0.5 decisions.
        agree = float(np.mean((oof_cal >= 0.5) == (a_scores >= 0.5)))
        print("\n=== decorrelation vs Model A ===")
        print(f"  Pearson r  = {pear:.3f}")
        print(f"  Spearman r = {spear:.3f}")
        print(f"  hard-decision agreement = {agree:.3f}")
        print("  (lower correlation = more genuinely independent predictions)")

    # --- Final models on everything ------------------------------------------
    final = fit(X, y)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    import pickle

    with open(out_dir / "model_b.pkl", "wb") as fh:
        pickle.dump(final, fh)

    net_saved = False
    if net_weight:
        idx = np.arange(len(y))
        rng = np.random.RandomState(SEED)
        rng.shuffle(idx)
        cut = int(len(idx) * 0.9)
        final_net = train_net(
            [encoded[i] for i in idx[:cut]], y[idx[:cut]],
            [encoded[i] for i in idx[cut:]], y[idx[cut:]],
        )
        torch.save(final_net.state_dict(), str(out_dir / "deepset_b.pt"))
        net_saved = True

    (out_dir / "model_b_meta.json").write_text(
        json.dumps(
            {
                "feature_names": feature_names,
                "calibration": calib,
                "n_train_examples": int(len(y)),
                "release_dates": unique_dates,
                "hgb_params": {k: v for k, v in HGB_PARAMS.items()},
                "seed": SEED,
                "use_relative": bool(USE_RELATIVE_B),
                "net_blend_weight": float(net_weight) if net_saved else 0.0,
            },
            indent=2,
        )
    )
    print(f"\nsaved Model B to {out_dir} (deepset={'yes' if net_saved else 'no'})")


if __name__ == "__main__":
    main()
