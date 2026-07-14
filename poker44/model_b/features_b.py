"""Feature extractor for Model B (commitment & interaction geometry).

Deliberately disjoint from ``poker44.model.features``: it does not compute
action-type ratios, entropy/run regularity, or signature-replay tuples. Instead
it reads under-exploited raw signals -- effective-stack / SPR, bet sizing
relative to the effective stack, raise_to / call_to magnitudes, the inter-actor
response graph, and showdown dynamics -- and aggregates them with different
distribution moments (mean/std/median/p10/p90).

Everything is noise-robust by construction: the validator injects quantization
noise into ``pot_before`` / ``pot_after``, so we avoid trusting per-action pot
deltas and prefer stack-relative and response-structure signals.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Dict, List

import numpy as np

STREETS = ("preflop", "flop", "turn", "river")
_STREET_ORDER = {s: i for i, s in enumerate(STREETS)}
_AGGRO = {"bet", "raise"}
_MEANINGFUL = {"fold", "call", "check", "bet", "raise"}


def _safe(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(out) or math.isinf(out):
        return default
    return out


def _moments(values: List[float], prefix: str, feats: Dict[str, float]) -> None:
    """Aggregate a per-hand value list into distribution moments.

    Uses mean/std/median/p10/p90/min/max -- a richer 7-moment view (the top
    Poker44 miner's aggregation depth) over Model B's geometry signals. The
    signals themselves stay distinct from the primary model's, so this adds
    resolution without importing the pattern family.
    """
    if values:
        arr = np.asarray(values, dtype=float)
        feats[f"{prefix}_mean"] = float(np.mean(arr))
        feats[f"{prefix}_std"] = float(np.std(arr))
        feats[f"{prefix}_med"] = float(np.median(arr))
        feats[f"{prefix}_p10"] = float(np.percentile(arr, 10))
        feats[f"{prefix}_p90"] = float(np.percentile(arr, 90))
        feats[f"{prefix}_min"] = float(np.min(arr))
        feats[f"{prefix}_max"] = float(np.max(arr))
    else:
        for suffix in ("mean", "std", "med", "p10", "p90", "min", "max"):
            feats[f"{prefix}_{suffix}"] = 0.0


def hand_geometry(hand: Dict[str, Any]) -> Dict[str, float]:
    """Per-hand commitment / geometry / interaction signals."""
    meta = hand.get("metadata") or {}
    players = hand.get("players") or []
    actions = hand.get("actions") or []
    outcome = hand.get("outcome") or {}

    bb = _safe(meta.get("bb"), 0.0) or 1.0
    hero = meta.get("hero_seat")

    f: Dict[str, float] = {}

    # --- Stack geometry (effective stack, SPR) --------------------------------
    stacks_bb = [
        _safe(p.get("starting_stack")) / bb
        for p in players
        if _safe(p.get("starting_stack")) > 0
    ]
    hero_stack = next(
        (_safe(p.get("starting_stack")) / bb for p in players if p.get("seat") == hero),
        0.0,
    )
    villain_stacks = [
        _safe(p.get("starting_stack")) / bb
        for p in players
        if p.get("seat") != hero and _safe(p.get("starting_stack")) > 0
    ]
    eff_stack = 0.0
    if hero_stack > 0 and villain_stacks:
        eff_stack = min(hero_stack, max(villain_stacks))
    f["eff_stack_bb"] = eff_stack
    f["hero_stack_bb"] = hero_stack
    f["stack_spread"] = float(np.std(stacks_bb)) if len(stacks_bb) > 1 else 0.0
    f["stack_ratio"] = (
        hero_stack / (float(np.mean(villain_stacks)) + 1e-9) if villain_stacks else 1.0
    )
    f["deep_flag"] = 1.0 if eff_stack >= 100 else 0.0
    f["short_flag"] = 1.0 if 0 < eff_stack <= 25 else 0.0

    # --- Commitment: bet size relative to the EFFECTIVE STACK (not the pot) ----
    hero_commit, table_commit, allins = [], [], 0
    raise_to_bb, call_to_bb, raise_over_call = [], [], []
    n_actions = len(actions)
    for act in actions:
        atype = act.get("action_type")
        amt_bb = _safe(act.get("normalized_amount_bb"))
        if atype in _AGGRO and amt_bb > 0:
            commit = amt_bb / (eff_stack + 1e-9) if eff_stack > 0 else 0.0
            table_commit.append(commit)
            if act.get("actor_seat") == hero:
                hero_commit.append(commit)
            if eff_stack > 0 and amt_bb >= 0.9 * eff_stack:
                allins += 1
        rt = _safe(act.get("raise_to"))
        ct = _safe(act.get("call_to"))
        if rt > 0:
            raise_to_bb.append(rt / bb)
        if ct > 0:
            call_to_bb.append(ct / bb)
        if rt > 0 and ct > 0:
            raise_over_call.append(rt / (ct + 1e-9))

    _moments(hero_commit, "hero_commit", f)
    _moments(table_commit, "table_commit", f)
    _moments(raise_to_bb, "raise_to_bb", f)
    _moments(call_to_bb, "call_to_bb", f)
    _moments(raise_over_call, "raise_over_call", f)
    f["allin_rate"] = allins / max(1, n_actions)

    # --- Inter-actor response graph -------------------------------------------
    # Walk actions per street and characterise how players respond to aggression.
    by_street: Dict[str, List[dict]] = defaultdict(list)
    for act in actions:
        by_street[act.get("street") or ""].append(act)

    reraise = fold_to_raise = call_to_raise = check_raise = 0
    raises_faced = 0
    donk = 0
    prev_street_aggressor = None
    for s in STREETS:
        seq = by_street.get(s, [])
        street_aggressor = None
        checked_seats = set()
        last_was_raise = False
        for i, act in enumerate(seq):
            atype = act.get("action_type")
            seat = act.get("actor_seat")
            if atype == "check":
                checked_seats.add(seat)
            if last_was_raise:
                raises_faced += 1
                if atype == "fold":
                    fold_to_raise += 1
                elif atype == "call":
                    call_to_raise += 1
                elif atype == "raise":
                    reraise += 1
            if atype == "raise" and seat in checked_seats:
                check_raise += 1
            # Donk: first aggressor on this street is NOT the previous street's
            # aggressor (betting into the prior initiative).
            if atype in _AGGRO and street_aggressor is None:
                street_aggressor = seat
                if (
                    prev_street_aggressor is not None
                    and seat != prev_street_aggressor
                    and s != "preflop"
                ):
                    donk += 1
            last_was_raise = atype == "raise"
        if street_aggressor is not None:
            prev_street_aggressor = street_aggressor

    faced = max(1, raises_faced)
    f["reraise_rate"] = reraise / faced
    f["fold_to_raise_rate"] = fold_to_raise / faced
    f["call_to_raise_rate"] = call_to_raise / faced
    f["check_raise_count"] = float(check_raise)
    f["donk_count"] = float(donk)
    f["raises_faced"] = float(raises_faced)

    # --- Street reach / initiative --------------------------------------------
    reached = max(
        (_STREET_ORDER.get(a.get("street") or "", 0) for a in actions), default=0
    )
    f["street_reached"] = float(reached)
    # Initiative retention: same seat is first aggressor on consecutive streets.
    first_aggr = {}
    for s in STREETS:
        for act in by_street.get(s, []):
            if act.get("action_type") in _AGGRO:
                first_aggr[s] = act.get("actor_seat")
                break
    retained = sum(
        1
        for a, b in zip(STREETS, STREETS[1:])
        if a in first_aggr and b in first_aggr and first_aggr[a] == first_aggr[b]
    )
    f["initiative_retained"] = float(retained)

    # --- Showdown dynamics ----------------------------------------------------
    showdown = 1.0 if outcome.get("showdown") else 0.0
    showed = sum(1 for p in players if p.get("showed_hand"))
    hero_showed = 1.0 if any(
        p.get("seat") == hero and p.get("showed_hand") for p in players
    ) else 0.0
    f["showdown"] = showdown
    f["showed_frac"] = showed / max(1, len(players))
    f["hero_showed"] = hero_showed
    # Went-to-showdown having invested aggression (aggressive showdown).
    f["aggro_showdown"] = showdown if hero_commit else 0.0

    return f


def chunk_features_b(chunk: List[Dict[str, Any]]) -> Dict[str, float]:
    """Aggregate per-hand geometry into a chunk-level feature dict."""
    hands = chunk or []
    if not hands:
        return {"n_hands_b": 0.0}

    per_hand = [hand_geometry(h) for h in hands]
    keys = sorted({k for row in per_hand for k in row})

    feats: Dict[str, float] = {"n_hands_b": float(len(hands))}
    for key in keys:
        vals = [row.get(key, 0.0) for row in per_hand]
        _moments(vals, key, feats)

    # --- Chunk-level cross-hand consistency of hero commitment ----------------
    hero_commit_means = [
        row.get("hero_commit_mean", 0.0)
        for row in per_hand
        if row.get("hero_commit_mean", 0.0) > 0
    ]
    if len(hero_commit_means) >= 2:
        cm = np.asarray(hero_commit_means)
        feats["hero_commit_cv"] = float(np.std(cm) / (np.mean(cm) + 1e-9))
        feats["hero_commit_range"] = float(np.ptp(cm))
    else:
        feats["hero_commit_cv"] = 0.0
        feats["hero_commit_range"] = 0.0

    # Fraction of hands that reached showdown / went deep -- session texture.
    reached = np.asarray([row.get("street_reached", 0.0) for row in per_hand])
    feats["deep_hand_frac"] = float(np.mean(reached >= 2))
    feats["showdown_frac_chunk"] = float(
        np.mean([row.get("showdown", 0.0) for row in per_hand])
    )
    faced = np.asarray([row.get("raises_faced", 0.0) for row in per_hand])
    feats["aggression_density"] = float(np.mean(faced))

    return feats


def batch_relative_matrix_b(absolute: np.ndarray) -> np.ndarray:
    """Within-batch rank-percentile view of each feature column (values in [0,1]).

    Magnitude-shift invariant: survives the benchmark->live sizing shift. Returns
    an all-0.5 matrix if the batch has fewer than two rows.
    """
    n = absolute.shape[0]
    if n < 2:
        return np.full_like(absolute, 0.5)
    try:
        from scipy.stats import rankdata

        out = np.empty_like(absolute, dtype=float)
        for j in range(absolute.shape[1]):
            out[:, j] = (rankdata(absolute[:, j], method="average") - 1.0) / (n - 1)
        return out
    except Exception:  # noqa: BLE001 -- scipy optional
        order = np.argsort(np.argsort(absolute, axis=0), axis=0)
        return order / (n - 1)


def build_feature_matrix_b(
    chunks: List[List[Dict[str, Any]]],
    feature_names: List[str],
    use_relative: bool = False,
) -> np.ndarray:
    """Project chunks onto a fixed feature-name ordering (missing -> 0.0).

    When ``use_relative`` is set, append the within-batch rank view so the
    column count doubles (absolute block followed by its relative block),
    matching the training layout.
    """
    rows = [chunk_features_b(c) for c in chunks]
    absolute = np.asarray(
        [[row.get(name, 0.0) for name in feature_names] for row in rows], dtype=float
    )
    if not use_relative or absolute.size == 0:
        return absolute
    return np.hstack([absolute, batch_relative_matrix_b(absolute)])
