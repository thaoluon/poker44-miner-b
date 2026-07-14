"""Attention-pooled DeepSets scorer for Model B.

Architecturally distinct from the primary model's sequence net:

* the primary net runs a GRU over each hand's ORDERED action sequence, then
  mean/max-pools hands;
* this net encodes each action independently through an MLP and pools actions
  within a hand with a permutation-invariant mean+max (DeepSets -- no
  recurrence, so no within-hand order signal), then pools hands with a learned
  ATTENTION weight.

It also reads a different channel set: the geometry / commitment values
(amount over effective stack, raise_to / call_to in bb, pot fraction) that the
primary GRU under-weights. The blend therefore adds orthogonal signal rather
than duplicating the primary model.

Torch is optional: if unavailable the caller falls back to GBDT-only scoring.
"""

from __future__ import annotations

import numpy as np

from poker44.model_b.features_b import _safe

MAX_H, MAX_A = 40, 14
_ATYPE = {"fold": 0, "call": 1, "check": 2, "bet": 3, "raise": 4}
_STR = {"preflop": 0, "flop": 1, "turn": 2, "river": 3}
BLEND_NET_WEIGHT = 0.25  # tuned on OOF; keeps the GBDT as the primary signal


def encode_chunk_b(chunk: list[dict]) -> tuple:
    """Encode a chunk into fixed-shape geometry/commitment tensors + masks."""
    atype = np.full((MAX_H, MAX_A), 5, np.int64)     # 5 = pad/other
    street = np.full((MAX_H, MAX_A), 4, np.int64)    # 4 = pad
    cont = np.zeros((MAX_H, MAX_A, 5), np.float32)   # geometry channels
    is_hero = np.zeros((MAX_H, MAX_A), np.float32)
    amask = np.zeros((MAX_H, MAX_A), np.float32)
    hmask = np.zeros(MAX_H, np.float32)
    for h, hand in enumerate((chunk or [])[:MAX_H]):
        meta = hand.get("metadata") or {}
        players = hand.get("players") or []
        hero = meta.get("hero_seat")
        bb = _safe(meta.get("bb"), 0.0) or 1.0
        hero_stack = next(
            (_safe(p.get("starting_stack")) for p in players if p.get("seat") == hero),
            0.0,
        )
        vill = [
            _safe(p.get("starting_stack"))
            for p in players
            if p.get("seat") != hero and _safe(p.get("starting_stack")) > 0
        ]
        eff = min(hero_stack, max(vill)) if hero_stack > 0 and vill else 0.0
        eff_bb = eff / bb if eff > 0 else 0.0
        acts = hand.get("actions") or []
        if acts:
            hmask[h] = 1.0
        for a, act in enumerate(acts[:MAX_A]):
            atype[h, a] = _ATYPE.get(act.get("action_type") or "", 5)
            street[h, a] = _STR.get(act.get("street") or "", 4)
            is_hero[h, a] = 1.0 if act.get("actor_seat") == hero else 0.0
            amt = _safe(act.get("normalized_amount_bb"))
            pot_before = _safe(act.get("pot_before")) / bb
            cont[h, a] = [
                amt / 50.0,
                (amt / (eff_bb + 1e-6)) if eff_bb > 0 else 0.0,   # commitment
                _safe(act.get("raise_to")) / bb / 50.0,
                _safe(act.get("call_to")) / bb / 50.0,
                (amt / (pot_before + 1e-6)) if pot_before > 0 else 0.0,  # pot frac
            ]
            amask[h, a] = 1.0
    return atype, street, is_hero, cont, amask, hmask


def _build_module(d: int = 32):
    import torch.nn as nn

    class DeepSetAttn(nn.Module):
        def __init__(self):
            super().__init__()
            self.ea = nn.Embedding(6, 6)
            self.es = nn.Embedding(5, 4)
            self.phi = nn.Sequential(
                nn.Linear(6 + 4 + 1 + 5, d), nn.ReLU(), nn.Linear(d, d), nn.ReLU()
            )
            self.hand_attn = nn.Linear(2 * d, 1)   # attention over hands
            self.head = nn.Sequential(
                nn.Linear(2 * d, d), nn.ReLU(), nn.Dropout(0.3), nn.Linear(d, 1)
            )

        def forward(self, atype, street, is_hero, cont, amask, hmask):
            import torch

            x = torch.cat(
                [self.ea(atype), self.es(street), is_hero.unsqueeze(-1), cont], -1
            )
            x = self.phi(x)                                   # (B,H,A,d)
            am = amask.unsqueeze(-1)
            mean_a = (x * am).sum(2) / am.sum(2).clamp(min=1)  # (B,H,d)
            mx_a = (x + (am - 1) * 1e9).max(2).values          # (B,H,d)
            hand = torch.cat([mean_a, mx_a], -1)               # (B,H,2d)
            # Attention pooling over hands (masked softmax).
            logits = self.hand_attn(hand).squeeze(-1)
            logits = logits + (hmask - 1) * 1e9
            w = torch.softmax(logits, dim=1).unsqueeze(-1)
            chunk = (hand * w).sum(1)                           # (B,2d)
            return self.head(chunk).squeeze(-1)

    return DeepSetAttn()


def _batch(enc_list, idxs):
    import torch

    b = [enc_list[i] for i in idxs]
    dtypes = [torch.long, torch.long, torch.float32, torch.float32, torch.float32, torch.float32]
    return [
        torch.tensor(np.stack([x[k] for x in b]), dtype=dtypes[k]) for k in range(6)
    ]


def train_net(enc_train, y_train, enc_val, y_val, *, seed: int = 7, max_epochs: int = 40):
    """Train the DeepSets net; early-stop on validation loss."""
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)
    model = _build_module()
    opt = torch.optim.Adam(model.parameters(), 1e-3, weight_decay=1e-4)
    lossf = nn.BCEWithLogitsLoss()
    n = len(y_train)
    best, best_state, patience = 1e9, None, 0
    rng = np.random.RandomState(seed)
    for _ in range(max_epochs):
        model.train()
        perm = rng.permutation(n)
        for s in range(0, n, 128):
            gi = perm[s:s + 128].tolist()
            opt.zero_grad()
            out = model(*_batch(enc_train, gi))
            loss = lossf(out, torch.tensor(y_train[gi], dtype=torch.float32))
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vout = model(*_batch(enc_val, list(range(len(y_val)))))
            vl = lossf(vout, torch.tensor(y_val, dtype=torch.float32)).item()
        if vl < best - 1e-4:
            best = vl
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= 5:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model


def predict_net(model, enc_list) -> np.ndarray:
    import torch

    if not enc_list:
        return np.zeros(0, dtype=float)
    with torch.no_grad():
        return torch.sigmoid(model(*_batch(enc_list, list(range(len(enc_list)))))).numpy()


class NetScorer:
    """Loads a saved DeepSets net for inference; None if torch/model absent."""

    def __init__(self, weights_path):
        self.model = None
        try:
            import torch

            model = _build_module()
            model.load_state_dict(torch.load(str(weights_path), map_location="cpu"))
            model.eval()
            self.model = model
        except Exception:  # noqa: BLE001
            self.model = None

    @property
    def ready(self) -> bool:
        return self.model is not None

    def score(self, chunks: list[list[dict]]) -> np.ndarray:
        enc = [encode_chunk_b(c or []) for c in chunks]
        return predict_net(self.model, enc)
