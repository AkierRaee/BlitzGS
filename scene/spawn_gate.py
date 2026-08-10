"""Spawn pruning gate: veto densification spawn candidates predicted to die.

Loads a survival model fitted offline on birth/death lineage logs
and, at each densify_and_clone / densify_and_split call, ranks the selected
candidates by predicted survival probability and vetoes the bottom k
fraction of that cohort. Ranking (not an absolute threshold) keeps the
gate scale-free across scenes: k means "the worst k% of what THIS scene
generated this event", regardless of candidate volume.

Feature assembly must match training exactly: the model was fitted on the
spawn features (BIRTH_FEATURES order) + birth_iter + is_split, on RAW
(un-normalized) values — per-scene z-scoring hurts tree models at spawn
time.
"""

import pickle

import numpy as np
import torch

BIRTH_FEATURES = ["parent_grad", "parent_opacity", "parent_scale",
                  "parent_age", "parent_denom", "lod_level"]

FEATURE_ORDER = BIRTH_FEATURES + ["birth_iter", "is_split"]


class SpawnGate:
    def __init__(self, model_pkl, k, start_iter):
        with open(model_pkl, "rb") as f:
            blob = pickle.load(f)
        if blob["feature_cols"] != FEATURE_ORDER:
            raise ValueError(
                f"gate model feature mismatch: {blob['feature_cols']} vs {FEATURE_ORDER}")
        self.model = blob["model"]
        self.meta = {k_: v for k_, v in blob.items() if k_ != "model"}
        self.k = float(k)
        self.start_iter = int(start_iter)
        self.n_cand = 0
        self.n_veto = 0

    def veto(self, feats, iteration, is_split):
        """feats: dict from GaussianModel._spawn_features for the selected
        candidates. Returns a cuda bool tensor over candidates (True = veto),
        or None when the gate is inactive / has nothing to do."""
        if feats is None or self.k <= 0 or iteration < self.start_iter:
            return None
        n = int(feats["parent_grad"].shape[0])
        n_veto = int(self.k * n)
        if n_veto == 0:
            return None
        cols = [feats[name].detach().float().cpu().numpy() for name in BIRTH_FEATURES]
        cols.append(np.full(n, iteration, dtype=np.float32))
        cols.append(np.full(n, 1.0 if is_split else 0.0, dtype=np.float32))
        X = np.stack(cols, axis=1)
        p_survive = self.model.predict_proba(X)[:, 1]
        idx = np.argpartition(p_survive, n_veto - 1)[:n_veto]
        veto = np.zeros(n, dtype=bool)
        veto[idx] = True
        self.n_cand += n
        self.n_veto += n_veto
        return torch.from_numpy(veto).cuda()
