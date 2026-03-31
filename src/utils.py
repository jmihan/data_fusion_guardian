import gc
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from scipy.stats import rankdata
from scipy.optimize import minimize, differential_evolution
from scipy.special import softmax

from src.config import (
    NEG_SAMPLE_BORDER_STR, NEG_SAMPLE_MOD_RECENT, NEG_SAMPLE_MOD_OLD,
    RECENCY_WEIGHT_START, RANDOM_SEED,
)


def make_weights(raw_target: np.ndarray, event_ts=None) -> np.ndarray:
    w = np.where(raw_target == 1, 10.0, np.where(raw_target == 0, 2.5, 1.0)).astype(np.float32)
    if event_ts is not None:
        ts = pd.to_datetime(event_ts)
        border = pd.Timestamp(NEG_SAMPLE_BORDER_STR)
        g = raw_target == -1
        w[g & (ts >= border)] = 1.8
        w[g & (ts < border)]  = 3.0
        w[ts >= RECENCY_WEIGHT_START] *= 1.15
    return w


def downcast_pandas(df, cat_cols_set=None):
    if cat_cols_set is None:
        cat_cols_set = set()
    for c in df.select_dtypes(include=['int64', 'int32']).columns:
        if c in cat_cols_set:
            continue
        col_min, col_max = df[c].min(), df[c].max()
        if pd.isna(col_min):
            continue
        if col_min >= -128 and col_max <= 127:
            df[c] = df[c].astype(np.int8)
        elif col_min >= -32768 and col_max <= 32767:
            df[c] = df[c].astype(np.int16)
        elif col_min >= -2147483648 and col_max <= 2147483647:
            df[c] = df[c].astype(np.int32)
    for c in df.select_dtypes(include=['float64']).columns:
        df[c] = df[c].astype(np.float32)
    return df


def rank_norm(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    return rankdata(x, method="average") / (len(x) + 1.0)


def _fast_ap(y_true, y_score):
    desc = np.argsort(y_score)[::-1]
    y_sorted = y_true[desc]
    tp = np.cumsum(y_sorted)
    n_pos = tp[-1]
    if n_pos == 0:
        return 0.0
    prec = tp / np.arange(1, len(y_sorted) + 1)
    return float(np.sum(prec * y_sorted) / n_pos)


def optimize_blend_weights(heads: Dict[str, np.ndarray], y_true: np.ndarray, method: str = "nelder-mead"):
    keys = list(heads.keys())
    preds = [heads[k] for k in keys]
    y_np = np.asarray(y_true, dtype=np.int8)

    if method == "grid-search" and len(keys) <= 4:
        return _grid_search_blend(keys, preds, y_np)
    elif method == "diff-evolution":
        return _diff_evolution_blend(keys, preds, y_np)
    else:
        return _nelder_mead_blend(keys, preds, y_np)


def _nelder_mead_blend(keys, preds, y_true):
    def neg_ap(logits):
        w = softmax(logits)
        blend = sum(w[i] * preds[i] for i in range(len(keys)))
        return -average_precision_score(y_true, blend)
    x0 = np.zeros(len(keys), dtype=np.float64)
    res = minimize(neg_ap, x0, method="Nelder-Mead",
                   options={"maxiter": 1500, "xatol": 1e-4, "fatol": 1e-6})
    best_w = softmax(res.x).astype(np.float32)
    best_ap = float(-res.fun)
    return keys, best_w, best_ap


def _grid_search_blend(keys, preds, y_true, step=0.01):
    n = len(keys)
    if n == 2:
        best_ap, best_w = -1.0, np.array([0.5, 0.5], dtype=np.float32)
        for w0 in np.arange(0, 1 + step / 2, step):
            w1 = 1.0 - w0
            blend = w0 * preds[0] + w1 * preds[1]
            ap = _fast_ap(y_true, blend)
            if ap > best_ap:
                best_ap = ap
                best_w = np.array([w0, w1], dtype=np.float32)
        return keys, best_w, float(best_ap)
    elif n == 3:
        best_ap, best_w = -1.0, np.array([1/3]*3, dtype=np.float32)
        grid = np.arange(0, 1 + step / 2, step)
        for w0 in grid:
            for w1 in grid:
                w2 = 1.0 - w0 - w1
                if w2 < -1e-6:
                    break
                w2 = max(0.0, w2)
                blend = w0 * preds[0] + w1 * preds[1] + w2 * preds[2]
                ap = _fast_ap(y_true, blend)
                if ap > best_ap:
                    best_ap = ap
                    best_w = np.array([w0, w1, w2], dtype=np.float32)
        return keys, best_w, float(best_ap)
    elif n == 4:
        best_ap, best_w = -1.0, np.array([0.25]*4, dtype=np.float32)
        grid = np.arange(0, 1 + step / 2, step)
        for w0 in grid:
            for w1 in grid:
                if w0 + w1 > 1.0 + 1e-6:
                    break
                for w2 in grid:
                    w3 = 1.0 - w0 - w1 - w2
                    if w3 < -1e-6:
                        break
                    w3 = max(0.0, w3)
                    blend = w0 * preds[0] + w1 * preds[1] + w2 * preds[2] + w3 * preds[3]
                    ap = _fast_ap(y_true, blend)
                    if ap > best_ap:
                        best_ap = ap
                        best_w = np.array([w0, w1, w2, w3], dtype=np.float32)
        return keys, best_w, float(best_ap)
    else:
        return _nelder_mead_blend(keys, preds, y_true)


def _diff_evolution_blend(keys, preds, y_true):
    n = len(keys)
    def neg_ap(weights):
        w = weights / weights.sum()
        blend = sum(w[i] * preds[i] for i in range(n))
        return -average_precision_score(y_true, blend)
    bounds = [(0.0, 1.0)] * n
    res = differential_evolution(neg_ap, bounds, maxiter=500, seed=RANDOM_SEED, tol=1e-6, polish=True)
    best_w = (res.x / res.x.sum()).astype(np.float32)
    best_ap = float(-res.fun)
    return keys, best_w, best_ap


def dedupe(items):
    seen = set()
    out = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def lgb_prepare(X: pd.DataFrame, cat_cols: List[str]) -> pd.DataFrame:
    Xc = X.copy()
    for col in cat_cols:
        Xc[col] = Xc[col].fillna(-1).astype(np.int64) + 1
    return Xc


def lgb_ap_metric(preds, dataset):
    y_true = dataset.get_label()
    return "ap", average_precision_score(y_true, preds), True


# ── Logit z-score helpers (for mega-ensemble) ───────────────────────────────

def logit(p: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def zscore(x: np.ndarray) -> np.ndarray:
    return (x - x.mean()) / (x.std() + 1e-12)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def mega_logit_zscore_blend(
    preds: Dict[str, np.ndarray],
    weights: Dict[str, float],
    eps: float = 1e-7,
) -> np.ndarray:
    """Multi-way logit z-score blend: logit -> zscore -> weighted avg -> sigmoid."""
    total_w = sum(weights[k] for k in preds)
    z_blend = np.zeros(len(next(iter(preds.values()))), dtype=np.float64)
    for name, pred in preds.items():
        w = weights[name] / total_w
        z_i = zscore(logit(pred, eps))
        nan_cnt = int(np.isnan(z_i).sum())
        if nan_cnt > 0:
            print(f"  WARNING: {name} has {nan_cnt} NaN after logit+zscore, filling with 0")
            z_i = np.nan_to_num(z_i, nan=0.0)
        z_blend += w * z_i
    return sigmoid(z_blend)
