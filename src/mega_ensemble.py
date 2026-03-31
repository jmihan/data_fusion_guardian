from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.config import DATA_DIR, SUBMISSION_DIR, SOL154_GRID
from src.utils import mega_logit_zscore_blend


def run_mega_ensemble(
    sol154_blend_path: Path,
    val_scores: Dict[str, float],
    best_combo_name: str,
):
    """
    Mega-ensemble: blend meta3b1n submissions with sol_154 blend.
    Produces 18 submissions (3 strategies x 6 sol154 weight grids).

    Args:
        sol154_blend_path: path to sub_totalblend.csv
        val_scores: dict {model_name: val_ap} for 5 individual models
        best_combo_name: name of the best blend combo (e.g. "3boost")
    """
    sol154_df = pd.read_csv(sol154_blend_path)
    print(f"Loaded sol_154 blend: {sol154_blend_path} ({len(sol154_df):,} rows)")

    mega_sub_sources = {
        "catboost":   SUBMISSION_DIR / "submission_cb.csv",
        "lightgbm":   SUBMISSION_DIR / "submission_lgb.csv",
        "xgboost":    SUBMISSION_DIR / "submission_xgb.csv",
        "extratrees": SUBMISSION_DIR / "submission_et.csv",
        "mlp":        SUBMISSION_DIR / "submission_mlp.csv",
        f"my_best_blend ({best_combo_name})": SUBMISSION_DIR / f"submission_blend_{best_combo_name}.csv",
    }

    sample_submit = pd.read_csv(DATA_DIR / "sample_submit.csv")
    mega_df = sample_submit[["event_id"]].copy()

    for label, path in mega_sub_sources.items():
        sub = pd.read_csv(path)
        mega_df = mega_df.merge(sub.rename(columns={"predict": label}), on="event_id", how="left")
        n_miss = int(mega_df[label].isna().sum())
        print(f"  {label:40s}: loaded {len(sub):,} rows, missing after merge: {n_miss}")

    mega_df = mega_df.merge(
        sol154_df.rename(columns={"predict": "sol154"}), on="event_id", how="left"
    )
    n_miss_sol = int(mega_df["sol154"].isna().sum())
    print(f"  {'sol154':40s}: loaded {len(sol154_df):,} rows, missing after merge: {n_miss_sol}")

    mega_names = list(mega_sub_sources.keys()) + ["sol154"]
    mega_inputs = {}
    for name in mega_names:
        vals = mega_df[name].values.astype(np.float64)
        n_nan = int(np.isnan(vals).sum())
        if n_nan > 0:
            med = np.nanmedian(vals)
            print(f"  WARNING: {name} has {n_nan} NaN, filling with median={med:.6f}")
            vals = np.where(np.isnan(vals), med, vals)
        mega_inputs[name] = vals

    # Spearman correlations
    print("\nSpearman rank correlations (pairwise):")
    for i in range(len(mega_names)):
        for j in range(i + 1, len(mega_names)):
            r, _ = spearmanr(mega_inputs[mega_names[i]], mega_inputs[mega_names[j]])
            print(f"  {mega_names[i]:30s} vs {mega_names[j]:30s}: {r:.4f}")

    best_blend_label = f"my_best_blend ({best_combo_name})"

    # Strategy A: diverse (low correlation with sol154)
    diverse_inputs = {k: mega_inputs[k] for k in ["catboost", "extratrees", "mlp", "sol154"]}

    # Strategy B: 2-way (best_blend + sol154)
    two_way_inputs = {best_blend_label: mega_inputs[best_blend_label], "sol154": mega_inputs["sol154"]}

    # Strategy C: all 5 models + sol154
    no_blend_inputs = {k: mega_inputs[k] for k in ["catboost", "lightgbm", "xgboost", "extratrees", "mlp", "sol154"]}

    STRATEGIES = {
        "diverse": (diverse_inputs, {"catboost": 1.0, "extratrees": 1.0, "mlp": 1.0, "sol154": 1.0}),
        "2way":    (two_way_inputs, {best_blend_label: 1.0, "sol154": 1.0}),
        "5models": (no_blend_inputs, {"catboost": 1.0, "lightgbm": 1.0, "xgboost": 1.0, "extratrees": 1.0, "mlp": 1.0, "sol154": 1.0}),
    }

    print(f"\nGenerating mega-ensemble grid ({len(STRATEGIES)} strategies x {len(SOL154_GRID)} weights):")
    for strat_name, (strat_inputs, base_weights) in STRATEGIES.items():
        my_keys = [k for k in base_weights if k != "sol154"]
        my_base_total = sum(base_weights[k] for k in my_keys)
        for sol_share in SOL154_GRID:
            my_share = 1.0 - sol_share
            grid_w = {k: my_share * (base_weights[k] / my_base_total) for k in my_keys}
            grid_w["sol154"] = sol_share

            pred = mega_logit_zscore_blend(strat_inputs, grid_w, eps=1e-7)
            tag = f"mega_{strat_name}_sol{int(sol_share * 100)}"
            out = mega_df[["event_id"]].copy()
            out["predict"] = pred
            out_path = SUBMISSION_DIR / f"submission_{tag}.csv"
            out.to_csv(out_path, index=False)
            print(f"  {tag:35s} -> {out_path.name}")

    print(f"\nMega-ensemble: {len(STRATEGIES) * len(SOL154_GRID)} submissions generated!")
