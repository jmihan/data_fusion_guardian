import gc
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import polars as pl

from src.config import (
    DATA_DIR, CACHE_DIR, VAL_START,
    FORCE_REBUILD_FEATURES, FORCE_REBUILD_PRIORS,
)
from src.features.columns import (
    FINAL_FEATURE_COLS, CAT_COLS, META_COLS, MODEL_DROP_COLS,
    RISKY_LABEL_FEATURES, RISKY_PRIOR_SUFFIXES,
)
from src.features.engineering import build_features_for_part
from src.features.priors import (
    DEVICE_STAT_COLS, build_prior_tables, build_device_stats_table,
    attach_global_features,
)
from src.utils import downcast_pandas, dedupe, make_weights


def audit_and_select_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str],
    cat_cols: List[str],
) -> Tuple[List[str], pd.DataFrame]:
    rows = []
    keep_cols = []

    for col in feature_cols:
        if col not in train_df.columns or col not in test_df.columns:
            rows.append({
                "feature": col,
                "drop_reason": "missing_in_split",
                "train_nunique": np.nan,
                "test_nunique": np.nan,
                "train_missing_rate": np.nan,
                "test_missing_rate": np.nan,
                "dominant_share": np.nan,
            })
            continue

        s_tr = train_df[col]
        s_te = test_df[col]

        train_nunique = int(s_tr.nunique(dropna=False))
        test_nunique = int(s_te.nunique(dropna=False))
        train_missing_rate = float(s_tr.isna().mean())
        test_missing_rate = float(s_te.isna().mean())

        dominant_share = 0.0
        if (col in cat_cols) or (train_nunique <= 128) or (train_missing_rate >= 0.95):
            vc = s_tr.astype("object").value_counts(dropna=False, normalize=True)
            dominant_share = float(vc.iloc[0]) if len(vc) else 1.0

        drop_reason = "keep"
        if col in RISKY_LABEL_FEATURES:
            drop_reason = "risky_label_feature"
        elif any(col.endswith(sfx) for sfx in RISKY_PRIOR_SUFFIXES):
            drop_reason = "risky_label_prior"
        elif train_nunique <= 1:
            drop_reason = "constant"
        elif dominant_share >= 0.9995:
            drop_reason = "almost_constant"
        elif train_missing_rate >= 0.995 and test_missing_rate >= 0.995:
            drop_reason = "too_sparse"
        elif (col in cat_cols) and (train_missing_rate >= 0.85) and (train_nunique >= 20) and (test_nunique <= 2):
            drop_reason = "collapsed_in_test"

        rows.append({
            "feature": col,
            "drop_reason": drop_reason,
            "train_nunique": train_nunique,
            "test_nunique": test_nunique,
            "train_missing_rate": train_missing_rate,
            "test_missing_rate": test_missing_rate,
            "dominant_share": dominant_share,
        })

        if drop_reason == "keep":
            keep_cols.append(col)

    audit_df = pd.DataFrame(rows).sort_values(
        ["drop_reason", "train_missing_rate", "dominant_share", "feature"],
        ascending=[True, False, False, True],
    ).reset_index(drop=True)
    return keep_cols, audit_df


def prepare_feature_matrices(
    fit_df: pd.DataFrame,
    apply_dfs: List[pd.DataFrame],
    feature_cols: List[str],
    cat_cols: List[str],
) -> Tuple[pd.DataFrame, ...]:
    fit_X = fit_df[feature_cols].copy()
    apply_Xs = [df[feature_cols].copy() for df in apply_dfs]

    for c in cat_cols:
        fit_X[c] = fit_X[c].fillna(-1).astype(np.int64)
        for X in apply_Xs:
            X[c] = X[c].fillna(-1).astype(np.int64)

    num_cols = [c for c in feature_cols if c not in cat_cols]
    if num_cols:
        medians = fit_X[num_cols].median(numeric_only=True)
        fit_X[num_cols] = fit_X[num_cols].fillna(medians)
        for X in apply_Xs:
            X[num_cols] = X[num_cols].fillna(medians)

    return (fit_X, *apply_Xs)


def run_preparation() -> dict:
    """Full preparation pipeline: features -> split -> globals -> matrices."""

    # 1. Build features for all 3 parts
    feature_paths = []
    for part_id in [1, 2, 3]:
        feature_paths.append(build_features_for_part(part_id, force=FORCE_REBUILD_FEATURES))

    features = pl.concat([pl.scan_parquet(p) for p in feature_paths], how="vertical_relaxed").collect()
    print("Feature table shape:", features.shape)

    train_base_df = features.filter(pl.col("is_train_sample")).with_columns(
        pl.col("target_bin").cast(pl.Int8)
    ).to_pandas()
    test_base_df = features.filter(pl.col("is_test")).to_pandas()
    del features; gc.collect()

    train_base_df["event_ts"] = pd.to_datetime(train_base_df["event_ts"])
    test_base_df["event_ts"]  = pd.to_datetime(test_base_df["event_ts"])

    train_base_df = train_base_df.sort_values(["event_ts", "event_id"]).reset_index(drop=True)
    test_base_df  = test_base_df.sort_values(["event_ts", "event_id"]).reset_index(drop=True)

    downcast_pandas(train_base_df)
    downcast_pandas(test_base_df)

    BASE_FEATURE_COLS = [c for c in train_base_df.columns if c not in META_COLS and c not in MODEL_DROP_COLS]
    BASE_CAT_COLS = [c for c in CAT_COLS if c in BASE_FEATURE_COLS]

    print(f"Base features: {len(BASE_FEATURE_COLS)} | cat: {len(BASE_CAT_COLS)}")
    print(f"Train memory: {train_base_df.memory_usage(deep=True).sum() / 1e6:.0f} MB")

    # 2. Build causal globals for local validation
    print("Building causal globals for local validation...")
    device_stats_local = build_device_stats_table(tag="pre_val", cutoff=VAL_START, include_pretest=False, force=FORCE_REBUILD_PRIORS)
    prior_tables_local, prior_feature_cols = build_prior_tables(tag="pre_val", cutoff=VAL_START, force=FORCE_REBUILD_PRIORS)

    train_local_df = attach_global_features(train_base_df, device_stats_local, prior_tables_local)
    test_local_df = attach_global_features(test_base_df, device_stats_local, prior_tables_local)
    del device_stats_local, prior_tables_local; gc.collect()

    raw_feature_cols = dedupe(BASE_FEATURE_COLS + DEVICE_STAT_COLS + prior_feature_cols)
    raw_cat_feature_cols = [c for c in CAT_COLS if c in raw_feature_cols]

    FEATURE_COLS, feature_audit_df = audit_and_select_features(train_local_df, test_local_df, raw_feature_cols, raw_cat_feature_cols)
    CAT_FEATURE_COLS = [c for c in CAT_COLS if c in FEATURE_COLS]
    NUM_FEATURE_COLS = [c for c in FEATURE_COLS if c not in CAT_FEATURE_COLS]

    val_mask = train_local_df["event_ts"] >= VAL_START

    print(f"Features after cleaning: {len(FEATURE_COLS)} | cat: {len(CAT_FEATURE_COLS)} | num: {len(NUM_FEATURE_COLS)}")
    print("Dropped features by reason:")
    print(feature_audit_df.loc[feature_audit_df["drop_reason"] != "keep", "drop_reason"].value_counts())

    # 3. Split train / val
    main_train_df = train_local_df.loc[~val_mask].reset_index(drop=True)
    main_val_df = train_local_df.loc[val_mask].reset_index(drop=True)
    print(f"Main train: {len(main_train_df):,}  Main val: {len(main_val_df):,}")

    X_main_tr, X_main_val, X_test = prepare_feature_matrices(
        main_train_df, [main_val_df, test_local_df], FEATURE_COLS, CAT_FEATURE_COLS
    )

    y_main_tr = main_train_df["target_bin"].astype(np.int8).values
    y_main_val = main_val_df["target_bin"].astype(np.int8).values
    w_main_tr = make_weights(main_train_df["train_target_raw"].values, main_train_df["event_ts"].values)
    w_main_val = make_weights(main_val_df["train_target_raw"].values, main_val_df["event_ts"].values)

    downcast_pandas(X_main_tr, set(CAT_FEATURE_COLS))
    downcast_pandas(X_main_val, set(CAT_FEATURE_COLS))
    downcast_pandas(X_test, set(CAT_FEATURE_COLS))

    print(f"Features: {len(FEATURE_COLS)} | X_main_tr shape: {X_main_tr.shape}")
    print(f"Test shape: {X_test.shape}")

    return {
        "train_base_df": train_base_df,
        "test_base_df": test_base_df,
        "X_main_tr": X_main_tr,
        "X_main_val": X_main_val,
        "X_test": X_test,
        "y_main_tr": y_main_tr,
        "y_main_val": y_main_val,
        "w_main_tr": w_main_tr,
        "w_main_val": w_main_val,
        "FEATURE_COLS": FEATURE_COLS,
        "CAT_FEATURE_COLS": CAT_FEATURE_COLS,
        "main_train_df": main_train_df,
        "main_val_df": main_val_df,
        "test_local_df": test_local_df,
    }
