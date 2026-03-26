"""
Anti-fraud classification pipeline (Strazh hackathon).

Fixes applied vs original notebook:
  1. REGEX: double-backslash-d -> backslash-d (battery, os_ver, screen were always -1)
  2. accept_language restored as feature
  3. Data-leaky z-scores removed, replaced with causal amt_zscore_cust
  4. Pretest deduplication (130k duplicate rows)
  5. screen_pixels guarded against missing (-1 * -1 = 1 bug)
  6. Cross-features handled cleanly (no double int->str conversion)
  7. CatBoost submission actually generates output
  8. session_unique_mcc removed (future leakage within session)

Usage:
    python pipeline.py
"""
# %% ==================== IMPORTS ====================
from pathlib import Path
import os
import gc
import warnings

import math

import numpy as np
import pandas as pd
import polars as pl
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from catboost import CatBoostClassifier, Pool
from scipy.stats import rankdata
import optuna

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
from tqdm import tqdm

warnings.filterwarnings("ignore")
pl.Config.set_tbl_rows(12)

# %% ==================== CONFIG ====================
DATA_DIR = Path("data/raw")
CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Negative sampling: keep 1/N green events from train
NEG_SAMPLE_MOD_RECENT = 10   # after NEG_SAMPLE_BORDER
NEG_SAMPLE_MOD_OLD = 30      # before NEG_SAMPLE_BORDER
NEG_SAMPLE_BORDER_STR = "2025-04-01 00:00:00"

# Time splits
VAL_START = pd.Timestamp("2025-05-01")
RECENT_BORDER = pd.Timestamp("2025-02-01")

RANDOM_SEED = 42
FORCE_REBUILD_FEATURES = True
FORCE_REBUILD_PRIORS = True
USE_GPU = torch.cuda.is_available()
RETRAIN_ON_FULL = False
USE_OPTUNA = False
INFERENCE_ONLY = False   # True = skip all training, load models/blend from cache
OPTUNA_N_TRIALS = 40

# Chunked feature building: customers per chunk (~5-6M rows per chunk)
FEATURE_CHUNK_SIZE = 3000

# FT-Transformer
FTT_ENABLED = True
FTT_EPOCHS = 15
FTT_BATCH_SIZE = 2048
FTT_D_MODEL = 192
FTT_N_HEADS = 8
FTT_N_LAYERS = 4
FTT_ATTN_DROPOUT = 0.3
FTT_FF_DROPOUT = 0.2
FTT_MAX_LR = 1e-4
FTT_WEIGHT_DECAY = 1e-4
FTT_PATIENCE = 3
FTT_FOCAL_ALPHA = 0.9
FTT_FOCAL_GAMMA = 2.0
FTT_CB_BLEND_WEIGHT = 0.7  # weight for CatBoost in final CatBoost+FTT blend
FTT_N_SEEDS = 1            # multi-seed ensemble
FTT_MIXUP_ALPHA = 0.2      # mixup Beta distribution parameter (0 = disabled)

# %% ==================== COLUMN DEFINITIONS ====================
# Raw columns loaded from parquet (accept_language restored)
BASE_COLS = [
    "customer_id", "event_id", "event_dttm",
    "event_type_nm", "event_desc",
    "channel_indicator_type", "channel_indicator_sub_type",
    "operaton_amt", "currency_iso_cd", "mcc_code", "pos_cd",
    "accept_language",
    "timezone", "session_id", "operating_system_type",
    "battery", "device_system_version", "screen_size",
    "developer_tools", "phone_voip_call_state",
    "web_rdp_connection", "compromised",
]

# Final feature columns for modelling
FEATURE_COLS = [
    # --- categorical (string for CatBoost) ---
    # NOTE: customer_id moved to META_COLS (was 36% importance in rg model = memorisation)
    "event_type_nm", "event_desc",
    "channel_indicator_sub_type",
    "mcc_code_i", "pos_cd", "timezone",
    "operating_system_type", "phone_voip_call_state",
    "lang_primary",
    "cross_type_mcc", "cross_mcc_channel", "cross_os_compromised",
    "cross_hour_channel", "cross_type_currency",
    # --- numeric ---
    "amt", "amt_log_abs", "amt_is_negative",
    "hour", "weekday", "day", "month", "is_weekend",
    "battery_pct",
    "os_ver_major", "screen_w", "screen_h", "screen_pixels", "screen_ratio",
    "is_night", "is_late_night", "is_evening_peak", "is_suspicious_lang",
    # accept_language & compromised null features
    "is_accept_lang_null", "lang_count", "is_compromised_null",
    # sequential (customer history, causal only)
    "cust_prev_events", "cust_prev_amt_mean", "cust_prev_amt_std",
    "sec_since_prev_event", "amt_delta_prev", "amt_zscore_cust",
    "cnt_prev_same_type", "cnt_prev_same_desc", "cnt_prev_same_mcc",
    "cnt_prev_same_subtype", "cnt_prev_same_session",
    "sec_since_prev_same_type", "sec_since_prev_same_desc",
    "events_before_today", "minutes_from_session_start",
    # novelty features (first-time actions)
    "is_new_mcc", "is_new_type", "is_new_desc", "is_new_subtype",
    # burst / velocity features
    "is_burst_5min", "is_rapid_1min", "events_this_hour", "event_rate_today",
    "events_last_5min", "events_last_60min", "amt_sum_last_60min",
    # amount pattern features
    "amt_vs_cust_max", "is_round_amount",
    # account age
    "days_since_first_event",
]

CAT_FEATURES = [
    "event_type_nm", "event_desc",
    "channel_indicator_sub_type",
    "mcc_code_i", "pos_cd", "timezone",
    "operating_system_type", "phone_voip_call_state",
    "lang_primary",
    "cross_type_mcc", "cross_mcc_channel", "cross_os_compromised",
    "cross_hour_channel", "cross_type_currency",
]

META_COLS = [
    "customer_id",  # needed for prior joins & OOF splits, NOT a model feature
    "event_id", "period", "event_ts",
    "is_train_sample", "is_test", "train_target_raw", "target_bin",
]


# %% ==================== 1. FEATURE ENGINEERING ====================
def _load_all_periods_chunked(part_id: int, cust_series: pl.Series) -> pl.LazyFrame:
    """Load pretrain+train+pretest+test for a SUBSET of customers in one partition."""
    pretrain = (
        pl.scan_parquet(DATA_DIR / f"pretrain_part_{part_id}.parquet")
        .filter(pl.col("customer_id").is_in(cust_series))
        .select(BASE_COLS)
        .with_columns(pl.lit("pretrain").alias("period"))
    )
    train = (
        pl.scan_parquet(DATA_DIR / f"train_part_{part_id}.parquet")
        .filter(pl.col("customer_id").is_in(cust_series))
        .select(BASE_COLS)
        .with_columns(pl.lit("train").alias("period"))
    )
    # FIX: deduplicate pretest (130k duplicate rows in source data)
    pretest = (
        pl.scan_parquet(DATA_DIR / "pretest.parquet")
        .filter(pl.col("customer_id").is_in(cust_series))
        .select(BASE_COLS)
        .unique()
        .with_columns(pl.lit("pretest").alias("period"))
    )
    test = (
        pl.scan_parquet(DATA_DIR / "test.parquet")
        .filter(pl.col("customer_id").is_in(cust_series))
        .select(BASE_COLS)
        .with_columns(pl.lit("test").alias("period"))
    )
    return pl.concat([pretrain, train, pretest, test], how="vertical_relaxed")


def _build_features_for_chunk(lf: pl.LazyFrame, labels_lf: pl.LazyFrame) -> pl.DataFrame:
    """Build all features for a LazyFrame chunk. Returns collected DataFrame of train+test rows only."""

    # ---- Parse raw columns (use smaller types where possible) ----
    lf = lf.with_columns([
        pl.col("event_dttm").str.strptime(
            pl.Datetime, format="%Y-%m-%d %H:%M:%S", strict=False
        ).alias("event_ts"),

        pl.col("operaton_amt").cast(pl.Float32).alias("amt"),
        pl.col("session_id").cast(pl.Int32, strict=False).fill_null(-1),

        pl.col("event_type_nm").cast(pl.Int32, strict=False).fill_null(-1),
        pl.col("event_desc").cast(pl.Int32, strict=False).fill_null(-1),
        pl.col("channel_indicator_type").cast(pl.Int16, strict=False).fill_null(-1),
        pl.col("channel_indicator_sub_type").cast(pl.Int16, strict=False).fill_null(-1),
        pl.col("currency_iso_cd").cast(pl.Int16, strict=False).fill_null(-1),
        pl.col("pos_cd").cast(pl.Int16, strict=False).fill_null(-1),
        pl.col("timezone").cast(pl.Int32, strict=False).fill_null(-1),
        pl.col("operating_system_type").cast(pl.Int16, strict=False).fill_null(-1),
        pl.col("phone_voip_call_state").cast(pl.Int8, strict=False).fill_null(-1),
        pl.col("web_rdp_connection").cast(pl.Int8, strict=False).fill_null(-1),

        pl.col("mcc_code").cast(pl.Int32, strict=False).fill_null(-1).alias("mcc_code_i"),

        pl.col("battery").str.extract(r"(\d{1,3})", 1)
            .cast(pl.Int16, strict=False).fill_null(-1).alias("battery_pct"),
        pl.col("device_system_version").str.extract(r"^(\d+)", 1)
            .cast(pl.Int16, strict=False).fill_null(-1).alias("os_ver_major"),
        pl.col("screen_size").str.extract(r"^(\d+)", 1)
            .cast(pl.Int16, strict=False).fill_null(-1).alias("screen_w"),
        pl.col("screen_size").str.extract(r"x(\d+)$", 1)
            .cast(pl.Int16, strict=False).fill_null(-1).alias("screen_h"),

        pl.col("developer_tools").cast(pl.Int8, strict=False).fill_null(-1).alias("developer_tools_i"),
        pl.col("compromised").cast(pl.Int8, strict=False).fill_null(-1).alias("compromised_i"),

        pl.col("accept_language")
            .str.extract(r"^([a-zA-Z]{2}(?:-[a-zA-Z]{2})?)", 1)
            .fill_null("unknown")
            .alias("lang_primary"),
        (pl.col("accept_language").fill_null("").str.to_lowercase()
            .str.contains(r"(?:en-us|\*)")).cast(pl.Int8)
            .alias("is_suspicious_lang"),
        pl.col("accept_language").is_null().cast(pl.Int8)
            .alias("is_accept_lang_null"),
        pl.when(pl.col("accept_language").is_not_null())
            .then(pl.col("accept_language").str.count_matches(r",") + 1)
            .otherwise(pl.lit(0))
            .cast(pl.Int8).alias("lang_count"),

        pl.col("compromised").is_null().cast(pl.Int8)
            .alias("is_compromised_null"),
    ])

    lf = lf.drop([
        "event_dttm", "operaton_amt", "mcc_code", "battery",
        "device_system_version", "screen_size", "developer_tools",
        "compromised", "accept_language",
    ])

    lf = lf.sort(["customer_id", "event_ts", "event_id"])

    # ---- Join labels ----
    lf = lf.join(labels_lf, on="event_id", how="left")
    lf = lf.with_columns(
        pl.when(pl.col("period") == "train")
          .then(
              pl.when(pl.col("target").is_null())
                .then(pl.lit(-1))
                .otherwise(pl.col("target"))
          )
          .otherwise(pl.lit(None))
          .alias("train_target_raw")
    )

    # ---- Negative sampling ----
    border_expr = pl.lit(NEG_SAMPLE_BORDER_STR).str.strptime(
        pl.Datetime, format="%Y-%m-%d %H:%M:%S", strict=False
    )
    lf = lf.with_columns(
        (
            (pl.col("period") == "train")
            & (pl.col("train_target_raw") == -1)
            & (
                (
                    (pl.col("event_ts") >= border_expr)
                    & ((pl.struct(["event_id", "customer_id"]).hash(seed=RANDOM_SEED)
                        % NEG_SAMPLE_MOD_RECENT) == 0)
                ) | (
                    (pl.col("event_ts") < border_expr)
                    & ((pl.struct(["event_id", "customer_id"]).hash(seed=RANDOM_SEED + 17)
                        % NEG_SAMPLE_MOD_OLD) == 0)
                )
            )
        ).alias("keep_green")
    )

    lf = lf.with_columns([
        (
            (pl.col("period") == "train")
            & ((pl.col("train_target_raw") != -1) | pl.col("keep_green"))
        ).alias("is_train_sample"),
        (pl.col("period") == "test").alias("is_test"),
    ])

    # ---- Time features ----
    lf = lf.with_columns([
        pl.col("event_ts").dt.hour().cast(pl.Int8).alias("hour"),
        pl.col("event_ts").dt.weekday().cast(pl.Int8).alias("weekday"),
        pl.col("event_ts").dt.day().cast(pl.Int8).alias("day"),
        pl.col("event_ts").dt.month().cast(pl.Int8).alias("month"),
        (pl.col("event_ts").dt.weekday() >= 5).cast(pl.Int8).alias("is_weekend"),
        (pl.col("event_ts").dt.epoch("s") // 86400).cast(pl.Int32).alias("event_day_number"),
        pl.col("event_ts").dt.date().alias("event_date"),
        pl.col("event_ts").dt.hour()
            .is_in([0, 1, 2, 3, 4, 5, 22, 23]).cast(pl.Int8).alias("is_night"),
        pl.col("event_ts").dt.hour()
            .is_in([0, 1, 2, 3]).cast(pl.Int8).alias("is_late_night"),
        pl.col("event_ts").dt.hour()
            .is_in([19, 20, 21, 22, 23]).cast(pl.Int8).alias("is_evening_peak"),
    ])

    # ---- Amount features ----
    lf = lf.with_columns([
        pl.col("amt").abs().log1p().cast(pl.Float32).alias("amt_log_abs"),
        (pl.col("amt") < 0).cast(pl.Int8).alias("amt_is_negative"),
    ])

    # ---- Screen features ----
    lf = lf.with_columns([
        pl.when((pl.col("screen_w") > 0) & (pl.col("screen_h") > 0))
            .then(pl.col("screen_w").cast(pl.Int32) * pl.col("screen_h").cast(pl.Int32))
            .otherwise(-1)
            .alias("screen_pixels"),
        pl.when((pl.col("screen_w") > 0) & (pl.col("screen_h") > 0))
            .then(pl.col("screen_w").cast(pl.Float32) / pl.col("screen_h").cast(pl.Float32))
            .otherwise(0.0)
            .alias("screen_ratio"),
    ])

    # ---- Sequential customer history features (strictly causal) ----
    lf = lf.with_columns([
        pl.cum_count("event_id").over("customer_id")
            .cast(pl.Int32).alias("cust_event_idx"),
        pl.col("amt").cum_sum().over("customer_id").alias("cust_cum_amt"),
        (pl.col("amt") * pl.col("amt")).cum_sum().over("customer_id")
            .alias("cust_cum_amt_sq"),
        pl.col("event_ts").shift(1).over("customer_id").alias("prev_event_ts"),
        pl.col("amt").shift(1).over("customer_id").alias("prev_amt"),

        (pl.cum_count("event_id").over(["customer_id", "event_type_nm"]) - 1)
            .cast(pl.Int16).alias("cnt_prev_same_type"),
        (pl.cum_count("event_id").over(["customer_id", "event_desc"]) - 1)
            .cast(pl.Int16).alias("cnt_prev_same_desc"),
        (pl.cum_count("event_id").over(["customer_id", "mcc_code_i"]) - 1)
            .cast(pl.Int16).alias("cnt_prev_same_mcc"),
        (pl.cum_count("event_id").over(["customer_id", "channel_indicator_sub_type"]) - 1)
            .cast(pl.Int16).alias("cnt_prev_same_subtype"),
        (pl.cum_count("event_id").over(["customer_id", "session_id"]) - 1)
            .cast(pl.Int16).alias("cnt_prev_same_session"),

        pl.col("event_ts").shift(1).over(["customer_id", "event_type_nm"])
            .alias("prev_same_type_ts"),
        pl.col("event_ts").shift(1).over(["customer_id", "event_desc"])
            .alias("prev_same_desc_ts"),

        (pl.cum_count("event_id").over(["customer_id", "event_date"]) - 1)
            .cast(pl.Int16).alias("events_before_today"),

        (pl.cum_count("event_id").over(["customer_id", "event_date", "hour"]) - 1)
            .cast(pl.Int16).alias("events_this_hour"),

        pl.col("event_ts").min().over("customer_id").alias("_cust_first_ts"),

        pl.col("amt").cum_max().over("customer_id").alias("_cust_cum_max_incl"),
    ])

    # ---- Derived sequential features ----
    lf = lf.with_columns([
        (pl.col("cust_event_idx") - 1).cast(pl.Int32).alias("cust_prev_events"),

        pl.when(pl.col("cust_event_idx") > 1)
            .then(
                (pl.col("cust_cum_amt") - pl.col("amt"))
                / (pl.col("cust_event_idx") - 1)
            )
            .otherwise(0.0)
            .cast(pl.Float32)
            .alias("cust_prev_amt_mean"),

        pl.when(pl.col("prev_event_ts").is_not_null())
            .then((pl.col("event_ts") - pl.col("prev_event_ts")).dt.total_seconds())
            .otherwise(-1)
            .cast(pl.Int32)
            .alias("sec_since_prev_event"),

        (pl.col("amt") - pl.col("prev_amt").fill_null(0.0))
            .cast(pl.Float32).alias("amt_delta_prev"),

        pl.when(pl.col("prev_same_type_ts").is_not_null())
            .then((pl.col("event_ts") - pl.col("prev_same_type_ts")).dt.total_seconds())
            .otherwise(-1)
            .cast(pl.Int32)
            .alias("sec_since_prev_same_type"),

        pl.when(pl.col("prev_same_desc_ts").is_not_null())
            .then((pl.col("event_ts") - pl.col("prev_same_desc_ts")).dt.total_seconds())
            .otherwise(-1)
            .cast(pl.Int32)
            .alias("sec_since_prev_same_desc"),

        pl.when(pl.col("session_id") != -1)
            .then(
                (pl.col("event_ts") - pl.col("event_ts").min().over("session_id"))
                .dt.total_minutes()
            )
            .otherwise(-1.0)
            .cast(pl.Float32)
            .alias("minutes_from_session_start"),
    ])

    # ---- Running std of past amounts ----
    lf = lf.with_columns(
        pl.when(pl.col("cust_event_idx") > 2)
            .then(
                (
                    (
                        (pl.col("cust_cum_amt_sq") - pl.col("amt") * pl.col("amt"))
                        / (pl.col("cust_event_idx") - 1)
                    )
                    - pl.col("cust_prev_amt_mean") ** 2
                ).clip(lower_bound=0).sqrt()
            )
            .otherwise(0.0)
            .cast(pl.Float32)
            .alias("cust_prev_amt_std")
    )

    lf = lf.with_columns(
        pl.when(pl.col("cust_prev_amt_std") > 0)
            .then(
                (pl.col("amt") - pl.col("cust_prev_amt_mean"))
                / (pl.col("cust_prev_amt_std") + 1e-6)
            )
            .otherwise(0.0)
            .cast(pl.Float32)
            .alias("amt_zscore_cust")
    )

    # ---- Novelty features ----
    lf = lf.with_columns([
        (pl.col("cnt_prev_same_mcc") == 0).cast(pl.Int8).alias("is_new_mcc"),
        (pl.col("cnt_prev_same_type") == 0).cast(pl.Int8).alias("is_new_type"),
        (pl.col("cnt_prev_same_desc") == 0).cast(pl.Int8).alias("is_new_desc"),
        (pl.col("cnt_prev_same_subtype") == 0).cast(pl.Int8).alias("is_new_subtype"),
    ])

    # ---- Burst / velocity proxy features ----
    lf = lf.with_columns([
        pl.when((pl.col("sec_since_prev_event") > 0) & (pl.col("sec_since_prev_event") < 300))
            .then(pl.lit(1)).otherwise(pl.lit(0)).cast(pl.Int8).alias("is_burst_5min"),
        pl.when((pl.col("sec_since_prev_event") > 0) & (pl.col("sec_since_prev_event") < 60))
            .then(pl.lit(1)).otherwise(pl.lit(0)).cast(pl.Int8).alias("is_rapid_1min"),
        pl.when(pl.col("hour") > 0)
            .then(pl.col("events_before_today").cast(pl.Float32) / pl.col("hour").cast(pl.Float32))
            .otherwise(pl.col("events_before_today").cast(pl.Float32))
            .alias("event_rate_today"),
    ])

    # ---- Amount pattern features ----
    lf = lf.with_columns([
        pl.col("_cust_cum_max_incl").shift(1).over("customer_id").alias("_cust_prev_max"),
    ])
    lf = lf.with_columns([
        pl.when(pl.col("_cust_prev_max").is_not_null() & (pl.col("_cust_prev_max").abs() > 1.0))
            .then(pl.col("amt") / (pl.col("_cust_prev_max") + 1e-6))
            .otherwise(0.0)
            .cast(pl.Float32).alias("amt_vs_cust_max"),
        pl.when(pl.col("amt").abs() > 0)
            .then((pl.col("amt").abs() % 10000 == 0).cast(pl.Int8))
            .otherwise(pl.lit(0).cast(pl.Int8))
            .alias("is_round_amount"),
    ])

    # ---- Account age ----
    lf = lf.with_columns(
        pl.when(pl.col("_cust_first_ts").is_not_null())
            .then((pl.col("event_ts") - pl.col("_cust_first_ts")).dt.total_days())
            .otherwise(0)
            .cast(pl.Int16).alias("days_since_first_event")
    )

    # ---- Drop intermediate columns ----
    lf = lf.drop(["_cust_cum_max_incl", "_cust_prev_max", "_cust_first_ts"])

    # ---- Velocity features via join_asof ----
    _vel_right = lf.select([
        pl.col("customer_id"),
        pl.col("event_ts").alias("_ref_ts"),
        pl.col("cust_event_idx").alias("_ref_cidx"),
        pl.col("cust_cum_amt").alias("_ref_cum_amt"),
    ])

    for period_min, col_name, amt_col in [
        (5,  "events_last_5min",  None),
        (60, "events_last_60min", "amt_sum_last_60min"),
    ]:
        _vel_left = lf.select([
            "customer_id", "event_id",
            (pl.col("event_ts") - pl.duration(minutes=period_min)).alias("_ts_ago"),
            pl.col("cust_event_idx"),
            pl.col("cust_cum_amt"),
            pl.col("amt"),
        ])
        _vel_joined = _vel_left.join_asof(
            _vel_right,
            left_on="_ts_ago",
            right_on="_ref_ts",
            by="customer_id",
            strategy="backward",
        )
        agg_exprs = [
            pl.when(pl.col("_ref_cidx").is_not_null())
                .then(pl.col("cust_event_idx") - pl.col("_ref_cidx") - 1)
                .otherwise(pl.col("cust_event_idx") - 1)
                .cast(pl.Int16).alias(col_name),
        ]
        if amt_col:
            agg_exprs.append(
                pl.when(pl.col("_ref_cum_amt").is_not_null())
                    .then(pl.col("cust_cum_amt") - pl.col("amt") - pl.col("_ref_cum_amt"))
                    .otherwise(pl.col("cust_cum_amt") - pl.col("amt"))
                    .cast(pl.Float32).alias(amt_col)
            )
        sel_cols = ["event_id", col_name] + ([amt_col] if amt_col else [])
        _vel_result = _vel_joined.with_columns(agg_exprs).select(sel_cols)
        lf = lf.join(_vel_result, on="event_id", how="left")

    # ---- Cross features ----
    lf = lf.with_columns([
        (pl.col("event_type_nm").cast(pl.Utf8) + "_x_"
         + pl.col("mcc_code_i").cast(pl.Utf8)).alias("cross_type_mcc"),
        (pl.col("mcc_code_i").cast(pl.Utf8) + "_x_"
         + pl.col("channel_indicator_type").cast(pl.Utf8)).alias("cross_mcc_channel"),
        (pl.col("operating_system_type").cast(pl.Utf8) + "_x_"
         + pl.col("compromised_i").cast(pl.Utf8)).alias("cross_os_compromised"),
        (pl.col("hour").cast(pl.Utf8) + "_x_"
         + pl.col("channel_indicator_type").cast(pl.Utf8)).alias("cross_hour_channel"),
        (pl.col("event_type_nm").cast(pl.Utf8) + "_x_"
         + pl.col("currency_iso_cd").cast(pl.Utf8)).alias("cross_type_currency"),
    ])

    # ---- Target binary ----
    lf = lf.with_columns(
        pl.when(pl.col("is_train_sample"))
            .then((pl.col("train_target_raw") == 1).cast(pl.Int8))
            .otherwise(pl.lit(None))
            .alias("target_bin")
    )

    # ---- Select train+test rows only, collect ----
    select_cols = META_COLS + FEATURE_COLS
    return (
        lf.filter(pl.col("is_train_sample") | pl.col("is_test"))
          .select(select_cols)
          .collect()
    )


def build_features_for_part(
    part_id: int, labels_lf: pl.LazyFrame, force: bool = False,
    chunk_size: int = FEATURE_CHUNK_SIZE,
) -> Path:
    """Build features for one partition, processing customers in chunks to limit memory."""
    out_path = CACHE_DIR / f"features_v2_part_{part_id}.parquet"
    if out_path.exists() and not force:
        print(f"  [part {part_id}] cache hit: {out_path.name}")
        return out_path

    print(f"  [part {part_id}] building features (chunked, {chunk_size} customers/chunk) ...")

    # Get all unique customer IDs for this partition
    custs = (
        pl.scan_parquet(DATA_DIR / f"pretrain_part_{part_id}.parquet")
        .select("customer_id").unique().collect()
        .get_column("customer_id").to_list()
    )
    print(f"  [part {part_id}] {len(custs):,} customers total")

    chunks = [custs[i:i+chunk_size] for i in range(0, len(custs), chunk_size)]
    results = []

    for ci, chunk_custs in enumerate(chunks):
        print(f"  [part {part_id}] chunk {ci+1}/{len(chunks)} "
              f"({len(chunk_custs)} customers) ...", end=" ", flush=True)

        cust_series = pl.Series("customer_id", chunk_custs)
        lf = _load_all_periods_chunked(part_id, cust_series)
        chunk_df = _build_features_for_chunk(lf, labels_lf)

        print(f"-> {chunk_df.height:,} rows")
        results.append(chunk_df)

        del chunk_df, lf
        gc.collect()

    out_df = pl.concat(results)
    out_df.write_parquet(out_path, compression="zstd")

    n_tr = out_df.filter(pl.col("is_train_sample")).height
    n_te = out_df.filter(pl.col("is_test")).height
    print(f"  [part {part_id}] done: {out_df.height:,} rows (train={n_tr:,}, test={n_te:,})")

    del out_df, results
    gc.collect()
    return out_path


# %% ==================== 2. CATEGORY PRIORS ====================
PRIOR_DEFS = {
    "event_desc":
        pl.col("event_desc").cast(pl.Int32, strict=False).fill_null(-1),
    "mcc_code_i":
        pl.col("mcc_code").cast(pl.Int32, strict=False).fill_null(-1).alias("mcc_code_i"),
    "timezone":
        pl.col("timezone").cast(pl.Int32, strict=False).fill_null(-1),
    "operating_system_type":
        pl.col("operating_system_type").cast(pl.Int16, strict=False).fill_null(-1),
    "channel_indicator_sub_type":
        pl.col("channel_indicator_sub_type").cast(pl.Int16, strict=False).fill_null(-1),
    "event_type_nm":
        pl.col("event_type_nm").cast(pl.Int32, strict=False).fill_null(-1),
    "pos_cd":
        pl.col("pos_cd").cast(pl.Int16, strict=False).fill_null(-1),
    # cross priors
    "cross_type_mcc": (
        pl.col("event_type_nm").cast(pl.Int32, strict=False).fill_null(-1).cast(pl.Utf8)
        + pl.lit("_x_")
        + pl.col("mcc_code").cast(pl.Int32, strict=False).fill_null(-1).cast(pl.Utf8)
    ).alias("cross_type_mcc"),
    "cross_mcc_channel": (
        pl.col("mcc_code").cast(pl.Int32, strict=False).fill_null(-1).cast(pl.Utf8)
        + pl.lit("_x_")
        + pl.col("channel_indicator_type").cast(pl.Int16, strict=False).fill_null(-1).cast(pl.Utf8)
    ).alias("cross_mcc_channel"),
    "cross_os_compromised": (
        pl.col("operating_system_type").cast(pl.Int16, strict=False).fill_null(-1).cast(pl.Utf8)
        + pl.lit("_x_")
        + pl.col("compromised").cast(pl.Int8, strict=False).fill_null(-1).cast(pl.Utf8)
    ).alias("cross_os_compromised"),
}


def build_prior_table(
    key_name: str,
    expr: pl.Expr,
    labels_lf: pl.LazyFrame,
    force: bool = False,
) -> pl.DataFrame:
    """Compute smoothed target-encoding priors for a categorical key."""
    out_path = CACHE_DIR / f"prior_v2_{key_name}.parquet"
    if out_path.exists() and not force:
        return pl.read_parquet(out_path)

    print(f"  Building priors: {key_name}")
    lf = pl.concat([
        pl.scan_parquet(DATA_DIR / f"train_part_{i}.parquet")
          .select([pl.col("event_id"), expr])
        for i in [1, 2, 3]
    ], how="vertical_relaxed")

    cnt_col = f"prior_{key_name}_cnt"
    lbl_col = f"prior_{key_name}_lbl"
    red_col = f"prior_{key_name}_red"

    total = lf.group_by(key_name).len().rename({"len": cnt_col})

    labeled = (
        lf.join(labels_lf, on="event_id", how="inner")
          .group_by(key_name)
          .agg([
              pl.len().alias(lbl_col),
              pl.sum("target").cast(pl.Float64).alias(red_col),
          ])
    )

    prior = (
        total.join(labeled, on=key_name, how="left")
            .with_columns([
                pl.col(lbl_col).fill_null(0.0),
                pl.col(red_col).fill_null(0.0),
            ])
            .with_columns([
                # Smoothed rates (Laplace-like smoothing)
                ((pl.col(red_col) + 1.0) / (pl.col(cnt_col) + 200.0))
                    .cast(pl.Float32).alias(f"prior_{key_name}_red_rate"),
                ((pl.col(lbl_col) + 1.0) / (pl.col(cnt_col) + 200.0))
                    .cast(pl.Float32).alias(f"prior_{key_name}_labeled_rate"),
                ((pl.col(red_col) + 1.0) / (pl.col(lbl_col) + 2.0))
                    .cast(pl.Float32).alias(f"prior_{key_name}_red_share"),
            ])
            .select([
                key_name, cnt_col,
                f"prior_{key_name}_red_rate",
                f"prior_{key_name}_labeled_rate",
                f"prior_{key_name}_red_share",
            ])
            .collect()
    )

    prior.write_parquet(out_path, compression="zstd")
    return prior


# %% ==================== 3. DATA PREPARATION ====================
def downcast_pandas(df, cat_cols_set=None):
    """Downcast numeric columns to smallest possible types to save memory."""
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


def prepare_train_test(
    train_pl: pl.DataFrame, test_pl: pl.DataFrame, prior_cols: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str], list[str]]:
    """Convert polars to pandas, handle types, split features by kind."""
    print(f"  Train: {train_pl.shape}, Test: {test_pl.shape}")

    train_df = train_pl.to_pandas()
    del train_pl
    gc.collect()

    test_df = test_pl.to_pandas()
    del test_pl
    gc.collect()

    train_df["event_ts"] = pd.to_datetime(train_df["event_ts"])
    test_df["event_ts"] = pd.to_datetime(test_df["event_ts"])

    # Build column lists
    feature_cols = [c for c in FEATURE_COLS if c in train_df.columns]
    feature_cols += [c for c in prior_cols if c in train_df.columns]
    cat_cols = [c for c in CAT_FEATURES if c in feature_cols]
    num_cols = [c for c in feature_cols if c not in cat_cols]

    # Categoricals -> string (CatBoost native cat handling)
    for c in cat_cols:
        for df in [train_df, test_df]:
            df[c] = (
                df[c].astype(str)
                     .replace({"nan": "MISSING", "None": "MISSING", "": "MISSING"})
                     .fillna("MISSING")
            )

    # Numerics -> float, fill NaN with train medians
    for c in num_cols:
        for df in [train_df, test_df]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    medians = train_df[num_cols].median(numeric_only=True)
    train_df[num_cols] = train_df[num_cols].fillna(medians)
    test_df[num_cols] = test_df[num_cols].fillna(medians)

    # --- Post-cache derived features (no cache rebuild needed) ---
    for df in [train_df, test_df]:
        abs_amt = df["amt"].abs()
        # Round amount tiers (10000 alone misses 99% of round amounts)
        df["is_round_100"] = ((abs_amt > 0) & (abs_amt % 100 < 0.01)).astype(np.int8)
        df["is_round_1000"] = ((abs_amt > 0) & (abs_amt % 1000 < 0.01)).astype(np.int8)
        # Amount × velocity (fraud = rapid high-value bursts)
        df["amt_x_velocity_60"] = (
            df["amt_log_abs"] * df["events_last_60min"].fillna(0)
        ).astype(np.float32)
        # Amount at night (fraud peaks at night with larger amounts)
        df["amt_x_night"] = (
            df["amt_log_abs"] * df["is_night"]
        ).astype(np.float32)
        # Transaction speed: high amt with short inter-event gap
        spe = df["sec_since_prev_event"].astype(np.float32)
        df["amt_speed"] = np.where(
            spe > 0, df["amt_log_abs"] / (spe + 1.0), 0.0
        ).astype(np.float32)

    post_num_cols = [
        "is_round_100", "is_round_1000",
        "amt_x_velocity_60", "amt_x_night", "amt_speed",
    ]
    feature_cols.extend(post_num_cols)
    num_cols.extend(post_num_cols)

    # Downcast numeric types to save memory
    cat_cols_set = set(cat_cols)
    train_df = downcast_pandas(train_df, cat_cols_set)
    test_df = downcast_pandas(test_df, cat_cols_set)

    # Sort chronologically for time-based validation
    train_df = train_df.sort_values("event_ts").reset_index(drop=True)

    print(f"  Features: {len(feature_cols)} (cat={len(cat_cols)}, num={len(num_cols)})")
    print(f"  Train memory: {train_df.memory_usage(deep=True).sum() / 1024**3:.2f} GB")
    print(f"  Test memory: {test_df.memory_usage(deep=True).sum() / 1024**3:.2f} GB")
    return train_df, test_df, feature_cols, cat_cols, num_cols


# %% ==================== 4. CATBOOST UTILS ====================
def make_weights(
    raw_target: np.ndarray,
    w_red: float = 10.0,
    w_yellow: float = 2.5,
) -> np.ndarray:
    """Sample weights: fraud=w_red, confirmed-suspicious=w_yellow, green=1."""
    return np.where(
        raw_target == 1, w_red,
        np.where(raw_target == 0, w_yellow, 1.0)
    ).astype(np.float32)


def train_catboost(
    X_tr, y_tr, w_tr,
    X_val, y_val, w_val,
    cat_cols: list[str],
    params: dict,
    use_gpu: bool = True,
    name: str = "model",
    save_path: str | None = None,
    full_val_X=None,
) -> tuple[str | None, int, float, dict, np.ndarray]:
    """Train CatBoost with holdout + early stopping, or load from cache.

    Returns (save_path, best_iter, val_ap, params, blend_val_pred).
    If full_val_X is provided, blend_val_pred is predictions on full_val_X
    (for blend optimization when model trains on a subset).
    Model is saved to disk and GPU memory is freed immediately.
    """
    if save_path is None:
        save_path = f"cache/cb_{name}.cbm"

    val_pool = Pool(X_val, y_val, weight=w_val, cat_features=cat_cols)

    # --- Load from cache if exists ---
    if Path(save_path).exists():
        print(f"  [{name}] loading cached model from {save_path}")
        model = CatBoostClassifier()
        model.load_model(save_path)
    else:
        # --- Train from scratch ---
        params = params.copy()
        params.update({
            "loss_function": "Logloss",
            "eval_metric": "PRAUC",
            "random_seed": RANDOM_SEED,
            "allow_writing_files": False,
            "verbose": 200,
            "metric_period": 100,
        })
        if use_gpu:
            params.update({
                "task_type": "GPU",
                "devices": "0",
                "gpu_ram_part": 0.8,
                "border_count": 64,
                "max_ctr_complexity": 1,
            })
        else:
            params.update({
                "task_type": "CPU",
                "thread_count": max(1, (os.cpu_count() or 4) - 1),
            })

        train_pool = Pool(X_tr, y_tr, weight=w_tr, cat_features=cat_cols)

        try:
            model = CatBoostClassifier(**params)
            model.fit(train_pool, eval_set=val_pool, use_best_model=True)
        except Exception as e:
            print(f"  [{name}] GPU failed ({e}), fallback to CPU")
            params["task_type"] = "CPU"
            params.pop("devices", None)
            params.pop("gpu_ram_part", None)
            params.pop("border_count", None)
            params.pop("max_ctr_complexity", None)
            params["thread_count"] = max(1, (os.cpu_count() or 4) - 1)
            model = CatBoostClassifier(**params)
            model.fit(train_pool, eval_set=val_pool, use_best_model=True)

        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        model.save_model(save_path)
        del train_pool

    val_pred = model.predict(val_pool, prediction_type="RawFormulaVal")
    val_ap = average_precision_score(y_val, val_pred)
    best_iter = model.get_best_iteration()
    if best_iter is None or best_iter <= 0:
        best_iter = params.get("iterations", 1000)

    # Predictions for blend optimization (on full val set if model was trained on subset)
    if full_val_X is not None:
        blend_pred = model.predict(
            Pool(full_val_X, cat_features=cat_cols), prediction_type="RawFormulaVal"
        )
    else:
        blend_pred = val_pred

    print(f"  [{name}] best_iter={best_iter}, val PR-AUC={val_ap:.6f}")

    del model, val_pool
    gc.collect()

    return save_path, best_iter, val_ap, params, blend_pred


def refit_catboost(
    X, y, w,
    cat_cols: list[str],
    base_params: dict,
    best_iter: int,
) -> CatBoostClassifier:
    """Retrain on full data without early stopping."""
    params = base_params.copy()
    params.pop("od_type", None)
    params.pop("od_wait", None)
    params["iterations"] = max(300, int(best_iter))

    y_arr = np.asarray(y)
    w_arr = np.asarray(w, dtype=np.float32)
    if w_arr.ndim == 0 or w_arr.shape[0] != len(y_arr):
        w_arr = np.ones(len(y_arr), dtype=np.float32)

    pool = Pool(X, y_arr, weight=w_arr, cat_features=cat_cols)
    model = CatBoostClassifier(**params)
    model.fit(pool, verbose=200)
    return model


# %% ==================== 4b. OPTUNA TUNING ====================
def run_optuna_tuning(
    X_tr, y_tr, raw_target_tr,
    X_val, y_val,
    cat_cols: list[str],
    n_trials: int = 40,
    use_gpu: bool = True,
) -> tuple[dict, float, float]:
    """Run Optuna to find optimal CatBoost hyperparameters + sample weights."""
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # Subsample green events for faster tuning (keep 20%)
    labeled_idx = np.where(raw_target_tr != -1)[0]
    green_idx = np.where(raw_target_tr == -1)[0]
    np.random.seed(RANDOM_SEED)
    sampled_green = np.random.choice(
        green_idx, size=int(len(green_idx) * 0.2), replace=False
    )
    tune_idx = np.concatenate([labeled_idx, sampled_green])
    X_tune = X_tr.iloc[tune_idx]
    y_tune = y_tr[tune_idx]
    raw_tune = raw_target_tr[tune_idx]

    val_pool = Pool(X_val, y_val, cat_features=cat_cols)

    def objective(trial):
        w_red = trial.suggest_float("w_red", 5.0, 25.0)
        w_yellow = trial.suggest_float("w_yellow", 1.0, 6.0)
        w_tune = make_weights(raw_tune, w_red=w_red, w_yellow=w_yellow)

        params = {
            "iterations": 800,
            "depth": trial.suggest_int("depth", 5, 9),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.12, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 30.0),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 3.0),
            "loss_function": "Logloss",
            "eval_metric": "PRAUC",
            "random_seed": RANDOM_SEED,
            "allow_writing_files": False,
            "verbose": 0,
            "od_type": "Iter",
            "od_wait": 80,
        }
        if use_gpu:
            params.update({
                "task_type": "GPU",
                "devices": "0",
                "gpu_ram_part": 0.8,
                "border_count": 64,
                "max_ctr_complexity": 1,
            })
        else:
            params.update({"task_type": "CPU",
                           "thread_count": max(1, (os.cpu_count() or 4) - 1)})

        tune_pool = Pool(X_tune, y_tune, weight=w_tune, cat_features=cat_cols)
        model = CatBoostClassifier(**params)
        model.fit(tune_pool, eval_set=val_pool, use_best_model=True, verbose=False)

        preds = model.predict(val_pool, prediction_type="RawFormulaVal")
        return average_precision_score(y_val, preds)

    study = optuna.create_study(direction="maximize", study_name="CatBoost_AntiFraud")
    study.optimize(objective, n_trials=n_trials)

    print(f"  Optuna best PR-AUC: {study.best_value:.6f}")
    print(f"  Best params: {study.best_params}")

    # Return full CatBoost params with optimal values
    bp = study.best_params
    result = {
        "iterations": 5000,
        "depth": bp["depth"],
        "learning_rate": bp["learning_rate"],
        "l2_leaf_reg": bp["l2_leaf_reg"],
        "bagging_temperature": bp["bagging_temperature"],
        "od_type": "Iter",
        "od_wait": 300,
    }
    return result, bp["w_red"], bp["w_yellow"]


# %% ==================== 5. BLEND HELPERS ====================
def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -40, 40)
    return 1.0 / (1.0 + np.exp(-x))


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-8, 1 - 1e-8)
    return np.log(p / (1 - p))


def find_best_blend_3(
    pred_main: np.ndarray,
    pred_recent: np.ndarray,
    pred_prod: np.ndarray,
    y_true: np.ndarray,
    step: float = 0.05,
) -> tuple[tuple[float, float, float], float]:
    """Grid-search blend weights for 3 model predictions."""
    best_ap, best_w = -1.0, (1.0, 0.0, 0.0)
    for w_m in np.arange(0.20, 0.95, step):
        for w_r in np.arange(0.00, 0.45, step):
            w_p = 1.0 - w_m - w_r
            if w_p < -0.01:
                continue
            w_p = max(w_p, 0.0)
            blend = w_m * pred_main + w_r * pred_recent + w_p * pred_prod
            ap = average_precision_score(y_true, blend)
            if ap > best_ap:
                best_ap = ap
                best_w = (float(w_m), float(w_r), float(w_p))
    return best_w, best_ap


# %% ==================== 6. FT-TRANSFORMER ====================
class FocalLoss(nn.Module):
    """Focal Loss for extreme class imbalance.
    alpha weights the positive class. Optional per-sample weights."""
    def __init__(self, alpha=0.75, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets, sample_weights=None):
        p = torch.sigmoid(logits)
        targets = targets.type_as(logits)
        pt = (1 - p) * (1 - targets) + p * targets
        bce = -torch.log(torch.clamp(pt, min=1e-8, max=1 - 1e-8))
        focal_weight = (1 - pt) ** self.gamma
        alpha_weight = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        loss = alpha_weight * focal_weight * bce
        if sample_weights is not None:
            loss = loss * sample_weights
        return loss.mean()


class PeriodicEmbedding(nn.Module):
    """Periodic (sin/cos) embeddings for numerical features.

    From 'On Embeddings for Numerical Features in Tabular Deep Learning'
    (Gorishniy et al., 2022).  Maps each scalar feature to d_model dims via
    learned per-feature frequencies, capturing non-linear relationships far
    better than a simple nn.Linear(1, d_model).
    """
    def __init__(self, n_features, d_model, n_frequencies=48, sigma=1.0):
        super().__init__()
        # Per-feature learnable frequency vectors
        self.coefficients = nn.Parameter(
            torch.randn(n_features, n_frequencies) * sigma
        )
        # Shared projection from 2*n_frequencies (sin+cos) -> d_model
        self.proj = nn.Linear(2 * n_frequencies, d_model)

    def forward(self, x):
        # x: (B, n_features)
        v = 2 * math.pi * x.unsqueeze(-1) * self.coefficients   # (B, F, n_freq)
        v = torch.cat([torch.sin(v), torch.cos(v)], dim=-1)      # (B, F, 2*n_freq)
        return self.proj(v)                                       # (B, F, d_model)


class FTTransformerDataset(Dataset):
    """Memory-efficient dataset: keeps numpy arrays, converts per-batch."""
    def __init__(self, x_num, x_cat, y, weights=None):
        self.x_num = x_num.astype(np.float32) if x_num.dtype != np.float32 else x_num
        self.x_cat = x_cat.astype(np.int64) if x_cat.dtype != np.int64 else x_cat
        self.y = y.astype(np.float32) if not isinstance(y, np.ndarray) or y.dtype != np.float32 else y
        self.weights = weights.astype(np.float32) if weights is not None and weights.dtype != np.float32 else weights

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        item = {
            "x_num": torch.as_tensor(self.x_num[idx]),
            "x_cat": torch.as_tensor(self.x_cat[idx]),
            "y": torch.as_tensor(self.y[idx]),
        }
        if self.weights is not None:
            item["weight"] = torch.as_tensor(self.weights[idx])
        return item


class FTTransformer(nn.Module):
    """Feature Tokenizer + Transformer for tabular data.

    Improvements over vanilla FTT:
      - PeriodicEmbedding for numerics (non-linear feature tokenisation)
      - GELU activation in transformer blocks
      - Dual pooling: CLS token + mean of feature tokens → deeper MLP head
    """
    def __init__(
        self, num_cont_features, cat_cardinalities,
        d_model=192, n_heads=8, n_layers=3,
        attn_dropout=0.2, ff_dropout=0.1,
    ):
        super().__init__()
        # Periodic embeddings for numerics (replaces per-feature Linear)
        self.num_embed = (
            PeriodicEmbedding(num_cont_features, d_model, n_frequencies=48, sigma=1.0)
            if num_cont_features > 0 else None
        )
        # Per-feature embeddings for categoricals
        self.cat_embs = (
            nn.ModuleList([nn.Embedding(dim, d_model, padding_idx=0) for dim in cat_cardinalities])
            if len(cat_cardinalities) > 0 else None
        )
        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.cls_token, std=0.01)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * 4, dropout=attn_dropout,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Head: CLS + mean pooling concatenated → 2-layer MLP
        self.head = nn.Sequential(
            nn.LayerNorm(d_model * 2),
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(ff_dropout),
            nn.Linear(d_model, 1),
        )
        # Init output bias for rare positive class
        pi = 0.01
        nn.init.constant_(self.head[-1].bias, -math.log((1 - pi) / pi))

    def forward(self, x_num, x_cat):
        B = x_num.shape[0] if x_num is not None else x_cat.shape[0]
        tokens = []
        if self.num_embed is not None and x_num is not None:
            tokens.append(self.num_embed(x_num))       # (B, n_num, d_model)
        if self.cat_embs is not None and x_cat is not None:
            cat_tokens = torch.stack(
                [self.cat_embs[i](x_cat[:, i]) for i in range(x_cat.shape[1])],
                dim=1,
            )                                           # (B, n_cat, d_model)
            tokens.append(cat_tokens)
        x = torch.cat(tokens, dim=1)                   # (B, n_features, d_model)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)                 # (B, 1+n_features, d_model)
        out = self.transformer(x)

        # Dual pooling: CLS + mean of feature tokens
        cls_out = out[:, 0, :]                          # (B, d_model)
        feat_mean = out[:, 1:, :].mean(dim=1)           # (B, d_model)
        pooled = torch.cat([cls_out, feat_mean], dim=-1) # (B, 2*d_model)
        return self.head(pooled).squeeze(-1)


def prepare_ftt_data(train_df, test_df, val_mask, feature_cols, cat_cols, raw_target,
                     w_red=3.0, w_yellow=2.0):
    """Prepare scaled/encoded arrays for FT-Transformer.

    Returns (X_tr_num, X_tr_cat, y_tr, w_tr,
             X_val_num, X_val_cat, y_val,
             X_test_num, X_test_cat, cat_dims, ftt_num_cols, ftt_cat_cols).
    """
    ftt_cat_cols = list(cat_cols)
    ftt_num_cols = [c for c in feature_cols if c not in cat_cols]

    # Split
    tr_df = train_df.loc[~val_mask]
    val_df = train_df.loc[val_mask]

    # Numerics: StandardScaler
    scaler = StandardScaler()
    X_tr_num = scaler.fit_transform(tr_df[ftt_num_cols].values.astype(np.float32))
    X_val_num = scaler.transform(val_df[ftt_num_cols].values.astype(np.float32))
    X_test_num = scaler.transform(test_df[ftt_num_cols].values.astype(np.float32))

    # Categoricals: OrdinalEncoder -> +1 (0 reserved for unknown/padding)
    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    X_tr_cat = encoder.fit_transform(tr_df[ftt_cat_cols].values.astype(str)).astype(np.int64) + 1
    X_val_cat = encoder.transform(val_df[ftt_cat_cols].values.astype(str)).astype(np.int64) + 1
    X_test_cat = encoder.transform(test_df[ftt_cat_cols].values.astype(str)).astype(np.int64) + 1
    # -1 + 1 = 0 for unknowns, which is padding_idx
    X_tr_cat[X_tr_cat == 0] = 0
    X_val_cat[X_val_cat == 0] = 0
    X_test_cat[X_test_cat == 0] = 0

    cat_dims = [len(encoder.categories_[i]) + 1 for i in range(len(ftt_cat_cols))]

    y_tr = tr_df["target_bin"].values.astype(np.float32)
    y_val = val_df["target_bin"].values.astype(np.float32)

    # FIX: original had weights=None for FTT.
    # Use moderate weights (FocalLoss alpha already handles class imbalance,
    # these mainly upweight yellow hard-negatives vs green background).
    w_tr = make_weights(raw_target[~val_mask], w_red=w_red, w_yellow=w_yellow)

    return (X_tr_num, X_tr_cat, y_tr, w_tr,
            X_val_num, X_val_cat, y_val,
            X_test_num, X_test_cat, cat_dims,
            ftt_num_cols, ftt_cat_cols)


def train_ft_transformer(
    X_tr_num, X_tr_cat, y_tr, w_tr,
    X_val_num, X_val_cat, y_val,
    cat_dims,
    seed=42,
    checkpoint_path: str | None = None,
):
    """Train FT-Transformer with early stopping on PR-AUC. Returns (model, best_pr_auc).

    If checkpoint_path exists, loads the model and skips training entirely.
    Best checkpoint is saved to disk whenever val PR-AUC improves.
    """
    if checkpoint_path is None:
        checkpoint_path = f"cache/ftt_seed_{seed}.pt"
    ckpt_path = Path(checkpoint_path)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  FTT device: {device}")

    use_cuda = device.type == "cuda"

    train_ds = FTTransformerDataset(X_tr_num, X_tr_cat, y_tr, w_tr)
    val_ds = FTTransformerDataset(X_val_num, X_val_cat, y_val)

    train_loader = DataLoader(
        train_ds, batch_size=FTT_BATCH_SIZE, shuffle=True,
        num_workers=0, pin_memory=use_cuda,
    )
    val_loader = DataLoader(
        val_ds, batch_size=FTT_BATCH_SIZE, shuffle=False,
        num_workers=0, pin_memory=use_cuda,
    )

    model = FTTransformer(
        num_cont_features=X_tr_num.shape[1],
        cat_cardinalities=cat_dims,
        d_model=FTT_D_MODEL, n_heads=FTT_N_HEADS, n_layers=FTT_N_LAYERS,
        attn_dropout=FTT_ATTN_DROPOUT, ff_dropout=FTT_FF_DROPOUT,
    ).to(device)

    # --- Load from checkpoint if exists ---
    if ckpt_path.exists():
        print(f"  FTT loading checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state"])
        best_pr_auc = ckpt["best_pr_auc"]
        model.eval()
        print(f"  FTT best val PR-AUC (from checkpoint): {best_pr_auc:.6f}")
        return model, best_pr_auc

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=FTT_MAX_LR / 10, weight_decay=FTT_WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=FTT_MAX_LR,
        steps_per_epoch=len(train_loader), epochs=FTT_EPOCHS,
        pct_start=0.2, anneal_strategy="cos",
    )
    criterion = FocalLoss(alpha=FTT_FOCAL_ALPHA, gamma=FTT_FOCAL_GAMMA)

    # AMP scaler for GPU mixed precision
    scaler = GradScaler('cuda') if use_cuda else None

    best_pr_auc = 0.0
    epochs_no_improve = 0
    best_state = None

    for epoch in range(FTT_EPOCHS):
        # --- Train ---
        model.train()
        train_loss = 0.0
        for batch in tqdm(train_loader, desc=f"  Epoch {epoch+1}/{FTT_EPOCHS} [train]", leave=False):
            x_num = batch["x_num"].to(device)
            x_cat = batch["x_cat"].to(device)
            y = batch["y"].to(device)
            w = batch["weight"].to(device) if "weight" in batch else None

            # Mixup on numeric features (categoricals stay unchanged)
            if FTT_MIXUP_ALPHA > 0:
                lam = np.random.beta(FTT_MIXUP_ALPHA, FTT_MIXUP_ALPHA)
                idx = torch.randperm(x_num.size(0), device=device)
                x_num = lam * x_num + (1 - lam) * x_num[idx]
                y = lam * y + (1 - lam) * y[idx]
                if w is not None:
                    w = lam * w + (1 - lam) * w[idx]

            optimizer.zero_grad()

            if scaler is not None:
                with autocast('cuda'):
                    logits = model(x_num, x_cat)
                    loss = criterion(logits, y, sample_weights=w)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(x_num, x_cat)
                loss = criterion(logits, y, sample_weights=w)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            scheduler.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # --- Validate ---
        model.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for batch in val_loader:
                x_num = batch["x_num"].to(device)
                x_cat = batch["x_cat"].to(device)
                if use_cuda:
                    with autocast('cuda'):
                        logits = model(x_num, x_cat)
                else:
                    logits = model(x_num, x_cat)
                all_preds.extend(torch.sigmoid(logits.float()).cpu().numpy())
                all_targets.extend(batch["y"].numpy())

        val_preds = np.array(all_preds)
        val_targets = np.array(all_targets)
        val_pr_auc = average_precision_score(val_targets, val_preds)

        print(f"  Epoch {epoch+1}/{FTT_EPOCHS}: "
              f"train_loss={avg_train_loss:.5f}, val_PR-AUC={val_pr_auc:.6f}")

        if val_pr_auc > best_pr_auc:
            best_pr_auc = val_pr_auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
            # Save checkpoint to disk
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model_state": best_state, "best_pr_auc": best_pr_auc}, ckpt_path)
            print(f"  [checkpoint saved] epoch={epoch+1}, PR-AUC={best_pr_auc:.6f} → {ckpt_path}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= FTT_PATIENCE:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    print(f"  Best FTT val PR-AUC: {best_pr_auc:.6f}")
    return model, best_pr_auc


def predict_ftt(model, X_num, X_cat, batch_size=4096):
    """Run FTT inference, return logits (not probabilities)."""
    device = next(model.parameters()).device
    use_cuda = device.type == "cuda"
    ds = FTTransformerDataset(X_num, X_cat, np.zeros(len(X_num), dtype=np.float32))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    logits_list = []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc="  FTT inference", leave=False):
            x_num = batch["x_num"].to(device)
            x_cat = batch["x_cat"].to(device)
            if use_cuda:
                with autocast('cuda'):
                    logits = model(x_num, x_cat)
            else:
                logits = model(x_num, x_cat)
            logits_list.extend(logits.float().cpu().numpy())
    return np.array(logits_list)


# %% ==================== 7. MAIN PIPELINE ====================
def main():
    print(f"USE_GPU: {USE_GPU}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # ============================================================
    # STEP 1: Load labels
    # ============================================================
    print("\n" + "=" * 60)
    print("STEP 1: Load labels")
    print("=" * 60)
    labels_lf = pl.scan_parquet(DATA_DIR / "train_labels.parquet")

    # ============================================================
    # STEP 2: Build features (cached per partition)
    # ============================================================
    print("\n" + "=" * 60)
    print("STEP 2: Build features")
    print("=" * 60)
    feature_paths = []
    for pid in [1, 2, 3]:
        p = build_features_for_part(pid, labels_lf, force=FORCE_REBUILD_FEATURES)
        feature_paths.append(p)
        gc.collect()

    # ============================================================
    # STEP 3: Category priors
    # ============================================================
    print("\n" + "=" * 60)
    print("STEP 3: Category priors")
    print("=" * 60)
    # Build prior lookup tables (small, few KB each)
    prior_tables = {}
    prior_feature_cols = []
    for key_name, expr in PRIOR_DEFS.items():
        prior_df = build_prior_table(key_name, expr, labels_lf, force=FORCE_REBUILD_PRIORS)
        prior_tables[key_name] = prior_df
        prior_feature_cols.extend([c for c in prior_df.columns if c != key_name])

    print(f"  Built {len(prior_tables)} prior tables, {len(prior_feature_cols)} prior features")

    # ============================================================
    # STEP 4: Prepare DataFrames (load train/test separately, apply priors)
    # ============================================================
    print("\n" + "=" * 60)
    print("STEP 4: Prepare DataFrames")
    print("=" * 60)

    # Load TRAIN data lazily, apply priors, collect
    print("  Loading train split ...")
    train_lf = pl.concat(
        [pl.scan_parquet(p) for p in feature_paths],
        how="vertical_relaxed",
    ).filter(pl.col("is_train_sample"))

    for key_name, prior_df in prior_tables.items():
        train_lf = train_lf.join(prior_df.lazy(), on=key_name, how="left")

    train_pl = train_lf.collect()
    del train_lf
    gc.collect()

    # Fill null priors with train means
    if prior_feature_cols:
        train_pl = train_pl.with_columns([
            pl.col(c).fill_null(pl.col(c).mean()) for c in prior_feature_cols
        ])

    # Compute prior fill values from train for use on test
    prior_fill_values = {}
    if prior_feature_cols:
        for c in prior_feature_cols:
            prior_fill_values[c] = train_pl.get_column(c).mean()

    train_pl = train_pl.with_columns(pl.col("target_bin").cast(pl.Int8))

    # Load TEST data lazily, apply priors, collect
    print("  Loading test split ...")
    test_lf = pl.concat(
        [pl.scan_parquet(p) for p in feature_paths],
        how="vertical_relaxed",
    ).filter(pl.col("is_test")).unique(subset=["event_id"])

    for key_name, prior_df in prior_tables.items():
        test_lf = test_lf.join(prior_df.lazy(), on=key_name, how="left")

    test_pl = test_lf.collect()
    del test_lf
    gc.collect()

    # Fill test priors with TRAIN means (no leakage)
    if prior_feature_cols:
        test_pl = test_pl.with_columns([
            pl.col(c).fill_null(pl.lit(prior_fill_values[c])) for c in prior_feature_cols
        ])

    del prior_tables
    gc.collect()

    # Convert to pandas with downcasting
    print("  Converting to pandas ...")
    train_df, test_df, feature_cols, cat_cols, num_cols = prepare_train_test(
        train_pl, test_pl, prior_feature_cols
    )
    gc.collect()

    # Validation mask (time-based)
    val_mask = (train_df["event_ts"] >= VAL_START).values
    raw_target = train_df["train_target_raw"].values
    print(f"  Train: {(~val_mask).sum():,}, Val: {val_mask.sum():,}")

    # ============================================================
    # STEP 5: Train CatBoost models
    # ============================================================
    print("\n" + "=" * 60)
    print("STEP 5: Train CatBoost models")
    print("=" * 60)
    X_all = train_df[feature_cols]
    y_all = train_df["target_bin"].astype(np.int8).values

    X_tr, X_val = X_all.loc[~val_mask], X_all.loc[val_mask]
    y_tr, y_val = y_all[~val_mask], y_all[val_mask]

    # --- Optuna hyperparameter tuning ---
    CB_PARAMS = {
        "iterations": 5000,
        "learning_rate": 0.02,
        "depth": 6,
        "l2_leaf_reg": 7,
        "random_strength": 2,
        "od_type": "Iter",
        "od_wait": 500,
    }
    opt_w_red, opt_w_yellow = 10.0, 2.5  # defaults

    if USE_OPTUNA and not INFERENCE_ONLY:
        print("\n--- Optuna tuning ---")
        CB_PARAMS, opt_w_red, opt_w_yellow = run_optuna_tuning(
            X_tr, y_tr, raw_target[~val_mask],
            X_val, y_val,
            cat_cols,
            n_trials=OPTUNA_N_TRIALS,
            use_gpu=USE_GPU,
        )

    w_all = make_weights(raw_target, w_red=opt_w_red, w_yellow=opt_w_yellow)
    w_tr, w_val = w_all[~val_mask], w_all[val_mask]

    # --- Model 1: Main (fraud vs all) ---
    print("\n--- [1/4] Main model ---")
    path_main, iter_main, ap_main, params_main, pred_main_v = train_catboost(
        X_tr, y_tr, w_tr, X_val, y_val, w_val,
        cat_cols, CB_PARAMS, USE_GPU, name="main",
    )

    # --- Model 2: Recent (data from RECENT_BORDER + all labeled) ---
    print("\n--- [2/4] Recent model ---")
    recent_mask = (
        (train_df["event_ts"] >= RECENT_BORDER).values | (raw_target != -1)
    )
    rec_tr = recent_mask & ~val_mask
    rec_val = recent_mask & val_mask

    path_recent, iter_recent, ap_recent, params_recent, pred_recent_v = train_catboost(
        X_all.loc[rec_tr], y_all[rec_tr],
        make_weights(raw_target[rec_tr], w_red=opt_w_red, w_yellow=opt_w_yellow),
        X_all.loc[rec_val], y_all[rec_val],
        make_weights(raw_target[rec_val], w_red=opt_w_red, w_yellow=opt_w_yellow),
        cat_cols, CB_PARAMS, USE_GPU, name="recent",
        full_val_X=X_val,
    )

    # --- Model 3: Suspicious (red+yellow vs green) ---
    print("\n--- [3/4] Suspicious model ---")
    y_susp = (raw_target != -1).astype(np.int8)
    w_susp = np.where(raw_target != -1, 6.0, 1.2).astype(np.float32)

    path_susp, iter_susp, _, params_susp, pred_susp_v = train_catboost(
        X_tr, y_susp[~val_mask], w_susp[~val_mask],
        X_val, y_susp[val_mask], w_susp[val_mask],
        cat_cols, CB_PARAMS, USE_GPU, name="suspicious",
    )

    # --- Model 4: Red vs Yellow (labeled data only) ---
    print("\n--- [4/4] Red|Yellow model ---")
    labeled_mask = raw_target != -1
    lbl_tr = labeled_mask & ~val_mask
    lbl_val = labeled_mask & val_mask

    y_rg = y_all[labeled_mask]
    w_rg = np.where(raw_target[labeled_mask] == 1, 2.2, 1.0).astype(np.float32)

    path_rg, iter_rg, _, params_rg, pred_rg_v = train_catboost(
        X_all.loc[lbl_tr], y_all[lbl_tr],
        np.where(raw_target[lbl_tr] == 1, 2.2, 1.0).astype(np.float32),
        X_all.loc[lbl_val], y_all[lbl_val],
        np.where(raw_target[lbl_val] == 1, 2.2, 1.0).astype(np.float32),
        cat_cols, CB_PARAMS, USE_GPU, name="red_vs_yellow",
        full_val_X=X_val,
    )

    # ============================================================
    # STEP 6: Find best blend weights on holdout
    # ============================================================
    print("\n" + "=" * 60)
    print("STEP 6: Blend optimization")
    print("=" * 60)

    import json as _json
    _blend_weights_path = Path("cache/blend_weights.json")

    if INFERENCE_ONLY and _blend_weights_path.exists():
        # Load previously saved weights — skip grid search
        _bw = _json.loads(_blend_weights_path.read_text())
        w_m, w_r, w_p = _bw["w_m"], _bw["w_r"], _bw["w_p"]
        print(f"  Loaded blend weights from cache: w_m={w_m:.2f}, w_r={w_r:.2f}, w_p={w_p:.2f}")
        pred_prod_v = logit(sigmoid(pred_susp_v) * sigmoid(pred_rg_v))
    else:
        # Two-stage product: P(fraud) ~ P(suspicious) * P(red|suspicious)
        pred_prod_v = logit(sigmoid(pred_susp_v) * sigmoid(pred_rg_v))

        best_w, best_ap = find_best_blend_3(
            pred_main_v, pred_recent_v, pred_prod_v, y_val
        )
        w_m, w_r, w_p = best_w

        print(f"  Main    val PR-AUC: {average_precision_score(y_val, pred_main_v):.6f}")
        print(f"  Recent  val PR-AUC: {average_precision_score(y_val, pred_recent_v):.6f}")
        print(f"  Product val PR-AUC: {average_precision_score(y_val, pred_prod_v):.6f}")
        print(f"  Blend weights (main, recent, product): ({w_m:.2f}, {w_r:.2f}, {w_p:.2f})")
        print(f"  Blend   val PR-AUC: {best_ap:.6f}")

        # Save blend weights to disk for INFERENCE_ONLY re-runs
        _blend_weights_path.parent.mkdir(parents=True, exist_ok=True)
        _blend_weights_path.write_text(_json.dumps({"w_m": w_m, "w_r": w_r, "w_p": w_p}))

    # Free susp/rg val predictions (prod_v already computed from them)
    del pred_susp_v, pred_rg_v
    gc.collect()
    # NOTE: pred_main_v, pred_recent_v, pred_prod_v kept alive — needed in Step 8 (FTT blend)

    # ============================================================
    # STEP 7: Refit on full train data & generate submission
    # ============================================================
    print("\n" + "=" * 60)
    print("STEP 7: Refit & submission")
    print("=" * 60)

    X_test = test_df[feature_cols]
    test_pool = Pool(X_test, cat_features=cat_cols)

    def _load_or_refit(path, refit_X, refit_y, refit_w, params, best_iter, label):
        """Load saved model or refit on full data. Predict on test, then free."""
        if RETRAIN_ON_FULL:
            print(f"  Refitting {label} on full train ...")
            model = refit_catboost(refit_X, refit_y, refit_w, cat_cols, params, best_iter)
        else:
            model = CatBoostClassifier()
            model.load_model(path)
        pred = model.predict(test_pool, prediction_type="RawFormulaVal")
        del model
        gc.collect()
        return pred

    pred_main_t = _load_or_refit(
        path_main, X_all, y_all, w_all,
        params_main, iter_main, "main",
    )
    pred_recent_t = _load_or_refit(
        path_recent,
        X_all.loc[recent_mask], y_all[recent_mask],
        make_weights(raw_target[recent_mask], w_red=opt_w_red, w_yellow=opt_w_yellow),
        params_recent, iter_recent, "recent",
    )
    pred_susp_t = _load_or_refit(
        path_susp, X_all, y_susp, w_susp,
        params_susp, iter_susp, "suspicious",
    )
    pred_rg_t = _load_or_refit(
        path_rg,
        X_all.loc[labeled_mask], y_rg, w_rg,
        params_rg, iter_rg, "red_vs_yellow",
    )

    pred_prod_t = logit(sigmoid(pred_susp_t) * sigmoid(pred_rg_t))
    pred_cb_blend = w_m * pred_main_t + w_r * pred_recent_t + w_p * pred_prod_t

    # --- Save CatBoost-only submission ---
    _save_submission(test_df, pred_cb_blend, "submission_catboost.csv")

    # Free individual test predictions
    del pred_susp_t, pred_rg_t, pred_recent_t, test_pool
    gc.collect()

    # ============================================================
    # STEP 8: FT-Transformer (optional stacking)
    # ============================================================
    if FTT_ENABLED:
        print("\n" + "=" * 60)
        print("STEP 8: FT-Transformer")
        print("=" * 60)

        # Add CatBoost meta-features using 3-fold OOF to avoid train leakage.
        _oof_cache = Path("cache/oof_preds.npz")

        if INFERENCE_ONLY and _oof_cache.exists():
            print("  Loading OOF meta-features from cache ...")
            _oof_data = np.load(_oof_cache)
            train_df["cb_main"] = _oof_data["oof_main"]
            train_df["cb_susp"] = _oof_data["oof_susp"]
            train_df["cb_rg"]   = _oof_data["oof_rg"]
            test_df["cb_main"]  = _oof_data["test_main"]
            test_df["cb_susp"]  = _oof_data["test_susp"]
            test_df["cb_rg"]    = _oof_data["test_rg"]
        elif INFERENCE_ONLY:
            # No OOF cache: load holdout CB models and predict directly.
            # Leakage doesn't matter — FTT is loaded from checkpoint, not trained.
            print("  INFERENCE_ONLY: no OOF cache, using holdout models for CB meta-features ...")
            _pool_tr = Pool(X_all[feature_cols], cat_features=cat_cols)
            _pool_te = Pool(test_df[feature_cols], cat_features=cat_cols)
            for _cb_name, _col in [("main", "cb_main"), ("suspicious", "cb_susp"), ("red_vs_yellow", "cb_rg")]:
                _m = CatBoostClassifier()
                _m.load_model(f"cache/cb_{_cb_name}.cbm")
                train_df[_col] = _m.predict(_pool_tr, prediction_type="RawFormulaVal").astype(np.float32)
                test_df[_col]  = _m.predict(_pool_te, prediction_type="RawFormulaVal").astype(np.float32)
                del _m; gc.collect()
            del _pool_tr, _pool_te; gc.collect()
        else:
            # Time-based expanding-window OOF: train on earlier months,
            # predict the next month.  Avoids temporal leakage that random
            # KFold introduces in time-series fraud data.
            print("  Building OOF CatBoost meta-features (time-based) ...")

            # Initialize OOF arrays
            oof_main = np.zeros(len(train_df), dtype=np.float32)
            oof_susp = np.zeros(len(train_df), dtype=np.float32)
            oof_rg = np.zeros(len(train_df), dtype=np.float32)
            test_pool_all = Pool(test_df[feature_cols], cat_features=cat_cols)
            test_main_sum = np.zeros(len(test_df), dtype=np.float64)
            test_susp_sum = np.zeros(len(test_df), dtype=np.float64)
            test_rg_sum = np.zeros(len(test_df), dtype=np.float64)
            n_test_models = 0

            oof_params = CB_PARAMS.copy()
            oof_params["iterations"] = max(300, iter_main)  # use best iter from holdout
            oof_params.pop("od_type", None)
            oof_params.pop("od_wait", None)
            if USE_GPU:
                oof_params.update({
                    "task_type": "GPU",
                    "devices": "0",
                    "gpu_ram_part": 0.8,
                    "border_count": 64,
                    "max_ctr_complexity": 1,
                })
            else:
                oof_params.update({"task_type": "CPU",
                                   "thread_count": max(1, (os.cpu_count() or 4) - 1)})
            oof_params.update({
                "loss_function": "Logloss", "eval_metric": "PRAUC",
                "random_seed": RANDOM_SEED, "allow_writing_files": False, "verbose": 0,
            })

            # Monthly expanding window: train on months < m, predict month m
            train_months = train_df["event_ts"].dt.to_period("M")
            unique_months = sorted(train_months.unique())
            print(f"    Time periods: {[str(m) for m in unique_months]}")

            for pred_month in unique_months:
                past_mask = train_months < pred_month
                curr_mask = train_months == pred_month
                oof_val_idx = np.where(curr_mask.values)[0]
                oof_tr_idx = np.where(past_mask.values)[0]

                if len(oof_tr_idx) < 100:
                    # First month — no past data, leave OOF as zeros (prior)
                    print(f"    Month {pred_month}: skip (no past data), "
                          f"{len(oof_val_idx):,} rows get prior=0")
                    continue

                print(f"    Month {pred_month}: train on {len(oof_tr_idx):,} past rows, "
                      f"predict {len(oof_val_idx):,} rows")
                X_oof_tr = X_all.iloc[oof_tr_idx]
                X_oof_val = X_all.iloc[oof_val_idx]
                oof_val_pool = Pool(X_oof_val, cat_features=cat_cols)

                # Main model OOF
                pool_tr = Pool(X_oof_tr, y_all[oof_tr_idx],
                               weight=w_all[oof_tr_idx], cat_features=cat_cols)
                m = CatBoostClassifier(**oof_params)
                m.fit(pool_tr, verbose=False)
                oof_main[oof_val_idx] = m.predict(oof_val_pool, prediction_type="RawFormulaVal")
                test_main_sum += m.predict(test_pool_all, prediction_type="RawFormulaVal")
                del m, pool_tr; gc.collect()

                # Suspicious model OOF
                pool_s = Pool(X_oof_tr, y_susp[oof_tr_idx],
                              weight=w_susp[oof_tr_idx], cat_features=cat_cols)
                m_s = CatBoostClassifier(**oof_params)
                m_s.fit(pool_s, verbose=False)
                oof_susp[oof_val_idx] = m_s.predict(oof_val_pool, prediction_type="RawFormulaVal")
                test_susp_sum += m_s.predict(test_pool_all, prediction_type="RawFormulaVal")
                del m_s, pool_s; gc.collect()

                # Red vs Yellow OOF (only labeled data in this fold)
                lbl_in_fold = raw_target[oof_tr_idx] != -1
                if lbl_in_fold.sum() > 50:
                    pool_r = Pool(
                        X_oof_tr.iloc[lbl_in_fold], y_all[oof_tr_idx][lbl_in_fold],
                        weight=np.where(raw_target[oof_tr_idx][lbl_in_fold] == 1, 2.2, 1.0).astype(np.float32),
                        cat_features=cat_cols,
                    )
                    m_r = CatBoostClassifier(**oof_params)
                    m_r.fit(pool_r, verbose=False)
                    oof_rg[oof_val_idx] = m_r.predict(oof_val_pool, prediction_type="RawFormulaVal")
                    test_rg_sum += m_r.predict(test_pool_all, prediction_type="RawFormulaVal")
                    del m_r, pool_r; gc.collect()

                n_test_models += 1
                del oof_val_pool; gc.collect()

            if n_test_models == 0:
                n_test_models = 1  # safety
            test_main = (test_main_sum / n_test_models).astype(np.float32)
            test_susp = (test_susp_sum / n_test_models).astype(np.float32)
            test_rg   = (test_rg_sum   / n_test_models).astype(np.float32)

            train_df["cb_main"] = oof_main
            train_df["cb_susp"] = oof_susp
            train_df["cb_rg"]   = oof_rg
            test_df["cb_main"]  = test_main
            test_df["cb_susp"]  = test_susp
            test_df["cb_rg"]    = test_rg

            # Save OOF predictions to disk for INFERENCE_ONLY re-runs
            _oof_cache.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                _oof_cache,
                oof_main=oof_main, oof_susp=oof_susp, oof_rg=oof_rg,
                test_main=test_main, test_susp=test_susp, test_rg=test_rg,
            )
            print(f"  OOF predictions saved to {_oof_cache}")

            # Free OOF intermediates
            del oof_main, oof_susp, oof_rg, test_main_sum, test_susp_sum, test_rg_sum
            del test_main, test_susp, test_rg, test_pool_all
            gc.collect()



        cb_meta_cols = ["cb_main", "cb_susp", "cb_rg"]
        ftt_feature_cols = feature_cols + cb_meta_cols

        # Prepare FTT data
        print("  Preparing FTT data ...")
        (
            X_tr_num, X_tr_cat, y_ftt_tr, w_ftt_tr,
            X_val_num, X_val_cat, y_ftt_val,
            X_test_num, X_test_cat, cat_dims,
            ftt_num_cols, ftt_cat_cols
        ) = prepare_ftt_data(
            train_df, test_df, val_mask,
            ftt_feature_cols, cat_cols, raw_target,
            w_red=opt_w_red, w_yellow=opt_w_yellow,
        )
        print(f"  FTT features: {X_tr_num.shape[1]} num + {X_tr_cat.shape[1]} cat")
        print(f"  FTT cat dims: {cat_dims}")

        # Train multi-seed FTT ensemble
        print(f"  Training FT-Transformer ({FTT_N_SEEDS} seeds) ...")
        ftt_val_logits_list = []
        ftt_test_logits_list = []

        for seed_i in range(FTT_N_SEEDS):
            seed = RANDOM_SEED + seed_i * 111
            print(f"\n  --- FTT seed {seed_i+1}/{FTT_N_SEEDS} (seed={seed}) ---")
            ftt_model, ftt_val_ap = train_ft_transformer(
                X_tr_num, X_tr_cat, y_ftt_tr, w_ftt_tr,
                X_val_num, X_val_cat, y_ftt_val,
                cat_dims, seed=seed,
                checkpoint_path=f"cache/ftt_seed_{seed}.pt",
            )
            ftt_val_logits_list.append(predict_ftt(ftt_model, X_val_num, X_val_cat))
            ftt_test_logits_list.append(predict_ftt(ftt_model, X_test_num, X_test_cat))
            del ftt_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Free FTT training arrays
        del X_tr_num, X_tr_cat, y_ftt_tr, w_ftt_tr
        gc.collect()

        ftt_val_logits = np.mean(ftt_val_logits_list, axis=0)
        ftt_test_logits = np.mean(ftt_test_logits_list, axis=0)
        print(f"\n  FTT ensemble ({FTT_N_SEEDS} seeds) val PR-AUC: "
              f"{average_precision_score(y_ftt_val, ftt_val_logits):.6f}")

        # --- Find CatBoost+FTT blend weight on val ---
        # Try both logit-space and rank-space blends (rank is more robust
        # when CatBoost and FTT have very different score distributions).
        cb_val_blend = w_m * pred_main_v + w_r * pred_recent_v + w_p * pred_prod_v
        n_val = len(cb_val_blend)
        cb_val_ranks = rankdata(cb_val_blend) / n_val
        ftt_val_ranks = rankdata(ftt_val_logits) / n_val

        best_cb_w, best_final_ap, best_blend_mode = 0.7, -1.0, "logit"
        for mode, (a, b) in [("logit", (cb_val_blend, ftt_val_logits)),
                              ("rank",  (cb_val_ranks, ftt_val_ranks))]:
            for alpha in np.arange(0.0, 1.01, 0.05):
                final_blend = alpha * a + (1 - alpha) * b
                ap = average_precision_score(y_val, final_blend)
                if ap > best_final_ap:
                    best_final_ap = ap
                    best_cb_w = float(alpha)
                    best_blend_mode = mode

        print(f"  CatBoost+FTT blend: alpha_cb={best_cb_w:.2f}, "
              f"mode={best_blend_mode}, val PR-AUC={best_final_ap:.6f}")

        # Final test prediction (same blend mode as chosen on val)
        if best_blend_mode == "rank":
            n_test = len(pred_cb_blend)
            pred_final = (best_cb_w * rankdata(pred_cb_blend) / n_test
                          + (1 - best_cb_w) * rankdata(ftt_test_logits) / n_test)
        else:
            pred_final = best_cb_w * pred_cb_blend + (1 - best_cb_w) * ftt_test_logits

        # --- Save blended submission ---
        _save_submission(test_df, pred_final, "submission.csv")

        # --- Save FTT-only submission ---
        _save_submission(test_df, ftt_test_logits, "submission_ftt.csv")

        # Cleanup FTT intermediates
        del X_val_num, X_val_cat, X_test_num, X_test_cat
        del ftt_val_logits, ftt_test_logits, ftt_val_logits_list, ftt_test_logits_list
        del pred_main_v, pred_recent_v, pred_prod_v
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    else:
        # No FTT, just rename CatBoost submission
        _save_submission(test_df, pred_cb_blend, "submission.csv")

    print("\n  Done!")


def _save_submission(test_df, predictions, filename):
    """Build and save submission CSV, ensuring all test events are present."""
    pred_df = pd.DataFrame({
        "event_id": test_df["event_id"].values,
        "predict": predictions,
    }).drop_duplicates("event_id")   # guard against duplicate event_ids in processed test

    test_all = pl.read_parquet(DATA_DIR / "test.parquet", columns=["event_id"])
    template = test_all.to_pandas().drop_duplicates("event_id")  # dedup raw template too
    submission = template[["event_id"]].merge(pred_df, on="event_id", how="left")

    dups = int(submission.duplicated("event_id").sum())
    if dups > 0:
        print(f"  WARNING: {dups} duplicate event_ids found after merge, keeping first")
        submission = submission.drop_duplicates("event_id")

    missing = int(submission["predict"].isna().sum())
    if missing > 0:
        print(f"  WARNING: {missing} events missing in {filename}, filling with median")
        submission["predict"] = submission["predict"].fillna(submission["predict"].median())

    submission.to_csv(filename, index=False)
    print(f"  Saved {filename} ({len(submission):,} rows)")


if __name__ == "__main__":
    main()
