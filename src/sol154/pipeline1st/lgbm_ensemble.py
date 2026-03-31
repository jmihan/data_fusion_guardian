"""pipeline1st: LGB+CatBoost 4-fold temporal CV with multi-head blend.

Converted from sol_154/pipeline1st.ipynb.
Produces submission_MINE.csv via 4-head blend (lgb, cb, cb_lbl, lgb_susp)
with Nelder-Mead optimized weights + full refit.
"""
import gc
import json
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl
from catboost import CatBoostClassifier, Pool
from scipy.optimize import minimize
from scipy.special import softmax
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score

warnings.filterwarnings("ignore")

from src.sol154.config import DATA_DIR, SOL154_PIPELINE1ST_CACHE as CACHE_DIR

# ── Constants ──

RANDOM_SEED = 42
SMOKE_RUN = False
SAMPLE_ROWS = 200_000
FORCE_REBUILD_FEATURES = False
USE_GPU = True
GPU_DEVICE = 0

NEG_SAMPLE_BORDER_STR = "2025-03-01 00:00:00"
NEG_SAMPLE_MOD_RECENT = 6
NEG_SAMPLE_MOD_OLD = 20
USE_PRETRAIN_HISTORY = True
HISTORY_MODE_TAG = "with_pretrain" if USE_PRETRAIN_HISTORY else "train_only"

BASE_FEATURE_TAG = f"blend_0135_01367_v016_features_{HISTORY_MODE_TAG}"
BASE_RUN_TAG = f"blend_0135_01367_v016_susp_blend_{HISTORY_MODE_TAG}"
FEATURE_TAG = BASE_FEATURE_TAG
RUN_TAG = BASE_RUN_TAG
SUBMISSION_FILENAME = "submission_MINE.csv"
MERGE_ALL = True
USE_FULL_MERGED = True
REFIT_ITER_MULT = 1.08

MANUAL_DROP_COLS = [
    "session_id", "browser_language_missing", "cnt_events_this_hour",
    "cust_prev_same_channel", "cust_prev_same_channel_sub", "cust_prev_same_device",
    "prior_browser_language_i_red_rate", "session_event_idx",
]
TIME_DECAY_MIN_MULT = 0.85
TIME_DECAY_MAX_MULT = 1.25
ENABLE_CB_LABEL_HEAD = True
ENABLE_LGB_SUSP_HEAD = True
PART_IDS = [1, 2, 3]

# ── Column definitions ──

BASE_COLS = [
    "customer_id", "event_id", "event_dttm", "event_type_nm", "event_desc",
    "channel_indicator_type", "channel_indicator_sub_type", "operaton_amt", "currency_iso_cd",
    "mcc_code", "pos_cd",
    "accept_language", "browser_language",
    "timezone", "session_id", "operating_system_type",
    "battery", "device_system_version", "screen_size", "developer_tools",
    "phone_voip_call_state", "web_rdp_connection", "compromised",
]

SAVE_META_COLS = [
    "event_id", "event_ts", "period",
    "target", "train_target_raw", "target_bin",
    "is_train_sample", "is_test", "keep_green",
]

MODEL_FEATURE_COLS = [
    "session_id", "event_type_nm", "event_desc", "channel_indicator_type",
    "channel_indicator_sub_type", "currency_iso_cd", "mcc_code_i", "pos_cd",
    "timezone", "operating_system_type", "phone_voip_call_state",
    "web_rdp_connection", "developer_tools_i", "compromised_i",
    "accept_language_i", "browser_language_i", "device_fp_i",
    "amt", "amt_abs", "amt_log_abs", "amt_bucket", "amt_is_negative",
    "hour", "weekday", "day", "month", "week_of_year", "is_weekend", "is_night",
    "hour_sin", "hour_cos", "weekday_sin", "weekday_cos", "month_sin", "month_cos",
    "battery_pct", "os_ver_major", "screen_w", "screen_h", "screen_pixels", "screen_ratio",
    "battery_missing", "screen_missing", "os_ver_missing",
    "accept_language_missing", "browser_language_missing",
    "dev_tools_missing", "compromised_missing",
    "cust_prev_events", "days_since_first_event", "days_since_prev_event", "sec_since_prev_event",
    "cnt_prev_same_desc", "cnt_prev_same_type", "cnt_prev_same_mcc", "cnt_prev_same_subtype",
    "cnt_prev_same_channel_type", "cnt_prev_same_currency", "cnt_prev_same_device",
    "cnt_prev_same_session", "events_before_today", "events_before_hour", "cnt_events_this_hour",
    "cust_prev_amt_mean", "cust_prev_amt_std", "cust_prev_max_amt",
    "amt_delta_prev", "sec_since_prev_same_desc", "sec_since_prev_same_type",
    "sec_since_prev_same_device", "sec_since_prev_same_mcc",
    "sec_since_prev_same_channel_subtype", "sec_since_prev_same_channel_type",
    "sec_since_prev_same_currency",
    "amt_zscore", "amt_vs_prev_max", "amt_vs_prev_mean",
    "sec_since_session_start", "session_amt_before", "session_event_idx",
    "cust_prev_same_device", "cust_prev_same_timezone", "cust_prev_same_os",
    "cust_prev_same_channel", "cust_prev_same_channel_sub",
    "device_prev_same_customer", "device_prev_same_desc", "device_prev_same_mcc",
    "device_prev_same_timezone", "device_prev_same_subtype",
    "is_new_device_for_customer", "is_new_timezone_for_customer", "is_new_desc_for_customer",
    "is_new_mcc_for_customer", "is_new_subtype_for_customer", "is_new_os_for_customer",
    "device_prev_ops_log", "device_prev_unique_customers_log", "device_prev_unique_sessions_log",
    "device_customer_diversity",
    "cust_prev_red_lbl_cnt", "cust_prev_yellow_lbl_cnt", "cust_prev_labeled_cnt",
    "sec_since_prev_red_lbl", "sec_since_prev_yellow_lbl",
    "cust_prev_red_lbl_rate", "cust_prev_yellow_lbl_rate", "cust_prev_susp_lbl_rate",
    "cust_prev_any_red_flag", "cust_prev_any_yellow_flag",
    "prior_event_desc_red_rate", "prior_mcc_code_i_red_rate", "prior_timezone_red_rate",
    "prior_event_type_nm_red_rate", "prior_channel_indicator_type_red_rate",
    "prior_channel_indicator_sub_type_red_rate", "prior_pos_cd_red_rate",
    "prior_operating_system_type_red_rate", "prior_accept_language_i_red_rate",
    "prior_browser_language_i_red_rate", "prior_device_fp_i_red_rate",
    "risk_new_desc_x_prior", "risk_new_device_x_prior", "risk_new_tz_x_prior", "risk_new_mcc_x_prior",
    "cust_prev_mean_amt_same_desc", "cust_prev_mean_amt_same_device",
    "amt_vs_same_desc_mean", "amt_vs_same_device_mean",
    "amt_sum_last_1h", "cnt_last_1h",
    "amt_sum_last_24h", "cnt_last_24h", "max_amt_last_24h",
    "amt_vs_1h_sum", "amt_vs_24h_sum",
]

CAT_COLS = [
    "session_id", "event_type_nm", "event_desc", "channel_indicator_type",
    "channel_indicator_sub_type", "currency_iso_cd", "mcc_code_i", "pos_cd",
    "timezone", "operating_system_type", "phone_voip_call_state",
    "web_rdp_connection", "developer_tools_i", "compromised_i",
    "accept_language_i", "browser_language_i", "device_fp_i",
]


# ── Helper functions ──

def dedupe(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _exists(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def pr_auc(y_true, y_pred):
    return average_precision_score(y_true, y_pred)


def rank_norm(x):
    x = np.asarray(x)
    return rankdata(x, method="average") / (len(x) + 1.0)


def optimize_blend_weights(heads: Dict[str, np.ndarray], y_true: np.ndarray):
    keys = list(heads.keys())
    preds = [heads[k] for k in keys]

    def neg_ap(logits):
        w = softmax(logits)
        blend = sum(w[i] * preds[i] for i in range(len(keys)))
        return -average_precision_score(y_true, blend)

    starts = [np.zeros(len(keys), dtype=np.float64)]
    for i in range(len(keys)):
        x0 = np.full(len(keys), -2.0, dtype=np.float64)
        x0[i] = 2.5
        starts.append(x0)

    rng = np.random.default_rng(RANDOM_SEED)
    for _ in range(12):
        starts.append(rng.normal(0.0, 1.0, size=len(keys)).astype(np.float64))

    best_res = None
    best_fun = None
    for x0 in starts:
        res = minimize(neg_ap, x0, method="Nelder-Mead",
                       options={"maxiter": 2000, "xatol": 1e-4, "fatol": 1e-6})
        if best_fun is None or float(res.fun) < best_fun:
            best_res = res
            best_fun = float(res.fun)

    best_w = softmax(best_res.x).astype(np.float32)
    best_ap = float(-best_res.fun)
    return keys, best_w, best_ap


def build_folds():
    return [
        ("f1", pd.Timestamp("2025-02-01"), pd.Timestamp("2025-02-28 23:59:59")),
        ("f2", pd.Timestamp("2025-03-01"), pd.Timestamp("2025-03-31 23:59:59")),
        ("f3", pd.Timestamp("2025-04-01"), pd.Timestamp("2025-04-30 23:59:59")),
        ("f4", pd.Timestamp("2025-05-01"), pd.Timestamp("2025-05-31 23:59:59")),
    ]


def make_time_decay_multiplier(event_ts):
    ts = pd.to_datetime(event_ts)
    if len(ts) > 1:
        ts_min = ts.min()
        ts_max = ts.max()
        span_days = max((ts_max - ts_min).total_seconds() / 86400.0, 1.0)
        recency = ((ts - ts_min) / np.timedelta64(1, "D")) / span_days
        recency = np.asarray(recency, dtype=np.float32)
        return TIME_DECAY_MIN_MULT + (TIME_DECAY_MAX_MULT - TIME_DECAY_MIN_MULT) * recency
    return np.ones(len(ts), dtype=np.float32)


def _green_sample_prob(raw_target, event_ts):
    ts = pd.to_datetime(event_ts)
    border = pd.Timestamp(NEG_SAMPLE_BORDER_STR)
    sample_prob = np.ones(len(raw_target), dtype=np.float32)
    is_green = raw_target == -1
    sample_prob[is_green & (ts >= border)] = 1.0 / NEG_SAMPLE_MOD_RECENT
    sample_prob[is_green & (ts < border)] = 1.0 / NEG_SAMPLE_MOD_OLD
    return sample_prob


def make_importance_weights(raw_target, event_ts):
    cls_mult = np.ones(len(raw_target), dtype=np.float32)
    cls_mult[raw_target == 0] = 2.8
    cls_mult[raw_target == 1] = 6.5
    w = (1.0 / _green_sample_prob(raw_target, event_ts)) * cls_mult * make_time_decay_multiplier(event_ts)
    w = w / np.mean(w)
    return w.astype(np.float32)


def make_labeled_importance_weights(raw_target, event_ts):
    cls_mult = np.ones(len(raw_target), dtype=np.float32)
    cls_mult[raw_target == 0] = 1.35
    cls_mult[raw_target == 1] = 2.6
    w = cls_mult * make_time_decay_multiplier(event_ts)
    mask = raw_target != -1
    if mask.any():
        w = w / np.mean(w[mask])
    return w.astype(np.float32)


def make_suspicious_weights(raw_target, event_ts):
    cls_mult = np.ones(len(raw_target), dtype=np.float32)
    cls_mult[raw_target != -1] = 3.0
    w = (1.0 / _green_sample_prob(raw_target, event_ts)) * cls_mult * make_time_decay_multiplier(event_ts)
    w = w / np.mean(w)
    return w.astype(np.float32)


def audit_and_select_features(
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    feature_cols: List[str],
    cat_cols: List[str],
) -> Tuple[List[str], pd.DataFrame]:
    rows = []
    keep = []
    for col in feature_cols:
        s_tr = train_frame[col]
        s_te = test_frame[col]
        train_nunique = int(s_tr.nunique(dropna=False))
        test_nunique = int(s_te.nunique(dropna=False))
        train_missing = float(s_tr.isna().mean())
        test_missing = float(s_te.isna().mean())
        dominant_share = 0.0
        if (col in cat_cols) or (train_nunique <= 128) or (train_missing >= 0.95):
            vc = s_tr.astype("object").value_counts(dropna=False, normalize=True)
            dominant_share = float(vc.iloc[0]) if len(vc) else 1.0
        drop_reason = "keep"
        if train_nunique <= 1:
            drop_reason = "constant"
        elif dominant_share >= 0.9997:
            drop_reason = "almost_constant"
        elif train_missing >= 0.997 and test_missing >= 0.997:
            drop_reason = "too_sparse"
        elif (col in cat_cols) and (train_nunique >= 50) and (test_nunique <= 2) and (train_missing >= 0.85):
            drop_reason = "collapsed_in_test"
        rows.append({"feature": col, "drop_reason": drop_reason, "train_nunique": train_nunique,
                      "test_nunique": test_nunique, "train_missing_rate": train_missing,
                      "test_missing_rate": test_missing, "dominant_share": dominant_share})
        if drop_reason == "keep":
            keep.append(col)
    audit_df = pd.DataFrame(rows).sort_values(
        ["drop_reason", "train_missing_rate", "dominant_share", "feature"],
        ascending=[True, False, False, True],
    ).reset_index(drop=True)
    return keep, audit_df


def prepare_feature_frames(
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    feature_cols: List[str],
    cat_cols: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    X_train = train_frame[feature_cols].copy()
    X_test = test_frame[feature_cols].copy()
    for c in cat_cols:
        X_train[c] = X_train[c].fillna(-1).astype(np.int64)
        X_test[c] = X_test[c].fillna(-1).astype(np.int64)
    num_cols = [c for c in feature_cols if c not in cat_cols]
    if num_cols:
        medians = X_train[num_cols].median(numeric_only=True)
        X_train[num_cols] = X_train[num_cols].fillna(medians)
        X_test[num_cols] = X_test[num_cols].fillna(medians)
    return X_train, X_test


def prep_lgb_cats(train_frame: pd.DataFrame, test_frame: pd.DataFrame, cat_cols: List[str]):
    train_lgb = train_frame.copy()
    test_lgb = test_frame.copy()
    for col in cat_cols:
        cat = pd.Categorical(train_lgb[col])
        train_lgb[col] = cat.codes.astype(np.int32) + 1
        test_lgb[col] = pd.Categorical(test_lgb[col], categories=cat.categories).codes.astype(np.int32) + 1
    return train_lgb, test_lgb


# ── Feature Engineering ──

def _load_period(path: Path, period_name: str) -> pl.LazyFrame:
    cols_present = [c for c in BASE_COLS if c in pl.scan_parquet(path).columns]
    lf = pl.scan_parquet(path).select(cols_present).with_columns(pl.lit(period_name).alias("period"))
    return lf


def period_frames_for_part(part_id: int) -> pl.LazyFrame:
    custs_lf = pl.scan_parquet(DATA_DIR / f"train_part_{part_id}.parquet").select("customer_id").unique()
    period_frames = []
    if USE_PRETRAIN_HISTORY:
        period_frames.append(_load_period(DATA_DIR / f"pretrain_part_{part_id}.parquet", "pretrain"))
    period_frames.append(_load_period(DATA_DIR / f"train_part_{part_id}.parquet", "train"))
    period_frames.append(
        _load_period(DATA_DIR / "pretest.parquet", "pretest").join(custs_lf, on="customer_id", how="inner")
    )
    period_frames.append(
        _load_period(DATA_DIR / "test.parquet", "test").join(custs_lf, on="customer_id", how="inner")
    )
    return pl.concat(period_frames, how="diagonal_relaxed")


def build_features_for_part(part_id: int, labels_lf: pl.LazyFrame, force: bool = False) -> Tuple[Path, Path]:
    train_out = CACHE_DIR / f"train_features_{FEATURE_TAG}_part_{part_id}.parquet"
    test_out = CACHE_DIR / f"test_features_{FEATURE_TAG}_part_{part_id}.parquet"

    if _exists(train_out) and _exists(test_out) and not force:
        print(f"[part {part_id}] cached")
        return train_out, test_out

    print(f"[part {part_id}] building features...")
    lf = period_frames_for_part(part_id)

    border_expr = pl.lit(NEG_SAMPLE_BORDER_STR).str.strptime(
        pl.Datetime, format="%Y-%m-%d %H:%M:%S", strict=False
    )

    # Missingness flags
    lf = lf.with_columns([
        pl.col("battery").is_null().cast(pl.Int8).alias("battery_missing"),
        pl.col("screen_size").is_null().cast(pl.Int8).alias("screen_missing"),
        pl.col("device_system_version").is_null().cast(pl.Int8).alias("os_ver_missing"),
        pl.col("accept_language").is_null().cast(pl.Int8).alias("accept_language_missing"),
        pl.col("browser_language").is_null().cast(pl.Int8).alias("browser_language_missing"),
        pl.col("developer_tools").is_null().cast(pl.Int8).alias("dev_tools_missing"),
        pl.col("compromised").is_null().cast(pl.Int8).alias("compromised_missing"),
    ])

    # Parse and cast columns
    lf = (
        lf
        .with_columns([
            pl.col("event_dttm").str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M:%S", strict=False).alias("event_ts"),
            pl.col("operaton_amt").cast(pl.Float64).fill_null(0.0).alias("amt"),
            pl.col("session_id").cast(pl.Int64, strict=False).fill_null(-1).alias("session_id"),
            pl.col("event_type_nm").cast(pl.Int32, strict=False).fill_null(-1).alias("event_type_nm"),
            pl.col("event_desc").cast(pl.Int32, strict=False).fill_null(-1).alias("event_desc"),
            pl.col("channel_indicator_type").cast(pl.Int16, strict=False).fill_null(-1).alias("channel_indicator_type"),
            pl.col("channel_indicator_sub_type").cast(pl.Int16, strict=False).fill_null(-1).alias("channel_indicator_sub_type"),
            pl.col("currency_iso_cd").cast(pl.Int16, strict=False).fill_null(-1).alias("currency_iso_cd"),
            pl.col("mcc_code").cast(pl.Int32, strict=False).fill_null(-1).alias("mcc_code_i"),
            pl.col("pos_cd").cast(pl.Int16, strict=False).fill_null(-1).alias("pos_cd"),
            pl.col("timezone").cast(pl.Int32, strict=False).fill_null(-1).alias("timezone"),
            pl.col("operating_system_type").cast(pl.Int16, strict=False).fill_null(-1).alias("operating_system_type"),
            pl.col("phone_voip_call_state").cast(pl.Int8, strict=False).fill_null(-1).alias("phone_voip_call_state"),
            pl.col("web_rdp_connection").cast(pl.Int8, strict=False).fill_null(-1).alias("web_rdp_connection"),
            pl.col("developer_tools").cast(pl.Int8, strict=False).fill_null(-1).alias("developer_tools_i"),
            pl.col("compromised").cast(pl.Int8, strict=False).fill_null(-1).alias("compromised_i"),
            pl.col("accept_language").cast(pl.Int32, strict=False).fill_null(-1).alias("accept_language_i"),
            pl.col("browser_language").cast(pl.Int32, strict=False).fill_null(-1).alias("browser_language_i"),
            pl.col("battery").str.extract(r"(\d{1,3})", 1).cast(pl.Int16, strict=False).fill_null(-1).alias("battery_pct"),
            pl.col("device_system_version").str.extract(r"^(\d+)", 1).cast(pl.Int16, strict=False).fill_null(-1).alias("os_ver_major"),
            pl.col("screen_size").str.extract(r"^(\d+)", 1).cast(pl.Int16, strict=False).fill_null(-1).alias("screen_w"),
            pl.col("screen_size").str.extract(r"x(\d+)$", 1).cast(pl.Int16, strict=False).fill_null(-1).alias("screen_h"),
        ])
        .drop([
            "event_dttm", "operaton_amt", "mcc_code",
            "battery", "device_system_version", "screen_size",
            "developer_tools", "compromised", "accept_language", "browser_language",
        ])
        .with_columns([
            (
                pl.col("screen_w").cast(pl.Int64) * 100_000_000
                + pl.col("screen_h").cast(pl.Int64) * 100_000
                + pl.col("operating_system_type").cast(pl.Int64) * 1000
                + (pl.col("accept_language_i").cast(pl.Int64) % 1000)
            ).alias("device_fp_i")
        ])
        .join(labels_lf, on="event_id", how="left")
        .with_columns([
            pl.when(pl.col("period") == "train")
              .then(pl.when(pl.col("target").is_null()).then(pl.lit(-1)).otherwise(pl.col("target")))
              .otherwise(pl.lit(None))
              .cast(pl.Int8)
              .alias("train_target_raw")
        ])
        .with_columns([
            (
                (pl.col("period") == "train") &
                (pl.col("train_target_raw") == -1) &
                (
                    (
                        (pl.col("event_ts") >= border_expr) &
                        ((pl.struct(["event_id", "customer_id"]).hash(seed=RANDOM_SEED) % NEG_SAMPLE_MOD_RECENT) == 0)
                    )
                    |
                    (
                        (pl.col("event_ts") < border_expr) &
                        ((pl.struct(["event_id", "customer_id"]).hash(seed=RANDOM_SEED + 17) % NEG_SAMPLE_MOD_OLD) == 0)
                    )
                )
            ).alias("keep_green")
        ])
        .with_columns([
            ((pl.col("period") == "train") & ((pl.col("train_target_raw") != -1) | pl.col("keep_green"))).alias("is_train_sample"),
            (pl.col("period") == "test").alias("is_test"),
        ])
    )

    # Temporal features
    two_pi = float(2 * np.pi)
    lf = lf.with_columns([
        pl.col("event_ts").dt.hour().cast(pl.Int8).alias("hour"),
        pl.col("event_ts").dt.weekday().cast(pl.Int8).alias("weekday"),
        pl.col("event_ts").dt.day().cast(pl.Int8).alias("day"),
        pl.col("event_ts").dt.month().cast(pl.Int8).alias("month"),
        pl.col("event_ts").dt.week().cast(pl.Int16).alias("week_of_year"),
        (pl.col("event_ts").dt.weekday() >= 5).cast(pl.Int8).alias("is_weekend"),
        (pl.col("event_ts").dt.hour().is_in([0, 1, 2, 3, 4, 5])).cast(pl.Int8).alias("is_night"),
        pl.col("amt").abs().cast(pl.Float32).alias("amt_abs"),
        pl.col("amt").abs().log1p().cast(pl.Float32).alias("amt_log_abs"),
        (pl.col("amt").abs().log1p() * 4.0).floor().clip(0, 63).cast(pl.Int16).alias("amt_bucket"),
        (pl.col("amt") < 0).cast(pl.Int8).alias("amt_is_negative"),
        (pl.col("screen_w").cast(pl.Int32) * pl.col("screen_h").cast(pl.Int32)).alias("screen_pixels"),
        pl.when((pl.col("screen_w") > 0) & (pl.col("screen_h") > 0))
          .then(pl.col("screen_w").cast(pl.Float32) / (pl.col("screen_h").cast(pl.Float32) + 1e-6))
          .otherwise(0.0).cast(pl.Float32).alias("screen_ratio"),
        pl.col("event_ts").dt.date().alias("event_date"),
        pl.col("event_ts").dt.truncate("1h").alias("event_hour_trunc"),
        (pl.col("event_ts").dt.hour().cast(pl.Float32) * (two_pi / 24.0)).sin().cast(pl.Float32).alias("hour_sin"),
        (pl.col("event_ts").dt.hour().cast(pl.Float32) * (two_pi / 24.0)).cos().cast(pl.Float32).alias("hour_cos"),
        (pl.col("event_ts").dt.weekday().cast(pl.Float32) * (two_pi / 7.0)).sin().cast(pl.Float32).alias("weekday_sin"),
        (pl.col("event_ts").dt.weekday().cast(pl.Float32) * (two_pi / 7.0)).cos().cast(pl.Float32).alias("weekday_cos"),
        (pl.col("event_ts").dt.month().cast(pl.Float32) * (two_pi / 12.0)).sin().cast(pl.Float32).alias("month_sin"),
        (pl.col("event_ts").dt.month().cast(pl.Float32) * (two_pi / 12.0)).cos().cast(pl.Float32).alias("month_cos"),
    ])

    lf = lf.sort(["customer_id", "event_ts", "event_id"])

    # Customer sequential features
    lf = lf.with_columns([
        pl.cum_count("event_id").over("customer_id").cast(pl.Int32).alias("cust_event_idx"),
        pl.col("amt").cum_sum().over("customer_id").alias("cust_cum_amt"),
        (pl.col("amt") * pl.col("amt")).cum_sum().over("customer_id").alias("cust_cum_amt_sq"),
        pl.col("amt").cum_max().over("customer_id").alias("cust_cum_max_amt"),
        pl.col("event_ts").shift(1).over("customer_id").alias("prev_event_ts"),
        pl.col("event_date").shift(1).over("customer_id").alias("prev_event_date"),
        pl.col("amt").shift(1).over("customer_id").alias("prev_amt"),
        pl.col("event_ts").min().over("customer_id").alias("cust_first_ts"),
        pl.col("event_ts").shift(1).over(["customer_id", "event_desc"]).alias("prev_same_desc_ts"),
        pl.col("event_ts").shift(1).over(["customer_id", "event_type_nm"]).alias("prev_same_type_ts"),
        pl.col("event_ts").shift(1).over(["customer_id", "channel_indicator_type"]).alias("prev_same_channel_type_ts"),
        pl.col("event_ts").shift(1).over(["customer_id", "currency_iso_cd"]).alias("prev_same_currency_ts"),
        pl.col("event_ts").shift(1).over(["customer_id", "device_fp_i"]).alias("prev_same_device_ts"),
        pl.col("event_ts").shift(1).over(["customer_id", "mcc_code_i"]).alias("prev_same_mcc_ts"),
        pl.col("event_ts").shift(1).over(["customer_id", "channel_indicator_sub_type"]).alias("prev_same_subtype_ts"),
        (pl.cum_count("event_id").over(["customer_id", "event_desc"]) - 1).cast(pl.Int32).alias("cnt_prev_same_desc"),
        (pl.cum_count("event_id").over(["customer_id", "event_type_nm"]) - 1).cast(pl.Int32).alias("cnt_prev_same_type"),
        (pl.cum_count("event_id").over(["customer_id", "mcc_code_i"]) - 1).cast(pl.Int32).alias("cnt_prev_same_mcc"),
        (pl.cum_count("event_id").over(["customer_id", "channel_indicator_sub_type"]) - 1).cast(pl.Int32).alias("cnt_prev_same_subtype"),
        (pl.cum_count("event_id").over(["customer_id", "channel_indicator_type"]) - 1).cast(pl.Int32).alias("cnt_prev_same_channel_type"),
        (pl.cum_count("event_id").over(["customer_id", "currency_iso_cd"]) - 1).cast(pl.Int32).alias("cnt_prev_same_currency"),
        (pl.cum_count("event_id").over(["customer_id", "device_fp_i"]) - 1).cast(pl.Int32).alias("cnt_prev_same_device"),
        (pl.cum_count("event_id").over(["customer_id", "session_id"]) - 1).cast(pl.Int32).alias("cnt_prev_same_session"),
        (pl.cum_count("event_id").over(["customer_id", "event_date"]) - 1).cast(pl.Int32).alias("events_before_today"),
        (pl.cum_count("event_id").over(["customer_id", "event_hour_trunc"]) - 1).cast(pl.Int32).alias("events_before_hour"),
        (pl.cum_count("event_id").over(["customer_id", "event_hour_trunc"]) - 1).cast(pl.Int32).alias("cnt_events_this_hour"),
        (pl.cum_count("event_id").over(["customer_id", "session_id"]) - 1).cast(pl.Int32).alias("session_event_idx"),
        pl.col("event_ts").min().over(["customer_id", "session_id"]).alias("session_start_ts"),
        pl.col("amt").cum_sum().over(["customer_id", "session_id"]).alias("session_cum_amt"),
    ])

    # Derived sequential features
    lf = lf.with_columns([
        (pl.col("cust_event_idx") - 1).cast(pl.Int32).alias("cust_prev_events"),
        ((pl.col("event_ts") - pl.col("cust_first_ts")).dt.total_seconds() / 86400.0).fill_null(0).cast(pl.Float32).alias("days_since_first_event"),
        pl.when(pl.col("prev_event_ts").is_not_null())
          .then((pl.col("event_ts") - pl.col("prev_event_ts")).dt.total_seconds())
          .otherwise(-1).cast(pl.Int32).alias("sec_since_prev_event"),
        (pl.col("amt") - pl.col("prev_amt").fill_null(0.0)).cast(pl.Float32).alias("amt_delta_prev"),
        pl.when(pl.col("cust_event_idx") > 1)
          .then((pl.col("cust_cum_amt") - pl.col("amt")) / (pl.col("cust_event_idx") - 1))
          .otherwise(0.0).cast(pl.Float32).alias("cust_prev_amt_mean"),
        pl.col("cust_cum_max_amt").shift(1).over("customer_id").fill_null(0.0).cast(pl.Float32).alias("cust_prev_max_amt"),
        pl.when(pl.col("prev_same_desc_ts").is_not_null())
          .then((pl.col("event_ts") - pl.col("prev_same_desc_ts")).dt.total_seconds())
          .otherwise(-1).cast(pl.Int32).alias("sec_since_prev_same_desc"),
        pl.when(pl.col("prev_same_type_ts").is_not_null())
          .then((pl.col("event_ts") - pl.col("prev_same_type_ts")).dt.total_seconds())
          .otherwise(-1).cast(pl.Int32).alias("sec_since_prev_same_type"),
        pl.when(pl.col("prev_same_channel_type_ts").is_not_null())
          .then((pl.col("event_ts") - pl.col("prev_same_channel_type_ts")).dt.total_seconds())
          .otherwise(-1).cast(pl.Int32).alias("sec_since_prev_same_channel_type"),
        pl.when(pl.col("prev_same_currency_ts").is_not_null())
          .then((pl.col("event_ts") - pl.col("prev_same_currency_ts")).dt.total_seconds())
          .otherwise(-1).cast(pl.Int32).alias("sec_since_prev_same_currency"),
        pl.when(pl.col("prev_same_device_ts").is_not_null())
          .then((pl.col("event_ts") - pl.col("prev_same_device_ts")).dt.total_seconds())
          .otherwise(-1).cast(pl.Int32).alias("sec_since_prev_same_device"),
        pl.when(pl.col("prev_same_mcc_ts").is_not_null())
          .then((pl.col("event_ts") - pl.col("prev_same_mcc_ts")).dt.total_seconds())
          .otherwise(-1).cast(pl.Int32).alias("sec_since_prev_same_mcc"),
        pl.when(pl.col("prev_same_subtype_ts").is_not_null())
          .then((pl.col("event_ts") - pl.col("prev_same_subtype_ts")).dt.total_seconds())
          .otherwise(-1).cast(pl.Int32).alias("sec_since_prev_same_channel_subtype"),
        pl.when(pl.col("session_start_ts").is_not_null())
          .then((pl.col("event_ts") - pl.col("session_start_ts")).dt.total_seconds())
          .otherwise(-1).cast(pl.Int32).alias("sec_since_session_start"),
        (pl.col("session_cum_amt") - pl.col("amt")).cast(pl.Float32).alias("session_amt_before"),
    ])

    lf = lf.with_columns([
        pl.when(pl.col("sec_since_prev_event") >= 0)
          .then(pl.col("sec_since_prev_event").cast(pl.Float32) / 86400.0)
          .otherwise(-1.0).cast(pl.Float32).alias("days_since_prev_event"),
        pl.when(pl.col("cust_event_idx") > 2)
          .then(
              (
                  ((pl.col("cust_cum_amt_sq") - pl.col("amt") * pl.col("amt")) / (pl.col("cust_event_idx") - 1))
                  - pl.col("cust_prev_amt_mean") * pl.col("cust_prev_amt_mean")
              ).clip(lower_bound=0.0).sqrt()
          )
          .otherwise(0.0).cast(pl.Float32).alias("cust_prev_amt_std"),
    ])

    lf = lf.with_columns([
        pl.when(pl.col("cust_prev_amt_std") > 1e-6)
          .then((pl.col("amt") - pl.col("cust_prev_amt_mean")) / (pl.col("cust_prev_amt_std") + 1e-6))
          .otherwise(0.0).cast(pl.Float32).alias("amt_zscore"),
        pl.when(pl.col("cust_prev_max_amt").abs() > 1e-6)
          .then(pl.col("amt") / (pl.col("cust_prev_max_amt") + 1e-6))
          .otherwise(0.0).cast(pl.Float32).alias("amt_vs_prev_max"),
        pl.when(pl.col("cust_prev_amt_mean").abs() > 1e-6)
          .then(pl.col("amt") / (pl.col("cust_prev_amt_mean") + 1e-6))
          .otherwise(0.0).cast(pl.Float32).alias("amt_vs_prev_mean"),
    ])

    # Device novelty features
    lf = lf.with_columns([
        (pl.cum_count("event_id").over(["customer_id", "device_fp_i"]) - 1).cast(pl.Int32).alias("cust_prev_same_device"),
        (pl.cum_count("event_id").over(["customer_id", "timezone"]) - 1).cast(pl.Int32).alias("cust_prev_same_timezone"),
        (pl.cum_count("event_id").over(["customer_id", "operating_system_type"]) - 1).cast(pl.Int32).alias("cust_prev_same_os"),
        (pl.cum_count("event_id").over(["customer_id", "channel_indicator_type"]) - 1).cast(pl.Int32).alias("cust_prev_same_channel"),
        (pl.cum_count("event_id").over(["customer_id", "channel_indicator_sub_type"]) - 1).cast(pl.Int32).alias("cust_prev_same_channel_sub"),
    ])

    lf = lf.with_columns([
        (pl.col("cust_prev_same_device") == 0).cast(pl.Int8).alias("is_new_device_for_customer"),
        (pl.col("cust_prev_same_timezone") == 0).cast(pl.Int8).alias("is_new_timezone_for_customer"),
        (pl.col("cnt_prev_same_desc") == 0).cast(pl.Int8).alias("is_new_desc_for_customer"),
        (pl.col("cnt_prev_same_mcc") == 0).cast(pl.Int8).alias("is_new_mcc_for_customer"),
        (pl.col("cust_prev_same_channel_sub") == 0).cast(pl.Int8).alias("is_new_subtype_for_customer"),
        (pl.col("cust_prev_same_os") == 0).cast(pl.Int8).alias("is_new_os_for_customer"),
    ])

    # Label history features
    lf = lf.with_columns([
        ((pl.col("period") == "train") & (pl.col("train_target_raw") == 1)).cast(pl.Int8).alias("is_red_lbl"),
        ((pl.col("period") == "train") & (pl.col("train_target_raw") == 0)).cast(pl.Int8).alias("is_yellow_lbl"),
    ]).with_columns([
        (pl.col("is_red_lbl") + pl.col("is_yellow_lbl")).cast(pl.Int8).alias("is_labeled_fb")
    ])

    lf = lf.with_columns([
        pl.col("is_red_lbl").cum_sum().over("customer_id").cast(pl.Int32).alias("_cust_red_cum"),
        pl.col("is_yellow_lbl").cum_sum().over("customer_id").cast(pl.Int32).alias("_cust_yellow_cum"),
        pl.col("is_labeled_fb").cum_sum().over("customer_id").cast(pl.Int32).alias("_cust_lab_cum"),
        pl.when(pl.col("is_red_lbl") == 1).then(pl.col("event_ts")).otherwise(None).alias("red_lbl_ts"),
        pl.when(pl.col("is_yellow_lbl") == 1).then(pl.col("event_ts")).otherwise(None).alias("yellow_lbl_ts"),
    ])

    lf = lf.with_columns([
        (pl.col("_cust_red_cum") - pl.col("is_red_lbl")).cast(pl.Int32).alias("cust_prev_red_lbl_cnt"),
        (pl.col("_cust_yellow_cum") - pl.col("is_yellow_lbl")).cast(pl.Int32).alias("cust_prev_yellow_lbl_cnt"),
        (pl.col("_cust_lab_cum") - pl.col("is_labeled_fb")).cast(pl.Int32).alias("cust_prev_labeled_cnt"),
    ])

    lf = lf.with_columns([
        pl.col("red_lbl_ts").shift(1).over("customer_id").alias("_prev_red_shift"),
        pl.col("yellow_lbl_ts").shift(1).over("customer_id").alias("_prev_yellow_shift"),
    ]).with_columns([
        pl.col("_prev_red_shift").forward_fill().over("customer_id").alias("prev_red_lbl_ts"),
        pl.col("_prev_yellow_shift").forward_fill().over("customer_id").alias("prev_yellow_lbl_ts"),
    ]).drop(["_prev_red_shift", "_prev_yellow_shift"])

    lf = lf.with_columns([
        pl.when(pl.col("prev_red_lbl_ts").is_not_null())
          .then((pl.col("event_ts") - pl.col("prev_red_lbl_ts")).dt.total_seconds())
          .otherwise(-1).cast(pl.Int32).alias("sec_since_prev_red_lbl"),
        pl.when(pl.col("prev_yellow_lbl_ts").is_not_null())
          .then((pl.col("event_ts") - pl.col("prev_yellow_lbl_ts")).dt.total_seconds())
          .otherwise(-1).cast(pl.Int32).alias("sec_since_prev_yellow_lbl"),
        ((pl.col("cust_prev_red_lbl_cnt") + 0.1) / (pl.col("cust_prev_labeled_cnt") + 1.0)).cast(pl.Float32).alias("cust_prev_red_lbl_rate"),
        ((pl.col("cust_prev_yellow_lbl_cnt") + 0.1) / (pl.col("cust_prev_labeled_cnt") + 1.0)).cast(pl.Float32).alias("cust_prev_yellow_lbl_rate"),
        (((pl.col("cust_prev_red_lbl_cnt") + pl.col("cust_prev_yellow_lbl_cnt")) + 0.1) / (pl.col("cust_prev_events") + 1.0)).cast(pl.Float32).alias("cust_prev_susp_lbl_rate"),
        (pl.col("cust_prev_red_lbl_cnt") > 0).cast(pl.Int8).alias("cust_prev_any_red_flag"),
        (pl.col("cust_prev_yellow_lbl_cnt") > 0).cast(pl.Int8).alias("cust_prev_any_yellow_flag"),
    ])

    # Amount aggregations per desc/device
    lf = lf.with_columns([
        pl.col("amt").cum_sum().over(["customer_id", "event_desc"]).alias("_cum_amt_same_desc"),
        pl.col("amt").cum_sum().over(["customer_id", "device_fp_i"]).alias("_cum_amt_same_device"),
        pl.cum_count("event_id").over(["customer_id", "event_desc"]).cast(pl.Int32).alias("_cnt_same_desc"),
        pl.cum_count("event_id").over(["customer_id", "device_fp_i"]).cast(pl.Int32).alias("_cnt_same_device"),
    ]).with_columns([
        pl.when(pl.col("_cnt_same_desc") > 1)
          .then((pl.col("_cum_amt_same_desc") - pl.col("amt")) / (pl.col("_cnt_same_desc") - 1))
          .otherwise(0.0).cast(pl.Float32).alias("cust_prev_mean_amt_same_desc"),
        pl.when(pl.col("_cnt_same_device") > 1)
          .then((pl.col("_cum_amt_same_device") - pl.col("amt")) / (pl.col("_cnt_same_device") - 1))
          .otherwise(0.0).cast(pl.Float32).alias("cust_prev_mean_amt_same_device"),
    ]).drop(["_cum_amt_same_desc", "_cum_amt_same_device", "_cnt_same_desc", "_cnt_same_device"])

    lf = lf.with_columns([
        pl.when(pl.col("cust_prev_mean_amt_same_desc").abs() > 1e-6)
          .then(pl.col("amt") / (pl.col("cust_prev_mean_amt_same_desc") + 1e-6))
          .otherwise(0.0).cast(pl.Float32).alias("amt_vs_same_desc_mean"),
        pl.when(pl.col("cust_prev_mean_amt_same_device").abs() > 1e-6)
          .then(pl.col("amt") / (pl.col("cust_prev_mean_amt_same_device") + 1e-6))
          .otherwise(0.0).cast(pl.Float32).alias("amt_vs_same_device_mean"),
    ])

    lf = lf.with_columns([
        pl.when(pl.col("is_train_sample"))
          .then((pl.col("train_target_raw") == 1).cast(pl.Int8))
          .otherwise(pl.lit(None))
          .alias("target_bin")
    ])

    # Device-level features (sorted by event_ts globally)
    lf = lf.sort(["event_ts", "event_id"])

    lf = lf.with_columns([
        (pl.cum_count("event_id").over("device_fp_i") - 1).cast(pl.Int32).alias("device_prev_ops"),
        (pl.cum_count("event_id").over(["device_fp_i", "customer_id"]) == 1).cast(pl.Int8).alias("_is_first_customer_on_device"),
        (pl.cum_count("event_id").over(["device_fp_i", "session_id"]) == 1).cast(pl.Int8).alias("_is_first_session_on_device"),
    ])

    lf = lf.with_columns([
        (pl.col("_is_first_customer_on_device").cum_sum().over("device_fp_i") - pl.col("_is_first_customer_on_device")).cast(pl.Int32).alias("device_prev_unique_customers"),
        (pl.col("_is_first_session_on_device").cum_sum().over("device_fp_i") - pl.col("_is_first_session_on_device")).cast(pl.Int32).alias("device_prev_unique_sessions"),
    ]).drop(["_is_first_customer_on_device", "_is_first_session_on_device"])

    lf = lf.with_columns([
        pl.col("device_prev_ops").log1p().cast(pl.Float32).alias("device_prev_ops_log"),
        pl.col("device_prev_unique_customers").log1p().cast(pl.Float32).alias("device_prev_unique_customers_log"),
        pl.col("device_prev_unique_sessions").log1p().cast(pl.Float32).alias("device_prev_unique_sessions_log"),
        pl.when(pl.col("device_prev_ops") > 0)
          .then(pl.col("device_prev_unique_customers") / (pl.col("device_prev_ops") + 1e-6))
          .otherwise(0.0).cast(pl.Float32).alias("device_customer_diversity"),
    ])

    lf = lf.with_columns([
        (pl.cum_count("event_id").over(["device_fp_i", "customer_id"]) - 1).cast(pl.Int32).alias("device_prev_same_customer"),
        (pl.cum_count("event_id").over(["device_fp_i", "event_desc"]) - 1).cast(pl.Int32).alias("device_prev_same_desc"),
        (pl.cum_count("event_id").over(["device_fp_i", "mcc_code_i"]) - 1).cast(pl.Int32).alias("device_prev_same_mcc"),
        (pl.cum_count("event_id").over(["device_fp_i", "timezone"]) - 1).cast(pl.Int32).alias("device_prev_same_timezone"),
        (pl.cum_count("event_id").over(["device_fp_i", "channel_indicator_sub_type"]) - 1).cast(pl.Int32).alias("device_prev_same_subtype"),
    ])

    # Prior red rates
    prior_keys = [
        "event_desc", "mcc_code_i", "timezone", "event_type_nm",
        "channel_indicator_type", "channel_indicator_sub_type", "pos_cd",
        "operating_system_type", "accept_language_i", "browser_language_i", "device_fp_i",
    ]

    for key in prior_keys:
        lf = lf.with_columns([
            pl.col("is_red_lbl").cum_sum().over(key).cast(pl.Int32).alias(f"_{key}_red_cum"),
            pl.col("is_labeled_fb").cum_sum().over(key).cast(pl.Int32).alias(f"_{key}_lab_cum"),
        ]).with_columns([
            (pl.col(f"_{key}_red_cum") - pl.col("is_red_lbl")).cast(pl.Int32).alias(f"{key}_prev_red_cnt"),
            (pl.col(f"_{key}_lab_cum") - pl.col("is_labeled_fb")).cast(pl.Int32).alias(f"{key}_prev_lab_cnt"),
            ((pl.col(f"_{key}_red_cum") - pl.col("is_red_lbl") + 0.25) / (pl.col(f"_{key}_lab_cum") - pl.col("is_labeled_fb") + 2.0)).cast(pl.Float32).alias(f"prior_{key}_red_rate"),
        ]).drop([f"_{key}_red_cum", f"_{key}_lab_cum"])

    lf = lf.with_columns([
        (pl.col("prior_event_desc_red_rate") * (1 + pl.col("is_new_desc_for_customer").cast(pl.Float32))).cast(pl.Float32).alias("risk_new_desc_x_prior"),
        (pl.col("prior_device_fp_i_red_rate") * (1 + pl.col("is_new_device_for_customer").cast(pl.Float32))).cast(pl.Float32).alias("risk_new_device_x_prior"),
        (pl.col("prior_timezone_red_rate") * (1 + pl.col("is_new_timezone_for_customer").cast(pl.Float32))).cast(pl.Float32).alias("risk_new_tz_x_prior"),
        (pl.col("prior_mcc_code_i_red_rate") * (1 + pl.col("is_new_mcc_for_customer").cast(pl.Float32))).cast(pl.Float32).alias("risk_new_mcc_x_prior"),
    ])

    # Rolling windows
    print(f"[part {part_id}] computing rolling windows with Polars...")
    lf = lf.sort(["customer_id", "event_ts", "event_id"])
    rolling_count_expr = pl.col("amt").is_not_null().cast(pl.Int32)
    lf = lf.with_columns([
        pl.col("amt").rolling_sum_by("event_ts", window_size="1h", closed="left").over("customer_id").fill_null(0.0).cast(pl.Float32).alias("amt_sum_last_1h"),
        rolling_count_expr.rolling_sum_by("event_ts", window_size="1h", closed="left").over("customer_id").fill_null(0).cast(pl.Int32).alias("cnt_last_1h"),
        pl.col("amt").rolling_sum_by("event_ts", window_size="24h", closed="left").over("customer_id").fill_null(0.0).cast(pl.Float32).alias("amt_sum_last_24h"),
        rolling_count_expr.rolling_sum_by("event_ts", window_size="24h", closed="left").over("customer_id").fill_null(0).cast(pl.Int32).alias("cnt_last_24h"),
        pl.col("amt").rolling_max_by("event_ts", window_size="24h", closed="left").over("customer_id").fill_null(0.0).cast(pl.Float32).alias("max_amt_last_24h"),
    ])

    lf = lf.with_columns([
        pl.when(pl.col("amt_sum_last_1h").abs() > 1.0)
          .then(pl.col("amt") / (pl.col("amt_sum_last_1h").abs() + 1.0))
          .otherwise(0.0).cast(pl.Float32).alias("amt_vs_1h_sum"),
        pl.when(pl.col("amt_sum_last_24h").abs() > 1.0)
          .then(pl.col("amt") / (pl.col("amt_sum_last_24h").abs() + 1.0))
          .otherwise(0.0).cast(pl.Float32).alias("amt_vs_24h_sum"),
    ])

    gc.collect()

    final_cols = dedupe(SAVE_META_COLS + MODEL_FEATURE_COLS)
    train_lf = lf.filter(pl.col("is_train_sample")).select(final_cols)
    test_lf = lf.filter(pl.col("is_test")).select(final_cols)

    train_lf.sink_parquet(train_out)
    test_lf.sink_parquet(test_out)

    train_rows = pl.scan_parquet(train_out).select(pl.len()).collect().item()
    test_rows = pl.scan_parquet(test_out).select(pl.len()).collect().item()

    print(f"[part {part_id}] train rows: {train_rows:,}")
    print(f"[part {part_id}] test  rows: {test_rows:,}")

    del train_lf, test_lf, lf
    gc.collect()
    return train_out, test_out


# ── Main pipeline ──

def run() -> Path:
    """Run the full pipeline1st pipeline and return path to submission_MINE.csv."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.random.seed(RANDOM_SEED)

    labels_lf = (
        pl.scan_parquet(DATA_DIR / "train_labels.parquet")
        .select([pl.col("event_id"), pl.col("target").cast(pl.Int8)])
    )

    # ── Step 1: Build features ──
    merged_train_path = CACHE_DIR / f"train_features_{FEATURE_TAG}_full.parquet"
    merged_test_path = CACHE_DIR / f"test_features_{FEATURE_TAG}_full.parquet"

    reuse_full_cache = _exists(merged_train_path) and _exists(merged_test_path) and (not FORCE_REBUILD_FEATURES)
    if reuse_full_cache:
        print("[cache] found merged full feature files, skipping part rebuild")
    else:
        train_paths = []
        test_paths = []
        for part_id in PART_IDS:
            tr_path, te_path = build_features_for_part(part_id, labels_lf, force=FORCE_REBUILD_FEATURES)
            train_paths.append(str(tr_path))
            test_paths.append(str(te_path))
            gc.collect()

        if MERGE_ALL:
            if not _exists(merged_train_path) or FORCE_REBUILD_FEATURES:
                print("[merge] train...")
                pl.concat([pl.scan_parquet(p) for p in train_paths], how="vertical_relaxed").sink_parquet(merged_train_path)
            if not _exists(merged_test_path) or FORCE_REBUILD_FEATURES:
                print("[merge] test...")
                pl.concat([pl.scan_parquet(p) for p in test_paths], how="vertical_relaxed").sink_parquet(merged_test_path)

    # ── Step 2: Load into pandas ──
    print("[cache] loading merged feature files")
    train_df = pd.read_parquet(merged_train_path)
    test_df = pd.read_parquet(merged_test_path)

    train_df["event_ts"] = pd.to_datetime(train_df["event_ts"])
    test_df["event_ts"] = pd.to_datetime(test_df["event_ts"])
    train_df = train_df.sort_values(["event_ts", "event_id"]).reset_index(drop=True)
    test_df = test_df.sort_values(["event_ts", "event_id"]).reset_index(drop=True)

    print("train_df.shape =", train_df.shape)
    print("test_df.shape  =", test_df.shape)

    # ── Step 3: Feature audit + prepare ──
    raw_feature_cols = [c for c in train_df.columns if c not in SAVE_META_COLS]
    feature_cols, feature_audit_df = audit_and_select_features(train_df, test_df, raw_feature_cols, CAT_COLS)
    manual_drop_cols = [c for c in MANUAL_DROP_COLS if c in feature_cols]
    if manual_drop_cols:
        feature_cols = [c for c in feature_cols if c not in manual_drop_cols]
        print("Manual dropped features:", manual_drop_cols)
    cat_feature_cols = [c for c in CAT_COLS if c in feature_cols]

    print("Feature count after audit:", len(feature_cols))
    print("Categorical feature count:", len(cat_feature_cols))

    X_all, X_test = prepare_feature_frames(train_df, test_df, feature_cols, cat_feature_cols)

    for c in feature_cols:
        if c in cat_feature_cols:
            X_all[c] = X_all[c].astype(np.int32)
            X_test[c] = X_test[c].astype(np.int32)
        elif pd.api.types.is_integer_dtype(X_all[c]):
            X_all[c] = X_all[c].astype(np.int32)
            X_test[c] = X_test[c].astype(np.int32)
        else:
            X_all[c] = X_all[c].astype(np.float32)
            X_test[c] = X_test[c].astype(np.float32)

    y_main_all = train_df["target_bin"].fillna(0).astype(np.int8).values
    raw_target = train_df["train_target_raw"].astype(np.int8).values
    w_main_all = make_importance_weights(raw_target, train_df["event_ts"].values)
    w_labeled_all = make_labeled_importance_weights(raw_target, train_df["event_ts"].values)
    y_susp_all = (raw_target != -1).astype(np.int8)
    w_susp_all = make_suspicious_weights(raw_target, train_df["event_ts"].values)

    X_all_lgb, X_test_lgb = prep_lgb_cats(X_all, X_test, cat_feature_cols)
    folds = build_folds()

    oof = pd.DataFrame({
        "event_id": train_df["event_id"].values,
        "event_ts": train_df["event_ts"].values,
        "y": y_main_all,
        "raw_target": raw_target,
        "lgb": np.nan, "cb": np.nan, "cb_lbl": np.nan, "lgb_susp": np.nan,
    })

    # ── Step 4: LGB main folds ──
    params_lgb = {
        "n_estimators": 3000, "learning_rate": 0.03, "num_leaves": 255, "max_depth": -1,
        "min_child_samples": 100, "subsample": 0.8, "subsample_freq": 1, "colsample_bytree": 0.8,
        "reg_alpha": 0.2, "reg_lambda": 5.0, "objective": "binary",
        "random_state": RANDOM_SEED, "n_jobs": -1, "max_bin": 255, "force_col_wise": True,
    }
    if USE_GPU:
        params_lgb.update({"device_type": "gpu", "gpu_platform_id": 0, "gpu_device_id": GPU_DEVICE})

    lgb_best_iters = []
    for fold_name, val_start, val_end in folds:
        val_mask = (train_df["event_ts"] >= val_start) & (train_df["event_ts"] <= val_end)
        tr_mask = train_df["event_ts"] < val_start
        if val_mask.sum() == 0 or tr_mask.sum() == 0:
            continue
        if np.unique(y_main_all[tr_mask]).size < 2 or np.unique(y_main_all[val_mask]).size < 2:
            continue
        print(f"{fold_name}: LGB valid [{val_start} .. {val_end}]")
        model = lgb.LGBMClassifier(**params_lgb)
        try:
            model.fit(
                X_all_lgb.loc[tr_mask], y_main_all[tr_mask], sample_weight=w_main_all[tr_mask],
                eval_set=[(X_all_lgb.loc[val_mask], y_main_all[val_mask])],
                eval_sample_weight=[w_main_all[val_mask]],
                eval_metric=lambda yt, yp: ("ap", average_precision_score(yt, yp), True),
                categorical_feature=cat_feature_cols,
                callbacks=[lgb.early_stopping(200), lgb.log_evaluation(200)],
            )
        except Exception as e:
            if USE_GPU:
                print("LGBM GPU failed, fallback to CPU:", e)
                params_cpu = {k: v for k, v in params_lgb.items() if k not in ("device_type", "gpu_platform_id", "gpu_device_id")}
                model = lgb.LGBMClassifier(**params_cpu)
                model.fit(
                    X_all_lgb.loc[tr_mask], y_main_all[tr_mask], sample_weight=w_main_all[tr_mask],
                    eval_set=[(X_all_lgb.loc[val_mask], y_main_all[val_mask])],
                    eval_sample_weight=[w_main_all[val_mask]],
                    eval_metric=lambda yt, yp: ("ap", average_precision_score(yt, yp), True),
                    categorical_feature=cat_feature_cols,
                    callbacks=[lgb.early_stopping(200), lgb.log_evaluation(200)],
                )
            else:
                raise
        pred_val = model.predict_proba(X_all_lgb.loc[val_mask])[:, 1]
        ap = pr_auc(y_main_all[val_mask], pred_val)
        best_iter = int(getattr(model, "best_iteration_", 0) or params_lgb["n_estimators"])
        oof.loc[val_mask, "lgb"] = pred_val
        lgb_best_iters.append(best_iter)
        print({"ap_lgb": round(ap, 6), "best_iter_lgb": best_iter})
        del model; gc.collect()

    # ── Step 5: LGB suspicious folds ──
    lgb_susp_best_iters = []
    if ENABLE_LGB_SUSP_HEAD:
        params_lgb_susp = dict(params_lgb)
        params_lgb_susp.update({
            "n_estimators": 2200, "learning_rate": 0.04, "num_leaves": 191,
            "min_child_samples": 140, "subsample": 0.85, "colsample_bytree": 0.85,
            "reg_alpha": 0.1, "reg_lambda": 4.0,
        })
        for fold_name, val_start, val_end in folds:
            val_mask = (train_df["event_ts"] >= val_start) & (train_df["event_ts"] <= val_end)
            tr_mask = train_df["event_ts"] < val_start
            if val_mask.sum() == 0 or tr_mask.sum() == 0:
                continue
            if np.unique(y_susp_all[tr_mask]).size < 2 or np.unique(y_susp_all[val_mask]).size < 2:
                continue
            print(f"{fold_name}: suspicious LGB valid [{val_start} .. {val_end}]")
            model = lgb.LGBMClassifier(**params_lgb_susp)
            try:
                model.fit(
                    X_all_lgb.loc[tr_mask], y_susp_all[tr_mask], sample_weight=w_susp_all[tr_mask],
                    eval_set=[(X_all_lgb.loc[val_mask], y_susp_all[val_mask])],
                    eval_sample_weight=[w_susp_all[val_mask]],
                    eval_metric=lambda yt, yp: ("ap", average_precision_score(yt, yp), True),
                    categorical_feature=cat_feature_cols,
                    callbacks=[lgb.early_stopping(150), lgb.log_evaluation(200)],
                )
            except Exception as e:
                if USE_GPU:
                    print("LGB suspicious GPU failed, fallback to CPU:", e)
                    params_cpu = {k: v for k, v in params_lgb_susp.items() if k not in ("device_type", "gpu_platform_id", "gpu_device_id")}
                    model = lgb.LGBMClassifier(**params_cpu)
                    model.fit(
                        X_all_lgb.loc[tr_mask], y_susp_all[tr_mask], sample_weight=w_susp_all[tr_mask],
                        eval_set=[(X_all_lgb.loc[val_mask], y_susp_all[val_mask])],
                        eval_sample_weight=[w_susp_all[val_mask]],
                        eval_metric=lambda yt, yp: ("ap", average_precision_score(yt, yp), True),
                        categorical_feature=cat_feature_cols,
                        callbacks=[lgb.early_stopping(150), lgb.log_evaluation(200)],
                    )
                else:
                    raise
            pred_val = model.predict_proba(X_all_lgb.loc[val_mask])[:, 1]
            best_iter = int(getattr(model, "best_iteration_", 0) or params_lgb_susp["n_estimators"])
            oof.loc[val_mask, "lgb_susp"] = pred_val
            lgb_susp_best_iters.append(best_iter)
            del model; gc.collect()
    else:
        oof["lgb_susp"] = oof["lgb"]
    if oof["lgb_susp"].notna().sum() == 0:
        oof["lgb_susp"] = oof["lgb"]

    # ── Step 6: CatBoost main folds ──
    params_cb = {
        "iterations": 2500, "learning_rate": 0.06, "depth": 9, "l2_leaf_reg": 3.0,
        "bootstrap_type": "Bernoulli", "subsample": 0.8, "loss_function": "Logloss",
        "eval_metric": "PRAUC", "random_seed": RANDOM_SEED, "od_type": "Iter", "od_wait": 250,
        "verbose": 200, "allow_writing_files": False, "task_type": "CPU",
    }
    if USE_GPU:
        params_cb["task_type"] = "GPU"
        params_cb["devices"] = str(GPU_DEVICE)

    cb_best_iters = []
    for fold_name, val_start, val_end in folds:
        val_mask = (train_df["event_ts"] >= val_start) & (train_df["event_ts"] <= val_end)
        tr_mask = train_df["event_ts"] < val_start
        if val_mask.sum() == 0 or tr_mask.sum() == 0:
            continue
        if np.unique(y_main_all[tr_mask]).size < 2 or np.unique(y_main_all[val_mask]).size < 2:
            continue
        print(f"{fold_name}: CatBoost valid [{val_start} .. {val_end}]")
        train_pool = Pool(X_all.loc[tr_mask], y_main_all[tr_mask], cat_features=cat_feature_cols, weight=w_main_all[tr_mask])
        val_pool = Pool(X_all.loc[val_mask], y_main_all[val_mask], cat_features=cat_feature_cols, weight=w_main_all[val_mask])
        model = CatBoostClassifier(**params_cb)
        try:
            model.fit(train_pool, eval_set=val_pool, use_best_model=True)
        except Exception as e:
            print("CatBoost fallback:", e)
            params_try = dict(params_cb)
            if params_try.get("task_type") == "GPU":
                params_try["task_type"] = "CPU"
                params_try.pop("devices", None)
            try:
                model = CatBoostClassifier(**params_try)
                model.fit(train_pool, eval_set=val_pool, use_best_model=True)
            except Exception as e2:
                params_try["eval_metric"] = "AUC"
                model = CatBoostClassifier(**params_try)
                model.fit(train_pool, eval_set=val_pool, use_best_model=True)
        pred_val = model.predict_proba(X_all.loc[val_mask])[:, 1]
        ap = pr_auc(y_main_all[val_mask], pred_val)
        best_iter = int((model.get_best_iteration() or params_cb["iterations"]) + (1 if model.get_best_iteration() is not None else 0))
        oof.loc[val_mask, "cb"] = pred_val
        cb_best_iters.append(best_iter)
        print({"ap_cb": round(ap, 6), "best_iter_cb": best_iter})
        del model, train_pool, val_pool; gc.collect()

    # ── Step 7: CatBoost label-head folds ──
    cb_lbl_best_iters = []
    if ENABLE_CB_LABEL_HEAD:
        labeled_mask_all = raw_target != -1
        params_cb_lbl = {
            "iterations": 1800, "learning_rate": 0.05, "depth": 8, "l2_leaf_reg": 4.0,
            "bootstrap_type": "Bernoulli", "subsample": 0.85, "loss_function": "Logloss",
            "eval_metric": "PRAUC", "random_seed": RANDOM_SEED, "od_type": "Iter", "od_wait": 200,
            "verbose": 200, "allow_writing_files": False, "task_type": "CPU",
        }
        if USE_GPU:
            params_cb_lbl["task_type"] = "GPU"
            params_cb_lbl["devices"] = str(GPU_DEVICE)

        for fold_name, val_start, val_end in folds:
            val_mask_full = (train_df["event_ts"] >= val_start) & (train_df["event_ts"] <= val_end)
            tr_mask_full = train_df["event_ts"] < val_start
            val_mask = val_mask_full & labeled_mask_all
            tr_mask = tr_mask_full & labeled_mask_all
            if val_mask.sum() == 0 or tr_mask.sum() == 0:
                continue
            if np.unique(y_main_all[tr_mask]).size < 2 or np.unique(y_main_all[val_mask]).size < 2:
                continue
            print(f"{fold_name}: CatBoost red-vs-yellow valid [{val_start} .. {val_end}]")
            train_pool = Pool(X_all.loc[tr_mask], y_main_all[tr_mask], cat_features=cat_feature_cols, weight=w_labeled_all[tr_mask])
            val_pool = Pool(X_all.loc[val_mask], y_main_all[val_mask], cat_features=cat_feature_cols, weight=w_labeled_all[val_mask])
            model = CatBoostClassifier(**params_cb_lbl)
            try:
                model.fit(train_pool, eval_set=val_pool, use_best_model=True)
            except Exception as e:
                params_try = dict(params_cb_lbl)
                if params_try.get("task_type") == "GPU":
                    params_try["task_type"] = "CPU"
                    params_try.pop("devices", None)
                try:
                    model = CatBoostClassifier(**params_try)
                    model.fit(train_pool, eval_set=val_pool, use_best_model=True)
                except Exception as e2:
                    params_try["eval_metric"] = "AUC"
                    model = CatBoostClassifier(**params_try)
                    model.fit(train_pool, eval_set=val_pool, use_best_model=True)
            pred_val_full = model.predict_proba(X_all.loc[val_mask_full])[:, 1]
            best_iter = int((model.get_best_iteration() or params_cb_lbl["iterations"]) + (1 if model.get_best_iteration() is not None else 0))
            oof.loc[val_mask_full, "cb_lbl"] = pred_val_full
            cb_lbl_best_iters.append(best_iter)
            del model, train_pool, val_pool; gc.collect()
    else:
        oof["cb_lbl"] = oof["cb"]

    # ── Step 8: Blend + full refit + submission ──
    valid_mask = oof[["lgb", "cb", "cb_lbl", "lgb_susp"]].notna().all(axis=1)
    if int(valid_mask.sum()) == 0:
        raise RuntimeError("No valid OOF rows for blending.")

    lgb_head = rank_norm(oof.loc[valid_mask, "lgb"].values)
    cb_head = rank_norm(oof.loc[valid_mask, "cb"].values)
    consensus_head = np.sqrt(np.clip(lgb_head, 1e-6, 1.0) * np.clip(cb_head, 1e-6, 1.0))
    susp_head = rank_norm(oof.loc[valid_mask, "lgb_susp"].values)
    lbl_head = rank_norm(oof.loc[valid_mask, "cb_lbl"].values)
    susp_gate_head = np.sqrt(np.clip(consensus_head, 1e-6, 1.0) * np.clip(susp_head, 1e-6, 1.0))
    lbl_gate_head = np.sqrt(np.clip(consensus_head, 1e-6, 1.0) * np.clip(lbl_head, 1e-6, 1.0))
    dual_gate_head = np.sqrt(np.clip(consensus_head, 1e-6, 1.0) * np.clip(np.sqrt(np.clip(susp_head, 1e-6, 1.0) * np.clip(lbl_head, 1e-6, 1.0)), 1e-6, 1.0))

    heads = {
        "lgb": lgb_head, "cb": cb_head, "consensus": consensus_head,
        "susp_gate": susp_gate_head, "lbl_gate": lbl_gate_head, "dual_gate": dual_gate_head,
    }
    blend_keys, best_w, best_ap = optimize_blend_weights(heads, oof.loc[valid_mask, "y"].values)
    print("Best blend weights:", {blend_keys[i]: round(float(best_w[i]), 4) for i in range(len(blend_keys))})
    print("Best OOF AP:", round(best_ap, 6))

    refit_iter_lgb = int(max(300, round(np.median(lgb_best_iters) * REFIT_ITER_MULT))) if lgb_best_iters else params_lgb["n_estimators"]
    refit_iter_cb = int(max(300, round(np.median(cb_best_iters) * REFIT_ITER_MULT))) if cb_best_iters else params_cb["iterations"]
    refit_iter_lgb_susp = int(max(250, round(np.median(lgb_susp_best_iters) * REFIT_ITER_MULT))) if lgb_susp_best_iters else 600
    refit_iter_cb_lbl = int(max(250, round(np.median(cb_lbl_best_iters) * REFIT_ITER_MULT))) if cb_lbl_best_iters else 1200

    print("Refit iterations:", {"lgb": refit_iter_lgb, "cb": refit_iter_cb, "lgb_susp": refit_iter_lgb_susp, "cb_lbl": refit_iter_cb_lbl})

    # Full refit LGB
    params_lgb_full = dict(params_lgb)
    params_lgb_full["n_estimators"] = refit_iter_lgb
    lgb_full = lgb.LGBMClassifier(**params_lgb_full)
    try:
        lgb_full.fit(X_all_lgb, y_main_all, sample_weight=w_main_all, categorical_feature=cat_feature_cols)
    except Exception:
        params_cpu = {k: v for k, v in params_lgb_full.items() if k not in ("device_type", "gpu_platform_id", "gpu_device_id")}
        lgb_full = lgb.LGBMClassifier(**params_cpu)
        lgb_full.fit(X_all_lgb, y_main_all, sample_weight=w_main_all, categorical_feature=cat_feature_cols)

    # Full refit CB
    params_cb_full = {k: v for k, v in params_cb.items() if k not in ("od_type", "od_wait")}
    params_cb_full["iterations"] = refit_iter_cb
    cb_full = CatBoostClassifier(**params_cb_full)
    try:
        cb_full.fit(Pool(X_all, y_main_all, cat_features=cat_feature_cols, weight=w_main_all), verbose=200)
    except Exception:
        params_cpu = dict(params_cb_full)
        params_cpu["task_type"] = "CPU"
        params_cpu.pop("devices", None)
        cb_full = CatBoostClassifier(**params_cpu)
        cb_full.fit(Pool(X_all, y_main_all, cat_features=cat_feature_cols, weight=w_main_all), verbose=200)

    pred_lgb_test = lgb_full.predict_proba(X_test_lgb)[:, 1]
    pred_cb_test = cb_full.predict_proba(X_test)[:, 1]

    # Full refit LGB suspicious
    pred_lgb_susp_test = pred_lgb_test.copy()
    if ENABLE_LGB_SUSP_HEAD:
        params_lgb_susp_full = dict(params_lgb_susp)
        params_lgb_susp_full["n_estimators"] = refit_iter_lgb_susp
        lgb_susp_full = lgb.LGBMClassifier(**params_lgb_susp_full)
        try:
            lgb_susp_full.fit(X_all_lgb, y_susp_all, sample_weight=w_susp_all, categorical_feature=cat_feature_cols)
        except Exception:
            params_cpu = {k: v for k, v in params_lgb_susp_full.items() if k not in ("device_type", "gpu_platform_id", "gpu_device_id")}
            lgb_susp_full = lgb.LGBMClassifier(**params_cpu)
            lgb_susp_full.fit(X_all_lgb, y_susp_all, sample_weight=w_susp_all, categorical_feature=cat_feature_cols)
        pred_lgb_susp_test = lgb_susp_full.predict_proba(X_test_lgb)[:, 1]

    # Full refit CB label-head
    pred_cb_lbl_test = pred_cb_test.copy()
    if ENABLE_CB_LABEL_HEAD:
        labeled_mask_all = raw_target != -1
        params_cb_lbl_full = {k: v for k, v in params_cb_lbl.items() if k not in ("od_type", "od_wait")}
        params_cb_lbl_full["iterations"] = refit_iter_cb_lbl
        cb_lbl_full = CatBoostClassifier(**params_cb_lbl_full)
        try:
            cb_lbl_full.fit(
                Pool(X_all.loc[labeled_mask_all], y_main_all[labeled_mask_all], cat_features=cat_feature_cols, weight=w_labeled_all[labeled_mask_all]),
                verbose=200,
            )
        except Exception:
            params_cpu = dict(params_cb_lbl_full)
            params_cpu["task_type"] = "CPU"
            params_cpu.pop("devices", None)
            cb_lbl_full = CatBoostClassifier(**params_cpu)
            cb_lbl_full.fit(
                Pool(X_all.loc[labeled_mask_all], y_main_all[labeled_mask_all], cat_features=cat_feature_cols, weight=w_labeled_all[labeled_mask_all]),
                verbose=200,
            )
        pred_cb_lbl_test = cb_lbl_full.predict_proba(X_test)[:, 1]

    # Build test heads + blend
    test_lgb_head = rank_norm(pred_lgb_test)
    test_cb_head = rank_norm(pred_cb_test)
    test_consensus_head = np.sqrt(np.clip(test_lgb_head, 1e-6, 1.0) * np.clip(test_cb_head, 1e-6, 1.0))
    test_susp_head = rank_norm(pred_lgb_susp_test)
    test_lbl_head = rank_norm(pred_cb_lbl_test)
    test_susp_gate = np.sqrt(np.clip(test_consensus_head, 1e-6, 1.0) * np.clip(test_susp_head, 1e-6, 1.0))
    test_lbl_gate = np.sqrt(np.clip(test_consensus_head, 1e-6, 1.0) * np.clip(test_lbl_head, 1e-6, 1.0))
    test_dual_gate = np.sqrt(np.clip(test_consensus_head, 1e-6, 1.0) * np.clip(np.sqrt(np.clip(test_susp_head, 1e-6, 1.0) * np.clip(test_lbl_head, 1e-6, 1.0)), 1e-6, 1.0))

    test_heads = {
        "lgb": test_lgb_head, "cb": test_cb_head, "consensus": test_consensus_head,
        "susp_gate": test_susp_gate, "lbl_gate": test_lbl_gate, "dual_gate": test_dual_gate,
    }
    final_pred = sum(best_w[i] * test_heads[blend_keys[i]] for i in range(len(blend_keys)))

    # Save submission
    sample_submit = pd.read_csv(DATA_DIR / "sample_submit.csv")
    pred_df = pd.DataFrame({"event_id": test_df["event_id"].values, "predict": final_pred.astype(np.float64)})
    submission = sample_submit[["event_id"]].merge(pred_df, on="event_id", how="left")
    assert submission["predict"].isna().sum() == 0, "Missing test predictions"
    assert len(submission) == len(sample_submit), "Row count mismatch"

    sub_path = CACHE_DIR / SUBMISSION_FILENAME
    submission.to_csv(sub_path, index=False)
    print("Saved ->", sub_path)
    return sub_path


if __name__ == "__main__":
    run()
