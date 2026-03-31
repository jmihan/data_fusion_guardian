import gc
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import polars as pl

from src.config import DATA_DIR, CACHE_DIR, GLOBAL_CACHE_TAG
from src.features.engineering import labels_lf
from src.utils import dedupe


def _device_fp_expr_from_raw() -> pl.Expr:
    return (
        (
            pl.col("screen_size").str.extract(r"^(\d+)", 1).cast(pl.Int16, strict=False).fill_null(-1).cast(pl.Int64) * 100_000_000
            + pl.col("screen_size").str.extract(r"x(\d+)$", 1).cast(pl.Int16, strict=False).fill_null(-1).cast(pl.Int64) * 100_000
            + pl.col("operating_system_type").cast(pl.Int16, strict=False).fill_null(-1).cast(pl.Int64) * 1000
            + (pl.col("accept_language").cast(pl.Int32, strict=False).fill_null(-1).cast(pl.Int64) % 1000)
        ).alias("device_fp_i")
    )


PRIOR_COL_DEFS = {
    "event_desc":                 pl.col("event_desc").cast(pl.Int32, strict=False).fill_null(-1).alias("event_desc"),
    "mcc_code_i":                 pl.col("mcc_code").cast(pl.Int32, strict=False).fill_null(-1).alias("mcc_code_i"),
    "timezone":                   pl.col("timezone").cast(pl.Int32, strict=False).fill_null(-1).alias("timezone"),
    "operating_system_type":      pl.col("operating_system_type").cast(pl.Int16, strict=False).fill_null(-1).alias("operating_system_type"),
    "channel_indicator_sub_type": pl.col("channel_indicator_sub_type").cast(pl.Int16, strict=False).fill_null(-1).alias("channel_indicator_sub_type"),
    "event_type_nm":              pl.col("event_type_nm").cast(pl.Int32, strict=False).fill_null(-1).alias("event_type_nm"),
    "pos_cd":                     pl.col("pos_cd").cast(pl.Int16, strict=False).fill_null(-1).alias("pos_cd"),
    "channel_indicator_type":     pl.col("channel_indicator_type").cast(pl.Int16, strict=False).fill_null(-1).alias("channel_indicator_type"),
    "accept_language_i":          pl.col("accept_language").cast(pl.Int32, strict=False).fill_null(-1).alias("accept_language_i"),
    "browser_language_i":         pl.col("browser_language").cast(pl.Int32, strict=False).fill_null(-1).alias("browser_language_i"),
    "device_fp_i":                _device_fp_expr_from_raw(),
}

DEVICE_STAT_COLS = [
    "device_n_distinct_customers",
    "device_n_distinct_sessions",
    "device_n_total_ops",
]


def _scan_train_prior_source(expr: pl.Expr, key_name: str, cutoff: pd.Timestamp | None = None) -> pl.LazyFrame:
    frames = []
    for i in [1, 2, 3]:
        lf = pl.scan_parquet(DATA_DIR / f"train_part_{i}.parquet").select([
            pl.col("event_id"),
            pl.col("event_dttm").str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M:%S", strict=False).alias("event_ts"),
            expr,
        ])
        if cutoff is not None:
            lf = lf.filter(pl.col("event_ts") < pl.lit(cutoff.to_pydatetime()))
        frames.append(lf.select(["event_id", key_name]))
    return pl.concat(frames, how="vertical_relaxed")


def _make_prior_from_source(lf: pl.LazyFrame, key_name: str) -> pl.DataFrame:
    cnt_col = f"prior_{key_name}_cnt"
    lbl_cnt_col = f"prior_{key_name}_lbl_cnt"
    red_cnt_col = f"prior_{key_name}_red_cnt"

    total = lf.group_by(key_name).len().rename({"len": cnt_col})
    labeled = (
        lf.join(labels_lf, on="event_id", how="inner")
          .group_by(key_name)
          .agg([
              pl.len().alias(lbl_cnt_col),
              pl.sum("target").cast(pl.Float64).alias(red_cnt_col),
          ])
    )

    return (
        total.join(labeled, on=key_name, how="left")
             .with_columns([
                 pl.col(lbl_cnt_col).fill_null(0.0),
                 pl.col(red_cnt_col).fill_null(0.0),
             ])
             .with_columns([
                 ((pl.col(red_cnt_col) + 1.0) / (pl.col(cnt_col) + 200.0)).cast(pl.Float32).alias(f"prior_{key_name}_red_rate_all"),
                 ((pl.col(lbl_cnt_col) + 1.0) / (pl.col(cnt_col) + 200.0)).cast(pl.Float32).alias(f"prior_{key_name}_labeled_rate_all"),
                 ((pl.col(red_cnt_col) + 1.0) / (pl.col(lbl_cnt_col) + 2.0)).cast(pl.Float32).alias(f"prior_{key_name}_red_share_labeled"),
             ])
             .select([
                 key_name,
                 cnt_col,
                 f"prior_{key_name}_red_rate_all",
                 f"prior_{key_name}_labeled_rate_all",
                 f"prior_{key_name}_red_share_labeled",
             ])
             .collect()
    )


def build_prior_tables(tag: str, cutoff: pd.Timestamp | None = None, force: bool = False) -> Tuple[Dict[str, pd.DataFrame], List[str]]:
    tables: Dict[str, pd.DataFrame] = {}
    prior_feature_cols: List[str] = []

    for key_name, expr in PRIOR_COL_DEFS.items():
        out_path = CACHE_DIR / f"prior_{GLOBAL_CACHE_TAG}_{tag}_{key_name}.parquet"
        if out_path.exists() and not force:
            prior_df = pl.read_parquet(out_path)
        else:
            print(f"Building prior table: {tag} | {key_name}")
            source_lf = _scan_train_prior_source(expr, key_name, cutoff=cutoff)
            prior_df = _make_prior_from_source(source_lf, key_name)
            prior_df.write_parquet(out_path)

        tables[key_name] = prior_df.to_pandas()
        prior_feature_cols.extend([c for c in prior_df.columns if c != key_name])

    return tables, dedupe(prior_feature_cols)


def _scan_device_history(include_pretest: bool, cutoff: pd.Timestamp | None = None) -> pl.LazyFrame:
    base_cols = [
        "customer_id", "event_dttm", "session_id",
        "screen_size", "operating_system_type", "accept_language",
    ]
    frames = []
    for i in [1, 2, 3]:
        frames.append(pl.scan_parquet(DATA_DIR / f"pretrain_part_{i}.parquet").select(base_cols))
        frames.append(pl.scan_parquet(DATA_DIR / f"train_part_{i}.parquet").select(base_cols))

    if include_pretest:
        frames.append(pl.scan_parquet(DATA_DIR / "pretest.parquet").select(base_cols))

    lf = pl.concat(frames, how="diagonal_relaxed").with_columns([
        pl.col("event_dttm").str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M:%S", strict=False).alias("event_ts"),
        pl.col("session_id").cast(pl.Int64, strict=False).fill_null(-1).alias("session_id"),
        _device_fp_expr_from_raw(),
    ])

    if cutoff is not None:
        lf = lf.filter(pl.col("event_ts") < pl.lit(cutoff.to_pydatetime()))

    return lf.select(["device_fp_i", "customer_id", "session_id"])


def build_device_stats_table(tag: str, cutoff: pd.Timestamp | None = None, include_pretest: bool = False, force: bool = False) -> pd.DataFrame:
    out_path = CACHE_DIR / f"device_stats_{GLOBAL_CACHE_TAG}_{tag}.parquet"
    if out_path.exists() and not force:
        return pl.read_parquet(out_path).to_pandas()

    print(f"Building device stats: {tag}")
    lf = _scan_device_history(include_pretest=include_pretest, cutoff=cutoff)
    stats_df = (
        lf.group_by("device_fp_i")
          .agg([
              pl.n_unique("customer_id").cast(pl.Int32).alias("device_n_distinct_customers"),
              pl.n_unique("session_id").cast(pl.Int32).alias("device_n_distinct_sessions"),
              pl.len().cast(pl.Int32).alias("device_n_total_ops"),
          ])
          .with_columns([
              pl.col("device_n_distinct_customers").log1p().cast(pl.Float32).alias("device_n_distinct_customers"),
              pl.col("device_n_distinct_sessions").log1p().cast(pl.Float32).alias("device_n_distinct_sessions"),
              pl.col("device_n_total_ops").log1p().cast(pl.Float32).alias("device_n_total_ops"),
          ])
          .collect()
    )
    stats_df.write_parquet(out_path)
    return stats_df.to_pandas()


def attach_global_features(base_df: pd.DataFrame, device_stats_df: pd.DataFrame, prior_tables: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    out = base_df.merge(device_stats_df, on="device_fp_i", how="left")
    for key_name, prior_df in prior_tables.items():
        out = out.merge(prior_df, on=key_name, how="left")
    return out
