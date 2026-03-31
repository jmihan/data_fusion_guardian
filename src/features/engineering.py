import gc
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

from src.config import (
    DATA_DIR, CACHE_DIR, FEATURE_CACHE_TAG, FEATURE_CHUNK_SIZE,
    NEG_SAMPLE_BORDER_STR, NEG_SAMPLE_MOD_RECENT, NEG_SAMPLE_MOD_OLD,
    RANDOM_SEED,
)
from src.features.columns import BASE_COLS, FINAL_FEATURE_COLS, META_COLS


# ── Labels (loaded once) ────────────────────────────────────────────────────
labels_lf = pl.scan_parquet(DATA_DIR / "train_labels.parquet")


def _load_periods_for_chunk(part_id: int, cust_series: pl.Series) -> pl.LazyFrame:
    pretrain_lf = (
        pl.scan_parquet(DATA_DIR / f"pretrain_part_{part_id}.parquet")
        .filter(pl.col("customer_id").is_in(cust_series))
        .select(BASE_COLS)
        .with_columns(pl.lit("pretrain").alias("period"))
    )
    train_lf = (
        pl.scan_parquet(DATA_DIR / f"train_part_{part_id}.parquet")
        .filter(pl.col("customer_id").is_in(cust_series))
        .select(BASE_COLS)
        .with_columns(pl.lit("train").alias("period"))
    )
    pretest_lf = (
        pl.scan_parquet(DATA_DIR / "pretest.parquet")
        .filter(pl.col("customer_id").is_in(cust_series))
        .select(BASE_COLS)
        .unique()  # FIX: 130,360 duplicate rows in pretest
        .with_columns(pl.lit("pretest").alias("period"))
    )
    test_lf = (
        pl.scan_parquet(DATA_DIR / "test.parquet")
        .filter(pl.col("customer_id").is_in(cust_series))
        .select(BASE_COLS)
        .with_columns(pl.lit("test").alias("period"))
    )
    return pl.concat([pretrain_lf, train_lf, pretest_lf, test_lf], how="vertical_relaxed")


def _build_features_for_chunk(lf: pl.LazyFrame) -> pl.DataFrame:
    lf = (
        lf.with_columns([
            pl.col("event_dttm").str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M:%S", strict=False).alias("event_ts"),
            pl.col("operaton_amt").cast(pl.Float64).alias("amt"),
            pl.col("session_id").cast(pl.Int64, strict=False).fill_null(-1).alias("session_id"),
            pl.col("event_type_nm").cast(pl.Int32, strict=False).fill_null(-1).alias("event_type_nm"),
            pl.col("event_desc").cast(pl.Int32, strict=False).fill_null(-1).alias("event_desc"),
            pl.col("channel_indicator_type").cast(pl.Int16, strict=False).fill_null(-1).alias("channel_indicator_type"),
            pl.col("channel_indicator_sub_type").cast(pl.Int16, strict=False).fill_null(-1).alias("channel_indicator_sub_type"),
            pl.col("currency_iso_cd").cast(pl.Int16, strict=False).fill_null(-1).alias("currency_iso_cd"),
            pl.col("pos_cd").cast(pl.Int16, strict=False).fill_null(-1).alias("pos_cd"),
            pl.col("timezone").cast(pl.Int32, strict=False).fill_null(-1).alias("timezone"),
            pl.col("operating_system_type").cast(pl.Int16, strict=False).fill_null(-1).alias("operating_system_type"),
            pl.col("phone_voip_call_state").cast(pl.Int8, strict=False).fill_null(-1).alias("phone_voip_call_state"),
            pl.col("web_rdp_connection").cast(pl.Int8, strict=False).fill_null(-1).alias("web_rdp_connection"),
            pl.col("mcc_code").cast(pl.Int32, strict=False).fill_null(-1).alias("mcc_code_i"),
            pl.col("battery").str.extract(r"(\d{1,3})", 1).cast(pl.Int16, strict=False).fill_null(-1).alias("battery_pct"),
            pl.col("device_system_version").str.extract(r"^(\d+)", 1).cast(pl.Int16, strict=False).fill_null(-1).alias("os_ver_major"),
            pl.col("screen_size").str.extract(r"^(\d+)", 1).cast(pl.Int16, strict=False).fill_null(-1).alias("screen_w"),
            pl.col("screen_size").str.extract(r"x(\d+)$", 1).cast(pl.Int16, strict=False).fill_null(-1).alias("screen_h"),
            pl.col("developer_tools").cast(pl.Int8, strict=False).fill_null(-1).alias("developer_tools_i"),
            pl.col("compromised").cast(pl.Int8, strict=False).fill_null(-1).alias("compromised_i"),
            pl.col("accept_language").cast(pl.Int32, strict=False).fill_null(-1).alias("accept_language_i"),
            pl.col("browser_language").cast(pl.Int32, strict=False).fill_null(-1).alias("browser_language_i"),

            pl.col("operaton_amt").is_null().cast(pl.Int8).alias("amt_missing"),
            pl.col("currency_iso_cd").is_null().cast(pl.Int8).alias("currency_missing"),
            pl.col("mcc_code").is_null().cast(pl.Int8).alias("mcc_missing"),
            pl.col("pos_cd").is_null().cast(pl.Int8).alias("pos_missing"),
            pl.col("accept_language").is_null().cast(pl.Int8).alias("accept_language_missing"),
            pl.col("browser_language").is_null().cast(pl.Int8).alias("browser_language_missing"),
            pl.col("timezone").is_null().cast(pl.Int8).alias("timezone_missing"),
            pl.col("session_id").is_null().cast(pl.Int8).alias("session_id_missing"),
            pl.col("operating_system_type").is_null().cast(pl.Int8).alias("operating_system_missing"),
            pl.col("battery").is_null().cast(pl.Int8).alias("battery_missing"),
            pl.col("device_system_version").is_null().cast(pl.Int8).alias("device_system_version_missing"),
            pl.col("screen_size").is_null().cast(pl.Int8).alias("screen_size_missing"),
            pl.col("developer_tools").is_null().cast(pl.Int8).alias("developer_tools_missing"),
            pl.col("phone_voip_call_state").is_null().cast(pl.Int8).alias("phone_voip_missing"),
            pl.col("web_rdp_connection").is_null().cast(pl.Int8).alias("web_rdp_missing"),
            pl.col("compromised").is_null().cast(pl.Int8).alias("compromised_missing"),
        ])
        .drop([
            "event_dttm", "operaton_amt", "mcc_code", "battery", "device_system_version",
            "screen_size", "developer_tools", "compromised", "accept_language", "browser_language",
        ])
        .sort(["customer_id", "event_ts", "event_id"])
    )

    lf = lf.with_columns([
        (
            pl.col("screen_w").cast(pl.Int64) * 100_000_000
            + pl.col("screen_h").cast(pl.Int64) * 100_000
            + pl.col("operating_system_type").cast(pl.Int64) * 1000
            + pl.col("accept_language_i").cast(pl.Int64) % 1000
        ).alias("device_fp_i"),
    ])

    lf = lf.join(labels_lf, on="event_id", how="left")
    lf = lf.with_columns([
        pl.when(pl.col("period") == "train")
          .then(pl.when(pl.col("target").is_null()).then(pl.lit(-1)).otherwise(pl.col("target")))
          .otherwise(pl.lit(None))
          .alias("train_target_raw")
    ])

    border_expr = pl.lit(NEG_SAMPLE_BORDER_STR).str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M:%S", strict=False)
    lf = lf.with_columns([
        ((pl.col("period") == "train") &
         (pl.col("train_target_raw") == -1) &
         (((pl.col("event_ts") >= border_expr) & ((pl.struct(["event_id", "customer_id"]).hash(seed=RANDOM_SEED) % NEG_SAMPLE_MOD_RECENT) == 0)) |
          ((pl.col("event_ts") < border_expr)  & ((pl.struct(["event_id", "customer_id"]).hash(seed=RANDOM_SEED + 17) % NEG_SAMPLE_MOD_OLD) == 0))))
          .alias("keep_green")
    ])
    lf = lf.with_columns([
        ((pl.col("period") == "train") & ((pl.col("train_target_raw") != -1) | pl.col("keep_green"))).alias("is_train_sample"),
        (pl.col("period") == "test").alias("is_test"),
        pl.col("event_ts").dt.hour().cast(pl.Int8).alias("hour"),
        pl.col("event_ts").dt.weekday().cast(pl.Int8).alias("weekday"),
        pl.col("event_ts").dt.day().cast(pl.Int8).alias("day"),
        pl.col("event_ts").dt.month().cast(pl.Int8).alias("month"),
        (pl.col("event_ts").dt.weekday() >= 5).cast(pl.Int8).alias("is_weekend"),
        (pl.col("event_ts").dt.hour().is_in([0, 1, 2, 3, 4, 5])).cast(pl.Int8).alias("is_night"),
        (pl.col("event_ts").dt.epoch("s") // 86400).cast(pl.Int32).alias("event_day_number"),
        pl.col("event_ts").dt.date().alias("event_date"),
        pl.col("event_ts").dt.truncate("1h").alias("event_hour_trunc"),
        pl.col("amt").abs().cast(pl.Float32).alias("amt_abs"),
        pl.col("amt").abs().log1p().cast(pl.Float32).alias("amt_log_abs"),
        (pl.col("amt").abs().log1p() * 4.0).floor().clip(0, 63).cast(pl.Int16).alias("amt_bucket"),
        (pl.col("amt") < 0).cast(pl.Int8).alias("amt_is_negative"),
        (pl.col("screen_w").cast(pl.Int32) * pl.col("screen_h").cast(pl.Int32)).alias("screen_pixels"),
        pl.when((pl.col("screen_h") > 0) & (pl.col("screen_w") > 0))
          .then(pl.col("screen_w").cast(pl.Float32) / pl.col("screen_h").cast(pl.Float32))
          .otherwise(0.0).alias("screen_ratio"),
        (((pl.col("event_ts").dt.hour().cast(pl.Float32) / 24.0) * (2.0 * np.pi)).sin()).cast(pl.Float32).alias("hour_sin"),
        (((pl.col("event_ts").dt.hour().cast(pl.Float32) / 24.0) * (2.0 * np.pi)).cos()).cast(pl.Float32).alias("hour_cos"),
        (((pl.col("event_ts").dt.weekday().cast(pl.Float32) / 7.0) * (2.0 * np.pi)).sin()).cast(pl.Float32).alias("weekday_sin"),
        (((pl.col("event_ts").dt.weekday().cast(pl.Float32) / 7.0) * (2.0 * np.pi)).cos()).cast(pl.Float32).alias("weekday_cos"),
        (((pl.col("event_ts").dt.month().cast(pl.Float32) / 12.0) * (2.0 * np.pi)).sin()).cast(pl.Float32).alias("month_sin"),
        (((pl.col("event_ts").dt.month().cast(pl.Float32) / 12.0) * (2.0 * np.pi)).cos()).cast(pl.Float32).alias("month_cos"),
    ])

    lf = lf.with_columns([
        pl.col("event_ts").min().over("customer_id").alias("_cust_first_ts"),
    ])
    lf = lf.with_columns([
        ((pl.col("event_ts") - pl.col("_cust_first_ts")).dt.total_seconds() // 86400)
          .cast(pl.Int32).alias("days_since_first_event"),
    ])
    lf = lf.drop("_cust_first_ts")

    lf = lf.with_columns([
        pl.col("event_ts").min().over(["customer_id", "session_id"]).alias("_session_first_ts"),
        pl.col("amt").cum_sum().over(["customer_id", "session_id"]).alias("_session_cum_amt"),
    ])
    lf = lf.with_columns([
        ((pl.col("event_ts") - pl.col("_session_first_ts")).dt.total_seconds())
          .cast(pl.Int32).alias("sec_since_session_start"),
        (pl.col("_session_cum_amt") - pl.col("amt")).cast(pl.Float32).alias("session_amt_before"),
    ])
    lf = lf.drop(["_session_first_ts", "_session_cum_amt"])

    lf = lf.with_columns([
        ((pl.col("period") == "train") & (pl.col("train_target_raw") == 1)).cast(pl.Int8).alias("is_red_lbl"),
        ((pl.col("period") == "train") & (pl.col("train_target_raw") == 0)).cast(pl.Int8).alias("is_yellow_lbl"),
    ])
    lf = lf.with_columns([
        (pl.col("is_red_lbl") + pl.col("is_yellow_lbl")).cast(pl.Int8).alias("is_labeled_fb")
    ])

    lf = lf.with_columns([
        pl.cum_count("event_id").over("customer_id").cast(pl.Int32).alias("cust_event_idx"),
        pl.col("amt").cum_sum().over("customer_id").alias("cust_cum_amt"),
        (pl.col("amt") * pl.col("amt")).cum_sum().over("customer_id").alias("cust_cum_amt_sq"),
        pl.col("amt").cum_max().over("customer_id").alias("cust_cum_max_amt"),
        pl.col("event_ts").shift(1).over("customer_id").alias("prev_event_ts"),
        pl.col("event_date").shift(1).over("customer_id").alias("prev_event_date"),
        pl.col("amt").shift(1).over("customer_id").alias("prev_amt"),
        (pl.cum_count("event_id").over(["customer_id", "event_type_nm"]) - 1).cast(pl.Int16).alias("cnt_prev_same_type"),
        (pl.cum_count("event_id").over(["customer_id", "event_desc"]) - 1).cast(pl.Int16).alias("cnt_prev_same_desc"),
        (pl.cum_count("event_id").over(["customer_id", "mcc_code_i"]) - 1).cast(pl.Int16).alias("cnt_prev_same_mcc"),
        (pl.cum_count("event_id").over(["customer_id", "channel_indicator_sub_type"]) - 1).cast(pl.Int16).alias("cnt_prev_same_subtype"),
        (pl.cum_count("event_id").over(["customer_id", "channel_indicator_type"]) - 1).cast(pl.Int16).alias("cnt_prev_same_channel_type"),
        (pl.cum_count("event_id").over(["customer_id", "currency_iso_cd"]) - 1).cast(pl.Int16).alias("cnt_prev_same_currency"),
        (pl.cum_count("event_id").over(["customer_id", "device_fp_i"]) - 1).cast(pl.Int16).alias("cnt_prev_same_device"),
        (pl.cum_count("event_id").over(["customer_id", "session_id"]) - 1).cast(pl.Int16).alias("cnt_prev_same_session"),
        pl.col("event_ts").shift(1).over(["customer_id", "event_type_nm"]).alias("prev_same_type_ts"),
        pl.col("event_ts").shift(1).over(["customer_id", "event_desc"]).alias("prev_same_desc_ts"),
        pl.col("event_ts").shift(1).over(["customer_id", "channel_indicator_type"]).alias("prev_same_channel_type_ts"),
        pl.col("event_ts").shift(1).over(["customer_id", "currency_iso_cd"]).alias("prev_same_currency_ts"),
        pl.col("event_ts").shift(1).over(["customer_id", "device_fp_i"]).alias("prev_same_device_ts"),
        pl.col("is_red_lbl").cum_sum().over("customer_id").cast(pl.Int32).alias("cust_red_lbl_cum"),
        pl.col("is_yellow_lbl").cum_sum().over("customer_id").cast(pl.Int32).alias("cust_yellow_lbl_cum"),
        pl.col("is_labeled_fb").cum_sum().over("customer_id").cast(pl.Int32).alias("cust_labeled_fb_cum"),
        pl.col("is_red_lbl").cum_sum().over(["customer_id", "event_desc"]).cast(pl.Int16).alias("desc_red_lbl_cum"),
        pl.col("is_yellow_lbl").cum_sum().over(["customer_id", "event_desc"]).cast(pl.Int16).alias("desc_yellow_lbl_cum"),
        pl.col("is_labeled_fb").cum_sum().over(["customer_id", "event_desc"]).cast(pl.Int16).alias("desc_labeled_fb_cum"),
        pl.col("is_red_lbl").cum_sum().over(["customer_id", "channel_indicator_type"]).cast(pl.Int16).alias("chan_red_lbl_cum"),
        pl.col("is_labeled_fb").cum_sum().over(["customer_id", "channel_indicator_type"]).cast(pl.Int16).alias("chan_labeled_fb_cum"),
        pl.col("is_red_lbl").cum_sum().over(["customer_id", "event_type_nm"]).cast(pl.Int16).alias("type_red_lbl_cum"),
        pl.col("is_labeled_fb").cum_sum().over(["customer_id", "event_type_nm"]).cast(pl.Int16).alias("type_labeled_fb_cum"),
        pl.when(pl.col("is_red_lbl") == 1).then(pl.col("event_ts")).otherwise(None).alias("red_lbl_ts"),
        pl.when(pl.col("is_yellow_lbl") == 1).then(pl.col("event_ts")).otherwise(None).alias("yellow_lbl_ts"),
    ])

    lf = lf.with_columns([
        (pl.cum_count("event_id").over(["customer_id", "event_hour_trunc"]) - 1).cast(pl.Int16).alias("cnt_events_this_hour"),
    ])

    lf = lf.with_columns([
        pl.col("red_lbl_ts").shift(1).over("customer_id").alias("_red_shifted"),
        pl.col("yellow_lbl_ts").shift(1).over("customer_id").alias("_yellow_shifted"),
    ])
    lf = lf.with_columns([
        pl.col("_red_shifted").forward_fill().over("customer_id").alias("prev_red_lbl_ts"),
        pl.col("_yellow_shifted").forward_fill().over("customer_id").alias("prev_yellow_lbl_ts"),
    ])
    lf = lf.drop(["_red_shifted", "_yellow_shifted"])

    lf = lf.with_columns([
        (pl.col("cust_event_idx") - 1).cast(pl.Int32).alias("cust_prev_events"),
        pl.when(pl.col("cust_event_idx") > 1)
          .then((pl.col("cust_cum_amt") - pl.col("amt")) / (pl.col("cust_event_idx") - 1))
          .otherwise(0.0).cast(pl.Float32).alias("cust_prev_amt_mean"),
        pl.col("cust_cum_max_amt").shift(1).over("customer_id").fill_null(0.0).cast(pl.Float32).alias("cust_prev_max_amt"),
        pl.when(pl.col("prev_event_ts").is_not_null())
          .then((pl.col("event_ts") - pl.col("prev_event_ts")).dt.total_seconds())
          .otherwise(-1).cast(pl.Int32).alias("sec_since_prev_event"),
        (pl.col("amt") - pl.col("prev_amt").fill_null(0.0)).cast(pl.Float32).alias("amt_delta_prev"),
        pl.when(pl.col("prev_same_type_ts").is_not_null())
          .then((pl.col("event_ts") - pl.col("prev_same_type_ts")).dt.total_seconds())
          .otherwise(-1).cast(pl.Int32).alias("sec_since_prev_same_type"),
        pl.when(pl.col("prev_same_desc_ts").is_not_null())
          .then((pl.col("event_ts") - pl.col("prev_same_desc_ts")).dt.total_seconds())
          .otherwise(-1).cast(pl.Int32).alias("sec_since_prev_same_desc"),
        pl.when(pl.col("prev_same_channel_type_ts").is_not_null())
          .then((pl.col("event_ts") - pl.col("prev_same_channel_type_ts")).dt.total_seconds())
          .otherwise(-1).cast(pl.Int32).alias("sec_since_prev_same_channel_type"),
        pl.when(pl.col("prev_same_currency_ts").is_not_null())
          .then((pl.col("event_ts") - pl.col("prev_same_currency_ts")).dt.total_seconds())
          .otherwise(-1).cast(pl.Int32).alias("sec_since_prev_same_currency"),
        pl.when(pl.col("prev_same_device_ts").is_not_null())
          .then((pl.col("event_ts") - pl.col("prev_same_device_ts")).dt.total_seconds())
          .otherwise(-1).cast(pl.Int32).alias("sec_since_prev_same_device"),
        (pl.cum_count("event_id").over(["customer_id", "event_date"]) - 1).cast(pl.Int16).alias("events_before_today"),
        (pl.col("cust_red_lbl_cum") - pl.col("is_red_lbl")).cast(pl.Int32).alias("cust_prev_red_lbl_cnt"),
        (pl.col("cust_yellow_lbl_cum") - pl.col("is_yellow_lbl")).cast(pl.Int32).alias("cust_prev_yellow_lbl_cnt"),
        (pl.col("cust_labeled_fb_cum") - pl.col("is_labeled_fb")).cast(pl.Int32).alias("cust_prev_labeled_cnt"),
        (pl.col("desc_labeled_fb_cum") - pl.col("is_labeled_fb")).cast(pl.Int16).alias("cnt_prev_labeled_same_desc"),
        (pl.col("desc_red_lbl_cum") - pl.col("is_red_lbl")).cast(pl.Int16).alias("cnt_prev_red_same_desc_lbl"),
        (pl.col("desc_yellow_lbl_cum") - pl.col("is_yellow_lbl")).cast(pl.Int16).alias("cnt_prev_yellow_same_desc_lbl"),
        (pl.col("chan_red_lbl_cum") - pl.col("is_red_lbl")).cast(pl.Int16).alias("cnt_prev_red_same_channel"),
        (pl.col("chan_labeled_fb_cum") - pl.col("is_labeled_fb")).cast(pl.Int16).alias("cnt_prev_labeled_same_channel"),
        (pl.col("type_red_lbl_cum") - pl.col("is_red_lbl")).cast(pl.Int16).alias("cnt_prev_red_same_type_lbl"),
        (pl.col("type_labeled_fb_cum") - pl.col("is_labeled_fb")).cast(pl.Int16).alias("cnt_prev_labeled_same_type_lbl"),
        pl.when(pl.col("prev_red_lbl_ts").is_not_null())
          .then((pl.col("event_ts") - pl.col("prev_red_lbl_ts")).dt.total_seconds())
          .otherwise(-1).cast(pl.Int32).alias("sec_since_prev_red_lbl"),
        pl.when(pl.col("prev_yellow_lbl_ts").is_not_null())
          .then((pl.col("event_ts") - pl.col("prev_yellow_lbl_ts")).dt.total_seconds())
          .otherwise(-1).cast(pl.Int32).alias("sec_since_prev_yellow_lbl"),
    ])

    lf = lf.with_columns([
        pl.when(pl.col("cust_prev_events") > 0)
          .then((pl.col("prev_event_date").is_null() | (pl.col("event_date") != pl.col("prev_event_date"))).cast(pl.Int8))
          .otherwise(1).cast(pl.Int8).alias("_is_new_customer_day"),
        pl.when(pl.col("cust_event_idx") > 2)
          .then(
              ((pl.col("cust_cum_amt_sq") - pl.col("amt") * pl.col("amt")) / (pl.col("cust_event_idx") - 1)
               - pl.col("cust_prev_amt_mean") * pl.col("cust_prev_amt_mean"))
              .clip(lower_bound=0).sqrt())
          .otherwise(0.0).cast(pl.Float32).alias("cust_prev_amt_std"),
        ((pl.col("cust_prev_red_lbl_cnt") + 0.1) / (pl.col("cust_prev_labeled_cnt") + 1.0)).cast(pl.Float32).alias("cust_prev_red_lbl_rate"),
        ((pl.col("cust_prev_yellow_lbl_cnt") + 0.1) / (pl.col("cust_prev_labeled_cnt") + 1.0)).cast(pl.Float32).alias("cust_prev_yellow_lbl_rate"),
        (((pl.col("cust_prev_red_lbl_cnt") + pl.col("cust_prev_yellow_lbl_cnt")) + 0.1) / (pl.col("cust_prev_events") + 1.0)).cast(pl.Float32).alias("cust_prev_susp_lbl_rate"),
        (pl.col("cust_prev_red_lbl_cnt") > 0).cast(pl.Int8).alias("cust_prev_any_red_flag"),
        (pl.col("cust_prev_yellow_lbl_cnt") > 0).cast(pl.Int8).alias("cust_prev_any_yellow_flag"),
        ((pl.col("cnt_prev_red_same_desc_lbl") + 0.1) / (pl.col("cnt_prev_labeled_same_desc") + 1.0)).cast(pl.Float32).alias("red_rate_prev_same_desc_lbl"),
        ((pl.col("cnt_prev_red_same_channel") + 0.1) / (pl.col("cnt_prev_labeled_same_channel") + 1.0)).cast(pl.Float32).alias("red_rate_prev_same_channel"),
        ((pl.col("cnt_prev_red_same_type_lbl") + 0.1) / (pl.col("cnt_prev_labeled_same_type_lbl") + 1.0)).cast(pl.Float32).alias("red_rate_prev_same_type"),
    ])

    lf = lf.with_columns([
        (pl.col("_is_new_customer_day").cum_sum().over("customer_id") - 1).cast(pl.Int16).alias("cust_active_days"),
        pl.when(pl.col("cust_prev_amt_std") > 0)
          .then((pl.col("amt") - pl.col("cust_prev_amt_mean")) / (pl.col("cust_prev_amt_std") + 1.0))
          .otherwise(0.0).cast(pl.Float32).alias("amt_zscore"),
        pl.when(pl.col("cust_prev_max_amt").abs() > 1.0)
          .then(pl.col("amt") / (pl.col("cust_prev_max_amt").abs() + 1.0))
          .otherwise(0.0).cast(pl.Float32).alias("amt_vs_personal_max"),
        pl.when(pl.col("sec_since_prev_event") >= 0)
          .then(pl.col("sec_since_prev_event").cast(pl.Float32) / 86400.0)
          .otherwise(-1.0).cast(pl.Float32).alias("days_since_prev_event"),
        (pl.col("cnt_prev_same_type") == 0).cast(pl.Int8).alias("is_new_event_type"),
        (pl.col("cnt_prev_same_desc") == 0).cast(pl.Int8).alias("is_new_event_desc"),
        (pl.col("cnt_prev_same_subtype") == 0).cast(pl.Int8).alias("is_new_channel_sub"),
        (pl.col("cnt_prev_same_channel_type") == 0).cast(pl.Int8).alias("is_new_channel_type"),
        (pl.col("cnt_prev_same_mcc") == 0).cast(pl.Int8).alias("is_new_mcc"),
        (pl.col("cnt_prev_same_currency") == 0).cast(pl.Int8).alias("is_new_currency"),
        (pl.col("cnt_prev_same_device") == 0).cast(pl.Int8).alias("is_new_device_fp"),
        (pl.col("cnt_prev_same_session") == 0).cast(pl.Int8).alias("is_first_in_session"),
    ])

    lf = lf.with_columns([
        (pl.col("cust_prev_events") / (pl.col("cust_active_days") + 1.0)).cast(pl.Float32).alias("avg_prev_events_per_active_day"),
    ])

    lf = lf.with_columns([
        pl.when(pl.col("is_train_sample")).then((pl.col("train_target_raw") == 1).cast(pl.Int8)).otherwise(pl.lit(None)).alias("target_bin")
    ])

    # Rolling features (pandas-based for time-window rolling)
    _tmp = lf.select(["event_id", "customer_id", "event_ts", "amt"]).collect().to_pandas()
    _tmp["event_ts"] = pd.to_datetime(_tmp["event_ts"])
    _tmp = _tmp.sort_values(["customer_id", "event_ts", "event_id"]).reset_index(drop=True).set_index("event_ts")

    _results = []
    for _cust, _grp in _tmp.groupby("customer_id", sort=False):
        _s = _grp["amt"]
        _r = pd.DataFrame(index=_grp.index)
        _r["amt_sum_last_1h"]  = _s.rolling("1h",  closed="left").sum()
        _r["cnt_last_1h"]      = _s.rolling("1h",  closed="left").count()
        _r["amt_sum_last_24h"] = _s.rolling("24h", closed="left").sum()
        _r["cnt_last_24h"]     = _s.rolling("24h", closed="left").count()
        _r["max_amt_last_24h"] = _s.rolling("24h", closed="left").max()
        _r["event_id"] = _grp["event_id"].values
        _results.append(_r)

    _roll_df = pd.concat(_results).reset_index(drop=True).fillna(0)
    for c in ["amt_sum_last_1h", "amt_sum_last_24h", "max_amt_last_24h"]:
        _roll_df[c] = _roll_df[c].astype("float32")
    for c in ["cnt_last_1h", "cnt_last_24h"]:
        _roll_df[c] = _roll_df[c].astype("int16")
    _rolling_cols = ["amt_sum_last_1h", "cnt_last_1h", "amt_sum_last_24h", "cnt_last_24h", "max_amt_last_24h"]
    _rolling_lf = pl.from_pandas(_roll_df[["event_id"] + _rolling_cols].drop_duplicates(subset=["event_id"])).lazy()
    lf = lf.join(_rolling_lf, on="event_id", how="left")
    del _tmp, _roll_df, _results, _rolling_lf; gc.collect()

    lf = lf.with_columns([
        pl.when(pl.col("amt_sum_last_1h").abs() > 1.0)
          .then(pl.col("amt") / (pl.col("amt_sum_last_1h").abs() + 1.0))
          .otherwise(0.0).cast(pl.Float32).alias("amt_vs_1h_sum"),
        pl.when(pl.col("amt_sum_last_24h").abs() > 1.0)
          .then(pl.col("amt") / (pl.col("amt_sum_last_24h").abs() + 1.0))
          .otherwise(0.0).cast(pl.Float32).alias("amt_vs_24h_sum"),
    ])

    lf = lf.drop([
        "prev_event_date",
        "prev_same_type_ts", "prev_same_desc_ts",
        "prev_same_channel_type_ts", "prev_same_currency_ts", "prev_same_device_ts",
        "_is_new_customer_day",
    ])

    select_cols = META_COLS + FINAL_FEATURE_COLS
    seen = set()
    select_cols = [c for c in select_cols if not (c in seen or seen.add(c))]
    present = [c for c in select_cols if c in lf.columns]

    out_df = lf.filter(pl.col("is_train_sample") | pl.col("is_test")).select(present).collect()
    return out_df


def build_features_for_part(part_id: int, force: bool = False) -> Path:
    out_path = CACHE_DIR / f"features_{FEATURE_CACHE_TAG}_part_{part_id}.parquet"
    if out_path.exists() and not force:
        print(f"[part {part_id}] use cache -> {out_path.name}")
        return out_path

    print(f"[part {part_id}] building features (chunked, {FEATURE_CHUNK_SIZE} custs/chunk)...")
    custs = (
        pl.scan_parquet(DATA_DIR / f"pretrain_part_{part_id}.parquet")
        .select("customer_id").unique().collect()
        .get_column("customer_id").to_list()
    )
    print(f"  [part {part_id}] {len(custs):,} customers total")

    chunks = [custs[i:i+FEATURE_CHUNK_SIZE] for i in range(0, len(custs), FEATURE_CHUNK_SIZE)]
    results = []

    for ci, chunk_custs in enumerate(chunks):
        print(f"  [part {part_id}] chunk {ci+1}/{len(chunks)} ({len(chunk_custs)} custs) ...", end=" ", flush=True)
        cust_series = pl.Series("customer_id", chunk_custs)
        lf = _load_periods_for_chunk(part_id, cust_series)
        chunk_df = _build_features_for_chunk(lf)
        print(f"-> {chunk_df.height:,} rows")
        results.append(chunk_df)
        del chunk_df, lf
        gc.collect()

    out_df = pl.concat(results)
    out_df.write_parquet(out_path, compression="zstd")
    n_tr = out_df.filter(pl.col("is_train_sample")).height
    n_te = out_df.filter(pl.col("is_test")).height
    print(f"  [part {part_id}] saved: {out_df.shape} (train={n_tr:,}, test={n_te:,})")
    del out_df, results
    gc.collect()
    return out_path
