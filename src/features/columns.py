BASE_COLS = [
    "customer_id", "event_id", "event_dttm", "event_type_nm", "event_desc",
    "channel_indicator_type", "channel_indicator_sub_type", "operaton_amt", "currency_iso_cd",
    "mcc_code", "pos_cd",
    "accept_language", "browser_language",
    "timezone", "session_id", "operating_system_type",
    "battery", "device_system_version", "screen_size", "developer_tools",
    "phone_voip_call_state", "web_rdp_connection", "compromised",
]

FINAL_FEATURE_COLS = [
    # категориальные идентификаторы
    "customer_id", "event_type_nm", "event_desc", "channel_indicator_type",
    "channel_indicator_sub_type", "currency_iso_cd", "mcc_code_i", "pos_cd", "timezone",
    "operating_system_type", "phone_voip_call_state", "web_rdp_connection",
    "developer_tools_i", "compromised_i",
    "accept_language_i", "browser_language_i",
    "device_fp_i",

    # amount + missingness
    "amt", "amt_abs", "amt_log_abs", "amt_is_negative", "amt_bucket",
    "amt_missing", "currency_missing", "mcc_missing", "pos_missing",
    "accept_language_missing", "browser_language_missing", "timezone_missing",
    "session_id_missing", "operating_system_missing", "battery_missing",
    "device_system_version_missing", "screen_size_missing", "developer_tools_missing",
    "phone_voip_missing", "web_rdp_missing", "compromised_missing",

    # временные
    "hour", "weekday", "day", "is_weekend", "is_night",
    "hour_sin", "hour_cos", "weekday_sin", "weekday_cos", "month_sin", "month_cos",

    # устройство
    "battery_pct", "os_ver_major", "screen_w", "screen_h", "screen_pixels", "screen_ratio",

    # возраст аккаунта и активность
    "days_since_first_event", "cust_active_days", "avg_prev_events_per_active_day",

    # клиентская история объёмов
    "cust_prev_events", "cust_prev_amt_mean", "cust_prev_amt_std", "cust_prev_max_amt",
    "amt_vs_personal_max",
    "sec_since_prev_event", "days_since_prev_event",
    "amt_delta_prev", "amt_zscore",

    # сессионные
    "sec_since_session_start", "session_amt_before", "cnt_prev_same_session",

    # velocity-счётчики
    "cnt_prev_same_type", "cnt_prev_same_desc", "cnt_prev_same_mcc",
    "cnt_prev_same_subtype", "cnt_prev_same_channel_type",
    "cnt_prev_same_currency", "cnt_prev_same_device",
    "sec_since_prev_same_type", "sec_since_prev_same_desc",
    "sec_since_prev_same_channel_type", "sec_since_prev_same_currency", "sec_since_prev_same_device",
    "events_before_today", "cnt_events_this_hour",

    # флаги «впервые»
    "is_new_event_type", "is_new_event_desc", "is_new_channel_sub",
    "is_new_channel_type", "is_new_mcc", "is_new_currency",
    "is_new_device_fp", "is_first_in_session",

    # rolling
    "amt_sum_last_1h", "cnt_last_1h",
    "amt_sum_last_24h", "cnt_last_24h", "max_amt_last_24h",
    "amt_vs_1h_sum", "amt_vs_24h_sum",

    # история меток клиента
    "cust_prev_red_lbl_cnt", "cust_prev_yellow_lbl_cnt", "cust_prev_labeled_cnt",
    "cust_prev_red_lbl_rate", "cust_prev_yellow_lbl_rate", "cust_prev_susp_lbl_rate",
    "cust_prev_any_red_flag", "cust_prev_any_yellow_flag",
    "sec_since_prev_red_lbl", "sec_since_prev_yellow_lbl",

    # fine-grained label history
    "cnt_prev_labeled_same_desc", "cnt_prev_red_same_desc_lbl", "cnt_prev_yellow_same_desc_lbl",
    "red_rate_prev_same_desc_lbl",
    "cnt_prev_red_same_channel", "cnt_prev_labeled_same_channel", "red_rate_prev_same_channel",
    "cnt_prev_red_same_type_lbl", "cnt_prev_labeled_same_type_lbl", "red_rate_prev_same_type",
]

CAT_COLS = [
    "customer_id", "event_type_nm", "event_desc", "channel_indicator_type",
    "channel_indicator_sub_type", "currency_iso_cd", "mcc_code_i", "pos_cd",
    "timezone", "operating_system_type", "phone_voip_call_state", "web_rdp_connection",
    "developer_tools_i", "compromised_i",
    "accept_language_i", "browser_language_i",
    "device_fp_i",
]

META_COLS = [
    "event_id", "period", "event_ts",
    "is_train_sample", "is_test",
    "train_target_raw", "target_bin",
]

MODEL_DROP_COLS = {
    "target", "keep_green", "event_date", "event_hour_trunc",
    "month", "event_day_number",
    "session_id",
}

RISKY_LABEL_FEATURES = {
    "cnt_prev_labeled_same_desc", "cnt_prev_red_same_desc_lbl", "cnt_prev_yellow_same_desc_lbl",
    "red_rate_prev_same_desc_lbl",
    "cnt_prev_red_same_channel", "cnt_prev_labeled_same_channel", "red_rate_prev_same_channel",
    "cnt_prev_red_same_type_lbl", "cnt_prev_labeled_same_type_lbl", "red_rate_prev_same_type",
}

RISKY_PRIOR_SUFFIXES = (
    "_red_rate_all",
    "_labeled_rate_all",
    "_red_share_labeled",
)
