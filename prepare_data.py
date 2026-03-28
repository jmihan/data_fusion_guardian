import argparse
import gc
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert raw data to single all-in-one table."
    )

    parser.add_argument("input_dir", type=Path)
    parser.add_argument("output_path", type=Path)

    args = parser.parse_args()

    if not args.input_dir.exists():
        raise ValueError("input_dir not found")

    return args


def convert_train_labels(data: pd.DataFrame) -> pd.DataFrame:
    data["target"] = data["target"].astype(np.int8)
    return data


def _safe_fillna_uint8(series: pd.Series, fill_val: int = 0) -> np.ndarray:
    """fillna + astype for nullable numeric -> uint8, no string intermediaries."""
    arr = series.to_numpy(dtype=np.float32, na_value=np.nan)
    mask = np.isnan(arr)
    arr[mask] = fill_val
    return arr.astype(np.uint8)


def convert_events(data: pd.DataFrame) -> pd.DataFrame:
    """In-place conversion — no .copy() to save memory."""

    # event_dttm
    data["event_dttm"] = pd.to_datetime(data["event_dttm"], format="%Y-%m-%d %H:%M:%S")

    # event_type_nm
    data["event_type_nm"] = data["event_type_nm"].astype(np.uint8)

    # event_desc
    data["event_desc"] = data["event_desc"].astype(np.uint8)

    # channel_indicator_type
    data["channel_indicator_type"] = data["channel_indicator_type"].astype(np.uint8)

    # channel_indicator_sub_type
    data["channel_indicator_sub_type"] = data["channel_indicator_sub_type"].astype(np.uint8)

    # currency_iso_cd  (NaN -> 0, else val+1)
    data["currency_iso_cd"] = _safe_fillna_uint8(data["currency_iso_cd"].fillna(-1.0) + 1)

    # mcc_code  (string col: NaN -> 0, else int(val)+1)
    mcc = data["mcc_code"]
    if mcc.dtype == object or hasattr(mcc, 'str'):
        mcc_arr = pd.to_numeric(mcc, errors="coerce")
    else:
        mcc_arr = mcc.astype(np.float32)
    data["mcc_code"] = _safe_fillna_uint8(mcc_arr + 1)
    del mcc, mcc_arr

    # pos_cd
    data["pos_cd"] = _safe_fillna_uint8(data["pos_cd"].fillna(-1.0) + 1)

    # browser_language
    def browser_language_to_int(s) -> int:
        if not isinstance(s, str):
            return 0
        if s == "not available":
            return 1
        return 2

    data["browser_language"] = data["browser_language"].apply(browser_language_to_int).astype(np.uint8)

    # timezone
    tz = data["timezone"].to_numpy(dtype=np.float32, na_value=np.nan)
    tz_mask = np.isnan(tz)
    tz[tz_mask] = -1.0
    data["timezone"] = (tz + 1).astype(np.uint16)
    del tz, tz_mask

    # operating_system_type
    data["operating_system_type"] = _safe_fillna_uint8(data["operating_system_type"].fillna(-1.0) + 1)

    # battery
    def battery_to_float(s) -> float:
        if not isinstance(s, str):
            return np.nan
        if s == "NaN%":
            return -2.0
        if s == "not available":
            return -1.0
        if s.startswith(":"):
            return float(s[1:].partition("%")[0])
        return float(s.removesuffix("%"))

    data["battery"] = data["battery"].apply(battery_to_float).astype(np.float32)

    # device_system_version
    def version_to_int(s) -> int:
        if not isinstance(s, str):
            return 0
        x = s.split(".")
        a = int(x[0])
        b = int(x[1]) if len(x) > 1 else 0
        c = int(x[2]) if len(x) > 2 else 0
        return a * 10000 + b * 100 + c

    dsv = data["device_system_version"]
    parts = dsv.apply(lambda x: 0 if not isinstance(x, str) else x.count(".") + 1).astype(np.uint8)
    version = dsv.apply(version_to_int).astype(np.uint32)

    data.drop(columns=["device_system_version"], inplace=True)
    data["device_system_version"] = version
    data["device_system_version_parts"] = parts
    del dsv, parts, version

    # screen_size
    ss = data["screen_size"]
    s1 = ss.apply(lambda x: int(x.split("x")[0]) if isinstance(x, str) else 0).astype(np.uint16)
    s2 = ss.apply(lambda x: int(x.split("x")[1]) if isinstance(x, str) else 0).astype(np.uint16)
    data.drop(columns=["screen_size"], inplace=True)
    data["screen_size_1"] = s1
    data["screen_size_2"] = s2
    del ss, s1, s2

    # developer_tools  (string col: NaN -> 0, else int(val)+1)
    dt = data["developer_tools"]
    if dt.dtype == object or hasattr(dt, 'str'):
        dt_arr = pd.to_numeric(dt, errors="coerce")
    else:
        dt_arr = dt.astype(np.float32)
    data["developer_tools"] = _safe_fillna_uint8(dt_arr + 1)
    del dt, dt_arr

    # phone_voip_call_state
    data["phone_voip_call_state"] = _safe_fillna_uint8(data["phone_voip_call_state"].fillna(-1.0) + 1)

    # web_rdp_connection
    data["web_rdp_connection"] = _safe_fillna_uint8(data["web_rdp_connection"].fillna(-1.0) + 1)

    # compromised  (string col: NaN -> 0, else int(val)+1)
    comp = data["compromised"]
    if comp.dtype == object or hasattr(comp, 'str'):
        comp_arr = pd.to_numeric(comp, errors="coerce")
    else:
        comp_arr = comp.astype(np.float32)
    data["compromised"] = _safe_fillna_uint8(comp_arr + 1)
    del comp, comp_arr

    return data


def prepare_data(input_dir: Path, output_path: Path):
    parts = []

    # --- Process each dataset separately to limit peak memory ---

    # Pretest
    print("Load & convert pretest...")
    df = pd.read_parquet(input_dir / "pretest.parquet")
    df = convert_events(df)
    df["original_index"] = df.index.astype(np.uint32)
    df["dataset"] = np.uint8(2)
    df["target"] = np.uint8(0)
    parts.append(df)
    del df; gc.collect()

    # Test
    print("Load & convert test...")
    df = pd.read_parquet(input_dir / "test.parquet")
    df = convert_events(df)
    df["original_index"] = df.index.astype(np.uint32)
    df["dataset"] = np.uint8(3)
    df["target"] = np.uint8(0)
    parts.append(df)
    del df; gc.collect()

    # Pretrain (3 parts — load and convert one at a time, then concat)
    print("Load & convert pretrain...")
    pretrain_parts = []
    for i in range(1, 4):
        print(f"  pretrain_part_{i}...")
        chunk = pd.read_parquet(input_dir / f"pretrain_part_{i}.parquet")
        chunk = convert_events(chunk)
        pretrain_parts.append(chunk)
        del chunk; gc.collect()
    df = pd.concat(pretrain_parts, ignore_index=True)
    del pretrain_parts; gc.collect()
    df["original_index"] = df.index.astype(np.uint32)
    df["dataset"] = np.uint8(0)
    df["target"] = np.uint8(0)
    parts.append(df)
    del df; gc.collect()

    # Train (3 parts)
    print("Load & convert train...")
    train_parts = []
    for i in range(1, 4):
        print(f"  train_part_{i}...")
        chunk = pd.read_parquet(input_dir / f"train_part_{i}.parquet")
        chunk = convert_events(chunk)
        train_parts.append(chunk)
        del chunk; gc.collect()
    df = pd.concat(train_parts, ignore_index=True)
    del train_parts; gc.collect()
    df["original_index"] = df.index.astype(np.uint32)
    df["dataset"] = np.uint8(1)
    df["target"] = np.uint8(1)

    # Apply train labels
    print("Apply train labels...")
    train_labels = pd.read_parquet(input_dir / "train_labels.parquet")
    events0 = set(train_labels.loc[train_labels["target"] == 0, "event_id"])
    events1 = set(train_labels.loc[train_labels["target"] == 1, "event_id"])
    del train_labels; gc.collect()

    df.loc[df["event_id"].isin(events0), "target"] = np.uint8(2)
    df.loc[df["event_id"].isin(events1), "target"] = np.uint8(3)
    del events0, events1
    parts.append(df)
    del df; gc.collect()

    # Concat all
    print("Combine all data...")
    all_data = pd.concat(parts, ignore_index=True)
    del parts; gc.collect()

    all_data["dataset"] = all_data["dataset"].astype(np.uint8)
    all_data["target"] = all_data["target"].astype(np.uint8)

    print("Sort...")
    all_data = all_data.sort_values(["customer_id", "event_dttm", "dataset"], kind="stable", ignore_index=True)

    print("Drop duplicates...")
    all_data["duplicated_event"] = all_data["event_id"].duplicated(keep="first")
    all_data = all_data.drop_duplicates("event_id", keep="last", ignore_index=True)

    print("Save data...")
    all_data.to_parquet(output_path, compression="zstd")
    print(f"Done: {len(all_data):,} rows -> {output_path}")


def main():
    args = parse_args()
    prepare_data(args.input_dir, args.output_path)


if __name__ == "__main__":
    main()
