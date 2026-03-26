# 1. Импорты и настройки

import polars as pl
from pathlib import Path

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(exist_ok=True, parents=True)

# Список файлов для обработки
pretrain_files = [
    "pretrain_part_1.parquet",
    "pretrain_part_2.parquet",
    "pretrain_part_3.parquet"
]
train_files = [
    "train_part_1.parquet",
    "train_part_2.parquet",
    "train_part_3.parquet"
]
pretest_file = "pretest.parquet"
test_file = "test.parquet"
labels_file = "train_labels.parquet"

# 2. Загрузка меток для train

# Загружаем метки (event_id, target)
labels_lf = pl.scan_parquet(RAW_DIR / labels_file).select(["event_id", "target"])
# Соберём в память, так как меток мало (~87k строк)
labels_df = labels_lf.collect()
print(f"Загружено меток: {len(labels_df)}")

# 3. Универсальная функция обработки одного файла

def process_file(
    input_path: Path,
    output_path: Path,
    labels: pl.DataFrame | None = None,
    drop_duplicates: bool = False
):
    """
    Загружает и обрабатывает один файл.
    
    Parameters
    ----------
    input_path : Path
        путь к исходному parquet
    output_path : Path
        путь для сохранения обработанного parquet
    labels : pl.DataFrame, optional
        датафрейм с колонками ['event_id', 'target'] для присоединения меток
    drop_duplicates : bool
        удалять ли полные дубликаты строк
    """
    lf = pl.scan_parquet(input_path)

    if drop_duplicates:
        lf = lf.unique()

    # Удаляем browser_language (бесполезный признак)
    lf = lf.drop("browser_language")

    original_cols = set(lf.collect_schema().names())

    # Словарь новых колонок для добавления
    exprs = []

    # 1. event_dttm -> datetime
    if "event_dttm" in original_cols:
        exprs.append(pl.col("event_dttm").str.to_datetime(format="%Y-%m-%d %H:%M:%S").alias("event_dttm"))

    # 2. mcc_code -> int64
    if "mcc_code" in original_cols:
        exprs.append(pl.col("mcc_code").cast(pl.Int64, strict=False).alias("mcc_code"))

    # 3. accept_language: чистка и создание дополнительных признаков
    if "accept_language" in original_cols:
        # Очищаем: заменяем пропуски и 'not available' на None
        accept_clean = (
            pl.when(pl.col("accept_language").str.to_lowercase().is_in(["", "not available", "null"]))
            .then(None)
            .otherwise(pl.col("accept_language"))
            .alias("accept_language")
        )
        exprs.append(accept_clean)
        
        # Первичный язык (первые два символа)
        exprs.append(
            pl.col("accept_language")
            .str.to_lowercase()
            .str.replace_all(r"[;,\s].*", "")
            .str.slice(0, 2)
            .alias("accept_language_primary")
        )
        
        # Флаг "*"
        exprs.append(
            (pl.col("accept_language") == "*").alias("accept_language_is_star")
        )
        
        # Флаг наличия запятой (несколько языков)
        exprs.append(
            pl.col("accept_language").str.contains(",").alias("accept_language_has_comma")
        )

    # 4. battery -> battery_value (число)
    if "battery" in original_cols:
        exprs.append(
            pl.col("battery")
            .str.replace("%", "")
            .str.strip_chars()
            .cast(pl.Float64, strict=False)
            .alias("battery_value")
        )

    # 5. device_system_version -> version_major (первое число до точки)
    if "device_system_version" in original_cols:
        exprs.append(
            pl.col("device_system_version")
            .str.split(".")
            .list.first()
            .cast(pl.Int64, strict=False)
            .alias("version_major")
        )
        # исходную колонку оставляем как есть

    # 6. screen_size -> screen_width, screen_height
    if "screen_size" in original_cols:
        exprs.append(
            pl.col("screen_size")
            .str.split("x")
            .list.get(0)
            .cast(pl.Int64, strict=False)
            .alias("screen_width")
        )
        exprs.append(
            pl.col("screen_size")
            .str.split("x")
            .list.get(1)
            .cast(pl.Int64, strict=False)
            .alias("screen_height")
        )

    # 7. developer_tools -> int64
    if "developer_tools" in original_cols:
        exprs.append(pl.col("developer_tools").cast(pl.Int64, strict=False).alias("developer_tools"))

    # 8. compromised -> int64
    if "compromised" in original_cols:
        exprs.append(pl.col("compromised").cast(pl.Int64, strict=False).alias("compromised"))

    # Применяем все выражения
    if exprs:
        lf = lf.with_columns(exprs)

    # Удаляем исходные колонки, которые заменили новыми
    drop_cols = []
    if "battery" in original_cols:
        drop_cols.append("battery")
    if "screen_size" in original_cols:
        drop_cols.append("screen_size")
    if drop_cols:
        lf = lf.drop(drop_cols)

    # Присоединяем метки, если переданы
    if labels is not None:
        lf = lf.join(labels.lazy(), on="event_id", how="left")
        lf = lf.with_columns(pl.col("target").fill_null(-1).alias("target"))

    # Получаем актуальную схему
    current_schema = lf.collect_schema()
    current_cols = set(current_schema.names())

    # Финальный список обязательных колонок (без target)
    final_cols = [
        "customer_id", "event_id", "event_dttm", "event_type_nm", "event_desc",
        "channel_indicator_type", "channel_indicator_sub_type", "operaton_amt",
        "currency_iso_cd", "mcc_code", "pos_cd", "timezone", "session_id",
        "operating_system_type", "phone_voip_call_state", "web_rdp_connection",
        "accept_language", "accept_language_primary", "accept_language_is_star",
        "accept_language_has_comma", "battery_value", "device_system_version",
        "version_major", "screen_width", "screen_height", "developer_tools", "compromised"
    ]

    # Добавляем target, если он был присоединён или присутствует в исходных
    if "target" in current_cols:
        final_cols.append("target")

    # Добавляем недостающие колонки со значением NULL
    missing = [col for col in final_cols if col not in current_cols]
    if missing:
        lf = lf.with_columns([pl.lit(None).alias(col) for col in missing])

    # Оставляем только нужные колонки
    lf = lf.select(final_cols)

    # Сохраняем
    lf.sink_parquet(output_path, compression="zstd")
    print(f"Обработан и сохранён: {output_path}")


# 4. Обработка pretrain частей (меток нет, дубликатов нет)

for fname in pretrain_files:
    in_path = RAW_DIR / fname
    out_path = PROCESSED_DIR / fname.replace(".parquet", "_processed.parquet")
    process_file(in_path, out_path, labels=None, drop_duplicates=False)

# 5. Обработка train частей (присоединяем метки)

for fname in train_files:
    in_path = RAW_DIR / fname
    out_path = PROCESSED_DIR / fname.replace(".parquet", "_processed.parquet")
    process_file(in_path, out_path, labels=labels_df, drop_duplicates=False)

# 6. Обработка pretest (удаляем дубликаты, меток нет)

in_path = RAW_DIR / pretest_file
out_path = PROCESSED_DIR / "pretest_processed.parquet"
process_file(in_path, out_path, labels=None, drop_duplicates=True)

# 7. Обработка test (меток нет, дубликатов нет)

in_path = RAW_DIR / test_file
out_path = PROCESSED_DIR / "test_processed.parquet"
process_file(in_path, out_path, labels=None, drop_duplicates=False)

print("Все файлы успешно обработаны.")
