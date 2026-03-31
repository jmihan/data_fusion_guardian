"""AGI blend: z-score логитов -> взвешенная смесь -> sigmoid.

4-шаговый блендинг:
  DL + ICEQ -> BLEND1
  BLEND1 + MINE -> BLEND2
  BLEND2 + BLEND1 -> BLEND4
  BLEND1 + BLEND4 -> sub_totalblend.csv
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.sol154.config import SOL154_AGI_DIR


def logit(p: np.ndarray, eps: float) -> np.ndarray:
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return (x - x.mean()) / (x.std() + 1e-12)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def blend_two_csvs(
    path_a: Path,
    path_b: Path,
    out_path: Path,
    weight_a: float,
    eps: float = 1e-7,
) -> pd.DataFrame:
    a = pd.read_csv(path_a).rename(columns={"predict": "pred_a"})
    b = pd.read_csv(path_b).rename(columns={"predict": "pred_b"})
    df = a.merge(b, on="event_id", how="inner")
    za = zscore(logit(df["pred_a"].values, eps))
    zb = zscore(logit(df["pred_b"].values, eps))
    z = weight_a * za + (1.0 - weight_a) * zb
    out = df[["event_id"]].copy()
    out["predict"] = sigmoid(z)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"OK -> {out_path}  ({len(out):,} rows)")
    return out


BLEND_STEPS = [
    ("submission_DL_PUBLIC.csv", "submission_ICEQ_PUBLIC.csv", "BLEND1.csv", 0.55, 1e-7),
    ("BLEND1.csv", "submission_MINE.csv", "BLEND2.csv", 0.55, 1e-7),
    ("BLEND2.csv", "BLEND1.csv", "BLEND4.csv", 0.5485, 1e-8),
    ("BLEND1.csv", "BLEND4.csv", "sub_totalblend.csv", 0.541415926, 1e-9),
]


def run(work_dir: Path = None) -> Path:
    """Run the 4-step AGI blend.

    Args:
        work_dir: directory containing submission_*.csv inputs.
                  Defaults to SOL154_AGI_DIR.

    Returns:
        Path to sub_totalblend.csv
    """
    if work_dir is None:
        work_dir = SOL154_AGI_DIR
    work_dir = work_dir.resolve()

    for pa, pb, outp, w, ep in BLEND_STEPS:
        blend_two_csvs(work_dir / pa, work_dir / pb, work_dir / outp, w, ep)

    return work_dir / "sub_totalblend.csv"


if __name__ == "__main__":
    run()
