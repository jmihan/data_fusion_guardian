"""sol154 pipeline orchestrator.

Steps:
1. solution1: CatBoost 4-sub-model pipeline -> submission_ICEQ_PUBLIC.csv
2. solution1/coles: CoLES embeddings + refit -> coles_seed_fb*.csv
3. solution2: DL LSTM/GRU multitask -> submission_DL_PUBLIC.csv
4. pipeline1st: LGB+CB 4-fold ensemble -> submission_MINE.csv
5. agi_blend: z-score blend of 3 inputs -> sub_totalblend.csv

If intermediates exist in src/sol154/intermediates/, they are used as-is
(skipping the corresponding compute steps).
"""
import shutil
from pathlib import Path

from src.sol154.config import (
    SOL154_INTERMEDIATES,
    SOL154_AGI_DIR,
    SOL154_CACHE_DIR,
)


def _have_intermediate(name: str) -> bool:
    p = SOL154_INTERMEDIATES / name
    return p.exists() and p.stat().st_size > 0


def _copy_intermediate(name: str, dest_dir: Path) -> Path:
    src = SOL154_INTERMEDIATES / name
    dst = dest_dir / name
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    print(f"  [intermediate] {name} -> {dst}")
    return dst


def run(force_recompute: bool = False) -> Path:
    """Run the full sol154 pipeline, returning path to sub_totalblend.csv.

    Args:
        force_recompute: if True, ignore intermediates and recompute everything.

    Returns:
        Path to sub_totalblend.csv
    """
    SOL154_AGI_DIR.mkdir(parents=True, exist_ok=True)

    # Check for pre-computed totalblend shortcut
    if not force_recompute and _have_intermediate("sub_totalblend.csv"):
        print("sol154: using pre-computed sub_totalblend.csv from intermediates/")
        return _copy_intermediate("sub_totalblend.csv", SOL154_AGI_DIR)

    # ── Step 1: solution1 (CatBoost pipeline) -> submission_ICEQ_PUBLIC.csv ──
    if not force_recompute and _have_intermediate("submission_ICEQ_PUBLIC.csv"):
        print("sol154 step 1: using intermediate submission_ICEQ_PUBLIC.csv")
        _copy_intermediate("submission_ICEQ_PUBLIC.csv", SOL154_AGI_DIR)
    else:
        print("sol154 step 1: running solution1/catboost pipeline...")
        from src.sol154.solution1.run_catboost import run as run_catboost
        run_catboost()
        # The catboost pipeline outputs to SUBMISSIONS dir; copy to AGI
        from src.sol154.config import SOL154_SOLUTION1_SUBMISSIONS
        iceq_src = SOL154_SOLUTION1_SUBMISSIONS / "submission_ICEQ_PUBLIC.csv"
        if not iceq_src.exists():
            # May be named differently; look for the best submission
            import glob
            candidates = list(SOL154_SOLUTION1_SUBMISSIONS.glob("*.csv"))
            if candidates:
                iceq_src = candidates[0]
        if iceq_src.exists():
            shutil.copy2(iceq_src, SOL154_AGI_DIR / "submission_ICEQ_PUBLIC.csv")

    # ── Optional: CoLES embeddings + refit ──
    # (CoLES improves ICEQ but is optional — intermediates bypass this)

    # ── Step 2: solution2 (DL) -> submission_DL_PUBLIC.csv ──
    if not force_recompute and _have_intermediate("submission_DL_PUBLIC.csv"):
        print("sol154 step 2: using intermediate submission_DL_PUBLIC.csv")
        _copy_intermediate("submission_DL_PUBLIC.csv", SOL154_AGI_DIR)
    else:
        print("sol154 step 2: running solution2/train_pooling...")
        from src.sol154.solution2.train_last_n_pooling import RunConfig, run as run_dl
        config = RunConfig()
        dl_path = run_dl(config)
        shutil.copy2(dl_path, SOL154_AGI_DIR / "submission_DL_PUBLIC.csv")

    # ── Step 3: pipeline1st (LGB+CB) -> submission_MINE.csv ──
    if not force_recompute and _have_intermediate("submission_MINE.csv"):
        print("sol154 step 3: using intermediate submission_MINE.csv")
        _copy_intermediate("submission_MINE.csv", SOL154_AGI_DIR)
    else:
        print("sol154 step 3: running pipeline1st/lgbm_ensemble...")
        from src.sol154.pipeline1st.lgbm_ensemble import run as run_lgbm
        mine_path = run_lgbm()
        shutil.copy2(mine_path, SOL154_AGI_DIR / "submission_MINE.csv")

    # ── Step 4: AGI blend -> sub_totalblend.csv ──
    print("sol154 step 4: running agi_blend...")
    from src.sol154.agi_blend import run as run_agi
    totalblend_path = run_agi(work_dir=SOL154_AGI_DIR)

    print(f"sol154 pipeline complete: {totalblend_path}")
    return totalblend_path


if __name__ == "__main__":
    run()
