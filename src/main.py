"""Entry point for the full pipeline.

Usage:
    python -m src.main                    # all stages
    python -m src.main --stage meta       # only 5 models + 8 blends (13 CSV)
    python -m src.main --stage sol154     # only sol154 pipeline
    python -m src.main --stage mega       # only mega-ensemble (needs ready CSVs)
"""
import argparse
import time


def run_meta_stage():
    """Train 5 models, optimize blends, generate 13 submissions."""
    from src.features.preparation import run_preparation
    from src.models.catboost_model import run_catboost_train
    from src.models.lightgbm_model import run_lightgbm_train
    from src.models.xgboost_model import run_xgboost_train
    from src.models.extratrees_model import run_extratrees_train
    from src.models.mlp_model import run_mlp_train
    from src.blend import run_blend_optimization, run_full_refit_and_submission
    from src.evaluate import run_feature_importance, run_results_table
    from src.config import (
        TRAIN_CB, TRAIN_LGB, TRAIN_XGB, TRAIN_ET, TRAIN_MLP,
    )

    print("=" * 60)
    print("STAGE: META (5 models + blends)")
    print("=" * 60)

    # Step 1: Feature engineering + preparation
    prep = run_preparation()

    X_main_tr = prep["X_main_tr"]
    y_main_tr = prep["y_main_tr"]
    w_main_tr = prep["w_main_tr"]
    X_main_val = prep["X_main_val"]
    y_main_val = prep["y_main_val"]
    FULL_FEATURE_COLS = prep["FULL_FEATURE_COLS"]

    # Step 2: Train models
    pred_cb_val, best_iter_cb, ap_cb = run_catboost_train(
        X_main_tr, y_main_tr, w_main_tr, X_main_val, y_main_val, train=TRAIN_CB
    )
    pred_lgb_val, best_iter_lgb, ap_lgb = run_lightgbm_train(
        X_main_tr, y_main_tr, w_main_tr, X_main_val, y_main_val, train=TRAIN_LGB
    )
    pred_xgb_val, best_iter_xgb, ap_xgb = run_xgboost_train(
        X_main_tr, y_main_tr, w_main_tr, X_main_val, y_main_val, train=TRAIN_XGB
    )
    pred_et_val, ap_et = run_extratrees_train(
        X_main_tr, y_main_tr, w_main_tr, X_main_val, y_main_val, train=TRAIN_ET
    )
    pred_mlp_val, best_epoch_mlp, ap_mlp = run_mlp_train(
        X_main_tr, y_main_tr, w_main_tr, X_main_val, y_main_val, train=TRAIN_MLP
    )

    # Step 3: Blend optimization
    val_preds = {
        "catboost": pred_cb_val,
        "lightgbm": pred_lgb_val,
        "xgboost": pred_xgb_val,
        "extratrees": pred_et_val,
        "mlp": pred_mlp_val,
    }
    val_scores = {
        "catboost": ap_cb,
        "lightgbm": ap_lgb,
        "xgboost": ap_xgb,
        "extratrees": ap_et,
        "mlp": ap_mlp,
    }
    best_iters = {
        "catboost": best_iter_cb,
        "lightgbm": best_iter_lgb,
        "xgboost": best_iter_xgb,
    }

    blend_results, best_combo_name = run_blend_optimization(val_preds, y_main_val)

    # Step 4: Feature importance + results table
    run_feature_importance(FULL_FEATURE_COLS)
    run_results_table(
        ap_cb, ap_lgb, ap_xgb, ap_et, ap_mlp,
        best_iter_cb, best_iter_lgb, best_iter_xgb, best_epoch_mlp,
        blend_results, best_combo_name,
    )

    # Step 5: Full refit + generate submissions
    run_full_refit_and_submission(
        prep, val_scores, best_iters, blend_results, best_combo_name
    )

    return val_scores, best_combo_name


def run_sol154_stage():
    """Run the sol154 pipeline (or use intermediates)."""
    print("=" * 60)
    print("STAGE: SOL154")
    print("=" * 60)
    from src.sol154.runner import run as run_sol154
    totalblend_path = run_sol154()
    return totalblend_path


def run_mega_stage(val_scores=None, best_combo_name=None):
    """Run mega-ensemble blending meta3b1n + sol154."""
    print("=" * 60)
    print("STAGE: MEGA-ENSEMBLE")
    print("=" * 60)
    from src.mega_ensemble import run_mega_ensemble
    from src.sol154.config import SOL154_AGI_DIR, SOL154_INTERMEDIATES

    # Find sol154 blend
    for candidate in [
        SOL154_AGI_DIR / "sub_totalblend.csv",
        SOL154_INTERMEDIATES / "sub_totalblend.csv",
    ]:
        if candidate.exists():
            sol154_blend_path = candidate
            break
    else:
        raise FileNotFoundError(
            "sub_totalblend.csv not found. Run --stage sol154 first, "
            "or place it in src/sol154/intermediates/"
        )

    # If val_scores not provided, use placeholder values
    if val_scores is None:
        print("WARNING: val_scores not provided, using defaults for mega-ensemble")
        val_scores = {
            "catboost": 0.0, "lightgbm": 0.0, "xgboost": 0.0,
            "extratrees": 0.0, "mlp": 0.0,
        }
    if best_combo_name is None:
        best_combo_name = "3boost"

    run_mega_ensemble(sol154_blend_path, val_scores, best_combo_name)


def main():
    parser = argparse.ArgumentParser(description="Data Fusion Anti-Fraud Pipeline")
    parser.add_argument(
        "--stage",
        choices=["all", "meta", "sol154", "mega"],
        default="all",
        help="Which stage to run (default: all)",
    )
    args = parser.parse_args()

    t0 = time.time()

    if args.stage == "all":
        val_scores, best_combo_name = run_meta_stage()
        run_sol154_stage()
        run_mega_stage(val_scores, best_combo_name)

    elif args.stage == "meta":
        run_meta_stage()

    elif args.stage == "sol154":
        run_sol154_stage()

    elif args.stage == "mega":
        run_mega_stage()

    elapsed = (time.time() - t0) / 60
    print(f"\nTotal time: {elapsed:.1f} min")


if __name__ == "__main__":
    main()
