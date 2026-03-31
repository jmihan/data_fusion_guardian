import gc

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from catboost import CatBoostClassifier
import lightgbm as lgb
import xgboost as xgb

from src.config import (
    CACHE_DIR,
    RETRAIN_ON_FULL_CB, RETRAIN_ON_FULL_LGB, RETRAIN_ON_FULL_XGB,
    ET_PARAMS,
)


def run_feature_importance(FULL_FEATURE_COLS):
    _cb_imp_path = CACHE_DIR / ("cb_main_full.cbm" if RETRAIN_ON_FULL_CB else "cb_main.cbm")
    _model_cb_imp = CatBoostClassifier().load_model(str(_cb_imp_path))
    cb_imp = _model_cb_imp.get_feature_importance()
    cb_feat_names = _model_cb_imp.feature_names_ if hasattr(_model_cb_imp, "feature_names_") else FULL_FEATURE_COLS
    cb_feat_imp = pd.DataFrame({"feature": cb_feat_names, "importance": cb_imp}).sort_values("importance", ascending=False)
    del _model_cb_imp

    _lgb_imp_path = CACHE_DIR / ("lgb_main_full.txt" if RETRAIN_ON_FULL_LGB else "lgb_main.txt")
    _model_lgb_imp = lgb.Booster(model_file=str(_lgb_imp_path))
    lgb_imp_raw = _model_lgb_imp.feature_importance(importance_type="gain")
    lgb_feat_imp = pd.DataFrame({
        "feature": FULL_FEATURE_COLS[:len(lgb_imp_raw)],
        "importance": lgb_imp_raw,
    }).sort_values("importance", ascending=False)
    del _model_lgb_imp

    _xgb_imp_path = CACHE_DIR / ("xgb_main_full.json" if RETRAIN_ON_FULL_XGB else "xgb_main.json")
    _model_xgb_imp = xgb.Booster()
    _model_xgb_imp.load_model(str(_xgb_imp_path))
    xgb_imp_dict = _model_xgb_imp.get_score(importance_type="gain")
    xgb_feat_imp = pd.DataFrame([
        {"feature": f, "importance": xgb_imp_dict.get(f, 0.0)} for f in FULL_FEATURE_COLS
    ]).sort_values("importance", ascending=False)
    del _model_xgb_imp; gc.collect()

    fig, axes = plt.subplots(1, 3, figsize=(20, 8))
    for ax, df_imp, title in zip(
        axes,
        [cb_feat_imp, lgb_feat_imp, xgb_feat_imp],
        ["CatBoost", "LightGBM", "XGBoost"],
    ):
        top = df_imp.head(20)
        ax.barh(top["feature"].values[::-1], top["importance"].values[::-1])
        ax.set_title(f"{title} Top-20 Features (gain)")
    plt.tight_layout()
    plt.show()


def run_results_table(ap_cb, ap_lgb, ap_xgb, ap_et, ap_mlp,
                      best_iter_cb, best_iter_lgb, best_iter_xgb, best_epoch_mlp,
                      blend_results, best_combo_name):
    results_rows = [
        ("CatBoost", ap_cb, str(best_iter_cb)),
        ("LightGBM", ap_lgb, str(best_iter_lgb)),
        ("XGBoost", ap_xgb, str(best_iter_xgb)),
        ("ExtraTrees", ap_et, str(ET_PARAMS["n_estimators"])),
        ("Simple MLP", ap_mlp, str(best_epoch_mlp)),
    ]
    for combo_name, res in blend_results.items():
        results_rows.append((f"Blend: {combo_name}", res["val_ap"], "-"))

    results = pd.DataFrame(results_rows, columns=["Model", "Val PR-AUC", "Best Iter/Epoch"])
    results = results.sort_values("Val PR-AUC", ascending=False)
    results["Val PR-AUC"] = results["Val PR-AUC"].apply(lambda x: f"{x:.6f}" if isinstance(x, float) else x)
    print("\n" + results.to_string(index=False))

    best_blend = blend_results[best_combo_name]
    print(f"\nBest blend: {best_combo_name} (PR-AUC = {best_blend['val_ap']:.6f})")
    print(f"Blend weights: { {best_blend['keys'][i]: round(float(best_blend['weights'][i]), 4) for i in range(len(best_blend['keys']))} }")
