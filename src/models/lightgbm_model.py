import gc
import json

import numpy as np
from sklearn.metrics import average_precision_score
import lightgbm as lgb

from src.config import CACHE_DIR, RANDOM_SEED, REFIT_ITER_MULT, LGB_PARAMS
from src.utils import lgb_prepare, lgb_ap_metric


def fit_lgb_with_holdout(X_tr, y_tr, w_tr, X_val, y_val, w_val, cat_cols, params):
    p = params.copy()
    n_estimators = int(p.pop("n_estimators", 5000))
    early_stopping_rounds = int(p.pop("early_stopping_rounds", 200))
    p.update({
        "objective": "binary",
        "metric": "None",
        "seed": RANDOM_SEED,
        "verbosity": -1,
        "n_jobs": -1,
        "feature_pre_filter": False,
    })

    Xtr_ = lgb_prepare(X_tr, cat_cols)
    Xval_ = lgb_prepare(X_val, cat_cols)

    tr_ds = lgb.Dataset(Xtr_, label=y_tr, weight=w_tr, categorical_feature=cat_cols, free_raw_data=False)
    val_ds = lgb.Dataset(Xval_, label=y_val, weight=w_val, reference=tr_ds, categorical_feature=cat_cols, free_raw_data=False)

    callbacks = [
        lgb.early_stopping(early_stopping_rounds, first_metric_only=True, verbose=False),
        lgb.log_evaluation(200),
    ]

    model = lgb.train(
        p, tr_ds, num_boost_round=n_estimators,
        valid_sets=[val_ds], valid_names=["valid"],
        feval=lgb_ap_metric, callbacks=callbacks,
    )

    best_iter = model.best_iteration or n_estimators
    val_raw = model.predict(Xval_, num_iteration=best_iter)
    val_ap = average_precision_score(y_val, val_raw)
    print(f"LGB best_iter={best_iter}, val_pr_auc={val_ap:.6f}")
    return model, int(best_iter), float(val_ap), p


def refit_full_lgb(X, y, w, cat_cols, base_params, best_iter):
    p = {k: v for k, v in base_params.items() if k not in ("n_estimators", "early_stopping_rounds")}
    p.update({
        "objective": "binary",
        "metric": "None",
        "seed": RANDOM_SEED,
        "verbosity": -1,
        "n_jobs": -1,
        "feature_pre_filter": False,
    })
    X_ = lgb_prepare(X, cat_cols)
    ds = lgb.Dataset(X_, label=y, weight=w, categorical_feature=cat_cols, free_raw_data=False)
    model = lgb.train(p, ds, num_boost_round=int(max(100, round(best_iter * REFIT_ITER_MULT))))
    return model


def run_lightgbm_train(X_main_tr, y_main_tr, w_main_tr, X_main_val, y_main_val, w_main_val, CAT_FEATURE_COLS, train=True):
    if train:
        print("=" * 60)
        print("Training LightGBM MAIN model")
        print("=" * 60)
        model_lgb, best_iter_lgb, ap_lgb, used_lgb = fit_lgb_with_holdout(
            X_main_tr, y_main_tr, w_main_tr,
            X_main_val, y_main_val, w_main_val,
            cat_cols=CAT_FEATURE_COLS,
            params=LGB_PARAMS,
        )
        model_lgb.save_model(str(CACHE_DIR / "lgb_main.txt"))

        pred_lgb_val = model_lgb.predict(
            lgb_prepare(X_main_val, CAT_FEATURE_COLS),
            num_iteration=best_iter_lgb,
        )
        del model_lgb; gc.collect()

        lgb_meta = {"best_iter": best_iter_lgb, "val_ap": ap_lgb}
        with open(CACHE_DIR / "lgb_meta.json", "w") as f:
            json.dump(lgb_meta, f, indent=2)
        print(f"LightGBM val PR-AUC: {ap_lgb:.6f}")

    else:
        print("Loading LightGBM from cache...")
        with open(CACHE_DIR / "lgb_meta.json") as f:
            lgb_meta = json.load(f)
        best_iter_lgb = lgb_meta["best_iter"]
        ap_lgb = lgb_meta["val_ap"]

        model_lgb = lgb.Booster(model_file=str(CACHE_DIR / "lgb_main.txt"))
        pred_lgb_val = model_lgb.predict(
            lgb_prepare(X_main_val, CAT_FEATURE_COLS),
            num_iteration=best_iter_lgb,
        )
        del model_lgb; gc.collect()
        print(f"LightGBM val PR-AUC: {ap_lgb:.6f} (cached)")

    return pred_lgb_val, best_iter_lgb, ap_lgb
