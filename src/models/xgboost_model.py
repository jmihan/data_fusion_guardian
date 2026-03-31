import gc
import json

import numpy as np
from sklearn.metrics import average_precision_score
import xgboost as xgb

from src.config import CACHE_DIR, REFIT_ITER_MULT, XGB_PARAMS, USE_GPU


def fit_xgb_with_holdout(X_tr, y_tr, w_tr, X_val, y_val, w_val, params, n_estimators=6000):
    dtrain = xgb.DMatrix(X_tr, label=y_tr, weight=w_tr)
    dval = xgb.DMatrix(X_val, label=y_val, weight=w_val)

    try:
        model = xgb.train(
            params, dtrain, num_boost_round=n_estimators,
            evals=[(dval, "val")],
            early_stopping_rounds=300,
            verbose_eval=200,
        )
    except Exception as e:
        print(f"XGB GPU failed, fallback to CPU: {e}")
        params_cpu = params.copy()
        params_cpu["device"] = "cpu"
        params_cpu["tree_method"] = "hist"
        model = xgb.train(
            params_cpu, dtrain, num_boost_round=n_estimators,
            evals=[(dval, "val")],
            early_stopping_rounds=300,
            verbose_eval=200,
        )

    val_pred = model.predict(dval)
    val_ap = average_precision_score(y_val, val_pred)
    best_iter = model.best_iteration
    print(f"XGB best_iter={best_iter}, val_pr_auc={val_ap:.6f}")
    return model, best_iter, val_ap, val_pred


def refit_full_xgb(X, y, w, params, best_iter):
    n_iter = int(max(100, round(best_iter * REFIT_ITER_MULT)))
    dtrain = xgb.DMatrix(X, label=y, weight=w)
    model = xgb.train(params, dtrain, num_boost_round=n_iter, verbose_eval=200)
    return model


def run_xgboost_train(X_main_tr, y_main_tr, w_main_tr, X_main_val, y_main_val, w_main_val, train=True):
    if train:
        print("=" * 60)
        print("Training XGBoost MAIN model")
        print("=" * 60)
        model_xgb, best_iter_xgb, ap_xgb, pred_xgb_val = fit_xgb_with_holdout(
            X_main_tr, y_main_tr, w_main_tr,
            X_main_val, y_main_val, w_main_val,
            params=XGB_PARAMS,
        )
        model_xgb.save_model(str(CACHE_DIR / "xgb_main.json"))
        del model_xgb; gc.collect()

        xgb_meta = {"best_iter": best_iter_xgb, "val_ap": ap_xgb}
        with open(CACHE_DIR / "xgb_meta.json", "w") as f:
            json.dump(xgb_meta, f, indent=2)
        print(f"XGBoost val PR-AUC: {ap_xgb:.6f}")

    else:
        print("Loading XGBoost from cache...")
        with open(CACHE_DIR / "xgb_meta.json") as f:
            xgb_meta = json.load(f)
        best_iter_xgb = xgb_meta["best_iter"]
        ap_xgb = xgb_meta["val_ap"]

        model_xgb = xgb.Booster()
        model_xgb.load_model(str(CACHE_DIR / "xgb_main.json"))
        dval = xgb.DMatrix(X_main_val)
        pred_xgb_val = model_xgb.predict(dval)
        del model_xgb, dval; gc.collect()
        print(f"XGBoost val PR-AUC: {ap_xgb:.6f} (cached)")

    return pred_xgb_val, best_iter_xgb, ap_xgb
