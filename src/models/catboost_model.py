import gc
import json
import os

import numpy as np
from sklearn.metrics import average_precision_score
from catboost import CatBoostClassifier, Pool

from src.config import CACHE_DIR, RANDOM_SEED, REFIT_ITER_MULT, CB_PARAMS, USE_GPU


def fit_catboost_with_holdout(X_tr, y_tr, w_tr, X_val, y_val, w_val, cat_cols, params, use_gpu=True):
    params = params.copy()
    params.update({
        "loss_function": "Logloss",
        "eval_metric": "PRAUC",
        "random_seed": RANDOM_SEED,
        "allow_writing_files": False,
        "verbose": 200,
        "metric_period": 100,
    })

    if use_gpu:
        params.update({"task_type": "GPU", "devices": "0"})
    else:
        params.update({"task_type": "CPU", "thread_count": max(1, (os.cpu_count() or 4) - 1)})

    tr_pool = Pool(X_tr, y_tr, weight=w_tr, cat_features=cat_cols)
    val_pool = Pool(X_val, y_val, weight=w_val, cat_features=cat_cols)

    def _fit_with_params(_params):
        model = CatBoostClassifier(**_params)
        model.fit(tr_pool, eval_set=val_pool, use_best_model=True)
        return model

    try:
        model = _fit_with_params(params)
    except Exception as e:
        print("CatBoost fallback:", e)
        params.pop("devices", None)
        params["task_type"] = "CPU"
        params["thread_count"] = max(1, (os.cpu_count() or 4) - 1)
        try:
            model = _fit_with_params(params)
        except Exception as e2:
            print("CatBoost PRAUC fallback -> AUC:", e2)
            params["eval_metric"] = "AUC"
            model = _fit_with_params(params)

    val_raw = model.predict(val_pool, prediction_type="RawFormulaVal")
    val_ap = average_precision_score(y_val, val_raw)

    best_iter = model.get_best_iteration()
    if best_iter is None:
        best_iter = int(params.get("iterations", 1000))
    else:
        best_iter = int(best_iter) + 1

    print(f"CB best_iter={best_iter}, val_pr_auc={val_ap:.6f}")
    return model, int(best_iter), float(val_ap), params


def refit_full_catboost(X, y, w, cat_cols, base_params, best_iter):
    params = {k: v for k, v in base_params.items() if k not in ("od_type", "od_wait")}
    params["iterations"] = int(max(100, round(best_iter * REFIT_ITER_MULT)))

    y_arr = np.asarray(y)
    w_arr = np.asarray(w, dtype=np.float32)
    if w_arr.ndim == 0 or w_arr.shape[0] != len(y_arr):
        w_arr = np.full(len(y_arr), float(np.nanmean(w_arr)) if w_arr.size > 0 else 1.0, dtype=np.float32)

    model = CatBoostClassifier(**params)
    model.fit(Pool(X, y_arr, weight=w_arr, cat_features=cat_cols), verbose=200)
    return model


def run_catboost_train(X_main_tr, y_main_tr, w_main_tr, X_main_val, y_main_val, w_main_val, CAT_FEATURE_COLS, train=True):
    if train:
        print("=" * 60)
        print("Training CatBoost MAIN model")
        print("=" * 60)
        model_cb, best_iter_cb, ap_cb, used_cb = fit_catboost_with_holdout(
            X_main_tr, y_main_tr, w_main_tr,
            X_main_val, y_main_val, w_main_val,
            cat_cols=CAT_FEATURE_COLS,
            params=CB_PARAMS,
            use_gpu=USE_GPU,
        )
        model_cb.save_model(str(CACHE_DIR / "cb_main.cbm"))

        val_pool = Pool(X_main_val, y_main_val, cat_features=CAT_FEATURE_COLS)
        pred_cb_val = model_cb.predict(val_pool, prediction_type="RawFormulaVal")
        del model_cb, val_pool; gc.collect()

        cb_meta = {
            "best_iter": best_iter_cb,
            "val_ap": ap_cb,
            "used_params": {k: str(v) for k, v in used_cb.items()},
        }
        with open(CACHE_DIR / "cb_meta.json", "w") as f:
            json.dump(cb_meta, f, indent=2)
        print(f"CatBoost val PR-AUC: {ap_cb:.6f}")

    else:
        print("Loading CatBoost from cache...")
        with open(CACHE_DIR / "cb_meta.json") as f:
            cb_meta = json.load(f)
        best_iter_cb = cb_meta["best_iter"]
        ap_cb = cb_meta["val_ap"]

        model_cb = CatBoostClassifier().load_model(str(CACHE_DIR / "cb_main.cbm"))
        val_pool = Pool(X_main_val, y_main_val, cat_features=CAT_FEATURE_COLS)
        pred_cb_val = model_cb.predict(val_pool, prediction_type="RawFormulaVal")
        used_cb = cb_meta.get("used_params", CB_PARAMS)
        del model_cb, val_pool; gc.collect()
        print(f"CatBoost val PR-AUC: {ap_cb:.6f} (cached)")

    return pred_cb_val, best_iter_cb, ap_cb
