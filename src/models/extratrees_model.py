import gc
import json

import numpy as np
from sklearn.metrics import average_precision_score
from sklearn.ensemble import ExtraTreesClassifier
import joblib

from src.config import CACHE_DIR, ET_PARAMS


def run_extratrees_train(X_main_tr, y_main_tr, w_main_tr, X_main_val, y_main_val, train=True):
    if train:
        print("=" * 60)
        print("Training ExtraTrees...")
        print("=" * 60)

        model_et = ExtraTreesClassifier(**ET_PARAMS)
        model_et.fit(X_main_tr.values, y_main_tr, sample_weight=w_main_tr)

        pred_et_val = model_et.predict_proba(X_main_val.values)[:, 1]
        ap_et = average_precision_score(y_main_val, pred_et_val)
        best_iter_et = ET_PARAMS["n_estimators"]

        joblib.dump(model_et, CACHE_DIR / "et_main.joblib", compress=3)
        with open(CACHE_DIR / "et_meta.json", "w") as f:
            json.dump({"n_estimators": best_iter_et, "val_ap": float(ap_et)}, f, indent=2)

        del model_et; gc.collect()
        print(f"ExtraTrees val PR-AUC: {ap_et:.6f}")

    else:
        print("Loading ExtraTrees from cache...")
        with open(CACHE_DIR / "et_meta.json") as f:
            et_meta = json.load(f)
        ap_et = et_meta["val_ap"]
        best_iter_et = et_meta["n_estimators"]

        model_et = joblib.load(CACHE_DIR / "et_main.joblib")
        pred_et_val = model_et.predict_proba(X_main_val.values)[:, 1]
        del model_et; gc.collect()
        print(f"ExtraTrees val PR-AUC: {ap_et:.6f} (cached)")

    return pred_et_val, best_iter_et, ap_et
