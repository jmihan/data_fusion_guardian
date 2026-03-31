import gc
import json
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostClassifier, Pool
import lightgbm as lgb
import xgboost as xgb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import joblib

from src.config import (
    DATA_DIR, CACHE_DIR, SUBMISSION_DIR, BLEND_METHOD,
    CB_PARAMS, LGB_PARAMS, XGB_PARAMS, ET_PARAMS,
    USE_GPU, RANDOM_SEED, REFIT_ITER_MULT, FORCE_REBUILD_PRIORS,
    MLP_HIDDEN_DIMS, MLP_DROPOUT, MLP_LR, MLP_WEIGHT_DECAY, MLP_BATCH_SIZE,
    RETRAIN_ON_FULL_CB, RETRAIN_ON_FULL_LGB, RETRAIN_ON_FULL_XGB,
    RETRAIN_ON_FULL_ET, RETRAIN_ON_FULL_MLP,
)
from src.features.columns import CAT_COLS
from src.features.priors import (
    DEVICE_STAT_COLS, build_prior_tables, build_device_stats_table,
    attach_global_features,
)
from src.features.preparation import prepare_feature_matrices
from src.utils import (
    rank_norm, optimize_blend_weights, dedupe, make_weights,
    downcast_pandas, lgb_prepare,
)
from src.models.catboost_model import refit_full_catboost
from src.models.lightgbm_model import refit_full_lgb
from src.models.xgboost_model import refit_full_xgb
from src.models.mlp_model import SimpleMLP


BLEND_COMBOS = [
    ("cb_lgb", ["catboost", "lightgbm"]),
    ("cb_xgb", ["catboost", "xgboost"]),
    ("lgb_xgb", ["lightgbm", "xgboost"]),
    ("3boost", ["catboost", "lightgbm", "xgboost"]),
    ("3boost_et", ["catboost", "lightgbm", "xgboost", "extratrees"]),
    ("3boost_mlp", ["catboost", "lightgbm", "xgboost", "mlp"]),
    ("all5", ["catboost", "lightgbm", "xgboost", "extratrees", "mlp"]),
    ("et_mlp", ["extratrees", "mlp"]),
]


def run_blend_optimization(pred_cb_val, pred_lgb_val, pred_xgb_val, pred_et_val, pred_mlp_val, y_main_val):
    all_val_heads = {
        "catboost": rank_norm(pred_cb_val),
        "lightgbm": rank_norm(pred_lgb_val),
        "xgboost": rank_norm(pred_xgb_val),
        "extratrees": rank_norm(pred_et_val),
        "mlp": rank_norm(pred_mlp_val),
    }

    blend_results = {}
    for combo_name, combo_keys in BLEND_COMBOS:
        heads_sub = {k: all_val_heads[k] for k in combo_keys}
        b_keys, b_w, b_ap = optimize_blend_weights(heads_sub, y_main_val, method=BLEND_METHOD)
        blend_results[combo_name] = {"keys": b_keys, "weights": b_w, "val_ap": b_ap}
        w_str = ", ".join(f"{b_keys[i]}={b_w[i]:.4f}" for i in range(len(b_keys)))
        print(f"  {combo_name:15s}  PR-AUC={b_ap:.6f}  [{w_str}]")

    best_combo_name = max(blend_results, key=lambda k: blend_results[k]["val_ap"])
    best_blend = blend_results[best_combo_name]

    all_blend_config = {}
    for combo_name, res in blend_results.items():
        all_blend_config[combo_name] = {
            "method": BLEND_METHOD,
            "keys": res["keys"],
            "weights": [float(w) for w in res["weights"]],
            "val_ap": float(res["val_ap"]),
        }
    with open(CACHE_DIR / "blend_results.json", "w") as f:
        json.dump(all_blend_config, f, indent=2)

    print(f"\nBest blend: {best_combo_name} -> PR-AUC = {best_blend['val_ap']:.6f}")
    return blend_results, best_combo_name


def run_full_refit_and_submission(
    train_base_df, test_base_df, FEATURE_COLS, CAT_FEATURE_COLS,
    best_iter_cb, best_iter_lgb, best_iter_xgb, best_epoch_mlp,
    blend_results, best_combo_name,
):
    print("=" * 60)
    print("Building causal globals for full fit / test...")
    print("=" * 60)

    device_stats_full = build_device_stats_table(
        tag="full_history", cutoff=None, include_pretest=True, force=FORCE_REBUILD_PRIORS,
    )
    prior_tables_full, prior_feature_cols_full = build_prior_tables(
        tag="full_train", cutoff=None, force=FORCE_REBUILD_PRIORS,
    )

    train_full_df = attach_global_features(train_base_df, device_stats_full, prior_tables_full)
    test_full_df = attach_global_features(test_base_df, device_stats_full, prior_tables_full)
    del device_stats_full, prior_tables_full; gc.collect()

    FULL_FEATURE_COLS = [c for c in FEATURE_COLS if c in train_full_df.columns and c in test_full_df.columns]
    FULL_CAT_FEATURE_COLS = [c for c in CAT_COLS if c in FULL_FEATURE_COLS]

    X_main_full, X_test_full = prepare_feature_matrices(
        train_full_df, [test_full_df], FULL_FEATURE_COLS, FULL_CAT_FEATURE_COLS,
    )

    y_main_full = train_full_df["target_bin"].astype(np.int8).values
    w_main_full = make_weights(train_full_df["train_target_raw"].values, train_full_df["event_ts"].values)

    downcast_pandas(X_main_full, set(FULL_CAT_FEATURE_COLS))
    downcast_pandas(X_test_full, set(FULL_CAT_FEATURE_COLS))

    print(f"Full features: {len(FULL_FEATURE_COLS)} | X_train: {X_main_full.shape} | X_test: {X_test_full.shape}")

    # ── CatBoost refit ───────────────────────────────────────────────────
    if RETRAIN_ON_FULL_CB:
        print("Refit CatBoost on full train...")
        model_cb_full = refit_full_catboost(
            X_main_full, y_main_full, w_main_full,
            FULL_CAT_FEATURE_COLS, CB_PARAMS, best_iter_cb,
        )
        model_cb_full.save_model(str(CACHE_DIR / "cb_main_full.cbm"))
    else:
        model_cb_full = CatBoostClassifier().load_model(str(CACHE_DIR / "cb_main.cbm"))

    test_pool = Pool(X_test_full, cat_features=FULL_CAT_FEATURE_COLS)
    pred_cb_test = model_cb_full.predict(test_pool, prediction_type="RawFormulaVal")
    del model_cb_full, test_pool; gc.collect()

    # ── LightGBM refit ───────────────────────────────────────────────────
    if RETRAIN_ON_FULL_LGB:
        print("Refit LightGBM on full train...")
        model_lgb_full = refit_full_lgb(
            X_main_full, y_main_full, w_main_full,
            FULL_CAT_FEATURE_COLS, LGB_PARAMS, best_iter_lgb,
        )
        model_lgb_full.save_model(str(CACHE_DIR / "lgb_main_full.txt"))
    else:
        model_lgb_full = lgb.Booster(model_file=str(CACHE_DIR / "lgb_main.txt"))

    pred_lgb_test = model_lgb_full.predict(lgb_prepare(X_test_full, FULL_CAT_FEATURE_COLS))
    del model_lgb_full; gc.collect()

    # ── XGBoost refit ────────────────────────────────────────────────────
    if RETRAIN_ON_FULL_XGB:
        print("Refit XGBoost on full train...")
        model_xgb_full = refit_full_xgb(
            X_main_full, y_main_full, w_main_full,
            XGB_PARAMS, best_iter_xgb,
        )
        model_xgb_full.save_model(str(CACHE_DIR / "xgb_main_full.json"))
    else:
        model_xgb_full = xgb.Booster()
        model_xgb_full.load_model(str(CACHE_DIR / "xgb_main.json"))

    dtest = xgb.DMatrix(X_test_full)
    pred_xgb_test = model_xgb_full.predict(dtest)
    del model_xgb_full, dtest; gc.collect()

    print("Booster test predictions ready.")

    # ── ExtraTrees refit ─────────────────────────────────────────────────
    if RETRAIN_ON_FULL_ET:
        print("Refit ExtraTrees on full train...")
        model_et_full = ExtraTreesClassifier(**ET_PARAMS)
        model_et_full.fit(X_main_full.values, y_main_full, sample_weight=w_main_full)
        joblib.dump(model_et_full, CACHE_DIR / "et_main_full.joblib", compress=3)
    else:
        model_et_full = joblib.load(CACHE_DIR / "et_main.joblib")

    pred_et_test = model_et_full.predict_proba(X_test_full.values)[:, 1]
    del model_et_full; gc.collect()

    # ── MLP refit ────────────────────────────────────────────────────────
    mlp_device = torch.device("cuda" if torch.cuda.is_available() and USE_GPU else "cpu")

    if RETRAIN_ON_FULL_MLP:
        print("Refit MLP on full train...")
        mlp_scaler_full = StandardScaler()
        X_full_scaled = mlp_scaler_full.fit_transform(X_main_full.values).astype(np.float32)
        X_full_scaled = np.nan_to_num(X_full_scaled, nan=0.0, posinf=0.0, neginf=0.0)
        X_test_scaled = mlp_scaler_full.transform(X_test_full.values).astype(np.float32)
        X_test_scaled = np.nan_to_num(X_test_scaled, nan=0.0, posinf=0.0, neginf=0.0)

        t_X_full = torch.from_numpy(X_full_scaled)
        t_y_full = torch.from_numpy(y_main_full.astype(np.float32))
        t_w_full = torch.from_numpy(w_main_full.astype(np.float32))

        full_ds = TensorDataset(t_X_full, t_y_full, t_w_full)
        full_loader = DataLoader(full_ds, batch_size=MLP_BATCH_SIZE, shuffle=True,
                                 pin_memory=(mlp_device.type == "cuda"), drop_last=False)

        mlp_model_full = SimpleMLP(X_full_scaled.shape[1], MLP_HIDDEN_DIMS, MLP_DROPOUT).to(mlp_device)
        mlp_opt_full = torch.optim.AdamW(mlp_model_full.parameters(), lr=MLP_LR, weight_decay=MLP_WEIGHT_DECAY)
        mlp_sched_full = torch.optim.lr_scheduler.OneCycleLR(
            mlp_opt_full, max_lr=MLP_LR, steps_per_epoch=len(full_loader), epochs=max(1, best_epoch_mlp),
        )
        mlp_gs_full = torch.amp.GradScaler("cuda", enabled=(mlp_device.type == "cuda"))

        n_pos = float(y_main_full.sum())
        n_neg = float(len(y_main_full) - n_pos)
        pw_mlp = torch.tensor([min(n_neg / max(n_pos, 1.0), 500.0)], device=mlp_device)

        n_refit_epochs = max(1, best_epoch_mlp)
        for epoch in range(1, n_refit_epochs + 1):
            mlp_model_full.train()
            epoch_loss = 0.0
            n_b = 0
            for bX, by, bw in full_loader:
                bX = bX.to(mlp_device, non_blocking=True)
                by = by.to(mlp_device, non_blocking=True)
                bw = bw.to(mlp_device, non_blocking=True)
                mlp_opt_full.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=(mlp_device.type == "cuda")):
                    logits = mlp_model_full(bX)
                    loss = (F.binary_cross_entropy_with_logits(
                        logits, by, pos_weight=pw_mlp, reduction="none") * bw).mean()
                mlp_gs_full.scale(loss).backward()
                mlp_gs_full.unscale_(mlp_opt_full)
                torch.nn.utils.clip_grad_norm_(mlp_model_full.parameters(), 1.0)
                mlp_gs_full.step(mlp_opt_full)
                mlp_gs_full.update()
                mlp_sched_full.step()
                epoch_loss += loss.item()
                n_b += 1
            print(f"  MLP refit epoch {epoch}/{n_refit_epochs} | loss={epoch_loss / max(n_b, 1):.4f}")
            gc.collect()
            if mlp_device.type == "cuda":
                torch.cuda.empty_cache()

        mlp_model_full.eval()
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=(mlp_device.type == "cuda")):
            pred_mlp_test = torch.sigmoid(
                mlp_model_full(torch.from_numpy(X_test_scaled).to(mlp_device))
            ).cpu().numpy()

        torch.save({"state_dict": mlp_model_full.state_dict()}, CACHE_DIR / "mlp_full.pt")
        joblib.dump(mlp_scaler_full, CACHE_DIR / "mlp_scaler_full.joblib")
        del mlp_model_full, mlp_opt_full, mlp_gs_full, mlp_sched_full; gc.collect()
        if mlp_device.type == "cuda":
            torch.cuda.empty_cache()

    else:
        mlp_scaler_test = joblib.load(CACHE_DIR / "mlp_scaler.joblib")
        ckpt = torch.load(CACHE_DIR / "mlp_val.pt", map_location=mlp_device, weights_only=True)
        mlp_model_test = SimpleMLP(ckpt["input_dim"], ckpt["hidden_dims"], ckpt["dropout"]).to(mlp_device)
        mlp_model_test.load_state_dict(ckpt["state_dict"])
        mlp_model_test.eval()

        X_test_scaled = mlp_scaler_test.transform(X_test_full.values).astype(np.float32)
        X_test_scaled = np.nan_to_num(X_test_scaled, nan=0.0, posinf=0.0, neginf=0.0)
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=(mlp_device.type == "cuda")):
            pred_mlp_test = torch.sigmoid(
                mlp_model_test(torch.from_numpy(X_test_scaled).to(mlp_device))
            ).cpu().numpy()

        del mlp_model_test, ckpt; gc.collect()

    print("ET + MLP test predictions ready.")

    # ── Submissions ──────────────────────────────────────────────────────
    sample_submit = pd.read_csv(DATA_DIR / "sample_submit.csv")
    test_event_ids = test_full_df["event_id"].values

    def make_submission(pred, name, event_ids=test_event_ids):
        pred_df = pd.DataFrame({"event_id": event_ids, "predict": pred})
        sub = sample_submit[["event_id"]].merge(pred_df, on="event_id", how="left")
        missing = int(sub["predict"].isna().sum())
        if missing > 0:
            print(f"WARNING: {name} has {missing} missing predictions, filling with median")
            sub["predict"] = sub["predict"].fillna(sub["predict"].median())
        sub = sub.drop_duplicates(subset="event_id", keep="first")
        out_path = SUBMISSION_DIR / f"submission_{name}.csv"
        sub.to_csv(out_path, index=False)
        print(f"Saved {out_path} ({len(sub):,} rows)")
        return sub

    make_submission(pred_cb_test, "cb")
    make_submission(pred_lgb_test, "lgb")
    make_submission(pred_xgb_test, "xgb")
    make_submission(pred_et_test, "et")
    make_submission(pred_mlp_test, "mlp")

    all_test_heads = {
        "catboost": rank_norm(pred_cb_test),
        "lightgbm": rank_norm(pred_lgb_test),
        "xgboost": rank_norm(pred_xgb_test),
        "extratrees": rank_norm(pred_et_test),
        "mlp": rank_norm(pred_mlp_test),
    }

    for combo_name, res in blend_results.items():
        bk, bw = res["keys"], res["weights"]
        blend_pred = sum(bw[i] * all_test_heads[bk[i]] for i in range(len(bk)))
        make_submission(blend_pred, f"blend_{combo_name}")

    print("\nAll submissions generated!")
    return FULL_FEATURE_COLS, FULL_CAT_FEATURE_COLS
