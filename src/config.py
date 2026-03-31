from pathlib import Path
import gc
import json
import os
import warnings
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import polars as pl
from sklearn.metrics import average_precision_score
from scipy.stats import rankdata
from scipy.optimize import minimize, differential_evolution
from scipy.special import softmax

pl.Config.set_tbl_rows(12)
pl.Config.set_tbl_cols(200)
warnings.filterwarnings("ignore")

# ── Directories ──────────────────────────────────────────────────────────────
DATA_DIR = Path("data/raw")
CACHE_DIR = Path("cache/meta3b1n")
SUBMISSION_DIR = Path("submission_meta")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)

# ── Sampling ─────────────────────────────────────────────────────────────────
NEG_SAMPLE_MOD_RECENT = 10
NEG_SAMPLE_MOD_OLD = 30
NEG_SAMPLE_BORDER_STR = "2025-04-01 00:00:00"

# ── Temporal splits ──────────────────────────────────────────────────────────
VAL_START = pd.Timestamp("2025-05-01")
RECENCY_WEIGHT_START = pd.Timestamp("2025-02-01")

# ── Global settings ──────────────────────────────────────────────────────────
REFIT_ITER_MULT = 1.10
RANDOM_SEED = 67

# ── Feature caching ─────────────────────────────────────────────────────────
FEATURE_CACHE_TAG = "v11_robust_features"
GLOBAL_CACHE_TAG = "v3_honest_eval"
FORCE_REBUILD_FEATURES = False
FORCE_REBUILD_PRIORS = False

# ── GPU ──────────────────────────────────────────────────────────────────────
USE_GPU = True

# ── Train flags ──────────────────────────────────────────────────────────────
TRAIN_CB = True
TRAIN_LGB = True
TRAIN_XGB = True
TRAIN_ET = True
TRAIN_MLP = True

RETRAIN_ON_FULL_CB = True
RETRAIN_ON_FULL_LGB = True
RETRAIN_ON_FULL_XGB = True
RETRAIN_ON_FULL_ET = True
RETRAIN_ON_FULL_MLP = True

# ── Blend ────────────────────────────────────────────────────────────────────
BLEND_METHOD = "nelder-mead"  # "nelder-mead", "grid-search", "diff-evolution"

# ── Feature chunk size ───────────────────────────────────────────────────────
FEATURE_CHUNK_SIZE = 3000  # customers per chunk

# ── CatBoost ─────────────────────────────────────────────────────────────────
CB_PARAMS = {
    "iterations": 6000,
    "learning_rate": 0.04,
    "depth": 8,
    "l2_leaf_reg": 10.0,
    "random_strength": 0.5,
    "od_type": "Iter",
    "od_wait": 300,
}

# ── LightGBM ─────────────────────────────────────────────────────────────────
LGB_PARAMS = {
    "n_estimators": 6000,
    "early_stopping_rounds": 300,
    "learning_rate": 0.03,
    "num_leaves": 255,
    "max_depth": -1,
    "min_child_samples": 80,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.75,
    "reg_alpha": 0.2,
    "reg_lambda": 8.0,
    "min_split_gain": 0.0,
    "max_bin": 255,
    "device": "cpu",
}

# ── XGBoost ──────────────────────────────────────────────────────────────────
XGB_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "aucpr",
    "learning_rate": 0.04,
    "max_depth": 8,
    "min_child_weight": 1,
    "subsample": 1.0,
    "colsample_bytree": 1.0,
    "reg_alpha": 0.0,
    "reg_lambda": 10.0,
    "seed": RANDOM_SEED,
    "verbosity": 1,
    "tree_method": "hist",
    "device": "cuda",
    "max_bin": 255,
}

# ── ExtraTrees ───────────────────────────────────────────────────────────────
ET_PARAMS = {
    "n_estimators": 200,
    "max_depth": 16,
    "min_samples_leaf": 50,
    "max_features": "sqrt",
    "n_jobs": -1,
    "random_state": RANDOM_SEED,
    "class_weight": "balanced_subsample",
    "warm_start": False,
}

# ── MLP ──────────────────────────────────────────────────────────────────────
MLP_HIDDEN_DIMS = [256, 128]
MLP_DROPOUT = 0.2
MLP_LR = 1e-3
MLP_WEIGHT_DECAY = 1e-4
MLP_BATCH_SIZE = 4096
MLP_MAX_EPOCHS = 20
MLP_PATIENCE = 5

# ── Mega-ensemble ────────────────────────────────────────────────────────────
SOL154_GRID = [0.55, 0.65, 0.75, 0.85, 0.90, 0.95]

# ── GPU detection ────────────────────────────────────────────────────────────
try:
    import torch
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_properties(0)
        vram_gb = gpu.total_memory / 1024**3
        print(f"GPU: {gpu.name}, VRAM: {vram_gb:.1f} GB")
    else:
        print("CUDA not available, will use CPU")
        USE_GPU = False
except ImportError:
    print("torch not installed, GPU check skipped")

np.random.seed(RANDOM_SEED)

print("DATA_DIR:", DATA_DIR.resolve())
print("CACHE_DIR:", CACHE_DIR.resolve())
