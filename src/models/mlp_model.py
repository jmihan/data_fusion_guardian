import gc
import json

import numpy as np
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import joblib

from src.config import (
    CACHE_DIR, USE_GPU,
    MLP_HIDDEN_DIMS, MLP_DROPOUT, MLP_LR, MLP_WEIGHT_DECAY,
    MLP_BATCH_SIZE, MLP_MAX_EPOCHS, MLP_PATIENCE,
)


class SimpleMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, dropout=0.2):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def _train_mlp(
    X_tr_np, y_tr_np, w_tr_np, X_val_np, y_val_np,
    hidden_dims, dropout, lr, weight_decay, batch_size, max_epochs, patience, device,
):
    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr_np).astype(np.float32)
    X_val_scaled = scaler.transform(X_val_np).astype(np.float32)

    X_tr_scaled = np.nan_to_num(X_tr_scaled, nan=0.0, posinf=0.0, neginf=0.0)
    X_val_scaled = np.nan_to_num(X_val_scaled, nan=0.0, posinf=0.0, neginf=0.0)

    t_X_tr = torch.from_numpy(X_tr_scaled)
    t_y_tr = torch.from_numpy(y_tr_np.astype(np.float32))
    t_w_tr = torch.from_numpy(w_tr_np.astype(np.float32))
    t_X_val = torch.from_numpy(X_val_scaled)

    train_ds = TensorDataset(t_X_tr, t_y_tr, t_w_tr)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              pin_memory=(device.type == "cuda"), drop_last=False)

    model = SimpleMLP(X_tr_scaled.shape[1], hidden_dims, dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, steps_per_epoch=len(train_loader), epochs=max_epochs,
    )
    grad_scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    n_pos = float(y_tr_np.sum())
    n_neg = float(len(y_tr_np) - n_pos)
    pos_weight = torch.tensor([min(n_neg / max(n_pos, 1.0), 500.0)], device=device)

    best_ap = -1.0
    best_epoch = 0
    best_state = None
    wait = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for batch_X, batch_y, batch_w in train_loader:
            batch_X = batch_X.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            batch_w = batch_w.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                logits = model(batch_X)
                loss_per_sample = F.binary_cross_entropy_with_logits(
                    logits, batch_y, pos_weight=pos_weight, reduction="none",
                )
                loss = (loss_per_sample * batch_w).mean()

            grad_scaler.scale(loss).backward()
            grad_scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            grad_scaler.step(optimizer)
            grad_scaler.update()
            scheduler.step()

            epoch_loss += loss.item()
            n_batches += 1

        model.eval()
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            val_logits = model(t_X_val.to(device))
            val_probs = torch.sigmoid(val_logits).cpu().numpy()

        val_ap = average_precision_score(y_val_np, val_probs)
        avg_loss = epoch_loss / max(n_batches, 1)
        print(f"  Epoch {epoch}/{max_epochs} | loss={avg_loss:.4f} | val PR-AUC={val_ap:.6f}")

        if val_ap > best_ap:
            best_ap = val_ap
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"  Early stopping at epoch {epoch} (patience={patience})")
                break

        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad(), torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
        val_logits = model(t_X_val.to(device))
        val_preds = torch.sigmoid(val_logits).cpu().numpy()

    del model, optimizer, grad_scaler, scheduler
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return val_preds, best_ap, best_epoch, best_state, scaler


def run_mlp_train(X_main_tr, y_main_tr, w_main_tr, X_main_val, y_main_val, train=True):
    if train:
        print("=" * 60)
        print("Training Simple MLP...")
        print("=" * 60)

        mlp_device = torch.device("cuda" if torch.cuda.is_available() and USE_GPU else "cpu")
        print(f"MLP device: {mlp_device}")

        pred_mlp_val, ap_mlp, best_epoch_mlp, mlp_best_state, mlp_scaler = _train_mlp(
            X_main_tr.values, y_main_tr, w_main_tr,
            X_main_val.values, y_main_val,
            MLP_HIDDEN_DIMS, MLP_DROPOUT, MLP_LR, MLP_WEIGHT_DECAY,
            MLP_BATCH_SIZE, MLP_MAX_EPOCHS, MLP_PATIENCE, mlp_device,
        )

        torch.save({
            "state_dict": mlp_best_state,
            "input_dim": X_main_tr.shape[1],
            "hidden_dims": MLP_HIDDEN_DIMS,
            "dropout": MLP_DROPOUT,
            "best_epoch": best_epoch_mlp,
            "best_ap": ap_mlp,
        }, CACHE_DIR / "mlp_val.pt")
        joblib.dump(mlp_scaler, CACHE_DIR / "mlp_scaler.joblib")

        with open(CACHE_DIR / "mlp_meta.json", "w") as f:
            json.dump({"best_epoch": best_epoch_mlp, "val_ap": float(ap_mlp)}, f, indent=2)

        del mlp_best_state; gc.collect()
        print(f"MLP best epoch: {best_epoch_mlp}, val PR-AUC: {ap_mlp:.6f}")

    else:
        print("Loading MLP from cache...")
        with open(CACHE_DIR / "mlp_meta.json") as f:
            mlp_meta = json.load(f)
        best_epoch_mlp = mlp_meta["best_epoch"]
        ap_mlp = mlp_meta["val_ap"]

        mlp_device = torch.device("cuda" if torch.cuda.is_available() and USE_GPU else "cpu")
        mlp_scaler = joblib.load(CACHE_DIR / "mlp_scaler.joblib")
        ckpt = torch.load(CACHE_DIR / "mlp_val.pt", map_location=mlp_device, weights_only=True)
        mlp_model_cached = SimpleMLP(ckpt["input_dim"], ckpt["hidden_dims"], ckpt["dropout"]).to(mlp_device)
        mlp_model_cached.load_state_dict(ckpt["state_dict"])
        mlp_model_cached.eval()

        X_val_scaled = mlp_scaler.transform(X_main_val.values).astype(np.float32)
        X_val_scaled = np.nan_to_num(X_val_scaled, nan=0.0, posinf=0.0, neginf=0.0)
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=(mlp_device.type == "cuda")):
            pred_mlp_val = torch.sigmoid(
                mlp_model_cached(torch.from_numpy(X_val_scaled).to(mlp_device))
            ).cpu().numpy()

        del mlp_model_cached, ckpt; gc.collect()
        print(f"MLP val PR-AUC: {ap_mlp:.6f} (cached)")

    return pred_mlp_val, best_epoch_mlp, ap_mlp
