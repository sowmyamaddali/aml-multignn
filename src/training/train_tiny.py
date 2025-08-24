# src/training/train_sanity.py
import argparse, json, random
import numpy as np
import torch
import torch.nn.functional as F
from torch import optim

# Optional metrics & calibration (will auto-disable if sklearn not present)
try:
    from sklearn.metrics import average_precision_score, roc_auc_score
    from sklearn.linear_model import LogisticRegression
    _HAS_SK = True
except Exception:
    _HAS_SK = False

from src.utils.data_loading import get_data
from src.models.layers.graph_data import GINe, GATe, PNA  # RGCN needs relation ids; skipping here.


# Helper Functions

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def stratified_edges(data, max_pos=2000, neg_per_pos=20, shuffle=True):
    """
    Keep up to max_pos positives, and neg_per_pos negatives per positive.
    We use this ONLY for TRAIN to speed up local iteration.
    """
    y = data.y
    pos_idx = torch.where(y == 1)[0]
    neg_idx = torch.where(y == 0)[0]

    if shuffle:
        pos_idx = pos_idx[torch.randperm(pos_idx.numel())]
        neg_idx = neg_idx[torch.randperm(neg_idx.numel())]

    k_pos = min(max_pos, pos_idx.numel())
    sel_pos = pos_idx[:k_pos]
    k_neg = min(neg_idx.numel(), k_pos * neg_per_pos)
    sel_neg = neg_idx[:k_neg]

    idx = torch.cat([sel_pos, sel_neg])
    if shuffle:
        idx = idx[torch.randperm(idx.numel())]

    data.edge_index = data.edge_index[:, idx]
    data.edge_attr  = data.edge_attr[idx]
    data.y          = data.y[idx]
    if getattr(data, "timestamps", None) is not None:
        data.timestamps = data.timestamps[idx]
    return data


def pick_model(name, num_features, edge_dim):
    name = name.lower()
    if name == "gin":
        return GINe(num_features=num_features, edge_dim=edge_dim,
                    num_gnn_layers=3, n_hidden=128, n_classes=2,
                    edge_updates=True, final_dropout=0.5)
    if name == "gat":
        return GATe(num_features=num_features, edge_dim=edge_dim,
                    num_gnn_layers=3, n_hidden=128, n_classes=2,
                    edge_updates=True, final_dropout=0.5)
    if name == "pna":
        return PNA(num_features=num_features, edge_dim=edge_dim,
                   num_gnn_layers=3, n_hidden=128, n_classes=2,
                   edge_updates=True, final_dropout=0.5, deg=None)
    raise ValueError(f"Unknown model: {name}")


def _importance_weight(pi_pop: float, pi_samp: float, cap: float = 50.0) -> float:
    """
    Compute positive-class weight that maps sample prior -> population prior.
    w_pos ~ ((1 - pi_pop)/pi_pop) * (pi_samp/(1 - pi_samp))
    Capped for numerical stability.
    """
    eps = 1e-6
    pi_pop = float(max(min(pi_pop, 1 - eps), eps))
    pi_samp = float(max(min(pi_samp, 1 - eps), eps))
    odds_pop = (1 - pi_pop) / pi_pop
    odds_samp = pi_samp / (1 - pi_samp)
    w_pos = odds_pop * odds_samp
    return float(min(max(w_pos, 0.0), cap))


def run_one(model, data, optimizer=None, train=False, device="cpu"):
    """
    One pass. Uses importance weighting against the population prior (model._pi_pop),
    so training on a stratified sample matches the target distribution.
    """
    model.train(train)
    x  = data.x.to(device)
    ei = data.edge_index.to(device)
    ea = data.edge_attr.to(device)
    y  = data.y.to(device)

    logits = model(x, ei, ea)  # [E, 2]

    # Importance-weighted CE so stratified sampling matches population prior
    assert hasattr(model, "_pi_pop") and model._pi_pop is not None, \
        "model._pi_pop must be set from the population (validation) split."
    pi_samp = y.float().mean().item() if y.numel() else 0.5
    w_pos = _importance_weight(model._pi_pop, pi_samp, cap=50.0)
    weights = torch.tensor([1.0, w_pos], dtype=torch.float32, device=device)
    loss = F.cross_entropy(logits, y, weight=weights)

    with torch.no_grad():
        preds = logits.argmax(1)
        acc = (preds == y).float().mean().item()
        tp = int(((preds == 1) & (y == 1)).sum().item())
        fp = int(((preds == 1) & (y == 0)).sum().item())
        fn = int(((preds == 0) & (y == 1)).sum().item())
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall    = tp / (tp + fn) if (tp + fn) else 0.0

    if train:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Return loss/acc + counts + P/R
    return loss.item(), acc, tp, fp, fn, precision, recall


def metrics_from_probs(probs, y, thr):
    preds = (probs >= thr).long()
    acc = (preds == y).float().mean().item()
    tp = int(((preds == 1) & (y == 1)).sum().item())
    fp = int(((preds == 1) & (y == 0)).sum().item())
    fn = int(((preds == 0) & (y == 1)).sum().item())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall    = tp / (tp + fn) if (tp + fn) else 0.0
    f1        = 2*precision*recall/(precision+recall) if (precision+recall) else 0.0
    return acc, tp, fp, fn, precision, recall, f1


@torch.no_grad()
def forward_probs(model, data, device="cpu"):
    """Single-shot (non-batched) forward to probabilities."""
    model.eval()
    x  = data.x.to(device)
    ei = data.edge_index.to(device)
    ea = data.edge_attr.to(device)
    y  = data.y.to(device)
    logits = model(x, ei, ea)
    probs = torch.softmax(logits, dim=1)[:, 1]  # P(y=1)
    return probs.cpu(), y.cpu()


@torch.no_grad()
def forward_probs_batched(model, data, device="cpu", batch_size=300_000):
    """Batched forward for very large edge sets to avoid OOM on MPS/CPU."""
    model.eval()
    E = data.edge_index.shape[1]
    probs_list = []
    for start in range(0, E, batch_size):
        end = min(E, start + batch_size)
        sl  = slice(start, end)
        x   = data.x.to(device)
        ei  = data.edge_index[:, sl].to(device)
        ea  = data.edge_attr[sl].to(device)
        logits = model(x, ei, ea)
        probs_list.append(torch.softmax(logits, dim=1)[:, 1].cpu())
        # free memory
        del ei, ea, logits
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    probs = torch.cat(probs_list, dim=0)
    y = data.y.cpu()
    return probs, y


def fit_platt(probs_np, y_np):
    """
    Platt scaling on validation (population) to calibrate probabilities.
    Returns a callable that maps probs->calibrated_probs.
    Auto-noops if sklearn isn't present.
    """
    if not _HAS_SK:
        return lambda p: p  # identity
    p = np.clip(probs_np, 1e-6, 1-1e-6).reshape(-1, 1)
    logit = np.log(p / (1 - p))
    lr = LogisticRegression(solver="lbfgs")
    lr.fit(logit, y_np.astype(int))
    def apply(p_new: np.ndarray):
        p_new = np.clip(p_new, 1e-6, 1-1e-6).reshape(-1, 1)
        logit_new = np.log(p_new / (1 - p_new))
        return lr.predict_proba(logit_new)[:, 1]
    return apply


# Main Function

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Base name, e.g. HI-Small_Trans")
    ap.add_argument("--model", default="gin", choices=["gin", "gat", "pna"])
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--config", default="src/configs/data_config.json")
    ap.add_argument("--ports", action="store_true")
    ap.add_argument("--tds", action="store_true")
    ap.add_argument("--reverse_mp", action="store_true")
    # stratified TRAIN knobs (val/test remain population)
    ap.add_argument("--max_pos_tr", type=int, default=2000)
    ap.add_argument("--neg_per_pos", type=int, default=20)
    # eval / misc
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--eval_batch", type=int, default=300000, help="Edges per batch during eval forward")
    args = ap.parse_args()

    set_seed(args.seed)

    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    with open(args.config) as f:
        data_config = json.load(f)

    # Load FULL splits (population distribution)
    tr_full, val_full, te_full, *_ = get_data(args, data_config)

    # Stratify ONLY TRAIN to make local runs feasible
    tr = stratified_edges(tr_full, max_pos=args.max_pos_tr, neg_per_pos=args.neg_per_pos)
    val = val_full  # population prior
    te  = te_full   # population prior

    num_features = tr.x.size(1)
    edge_dim     = tr.edge_attr.size(1)
    print(f"num_features={num_features}, edge_dim={edge_dim}")

    model = pick_model(args.model, num_features, edge_dim).to(device)

    # Set population prior for importance-weighted loss from VALIDATION (population)
    pi_pop = float(val.y.float().mean().item())
    model._pi_pop = pi_pop
    print(f"Estimated population positive rate (from val): {pi_pop:.8f}")

    opt = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    best_thr = 0.50  # will update from population validation

    # --------- Training loop ---------
    for ep in range(1, args.epochs + 1):
        tr_loss, tr_acc, tr_tp, tr_fp, tr_fn, tr_p, tr_r = run_one(model, tr, opt,  train=True,  device=device)

        # quick snapshot on population val at argmax=0.5 (not tuned)
        val_loss, val_acc, val_tp, val_fp, val_fn, val_p, val_r = run_one(model, val, None, train=False, device=device)

        print(
            f"[{ep:02d}] "
            f"train loss={tr_loss:.4f} acc={tr_acc:.3f} P={tr_p:.3f} R={tr_r:.3f} TP/FP/FN={tr_tp}/{tr_fp}/{tr_fn} | "
            f"val(loss@0.5)={val_loss:.4f} acc={val_acc:.3f} P={val_p:.3f} R={val_r:.3f} TP/FP/FN={val_tp}/{val_fp}/{val_fn}"
        )

        # Tune threshold on population validation (maximize F1)
        val_probs, val_y = forward_probs_batched(model, val, device, batch_size=args.eval_batch)
        candidates = [round(t, 3) for t in (0.50, 0.48, 0.46, 0.44, 0.42, 0.40, 0.38, 0.36, 0.34, 0.32, 0.30, 0.28, 0.26, 0.24, 0.22, 0.20)]
        best = {"thr": best_thr, "f1": -1}
        for thr in candidates:
            acc_b, tp_b, fp_b, fn_b, p_b, r_b, f1_b = metrics_from_probs(val_probs, val_y, thr)
            if f1_b > best["f1"]:
                best = {"thr": thr, "acc": acc_b, "tp": tp_b, "fp": fp_b, "fn": fn_b,
                        "p": p_b, "r": r_b, "f1": f1_b}
        best_thr = best["thr"]

        # AUC-PR/AUROC on population val (if sklearn available)
        if _HAS_SK:
            val_aucpr = average_precision_score(val_y.numpy(), val_probs.numpy())
            val_auroc = roc_auc_score(val_y.numpy(), val_probs.numpy())
            extra = f" | AUC-PR={val_aucpr:.3f} AUROC={val_auroc:.3f}"
        else:
            extra = ""

        print(
            f"      best@thr={best['thr']:.2f} "
            f"P={best['p']:.3f} R={best['r']:.3f} F1={best['f1']:.3f} "
            f"TP/FP/FN={best['tp']}/{best['fp']}/{best['fn']}{extra}"
        )

    # Calibration on population validation
    # Fit Platt scaling to calibrate probabilities (no-op if sklearn not present)
    val_probs_np = val_probs.numpy()
    val_y_np     = val_y.numpy().astype(int)
    apply_cal = fit_platt(val_probs_np, val_y_np)

    # Final test on population distribution with tuned threshold
    te_probs, te_y = forward_probs_batched(model, te, device, batch_size=args.eval_batch)
    te_probs_cal = torch.from_numpy(apply_cal(te_probs.numpy())).float()
    acc, tp, fp, fn, p, r, f1 = metrics_from_probs(te_probs_cal, te_y, best_thr)

    # Optional AUCs on test
    if _HAS_SK:
        te_aucpr = average_precision_score(te_y.numpy(), te_probs_cal.numpy())
        te_auroc = roc_auc_score(te_y.numpy(), te_probs_cal.numpy())
        auc_line = f" | AUC-PR={te_aucpr:.3f} AUROC={te_auroc:.3f}"
    else:
        auc_line = ""

    print(f"[TEST-Cal @thr={best_thr:.2f}] acc={acc:.3f} P={p:.3f} R={r:.3f} F1={f1:.3f} TP/FP/FN={tp}/{fp}/{fn}{auc_line}")


if __name__ == "__main__":
    main()