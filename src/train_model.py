import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from src.ds import Track3CachedDataset, collate_fn
from src.model import TriStreamModel

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)

D_FACE = 1280

sample_ds = Track3CachedDataset(train_ok.iloc[:1], feat_dir=FEAT_DIR, phase=1, has_label=True)
sample = sample_ds[0]
d_skel_in = sample["skel"].shape[1]
print("d_skel_in:", d_skel_in)

class FocalLoss(nn.Module):
    def __init__(self, gamma=1.0, pos_weight=None):
        super().__init__()
        self.gamma = gamma
        self.pos_weight = pos_weight
    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none", pos_weight=self.pos_weight)
        p = torch.sigmoid(logits)
        pt = torch.where(targets == 1, p, 1 - p).clamp(1e-6, 1-1e-6)
        return (((1 - pt) ** self.gamma) * bce).mean()

def eval_auc(model, loader):
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for b in loader:
            logit = model(
                b["ctx"].to(device, non_blocking=True),
                b["face"].to(device, non_blocking=True),
                b["skel"].to(device, non_blocking=True),
                b["time_mask"].to(device, non_blocking=True),
                b["face_mask"].to(device, non_blocking=True),
                b["skel_mask"].to(device, non_blocking=True),
            )
            prob = torch.sigmoid(logit).detach().cpu().numpy()
            ys.append(b["y"].numpy())
            ps.append(prob)
    ys = np.concatenate(ys); ps = np.concatenate(ps)
    if len(np.unique(ys)) < 2:
        return float("nan")
    return roc_auc_score(ys, ps)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
y = train_ok["label"].values

best_auc = -1
best_state = None

for fold, (tr_idx, va_idx) in enumerate(skf.split(train_ok, y), 1):
    tr_df = train_ok.iloc[tr_idx].reset_index(drop=True)
    va_df = train_ok.iloc[va_idx].reset_index(drop=True)

    tr_ds = Track3CachedDataset(tr_df, feat_dir=FEAT_DIR, phase=1, has_label=True)
    va_ds = Track3CachedDataset(va_df, feat_dir=FEAT_DIR, phase=1, has_label=True)

    tr_loader = DataLoader(tr_ds, batch_size=4, shuffle=True, num_workers=0, collate_fn=collate_fn, pin_memory=True)
    va_loader = DataLoader(va_ds, batch_size=4, shuffle=False, num_workers=0, collate_fn=collate_fn, pin_memory=True)

    pos = (tr_df["label"] == 1).sum()
    neg = (tr_df["label"] == 0).sum()
    pos_weight = torch.tensor([neg / max(pos, 1)], dtype=torch.float32).to(device)

    model = TriStreamModel(d_ctx_in=512, d_face_in=D_FACE, d_skel_in=d_skel_in,
                          d=512, n_layers=4, n_heads=4, dropout=0.3).to(device)

    crit = FocalLoss(gamma=1.0, pos_weight=pos_weight)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)

    scaler = torch.cuda.amp.GradScaler(enabled=(device=="cuda"))
    GRAD_ACCUM = 2
    patience = 4
    bad = 0
    fold_best = -1

    for epoch in range(1, 21):
        model.train()
        opt.zero_grad(set_to_none=True)
        losses = []

        for step, b in enumerate(tr_loader, 1):
            with torch.cuda.amp.autocast(enabled=(device=="cuda")):
                logit = model(
                    b["ctx"].to(device, non_blocking=True),
                    b["face"].to(device, non_blocking=True),
                    b["skel"].to(device, non_blocking=True),
                    b["time_mask"].to(device, non_blocking=True),
                    b["face_mask"].to(device, non_blocking=True),
                    b["skel_mask"].to(device, non_blocking=True),
                )
                loss = crit(logit, b["y"].to(device, non_blocking=True)) / GRAD_ACCUM

            scaler.scale(loss).backward()

            if step % GRAD_ACCUM == 0:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt); scaler.update()
                opt.zero_grad(set_to_none=True)

            losses.append(loss.item() * GRAD_ACCUM)

        auc = eval_auc(model, va_loader)
        print(f"fold {fold} epoch {epoch}: loss={np.mean(losses):.4f} val_auc={auc:.4f}")

        if np.isfinite(auc) and auc > fold_best + 1e-4:
            fold_best = auc
            bad = 0
            if auc > best_auc:
                best_auc = auc
                best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break

print("Best CV AUC:", best_auc)

if best_state is not None:
    out = "/content/drive/MyDrive/best_tristream_cv_agcn.pt"
    torch.save(best_state, out)
    print("Saved:", out)
