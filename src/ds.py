import os
import torch
from torch.utils.data import Dataset

from .feature_extract import cache_paths


_CTX_PROJ = None

def _ctx_to_512(x: torch.Tensor) -> torch.Tensor:
    """
    x: (T, D) on CPU
    returns (T, 512) on CPU
    """
    global _CTX_PROJ
    if x.ndim != 2:
        raise ValueError(f"ctx must be (T,D), got {tuple(x.shape)}")

    T, D = x.shape
    if D == 512:
        return x
    if D == 768:

        if _CTX_PROJ is None:
            _CTX_PROJ = torch.nn.Linear(768, 512, bias=False)
            torch.nn.init.orthogonal_(_CTX_PROJ.weight)
            _CTX_PROJ.eval()
        with torch.no_grad():
            return _CTX_PROJ(x.float()).to(dtype=torch.float32)
    raise ValueError(f"Unexpected ctx dim: {D} (expected 512 or 768)")

def _ensure_float32(x: torch.Tensor) -> torch.Tensor:
    return x.float() if x.dtype != torch.float32 else x


class Track3CachedDataset(Dataset):
    def __init__(self, df, feat_dir: str, phase: int = 1, has_label: bool = True):
        self.df = df.reset_index(drop=True)
        self.feat_dir = feat_dir
        self.phase = phase
        self.has_label = has_label

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        vid = int(row["video_id"])
        split = row.get("split", "test")

        ps = cache_paths(self.feat_dir, vid, split, self.phase)

        ctx  = torch.load(ps["ctx"],  map_location="cpu")
        face = torch.load(ps["face"], map_location="cpu")
        skel = torch.load(ps["skel"], map_location="cpu")

        fmask = torch.load(ps["fmask"], map_location="cpu")
        smask = torch.load(ps["smask"], map_location="cpu")

        ctx  = _ensure_float32(ctx)
        face = _ensure_float32(face)
        skel = _ensure_float32(skel)

        ctx = _ctx_to_512(ctx)

        T = min(ctx.shape[0], face.shape[0], skel.shape[0])
        ctx, face, skel = ctx[:T], face[:T], skel[:T]
        fmask, smask = fmask[:T].bool(), smask[:T].bool()

        out = {
            "id": vid,
            "ctx": ctx,       # (T,512)
            "face": face,     # (T,1280)
            "skel": skel,     # (T,512)
            "time_mask": torch.ones((T,), dtype=torch.bool),
            "face_mask": fmask,
            "skel_mask": smask,
        }
        if self.has_label:
            out["y"] = torch.tensor(float(row["label"]), dtype=torch.float32)
        return out



def pad_seq(list_TD, pad_value=0.0):
    T = max(x.shape[0] for x in list_TD)
    D = list_TD[0].shape[1]
    out = torch.full((len(list_TD), T, D), pad_value, dtype=torch.float32)
    tm  = torch.zeros((len(list_TD), T), dtype=torch.bool)
    for i, x in enumerate(list_TD):
        out[i, :x.shape[0]] = x.float()
        tm[i, :x.shape[0]] = True
    return out, tm


def collate_fn(batch):
    ctx, time_mask = pad_seq([b["ctx"] for b in batch])
    face, _        = pad_seq([b["face"] for b in batch])
    skel, _        = pad_seq([b["skel"] for b in batch])

    face_mask = torch.zeros_like(time_mask)
    skel_mask = torch.zeros_like(time_mask)

    for i, b in enumerate(batch):
        face_mask[i, :len(b["face_mask"])] = b["face_mask"]
        skel_mask[i, :len(b["skel_mask"])] = b["skel_mask"]

    out = {
        "id": [b["id"] for b in batch],
        "ctx": ctx,
        "face": face,
        "skel": skel,
        "time_mask": time_mask,
        "face_mask": face_mask,
        "skel_mask": skel_mask,
    }
    if "y" in batch[0]:
        out["y"] = torch.stack([b["y"] for b in batch])
    return out
