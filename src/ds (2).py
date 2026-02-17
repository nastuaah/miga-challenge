import os
import torch
import numpy as np
from torch.utils.data import Dataset

def cache_key(video_id, split, phase):
    s = f"{int(video_id)}|{split}|{phase}"
    import hashlib
    return hashlib.md5(s.encode("utf-8")).hexdigest()[:12]

class Track3CachedDataset(Dataset):
    def __init__(
        self,
        df,
        ctx_dir=None,
        face_dir=None,
        skel_dir=None,
        feat_dir=None,
        phase=1,
        has_label=True,
        use_ctx=True,
        use_face=True,
        use_skel=True
    ):
        self.df = df.reset_index(drop=True)
        self.phase = phase
        self.has_label = has_label
        self.use_ctx = use_ctx
        self.use_face = use_face
        self.use_skel = use_skel

        if feat_dir is not None:
            self.ctx_dir = feat_dir
            self.face_dir = feat_dir
            self.skel_dir = feat_dir
        else:
            self.ctx_dir = ctx_dir
            self.face_dir = face_dir
            self.skel_dir = skel_dir

        self.keys = []
        self.ids = []
        self.splits = []
        for _, row in self.df.iterrows():
            video_id = int(row["video_id"])
            split = row.get("split", "train")
            key = cache_key(video_id, split, self.phase)
            self.keys.append(key)
            self.ids.append(video_id)
            self.splits.append(split)

        if has_label and "label" in self.df.columns:
            self.labels = self.df["label"].values.astype(np.float32)
        else:
            self.labels = None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        key = self.keys[idx]
        out = {"id": self.ids[idx], "split": self.splits[idx]}

        # Контекст
        if self.use_ctx and self.ctx_dir:
            ctx_path = os.path.join(self.ctx_dir, f"{key}_ctx.pt")
            if os.path.exists(ctx_path):
                ctx = torch.load(ctx_path)
                out["ctx"] = ctx.float()
                out["time_mask"] = torch.ones(len(ctx), dtype=torch.bool)
            else:
                out["ctx"] = torch.zeros((0, 512), dtype=torch.float32)
                out["time_mask"] = torch.zeros(0, dtype=torch.bool)
        else:
            out["ctx"] = torch.zeros((0, 512), dtype=torch.float32)
            out["time_mask"] = torch.zeros(0, dtype=torch.bool)

        # Лицо
        if self.use_face and self.face_dir:
            face_path = os.path.join(self.face_dir, f"{key}_face.pt")
            fmask_path = os.path.join(self.face_dir, f"{key}_facemask.pt")
            if os.path.exists(face_path) and os.path.exists(fmask_path):
                face = torch.load(face_path)
                fmask = torch.load(fmask_path)
                out["face"] = face.float()
                out["face_mask"] = fmask.bool()
            else:
                out["face"] = torch.zeros((0, 1280), dtype=torch.float32)
                out["face_mask"] = torch.zeros(0, dtype=torch.bool)
        else:
            out["face"] = torch.zeros((0, 1280), dtype=torch.float32)
            out["face_mask"] = torch.zeros(0, dtype=torch.bool)

        # Скелет
        if self.use_skel and self.skel_dir:
            skel_path = os.path.join(self.skel_dir, f"{key}_skel.pt")
            smask_path = os.path.join(self.skel_dir, f"{key}_skelmask.pt")
            if os.path.exists(skel_path) and os.path.exists(smask_path):
                skel = torch.load(skel_path)
                smask = torch.load(smask_path)
                out["skel"] = skel.float()
                out["skel_mask"] = smask.bool()
            else:
                out["skel"] = torch.zeros((0, 512), dtype=torch.float32)
                out["skel_mask"] = torch.zeros(0, dtype=torch.bool)
        else:
            out["skel"] = torch.zeros((0, 512), dtype=torch.float32)
            out["skel_mask"] = torch.zeros(0, dtype=torch.bool)

        if self.labels is not None:
            out["y"] = torch.tensor(self.labels[idx], dtype=torch.float32)

        return out

def pad_seq(list_TD, pad_value=0.0):
    if len(list_TD) == 0:
        return torch.empty(0, 0, 0), torch.empty(0, 0, dtype=torch.bool)
    T = max(x.shape[0] for x in list_TD)
    D = list_TD[0].shape[1] if len(list_TD[0].shape) > 1 else 0
    out = torch.full((len(list_TD), T, D), pad_value, dtype=torch.float32)
    mask = torch.zeros((len(list_TD), T), dtype=torch.bool)
    for i, x in enumerate(list_TD):
        if x.shape[0] > 0:
            out[i, :x.shape[0]] = x.float()
            mask[i, :x.shape[0]] = True
    return out, mask

def collate_fn(batch):
    # Контекст – всегда есть (может быть пустым, но для тренировки не пуст)
    ctx_list = [b["ctx"] for b in batch]
    ctx_pad, time_mask = pad_seq(ctx_list)  # (B, T_ctx, 512)
    B, T_ctx, _ = ctx_pad.shape

    # Лицо: если хотя бы один образец имеет ненулевую длину, паддим до T_ctx,
    # иначе возвращаем пустой тензор и пустую маску.
    face_lens = [b["face"].size(0) for b in batch]
    if any(l > 0 for l in face_lens):
        face_pad = torch.zeros(B, T_ctx, 1280, dtype=torch.float32)
        face_mask = torch.zeros(B, T_ctx, dtype=torch.bool)
        for i, b in enumerate(batch):
            f = b["face"]
            if f.size(0) > 0:
                copy_len = min(T_ctx, f.size(0))
                face_pad[i, :copy_len] = f[:copy_len]
                fm = b.get("face_mask", torch.ones(copy_len, dtype=torch.bool))
                face_mask[i, :copy_len] = fm[:copy_len]
    else:
        # Все образцы не имеют лица -> возвращаем пустые тензоры
        face_pad = torch.zeros(B, 0, 1280, dtype=torch.float32)
        face_mask = torch.zeros(B, 0, dtype=torch.bool)

    # Скелет аналогично
    skel_lens = [b["skel"].size(0) for b in batch]
    if any(l > 0 for l in skel_lens):
        skel_pad = torch.zeros(B, T_ctx, 512, dtype=torch.float32)
        skel_mask = torch.zeros(B, T_ctx, dtype=torch.bool)
        for i, b in enumerate(batch):
            s = b["skel"]
            if s.size(0) > 0:
                copy_len = min(T_ctx, s.size(0))
                skel_pad[i, :copy_len] = s[:copy_len]
                sm = b.get("skel_mask", torch.ones(copy_len, dtype=torch.bool))
                skel_mask[i, :copy_len] = sm[:copy_len]
    else:
        skel_pad = torch.zeros(B, 0, 512, dtype=torch.float32)
        skel_mask = torch.zeros(B, 0, dtype=torch.bool)

    out = {
        "id": [b["id"] for b in batch],
        "ctx": ctx_pad,
        "face": face_pad,
        "skel": skel_pad,
        "time_mask": time_mask,
        "face_mask": face_mask,
        "skel_mask": skel_mask,
    }

    if "y" in batch[0]:
        out["y"] = torch.stack([b["y"] for b in batch])

    return out
