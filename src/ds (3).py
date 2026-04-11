import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import WeightedRandomSampler

def cache_key(video_id, split, phase):
    s = f"{int(video_id)}|{split}|{phase}"
    import hashlib
    return hashlib.md5(s.encode("utf-8")).hexdigest()[:12]

class Track3CachedDataset(Dataset):
    def __init__(self, df, ctx_dir=None, face_dir=None, skel_dir=None, phase=1, has_label=True, use_ctx=True, use_face=True, use_skel=True, ctx_dim=512, face_dim=1280, skel_dim=512):
        self.df = df.reset_index(drop=True)
        self.phase = phase
        self.has_label = has_label
        self.use_ctx = use_ctx
        self.use_face = use_face
        self.use_skel = use_skel
        self.ctx_dim = ctx_dim
        self.face_dim = face_dim
        self.skel_dim = skel_dim

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

        if self.use_ctx and self.ctx_dir:
            ctx_path = os.path.join(self.ctx_dir, f"{key}_ctx.pt")
            if os.path.exists(ctx_path):
                ctx = torch.load(ctx_path)
                out["ctx"] = ctx.float()
            else:
                out["ctx"] = torch.zeros((1, self.ctx_dim), dtype=torch.float32)
        else:
            out["ctx"] = torch.zeros((1, self.ctx_dim), dtype=torch.float32)

        if self.use_face and self.face_dir:
            face_path = os.path.join(self.face_dir, f"{key}_face.pt")
            fmask_path = os.path.join(self.face_dir, f"{key}_facemask.pt")
            if os.path.exists(face_path) and os.path.exists(fmask_path):
                face = torch.load(face_path)
                fmask = torch.load(fmask_path)
                out["face"] = face.float()
                out["face_mask"] = fmask.bool()
            else:
                out["face"] = torch.zeros((1, self.face_dim), dtype=torch.float32)
                out["face_mask"] = torch.zeros(1, dtype=torch.bool)
        else:
            out["face"] = torch.zeros((1, self.face_dim), dtype=torch.float32)
            out["face_mask"] = torch.zeros(1, dtype=torch.bool)

        if self.use_skel and self.skel_dir:
            skel_path = os.path.join(self.skel_dir, f"{key}_skel.pt")
            smask_path = os.path.join(self.skel_dir, f"{key}_skelmask.pt")
            if os.path.exists(skel_path) and os.path.exists(smask_path):
                skel = torch.load(skel_path)
                smask = torch.load(smask_path)
                out["skel"] = skel.float()
                out["skel_mask"] = smask.bool()
            else:
                out["skel"] = torch.zeros((1, self.skel_dim), dtype=torch.float32)
                out["skel_mask"] = torch.zeros(1, dtype=torch.bool)
        else:
            out["skel"] = torch.zeros((1, self.skel_dim), dtype=torch.float32)
            out["skel_mask"] = torch.zeros(1, dtype=torch.bool)

        if self.labels is not None:
            out["y"] = torch.tensor(self.labels[idx], dtype=torch.float32)

        return out
