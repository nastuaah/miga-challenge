import os
import glob
import hashlib
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from decord import VideoReader, cpu


MIGA_ROOT = os.environ.get("MIGA_ROOT", "/content/miga_data")

RGB_P1_TRAIN = f"{MIGA_ROOT}/imigue_rgb_phase1/train_data"
RGB_P1_VAL   = f"{MIGA_ROOT}/imigue_rgb_phase1/validation_data"
RGB_P2       = f"{MIGA_ROOT}/imigue_rgb_phase2"

SK_P1_TRAIN  = f"{MIGA_ROOT}/imigue_data_phase1/datasets/imigue_skeleton_train"
SK_P1_VAL    = f"{MIGA_ROOT}/imigue_data_phase1/datasets/imigue_skeleton_validate"
SK_P2_TEST   = f"{MIGA_ROOT}/imigue_data_phase2/imigue_skeleton_test"

# ---------------------------
# EXP SWITCHES (defaults = baseline)
# ---------------------------
CTX_BACKBONE  = os.environ.get("CTX_BACKBONE", "r3d_18")     # r3d_18 | mc3_18 | r2plus1d_18
FACE_SAMPLING = os.environ.get("FACE_SAMPLING", "mid")      # mid | 3frames_avg
FACE_DETECT   = os.environ.get("FACE_DETECT", "yolo")       # yolo | center
SKEL_AGG      = os.environ.get("SKEL_AGG", "mean")          # mean | mean_std | max


def vid4(x: int) -> str:
    return f"{int(x):04d}"


def resolve_video_path_phase1(video_id: int, split: str) -> Optional[str]:
    v = vid4(video_id)
    if split == "train":
        p = os.path.join(RGB_P1_TRAIN, v, f"{v}.mp4")
        return p if os.path.exists(p) else None
    if split == "val":
        p = os.path.join(RGB_P1_VAL, v, f"{v}.mp4")
        return p if os.path.exists(p) else None
    for p in [
        os.path.join(RGB_P1_TRAIN, v, f"{v}.mp4"),
        os.path.join(RGB_P1_VAL, v, f"{v}.mp4"),
    ]:
        if os.path.exists(p):
            return p
    return None


def resolve_video_path_phase2(video_id: int) -> Optional[str]:
    v = vid4(video_id)
    p = os.path.join(RGB_P2, v, f"{v}.mp4")
    if os.path.exists(p):
        return p
    hits = glob.glob(os.path.join(RGB_P2, "**", f"{v}.mp4"), recursive=True)
    return hits[0] if hits else None


def resolve_skeleton_path_phase1(video_id: int, split: str, prefer_hand: bool = True) -> Optional[str]:
    v = vid4(video_id)
    base = SK_P1_TRAIN if split == "train" else SK_P1_VAL
    p_hand  = os.path.join(base, v, f"{v}_light_hand.csv")
    p_light = os.path.join(base, v, f"{v}_light.csv")
    if prefer_hand and os.path.exists(p_hand): return p_hand
    if os.path.exists(p_light): return p_light
    if os.path.exists(p_hand):  return p_hand
    return None


def resolve_skeleton_path_phase2(video_id: int, prefer_hand: bool = True) -> Optional[str]:
    v = vid4(video_id)
    p_hand  = os.path.join(SK_P2_TEST, v, f"{v}_light_hand.csv")
    p_light = os.path.join(SK_P2_TEST, v, f"{v}_light.csv")
    if prefer_hand and os.path.exists(p_hand): return p_hand
    if os.path.exists(p_light): return p_light
    if os.path.exists(p_hand):  return p_hand
    return None


def cache_key(video_id: int, split: str, phase: int) -> str:
    s = f"{int(video_id)}|{split}|{phase}"
    return hashlib.md5(s.encode("utf-8")).hexdigest()[:12]


def cache_paths(feat_dir: str, video_id: int, split: str, phase: int) -> Dict[str, str]:
    key = cache_key(video_id, split, phase)
    d = Path(feat_dir)
    d.mkdir(parents=True, exist_ok=True)
    return {
        "ctx":   str(d / f"{key}_ctx.pt"),
        "face":  str(d / f"{key}_face.pt"),
        "fmask": str(d / f"{key}_facemask.pt"),
        "skel":  str(d / f"{key}_skel.pt"),
        "smask": str(d / f"{key}_skelmask.pt"),
    }


def _vr(path: str) -> VideoReader:
    return VideoReader(path, ctx=cpu(0))


# ---------------------------
# CTX
# ---------------------------
_CTX_MODEL = None

def build_ctx_model(device: str = "cuda"):
    global _CTX_MODEL
    if _CTX_MODEL is not None:
        _CTX_MODEL.to(device).eval()
        return _CTX_MODEL

    if CTX_BACKBONE == "mc3_18":
        from torchvision.models.video import mc3_18, MC3_18_Weights
        weights = MC3_18_Weights.DEFAULT
        model = mc3_18(weights=weights)
        model.fc = nn.Identity()
    elif CTX_BACKBONE == "r2plus1d_18":
        from torchvision.models.video import r2plus1d_18, R2Plus1D_18_Weights
        weights = R2Plus1D_18_Weights.DEFAULT
        model = r2plus1d_18(weights=weights)
        model.fc = nn.Identity()
    else:
        from torchvision.models.video import r3d_18, R3D_18_Weights
        weights = R3D_18_Weights.DEFAULT
        model = r3d_18(weights=weights)
        model.fc = nn.Identity()

    model = model.to(device).eval()
    _CTX_MODEL = model
    return _CTX_MODEL


def _preprocess_clip_torch(clip_rgb: np.ndarray, device: str) -> torch.Tensor:
    x = torch.from_numpy(clip_rgb).to(torch.float32) / 255.0
    x = x.permute(3, 0, 1, 2)  # (C,T,H,W)
    x = x.unsqueeze(0)         # (1,C,T,H,W)
    x = F.interpolate(x, size=(x.shape[2], 112, 112), mode="trilinear", align_corners=False)

    mean = torch.tensor([0.43216, 0.394666, 0.37645], device=x.device).view(1,3,1,1,1)
    std  = torch.tensor([0.22803, 0.22145, 0.216989], device=x.device).view(1,3,1,1,1)
    x = (x - mean) / std
    return x.to(device)


@torch.no_grad()
def extract_ctx_stream(video_path: str, device: str = "cuda", chunk: int = 32) -> torch.Tensor:
    model = build_ctx_model(device=device)
    vr = _vr(video_path)
    T = len(vr)
    if T <= 0:
        return torch.zeros((0, 512), dtype=torch.float32)

    pad = (-T) % chunk
    total = T + pad

    feats = []
    for i in range(0, total, chunk):
        idxs = list(range(i, min(i + chunk, T)))
        if len(idxs) == 0:
            break
        frames = vr.get_batch(idxs).asnumpy()
        if frames.shape[0] < chunk:
            frames = np.concatenate([frames, np.repeat(frames[-1:], chunk - frames.shape[0], axis=0)], axis=0)

        x = _preprocess_clip_torch(frames, device=device)
        f = model(x).squeeze(0).detach().cpu()
        feats.append(f)

    return torch.stack(feats, dim=0).to(torch.float32)


# ---------------------------
# FACE
# ---------------------------
_FACE_MODEL = None
_EMO_REC = None

def _init_yolo():
    global _FACE_MODEL
    if _FACE_MODEL is not None:
        return _FACE_MODEL
    try:
        from ultralytics import YOLO
        for w in ["yolov8n-face.pt", "yolov8n.pt"]:
            try:
                _FACE_MODEL = YOLO(w)
                break
            except Exception:
                _FACE_MODEL = None
    except Exception:
        _FACE_MODEL = None
    return _FACE_MODEL


def _center_crop(img, size=224):
    h, w = img.shape[:2]
    s = min(h, w)
    y0 = (h - s) // 2
    x0 = (w - s) // 2
    crop = img[y0:y0+s, x0:x0+s]
    import cv2
    crop = cv2.resize(crop, (size, size))
    return crop


def _detect_face_crop(img_rgb, size=224):
    if FACE_DETECT == "center":
        return _center_crop(img_rgb, size=size), True

    model = _init_yolo()
    if model is None:
        return _center_crop(img_rgb, size=size), False

    try:
        res = model.predict(source=img_rgb, verbose=False)
        if len(res) == 0 or res[0].boxes is None or len(res[0].boxes) == 0:
            return _center_crop(img_rgb, size=size), False
        boxes = res[0].boxes.xyxy.detach().cpu().numpy()
        areas = (boxes[:,2]-boxes[:,0])*(boxes[:,3]-boxes[:,1])
        b = boxes[int(np.argmax(areas))]
        x1, y1, x2, y2 = [int(max(0, v)) for v in b]
        x2 = min(x2, img_rgb.shape[1]-1)
        y2 = min(y2, img_rgb.shape[0]-1)
        crop = img_rgb[y1:y2, x1:x2]
        if crop.size == 0:
            return _center_crop(img_rgb, size=size), False
        import cv2
        crop = cv2.resize(crop, (size, size))
        return crop, True
    except Exception:
        return _center_crop(img_rgb, size=size), False


def _init_emotieff(device: str):
    global _EMO_REC
    if _EMO_REC is not None:
        return _EMO_REC
    from emotiefflib.facial_analysis import EmotiEffLibRecognizerTorch
    try:
        _EMO_REC = EmotiEffLibRecognizerTorch(device=device)
    except TypeError:
        _EMO_REC = EmotiEffLibRecognizerTorch()
    return _EMO_REC


def _as_rgb_uint8(face_rgb_224):
    x = np.asarray(face_rgb_224)
    if x.dtype != np.uint8:
        x = np.clip(x, 0, 255).astype(np.uint8)
    return x


@torch.no_grad()
def emotieff_embed(face_rgb_224: np.ndarray, device: str = "cuda") -> np.ndarray:
    rec = _init_emotieff(device=device)
    x = _as_rgb_uint8(face_rgb_224)

    for name in ["get_face_embedding","get_embedding","get_embeddings","extract_embedding","extract_embeddings","get_features","extract_features"]:
        if hasattr(rec, name) and callable(getattr(rec, name)):
            out = getattr(rec, name)(x)
            return np.asarray(out).reshape(-1).astype(np.float32)

    if callable(rec):
        out = rec(x)
        if isinstance(out, dict):
            for k in ["embedding","emb","features","feature"]:
                if k in out:
                    return np.asarray(out[k]).reshape(-1).astype(np.float32)
        return np.asarray(out).reshape(-1).astype(np.float32)

    raise RuntimeError("EmotiEffLib: не удалось получить embedding")


@torch.no_grad()
def extract_face_windows(video_path: str, device: str = "cuda", chunk: int = 32) -> Tuple[torch.Tensor, torch.Tensor]:
    vr = _vr(video_path)
    T = len(vr)
    if T <= 0:
        return torch.zeros((0, 1280), dtype=torch.float32), torch.zeros((0,), dtype=torch.bool)

    pad = (-T) % chunk
    total = T + pad

    feats = []
    mask = []
    D = None

    for i in range(0, total, chunk):
        if FACE_SAMPLING == "3frames_avg":
            idxs = [
                min(i + max(0, chunk // 4), T - 1),
                min(i + max(0, chunk // 2), T - 1),
                min(i + max(0, (3 * chunk) // 4), T - 1),
            ]
        else:
            idxs = [min(i + chunk // 2, T - 1)]

        embs = []
        oks = []
        for idx in idxs:
            frame = vr[idx].asnumpy()
            crop, ok = _detect_face_crop(frame, size=224)
            if ok:
                embs.append(emotieff_embed(crop, device=device))
                oks.append(True)
            else:
                oks.append(False)

        if any(oks):
            emb = np.mean(embs, axis=0).astype(np.float32)
            if D is None:
                D = int(emb.shape[0])
            feats.append(torch.from_numpy(emb).to(torch.float32))
            mask.append(True)
        else:
            if D is None:
                D = 1280
            feats.append(torch.zeros((D,), dtype=torch.float32))
            mask.append(False)

    face = torch.stack(feats, dim=0)
    fmask = torch.tensor(mask, dtype=torch.bool)
    return face, fmask


# ---------------------------
# SKEL
# ---------------------------
_SKEL_PROJ = None

def _skel_projector(in_dim: int, out_dim: int = 512) -> nn.Module:
    global _SKEL_PROJ
    if _SKEL_PROJ is not None and getattr(_SKEL_PROJ, "_in_dim", None) == in_dim:
        return _SKEL_PROJ
    torch.manual_seed(42)
    proj = nn.Sequential(
        nn.Linear(in_dim, out_dim),
        nn.LayerNorm(out_dim),
        nn.GELU(),
    )
    proj._in_dim = in_dim
    _SKEL_PROJ = proj.eval()
    return _SKEL_PROJ


def _load_skeleton_csv(path: str) -> np.ndarray:
    df = pd.read_csv(path)
    return df.values.astype(np.float32)  # (T,D)


@torch.no_grad()
def extract_skeleton_windows(skel_csv_path: str, device: str = "cpu", chunk: int = 32) -> Tuple[torch.Tensor, torch.Tensor]:
    arr = _load_skeleton_csv(skel_csv_path)
    T, D = arr.shape
    if T <= 0:
        return torch.zeros((0, 512), dtype=torch.float32), torch.zeros((0,), dtype=torch.bool)

    pad = (-T) % chunk
    total = T + pad
    if pad:
        arr = np.concatenate([arr, np.repeat(arr[-1:], pad, axis=0)], axis=0)

    # aggregator output dim
    if SKEL_AGG == "mean_std":
        in_dim = 2 * D
    else:
        in_dim = D

    proj = _skel_projector(in_dim, 512)

    feats = []
    mask = []
    for i in range(0, total, chunk):
        w = arr[i:i+chunk]  # (chunk,D)
        valid = not np.allclose(w, 0)

        if SKEL_AGG == "max":
            v = w.max(axis=0)
        elif SKEL_AGG == "mean_std":
            mu = w.mean(axis=0)
            sd = w.std(axis=0)
            v = np.concatenate([mu, sd], axis=0)
        else:
            v = w.mean(axis=0)

        x = torch.from_numpy(v).to(torch.float32)
        z = proj(x).detach().cpu()
        feats.append(z)
        mask.append(valid)

    skel = torch.stack(feats, dim=0).to(torch.float32)
    smask = torch.tensor(mask, dtype=torch.bool)
    return skel, smask


@torch.no_grad()
def build_and_cache_one(
    video_id: int,
    split: str,
    phase: int,
    feat_dir: str,
    chunk: int = 32,
    vpath: Optional[str] = None,
    spath: Optional[str] = None,
    do_ctx: bool = True,
    do_face: bool = True,
    do_skel: bool = True,
    device: str = "cuda",
):
    ps = cache_paths(feat_dir, video_id, split, phase)

    if vpath is None:
        vpath = resolve_video_path_phase1(video_id, split) if phase == 1 else resolve_video_path_phase2(video_id)

    if spath is None:
        spath = resolve_skeleton_path_phase1(video_id, split) if phase == 1 else resolve_skeleton_path_phase2(video_id)

    if do_ctx or do_face:
        if vpath is None or (not os.path.exists(vpath)):
            raise FileNotFoundError(f"RGB video not found for id={video_id} split={split} phase={phase}")

    if do_skel:
        if spath is None or (not os.path.exists(spath)):
            raise FileNotFoundError(f"Skeleton csv not found for id={video_id} split={split} phase={phase}")

    if do_ctx and (not os.path.exists(ps["ctx"])):
        ctx = extract_ctx_stream(vpath, device=device, chunk=chunk)
        torch.save(ctx, ps["ctx"])

    if do_face and (not os.path.exists(ps["face"])) and (not os.path.exists(ps["fmask"])):
        face, fmask = extract_face_windows(vpath, device=device, chunk=chunk)
        torch.save(face, ps["face"])
        torch.save(fmask, ps["fmask"])

    if do_skel and (not os.path.exists(ps["skel"])) and (not os.path.exists(ps["smask"])):
        skel, smask = extract_skeleton_windows(spath, device="cpu", chunk=chunk)
        torch.save(skel, ps["skel"])
        torch.save(smask, ps["smask"])

    return ps
