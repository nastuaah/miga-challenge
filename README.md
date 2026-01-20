### **0. Environment and Execution**
   Google Colab Pro with GPU A100

### **1. Dataset**
   The official challenge data from Kaggle (RGB videos + skeleton CSVs):
  - https://miga3.a3s.fi/imigue_skeleton_phase1.zip
  - https://miga3.a3s.fi/imigue_rgb_phase1.zip
  - https://miga3.a3s.fi/imigue_skeleton_phase2.zip
  - https://miga3.a3s.fi/imigue_rgb_phase2.zip

### **2. Project Code Layout**
   All core logic is implemented as Python modules inside `src/`:
- `src/paths.py`  
  Path resolution helpers for video and skeleton files.
- `src/feature_extract.py`  
  Feature extraction and caching (ctx / face / skeleton).
- `src/agcn.py`  
  Skeleton embedding model (AGCN-based) producing** 512-D features.
- `src/cache_features_cli.py`  
  CLI script to run caching on CSV shards.
- `src/ds.py`  
  Dataset reading cached features (`*_ctx.pt`, `*_face.pt`, `*_skel.pt`) and masks.
- `src/model.py`  
  `TriStreamModel`: three transformer encoders + attention pooling + binary head.
- `src/train_model.py` / `src/inference.py`  
  Files for training/inference

### **3. Feature Caching **
**3.1 Context stream (`*_ctx.pt`)**
- Video is split into windows of length `chunk` frames.
- Each window is embedded by a video backbone into a fixed vector.
- Output:
  - `*_ctx.pt` with shape (W, 512)

**3.2 Face stream (`*_face.pt` + `*_facemask.pt`)**
- Face detection: YOLOv8 (Ultralytics)
- Face embedding: EmotiEffLib model
- Output:
  - `*_face.pt` with shape (W, 1280)
  - `*_facemask.pt` with shape (W, )

**3.3 Skeleton stream (`*_skel.pt` + `*_skelmask.pt`)**
- Skeleton is loaded from CSV per video.
- The AGCN-based encoder maps skeleton input → 512-D embedding per window.
- Output:
  - `*_skel.pt` with shape (W, 512)
  - `*_skelmask.pt` with shape **(W, )

## 4. Phase 1 CSV Preparation 
Building `/content/phase1_all_with_split.csv`:
- `video_id`
- `label` (only Phase 1)
- `split` in `{train, val}`

## 5. Phase 1 Feature Caching

Cached features are written to Google Drive.
- Phase 1 cache directory:
  - `/content/drive/MyDrive/miga_features_cache_agcn`
 
**5.1 Caching context features (ctx)**
```bash
PYTHONPATH=/content PYTHONDONTWRITEBYTECODE=1 python -u /content/src/cache_features_cli.py \
  --csv /content/phase1_all_with_split.csv --phase 1 \
  --feat_dir /content/drive/MyDrive/miga_features_cache_agcn \
  --chunk 32 --device cuda --do_ctx
```

**5.2 Caching face features (face)**
```bash
PYTHONPATH=/content PYTHONDONTWRITEBYTECODE=1 python -u /content/src/cache_features_cli.py \
  --csv /content/phase1_all_with_split.csv --phase 1 \
  --feat_dir /content/drive/MyDrive/miga_features_cache_agcn \
  --chunk 32 --device cuda --do_face
  ```

**5.3 Caching skeleton features (skel)**
```bash
PYTHONPATH=/content PYTHONDONTWRITEBYTECODE=1 python -u /content/src/cache_features_cli.py \
  --csv /content/phase1_all_with_split.csv --phase 1 \
  --feat_dir /content/drive/MyDrive/miga_features_cache_agcn \
  --chunk 32 --device cuda --do_skel
  ```
For large caching runs, the CSV range was splitted into shards 10-25 items each.
