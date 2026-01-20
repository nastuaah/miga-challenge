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

### 3. Feature Caching 
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
- `label` 
- `split` in `{train, val}`

## 5. Phase 1 Feature Caching

Cached features are written to Google Drive.
- Phase 1 cache directory:
  - [`/content/drive/MyDrive/miga_features_cache_agcn`](https://drive.google.com/drive/folders/1wx77jdvzYPDFmXWWjKo76mGUg_Ghj3eR?usp=sharing)
 
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

## 6. TriStreamModel
TriStreamModel processes three aligned sequences:
- Context: (B, W, 512)
- Face: (B, W, 1280)
- Skeleton: (B, W, 512)
- Linear projection → common hidden dim d=512
- Transformer encoder per stream (n_layers=4, n_heads=4)
- Concatenation of pooled embeddings → MLP head → single logit
- Output:logit of shape (B,) or (B, 1) (implementation returns (B,))

## 7. Training (Phase 1) 
- Focal loss with pos_weight (class imbalance handling)
- Validation metric: ROC AUC on validation fold
- Cross-validation: stratifiedKFold ensures class ratio is preserved in each fold.
- Best checkpoint saved to: [`/content/drive/MyDrive/best_tristream_cv_agcn.pt`](https://drive.google.com/file/d/1fCPdkaUaZJNtd9BNaSROGlS8ZNfJpzcl/view?usp=sharing)
- **Best CV AUC: 0.6947368421052631**

## 8. Phase 2 (Test) CSV Preparation 
Building /content/phase2_all_with_paths.csv:
- `video_id`
- `split="test"`
- `video_path`
- `skeleton_path`

## 9. Phase 2 Feature Caching
Phase 2 cached features are written to: [`/content/drive/MyDrive/miga_features_cache_agcn_p2`](https://drive.google.com/drive/folders/11bBvaHCe4l2YAy4m2Y9e6Lwa2lfvyUqY?usp=sharing)

## 10. Inference on Phase 2 (Test)
**10.1 Load model checkpoint**
```bash
from src.model import TriStreamModel
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "/content/drive/MyDrive/best_tristream_cv_agcn.pt"

model = TriStreamModel(
    d_ctx_in=512,
    d_face_in=1280,
    d_skel_in=512,
    d=512,
    n_layers=4,
    n_heads=4,
    dropout=0.3
).to(device)

state = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(state)
model.eval()
  ```

**10.2 Run inference**
- For each batch:
  - forward → logits
  - probability = sigmoid(logit) (binary class probability)
    
**10.3 Create submission.csv**
- The submission file contain:
  - video_id: integer ID
  - label: predicted probability in [0, 1]
  - submission file: [/content/drive/MyDrive/best_tristream_cv_agcn.pt](https://drive.google.com/file/d/1fCPdkaUaZJNtd9BNaSROGlS8ZNfJpzcl/view?usp=sharing)
<img width="600" height="200" alt="image" src="https://github.com/user-attachments/assets/dc1ee081-41fa-4ae6-a5ba-3657ce09c1f7" />
