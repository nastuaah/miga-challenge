**0. Environment and Execution**
   Google Colab Pro with GPU A100

**1. Dataset**
   The official challenge data from Kaggle (RGB videos + skeleton CSVs):
  - https://miga3.a3s.fi/imigue_skeleton_phase1.zip
  - https://miga3.a3s.fi/imigue_rgb_phase1.zip
  - https://miga3.a3s.fi/imigue_skeleton_phase2.zip
  - https://miga3.a3s.fi/imigue_rgb_phase2.zip

**2. Project Code Layout**
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
