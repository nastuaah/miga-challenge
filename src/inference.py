import os
import pickle
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from src.ds import Track3CachedDataset, collate_fn
from src.model import TriStreamModel

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)

BASE_P2 = "/content/drive/MyDrive/miga_features_cache_agcn_p2"
test_ctx_dir = f"{BASE_P2}/exp_ctx_r2plus1d"
test_face_dir = f"{BASE_P2}/baseline_face"
test_skel_dir = f"{BASE_P2}/exp_skel_mean_std"

model_state_path = "full_model_results.pkl"  

with open(model_state_path, 'rb') as f:
    model_data = pickle.load(f)
best_state = model_data['best_state']
print("Модель загружена. Ключи:", best_state.keys() if best_state else None)


test_csv = "/content/phase2_all_with_paths.csv"  
test_df = pd.read_csv(test_csv)
print(f"Загружено тестовых образцов: {len(test_df)}")


test_ds = Track3CachedDataset(
    test_df,
    ctx_dir=test_ctx_dir,
    face_dir=test_face_dir,
    skel_dir=test_skel_dir,
    phase=2,
    has_label=False,  
    use_ctx=True,
    use_face=True,
    use_skel=True
)

test_loader = DataLoader(
    test_ds,
    batch_size=16,
    shuffle=False,
    num_workers=0,
    collate_fn=collate_fn,
    pin_memory=True
)

# ---- создаём модель и загружаем веса ----
model = TriStreamModel(
    d_ctx_in=d_ctx_in,
    d_face_in=D_FACE,
    d_skel_in=d_skel_in,
    d=512,
    n_layers=4,
    n_heads=4,
    dropout=0.3,
).to(device)

model.load_state_dict(best_state)
model.eval()
print("Модель инициализирована и веса загружены.")

# ---- инференс ----
all_ids = []
all_probs = []

with torch.no_grad():
    for batch in test_loader:
        ids = batch["id"]
        logits = model(
            batch["ctx"].to(device, non_blocking=True),
            batch["face"].to(device, non_blocking=True),
            batch["skel"].to(device, non_blocking=True),
            batch["time_mask"].to(device, non_blocking=True),
            batch["face_mask"].to(device, non_blocking=True),
            batch["skel_mask"].to(device, non_blocking=True),
        )
        probs = torch.sigmoid(logits).cpu().numpy()
        all_ids.extend(ids)
        all_probs.extend(probs)

results_df = pd.DataFrame({
    "video_id": all_ids,
    "prediction": all_probs 
})


output_path = "/content/phase2_predictions.csv"
results_df.to_csv(output_path, index=False)
print(f"Предсказания сохранены в {output_path}")
print(results_df.head())
