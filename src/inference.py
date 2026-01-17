import numpy as np
from tqdm import tqdm

all_ids = []
all_probs = []

with torch.no_grad():
    for b in tqdm(test_loader):
        logit = model(
            b["ctx"].to(device, non_blocking=True),
            b["face"].to(device, non_blocking=True),
            b["skel"].to(device, non_blocking=True),
            b["time_mask"].to(device, non_blocking=True),
            b["face_mask"].to(device, non_blocking=True),
            b["skel_mask"].to(device, non_blocking=True),
        )

        print(f"logit shape: {logit.shape}")

        prob = torch.sigmoid(logit).cpu().numpy().flatten()  

        video_ids = np.array(b["id"])

        all_probs.append(prob)
        all_ids.append(video_ids)

all_probs = np.concatenate(all_probs)
all_ids   = np.concatenate(all_ids)

print("Predictions:", all_probs.shape)
print("IDs:", all_ids.shape)
