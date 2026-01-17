import os, glob

ROOT = "/content/miga_data"

RGB_P1_TRAIN = f"{ROOT}/imigue_rgb_phase1/train_data"
RGB_P1_VAL   = f"{ROOT}/imigue_rgb_phase1/validation_data"
RGB_P2       = f"{ROOT}/imigue_rgb_phase2"

SK_P1_TRAIN  = f"{ROOT}/imigue_data_phase1/datasets/imigue_skeleton_train"
SK_P1_VAL    = f"{ROOT}/imigue_data_phase1/datasets/imigue_skeleton_validate"
SK_P2_TEST   = f"{ROOT}/imigue_data_phase2/imigue_skeleton_test"

def vid4(x): return f"{int(x):04d}"

def resolve_video_path_phase1(video_id, split):
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
        if os.path.exists(p): return p
    return None

def resolve_video_path_phase2(video_id):
    v = vid4(video_id)
    p = os.path.join(RGB_P2, v, f"{v}.mp4")
    if os.path.exists(p): return p
    hits = glob.glob(os.path.join(RGB_P2, "**", f"{v}.mp4"), recursive=True)
    return hits[0] if hits else None

def resolve_skeleton_path_phase1(video_id, split, prefer_hand=True):
    v = vid4(video_id)
    base = SK_P1_TRAIN if split=="train" else SK_P1_VAL
    p_hand  = os.path.join(base, v, f"{v}_light_hand.csv")
    p_light = os.path.join(base, v, f"{v}_light.csv")
    if prefer_hand and os.path.exists(p_hand): return p_hand
    if os.path.exists(p_light): return p_light
    if os.path.exists(p_hand):  return p_hand
    return None

def resolve_skeleton_path_phase2(video_id, prefer_hand=True):
    v = vid4(video_id)
    p_hand  = os.path.join(SK_P2_TEST, v, f"{v}_light_hand.csv")
    p_light = os.path.join(SK_P2_TEST, v, f"{v}_light.csv")
    if prefer_hand and os.path.exists(p_hand): return p_hand
    if os.path.exists(p_light): return p_light
    if os.path.exists(p_hand):  return p_hand
    return None
