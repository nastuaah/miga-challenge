import argparse
import pandas as pd
from tqdm import tqdm
import torch
import traceback

from src.feature_extract import build_and_cache_one

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--phase", type=int, required=True)
    ap.add_argument("--feat_dir", required=True)
    ap.add_argument("--chunk", type=int, default=32)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=999999)
    ap.add_argument("--split_col", type=str, default="split")
    ap.add_argument("--device", choices=["auto","cpu","cuda"], default="auto")
    ap.add_argument("--do_ctx", action="store_true")
    ap.add_argument("--do_face", action="store_true")
    ap.add_argument("--do_skel", action="store_true")
    args = ap.parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    if not (args.do_ctx or args.do_face or args.do_skel):
        args.do_ctx = args.do_face = args.do_skel = True

    print("DEVICE:", device)
    print("FEAT_DIR:", args.feat_dir)
    print("MODES: ctx=", args.do_ctx, "face=", args.do_face, "skel=", args.do_skel)

    df = pd.read_csv(args.csv)
    df = df.iloc[args.start:args.end].reset_index(drop=True)

    ok, bad = 0, 0

    for i, r in tqdm(df.iterrows(), total=len(df)):
        try:
            vid = int(r["video_id"])
            split = r.get(args.split_col, "test")

            build_and_cache_one(
                video_id=vid,
                split=split,
                phase=args.phase,
                feat_dir=args.feat_dir,
                chunk=args.chunk,
                vpath=None,
                spath=r.get("skeleton_path", None),
                do_ctx=args.do_ctx,
                do_face=args.do_face,
                do_skel=args.do_skel,
                device=device,
            )
            ok += 1

        except Exception:
            bad += 1
            print(" ERROR")
            print("row:", i, "video_id:", r.get("video_id"), "split:", r.get(args.split_col, ""))
            if "skeleton_path" in r:
                print("skeleton_path:", r.get("skeleton_path"))
            traceback.print_exc()

    print(f"\nDONE shard [{args.start}:{args.end}) ok={ok} bad={bad}")

if __name__ == "__main__":
    main()
