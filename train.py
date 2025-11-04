# train.py
import argparse, yaml
from pprint import pprint
import torch
from torch.utils.data import DataLoader
from data import build_dataset

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--split", type=str, default="train", choices=["train","valid","test","reference"])
    ap.add_argument("--batch_size", type=int, default=4)
    return ap.parse_args()

def main():
    args = parse_args()
    with open(args.config, "r") as f:
        opt = yaml.safe_load(f)

    ds = build_dataset(opt, split=args.split)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    print("\n=== Sanity check ===")
    print(f"Split: {args.split}, N={len(ds)}")
    first = next(iter(dl))

    def shape(x):
        if x is None: return None
        if isinstance(x, torch.Tensor): return tuple(x.shape)
        return f"type={type(x)}"

    summary = {
        "id": first["id"][:3],
        "receiver_pos": shape(first["receiver_pos"]),  # (B, 3)
        "source_pos":   shape(first["source_pos"]),    # (B, 3)
        "orientation":  shape(first["orientation"]),   # (B, 3)
        "stft":         shape(first["stft"]),          # (B, C, F, T)
        "wav":          shape(first["wav"]),           # (B, C, T)
        "feat_rgb":     shape(first.get("feat_rgb", None)),
        "feat_depth":   shape(first.get("feat_depth", None)),
    }
    pprint(summary)

    B = min(2, len(first["id"]))
    for i in range(B):
        print(f"\n--- Sample {i} ({first['id'][i]}) ---")
        print("receiver_pos:", first["receiver_pos"][i].tolist())
        print("source_pos  :", first["source_pos"][i].tolist())
        print("orientation :", first["orientation"][i].tolist())
        print("stft shape  :", tuple(first["stft"][i].shape))
        print("wav shape   :", tuple(first["wav"][i].shape))
        if "feat_rgb" in first:
            print("feat_rgb   :", tuple(first["feat_rgb"][i].shape))
            print("feat_depth :", tuple(first["feat_depth"][i].shape))

    print("\nOK. Dataset wiring looks good.")

if __name__ == "__main__":
    main()