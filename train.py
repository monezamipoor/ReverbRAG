# train.py
import argparse, yaml
from pprint import pprint
import torch
from torch.utils.data import DataLoader, get_worker_info
from data import build_dataset
from dataloader import RandomSliceBatchSampler, EDCFullBatchSampler

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--split", type=str, default="train", choices=["train","valid","test","reference"])
    return ap.parse_args()

def _worker_init_fn(worker_id):
    # Ensure each worker starts with a fresh, small LRU cache
    info = get_worker_info()
    ds = info.dataset  # this worker's dataset copy
    # Reset cache structures (already created in __init__, but ensure clean)
    try:
        ds._cache.clear()
    except Exception:
        ds._cache = OrderedDict()  # fallback if needed

def main():
    args = parse_args()
    with open(args.config, "r") as f:
        opt = yaml.safe_load(f)

    ds = build_dataset(opt, split=args.split)

    sampler_cfg = opt.get("sampler", {})
    dataset_mode = sampler_cfg.get("dataset_mode", "full")
    batching = sampler_cfg.get("batching", "random")
    bs = int(sampler_cfg.get("batch_size", 4))
    shuffle = bool(sampler_cfg.get("shuffle", True))
    num_workers = int(sampler_cfg.get("num_workers", 4))

    # Build DataLoader with persistent workers to keep caches warm
    if dataset_mode == "full":
        if batching == "edc_full":
            raise ValueError("edc_full batching requires dataset_mode='slice'.")
        dl = DataLoader(
            ds, batch_size=bs, shuffle=shuffle, num_workers=num_workers,
            pin_memory=True, persistent_workers=(num_workers > 0),
            worker_init_fn=_worker_init_fn,
        )
    else:
        if batching == "random":
            batch_sampler = RandomSliceBatchSampler(len(ds), batch_size=bs, drop_last=False, shuffle=shuffle)
        elif batching == "edc_full":
            batch_sampler = EDCFullBatchSampler(ids=ds.ids, max_frames=ds.max_frames,
                                                batch_size_rirs=bs, drop_last=False, shuffle=shuffle)
        else:
            raise ValueError(f"Unknown batching mode '{batching}'")
        dl = DataLoader(
            ds, batch_sampler=batch_sampler, num_workers=num_workers,
            pin_memory=True, persistent_workers=(num_workers > 0),
            worker_init_fn=_worker_init_fn,
        )

    print("\n=== Sanity check (caching on) ===")
    print(f"Split: {args.split}, N_items={len(ds)}, mode={dataset_mode}, batching={batching}, batch_size={bs}")
    batch = next(iter(dl))

    def shape(x):
        if x is None: return None
        if isinstance(x, torch.Tensor): return tuple(x.shape)
        return f"type={type(x)}"

    if dataset_mode == "full":
        # Replace the "summary" section inside the `if dataset_mode == "full":` branch:

        summary = {
            "batch_size": len(batch["id"]),
            "ids": batch["id"],                         # show all ids
            "receiver_pos": shape(batch["receiver_pos"]),  # (B, 3)
            "source_pos":   shape(batch["source_pos"]),    # (B, 3)
            "orientation":  shape(batch["orientation"]),   # (B, 3)
            "wav":          shape(batch["wav"]),           # (B, 1, T)
            "stft":         shape(batch["stft"]),          # (B, 1, 513, 60)
            "feat_rgb":     shape(batch.get("feat_rgb", None)),
            "feat_depth":   shape(batch.get("feat_depth", None)),
        }
        
        B = min(2, len(batch["id"]))
        for i in range(B):
            print(f"\n--- FULL Sample {i} ({batch['id'][i]}) ---")
            print("receiver_pos:", batch["receiver_pos"][i].tolist())
            print("source_pos  :", batch["source_pos"][i].tolist())
            print("orientation :", batch["orientation"][i].tolist())
            print("wav shape   :", tuple(batch["wav"][i].shape))
            print("stft shape  :", tuple(batch["stft"][i].shape))
            if "feat_rgb" in batch:
                print("feat_rgb   :", tuple(batch["feat_rgb"][i].shape))
                print("feat_depth :", tuple(batch["feat_depth"][i].shape))


    else:
        summary = {
            "id": batch["id"][:6],
            "slice_t": list(map(int, batch["slice_t"][:6])),
            "wav_slice": shape(batch["wav_slice"]),   # (B, 1, win’)
            "stft_slice": shape(batch["stft_slice"]), # (B, 1, F)
        }
    pprint(summary)
    print("\nOK. Loader + cache wiring looks good.")

if __name__ == "__main__":
    main()
