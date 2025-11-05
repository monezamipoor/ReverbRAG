# train.py
# Dataset-backed smoke test for Unified ReverbRAG model.
# - Builds the real RAF dataset and a DataLoader using your samplers.
# - Pulls ONE batch and runs a forward pass (no training).
# - Abstracts model choice from YAML: run.model = {neraf | avnerf}
#
# Usage:
#   python train.py --config configs/base.yml

import argparse
import os
import yaml
import torch
from torch.utils.data import DataLoader, get_worker_info
from collections import OrderedDict

from model import UnifiedReverbRAGModel, ModelConfig
from data import build_dataset
from dataloader import RandomSliceBatchSampler, EDCFullBatchSampler


def load_global_feat(root: str, scene: str, device: torch.device) -> torch.Tensor:
    feat_path = os.path.join(root, scene, "feats", "global.pt")
    if not os.path.isfile(feat_path):
        raise FileNotFoundError(f"Could not find visual features at: {feat_path}")
    feat = torch.load(feat_path, map_location=device)
    if isinstance(feat, dict) and "feat" in feat:
        feat = feat["feat"]
    feat = feat.float()
    if feat.ndim == 1:
        feat = feat.unsqueeze(0)  # [1,1024]
    return feat  # [1,1024] or [B,1024]


def _worker_init_fn(_):
    # Ensure each worker starts with a fresh, small LRU cache in RAFDataset
    info = get_worker_info()
    if info is not None:
        ds = info.dataset
        try:
            ds._cache.clear()
        except Exception:
            ds._cache = OrderedDict()


def build_loader(cfg, dataset):
    sampler_cfg = cfg.get("sampler", {})
    dataset_mode = sampler_cfg.get("dataset_mode", "full").lower()
    batching = sampler_cfg.get("batching", "random").lower()
    bs = int(sampler_cfg.get("batch_size", 4))
    shuffle = bool(sampler_cfg.get("shuffle", True))
    num_workers = int(sampler_cfg.get("num_workers", 0))
    use_persist = num_workers > 0

    if dataset_mode == "full":
        if batching == "edc_full":
            raise ValueError("edc_full batching requires dataset_mode='slice'.")
        return DataLoader(
            dataset,
            batch_size=bs,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=use_persist,
            worker_init_fn=_worker_init_fn,
            drop_last=False,
        ), dataset_mode

    # slice mode → use custom batch samplers
    if batching == "random":
        batch_sampler = RandomSliceBatchSampler(len(dataset), batch_size=bs, drop_last=False, shuffle=shuffle)
    elif batching == "edc_full":
        batch_sampler = EDCFullBatchSampler(
            ids=dataset.ids,
            max_frames=dataset.max_frames,
            batch_size_rirs=bs,   # counts RIRs; effective elements = bs * T
            drop_last=False,
            shuffle=shuffle,
        )
    else:
        raise ValueError(f"Unknown batching mode '{batching}'")

    loader = DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=use_persist,
        worker_init_fn=_worker_init_fn,
    )
    return loader, dataset_mode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    run = cfg.get("run", {})
    dbs = cfg.get("databases", {})
    sampler_cfg = cfg.get("sampler", {})

    baseline = run.get("model", "neraf").lower()      # "neraf" | "avnerf"
    database = run.get("database", "raf").lower()     # "raf" | "soundspaces"
    scene = run.get("scene", "FurnishedRoom")
    sample_rate = int(run.get("sample_rate", 48000))

    # Resolve dataset root
    if database == "raf":
        root = dbs["raf"]["root"]
    elif database == "soundspaces":
        root = dbs["soundspaces"]["root"]
    else:
        raise ValueError(f"Unknown database {database}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build dataset + dataloader
    dataset = build_dataset(cfg, split="train")
    loader, dataset_mode = build_loader(cfg, dataset)

    # Build model (abstracted by baseline)
    mcfg = ModelConfig(
        baseline=baseline,
        database=database,
        scene_root=root,
        scene_name=scene,
        sample_rate=sample_rate,
        W_field=1024,
        scene_aabb=dataset.scene_box.aabb
    )
    model = UnifiedReverbRAGModel(mcfg).to(device).eval()

    # --- pull one real batch ---
    batch = next(iter(loader))

    # Common pose inputs
    mic_xyz = batch["receiver_pos"].to(device).float()   # [B,3]
    src_xyz = batch["source_pos"].to(device).float()     # [B,3]
    head_dir = batch["orientation"].to(device).float()   # [B,3]
    stft = batch["stft"].to(device).float()            # [B,C,F,T]
    B = mic_xyz.shape[0]
    C = 1 if database == "raf" else 2

    # Time argument to model (raw indices; model will normalize)
    if dataset_mode == "full":
        T = int(dataset.max_frames)  # 60
        t_idx = torch.arange(T, device=device, dtype=torch.float32).view(1, T, 1).expand(B, T, 1)  # [B,T,1] = 0..T-1
    else:
        slice_t = batch["slice_t"].to(device).long().view(B, 1, 1)  # [B,1,1]
        T = 1
        t_idx = slice_t.float()                                     # raw index, model scales

    # Visual features:
    # - NeRAF mode: use global 1024-D features (as before).
    # - AV-NeRF mode: prefer per-pose features if present; fallback to global.pt.
    if baseline == "avnerf":
        # try per-pose RGB and/or depth vectors
        feats = [batch.get(k, None) for k in ("feat_rgb", "feat_depth")]
        visual_feat = torch.cat(
            [f.to(device).float() for f in feats],dim=-1)
    else:
        vis_1x = load_global_feat(root, scene, device)  # [1,1024]
        visual_feat = vis_1x.expand(B, -1).contiguous()  # [B,1024]

    # ---- Forward (no grad; smoke test only)
    with torch.no_grad():
        out = model(
            mic_xyz=mic_xyz,
            src_xyz=src_xyz,
            head_dir=head_dir,
            t_idx=t_idx,            # [B,T,1] (0..T-1); model handles normalization
            visual_feat=visual_feat,
        )

    # Expect shapes:
    # full   → [B, C, F, 60]
    # slice  → [B, C, F, 1]
    print("==== Dataset Smoke Test ====")
    print(f"Baseline     : {baseline}")
    print(f"Database     : {database}")
    print(f"Scene        : {scene}")
    print(f"Sample rate  : {sample_rate}")
    print(f"Mode         : {dataset_mode}")
    # Print batch semantics depending on the batching mode
    batching = sampler_cfg.get("batching", "random").lower()
    bs_cfg = int(sampler_cfg.get("batch_size", 4))
    if dataset_mode == "slice" and batching == "edc_full":
        print(f"Batch size   : {bs_cfg} RIRs  → effective rows = {B} slices (should be {bs_cfg}*{dataset.max_frames})")
    else:
        print(f"Batch size   : {B} items")

    print(f"Input shapes : mic={tuple(mic_xyz.shape)}, src={tuple(src_xyz.shape)}, "
          f"head={tuple(head_dir.shape)}, t_idx={tuple(t_idx.shape)}, visual={tuple(visual_feat.shape)}")
    print(f"Output shape : {tuple(out.shape)}   (expect [B={B}, C={C}, F, T={T}])")
    print(f"stft shape   : {tuple(stft.shape)}")
    assert out.ndim == 4 and out.shape[-1] == T and out.shape[1] == C, "Output shape mismatch"
    print("OK ✅")


if __name__ == "__main__":
    main()