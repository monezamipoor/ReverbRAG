# main.py
from datetime import datetime
import os
import argparse
import copy
import shutil
import yaml
import random
import numpy as np
from typing import Dict, Any

import torch
from torch.utils.data import DataLoader, get_worker_info
from tqdm.auto import tqdm

# local
from data import build_dataset
from dataloader import RandomSliceBatchSampler, EDCFullBatchSampler
from model import UnifiedReverbRAGModel, ModelConfig
from trainer import Trainer, fs_to_stft_params

# --------- small utils ----------
def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def _worker_init_fn(_):
    # reset per-worker dataset cache if present
    info = get_worker_info()
    if info is not None and hasattr(info.dataset, "_cache"):
        try: info.dataset._cache.clear()
        except Exception: info.dataset._cache = {}

def _build_loader(cfg: Dict[str, Any], dataset, force_full: bool = False):
    sc = copy.deepcopy(cfg.get("sampler", {}))
    if force_full:
        sc["dataset_mode"] = "full"
        sc["batching"] = "random"

    dataset_mode = sc.get("dataset_mode", "full").lower()
    batching = sc.get("batching", "random").lower()
    bs = int(sc.get("batch_size", 4))
    shuffle = bool(sc.get("shuffle", True))
    num_workers = int(sc.get("num_workers", 0))
    use_persist = num_workers > 0

    if dataset_mode == "full":
        if batching == "edc_full":
            raise ValueError("edc_full batching requires dataset_mode='slice'.")
        loader = DataLoader(
            dataset, batch_size=bs, shuffle=shuffle, num_workers=num_workers,
            pin_memory=True, persistent_workers=use_persist, worker_init_fn=_worker_init_fn,
            drop_last=False,
        )
        return loader, dataset_mode

    # slice mode with custom samplers
    if batching == "random":
        batch_sampler = RandomSliceBatchSampler(len(dataset), batch_size=bs, drop_last=False, shuffle=shuffle)
    elif batching == "edc_full":
        batch_sampler = EDCFullBatchSampler(
            ids=dataset.ids, max_frames=dataset.max_frames,
            batch_size_rirs=bs, drop_last=False, shuffle=shuffle
        )
    else:
        raise ValueError(f"Unknown batching mode '{batching}'")

    loader = DataLoader(
        dataset, batch_sampler=batch_sampler, num_workers=num_workers,
        pin_memory=True, persistent_workers=use_persist, worker_init_fn=_worker_init_fn,
    )
    return loader, dataset_mode

def build_visual_feat_builder(baseline: str):
    """
    Returns a callable batch->visual_feat:
      - AV-NeRF: concat per-pose features if present (feat_rgb, feat_depth).
      - NeRAF  : use global 1024-D; dataset already expands to per-item in RAFDataset.
                 We simply fall back to (1x1024)->repeat if needed.
    """
    def _builder(batch, B):
        if baseline == "avnerf":
            feats = []
            if "feat_rgb" in batch and batch["feat_rgb"].numel() > 0: feats.append(batch["feat_rgb"])
            if "feat_depth" in batch and batch["feat_depth"].numel() > 0: feats.append(batch["feat_depth"])
            if len(feats) == 0:
                # fallback: zeros if per-pose features are absent
                return torch.zeros(B, 0)
            return torch.cat(feats, dim=-1)
        # NeRAF (global 1024 already expanded in dataset build or cached per sid)
        # If your dataset returns nothing, just zeros(1024).
        f = batch.get("feat_rgb", None)  # RAFDataset does not provide global.pt here
        if f is None or f.numel() == 0:
            return torch.zeros(B, 1024)
        return f
    return _builder
# ---------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # Seed
    seed = int(cfg.get("seed", 42))
    set_seed(seed)

    # Run config
    run = cfg.get("run", {})
    baseline = run.get("model", "neraf").lower()          # {neraf|avnerf}
    database = run.get("database", "raf").lower()         # {raf|soundspaces}
    scene = run.get("scene", "FurnishedRoom")
    fs = int(run.get("sample_rate", 48000))
    epochs = int(run.get("epochs", 20))

    # Databases roots
    dbs = cfg.get("databases", {})
    if database == "raf":
        root = dbs["raf"]["root"]
    elif database == "soundspaces":
        root = dbs["soundspaces"]["root"]
    else:
        raise ValueError(f"Unknown database {database}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --------- Build datasets/loaders ----------
    train_ds = build_dataset(cfg, split="train")
    train_loader, dataset_mode = _build_loader(cfg, train_ds, force_full=False)
    ref_bank = getattr(train_ds, "ref_bank_stft", torch.empty(0))
    ref_bank_ids = getattr(train_ds, "ref_bank_ids", [])

    # Val: force dataset_mode='full' as requested
    val_cfg = copy.deepcopy(cfg)
    val_cfg.setdefault("sampler", {})
    val_cfg["sampler"]["dataset_mode"] = "full"
    val_ds = build_dataset(val_cfg, split="validation")
    val_loader, _ = _build_loader(val_cfg, val_ds, force_full=True)

    # --------- Build model ----------
    mcfg = ModelConfig(
        baseline=baseline, database=database,
        scene_root=root, scene_name=scene,
        sample_rate=fs, W_field=1024, scene_aabb=train_ds.scene_box.aabb,
    )
    model = UnifiedReverbRAGModel(mcfg).to(device)

    # --------- Trainer ----------
    stft_params = fs_to_stft_params(fs)
    visual_builder = build_visual_feat_builder(baseline)
    trainer = Trainer(
        model=model,
        opt_cfg={},                 # not needed (kept for future)
        run_cfg=run,
        device=device,
        baseline=baseline,
        visual_feat_builder=visual_builder,
        stft_params=stft_params,
        ref_bank=ref_bank,          
        ref_bank_ids=ref_bank_ids,
    )

    # Derive run_name from config filename (no need for YAML run_name)
    cfg_path = args.config
    cfg_file = os.path.splitext(os.path.basename(cfg_path))[0]   # e.g., "base"
    date_tag = datetime.now().strftime("%Y%m%d-%H%M")
    run_name = cfg_file

    # Build an output directory: ./runs/<db>/<scene>/<date>_<runname>/
    save_root = run.get("save_root", "./runs")
    out_dir = os.path.join(save_root, database, scene, f"{date_tag}_{run_name}")
    os.makedirs(out_dir, exist_ok=True)

    # Copy the exact config into the run folder for reproducibility
    shutil.copy2(cfg_path, os.path.join(out_dir, f"{run_name}.yml"))

    # --------- W&B (optional) ----------
    wandb_run = None
    wb = cfg.get("wandb", {})
    try:
        import wandb
        if wb.get("project", None):
            wandb_run = wandb.init(project=wb["project"], name=run_name)
            wandb_run.config.update(cfg)
    except Exception as e:
        print(f"[wandb] disabled ({e})")

    # --------- Resume (optional) ----------
    resume_cfg = run.get("resume", {}) or {}
    resume_path = resume_cfg.get("checkpoint", None)
    load_opt = bool(resume_cfg.get("load_optimizer", False))
    lr_override = resume_cfg.get("lr_override", None)

    # --------- Train + Eval + internal checkpointing ----------
    trainer.train(
        train_loader, val_loader,
        dataset_mode=dataset_mode,
        epochs=epochs,
        wandb_run=wandb_run,
        save_dir=out_dir,
        save_every=int(run.get("save_every", 10)),
        cfg_copy_path=os.path.join(out_dir, f"{run_name}.yml"),
        resume_ckpt=resume_path,
        resume_load_optimizer=load_opt,
        resume_lr_override=lr_override,
    )


if __name__ == "__main__":
    main()