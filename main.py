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
from typing import Optional
import math

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

# --- add these helper builders somewhere above main() ---

def build_train_loader(cfg, dataset):
    """
    Train loader:
      - Always SLICE mode for train.
      - If run.edc_loss > 0: force EDCFullBatchSampler with rirs_per_batch knob.
      - Else: RandomSliceBatchSampler with 'batch_size' (slices).
    Prints clear diagnostics and estimated steps/epoch.
    """
    sc   = copy.deepcopy(cfg.get("sampler", {}))
    cfg  = cfg.get("run", {}) or {}
    loss_cfg = cfg.get("losses", {}) or {}
    edc_w     = float(loss_cfg.get("edc", 0.0))
    env_rms_w = float(loss_cfg.get("env_rms", 0.0))
    # envelope_enabled = cfg["model"]["envelope"]["enabled"]
    shuffle      = bool(sc.get("shuffle", True))
    num_workers  = int(sc.get("num_workers", 0))
    use_persist  = num_workers > 0

    T = int(getattr(dataset, "max_frames", 60))
    total_slices = len(dataset)
    est_num_rirs = total_slices // T

    if (edc_w > 0.0) or (env_rms_w > 0.0):
        # EDC mode: use RIRs-per-batch knob
        rpb = int(sc.get("rirs_per_batch", 40))
        bs_slices = rpb * T
        batch_sampler = EDCFullBatchSampler(
            ids=dataset.ids,
            max_frames=T,
            batch_size_rirs=rpb,
            drop_last=False,
            shuffle=shuffle,
        )
        loader = DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=use_persist,
            worker_init_fn=_worker_init_fn,
        )
        steps = max(1, math.ceil(est_num_rirs / max(1, rpb)))
        print(f"[dataloader][TRAIN] mode=SLICE | sampler=EDCFullBatchSampler | T={T} | "
              f"rirs_per_batch={rpb} | bs={bs_slices} (slices) | workers={num_workers} | shuffle={shuffle}")
        print(f"[dataset][TRAIN] slices={total_slices} ≈ RIRs={est_num_rirs} | ~steps/epoch={steps}")
        return loader

    # Non-EDC mode: batch by raw slices
    bs_slices = int(sc.get("batch_size", 4096))
    batch_sampler = RandomSliceBatchSampler(
        len(dataset), batch_size=bs_slices, drop_last=False, shuffle=shuffle
    )
    loader = DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=use_persist,
        worker_init_fn=_worker_init_fn,
    )
    steps = max(1, math.ceil(total_slices / max(1, bs_slices)))
    print(f"[dataloader][TRAIN] mode=SLICE | sampler=RandomSliceBatchSampler | bs={bs_slices} (slices) | "
          f"workers={num_workers} | shuffle={shuffle}")
    print(f"[dataset][TRAIN] slices={total_slices} ≈ RIRs={est_num_rirs} | ~steps/epoch={steps}")
    return loader


def build_eval_loader(cfg, dataset):
    """
    Val/Test loader:
      - Always FULL mode (no EDC packing here).
      - Uses sampler.batch_size if present (default 2048).
    """
    sc  = copy.deepcopy(cfg.get("sampler", {}))
    bs  = int(sc.get("batch_size", 2048))
    shuffle = bool(sc.get("shuffle", True))
    num_workers = int(sc.get("num_workers", 0))
    use_persist = num_workers > 0

    loader = DataLoader(
        dataset,
        batch_size=bs,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=use_persist,
        worker_init_fn=_worker_init_fn,
        drop_last=False,
    )
    print(f"[dataloader][EVAL ] mode=FULL | bs={bs} | workers={num_workers} | shuffle={shuffle}")
    return loader

# ---------------------------------



def build_visual_feat_builder(baseline: str, neraf_global_vec: Optional[torch.Tensor]):
    """
    Strict visual feature wiring (no fallbacks):
      - AV-NeRF: require per-pose feats in batch (feat_rgb and/or feat_depth). If both missing -> raise.
      - NeRAF : require a global scene feature vector (e.g., 1024-D) passed in; expand to (B, D).
    """
    def _builder(batch, B):
        if baseline == "avnerf":
            feats = []
            f_rgb = batch.get("feat_rgb", None)
            f_dep = batch.get("feat_depth", None)
            if f_rgb is not None and f_rgb.numel() > 0:
                feats.append(f_rgb)
            if f_dep is not None and f_dep.numel() > 0:
                feats.append(f_dep)
            if len(feats) == 0:
                raise RuntimeError(
                    "[AV-NeRF] Missing per-pose visual features in batch: expected 'feat_rgb' and/or 'feat_depth'. "
                    "Check your RAFDataset build for this split."
                )
            return torch.cat(feats, dim=-1)

        # NeRAF path: must have a single global scene context vector loaded beforehand
        if neraf_global_vec is None:
            raise RuntimeError(
                "[NeRAF] Global scene feature vector is None. "
                "Expected to load from <root>/<scene>/feats/global.pt before training."
            )
        if neraf_global_vec.ndim != 1:
            raise RuntimeError(f"[NeRAF] global.pt must be 1-D (got {neraf_global_vec.shape}).")
        D = neraf_global_vec.numel()
        return neraf_global_vec.view(1, D).expand(B, D).to(batch["stft_slice"].device if "stft_slice" in batch else batch["stft"].device)

    return _builder


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
    
    # Fetch Visual Features for each baseline
    neraf_global_vec = None
    if baseline == "neraf":
        global_path = os.path.join(root, scene, "feats", "global.pt")  # e.g. ../NeRAF/data/RAF/EmptyRoom/feats/global.pt
        if not os.path.isfile(global_path):
            raise FileNotFoundError(f"[NeRAF] Required global feature file not found: {global_path}")
        obj = torch.load(global_path, map_location="cpu")
        if isinstance(obj, dict) and "global" in obj:
            obj = obj["global"]
        if not torch.is_tensor(obj):
            raise RuntimeError(f"[NeRAF] {global_path} must contain a 1-D tensor (or dict['global']). Got: {type(obj)}.")
        if obj.ndim != 1:
            raise RuntimeError(f"[NeRAF] {global_path} must be 1-D. Got shape: {tuple(obj.shape)}.")
        neraf_global_vec = obj.float()
        print(f"[features] Loaded NeRAF global feature {tuple(neraf_global_vec.shape)} from {global_path}")

    # --------- Build datasets/loaders ----------

    # TRAIN: always SLICE dataset; loader auto-picks EDC sampler if edc_loss>0
    train_cfg = copy.deepcopy(cfg)
    train_cfg.setdefault("sampler", {})
    train_cfg["sampler"]["dataset_mode"] = "slice"   # ensure dataset yields time slices
    train_ds = build_dataset(train_cfg, split="train")
    train_loader = build_train_loader(cfg, train_ds)  # our build_train_loader ignores dataset_mode and is EDC-aware

    # VAL: always FULL dataset + FULL loader (batch_size from val_cfg)
    val_cfg = copy.deepcopy(cfg)
    val_cfg.setdefault("sampler", {})
    val_cfg["sampler"]["dataset_mode"] = "full"
    val_cfg["sampler"]["batch_size"] = 2048
    val_ds = build_dataset(val_cfg, split="validation")
    val_loader = build_eval_loader(val_cfg, val_ds)   # <-- pass val_cfg, not cfg

    # (Optional) TEST: same as VAL if you have it
    # test_cfg = copy.deepcopy(val_cfg)
    # test_ds = build_dataset(test_cfg, split="test")
    # test_loader = build_eval_loader(test_cfg, test_ds)

    # Reference bank (from the train dataset)
    ref_bank = getattr(train_ds, "ref_bank_stft", torch.empty(0))
    ref_bank_ids = getattr(train_ds, "ref_bank_ids", [])


    # --------- Build model ----------
    mcfg = ModelConfig(
        baseline=baseline, database=database,
        scene_root=root, scene_name=scene,
        sample_rate=fs, W_field=1024, scene_aabb=train_ds.scene_box.aabb,
        opt=cfg.get("model", {}),
    )

    model = UnifiedReverbRAGModel(mcfg).to(device)

    # --------- Trainer ----------
    stft_params = fs_to_stft_params(fs)
    visual_builder = build_visual_feat_builder(baseline, neraf_global_vec)
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