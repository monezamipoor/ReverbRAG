# render.py
import os, sys, glob, json, argparse, copy, shutil
import numpy as np
import yaml
import torch
from tqdm.auto import tqdm

# local imports from your project
from data import build_dataset
from model import UnifiedReverbRAGModel, ModelConfig
from trainer import Trainer, fs_to_stft_params
from main import build_eval_loader, build_visual_feat_builder
from evaluator import UnifiedEvaluator  # we will reuse its ISTFT + metric logic

def _find_yaml_next_to(ckpt_path: str):
    d = os.path.dirname(os.path.abspath(ckpt_path))
    cands = sorted(glob.glob(os.path.join(d, "*.yml")) + glob.glob(os.path.join(d, "*.yaml")))
    if not cands:
        raise FileNotFoundError(f"No YAML config found next to checkpoint: {d}")
    return cands[0]

@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, type=str)
    p.add_argument("--save-path", required=True, type=str)
    p.add_argument("--config", default=None, type=str, help="Optional explicit YAML path")
    p.add_argument("--test-split", default="test", choices=["test", "validation"])
    args = p.parse_args()

    ckpt = os.path.abspath(args.checkpoint)
    if not os.path.isfile(ckpt):
        raise FileNotFoundError(ckpt)

    cfg_path = args.config or _find_yaml_next_to(ckpt)
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    # --- run/db basics like main.py ---
    run = cfg.get("run", {}) or {}
    baseline = run.get("model", "neraf").lower()
    database = run.get("database", "raf").lower()
    scene = run.get("scene", "FurnishedRoom")
    fs = int(run.get("sample_rate", 48000))

    dbs = cfg.get("databases", {})
    if database == "raf":
        root = dbs["raf"]["root"]
    elif database == "soundspaces":
        root = dbs["soundspaces"]["root"]
    else:
        raise ValueError(f"Unknown database {database}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Build TRAIN briefly to get ref bank for RAG parity (same as main flow) ---
    train_cfg = copy.deepcopy(cfg)
    train_cfg.setdefault("sampler", {})
    train_cfg["sampler"]["dataset_mode"] = "slice"
    train_ds = build_dataset(train_cfg, split="train")

    ref_bank = getattr(train_ds, "ref_bank_stft", torch.empty(0))
    ref_bank_ids = getattr(train_ds, "ref_bank_ids", [])

    # --- Visual features (strict, like main.py) ---
    neraf_global_vec = None
    if baseline == "neraf":
        global_path = os.path.join(root, scene, "feats", "global.pt")
        obj = torch.load(global_path, map_location="cpu")
        if isinstance(obj, dict) and "global" in obj: obj = obj["global"]
        if not torch.is_tensor(obj) or obj.ndim != 1:
            raise RuntimeError(f"[NeRAF] {global_path} must be a 1-D tensor or dict['global'].")
        neraf_global_vec = obj.float()
    visual_builder = build_visual_feat_builder(baseline, neraf_global_vec)  # strict wiring. :contentReference[oaicite:1]{index=1}

    # --- TEST dataset/loader (FULL mode) ---
    test_cfg = copy.deepcopy(cfg)
    test_cfg.setdefault("sampler", {})
    test_cfg["sampler"]["dataset_mode"] = "full"
    test_ds = build_dataset(test_cfg, split=args.test_split)
    test_loader = build_eval_loader(test_cfg, test_ds)  # FULL loader. :contentReference[oaicite:2]{index=2}

    # Optional: id->index mapping
    id2idx = {sid: i for i, sid in enumerate(getattr(test_ds, "ids", []))} if hasattr(test_ds, "ids") else None

    # --- Model + Trainer skeleton (for forward path) ---
    stftp = fs_to_stft_params(fs)  # provides N_freq/win/hop. :contentReference[oaicite:3]{index=3}
    mcfg = ModelConfig(
        baseline=baseline, database=database,
        scene_root=root, scene_name=scene,
        sample_rate=fs, W_field=1024,
        scene_aabb=getattr(train_ds, "scene_box", None).aabb if hasattr(train_ds, "scene_box") else None,
        reverbrag=cfg.get("reverbrag", {}),
    )
    model = UnifiedReverbRAGModel(mcfg).to(device)
    trainer = Trainer(
        model=model, opt_cfg={}, run_cfg=run, device=device, baseline=baseline,
        visual_feat_builder=visual_builder, stft_params=stftp,
        ref_bank=ref_bank, ref_bank_ids=ref_bank_ids,
    )  # sets evaluator, etc. :contentReference[oaicite:4]{index=4}

    # --- Load checkpoint robustly ---
    print(f"[ckpt] loading {ckpt}")
    state = torch.load(ckpt, map_location="cpu")
    sd = state.get("model", state)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[ckpt] loaded with missing={len(missing)} unexpected={len(unexpected)}")

    # --- Output dirs (tidy layout) ---
    base_dir = os.path.abspath(args.save_path)
    renders_dir = os.path.join(base_dir, "renders")
    os.makedirs(renders_dir, exist_ok=True)
    # copy YAML for provenance
    try:
        dst_yaml = os.path.join(base_dir, os.path.basename(cfg_path))
        if os.path.abspath(dst_yaml) != os.path.abspath(cfg_path):
            shutil.copy2(cfg_path, dst_yaml)
    except Exception as e:
        print(f"[warn] cannot copy config: {e}")

    # --- Evaluator (we'll reuse its ISTFT pipeline for waveform) ---
    evaluator = UnifiedEvaluator(fs=fs, edc_bins=60, edc_dist="l1")  # same as trainer. :contentReference[oaicite:5]{index=5}

    # --- Rendering loop ---
    model.eval()
    counter = 0
    manifest = []
    for batch in tqdm(test_loader, desc="[render:test]"):
        # Forward in FULL mode — matches eval path in Trainer
        pred_log = trainer._forward_batch(batch, mode_full=True).float()  # [B,C,F,60]
        gt_log   = batch["stft"].to(device).float() if "stft" in batch else None

        # Build dataset indices
        batch_ids = batch.get("id", None)
        if batch_ids is not None and id2idx is not None:
            batch_idx = [id2idx[sid] for sid in batch_ids]
        elif batch_ids is not None:
            # if IDs are numeric strings
            try:
                batch_idx = [int(sid) for sid in batch_ids]
            except Exception:
                batch_idx = list(range(counter, counter + pred_log.shape[0]))
        else:
            batch_idx = list(range(counter, counter + pred_log.shape[0]))

        # Waveforms via the same ISTFT logic used by evaluator (avoids NaNs/zeros)
        # evaluator._waveforms_from_logmag expects [B,C,F,T] on the correct device. :contentReference[oaicite:6]{index=6}
        wav_pred = evaluator._waveforms_from_logmag(pred_log)  # -> np.ndarray [B,C,Tw]

        # Save each sample
        B = pred_log.shape[0]
        for b in range(B):
            # determine sample SID instead of numeric index
            if batch_ids is not None:
                sid = str(batch_ids[b])
            elif hasattr(test_ds, "ids") and batch_idx[b] < len(test_ds.ids):
                sid = test_ds.ids[batch_idx[b]]
            else:
                sid = f"{batch_idx[b]:06d}"

            rec = {
                "sid": sid,
                "pred_stft": pred_log[b, 0].detach().cpu().numpy() if pred_log.shape[1] >= 1 else pred_log[b].mean(0).cpu().numpy(),
                "pred_wav":  wav_pred[b, 0] if wav_pred.shape[1] >= 1 else wav_pred[b].mean(0),
            }
            out_path = os.path.join(renders_dir, f"{counter:06d}.npy")
            np.save(out_path, rec)
            manifest.append({"file": f"{counter:06d}.npy", "sid": sid})
            counter += 1

    with open(os.path.join(base_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    # --- Final evaluation (exact same metrics as during training eval) ---
    # We evaluate fresh tensors (not reloading .npy); evaluator uses the same GL + metric formulas. :contentReference[oaicite:7]{index=7}
    model.eval()
    agg, n_batches = None, 0
    for batch in tqdm(test_loader, desc="[eval]"):
        pred_log = trainer._forward_batch(batch, mode_full=True).float()
        gt_log   = batch["stft"].to(device).float()
        m = evaluator.evaluate(pred_log, gt_log)  # dict: stft/edc/t60/edt/c50/invalid_t60 :contentReference[oaicite:8]{index=8}
        if agg is None:
            agg = {k: float(v) for k, v in m.items()}
        else:
            for k, v in m.items():
                agg[k] += float(v)
        n_batches += 1
    if agg is None: agg = {}
    final = {k: (v / max(1, n_batches)) for k, v in agg.items()}
    with open(os.path.join(base_dir, "eval.json"), "w") as f:
        json.dump(final, f, indent=2)

    print(f"[done] wrote {counter} npy files → {renders_dir}")
    print(f"[done] eval → {os.path.join(base_dir, 'eval.json')}")

if __name__ == "__main__":
    main()