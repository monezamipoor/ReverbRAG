# render_and_analyze.py

import os, sys, glob, json, argparse, copy, shutil
import numpy as np
import yaml
import torch
from tqdm.auto import tqdm

# For headless plotting
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# local imports from your project
from data import build_dataset
from model import UnifiedReverbRAGModel, ModelConfig
from trainer import Trainer, fs_to_stft_params
from main import build_eval_loader, build_visual_feat_builder
from evaluator import UnifiedEvaluator  # ISTFT + metrics

def _find_yaml_next_to(ckpt_path: str):
    d = os.path.dirname(os.path.abspath(ckpt_path))
    cands = sorted(glob.glob(os.path.join(d, "*.yml")) + glob.glob(os.path.join(d, "*.yaml")))
    if not cands:
        raise FileNotFoundError(f"No YAML config found next to checkpoint: {d}")
    return cands[0]

# ---------- EDC helpers (from your diagnostics script) ----------

def edc_from_stft_mag(mag: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    mag: [F, T] linear magnitude
    Returns: Schroeder EDC curve [T] in dB (0 dB at t=0, decreasing).
    """
    power_t = (mag ** 2).sum(dim=0)                      # [T]
    total = power_t.sum() + eps
    rev_cumsum = torch.flip(torch.cumsum(torch.flip(power_t, dims=[0]), dim=0), dims=[0])
    edc = 10.0 * torch.log10(rev_cumsum / total + eps)
    return edc  # [T]

def normalize_edc(edc: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    edc: [T] in dB.
    Returns: normalized EDC [T], 0 dB at t=0 and z-scored over time.
    Focuses on *shape* of decay only.
    """
    edc_rel = edc - edc[0]               # 0 dB at t=0
    std = edc_rel.std().clamp_min(eps)
    return edc_rel / std

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
        if isinstance(obj, dict) and "global" in obj:
            obj = obj["global"]
        if not torch.is_tensor(obj) or obj.ndim != 1:
            raise RuntimeError(f"[NeRAF] {global_path} must be a 1-D tensor or dict['global'].")
        neraf_global_vec = obj.float()
    visual_builder = build_visual_feat_builder(baseline, neraf_global_vec)

    # --- TEST dataset/loader (FULL mode) ---
    test_cfg = copy.deepcopy(cfg)
    test_cfg.setdefault("sampler", {})
    test_cfg["sampler"]["dataset_mode"] = "full"
    test_ds = build_dataset(test_cfg, split=args.test_split)
    test_loader = build_eval_loader(test_cfg, test_ds)  # FULL loader.

    # Optional: id->index mapping
    id2idx = {sid: i for i, sid in enumerate(getattr(test_ds, "ids", []))} if hasattr(test_ds, "ids") else None

    # --- Model + Trainer skeleton (for forward path) ---
    stftp = fs_to_stft_params(fs)  # provides N_freq/win/hop.
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
    )  # sets evaluator, etc.

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

    # --- Evaluator (ISTFT + metric logic) ---
    evaluator = UnifiedEvaluator(fs=fs, edc_bins=60, edc_dist="l1")

    # --- Rendering loop: write npy packs with pred + GT STFT + pred wav ---
    model.eval()
    counter = 0
    manifest = []
    eps = 1e-6

    for batch in tqdm(test_loader, desc="[render:test]"):
        # Forward in FULL mode — matches eval path in Trainer
        pred_log = trainer._forward_batch(batch, mode_full=True).float().to(device)  # [B,C,F,T]
        gt_log = batch["stft"].to(device).float() if "stft" in batch else None

        # Waveforms via the same ISTFT logic used by evaluator
        wav_pred = evaluator._waveforms_from_logmag(pred_log)  # -> np.ndarray [B,C,Tw]

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

        B = pred_log.shape[0]
        for b in range(B):
            # determine sample SID instead of numeric index
            if batch_ids is not None:
                sid = str(batch_ids[b])
            elif hasattr(test_ds, "ids") and batch_idx[b] < len(test_ds.ids):
                sid = test_ds.ids[batch_idx[b]]
            else:
                sid = f"{batch_idx[b]:06d}"

            # choose first channel if multi-channel
            if pred_log.shape[1] >= 1:
                pred_stft_b = pred_log[b, 0].detach().cpu().numpy()
            else:
                pred_stft_b = pred_log[b].mean(0).detach().cpu().numpy()

            if gt_log is not None:
                if gt_log.shape[1] >= 1:
                    gt_stft_b = gt_log[b, 0].detach().cpu().numpy()
                else:
                    gt_stft_b = gt_log[b].mean(0).detach().cpu().numpy()
            else:
                gt_stft_b = None

            if wav_pred.shape[1] >= 1:
                wav_pred_b = wav_pred[b, 0]
            else:
                wav_pred_b = wav_pred[b].mean(0)

            rec = {
                "sid": sid,
                "pred_stft": pred_stft_b,
                "pred_wav": wav_pred_b,
            }
            # also store GT log-STFT in NeRAF-style key "data" for compatibility
            if gt_stft_b is not None:
                rec["data"] = gt_stft_b

            out_path = os.path.join(renders_dir, f"{counter:06d}.npy")
            np.save(out_path, rec)
            manifest.append({"file": f"{counter:06d}.npy", "sid": sid})
            counter += 1

    with open(os.path.join(base_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    # --- Final evaluation + diagnostics (STFT/EDC) ---
    model.eval()
    agg, n_batches = None, 0

    # accumulators for diagnostics (per-sample)
    stft_err_per_time = []   # [N, T]
    edc_err_per_time = []    # [N, T]
    slope_err_per_time = []  # [N, T-1]
    edc_gt_all = []          # [N, T]
    edc_pr_all = []          # [N, T]

    for batch in tqdm(test_loader, desc="[eval]"):
        pred_log = trainer._forward_batch(batch, mode_full=True).float().to(device)  # [B,C,F,T]
        gt_log = batch["stft"].to(device).float()                                    # [B,C,F,T]

        # metrics (same as training eval)
        m = evaluator.evaluate(pred_log, gt_log)  # dict: stft/edc/t60/edt/c50/invalid_t60
        if agg is None:
            agg = {k: float(v) for k, v in m.items()}
        else:
            for k, v in m.items():
                agg[k] += float(v)
        n_batches += 1

        # ----- Diagnostics (per-sample) -----
        B, C, F, T = gt_log.shape
        for b in range(B):
            gt_log_b = gt_log[b, 0] if C >= 1 else gt_log[b].mean(0)   # [F,T]
            pr_log_b = pred_log[b, 0] if C >= 1 else pred_log[b].mean(0)

            # STFT relative error
            stft_rel = (pr_log_b - gt_log_b).abs() / (gt_log_b.abs() + eps)  # [F,T]
            stft_err_t = stft_rel.mean(dim=0)                                # [T]
            stft_err_per_time.append(stft_err_t.unsqueeze(0))

            # EDC side
            gt_mag = torch.clamp(torch.exp(gt_log_b) - 1e-3, min=0.0)
            pr_mag = torch.clamp(torch.exp(pr_log_b) - 1e-3, min=0.0)

            edc_gt = edc_from_stft_mag(gt_mag)   # [T] dB
            edc_pr = edc_from_stft_mag(pr_mag)   # [T] dB

            edc_gt_n = normalize_edc(edc_gt)
            edc_pr_n = normalize_edc(edc_pr)

            edc_err_t = (edc_pr_n - edc_gt_n).abs()  # [T]
            edc_err_per_time.append(edc_err_t.unsqueeze(0))

            slope_gt = edc_gt_n[1:] - edc_gt_n[:-1]  # [T-1]
            slope_pr = edc_pr_n[1:] - edc_pr_n[:-1]
            slope_err = (slope_pr - slope_gt).abs()  # [T-1]
            slope_err_per_time.append(slope_err.unsqueeze(0))

            edc_gt_all.append(edc_gt_n.unsqueeze(0))
            edc_pr_all.append(edc_pr_n.unsqueeze(0))

    if agg is None:
        agg = {}
    final = {k: (v / max(1, n_batches)) for k, v in agg.items()}
    with open(os.path.join(base_dir, "eval.json"), "w") as f:
        json.dump(final, f, indent=2)

    print(f"[done] wrote {counter} npy files → {renders_dir}")
    print(f"[done] eval → {os.path.join(base_dir, 'eval.json')}")

    # ---------- Stack & average diagnostics, plot to analysis.png ----------
    if len(stft_err_per_time) > 0:
        stft_err_per_time = torch.cat(stft_err_per_time, dim=0)   # [N, T]
        edc_err_per_time = torch.cat(edc_err_per_time, dim=0)     # [N, T]
        slope_err_per_time = torch.cat(slope_err_per_time, dim=0) # [N, T-1]
        edc_gt_all = torch.cat(edc_gt_all, dim=0)                 # [N, T]
        edc_pr_all = torch.cat(edc_pr_all, dim=0)                 # [N, T]

        mean_stft_err = stft_err_per_time.mean(dim=0).cpu().numpy()
        mean_edc_err = edc_err_per_time.mean(dim=0).cpu().numpy()
        mean_slope_err = slope_err_per_time.mean(dim=0).cpu().numpy()

        print(f"[info] STFT error matrix : {stft_err_per_time.shape}")
        print(f"[info] EDC error matrix  : {edc_err_per_time.shape}")
        print(f"[info] Slope error matrix: {slope_err_per_time.shape}")

        fig, axes = plt.subplots(2, 2, figsize=(14, 9))
        (ax1, ax2), (ax3, ax4) = axes

        # 1) STFT relative error vs time
        ax1.plot(mean_stft_err)
        ax1.set_title("Mean Relative Log-STFT Error per Time Slice")
        ax1.set_xlabel("Time slice index")
        ax1.set_ylabel("Rel. STFT Error (arb.)")
        ax1.grid(True)

        # 2) EDC shape error vs time
        ax2.plot(mean_edc_err)
        ax2.set_title("Mean |EDC_pred - EDC_gt| per Time Slice (normalized EDC)")
        ax2.set_xlabel("Time slice index")
        ax2.set_ylabel("EDC Shape Error (arb.)")
        ax2.grid(True)

        # 3) Decay slope mismatch vs time
        ax3.plot(mean_slope_err)
        ax3.set_title("Mean Decay Slope Mismatch per Time Slice")
        ax3.set_xlabel("Time slice index (between t and t+1)")
        ax3.set_ylabel("|Δ slope_pred - slope_gt|")
        ax3.grid(True)

        # 4) Example normalized EDC curves (one sample)
        example_idx = 0
        ax4.plot(edc_gt_all[example_idx].cpu().numpy(), label="GT (norm)")
        ax4.plot(edc_pr_all[example_idx].cpu().numpy(), label="Pred (norm)")
        ax4.set_title(f"Normalized EDC Curves — Example {example_idx}")
        ax4.set_xlabel("Time slice index")
        ax4.set_ylabel("Norm. EDC (0 dB at t=0, z-scored)")
        ax4.legend()
        ax4.grid(True)

        plt.tight_layout()
        out_png = os.path.join(base_dir, "analysis.png")
        plt.savefig(out_png, dpi=150)
        plt.close(fig)
        print(f"[done] analysis plot → {out_png}")
    else:
        print("[warn] No diagnostics collected (empty test set?). Skipping analysis.png.")

if __name__ == "__main__":
    main()