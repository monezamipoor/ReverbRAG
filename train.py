# train.py
# Minimal smoke-test runner for Unified ReverbRAG NVAS model
# Usage:
#   python train.py --config configs/base.yml
#
# Reads YAML, loads {root}/{scene}/feats/global.pt (1024-D), builds model,
# creates a synthetic batch matching your loader shapes, runs forward pass,
# and prints the resulting tensor sizes.

import argparse
import os
import yaml
import torch

from model import UnifiedReverbRAGModel, ModelConfig


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
    return feat  # [1,1024] or [B,1024] if pre-batched


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg_yaml = yaml.safe_load(f)

    run = cfg_yaml.get("run", {})
    dbs = cfg_yaml.get("databases", {})

    baseline = run.get("model", "neraf").lower()      # "neraf" | "avnerf"
    database = run.get("database", "raf").lower()     # "raf" | "soundspaces"
    scene = run.get("scene", "FurnishedRoom")
    sample_rate = int(run.get("sample_rate", 48000))

    # Resolve database root
    if database == "raf":
        root = dbs["raf"]["root"]
    elif database == "soundspaces":
        root = dbs["soundspaces"]["root"]
    else:
        raise ValueError(f"Unknown database {database}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build model config and model
    mcfg = ModelConfig(
        baseline=baseline,
        database=database,
        scene_root=root,
        scene_name=scene,
        sample_rate=sample_rate,
        W_field=1024,  # matches your NeRAF field width symbol W
    )
    model = UnifiedReverbRAGModel(mcfg).to(device)
    model.eval()

    # Load mandatory visual feature (global.pt, 1024-D)
    vis_1x = load_global_feat(root, scene, device)  # [1,1024]

    # Create a small synthetic batch compatible with your loader shapes
    # From your notes: RAF slice mode often uses T=60
    T = 60
    if database == "raf":
        C = 1
    else:
        C = 2

    B = int(cfg_yaml.get("sampler", {}).get("batch_size", 4))
    B = max(1, min(B, 8))  # keep test small

    # Inputs
    mic_xyz = torch.rand(B, 3, device=device) * 2 - 1         # [-1,1]
    src_xyz = torch.rand(B, 3, device=device) * 2 - 1
    head_dir = torch.randn(B, 3, device=device)
    head_dir = head_dir / (head_dir.norm(dim=-1, keepdim=True) + 1e-8)

    t_norm = torch.linspace(0, 1, steps=T, device=device).view(1, T, 1).expand(B, T, 1)
    visual_feat = vis_1x.expand(B, -1).contiguous()           # [B,1024]

    # Forward
    with torch.no_grad():
        out = model(mic_xyz=mic_xyz,
                    src_xyz=src_xyz,
                    head_dir=head_dir,
                    t_norm=t_norm,
                    visual_feat=visual_feat)

    # Expect [B, C, F, T]
    print("==== Smoke Test ====")
    print(f"Baseline     : {baseline}")
    print(f"Database     : {database}")
    print(f"Scene        : {scene}")
    print(f"Sample rate  : {sample_rate}")
    print(f"Input shapes : mic={tuple(mic_xyz.shape)}, src={tuple(src_xyz.shape)}, head={tuple(head_dir.shape)}, t_norm={tuple(t_norm.shape)}, visual={tuple(visual_feat.shape)}")
    print(f"Output shape : {tuple(out.shape)}   (expect [B={B}, C={C}, F, T={T}])")

    # A couple of cheap assertions
    assert out.ndim == 4 and out.shape[0] == B and out.shape[-1] == T and out.shape[1] == C, "Output shape mismatch"
    print("OK ✅")


if __name__ == "__main__":
    main()