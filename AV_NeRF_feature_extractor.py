import os
import PIL
import glob
import pickle
import argparse
import numpy as np
from tqdm import tqdm
import torch
from torchvision.models import resnet18, ResNet18_Weights


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, required=True, help="Path to folder containing rgb/ and depth/ subfolders")
    parser.add_argument('--save-dir', type=str, required=True, help="Path to save features pickle file")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Load pretrained ResNet18
    weights = ResNet18_Weights.DEFAULT
    original_resnet = resnet18(weights=weights, progress=False).eval()
    layers = list(original_resnet.children())[:-1]
    model = torch.nn.Sequential(*layers)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model.to(device)
    transforms = weights.transforms()

    # Gather image paths
    rgb_list = sorted(glob.glob(os.path.join(args.data_dir, "rgb", "*.png")))
    depth_list = sorted(glob.glob(os.path.join(args.data_dir, "depth", "*.png")))

    features = {"rgb": [], "depth": []}

    # --- RGB features ---
    print(f"Extracting RGB features from {len(rgb_list)} images...")
    for rgb_path in tqdm(rgb_list, desc="RGB", unit="img"):
        rgb = PIL.Image.open(rgb_path).convert('RGB')
        rgb_tensor = transforms(rgb).unsqueeze(0)
        with torch.no_grad():
            feature = model(rgb_tensor.to(device)).squeeze().cpu().numpy()
        features["rgb"].append(feature)

    # --- Depth features ---
    print(f"Extracting Depth features from {len(depth_list)} images...")
    for depth_path in tqdm(depth_list, desc="Depth", unit="img"):
        depth = PIL.Image.open(depth_path).convert('RGB')
        depth_tensor = transforms(depth).unsqueeze(0)
        with torch.no_grad():
            feature = model(depth_tensor.to(device)).squeeze().cpu().numpy()
        features["depth"].append(feature)

    # Save everything in one file
    os.makedirs(args.save_dir, exist_ok=True)
    out_path = os.path.join(args.save_dir, "feats.pkl")
    pickle.dump(features, open(out_path, "wb"))
    print(f"\n✅ Saved features to: {out_path}")