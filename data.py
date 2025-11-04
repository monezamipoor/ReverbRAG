# data.py
import os, json, pickle
from typing import Dict, Any
import numpy as np
import torch
from torch.utils.data import Dataset
import torchaudio
from scipy.spatial.transform import Rotation as R
from nerfstudio.data.scene_box import SceneBox


def _load_yaml_like_dict(opt: Dict[str, Any], *keys, default=None):
    cur = opt
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur

def _stft_transform(sr: int, center: bool = True):
    if sr == 48000:
        n_fft, win_length, hop_length = 1024, 512, 256
    elif sr == 16000:
        n_fft, win_length, hop_length = 512, 256, 128
    else:
        raise ValueError(f"Unsupported sample rate {sr}")
    return torchaudio.transforms.Spectrogram(
        n_fft=n_fft, win_length=win_length, hop_length=hop_length,
        power=None, center=center, pad_mode='reflect'
    ), n_fft // 2 + 1, hop_length, win_length

def _orientation_from_quat_xyzW(quat_xyzw: np.ndarray) -> torch.Tensor:
    r = R.from_quat(quat_xyzw)
    yaw_deg = float(r.as_euler('yxz', degrees=True)[0])
    yaw_deg = np.round(yaw_deg, 0)
    yaw_rad = np.deg2rad(yaw_deg)
    vec = np.array([np.cos(yaw_rad), 0.0, np.sin(yaw_rad)], dtype=np.float32)
    vec01 = (vec + 1.0) / 2.0
    return torch.from_numpy(vec01)

def _compute_log_mag(spec: torch.Tensor) -> torch.Tensor:
    return torch.log(spec.abs() + 1e-3)


class RAFDataset(Dataset):
    """
    Returns per sample:
      id: str
      receiver_pos: Tensor[3]      (SceneBox-normalized)
      source_pos:   Tensor[3]      (SceneBox-normalized)
      orientation:  Tensor[3]      (NeRAF-style)
      stft:         Tensor[C, F, T]  (C=1 for RAF)
      wav:          Tensor[C, T]     (C=1 for RAF)
      feat_rgb:     Tensor[...], empty tensor if model != avnerf or missing
      feat_depth:   Tensor[...], empty tensor if model != avnerf or missing
    """
    def __init__(self, scene_root: str, split: str, model_kind: str,
                 sample_rate: int = 48000, max_len_time: float = 0.32,
                 center_stft: bool = True):
        super().__init__()
        self.scene_root = scene_root
        self.data_dir = os.path.join(scene_root, "data")
        self.meta_dir = os.path.join(scene_root, "metadata")
        self.feats_dir = os.path.join(scene_root, "feats")
        self.model_kind = model_kind.lower()
        self.sample_rate = sample_rate
        self.max_len_time = max_len_time
        self.stft_center = center_stft
        self.n_channels = 1  # RAF is mono

        # Splits: expect metadata/data-split.json with train/valid/test/reference
        split_path = os.path.join(self.meta_dir, "data-split.json")
        with open(split_path, "r") as f:
            splits = json.load(f)
        if split not in splits:
            raise ValueError(f"Split '{split}' not found in {split_path}")
        raw = splits[split]
        self.ids = raw[0] if (isinstance(raw, list) and raw and isinstance(raw[0], list)) else raw
        self.ids = [f"{int(sid):06d}" if str(sid).isdigit() else str(sid) for sid in self.ids]

        # AABB over all IDs (for normalization)
        all_ids = []
        for v in splits.values():
            block = v[0] if (isinstance(v, list) and v and isinstance(v[0], list)) else v
            all_ids.extend(block)
        all_ids = [f"{int(sid):06d}" if str(sid).isdigit() else str(sid) for sid in all_ids]

        poses = []
        for sid in all_ids:
            with open(os.path.join(self.data_dir, sid, "rx_pos.txt"), "r") as fh:
                rx = [float(x) for x in fh.read().split(",")]
            poses.append(rx)
        poses = np.stack(poses, axis=0).astype(np.float32)
        aabb = np.vstack([poses.min(0), poses.max(0)])
        aabb[0] -= 1.0; aabb[1] += 1.0
        self.scene_box = SceneBox(aabb=torch.tensor(aabb, dtype=torch.float32))
        self.id2idx = {sid: i for i, sid in enumerate(self.ids)}

        # STFT config; fixed max frames 60 to match your code
        self.stft, self.F_bins, self.hop, self.win = _stft_transform(self.sample_rate, center=self.stft_center)
        self.max_frames = 60

        # ---- AV features (only if model == avnerf) ----
        self.has_feats = False
        if self.model_kind == "avnerf":
            feats_pkl = os.path.join(self.feats_dir, "feats.pkl")
            if os.path.exists(feats_pkl):
                with open(feats_pkl, "rb") as f:
                    self.feats = pickle.load(f)  # { "rgb": [np.ndarray...], "depth": [np.ndarray...] }
                # basic sanity
                self._feat_len = min(len(self.feats.get("rgb", [])), len(self.feats.get("depth", [])))
                self.has_feats = self._feat_len > 0
            else:
                self.feats = {"rgb": [], "depth": []}
        else:
            self.feats = {"rgb": [], "depth": []}

    def __len__(self): return len(self.ids)

    def _compute_log_magnitude_cropped(self, wav_1d: torch.Tensor) -> torch.Tensor:
        spec = self.stft(wav_1d.unsqueeze(0))  # (1, F, T) for mono input
        _, F, T = spec.shape
        if T > self.max_frames:
            spec = spec[:, :, :self.max_frames]
        elif T < self.max_frames:
            minval = spec.abs().min()
            pad = self.max_frames - T
            spec = torch.nn.functional.pad(spec, (0, pad), mode="constant", value=minval)
        # spec is (1, F, 60) for mono ⇒ add channel dim explicitly as C=1
        logmag = _compute_log_mag(spec)  # (1, F, 60)
        return logmag  # (C=1, F, T)

    def _feat_index_from_sid(self, sid: str) -> int:
        # Common case: ids are zero-padded numerics; feats are aligned by numeric index
        try:
            return int(sid)
        except Exception:
            # fallback: use position within this split (not ideal, but avoids crash)
            return self.id2idx[sid]

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sid = self.ids[idx]
        folder = os.path.join(self.data_dir, sid)

        # positions + orientation
        with open(os.path.join(folder, "rx_pos.txt"), "r") as f:
            rx = torch.tensor([float(x) for x in f.read().split(",")], dtype=torch.float32)
        with open(os.path.join(folder, "tx_pos.txt"), "r") as f:
            vals = [float(x) for x in f.read().split(",")]
        quat = np.array(vals[:4], dtype=np.float32)  # xyzW
        tx   = torch.tensor(vals[4:], dtype=torch.float32)

        orientation = _orientation_from_quat_xyzW(quat)
        rx_n = SceneBox.get_normalized_positions(rx, self.scene_box.aabb)
        tx_n = SceneBox.get_normalized_positions(tx, self.scene_box.aabb)

        # WAV (keep channel dim): torchaudio returns [1, T] for mono
        wav, sr = torchaudio.load(os.path.join(folder, "rir.wav"))  # [1, T]
        if sr != self.sample_rate:
            wav = torchaudio.functional.resample(wav, orig_freq=sr, new_freq=self.sample_rate)
        wav = wav[:, : int(self.max_len_time * self.sample_rate)]  # keep shape [1, T]

        # STFT with channel dim: [1, F, T]
        stft = self._compute_log_magnitude_cropped(wav.squeeze(0))

        # AV features
        if self.model_kind == "avnerf" and self.has_feats:
            fi = self._feat_index_from_sid(sid)
            feat_rgb_np = self.feats["rgb"][fi] if fi < len(self.feats["rgb"]) else None
            feat_depth_np = self.feats["depth"][fi] if fi < len(self.feats["depth"]) else None
            if isinstance(feat_rgb_np, np.ndarray):
                feat_rgb = torch.from_numpy(feat_rgb_np).float()
            else:
                feat_rgb = torch.empty(0)
            if isinstance(feat_depth_np, np.ndarray):
                feat_depth = torch.from_numpy(feat_depth_np).float()
            else:
                feat_depth = torch.empty(0)
        else:
            feat_rgb = torch.empty(0)
            feat_depth = torch.empty(0)

        return {
            "id": sid,
            "receiver_pos": rx_n,          # [3]
            "source_pos": tx_n,            # [3]
            "orientation": orientation,    # [3] in [0,1]
            "stft": stft,                  # [C=1, F, T]
            "wav": wav,                    # [C=1, T]
            "feat_rgb": feat_rgb,          # tensor or empty tensor
            "feat_depth": feat_depth,      # tensor or empty tensor
        }


class SoundSpacesDataset(Dataset):
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("SoundSpacesDataset is not yet implemented.")


def build_dataset(opt: Dict[str, Any], split: str):
    model_kind = _load_yaml_like_dict(opt, "run", "model", default="neraf")
    db = _load_yaml_like_dict(opt, "run", "database", default="raf").lower()
    scene = _load_yaml_like_dict(opt, "run", "scene", default=None)

    if db == "raf":
        base = _load_yaml_like_dict(opt, "databases", "raf", "root", default=None)
        if base is None or scene is None:
            raise ValueError("databases.raf.root and run.scene are required for RAF.")
        scene_root = os.path.join(base, scene)
        sr = _load_yaml_like_dict(opt, "run", "sample_rate", default=48000)
        return RAFDataset(scene_root=scene_root, split=split, model_kind=model_kind, sample_rate=sr)

    elif db == "soundspaces":
        base = _load_yaml_like_dict(opt, "databases", "soundspaces", "root", default=None)
        if base is None or scene is None:
            raise ValueError("databases.soundspaces.root and run.scene are required for SoundSpaces.")
        scene_root = os.path.join(base, scene)
        sr = _load_yaml_like_dict(opt, "run", "sample_rate", default=16000)
        return SoundSpacesDataset(scene_root=scene_root, split=split, model_kind=model_kind, sample_rate=sr)

    else:
        raise ValueError(f"Unknown database '{db}'.")