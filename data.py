# data.py
import os, json, pickle
from collections import OrderedDict
from typing import Dict, Any, List, Tuple
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
    return torch.from_numpy((vec + 1.0) / 2.0)

def _logmag(x: torch.Tensor) -> torch.Tensor:
    return torch.log(x.abs() + 1e-3)


class RAFDataset(Dataset):
    """
    dataset_mode = "full":
        -> { id, receiver_pos[3], source_pos[3], orientation[3], wav[1,T], stft[1,F,60], feat_rgb, feat_depth }

    dataset_mode = "slice":
        -> { id, receiver_pos[3], source_pos[3], orientation[3], slice_t, T,
             wav_slice[1,win’], stft_slice[1,F], feat_rgb, feat_depth }

    Per-worker LRU cache: stores (wav_full, stft_full) keyed by sid.
    """
    def __init__(self, scene_root: str, split: str, model_kind: str,
                 sample_rate: int = 48000, max_len_time: float = 0.32,
                 center_stft: bool = True, dataset_mode: str = "full",
                 cache_capacity: int = 8):
        super().__init__()
        assert dataset_mode in ("full", "slice")
        self.scene_root = scene_root
        self.data_dir = os.path.join(scene_root, "data")
        self.meta_dir = os.path.join(scene_root, "metadata")
        self.feats_dir = os.path.join(scene_root, "feats")
        self.model_kind = model_kind.lower()
        self.sample_rate = sample_rate
        self.max_len_time = max_len_time
        self.stft_center = center_stft
        self.dataset_mode = dataset_mode
        self.n_channels = 1  # RAF mono

        # --- splits
        split_path = os.path.join(self.meta_dir, "data-split.json")
        with open(split_path, "r") as f:
            splits = json.load(f)
        if split not in splits:
            raise ValueError(f"Split '{split}' not found in {split_path}")
        raw = splits[split]
        self.ids: List[str] = raw[0] if (isinstance(raw, list) and raw and isinstance(raw[0], list)) else raw
        self.ids = [f"{int(sid):06d}" if str(sid).isdigit() else str(sid) for sid in self.ids]

        # --- SceneBox
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
        aabb = np.vstack([poses.min(0), poses.max(0)]); aabb[0] -= 1.0; aabb[1] += 1.0
        self.scene_box = SceneBox(aabb=torch.tensor(aabb, dtype=torch.float32))
        self.id2idx = {sid: i for i, sid in enumerate(self.ids)}

        # --- STFT
        self.stft, self.F_bins, self.hop, self.win = _stft_transform(self.sample_rate, center=self.stft_center)
        self.max_frames = 60  # fixed

        # --- AV feats
        self.has_feats = False
        if self.model_kind == "avnerf":
            feats_pkl = os.path.join(self.feats_dir, "feats.pkl")
            if os.path.exists(feats_pkl):
                with open(feats_pkl, "rb") as f:
                    self.feats = pickle.load(f)
                self._feat_len = min(len(self.feats.get("rgb", [])), len(self.feats.get("depth", [])))
                self.has_feats = self._feat_len > 0
            else:
                self.feats = {"rgb": [], "depth": []}
        else:
            self.feats = {"rgb": [], "depth": []}

        # --- slice indexing (for dataset_mode='slice')
        if self.dataset_mode == "slice":
            self._lin2pair: List[Tuple[int, int]] = []
            for sid_idx in range(len(self.ids)):
                for t in range(self.max_frames):
                    self._lin2pair.append((sid_idx, t))

        # --- CACHING (per-worker)
        self._cache: OrderedDict[str, Tuple[torch.Tensor, torch.Tensor]] = OrderedDict()
        self._cache_capacity = int(cache_capacity)

    def __len__(self):
        return len(self.ids) if self.dataset_mode == "full" else len(self.ids) * self.max_frames

    def _feat_index_from_sid(self, sid: str) -> int:
        try: return int(sid)
        except Exception: return self.id2idx[sid]

    def _load_positions_and_ori(self, sid: str):
        folder = os.path.join(self.data_dir, sid)
        with open(os.path.join(folder, "rx_pos.txt"), "r") as f:
            rx = torch.tensor([float(x) for x in f.read().split(",")], dtype=torch.float32)
        with open(os.path.join(folder, "tx_pos.txt"), "r") as f:
            vals = [float(x) for x in f.read().split(",")]
        quat = np.array(vals[:4], dtype=np.float32)
        tx   = torch.tensor(vals[4:], dtype=torch.float32)
        orientation = _orientation_from_quat_xyzW(quat)
        return rx, tx, orientation

    def _load_wav(self, sid: str):
        wav, sr = torchaudio.load(os.path.join(self.data_dir, sid, "rir.wav"))  # [1, T]
        if sr != self.sample_rate:
            wav = torchaudio.functional.resample(wav, orig_freq=sr, new_freq=self.sample_rate)
        wav = wav[:, : int(self.max_len_time * self.sample_rate)]
        return wav  # [1, T]

    def _stft_full(self, wav: torch.Tensor):
        spec = self.stft(wav)  # [1, F, T_frames]
        T = spec.shape[-1]
        if T > self.max_frames:
            spec = spec[:, :, : self.max_frames]
        elif T < self.max_frames:
            minval = spec.abs().min()
            spec = torch.nn.functional.pad(spec, (0, self.max_frames - T), mode="constant", value=minval)
        return _logmag(spec)  # [1, F, 60]

    # -------- LRU cache helpers --------
    def _cache_get_full(self, sid: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return (wav_full[1,T], stft_full[1,F,60]) from per-worker cache, computing if missing.
        """
        if sid in self._cache:
            wav_full, stft_full = self._cache.pop(sid)  # mark as recently used
            self._cache[sid] = (wav_full, stft_full)
            return wav_full, stft_full

        # miss: compute once
        wav_full = self._load_wav(sid)
        stft_full = self._stft_full(wav_full)

        # insert with eviction
        self._cache[sid] = (wav_full, stft_full)
        if len(self._cache) > self._cache_capacity:
            self._cache.popitem(last=False)  # evict LRU

        return wav_full, stft_full
    # -----------------------------------

    def _slice_from_wav(self, wav: torch.Tensor, t: int):
        start = int(t * self.hop)
        end = start + self.win
        end = min(end, wav.shape[-1])
        start = max(0, end - self.win)
        return wav[:, start:end]  # [1, win’]

    def _av_feats(self, sid: str):
        if self.model_kind == "avnerf" and self.has_feats:
            fi = self._feat_index_from_sid(sid)
            rgb = self.feats["rgb"][fi] if fi < len(self.feats["rgb"]) else None
            dep = self.feats["depth"][fi] if fi < len(self.feats["depth"]) else None
            rgb = torch.from_numpy(rgb).float() if isinstance(rgb, np.ndarray) else torch.empty(0)
            dep = torch.from_numpy(dep).float() if isinstance(dep, np.ndarray) else torch.empty(0)
            return rgb, dep
        return torch.empty(0), torch.empty(0)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        if self.dataset_mode == "full":
            sid = self.ids[index]
            rx, tx, orientation = self._load_positions_and_ori(sid)
            wav_full, stft_full = self._cache_get_full(sid)  # <-- cached
            frgb, fdep = self._av_feats(sid)
            return {
                "id": sid,
                "receiver_pos": rx,
                "source_pos": tx,
                "orientation": orientation,
                "wav": wav_full,      # [1, T]
                "stft": stft_full,    # [1, F, 60]
                "feat_rgb": frgb,
                "feat_depth": fdep,
            }

        # slice mode
        sid_idx, t = self._lin2pair[index]
        sid = self.ids[sid_idx]
        rx, tx, orientation = self._load_positions_and_ori(sid)
        wav_full, stft_full = self._cache_get_full(sid)      # <-- cached
        wav_slice = self._slice_from_wav(wav_full, t)        # [1, win’]
        stft_slice = stft_full[:, :, t]                      # [1, F]
        frgb, fdep = self._av_feats(sid)
        return {
            "id": sid,
            "receiver_pos": rx,
            "source_pos": tx,
            "orientation": orientation,
            "slice_t": t,
            "wav_slice": wav_slice,      # [1, win’]
            "stft_slice": stft_slice,    # [1, F]
            "feat_rgb": frgb,
            "feat_depth": fdep,
        }


class SoundSpacesDataset(Dataset):
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("SoundSpacesDataset is not yet implemented.")


def build_dataset(opt: Dict[str, Any], split: str):
    model_kind = _load_yaml_like_dict(opt, "run", "model", default="neraf")
    db = _load_yaml_like_dict(opt, "run", "database", default="raf").lower()
    scene = _load_yaml_like_dict(opt, "run", "scene", default=None)
    dataset_mode = _load_yaml_like_dict(opt, "sampler", "dataset_mode", default="full")
    cache_capacity = _load_yaml_like_dict(opt, "sampler", "cache_capacity", default=8)

    if db == "raf":
        base = _load_yaml_like_dict(opt, "databases", "raf", "root", default=None)
        if base is None or scene is None:
            raise ValueError("databases.raf.root and run.scene are required for RAF.")
        scene_root = os.path.join(base, scene)
        sr = _load_yaml_like_dict(opt, "run", "sample_rate", default=48000)
        return RAFDataset(scene_root=scene_root, split=split, model_kind=model_kind,
                          sample_rate=sr, dataset_mode=dataset_mode,
                          cache_capacity=cache_capacity)

    elif db == "soundspaces":
        base = _load_yaml_like_dict(opt, "databases", "soundspaces", "root", default=None)
        if base is None or scene is None:
            raise ValueError("databases.soundspaces.root and run.scene are required for SoundSpaces.")
        scene_root = os.path.join(base, scene)
        sr = _load_yaml_like_dict(opt, "run", "sample_rate", default=16000)
        return SoundSpacesDataset(scene_root=scene_root, split=split, model_kind=model_kind,
                                  sample_rate=sr, dataset_mode=dataset_mode,
                                  cache_capacity=cache_capacity)
    else:
        raise ValueError(f"Unknown database '{db}'.")
