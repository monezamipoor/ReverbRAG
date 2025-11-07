# data.py
import os, json, pickle
from collections import OrderedDict
from typing import Dict, Any, List, Tuple
import numpy as np
from utils import _take_topk_refs
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
                 reverbrag_cfg: dict = None):
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
        
        # --- Sharded memmaps (STFT + EDC)
        self.mm_index_path = os.path.join(self.feats_dir, "index.json")
        self._st_shards = []
        self._edc_shards = []
        self._sid_ptr = None  # {sid: (shard_id, row)}

        if os.path.exists(self.mm_index_path):
            with open(self.mm_index_path, "r") as f:
                idx = json.load(f)
            # open shard files once (read-only)
            for sh in idx["shards"]:
                # STFT shards: (count, F, T) in float16
                self._st_shards.append(np.memmap(
                    sh["stft"], dtype=np.float16, mode="r", shape=(sh["count"], self.F_bins, self.max_frames)
                ))
                # EDC shards: optional; (count, T) in float32
                if "edc" in sh and os.path.exists(sh["edc"]):
                    self._edc_shards.append(np.memmap(
                        sh["edc"], dtype=np.float32, mode="r", shape=(sh["count"], self.max_frames)
                    ))
                else:
                    self._edc_shards.append(None)
            # sid → (shard_id, row)
            self._sid_ptr = {k: tuple(v) for k, v in idx["sid_to_ptr"].items()}
            
        # --- ReverbRAG references
        self.reverbrag_enabled = bool((reverbrag_cfg or {}).get("enabled", False))
        self.rag_topk = int((reverbrag_cfg or {}).get("k", 0))
        
        # ---- ReverbRAG reference bank ----
        self.ref_bank_ids: List[str] = []
        self.ref_bank_stft: torch.Tensor = torch.empty(0)   # [R, 1, F, 60] log-mags
        self._sid_to_refidx: Dict[str, List[int]] = {}

        if self.reverbrag_enabled and self.rag_topk > 0:
            ref_json_path = os.path.join(self.meta_dir, "references.json")
            if os.path.exists(ref_json_path):
                with open(ref_json_path, "r") as f:
                    ref_map_raw = json.load(f)              # may be flat or split dict
                # If file is {"train": {...}, "validation": {...}}, pick the current split if present
                ref_map = ref_map_raw.get(split, ref_map_raw) if isinstance(ref_map_raw, dict) else ref_map_raw

                # Collect unique reference IDs from the fixed reference set
                ref_set = set()
                for sid, lst in ref_map.items():
                    sidn = f"{int(sid):06d}" if str(sid).isdigit() else str(sid)
                    topk = _take_topk_refs(lst, self.rag_topk)
                    ref_set.update(topk)
                self.ref_bank_ids = sorted(list(ref_set))

                # Build id -> bank index
                ref_id2idx = {rid: i for i, rid in enumerate(self.ref_bank_ids)}

                # Preload STFT bank ONCE (static across epochs)
                # Stored as [R, 1, F, 60] log-magnitude (same convention as stft_full)
                bank = []
                for rid in self.ref_bank_ids:
                    st = self._mm_fetch_stft(rid)           # [1, F, 60]
                    bank.append(st)
                if len(bank):
                    self.ref_bank_stft = torch.stack(bank, dim=0)  # [R, 1, F, 60]

                # Map each sample id to its list of ref indices
                for sid, lst in ref_map.items():
                    sidn = f"{int(sid):06d}" if str(sid).isdigit() else str(sid)
                    ids = _take_topk_refs(lst, self.rag_topk)
                    arr = [(rid, ref_id2idx.get(rid, -1)) for rid in ids]
                    idxs = [j for (_, j) in arr if j >= 0]
                    while len(idxs) < self.rag_topk:
                        idxs.append(-1)
                    self._sid_to_refidx[sidn] = idxs

        # expose for samplers
        self.ref_bank_size = len(self.ref_bank_ids)
            
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

    # -------- Fast-path memmap fetchers (no cache) --------
    def _mm_fetch_stft(self, sid: str) -> torch.Tensor:
        """
        Return [1, F, 60] log-magnitude STFT tensor.
        Uses sharded memmap if available; else recomputes.
        """
        ptr = getattr(self, "_sid_ptr", None)
        if ptr is not None:
            k, row = ptr[sid]
            a = self._st_shards[k][row]        # (F, 60) float16
            x = torch.from_numpy(a.astype(np.float32))  # to fp32
            return x.unsqueeze(0)              # [1, F, 60]
        # fallback: recompute once (no caching)
        wav = self._load_wav(sid)
        return self._stft_full(wav)

    def _mm_fetch_edc(self, sid: str) -> torch.Tensor:
        """
        Return [60] EDC(dB) curve if available, else empty tensor.
        """
        if self._sid_ptr is not None and self._edc_shards:
            k, row = self._sid_ptr[sid]
            edc_mm = self._edc_shards[k]
            if edc_mm is not None:
                return torch.from_numpy(edc_mm[row].astype(np.float32))
        return torch.empty(0)

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
            wav_full = self._load_wav(sid)
            stft_full = self._mm_fetch_stft(sid)
            frgb, fdep = self._av_feats(sid)
            out = {
                "id": sid,
                "receiver_pos": rx, 
                "source_pos": tx, 
                "orientation": orientation,
                "wav": wav_full, 
                "stft": stft_full,
                "feat_rgb": frgb, 
                "feat_depth": fdep,
            }
        else:
            sid_idx, t = self._lin2pair[index]
            sid = self.ids[sid_idx]
            rx, tx, orientation = self._load_positions_and_ori(sid)
            wav_full = self._load_wav(sid)
            stft_full = self._mm_fetch_stft(sid)
            frgb, fdep = self._av_feats(sid)
            out = {
                "id": sid,
                "receiver_pos": rx, 
                "source_pos": tx, 
                "orientation": orientation,
                "slice_t": t,
                "wav": wav_full,
                "stft_slice": stft_full[:, :, t],
                "feat_rgb": frgb, 
                "feat_depth": fdep,
            }

        # attach top-K reference indices if available
        if self.reverbrag_enabled and self.rag_topk > 0:
            idxs = self._sid_to_refidx.get(sid, [])
            if idxs:
                out["ref_indices"] = torch.tensor(idxs, dtype=torch.long)   # [K] in bank
        return out


class SoundSpacesDataset(Dataset):
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("SoundSpacesDataset is not yet implemented.")


def build_dataset(opt: Dict[str, Any], split: str):
    model_kind = _load_yaml_like_dict(opt, "run", "model", default="neraf")
    db = _load_yaml_like_dict(opt, "run", "database", default="raf").lower()
    scene = _load_yaml_like_dict(opt, "run", "scene", default=None)
    dataset_mode = _load_yaml_like_dict(opt, "sampler", "dataset_mode", default="full")
    cache_capacity = _load_yaml_like_dict(opt, "sampler", "cache_capacity", default=8)
    rag_cfg = {
        "enabled": bool(_load_yaml_like_dict(opt, "run", "reverbrag", default=False)),
        "k": int(_load_yaml_like_dict(opt, "reverbrag", "k", default=0)),
    }

    if db == "raf":
        base = _load_yaml_like_dict(opt, "databases", "raf", "root", default=None)
        if base is None or scene is None:
            raise ValueError("databases.raf.root and run.scene are required for RAF.")
        scene_root = os.path.join(base, scene)
        sr = _load_yaml_like_dict(opt, "run", "sample_rate", default=48000)
        return RAFDataset(scene_root=scene_root, split=split, model_kind=model_kind,
                          sample_rate=sr, dataset_mode=dataset_mode,
                        reverbrag_cfg=rag_cfg)

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
