import os
import json
import argparse
from evaluator import _EvalNPYCache, _pair_metrics_with_edc, compute_audio_distance
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchaudio
import math
from tqdm.auto import tqdm
from torch.utils.data import Dataset, DataLoader
from scipy.spatial.transform import Rotation as R
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.field_components.encodings import NeRFEncoding, SHEncoding
import matplotlib.pyplot as plt
import random
from datetime import datetime

import sys
sys.path.append('../NeRAF')
from NeRAF_helper import compute_t60 as _helper_compute_t60
from NeRAF_helper import evaluate_edt as _helper_evaluate_edt
from NeRAF_helper import evaluate_clarity as _helper_evaluate_clarity


# Add reproducibility
def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ---------------- Dataset ----------------
class FurnishedRoomSTFTDataset(Dataset):
    def __init__(self, root_dir, split='train', sample_rate=48000, return_wav=False, mode="normal"):
        self.root_dir = root_dir
        self.data_dir = os.path.join(root_dir, 'data')
        self.return_wav = return_wav
        self.max_len_time = 0.32
        self.max_len = 60
        self.stft_center = True
        split_file = 'data-split-original.json' if mode=="normal" else 'data-split-references.json'
        with open(os.path.join(root_dir, 'metadata', split_file)) as f:
            all_splits = json.load(f)
        all_ids = []
        for subset_raw in all_splits.values():
            part = subset_raw[0] if isinstance(subset_raw[0], list) else subset_raw
            all_ids.extend(part)
        poses = []
        for sid in all_ids:
            pts = open(os.path.join(self.data_dir, sid, 'rx_pos.txt')).read().split(',')
            poses.append([float(x) for x in pts])
        poses = np.stack(poses)
        aabb = np.vstack([poses.min(0), poses.max(0)])
        aabb[0] -= 1.0; aabb[1] += 1.0
        self.scene_box = SceneBox(aabb=torch.tensor(aabb, dtype=torch.float32))
        print(f"[SceneBox] AABB: {self.scene_box.aabb}")
        # ——— now select the split you actually want ———
        if split == 'all':
            self.ids = all_ids
        elif mode!='normal' and split == 'train':
            # Combine 'train' and 'references' sets
            train_raw = all_splits['train']
            refs_raw = all_splits['reference']
            train_ids = train_raw[0] if isinstance(train_raw[0], list) else train_raw
            refs_ids = refs_raw[0] if isinstance(refs_raw[0], list) else refs_raw
            self.ids = train_ids + refs_ids
        else:
            raw = all_splits[split]
            self.ids = raw[0] if isinstance(raw[0], list) else raw

        self.id2idx = {sid: i for i, sid in enumerate(self.ids)}
        self.sample_rate = sample_rate
        if sample_rate == 48000:
            self.n_fft, self.win_length, self.hop_length = 1024, 512, 256
        elif sample_rate == 16000:
            self.n_fft, self.win_length, self.hop_length = 512, 256, 128
        else:
            raise ValueError(f"Unsupported sample rate {sample_rate}")
        self.stft = torchaudio.transforms.Spectrogram(
            n_fft=self.n_fft, win_length=self.win_length,
            hop_length=self.hop_length, power=None, center=self.stft_center, pad_mode='reflect'
        )
        # Load precomputed EDC curves (sid -> tensor[T_target])
        features_path = os.path.join(root_dir, "edc_decay_features.pt")
        if os.path.exists(features_path):
            self.edc_decay = torch.load(features_path)
        else:
            self.edc_decay = None



    def __len__(self):
        return len(self.ids)

    def compute_log_magnitude(self, wav: torch.Tensor) -> torch.Tensor:
        # 1) full STFT: (1, F, T_full)
        spec = self.stft(wav.unsqueeze(0))
        # 2) crop or pad time‐axis to exactly max_len frames
        T_full = spec.shape[2]
        if T_full > self.max_len:
            spec = spec[:, :, : self.max_len]
        elif T_full < self.max_len:
            # pad with the minimum magnitude so it matches RAFDataset convention
            minval = spec.abs().min()
            pad_amt = self.max_len - T_full
            spec = torch.nn.functional.pad(
                spec,
                (0, pad_amt),          # pad on the right
                mode='constant',
                value=minval
            )
        # 3) magnitude and log
        mag = spec.abs()
        return torch.log(mag + 1e-3).squeeze(0)  # → (F, max_len)

    def __getitem__(self, idx):
        sid = self.ids[idx]
        folder = os.path.join(self.data_dir, sid)
        mic = torch.tensor([float(x) for x in open(os.path.join(folder,'rx_pos.txt')).read().split(',')], dtype=torch.float32)
        vals = [float(x) for x in open(os.path.join(folder,'tx_pos.txt')).read().split(',')]
        quat = np.array(vals[:4], dtype=np.float32)
        src = torch.tensor(vals[4:], dtype=torch.float32)
        r = R.from_quat(quat)
        yaw_deg = float(r.as_euler('yxz', degrees=True)[0])
        yaw_rad = np.deg2rad(round(yaw_deg,0))
        rot3 = np.array([np.cos(yaw_rad),0.0,np.sin(yaw_rad)], dtype=np.float32)
        rot = torch.from_numpy((rot3+1.0)/2.0)

        mic = SceneBox.get_normalized_positions(mic, self.scene_box.aabb)
        src = SceneBox.get_normalized_positions(src, self.scene_box.aabb)
        wav, sr = torchaudio.load(os.path.join(folder,'rir.wav'))
        wav = wav.squeeze(0)
        if sr != self.sample_rate:
            wav = torchaudio.functional.resample(wav, orig_freq=sr, new_freq=self.sample_rate)
        max_samples = int(self.max_len_time * self.sample_rate)
        wav = wav[:max_samples]
        stft = self.compute_log_magnitude(wav)
        out = {'id': sid, 'mic_pose': mic, 'source_pose': src, 'rot': rot, 'stft': stft}
        if self.return_wav:
            out['wav'] = wav
        if self.edc_decay is not None and sid in self.edc_decay:
            out['edc'] = self.edc_decay[sid]['edc']            # [T_edc]
            out['decay_feats'] = self.edc_decay[sid]['decay_feats']  # [3] = [T60, C50, EDT]
        else:
            out['edc'] = None
            out['decay_feats'] = None
        return out

# ---------------- Model ----------------
class RIRRetrievalMLP(nn.Module):
    def __init__(self, grid_feat_dim=1024, pos_freqs=10, rot_levels=4, hidden_dims=[1024,512,256]):
        super().__init__()
        self.position_encoding = NeRFEncoding(in_dim=3, num_frequencies=pos_freqs,
                                              min_freq_exp=0.0, max_freq_exp=8.0,
                                              include_input=True)
        self.rot_encoding = SHEncoding(levels=rot_levels, implementation='tcnn')
        pe_dim = self.position_encoding.get_out_dim()
        r_dim = self.rot_encoding.get_out_dim()
        inp_dim = grid_feat_dim + 2*pe_dim + r_dim
        dims = [inp_dim] + hidden_dims
        layers = []
        for i in range(len(dims)-1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims)-2:
                layers.append(nn.LeakyReLU(0.1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, grid, mic, src, rot):
        me = self.position_encoding(mic)
        se = self.position_encoding(src)
        re = self.rot_encoding(rot)
        x = torch.cat([grid, me, se, re], dim=-1)
        out = self.mlp(x)
        return F.normalize(out, p=2, dim=-1)

# ---------------- Loss ----------------
def nt_xent_loss(z, pos_idx, tau=0.1):
    # uses cosine similarity on normalized embeddings
    B = z.size(0)
    sim = torch.matmul(z, z.t()) / tau
    mask = torch.eye(B, device=z.device).bool()
    sim = sim.masked_fill(mask, float('-inf'))
    logp = F.log_softmax(sim, dim=1)
    return -logp[torch.arange(B, device=z.device), pos_idx].mean()

# ---------------- Soft-label NT-Xent ----------------
def nt_xent_loss_soft(z, D_batch, tau=0.1, t_m=0.5, eps=1e-8):
    """
    z:       [B, d] L2-normalized embeddings
    D_batch: [B, B] distances for this batch block (e.g., from MIXED or a single metric)
    tau:     similarity temperature (same as your hard NT-Xent)
    t_m:     metric temperature for turning distances into soft targets
    """
    B = z.size(0)
    # logits from similarities
    sim = torch.matmul(z, z.t()) / tau                      # [B,B]
    mask_eye = torch.eye(B, device=z.device, dtype=torch.bool)
    sim = sim.masked_fill(mask_eye, float('-inf'))          # no self-match
    logp = F.log_softmax(sim, dim=1)                        # [B,B]

    # soft targets q_ij ∝ exp(-D_ij / t_m), with q_ii = 0, and ignoring infs
    D = D_batch.to(z.device)
    q = torch.exp((-D) / t_m)                               # [B,B]
    q = q.masked_fill(mask_eye | ~torch.isfinite(D), 0.0)   # kill diag and infs

    row_sum = q.sum(dim=1, keepdim=True)                    # [B,1]
    # handle rows that sum to zero (e.g., all inf): fall back to hard argmin
    zero_rows = (row_sum <= eps).squeeze(1)                 # [B]
    if zero_rows.any():
        # put all mass on argmin finite distance
        D_safe = D.clone()
        D_safe = D_safe.masked_fill(mask_eye | ~torch.isfinite(D_safe), float('inf'))
        hard_pos = D_safe.argmin(dim=1)                     # [B]
        q[zero_rows] = 0.0
        q[torch.where(zero_rows)[0], hard_pos[zero_rows]] = 1.0
        row_sum = q.sum(dim=1, keepdim=True)

    q = q / (row_sum + eps)

    # cross-entropy with soft targets
    loss = -(q * logp).sum(dim=1).mean()
    return loss

MIXED_DIST = None
MIXED_IDS  = None

def metric_list_to_name(metric_weights):
    """Convert ['MAG',0.3,'SPL',0.1,...] → 'MAG_0.300__SPL_0.100__...'."""
    parts = []
    for m, w in zip(metric_weights[0::2], metric_weights[1::2]):
        parts.append(f"{m}_{w:.3f}")
    return "__".join(parts)

def get_or_create_mixed_dist(dataloader, metric_weights, cache_dir="mixed_cache", device="cpu", eps=1e-6):
    os.makedirs(cache_dir, exist_ok=True)
    fname = metric_list_to_name(metric_weights) + ".pt"
    path  = os.path.join(cache_dir, fname)

    if os.path.exists(path):
        print(f"[MIXED] Loading cached distance matrix from {path}")
        data = torch.load(path, map_location=device)
        return data['dist'], data['ids']

    print(f"[MIXED] No cache found for {fname}, computing...")

    from retriever import compute_audio_distance

    # Load dataset fully into RAM
    all_ids, all_stfts, all_wavs, all_edcs, all_decays = [], [], [], [], []
    print("[MIXED] Loading dataset to RAM...")
    for batch in tqdm(dataloader, desc="Loading dataset"):
        all_ids.extend(batch['id'])
        all_stfts.append(batch['stft'])
        all_wavs.append(batch.get('wav'))
        all_edcs.append(batch.get('edc'))
        all_decays.append(batch.get('decay_feats'))

    stfts = torch.cat(all_stfts, dim=0).cpu()
    wavs   = torch.cat([w for w in all_wavs if w is not None], dim=0).cpu() if all_wavs[0] is not None else None
    edcs   = torch.cat([e for e in all_edcs if e is not None], dim=0).cpu() if all_edcs[0] is not None else None
    decays = torch.cat([d for d in all_decays if d is not None], dim=0).cpu() if all_decays[0] is not None else None

    N = stfts.size(0)
    dist_mats = []

    for mname, w in zip(metric_weights[0::2], metric_weights[1::2]):
        print(f"[MIXED] Computing metric: {mname} (weight={w})")
        D = compute_audio_distance(stfts, wavs, edcs, decays, metric=mname, eps=eps).cpu()
        mask = D != float('inf')
        mean, std = D[mask].mean(), D[mask].std()
        D = (D - mean) / (std + eps)
        dist_mats.append(D * w)

    mixed_dist = sum(dist_mats)
    idx = torch.arange(N)
    mixed_dist[idx, idx] = float('inf')

    torch.save({'dist': mixed_dist, 'ids': all_ids}, path)
    print(f"[MIXED] Saved cached distance matrix to {path}")

    return mixed_dist, all_ids

def parse_metric_weights(s):
    if s is None:
        return None
    parts = s.split(',')
    assert len(parts) % 2 == 0, "metric_weights must be in metric,weight pairs"
    metrics = []
    for i in range(0, len(parts), 2):
        metrics.append(parts[i])
        metrics.append(float(parts[i+1]))
    return metrics

def eval_metric_names(args):
    """Return the list of metric names to evaluate on this epoch."""
    names = []
    mw = parse_metric_weights(args.metric_weights) if hasattr(args, 'metric_weights') else None
    if args.metric.upper() == 'MIXED':
        # Evaluate once per component in the mixed spec
        if mw is None:
            raise ValueError("MIXED metric requires --metric_weights")
        names = [m for m in mw[0::2]]
    elif getattr(args, 'loss_mode', 'single') == 'multi':
        # Even if training wasn't MIXED, user wants multi-loss: eval each specified metric
        if mw is None:
            raise ValueError("loss_mode=multi requires --metric_weights")
        names = [m for m in mw[0::2]]
    else:
        names = [args.metric]
    return names

# Config knobs (edit here; no CLI args necessary)
def _fmt_val(k, v):
    return f"{v:.2e}" if (k == "MSE" or k == "MAG2") else f"{v:.3f}"

def _fmt_val_signed(k, v):
    return f"{v:+.2e}" if (k == "MSE" or k == "MAG2") else f"{v:+.3f}"
# ---------------- Training Loop ----------------
def train(train_ds, val_ds, feats_map,
          grid_vec=None, use_global_grid=False,
          num_epochs=10, batch_size=2048, lr=1e-4,
          tau=0.1, K=3,
          audio_metric='MAG', sim_metric='cosine',
          scheduler_type='step', step_size=5, gamma=0.1, T_max=50,
          hidden_dims=[1024,512,256], args=None):
    # Create a timestamped output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = os.path.join('outputs', timestamp)
    os.makedirs(out_dir, exist_ok=True)

    # Save run parameters
    if args is not None:
        with open(os.path.join(out_dir, 'run_params.txt'), 'w') as f:
            json.dump(vars(args), f, indent=4)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4)
    EVAL_NPY_PATTERN = "../eval_results/emptyroom/emptyroom/renders/eval_*.npy"
    EVAL_NPY_MAX     = 200         # <- change if you want more/less
    EVAL_NPY_SEED    = 1337       # single seed for deterministic sampling
    npy_ds = FurnishedRoomSTFTDataset(
        root_dir=val_ds.root_dir, split="test", sample_rate=val_ds.sample_rate, return_wav=True, mode="normal")

    # Build the eval-npy cache ONCE using npy_ds (not val_ds)
    eval_npy_cache = _EvalNPYCache(npy_ds, sample_rate=npy_ds.sample_rate,
                                   pattern=EVAL_NPY_PATTERN, max_files=EVAL_NPY_MAX, seed=EVAL_NPY_SEED)
    
    metric_weights = parse_metric_weights(args.metric_weights)
    
    if args.metric.upper() == 'MIXED':
        assert metric_weights is not None, "Must provide --metric_weights for MIXED metric"
        # Precompute/load for train
        MIXED_DIST_TRAIN, MIXED_IDS_TRAIN = get_or_create_mixed_dist(
            DataLoader(train_ds, batch_size=args.batch_size, shuffle=False, num_workers=4),
            metric_weights,
            cache_dir="mixed_cache_train"
        )
        # Precompute/load for val
        MIXED_DIST_VAL, MIXED_IDS_VAL = get_or_create_mixed_dist(
            DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4),
            metric_weights,
            cache_dir="mixed_cache_val"
        )
        id2idx_train = {sid: i for i, sid in enumerate(MIXED_IDS_TRAIN)}
        id2idx_val   = {sid: i for i, sid in enumerate(MIXED_IDS_VAL)}
    
    model = RIRRetrievalMLP(grid_feat_dim=(grid_vec.shape[0] if use_global_grid else 0),
                            hidden_dims=hidden_dims).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = (torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=step_size, gamma=gamma)
                 if scheduler_type=='step'
                 else torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max))

    history = {
    'train_loss':[], 'train_map1':[], 'train_map5':[], 'train_acc1':[], 'train_acc5':[], 'train_mean_rank':[],
    'val_loss':[],   'val_map1':[],   'val_map5':[],   'val_acc1':[],   'val_acc5':[],   'val_mean_rank':[]}


    best_mean_rank = float('inf')  # Track best (lowest) validation mean rank
    best_model_path = os.path.join(out_dir, 'rir_retrieval_model.ckpt')
    best_accum_map = float("-inf")
    best_accum_map_epoch = -1
    
    for epoch in range(1, num_epochs+1):
        model.train()
        t_loss = 0
        t_map1 = 0  # hits@1
        t_map5 = 0  # hits@5
        t_acc1 = 0  # == hits@1
        t_acc5 = 0  # == hits@5
        t_rank_sum = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs} [Train]"):
            # move to device
            batch = {k:(v.to(device) if torch.is_tensor(v) else v) for k,v in batch.items()}
            B = batch['stft'].size(0)
            G = (grid_vec.unsqueeze(0).expand(B,-1).to(device)
                 if use_global_grid else torch.stack([feats_map[i] for i in batch['id']]).to(device))
            Z = model(G, batch['mic_pose'], batch['source_pose'], batch['rot'])
            if args.metric.upper() == 'MIXED':
                # get global indices for this batch (CPU tensor is fine)
                batch_idx_cpu = torch.tensor([id2idx_train[sid] for sid in batch['id']], dtype=torch.long)

                # slice CPU big matrix, then move the small BxB block to the model device
                D_a = MIXED_DIST_TRAIN.index_select(0, batch_idx_cpu).index_select(1, batch_idx_cpu).to(device)

                # compute pos ON THE SAME DEVICE as ranks/Z
                pos = D_a.argmin(dim=1)  # stays on `device`
            else:
                D_a = compute_audio_distance(
                stft=batch['stft'],
                wavs=batch.get('wav'),
                edc_curves=batch.get('edc'),
                decay_feats=batch.get('decay_feats'),
                metric=audio_metric
                )
                
            # === Choose positive targets & loss ===
            if args.loss_mode == 'single':
                
                pos = D_a.argmin(dim=1)

                if args.loss_type == 'soft':
                    loss = nt_xent_loss_soft(Z, D_a, tau=args.tau, t_m=args.t_m)
                else:
                    loss = nt_xent_loss(Z, pos, tau=args.tau)

            else:  # args.loss_mode == 'multi'
                # Use multiple metrics/losses on the same embeddings
                metric_weights = parse_metric_weights(args.metric_weights)
                assert metric_weights is not None and len(metric_weights) >= 2, \
                    "multi-loss needs --metric_weights like 'EDC,0.6,SPL,0.4'"

                # Build per-metric distance blocks (NO normalization) on-the-fly
                D_list, W_list = [], []
                for mname, w in zip(metric_weights[0::2], metric_weights[1::2]):
                    Dm = compute_audio_distance(
                        stft=batch['stft'],
                        wavs=batch.get('wav'),
                        edc_curves=batch.get('edc'),
                        decay_feats=batch.get('decay_feats'),
                        metric=mname
                    ).to(Z.device)
                    D_list.append(Dm)
                    W_list.append(float(w))

                # Weighted sum of per-metric losses (no z-score)
                loss = 0.0
                if args.loss_type == 'soft':
                    for Dm, w in zip(D_list, W_list):
                        loss = loss + w * nt_xent_loss_soft(Z, Dm, tau=args.tau, t_m=args.t_m)
                else:
                    for Dm, w in zip(D_list, W_list):
                        pos_m = Dm.argmin(dim=1)
                        loss = loss + w * nt_xent_loss(Z, pos_m, tau=args.tau)

                # For reporting ranks, use a *raw* weighted sum (no normalization)
                D_rawmix = torch.zeros_like(D_list[0])
                for Dm, w in zip(D_list, W_list):
                    D_rawmix = D_rawmix + w * Dm
                pos = D_rawmix.argmin(dim=1)

            optimizer.zero_grad(); loss.backward(); optimizer.step()

            # metrics
            # metrics
            if sim_metric=='cosine':
                simm = Z @ Z.t(); simm.fill_diagonal_(-1); ranks = simm.argsort(dim=1, descending=True)
            else:
                distm = torch.cdist(Z,Z,p=2); distm.fill_diagonal_(float('inf')); ranks = distm.argsort(dim=1)

            hits1 = (ranks[:, 0] == pos).float().sum().item()
            hits5 = (ranks[:, :5] == pos.unsqueeze(1)).any(dim=1).float().sum().item()

            ranks_pos = (ranks == pos.unsqueeze(1)).nonzero(as_tuple=True)[1]

            t_loss     += loss.item()
            t_map1     += hits1
            t_map5     += hits5
            t_acc1     += hits1           # acc@1 == hits@1 / N
            t_acc5     += hits5           # acc@5 == hits@5 / N
            t_rank_sum += ranks_pos.sum().item()

        n = len(train_loader.dataset)
        history['train_loss'].append(t_loss/len(train_loader))
        history['train_map1'].append(t_map1/n)
        history['train_map5'].append(t_map5/n)
        history['train_acc1'].append(t_acc1/n)
        history['train_acc5'].append(t_acc5/n)
        history['train_mean_rank'].append(t_rank_sum/n)


        # Validation
        model.eval()
        eval_names = eval_metric_names(args)  # which metrics to print on this epoch
        _last_Z_gallery = None
        _last_all_val_ids = None

        last_metric_mean_rank = None  # for saving by "last one"
        for mi, metric_name in enumerate(eval_names):
            v_loss = 0
            v_map1 = 0
            v_map5 = 0
            v_acc1 = 0
            v_acc5 = 0
            v_rank_sum = 0
            all_val_ids = []
            all_val_Z   = []
            with torch.no_grad():
                for batch in tqdm(val_loader, desc=f"Epoch {epoch}/{num_epochs} [Val:{metric_name}]"):
                    batch = {k:(v.to(device) if torch.is_tensor(v) else v) for k,v in batch.items()}
                    B = batch['stft'].size(0)
                    G = (grid_vec.unsqueeze(0).expand(B,-1).to(device)
                         if use_global_grid else torch.stack([feats_map[i] for i in batch['id']]).to(device))
                    Z = model(G, batch['mic_pose'], batch['source_pose'], batch['rot'])
                    all_val_ids.extend(batch['id'])
                    all_val_Z.append(Z.detach().cpu())

                    # Build D_a for THIS metric_name (no normalization, no mixed cache)
                    D_a = compute_audio_distance(
                        stft=batch['stft'],
                        wavs=batch.get('wav'),
                        edc_curves=batch.get('edc'),
                        decay_feats=batch.get('decay_feats'),
                        metric=metric_name
                    )

                    pos = D_a.argmin(dim=1)

                    # Loss matches args.loss_type
                    if args.loss_type == 'soft':
                        loss = nt_xent_loss_soft(Z, D_a, tau=args.tau, t_m=args.t_m)
                    else:
                        loss = nt_xent_loss(Z, pos, tau=args.tau)

                    # ranks on the embedding space
                    # ranks on the embedding space
                    if sim_metric=='cosine':
                        simm = Z @ Z.t(); simm.fill_diagonal_(-1); ranks = simm.argsort(dim=1, descending=True)
                    else:
                        distm = torch.cdist(Z,Z,p=2); distm.fill_diagonal_(float('inf')); ranks = distm.argsort(dim=1)

                    hits1 = (ranks[:, 0] == pos).float().sum().item()
                    hits5 = (ranks[:, :5] == pos.unsqueeze(1)).any(dim=1).float().sum().item()
                    ranks_pos = (ranks == pos.unsqueeze(1)).nonzero(as_tuple=True)[1]

                    v_loss     += loss.item()
                    v_map1     += hits1
                    v_map5     += hits5
                    v_acc1     += hits1
                    v_acc5     += hits5
                    v_rank_sum += ranks_pos.sum().item()

                
                # remember this metric's gallery; we'll use the last one after the loop
            _last_Z_gallery = torch.cat(all_val_Z, dim=0) if len(all_val_Z) else None
            _last_all_val_ids = list(all_val_ids)
            
            m = len(val_loader.dataset)
            val_loss_epoch      = v_loss/len(val_loader)
            val_map1_epoch      = v_map1 / m
            val_map5_epoch      = v_map5 / m
            val_acc1_epoch      = v_acc1 / m
            val_acc5_epoch      = v_acc5 / m
            val_mean_rank_epoch = v_rank_sum / m

            history['val_loss'].append(val_loss_epoch)
            history['val_map1'].append(val_map1_epoch)
            history['val_map5'].append(val_map5_epoch)
            history['val_acc1'].append(val_acc1_epoch)
            history['val_acc5'].append(val_acc5_epoch)
            history['val_mean_rank'].append(val_mean_rank_epoch)

            print(
                f"           Val[{metric_name}] "
                f"Loss {val_loss_epoch:.4f}, "
                f"mAP1 {val_map1_epoch:.4f}, mAP5 {val_map5_epoch:.4f}, "
                f"acc@1 {val_acc1_epoch:.4f}, acc@5 {val_acc5_epoch:.4f}, "
                f"mean_rank {val_mean_rank_epoch:.2f}"
            )


            # Remember the last one's mean rank for saving
            last_metric_mean_rank = val_mean_rank_epoch

                # ───────── Eval-vs-NPY: run once per epoch, using the last built gallery ─────────
        if eval_npy_cache.records and (_last_Z_gallery is not None):
            Zg = _last_Z_gallery  # [N, d] on CPU
            id_list = _last_all_val_ids
            id2idx_val = {sid: i for i, sid in enumerate(id_list)}

            agg_pred = {k:0.0 for k in ('MSE','SPL','MAG','MAG2','EDC','T60','C50','EDT')}
            agg_retr = {k:0.0 for k in ('MSE','SPL','MAG','MAG2','EDC','T60','C50','EDT')}
            cnt_pred = 0
            cnt_retr = 0

            # For each cached NPY record (from TEST split), build a query embedding and compare to VAL gallery
            for rec in eval_npy_cache.records:
                # 1) pred vs GT: compute once, cached
                if rec['metrics_pred_vs_gt'] is None:
                    rec['metrics_pred_vs_gt'] = _pair_metrics_with_edc(
                        rec['stft_gt'], rec['wav_gt'],
                        rec['stft_pred'], rec['wav_pred'],
                        rec.get('edc_gt'), None,
                        fs=val_ds.sample_rate
                    )
                for k in agg_pred: agg_pred[k] += float(rec['metrics_pred_vs_gt'][k])
                cnt_pred += 1

                # 2) model retrieval vs GT: embed the TEST item as query, retrieve from VAL gallery
                #    (don’t try to find it inside VAL by id — they are different splits)
                test_item = npy_ds[npy_ds.id2idx[rec['id']]]
                if use_global_grid:
                    Gq = grid_vec.unsqueeze(0).to(device)
                else:
                    Gq = feats_map[test_item['id']].unsqueeze(0).to(device)

                with torch.no_grad():
                    Zq = model(Gq.to(device),
                            test_item['mic_pose'].unsqueeze(0).to(device),
                            test_item['source_pose'].unsqueeze(0).to(device),
                            test_item['rot'].unsqueeze(0).to(device)).detach().cpu()  # [1,d]

                sims = (Zq @ Zg.T).squeeze(0)     # [N] on CPU
                # there is no “self” in gallery (different split), so no need to mask diagonal
                j = int(torch.argmax(sims).item())
                rid = id_list[j]

                # fetch retrieved VAL sample tensors
                r_item = val_ds[val_ds.id2idx[rid]]

                mr = _pair_metrics_with_edc(
                    rec['stft_gt'], rec['wav_gt'],
                    r_item['stft'], r_item['wav'],
                    rec.get('edc_gt'), r_item.get('edc'),
                    fs=val_ds.sample_rate
                )
                for k in agg_retr: agg_retr[k] += float(mr[k])
                cnt_retr += 1

            if cnt_pred > 0 and cnt_retr > 0:
                avg_pred = {k: agg_pred[k]/cnt_pred for k in agg_pred}
                avg_retr = {k: agg_retr[k]/cnt_retr for k in agg_retr}
                deltas   = {k: (avg_retr[k] - avg_pred[k]) for k in agg_retr}

                order = ('SPL','MSE','MAG','MAG2','EDC','T60','C50','EDT')
                # print("\n  ── Eval-vs-NPY (averages over deterministic subset) ──")
                # print("     Pred vs GT:   " + "  ".join(f"{k}:{_fmt_val(k, avg_pred[k])}" for k in order))
                # print("     Model vs GT:  " + "  ".join(f"{k}:{_fmt_val(k, avg_retr[k])}" for k in order))
                print("     Δ(ref−pred):  " + "  ".join(f"{k}:{_fmt_val_signed(k, deltas[k])}" for k in order))
            else:
                print("\n  ── Eval-vs-NPY: no usable pairs this epoch (check splits/ids). ──")

        # keep train prints as they are
        print(
            f"Epoch {epoch}: "
            f"Train Loss {history['train_loss'][-1]:.4f}, "
            f"mAP1 {history['train_map1'][-1]:.4f}, mAP5 {history['train_map5'][-1]:.4f}, "
            f"acc@1 {history['train_acc1'][-1]:.4f}, acc@5 {history['train_acc5'][-1]:.4f}, "
            f"mean_rank {history['train_mean_rank'][-1]:.2f}"
        )


        # Step LR **once per epoch** (after all val metrics)
        scheduler.step()

        # Save model based on the *last metric's* mean rank (simplest rule as requested)
        if last_metric_mean_rank is not None and last_metric_mean_rank <= best_mean_rank:
            best_mean_rank = last_metric_mean_rank
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_mean_rank': best_mean_rank,
                'run_args': vars(args) if args is not None else None,
                'model_config': {
                    'grid_feat_dim': 1024,
                    'hidden_dims': hidden_dims,
                }
            }
            torch.save(checkpoint, best_model_path)


    # Plot and save metrics
    epochs = range(1, num_epochs+1)
    fig, axs = plt.subplots(2,2, figsize=(12,6))
    axs = axs.flatten()
# Loss
    axs[0].plot(epochs, history['train_loss'], label='Train')
    axs[0].plot(epochs, history['val_loss'],   label='Val')
    axs[0].set_title('Loss'); axs[0].legend()

    # mAP@1 and mAP@5
    axs[1].plot(epochs, history['train_map1'], label='Train mAP@1')
    axs[1].plot(epochs, history['val_map1'],   label='Val mAP@1')
    axs[1].plot(epochs, history['train_map5'], label='Train mAP@5')
    axs[1].plot(epochs, history['val_map5'],   label='Val mAP@5')
    axs[1].set_title('mAP@1 and mAP@5'); axs[1].legend()

    # Acc@1 and Acc@5
    axs[2].plot(epochs, history['train_acc1'], label='Train acc@1')
    axs[2].plot(epochs, history['val_acc1'],   label='Val acc@1')
    axs[2].plot(epochs, history['train_acc5'], label='Train acc@5')
    axs[2].plot(epochs, history['val_acc5'],   label='Val acc@5')
    axs[2].set_title('Acc@1 and Acc@5'); axs[2].legend()

    # Mean Rank
    axs[3].plot(epochs, history['train_mean_rank'], label='Train')
    axs[3].plot(epochs, history['val_mean_rank'],   label='Val')
    axs[3].set_title('Mean Rank'); axs[3].legend()

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'training_metrics_extended.png'))
    plt.close(fig)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='../data/RAF/EmptyRoom')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--metric', choices=['MAG','ENV','SNR','SC','LSD', 'T60', 'C50', 'EDT', 'SPL', 'MIXED'], default='MIXED')
    parser.add_argument('--metric_weights', type=str, help="Comma-separated metric and weight pairs", default='EDC,0.6,SPL,0.4')
    parser.add_argument('--sim_metric', choices=['cosine','l2'], default='cosine')
    parser.add_argument('--scheduler', choices=['step','cosine'], default='cosine')
    parser.add_argument('--step_size', type=int, nargs='+', default=[2,4])
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--T_max', type=int, default=180)
    parser.add_argument('--hidden_dims', type=int, nargs='+', default=[5029,2048,1024,1024,256])
    parser.add_argument('--K', type=int, default=10)
    parser.add_argument('--tau', type=float, default=0.05)
    parser.add_argument('--loss_type', choices=['hard','soft'], default='soft',
                    help='hard = standard NT-Xent; soft = soft-label NT-Xent from mixed/single metric')
    parser.add_argument('--t_m', type=float, default=0.1,
                    help='Metric temperature for soft targets (lower = sharper)')
    parser.add_argument('--loss_mode', choices=['single','multi'], default='single',
    help='single = one loss (MAG/EDC/... or MIXED); multi = sum of per-metric losses from --metric_weights (no normalization)')


    args = parser.parse_args()

    # Set fixed seed for reproducibility
    set_seed(42)

    train_ds = FurnishedRoomSTFTDataset(args.root, split='train', return_wav=True)
    val_ds   = FurnishedRoomSTFTDataset(args.root, split='validation', return_wav=True)
    feats = torch.load('./features.pt')
    use_global = isinstance(feats, torch.Tensor) and feats.ndim==1
    grid_vec = feats.to('cuda' if torch.cuda.is_available() else 'cpu') if use_global else None

    # Yes, DataLoader shuffle=True ensures different batch order each epoch
    train(train_ds, val_ds, feats_map=feats, grid_vec=grid_vec,
          use_global_grid=use_global,
          num_epochs=args.epochs,
          batch_size=args.batch_size,
          lr=args.lr,
          tau=args.tau,
          K=args.K,
          audio_metric=args.metric,
          sim_metric=args.sim_metric,
          scheduler_type=args.scheduler,
          step_size=args.step_size,
          gamma=args.gamma,
          T_max=args.T_max,
          hidden_dims=args.hidden_dims,
          args=args)