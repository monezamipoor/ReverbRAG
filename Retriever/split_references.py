import os
import json
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from sklearn_extra.cluster import KMedoids

from retriever import FurnishedRoomSTFTDataset, compute_audio_distance, RIRRetrievalMLP


def load_splits(root_dir, split_file):
    path = os.path.join(root_dir, 'metadata', 'data-split-original.json')
    with open(path, 'r') as f:
        splits = json.load(f)
    return splits


def save_splits(root_dir, splits, out_name):
    path = os.path.join(root_dir, 'metadata', out_name)
    with open(path, 'w') as f:
        json.dump(splits, f, indent=2)
    print(f"Saved new splits JSON to {path}")


def build_global_dataset(root_dir, device):
    ds_all = FurnishedRoomSTFTDataset(
        root_dir, split='train', return_wav=True, mode='normal'
    )

    ids = ds_all.ids
    stfts, wavs, edcs_list, decays_list = [], [], [], []
    mic_list, src_list, rot_list = [], [], []

    for idx in tqdm(range(len(ds_all)), desc="Loading features"):
        item = ds_all[idx]
        stfts.append(item['stft'])          # [F,T]
        wavs.append(item['wav'])            # [T]
        edcs_list.append(item['edc'])       # [T_edc] or None
        decays_list.append(item['decay_feats'])  # [D] or None

        # needed for retriever embeddings
        mic_list.append(item['mic_pose'])
        src_list.append(item['source_pose'])
        rot_list.append(item['rot'])

    stfts = torch.stack(stfts, dim=0).to(device)
    wavs  = torch.stack(wavs,  dim=0).to(device)
    mic   = torch.stack(mic_list, dim=0).to(device)
    src   = torch.stack(src_list, dim=0).to(device)
    rot   = torch.stack(rot_list, dim=0).to(device)

    if all(e is not None for e in edcs_list):
        edcs = torch.stack(edcs_list, dim=0).to(device)
    else:
        edcs = None

    if all(d is not None for d in decays_list):
        decays = torch.stack(decays_list, dim=0).to(device)
    else:
        decays = None

    return ids, stfts, wavs, edcs, decays, mic, src, rot, ds_all.sample_rate

# ---------- Embedding backend helpers ----------

def compute_embeddings_all(ckpt_path, grid_vec_path, mic, src, rot, device):
    """Load ckpt+global grid, compute L2-normalized embeddings for all items."""
    ckpt  = torch.load(ckpt_path, map_location=device)
    model = RIRRetrievalMLP(**ckpt["model_config"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    grid_vec = torch.load(grid_vec_path, map_location=device).to(device)  # use_global_grid=True
    N = mic.size(0)
    with torch.no_grad():
        Gg  = grid_vec.unsqueeze(0).expand(N, -1).to(device)
        Z   = model(Gg, mic.to(device), src.to(device), rot.to(device))   # [N, D]
        Z   = F.normalize(Z, p=2, dim=1)
    return Z


def fps_select_embeddings(Z, n_refs):
    """Farthest-Point Sampling in cosine distance: d(i,j)=1 - z_i·z_j."""
    N = Z.size(0)
    device = Z.device
    first = torch.randint(0, N, (1,), device=device).item()
    selected = [first]
    min_dists = torch.full((N,), float('inf'), device=device)
    min_dists[first] = -1.0

    for _ in tqdm(range(1, n_refs), desc='FPS sampling (embeddings)'):
        last = selected[-1]
        sims = (Z @ Z[last].unsqueeze(1)).squeeze(1)   # [N]
        d    = 1.0 - sims
        min_dists = torch.minimum(min_dists, d)
        min_dists[selected] = -1.0
        nxt = int(min_dists.argmax().item())
        selected.append(nxt)
        min_dists[nxt] = -1.0
    return selected


def kmedoids_select_embeddings(Z, n_refs):
    """k-medoids on cosine distances from embeddings."""
    Z = F.normalize(Z, p=2, dim=1)
    S = Z @ Z.t()              # [N,N], cosine similarity
    D = (1.0 - S).detach().cpu().numpy().astype('float32')
    np.fill_diagonal(D, 0.0)
    km = KMedoids(n_clusters=n_refs, metric='precomputed', init='k-medoids++')
    labels = km.fit_predict(D)
    return km.medoid_indices_


def kmedoids_select(distance_memmap, n_refs):
    # distance_memmap is a (N,N) numpy array on disk
    km = KMedoids(n_clusters=n_refs, metric='precomputed', init='k-medoids++')
    labels = km.fit_predict(distance_memmap)
    medoid_indices = km.medoid_indices_
    return medoid_indices


def _cat_or_none(a, b):
    if a is None or b is None:
        return None
    return torch.cat([a, b], dim=0)


def fps_select(stfts, wavs, edcs, decays, n_refs, metric, batch_size, fs):
    """
    Farthest Point Sampling using correct single-set compute_audio_distance call.
    We compute distances from the current seed (1 sample) to a block by concatenating
    [seed; block] and then taking the cross row of the resulting [B×B].
    """
    N = stfts.size(0)
    device = stfts.device

    first = torch.randint(0, N, (1,), device=device).item()
    selected = [first]

    min_dists = torch.full((N,), float('inf'), device=device)
    min_dists[first] = -1.0

    for _ in tqdm(range(1, n_refs), desc='FPS sampling'):
        last = selected[-1]

        # Compute distances from single sample `last` to all in batches
        new_d = torch.full((N,), float('inf'), device=device)
        seed_stft   = stfts[last:last+1]
        seed_wav    = wavs[last:last+1] if wavs is not None else None
        seed_edc    = edcs[last:last+1] if edcs is not None else None
        seed_decay  = decays[last:last+1] if decays is not None else None

        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)

            block_stft  = stfts[start:end]
            block_wav   = wavs[start:end] if wavs is not None else None
            block_edc   = edcs[start:end] if edcs is not None else None
            block_decay = decays[start:end] if decays is not None else None

            # Concatenate [seed; block] and compute full [B×B]
            stft_ij  = torch.cat([seed_stft, block_stft], dim=0)
            wav_ij   = _cat_or_none(seed_wav, block_wav)
            edc_ij   = _cat_or_none(seed_edc, block_edc)
            decay_ij = _cat_or_none(seed_decay, block_decay)

            D_full = compute_audio_distance(
                stft=stft_ij,
                wavs=wav_ij,
                edc_curves=edc_ij,
                decay_feats=decay_ij,
                metric=metric,
                fs=fs
            )
            # Distances from the seed (row 0) to the block live in columns [1:1+block_size]
            d_block = D_full[0, 1:1 + (end - start)]
            new_d[start:end] = d_block

        min_dists = torch.minimum(min_dists, new_d)
        min_dists[selected] = -1.0

        nxt = int(min_dists.argmax().item())
        selected.append(nxt)
        min_dists[nxt] = -1.0

    return selected


def compute_full_distance_memmap(stfts, wavs, edcs, decays, metric, batch_size, memmap_path, fs):
    """
    Build the full [N×N] distance matrix using block concatenation.
    For i==j, compute within-block; for i<j, concatenate [block_i; block_j] once
    and slice the cross distances to fill both upper and lower triangles.
    """
    N = stfts.size(0)
    D = np.memmap(memmap_path, dtype='float32', mode='w+', shape=(N, N))

    device = stfts.device

    for i in tqdm(range(0, N, batch_size), desc='Building distance matrix'):
        end_i = min(i + batch_size, N)

        # Within-block (i,i)
        stft_i  = stfts[i:end_i]
        wav_i   = wavs[i:end_i] if wavs is not None else None
        edc_i   = edcs[i:end_i] if edcs is not None else None
        decay_i = decays[i:end_i] if decays is not None else None

        D_ii = compute_audio_distance(
            stft=stft_i, wavs=wav_i, edc_curves=edc_i, decay_feats=decay_i,
            metric=metric, fs=fs
        ).detach().cpu().numpy()
        D[i:end_i, i:end_i] = D_ii

        for j in range(i + batch_size, N, batch_size):
            end_j = min(j + batch_size, N)

            stft_j  = stfts[j:end_j]
            wav_j   = wavs[j:end_j] if wavs is not None else None
            edc_j   = edcs[j:end_j] if edcs is not None else None
            decay_j = decays[j:end_j] if decays is not None else None

            # One call over [block_i; block_j]
            stft_ij  = torch.cat([stft_i, stft_j], dim=0)
            wav_ij   = _cat_or_none(wav_i,   wav_j)
            edc_ij   = _cat_or_none(edc_i,   edc_j)
            decay_ij = _cat_or_none(decay_i, decay_j)

            D_full = compute_audio_distance(
                stft=stft_ij, wavs=wav_ij, edc_curves=edc_ij, decay_feats=decay_ij,
                metric=metric, fs=fs
            ).detach().cpu().numpy()

            bi = end_i - i
            bj = end_j - j
            cross = D_full[:bi, bi:bi + bj]  # distances i→j
            D[i:end_i, j:end_j] = cross
            D[j:end_j, i:end_i] = cross.T

    # mask diagonal to +inf (self matches)
    np.fill_diagonal(D, 0.0)
    return D


def build_reference_split(root_dir, n_refs, method, metric, batch_size, device, out_name, backend, ckpt_path, grid_vec_path):
    # load and index splits
    orig = load_splits(root_dir, 'data-split.json')
    id_to_split = {}
    for split, lst in orig.items():
        flat_ids = np.array(lst).squeeze().tolist()
        if isinstance(flat_ids, str):
            flat_ids = [flat_ids]
        for sid in flat_ids:
            id_to_split[sid] = split

    # load global tensors (now includes wavs/edc/decays)
    ids, stfts, wavs, edcs, decays, mic, src, rot, fs = build_global_dataset(root_dir, device)
    N = len(ids)

    if backend == 'embedding':
        # compute all embeddings once
        Z = compute_embeddings_all(ckpt_path, grid_vec_path, mic, src, rot, device)

        if method == 'kmeans':
            sel_idx = kmedoids_select_embeddings(Z, n_refs)
        else:  # 'minmax' (FPS)
            sel_idx = fps_select_embeddings(Z, n_refs)

    else:  # backend == 'metric'
        if method == 'kmeans':
            memmap_path = os.path.join(root_dir, 'metadata', 'distance.memmap')
            D = compute_full_distance_memmap(
                stfts, wavs, edcs, decays,
                metric=metric, batch_size=batch_size,
                memmap_path=memmap_path, fs=fs
            )
            sel_idx = kmedoids_select(D, n_refs)
        else:  # 'minmax'
            sel_idx = fps_select(
                stfts, wavs, edcs, decays,
                n_refs=n_refs, metric=metric, batch_size=batch_size, fs=fs
            )


    ref_ids = [ids[i] for i in sel_idx]

    # remove from original splits and add to 'reference'
    new_splits = {k: np.array(v).squeeze().tolist() for k, v in orig.items()}
    new_splits['reference'] = []
    for rid in ref_ids:
        spl = id_to_split[rid]
        if rid in new_splits[spl]:
            new_splits[spl].remove(rid)
        new_splits['reference'].append(rid)

    save_splits(root_dir, new_splits, out_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, default='../data/RAF/EmptyRoom')
    parser.add_argument('--n_refs', type=int, default=25000)
    parser.add_argument('--method', choices=['kmeans', 'minmax'], default='minmax')
    # Expand to the metrics actually supported by compute_audio_distance
    parser.add_argument(
        '--metric',
        choices=['MAG', 'MAG_HELPER', 'MAG2', 'MSE', 'SC', 'LSD', 'ENV', 'SPL', 'EDC', 'T60', 'T60PCT', 'C50', 'EDT', 'DR'],
        default='SPL'
    )
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--out_name', type=str, default='data-split-references.json')
    parser.add_argument('--backend', choices=['metric','embedding'], default='embedding',
                    help='metric = audio-domain distance (compute_audio_distance); embedding = cosine in retriever space')
    parser.add_argument('--ckpt_path', type=str, default='./outputs/20250906_184700/rir_retrieval_model.ckpt')
    parser.add_argument('--grid_vec_path', type=str, default='./features.pt')
    args = parser.parse_args()
# 20250906_184700 20250812_204815
    device = torch.device(args.device if torch.cuda.is_available() and 'cuda' in args.device else 'cpu')
    build_reference_split(
        root_dir=args.root_dir,
        n_refs=args.n_refs,
        method=args.method,
        metric=args.metric,
        batch_size=args.batch_size,
        device=device,
        out_name=args.out_name,
        backend=args.backend, 
        ckpt_path=args.ckpt_path, 
        grid_vec_path=args.grid_vec_path
    )