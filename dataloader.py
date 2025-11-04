# dataloader.py
import math
import random
from typing import Iterator, List, Sequence, Tuple, Optional
from torch.utils.data import Sampler, BatchSampler


class RandomSliceBatchSampler(BatchSampler):
    """
    For dataset_mode='slice': draw individual slice indices at random.
    batch_size counts SLICES.
    """
    def __init__(self, dataset_len: int, batch_size: int, drop_last: bool, shuffle: bool = True):
        self.dataset_len = dataset_len
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        self._indices = list(range(dataset_len))

    def __iter__(self) -> Iterator[List[int]]:
        idx = self._indices.copy()
        if self.shuffle:
            random.shuffle(idx)
        for i in range(0, len(idx), self.batch_size):
            batch = idx[i:i + self.batch_size]
            if len(batch) == self.batch_size or not self.drop_last:
                yield batch

    def __len__(self) -> int:
        n = math.ceil(self.dataset_len / self.batch_size)
        return n if not self.drop_last else self.dataset_len // self.batch_size


class EDCFullBatchSampler(BatchSampler):
    """
    Groups all T slices of each SID together.
    - dataset must expose: ids (list[str]), max_frames, and an index mapping like data._lin2pair (sid_idx, t)
    - batch_size counts RIRs (SIDs). Each yielded batch contains batch_size * T contiguous indices.
    """
    def __init__(self, ids: Sequence[str], max_frames: int, batch_size_rirs: int,
                 drop_last: bool, shuffle: bool = True):
        self.ids = list(ids)
        self.T = int(max_frames)
        self.batch_size_rirs = int(batch_size_rirs)
        self.drop_last = drop_last
        self.shuffle = shuffle

        # Build base offsets: SID j occupies [j*T, (j+1)*T) in linear slice-index space
        self._sid_offsets = [j * self.T for j in range(len(self.ids))]
        self._sid_indices = list(range(len(self.ids)))

    def __iter__(self) -> Iterator[List[int]]:
        order = self._sid_indices.copy()
        if self.shuffle:
            random.shuffle(order)
        # step over SIDs in groups of batch_size_rirs
        for i in range(0, len(order), self.batch_size_rirs):
            sid_group = order[i:i + self.batch_size_rirs]
            if len(sid_group) < self.batch_size_rirs and self.drop_last:
                break
            batch: List[int] = []
            for sid_idx in sid_group:
                base = self._sid_offsets[sid_idx]
                batch.extend(list(range(base, base + self.T)))  # all T slices in order
            yield batch

    def __len__(self) -> int:
        n = math.ceil(len(self.ids) / self.batch_size_rirs)
        return n if not self.drop_last else len(self.ids) // self.batch_size_rirs