"""
Audio Datamanager for NeRAF
"""
from dataclasses import dataclass, field
from typing import Dict, Literal, Tuple, Type, Union, Callable, Any
from pathlib import Path

from .NeRAF_helper import WindowBatchSampler, collate_with_windowptr_varlen
from nerfstudio.cameras.cameras import Cameras
import torch

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.data.datamanagers.base_datamanager import (
    DataManager,
    DataManagerConfig,
)
from nerfstudio.data.utils.nerfstudio_collate import nerfstudio_collate
from torch.utils.data import DataLoader
from NeRAF.NeRAF_dataset import SoundSpacesDataset, RAFDataset
from NeRAF.NeRAF_dataparser import SoundSpacesDataparserOutputs, SoundSpacesDataParserConfig, RAFDataparserOutputs, RAFDataParserConfig, NeRAFDataparserOutputs, NeRAFDataParserConfig

import os
import numpy as np
import pickle as pkl

from nerfstudio.utils.rich_utils import CONSOLE

""""""""""""" NeRAF General DM """""""""""""
@dataclass
class NeRAFDataManagerConfig(DataManagerConfig):
    """
    NeRAF DataManager Config
    """

    _target: Type = field(default_factory=lambda: NeRAFDataManager)
    dataparser: Type = field(default_factory=lambda: NeRAFDataParserConfig)
    train_num_rays_per_batch: int = 4096
    eval_num_rays_per_batch: int = 4096
    collate_fn: Callable[[Any], Any] = nerfstudio_collate
    max_len: int = 156
    hop_len: int = 128
    fs: int = 22050
    window_size: int = 60
    use_windowed_edc: bool = True
    min_len_keep: int = 10

class NeRAFDataManager(DataManager):
    config: NeRAFDataManagerConfig
    train_dataset: RAFDataset
    eval_dataset: RAFDataset
    train_dataparser_outputs: RAFDataparserOutputs

    def __init__(
        self,
        config: NeRAFDataManagerConfig,
        device: Union[torch.device, str] = "cpu",
        test_mode: Literal["test", "val", "inference"] = "test",
        world_size: int = 1,
        local_rank: int = 0,
        num_workers: int = 16,
        **kwargs,  # pylint: disable=unused-argument
    ):
        self.config = config
        self.device = device
        self.world_size = world_size
        self.local_rank = local_rank
        self.test_mode = test_mode
        self.num_workers = num_workers

        self.train_count = 0
        self.eval_count = 0
        self.eval_image_count = 0
        
        super().__init__()

    def setup_dataparser(self):
        self.dataparser_config = self.config.dataparser
        if self.config.data is not None:
            self.config.dataparser.data = Path(self.config.data)
        else:
            self.config.data = self.config.dataparser.data
        self.dataparser = self.dataparser_config.setup()

    def setup_train(self):
        """Sets up the data loaders for training"""
        assert self.train_dataset is not None
        CONSOLE.print("Setting up training dataset...")
        
        if self.config.use_windowed_edc:
            sampler = WindowBatchSampler(
                self.train_dataset,
                batch_size=self.config.train_num_rays_per_batch,
                window_size=self.config.window_size,
                shuffle=True,
                drop_last=False,
            )
            self.train_audio_dataloader = DataLoader(
                self.train_dataset,
                batch_sampler=sampler,
                collate_fn=lambda s: collate_with_windowptr_varlen(s, min_len_keep=self.config.min_len_keep or 8),
                num_workers=self.num_workers,
                pin_memory=True,
            )
            self.train_epoch = 0
            self.train_audio_dataloader.batch_sampler.set_epoch(self.train_epoch)
        else:
            self.train_audio_dataloader = DataLoader(
                self.train_dataset,
                batch_size=self.config.train_num_rays_per_batch,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=True,
            )
        self.iter_train_audio_dataloader = iter(self.train_audio_dataloader)
    
    def setup_eval(self):
        """Sets up the data loader for evaluation"""
        assert self.eval_dataset is not None
        CONSOLE.print("Setting up evaluation dataset...")
        self.eval_audio_dataloader = DataLoader(
            self.eval_dataset,
            batch_size=self.config.eval_num_rays_per_batch,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        self.iter_eval_image_dataloader = iter(self.eval_audio_dataloader)

    def next_train(self, step: int) -> Tuple[RayBundle, Dict]:
        """Returns the next batch of data from the train dataloader."""
        self.train_count += 1

        try:
            batch = next(self.iter_train_audio_dataloader)
            batch["references"] = self.reference_bank
        except StopIteration:
            self.store_idx = []
            if self.config.use_windowed_edc:
                self.train_epoch += 1
                self.train_audio_dataloader.batch_sampler.set_epoch(self.train_epoch, shuffle_start=self.config.shuffle_start)
            self.iter_train_audio_dataloader = iter(self.train_audio_dataloader)
            batch = next(self.iter_train_audio_dataloader)
            batch["references"] = self.reference_bank

        assert isinstance(batch, dict)

        return None, batch

    def next_eval(self, step: int) -> Tuple[RayBundle, Dict]:
        """Returns the next batch of data from the eval dataloader."""
        # case where we eval a batch of stft slices (not a full audio)
        self.eval_count += 1
        try:
            batch = next(self.iter_eval_image_dataloader)
            batch["references"] = self.reference_bank
        except StopIteration:
            self.iter_eval_image_dataloader = iter(self.eval_audio_dataloader)
            batch = next(self.iter_eval_image_dataloader)
            batch["references"] = self.reference_bank

        assert isinstance(batch, dict)

        return None, batch

    def next_eval_image(self, step):
        # case where we eval a full audio
        i = self.eval_image_count
        self.eval_dataset.mode = 'eval_image'
        batch = self.eval_dataset[i]
        batch["references"] = self.reference_bank
        self.eval_image_count += 1
        self.eval_dataset.mode = 'eval'

        return None, batch

    

""""""""""""" RAF """""""""""""

@dataclass
class RAFDataManagerConfig(NeRAFDataManagerConfig):
    """
    RAF DataManager Config
    """

    _target: Type = field(default_factory=lambda: RAFDataManager)
    dataparser: Type = field(default_factory=lambda: RAFDataParserConfig)

    train_num_rays_per_batch: int = 4096
    eval_num_rays_per_batch: int = 4096

    collate_fn: Callable[[Any], Any] = nerfstudio_collate
    
    max_len: int = 0.32
    hop_len: int = 256
    fs: int = 48000
    test_mode: Literal["test", "val", "inference"] = "test"
    reference_num:    int  = 0      # how many references to pull
    reference_mode: bool = False
    shuffle_start: bool = False


class RAFDataManager(NeRAFDataManager):
    """Template DataManager

    Args:
        config: the DataManagerConfig used to instantiate class
    """

    config: RAFDataManagerConfig
    train_dataset: RAFDataset
    eval_dataset: RAFDataset
    train_dataparser_outputs: RAFDataparserOutputs

    def __init__(
        self,
        config: RAFDataManagerConfig,
        device: Union[torch.device, str] = "cpu",
        test_mode: Literal["test", "val", "inference"] = "test",
        world_size: int = 1,
        local_rank: int = 0,
        num_workers: int = 16,
        **kwargs,  # pylint: disable=unused-argument
    ):

        self.config = config
        self.device = device
        self.world_size = world_size
        self.local_rank = local_rank
        self.test_mode = test_mode
        self.num_workers = num_workers


        if "AVN_RENDER_POSES" in os.environ:
            self.test_mode = "inference"
        else: 
            self.test_mode = "test"

        self.setup_dataparser()
        
        self.fs = config.fs
        if self.fs == 48000:
            self.hop_len = 256
        elif self.fs == 16000:
            self.hop_len = 128

        self.max_len = int(config.max_len * self.fs / self.hop_len)
        self.max_len_time = int(config.max_len * self.fs)
        self.wav_path = os.path.join(self.config.data,'data')
    
        self.train_dataparser_outputs: RAFDataparserOutputs = self.dataparser.get_dataparser_outputs(split="train")
        self.eval_dataparser_outputs: RAFDataparserOutputs = self.dataparser.get_dataparser_outputs(split=self.test_mode)
        self.train_dataset = self.create_train_dataset()
        self.eval_dataset = self.create_eval_dataset()
        print('N train instances:', len(self.train_dataset))
        print('N eval instances:', int(len(self.eval_dataset) / self.max_len))
        # References
        if self.config.reference_mode:
            self.refs_dataparser_outputs = self.dataparser.get_dataparser_outputs(split="reference")
            self.refs_dataset = self.create_references_dataset()
            # fetch every single reference item (no DataLoader needed)
            all_stft, all_mic, all_src, all_rot = [], [], [], []
            Nref = len(self.refs_dataset)
            for flat_idx in range(Nref):
                rd = self.refs_dataset.get_data(flat_idx)
                all_stft.append(   rd["data"]       )  # (F,)
                all_mic.append(    rd["mic_pose"]   )  # (3,)
                all_src.append(    rd["source_pose"] )  # (3,)
                all_rot.append(    rd["rot"]         )  # (R,)

            # stack into one big tensor per field
            self.reference_bank = {
                "data":        torch.stack([x.squeeze(0) for x in all_stft], dim=0),  # (Nref, F, T)
                "mic_pose":    torch.stack(all_mic,   dim=0),  # (Nref, 3)
                "source_pose": torch.stack(all_src,   dim=0),  # (Nref, 3)
                "rot":         torch.stack(all_rot,   dim=0),  # (Nref, R)
            }
        else:
            self.reference_bank = []
        
        super().__init__(config, device, self.test_mode, world_size, local_rank, **kwargs)

    
    def create_train_dataset(self) -> RAFDataset:
        """Sets up the data loaders for training"""
        return RAFDataset(
            mode='train',
            dataparser_outputs=self.train_dataparser_outputs,
            max_len=self.max_len,
            max_len_time=self.max_len_time, 
            wav_path=self.wav_path,
            fs=self.fs,
            hop_len=self.hop_len,
            reference_num=self.config.reference_num,
            reference_mode = self.config.reference_mode
        )

    def create_eval_dataset(self) -> RAFDataset:
        """Sets up the data loaders for evaluation"""
        return RAFDataset(
            mode='eval' if self.test_mode == 'test' else 'inference',
            dataparser_outputs=self.eval_dataparser_outputs,
            max_len=self.max_len,
            max_len_time=self.max_len_time, 
            wav_path=self.wav_path,
            fs=self.fs,
            hop_len=self.hop_len, 
            reference_num=self.config.reference_num,
            reference_mode = self.config.reference_mode
        )
        
    def create_references_dataset(self) -> RAFDataset:
        """Sets up the small 2k reference dataset."""
        return RAFDataset(
            mode="reference",
            dataparser_outputs=self.refs_dataparser_outputs,
            max_len=self.max_len,
            max_len_time=self.max_len_time,
            wav_path=self.wav_path,
            fs=self.fs,
            hop_len=self.hop_len,
            reference_num=0,          # no nested references
            reference_mode=False,
        )

""""""""""""" Sound Spaces """""""""""""
@dataclass
class SoundSpacesDataManagerConfig(NeRAFDataManagerConfig):
    """NeRAF DataManager Config
    """

    _target: Type = field(default_factory=lambda: SoundSpacesDataManager)
    dataparser: Type = field(default_factory=lambda: SoundSpacesDataParserConfig)
    train_num_rays_per_batch: int = 4096
    eval_num_rays_per_batch: int = 4096
    collate_fn: Callable[[Any], Any] = nerfstudio_collate
    max_len: int = 156
    hop_len: int = 128
    fs: int = 22050


class SoundSpacesDataManager(NeRAFDataManager):
    """NeRAF DataManager

    Args:
        config: the DataManagerConfig used to instantiate class
    """

    config: SoundSpacesDataManagerConfig
    train_dataset: SoundSpacesDataset
    eval_dataset: SoundSpacesDataset
    train_dataparser_outputs: SoundSpacesDataparserOutputs

    def __init__(
        self,
        config: SoundSpacesDataManagerConfig,
        device: Union[torch.device, str] = "cpu",
        test_mode: Literal["test", "val", "inference"] = "test",
        world_size: int = 1,
        local_rank: int = 0,
        num_workers: int = 16,
        **kwargs,  # pylint: disable=unused-argument
    ):
        
        self.config = config
        self.device = device
        self.world_size = world_size
        self.local_rank = local_rank
        self.test_mode = test_mode
        self.num_workers = num_workers


        if "AVN_RENDER_POSES" in os.environ:
            self.test_mode = "inference"
        else: 
            self.test_mode = "test"

        self.setup_dataparser()

        self.max_len = config.max_len
        self.hop_len = config.hop_len
        self.fs = config.fs
        self.max_len_time = self.max_len * self.hop_len

        if self.fs == 44100:
            self.mag_path = os.path.join(self.config.data, 'binaural_magnitudes')
        else:
            self.mag_path = os.path.join(self.config.data, 'binaural_magnitudes_sr22050')
        self.wav_path = os.path.join(self.config.data, 'binaural_rirs')
        
        self.train_dataparser_outputs: SoundSpacesDataparserOutputs = self.dataparser.get_dataparser_outputs(split="train")
        self.eval_dataparser_outputs: SoundSpacesDataparserOutputs = self.dataparser.get_dataparser_outputs(split=self.test_mode)
        self.train_dataset = self.create_train_dataset()
        self.eval_dataset = self.create_eval_dataset()
        print('N train instances:', len(self.train_dataset))
        if self.test_mode == 'inference':
            print('N eval instances:', len(self.eval_dataset))
        else:
            print('N eval instances:', int(len(self.eval_dataset) / self.max_len))
        
        super().__init__(config, device, self.test_mode, world_size, local_rank, num_workers, **kwargs)

        

    
    def create_train_dataset(self) -> SoundSpacesDataset:
        """Sets up the data loaders for training"""
        return SoundSpacesDataset(
            mode='train',
            dataparser_outputs=self.train_dataparser_outputs,
            max_len=self.max_len,
            mag_path=self.mag_path,
            wav_path=self.wav_path,
            fs=self.fs,
            hop_len=self.hop_len, 
        )

    def create_eval_dataset(self) -> SoundSpacesDataset:
        """Sets up the data loaders for evaluation"""
        return SoundSpacesDataset(
            mode='eval' if self.test_mode == 'test' else 'inference',
            dataparser_outputs=self.eval_dataparser_outputs,
            max_len=self.max_len,
            mag_path=self.mag_path,
            wav_path=self.wav_path,
            fs=self.fs,
            hop_len=self.hop_len, 
        )
    