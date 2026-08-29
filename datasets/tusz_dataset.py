"""EvoBrain-compatible marker/HDF5 TUSZ loader."""

from __future__ import annotations

from pathlib import Path
import random

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .evobrain_contract import (
    FREQUENCY,
    TIME_STEP_SECONDS,
    TUSZ_CHANNELS,
    WINDOW_SECONDS,
    dynamic_adjacency,
    load_scalar,
    require_raw_mode,
)


def _marker_entries(marker_dir, split, clip_len, seed):
    marker_root = Path(marker_dir)
    seizure_file = marker_root / f"{split}Set_seq2seq_{clip_len}s_sz.txt"
    nonseizure_file = marker_root / f"{split}Set_seq2seq_{clip_len}s_nosz.txt"
    if not seizure_file.exists() or not nonseizure_file.exists():
        raise FileNotFoundError(f"Missing TUSZ marker files: {seizure_file}, {nonseizure_file}")
    seizure = [line.strip() for line in seizure_file.read_text().splitlines() if line.strip()]
    nonseizure = [line.strip() for line in nonseizure_file.read_text().splitlines() if line.strip()]
    rng = random.Random(seed)
    if split == "train":
        rng.shuffle(seizure)
        rng.shuffle(nonseizure)
        nonseizure = nonseizure[:len(seizure)]
    combined = seizure + nonseizure
    rng.shuffle(combined)
    return [tuple(item.rsplit(",", 1)) for item in combined]


def _resampled_path(input_dir, split, marker_name):
    base_name = marker_name.split(".edf")[0] + ".h5"
    aliases = ("test", "eval") if split == "test" else (split,)
    candidates = [Path(input_dir) / alias / base_name for alias in aliases]
    candidates.append(Path(input_dir) / base_name)
    match = next((path for path in candidates if path.exists()), None)
    if match is None:
        raise FileNotFoundError(f"No resampled HDF5 for {marker_name}; checked {candidates}")
    return match


class TUSZDataset(Dataset):
    """Index raw EDFs using EvoBrain's non-overlapping marker semantics."""

    def __init__(
        self,
        input_dir: str,
        marker_dir: str,
        split: str,
        max_seq_len: int = WINDOW_SECONDS,
        time_step_size: int = TIME_STEP_SECONDS,
        standardize: bool = True,
        mean_path: str | None = None,
        std_path: str | None = None,
        data_augment: bool = False,
        top_k: int = 3,
        seed: int = 123,
        use_fft: bool = False,
    ):
        require_raw_mode(use_fft)
        if max_seq_len != WINDOW_SECONDS or time_step_size != TIME_STEP_SECONDS:
            raise ValueError("The mirrored seizure contract requires 10-second clips and one-second steps")
        if input_dir is None or marker_dir is None:
            raise ValueError("input_dir and marker_dir are required for TUSZ")
        self.input_dir = Path(input_dir)
        self.marker_dir = Path(marker_dir)
        self.split = split
        self.max_seq_len = max_seq_len
        self.time_step_size = time_step_size
        self.standardize = standardize
        self.mean = load_scalar(mean_path, "mean") if standardize else None
        self.std = load_scalar(std_path, "std") if standardize else None
        if self.std == 0:
            raise ValueError("std must be non-zero")
        self.data_augment = data_augment
        self.top_k = top_k

        self.entries = []
        for marker_name, label_text in _marker_entries(self.marker_dir, split, max_seq_len, seed):
            clip_index = int(marker_name.rsplit("_", 1)[1].split(".h5")[0])
            self.entries.append(
                (_resampled_path(self.input_dir, split, marker_name), clip_index,
                 int(label_text), marker_name.split(".h5")[0])
            )
        self.num_nodes = len(TUSZ_CHANNELS)
        sample = self.entries[:min(3000, len(self.entries))]
        positives = sum(label == 1 for _, _, label, _ in sample)
        negatives = len(sample) - positives
        self.pos_weight = float(negatives) / positives if positives else 260.0

    def __len__(self) -> int:
        return len(self.entries)

    def _read_window(self, h5_path: Path, clip_index: int) -> np.ndarray:
        physical_length = self.max_seq_len * FREQUENCY
        start = clip_index * physical_length
        with h5py.File(h5_path, "r") as handle:
            if int(handle["resample_freq"][()]) != FREQUENCY:
                raise ValueError(f"Unexpected resample frequency in {h5_path}")
            window = handle["resampled_signal"][:, start:start + physical_length]
        if window.shape[0] != self.num_nodes:
            raise ValueError(f"Expected {self.num_nodes} TUSZ channels, got {window.shape[0]}")
        if window.shape[-1] < physical_length:
            if window.shape[-1] == 0:
                raise ValueError(f"Empty TUSZ window {h5_path.name}:{clip_index}")
            window = np.pad(window, ((0, 0), (0, physical_length - window.shape[-1])), mode="edge")
        step = self.time_step_size * FREQUENCY
        return np.stack([window[:, offset:offset + step] for offset in range(0, physical_length, step)])

    def __getitem__(self, index: int):
        h5_path, clip_index, label, writeout_fn = self.entries[index]
        eeg_clip = self._read_window(h5_path, clip_index)
        feature = eeg_clip.copy()
        if self.data_augment:
            pairs = ((0, 1), (2, 3), (10, 11), (4, 5), (12, 13), (14, 15), (8, 9))
            if np.random.choice((True, False)):
                for left, right in pairs:
                    feature[:, [left, right], :] = feature[:, [right, left], :]
            feature *= np.random.uniform(0.8, 1.2)
        if self.standardize:
            feature = (feature - self.mean) / self.std

        x = torch.as_tensor(feature, dtype=torch.float32)
        y = torch.tensor([label], dtype=torch.float32)
        seq_len = torch.tensor([self.max_seq_len], dtype=torch.int64)
        adjacency = dynamic_adjacency(eeg_clip, top_k=self.top_k)
        supports = torch.zeros(
            self.max_seq_len, 2, self.num_nodes, self.num_nodes, dtype=torch.float32
        )
        return x, y, seq_len, supports, adjacency, writeout_fn


class LoadDataset:
    def __init__(self, params):
        self.params = params

    def get_data_loader(self):
        loaders = {}
        for split in ("train", "dev", "test"):
            dataset = TUSZDataset(
                input_dir=self.params.input_dir,
                marker_dir=self.params.marker_dir,
                split=split,
                max_seq_len=self.params.max_seq_len,
                time_step_size=self.params.time_step_size,
                standardize=self.params.standardize,
                mean_path=self.params.scaler_mean_path,
                std_path=self.params.scaler_std_path,
                data_augment=self.params.data_augment if split == "train" else False,
                top_k=self.params.top_k,
                seed=self.params.seed,
                use_fft=self.params.use_fft,
            )
            key = "val" if split == "dev" else split
            loaders[key] = DataLoader(
                dataset,
                batch_size=self.params.batch_size,
                shuffle=split == "train",
                num_workers=self.params.num_workers,
            )
        return loaders
