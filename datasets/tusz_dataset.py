"""Raw continuous TUSZ loader with EvoBrain-compatible 10-second windows."""

from __future__ import annotations

from math import gcd
from pathlib import Path
import random

import numpy as np
import torch
from scipy.signal import resample_poly
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


MODERN_ALIASES = {
    "T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8",
    "T7": "T3", "T8": "T4", "P7": "T5", "P8": "T6",
}


def _annotation_path(edf_path: Path) -> Path | None:
    for suffix in (".tse_bi", ".csv_bi", ".tse", ".csv"):
        candidate = edf_path.with_suffix(suffix)
        if candidate.exists():
            return candidate
    return None


def _seizures_and_end(annotation_path: Path) -> tuple[list[tuple[float, float]], float]:
    seizures: list[tuple[float, float]] = []
    max_time = 0.0
    with annotation_path.open("r", errors="ignore") as handle:
        for line in handle:
            if "version" in line or line.startswith("#") or not line.strip() or "start_time" in line:
                continue
            parts = line.strip().replace(",", " ").split()
            try:
                if len(parts) >= 4 and parts[0].upper() == "TERM":
                    start, end, label = float(parts[1]), float(parts[2]), parts[3]
                elif len(parts) >= 3:
                    start, end, label = float(parts[0]), float(parts[1]), parts[2]
                elif len(parts) >= 2 and any(
                    key in line.lower()
                    for key in ("seiz", "fnsz", "gnsz", "cpsz", "spsz", "tcsz")
                ):
                    start, end, label = float(parts[0]), float(parts[1]), "seiz"
                else:
                    continue
            except ValueError:
                continue
            max_time = max(max_time, end)
            if label.lower() != "bckg" or "seiz" in line.lower():
                seizures.append((start, end))
    return seizures, max_time


def _split_directory(raw_dir: Path, split: str) -> Path:
    if split == "test":
        for alias in ("eval", "test", "dev"):
            candidate = raw_dir / alias
            if candidate.is_dir():
                return candidate
    candidate = raw_dir / split
    return candidate if candidate.is_dir() else raw_dir


def _clean_tusz_label(label: str) -> str:
    return label.split("-")[0].strip().upper().replace("EEG ", "")


def _ordered_channel_indices(labels: list[str]) -> list[int]:
    cleaned = [_clean_tusz_label(label) for label in labels]
    indices = []
    for requested in TUSZ_CHANNELS:
        name = requested.replace("EEG ", "")
        candidates = (name, MODERN_ALIASES.get(name, ""))
        match = next((cleaned.index(item) for item in candidates if item in cleaned), None)
        if match is None:
            raise ValueError(f"Required TUSZ channel {requested!r} is absent")
        indices.append(match)
    return indices


class TUSZDataset(Dataset):
    """Index raw EDFs using EvoBrain's non-overlapping marker semantics."""

    def __init__(
        self,
        raw_data_dir: str,
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
        self.raw_data_dir = Path(raw_data_dir)
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

        split_dir = _split_directory(self.raw_data_dir, split)
        positive = []
        negative = []
        for edf_path in sorted(split_dir.rglob("*.edf")):
            annotation = _annotation_path(edf_path)
            if annotation is None:
                continue
            seizures, max_time = _seizures_and_end(annotation)
            for clip_index in range(int(max_time // max_seq_len)):
                start = clip_index * max_seq_len
                end = (clip_index + 1) * max_seq_len
                label = int(any(max(start, onset) < min(end, offset) for onset, offset in seizures))
                entry = (edf_path, clip_index, label, f"{edf_path.name}_{clip_index}")
                (positive if label else negative).append(entry)

        rng = random.Random(seed)
        if split == "train":
            rng.shuffle(positive)
            rng.shuffle(negative)
            negative = negative[:len(positive)]
        self.entries = positive + negative
        rng.shuffle(self.entries)
        self.num_nodes = len(TUSZ_CHANNELS)
        self.pos_weight = None

    def __len__(self) -> int:
        return len(self.entries)

    def _read_window(self, edf_path: Path, clip_index: int) -> np.ndarray:
        import pyedflib

        reader = pyedflib.EdfReader(str(edf_path))
        try:
            indices = _ordered_channel_indices(list(reader.getSignalLabels()))
            original_frequency = int(round(reader.getSampleFrequency(0)))
            signal = np.stack([reader.readSignal(index) for index in indices]).astype(np.float32)
        finally:
            reader.close()
        if original_frequency != FREQUENCY:
            common = gcd(original_frequency, FREQUENCY)
            signal = resample_poly(
                signal,
                up=FREQUENCY // common,
                down=original_frequency // common,
                axis=-1,
            ).astype(np.float32)
        physical_length = self.max_seq_len * FREQUENCY
        start = clip_index * physical_length
        window = signal[:, start:start + physical_length]
        if window.shape[-1] < physical_length:
            if window.shape[-1] == 0:
                raise ValueError(f"Empty TUSZ window {edf_path.name}:{clip_index}")
            window = np.pad(window, ((0, 0), (0, physical_length - window.shape[-1])), mode="edge")
        step = self.time_step_size * FREQUENCY
        return np.stack([window[:, offset:offset + step] for offset in range(0, physical_length, step)])

    def __getitem__(self, index: int):
        edf_path, clip_index, label, writeout_fn = self.entries[index]
        eeg_clip = self._read_window(edf_path, clip_index)
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
                raw_data_dir=self.params.raw_data_dir,
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

