"""EvoBrain-compatible loader for already-segmented CHB-MIT PKL files."""

from __future__ import annotations

from pathlib import Path
import pickle

import numpy as np
import torch
from scipy.signal import resample_poly
from torch.utils.data import DataLoader, Dataset

from .evobrain_contract import (
    FREQUENCY,
    TIME_STEP_SECONDS,
    WINDOW_SECONDS,
    empty_supports,
    require_raw_mode,
)


class CustomDataset(Dataset):
    def __init__(
        self,
        data_dir,
        mode="train",
        max_seq_len=WINDOW_SECONDS,
        time_step_size=TIME_STEP_SECONDS,
        use_fft=False,
    ):
        require_raw_mode(use_fft)
        if max_seq_len != WINDOW_SECONDS or time_step_size != TIME_STEP_SECONDS:
            raise ValueError("The mirrored seizure contract requires 10-second clips and one-second steps")
        self.data_dir = Path(data_dir)
        self.mode = mode
        self.max_seq_len = max_seq_len
        self.time_step_size = time_step_size
        aliases = (mode, "val") if mode == "dev" else (mode,)
        split_dir = next((self.data_dir / name for name in aliases if (self.data_dir / name).is_dir()), None)
        if split_dir is not None:
            self.files = sorted(path for path in split_dir.rglob("*.pkl") if not path.name.startswith("."))
        else:
            self.files = sorted(path for path in self.data_dir.glob("*.pkl") if not path.name.startswith("."))
        self.num_nodes = self._detect_num_nodes()
        self.pos_weight = self._estimate_pos_weight() if mode == "train" and self.files else None

    @staticmethod
    def _decode(data):
        if isinstance(data, dict):
            raw = data.get("X", data.get("data"))
            label = int(data.get("y", data.get("label", 0)))
        elif isinstance(data, (tuple, list)):
            raw, label = data[0], int(data[1])
        else:
            raw, label = data, 0
        if raw is None:
            raise ValueError("PKL sample does not contain X or data")
        return np.asarray(raw, dtype=np.float32), label

    def _read(self, path: Path):
        with path.open("rb") as handle:
            return self._decode(pickle.load(handle))

    def _detect_num_nodes(self):
        if not self.files:
            return 16  # EvoBrain/data/dataloader_chb.py:498 fallback.
        try:
            raw, _ = self._read(self.files[0])
            return raw.shape[0] if raw.ndim >= 2 else 16
        except Exception:
            return 16

    def _estimate_pos_weight(self):
        # EvoBrain/data/dataloader_chb.py:541-563.
        sample_size = min(3000, len(self.files))
        indices = np.random.RandomState(123).choice(len(self.files), sample_size, replace=False)
        positives = 0
        try:
            for index in indices:
                try:
                    _, label = self._read(self.files[int(index)])
                    positives += int(label == 1)
                except Exception:
                    pass
            negatives = sample_size - positives
            return float(negatives) / float(positives) if positives > 0 else 260.0
        except Exception:
            return 260.0

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        path = self.files[index]
        raw_data, label = self._read(path)
        raw_data = np.nan_to_num(raw_data, nan=0.0, posinf=0.0, neginf=0.0)

        if raw_data.ndim == 2:
            channels, samples = raw_data.shape
            self.num_nodes = channels
            target_samples = self.max_seq_len * FREQUENCY
            if samples != target_samples:
                raw_data = resample_poly(raw_data, up=target_samples, down=samples, axis=1)
                raw_data = np.nan_to_num(raw_data, nan=0.0, posinf=0.0, neginf=0.0)
            step = self.time_step_size * FREQUENCY
            eeg_clip = np.stack(
                [raw_data[:, offset:offset + step] for offset in range(0, target_samples, step)]
            )
        elif raw_data.ndim == 3:
            if raw_data.shape[1] == self.max_seq_len:
                raw_data = raw_data.transpose(1, 0, 2)
            eeg_clip = raw_data
        else:
            raise ValueError(f"Expected a 2-D or 3-D PKL array, got shape {raw_data.shape}")

        feature = np.nan_to_num(eeg_clip.copy(), nan=0.0, posinf=0.0, neginf=0.0)
        mean = np.mean(feature)
        std = np.std(feature)
        if std > 1e-5:
            feature = (feature - mean) / std
        feature = np.nan_to_num(feature, nan=0.0, posinf=0.0, neginf=0.0)

        norms = np.maximum(np.linalg.norm(feature, axis=-1, keepdims=True), 1e-5)
        normalized = feature / norms
        adjacency = np.abs(normalized @ normalized.swapaxes(-1, -2)).astype(np.float32)
        adjacency = np.nan_to_num(adjacency, nan=0.0, posinf=1.0, neginf=0.0)
        adjacency = np.clip(adjacency, 0.0, 1.0)
        for step_index in range(adjacency.shape[0]):
            np.fill_diagonal(adjacency[step_index], 1.0)

        return (
            torch.as_tensor(feature, dtype=torch.float32),
            torch.tensor([label], dtype=torch.float32),
            torch.tensor([self.max_seq_len], dtype=torch.int64),
            empty_supports(),
            torch.from_numpy(adjacency),
            path.stem,
        )


class LoadDataset:
    def __init__(self, params):
        self.params = params

    def get_data_loader(self):
        loaders = {}
        for split in ("train", "dev", "test"):
            dataset = CustomDataset(
                self.params.datasets_dir,
                mode=split,
                max_seq_len=self.params.max_seq_len,
                time_step_size=self.params.time_step_size,
                use_fft=self.params.use_fft,
            )
            key = "val" if split == "dev" else split
            loaders[key] = DataLoader(
                dataset,
                batch_size=self.params.batch_size,
                shuffle=split == "train",
                num_workers=self.params.num_workers,
                pin_memory=torch.cuda.is_available(),
            )
        return loaders
