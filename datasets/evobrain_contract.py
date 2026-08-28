"""Shared six-field seizure-detection contract mirrored from EvoBrain.

The constants below are sourced from EvoBrain/constants.py:2-32 and
EvoBrain/args.py:67-79. CBraMod is deliberately raw-only: the native
backbone consumes 200 time-domain samples per one-second patch.
"""

from __future__ import annotations

from pathlib import Path
import pickle

import numpy as np
import torch


FREQUENCY = 200
WINDOW_SECONDS = 10
TIME_STEP_SECONDS = 1
TUSZ_CHANNELS = (
    "EEG FP1", "EEG FP2", "EEG F3", "EEG F4", "EEG C3", "EEG C4",
    "EEG P3", "EEG P4", "EEG O1", "EEG O2", "EEG F7", "EEG F8",
    "EEG T3", "EEG T4", "EEG T5", "EEG T6", "EEG FZ", "EEG CZ",
    "EEG PZ",
)


def require_raw_mode(use_fft: bool) -> None:
    """Reject phase-discarding FFT inputs at the CBraMod boundary."""
    if use_fft:
        raise ValueError(
            "CBraMod requires raw time-domain input; configure use_fft=False. "
            "The 100-bin FFT contract cannot be faithfully converted to a "
            "200-point raw patch because phase information was discarded."
        )


def load_scalar(path: str | Path | None, name: str) -> float:
    """Load a data-derived scalar without inventing a fallback value."""
    if path is None:
        raise ValueError(f"{name} path is required when standardize=True")
    with Path(path).open("rb") as handle:
        value = pickle.load(handle)
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError(f"{name} must contain one scalar, got shape {array.shape}")
    return float(array.reshape(-1)[0])


def dynamic_adjacency(eeg_clip: np.ndarray, top_k: int = 3) -> torch.Tensor:
    """Build EvoBrain-style per-step absolute cosine adjacency with self-loops."""
    norms = np.linalg.norm(eeg_clip, axis=-1, keepdims=True)
    norms[norms == 0] = 1e-8
    normalized = eeg_clip / norms
    adjacency = np.abs(normalized @ normalized.swapaxes(-1, -2)).astype(np.float32)
    for step in range(adjacency.shape[0]):
        np.fill_diagonal(adjacency[step], 1.0)
        if top_k is not None and top_k < adjacency.shape[1]:
            keep = np.argpartition(adjacency[step], -top_k, axis=1)[:, -top_k:]
            mask = np.zeros_like(adjacency[step], dtype=bool)
            np.put_along_axis(mask, keep, True, axis=1)
            adjacency[step] = np.where(mask, adjacency[step], 0.0)
            np.fill_diagonal(adjacency[step], 1.0)
    return torch.from_numpy(adjacency)


def empty_supports() -> torch.Tensor:
    """CHB-MIT-PKL support placeholder from EvoBrain's PKL contract."""
    return torch.empty(0, dtype=torch.float32)

