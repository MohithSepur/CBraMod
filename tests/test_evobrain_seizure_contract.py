import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path
from types import SimpleNamespace
import pickle

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from models.cbramod import CBraMod
from models.model_for_chb import Model, SeizureModelOutput
from datasets.chb_dataset import CustomDataset, pin_memory_enabled
from datasets.tusz_dataset import TUSZDataset
from seizure_detection import (
    configure_device,
    configure_amp,
    evaluate_and_save,
    resolve_device,
    smoothed_pos_weight,
    train_one_batch,
    training_criterion,
)


def synthetic_batch(batch_size, channels, pkl_contract=False):
    # Shapes and dtypes mirror the accepted Step A raw contracts exactly.
    x = torch.randn(batch_size, 10, channels, 200, dtype=torch.float32)
    y = torch.tensor([[0.0], [1.0]], dtype=torch.float32)[:batch_size]
    seq_len = torch.full((batch_size, 1), 10, dtype=torch.int64)
    supports = (
        torch.empty(batch_size, 0, dtype=torch.float32)
        if pkl_contract
        else torch.zeros(batch_size, 10, 2, channels, channels, dtype=torch.float32)
    )
    adj_mat = torch.eye(channels, dtype=torch.float32).repeat(batch_size, 10, 1, 1)
    names = tuple(f"sample-{index}" for index in range(batch_size))
    return x, y, seq_len, supports, adj_mat, names


class _DatasetWithPosWeight:
    pos_weight = 1.0


class _StaticDataset(Dataset):
    def __init__(self):
        self.samples = []
        for index, label in enumerate((0.0, 1.0, 0.0, 1.0)):
            sample = synthetic_batch(2, 16, pkl_contract=True)
            self.samples.append(tuple(
                value[index % 2] if torch.is_tensor(value) else value[index % 2]
                for value in sample
            ))
            values = list(self.samples[-1])
            values[1] = torch.tensor([label], dtype=torch.float32)
            values[5] = f"eval-{index}"
            self.samples[-1] = tuple(values)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


class _StaticModel(torch.nn.Module):
    def forward_contract(self, batch):
        x, y, seq_len, supports, adj_mat, names = batch
        logits = torch.where(y.reshape(-1) > 0, torch.tensor(2.0), torch.tensor(-2.0)).to(x)
        return SeizureModelOutput(logits, y, seq_len, supports, adj_mat, tuple(names))


class EvoBrainContractTest(unittest.TestCase):
    def test_cuda_available_device_and_amp_configuration_branch(self):
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.set_device") as set_device,
        ):
            device = resolve_device(cuda_index=2)
            use_amp, scaler = configure_amp(device, requested=True)
            self.assertEqual(device, torch.device("cuda:2"))
            self.assertTrue(use_amp)
            self.assertTrue(scaler.is_enabled())
            self.assertTrue(pin_memory_enabled())
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                pass
            self.assertEqual(configure_device(2), device)
            set_device.assert_called_once_with(2)

    def _model(self):
        params = SimpleNamespace(
            use_pretrained_weights=False,
            classifier="avgpooling_patch_reps",
            dropout=0.0,
            cuda=0,
        )
        tiny_backbone = CBraMod(
            in_dim=200,
            out_dim=200,
            d_model=200,
            dim_feedforward=200,
            seq_len=10,
            n_layer=1,
            nhead=8,
        )
        return Model(params, backbone=tiny_backbone)

    def test_tusz_forward_backward_optimizer_step(self):
        torch.manual_seed(123)
        model = self._model()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        criterion = training_criterion(_DatasetWithPosWeight(), torch.device("cpu"))
        batch = synthetic_batch(2, 19, pkl_contract=False)

        loss, stepped, names = train_one_batch(
            model=model,
            batch=batch,
            optimizer=optimizer,
            criterion=criterion,
            device=torch.device("cpu"),
            max_grad_norm=5.0,
        )

        self.assertTrue(np.isfinite(loss))
        self.assertTrue(stepped)
        self.assertEqual(names, ("sample-0", "sample-1"))

    def test_chb_auto_detected_channel_shape_and_fft_rejection(self):
        model = self._model()
        output = model.forward_contract(synthetic_batch(2, 16, pkl_contract=True))
        self.assertEqual(output.logits.shape, (2,))
        self.assertEqual(output.supports.shape, (2, 0))

        fft_batch = list(synthetic_batch(2, 16, pkl_contract=True))
        fft_batch[0] = torch.randn(2, 10, 16, 100, dtype=torch.float32)
        with self.assertRaisesRegex(ValueError, "raw-only"):
            model.forward_contract(fft_batch)

    def test_chb_pkl_contract_and_dynamic_pos_weight(self):
        with tempfile.TemporaryDirectory() as directory:
            train_dir = Path(directory) / "train"
            train_dir.mkdir()
            for index, label in enumerate((1, 0, 0, 0)):
                with (train_dir / f"sample-{index}.pkl").open("wb") as handle:
                    pickle.dump(
                        {"X": np.random.randn(16, 2560).astype(np.float32), "y": label},
                        handle,
                    )
            dataset = CustomDataset(directory, mode="train", use_fft=False)
            self.assertEqual(dataset.pos_weight, 3.0)
            self.assertEqual(smoothed_pos_weight(dataset.pos_weight), np.sqrt(3.0))
            x, y, seq_len, supports, adj_mat, name = dataset[0]
            self.assertEqual(x.shape, (10, 16, 200))
            self.assertEqual(x.dtype, torch.float32)
            self.assertEqual(y.shape, (1,))
            self.assertEqual(seq_len.dtype, torch.int64)
            self.assertEqual(supports.shape, (0,))
            self.assertEqual(adj_mat.shape, (10, 16, 16))
            self.assertEqual(name, "sample-0")

    def test_tusz_index_uses_non_overlapping_strict_overlap_windows(self):
        with tempfile.TemporaryDirectory() as directory:
            raw_dir = Path(directory) / "dev"
            raw_dir.mkdir()
            (raw_dir / "record.edf").touch()
            (raw_dir / "record.tse").write_text(
                "0 5 bckg\n5 8 fnsz\n8 25 bckg\n",
                encoding="utf-8",
            )
            dataset = TUSZDataset(
                raw_data_dir=directory,
                split="dev",
                standardize=False,
                use_fft=False,
            )
            self.assertEqual(len(dataset), 2)
            indexed = sorted((clip_index, label) for _, clip_index, label, _ in dataset.entries)
            self.assertEqual(indexed, [(0, 1), (1, 0)])

    def test_traceable_dev_threshold_is_reused_for_test(self):
        loader = DataLoader(_StaticDataset(), batch_size=2, shuffle=False)
        model = _StaticModel()
        with tempfile.TemporaryDirectory() as directory:
            dev = evaluate_and_save(model, loader, torch.device("cpu"), directory, "dev")
            test = evaluate_and_save(
                model, loader, torch.device("cpu"), directory, "test", threshold=dev["threshold"]
            )
            self.assertEqual(test["threshold"], dev["threshold"])
            self.assertEqual(dev["f1"], 1.0)
            self.assertEqual(test["f1"], 1.0)
            for split in ("dev", "test"):
                result = np.load(Path(directory) / f"{split}_results.npz")
                self.assertEqual(result["file_names"].tolist(), [f"eval-{i}" for i in range(4)])
                for key in (
                    "labels", "probabilities", "predictions", "accuracy",
                    "balanced_accuracy", "f1", "recall", "precision",
                    "specificity", "auroc", "pr_auc",
                ):
                    self.assertIn(key, result.files)


if __name__ == "__main__":
    unittest.main()
