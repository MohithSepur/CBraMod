import torch
import torch.nn as nn
from typing import NamedTuple
from .cbramod import CBraMod


class AdaptedSeizureBatch(NamedTuple):
    x: torch.Tensor
    y: torch.Tensor
    seq_len: torch.Tensor
    supports: torch.Tensor
    adj_mat: torch.Tensor
    writeout_fn: tuple[str, ...]


class SeizureModelOutput(NamedTuple):
    logits: torch.Tensor
    y: torch.Tensor
    seq_len: torch.Tensor
    supports: torch.Tensor
    adj_mat: torch.Tensor
    writeout_fn: tuple[str, ...]


class EvoBrainRawAdapter(nn.Module):
    """Adapt `[B,S,C,200]` raw contract batches to CBraMod `[B,C,S,200]`."""

    def forward(self, batch):
        if not isinstance(batch, (tuple, list)) or len(batch) != 6:
            raise ValueError("Expected (x, y, seq_len, supports, adj_mat, writeout_fn)")
        x, y, seq_len, supports, adj_mat, writeout_fn = batch
        if not torch.is_tensor(x) or x.ndim != 4:
            raise ValueError("x must be a four-dimensional [B,S,C,P] tensor")
        if x.shape[1] != 10:
            raise ValueError(f"Expected 10 one-second steps, got {x.shape[1]}")
        if x.shape[-1] != 200:
            representation = "100-bin FFT" if x.shape[-1] == 100 else f"{x.shape[-1]} features"
            raise ValueError(
                f"CBraMod is raw-only and cannot accept {representation}; configure use_fft=False"
            )
        if x.dtype != torch.float32:
            raise TypeError(f"x must be torch.float32, got {x.dtype}")
        filenames = tuple(str(name) for name in writeout_fn)
        return AdaptedSeizureBatch(
            x=x.permute(0, 2, 1, 3).contiguous(),
            y=y,
            seq_len=seq_len,
            supports=supports,
            adj_mat=adj_mat,
            writeout_fn=filenames,
        )


class MeanPatchClassifier(nn.Module):
    def __init__(self, embedding_dim=200):
        super().__init__()
        self.linear = nn.Linear(embedding_dim, 1)

    def forward(self, features):
        return self.linear(features.mean(dim=(1, 2))).reshape(-1)


class Model(nn.Module):
    def __init__(self, param, backbone=None):
        super(Model, self).__init__()
        self.adapter = EvoBrainRawAdapter()
        self.backbone = backbone if backbone is not None else CBraMod(
            in_dim=200, out_dim=200, d_model=200,
            dim_feedforward=800, seq_len=30,
            n_layer=12, nhead=8
        )
        if param.use_pretrained_weights and backbone is None:
            map_location = torch.device(
                f'cuda:{param.cuda}' if torch.cuda.is_available() else 'cpu'
            )
            self.backbone.load_state_dict(torch.load(param.foundation_dir, map_location=map_location))
        self.backbone.proj_out = nn.Identity()

        if param.classifier == 'avgpooling_patch_reps':
            self.classifier = MeanPatchClassifier(embedding_dim=200)
        elif param.classifier == 'all_patch_reps_onelayer':
            self.classifier = nn.Sequential(
                nn.Flatten(start_dim=1),
                nn.Linear(16*10*200, 1),
            )
        elif param.classifier == 'all_patch_reps_twolayer':
            self.classifier = nn.Sequential(
                nn.Flatten(start_dim=1),
                nn.Linear(16*10*200, 200),
                nn.ELU(),
                nn.Dropout(param.dropout),
                nn.Linear(200, 1),
            )
        elif param.classifier == 'all_patch_reps':
            self.classifier = nn.Sequential(
                nn.Flatten(start_dim=1),
                nn.Linear(16*10*200, 10*200),
                nn.ELU(),
                nn.Dropout(param.dropout),
                nn.Linear(10*200, 200),
                nn.ELU(),
                nn.Dropout(param.dropout),
                nn.Linear(200, 1),
            )

    def forward(self, x):
        _, _, _, _ = x.shape
        feats = self.backbone(x)
        out = self.classifier(feats)
        return out.reshape(-1)

    def forward_contract(self, batch):
        adapted = self.adapter(batch)
        logits = self.forward(adapted.x).reshape(-1)
        return SeizureModelOutput(
            logits=logits,
            y=adapted.y,
            seq_len=adapted.seq_len,
            supports=adapted.supports,
            adj_mat=adapted.adj_mat,
            writeout_fn=adapted.writeout_fn,
        )
