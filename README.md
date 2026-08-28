<div align="center">

# CBraMod


_A Criss-Cross Brain Foundation Model for EEG Decoding_


[![Paper](https://img.shields.io/badge/arXiv-2412.07236-red)](https://arxiv.org/abs/2412.07236)
[![Paper](https://img.shields.io/badge/Paper-ICLR-008B8B)](https://openreview.net/forum?id=NPNUHgHF2w)
[![huggingface](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-FFD21E)](https://huggingface.co/weighting666/CBraMod)
![GitHub Repo stars](https://img.shields.io/github/stars/wjq-learning/CBraMod)

</div>


<div align="center">
<img src="figure/CBraMod_logo.png" style="width: 15%;" />
</div>


<p align="center">
    🔍&nbsp;<a href="#-about">About</a>
    | 🔨&nbsp;<a href="#-setup">Setup</a>
    | 🚢&nbsp;<a href="#-pretrain">Pretrain</a>
    | ⛵&nbsp;<a href="#-finetune">Finetune</a>
    | 🚀&nbsp;<a href="#-quick-start">Quick Start</a>
    | 🔗&nbsp;<a href="#-citation">Citation</a>
</p>
🔥 NEWS: Thanks to over 100 stars! We've further refined the code for improved stability. Appreciate your patience as we refine the implementation — ongoing EEG research continues to shape the development of a standardized pipeline.

🔥 NEWS: The paper "_CBraMod: A Criss-Cross Brain Foundation Model for EEG Decoding_" has been accepted by ICLR 2025!

## 🔍 About
We propose **CBraMod**, a novel EEG foundation model, for EEG decoding on various clinical and BCI application.
The preprint version of our paper is available at [arXiv](https://arxiv.org/abs/2412.07236). 
The camera-ready version of the paper will be available at [OpenReview](https://openreview.net/forum?id=NPNUHgHF2w).
<div align="center">
<img src="figure/model.png" style="width:100%;" />
</div>



## 🔨 Setup
Install [Python](https://www.python.org/downloads/).

Install [PyTorch](https://pytorch.org/get-started/locally/).

Install other requirements:
```commandline
pip install -r requirements.txt
``` 


## 🚢 Pretrain
You can pretrain CBraMod on our pretraining dataset or your custom pretraining dataset using the following code:
```commandline
python pretrain_main.py
```
We have released a pretrained checkpoint on [Hugginface🤗](https://huggingface.co/weighting666/CBraMod).

## ⛵ Finetune
You can finetune CBraMod on our selected downstream datasets using the following code:
```commandline
python finetune_main.py
```

### EvoBrain-compatible seizure detection

The `TUSZ` and `CHB-MIT` fine-tuning paths use the six-field EvoBrain batch
contract `(x, y, seq_len, supports, adj_mat, writeout_fn)`. CBraMod consumes
raw one-second patches, so these paths enforce `--no-use_fft`; 100-bin FFT
features are rejected rather than transformed back into phase-less pseudo-raw
signals.

TUSZ reads continuous EDF recordings and creates non-overlapping 10-second
windows. Raw-domain, training-derived scaler files must be supplied explicitly
when standardization is enabled; do not pass EvoBrain's FFT-domain scaler to
this raw-only path. The repository does not invent replacement statistics.
CHB-MIT reads already-segmented PKLs and does not run a splitter.

```commandline
python finetune_main.py --downstream_dataset TUSZ \
  --raw_data_dir /path/to/TUSZ \
  --scaler_mean_path /path/to/mean.pkl \
  --scaler_std_path /path/to/std.pkl \
  --no-use_fft

python finetune_main.py --downstream_dataset CHB-MIT \
  --datasets_dir /path/to/chb-pkl-splits \
  --no-use_fft
```

Development plumbing can be checked without a dataset using:

```commandline
python -m unittest tests.test_evobrain_seizure_contract
```


## 🚀 Quick Start
You can fine-tune the pretrained CBraMod on your custom downstream dataset using the following example code:
```python
import torch
import torch.nn as nn
from models.cbramod import CBraMod
from einops.layers.torch import Rearrange

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = CBraMod().to(device)
model.load_state_dict(torch.load('pretrained_weights/pretrained_weights.pth', map_location=device))
model.proj_out = nn.Identity()
classifier = nn.Sequential(
  Rearrange('b c s p -> b (c s p)'),
  nn.Linear(22*4*200, 4*200),
  nn.ELU(),
  nn.Dropout(0.1),
  nn.Linear(4 * 200, 200),
  nn.ELU(),
  nn.Dropout(0.1),
  nn.Linear(200, 4),
).to(device)

# mock_eeg.shape = (batch_size, num_of_channels, time_segments, points_per_patch)
mock_eeg = torch.randn((8, 22, 4, 200)).to(device)

# logits.shape = (batch_size, num_of_classes)
logits = classifier(model(mock_eeg))
```



## 🔗 Citation
If you're using this repository in your research or applications, please cite using the following BibTeX:
```bibtex
@inproceedings{wang2025cbramod,
    title={{CB}raMod: A Criss-Cross Brain Foundation Model for {EEG} Decoding},
    author={Jiquan Wang and Sha Zhao and Zhiling Luo and Yangxuan Zhou and Haiteng Jiang and Shijian Li and Tao Li and Gang Pan},
    booktitle={The Thirteenth International Conference on Learning Representations},
    year={2025},
    url={https://openreview.net/forum?id=NPNUHgHF2w}
}
```

## ⭐ Star History
<div align="center">
    <a href="https://star-history.com/#wjq-learning/CBraMod&Date">
        <img src="https://api.star-history.com/svg?repos=wjq-learning/CBraMod&type=Date" style="width: 80%;" />
    </a>
</div>
