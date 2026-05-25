# LoSATok: Low-dimensional Semantic-Acoustic Tokenizer

[![github](https://img.shields.io/badge/Code-Repo-black?logo=github)](https://github.com/wxzyd123/LoSATok)
[![arXiv](https://img.shields.io/badge/%F0%9F%93%84%20ArXiv-Paper-red.svg)]()
[![model](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Model-yellow.svg)](https://huggingface.co/wxzyd123/LoSATok)

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![PyTorch 2.8](https://img.shields.io/badge/PyTorch-2.8-EE4C2C.svg?logo=pytorch)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**LoSATok** is a **continuous, low-dimensional, 25 Hz** audio tokenizer that jointly
models **semantic** and **acoustic** information in a single latent space. 
LoSATok can performance competitive cross-domain understanding and generation capabilities with a **128-dim** latent `z`. 

See our paper "[LoSATok: Low-dimensional Semantic-Acoustic Tokenizer for Cross-Domain Audio Understanding and Generation]()".

---
## ✨ Key Features
- **128-Dim Semantic-Acoustic Latent** — A compact continuous representation that unifies semantic and acoustic information for both understanding and generation.
- **Semantic Bottleneck (SemBo)** — Compresses frozen MiDashengLM features from 1280-dim to 128-dim with temporal relation preservation.
- **Dual-Level Semantic Supervision** — Uses both high- and low-dimensional semantic targets to balance semantic richness and acoustic reconstruction.
- **DiT-Efficient Generation** — Makes downstream diffusion modeling easier, improving convergence and generation quality for TTS, TTM, and TTA.
- **Cross-Domain Audio Modeling** — Works across speech, music, and general audio within a single tokenizer.

---

## 🛠️ Setup

We test our environment on **Ubuntu 20.04** with **Python 3.10** and **CUDA 12.x**.

### Env Setup

```bash
conda create -n losatok python=3.10 -y
conda activate losatok
```

### Basic Requirements

```bash
git clone https://github.com/wxzyd123/LoSATok.git
cd LoSATok

pip install torch==2.8.0 torchaudio==2.8.0
pip install -r requirements.txt

sudo apt install ffmpeg
```

---

## 📦 Checkpoints

LoSATok needs two checkpoints:

| File                          | Description                                                                                |
| ----------------------------- | ------------------------------------------------------------------------------------------ |
| `ckpts/semantic_encoder.pth`  | Frozen MiDashengLM semantic encoder + pretrained Semantic Bottleneck checkpoint.           |
| `ckpts/losatok_kl1e-3.pth` or  `ckpts/losatok_kl1e-2.pth`    | LoSATok checkpoint. `kl1e-3` and `kl1e-2` correspond to different KL clamp strengths.      |

Place them under the `ckpts/` directory so that the layout looks like:


The semantic encoder loads the [MiDashengLM-7B](https://huggingface.co/mispeech/midashenglm-7b)
backbone at construction time.

You can download these checkpoints from the anonymous huggingface links [🤗 here](https://huggingface.co/wxzyd123/LoSATok).

---

## 🚀 Usage

### 1. Reconstruction via CLI

The simplest way to test LoSATok is to reconstruct a wav file:

```bash
python infer.py \
    --config_path config/16k_16k_25Hz_losatok.yml \
    --model_path  ckpts/losatok_kl1e-2.pth \
    --input_path  example/en.wav \
    --output_path example/recon.wav \
    --save_features example/en_features.pt
```

**Arguments**

| Argument          | Description                                                                                  |
| ----------------- | -------------------------------------------------------------------------------------------- |
| `--config_path`   | YAML config that defines the `AudioVAE` skeleton. Default: `config/16k_16k_25Hz_losatok.yml`. |
| `--model_path`    | Trained LoSATok checkpoint (`*.pth`). If omitted, an untrained model is built (debug only).  |
| `--input_path`    | Input wav path.                                                                              |
| `--output_path`   | Reconstructed wav path.                                                                      |
| `--device`        | `cuda` (default) or `cpu`.                                                                   |
| `--sample`        | If set, use the reparameterized `z = mu + eps * std` instead of the deterministic `mu`.      |
| `--max_duration`  | Optional float, clip the input to the first *N* seconds (avoids OOM on long files).          |
| `--save_features` | Optional `.pt` path to also dump the full encoded feature dict.                              |

After running, the script prints the shapes of all encoded features, e.g.:

```text
[LoSATok] Encoded feature shapes:
  z                  (1, T_token, 128)
  mu                 (1, T_token, 128)
  logvar             (1, T_token, 128)
  semantic_emb       (1, T_token, 1280)
  acoustic_emb       (1, T_token, 1280)
  unified_emb        (1, T_token, 1280)
  semantic_emb_low   (1, T_token, 128)
  acoustic_emb_low   (1, T_token, 128)
  unified_emb_low    (1, T_token, 128)
```

### 2. Python API

```python
import librosa, torch
from infer import load_losatok, encode, decode

model = load_losatok(
    config_path="config/16k_16k_25Hz_losatok.yml",
    model_path="ckpts/losatok_kl1e-3.pth",
    device="cuda",
)

wav, sr = librosa.load("example/en.wav", sr=model.sample_rate, mono=True)
audio = torch.from_numpy(wav).unsqueeze(0)

# ---- Encode: waveform -> LoSATok tokens ----
features = encode(model, audio)

z                = features["z"]                   # (B, T_token, 128)  <- LoSATok tokens
mu               = features["mu"]                  # (B, T_token, 128)
logvar           = features["logvar"]              # (B, T_token, 128)
semantic_emb     = features["semantic_emb"]        # (B, T_token, 1280)
acoustic_emb     = features["acoustic_emb"]        # (B, T_token, 1280)
unified_emb      = features["unified_emb"]         # (B, T_token, 1280)
semantic_emb_low = features["semantic_emb_low"]    # (B, T_token, 128)
acoustic_emb_low = features["acoustic_emb_low"]    # (B, T_token, 128)
unified_emb_low  = features["unified_emb_low"]     # (B, T_token, 128)

# ---- Decode: LoSATok tokens -> waveform ----
audio_recon = decode(model, z)
```

`z` is the **128-dim continuous LoSATok token**, typically fed to downstream models
(LLMs, DiTs, etc.).



---

## ❤️ Acknowledgements

We sincerely thank these excellent open-source work:

- [DashengTokenizer](https://huggingface.co/mispeech/dashengtokenizer)
- [UniFlow-Audio](https://github.com/wsntxxn/UniFlow-Audio)
- [XARES](https://github.com/jimbozhang/xares)
- [Ming-UniAudio](https://github.com/inclusionAI/Ming-UniAudio)


---

## 📄 License

The code in this repository is released under the **MIT** license. See [LICENSE](LICENSE) for details.