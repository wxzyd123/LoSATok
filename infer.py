"""LoSATok inference utilities.

CLI example:
    python infer.py \
        --config_path config/16k_16k_25Hz_losatok.yml \
        --model_path  ckpts/losatok_kl1e-3.pth or ckpts/losatok_kl1e-3.pth \
        --input_path  example/en.wav \
        --output_path example/recon.wav
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, Optional

import librosa
import numpy as np
import torch
import torchaudio

from losatok import AudioVAE, load_model_from_yaml


def load_losatok(
    config_path: str,
    model_path: Optional[str] = None,
    device: str | torch.device = "cuda",
    strict: bool = True,
) -> AudioVAE:
    """Load a LoSATok model (``AudioVAE``) and switch it to inference mode.

    Args:
        config_path: Path to the yml config, e.g. ``config/16k_16k_25Hz_losatok.yml``.
        model_path:  Path to the trained ``vae.pth`` checkpoint. If ``None``,
                     only the network skeleton is built (useful for debugging).
        device:      Inference device.
        strict:      Whether ``load_state_dict`` should be strict.

    Returns:
        An ``AudioVAE`` in ``eval()`` mode, already moved to ``device``.
    """
    print(f"[LoSATok] Building model from config: {config_path}")
    model = load_model_from_yaml(config_path)

    if model_path is not None:
        print(f"[LoSATok] Loading checkpoint: {model_path}")
        state = torch.load(model_path, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state_dict = state["state_dict"]
        else:
            state_dict = state
        missing, unexpected = model.load_state_dict(state_dict, strict=strict)
        if not strict:
            if missing:
                print(f"[LoSATok] Missing keys: {len(missing)} (first 5: {missing[:5]})")
            if unexpected:
                print(f"[LoSATok] Unexpected keys: {len(unexpected)} (first 5: {unexpected[:5]})")

    model.eval().to(device)
    return model


def _to_batched_audio(
    audio: torch.Tensor | np.ndarray,
    device: str | torch.device,
) -> torch.Tensor:
    """Normalize any input waveform to a float32 tensor of shape ``(B, T)``."""
    if isinstance(audio, np.ndarray):
        audio = torch.from_numpy(audio)
    audio = audio.float()
    if audio.dim() == 1:
        audio = audio.unsqueeze(0)
    elif audio.dim() == 3:
        audio = audio.squeeze(1)
    elif audio.dim() != 2:
        raise ValueError(f"Unsupported audio shape {tuple(audio.shape)}, expect (T,), (B, T) or (B, 1, T).")
    return audio.to(device)


@torch.no_grad()
def encode(
    model: AudioVAE,
    audio: torch.Tensor | np.ndarray,
    *,
    sample: bool = False,
) -> Dict[str, torch.Tensor]:
    """Extract LoSATok encoded features from an input waveform.

    Args:
        model:   Model returned by ``load_losatok``.
        audio:   Waveform of shape ``(T,)`` / ``(B, T)`` / ``(B, 1, T)``. The
                 sample rate must equal ``model.sample_rate`` (16000 by default).
        sample:  If ``True``, use the reparameterized ``z = mu + eps * std`` as
                 the latent; otherwise use ``mu`` (more stable for inference).

    Returns:
        A dict of tensors with shape ``(B, T_token, D)`` unless stated otherwise:
            ``z``                final latent, ``D = decoder_dim``
            ``mu``                Gaussian mean
            ``logvar``            Gaussian log-variance
            ``semantic_emb``      high-dim semantic feature (``embed_dim``)
            ``acoustic_emb``      high-dim acoustic feature (``embed_dim``)
            ``unified_emb``       high-dim semantic + acoustic
            ``semantic_emb_low``  low-dim semantic feature (``decoder_dim``)
            ``acoustic_emb_low``  low-dim acoustic feature (``decoder_dim``)
            ``unified_emb_low``   low-dim semantic + acoustic
    """
    device = next(model.parameters()).device
    audio = _to_batched_audio(audio, device)

    embeds = model.encoder_forward(audio)
    unified_emb_low = embeds["unified_emb_low"]

    mu = model.fc_mu(unified_emb_low)
    logvar = model.fc_logvar(unified_emb_low)
    logvar = torch.clamp(logvar, min=-20, max=20)
    z = model.reparameterize(mu, logvar) if sample else mu

    out = {
        "z": z,
        "mu": mu,
        "logvar": logvar,
        "semantic_emb": embeds["semantic_emb"],
        "acoustic_emb": embeds["acoustic_emb"],
        "unified_emb": embeds["unified_emb"],
        "semantic_emb_low": embeds["semantic_emb_low"],
        "acoustic_emb_low": embeds["acoustic_emb_low"],
        "unified_emb_low": embeds["unified_emb_low"],
    }
    return out


@torch.no_grad()
def decode(model: AudioVAE, z: torch.Tensor) -> torch.Tensor:
    """Decode latent ``z`` back to a waveform of shape ``(B, 1, T)``."""
    device = next(model.parameters()).device
    z = z.to(device)
    audio = model.decode(z)
    return audio


@torch.no_grad()
def reconstruct(
    model: AudioVAE,
    audio: torch.Tensor | np.ndarray,
    *,
    sample: bool = False,
    return_features: bool = False,
):
    """Run a full encode -> decode pass on the input waveform.

    Args:
        model:           LoSATok model.
        audio:           Input waveform.
        sample:          See ``encode``.
        return_features: If ``True``, return ``(audio_recon, features_dict)``.

    Returns:
        The reconstructed waveform ``(B, 1, T)`` or ``(audio, features)``.
    """
    features = encode(model, audio, sample=sample)
    audio_recon = decode(model, features["z"])
    if return_features:
        return audio_recon, features
    return audio_recon


def _load_audio(path: str, target_sr: int) -> torch.Tensor:
    """Load an audio file and resample to ``target_sr``, returning shape ``(1, T)``."""
    wav, sr = librosa.load(path, sr=target_sr, mono=True)
    return torch.from_numpy(wav).unsqueeze(0)


def _save_audio(path: str, waveform: torch.Tensor, sample_rate: int) -> None:
    """Save ``waveform`` of shape ``(C, T)`` or ``(T,)`` to ``path`` (wav)."""
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    elif waveform.dim() == 3:
        waveform = waveform.squeeze(0)
    waveform = waveform.detach().cpu().float().clamp(-1.0, 1.0)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torchaudio.save(path, waveform, sample_rate)


@torch.no_grad()
def infer_file(
    model: AudioVAE,
    input_path: str,
    output_path: str,
    *,
    sample: bool = False,
    max_duration: Optional[float] = None,
    return_features: bool = False,
):
    """Read an audio file, run one LoSATok reconstruction, and save it back.

    Args:
        model:           LoSATok model.
        input_path:      Input wav path.
        output_path:     Output wav path.
        sample:          See ``encode``.
        max_duration:    If given, only the first ``max_duration`` seconds are
                         used, to avoid OOM on very long files.
        return_features: Whether to also return the encoded feature dict.

    Returns:
        The reconstructed 1D numpy waveform, or ``(wav, features)`` if
        ``return_features=True``.
    """
    sample_rate = model.sample_rate
    audio = _load_audio(input_path, sample_rate)
    if max_duration is not None:
        audio = audio[..., : int(max_duration * sample_rate)]

    recon, features = reconstruct(model, audio, sample=sample, return_features=True)
    recon_wav = recon.squeeze(0).squeeze(0).cpu()
    _save_audio(output_path, recon_wav, sample_rate)
    print(f"[LoSATok] Saved reconstruction to {output_path} (len={recon_wav.shape[-1]} samples)")

    if return_features:
        return recon_wav.numpy(), features
    return recon_wav.numpy()


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LoSATok inference (single audio file).")
    parser.add_argument("--config_path", type=str, default="config/16k_16k_25Hz_losatok.yml", help="Path to LoSATok yml config.")
    parser.add_argument("--model_path", type=str, default=None, help="Path to trained checkpoint (vae.pth).")
    parser.add_argument("--input_path", type=str, required=True, help="Input wav path.")
    parser.add_argument("--output_path", type=str, required=True, help="Output wav path.")
    parser.add_argument("--device", type=str, default="cuda", help="Inference device.")
    parser.add_argument("--sample", action="store_true", help="Use reparameterized z instead of mu.")
    parser.add_argument("--max_duration", type=float, default=None, help="Clip input to first N seconds.")
    parser.add_argument(
        "--save_features",
        type=str,
        default=None,
        help="Optional .pt path to also dump the encoded feature dict.",
    )
    return parser


def main() -> None:
    args = _build_argparser().parse_args()

    model = load_losatok(
        config_path=args.config_path,
        model_path=args.model_path,
        device=args.device,
    )

    _, features = infer_file(
        model,
        input_path=args.input_path,
        output_path=args.output_path,
        sample=args.sample,
        max_duration=args.max_duration,
        return_features=True,
    )

    print("[LoSATok] Encoded feature shapes:")
    for k, v in features.items():
        print(f"  {k:<18s} {tuple(v.shape)}")

    if args.save_features is not None:
        os.makedirs(os.path.dirname(os.path.abspath(args.save_features)), exist_ok=True)
        torch.save({k: v.detach().cpu() for k, v in features.items()}, args.save_features)
        print(f"[LoSATok] Saved features to {args.save_features}")


if __name__ == "__main__":
    main()
