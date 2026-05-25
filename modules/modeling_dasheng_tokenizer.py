from modules.configuration_dasheng_tokenizer import DashengTokenizerConfig
from modules.modeling_dasheng_encoder import DashengEncoder
from modules.vocos import VocosModel
from typing import Optional, Tuple, Union
import torch
import torch.nn as nn
from einops import rearrange
import torchaudio
from transformers import PreTrainedModel


class VocosMelSpec(torch.nn.Module):
    """MelSpectrogram frontend for Vocos."""
    def __init__(self, sample_rate=16000, n_fft=1024, hop_length=256, n_mels=100, padding="center"):
        super().__init__()
        if padding not in ["center", "same"]:
            raise ValueError("Padding must be 'center' or 'same'.")
        self.padding = padding
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        with torch.device("cpu"):
            self.mel_spec = torchaudio.transforms.MelSpectrogram(
                    sample_rate=self.sample_rate,
                    n_fft=self.n_fft,
                    hop_length=self.hop_length,
                    n_mels=self.n_mels,
                    center=self.padding == "center",
                    power=1,)

    def forward(self, audio, **kwargs):
        if self.padding == "same":
            pad = self.mel_spec.win_length - self.mel_spec.hop_length
            audio = torch.nn.functional.pad(audio, (pad // 2, pad // 2), mode="reflect")
        mel = self.mel_spec(audio)
        return torch.log(torch.clip(mel, min=1e-7))


class DashengTokenizerEncoder(torch.nn.Module):
    def __init__(
        self,
        embed_dim: int = 1280,
        depth:int = 32,
        num_heads: int = 16,
        n_mels_patch: int = 128,
        hop_length: int = 160,
        patch_stride_time: int = 4,
        **kwargs,
    ):
        super().__init__()
        self.model = DashengEncoder(embed_dim=embed_dim, depth=depth, num_heads=num_heads)
        self.embed_dim = int(self.model.embed_dim)
        self.model.outputlayer = torch.nn.Identity()

        self.front_end = VocosMelSpec(hop_length=hop_length, n_mels=n_mels_patch)
        self.patch_embed = torch.nn.Conv2d(
            1, self.model.embed_dim,
            (n_mels_patch, patch_stride_time),
            (n_mels_patch, patch_stride_time),
        )
        self.norm = torch.nn.LayerNorm(self.model.embed_dim)

        # Store parameters for reference
        self.n_fft = self.model.front_end.n_fft
        self.hop_size = self.model.front_end.hop_size

    # @torch.no_grad()
    def forward(
        self,
        input: torch.Tensor,
        input_attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Forward pass of the encoder.

        Args:
            input: Audio tensor of shape (batch_size, num_samples)
            input_attn_mask: Optional attention mask

        Returns:
            Combined embeddings of shape (batch_size, num_tokens, embed_dim)
        """
        input = input.squeeze(1)
        with torch.no_grad():
            semantic_emb = self.model(input, input_attn_mask)[0]

        # acoustic part
        mel = self.front_end(input).unsqueeze(1)
        mel_emb = self.patch_embed(mel)
        acoustic_emb = rearrange(mel_emb, "b c f t -> b (f t) c")
        acoustic_emb = self.norm(acoustic_emb)

        semantic_emb = semantic_emb[:, : acoustic_emb.shape[1], :]
        emb = semantic_emb + acoustic_emb

        result_dict = {
            "semantic_emb": semantic_emb,
            "acoustic_emb": acoustic_emb,
            "unified_emb": emb,
        }

        return result_dict


class DashengTokenizerPreTrainedModel(PreTrainedModel):

    config_class = DashengTokenizerConfig
    supports_gradient_checkpointing = True

class DashengTokenizerModel(DashengTokenizerPreTrainedModel):
    """
    HuggingFace-compatible DashEng Tokenizer Model (Encoder + Decoder).

    This model includes both the encoder and decoder for end-to-end audio processing.
    """

    def __init__(self, config: DashengTokenizerConfig):
        super().__init__(config)
        self.config = config

        self.encoder = DashengTokenizerEncoder(
            embed_dim=config.embed_dim,
            depth = config.depth,
            num_heads=config.num_heads,
            n_mels_patch=config.n_mels_patch,
            hop_length=config.hop_length,
            patch_stride_time=getattr(config, "patch_stride_time", 4),
        )

        self.embed_dim = self.encoder.embed_dim

        self.upsampler = None
        if config.upsample_tokens >= 1:
            self.upsampler = torch.nn.ConvTranspose1d(
                config.decoder_dim, self.embed_dim,
                kernel_size=config.upsample_tokens,
                stride=config.upsample_tokens
            )

        # Decoder
        self.decoder = VocosModel(
            input_channels=self.embed_dim,
            hidden_dim=config.decoder_embed_dim,
            intermediate_dim=config.decoder_intermediate_size,
            vocos_istft_hop=config.istft_hop,
            vocos_n_fft=config.istft_n_fft,
            num_layers=config.decoder_depth,
        )

        self.post_init()

    def encode(
        self,
        audio: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode audio into embeddings."""
        return self.encoder(audio, attention_mask)

    def decode(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Decode embeddings back to audio."""
        if self.upsampler is not None:
            embeddings = self.upsampler(embeddings.transpose(-2, -1)).transpose(-2, -1)
        output = self.decoder(embeddings.transpose(-2, -1))
        return output

    def forward(
        self,
        audio: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], dict]:
        """
        Forward pass of the DashEng tokenizer.

        Args:
            audio: Audio tensor of shape (batch_size, num_samples)
            attention_mask: Optional attention mask
            output_attentions: Whether to return attention weights
            output_hidden_states: Whether to return hidden states
            return_dict: Whether to return a dict

        Returns:
            Reconstructed audio of shape (batch_size, num_samples)
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Encode
        # embeddings = self.encoder(audio, attention_mask)
        result_embeddings = self.encoder(audio, attention_mask)
        semantic_emb = result_embeddings["semantic_emb"]
        acoustic_emb = result_embeddings["acoustic_emb"]
        embeddings = result_embeddings["unified_emb"]

        # Decode
        audio_reconstructed = self.decode(embeddings)

        if not return_dict:
            return (audio_reconstructed,)

        return {
            "audio": audio_reconstructed,
            "semantic_emb": semantic_emb,
            "acoustic_emb": acoustic_emb,
            "unified_emb": embeddings,
        }
    