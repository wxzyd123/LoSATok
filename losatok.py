import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


from modules.configuration_dasheng_tokenizer import DashengTokenizerConfig
from modules.modeling_dasheng_tokenizer import DashengTokenizerModel
from semantic_bottleneck import SemanticEncoder


class AudioVAE(nn.Module):
    SEMANTIC_NATIVE_HZ = 25

    def __init__(
        self,
        config_dict,
        semantic_encoder_path="ckpts/semantic_encoder.pth",
        sample_rate: int = 16000,
        exp_kl_clamp: float = None,
        exp_use_logvar_loss: bool = False,
        semantic_loss_mode: str = "both",
        frame_rate: int = 25,
    ):
        super().__init__()

        assert semantic_loss_mode in ("both", "high_only", "low_only"), (
            f"semantic_loss_mode must be one of 'both', 'high_only', 'low_only', "
            f"got {semantic_loss_mode!r}"
        )
        assert frame_rate % self.SEMANTIC_NATIVE_HZ == 0, (
            f"frame_rate must be an integer multiple of {self.SEMANTIC_NATIVE_HZ}, "
            f"got {frame_rate}"
        )

        self.config_dict = config_dict
        self.exp_kl_clamp = exp_kl_clamp
        self.exp_use_logvar_loss = exp_use_logvar_loss
        self.semantic_loss_mode = semantic_loss_mode
        self.frame_rate = frame_rate
        self.semantic_upsample_factor = frame_rate // self.SEMANTIC_NATIVE_HZ

        self.config = DashengTokenizerConfig(**config_dict)
        self.dasheng_tokenizer = DashengTokenizerModel(self.config)
        self.dasheng_tokenizer.encoder.model = nn.Identity()

        self.semantic_encoder = SemanticEncoder(
            high_dim=self.config_dict["embed_dim"],
            low_dim=self.config_dict["decoder_dim"],
            hidden_dim=512,
        )
        print(f"Loading semantic encoder from {semantic_encoder_path}")
        self.semantic_encoder.load_state_dict(torch.load(semantic_encoder_path)["state_dict"])
        self.freeze_semantic_encoder()

        self.fc_a = nn.Linear(self.config_dict["embed_dim"], self.config_dict["decoder_dim"])
        self.fc_mu = nn.Linear(self.config_dict["decoder_dim"], self.config_dict["decoder_dim"])
        self.fc_logvar = nn.Linear(self.config_dict["decoder_dim"], self.config_dict["decoder_dim"])

        self.sample_rate = sample_rate


    def freeze_semantic_encoder(self):
        self.semantic_encoder.eval()
        for param in self.semantic_encoder.parameters():
            param.requires_grad = False


    def _upsample_semantic(self, x: torch.Tensor, target_length: int) -> torch.Tensor:
        if x.shape[1] == target_length:
            return x

        return F.interpolate(
            x.transpose(1, 2),
            size=target_length,
            mode="linear",
            align_corners=False,
        ).transpose(1, 2)

    
    def encoder_forward(
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
            semantic_emb, semantic_emb_low, _, _ = self.semantic_encoder(input)

        mel = self.dasheng_tokenizer.encoder.front_end(input).unsqueeze(1)
        mel_emb = self.dasheng_tokenizer.encoder.patch_embed(mel)
        acoustic_emb = rearrange(mel_emb, "b c f t -> b (f t) c")
        acoustic_emb = self.dasheng_tokenizer.encoder.norm(acoustic_emb)

        if self.semantic_upsample_factor > 1:
            target_length = semantic_emb.shape[1] * self.semantic_upsample_factor
            semantic_emb = self._upsample_semantic(semantic_emb, target_length)
            semantic_emb_low = self._upsample_semantic(semantic_emb_low, target_length)

        
        min_time = min(semantic_emb.shape[1], acoustic_emb.shape[1])
        semantic_emb = semantic_emb[:, :min_time, :]
        semantic_emb_low = semantic_emb_low[:, :min_time, :]
        acoustic_emb = acoustic_emb[:, :min_time, :]
        emb = semantic_emb + acoustic_emb

        acoustic_emb_low = self.fc_a(acoustic_emb)
        unified_emb_low = acoustic_emb_low + semantic_emb_low

        result_dict = {
            "semantic_emb": semantic_emb,
            "acoustic_emb": acoustic_emb,
            "unified_emb": emb,
            "unified_emb_low": unified_emb_low,
            "semantic_emb_low": semantic_emb_low,
            "acoustic_emb_low": acoustic_emb_low,
        }

        return result_dict


    def decode(self, z):
        audio_reconstructed = self.dasheng_tokenizer.decode(z)

        return audio_reconstructed

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)

        return mu + eps * std

    def kl_loss(self, mu, logvar):
        kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        if self.exp_kl_clamp is not None:
            kl_per_dim = torch.clamp(kl_per_dim, min=self.exp_kl_clamp)
        return kl_per_dim.sum(dim=-1).mean()


    def forward(self, audio_data, **kwargs):
        result_embeddings = self.encoder_forward(audio_data, kwargs.get("attention_mask", None))
        semantic_emb = result_embeddings["semantic_emb"]
        acoustic_emb = result_embeddings["acoustic_emb"]
        unified_emb = result_embeddings["unified_emb"]
        acoustic_emb_low = result_embeddings["acoustic_emb_low"]
        semantic_emb_low = result_embeddings["semantic_emb_low"]
        unified_emb_low = result_embeddings["unified_emb_low"]

        semantic_loss_low = F.mse_loss(acoustic_emb_low, semantic_emb_low.detach())
        semantic_loss = F.mse_loss(acoustic_emb, semantic_emb.detach())
        if self.semantic_loss_mode == "both":
            loss = semantic_loss_low + semantic_loss
        elif self.semantic_loss_mode == "high_only":
            loss = semantic_loss
        elif self.semantic_loss_mode == "low_only":
            loss = semantic_loss_low
        else:
            raise ValueError(f"Unknown semantic_loss_mode: {self.semantic_loss_mode!r}")

        mu = self.fc_mu(unified_emb_low)
        logvar = self.fc_logvar(unified_emb_low)
        logvar = torch.clamp(logvar, min=-20, max=20)

        logvar_loss = None
        if self.exp_use_logvar_loss:
            logvar_loss = torch.mean(torch.sum(logvar.pow(2), dim=-1))

        z = self.reparameterize(mu, logvar)
        kl_loss = self.kl_loss(mu, logvar)

        target_emb = z
        audio_reconstructed = self.dasheng_tokenizer.decode(target_emb)

        return {
            "audio": audio_reconstructed,
            "z": z,
            "acoustic_emb": acoustic_emb,
            "semantic_emb": semantic_emb,
            "unified_emb": unified_emb,
            "unified_emb_low": unified_emb_low,
            "semantic_emb_low": semantic_emb_low,
            "acoustic_emb_low": acoustic_emb_low,
            "vae/mu": mu,
            "vae/logvar": logvar,
            "vae/semantic_loss": loss,
            "vae/kl_loss": kl_loss,
            "vae/logvar_loss": logvar_loss,
        }


def load_model_from_yaml(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    audio_vae_config = {
        k.split("AudioVAE.")[1]: v
        for k, v in config.items()
        if k.startswith("AudioVAE.")
    }
    model = AudioVAE(**audio_vae_config)

    return model


if __name__ == "__main__":
    config_dict = {
        "decoder_depth": 12,
        "decoder_embed_dim": 1280,
        "decoder_dim": 128,
        "decoder_intermediate_size": 5120,
        "depth": 32,
        "dtype": "float32",
        "embed_dim": 1280,
        "hop_length": 160,
        "istft_hop": 320,
        "istft_n_fft": 1280,
        "model_type": "dashengtokenizer",
        "n_mels_patch": 128,
        "num_heads": 16,
        "transformers_version": "5.1.0",
        "upsample_tokens": 2,
    }
    audio_vae = AudioVAE(config_dict)

    print(audio_vae)