from einops import rearrange
from einops.layers.torch import Rearrange
import torchaudio.transforms as audio_transforms
import torch
import torch.nn as nn
from typing import Optional, Type

class FrontEnd(nn.Sequential):
    def __init__(
        self,
        f_min: int = 0,
        sample_rate: int = 16000,
        win_size: int = 512,
        center: bool = True,
        n_fft: int = 512,
        f_max: Optional[int] = 8000,
        hop_size: int = 160,
        n_mels: int = 64,
    ):
        self.f_min = f_min
        self.sample_rate = sample_rate
        self.win_size = win_size
        self.center = center
        self.n_fft = n_fft
        self.f_max = f_max
        self.hop_size = hop_size
        self.n_mels = n_mels

        with torch.device("cpu"):
            super().__init__(
                audio_transforms.MelSpectrogram(
                    f_min=self.f_min,
                    sample_rate=self.sample_rate,
                    win_length=self.win_size,
                    center=self.center,
                    n_fft=self.n_fft,
                    f_max=self.f_max,
                    hop_length=self.hop_size,
                    n_mels=self.n_mels,
                ),
                audio_transforms.AmplitudeToDB(top_db=120),
            )

    @torch.autocast(enabled=False, device_type="cuda")
    def forward(self, x, attention_mask=None):
        """
        Forward pass of the frontend.

        Args:
            x: Audio tensor of shape (batch_size, num_samples)
            attention_mask: Optional attention mask of shape (batch_size, num_samples)

        Returns:
            features: Mel spectrogram features of shape (batch_size, n_mels, num_frames)
            attention_mask: Downsampled attention mask of shape (batch_size, num_frames)
        """
        features = super().forward(x)
        if attention_mask is not None:
            lengths = attention_mask.float().sum(-1) // self.hop_size
            attention_mask = (torch.arange(features.shape[-1], device=features.device) < lengths.unsqueeze(-1)).int()
        return features, attention_mask


class Mlp(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Type[torch.nn.Module] = nn.GELU,
        drop: float = 0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        causal: bool = False,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.causal = causal

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, mask: Optional[torch.Tensor] = None):
        B, N, C = x.shape
        # qkv: [3, B, heads, N, head_dim]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale

        # Apply Causal Mask
        if self.causal:
            c_mask = torch.ones(N, N, device=x.device, dtype=torch.bool).triu(1)
            attn = attn.masked_fill(c_mask, float("-inf"))

        # Apply Padding Mask (B, N) -> (B, 1, 1, N)
        if mask is not None:
            if mask.dtype != torch.bool:
                padding_mask = (mask == 0) 
            else:
                padding_mask = mask
            padding_mask = padding_mask.view(B, 1, 1, N)
            attn = attn.masked_fill(padding_mask, float("-inf"))
        attn = attn.softmax(dim=-1).nan_to_num()
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj_drop(self.proj(x))


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, num_heads, qkv_bias, attn_drop, drop)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=nn.GELU, drop=drop)

    def forward(self, x, mask=None):
        x = x + self.attn(self.norm1(x), mask=mask)
        x = x + self.mlp(self.norm2(x))
        return x


class AudioPatchEmbed(torch.nn.Module):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.stride = kwargs.get('stride', [None, 4])[-1]
        self.proj = nn.Conv2d(*args, **kwargs)

    def forward(self, x:torch.Tensor, attention_mask:torch.Tensor | None =None):
        x = self.proj(x)
        if attention_mask is not None:
            lengths = attention_mask.float().sum(-1) // self.stride
            attention_mask = (torch.arange(x.shape[-1], device=x.device) < lengths.unsqueeze(-1)).int()
        return x, attention_mask




class DashengEncoder(nn.Module):
    def __init__(
        self,
        embed_dim: int = 1280,
        depth: int = 32,
        num_heads: int = 20,
        patch_size=[64, 4],
        patch_stride=[64, 4],
        target_length=1008,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.time_patches = patch_stride[-1]
        self.front_end = FrontEnd()
        self.target_length = target_length
        self.max_t_tokens = target_length // patch_stride[-1]
        self.patch_embed = AudioPatchEmbed(1, embed_dim, kernel_size=patch_size, stride=patch_stride)
        self.init_bn = nn.Sequential(
            Rearrange("b c f t -> b f c t"),
            torch.nn.BatchNorm2d(self.front_end.n_mels, momentum=0.01),
            Rearrange("b f c t -> b c f t"),
        )

        self.time_pos_embed = nn.Parameter(torch.randn(1, embed_dim, 1, target_length // self.time_patches) * 0.02)
        self.freq_pos_embed = nn.Parameter(torch.randn(1, embed_dim, 1, 1) * 0.02)

        self.blocks = nn.ModuleList([Block(embed_dim, num_heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)

    def _forward_main(self, x, attention_mask, mask_to_zero:bool = False):
        x, attention_mask = self.patch_embed(x, attention_mask)
        t = x.shape[-1]
        x = x + self.time_pos_embed[:, :, :, :t] + self.freq_pos_embed
        x = rearrange(x, "b c f t -> b (f t) c")
        for block in self.blocks:
            x = block(x, mask=attention_mask)
        x = self.norm(x) 
        if attention_mask is not None and mask_to_zero:
            x = x * attention_mask.unsqueeze(-1) # Zero out all samples that were masked, but only after first chunk
        return x 


    def forward(self, x: torch.Tensor, attention_mask=None):
        """
        Forward pass of the AudioTransformer.

        Args:
            x: Audio tensor of shape (batch_size, num_samples)
            attention_mask: Optional attention mask of shape (batch_size, num_samples)
                           where True indicates valid samples and False indicates padding

        Returns:
            embeddings: Token embeddings of shape (batch_size, num_tokens, embed_dim)
        """
        # Process through frontend - returns features and downsampled mask
        x, attention_mask = self.front_end(x, attention_mask)

        # Rearrange features for patch embedding: (b f t) -> (b 1 f t)
        x = rearrange(x, "b f t -> b 1 f t")
        x = self.init_bn(x)

        input_splits = x.split(self.target_length, dim=-1)
        masks = [None for _ in range(len(input_splits))]
        if attention_mask is not None:
            masks = attention_mask.split(self.target_length, dim=-1)

        outputs = []
        for i, (input_split_x, mask) in enumerate(zip(input_splits, masks)):
            output = self._forward_main(input_split_x, attention_mask=mask, mask_to_zero=i != 0)
            outputs.append(output)
        x = torch.cat(outputs, dim=1)
        return x