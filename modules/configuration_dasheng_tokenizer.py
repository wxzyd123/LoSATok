"""
Dasheng Audio Tokenizer Configuration
"""

from transformers import PretrainedConfig

class DashengTokenizerConfig(PretrainedConfig):
    """
    Configuration class for DashEng Audio Tokenizer.

    This configuration is used to initialize the DashEng model with the same
    parameters as the original implementation in models.py.

    Args:
        target_nmels (int): Number of Mel bins for the frontend. Default: 100
        decoder_embed_dim (int): Decoder embedding dimension. Default: 768
        decoder_depth (int): Number of decoder layers. Default: 8
        decoder_intermediate_size (int): Decoder intermediate size. Default: 1536
        istft_n_fft (int): ISTFT n_fft parameter. Default: 1280
        istft_hop (int): ISTFT hop parameter. Default: 640
        upsample_tokens (int): Upsample factor for tokens. Default: 1
        n_mels_patch (int): Number of Mel bins for patch embedding. Default: 100
        hop_length (int): Hop length for Mel spectrogram. Default: 160
        patch_stride_time (int): Time-axis stride of patch_embed Conv2d.
            With hop_length=160 (mel @100Hz), stride=4 gives 25Hz acoustic_emb,
            stride=2 gives 50Hz acoustic_emb. Default: 4
    """

    model_type = "dashengtokenizer"

    def __init__(
        self,
        embed_dim: int = 1280,
        depth:int = 32,
        num_heads: int = 16,
        decoder_embed_dim: int = 1280,
        decoder_depth: int = 12,
        decoder_intermediate_size: int = 5120,
        istft_n_fft: int = 1280, 
        istft_hop: int = 320, # 20ms
        upsample_tokens: int = 2,
        n_mels_patch: int = 128, # acoustic nmel
        hop_length: int = 160, # acoustic hop
        patch_stride_time: int = 4, # acoustic time-stride of patch_embed
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.decoder_embed_dim = decoder_embed_dim
        self.decoder_depth = decoder_depth
        self.decoder_intermediate_size = decoder_intermediate_size
        self.istft_n_fft = istft_n_fft
        self.istft_hop = istft_hop
        self.upsample_tokens = upsample_tokens
        self.n_mels_patch = n_mels_patch
        self.hop_length = hop_length
        self.patch_stride_time = patch_stride_time
