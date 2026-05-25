import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModelForCausalLM


def semantic_bottleneck_loss(emb, emb_recon, emb_low):
    mse_loss = F.mse_loss(
            F.normalize(emb_recon, dim=-1),
            F.normalize(emb.detach(), dim=-1)
    )

    h_matrix = F.normalize(emb, dim=-1)
    l_matrix = F.normalize(emb_low, dim=-1)
    sim_H = h_matrix @ h_matrix.transpose(-1, -2)   # [B, T, T]
    sim_L = l_matrix @ l_matrix.transpose(-1, -2)   # [B, T, T]
    relation_loss = F.mse_loss(sim_L, sim_H.detach())

    total_loss = mse_loss + relation_loss
    
    loss_dict = {
        "mse_loss": mse_loss,
        "relation_loss": relation_loss,
        "total_loss": total_loss,
    }
    return loss_dict


class SemanticBottleneck(nn.Module):
    def __init__(
        self, 
        high_dim: int = 1280, 
        low_dim: int = 128,
        hidden_dim: int = 512,
    ):
        super().__init__()
        self.downsample = nn.Sequential(
            nn.Linear(high_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, low_dim),
        )
        self.upsample = nn.Sequential(
            nn.Linear(low_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, high_dim),
        )

    def forward(self, embeddings):
        z = self.downsample(embeddings)
        recon = self.upsample(z)

        return z, recon


class SemanticEncoder(nn.Module):
    def __init__(
        self,
        high_dim: int = 1280,
        low_dim: int = 128,
        hidden_dim: int = 512,
        semantic_encoder_path: str = "mispeech/midashenglm-7b-1021-fp32",
    ):
        super().__init__()

        self.high_dim = high_dim
        self.low_dim = low_dim
        self.hidden_dim = hidden_dim

        dashenglm = AutoModelForCausalLM.from_pretrained(semantic_encoder_path, trust_remote_code=True)
        self.encoder = dashenglm.audio_encoder
        del dashenglm

        self.bottleneck = SemanticBottleneck(self.high_dim, self.low_dim, self.hidden_dim)

        self.freeze_encoder()

    
    def freeze_encoder(self):
        self.encoder.eval()
        for param in self.encoder.parameters():
            param.requires_grad = False


    def forward(self, audio):
        with torch.no_grad():
            embeddings = self.encoder(audio)[0]
        embeddings = embeddings.detach()
        z, recon = self.bottleneck(embeddings)

        loss_dict = semantic_bottleneck_loss(embeddings, recon, z)

        return embeddings, z, recon, loss_dict


if __name__ == "__main__":
    bottleneck = SemanticBottleneck(high_dim=1280, low_dim=128, hidden_dim=512).cuda()

    embeddings = torch.randn(2, 25, 1280).cuda()
    z, recon = bottleneck(embeddings)

    loss_dict = semantic_bottleneck_loss(embeddings, recon, z)

    print(z.shape)
    print(recon.shape)
    print(loss_dict)