import torch
import torch.nn as nn
import torch.nn.functional as F

class KionTagStyleEncoder(nn.Module):
    """
    Dedicated emotion & style conditioning module for KionTTS.
    Converts 24 emotion/style tags and continuous intensities into StyleTTS2's
    256-dimensional style space:
      - First 128 dims: acoustic style (ref) for decoder AdaIN layers
      - Last 128 dims: prosodic style (s) for duration/pitch/energy predictors
    
    Tags:
      0-13 (14 Emotions):
        angry, annoyed, bored, concerned, confused, curious, disappointed,
        excited, frustrated, happy, heartbroken, overjoyed, sad, surprised
      14-23 (10 Delivery Styles):
        affectionate, authoritative, calm, deadpan, dramatic, playful,
        sarcasm, serious, soothing, teasing
    """
    def __init__(self, num_tags=24, emb_dim=64, style_dim=256):
        super().__init__()
        self.num_tags = num_tags
        self.emb_dim = emb_dim
        self.style_dim = style_dim

        # Learnable embedding table for each tag
        self.tag_embeddings = nn.Parameter(torch.randn(num_tags, emb_dim) * 0.02)

        # Baseline neutral embedding when intensity is zero
        self.neutral_embedding = nn.Parameter(torch.randn(1, emb_dim) * 0.02)

        # Multi-layer projection head mapping combined tag features to StyleTTS2 style space
        in_features = emb_dim + num_tags  # continuous embedding blend + raw intensity vector
        self.proj = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, style_dim),
            nn.LayerNorm(style_dim)
        )

    def forward(self, tag_vectors):
        """
        Args:
            tag_vectors: Tensor of shape (B, num_tags) with float intensities in [0.0, 1.0]
        Returns:
            s_tag: Tensor of shape (B, style_dim) where:
                   ref = s_tag[:, :128] (acoustic style for decoder)
                   s   = s_tag[:, 128:] (prosodic style for predictor)
        """
        # Weighted linear combination of tag embeddings: (B, num_tags) @ (num_tags, emb_dim) -> (B, emb_dim)
        blended_emb = torch.matmul(tag_vectors, self.tag_embeddings)

        # Check for unconditioned / neutral samples
        tag_sums = tag_vectors.sum(dim=-1, keepdim=True)  # (B, 1)
        is_neutral = (tag_sums < 1e-4).float()
        blended_emb = blended_emb + is_neutral * self.neutral_embedding

        # Concatenate embedding representation with explicit intensity vector
        feat = torch.cat([blended_emb, tag_vectors], dim=-1)  # (B, emb_dim + num_tags)

        # Project to 256-d StyleTTS2 style space
        s_tag = self.proj(feat)
        return s_tag


def compute_kion_style_loss(s_tag, s_audio, lambda_cos=0.5):
    """
    Computes grounded style alignment loss between predicted tag style (s_tag)
    and ground-truth reference audio style (s_audio).
    
    Guaranteed Pitfall Protection:
    - Never compares adjacent samples in a batch (avoids mode collapse).
    - Aligns both distance (MSE) and directional angle (Cosine Similarity).
    """
    loss_mse = F.mse_loss(s_tag, s_audio)
    cos_sim = F.cosine_similarity(s_tag, s_audio, dim=-1)
    loss_cos = (1.0 - cos_sim).mean()
    return loss_mse + lambda_cos * loss_cos
