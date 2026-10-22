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
    def __init__(
        self,
        num_tags: int = 24,
        emb_dim: int = 64,
        style_dim: int = 256,
        acoustic_scale: float = 1.0,
        prosodic_scale: float = 1.0,
    ):
        super().__init__()
        self.num_tags = num_tags
        self.emb_dim = emb_dim
        self.style_dim = style_dim
        self.acoustic_scale = float(acoustic_scale)
        self.prosodic_scale = float(prosodic_scale)

        # Learnable embedding table for each tag
        self.tag_embeddings = nn.Parameter(torch.randn(num_tags, emb_dim) * 0.02)

        # Baseline neutral embedding when intensity is zero
        self.neutral_embedding = nn.Parameter(torch.randn(1, emb_dim) * 0.02)

        # Multi-layer projection head mapping combined tag features to StyleTTS2 style space
        in_features = emb_dim + num_tags  # continuous embedding blend + raw intensity vector
        final_proj = nn.Linear(128, style_dim)
        # Initialize final projection with std=0.08 (matching teacher coordinate std ~0.29)
        # so fresh training runs start with healthy dynamic range matching the StyleTTS2 teacher manifold
        nn.init.normal_(final_proj.weight, mean=0.0, std=0.08)
        nn.init.zeros_(final_proj.bias)

        self.proj = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            final_proj
        )

    def set_scale_multipliers(self, acoustic: float = 2.5, prosodic: float = 3.1):
        """Sets scale multipliers to bridge stage 1 under-scaled representations."""
        self.acoustic_scale = float(acoustic)
        self.prosodic_scale = float(prosodic)

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

        # Apply acoustic / prosodic scale multipliers if configured (e.g. for Stage 2 bridging)
        if self.acoustic_scale != 1.0 or self.prosodic_scale != 1.0:
            s_tag = torch.cat([
                s_tag[:, :128] * self.acoustic_scale,
                s_tag[:, 128:] * self.prosodic_scale,
            ], dim=-1)

        return s_tag


def compute_kion_style_loss(s_tag, s_audio, lambda_cos=1.0, lambda_mse=10.0, lambda_l1=1.0, lambda_reg=1.0):
    """
    Unified style alignment loss helper.
    Delegates directly to KionStyleAlignmentLoss (the single source of truth in losses.py).
    """
    from .losses import KionStyleAlignmentLoss
    loss_fn = KionStyleAlignmentLoss(
        lambda_cos=lambda_cos,
        lambda_mse=lambda_mse,
        lambda_l1=lambda_l1,
        lambda_reg=lambda_reg,
    ).to(s_tag.device)
    return loss_fn(s_tag, s_audio)
