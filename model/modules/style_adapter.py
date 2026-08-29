"""
KionTTS Style Conditioning Adapter
Maps emotion/style intensity vectors to continuous latent style embeddings
for conditioning StyleTTS2 prosody predictors and acoustic decoders.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class KionStyleAdapter(nn.Module):
    def __init__(
        self,
        num_emotions: int = 14,
        num_styles: int = 10,
        tag_embed_dim: int = 64,
        latent_style_dim: int = 128,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_emotions = num_emotions
        self.num_styles = num_styles
        self.tag_embed_dim = tag_embed_dim
        self.latent_style_dim = latent_style_dim

        # Learnable canonical embeddings for each emotion and style
        self.emotion_embeddings = nn.Parameter(
            torch.randn(num_emotions, tag_embed_dim) * 0.02
        )
        self.style_embeddings = nn.Parameter(
            torch.randn(num_styles, tag_embed_dim) * 0.02
        )

        # Baseline neutral voice bias vector
        self.neutral_embedding = nn.Parameter(
            torch.randn(tag_embed_dim) * 0.02
        )

        # Multi-layer MLP to map from combined tag embedding space to latent style space
        input_dim = tag_embed_dim * 2
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_style_dim),
        )

    def forward(self, style_weights: torch.Tensor) -> torch.Tensor:
        """
        Args:
            style_weights: Float tensor of shape (B, num_emotions + num_styles)
                           representing continuous intensities in [0, 1].
        Returns:
            latent_style: Tensor of shape (B, latent_style_dim)
        """
        B = style_weights.size(0)
        
        # Split into emotion weights and style weights
        emo_weights = style_weights[:, : self.num_emotions]  # (B, 14)
        sty_weights = style_weights[:, self.num_emotions :]  # (B, 10)

        # Matrix multiply weights with embedding tables: (B, num_tags) @ (num_tags, dim) -> (B, dim)
        emo_vec = torch.matmul(emo_weights, self.emotion_embeddings)
        sty_vec = torch.matmul(sty_weights, self.style_embeddings)

        # Add neutral baseline bias if intensity is low
        total_intensity = torch.sum(emo_weights, dim=-1, keepdim=True) + torch.sum(sty_weights, dim=-1, keepdim=True)
        neutral_weight = torch.clamp(1.0 - total_intensity, min=0.0)
        
        emo_vec = emo_vec + neutral_weight * self.neutral_embedding
        sty_vec = sty_vec + neutral_weight * self.neutral_embedding

        # Concatenate and pass through projection MLP
        combined = torch.cat([emo_vec, sty_vec], dim=-1)  # (B, 2 * tag_embed_dim)
        latent_style = self.mlp(combined)  # (B, latent_style_dim)

        return latent_style


if __name__ == "__main__":
    adapter = KionStyleAdapter()
    dummy_weights = torch.zeros(2, 24)
    dummy_weights[0, 0] = 0.8  # angry = 0.8
    dummy_weights[1, 14] = 0.7  # affectionate = 0.7
    dummy_weights[1, 19] = 0.6  # playful = 0.6
    
    out = adapter(dummy_weights)
    print("Adapter output shape:", out.shape)
    assert out.shape == (2, 128), f"Expected shape (2, 128), got {out.shape}"
    print("KionStyleAdapter forward test PASSED.")
