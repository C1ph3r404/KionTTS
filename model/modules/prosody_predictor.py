"""
KionTTS Prosody Predictor
Predicts phoneme durations, continuous F0 pitch contours, and energy levels
conditioned on text representations and Kion style vectors.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class LengthRegulator(nn.Module):
    """
    Expands phoneme hidden states to acoustic frame hidden states according to durations.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, durations: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T_text, D)
            durations: (B, T_text) Long/Int tensor of durations per phoneme
        Returns:
            expanded: (B, T_frames, D)
        """
        output = []
        for i in range(x.size(0)):
            rep = torch.repeat_interleave(x[i], durations[i], dim=0)
            output.append(rep)

        max_len = max([r.size(0) for r in output]) if output else 0
        if max_len == 0:
            return torch.zeros(x.size(0), 1, x.size(-1), device=x.device, dtype=x.dtype)

        padded = torch.zeros(x.size(0), max_len, x.size(-1), device=x.device, dtype=x.dtype)
        for i, r in enumerate(output):
            padded[i, : r.size(0)] = r
        return padded


class ConvPredictor(nn.Module):
    """Generic 1D Convolutional predictor for duration, pitch, or energy."""

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int = 256,
        kernel_size: int = 3,
        dropout: float = 0.2,
        out_channels: int = 1,
    ):
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels,
            hidden_dim,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(
            hidden_dim,
            hidden_dim,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout2 = nn.Dropout(dropout)

        self.linear = nn.Linear(hidden_dim, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        out = self.conv1(x.transpose(1, 2)).transpose(1, 2)
        out = self.norm1(out)
        out = F.relu(out)
        out = self.dropout1(out)

        out = self.conv2(out.transpose(1, 2)).transpose(1, 2)
        out = self.norm2(out)
        out = F.relu(out)
        out = self.dropout2(out)

        out = self.linear(out)
        return out


class KionProsodyPredictor(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 256,
        latent_style_dim: int = 128,
        pitch_bins: int = 256,
        energy_bins: int = 256,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.style_proj = nn.Linear(latent_style_dim, hidden_dim)

        # Duration predictor operates on phoneme level
        self.duration_predictor = ConvPredictor(hidden_dim, hidden_dim=hidden_dim, out_channels=1)
        self.length_regulator = LengthRegulator()

        # Pitch and energy predictors operate on frame level
        self.pitch_predictor = ConvPredictor(hidden_dim, hidden_dim=hidden_dim, out_channels=1)
        self.energy_predictor = ConvPredictor(hidden_dim, hidden_dim=hidden_dim, out_channels=1)

        # Embeddings for conditioning generator on pitch and energy
        self.pitch_embedding = nn.Linear(1, hidden_dim)
        self.energy_embedding = nn.Linear(1, hidden_dim)

    def forward(
        self,
        text_encoded: torch.Tensor,
        latent_style: torch.Tensor,
        target_durations: Optional[torch.Tensor] = None,
        target_pitch: Optional[torch.Tensor] = None,
        target_energy: Optional[torch.Tensor] = None,
        pace: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            text_encoded: (B, T_text, hidden_dim)
            latent_style: (B, latent_style_dim)
            target_durations: (B, T_text) or None
            target_pitch: (B, T_frames) or None
            target_energy: (B, T_frames) or None
            pace: Speed factor (> 1.0 faster, < 1.0 slower)
        Returns:
            frame_features: (B, T_frames, hidden_dim)
            pred_log_durations: (B, T_text)
            pred_pitch: (B, T_frames)
            pred_energy: (B, T_frames)
            used_durations: (B, T_text)
        """
        # Inject style conditioning into phoneme representations
        style_feat = self.style_proj(latent_style).unsqueeze(1)  # (B, 1, hidden_dim)
        cond_text = text_encoded + style_feat

        # Predict durations
        pred_log_dur = self.duration_predictor(cond_text).squeeze(-1)  # (B, T_text)
        pred_dur = torch.clamp(torch.round(torch.exp(pred_log_dur) - 1.0) / pace, min=1.0).long()

        if target_durations is not None:
            dur_to_use = target_durations.long()
        else:
            dur_to_use = pred_dur

        # Expand phonemes to frame-level hidden representations
        expanded_features = self.length_regulator(cond_text, dur_to_use)

        # Predict Pitch and Energy at frame level
        pred_pitch = self.pitch_predictor(expanded_features).squeeze(-1)  # (B, T_frames)
        pred_energy = self.energy_predictor(expanded_features).squeeze(-1)  # (B, T_frames)

        pitch_to_embed = target_pitch.unsqueeze(-1) if target_pitch is not None else pred_pitch.unsqueeze(-1)
        energy_to_embed = target_energy.unsqueeze(-1) if target_energy is not None else pred_energy.unsqueeze(-1)

        p_emb = self.pitch_embedding(pitch_to_embed)
        e_emb = self.energy_embedding(energy_to_embed)

        # Enriched frame representations
        frame_features = expanded_features + p_emb + e_emb

        return frame_features, pred_log_dur, pred_pitch, pred_energy, dur_to_use


if __name__ == "__main__":
    pp = KionProsodyPredictor()
    dummy_text = torch.randn(2, 20, 256)
    dummy_style = torch.randn(2, 128)
    
    # Test inference mode (no targets)
    frame_feats, p_dur, p_pitch, p_energy, used_dur = pp(dummy_text, dummy_style)
    print("Prosody Predictor Frame Features shape:", frame_feats.shape)
    print("Predicted Pitch shape:", p_pitch.shape)
    print("Predicted Energy shape:", p_energy.shape)
    print("KionProsodyPredictor test PASSED.")
