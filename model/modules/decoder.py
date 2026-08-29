"""
KionDecoder Module
Acoustic decoder synthesizing Mel-spectrograms and high-fidelity speech waveforms
from prosody-enriched frame representations and Kion style conditioning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional


class ResBlock1(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3, dilation: Tuple[int, int, int] = (1, 3, 5)):
        super().__init__()
        self.convs1 = nn.ModuleList(
            [
                nn.Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    1,
                    dilation=d,
                    padding=(kernel_size * d - d) // 2,
                )
                for d in dilation
            ]
        )
        self.convs2 = nn.ModuleList(
            [
                nn.Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    1,
                    dilation=1,
                    padding=(kernel_size - 1) // 2,
                )
                for _ in dilation
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, 0.1)
            xt = c1(xt)
            xt = F.leaky_relu(xt, 0.1)
            xt = c2(xt)
            x = xt + x
        return x


class KionAcousticDecoder(nn.Module):
    """
    Acoustic decoder producing 80-band Mel-spectrograms from frame representations.
    """

    def __init__(
        self,
        in_dim: int = 256,
        mel_channels: int = 80,
        hidden_dim: int = 512,
        num_layers: int = 4,
        kernel_size: int = 5,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.in_proj = nn.Linear(in_dim, hidden_dim)

        self.convolutions = nn.ModuleList()
        for _ in range(num_layers):
            self.convolutions.append(
                nn.Sequential(
                    nn.Conv1d(
                        hidden_dim,
                        hidden_dim,
                        kernel_size=kernel_size,
                        padding=(kernel_size - 1) // 2,
                    ),
                    nn.BatchNorm1d(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
            )

        self.out_proj = nn.Linear(hidden_dim, mel_channels)

        # Post-net for Mel refinement
        self.postnet = nn.Sequential(
            nn.Conv1d(mel_channels, 256, 5, padding=2),
            nn.BatchNorm1d(256),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Conv1d(256, 256, 5, padding=2),
            nn.BatchNorm1d(256),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Conv1d(256, mel_channels, 5, padding=2),
        )

    def forward(self, frame_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            frame_features: (B, T_frames, in_dim)
        Returns:
            mel_output: (B, 80, T_frames)
            mel_postnet: (B, 80, T_frames)
        """
        x = self.in_proj(frame_features)  # (B, T, hidden_dim)
        x = x.transpose(1, 2)  # (B, hidden_dim, T)

        for conv in self.convolutions:
            x = conv(x) + x

        x = x.transpose(1, 2)  # (B, T, hidden_dim)
        mel_linear = self.out_proj(x).transpose(1, 2)  # (B, 80, T)

        residual = self.postnet(mel_linear)
        mel_refined = mel_linear + residual

        return mel_refined, mel_linear


if __name__ == "__main__":
    decoder = KionAcousticDecoder()
    dummy_feats = torch.randn(2, 60, 256)
    mel_out, mel_lin = decoder(dummy_feats)
    print("Mel Output Shape:", mel_out.shape)
    assert mel_out.shape == (2, 80, 60)
    print("KionAcousticDecoder test PASSED.")
