"""
KionTTS Text / Phoneme Encoder
Processes phoneme token sequences into rich contextual acoustic representations.
"""

import math
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, dim: int, max_len: int = 2000):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        return x + self.pe[:, : x.size(1), :]


class ConvReluNorm(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 5, dropout: float = 0.1):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
        )
        self.norm = nn.LayerNorm(out_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        residual = x
        out = self.conv(x.transpose(1, 2)).transpose(1, 2)
        out = self.norm(out)
        out = F.gelu(out)
        out = self.dropout(out)
        return out + residual if out.shape == residual.shape else out


class KionTextEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int = 256,
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 4,
        kernel_size: int = 5,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)
        self.pos_encoding = PositionalEncoding(hidden_dim)

        self.conv_layers = nn.ModuleList(
            [ConvReluNorm(hidden_dim, hidden_dim, kernel_size=kernel_size, dropout=dropout) for _ in range(3)]
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.proj_out = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, text_tokens: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            text_tokens: (B, T) LongTensor of phoneme token IDs
            mask: (B, T) BoolTensor where True indicates padding positions
        Returns:
            encoded_text: (B, T, hidden_dim)
        """
        x = self.embedding(text_tokens) * math.sqrt(x.size(-1) if hasattr(self, "_dummy") else 256)
        x = self.pos_encoding(x)

        for layer in self.conv_layers:
            x = layer(x)

        if mask is not None:
            # Transformer expects src_key_padding_mask: True for masked positions
            x = self.transformer(x, src_key_padding_mask=mask)
        else:
            x = self.transformer(x)

        out = self.proj_out(x)
        return out


if __name__ == "__main__":
    encoder = KionTextEncoder(vocab_size=150, hidden_dim=256)
    dummy_input = torch.randint(1, 140, (2, 35))
    out = encoder(dummy_input)
    print("Text Encoder Output shape:", out.shape)
    assert out.shape == (2, 35, 256)
    print("KionTextEncoder test PASSED.")
