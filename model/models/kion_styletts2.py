"""
KionStyleTTS2 Unified Architecture
Wraps the official StyleTTS2 neural vocoder generator, prosody predictor,
and text encoder with the custom KionStyleAdapter for continuous tag-conditioned speech synthesis.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from model.modules.style_adapter import KionStyleAdapter

# Ensure StyleTTS2 root is in sys.path for internal relative-style imports
styletts2_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../StyleTTS2"))
if styletts2_root not in sys.path:
    sys.path.insert(0, styletts2_root)

from models import TextEncoder, ProsodyPredictor, StyleEncoder
from Modules.istftnet import Decoder as iSTFTDecoder
from Modules.discriminators import (
    MultiPeriodDiscriminator,
    MultiResSpecDiscriminator,
)


class KionStyleTTS2(nn.Module):
    """
    KionTTS - StyleTTS2 End-to-End Expressive Single-Speaker Model.
    Connects KionStyleAdapter to StyleTTS2's acoustic pipeline.
    """

    def __init__(
        self,
        num_emotions: int = 14,
        num_styles: int = 10,
        style_dim: int = 128,
        hidden_dim: int = 512,
        n_token: int = 178,
        n_layer: int = 3,
        max_dur: int = 50,
        n_mels: int = 80,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_emotions = num_emotions
        self.num_styles = num_styles
        self.style_dim = style_dim
        self.hidden_dim = hidden_dim

        # 1. Kion Tag-to-Style Adapter
        self.style_adapter = KionStyleAdapter(
            num_emotions=num_emotions,
            num_styles=num_styles,
            latent_style_dim=style_dim,
            hidden_dim=hidden_dim // 2,
            dropout=dropout,
        )

        # 2. Text Encoder (Phoneme -> Text representations)
        self.text_encoder = TextEncoder(
            channels=hidden_dim,
            kernel_size=5,
            depth=n_layer,
            n_symbols=n_token,
        )

        # 3. Prosody Predictor (Duration, Pitch F0, Energy N)
        self.predictor = ProsodyPredictor(
            style_dim=style_dim,
            d_hid=hidden_dim,
            nlayers=n_layer,
            max_dur=max_dur,
            dropout=dropout,
        )

        # 4. Neural Vocoder Decoder (iSTFTNet / HiFi-GAN backbone)
        self.decoder = iSTFTDecoder(
            dim_in=hidden_dim,
            style_dim=style_dim,
            dim_out=n_mels,
            resblock_kernel_sizes=[3, 7, 11],
            upsample_rates=[8, 8],
            upsample_initial_channel=512,
            resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            upsample_kernel_sizes=[16, 16],
            gen_istft_n_fft=16,
            gen_istft_hop_size=4,
        )

        # 5. Acoustic Reference Style Encoder (used for audio style extraction during training)
        self.style_encoder = StyleEncoder(
            dim_in=48,
            style_dim=style_dim,
            max_conv_dim=hidden_dim,
        )

    def forward(
        self,
        text_tokens: torch.Tensor,
        text_lengths: torch.Tensor,
        text_mask: torch.Tensor,
        style_weights: torch.Tensor,
        alignment: Optional[torch.Tensor] = None,
        ref_mel: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """
        Training Forward Pass.
        Args:
            text_tokens: (B, T_text) LongTensor phoneme IDs
            text_lengths: (B,) LongTensor lengths of text
            text_mask: (B, T_text) BoolTensor mask
            style_weights: (B, num_emotions + num_styles) Tag intensity vector
            alignment: (B, T_text, T_frames) CTC/Monotonic alignment matrix (if available)
            ref_mel: (B, 1, 80, T_frames) Reference Mel for acoustic style extraction
        """
        # A. Map tags to continuous style representation
        s_tag = self.style_adapter(style_weights)  # (B, style_dim)

        # B. If reference mel is provided, extract acoustic style vector for alignment loss
        s_audio = None
        if ref_mel is not None:
            s_audio = self.style_encoder(ref_mel)

        # C. Encode text phonemes
        text_feats = self.text_encoder(text_tokens, text_lengths, text_mask)  # (B, hidden_dim, T_text)

        # D. Prosody prediction
        # Use predicted or acoustic style
        s_to_use = s_audio if (s_audio is not None and self.training) else s_tag

        pred_dur, en = self.predictor(text_feats, s_to_use, text_lengths, alignment, text_mask)

        return {
            "s_tag": s_tag,
            "s_audio": s_audio,
            "text_feats": text_feats,
            "pred_dur": pred_dur,
            "en": en,
        }

    @torch.no_grad()
    def synthesize(
        self,
        text_tokens: torch.Tensor,
        style_weights: torch.Tensor,
        pace: float = 1.0,
    ) -> torch.Tensor:
        """
        Direct Tag-Conditioned Inference (Tag -> Waveform). No reference audio required!
        Args:
            text_tokens: (B, T_text) LongTensor
            style_weights: (B, num_emotions + num_styles) FloatTensor in [0, 1]
            pace: Speed multiplier (> 1 faster, < 1 slower)
        Returns:
            waveform: (B, T_samples) 24kHz audio
        """
        self.eval()
        device = text_tokens.device
        B = text_tokens.size(0)

        # 1. Compute latent style from tags
        s = self.style_adapter(style_weights)  # (B, style_dim)

        # 2. Text encode
        text_lengths = torch.tensor([text_tokens.size(1)] * B, device=device, dtype=torch.long)
        text_mask = torch.zeros(B, text_tokens.size(1), device=device, dtype=torch.bool)
        text_feats = self.text_encoder(text_tokens, text_lengths, text_mask)

        # 3. Predict durations
        d = self.predictor.text_encoder(text_feats, s, text_lengths, text_mask)
        input_lengths = text_lengths.cpu().numpy()
        x = nn.utils.rnn.pack_padded_sequence(d, input_lengths, batch_first=True, enforce_sorted=False)
        self.predictor.lstm.flatten_parameters()
        x, _ = self.predictor.lstm(x)
        x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        pred_dur = self.predictor.duration_proj(x)
        pred_dur = torch.sigmoid(pred_dur).sum(axis=-1) / pace

        # 4. Length regulation to frame representations
        pred_dur = torch.round(pred_dur).clamp(min=1).long()
        total_frames = int(pred_dur.sum().item())
        if total_frames % 2 != 0:
            pred_dur[0, -1] += 1
            total_frames += 1

        pred_aln_trg = torch.zeros(text_tokens.size(1), total_frames, device=device)
        c_frame = 0
        for i in range(text_tokens.size(1)):
            dur_i = int(pred_dur[0, i].item())
            if c_frame + dur_i <= total_frames:
                pred_aln_trg[i, c_frame : c_frame + dur_i] = 1
            c_frame += dur_i

        # Acoustic and prosodic frame expansion
        aln_matrix = pred_aln_trg.unsqueeze(0)  # (1, T_text, T_frames)
        asr = text_feats @ aln_matrix            # (1, hidden_dim, T_frames)
        en = d.transpose(-1, -2) @ aln_matrix   # (1, hidden_dim, T_frames)

        # 5. Predict F0 and N curves
        F0_pred, N_pred = self.predictor.F0Ntrain(en, s)

        # 6. Synthesize audio waveform via Neural Vocoder Decoder
        waveform = self.decoder(asr, F0_pred, N_pred, s)
        return waveform.squeeze(1)
