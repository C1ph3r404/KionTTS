"""
KionTTS Unified Model Architecture
Combines KionTextEncoder, KionStyleAdapter, KionProsodyPredictor, and KionAcousticDecoder
into an expressive single-speaker TTS system with continuous emotion/style conditioning.
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional, Any

from model.modules.text_encoder import KionTextEncoder
from model.modules.style_adapter import KionStyleAdapter
from model.modules.prosody_predictor import KionProsodyPredictor
from model.modules.decoder import KionAcousticDecoder


class KionTTSModel(nn.Module):
    def __init__(
        self,
        vocab_size: int = 256,
        hidden_dim: int = 256,
        num_emotions: int = 14,
        num_styles: int = 10,
        latent_style_dim: int = 128,
        mel_channels: int = 80,
    ):
        super().__init__()
        self.style_adapter = KionStyleAdapter(
            num_emotions=num_emotions,
            num_styles=num_styles,
            latent_style_dim=latent_style_dim,
        )
        self.text_encoder = KionTextEncoder(
            vocab_size=vocab_size,
            hidden_dim=hidden_dim,
        )
        self.prosody_predictor = KionProsodyPredictor(
            hidden_dim=hidden_dim,
            latent_style_dim=latent_style_dim,
        )
        self.decoder = KionAcousticDecoder(
            in_dim=hidden_dim,
            mel_channels=mel_channels,
        )

    def forward(
        self,
        text_tokens: torch.Tensor,
        style_weights: torch.Tensor,
        target_durations: Optional[torch.Tensor] = None,
        target_pitch: Optional[torch.Tensor] = None,
        target_energy: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Training Forward Pass.
        """
        # 1. Encode text phonemes
        text_encoded = self.text_encoder(text_tokens, mask=text_mask)

        # 2. Map emotion/style weights to latent style vector
        latent_style = self.style_adapter(style_weights)

        # 3. Predict prosody features and expand phonemes to frames
        frame_feats, pred_log_dur, pred_pitch, pred_energy, used_dur = self.prosody_predictor(
            text_encoded=text_encoded,
            latent_style=latent_style,
            target_durations=target_durations,
            target_pitch=target_pitch,
            target_energy=target_energy,
        )

        # 4. Synthesize Mel-spectrogram
        mel_refined, mel_linear = self.decoder(frame_feats)

        return {
            "mel_refined": mel_refined,
            "mel_linear": mel_linear,
            "pred_log_durations": pred_log_dur,
            "pred_pitch": pred_pitch,
            "pred_energy": pred_energy,
            "latent_style": latent_style,
            "used_durations": used_dur,
        }

    @torch.no_grad()
    def inference(
        self,
        text_tokens: torch.Tensor,
        style_weights: torch.Tensor,
        pace: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        """
        Inference Forward Pass.
        """
        self.eval()
        text_encoded = self.text_encoder(text_tokens)
        latent_style = self.style_adapter(style_weights)

        frame_feats, pred_log_dur, pred_pitch, pred_energy, used_dur = self.prosody_predictor(
            text_encoded=text_encoded,
            latent_style=latent_style,
            pace=pace,
        )

        mel_refined, mel_linear = self.decoder(frame_feats)

        return {
            "mel": mel_refined,
            "pred_pitch": pred_pitch,
            "pred_energy": pred_energy,
            "pred_durations": used_dur,
            "latent_style": latent_style,
        }


if __name__ == "__main__":
    model = KionTTSModel()
    tokens = torch.randint(1, 100, (2, 25))
    weights = torch.zeros(2, 24)
    weights[0, 0] = 0.8  # angry
    weights[1, 14] = 0.7  # affectionate

    # Test training forward
    out = model(tokens, weights)
    print("Training Mel output shape:", out["mel_refined"].shape)
    assert out["mel_refined"].shape[0] == 2

    # Test inference
    inf_out = model.inference(tokens, weights)
    print("Inference Mel output shape:", inf_out["mel"].shape)
    print("KionTTSModel full integration test PASSED.")
