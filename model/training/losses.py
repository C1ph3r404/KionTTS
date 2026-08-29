"""
KionTTS Loss Functions
Computes acoustic reconstruction, duration, pitch, energy, and style consistency losses.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


class KionLoss(nn.Module):
    def __init__(
        self,
        mel_weight: float = 1.0,
        mel_linear_weight: float = 0.5,
        duration_weight: float = 0.5,
        pitch_weight: float = 0.2,
        energy_weight: float = 0.2,
    ):
        super().__init__()
        self.mel_weight = mel_weight
        self.mel_linear_weight = mel_linear_weight
        self.duration_weight = duration_weight
        self.pitch_weight = pitch_weight
        self.energy_weight = energy_weight

    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            predictions: dict containing 'mel_refined', 'mel_linear', 'pred_log_durations', 'pred_pitch', 'pred_energy'
            targets: dict containing 'target_mel', 'target_log_durations', 'target_pitch', 'target_energy'
        """
        losses = {}

        # 1. Mel Reconstruction Losses
        pred_mel = predictions["mel_refined"]
        target_mel = targets["target_mel"]

        # Truncate to matching length if there is slight length difference due to padding
        min_len = min(pred_mel.size(-1), target_mel.size(-1))
        pred_mel = pred_mel[..., :min_len]
        target_mel = target_mel[..., :min_len]

        losses["mel_loss"] = F.l1_loss(pred_mel, target_mel) * self.mel_weight

        if "mel_linear" in predictions:
            pred_linear = predictions["mel_linear"][..., :min_len]
            losses["mel_linear_loss"] = F.l1_loss(pred_linear, target_mel) * self.mel_linear_weight
        else:
            losses["mel_linear_loss"] = torch.tensor(0.0, device=pred_mel.device)

        # 2. Duration Loss
        if "target_log_durations" in targets and targets["target_log_durations"] is not None:
            losses["duration_loss"] = (
                F.mse_loss(predictions["pred_log_durations"], targets["target_log_durations"])
                * self.duration_weight
            )
        else:
            losses["duration_loss"] = torch.tensor(0.0, device=pred_mel.device)

        # 3. Pitch Loss
        if "target_pitch" in targets and targets["target_pitch"] is not None:
            t_pitch = targets["target_pitch"][..., :min_len]
            p_pitch = predictions["pred_pitch"][..., :min_len]
            losses["pitch_loss"] = F.smooth_l1_loss(p_pitch, t_pitch) * self.pitch_weight
        else:
            losses["pitch_loss"] = torch.tensor(0.0, device=pred_mel.device)

        # 4. Energy Loss
        if "target_energy" in targets and targets["target_energy"] is not None:
            t_energy = targets["target_energy"][..., :min_len]
            p_energy = predictions["pred_energy"][..., :min_len]
            losses["energy_loss"] = F.smooth_l1_loss(p_energy, t_energy) * self.energy_weight
        else:
            losses["energy_loss"] = torch.tensor(0.0, device=pred_mel.device)

        # Total combined loss
        total_loss = (
            losses["mel_loss"]
            + losses["mel_linear_loss"]
            + losses["duration_loss"]
            + losses["pitch_loss"]
            + losses["energy_loss"]
        )
        losses["total_loss"] = total_loss

        return losses


if __name__ == "__main__":
    criterion = KionLoss()
    preds = {
        "mel_refined": torch.randn(2, 80, 50),
        "mel_linear": torch.randn(2, 80, 50),
        "pred_log_durations": torch.randn(2, 20),
        "pred_pitch": torch.randn(2, 50),
        "pred_energy": torch.randn(2, 50),
    }
    targs = {
        "target_mel": torch.randn(2, 80, 50),
        "target_log_durations": torch.randn(2, 20),
        "target_pitch": torch.randn(2, 50),
        "target_energy": torch.randn(2, 50),
    }
    loss_dict = criterion(preds, targs)
    print("Computed Loss Dict:", {k: v.item() for k, v in loss_dict.items()})
    assert loss_dict["total_loss"] > 0
    print("KionLoss test PASSED.")
