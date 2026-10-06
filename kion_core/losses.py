"""
Production Loss Modules for KionTTS.
Includes:
- MultiResolutionSTFTLoss: Multi-resolution spectral convergence and log-STFT magnitude loss.
- GeneratorLoss & DiscriminatorLoss: Multi-Period (MPD) and Multi-Scale (MSD) GAN losses.
- KionStyleAlignmentLoss: Grounded cosine and L1 style alignment between tag embeddings and audio styles.
- WavLMLoss: Perceptual speech language model loss for human-like prosody and timbre.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from typing import Tuple, List, Optional


class SpectralConvergenceLoss(nn.Module):
    """Spectral convergence loss module."""
    def __init__(self):
        super().__init__()

    def forward(self, x_mag: torch.Tensor, y_mag: torch.Tensor) -> torch.Tensor:
        """
        x_mag: Predicted signal magnitude spectrogram (B, frames, freq_bins)
        y_mag: Groundtruth signal magnitude spectrogram (B, frames, freq_bins)
        """
        return torch.norm(y_mag - x_mag, p="fro") / (torch.norm(y_mag, p="fro") + 1e-7)


class STFTLoss(nn.Module):
    """Single resolution STFT loss with spectral convergence and log magnitude distance."""
    def __init__(self, fft_size=1024, shift_size=120, win_length=600, window=torch.hann_window):
        super().__init__()
        self.fft_size = fft_size
        self.shift_size = shift_size
        self.win_length = win_length
        self.to_mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=24000,
            n_fft=fft_size,
            win_length=win_length,
            hop_length=shift_size,
            window_fn=window,
            power=1.0,
        )
        self.spectral_convergence_loss = SpectralConvergenceLoss()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        x: Predicted waveform (B, T) or (B, 1, T)
        y: Ground truth waveform (B, T) or (B, 1, T)
        """
        if x.dim() == 3:
            x = x.squeeze(1)
        if y.dim() == 3:
            y = y.squeeze(1)

        x_mag = self.to_mel(x)
        y_mag = self.to_mel(y)

        mean, std = -4.0, 4.0
        x_log = (torch.log(x_mag.clamp(min=1e-5)) - mean) / std
        y_log = (torch.log(y_mag.clamp(min=1e-5)) - mean) / std

        sc_loss = self.spectral_convergence_loss(x_log, y_log)
        mag_loss = F.l1_loss(x_log, y_log)
        return sc_loss + mag_loss


class MultiResolutionSTFTLoss(nn.Module):
    """
    Multi-resolution STFT loss module across 3 different window and FFT resolutions:
    - High temporal resolution (small FFT)
    - Balanced resolution
    - High frequency resolution (large FFT)
    """
    def __init__(
        self,
        fft_sizes=[1024, 2048, 512],
        hop_sizes=[120, 240, 50],
        win_lengths=[600, 1200, 240],
        window=torch.hann_window,
    ):
        super().__init__()
        assert len(fft_sizes) == len(hop_sizes) == len(win_lengths)
        self.stft_losses = nn.ModuleList([
            STFTLoss(fs, ss, wl, window)
            for fs, ss, wl in zip(fft_sizes, hop_sizes, win_lengths)
        ])

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        total_loss = 0.0
        for loss_fn in self.stft_losses:
            total_loss = total_loss + loss_fn(x, y)
        return total_loss / len(self.stft_losses)


def feature_matching_loss(fmap_r: List[List[torch.Tensor]], fmap_g: List[List[torch.Tensor]]) -> torch.Tensor:
    """L1 feature matching loss across discriminator intermediate activations."""
    loss = 0.0
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            loss = loss + torch.mean(torch.abs(rl.detach() - gl))
    return loss * 2.0


def discriminator_loss(disc_real_outputs, disc_generated_outputs) -> Tuple[torch.Tensor, List[float], List[float]]:
    loss = 0.0
    r_losses = []
    g_losses = []
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        r_loss = torch.mean((1.0 - dr) ** 2)
        g_loss = torch.mean(dg ** 2)
        loss = loss + (r_loss + g_loss)
        r_losses.append(r_loss.item())
        g_losses.append(g_loss.item())
    return loss, r_losses, g_losses


def generator_loss(disc_outputs) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    loss = 0.0
    gen_losses = []
    for dg in disc_outputs:
        l = torch.mean((1.0 - dg) ** 2)
        gen_losses.append(l)
        loss = loss + l
    return loss, gen_losses


def generator_tprls_loss(disc_real_outputs, disc_generated_outputs, tau=0.04) -> torch.Tensor:
    """Relative discriminator loss (TPRLS) for stabilizing adversarial training."""
    loss = 0.0
    for dg, dr in zip(disc_real_outputs, disc_generated_outputs):
        diff = dr - dg
        m_dg = torch.median(diff)
        mask = dr < (dg + m_dg)
        if mask.any():
            l_rel = torch.mean(((diff - m_dg) ** 2)[mask])
            loss = loss + tau - F.relu(tau - l_rel)
    return loss


class GeneratorLoss(nn.Module):
    """
    Complete HiFi-GAN Generator Loss combining:
    - Multi-Period Discriminator (MPD) adversarial loss
    - Multi-Scale Spectrogram Discriminator (MSD) adversarial loss
    - Feature matching loss from MPD & MSD
    - Relative TPRLS adversarial loss
    """
    def __init__(self, mpd: nn.Module, msd: nn.Module):
        super().__init__()
        self.mpd = mpd
        self.msd = msd

    def forward(self, y_real: torch.Tensor, y_gen: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if y_real.dim() == 2:
            y_real = y_real.unsqueeze(1)
        if y_gen.dim() == 2:
            y_gen = y_gen.unsqueeze(1)

        y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = self.mpd(y_real, y_gen)
        y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = self.msd(y_real, y_gen)

        loss_fm_f = feature_matching_loss(fmap_f_r, fmap_f_g)
        loss_fm_s = feature_matching_loss(fmap_s_r, fmap_s_g)
        loss_fm = loss_fm_f + loss_fm_s

        loss_gen_f, _ = generator_loss(y_df_hat_g)
        loss_gen_s, _ = generator_loss(y_ds_hat_g)
        loss_gen_adv = loss_gen_f + loss_gen_s

        loss_rel = generator_tprls_loss(y_df_hat_r, y_df_hat_g) + generator_tprls_loss(y_ds_hat_r, y_ds_hat_g)
        total_gen_loss = loss_gen_adv + loss_fm + loss_rel

        return total_gen_loss, loss_gen_adv, loss_fm


class DiscriminatorLoss(nn.Module):
    """
    Complete HiFi-GAN Discriminator Loss combining MPD and MSD.
    """
    def __init__(self, mpd: nn.Module, msd: nn.Module):
        super().__init__()
        self.mpd = mpd
        self.msd = msd

    def forward(self, y_real: torch.Tensor, y_gen: torch.Tensor) -> torch.Tensor:
        if y_real.dim() == 2:
            y_real = y_real.unsqueeze(1)
        if y_gen.dim() == 2:
            y_gen = y_gen.unsqueeze(1)

        y_df_hat_r, y_df_hat_g, _, _ = self.mpd(y_real, y_gen.detach())
        loss_disc_f, _, _ = discriminator_loss(y_df_hat_r, y_df_hat_g)

        y_ds_hat_r, y_ds_hat_g, _, _ = self.msd(y_real, y_gen.detach())
        loss_disc_s, _, _ = discriminator_loss(y_ds_hat_r, y_ds_hat_g)

        return loss_disc_f + loss_disc_s


class KionStyleAlignmentLoss(nn.Module):
    """
    Grounded style alignment loss between predicted tag style (s_tag)
    and ground-truth reference audio style (s_audio).
    
    Protections:
    - Direct MSE coordinate loss: strictly enforces magnitude & manifold matching teacher (~0.54 norm).
    - Angular cosine distance: aligns the directional orientation in 256-d space.
    - Direct L1 loss: enforces fine-grained coordinate precision.
    - Soft upper-bound norm penalty: prevents any vector magnitude explosion above teacher scale.
    """
    def __init__(self, lambda_cos: float = 1.0, lambda_mse: float = 10.0, lambda_l1: float = 1.0, lambda_reg: float = 1.0):
        super().__init__()
        self.lambda_cos = lambda_cos
        self.lambda_mse = lambda_mse
        self.lambda_l1 = lambda_l1
        self.lambda_reg = lambda_reg

    def forward(self, s_tag: torch.Tensor, s_audio: torch.Tensor) -> torch.Tensor:
        s_tag_safe = torch.nan_to_num(s_tag, nan=0.0).clamp(-10.0, 10.0)
        s_audio_safe = torch.nan_to_num(s_audio.detach(), nan=0.0).clamp(-10.0, 10.0)

        # 1. Angular cosine distance between style vectors
        s_tag_norm = F.normalize(s_tag_safe, p=2, dim=-1, eps=1e-4)
        s_audio_norm = F.normalize(s_audio_safe, p=2, dim=-1, eps=1e-4)
        cos_sim = (s_tag_norm * s_audio_norm).sum(dim=-1).clamp(-1.0, 1.0)
        loss_cos = (1.0 - cos_sim).mean()

        # 2. Direct unnormalized MSE coordinate loss (crucial for keeping norm matching teacher ~0.54)
        loss_mse = F.mse_loss(s_tag_safe, s_audio_safe)

        # 3. Direct unnormalized L1 coordinate loss
        loss_l1 = F.l1_loss(s_tag_safe, s_audio_safe)

        # 4. Norm boundary regularization: penalize vectors if norm exceeds 1.5 (teacher norm is ~0.54)
        tag_norms = torch.norm(s_tag_safe, p=2, dim=-1)
        norm_penalty = torch.mean(torch.relu(tag_norms - 1.5) ** 2)

        total = self.lambda_cos * loss_cos + self.lambda_mse * loss_mse + self.lambda_l1 * loss_l1 + self.lambda_reg * norm_penalty
        if torch.isnan(total) or torch.isinf(total):
            return torch.tensor(0.0, device=s_tag.device, requires_grad=True)
        return total


class WavLMLoss(nn.Module):
    """
    Perceptual Speech Language Model feature-matching loss using WavLM.
    Extracts deep hidden representations from real and synthesized speech to ensure natural timbre.
    """
    def __init__(self, model_name="microsoft/wavlm-base-plus", wd=None, sr=24000, slm_sr=16000):
        super().__init__()
        from transformers import AutoModel
        self.wavlm = AutoModel.from_pretrained(model_name)
        self.wavlm.eval()
        for p in self.wavlm.parameters():
            p.requires_grad = False
        self.wd = wd
        self.resampler = torchaudio.transforms.Resample(sr, slm_sr)

    def forward(self, y_real: torch.Tensor, y_gen: torch.Tensor) -> torch.Tensor:
        if y_real.dim() == 2:
            y_real = y_real.squeeze(1)
        if y_gen.dim() == 2:
            y_gen = y_gen.squeeze(1)

        # Resample to 16kHz for WavLM
        y_real_16k = self.resampler(y_real)
        y_gen_16k = self.resampler(y_gen)

        with torch.no_grad():
            real_feats = self.wavlm(y_real_16k).last_hidden_state
        gen_feats = self.wavlm(y_gen_16k).last_hidden_state

        loss = F.l1_loss(gen_feats, real_feats)
        return loss
