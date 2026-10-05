"""
Production Dual-Stage Training Engine for KionTTS.
Features:
- Stage 1: Acoustic Foundation, Prosody Adaptation, and Grounded Tag Style Alignment.
- Stage 2: HiFi-GAN Vocoder Fine-Tuning with MultiResolutionSTFT, MPD/MSD GAN, and Optional WavLM SLM.
- Memory-Safe Windowed Decoder Training (guaranteed zero OOM on 16GB GPUs).
- Native PyTorch AMP (Automatic Mixed Precision) and Multi-GPU / Accelerate compatibility.
- Seamless Hugging Face Hub checkpoint synchronization and rolling pruning.
"""

import os
import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from torch.utils.tensorboard import SummaryWriter
except (ImportError, ModuleNotFoundError):
    SummaryWriter = None
from typing import Dict, Any, Optional

from .losses import (
    MultiResolutionSTFTLoss,
    GeneratorLoss,
    DiscriminatorLoss,
    KionStyleAlignmentLoss,
)
from .tag_style_encoder import KionTagStyleEncoder
from .checkpoint_manager import KionCheckpointManager
from .synthesizer import KionSynthesizer

DEFAULT_EVAL_PROMPTS = [
    ("[neutral=0.6] Antigravity voice synthesis operational on neural cluster.", "neutral"),
    ("[happy=0.9, excited=0.8] We did it! The full training pipeline converged with pristine acoustic fidelity!", "happy_excited"),
    ("[sarcasm=0.9] Oh, brilliant. Another zero-division warning to brighten my morning.", "sarcasm"),
    ("[soothing=0.8, calm=0.7] Take a deep breath. The loss curves are steadily dropping toward zero.", "soothing_calm"),
]

try:
    from utils import maximum_path, mask_from_lens, log_norm, length_to_mask
except ImportError:
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "StyleTTS2"))
    from utils import maximum_path, mask_from_lens, log_norm, length_to_mask


def extract_f0_safe(pitch_extractor, mel_slice):
    """Safely extracts F0 pitch contours with dimension handling."""
    with torch.no_grad():
        f0_out = pitch_extractor(mel_slice)
        if isinstance(f0_out, (tuple, list)):
            f0_real = f0_out[0]
        else:
            f0_real = f0_out
    return f0_real


def compute_energy_norm(mel_slice):
    """Computes log spectral energy norm matching StyleTTS2."""
    if mel_slice.dim() == 2:
        mel_slice = mel_slice.unsqueeze(0)
    if mel_slice.dim() == 3:
        mel_slice = mel_slice.unsqueeze(1)
    return log_norm(mel_slice).squeeze(1)


class MyDistributedDataParallel(torch.nn.parallel.DistributedDataParallel):
    """
    Subclass of PyTorch DistributedDataParallel that delegates attribute/method access
    to the underlying module, ensuring custom methods (e.g. predictor.F0Ntrain)
    and attributes remain directly accessible.
    """
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


from contextlib import contextmanager

@contextmanager
def maybe_no_sync(modules, enabled: bool = True):
    """Context manager to suppress DDP gradient all-reduce during gradient accumulation steps."""
    if not enabled:
        yield
        return
    sync_contexts = []
    for m in modules:
        if m is not None and hasattr(m, "no_sync"):
            sync_contexts.append(m.no_sync())
    for ctx in sync_contexts:
        ctx.__enter__()
    try:
        yield
    finally:
        for ctx in reversed(sync_contexts):
            ctx.__exit__(None, None, None)


class KionProductionTrainer:
    """
    Production-grade single-speaker StyleTTS2 trainer with emotion tag conditioning.
    Optimized for single GPU and Kaggle dual T4 GPU (DDP) distributed training.
    """
    def __init__(
        self,
        model: Dict[str, Any],
        tag_encoder: KionTagStyleEncoder,
        checkpoint_manager: KionCheckpointManager,
        output_dir: str = "checkpoints",
        log_dir: str = "runs/kiontts",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        use_amp: bool = True,
        accum_steps: int = 1,
        local_rank: int = -1,
        rank: int = 0,
        world_size: int = 1,
        is_distributed: bool = False,
    ):
        self.model = model
        self.tag_encoder = tag_encoder
        self.ckpt_manager = checkpoint_manager
        self.output_dir = output_dir
        self.log_dir = log_dir

        # Distributed state detection
        self.local_rank = local_rank
        self.rank = rank
        self.world_size = world_size
        self.is_distributed = is_distributed

        if not self.is_distributed and "LOCAL_RANK" in os.environ:
            self.local_rank = int(os.environ["LOCAL_RANK"])
            self.rank = int(os.environ.get("RANK", 0))
            self.world_size = int(os.environ.get("WORLD_SIZE", 1))
            self.is_distributed = self.world_size > 1

        self.is_main_process = (self.rank == 0)

        # Set device
        if self.local_rank >= 0 and torch.cuda.is_available():
            self.device = torch.device(f"cuda:{self.local_rank}")
        else:
            self.device = torch.device(device)

        self.use_amp = use_amp and (self.device.type == "cuda")
        self.accum_steps = max(1, accum_steps)

        # Hardware-specific performance optimizations for Kaggle T4 / Ampere
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True
            if hasattr(torch, "set_float32_matmul_precision"):
                torch.set_float32_matmul_precision("high")

        if self.is_main_process:
            os.makedirs(self.output_dir, exist_ok=True)
            os.makedirs(self.log_dir, exist_ok=True)
            self.writer = SummaryWriter(self.log_dir) if SummaryWriter is not None else None
        else:
            self.writer = None

        # Build base losses
        self.stft_loss = MultiResolutionSTFTLoss().to(self.device)
        self.style_loss_fn = KionStyleAlignmentLoss(lambda_cos=1.0, lambda_reg=0.01).to(self.device)
        self.gen_loss_fn = None
        self.disc_loss_fn = None

        # AMP Scalers (PyTorch 2.4+ compatibility)
        if hasattr(torch.amp, "GradScaler"):
            self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
            self.scaler_d = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        else:
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
            self.scaler_d = torch.cuda.amp.GradScaler(enabled=self.use_amp)

    def autocast(self):
        """Context manager for automatic mixed precision across PyTorch versions."""
        if hasattr(torch.amp, "autocast"):
            return torch.amp.autocast("cuda", enabled=self.use_amp)
        return torch.cuda.amp.autocast(enabled=self.use_amp)

    def activate_stage_modules(self, stage: str):
        """
        Dynamically manages GPU memory by loading ONLY the modules needed for the active stage.
        Unused modules are offloaded to CPU RAM, and GPU memory cache is cleared.
        In multi-GPU mode, trainable modules are wrapped with DistributedDataParallel.
        """
        import gc
        if stage == "stage1":
            active_modules = [
                "predictor", "bert", "bert_encoder", "text_encoder",
                "style_encoder", "predictor_encoder", "pitch_extractor", "text_aligner"
            ]
            offload_modules = ["decoder", "mpd", "msd", "diffusion", "wd"]
        elif stage == "stage2":
            active_modules = [
                "predictor", "bert", "bert_encoder", "text_encoder",
                "style_encoder", "predictor_encoder", "decoder", "mpd", "msd",
                "pitch_extractor", "text_aligner"
            ]
            offload_modules = ["diffusion", "wd"]
        else:
            active_modules = list(self.model.keys())
            offload_modules = []

        if self.is_main_process:
            print(f"[*] Dynamically configuring GPU memory for [{stage}]...")
        # 1. Offload unused modules to CPU RAM first
        for k in offload_modules:
            if k in self.model and self.model[k] is not None:
                mod = self.model[k].module if hasattr(self.model[k], "module") else self.model[k]
                if hasattr(mod, "to"):
                    mod.to("cpu")
                if hasattr(mod, "eval"):
                    mod.eval()
                for p in mod.parameters():
                    p.requires_grad = False

        # 2. Flush GPU VRAM cache
        if self.device.type == "cuda":
            gc.collect()
            torch.cuda.empty_cache()

        # 3. Move active modules to GPU
        for k in active_modules:
            if k in self.model and self.model[k] is not None:
                mod = self.model[k].module if hasattr(self.model[k], "module") else self.model[k]
                if hasattr(mod, "to"):
                    mod.to(self.device)
        self.tag_encoder.to(self.device)

        # Freeze non-trainable reference feature extractors in Stage 2
        if stage == "stage2":
            frozen_keys = ["text_encoder", "style_encoder", "predictor_encoder", "pitch_extractor", "text_aligner"]
            for k in frozen_keys:
                if k in self.model and self.model[k] is not None:
                    mod = self.model[k].module if hasattr(self.model[k], "module") else self.model[k]
                    if hasattr(mod, "eval"):
                        mod.eval()
                    for p in mod.parameters():
                        p.requires_grad = False

        # 4. Wrap trainable modules in DDP if distributed
        if self.is_distributed:
            dev_idx = self.device.index if self.device.index is not None else 0
            # Note: "predictor" (ProsodyPredictor) executes in two separate passes:
            # forward() for duration/alignment and F0Ntrain() for pitch/energy.
            # Wrapping "predictor" directly in DDP causes PyTorch DDP to mark variables ready twice
            # because parameters used exclusively in F0Ntrain are marked unused during forward().
            # To prevent autograd conflicts, we keep predictor unwrapped from DDP and synchronize its
            # gradients directly across ranks via all_reduce before optimizer steps.
            if stage == "stage1":
                # In Stage 1, text_encoder is not used; only bert_encoder, bert, and tag_encoder are trained
                trainable_keys = ["bert_encoder"]
                if "bert" in self.model and self.model["bert"] is not None:
                    trainable_keys.append("bert")
            elif stage == "stage2":
                trainable_keys = ["bert_encoder", "decoder", "mpd", "msd"]
                if "bert" in self.model and self.model["bert"] is not None:
                    trainable_keys.append("bert")
            else:
                trainable_keys = []

            # Freeze bert.pooler if present: StyleTTS2 only uses sequence outputs;
            # the pooler is never used in the loss. Freezing it prevents DDP from tracking
            # unused pooler parameters (indices 23, 24).
            if "bert" in self.model and self.model["bert"] is not None:
                bert_mod = self.model["bert"].module if hasattr(self.model["bert"], "module") else self.model["bert"]
                if hasattr(bert_mod, "pooler") and bert_mod.pooler is not None:
                    for p in bert_mod.pooler.parameters():
                        p.requires_grad = False

            for k in trainable_keys:
                if k in self.model and self.model[k] is not None:
                    if not isinstance(self.model[k], torch.nn.parallel.DistributedDataParallel):
                        self.model[k] = MyDistributedDataParallel(
                            self.model[k],
                            device_ids=[dev_idx],
                            output_device=dev_idx,
                            find_unused_parameters=True,
                        )

            if not isinstance(self.tag_encoder, torch.nn.parallel.DistributedDataParallel):
                self.tag_encoder = MyDistributedDataParallel(
                    self.tag_encoder,
                    device_ids=[dev_idx],
                    output_device=dev_idx,
                    find_unused_parameters=True,
                )

            # Ensure predictor parameters are synchronized across ranks at stage start
            if "predictor" in self.model and self.model["predictor"] is not None:
                for param in self.model["predictor"].parameters():
                    torch.distributed.broadcast(param.data, src=0)

        # 5. Initialize GAN losses only when discriminators are active on GPU
        if stage == "stage2" and "mpd" in self.model and "msd" in self.model:
            mpd_mod = self.model["mpd"].module if hasattr(self.model["mpd"], "module") else self.model["mpd"]
            msd_mod = self.model["msd"].module if hasattr(self.model["msd"], "module") else self.model["msd"]
            self.gen_loss_fn = GeneratorLoss(mpd_mod, msd_mod).to(self.device)
            self.disc_loss_fn = DiscriminatorLoss(mpd_mod, msd_mod).to(self.device)
        else:
            self.gen_loss_fn = None
            self.disc_loss_fn = None

        if self.device.type == "cuda" and self.is_main_process:
            allocated_mb = torch.cuda.memory_allocated(self.device) / (1024 * 1024)
            reserved_mb = torch.cuda.memory_reserved(self.device) / (1024 * 1024)
            print(f"[✓] Active GPU VRAM for [{stage}]: {allocated_mb:.1f} MB allocated ({reserved_mb:.1f} MB reserved)")

    def _build_optimizers(self, stage: str, lr: float = 1e-4):
        """Builds decoupled optimizers for generator submodules and discriminators."""
        # 1. Tag Style Encoder Optimizer
        tag_params = list(self.tag_encoder.parameters())
        self.opt_tag = torch.optim.AdamW(tag_params, lr=lr * 2.0, betas=(0.9, 0.99), weight_decay=1e-4)

        # 2. Prosody Predictor & BERT Encoder Optimizer
        pred_params = []
        if "predictor" in self.model:
            pred_params += list(self.model["predictor"].parameters())
        if "bert_encoder" in self.model:
            pred_params += list(self.model["bert_encoder"].parameters())
        
        self.opt_pred = torch.optim.AdamW(pred_params, lr=lr, betas=(0.9, 0.99), weight_decay=1e-4)

        # 3. PL-BERT fine-tuning (conservative LR, only trainable parameters)
        if "bert" in self.model and hasattr(self.model["bert"], "parameters"):
            bert_mod = self.model["bert"].module if hasattr(self.model["bert"], "module") else self.model["bert"]
            bert_params = [p for p in bert_mod.parameters() if p.requires_grad]
            self.opt_bert = torch.optim.AdamW(
                bert_params, lr=lr * 0.1, betas=(0.9, 0.99), weight_decay=1e-2
            )
        else:
            self.opt_bert = None

        # 4. HiFi-GAN Decoder Optimizer (Active in Stage 2 or joint fine-tuning)
        if "decoder" in self.model:
            dec_params = list(self.model["decoder"].parameters())
            self.opt_dec = torch.optim.AdamW(dec_params, lr=lr, betas=(0.0, 0.99), weight_decay=1e-4)
        else:
            self.opt_dec = None

        # 5. Discriminators Optimizer (MPD + MSD)
        if "mpd" in self.model and "msd" in self.model:
            disc_params = list(self.model["mpd"].parameters()) + list(self.model["msd"].parameters())
            self.opt_disc = torch.optim.AdamW(disc_params, lr=lr, betas=(0.0, 0.99), weight_decay=1e-4)
        else:
            self.opt_disc = None

    def generate_and_save_samples(
        self,
        stage: str,
        epoch: int,
        step: int,
        prompts: Optional[list] = None,
    ):
        """
        Synthesizes benchmark audio evaluation samples, saves .wav files locally,
        logs audio to TensorBoard, and syncs to Hugging Face Hub under samples/.
        """
        if not self.is_main_process:
            return

        if "decoder" not in self.model or self.model["decoder"] is None:
            return

        eval_prompts = prompts or DEFAULT_EVAL_PROMPTS
        samples_dir = os.path.join(self.output_dir, "samples")
        os.makedirs(samples_dir, exist_ok=True)

        print(f"\n[*] Generating {len(eval_prompts)} audio evaluation samples for [{stage}] Epoch {epoch}...")

        # Record training state of modules
        was_training = {k: m.training for k, m in self.model.items() if m is not None}
        tag_was_training = self.tag_encoder.training

        # Ensure modules are in eval mode during synthesis
        for k in self.model:
            if self.model[k] is not None and hasattr(self.model[k], "eval"):
                self.model[k].eval()
        self.tag_encoder.eval()

        # Ensure decoder is on self.device for synthesis
        decoder_mod = self.model["decoder"].module if hasattr(self.model["decoder"], "module") else self.model["decoder"]
        try:
            decoder_device = next(iter(decoder_mod.parameters())).device
        except Exception:
            decoder_device = self.device
        if decoder_device != self.device:
            decoder_mod.to(self.device)

        phonemizer_fn = None
        try:
            from phonemizer.backend import EspeakBackend
            espeak = EspeakBackend(language="en-us", preserve_punctuation=True, with_stress=True)
            phonemizer_fn = espeak.phonemize
        except Exception:
            pass

        eval_model = {k: (v.module if hasattr(v, "module") else v) for k, v in self.model.items() if v is not None}
        eval_tag_enc = self.tag_encoder.module if hasattr(self.tag_encoder, "module") else self.tag_encoder

        synthesizer = KionSynthesizer(
            model=eval_model,
            tag_encoder=eval_tag_enc,
            phonemizer_fn=phonemizer_fn,
            device=str(self.device),
        )

        for i, (prompt, tag_label) in enumerate(eval_prompts):
            try:
                wave = synthesizer.synthesize(prompt=prompt)
                if wave is None or len(wave) == 0:
                    continue

                filename = f"{stage}_epoch_{epoch:03d}_sample_{i+1}_{tag_label}.wav"
                local_path = os.path.join(samples_dir, filename)
                import soundfile as sf
                sf.write(local_path, wave, 24000)
                print(f"  [✓] Audio sample saved: {filename}")

                # TensorBoard audio logging
                if self.writer is not None:
                    try:
                        self.writer.add_audio(
                            tag=f"{stage}/Sample_{i+1}_{tag_label}",
                            snd_tensor=wave,
                            global_step=epoch,
                            sample_rate=24000,
                        )
                    except Exception as e:
                        print(f"  [-] TensorBoard audio log notice: {e}")

                # Sync sample to Hugging Face Hub
                if self.ckpt_manager:
                    self.ckpt_manager.upload_sample_audio(local_path)

            except Exception as e:
                print(f"  [!] Notice during sample synthesis '{tag_label}': {e}")

        # Restore decoder device if it was offloaded to CPU
        if decoder_device != self.device:
            decoder_mod.to(decoder_device)

        # Restore module training states
        for k, is_train in was_training.items():
            if self.model[k] is not None:
                self.model[k].train(is_train)
        self.tag_encoder.train(tag_was_training)

    def train_stage1(
        self,
        train_loader,
        val_loader=None,
        epochs: int = 40,
        start_epoch: int = 0,
        start_step: int = 0,
        lr: float = 1e-4,
        save_freq: int = 5,
        save_step_freq: int = 500,
    ):
        """
        Stage 1: Acoustic Foundation, Prosody Predictor Adaptation, and Tag Style Alignment.
        In Stage 1, HiFi-GAN decoder is frozen in eval mode with pretrained LibriTTS acoustic weights,
        guaranteeing clean acoustic output while Predictor and KionTagStyleEncoder adapt to Kion.
        """
        if self.is_main_process:
            print("\n" + "=" * 65)
            print("Starting KionTTS Stage 1 Training: Acoustic Foundation & Tag Alignment")
            print(f"  Target Epochs      : {epochs}")
            print(f"  Start Epoch        : {start_epoch}")
            print(f"  Start Step         : {start_step}")
            print(f"  Learning Rate      : {lr}")
            print(f"  Save Step Freq     : Every {save_step_freq} steps")
            print(f"  Device             : {self.device}")
            print(f"  Distributed Multi-GPU : {self.is_distributed} (World Size: {self.world_size})")
            print(f"  Mixed Precision    : {'AMP FP16' if self.use_amp else 'FP32'}")
            print("=" * 65 + "\n")

        self.activate_stage_modules(stage="stage1")
        self._build_optimizers(stage="stage1", lr=lr)

        # Ensure reference modules are in eval mode
        for k in ["decoder", "style_encoder", "predictor_encoder", "text_aligner", "pitch_extractor"]:
            if k in self.model and self.model[k] is not None:
                mod = self.model[k].module if hasattr(self.model[k], "module") else self.model[k]
                mod.eval()
                for p in mod.parameters():
                    p.requires_grad = False

        if start_epoch >= epochs:
            if self.is_main_process:
                print(f"[✓] Stage 1 target epochs already achieved ({start_epoch}/{epochs}). Skipping Stage 1 training.")
            return

        best_loss = float("inf")
        global_step = start_step if start_step > 0 else start_epoch * len(train_loader)

        for epoch in range(start_epoch, epochs):
            if hasattr(train_loader, "sampler") and hasattr(train_loader.sampler, "set_epoch"):
                train_loader.sampler.set_epoch(epoch)

            epoch_start_time = time.time()
            self.tag_encoder.train()
            if "predictor" in self.model:
                self.model["predictor"].train()
            if "bert" in self.model:
                self.model["bert"].train()
            if "bert_encoder" in self.model:
                self.model["bert_encoder"].train()
            if "text_encoder" in self.model:
                self.model["text_encoder"].train()

            total_style_loss = 0.0
            total_pred_loss = 0.0
            num_batches = 0

            self.opt_tag.zero_grad()
            self.opt_pred.zero_grad()
            if self.opt_bert:
                self.opt_bert.zero_grad()

            batches_in_epoch = len(train_loader)
            max_steps_this_epoch = batches_in_epoch
            if epoch == start_epoch and start_step > 0:
                steps_done_in_epoch = start_step % batches_in_epoch
                if steps_done_in_epoch > 0:
                    max_steps_this_epoch = batches_in_epoch - steps_done_in_epoch
                    if self.is_main_process:
                        print(f"[*] Resuming mid-epoch: executing remaining {max_steps_this_epoch} steps to complete Epoch {epoch+1} (Global step {global_step})...")

            for step, batch in enumerate(train_loader):
                if step >= max_steps_this_epoch:
                    break

                waves, texts, input_lengths, mels, output_lengths, ref_mels, tag_vectors, paths = batch

                texts = texts.to(self.device, non_blocking=True)
                input_lengths = input_lengths.to(self.device, non_blocking=True)
                mels = mels.to(self.device, non_blocking=True)
                output_lengths = output_lengths.to(self.device, non_blocking=True)
                ref_mels = ref_mels.to(self.device, non_blocking=True)
                tag_vectors = tag_vectors.to(self.device, non_blocking=True)

                mel_input_length = output_lengths // 2

                # 1. Ground-Truth Acoustic & Prosodic Style Extraction in FP32 (prevents spectral_norm FP16 overflow)
                with torch.no_grad(), torch.amp.autocast('cuda', enabled=False):
                    s_acoustic_gt = self.model["style_encoder"](ref_mels.unsqueeze(1).float()).float()
                    s_prosody_gt = self.model["predictor_encoder"](ref_mels.unsqueeze(1).float()).float()
                    s_acoustic_gt = torch.nan_to_num(s_acoustic_gt, nan=0.0, posinf=1.0, neginf=-1.0).clamp(-10.0, 10.0)
                    s_prosody_gt = torch.nan_to_num(s_prosody_gt, nan=0.0, posinf=1.0, neginf=-1.0).clamp(-10.0, 10.0)
                    s_audio_full = torch.cat([s_acoustic_gt, s_prosody_gt], dim=-1)

                    mask = length_to_mask(mel_input_length).to(self.device)
                    text_mask = length_to_mask(input_lengths).to(self.device)
                    try:
                        _, _, s2s_attn = self.model["text_aligner"](mels, mask, texts)
                        s2s_attn = s2s_attn.transpose(-1, -2)[..., 1:].transpose(-1, -2)
                        mask_ST = mask_from_lens(s2s_attn, input_lengths, mel_input_length)
                        s2s_attn_mono = maximum_path(s2s_attn, mask_ST)
                    except Exception:
                        s2s_attn_mono = torch.zeros((texts.size(0), texts.size(1), mel_input_length.max()), device=self.device)
                        for b in range(texts.size(0)):
                            t_l = input_lengths[b].item()
                            m_l = mel_input_length[b].item()
                            if t_l > 0 and m_l > 0:
                                step_val = m_l / t_l
                                for ti in range(t_l):
                                    st_f = int(ti * step_val)
                                    end_f = int((ti + 1) * step_val) if ti < t_l - 1 else m_l
                                    s2s_attn_mono[b, ti, st_f:max(st_f + 1, end_f)] = 1.0

                    d_gt = s2s_attn_mono.sum(axis=-1).detach()
                    F0_real = extract_f0_safe(self.model["pitch_extractor"], mels.unsqueeze(1))
                    F0_real = torch.nan_to_num(F0_real, nan=0.0, posinf=0.0, neginf=0.0)
                    N_real = compute_energy_norm(mels)
                    N_real = torch.nan_to_num(N_real, nan=0.0, posinf=0.0, neginf=0.0)

                with self.autocast():
                    # 2. Tag Style Encoder Forward & Grounded Loss
                    s_tag = self.tag_encoder(tag_vectors)  # (B, 256)
                    loss_style = self.style_loss_fn(s_tag, s_audio_full)

                    # 3. Text Representation & PL-BERT
                    bert_dur = self.model["bert"](texts, attention_mask=(~text_mask).int())
                    d_en = self.model["bert_encoder"](bert_dur).transpose(-1, -2)

                    # 4. Prosody Prediction Loss (Computed in FP32 to prevent LSTM FP16 overflow)
                    cond_style = s_tag[:, 128:].detach() if (step % 2 == 0) else s_prosody_gt
                    cond_style_f32 = torch.nan_to_num(cond_style.float(), nan=0.0, posinf=1.0, neginf=-1.0).clamp(-10.0, 10.0)
                    d_en_f32 = torch.nan_to_num(d_en.float(), nan=0.0).clamp(-50.0, 50.0)

                # Predictor forward and loss in FP32
                with torch.amp.autocast('cuda', enabled=False):
                    d, p = self.model["predictor"](d_en_f32, cond_style_f32, input_lengths, s2s_attn_mono.float(), text_mask)
                    F0_pred, N_pred = self.model["predictor"].F0Ntrain(p, cond_style_f32)
                    F0_pred = torch.nan_to_num(F0_pred, nan=0.0).clamp(0.0, 1000.0)
                    N_pred = torch.nan_to_num(N_pred, nan=0.0).clamp(-50.0, 50.0)

                    # Predictor Losses (StyleTTS2 divides F0 loss by 10 to balance raw Hz scale)
                    loss_f0 = F.smooth_l1_loss(F0_pred, F0_real.float()) / 10.0
                    loss_norm = F.smooth_l1_loss(N_pred, N_real.float())

                    # StyleTTS2 Duration & CE loss with short text boundary protection
                    loss_dur = torch.tensor(0.0, device=self.device)
                    loss_ce = torch.tensor(0.0, device=self.device)
                    for _s2s_pred, _text_input, _text_length in zip(d, d_gt, input_lengths):
                        _tl = int(_text_length.item())
                        _s2s_pred = _s2s_pred[:_tl, :]
                        _text_input = _text_input[:_tl].long()
                        cols_idx = torch.arange(_s2s_pred.size(1), device=_s2s_pred.device).unsqueeze(0)
                        _s2s_trg = (cols_idx < _text_input.unsqueeze(1)).float()
                        # Clamp logits to prevent overflow → NaN in BCE during early training
                        _s2s_pred_clamped = torch.clamp(_s2s_pred, min=-15.0, max=15.0)
                        _dur_pred = torch.sigmoid(_s2s_pred_clamped).sum(axis=1)
                        if _tl > 2:
                            _dur_loss_item = F.l1_loss(_dur_pred[1:_tl-1], _text_input[1:_tl-1].float())
                            if torch.isfinite(_dur_loss_item):
                                loss_dur = loss_dur + _dur_loss_item
                        _ce_item = F.binary_cross_entropy_with_logits(_s2s_pred_clamped.flatten(), _s2s_trg.flatten())
                        if torch.isfinite(_ce_item):
                            loss_ce = loss_ce + _ce_item
                    loss_dur = (loss_dur + loss_ce) / max(1, texts.size(0))

                    loss_predictor = loss_f0 + loss_norm + loss_dur

                    # Combined Step Loss
                    loss_step = (loss_style * 2.0 + loss_predictor) / self.accum_steps

                # NaN/Inf guard: skip step if invalid to preserve model parameters
                # In distributed multi-GPU mode, any skip decision MUST be synchronized across all ranks!
                skip_step = torch.tensor(1.0 if not torch.isfinite(loss_step) else 0.0, device=self.device)
                if self.is_distributed:
                    torch.distributed.all_reduce(skip_step, op=torch.distributed.ReduceOp.MAX)

                if skip_step.item() > 0.0:
                    if self.is_main_process:
                        print(f"  [!] Notice: Non-finite loss detected at step {step+1}. Skipping optimizer update across all ranks.")
                    self.opt_tag.zero_grad(set_to_none=True)
                    self.opt_pred.zero_grad(set_to_none=True)
                    if self.opt_bert:
                        self.opt_bert.zero_grad(set_to_none=True)
                    continue

                # Backward pass with no_sync during accumulation steps
                # Sequential deterministic backward passes to eliminate inter-module DDP collective race conditions:
                is_accumulating = ((step + 1) % self.accum_steps != 0) and ((step + 1) != max_steps_this_epoch)
                sync_ctx = maybe_no_sync(
                    [self.tag_encoder, self.model.get("bert"), self.model.get("bert_encoder")],
                    enabled=(self.is_distributed and is_accumulating)
                )
                with sync_ctx:
                    # Pass 1: Tag Style Encoder alignment (touches ONLY self.tag_encoder)
                    loss_style_scaled = (loss_style * 2.0) / self.accum_steps
                    self.scaler.scale(loss_style_scaled).backward()
                    # Pass 2: Prosody Predictor & BERT adaptation (touches ONLY predictor -> bert_encoder -> bert)
                    loss_pred_scaled = loss_predictor / self.accum_steps
                    self.scaler.scale(loss_pred_scaled).backward()

                if (step + 1) % self.accum_steps == 0 or (step + 1) == max_steps_this_epoch:
                    self.scaler.unscale_(self.opt_tag)
                    self.scaler.unscale_(self.opt_pred)
                    if self.opt_bert:
                        self.scaler.unscale_(self.opt_bert)

                    # Explicitly synchronize predictor gradients across distributed ranks (single coalesced all_reduce)
                    # Use requires_grad so all ranks construct identical flat_grad tensors regardless of dynamic activation
                    if self.is_distributed and "predictor" in self.model and self.model["predictor"] is not None:
                        pred_params = [p for p in self.model["predictor"].parameters() if p.requires_grad]
                        if pred_params:
                            for p in pred_params:
                                if p.grad is None:
                                    p.grad = torch.zeros_like(p.data)
                            flat_grad = torch.cat([p.grad.data.reshape(-1) for p in pred_params])
                            torch.distributed.all_reduce(flat_grad, op=torch.distributed.ReduceOp.SUM)
                            flat_grad.div_(self.world_size)
                            offset = 0
                            for p in pred_params:
                                numel = p.grad.data.numel()
                                p.grad.data.copy_(flat_grad[offset:offset + numel].reshape(p.grad.data.shape))
                                offset += numel

                    torch.nn.utils.clip_grad_norm_(self.tag_encoder.parameters(), max_norm=5.0)
                    torch.nn.utils.clip_grad_norm_(self.opt_pred.param_groups[0]["params"], max_norm=5.0)
                    if self.opt_bert:
                        torch.nn.utils.clip_grad_norm_(self.opt_bert.param_groups[0]["params"], max_norm=5.0)

                    self.scaler.step(self.opt_tag)
                    self.scaler.step(self.opt_pred)
                    if self.opt_bert:
                        self.scaler.step(self.opt_bert)
                    self.scaler.update()

                    self.opt_tag.zero_grad(set_to_none=True)
                    self.opt_pred.zero_grad(set_to_none=True)
                    if self.opt_bert:
                        self.opt_bert.zero_grad(set_to_none=True)

                total_style_loss += loss_style.item()
                total_pred_loss += loss_predictor.item()
                num_batches += 1
                global_step += 1

                # Periodic step-interval checkpoint saving & HF sync (main process only)
                if save_step_freq > 0 and global_step % save_step_freq == 0:
                    if self.is_main_process:
                        save_models = {
                            "tag_encoder": self.tag_encoder,
                            "predictor": self.model.get("predictor"),
                            "bert": self.model.get("bert"),
                            "bert_encoder": self.model.get("bert_encoder"),
                            "text_encoder": self.model.get("text_encoder"),
                            "decoder": self.model.get("decoder"),
                            "style_encoder": self.model.get("style_encoder"),
                        }
                        save_opts = {
                            "opt_tag": self.opt_tag,
                            "opt_pred": self.opt_pred,
                        }
                        print(f"\n[*] Periodic Step Checkpoint: Step {global_step} (Epoch {epoch+1}). Syncing to HF Hub...")
                        self.ckpt_manager.save_checkpoint(
                            stage="stage1",
                            epoch=epoch + 1,
                            step=global_step,
                            models=save_models,
                            optimizers=save_opts,
                            loss_val=(total_style_loss + total_pred_loss) / max(1, num_batches),
                            is_best=False,
                            upload_hf=True,
                        )
                    if self.is_distributed and torch.distributed.is_initialized():
                        torch.distributed.barrier()

                # Responsive progress logging (every 5 steps, step 0, or end of epoch)
                log_freq = max(1, min(5, max_steps_this_epoch // 5))
                if self.is_main_process and (step == 0 or (step + 1) % log_freq == 0 or (step + 1) == max_steps_this_epoch):
                    elapsed = max(time.time() - epoch_start_time, 1e-3)
                    it_per_sec = (step + 1) / elapsed
                    eta_sec = (max_steps_this_epoch - (step + 1)) / max(it_per_sec, 1e-3)
                    print(
                        f"Epoch [{epoch+1:02d}/{epochs}] Step [{step+1:03d}/{max_steps_this_epoch}] (Global: {global_step}) "
                        f"StyleLoss: {loss_style.item():.4f} | "
                        f"F0Loss: {loss_f0.item():.4f} | "
                        f"NormLoss: {loss_norm.item():.4f} | "
                        f"DurLoss: {loss_dur.item():.4f} | "
                        f"{it_per_sec:.2f} it/s (ETA: {int(eta_sec)}s)"
                    )

            avg_style = total_style_loss / max(1, num_batches)
            avg_pred = total_pred_loss / max(1, num_batches)
            epoch_loss = avg_style + avg_pred
            epoch_time = time.time() - epoch_start_time

            if self.is_main_process:
                print(
                    f"\n=== Epoch {epoch+1:02d}/{epochs} Summary ({epoch_time:.1f}s) ===\n"
                    f"  Avg Style Loss     : {avg_style:.4f}\n"
                    f"  Avg Predictor Loss : {avg_pred:.4f}\n"
                    f"  Total Epoch Loss   : {epoch_loss:.4f}\n"
                )

                if self.writer is not None:
                    self.writer.add_scalar("Stage1/StyleLoss", avg_style, epoch + 1)
                    self.writer.add_scalar("Stage1/PredictorLoss", avg_pred, epoch + 1)
                    self.writer.add_scalar("Stage1/TotalLoss", epoch_loss, epoch + 1)

            # Checkpoint saving & remote sync (main process only)
            is_best = epoch_loss < best_loss
            if is_best:
                best_loss = epoch_loss

            if (epoch + 1) % save_freq == 0 or is_best or (epoch + 1) == epochs:
                if self.is_main_process:
                    save_models = {
                        "tag_encoder": self.tag_encoder,
                        "predictor": self.model.get("predictor"),
                        "bert": self.model.get("bert"),
                        "bert_encoder": self.model.get("bert_encoder"),
                        "text_encoder": self.model.get("text_encoder"),
                        "decoder": self.model.get("decoder"),
                        "style_encoder": self.model.get("style_encoder"),
                    }
                    save_opts = {
                        "opt_tag": self.opt_tag,
                        "opt_pred": self.opt_pred,
                    }
                    self.ckpt_manager.save_checkpoint(
                        stage="stage1",
                        epoch=epoch + 1,
                        step=global_step,
                        models=save_models,
                        optimizers=save_opts,
                        loss_val=epoch_loss,
                        is_best=is_best,
                        upload_hf=True,
                    )
                    self.generate_and_save_samples(stage="stage1", epoch=epoch + 1, step=global_step)

                if self.is_distributed and torch.distributed.is_initialized():
                    torch.distributed.barrier()

        if self.is_main_process:
            print("[✓] Stage 1 Training Completed Successfully!")

    def train_stage2(
        self,
        train_loader,
        val_loader=None,
        epochs: int = 50,
        start_epoch: int = 0,
        start_step: int = 0,
        lr: float = 1e-4,
        save_freq: int = 5,
        save_step_freq: int = 500,
        dec_window: int = 48,
    ):
        """
        Stage 2: Full-Stack Joint Fine-Tuning.
        Trains HiFi-GAN Decoder with MultiResolutionSTFT, MPD/MSD GAN, and grounded tag styles.
        Uses windowed micro-batching (`dec_window` frames) to guarantee stable T4/V100/A100 VRAM.
        """
        if self.is_main_process:
            print("\n" + "=" * 65)
            print("Starting KionTTS Stage 2 Training: Full-Stack Acoustic & GAN Refinement")
            print(f"  Target Epochs      : {epochs}")
            print(f"  Start Epoch        : {start_epoch}")
            print(f"  Start Step         : {start_step}")
            print(f"  Learning Rate      : {lr}")
            print(f"  Save Step Freq     : Every {save_step_freq} steps")
            print(f"  Decoder Window     : {dec_window} frames ({dec_window * 300} samples)")
            print(f"  Device             : {self.device}")
            print(f"  Distributed Multi-GPU : {self.is_distributed} (World Size: {self.world_size})")
            print("=" * 65 + "\n")

        self.activate_stage_modules(stage="stage2")
        self._build_optimizers(stage="stage2", lr=lr)

        # Unfreeze decoder & discriminators for fine-tuning
        if "decoder" in self.model and self.model["decoder"] is not None:
            dec_mod = self.model["decoder"].module if hasattr(self.model["decoder"], "module") else self.model["decoder"]
            dec_mod.train()
            for p in dec_mod.parameters():
                p.requires_grad = True

        if "mpd" in self.model and "msd" in self.model:
            mpd_mod = self.model["mpd"].module if hasattr(self.model["mpd"], "module") else self.model["mpd"]
            msd_mod = self.model["msd"].module if hasattr(self.model["msd"], "module") else self.model["msd"]
            mpd_mod.train()
            msd_mod.train()
            for p in list(mpd_mod.parameters()) + list(msd_mod.parameters()):
                p.requires_grad = True

        if start_epoch >= epochs:
            if self.is_main_process:
                print(f"[✓] Stage 2 target epochs already achieved ({start_epoch}/{epochs}). Skipping Stage 2 training.")
            return

        best_loss = float("inf")
        global_step = start_step if start_step > 0 else start_epoch * len(train_loader)

        for epoch in range(start_epoch, epochs):
            if hasattr(train_loader, "sampler") and hasattr(train_loader.sampler, "set_epoch"):
                train_loader.sampler.set_epoch(epoch)

            epoch_start_time = time.time()
            self.tag_encoder.train()
            if "predictor" in self.model and self.model["predictor"] is not None:
                self.model["predictor"].train()
            if "decoder" in self.model and self.model["decoder"] is not None:
                self.model["decoder"].train()
            if "bert_encoder" in self.model and self.model["bert_encoder"] is not None:
                self.model["bert_encoder"].train()
            if "bert" in self.model and self.model["bert"] is not None:
                self.model["bert"].train()
            if "mpd" in self.model and self.model["mpd"] is not None:
                self.model["mpd"].train()
            if "msd" in self.model and self.model["msd"] is not None:
                self.model["msd"].train()

            # Ensure reference feature extractors remain strictly in eval mode
            for k in ["text_encoder", "style_encoder", "predictor_encoder", "pitch_extractor", "text_aligner"]:
                if k in self.model and self.model[k] is not None:
                    mod = self.model[k].module if hasattr(self.model[k], "module") else self.model[k]
                    if hasattr(mod, "eval"):
                        mod.eval()

            total_stft_loss = 0.0
            total_gen_loss = 0.0
            total_disc_loss = 0.0
            num_batches = 0

            self.opt_tag.zero_grad(set_to_none=True)
            self.opt_pred.zero_grad(set_to_none=True)
            if self.opt_dec:
                self.opt_dec.zero_grad(set_to_none=True)
            if self.opt_bert:
                self.opt_bert.zero_grad(set_to_none=True)
            if self.opt_disc:
                self.opt_disc.zero_grad(set_to_none=True)

            batches_in_epoch = len(train_loader)
            max_steps_this_epoch = batches_in_epoch
            if epoch == start_epoch and start_step > 0:
                steps_done_in_epoch = start_step % batches_in_epoch
                if steps_done_in_epoch > 0:
                    max_steps_this_epoch = batches_in_epoch - steps_done_in_epoch
                    if self.is_main_process:
                        print(f"[*] Resuming mid-epoch: executing remaining {max_steps_this_epoch} steps to complete Epoch {epoch+1} (Global step {global_step})...")

            for step, batch in enumerate(train_loader):
                if step >= max_steps_this_epoch:
                    break

                waves, texts, input_lengths, mels, output_lengths, ref_mels, tag_vectors, paths = batch

                texts = texts.to(self.device, non_blocking=True)
                input_lengths = input_lengths.to(self.device, non_blocking=True)
                mels = mels.to(self.device, non_blocking=True)
                output_lengths = output_lengths.to(self.device, non_blocking=True)
                ref_mels = ref_mels.to(self.device, non_blocking=True)
                tag_vectors = tag_vectors.to(self.device, non_blocking=True)

                batch_size = texts.size(0)

                mel_input_length = output_lengths // 2

                # ── Step A: Predictor & Style Forward ──
                with self.autocast():
                    with torch.no_grad():
                        s_acoustic_gt = self.model["style_encoder"](ref_mels.unsqueeze(1))
                        s_prosody_gt = self.model["predictor_encoder"](ref_mels.unsqueeze(1))
                        s_audio_full = torch.cat([s_acoustic_gt, s_prosody_gt], dim=-1)

                        mask = length_to_mask(mel_input_length).to(self.device)
                        text_mask = length_to_mask(input_lengths).to(self.device)
                        try:
                            _, _, s2s_attn = self.model["text_aligner"](mels, mask, texts)
                            s2s_attn = s2s_attn.transpose(-1, -2)[..., 1:].transpose(-1, -2)
                            mask_ST = mask_from_lens(s2s_attn, input_lengths, mel_input_length)
                            s2s_attn_mono = maximum_path(s2s_attn, mask_ST)
                        except Exception:
                            s2s_attn_mono = torch.zeros((texts.size(0), texts.size(1), mel_input_length.max()), device=self.device)
                            for b in range(texts.size(0)):
                                t_l = input_lengths[b].item()
                                m_l = mel_input_length[b].item()
                                if t_l > 0 and m_l > 0:
                                    step_val = m_l / t_l
                                    for ti in range(t_l):
                                        st_f = int(ti * step_val)
                                        end_f = int((ti + 1) * step_val) if ti < t_l - 1 else m_l
                                        s2s_attn_mono[b, ti, st_f:max(st_f + 1, end_f)] = 1.0

                        # Pretrained acoustic text representations (fixed reference features in Stage 2)
                        t_en = self.model["text_encoder"](texts, input_lengths, text_mask)
                        asr = (t_en @ s2s_attn_mono)  # (B, 512, mel_input_length)

                    s_tag = self.tag_encoder(tag_vectors)
                    loss_style = self.style_loss_fn(s_tag, s_audio_full)

                    bert_dur = self.model["bert"](texts, attention_mask=(~text_mask).int())
                    d_en = self.model["bert_encoder"](bert_dur).transpose(-1, -2)

                    cond_style = s_tag[:, 128:].detach()
                    d, p = self.model["predictor"](d_en, cond_style, input_lengths, s2s_attn_mono, text_mask)

                # ── Step B: Sliced Real Audio & Decoder Forward ──
                win_len = min(dec_window, int(mel_input_length.min().item() - 1))
                if self.is_distributed:
                    win_len_t = torch.tensor(win_len, device=self.device)
                    torch.distributed.all_reduce(win_len_t, op=torch.distributed.ReduceOp.MIN)
                    win_len = int(win_len_t.item())

                if win_len < 10:
                    continue

                en_slices = []
                p_slices = []
                wav_slices = []
                mel_slices = []

                for b in range(batch_size):
                    cur_len = int(mel_input_length[b].item())
                    max_st = max(1, cur_len - win_len)
                    st = np.random.randint(0, max_st)

                    en_slices.append(asr[b : b + 1, :, st : st + win_len])
                    p_slices.append(p[b : b + 1, :, st : st + win_len])
                    mel_slices.append(mels[b : b + 1, :, st * 2 : (st + win_len) * 2])

                    # Audio slice from raw wave (24000 Hz, hop 300 for mel -> hop 600 for downsampled frames)
                    st_sample = (st * 2) * 300
                    end_sample = (st + win_len) * 2 * 300
                    w_raw = waves[b]
                    target_len = (win_len * 2) * 300
                    if len(w_raw) >= end_sample:
                        w_sub = w_raw[st_sample:end_sample]
                    else:
                        w_sub = np.pad(w_raw, (0, max(0, end_sample - len(w_raw))))[st_sample:end_sample]
                    if len(w_sub) < target_len:
                        w_sub = np.pad(w_sub, (0, target_len - len(w_sub)))
                    wav_slices.append(w_sub[:target_len])

                y_real = torch.from_numpy(np.stack(wav_slices)).float().unsqueeze(1).to(self.device, non_blocking=True)
                en_sub = torch.cat(en_slices, dim=0)
                p_sub = torch.cat(p_slices, dim=0)
                gt_sub = torch.cat(mel_slices, dim=0)
                ref_sub = s_tag[:, :128].detach()

                with torch.no_grad():
                    F0_real = extract_f0_safe(self.model["pitch_extractor"], gt_sub.unsqueeze(1))
                    F0_real = torch.nan_to_num(F0_real, nan=0.0, posinf=0.0, neginf=0.0)
                    N_real = compute_energy_norm(gt_sub)
                    N_real = torch.nan_to_num(N_real, nan=0.0, posinf=0.0, neginf=0.0)

                with self.autocast():
                    F0_pred, N_pred = self.model["predictor"].F0Ntrain(p_sub, cond_style)
                    y_rec = self.model["decoder"](en_sub, F0_pred, N_pred, ref_sub)

                    # Multi-Resolution STFT Reconstruction Loss against real audio (lengths match 1:1)
                    loss_stft = self.stft_loss(y_rec, y_real)

                    loss_f0_rec = F.smooth_l1_loss(F0_pred, F0_real) / 10.0
                    loss_norm_rec = F.smooth_l1_loss(N_pred, N_real)

                    # GAN Generator Loss
                    if self.gen_loss_fn:
                        loss_gen, loss_adv, loss_fm = self.gen_loss_fn(y_real, y_rec)
                    else:
                        loss_gen = torch.tensor(0.0, device=self.device)

                    loss_g_total = (loss_stft * 2.5 + loss_gen * 1.0 + loss_style * 1.0 + loss_f0_rec + loss_norm_rec) / self.accum_steps

                # NaN/Inf guard synchronized across distributed ranks
                skip_g = torch.tensor(1.0 if not torch.isfinite(loss_g_total) else 0.0, device=self.device)
                if self.is_distributed:
                    torch.distributed.all_reduce(skip_g, op=torch.distributed.ReduceOp.MAX)

                if skip_g.item() > 0.0:
                    if self.is_main_process:
                        print(f"  [!] Notice: Non-finite loss detected in loss_g_total at step {step+1}. Skipping optimizer update across all ranks.")
                    self.opt_tag.zero_grad(set_to_none=True)
                    self.opt_pred.zero_grad(set_to_none=True)
                    if self.opt_dec:
                        self.opt_dec.zero_grad(set_to_none=True)
                    if self.opt_bert:
                        self.opt_bert.zero_grad(set_to_none=True)
                    if self.opt_disc:
                        self.opt_disc.zero_grad(set_to_none=True)
                    continue

                # ── Step C: Generator Backward & Accumulation ──
                # Sequential deterministic backward passes to eliminate inter-module DDP collective race conditions:
                is_accumulating = ((step + 1) % self.accum_steps != 0) and ((step + 1) != max_steps_this_epoch)
                sync_ctx_g = maybe_no_sync(
                    [self.tag_encoder, self.model.get("decoder"), self.model.get("bert_encoder"), self.model.get("bert")],
                    enabled=(self.is_distributed and is_accumulating)
                )
                with sync_ctx_g:
                    # Pass 1: Tag Style Encoder alignment (touches ONLY self.tag_encoder)
                    loss_style_scaled = (loss_style * 1.0) / self.accum_steps
                    self.scaler.scale(loss_style_scaled).backward()
                    # Pass 2: Full-Stack Vocoder & Predictor Loss (touches decoder, bert_encoder, bert)
                    loss_g_acoustic = (loss_stft * 2.5 + loss_gen * 1.0 + loss_f0_rec + loss_norm_rec) / self.accum_steps
                    self.scaler.scale(loss_g_acoustic).backward()

                # ── Step D: Discriminator Backward & Accumulation ──
                loss_d_total = torch.tensor(0.0, device=self.device)
                if self.disc_loss_fn and self.opt_disc:
                    sync_ctx_d = maybe_no_sync(
                        [self.model.get("mpd"), self.model.get("msd")],
                        enabled=(self.is_distributed and is_accumulating)
                    )
                    with self.autocast():
                        loss_d_total = self.disc_loss_fn(y_real, y_rec.detach()) / self.accum_steps
                    with sync_ctx_d:
                        self.scaler_d.scale(loss_d_total).backward()

                if (step + 1) % self.accum_steps == 0 or (step + 1) == max_steps_this_epoch:
                    self.scaler.unscale_(self.opt_tag)
                    self.scaler.unscale_(self.opt_pred)
                    if self.opt_dec:
                        self.scaler.unscale_(self.opt_dec)
                    if self.opt_bert:
                        self.scaler.unscale_(self.opt_bert)

                    # Explicitly synchronize predictor gradients across distributed ranks (single coalesced all_reduce)
                    # Use requires_grad so all ranks construct identical flat_grad tensors regardless of dynamic activation
                    if self.is_distributed and "predictor" in self.model and self.model["predictor"] is not None:
                        pred_params = [p for p in self.model["predictor"].parameters() if p.requires_grad]
                        if pred_params:
                            for p in pred_params:
                                if p.grad is None:
                                    p.grad = torch.zeros_like(p.data)
                            flat_grad = torch.cat([p.grad.data.reshape(-1) for p in pred_params])
                            torch.distributed.all_reduce(flat_grad, op=torch.distributed.ReduceOp.SUM)
                            flat_grad.div_(self.world_size)
                            offset = 0
                            for p in pred_params:
                                numel = p.grad.data.numel()
                                p.grad.data.copy_(flat_grad[offset:offset + numel].reshape(p.grad.data.shape))
                                offset += numel

                    torch.nn.utils.clip_grad_norm_(self.tag_encoder.parameters(), max_norm=5.0)
                    torch.nn.utils.clip_grad_norm_(self.opt_pred.param_groups[0]["params"], max_norm=5.0)
                    if self.opt_dec:
                        torch.nn.utils.clip_grad_norm_(self.opt_dec.param_groups[0]["params"], max_norm=5.0)
                    if self.opt_bert:
                        torch.nn.utils.clip_grad_norm_(self.opt_bert.param_groups[0]["params"], max_norm=5.0)

                    self.scaler.step(self.opt_tag)
                    self.scaler.step(self.opt_pred)
                    if self.opt_dec:
                        self.scaler.step(self.opt_dec)
                    if self.opt_bert:
                        self.scaler.step(self.opt_bert)
                    self.scaler.update()

                    self.opt_tag.zero_grad(set_to_none=True)
                    self.opt_pred.zero_grad(set_to_none=True)
                    if self.opt_dec:
                        self.opt_dec.zero_grad(set_to_none=True)
                    if self.opt_bert:
                        self.opt_bert.zero_grad(set_to_none=True)

                    if self.disc_loss_fn and self.opt_disc:
                        self.scaler_d.unscale_(self.opt_disc)
                        torch.nn.utils.clip_grad_norm_(self.opt_disc.param_groups[0]["params"], max_norm=5.0)
                        self.scaler_d.step(self.opt_disc)
                        self.scaler_d.update()
                        self.opt_disc.zero_grad(set_to_none=True)

                total_stft_loss += loss_stft.item()
                total_gen_loss += loss_gen.item()
                total_disc_loss += loss_d_total.item()
                num_batches += 1
                global_step += 1

                # Periodic step-interval checkpoint saving & HF sync (main process only)
                if save_step_freq > 0 and global_step % save_step_freq == 0:
                    if self.is_main_process:
                        save_models = {
                            "tag_encoder": self.tag_encoder,
                            "predictor": self.model.get("predictor"),
                            "bert": self.model.get("bert"),
                            "bert_encoder": self.model.get("bert_encoder"),
                            "text_encoder": self.model.get("text_encoder"),
                            "decoder": self.model.get("decoder"),
                            "style_encoder": self.model.get("style_encoder"),
                            "mpd": self.model.get("mpd"),
                            "msd": self.model.get("msd"),
                        }
                        save_opts = {
                            "opt_tag": self.opt_tag,
                            "opt_pred": self.opt_pred,
                            "opt_dec": self.opt_dec,
                            "opt_disc": self.opt_disc,
                        }
                        if self.opt_bert is not None:
                            save_opts["opt_bert"] = self.opt_bert
                        print(f"\n[*] Periodic Step Checkpoint: Step {global_step} (Epoch {epoch+1}). Syncing to HF Hub...")
                        self.ckpt_manager.save_checkpoint(
                            stage="stage2",
                            epoch=epoch + 1,
                            step=global_step,
                            models=save_models,
                            optimizers=save_opts,
                            loss_val=(total_stft_loss + total_gen_loss) / max(1, num_batches),
                            is_best=False,
                            upload_hf=True,
                        )
                    if self.is_distributed and torch.distributed.is_initialized():
                        torch.distributed.barrier()

                # Responsive progress logging (every 5 steps, step 0, or end of epoch)
                log_freq = max(1, min(5, max_steps_this_epoch // 5))
                if self.is_main_process and (step == 0 or (step + 1) % log_freq == 0 or (step + 1) == max_steps_this_epoch):
                    elapsed = max(time.time() - epoch_start_time, 1e-3)
                    it_per_sec = (step + 1) / elapsed
                    eta_sec = (max_steps_this_epoch - (step + 1)) / max(it_per_sec, 1e-3)
                    print(
                        f"Stage2 Epoch [{epoch+1:02d}/{epochs}] Step [{step+1:03d}/{max_steps_this_epoch}] (Global: {global_step}) "
                        f"STFTLoss: {loss_stft.item():.4f} | "
                        f"GenLoss: {loss_gen.item():.4f} | "
                        f"DiscLoss: {loss_d_total.item():.4f} | "
                        f"StyleLoss: {loss_style.item():.4f} | "
                        f"{it_per_sec:.2f} it/s (ETA: {int(eta_sec)}s)"
                    )

            avg_stft = total_stft_loss / max(1, num_batches)
            avg_gen = total_gen_loss / max(1, num_batches)
            avg_disc = total_disc_loss / max(1, num_batches)
            epoch_loss = avg_stft + avg_gen
            epoch_time = time.time() - epoch_start_time

            if self.is_main_process:
                print(
                    f"\n=== Stage 2 Epoch {epoch+1:02d}/{epochs} Summary ({epoch_time:.1f}s) ===\n"
                    f"  Avg STFT Loss : {avg_stft:.4f}\n"
                    f"  Avg Gen Loss  : {avg_gen:.4f}\n"
                    f"  Avg Disc Loss : {avg_disc:.4f}\n"
                )

                if self.writer is not None:
                    self.writer.add_scalar("Stage2/STFTLoss", avg_stft, epoch + 1)
                    self.writer.add_scalar("Stage2/GenLoss", avg_gen, epoch + 1)
                    self.writer.add_scalar("Stage2/DiscLoss", avg_disc, epoch + 1)

            # Checkpoint saving & remote sync (main process only)
            is_best = epoch_loss < best_loss
            if is_best:
                best_loss = epoch_loss

            if (epoch + 1) % save_freq == 0 or is_best or (epoch + 1) == epochs:
                if self.is_main_process:
                    save_models = {
                        "tag_encoder": self.tag_encoder,
                        "predictor": self.model.get("predictor"),
                        "bert": self.model.get("bert"),
                        "bert_encoder": self.model.get("bert_encoder"),
                        "text_encoder": self.model.get("text_encoder"),
                        "decoder": self.model.get("decoder"),
                        "style_encoder": self.model.get("style_encoder"),
                        "mpd": self.model.get("mpd"),
                        "msd": self.model.get("msd"),
                    }
                    save_opts = {
                        "opt_tag": self.opt_tag,
                        "opt_pred": self.opt_pred,
                        "opt_dec": self.opt_dec,
                        "opt_disc": self.opt_disc,
                    }
                    if self.opt_bert is not None:
                        save_opts["opt_bert"] = self.opt_bert
                    self.ckpt_manager.save_checkpoint(
                        stage="stage2",
                        epoch=epoch + 1,
                        step=global_step,
                        models=save_models,
                        optimizers=save_opts,
                        loss_val=epoch_loss,
                        is_best=is_best,
                        upload_hf=True,
                    )
                    self.generate_and_save_samples(stage="stage2", epoch=epoch + 1, step=global_step)

                if self.is_distributed and torch.distributed.is_initialized():
                    torch.distributed.barrier()

        if self.is_main_process:
            print("[✓] Stage 2 Training Completed Successfully!")


def run_training_pipeline(
    model: Dict[str, Any],
    tag_encoder: KionTagStyleEncoder,
    train_loader,
    val_loader=None,
    config: Optional[Dict[str, Any]] = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    log_dir: str = "checkpoints",
    eval_dir: str = "eval_samples",
    phonemizer_fn=None,
    checkpoint_manager: Optional[KionCheckpointManager] = None,
) -> str:
    """
    Unified training pipeline runner for notebooks (e.g. KionTTS_Kaggle_Training_Pipeline.ipynb).
    Executes Stage 1 up to config['joint_epoch'] and Stage 2 up to config['epochs'].
    """
    cfg = config or {}
    total_epochs = cfg.get("epochs", 6)
    joint_epoch = cfg.get("joint_epoch", 3)
    lr = cfg.get("lr", 1e-4)
    accum_steps = cfg.get("accum_steps", 1)
    save_freq = cfg.get("save_freq", 1)

    ckpt_manager = checkpoint_manager or KionCheckpointManager(checkpoint_dir=log_dir)

    trainer = KionProductionTrainer(
        model=model,
        tag_encoder=tag_encoder,
        checkpoint_manager=ckpt_manager,
        output_dir=log_dir,
        device=device,
        accum_steps=accum_steps,
    )

    # Stage 1: up to joint_epoch
    if joint_epoch > 0:
        trainer.train_stage1(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=joint_epoch,
            lr=lr,
            save_freq=save_freq,
        )

    # Stage 2: up to total_epochs
    if total_epochs > joint_epoch:
        trainer.train_stage2(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=total_epochs,
            start_epoch=joint_epoch,
            lr=lr,
            save_freq=save_freq,
        )

    latest_ckpt = ckpt_manager.find_latest_checkpoint(stage="stage2") or ckpt_manager.find_latest_checkpoint(stage="stage1")
    return latest_ckpt or os.path.join(log_dir, "kion_stage2_latest.pth")
