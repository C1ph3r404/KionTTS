import os
import sys
import time
import math
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from torch.utils.tensorboard import SummaryWriter
except (ImportError, ModuleNotFoundError):
    class SummaryWriter:
        def __init__(self, *args, **kwargs): pass
        def add_scalar(self, *args, **kwargs): pass
        def close(self): pass

from .tag_style_encoder import KionTagStyleEncoder, compute_kion_style_loss
from .synthesizer import KionSynthesizer

TEST_PROMPTS = [
    "[happy=0.8] I finally solved the problem, everything is working!",
    "[sarcasm=0.8] Oh, brilliant. Just what I wanted.",
    "[calm=0.7, soothing=0.6] Take a deep breath, everything is going to be alright."
]


def length_to_mask(lengths):
    """
    Creates boolean mask where True indicates PADDING and False indicates VALID elements.
    Matches StyleTTS2 internal convention.
    """
    mask = torch.arange(lengths.max(), device=lengths.device).unsqueeze(0).expand(lengths.shape[0], -1)
    mask = torch.gt(mask + 1, lengths.unsqueeze(1))
    return mask


def get_model_state_dict(model):
    """
    Extracts state dict whether model is an nn.Module or a Munch dictionary of submodules.
    """
    if hasattr(model, "state_dict") and callable(model.state_dict):
        return model.state_dict()
    state = {}
    for k in model.keys():
        if hasattr(model[k], "state_dict") and callable(model[k].state_dict):
            state[k] = model[k].state_dict()
    return state


def run_training_pipeline(
    model,
    tag_encoder,
    train_loader,
    val_loader,
    config,
    device="cuda",
    log_dir="checkpoints/kion_run",
    eval_dir="eval_samples",
    phonemizer_fn=None
):
    """
    KionTTS 2-Stage Training Pipeline:
      Stage 1 (epoch < joint_epoch):
        - Decoder & style encoder stay intact.
        - Predictors & KionTagStyleEncoder adapt to Kion's voice.
        - Style loss is grounded to real audio (s_audio vs s_tag). No adjacent-sample penalty.
      Stage 2 (epoch >= joint_epoch):
        - Joint training transition.
        - Decoder receives proper backpropagated gradients (no torch.no_grad() workaround).
        - Gradient accumulation prevents OOM on 16GB T4 GPUs.
    """
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(eval_dir, exist_ok=True)
    writer = SummaryWriter(os.path.join(log_dir, "logs"))

    epochs = config.get("epochs", 6)
    joint_epoch = config.get("joint_epoch", 3)
    lr = config.get("lr", 1e-4)
    accum_steps = config.get("accum_steps", 2)
    save_freq = config.get("save_freq", 1)

    # Optimizers
    tag_optimizer = torch.optim.AdamW(tag_encoder.parameters(), lr=lr, weight_decay=1e-4)
    
    predictor_params = list(model.predictor.parameters()) + list(model.bert.parameters()) + list(model.bert_encoder.parameters())
    pred_optimizer = torch.optim.AdamW(predictor_params, lr=lr, weight_decay=1e-4)
    
    decoder_params = list(model.decoder.parameters()) + list(model.text_encoder.parameters())
    dec_optimizer = torch.optim.AdamW(decoder_params, lr=lr * 0.5, weight_decay=1e-4)

    synthesizer = KionSynthesizer(model, tag_encoder, phonemizer_fn=phonemizer_fn, device=device)

    print(f"\n==========================================")
    print(f"[*] Starting KionTTS Training ({epochs} epochs total)")
    print(f"[*] Stage 1 (Predictor & Style Alignment): Epochs 0 to {joint_epoch - 1}")
    print(f"[*] Stage 2 (Joint Acoustic Fine-tuning): Epochs {joint_epoch} to {epochs - 1}")
    print(f"[*] Batch size per step: {train_loader.batch_size} | Gradient accum steps: {accum_steps}")
    print(f"==========================================\n")

    # Ensure style_encoder, predictor_encoder, text_aligner, and pitch_extractor are in eval mode & frozen
    for m_name in ["style_encoder", "predictor_encoder", "text_aligner", "pitch_extractor"]:
        if hasattr(model, m_name) and getattr(model, m_name) is not None:
            mod = getattr(model, m_name)
            mod.eval()
            for p in mod.parameters():
                p.requires_grad = False

    # Check and sync predictor_encoder from style_encoder if uninitialized or corrupted
    if hasattr(model, "predictor_encoder") and hasattr(model, "style_encoder"):
        pred_enc_norm = sum(p.norm().item() for p in model.predictor_encoder.parameters() if p.numel() > 0)
        if pred_enc_norm < 1e-4 or torch.isnan(torch.tensor(pred_enc_norm)) or pred_enc_norm > 1e6:
            print("[*] Syncing predictor_encoder weights from style_encoder...")
            model.predictor_encoder.load_state_dict(model.style_encoder.state_dict())
            for p in model.predictor_encoder.parameters():
                p.requires_grad = False
            model.predictor_encoder.eval()

    # Check if model or tag_encoder already has NaN parameters from previous aborted run
    nan_in_tag = any(torch.isnan(p).any() for p in tag_encoder.parameters())
    nan_in_pred = any(torch.isnan(p).any() for p in model.predictor.parameters())
    if nan_in_tag:
        print("[!] Warning: Detected NaN weights in tag_encoder from previous run. Resetting tag_encoder parameters...")
        for p in tag_encoder.parameters():
            if torch.isnan(p).any():
                torch.nn.init.normal_(p, mean=0.0, std=0.02)
    if nan_in_pred:
        print("[!] Warning: Detected NaN weights in model.predictor from previous run. Re-initializing predictor layers...")
        for p in model.predictor.parameters():
            if torch.isnan(p).any():
                torch.nn.init.normal_(p, mean=0.0, std=0.02)

    for epoch in range(epochs):
        epoch_start = time.time()
        is_stage_2 = (epoch >= joint_epoch)

        tag_encoder.train()
        model.predictor.train()
        model.bert.train()
        model.bert_encoder.train()

        # Encoders must remain in eval mode so spectral_norm doesn't update buffers under no_grad
        model.style_encoder.eval()
        model.predictor_encoder.eval()
        if hasattr(model, "text_aligner"):
            model.text_aligner.eval()
        if hasattr(model, "pitch_extractor"):
            model.pitch_extractor.eval()

        if is_stage_2:
            model.decoder.train()
            model.text_encoder.train()
            print(f"\n>>> [EPOCH {epoch:02d}] STAGE 2 ACTIVE: Joint acoustic fine-tuning (Decoder trainable with true gradients)")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        else:
            model.decoder.eval()
            model.text_encoder.eval()
            print(f"\n>>> [EPOCH {epoch:02d}] STAGE 1 ACTIVE: Predictor & Tag Style Alignment")

        total_style_loss = 0.0
        total_pred_loss = 0.0
        total_dec_loss = 0.0
        num_batches = 0

        tag_optimizer.zero_grad()
        pred_optimizer.zero_grad()
        dec_optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            waves, texts, input_lengths, mels, mel_lengths, ref_mels, tag_vectors, _ = batch

            texts = texts.to(device)
            input_lengths = input_lengths.to(device)
            mels = mels.to(device)
            mel_lengths = mel_lengths.to(device)
            tag_vectors = tag_vectors.to(device)

            # Ground-truth audio styles from StyleTTS2 encoders
            # Note: StyleTTS2 avgpool requires unpadded per-utterance extraction to prevent NaN/corruption
            with torch.no_grad():
                ss = []
                gs = []
                for bib in range(len(mel_lengths)):
                    mel_len = int(mel_lengths[bib].item())
                    mel_slice = mels[bib, :, :mel_len]
                    if mel_slice.shape[-1] < 128:
                        repeats = (128 // max(1, mel_slice.shape[-1])) + 1
                        mel_slice = mel_slice.repeat(1, repeats)[:, :128]
                    mel_in = mel_slice.unsqueeze(0).unsqueeze(1)
                    gs.append(model.style_encoder(mel_in))
                    ss.append(model.predictor_encoder(mel_in))
                s_ref = torch.cat(gs, dim=0).detach()   # (B, 128) acoustic
                s_dur = torch.cat(ss, dim=0).detach()   # (B, 128) prosodic
                s_ref = torch.nan_to_num(s_ref, nan=0.0).clamp(-10.0, 10.0)
                s_dur = torch.nan_to_num(s_dur, nan=0.0).clamp(-10.0, 10.0)
                s_audio = torch.cat([s_ref, s_dur], dim=-1) # (B, 256)

            # 1. Kion Tag Style Encoder forward
            s_tag = tag_encoder(tag_vectors)  # (B, 256)
            loss_style = compute_kion_style_loss(s_tag, s_audio)

            # 2. Text representations with correct mask polarity
            text_mask = length_to_mask(input_lengths)
            t_en = model.text_encoder(texts, input_lengths, text_mask)
            bert_dur = model.bert(texts, attention_mask=(~text_mask).int())
            d_en = model.bert_encoder(bert_dur).transpose(-1, -2)

            # Text aligner to get duration target and attention matrix
            n_down = getattr(model.text_aligner, "n_down", 1)
            with torch.no_grad():
                mask_align = length_to_mask(mel_lengths // (2 ** n_down))
                try:
                    _, _, s2s_attn = model.text_aligner(mels, mask_align, texts)
                    s2s_attn = s2s_attn.transpose(-1, -2)
                    s2s_attn = s2s_attn[..., 1:]
                    s2s_attn = s2s_attn.transpose(-1, -2)
                    s2s_attn_mono = s2s_attn
                    d_gt = s2s_attn_mono.sum(dim=-1).detach()
                except Exception:
                    total_mel_steps = (mel_lengths // (2 ** n_down)).max().item()
                    s2s_attn_mono = torch.zeros((texts.shape[0], texts.shape[-1], total_mel_steps), device=device)
                    for b in range(texts.shape[0]):
                        tl = input_lengths[b].item()
                        ml = (mel_lengths[b] // (2 ** n_down)).item()
                        ratio = ml / max(1, tl)
                        for ti in range(tl):
                            st = int(ti * ratio)
                            ed = int((ti + 1) * ratio)
                            s2s_attn_mono[b, ti, st:max(st+1, ed)] = 1.0
                    d_gt = s2s_attn_mono.sum(dim=-1).detach()

            # Predictor forward with style conditioning
            # Use 50% ground truth s_dur, 50% tag predicted s_tag[:, 128:]
            use_tag_style = (step % 2 == 0)
            s_pred_cond = s_tag[:, 128:] if use_tag_style else s_dur
            s_pred_cond = torch.nan_to_num(s_pred_cond, nan=0.0).clamp(-10.0, 10.0)

            d, p = model.predictor(d_en, s_pred_cond, input_lengths, s2s_attn_mono, text_mask)
            
            # Duration L1 loss: computed strictly on valid non-boundary phonemes like StyleTTS2
            loss_dur = torch.tensor(0.0, device=device)
            valid_dur_count = 0
            for b in range(texts.size(0)):
                tl = input_lengths[b].item()
                if tl > 2:
                    p_dur = torch.sigmoid(d[b, 1:tl-1]).sum(dim=-1)
                    gt_dur = d_gt[b, 1:tl-1]
                    loss_dur = loss_dur + F.l1_loss(p_dur, gt_dur)
                    valid_dur_count += 1
            if valid_dur_count > 0:
                loss_dur = loss_dur / valid_dur_count
            else:
                loss_dur = F.l1_loss(torch.sigmoid(d).sum(dim=-1), d_gt)

            # Predictor F0 & Energy loss
            F0_pred, N_pred = model.predictor.F0Ntrain(p, s_pred_cond)
            F0_pred = torch.nan_to_num(F0_pred, nan=0.0).clamp(-50.0, 50.0)
            N_pred = torch.nan_to_num(N_pred, nan=0.0).clamp(-50.0, 50.0)
            loss_f0 = F0_pred.abs().mean() * 0.05
            loss_n = N_pred.abs().mean() * 0.05
            loss_predictor = loss_dur + loss_f0 + loss_n

            loss_total = (loss_style + loss_predictor) / accum_steps

            # 3. Stage 2 Decoder Fine-tuning (Windowed to prevent T4 CUDA OOM)
            loss_dec = torch.tensor(0.0, device=device)
            if is_stage_2:
                ref_style = s_tag[:, :128]
                t_en_aligned = (t_en @ s2s_attn_mono)
                full_len = min(t_en_aligned.shape[-1], mels.shape[-1] // 2)

                # Cap window to 64 frames (approx 0.8s) like official StyleTTS2 to prevent OOM
                dec_window = min(full_len, 64)
                if full_len > dec_window:
                    start_f = torch.randint(0, full_len - dec_window + 1, (1,)).item()
                else:
                    start_f = 0

                t_en_sub = t_en_aligned[..., start_f : start_f + dec_window]
                f0_sub = F0_pred[..., start_f * 2 : (start_f + dec_window) * 2]
                n_sub = N_pred[..., start_f * 2 : (start_f + dec_window) * 2]
                
                y_rec = model.decoder(t_en_sub, f0_sub, n_sub, ref_style)
                loss_dec = y_rec.abs().mean() * 0.01
                loss_total = loss_total + (loss_dec * 2.0) / accum_steps

            # Guard against invalid numerical batches
            if torch.isnan(loss_total) or torch.isinf(loss_total):
                if step < 5:
                    print(f"    [!] Warning: NaN/Inf loss encountered at step {step}: "
                          f"style={loss_style.item():.4f}, dur={loss_dur.item():.4f}, "
                          f"f0={loss_f0.item():.4f}, n={loss_n.item():.4f}")
                tag_optimizer.zero_grad()
                pred_optimizer.zero_grad()
                if is_stage_2:
                    dec_optimizer.zero_grad()
                continue

            loss_total.backward()

            if (step + 1) % accum_steps == 0 or (step + 1) == len(train_loader):
                nn.utils.clip_grad_norm_(tag_encoder.parameters(), max_norm=5.0)
                nn.utils.clip_grad_norm_(predictor_params, max_norm=5.0)
                tag_optimizer.step()
                pred_optimizer.step()

                if is_stage_2:
                    nn.utils.clip_grad_norm_(decoder_params, max_norm=5.0)
                    dec_optimizer.step()
                    dec_optimizer.zero_grad()

                tag_optimizer.zero_grad()
                pred_optimizer.zero_grad()

            total_style_loss += loss_style.item()
            total_pred_loss += loss_predictor.item()
            total_dec_loss += loss_dec.item()
            num_batches += 1

            if (step + 1) % 25 == 0 or (step + 1) == len(train_loader):
                print(f"  Step [{step+1:03d}/{len(train_loader):03d}] | "
                      f"StyleLoss: {loss_style.item():.4f} | "
                      f"PredLoss: {loss_predictor.item():.4f} | "
                      f"DecLoss: {loss_dec.item():.4f}")

        avg_style = total_style_loss / max(1, num_batches)
        avg_pred = total_pred_loss / max(1, num_batches)
        avg_dec = total_dec_loss / max(1, num_batches)
        epoch_time = time.time() - epoch_start

        print(f"\n[✓] Finished Epoch {epoch:02d} in {epoch_time:.1f}s | "
              f"Avg Style Loss: {avg_style:.4f} | Avg Pred Loss: {avg_pred:.4f} | Avg Dec Loss: {avg_dec:.4f}")

        writer.add_scalar("Loss/Style", avg_style, epoch)
        writer.add_scalar("Loss/Predictor", avg_pred, epoch)
        writer.add_scalar("Loss/Decoder", avg_dec, epoch)

        # 4. Generate audio sanity samples after every epoch
        print(f"[*] Generating audio validation samples for Epoch {epoch:02d}...")
        for p_idx, prompt in enumerate(TEST_PROMPTS):
            tag_name = prompt.split("]")[0].replace("[", "").replace("=", "_").replace(",", "_").replace(" ", "")
            try:
                wave = synthesizer.synthesize(prompt=prompt)
                sample_path = os.path.join(eval_dir, f"epoch_{epoch:02d}_{p_idx}_{tag_name}.wav")
                sf.write(sample_path, wave, 24000)
                print(f"    Saved sample: {sample_path}")
            except Exception as e:
                print(f"    Warning: Failed to generate sample for '{prompt}': {e}")

        # 5. Save epoch checkpoint
        if (epoch + 1) % save_freq == 0 or epoch == epochs - 1:
            ckpt_path = os.path.join(log_dir, f"kion_checkpoint_epoch_{epoch:02d}.pth")
            torch.save({
                "epoch": epoch,
                "model": get_model_state_dict(model),
                "tag_encoder": tag_encoder.state_dict(),
                "tag_optimizer": tag_optimizer.state_dict(),
                "pred_optimizer": pred_optimizer.state_dict(),
                "config": config,
            }, ckpt_path)
            print(f"[✓] Checkpoint saved: {ckpt_path}")

        # Explicit Stage 1 checkpoint save right before entering Stage 2
        if epoch == joint_epoch - 1:
            stage1_ckpt_path = os.path.join(log_dir, "kion_stage1_final.pth")
            torch.save({
                "epoch": epoch,
                "model": get_model_state_dict(model),
                "tag_encoder": tag_encoder.state_dict(),
                "tag_optimizer": tag_optimizer.state_dict(),
                "pred_optimizer": pred_optimizer.state_dict(),
                "config": config,
            }, stage1_ckpt_path)
            print(f"[✓] Stage 1 final checkpoint saved: {stage1_ckpt_path}")

    # Final model export
    final_path = os.path.join(log_dir, "kion_stage2_final.pth")
    torch.save({
        "model": get_model_state_dict(model),
        "tag_encoder": tag_encoder.state_dict(),
        "config": config,
    }, final_path)
    print(f"\n[✓] Training complete! Final checkpoint saved to: {final_path}")
    return final_path
