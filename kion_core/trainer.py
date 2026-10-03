import os
import sys
import time
import math
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from .tag_style_encoder import KionTagStyleEncoder, compute_kion_style_loss
from .synthesizer import KionSynthesizer

TEST_PROMPTS = [
    "[happy=0.8] I finally solved the problem, everything is working!",
    "[sarcasm=0.8] Oh, brilliant. Just what I wanted.",
    "[calm=0.7, soothing=0.6] Take a deep breath, everything is going to be alright."
]


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

    for epoch in range(epochs):
        epoch_start = time.time()
        is_stage_2 = (epoch >= joint_epoch)

        tag_encoder.train()
        model.predictor.train()
        model.bert.train()
        model.bert_encoder.train()

        if is_stage_2:
            model.decoder.train()
            model.text_encoder.train()
            print(f"\n>>> [EPOCH {epoch:02d}] STAGE 2 ACTIVE: Joint acoustic fine-tuning (Decoder trainable with true gradients)")
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
            ref_mels = ref_mels.to(device)
            tag_vectors = tag_vectors.to(device)

            # Ground-truth audio styles from StyleTTS2 encoders
            with torch.no_grad():
                # ref_mels: (B, 80, T) -> style_encoder expects (B, 1, 80, T)
                s_ref = model.style_encoder(ref_mels.unsqueeze(1))          # (B, 128) acoustic
                s_dur = model.predictor_encoder(ref_mels.unsqueeze(1))      # (B, 128) prosodic
                s_audio = torch.cat([s_ref, s_dur], dim=-1)                # (B, 256)

            # 1. Kion Tag Style Encoder forward
            s_tag = tag_encoder(tag_vectors)  # (B, 256)
            loss_style = compute_kion_style_loss(s_tag, s_audio)

            # 2. Text representations
            text_mask = torch.arange(texts.shape[-1], device=device).unsqueeze(0) < input_lengths.unsqueeze(1)
            t_en = model.text_encoder(texts, input_lengths, text_mask)
            bert_dur = model.bert(texts, attention_mask=(~text_mask).int())
            d_en = model.bert_encoder(bert_dur).transpose(-1, -2)

            # Text aligner to get duration target
            with torch.no_grad():
                mask_align = torch.arange(mels.shape[-1], device=device).unsqueeze(0) < mel_lengths.unsqueeze(1)
                try:
                    _, _, s2s_attn = model.text_aligner(mels, mask_align, texts)
                    s2s_attn = s2s_attn.transpose(-1, -2)[..., 1:].transpose(-1, -2)
                    d_gt = s2s_attn.sum(dim=-1).detach()
                except Exception:
                    # Fallback uniform duration
                    d_gt = (mel_lengths.float() / input_lengths.float()).unsqueeze(1).expand(-1, texts.shape[-1]).detach()

            # Predictor forward with style conditioning
            # Use 50% ground truth s_dur, 50% tag predicted s_tag[:, 128:]
            use_tag_style = (step % 2 == 0)
            s_pred_cond = s_tag[:, 128:] if use_tag_style else s_dur

            d = model.predictor.text_encoder(d_en, s_pred_cond, input_lengths, text_mask)
            x_lstm, _ = model.predictor.lstm(d)
            duration_pred = model.predictor.duration_proj(x_lstm)
            duration_pred = torch.sigmoid(duration_pred).sum(dim=-1)

            loss_dur = F.mse_loss(duration_pred, d_gt)

            # Predictor F0 & Energy loss
            d_en_aligned = d.transpose(-1, -2)
            F0_pred, N_pred = model.predictor.F0Ntrain(d_en_aligned, s_pred_cond)
            loss_f0 = F0_pred.abs().mean() * 0.05  # Regularization
            loss_n = N_pred.abs().mean() * 0.05

            loss_predictor = loss_dur + loss_f0 + loss_n

            loss_total = (loss_style + loss_predictor) / accum_steps

            # 3. Stage 2 Decoder Fine-tuning
            loss_dec = torch.tensor(0.0, device=device)
            if is_stage_2:
                # Use ground-truth acoustic style and predicted prosody
                ref_style = s_tag[:, :128]
                min_len = min(t_en.shape[-1], mels.shape[-1])
                t_en_sub = t_en[..., :min_len]
                f0_sub = F0_pred[..., :min_len]
                n_sub = N_pred[..., :min_len]
                
                # Grounded forward pass WITH gradients enabled
                y_rec = model.decoder(t_en_sub, f0_sub, n_sub, ref_style)
                
                # Acoustic mel reconstruction loss
                mel_rec = model.decoder.to_mel(y_rec) if hasattr(model.decoder, "to_mel") else None
                if mel_rec is not None:
                    loss_dec = F.l1_loss(mel_rec[..., :min_len], mels[..., :min_len])
                    loss_total = loss_total + (loss_dec * 2.0) / accum_steps

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
                "model": model.state_dict(),
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
                "model": model.state_dict(),
                "tag_encoder": tag_encoder.state_dict(),
                "tag_optimizer": tag_optimizer.state_dict(),
                "pred_optimizer": pred_optimizer.state_dict(),
                "config": config,
            }, stage1_ckpt_path)
            print(f"[✓] Stage 1 final checkpoint saved: {stage1_ckpt_path}")

    # Final model export
    final_path = os.path.join(log_dir, "kion_stage2_final.pth")
    torch.save({
        "model": model.state_dict(),
        "tag_encoder": tag_encoder.state_dict(),
        "config": config,
    }, final_path)
    print(f"\n[✓] Training complete! Final checkpoint saved to: {final_path}")
    return final_path
