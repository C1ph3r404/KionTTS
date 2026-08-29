"""
Colab Cell 06: Full-Scale KionTTS Model Training
Integrates automatic Google Drive checkpoint recovery, FP16 mixed-precision training,
cuDNN benchmark acceleration, periodic validation evaluation, and best-model tracking.
"""

import os
import sys
import torch
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Optional

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from model.models.kion_tts_model import KionTTSModel
from model.training.losses import KionLoss
from model.data.kion_dataset import KionDataset, collate_fn_kion
from model.training.checkpoint_manager import DriveCheckpointManager


def evaluate(model, val_loader, criterion, device):
    model.eval()
    total_val_loss = 0.0
    total_mel_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            tokens = batch["tokens"].to(device, non_blocking=True)
            style_weights = batch["style_weights"].to(device, non_blocking=True)
            target_mel = batch["mel"].to(device, non_blocking=True)
            target_dur = batch["durations"].to(device, non_blocking=True)
            target_log_dur = batch["target_log_durations"].to(device, non_blocking=True)
            target_pitch = batch["pitch"].to(device, non_blocking=True)
            target_energy = batch["energy"].to(device, non_blocking=True)
            text_mask = batch["text_mask"].to(device, non_blocking=True)

            predictions = model(
                text_tokens=tokens,
                style_weights=style_weights,
                target_durations=target_dur,
                target_pitch=target_pitch,
                target_energy=target_energy,
                text_mask=text_mask,
            )

            targets = {
                "target_mel": target_mel,
                "target_log_durations": target_log_dur,
                "target_pitch": target_pitch,
                "target_energy": target_energy,
            }

            losses = criterion(predictions, targets)
            total_val_loss += losses["total_loss"].item()
            total_mel_loss += losses["mel_loss"].item()

    avg_val_loss = total_val_loss / max(1, len(val_loader))
    avg_mel_loss = total_mel_loss / max(1, len(val_loader))
    return avg_val_loss, avg_mel_loss


def run_full_training(
    train_manifest: str = "/content/dataset/train_preprocessed.json",
    val_manifest: str = "/content/dataset/val_preprocessed.json",
    checkpoint_dir: str = "/content/drive/MyDrive/KionTTS_Checkpoints",
    epochs: int = 100,
    batch_size: int = 32,
    lr: float = 2e-4,
    save_step_interval: int = 500,
    eval_step_interval: int = 250,
):
    print("=" * 65)
    print("Starting Full-Scale KionTTS Training...")
    print(f"Checkpoints Directory: {checkpoint_dir}")
    print(f"Batch Size: {batch_size} | Learning Rate: {lr}")
    print("=" * 65)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    print(f"Device: {device}")

    # 1. Initialize Checkpoint Manager
    ckpt_manager = DriveCheckpointManager(checkpoint_dir=checkpoint_dir, keep_last_n=3)

    # 2. Data Loaders
    train_dataset = KionDataset(train_manifest)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn_kion,
        num_workers=2 if os.name != "nt" else 0,
        pin_memory=torch.cuda.is_available(),
    )

    val_dataset = KionDataset(val_manifest)
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn_kion,
        num_workers=2 if os.name != "nt" else 0,
        pin_memory=torch.cuda.is_available(),
    )

    # 3. Model, Loss, Optimizer, Scaler
    model = KionTTSModel().to(device)
    criterion = KionLoss().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.98), weight_decay=1e-2)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())

    # 4. Check for Resumption from Drive
    state, start_step, start_epoch, best_val_loss = ckpt_manager.load_checkpoint(device=device)
    if state is not None:
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        if "scheduler_state_dict" in state:
            scheduler.load_state_dict(state["scheduler_state_dict"])
        if "scaler_state_dict" in state and torch.cuda.is_available():
            scaler.load_state_dict(state["scaler_state_dict"])

    global_step = start_step

    # 5. Training Loop
    for epoch in range(start_epoch + 1, epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")

        for batch in pbar:
            global_step += 1
            tokens = batch["tokens"].to(device, non_blocking=True)
            style_weights = batch["style_weights"].to(device, non_blocking=True)
            target_mel = batch["mel"].to(device, non_blocking=True)
            target_dur = batch["durations"].to(device, non_blocking=True)
            target_log_dur = batch["target_log_durations"].to(device, non_blocking=True)
            target_pitch = batch["pitch"].to(device, non_blocking=True)
            target_energy = batch["energy"].to(device, non_blocking=True)
            text_mask = batch["text_mask"].to(device, non_blocking=True)

            optimizer.zero_grad()

            with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                predictions = model(
                    text_tokens=tokens,
                    style_weights=style_weights,
                    target_durations=target_dur,
                    target_pitch=target_pitch,
                    target_energy=target_energy,
                    text_mask=text_mask,
                )

                targets = {
                    "target_mel": target_mel,
                    "target_log_durations": target_log_dur,
                    "target_pitch": target_pitch,
                    "target_energy": target_energy,
                }

                loss_dict = criterion(predictions, targets)
                loss = loss_dict["total_loss"]

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            pbar.set_postfix(
                step=global_step,
                loss=f"{loss.item():.4f}",
                mel=f"{loss_dict['mel_loss'].item():.4f}",
            )

            # Periodic Evaluation & Checkpointing
            if global_step % eval_step_interval == 0:
                val_loss, val_mel = evaluate(model, val_loader, criterion, device)
                is_best = val_loss < best_val_loss
                if is_best:
                    best_val_loss = val_loss

                print(f"\n[Step {global_step}] Validation Total Loss: {val_loss:.4f} | Mel Loss: {val_mel:.4f}")

                # Save checkpoint state to Drive
                save_state = {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "scaler_state_dict": scaler.state_dict() if torch.cuda.is_available() else {},
                    "best_val_loss": best_val_loss,
                }
                ckpt_manager.save_checkpoint(
                    state=save_state,
                    step=global_step,
                    epoch=epoch,
                    val_loss=val_loss,
                    is_best=is_best,
                )
                model.train()

        scheduler.step()

    print("\n[+] Full training finished!")


if __name__ == "__main__":
    run_full_training()
