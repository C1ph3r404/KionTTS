"""
Colab Cell 05: Fast Prototype Model Training (~1k Samples)
Runs an optimized validation experiment on 1,000 samples across dominant styles
using FP16 mixed precision and CUDA acceleration.
"""

import os
import sys
import json
import torch
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from model.models.kion_tts_model import KionTTSModel
from model.training.losses import KionLoss
from model.data.kion_dataset import KionDataset, collate_fn_kion


def run_prototype_training(
    preprocessed_manifest: str = "/content/dataset/train_preprocessed.json",
    checkpoint_dir: str = "/content/drive/MyDrive/KionTTS_Checkpoints",
    epochs: int = 20,
    batch_size: int = 32,
    lr: float = 2e-4,
    subset_size: int = 1000,
):
    print("=" * 60)
    print("Starting KionTTS Accelerated Prototype Training...")
    print(f"Target Subset Size: {subset_size} samples")
    print(f"Epochs: {epochs} | Batch Size: {batch_size} | Learning Rate: {lr}")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    print(f"Using compute device: {device}")

    # 1. Dataset & Subset DataLoader
    full_dataset = KionDataset(preprocessed_manifest)
    num_samples = min(subset_size, len(full_dataset))
    subset_indices = list(range(num_samples))
    dataset = Subset(full_dataset, subset_indices)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn_kion,
        num_workers=2 if os.name != "nt" else 0,
        pin_memory=torch.cuda.is_available(),
    )

    # 2. Model, Loss, Optimizer, Mixed Precision Scaler
    model = KionTTSModel().to(device)
    criterion = KionLoss().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = GradScaler(enabled=torch.cuda.is_available())

    os.makedirs(checkpoint_dir, exist_ok=True)
    prototype_save_path = os.path.join(checkpoint_dir, "prototype_kion.pt")

    # 3. Training Loop
    for epoch in range(1, epochs + 1):
        model.train()
        total_epoch_loss = 0.0
        total_mel_loss = 0.0
        total_dur_loss = 0.0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{epochs}")
        for batch in pbar:
            tokens = batch["tokens"].to(device, non_blocking=True)
            style_weights = batch["style_weights"].to(device, non_blocking=True)
            target_mel = batch["mel"].to(device, non_blocking=True)
            target_dur = batch["durations"].to(device, non_blocking=True)
            target_log_dur = batch["target_log_durations"].to(device, non_blocking=True)
            target_pitch = batch["pitch"].to(device, non_blocking=True)
            target_energy = batch["energy"].to(device, non_blocking=True)
            text_mask = batch["text_mask"].to(device, non_blocking=True)

            optimizer.zero_grad()

            with autocast(enabled=torch.cuda.is_available()):
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

            total_epoch_loss += loss.item()
            total_mel_loss += loss_dict["mel_loss"].item()
            total_dur_loss += loss_dict["duration_loss"].item()

            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                mel=f"{loss_dict['mel_loss'].item():.4f}",
                dur=f"{loss_dict['duration_loss'].item():.4f}",
            )

        scheduler.step()
        avg_loss = total_epoch_loss / len(dataloader)
        avg_mel = total_mel_loss / len(dataloader)
        print(f"[*] Epoch {epoch} Complete | Avg Total Loss: {avg_loss:.4f} | Avg Mel Loss: {avg_mel:.4f}")

    # 4. Save Prototype Model
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "epoch": epochs,
            "loss": avg_loss,
        },
        prototype_save_path,
    )
    print(f"\n[+] Prototype training completed successfully! Saved to: {prototype_save_path}")


if __name__ == "__main__":
    run_prototype_training()
