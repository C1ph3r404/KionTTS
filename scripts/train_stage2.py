#!/usr/bin/env python3
"""
Production Stage 2 Training Script for KionTTS.
Fine-tunes:
- HiFi-GAN Decoder with Multi-Resolution STFT Loss
- Multi-Period (MPD) and Multi-Scale (MSD) GAN Discriminators
- Grounded KionTagStyleEncoder & Prosody Predictor
- Windowed micro-batching prevents GPU out-of-memory.
"""

import os
import sys
import argparse
import yaml
import torch
from munch import Munch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "StyleTTS2"))

from kion_core.dataset import build_kion_dataloader
from kion_core.tag_style_encoder import KionTagStyleEncoder
from kion_core.checkpoint_manager import KionCheckpointManager
from kion_core.trainer import KionProductionTrainer

from models import build_model, load_checkpoint, load_ASR_models, load_F0_models
from utils import recursive_munch
from Utils.PLBERT.util import load_plbert


def load_styletts2_backbone(config_path: str, pretrained_ckpt: str, device: str = "cuda"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    args = recursive_munch(config["model_params"])

    asr_cfg = os.path.join(REPO_ROOT, "StyleTTS2", config["ASR_config"])
    asr_path = os.path.join(REPO_ROOT, "StyleTTS2", config["ASR_path"])
    f0_path = os.path.join(REPO_ROOT, "StyleTTS2", config["F0_path"])
    bert_dir = os.path.join(REPO_ROOT, "StyleTTS2", config["PLBERT_dir"])

    text_aligner = load_ASR_models(asr_path, asr_cfg)
    pitch_extractor = load_F0_models(f0_path)
    plbert = load_plbert(bert_dir)
    model = build_model(args, text_aligner, pitch_extractor, plbert)

    if os.path.exists(pretrained_ckpt):
        print(f"[*] Loading pretrained LibriTTS checkpoint from: {pretrained_ckpt}")
        model, _, _, _ = load_checkpoint(model, None, pretrained_ckpt, load_only_params=True)

    return model, config


def main():
    parser = argparse.ArgumentParser(description="Run KionTTS Stage 2 Training.")
    parser.add_argument("--config", type=str, default="StyleTTS2/Configs/config_ft.yml", help="Path to config YAML")
    parser.add_argument("--pretrained_ckpt", type=str, default="Models/LibriTTS/epochs_2nd_00020.pth", help="Pretrained LibriTTS checkpoint")
    parser.add_argument("--stage1_ckpt", type=str, default=None, help="Stage 1 checkpoint to initialize weights from")
    parser.add_argument("--manifest", type=str, default="DataSet/train_manifest.json", help="Path to train manifest JSON")
    parser.add_argument("--val_manifest", type=str, default="DataSet/val_manifest.json", help="Path to val manifest JSON")
    parser.add_argument("--data_root", type=str, default="DataSet", help="Root data directory containing wavs/")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--epochs", type=int, default=50, help="Total Stage 2 epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size per GPU")
    parser.add_argument("--dec_window", type=int, default=48, help="Decoder random window length in frames (48 frames = 14,400 samples)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--save_freq", type=int, default=5, help="Epoch save frequency")
    parser.add_argument("--save_step_freq", type=int, default=500, help="Step-interval save & HF sync frequency")
    parser.add_argument("--accum_steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training")
    parser.add_argument("--num_workers", type=int, default=2, help="DataLoader worker processes per GPU")
    parser.add_argument("--hf_repo", type=str, default="nate0001/KionTTS", help="Hugging Face repo for checkpoint sync")
    parser.add_argument("--hf_token", type=str, default=None, help="Hugging Face API token (defaults to Kaggle Secrets or env)")
    args = parser.parse_args()

    # Setup distributed multi-GPU (Kaggle dual T4 / torchrun / accelerate)
    local_rank = args.local_rank if args.local_rank != -1 else int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    is_distributed = (world_size > 1) or (local_rank != -1)

    if is_distributed:
        if local_rank == -1:
            local_rank = 0
        torch.cuda.set_device(local_rank)
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend="nccl")
        device = f"cuda:{local_rank}"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    is_main_process = (rank == 0)

    # 1. Load StyleTTS2 Backbone
    model, config = load_styletts2_backbone(args.config, args.pretrained_ckpt, device=device)

    # 2. Initialize KionTagStyleEncoder
    tag_encoder = KionTagStyleEncoder(num_tags=24, emb_dim=64, style_dim=256)

    # 3. Setup Checkpoint Manager
    ckpt_manager = KionCheckpointManager(
        checkpoint_dir=args.checkpoint_dir,
        repo_id=args.hf_repo,
        hf_token=args.hf_token,
    )

    # Check for existing Stage 2 checkpoint to resume, or initialize from Stage 1
    latest_stage2 = ckpt_manager.find_latest_checkpoint(stage="stage2")
    start_epoch = 0
    start_step = 0

    if latest_stage2:
        models_to_load = {"tag_encoder": tag_encoder}
        for k in ["predictor", "bert", "bert_encoder", "text_encoder", "decoder", "mpd", "msd"]:
            if k in model:
                models_to_load[k] = model[k]
        state_meta = ckpt_manager.load_checkpoint(latest_stage2, models=models_to_load, load_optimizers=False)
        start_epoch = state_meta.get("epoch", 0)
        start_step = state_meta.get("step", 0)
        if is_main_process:
            print(f"[✓] Resuming Stage 2 from epoch {start_epoch}, step {start_step}...")

        if start_epoch >= args.epochs:
            if is_main_process:
                print(f"[✓] Stage 2 target of {args.epochs} epochs has already been completed (found checkpoint at epoch {start_epoch}). Exiting Stage 2.")
            return
    else:
        # Load weights from Stage 1
        s1_ckpt = args.stage1_ckpt or ckpt_manager.find_latest_checkpoint(stage="stage1")
        if s1_ckpt and os.path.exists(s1_ckpt):
            if is_main_process:
                print(f"[*] Initializing Stage 2 from verified Stage 1 weights: {s1_ckpt}")
            models_to_load = {"tag_encoder": tag_encoder}
            for k in ["predictor", "bert", "bert_encoder", "text_encoder"]:
                if k in model:
                    models_to_load[k] = model[k]
            ckpt_manager.load_checkpoint(s1_ckpt, models=models_to_load, load_optimizers=False)

    # 4. Build DataLoaders (with DistributedSampler across dual T4s)
    train_loader = build_kion_dataloader(
        manifest_path=args.manifest,
        root_dir=args.data_root,
        batch_size=args.batch_size,
        validation=False,
        num_workers=args.num_workers,
        is_distributed=is_distributed,
        rank=rank,
        world_size=world_size,
    )
    val_loader = None
    if os.path.exists(args.val_manifest):
        val_loader = build_kion_dataloader(
            manifest_path=args.val_manifest,
            root_dir=args.data_root,
            batch_size=args.batch_size,
            validation=True,
            num_workers=args.num_workers,
            is_distributed=False,
        )

    # 5. Initialize Production Trainer
    trainer = KionProductionTrainer(
        model=model,
        tag_encoder=tag_encoder,
        checkpoint_manager=ckpt_manager,
        output_dir=args.checkpoint_dir,
        device=device,
        accum_steps=args.accum_steps,
        local_rank=local_rank,
        rank=rank,
        world_size=world_size,
        is_distributed=is_distributed,
    )

    try:
        # 6. Execute Stage 2 Training
        trainer.train_stage2(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=args.epochs,
            start_epoch=start_epoch,
            start_step=start_step,
            lr=args.lr,
            save_freq=args.save_freq,
            save_step_freq=args.save_step_freq,
            dec_window=args.dec_window,
        )
    finally:
        if is_distributed and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
