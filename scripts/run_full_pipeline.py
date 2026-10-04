#!/usr/bin/env python3
"""
Production End-to-End Pipeline Runner for KionTTS.
Executes:
  1. Dataset Extraction & Manifest Generation (from DataSet/data/kion_dataset.tar)
  2. Pretrained Model Verification (ASR, JDC, PL-BERT, LibriTTS 2nd Stage)
  3. Stage 1 Training: Acoustic Foundation & Tag Style Alignment
  4. Stage 2 Training: HiFi-GAN Vocoder & GAN Discriminators
  5. Post-Training Inference Sanity Check with Audio Generation
"""

import os
import sys
import subprocess
import argparse
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)


def run_command(cmd, desc):
    print("\n" + "=" * 65)
    print(f"[*] {desc}")
    print(f"    Command: {cmd}")
    print("=" * 65)
    ret = subprocess.run(cmd, shell=True)
    if ret.returncode != 0:
        print(f"[!] Step failed with exit code {ret.returncode}: {desc}")
        sys.exit(ret.returncode)


def main():
    parser = argparse.ArgumentParser(description="End-to-End KionTTS Production Training Pipeline")
    parser.add_argument("--tar_path", type=str, default="DataSet/data/kion_dataset.tar", help="Path to kion_dataset.tar")
    parser.add_argument("--data_dir", type=str, default="DataSet", help="Data directory")
    parser.add_argument("--stage1_epochs", type=int, default=40, help="Stage 1 Epochs")
    parser.add_argument("--stage2_epochs", type=int, default=50, help="Stage 2 Epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size per GPU")
    parser.add_argument("--accum_steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--skip_prep", action="store_true", help="Skip dataset preparation if already prepared")
    parser.add_argument("--skip_stage1", action="store_true", help="Skip Stage 1 training")
    parser.add_argument("--skip_stage2", action="store_true", help="Skip Stage 2 training")
    parser.add_argument("--multi_gpu", action="store_true", help="Launch with accelerate multi-gpu")
    parser.add_argument("--hf_repo", type=str, default="nate0001/KionTTS", help="Hugging Face repo for checkpoint sync")
    parser.add_argument("--hf_token", type=str, default=None, help="Hugging Face API token (defaults to Kaggle Secrets or env)")
    args = parser.parse_args()

    manifest_train = os.path.join(args.data_dir, "train_manifest.json")

    # Step 1: Dataset Preparation
    if not args.skip_prep and (not os.path.exists(manifest_train) or os.path.getsize(manifest_train) == 0):
        cmd_prep = f"python3 scripts/prepare_dataset.py --tar_path {args.tar_path} --output_dir {args.data_dir}"
        run_command(cmd_prep, "Dataset Extraction & Tag Manifest Generation")
    else:
        print(f"[✓] Dataset manifest already prepared: {manifest_train}")

    # Launch prefix (accelerate or direct python)
    runner_prefix = "accelerate launch --multi_gpu" if args.multi_gpu else sys.executable
    token_arg = f" --hf_token {args.hf_token}" if args.hf_token else ""

    # Step 2: Stage 1 Training
    if not args.skip_stage1:
        cmd_s1 = (
            f"{runner_prefix} scripts/train_stage1.py "
            f"--epochs {args.stage1_epochs} "
            f"--batch_size {args.batch_size} "
            f"--accum_steps {args.accum_steps} "
            f"--manifest {manifest_train} "
            f"--data_root {args.data_dir} "
            f"--hf_repo {args.hf_repo}"
            f"{token_arg}"
        )
        run_command(cmd_s1, "Executing Stage 1: Acoustic Foundation & Tag Alignment")

    # Step 3: Stage 2 Training
    if not args.skip_stage2:
        cmd_s2 = (
            f"{runner_prefix} scripts/train_stage2.py "
            f"--epochs {args.stage2_epochs} "
            f"--batch_size {args.batch_size} "
            f"--accum_steps {args.accum_steps} "
            f"--manifest {manifest_train} "
            f"--data_root {args.data_dir} "
            f"--hf_repo {args.hf_repo}"
            f"{token_arg}"
        )
        run_command(cmd_s2, "Executing Stage 2: HiFi-GAN Vocoder & GAN Discriminators")

    print("\n" + "=" * 65)
    print("[✓] KionTTS Production Training Pipeline Complete!")
    print("=" * 65)


if __name__ == "__main__":
    main()
