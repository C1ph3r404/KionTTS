#!/usr/bin/env python3
"""
Standalone Style Manifold Evaluator for KionTTS Stage 1.
Evaluates Teacher vs. Student style alignment on the validation set
using a saved checkpoint without disrupting active training.
"""

import os
import sys
import glob
import torch
import torch.nn.functional as F
import yaml
from munch import Munch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "StyleTTS2"))

from kion_core.dataset import build_kion_dataloader
from kion_core.tag_style_encoder import KionTagStyleEncoder
from models import build_model, load_checkpoint, load_ASR_models, load_F0_models
from utils import recursive_munch
from Utils.PLBERT.util import load_plbert


def evaluate_checkpoint(
    checkpoint_path: str = None,
    checkpoint_dir: str = "checkpoints",
    val_manifest: str = "/content/dataset/val_manifest.json",
    data_root: str = "/content/dataset",
    config_path: str = "StyleTTS2/Configs/config_ft.yml",
    pretrained_ckpt: str = "Models/LibriTTS/epochs_2nd_00020.pth",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    max_batches: int = 10,
):
    # 1. Locate checkpoint
    if not checkpoint_path:
        ckpts = sorted(glob.glob(os.path.join(checkpoint_dir, "*.pth")) + glob.glob(os.path.join(checkpoint_dir, "*.pt")))
        if not ckpts:
            print(f"[!] No checkpoints found in {checkpoint_dir} yet.")
            return None
        checkpoint_path = ckpts[-1]

    print(f"[*] Evaluating style alignment for checkpoint: {checkpoint_path}")

    # 2. Build Teacher models (frozen)
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
    teacher_model = build_model(args, text_aligner, pitch_extractor, plbert)

    if not os.path.isabs(pretrained_ckpt):
        pretrained_ckpt = os.path.join(REPO_ROOT, pretrained_ckpt)

    if os.path.exists(pretrained_ckpt):
        teacher_model, _, _, _ = load_checkpoint(teacher_model, None, pretrained_ckpt, load_only_params=True)
        print(f"[✓] Loaded pretrained teacher backbone from: {pretrained_ckpt}")
    else:
        print(f"[!] Warning: pretrained_ckpt not found at: {pretrained_ckpt}")

    style_encoder = teacher_model["style_encoder"].to(device).eval()
    predictor_encoder = teacher_model.get("predictor_encoder", style_encoder).to(device).eval()

    # 3. Build Student model (tag_encoder)
    tag_encoder = KionTagStyleEncoder().to(device).eval()
    ckpt_data = torch.load(checkpoint_path, map_location=device)
    if "net" in ckpt_data and "tag_encoder" in ckpt_data["net"]:
        tag_encoder.load_state_dict(ckpt_data["net"]["tag_encoder"])
        print("[✓] Loaded trained tag_encoder from ckpt['net']['tag_encoder']")
    elif "tag_encoder" in ckpt_data:
        tag_encoder.load_state_dict(ckpt_data["tag_encoder"])
        print("[✓] Loaded trained tag_encoder from ckpt['tag_encoder']")
    elif "model_state_dict" in ckpt_data:
        tag_encoder.load_state_dict(ckpt_data["model_state_dict"])
    elif "state_dict" in ckpt_data:
        tag_encoder.load_state_dict(ckpt_data["state_dict"])
    else:
        try:
            tag_encoder.load_state_dict(ckpt_data)
        except Exception:
            pass

    # 4. DataLoader
    val_loader = build_kion_dataloader(
        manifest_path=val_manifest,
        root_dir=data_root,
        batch_size=4,
        validation=True,
        num_workers=0,
    )

    # 5. Evaluate
    eval_batches = 0
    total_norm_t = 0.0
    total_norm_s = 0.0
    total_cos_full = 0.0
    total_cos_acoustic = 0.0
    total_cos_prosody = 0.0
    total_mse = 0.0
    max_coord_val = 0.0

    with torch.no_grad():
        for batch in val_loader:
            waves, texts, input_lengths, mels, output_lengths, ref_mels, tag_vectors, paths = batch
            ref_mels = ref_mels.to(device)
            tag_vectors = tag_vectors.to(device)

            ref_t = style_encoder(ref_mels.unsqueeze(1).float()).float()
            p_t = predictor_encoder(ref_mels.unsqueeze(1).float()).float()
            s_teacher = torch.cat([ref_t, p_t], dim=-1)

            s_student = tag_encoder(tag_vectors)

            norm_t = torch.norm(s_teacher, p=2, dim=-1).mean().item()
            norm_s = torch.norm(s_student, p=2, dim=-1).mean().item()
            cos_full = F.cosine_similarity(s_student, s_teacher, dim=-1).mean().item()
            cos_acoustic = F.cosine_similarity(s_student[:, :128], s_teacher[:, :128], dim=-1).mean().item()
            cos_prosody = F.cosine_similarity(s_student[:, 128:], s_teacher[:, 128:], dim=-1).mean().item()
            mse = F.mse_loss(s_student, s_teacher).item()

            total_norm_t += norm_t
            total_norm_s += norm_s
            total_cos_full += cos_full
            total_cos_acoustic += cos_acoustic
            total_cos_prosody += cos_prosody
            total_mse += mse
            max_coord_val = max(max_coord_val, s_student.abs().max().item())

            eval_batches += 1
            if eval_batches >= max_batches:
                break

    if eval_batches > 0:
        avg_norm_t = total_norm_t / eval_batches
        avg_norm_s = total_norm_s / eval_batches
        avg_cos_full = total_cos_full / eval_batches
        avg_cos_acoustic = total_cos_acoustic / eval_batches
        avg_cos_prosody = total_cos_prosody / eval_batches
        avg_mse = total_mse / eval_batches

        norm_status = "SAFE (Bounded ~0.5)" if avg_norm_s < 1.0 else "WARNING (Norm High)"
        trend_status = "IMPROVING (Aligning to Teacher)" if avg_cos_full > 0.1 else "INITIALIZING"

        step_label = os.path.basename(checkpoint_path)
        print("\n" + "═" * 84)
        print(f"  STAGE 1 [{step_label}] STYLE MANIFOLD EVALUATION (Student vs Official Teacher)")
        print("─" * 84)
        print(f"  Teacher Vector Norm : {avg_norm_t:.4f}         │ Student Vector Norm : {avg_norm_s:.4f} [{norm_status}]")
        print(f"  Overall Cosine Sim  : {avg_cos_full:+.4f}         │ Coordinate MSE Loss : {avg_mse:.6f}")
        print(f"  Acoustic Cosine Sim : {avg_cos_acoustic:+.4f}         │ Prosodic Cosine Sim : {avg_cos_prosody:+.4f}")
        print(f"  Max Student Coord   : {max_coord_val:.4f}         │ Training Status     : {trend_status}")
        print("═" * 84 + "\n")


if __name__ == "__main__":
    ckpt = sys.argv[1] if len(sys.argv) > 1 else None
    evaluate_checkpoint(checkpoint_path=ckpt)
