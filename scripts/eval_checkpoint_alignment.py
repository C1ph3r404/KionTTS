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

if "__file__" in globals() and os.path.exists(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "kion_core"))):
    REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
elif os.path.exists("/content/KionTTS"):
    REPO_ROOT = "/content/KionTTS"
else:
    REPO_ROOT = os.path.abspath(os.getcwd())

sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "StyleTTS2"))

from kion_core.dataset import build_kion_dataloader
from kion_core.tag_style_encoder import KionTagStyleEncoder
from models import build_model, StyleEncoder, load_ASR_models, load_F0_models
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
    max_batches: int = 15,
    acoustic_scale: float = 1.0,
    prosodic_scale: float = 1.0,
):
    # 1. Discover manifests & paths
    if not os.path.exists(val_manifest):
        for cand_manifest, cand_root in [
            (os.path.join(REPO_ROOT, "DataSet/val_manifest.json"), os.path.join(REPO_ROOT, "DataSet")),
            (os.path.join(REPO_ROOT, "dataset/val_manifest.json"), os.path.join(REPO_ROOT, "dataset")),
            ("/kaggle/working/KionTTS/DataSet/val_manifest.json", "/kaggle/working/KionTTS/DataSet"),
            ("/content/dataset/val_manifest.json", "/content/dataset"),
        ]:
            if os.path.exists(cand_manifest):
                val_manifest = cand_manifest
                data_root = cand_root
                break

    # Locate checkpoint
    if not checkpoint_path:
        check_dir = os.path.join(REPO_ROOT, checkpoint_dir) if not os.path.isabs(checkpoint_dir) else checkpoint_dir
        ckpts = sorted(glob.glob(os.path.join(check_dir, "*.pth")) + glob.glob(os.path.join(check_dir, "*.pt")))
        if not ckpts:
            print(f"[!] No checkpoints found in {check_dir} yet.")
            return None
        checkpoint_path = ckpts[-1]

    print(f"[*] Evaluating style alignment for checkpoint: {checkpoint_path}")

    # 2. Build Teacher models (frozen)
    if not os.path.isabs(config_path):
        config_path = os.path.join(REPO_ROOT, config_path)
    if not os.path.isabs(pretrained_ckpt):
        pretrained_ckpt = os.path.join(REPO_ROOT, pretrained_ckpt)

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    args = recursive_munch(config["model_params"])

    # Build Official Pretrained Teacher
    official_style_encoder = StyleEncoder(dim_in=args.dim_in, style_dim=args.style_dim, max_conv_dim=args.hidden_dim).to(device).eval()
    official_predictor_encoder = StyleEncoder(dim_in=args.dim_in, style_dim=args.style_dim, max_conv_dim=args.hidden_dim).to(device).eval()

    if os.path.exists(pretrained_ckpt):
        libri_state = torch.load(pretrained_ckpt, map_location="cpu", weights_only=False)
        libri_params = libri_state.get("net", libri_state)
        if "style_encoder" in libri_params:
            clean_se = {k.replace("module.", ""): v for k, v in libri_params["style_encoder"].items()}
            official_style_encoder.load_state_dict(clean_se, strict=False)
        if "predictor_encoder" in libri_params:
            clean_pe = {k.replace("module.", ""): v for k, v in libri_params["predictor_encoder"].items()}
            official_predictor_encoder.load_state_dict(clean_pe, strict=False)
        else:
            official_predictor_encoder.load_state_dict(official_style_encoder.state_dict())
        print(f"[✓] Loaded Official Pretrained Teacher backbone from: {pretrained_ckpt}")
    else:
        print(f"[!] Warning: Pretrained LibriTTS backbone not found at: {pretrained_ckpt}")

    # 3. Load Checkpoint and Student model (tag_encoder)
    ckpt_data = torch.load(checkpoint_path, map_location=device, weights_only=False)

    tag_encoder = KionTagStyleEncoder(acoustic_scale=acoustic_scale, prosodic_scale=prosodic_scale).to(device).eval()
    if "net" in ckpt_data and "tag_encoder" in ckpt_data["net"]:
        clean_tag = {k.replace("module.", ""): v for k, v in ckpt_data["net"]["tag_encoder"].items()}
        tag_encoder.load_state_dict(clean_tag)
        print("[✓] Loaded trained tag_encoder from ckpt['net']['tag_encoder']")
    elif "tag_encoder" in ckpt_data:
        clean_tag = {k.replace("module.", ""): v for k, v in ckpt_data["tag_encoder"].items()}
        tag_encoder.load_state_dict(clean_tag)
        print("[✓] Loaded trained tag_encoder from ckpt['tag_encoder']")
    else:
        print("[!] Warning: Could not locate tag_encoder keys in checkpoint")

    # Check if checkpoint also contains the training run's internal style_encoder
    has_internal_teacher = "net" in ckpt_data and "style_encoder" in ckpt_data["net"]
    internal_style_encoder = None
    if has_internal_teacher:
        internal_style_encoder = StyleEncoder(dim_in=args.dim_in, style_dim=args.style_dim, max_conv_dim=args.hidden_dim).to(device).eval()
        clean_ise = {k.replace("module.", ""): v for k, v in ckpt_data["net"]["style_encoder"].items()}
        internal_style_encoder.load_state_dict(clean_ise, strict=False)
        print("[✓] Loaded Checkpoint internal teacher style_encoder")

    # 4. DataLoader
    val_loader = build_kion_dataloader(
        manifest_path=val_manifest,
        root_dir=data_root,
        batch_size=4,
        validation=True,
        num_workers=0,
    )

    # Metrics
    eval_batches = 0
    total_norm_libri = 0.0
    total_norm_student = 0.0
    total_norm_s_ac = 0.0
    total_norm_s_pr = 0.0
    total_norm_t_ac = 0.0
    total_norm_t_pr = 0.0

    total_cos_libri = 0.0
    total_cos_acoustic_libri = 0.0
    total_cos_prosody_libri = 0.0
    total_mse_libri = 0.0

    total_norm_internal = 0.0
    total_cos_internal = 0.0
    total_mse_internal = 0.0

    max_coord_val = 0.0

    with torch.no_grad():
        for batch in val_loader:
            waves, texts, input_lengths, mels, output_lengths, ref_mels, tag_vectors, paths = batch
            ref_mels = ref_mels.to(device)
            tag_vectors = tag_vectors.to(device)

            # Student output
            s_student = tag_encoder(tag_vectors)  # (B, 256)

            # Official LibriTTS Teacher output
            ref_t_libri = official_style_encoder(ref_mels.unsqueeze(1).float()).float()
            p_t_libri = official_predictor_encoder(ref_mels.unsqueeze(1).float()).float()
            s_teacher_libri = torch.cat([ref_t_libri, p_t_libri], dim=-1)

            # Norms breakdown
            norm_l = torch.norm(s_teacher_libri, p=2, dim=-1).mean().item()
            norm_s = torch.norm(s_student, p=2, dim=-1).mean().item()
            norm_s_ac = torch.norm(s_student[:, :128], p=2, dim=-1).mean().item()
            norm_s_pr = torch.norm(s_student[:, 128:], p=2, dim=-1).mean().item()
            norm_t_ac = torch.norm(ref_t_libri, p=2, dim=-1).mean().item()
            norm_t_pr = torch.norm(p_t_libri, p=2, dim=-1).mean().item()

            cos_l = F.cosine_similarity(s_student, s_teacher_libri, dim=-1).mean().item()
            cos_ac_l = F.cosine_similarity(s_student[:, :128], s_teacher_libri[:, :128], dim=-1).mean().item()
            cos_pr_l = F.cosine_similarity(s_student[:, 128:], s_teacher_libri[:, 128:], dim=-1).mean().item()
            mse_l = F.mse_loss(s_student, s_teacher_libri).item()

            total_norm_libri += norm_l
            total_norm_student += norm_s
            total_norm_s_ac += norm_s_ac
            total_norm_s_pr += norm_s_pr
            total_norm_t_ac += norm_t_ac
            total_norm_t_pr += norm_t_pr

            total_cos_libri += cos_l
            total_cos_acoustic_libri += cos_ac_l
            total_cos_prosody_libri += cos_pr_l
            total_mse_libri += mse_l

            # Metrics vs Internal Training Teacher (if available)
            if internal_style_encoder is not None:
                ref_t_int = internal_style_encoder(ref_mels.unsqueeze(1).float()).float()
                s_teacher_int = torch.cat([ref_t_int, ref_t_int], dim=-1)
                norm_int = torch.norm(s_teacher_int, p=2, dim=-1).mean().item()
                cos_int = F.cosine_similarity(s_student, s_teacher_int, dim=-1).mean().item()
                mse_int = F.mse_loss(s_student, s_teacher_int).item()

                total_norm_internal += norm_int
                total_cos_internal += cos_int
                total_mse_internal += mse_int

            max_coord_val = max(max_coord_val, s_student.abs().max().item())

            eval_batches += 1
            if eval_batches >= max_batches:
                break

    if eval_batches > 0:
        avg_norm_l = total_norm_libri / eval_batches
        avg_norm_s = total_norm_student / eval_batches
        avg_norm_s_ac = total_norm_s_ac / eval_batches
        avg_norm_s_pr = total_norm_s_pr / eval_batches
        avg_norm_t_ac = total_norm_t_ac / eval_batches
        avg_norm_t_pr = total_norm_t_pr / eval_batches

        avg_cos_l = total_cos_libri / eval_batches
        avg_cos_ac_l = total_cos_acoustic_libri / eval_batches
        avg_cos_pr_l = total_cos_prosody_libri / eval_batches
        avg_mse_l = total_mse_libri / eval_batches

        step_label = os.path.basename(checkpoint_path)
        is_bridged = (tag_encoder.acoustic_scale != 1.0 or tag_encoder.prosodic_scale != 1.0)

        print("\n" + "═" * 84)
        print(f"  STAGE 1 [{step_label}] STYLE MANIFOLD EVALUATION REPORT")
        print("═" * 84)
        print(f"  Student Vector L2 Norm  : {avg_norm_s:.4f}  (Acoustic={avg_norm_s_ac:.4f} | Prosodic={avg_norm_s_pr:.4f})")
        print(f"  Max Student Coordinate  : {max_coord_val:.4f}")
        if is_bridged:
            print(f"  Active Scale Multipliers: Acoustic={tag_encoder.acoustic_scale:.2f}x | Prosodic={tag_encoder.prosodic_scale:.2f}x")
        print("─" * 84)
        print("  COMPARISON A: STUDENT vs OFFICIAL PRETRAINED TEACHER (LibriTTS Backbone)")
        print(f"    • Official Teacher Norm : {avg_norm_l:.4f}  (Acoustic={avg_norm_t_ac:.4f} | Prosodic={avg_norm_t_pr:.4f})")
        print(f"    • Overall Cosine Sim    : {avg_cos_l:+.4f}")
        print(f"    • Acoustic Cosine Sim   : {avg_cos_ac_l:+.4f} (Acoustic 128D)")
        print(f"    • Prosodic Cosine Sim   : {avg_cos_pr_l:+.4f} (Prosodic 128D)")
        print(f"    • Manifold MSE Loss     : {avg_mse_l:.6f}")
        if not is_bridged:
            rec_ac = avg_norm_t_ac / max(1e-4, avg_norm_s_ac)
            rec_pr = avg_norm_t_pr / max(1e-4, avg_norm_s_pr)
            print(f"    • Recommended Bridge    : --acoustic_scale {rec_ac:.2f} --prosodic_scale {rec_pr:.2f}")

        if internal_style_encoder is not None:
            avg_norm_int = total_norm_internal / eval_batches
            avg_cos_int = total_cos_internal / eval_batches
            avg_mse_int = total_mse_internal / eval_batches
            print("─" * 84)
            print("  COMPARISON B: STUDENT vs TRAINING RUN'S INTERNAL TEACHER")
            print(f"    • Internal Teacher Norm : {avg_norm_int:.4f}")
            print(f"    • Overall Cosine Sim    : {avg_cos_int:+.4f}")
            print(f"    • Internal MSE Loss     : {avg_mse_int:.6f}")

        print("═" * 84 + "\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Audit StyleTTS2 Student vs Teacher alignment.")
    parser.add_argument("checkpoint", nargs="?", default=None, help="Path to checkpoint .pth")
    parser.add_argument("--acoustic_scale", type=float, default=1.0, help="Acoustic scale multiplier")
    parser.add_argument("--prosodic_scale", type=float, default=1.0, help="Prosodic scale multiplier")
    cli_args = parser.parse_args()

    evaluate_checkpoint(
        checkpoint_path=cli_args.checkpoint,
        acoustic_scale=cli_args.acoustic_scale,
        prosodic_scale=cli_args.prosodic_scale,
    )
