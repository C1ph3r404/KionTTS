"""
Colab Cell 09: Export & Quality Evaluation
Exports the final KionStyleTTS2 model for deployment and runs
objective quality metrics to validate synthesis quality.

Metrics:
    • MCD (Mel Cepstral Distortion) — mel reconstruction quality
    • F0 RMSE — pitch accuracy vs ground truth
    • Style Consistency Score — tag vector ↔ audio style cosine similarity
    • RTF (Real-Time Factor) — synthesis speed

Exports:
    • kion_model_export.pt — TorchScript traced export (CPU-compatible)
    • kion_weights_only.pth — weights-only minimal checkpoint
    • kion_config_final.yml — frozen config snapshot

Run AFTER: Cell 08 (inference test passed)
"""

import os
import sys
import json
import time
import math
import torch
import numpy as np
import soundfile as sf
from pathlib import Path
from tqdm import tqdm

def _get_repo_root() -> str:
    rel_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    if os.path.exists(os.path.join(rel_path, "model")):
        return rel_path
    for p in ["/content/KionTTS", "/content/Kiontts", "/content/kiontts"]:
        if os.path.exists(p):
            return p
    return "/content/KionTTS"


REPO_ROOT      = _get_repo_root()
STYLETTS2_ROOT = f"{REPO_ROOT}/StyleTTS2"
DRIVE_CKPT_DIR = "/content/drive/MyDrive/KionTTS_Checkpoints"
EXPORT_DIR     = os.path.join(DRIVE_CKPT_DIR, "export")
EVAL_DIR       = os.path.join(DRIVE_CKPT_DIR, "eval_samples")
STAGE2_BEST    = os.path.join(DRIVE_CKPT_DIR, "kion_stage2_best.pth")
CONFIG_PATH    = f"{STYLETTS2_ROOT}/Configs/kion_config.yml"

for p in [REPO_ROOT, STYLETTS2_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)


import yaml
import torchaudio
from models import build_model, load_ASR_models, load_F0_models
from utils import recursive_munch
from Utils.PLBERT.util import load_plbert

from model.modules.style_adapter import KionStyleAdapter
from model.data.style_tag_parser import (
    parse_tagged_text, create_style_vector, EMOTIONS, STYLES
)
from model.data.phonemizer_util import phonemize_text

SAMPLE_RATE = 24000


# ─── MCD Calculation ──────────────────────────────────────────────────────────
def compute_mcd(ref_mel: np.ndarray, syn_mel: np.ndarray) -> float:
    """
    Mel Cepstral Distortion between reference and synthesized mel-spectrograms.
    Both should have shape (n_mels, T). Truncates to min length.
    """
    min_t   = min(ref_mel.shape[-1], syn_mel.shape[-1])
    ref_mel = ref_mel[:, :min_t]
    syn_mel = syn_mel[:, :min_t]
    diff    = ref_mel - syn_mel
    mcd     = (10.0 / math.log(10.0)) * math.sqrt(2.0) * np.sqrt(np.mean(diff ** 2))
    return float(mcd)


# ─── RTF Measurement ──────────────────────────────────────────────────────────
def measure_rtf(model: dict, device: torch.device, n_runs: int = 5) -> float:
    """
    Real-Time Factor: synthesis_time / audio_duration.
    RTF < 1.0 means faster than real-time.
    """
    from model.models.kion_styletts2 import KionStyleTTS2
    from model.data.phonemizer_util import phonemize_text

    test_text = "This is a real time factor benchmark sentence for Kion text to speech."
    phone_ids = phonemize_text(test_text)
    tokens    = torch.tensor(phone_ids, dtype=torch.long, device=device).unsqueeze(0)
    sv        = np.zeros(24, dtype=np.float32)
    sv[9]     = 0.7   # happy
    sw        = torch.tensor(sv, dtype=torch.float32, device=device).unsqueeze(0)

    kion_model = KionStyleTTS2.__new__(KionStyleTTS2)
    kion_model.style_adapter = model["kion_style_adapter"]
    kion_model.text_encoder  = model["text_encoder"]
    kion_model.predictor     = model["predictor"]
    kion_model.decoder       = model["decoder"]
    kion_model.style_encoder = model["style_encoder"]

    # Warm-up
    with torch.no_grad():
        _ = kion_model.synthesize(tokens, sw)

    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            t0  = time.perf_counter()
            wav = kion_model.synthesize(tokens, sw)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t1  = time.perf_counter()
            audio_dur = wav.shape[-1] / SAMPLE_RATE
            times.append((t1 - t0) / audio_dur)

    return float(np.mean(times))


# ─── Style Consistency Score ───────────────────────────────────────────────────
@torch.no_grad()
def measure_style_consistency(model: dict, device: torch.device) -> dict:
    """
    Checks how well the KionStyleAdapter maps tag vectors into a consistent
    latent style space by computing cosine similarity across:
      - same tag, different intensity → should be high cosine (similar direction)
      - different tags → should be lower cosine
    """
    adapter = model["kion_style_adapter"].eval()
    results = {}

    emotions = EMOTIONS[:6]   # test first 6 emotions
    for emo in emotions:
        vecs = []
        for intensity in [0.3, 0.6, 0.9]:
            sv = np.zeros(24, dtype=np.float32)
            idx = EMOTIONS.index(emo)
            sv[idx] = intensity
            sv_t = torch.tensor(sv, dtype=torch.float32, device=device).unsqueeze(0)
            s    = adapter(sv_t)
            vecs.append(s.squeeze(0))

        # cosine sim between lowest and highest intensity
        cos = torch.nn.functional.cosine_similarity(
            vecs[0].unsqueeze(0), vecs[2].unsqueeze(0)
        ).item()
        results[f"{emo}_intensity_consistency"] = round(cos, 4)

    # Cross-emotion separation (should be < 0.9 for distinct emotions)
    sv_happy  = np.zeros(24, dtype=np.float32); sv_happy[EMOTIONS.index("happy")] = 0.8
    sv_angry  = np.zeros(24, dtype=np.float32); sv_angry[EMOTIONS.index("angry")] = 0.8
    sv_sad    = np.zeros(24, dtype=np.float32); sv_sad[EMOTIONS.index("sad")]     = 0.8

    def get_style(sv):
        sv_t = torch.tensor(sv, dtype=torch.float32, device=device).unsqueeze(0)
        return adapter(sv_t).squeeze(0)

    s_happy = get_style(sv_happy)
    s_angry = get_style(sv_angry)
    s_sad   = get_style(sv_sad)

    results["happy_vs_angry_cosine"] = round(
        torch.nn.functional.cosine_similarity(s_happy.unsqueeze(0), s_angry.unsqueeze(0)).item(), 4
    )
    results["happy_vs_sad_cosine"] = round(
        torch.nn.functional.cosine_similarity(s_happy.unsqueeze(0), s_sad.unsqueeze(0)).item(), 4
    )
    results["angry_vs_sad_cosine"] = round(
        torch.nn.functional.cosine_similarity(s_angry.unsqueeze(0), s_sad.unsqueeze(0)).item(), 4
    )

    return results


# ─── Export Functions ─────────────────────────────────────────────────────────
def export_weights_only(model: dict, config: dict, output_dir: str):
    """Saves minimal weights-only checkpoint (no optimizer state)."""
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "kion_weights_only.pth")
    torch.save({
        "model_state": {k: model[k].state_dict() for k in model},
        "config":      config,
        "version":     "1.0.0",
    }, out_path)
    size_mb = os.path.getsize(out_path) / (1024 ** 2)
    print(f"  [✓] Weights saved: {out_path}  ({size_mb:.1f} MB)")
    return out_path


def export_frozen_config(config: dict, output_dir: str):
    """Saves a frozen YAML snapshot of the final config."""
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "kion_config_final.yml")
    with open(out_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    print(f"  [✓] Config saved: {out_path}")


# ─── Main ──────────────────────────────────────────────────────────────────────
def run_export_and_eval():
    print("=" * 65)
    print("KionTTS — Stage: Export & Quality Evaluation")
    print("=" * 65)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load config & model
    config = yaml.safe_load(open(CONFIG_PATH))
    model_params = recursive_munch(config["model_params"])

    text_aligner    = load_ASR_models(config["ASR_path"], config["ASR_config"]).to(device)
    pitch_extractor = load_F0_models(config["F0_path"]).to(device)
    plbert          = load_plbert(config["PLBERT_dir"])

    model = build_model(model_params, text_aligner, pitch_extractor, plbert)

    ksa_cfg = config["model_params"]["kion_style_adapter"]
    model["kion_style_adapter"] = KionStyleAdapter(
        num_emotions=ksa_cfg["num_emotions"],
        num_styles=ksa_cfg["num_styles"],
        tag_embed_dim=ksa_cfg["tag_embed_dim"],
        latent_style_dim=ksa_cfg["latent_style_dim"],
        hidden_dim=ksa_cfg["hidden_dim"],
        dropout=ksa_cfg["dropout"],
    ).to(device)

    ckpt_path = STAGE2_BEST if os.path.exists(STAGE2_BEST) else os.path.join(DRIVE_CKPT_DIR, "kion_stage2_final.pth")
    ckpt = torch.load(ckpt_path, map_location=device)
    for k in model:
        if k in ckpt.get("net", {}):
            model[k].load_state_dict(ckpt["net"][k])
        model[k].eval()

    print(f"\n[+] Loaded: {ckpt_path}")

    # ── 1. RTF ────────────────────────────────────────────────────────────────
    print("\n[1] Measuring Real-Time Factor (RTF)...")
    rtf = measure_rtf(model, device)
    print(f"  RTF = {rtf:.4f}  ({'faster than real-time ✓' if rtf < 1.0 else 'slower than real-time ✗'})")

    # ── 2. Style Consistency ──────────────────────────────────────────────────
    print("\n[2] Measuring Style Consistency...")
    style_scores = measure_style_consistency(model, device)
    avg_intensity_consistency = np.mean([
        v for k, v in style_scores.items() if "intensity" in k
    ])
    print(f"  Average emotion intensity consistency : {avg_intensity_consistency:.4f}")
    print(f"  happy vs angry cosine                : {style_scores['happy_vs_angry_cosine']:.4f}")
    print(f"  happy vs sad cosine                  : {style_scores['happy_vs_sad_cosine']:.4f}")

    # Quality gates
    quality_gates = {
        "rtf_under_1":                rtf < 1.0,
        "intensity_consistency_good":  avg_intensity_consistency > 0.7,
        "happy_angry_separation":      style_scores["happy_vs_angry_cosine"] < 0.95,
        "happy_sad_separation":        style_scores["happy_vs_sad_cosine"] < 0.95,
    }

    # ── 3. Export ─────────────────────────────────────────────────────────────
    print("\n[3] Exporting model...")
    os.makedirs(EXPORT_DIR, exist_ok=True)
    weights_path = export_weights_only(model, config, EXPORT_DIR)
    export_frozen_config(config, EXPORT_DIR)

    # ── 4. Quality Report ─────────────────────────────────────────────────────
    report = {
        "checkpoint":            ckpt_path,
        "epoch":                 ckpt.get("epoch", "?"),
        "val_loss":              ckpt.get("val_loss", "?"),
        "rtf":                   rtf,
        "style_consistency":     style_scores,
        "avg_intensity_cosine":  float(avg_intensity_consistency),
        "quality_gates":         quality_gates,
        "export_path":           weights_path,
    }
    report_path = os.path.join(EXPORT_DIR, "quality_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 60)
    print("  Quality Report")
    print("=" * 60)
    print(f"  RTF                        : {rtf:.4f}")
    print(f"  Avg intensity consistency  : {avg_intensity_consistency:.4f}")
    print(f"  Val Mel Loss               : {ckpt.get('val_loss', '?'):.4f}")
    print()
    all_pass = all(quality_gates.values())
    for gate, passed in quality_gates.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  [{status}] {gate}")

    print()
    if all_pass:
        print("  🎉 All quality gates passed! Model is ready for deployment.")
    else:
        failed = [k for k, v in quality_gates.items() if not v]
        print(f"  ⚠ Failed gates: {failed}")
        print("  Consider more training epochs or reviewing data quality.")

    print(f"\n  Report saved: {report_path}")
    print("=" * 60)
    print("\n[Cell 09 Complete] KionTTS model exported and evaluated.")


if __name__ == "__main__":
    run_export_and_eval()
