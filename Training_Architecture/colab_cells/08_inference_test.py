"""
Colab Cell 08: Inference Test & Quality Evaluation
Loads the fully trained KionStyleTTS2 model and runs:
    1. Tag-conditioned synthesis for all 14 emotions × 3 intensity levels
    2. Style blend tests ([playful=0.7,teasing=0.5] etc.)
    3. Speed / pace variation tests
    4. Audio saved to Google Drive for listening

Run AFTER: Cell 07 (Stage 2 complete)
"""

import os
import sys
import json
import torch
import numpy as np
import soundfile as sf
from pathlib import Path
from tqdm import tqdm

# ─── Colab Paths ──────────────────────────────────────────────────────────────
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
EVAL_DIR       = os.path.join(DRIVE_CKPT_DIR, "eval_samples")
STAGE2_FINAL   = os.path.join(DRIVE_CKPT_DIR, "kion_stage2_final.pth")
STAGE2_BEST    = os.path.join(DRIVE_CKPT_DIR, "kion_stage2_best.pth")
CONFIG_PATH    = f"{STYLETTS2_ROOT}/Configs/kion_config.yml"

for p in [REPO_ROOT, STYLETTS2_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)


import yaml
from munch import Munch
from models import build_model, load_ASR_models, load_F0_models, load_checkpoint
from utils import recursive_munch
from Utils.PLBERT.util import load_plbert

from model.models.kion_styletts2 import KionStyleTTS2
from model.modules.style_adapter import KionStyleAdapter
from model.data.style_tag_parser import (
    parse_tagged_text,
    create_style_vector,
    EMOTIONS,
    STYLES,
)
from model.data.phonemizer_util import phonemize_text


SAMPLE_RATE = 24000


# ─── Load Model ───────────────────────────────────────────────────────────────
def load_kion_model(config_path: str, checkpoint_path: str, device: torch.device):
    """Load KionStyleTTS2 from a Stage 2 checkpoint."""
    config = yaml.safe_load(open(config_path))
    model_params = recursive_munch(config["model_params"])

    text_aligner    = load_ASR_models(config["ASR_path"], config["ASR_config"]).to(device)
    pitch_extractor = load_F0_models(config["F0_path"]).to(device)
    plbert          = load_plbert(config["PLBERT_dir"])

    model = build_model(model_params, text_aligner, pitch_extractor, plbert)

    # Add KionStyleAdapter
    ksa_cfg = config["model_params"]["kion_style_adapter"]
    model["kion_style_adapter"] = KionStyleAdapter(
        num_emotions=ksa_cfg["num_emotions"],
        num_styles=ksa_cfg["num_styles"],
        tag_embed_dim=ksa_cfg["tag_embed_dim"],
        latent_style_dim=ksa_cfg["latent_style_dim"],
        hidden_dim=ksa_cfg["hidden_dim"],
        dropout=ksa_cfg["dropout"],
    ).to(device)

    # Load weights
    ckpt = torch.load(checkpoint_path, map_location=device)
    for k in model:
        if k in ckpt["net"]:
            model[k].load_state_dict(ckpt["net"][k])
            model[k].eval()

    print(f"[+] KionStyleTTS2 loaded from: {checkpoint_path}")
    print(f"    Epoch: {ckpt.get('epoch', '?')} | Val loss: {ckpt.get('val_loss', '?'):.4f}")
    return model, config


# ─── Synthesis Function ───────────────────────────────────────────────────────
@torch.no_grad()
def synthesize(
    model: dict,
    text: str,
    style_tag: str,
    device: torch.device,
    pace: float = 1.0,
    seed: int = 42,
) -> np.ndarray:
    """
    Synthesize speech from text + style tag.

    Args:
        model       : KionStyleTTS2 model dict
        text        : Raw text (may contain inline tags)
        style_tag   : Additional style override, e.g. '[happy=0.8]'
        device      : torch device
        pace        : Speaking pace (1.0 = normal, 0.8 = slower, 1.2 = faster)
        seed        : Random seed for reproducibility

    Returns:
        waveform as float32 numpy array at 24kHz
    """
    torch.manual_seed(seed)

    # Parse text and tags
    full_input = f"{style_tag} {text}".strip()
    cleaned_text, emotions, styles_dict = parse_tagged_text(full_input)
    style_vec = create_style_vector(emotions, styles_dict)

    # Phonemise
    phoneme_ids = phonemize_text(cleaned_text)   # returns list[int]
    tokens      = torch.tensor(phoneme_ids, dtype=torch.long, device=device).unsqueeze(0)  # (1, T)

    # Style weight tensor
    style_weights = torch.tensor(style_vec, dtype=torch.float32, device=device).unsqueeze(0)  # (1, 24)

    # Build KionStyleTTS2 wrapper for synthesize()
    kion_model = KionStyleTTS2.__new__(KionStyleTTS2)
    kion_model.style_adapter  = model["kion_style_adapter"]
    kion_model.text_encoder   = model["text_encoder"]
    kion_model.predictor      = model["predictor"]
    kion_model.decoder        = model["decoder"]
    kion_model.style_encoder  = model["style_encoder"]

    waveform = kion_model.synthesize(tokens, style_weights, pace=pace)  # (1, T_samples)
    return waveform.squeeze(0).cpu().numpy()


# ─── Evaluation Suite ─────────────────────────────────────────────────────────
def run_evaluation(model, config, device):
    os.makedirs(EVAL_DIR, exist_ok=True)
    results = []

    # ── Test sentences ────────────────────────────────────────────────────────
    sentences = {
        "neutral":    "Hello, I am Kion. How can I help you today?",
        "short":      "Wait, what?",
        "long":       "I have been sitting here for the past two hours trying to figure out what went wrong, and I still have absolutely no idea.",
        "question":   "Are you seriously telling me that right now?",
        "exclamation": "I cannot believe this actually worked!",
        "pensive":    "I am not sure what to think about all of this.",
    }

    # ── 1. Sweep all emotions at 3 intensities ────────────────────────────────
    print("\n[1] Sweeping all emotions...")
    for emotion in tqdm(EMOTIONS, desc="Emotions"):
        for intensity in [0.4, 0.7, 1.0]:
            for sent_key, sent_text in [("neutral_sentence", sentences["neutral"]), ("question", sentences["question"])]:
                tag = f"[{emotion}={intensity:.1f}]"
                try:
                    wav = synthesize(model, sent_text, tag, device, pace=1.0)
                    fname = f"emotion_{emotion}_{intensity:.1f}_{sent_key}.wav"
                    fpath = os.path.join(EVAL_DIR, fname)
                    sf.write(fpath, wav, SAMPLE_RATE)
                    results.append({"type": "emotion", "tag": tag, "text": sent_text, "file": fname, "status": "ok"})
                except Exception as e:
                    results.append({"type": "emotion", "tag": tag, "text": sent_text, "file": None, "status": str(e)})

    # ── 2. Sweep all delivery styles ──────────────────────────────────────────
    print("\n[2] Sweeping delivery styles...")
    for style in tqdm(STYLES, desc="Styles"):
        for intensity in [0.5, 0.9]:
            tag = f"[{style}={intensity:.1f}]"
            try:
                wav = synthesize(model, sentences["neutral"], tag, device)
                fname = f"style_{style}_{intensity:.1f}.wav"
                sf.write(os.path.join(EVAL_DIR, fname), wav, SAMPLE_RATE)
                results.append({"type": "style", "tag": tag, "file": fname, "status": "ok"})
            except Exception as e:
                results.append({"type": "style", "tag": tag, "file": None, "status": str(e)})

    # ── 3. Blend tests ────────────────────────────────────────────────────────
    print("\n[3] Running blend tests...")
    blends = [
        ("[playful=0.7,teasing=0.5]", "You really thought I would not notice that?"),
        ("[sarcasm=0.8,deadpan=0.6]", "Oh, wonderful. Just what I needed."),
        ("[happy=0.9,excited=0.7]",   "I finally fixed it! This is incredible!"),
        ("[sad=0.6,calm=0.4]",        "I understand. I think it is for the best."),
        ("[curious=0.8,playful=0.5]", "Wait, seriously? How does that even work?"),
        ("[angry=0.7,frustrated=0.6]","I have told you this three times already."),
        ("[affectionate=0.8,soothing=0.7]", "It is okay. Everything is going to be fine."),
        ("[sarcasm=0.9,dramatic=0.5]","Oh yes, absolutely brilliant plan you have there."),
    ]
    for tag, text in tqdm(blends, desc="Blends"):
        try:
            wav   = synthesize(model, text, tag, device)
            fname = f"blend_{tag.replace('[','').replace(']','').replace(',','_').replace('=','')[:40]}.wav"
            sf.write(os.path.join(EVAL_DIR, fname), wav, SAMPLE_RATE)
            results.append({"type": "blend", "tag": tag, "text": text, "file": fname, "status": "ok"})
        except Exception as e:
            results.append({"type": "blend", "tag": tag, "file": None, "status": str(e)})

    # ── 4. Pace / speed tests ─────────────────────────────────────────────────
    print("\n[4] Running pace tests...")
    for pace in [0.7, 1.0, 1.3]:
        tag = "[calm=0.5]"
        text = "This is a pace test. I want to hear how fast or slow the speech sounds."
        try:
            wav   = synthesize(model, text, tag, device, pace=pace)
            fname = f"pace_{pace:.1f}.wav"
            sf.write(os.path.join(EVAL_DIR, fname), wav, SAMPLE_RATE)
            results.append({"type": "pace", "pace": pace, "file": fname, "status": "ok"})
        except Exception as e:
            results.append({"type": "pace", "pace": pace, "file": None, "status": str(e)})

    # ── 5. OOD sentences ──────────────────────────────────────────────────────
    print("\n[5] OOD (Out-of-Distribution) sentences...")
    ood_tests = [
        ("[curious=0.7]", "She sells seashells by the seashore."),
        ("[playful=0.8]", "How much wood would a woodchuck chuck?"),
        ("[dramatic=0.9]", "Peter Piper picked a peck of pickled peppers."),
    ]
    for tag, text in tqdm(ood_tests, desc="OOD"):
        try:
            wav   = synthesize(model, text, tag, device)
            fname = f"ood_{tag.replace('[','').replace(']','').replace('=','_')}.wav"
            sf.write(os.path.join(EVAL_DIR, fname), wav, SAMPLE_RATE)
            results.append({"type": "ood", "tag": tag, "text": text, "file": fname, "status": "ok"})
        except Exception as e:
            results.append({"type": "ood", "tag": tag, "file": None, "status": str(e)})

    # ── Save evaluation manifest ──────────────────────────────────────────────
    manifest_path = os.path.join(EVAL_DIR, "eval_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(results, f, indent=2)

    ok_count   = sum(1 for r in results if r["status"] == "ok")
    fail_count = len(results) - ok_count

    print("\n" + "=" * 60)
    print(f"  Evaluation complete!")
    print(f"  ✓ Succeeded : {ok_count} / {len(results)}")
    print(f"  ✗ Failed    : {fail_count} / {len(results)}")
    print(f"  Audio saved : {EVAL_DIR}")
    print(f"  Manifest    : {manifest_path}")
    print("=" * 60)

    if fail_count > 0:
        print("\n  Failed samples:")
        for r in results:
            if r["status"] != "ok":
                print(f"    [{r['type']}] {r.get('tag','?')} — {r['status']}")

    return results


# ─── Entry Point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Pick best available checkpoint
    ckpt_path = STAGE2_BEST if os.path.exists(STAGE2_BEST) else STAGE2_FINAL
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"No Stage 2 checkpoint found.\n"
            f"Expected: {STAGE2_BEST} or {STAGE2_FINAL}\n"
            "Run Cell 07 first."
        )

    model, config = load_kion_model(CONFIG_PATH, ckpt_path, device)
    results = run_evaluation(model, config, device)

    print("\n[Cell 08 Complete] Listen to samples in Google Drive > KionTTS_Checkpoints > eval_samples")
