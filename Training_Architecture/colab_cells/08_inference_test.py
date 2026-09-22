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

# ─── PyTorch 2.6+ Compatibility ───────────────────────────────────────────────
_orig_torch_load = torch.load
def _compat_torch_load(*args, **kwargs):
    if "weights_only" not in kwargs:
        try:
            return _orig_torch_load(*args, weights_only=False, **kwargs)
        except TypeError:
            return _orig_torch_load(*args, **kwargs)
    return _orig_torch_load(*args, **kwargs)
torch.load = _compat_torch_load

# ─── Colab Paths ──────────────────────────────────────────────────────────────
def _get_repo_root() -> str:
    rel_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    if os.path.exists(os.path.join(rel_path, "model")):
        return rel_path
    for p in ["/content/KionTTS", "/content/Kiontts", "/content/kiontts"]:
        if os.path.exists(p):
            return p
    return "/content/KionTTS"


def _get_ckpt_dir() -> str:
    for p in ["/content/drive/MyDrive/KionTTS_Checkpoints", "/kaggle/working/checkpoints", "/kaggle/working", "/content/checkpoints"]:
        if os.path.exists(p):
            return p
    return "/content/drive/MyDrive/KionTTS_Checkpoints" if os.path.exists("/content") else "/kaggle/working/checkpoints"


REPO_ROOT      = _get_repo_root()
STYLETTS2_ROOT = f"{REPO_ROOT}/StyleTTS2"
DRIVE_CKPT_DIR = _get_ckpt_dir()
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
try:
    from utils import recursive_munch
except Exception:
    def recursive_munch(d):
        if isinstance(d, dict):
            return Munch((k, recursive_munch(v)) for k, v in d.items())
        elif isinstance(d, list):
            return [recursive_munch(v) for v in d]
        return d
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


def _resolve_config_paths(cfg, base_dir=STYLETTS2_ROOT):
    """Ensure relative paths to pretrained utility models resolve to STYLETTS2_ROOT or REPO_ROOT."""
    def _resolve(p):
        if not p or not isinstance(p, str):
            return p
        if os.path.isabs(p) and os.path.exists(p):
            return p
        if os.path.exists(p):
            return os.path.abspath(p)
        for root in [base_dir, REPO_ROOT, "/content/KionTTS/StyleTTS2", "/content/Kiontts/StyleTTS2", "/content/kiontts/StyleTTS2"]:
            cand = os.path.normpath(os.path.join(root, p))
            if os.path.exists(cand):
                return cand
        return p

    for key in ["ASR_config", "ASR_path", "F0_path", "PLBERT_dir", "pretrained_model"]:
        if key in cfg and cfg[key]:
            cfg[key] = _resolve(cfg[key])
    return cfg


def _download_file(url: str, dest_path: str, desc: str):
    """Download a file with progress reporting and curl fallback."""
    os.makedirs(os.path.dirname(os.path.abspath(dest_path)), exist_ok=True)
    print(f"[*] Downloading {desc} to {dest_path}...")
    try:
        import urllib.request
        urllib.request.urlretrieve(url, dest_path)
        print(f"[+] Downloaded: {dest_path}")
    except Exception as e:
        import subprocess
        subprocess.run(["curl", "-L", "-o", dest_path, url], check=True)
        print(f"[+] Downloaded via curl: {dest_path}")


def _ensure_pretrained_assets(cfg):
    """Auto-download required pretrained utility weights if missing on disk."""
    asr_path = cfg.get("ASR_path")
    if asr_path and not os.path.exists(asr_path):
        _download_file("https://github.com/yl4579/StyleTTS2/raw/main/Utils/ASR/epoch_00080.pth", asr_path, "ASR aligner")
    f0_path = cfg.get("F0_path")
    if f0_path and not os.path.exists(f0_path):
        _download_file("https://github.com/yl4579/StyleTTS2/raw/main/Utils/JDC/bst.t7", f0_path, "F0 pitch model")
    plbert_dir = cfg.get("PLBERT_dir")
    if plbert_dir:
        plbert_ckpt = os.path.join(plbert_dir, "step_1000000.t7")
        if not os.path.exists(plbert_ckpt):
            _download_file("https://github.com/yl4579/StyleTTS2/raw/main/Utils/PLBERT/step_1000000.t7", plbert_ckpt, "PL-BERT")


# ─── Load Model ───────────────────────────────────────────────────────────────
def load_kion_model(config_path: str, checkpoint_path: str, device: torch.device):
    """Load KionStyleTTS2 from a Stage 2 checkpoint."""
    config = yaml.safe_load(open(config_path))
    config = _resolve_config_paths(config)
    _ensure_pretrained_assets(config)
    model_params = recursive_munch(config["model_params"])

    text_aligner    = load_ASR_models(config["ASR_path"], config["ASR_config"]).to(device)
    pitch_extractor = load_F0_models(config["F0_path"]).to(device)
    plbert          = load_plbert(config["PLBERT_dir"])

    model = build_model(model_params, text_aligner, pitch_extractor, plbert)

    # Add KionStyleAdapter
    ksa_cfg = config["model_params"].get("kion_style_adapter", {
        "num_emotions": 14,
        "num_styles": 10,
        "tag_embed_dim": 64,
        "latent_style_dim": config["model_params"].get("style_dim", 128),
        "hidden_dim": config["model_params"].get("hidden_dim", 512) // 2,
        "dropout": 0.1,
    })
    model["kion_style_adapter"] = KionStyleAdapter(
        num_emotions=ksa_cfg.get("num_emotions", 14),
        num_styles=ksa_cfg.get("num_styles", 10),
        tag_embed_dim=ksa_cfg.get("tag_embed_dim", 64),
        latent_style_dim=ksa_cfg.get("latent_style_dim", 128),
        hidden_dim=ksa_cfg.get("hidden_dim", 256),
        dropout=ksa_cfg.get("dropout", 0.1),
    ).to(device)

    # Load weights, force float32 (AMP checkpoints save as fp16 which overflows at inference)
    ckpt = torch.load(checkpoint_path, map_location=device)
    for k in model:
        if k in ckpt["net"]:
            model[k].load_state_dict(ckpt["net"][k])
        model[k].float().to(device)
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

    # ── Parse text and style tags ──────────────────────────────────────────────
    full_input = f"{style_tag} {text}".strip()
    cleaned_text, emotions, styles_dict = parse_tagged_text(full_input)
    style_vec = create_style_vector(emotions, styles_dict)

    # ── Phonemise ──────────────────────────────────────────────────────────────
    phoneme_ids  = phonemize_text(cleaned_text)
    texts        = torch.tensor(phoneme_ids, dtype=torch.long, device=device).unsqueeze(0)  # (1, T)
    input_lengths = torch.tensor([texts.size(1)], dtype=torch.long, device=device)
    text_mask     = torch.zeros(1, texts.size(1), dtype=torch.bool, device=device)

    # ── Style weight tensor for KionStyleAdapter ───────────────────────────────
    style_weights = torch.tensor(style_vec, dtype=torch.float32, device=device).unsqueeze(0)  # (1, N_tags)

    # Shorthand refs — all already on device and in float32 from load_kion_model
    text_encoder      = model["text_encoder"]
    bert              = model["bert"]
    bert_encoder      = model["bert_encoder"]
    predictor         = model["predictor"]
    predictor_encoder = model["predictor_encoder"]
    style_encoder     = model["style_encoder"]
    kion_adapter      = model["kion_style_adapter"]
    decoder           = model["decoder"]

    # ── 1. Text encoding (matches training line: t_en = text_encoder(texts, ...)) ──
    t_en = text_encoder(texts, input_lengths, text_mask)  # (1, hidden, T_text)

    # ── 2. BERT prosody encoding (matches training: bert_dur → bert_encoder → d_en) ──
    bert_dur = bert(texts, attention_mask=(~text_mask).int())
    d_en     = bert_encoder(bert_dur).transpose(-1, -2)   # (1, hidden, T_text)

    # ── 3. Style vector for duration predictor ─────────────────────────────────
    # During training s_dur comes from predictor_encoder(mel). At inference we
    # use KionStyleAdapter to produce a style vector of the same dimension.
    s_dur   = kion_adapter(style_weights)   # (1, style_dim)

    # ── 4. Predict durations via predictor ────────────────────────────────────
    # predictor(d_en, s_dur, input_lengths, attn=None, text_mask)
    # returns (d, p) where d contains per-token duration logits
    d, _ = predictor(d_en, s_dur, input_lengths, None, text_mask)
    # d: (1, T_text, max_dur_bins) — sum sigmoid for expected duration per token
    pred_dur = torch.sigmoid(d).sum(axis=-1) / pace   # (1, T_text)

    # ── 5. Length regulation (build alignment matrix) ─────────────────────────
    pred_dur  = torch.round(pred_dur).clamp(min=1).long()
    T_text    = texts.size(1)
    total_frames = int(pred_dur.sum().item())
    if total_frames % 2 != 0:
        pred_dur[0, -1] += 1
        total_frames    += 1

    aln_hard = torch.zeros(T_text, total_frames, device=device)
    c = 0
    for i in range(T_text):
        dur_i = int(pred_dur[0, i].item())
        if c + dur_i <= total_frames:
            aln_hard[i, c:c + dur_i] = 1
        c += dur_i
    aln = aln_hard.unsqueeze(0)   # (1, T_text, T_frames)

    # ── 6. Frame-level acoustic + prosody features ────────────────────────────
    # asr: (1, hidden, T_frames)  —  matches training: asr = t_en @ aln
    asr = t_en @ aln          # (1, hidden, T_text) @ (1, T_text, T_frames)
    # p_en: (1, hidden, T_frames) — used for F0/N prediction
    p_en = (d_en @ aln)       # same shape

    # ── 7. F0 and Energy prediction ───────────────────────────────────────────
    F0_pred, N_pred = predictor.F0Ntrain(p_en, s_dur)

    # ── 8. Decoder (iSTFTNet vocoder) ─────────────────────────────────────────
    waveform = decoder(asr, F0_pred, N_pred, s_dur)
    wav = waveform.squeeze().cpu().float().numpy()

    # Sanitise any residual NaN/Inf
    wav = np.nan_to_num(wav, nan=0.0, posinf=0.0, neginf=0.0)

    # Diagnostics
    peak = float(np.abs(wav).max()) if wav.size > 0 else 0.0
    rms  = float(np.sqrt(np.mean(wav**2))) if wav.size > 0 else 0.0
    print(f"     [dbg] wav shape={wav.shape}, peak={peak:.6f}, rms={rms:.6f}")

    # Normalise to [-0.95, 0.95]
    if peak > 1e-6:
        wav = wav / peak * 0.95
    else:
        print("     [!] WARNING: near-silent output — check model/checkpoint")

    return wav


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
    ckpt_path = None
    for cand in [STAGE2_BEST, STAGE2_FINAL, os.path.join(DRIVE_CKPT_DIR, "kion_stage2_latest.pth")]:
        if os.path.exists(cand):
            ckpt_path = cand
            break
    if not ckpt_path:
        import glob
        steps = glob.glob(os.path.join(DRIVE_CKPT_DIR, "checkpoint-*.pth"))
        if steps:
            ckpt_path = sorted(steps, key=os.path.getmtime, reverse=True)[0]

    if not ckpt_path or not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"No Stage 2 checkpoint found in {DRIVE_CKPT_DIR}.\n"
            f"Expected: {STAGE2_BEST}, {STAGE2_FINAL}, or step checkpoints.\n"
            "Run Stage 2 training first or download checkpoint from Hugging Face."
        )

    model, config = load_kion_model(CONFIG_PATH, ckpt_path, device)
    results = run_evaluation(model, config, device)

    print("\n[Cell 08 Complete] Listen to samples in Google Drive > KionTTS_Checkpoints > eval_samples")
