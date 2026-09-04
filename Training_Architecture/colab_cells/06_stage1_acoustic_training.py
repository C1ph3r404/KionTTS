"""
Colab Cell 06: Stage 1 — Acoustic Foundation Training
Trains the KionStyleTTS2 acoustic backbone using the StyleTTS2 train_first.py logic:
    • TextEncoder (phoneme CNN + BiLSTM)
    • iSTFTNet Decoder (neural vocoder)
    • StyleEncoder (extracts acoustic style from reference mel)
    • MultiPeriodDiscriminator + MultiResSpecDiscriminator (GAN)
    • WavLM Speech Language Model discriminator (SLM)

Training Phases:
    Phase 1 (Epochs 0–TMA): mel reconstruction only
    Phase 2 (Epochs TMA–end): + GAN + monotonic alignment + S2S + SLM

Expected Duration: ~100–120 epochs
  T4 GPU (15GB): ~10–14 hours
  A100 (40GB):   ~3–5 hours

Run AFTER: Cell 05 (config builder)
Run BEFORE: Cell 07 (Stage 2)
"""

import os
import sys

# Prevent CUDA memory fragmentation on 16GB GPUs (like Colab T4)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import time
import random
import shutil
import logging
import warnings
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# ─── PyTorch 2.6+ Compatibility ───────────────────────────────────────────────
# PyTorch 2.6 changed torch.load default to weights_only=True, which breaks legacy StyleTTS2 checkpoints.
_orig_torch_load = torch.load
def _compat_torch_load(*args, **kwargs):
    if "weights_only" not in kwargs:
        try:
            return _orig_torch_load(*args, weights_only=False, **kwargs)
        except TypeError:
            return _orig_torch_load(*args, **kwargs)
    return _orig_torch_load(*args, **kwargs)
torch.load = _compat_torch_load

warnings.simplefilter("ignore")

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
CONFIG_PATH    = f"{STYLETTS2_ROOT}/Configs/kion_config.yml"

# Add StyleTTS2 to Python path
for p in [REPO_ROOT, STYLETTS2_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)



# ─── Ensure Dependencies (self-healing for active Colab runtimes) ────────────
for _pkg, _mod in [("munch", "munch"), ("einops-exts", "einops_exts"), ("einops", "einops"), ("pydub", "pydub"), ("nltk", "nltk")]:
    try:
        __import__(_mod)
    except ImportError:
        import subprocess
        print(f"[*] Installing missing package: {_pkg}...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", _pkg], check=False)

try:
    import monotonic_align
except ImportError:
    import subprocess
    print("[*] Installing missing package: monotonic_align from git...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "cython"], check=False)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "git+https://github.com/resemble-ai/monotonic_align.git"], check=False)

# ─── Imports (after path setup) ───────────────────────────────────────────────
import yaml
from munch import Munch
from accelerate import Accelerator
from accelerate.utils import LoggerType
from accelerate import DistributedDataParallelKwargs

from models import (
    build_model,
    load_ASR_models,
    load_F0_models,
    load_checkpoint,
)
from meldataset import build_dataloader
from losses import (
    MultiResolutionSTFTLoss,
    GeneratorLoss,
    DiscriminatorLoss,
    WavLMLoss,
)
from utils import (
    get_data_path_list,
    length_to_mask,
    log_norm,
    get_image,
    recursive_munch,
    log_print,
    mask_from_lens,
    maximum_path,
)
from optimizers import build_optimizer

# KionStyleAdapter for stage 1 style conditioning
from model.modules.style_adapter import KionStyleAdapter

log = logging.getLogger("KionStage1")
log.setLevel(logging.DEBUG)
_h = logging.StreamHandler()
_h.setLevel(logging.DEBUG)
log.addHandler(_h)


# ─── Checkpoint helpers ───────────────────────────────────────────────────────
def _is_valid_checkpoint(path: str) -> bool:
    """Verifies that a checkpoint exists, is non-empty, and can be unpickled cleanly."""
    if not path or not os.path.exists(path):
        return False
    try:
        if os.path.getsize(path) < 1024 * 1024:  # Must be at least 1MB
            return False
        state = torch.load(path, map_location="cpu")
        return isinstance(state, dict) and "net" in state
    except Exception as e:
        print(f"  [!] Integrity check failed for '{os.path.basename(path)}': {e}")
        return False


def _save_to_drive(state: dict, epoch: int, step: int = None, is_best: bool = False):
    os.makedirs(DRIVE_CKPT_DIR, exist_ok=True)
    if step is not None:
        # Fixed rolling slots A & B (overwritten in-place, zero files deleted into Drive Trash!)
        slot = "A" if (step // 1000) % 2 == 0 else "B"
        ckpt_path = os.path.join(DRIVE_CKPT_DIR, f"kion_stage1_step_slot_{slot}.pth")
    else:
        # Fixed rolling slots A & B for epochs (overwritten in-place)
        slot = "A" if epoch % 2 == 0 else "B"
        ckpt_path = os.path.join(DRIVE_CKPT_DIR, f"kion_stage1_epoch_slot_{slot}.pth")

    # Safe write: save to local /tmp first, verify integrity, then copy to Drive
    tmp_path = f"/tmp/kion_stage1_tmp_{int(time.time())}.pth"
    try:
        torch.save(state, tmp_path)
        if _is_valid_checkpoint(tmp_path):
            shutil.copyfile(tmp_path, ckpt_path)
            if step is not None:
                print(f"\n  [✓] Verified & saved step checkpoint {step} (Slot {slot}) → {ckpt_path}")
            else:
                print(f"\n  [✓] Verified & saved epoch checkpoint (Epoch {epoch}, Slot {slot}) → {ckpt_path}")
            # Update pointer file only on successful verified save
            meta_path = os.path.join(DRIVE_CKPT_DIR, "latest_stage1_checkpoint.txt")
            try:
                with open(meta_path, "w") as f:
                    f.write(f"{ckpt_path}\n")
            except Exception:
                pass
        else:
            print(f"  [!] Local save integrity check failed. Skipping copy to Drive.")
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass

    if is_best and os.path.exists(ckpt_path):
        best_path = os.path.join(DRIVE_CKPT_DIR, "kion_stage1_best.pth")
        shutil.copy2(ckpt_path, best_path)
        print(f"  [★] New best! Copied → {best_path}")


def _find_latest_drive_checkpoint() -> str | None:
    import glob
    candidates = []

    # 1. Check pointer file
    meta_path = os.path.join(DRIVE_CKPT_DIR, "latest_stage1_checkpoint.txt")
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r") as f:
                target = f.readline().strip()
                if target and os.path.exists(target):
                    candidates.append(target)
        except Exception:
            pass

    # 2. Check rolling slots and legacy numbered files (sorted newest first)
    patterns = [
        "kion_stage1_step_slot_*.pth",
        "kion_stage1_epoch_slot_*.pth",
        "kion_stage1_step_*.pth",
        "kion_stage1_epoch_*.pth",
    ]
    found = []
    for pat in patterns:
        found.extend(glob.glob(os.path.join(DRIVE_CKPT_DIR, pat)))

    valid_found = [c for c in found if not c.endswith("best.pth") and not c.endswith("final.pth")]
    valid_found.sort(key=os.path.getmtime, reverse=True)
    for c in valid_found:
        if c not in candidates:
            candidates.append(c)

    best_path = os.path.join(DRIVE_CKPT_DIR, "kion_stage1_best.pth")
    if os.path.exists(best_path) and best_path not in candidates:
        candidates.append(best_path)

    # 3. Test candidates for corruption; return the newest uncorrupted one
    for cand in candidates:
        print(f"  Verifying integrity of candidate: {os.path.basename(cand)}...")
        if _is_valid_checkpoint(cand):
            print(f"  [✓] Integrity confirmed: {os.path.basename(cand)}")
            return cand
        else:
            print(f"  [!] Checkpoint {os.path.basename(cand)} is damaged. Falling back to previous slot...")

    return None


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
            resolved = _resolve(cfg[key])
            if resolved != cfg[key]:
                print(f"[*] Resolved {key}: '{cfg[key]}' -> '{resolved}'")
            cfg[key] = resolved

    if "data_params" in cfg and isinstance(cfg["data_params"], dict):
        dp = cfg["data_params"]
        for dkey in ["train_data", "val_data", "OOD_data"]:
            if dkey in dp and dp[dkey]:
                dp[dkey] = _resolve(dp[dkey])
    return cfg


def _download_file(url: str, dest_path: str, desc: str):
    """Download a file with progress reporting and curl fallback."""
    os.makedirs(os.path.dirname(os.path.abspath(dest_path)), exist_ok=True)
    print(f"[*] Downloading {desc} to {dest_path}...")
    try:
        import urllib.request
        def _reporthook(blocknum, blocksize, totalsize):
            if totalsize > 0:
                percent = min(100, blocknum * blocksize * 100 / totalsize)
                sys.stdout.write(f"\r    Progress: {percent:.1f}% ({blocknum*blocksize/(1024*1024):.1f}/{totalsize/(1024*1024):.1f} MB)")
                sys.stdout.flush()
        urllib.request.urlretrieve(url, dest_path, reporthook=_reporthook)
        print(f"\n[+] Downloaded: {dest_path} ({os.path.getsize(dest_path)/(1024*1024):.1f} MB)")
    except Exception as e:
        print(f"\n[!] Python download failed ({e}), falling back to curl...")
        import subprocess
        subprocess.run(["curl", "-L", "-o", dest_path, url], check=True)
        print(f"[+] Downloaded via curl: {dest_path}")


def _ensure_pretrained_assets(cfg):
    """Auto-download required pretrained utility weights if missing on disk."""
    # 1. ASR model checkpoint (epoch_00080.pth)
    asr_path = cfg.get("ASR_path")
    if asr_path and not os.path.exists(asr_path):
        url = "https://github.com/yl4579/StyleTTS2/raw/main/Utils/ASR/epoch_00080.pth"
        _download_file(url, asr_path, "Pretrained ASR aligner (epoch_00080.pth)")

    # 2. F0 model checkpoint (bst.t7)
    f0_path = cfg.get("F0_path")
    if f0_path and not os.path.exists(f0_path):
        url = "https://github.com/yl4579/StyleTTS2/raw/main/Utils/JDC/bst.t7"
        _download_file(url, f0_path, "Pretrained F0 pitch extractor (bst.t7)")

    # 3. PL-BERT checkpoint (step_1000000.t7)
    plbert_dir = cfg.get("PLBERT_dir")
    if plbert_dir:
        plbert_ckpt = os.path.join(plbert_dir, "step_1000000.t7")
        if not os.path.exists(plbert_ckpt):
            url = "https://github.com/yl4579/StyleTTS2/raw/main/Utils/PLBERT/step_1000000.t7"
            _download_file(url, plbert_ckpt, "Pretrained PL-BERT (step_1000000.t7)")


# ─── Main Training Function ───────────────────────────────────────────────────
def run_stage1_training(config_path: str = CONFIG_PATH):
    print("=" * 65)
    print("KionTTS — Stage 1: Acoustic Foundation Training")
    print("=" * 65)

    # 1. Load config
    config = yaml.safe_load(open(config_path))
    config = _resolve_config_paths(config)
    _ensure_pretrained_assets(config)
    log_dir = config["log_dir"]
    os.makedirs(log_dir, exist_ok=True)
    shutil.copy(config_path, os.path.join(log_dir, os.path.basename(config_path)))

    # 2. Accelerator (single-GPU for Colab)
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        project_dir=log_dir,
        split_batches=True,
        kwargs_handlers=[ddp_kwargs],
    )
    device = accelerator.device

    if accelerator.is_main_process:
        writer = SummaryWriter(os.path.join(log_dir, "tensorboard"))
        file_handler = logging.FileHandler(os.path.join(log_dir, "stage1_train.log"))
        file_handler.setLevel(logging.DEBUG)
        log.addHandler(file_handler)

    # 3. Hyperparameters from config
    batch_size          = config.get("batch_size", 2)
    epochs              = config.get("epochs_1st", 120)
    save_freq           = 1  # Checkpoint every 1 epoch to ensure progress is saved to Drive
    save_step_interval  = config.get("save_step_interval", 1000)  # Checkpoint every 1000 steps
    log_interval        = config.get("log_interval", 10)
    max_len             = config.get("max_len", 200)
    loss_params         = Munch(config["loss_params"])
    TMA_epoch           = loss_params.TMA_epoch
    sr                  = config["preprocess_params"].get("sr", 24000)

    # ── Protect against CUDA OOM on Colab T4 (15GB VRAM) ──
    if torch.cuda.is_available():
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        if vram_gb < 14.0 and batch_size > 2:
            print(f"[*] Detected {vram_gb:.1f} GB VRAM (T4 / mid-VRAM GPU).")
            print(f"    Auto-clamping batch_size from {batch_size} -> 2 to prevent CUDA OutOfMemoryError.")
            batch_size = 2
        if vram_gb < 20.0 and max_len > 200:
            print(f"    Auto-clamping max_len from {max_len} -> 200 frames for memory safety.")
            max_len = 200
        torch.backends.cudnn.benchmark = True

    data_params = config["data_params"]
    train_list, val_list = get_data_path_list(
        data_params["train_data"], data_params["val_data"]
    )

    print(f"  Train samples : {len(train_list)}")
    print(f"  Val samples   : {len(val_list)}")
    print(f"  Batch size    : {batch_size}")
    print(f"  Max Mel Len   : {max_len}")
    print(f"  Epochs        : {epochs}  (TMA starts @ epoch {TMA_epoch})")
    print(f"  Device        : {device}")

    # 4. Build data loaders
    train_dataloader = build_dataloader(
        train_list,
        data_params["root_path"],
        OOD_data=data_params["OOD_data"],
        min_length=data_params["min_length"],
        batch_size=batch_size,
        num_workers=2,
        dataset_config={},
        device=device,
    )
    val_dataloader = build_dataloader(
        val_list,
        data_params["root_path"],
        OOD_data=data_params["OOD_data"],
        min_length=data_params["min_length"],
        batch_size=batch_size,
        validation=True,
        num_workers=0,
        dataset_config={},
        device=device,
    )

    # 5. Load pretrained utility models (ASR aligner, F0, PLBERT)
    with accelerator.main_process_first():
        print("  Loading pretrained ASR aligner...")
        text_aligner = load_ASR_models(config["ASR_path"], config["ASR_config"])

        print("  Loading pretrained F0 (JDC pitch) model...")
        pitch_extractor = load_F0_models(config["F0_path"])

        print("  Loading pretrained PL-BERT...")
        from Utils.PLBERT.util import load_plbert
        plbert = load_plbert(config["PLBERT_dir"])

    # 6. Build KionStyleTTS2 model components
    model_params = recursive_munch(config["model_params"])
    model = build_model(model_params, text_aligner, pitch_extractor, plbert)

    # Add KionStyleAdapter to the model dict (retained on CPU for Stage 1)
    ksa_cfg = config["model_params"]["kion_style_adapter"]
    kion_style_adapter = KionStyleAdapter(
        num_emotions=ksa_cfg["num_emotions"],
        num_styles=ksa_cfg["num_styles"],
        tag_embed_dim=ksa_cfg["tag_embed_dim"],
        latent_style_dim=ksa_cfg["latent_style_dim"],
        hidden_dim=ksa_cfg["hidden_dim"],
        dropout=ksa_cfg["dropout"],
    )
    model["kion_style_adapter"] = kion_style_adapter

    # ── Memory Optimization for Stage 1 ──────────────────────────────────────
    # Stage 1 only trains: text_encoder, style_encoder, decoder, mpd, msd, wd,
    # and (after TMA_epoch) text_aligner and pitch_extractor.
    # Stage 2 modules (bert, bert_encoder, predictor, predictor_encoder, diffusion,
    # and kion_style_adapter) are NEVER invoked in Stage 1. Keeping them on CPU
    # saves ~2.5 GB of GPU VRAM on T4, leaving headroom for the iSTFTNet decoder.
    stage1_active_keys = [
        "text_encoder", "style_encoder", "decoder",
        "text_aligner", "pitch_extractor",
        "mpd", "msd", "wd"
    ]

    for k in model:
        if k in stage1_active_keys:
            model[k] = model[k].to(device)
            model[k] = accelerator.prepare(model[k])
        else:
            model[k] = model[k].to("cpu")

    # 7. Scheduler + optimiser (only allocate optimizer states for Stage 1 modules)
    scheduler_params = {
        "max_lr":           float(config["optimizer_params"].get("lr", 1e-4)),
        "pct_start":        0.0,
        "epochs":           epochs,
        "steps_per_epoch":  len(train_dataloader),
    }
    optimizer = build_optimizer(
        {k: model[k].parameters() for k in stage1_active_keys if k in model},
        scheduler_params_dict={k: scheduler_params.copy() for k in stage1_active_keys if k in model},
        lr=float(config["optimizer_params"].get("lr", 1e-4)),
    )

    # 8. Prepare with Accelerator
    train_dataloader, val_dataloader = accelerator.prepare(
        train_dataloader, val_dataloader
    )
    for k in optimizer.optimizers:
        optimizer.optimizers[k] = accelerator.prepare(optimizer.optimizers[k])
        optimizer.schedulers[k] = accelerator.prepare(optimizer.schedulers[k])

    # 9. Resume from Drive checkpoint if available
    start_epoch = 0
    iters = 0
    best_loss = float("inf")

    with accelerator.main_process_first():
        latest_ckpt = _find_latest_drive_checkpoint()
        pretrained  = config.get("pretrained_model", "")

        if latest_ckpt:
            print(f"  Resuming from Drive checkpoint: {latest_ckpt}")
            model, optimizer, start_epoch, iters = load_checkpoint(
                model, optimizer, latest_ckpt, load_only_params=False
            )
            print(f"  [✓] Resumed state successfully: starting at epoch {start_epoch + 1}, step {iters}")
        elif pretrained:
            print(f"  Loading pretrained model: {pretrained}")
            model, optimizer, start_epoch, iters = load_checkpoint(
                model, optimizer, pretrained,
                load_only_params=config.get("load_only_params", True),
            )

    # 10. Loss modules
    try:
        n_down = model["text_aligner"].module.n_down
    except AttributeError:
        n_down = model["text_aligner"].n_down

    stft_loss = MultiResolutionSTFTLoss().to(device)
    gl        = GeneratorLoss(model["mpd"], model["msd"]).to(device)
    dl        = DiscriminatorLoss(model["mpd"], model["msd"]).to(device)
    wl        = WavLMLoss(
        model_params.slm.model,
        model["wd"],
        sr,
        model_params.slm.sr,
    ).to(device)

    # ── Speed Optimization: Freeze WavLM SLM backbone ────────────────────────
    # WavLM is an auxiliary feature extractor (94M params). Freezing its weights
    # prevents PyTorch from allocating gradients and tracking computation graphs
    # for all 13 transformer layers during generator backward passes.
    if hasattr(wl, "wavlm") and wl.wavlm is not None:
        wl.wavlm.eval()
        for p in wl.wavlm.parameters():
            p.requires_grad = False

    print(f"\n  Starting training from epoch {start_epoch + 1}...")
    torch.cuda.empty_cache()

    # ──────────────────────────────────────────────────────────────────────────
    # 11. Training Loop
    # ──────────────────────────────────────────────────────────────────────────
    for epoch in range(start_epoch, epochs):
        epoch_start = time.time()
        running_loss = 0.0
        _ = [model[k].train() for k in stage1_active_keys if k in model]

        total_steps_in_loader = len(train_dataloader)
        if epoch == start_epoch and (iters % total_steps_in_loader) != 0:
            remaining_steps = total_steps_in_loader - (iters % total_steps_in_loader)
        else:
            remaining_steps = total_steps_in_loader

        pbar = tqdm(
            train_dataloader,
            total=remaining_steps,
            desc=f"Epoch {epoch+1:03d}/{epochs} [Stage 1]",
            leave=False,
        )

        for i, batch in enumerate(pbar):
            if i >= remaining_steps:
                break

            waves = batch[0]
            batch = [b.to(device, non_blocking=True) for b in batch[1:]]
            texts, input_lengths, _, _, mels, mel_input_length, _ = batch

            with torch.no_grad():
                mask      = length_to_mask(mel_input_length // (2 ** n_down)).to(device)
                text_mask = length_to_mask(input_lengths).to(device)

            # ASR aligner forward
            # When epoch < TMA_epoch, text_aligner is frozen and no s2s/mono loss is backpropagated.
            if epoch < TMA_epoch:
                with torch.no_grad():
                    ppgs, s2s_pred, s2s_attn = model["text_aligner"](mels, mask, texts)
                    s2s_attn = s2s_attn.transpose(-1, -2)[..., 1:].transpose(-1, -2)
            else:
                ppgs, s2s_pred, s2s_attn = model["text_aligner"](mels, mask, texts)
                s2s_attn = s2s_attn.transpose(-1, -2)[..., 1:].transpose(-1, -2)

            with torch.no_grad():
                attn_mask = (
                    (~mask).unsqueeze(-1)
                    .expand(mask.shape[0], mask.shape[1], text_mask.shape[-1])
                    .float()
                    .transpose(-1, -2)
                ) * (
                    (~text_mask).unsqueeze(-1)
                    .expand(*text_mask.shape, mask.shape[-1])
                    .float()
                )
                attn_mask  = attn_mask < 1
            s2s_attn.masked_fill_(attn_mask, 0.0)

            with torch.no_grad():
                mask_ST       = mask_from_lens(s2s_attn, input_lengths, mel_input_length // (2 ** n_down))
                s2s_attn_mono = maximum_path(s2s_attn, mask_ST)

            # Text encode
            t_en = model["text_encoder"](texts, input_lengths, text_mask)

            # 50/50 soft vs monotonic alignment
            asr = (t_en @ s2s_attn) if random.getrandbits(1) else (t_en @ s2s_attn_mono)

            # Build random clips
            mel_input_length_all = accelerator.gather(mel_input_length)
            mel_len    = min(int(mel_input_length_all.min().item() / 2 - 1), max_len // 2)
            mel_len_st = int(mel_input_length.min().item() / 2 - 1)

            en, gt, wav, st = [], [], [], []
            for bib in range(len(mel_input_length)):
                mel_length = int(mel_input_length[bib].item() / 2)
                rs = np.random.randint(0, mel_length - mel_len)
                en.append(asr[bib, :, rs:rs + mel_len])
                gt.append(mels[bib, :, rs * 2:(rs + mel_len) * 2])
                y = waves[bib][rs * 2 * 300:(rs + mel_len) * 2 * 300]
                wav.append(torch.from_numpy(y).to(device, non_blocking=True))
                rs2 = np.random.randint(0, mel_length - mel_len_st)
                st.append(mels[bib, :, rs2 * 2:(rs2 + mel_len_st) * 2])

            en  = torch.stack(en)
            gt  = torch.stack(gt).detach()
            st  = torch.stack(st).detach()
            wav = torch.stack(wav).float().detach()

            if gt.shape[-1] < 80:
                continue

            # F0 and style extraction
            with torch.no_grad():
                real_norm        = log_norm(gt.unsqueeze(1)).squeeze(1).detach()
                F0_real, _, _    = model["pitch_extractor"](gt.unsqueeze(1))

            s = model["style_encoder"](gt.unsqueeze(1))  # acoustic style (training only)
            y_rec = model["decoder"](en, F0_real, real_norm, s)

            # ── Discriminator step ──
            if epoch >= TMA_epoch:
                optimizer.zero_grad("msd")
                optimizer.zero_grad("mpd")
                d_loss = dl(wav.detach().unsqueeze(1).float(), y_rec.detach()).mean()
                accelerator.backward(d_loss)
                optimizer.step("msd")
                optimizer.step("mpd")
            else:
                d_loss = 0

            # ── Generator step ──
            optimizer.zero_grad("text_encoder")
            optimizer.zero_grad("style_encoder")
            optimizer.zero_grad("decoder")
            if epoch >= TMA_epoch:
                optimizer.zero_grad("text_aligner")
                optimizer.zero_grad("pitch_extractor")

            loss_mel = stft_loss(y_rec.squeeze(), wav.detach())

            if epoch >= TMA_epoch:
                loss_s2s = sum(
                    F.cross_entropy(pred[:tl], txt[:tl])
                    for pred, txt, tl in zip(s2s_pred, texts, input_lengths)
                ) / texts.size(0)

                loss_mono    = F.l1_loss(s2s_attn, s2s_attn_mono) * 10
                loss_gen_all = gl(wav.detach().unsqueeze(1).float(), y_rec).mean()
                loss_slm     = wl(wav.detach(), y_rec).mean()
                g_loss = (
                    loss_params.lambda_mel  * loss_mel
                    + loss_params.lambda_mono * loss_mono
                    + loss_params.lambda_s2s  * loss_s2s
                    + loss_params.lambda_gen  * loss_gen_all
                    + loss_params.lambda_slm  * loss_slm
                )
            else:
                loss_s2s = loss_mono = loss_gen_all = loss_slm = 0
                g_loss = loss_mel

            running_loss += accelerator.gather(loss_mel).mean().item()
            accelerator.backward(g_loss)

            optimizer.step("text_encoder")
            optimizer.step("style_encoder")
            optimizer.step("decoder")
            if epoch >= TMA_epoch:
                optimizer.step("text_aligner")
                optimizer.step("pitch_extractor")

            # Step learning rate schedulers
            optimizer.scheduler()

            iters += 1
            _to_num = lambda v: float(v.item()) if hasattr(v, "item") else (float(v) if v is not None else 0.0)

            pbar.set_postfix(
                step=iters,
                mel=f"{_to_num(loss_mel):.4f}",
                gen=f"{_to_num(loss_gen_all):.4f}",
            )

            # ── Checkpoint every save_step_interval (1000) steps ──
            if iters % save_step_interval == 0 and accelerator.is_main_process:
                state = {
                    "net":       {k: accelerator.unwrap_model(model[k]).state_dict() for k in model},
                    "optimizer": optimizer.state_dict(),
                    "iters":     iters,
                    "val_loss":  running_loss / max((i % log_interval) + 1, 1),
                    "epoch":     epoch,  # In-progress epoch
                }
                _save_to_drive(state, epoch=epoch + 1, step=iters)

            if (i + 1) % log_interval == 0 and accelerator.is_main_process:
                avg = running_loss / log_interval
                writer.add_scalar("train/mel_loss",  avg,                  iters)
                writer.add_scalar("train/gen_loss",  _to_num(loss_gen_all), iters)
                writer.add_scalar("train/disc_loss", _to_num(d_loss),     iters)
                running_loss = 0.0

        # ── Validation ────────────────────────────────────────────────────
        _ = [model[k].eval() for k in stage1_active_keys if k in model]
        loss_test = 0.0
        iters_test = 0

        with torch.no_grad():
            for batch in val_dataloader:
                waves = batch[0]
                batch = [b.to(device, non_blocking=True) for b in batch[1:]]
                texts, input_lengths, _, _, mels, mel_input_length, _ = batch

                mask      = length_to_mask(mel_input_length // (2 ** n_down)).to(device)
                text_mask = length_to_mask(input_lengths).to(device)
                _, _, s2s_attn = model["text_aligner"](mels, mask, texts)
                s2s_attn = s2s_attn.transpose(-1, -2)[..., 1:].transpose(-1, -2)

                t_en = model["text_encoder"](texts, input_lengths, text_mask)
                asr  = t_en @ s2s_attn

                mel_len = min(int(mel_input_length.min().item() / 2 - 1), max_len // 2)
                en, gt, wav = [], [], []
                for bib in range(len(mel_input_length)):
                    ml = int(mel_input_length[bib].item() / 2)
                    rs = np.random.randint(0, ml - mel_len)
                    en.append(asr[bib, :, rs:rs + mel_len])
                    gt.append(mels[bib, :, rs * 2:(rs + mel_len) * 2])
                    y = waves[bib][rs * 2 * 300:(rs + mel_len) * 2 * 300]
                    wav.append(torch.from_numpy(y).to(device, non_blocking=True))

                en  = torch.stack(en)
                gt  = torch.stack(gt).detach()
                wav = torch.stack(wav).float().detach()

                F0_real, _, _ = model["pitch_extractor"](gt.unsqueeze(1))
                s             = model["style_encoder"](gt.unsqueeze(1))
                real_norm     = log_norm(gt.unsqueeze(1)).squeeze(1)
                y_rec         = model["decoder"](en, F0_real, real_norm, s)
                loss_mel      = stft_loss(y_rec.squeeze(), wav.detach())
                loss_test    += accelerator.gather(loss_mel).mean().item()
                iters_test   += 1

        torch.cuda.empty_cache()

        val_loss = loss_test / max(iters_test, 1)
        is_best  = val_loss < best_loss
        if is_best:
            best_loss = val_loss

        elapsed = time.time() - epoch_start
        if accelerator.is_main_process:
            print(
                f"  Epoch {epoch+1:03d} | val_mel={val_loss:.4f}"
                f" {'★BEST' if is_best else '     '} | {elapsed:.0f}s"
            )
            writer.add_scalar("eval/mel_loss", val_loss, epoch + 1)

            # Write sample audio to Google Drive and TensorBoard on best or every 5 epochs
            if (epoch + 1) % 5 == 0 or is_best:
                sample_dir = os.path.join(DRIVE_CKPT_DIR, "samples")
                os.makedirs(sample_dir, exist_ok=True)
                with torch.no_grad():
                    for bib in range(min(3, len(en))):
                        ml  = int(mel_input_length[bib].item())
                        g   = mels[bib, :, :ml].unsqueeze(0)
                        e   = asr[bib, :, :ml // 2].unsqueeze(0)
                        F0r, _, _ = model["pitch_extractor"](g.unsqueeze(1))
                        s_  = model["style_encoder"](g.unsqueeze(1))
                        nr  = log_norm(g.unsqueeze(1)).squeeze(1)
                        yr  = model["decoder"](e, F0r.unsqueeze(0), nr, s_)
                        audio_arr = yr.cpu().numpy().squeeze()
                        writer.add_audio(f"eval/synth_{bib}", audio_arr, epoch + 1, sample_rate=sr)
                        try:
                            import soundfile as sf
                            wav_path = os.path.join(sample_dir, f"kion_stage1_epoch_{epoch+1:03d}_sample_{bib+1}.wav")
                            sf.write(wav_path, audio_arr, sr)
                        except Exception:
                            pass
                print(f"  [♫] Saved {min(3, len(en))} audio samples → {sample_dir}")

            # Save checkpoint
            if (epoch + 1) % save_freq == 0 or is_best:
                state = {
                    "net":       {k: accelerator.unwrap_model(model[k]).state_dict() for k in model},
                    "optimizer": optimizer.state_dict(),
                    "iters":     iters,
                    "val_loss":  val_loss,
                    "epoch":     epoch + 1,  # Completed epoch: resume starts from next epoch
                }
                _save_to_drive(state, epoch + 1, is_best=is_best)

    # Final save
    if accelerator.is_main_process:
        print("\n[+] Stage 1 training complete!")
        state = {
            "net":       {k: accelerator.unwrap_model(model[k]).state_dict() for k in model},
            "optimizer": optimizer.state_dict(),
            "iters":     iters,
            "val_loss":  best_loss,
            "epoch":     epochs,
        }
        final_path = os.path.join(DRIVE_CKPT_DIR, "kion_stage1_final.pth")
        torch.save(state, final_path)
        print(f"  Final Stage 1 checkpoint → {final_path}")
        print("  Proceed to Cell 07 for Stage 2 (style diffusion) training.")


# ─── Colab Cell Entry Point ───────────────────────────────────────────────────
if __name__ == "__main__":
    from utils import maximum_path   # import after path setup
    run_stage1_training()
