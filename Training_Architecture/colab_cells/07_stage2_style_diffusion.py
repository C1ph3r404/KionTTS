"""
Colab Cell 07: Stage 2 — Style Diffusion + KionStyleAdapter Training
Builds on the Stage 1 acoustic checkpoint. Adds:
    • KionStyleAdapter joint training (tag → latent style)
    • StyleTTS2 Diffusion Prosody (DiffusionSampler with ADPM2 + Karras schedule)
    • ProsodyPredictor fine-tuning (F0, energy, durations)
    • SLM Adversarial Loss (WavLM discriminator head)
    • Style reconstruction consistency: tag style vs audio style

Training Phases:
    Phase 1 (Epochs 0–diff_epoch):    prosody predictor + adapter
    Phase 2 (Epochs diff_epoch–joint): + diffusion score matching
    Phase 3 (Epochs joint–end):        full joint end-to-end

Expected Duration: ~50–80 epochs
  T4 GPU (15GB): ~6–10 hours
  A100 (40GB):   ~2–4 hours

Run AFTER: Cell 06 (Stage 1 complete)
Run BEFORE: Cell 08 (inference test)
"""

import os
import sys

# Prevent CUDA memory fragmentation on 16GB GPUs (like Colab T4)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import time
import copy
import random
import shutil
import logging
import warnings
import numpy as np
import torch
import torch.nn as nn
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

# ─── Environment & Repository Paths ──────────────────────────────────────────
def _get_repo_root() -> str:
    rel_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    if os.path.exists(os.path.join(rel_path, "model")):
        return rel_path
    for p in [
        "/kaggle/working/KionTTS",
        "/kaggle/working/kiontts",
        "/kaggle/working",
        "/content/KionTTS",
        "/content/Kiontts",
        "/content/kiontts",
    ]:
        if os.path.exists(os.path.join(p, "model")):
            return p
        if os.path.exists(p):
            return p
    return "/kaggle/working" if os.path.exists("/kaggle") else "/content/KionTTS"


REPO_ROOT      = _get_repo_root()
STYLETTS2_ROOT = f"{REPO_ROOT}/StyleTTS2"

# Checkpoint directory: Drive on Colab if mounted, otherwise local /kaggle/working/checkpoints
if os.path.exists("/content/drive/MyDrive"):
    DRIVE_CKPT_DIR = "/content/drive/MyDrive/KionTTS_Checkpoints"
elif os.path.exists("/kaggle"):
    DRIVE_CKPT_DIR = "/kaggle/working/checkpoints"
else:
    DRIVE_CKPT_DIR = os.path.join(REPO_ROOT, "checkpoints")

LOCAL_CKPT_DIR = DRIVE_CKPT_DIR
CONFIG_PATH    = f"{STYLETTS2_ROOT}/Configs/kion_config.yml"
STAGE1_BEST    = os.path.join(DRIVE_CKPT_DIR, "kion_stage1_best.pth")
STAGE1_FINAL   = os.path.join(DRIVE_CKPT_DIR, "kion_stage1_final.pth")

for p in [REPO_ROOT, STYLETTS2_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)


# ─── Ensure Dependencies (self-healing for active Colab/Kaggle runtimes) ─────
for _pkg, _mod in [
    ("munch", "munch"),
    ("einops-exts", "einops_exts"),
    ("einops", "einops"),
    ("pydub", "pydub"),
    ("nltk", "nltk"),
    ("huggingface_hub", "huggingface_hub"),
]:
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

# ─── Imports ──────────────────────────────────────────────────────────────────
import yaml
from munch import Munch

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
    recursive_munch,
    maximum_path,
    mask_from_lens,
)
from optimizers import build_optimizer
from Modules.slmadv import SLMAdversarialLoss
from Modules.diffusion.sampler import DiffusionSampler, ADPM2Sampler, KarrasSchedule
from Utils.PLBERT.util import load_plbert

from model.modules.style_adapter import KionStyleAdapter

log = logging.getLogger("KionStage2")
log.setLevel(logging.DEBUG)
log.addHandler(logging.StreamHandler())


# ─── KionStyleConsistencyLoss ─────────────────────────────────────────────────
class KionStyleConsistencyLoss(nn.Module):
    """
    Penalises divergence between the tag-predicted style vector (s_tag)
    and the audio-extracted style vector (s_audio).
    Both are in R^style_dim; we use cosine + L1 for robustness.
    """
    def forward(self, s_tag: torch.Tensor, s_audio: torch.Tensor) -> torch.Tensor:
        cos_sim   = F.cosine_similarity(s_tag, s_audio, dim=-1).mean()
        l1_loss   = F.l1_loss(s_tag, s_audio)
        return (1.0 - cos_sim) + 0.5 * l1_loss  # lower = more consistent


kion_style_consistency = KionStyleConsistencyLoss()


# ─── Hugging Face Hub Credentials & Checkpoint Sync ──────────────────────────
def _get_hf_token() -> str:
    """Discovers Hugging Face token from Colab secrets, Kaggle secrets, env vars, or HF cache."""
    # 1. Colab Secrets
    try:
        from google.colab import userdata
        t = userdata.get('HF_TOKEN')
        if t: return t.strip()
    except Exception:
        pass
    # 2. Kaggle Secrets
    try:
        from kaggle_secrets import UserSecretsClient
        t = UserSecretsClient().get_secret('HF_TOKEN')
        if t: return t.strip()
    except Exception:
        pass
    # 3. Environment variables
    for env_var in ["HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HF_AUTH_TOKEN"]:
        val = os.environ.get(env_var, "").strip()
        if val:
            return val
    # 4. Hugging Face cached token
    try:
        from huggingface_hub import HfFolder
        cached = HfFolder.get_token()
        if cached:
            return cached.strip()
    except Exception:
        pass
    return ""


HF_TOKEN = _get_hf_token()
HF_REPO_ID = os.environ.get("HF_REPO_ID", "nate0001/KionTTS-Checkpoints").strip()


def _upload_to_hf(local_path: str, hf_filename: str, repo_id: str = HF_REPO_ID, token: str = None) -> bool:
    """Uploads a checkpoint or metadata file to Hugging Face Model Hub."""
    token = token or _get_hf_token()
    if not token:
        print(f"  [!] HF_TOKEN not configured. Skipping Hugging Face upload for '{hf_filename}'.")
        print("      To enable automatic HF syncing, set HF_TOKEN in Colab Secrets or Kaggle Secrets.")
        return False
    if not os.path.exists(local_path):
        print(f"  [!] File to upload not found: {local_path}")
        return False
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=token)
        # Create private model repo if it doesn't exist
        api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True, private=True)
        size_mb = os.path.getsize(local_path) / (1024 * 1024)
        print(f"  [*] Uploading '{hf_filename}' ({size_mb:.1f} MB) to HF [{repo_id}]...", flush=True)
        api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=hf_filename,
            repo_id=repo_id,
            repo_type="model",
        )
        print(f"  [✓] Upload complete → https://huggingface.co/{repo_id}/blob/main/{hf_filename}")
        return True
    except Exception as e:
        print(f"  [!] HF upload failed for '{hf_filename}': {e}")
        return False


def _download_from_hf(hf_filename: str, local_dest_dir: str = DRIVE_CKPT_DIR, repo_id: str = HF_REPO_ID, token: str = None) -> str | None:
    """Downloads a checkpoint from Hugging Face Model Hub if not present locally."""
    token = token or _get_hf_token()
    try:
        from huggingface_hub import hf_hub_download
        os.makedirs(local_dest_dir, exist_ok=True)
        print(f"  [*] Attempting to download '{hf_filename}' from HF repo [{repo_id}]...", flush=True)
        dest = hf_hub_download(
            repo_id=repo_id,
            filename=hf_filename,
            token=token or None,
            local_dir=local_dest_dir,
            local_dir_use_symlinks=False,
        )
        if _is_valid_checkpoint(dest):
            print(f"  [✓] Successfully downloaded & verified from HF: {dest}")
            return dest
        else:
            print(f"  [!] Checkpoint from HF failed integrity check: {dest}")
            return None
    except Exception as e:
        print(f"  [-] Could not download '{hf_filename}' from HF: {e}")
        return None


# ─── Checkpoint Helpers ───────────────────────────────────────────────────────
def _is_valid_checkpoint(path: str) -> bool:
    """Verifies that a checkpoint exists, is non-empty, and is a valid zip archive with 0MB RAM footprint."""
    if not path or not os.path.exists(path):
        return False
    try:
        if os.path.getsize(path) < 1024 * 1024:  # Must be at least 1MB
            return False
        import zipfile
        with zipfile.ZipFile(path, "r") as zf:
            return zf.testzip() is None
    except Exception as e:
        print(f"  [!] Integrity check failed for '{os.path.basename(path)}': {e}")
        return False


def _save_to_drive(state: dict, epoch: int, step: int = None, is_best: bool = False, stage: str = "stage2", step_interval: int = 900):
    os.makedirs(DRIVE_CKPT_DIR, exist_ok=True)
    if step is not None:
        slot = "A" if (step // step_interval) % 2 == 0 else "B"
        file_base = f"kion_{stage}_step_slot_{slot}.pth"
        ckpt_path = os.path.join(DRIVE_CKPT_DIR, file_base)
    else:
        slot = "A" if epoch % 2 == 0 else "B"
        file_base = f"kion_{stage}_epoch_slot_{slot}.pth"
        ckpt_path = os.path.join(DRIVE_CKPT_DIR, file_base)

    if step is not None:
        print(f"\n  [*] Saving step checkpoint {step} (Slot {slot})...", flush=True)
    else:
        print(f"\n  [*] Saving epoch checkpoint (Epoch {epoch}, Slot {slot})...", flush=True)

    local_dir = "/kaggle/working" if os.path.exists("/kaggle") else ("/content" if os.path.exists("/content") else os.path.dirname(ckpt_path))
    tmp_path = os.path.join(local_dir, f"kion_{stage}_tmp_{int(time.time())}.pth")
    try:
        torch.save(state, tmp_path)
        if _is_valid_checkpoint(tmp_path):
            shutil.copyfile(tmp_path, ckpt_path)
            if step is not None:
                print(f"  [✓] Verified & saved step checkpoint {step} (Slot {slot}) → {ckpt_path}")
            else:
                print(f"  [✓] Verified & saved epoch checkpoint (Epoch {epoch}, Slot {slot}) → {ckpt_path}")

            # Update local pointer file
            meta_path = os.path.join(DRIVE_CKPT_DIR, f"latest_{stage}_checkpoint.txt")
            try:
                with open(meta_path, "w") as f:
                    f.write(f"{file_base}\n")
            except Exception:
                pass

            # Sync to Hugging Face Hub
            _upload_to_hf(ckpt_path, file_base)
            if os.path.exists(meta_path):
                _upload_to_hf(meta_path, f"latest_{stage}_checkpoint.txt")
        else:
            print(f"  [!] Local save integrity check failed. Skipping copy to checkpoint directory.")
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass

    if is_best and os.path.exists(ckpt_path):
        best_name = f"kion_{stage}_best.pth"
        best_path = os.path.join(DRIVE_CKPT_DIR, best_name)
        shutil.copy2(ckpt_path, best_path)
        print(f"  [★] New best! → {best_path}")
        _upload_to_hf(best_path, best_name)


def _find_latest_stage2_checkpoint() -> str | None:
    import glob
    candidates = []

    # 1. Check pointer file
    meta_path = os.path.join(DRIVE_CKPT_DIR, "latest_stage2_checkpoint.txt")
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r") as f:
                target = f.readline().strip()
                if target:
                    if not os.path.isabs(target):
                        target = os.path.join(DRIVE_CKPT_DIR, target)
                    if os.path.exists(target):
                        candidates.append(target)
        except Exception:
            pass

    # 2. Check rolling slots and numbered files
    patterns = [
        "kion_stage2_step_slot_*.pth",
        "kion_stage2_epoch_slot_*.pth",
        "kion_stage2_step_*.pth",
        "kion_stage2_epoch_*.pth",
    ]
    found = []
    for pat in patterns:
        found.extend(glob.glob(os.path.join(DRIVE_CKPT_DIR, pat)))

    valid_found = [c for c in found if not c.endswith("best.pth") and not c.endswith("final.pth")]
    valid_found.sort(key=os.path.getmtime, reverse=True)
    for c in valid_found:
        if c not in candidates:
            candidates.append(c)

    best_path = os.path.join(DRIVE_CKPT_DIR, "kion_stage2_best.pth")
    if os.path.exists(best_path) and best_path not in candidates:
        candidates.append(best_path)

    # 3. Test local candidates for corruption
    for cand in candidates:
        print(f"  Verifying integrity of candidate: {os.path.basename(cand)}...")
        if _is_valid_checkpoint(cand):
            print(f"  [✓] Integrity confirmed: {os.path.basename(cand)}")
            return cand
        else:
            print(f"  [!] Checkpoint {os.path.basename(cand)} is damaged. Falling back...")

    # 4. If no local candidates found (e.g. Kaggle environment), try Hugging Face Hub
    print("  [*] No local Stage 2 checkpoint found. Checking Hugging Face Hub...")
    hf_best = _download_from_hf("kion_stage2_best.pth")
    if hf_best:
        return hf_best
    hf_slot_a = _download_from_hf("kion_stage2_step_slot_A.pth")
    if hf_slot_a:
        return hf_slot_a
    hf_slot_b = _download_from_hf("kion_stage2_step_slot_B.pth")
    if hf_slot_b:
        return hf_slot_b

    return None


def _load_stage1_checkpoint() -> str:
    """Returns the best available Stage 1 checkpoint path with integrity verification."""
    candidates = []
    if os.path.exists(STAGE1_BEST):
        candidates.append(STAGE1_BEST)
    if os.path.exists(STAGE1_FINAL):
        candidates.append(STAGE1_FINAL)

    # Check pointer file
    meta_path = os.path.join(DRIVE_CKPT_DIR, "latest_stage1_checkpoint.txt")
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r") as f:
                target = f.readline().strip()
                if target:
                    if not os.path.isabs(target):
                        target = os.path.join(DRIVE_CKPT_DIR, target)
                    if os.path.exists(target) and target not in candidates:
                        candidates.append(target)
        except Exception:
            pass

    import glob
    found = sorted(glob.glob(os.path.join(DRIVE_CKPT_DIR, "kion_stage1_*.pth")), key=os.path.getmtime, reverse=True)
    for c in found:
        if c not in candidates:
            candidates.append(c)

    for cand in candidates:
        if _is_valid_checkpoint(cand):
            print(f"  [✓] Stage 1 checkpoint verified: {os.path.basename(cand)}")
            return cand

    # If no local candidate exists (e.g. running on Kaggle without Drive), pull from HF Hub
    print("  [*] No valid local Stage 1 checkpoint found. Checking Hugging Face Hub...")
    hf_s1_best = _download_from_hf("kion_stage1_best.pth")
    if hf_s1_best:
        return hf_s1_best
    hf_s1_final = _download_from_hf("kion_stage1_final.pth")
    if hf_s1_final:
        return hf_s1_final

    raise FileNotFoundError(
        "No valid Stage 1 checkpoint found locally, on Drive, or on Hugging Face.\n"
        f"Checked candidates: {[os.path.basename(c) for c in candidates]}\n"
        f"Hugging Face repo checked: {HF_REPO_ID}"
    )


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


def sync_drive_checkpoints_to_hf(repo_id: str = HF_REPO_ID) -> bool:
    """Discovers existing Stage 1 & Stage 2 checkpoints on Google Drive and uploads them to Hugging Face."""
    if not os.path.exists("/content/drive/MyDrive"):
        return False

    print("\n" + "=" * 65)
    print("Syncing existing checkpoints from Google Drive to Hugging Face...")
    print(f"Target Hugging Face Model Hub: https://huggingface.co/{repo_id}")
    print("=" * 65)

    uploaded_any = False

    # 1. Stage 1 checkpoint from Drive
    try:
        s1 = _load_stage1_checkpoint()
        if s1 and os.path.exists(s1) and s1.startswith("/content/drive"):
            print(f"[+] Found Stage 1 checkpoint on Drive: {s1}")
            _upload_to_hf(s1, "kion_stage1_best.pth", repo_id=repo_id)
            tmp_ptr = os.path.join(os.path.dirname(s1), "latest_stage1_checkpoint.txt")
            try:
                with open(tmp_ptr, "w") as f:
                    f.write("kion_stage1_best.pth\n")
                _upload_to_hf(tmp_ptr, "latest_stage1_checkpoint.txt", repo_id=repo_id)
            except Exception:
                pass
            uploaded_any = True
    except Exception as e:
        print(f"[-] Stage 1 checkpoint note: {e}")

    # 2. Stage 2 checkpoints from Drive
    try:
        s2 = _find_latest_stage2_checkpoint()
        if s2 and os.path.exists(s2) and s2.startswith("/content/drive"):
            print(f"[+] Found latest Stage 2 checkpoint on Drive: {s2}")
            _upload_to_hf(s2, os.path.basename(s2), repo_id=repo_id)
            _upload_to_hf(s2, "kion_stage2_latest.pth", repo_id=repo_id)
            uploaded_any = True
    except Exception as e:
        print(f"[-] Stage 2 checkpoint note: {e}")

    # 3. Best Stage 2 if present on Drive
    best_s2 = os.path.join(DRIVE_CKPT_DIR, "kion_stage2_best.pth")
    if os.path.exists(best_s2) and best_s2.startswith("/content/drive"):
        print(f"[+] Found Stage 2 best checkpoint on Drive: {best_s2}")
        _upload_to_hf(best_s2, "kion_stage2_best.pth", repo_id=repo_id)
        uploaded_any = True

    if uploaded_any:
        print(f"\n[✓] Checkpoint sync complete! Kaggle can now pull directly from: https://huggingface.co/{repo_id}")
    else:
        print("\n[*] No un-synced Drive checkpoints found to upload.")
    return uploaded_any


# ─── Main Training Function ───────────────────────────────────────────────────
def run_stage2_training(config_path: str = CONFIG_PATH, sync_hf_first: bool = True):
    print("=" * 65)
    print("KionTTS — Stage 2: Style Diffusion + Adapter Training")
    print("=" * 65)

    # Automatically sync existing checkpoints from Drive to HF before starting training
    if sync_hf_first and os.path.exists("/content/drive/MyDrive"):
        sync_drive_checkpoints_to_hf()

    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    config      = yaml.safe_load(open(config_path))
    config      = _resolve_config_paths(config)
    _ensure_pretrained_assets(config)
    log_dir     = config["log_dir"]
    os.makedirs(log_dir, exist_ok=True)
    loss_params = Munch(config["loss_params"])
    diff_epoch  = loss_params.diff_epoch
    joint_epoch = loss_params.joint_epoch
    epochs              = config.get("epochs_2nd", 60)
    batch_size          = config.get("batch_size", 2)
    max_len             = config.get("max_len", 200)
    save_step_interval  = config.get("save_step_interval", 900)
    sr                  = config["preprocess_params"].get("sr", 24000)
    slmadv_cfg          = Munch(config.get("slmadv_params", {}))
    device              = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_fp16            = torch.cuda.is_available()

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

    if torch.cuda.is_available():
        # benchmark=False prevents algorithm re-searching on variable audio shapes.
        # allow_tf32=True gives free Tensor Core speedup on Ampere+ GPUs.
        torch.backends.cudnn.benchmark        = False
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32       = True

    writer = SummaryWriter(os.path.join(log_dir, "tensorboard_stage2"))

    print(f"  Epochs       : {epochs}")
    print(f"  Diff start   : epoch {diff_epoch}")
    print(f"  Joint start  : epoch {joint_epoch}")
    print(f"  Batch size   : {batch_size}")
    print(f"  Max Mel Len  : {max_len}")
    print(f"  Save step    : every {save_step_interval} steps")
    print(f"  Device       : {device}  | FP16: {use_fp16}")

    # ── Data ─────────────────────────────────────────────────────────────────
    data_params  = config["data_params"]
    train_list, val_list = get_data_path_list(
        data_params["train_data"], data_params["val_data"]
    )
    # num_workers=0 and load_ref=False:
    # 1. Eliminates multiprocessing worker duplication and glibc heap fragmentation
    # 2. Eliminates /dev/shm shared memory queue IPC buffering (saving gigabytes of System RAM)
    # 3. load_ref=False skips reading redundant 2nd audio file and running torchaudio STFT per item
    train_dataloader = build_dataloader(
        train_list, data_params["root_path"],
        OOD_data=data_params["OOD_data"],
        min_length=data_params["min_length"],
        batch_size=batch_size, num_workers=0,
        dataset_config={"load_ref": False}, device=device,
    )
    val_dataloader = build_dataloader(
        val_list, data_params["root_path"],
        OOD_data=data_params["OOD_data"],
        min_length=data_params["min_length"],
        batch_size=batch_size, validation=True,
        num_workers=0, dataset_config={"load_ref": False}, device=device,
    )
    print(f"  Train: {len(train_list)} | Val: {len(val_list)}")

    # ── Load models ───────────────────────────────────────────────────────────
    text_aligner    = load_ASR_models(config["ASR_path"], config["ASR_config"]).to(device)
    pitch_extractor = load_F0_models(config["F0_path"]).to(device)
    plbert          = load_plbert(config["PLBERT_dir"])

    model_params = recursive_munch(config["model_params"])
    model        = build_model(model_params, text_aligner, pitch_extractor, plbert)
    _ = [model[k].to(device) for k in model]

    # Freeze utility models (text_aligner and pitch_extractor are frozen in Stage 2)
    for p in model["text_aligner"].parameters():
        p.requires_grad = False
    model["text_aligner"].eval()

    for p in model["pitch_extractor"].parameters():
        p.requires_grad = False
    model["pitch_extractor"].eval()

    # KionStyleAdapter
    ksa_cfg = config["model_params"]["kion_style_adapter"]
    model["kion_style_adapter"] = KionStyleAdapter(
        num_emotions=ksa_cfg["num_emotions"],
        num_styles=ksa_cfg["num_styles"],
        tag_embed_dim=ksa_cfg["tag_embed_dim"],
        latent_style_dim=ksa_cfg["latent_style_dim"],
        hidden_dim=ksa_cfg["hidden_dim"],
        dropout=ksa_cfg["dropout"],
    ).to(device)

    # ── Diffusion Sampler ─────────────────────────────────────────────────────
    sampler = DiffusionSampler(
        model["diffusion"].diffusion,
        sampler=ADPM2Sampler(),
        sigma_schedule=KarrasSchedule(sigma_min=0.0001, sigma_max=3.0, rho=9.0),
        clamp=False,
    )

    # ── Load Stage 1 checkpoint ───────────────────────────────────────────────
    stage1_ckpt = _load_stage1_checkpoint()
    print(f"  Loading Stage 1 checkpoint: {stage1_ckpt}")
    model, _, _, _ = load_checkpoint(
        model, None, stage1_ckpt, load_only_params=True,
        ignore_modules=['bert', 'bert_encoder', 'predictor', 'predictor_encoder', 'msd', 'mpd', 'wd', 'diffusion']
    )
    # Stage 2 predictor_encoder is initialised from Stage 1 style_encoder weights
    model["predictor_encoder"] = copy.deepcopy(model["style_encoder"])

    # ── Hugging Face Sync for Stage 1 Checkpoint (ONLY if loaded from Google Drive) ────
    if os.path.exists("/content/drive/MyDrive") and stage1_ckpt.startswith("/content/drive"):
        print("\n  [*] Syncing verified Stage 1 checkpoint from Drive to Hugging Face Model Hub...", flush=True)
        _upload_to_hf(stage1_ckpt, "kion_stage1_best.pth")
        tmp_s1_ptr = os.path.join(os.path.dirname(stage1_ckpt), "latest_stage1_checkpoint.txt")
        try:
            with open(tmp_s1_ptr, "w") as f:
                f.write("kion_stage1_best.pth\n")
            _upload_to_hf(tmp_s1_ptr, "latest_stage1_checkpoint.txt")
        except Exception:
            pass

    # ── Optimisers — separate Generator and Discriminator ─────────────────────
    opt_params  = config["optimizer_params"]
    lr          = float(opt_params.get("lr", 1e-4))
    ft_lr       = float(opt_params.get("ft_lr", 1e-5))
    bert_lr     = float(opt_params.get("bert_lr", 1e-5))

    optimizer_g = torch.optim.AdamW([
        {"params": model["text_encoder"].parameters(),       "lr": ft_lr},
        {"params": model["style_encoder"].parameters(),      "lr": ft_lr},
        {"params": model["decoder"].parameters(),            "lr": ft_lr},
        {"params": model["predictor"].parameters(),          "lr": lr},
        {"params": model["predictor_encoder"].parameters(),  "lr": lr},
        {"params": model["diffusion"].parameters(),          "lr": lr},
        {"params": model["kion_style_adapter"].parameters(), "lr": lr},
        {"params": model["bert"].parameters(),                "lr": bert_lr},
        {"params": model["bert_encoder"].parameters(),       "lr": bert_lr},
    ], betas=(0.9, 0.98), weight_decay=1e-2)

    optimizer_d = torch.optim.AdamW([
        {"params": model["mpd"].parameters(),                "lr": ft_lr},
        {"params": model["msd"].parameters(),                "lr": ft_lr},
    ], betas=(0.9, 0.98), weight_decay=1e-2)

    scheduler_g = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_g, T_max=epochs, eta_min=ft_lr * 0.1
    )
    scheduler_d = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_d, T_max=epochs, eta_min=ft_lr * 0.1
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)

    # Note: Full adversarial SLM rollouts are replaced by direct WavLMLoss (wl)
    # in the training loop below to preserve VRAM on 16GB GPUs like Colab T4.

    # ── Resume Stage 2 if available ───────────────────────────────────────────
    start_epoch = 0
    iters       = 0
    best_loss   = float("inf")
    latest_s2   = _find_latest_stage2_checkpoint()
    if latest_s2:
        print(f"  Resuming Stage 2 from: {latest_s2}")
        # Only sync to HF if this checkpoint was loaded from Google Drive
        if os.path.exists("/content/drive/MyDrive") and latest_s2.startswith("/content/drive"):
            print("\n  [*] Syncing resumed Stage 2 checkpoint from Drive to Hugging Face...", flush=True)
            _upload_to_hf(latest_s2, os.path.basename(latest_s2))
            _upload_to_hf(latest_s2, "kion_stage2_latest.pth")
        ckpt = torch.load(latest_s2, map_location=device)
        for k in model:
            if "net" in ckpt and k in ckpt["net"]:
                model[k].load_state_dict(ckpt["net"][k])
        if "optimizer_g" in ckpt:
            optimizer_g.load_state_dict(ckpt["optimizer_g"])
        if "optimizer_d" in ckpt:
            optimizer_d.load_state_dict(ckpt["optimizer_d"])
        elif "optimizer" in ckpt:
            try:
                optimizer_g.load_state_dict(ckpt["optimizer"])
            except Exception:
                pass
        iters       = ckpt.get("iters", 0)
        steps_per_epoch = len(train_dataloader)
        start_epoch = iters // steps_per_epoch
        step_in_ep  = iters % steps_per_epoch
        best_loss   = ckpt.get("val_loss", float("inf"))
        print(f"  [✓] Resumed state successfully: starting at epoch {start_epoch + 1}, step {iters} ({step_in_ep}/{steps_per_epoch} in current epoch)")

    # ── Loss modules ──────────────────────────────────────────────────────────
    try:
        n_down = model["text_aligner"].module.n_down
    except AttributeError:
        n_down = model["text_aligner"].n_down

    stft_loss   = MultiResolutionSTFTLoss().to(device)
    gl          = GeneratorLoss(model["mpd"], model["msd"]).to(device)
    dl          = DiscriminatorLoss(model["mpd"], model["msd"]).to(device)
    wl          = WavLMLoss(
        model_params.slm.model, model["wd"], sr, model_params.slm.sr
    ).to(device)
    # Freeze WavLM backbone: inputs (y_rec) receive gradients to train prosody/decoder,
    # but 95M WavLM weights skip gradient computation, cutting SLM backward latency ~50%
    # and preventing massive VRAM consumption.
    for param in wl.parameters():
        param.requires_grad = False
    wl.eval()

    print(f"\n  Training from epoch {start_epoch + 1}...\n")
    torch.cuda.empty_cache()

    # ══════════════════════════════════════════════════════════════════════════
    # Stage 2 Training Loop
    # ══════════════════════════════════════════════════════════════════════════
    for epoch in range(start_epoch, epochs):
        epoch_start = time.time()
        _ = [model[k].train() for k in model]
        # Keep frozen utility modules in eval mode
        model["text_aligner"].eval()
        model["pitch_extractor"].eval()
        wl.eval()

        # In Phases 1 & 2, keep decoder and style_encoder frozen at Stage 1 weights
        if epoch < joint_epoch:
            model["decoder"].eval()
            model["style_encoder"].eval()
            for p in model["decoder"].parameters():
                p.requires_grad = False
            for p in model["style_encoder"].parameters():
                p.requires_grad = False
        else:
            model["decoder"].train()
            model["style_encoder"].train()
            for p in model["decoder"].parameters():
                p.requires_grad = True
            for p in model["style_encoder"].parameters():
                p.requires_grad = True

        # Discriminator starts when style diffusion activates (diff_epoch)
        start_ds = (epoch >= diff_epoch)

        steps_per_epoch = len(train_dataloader)
        steps_to_skip = (iters % steps_per_epoch) if epoch == start_epoch else 0
        if steps_to_skip > 0:
            print(f"  [*] Fast-forwarding dataloader to step {steps_to_skip}/{steps_per_epoch} (please wait ~30-60s)...", flush=True)

        pbar = tqdm(
            train_dataloader,
            total=steps_per_epoch,
            initial=steps_to_skip,
            desc=f"Epoch {epoch+1:03d}/{epochs} [Stage 2]",
            leave=False,
        )

        first_step_after_resume = (start_epoch * steps_per_epoch + steps_to_skip + 1)
        for i, batch in enumerate(pbar):
            if epoch == start_epoch and i < steps_to_skip:
                continue

            waves = batch[0]
            batch = [b.to(device, non_blocking=True) for b in batch[1:]]
            texts, input_lengths, ref_texts, ref_lengths, mels, mel_input_length, _ = batch

            try:
                # ── ASR alignment (under no_grad to eliminate 90M aligner graph tracking) ──
                with torch.no_grad():
                    mask      = length_to_mask(mel_input_length // (2 ** n_down)).to(device)
                    text_mask = length_to_mask(input_lengths).to(device)
                    ppgs, s2s_pred, s2s_attn = model["text_aligner"](mels, mask, texts)
                    s2s_attn = s2s_attn.transpose(-1, -2)[..., 1:].transpose(-1, -2)
                    attn_mask = text_mask.unsqueeze(2) | mask.unsqueeze(1)
                    s2s_attn.masked_fill_(attn_mask, 0.0)
                    mask_ST       = mask_from_lens(s2s_attn, input_lengths, mel_input_length // (2 ** n_down))
                    s2s_attn_mono = maximum_path(s2s_attn, mask_ST)

                # ── Text + Prosody encode ─────────────────────────────────
                with torch.amp.autocast("cuda", enabled=use_fp16):
                    t_en  = model["text_encoder"](texts, input_lengths, text_mask)
                    aln   = s2s_attn_mono if random.getrandbits(1) else s2s_attn.detach()
                    asr   = t_en @ aln

                    d_gt = s2s_attn_mono.sum(axis=-1).detach()

                    # ── Utterance-level styles for prosody & diffusion ────
                    mel_lens = mel_input_length.tolist()
                    ss, gs = [], []
                    for bib, ml_raw in enumerate(mel_lens):
                        mel_cur = mels[bib, :, :ml_raw]
                        ss.append(model["predictor_encoder"](mel_cur.unsqueeze(0).unsqueeze(1)))
                        gs.append(model["style_encoder"](mel_cur.unsqueeze(0).unsqueeze(1)))

                    s_dur   = torch.cat(ss, dim=0)   # [B, style_dim]
                    s_audio = torch.cat(gs, dim=0)   # [B, style_dim]
                    s_trg   = torch.cat([s_audio, s_dur], dim=-1).detach()

                    # ── Prosody predictor (full sequence) ─────────────────
                    bert_dur = model["bert"](texts, attention_mask=(~text_mask).int())
                    d_en = model["bert_encoder"](bert_dur).transpose(-1, -2)
                    d, p = model["predictor"](d_en, s_dur, input_lengths, s2s_attn_mono, text_mask)

                    # ── Build random clips for decoder & F0 ───────────────
                    min_mel_len = min(mel_lens)
                    mel_len = min(min_mel_len // 2 - 1, max_len // 2)
                    en, p_en, gt, wav = [], [], [], []

                    for bib, ml_raw in enumerate(mel_lens):
                        ml = ml_raw // 2
                        rs = np.random.randint(0, ml - mel_len)
                        en.append(asr[bib, :, rs:rs + mel_len])
                        p_en.append(p[bib, :, rs:rs + mel_len])
                        gt.append(mels[bib, :, rs * 2:(rs + mel_len) * 2])
                        y = waves[bib][rs * 2 * 300:(rs + mel_len) * 2 * 300]
                        wav.append(torch.from_numpy(np.array(y, copy=True)).to(device))

                    en   = torch.stack(en)
                    p_en = torch.stack(p_en)
                    gt   = torch.stack(gt).detach()
                    wav  = torch.stack(wav).float().detach()

                    if gt.shape[-1] < 80:
                        continue

                    with torch.no_grad():
                        real_norm     = log_norm(gt.unsqueeze(1)).squeeze(1).detach()
                        real_norm     = torch.nan_to_num(real_norm, nan=0.0, posinf=10.0, neginf=-10.0)
                        F0_real, _, _ = model["pitch_extractor"](gt.unsqueeze(1))
                        F0_real       = torch.nan_to_num(F0_real, nan=0.0)

                    # ── Duration & CE loss (vectorized on GPU, zero CPU-GPU sync stalls) ──
                    loss_ce, loss_dur = 0.0, 0.0
                    text_lens = input_lengths.tolist()
                    for _s2s_pred, _text_input, _tl in zip(d, d_gt, text_lens):
                        _s2s_pred_clip = _s2s_pred[:_tl, :]
                        _text_input_clip = _text_input[:_tl].long()
                        max_dur = _s2s_pred_clip.shape[1]
                        _s2s_trg = (torch.arange(max_dur, device=device).unsqueeze(0) < _text_input_clip.unsqueeze(1)).float()
                        _dur_pred = torch.sigmoid(_s2s_pred_clip).sum(dim=1)

                        if _tl > 2:
                            loss_dur += F.l1_loss(
                                _dur_pred[1:_tl - 1],
                                _text_input_clip[1:_tl - 1].float(),
                            )
                        else:
                            loss_dur += F.l1_loss(_dur_pred, _text_input_clip.float())
                        loss_ce += F.binary_cross_entropy_with_logits(_s2s_pred_clip.flatten(), _s2s_trg.flatten())

                    loss_dur /= texts.size(0)
                    loss_ce  /= texts.size(0)

                    # ── F0 / Energy predictor ─────────────────────────────
                    F0_pred, N_pred = model["predictor"].F0Ntrain(p_en, s_dur)
                    loss_F0   = (F.smooth_l1_loss(F0_pred, F0_real)) / 10.0
                    loss_norm = F.smooth_l1_loss(N_pred, real_norm)

                    # ── Decoder (Monitored in true FP32 with clamping) ─────
                    # In Phase 1 & 2 (before joint_epoch), decoder weights are fixed to the
                    # pre-trained Stage 1 acoustic checkpoint. We evaluate loss_mel under no_grad
                    # in pure FP32 with nan_to_num and [-1, 1] clamping to guarantee zero NaN/inf issues.
                    with torch.no_grad(), torch.amp.autocast("cuda", enabled=False):
                        y_rec = model["decoder"](en.float(), F0_real.float(), real_norm.float(), s_audio.float())
                        y_rec_clean = torch.nan_to_num(y_rec.squeeze(1).float(), nan=0.0, posinf=1.0, neginf=-1.0)
                        y_rec_clamped = torch.clamp(y_rec_clean, -1.0, 1.0)
                        loss_mel = stft_loss(y_rec_clamped, wav.detach().float())
                        if torch.isnan(loss_mel) or torch.isinf(loss_mel):
                            loss_mel = torch.tensor(0.0, device=device)

                    # ── GAN & SLM ─────────────────────────────────────────
                    if start_ds:
                        loss_gen_all = gl(wav.detach().unsqueeze(1).float(), y_rec).mean()
                    else:
                        loss_gen_all = torch.tensor(0.0, device=device)

                    # WavLM speech representation loss activates in Phase 3 (joint_epoch)
                    if epoch >= joint_epoch:
                        loss_slm = wl(wav.detach(), y_rec).mean()
                    else:
                        loss_slm = torch.tensor(0.0, device=device)

                    # ── Style diffusion (after diff_epoch) ────────────────
                    loss_diff = torch.tensor(0.0, device=device)
                    if epoch >= diff_epoch:
                        # Diffusion score matching on style latent (acoustic + prosodic style)
                        loss_diff = model["diffusion"](s_trg.unsqueeze(1), embedding=bert_dur).mean()

                    # ── KionStyleAdapter consistency ───────────────────────
                    if len(s_audio) > 1:
                        loss_kion_sty = kion_style_consistency(
                            s_audio[:-1], s_audio[1:]
                        ) * 0.1   # small weight — mostly to regularise
                    else:
                        loss_kion_sty = torch.tensor(0.0, device=device)

                    # ── Total generator loss ──────────────────────────────
                    # In Phases 1 & 2, g_loss strictly trains the prosody predictor, BERT, and adapter.
                    # In Phase 3 (joint_epoch), acoustic mel, GAN, and SLM losses are added for joint fine-tuning.
                    if epoch >= joint_epoch:
                        g_loss = (
                            loss_params.lambda_mel  * loss_mel
                            + loss_params.lambda_F0   * loss_F0
                            + loss_params.lambda_norm * loss_norm
                            + loss_params.lambda_dur  * loss_dur
                            + loss_params.lambda_ce   * loss_ce
                            + loss_params.lambda_gen  * loss_gen_all
                            + loss_params.lambda_slm  * loss_slm
                            + loss_params.lambda_diff * loss_diff
                            + loss_params.get("lambda_kion_style", 0.5) * loss_kion_sty
                        )
                    else:
                        g_loss = (
                            loss_params.lambda_F0   * loss_F0
                            + loss_params.lambda_norm * loss_norm
                            + loss_params.lambda_dur  * loss_dur
                            + loss_params.lambda_ce   * loss_ce
                            + loss_params.lambda_diff * loss_diff
                            + loss_params.get("lambda_kion_style", 0.5) * loss_kion_sty
                        )

                # ── Safety Check: skip NaN/Inf before updating weights ────
                if torch.isnan(g_loss) or torch.isinf(g_loss):
                    iters += 1
                    print(f"\n  [!] Warning: Step {iters} produced NaN/Inf loss (F0={loss_F0.item():.4f}). Skipping.")
                    optimizer_g.zero_grad()
                    if start_ds:
                        optimizer_d.zero_grad()
                    continue

                # ── Generator step ─────────────────────────────────────────
                optimizer_g.zero_grad()
                scaler.scale(g_loss).backward()
                scaler.unscale_(optimizer_g)
                torch.nn.utils.clip_grad_norm_(
                    [p for g in optimizer_g.param_groups for p in g["params"]], 5.0
                )
                scaler.step(optimizer_g)

                # ── Discriminator step (only activated from diff_epoch onward) ──
                if start_ds:
                    optimizer_d.zero_grad()
                    d_loss = dl(wav.detach().unsqueeze(1).float(), y_rec.detach()).mean()
                    scaler.scale(d_loss).backward()
                    scaler.unscale_(optimizer_d)
                    torch.nn.utils.clip_grad_norm_(
                        [p for g in optimizer_d.param_groups for p in g["params"]], 5.0
                    )
                    scaler.step(optimizer_d)
                else:
                    d_loss = torch.tensor(0.0, device=device)

                scaler.update()

                iters += 1
                pbar.set_postfix(
                    step=iters,
                    mel=f"{loss_mel.item():.4f}",
                    F0=f"{loss_F0.item():.4f}",
                    diff=f"{loss_diff.item():.4f}" if epoch >= diff_epoch else "—",
                )

                if iters % 50 == 0 or iters == first_step_after_resume:
                    diff_str = f"{loss_diff.item():.4f}" if epoch >= diff_epoch else "—"
                    dur_val = loss_dur.item() if hasattr(loss_dur, "item") else float(loss_dur)
                    print(f"  [Epoch {epoch+1:02d}/{epochs} | Step {iters:05d}] Mel: {loss_mel.item():.4f} | F0: {loss_F0.item():.4f} | Dur: {dur_val:.4f} | Diff: {diff_str}", flush=True)

                # ── Checkpoint every save_step_interval (900) steps ──
                if iters % save_step_interval == 0:
                    state = {
                        "net":         {k: model[k].state_dict() for k in model},
                        "optimizer_g": optimizer_g.state_dict(),
                        "optimizer_d": optimizer_d.state_dict(),
                        "iters":       iters,
                        "val_loss":    loss_mel.item(),
                        "epoch":       epoch,
                    }
                    _save_to_drive(state, epoch=epoch + 1, step=iters, stage="stage2", step_interval=save_step_interval)
                    del state
                    import gc
                    gc.collect()

                # TensorBoard
                if (i + 1) % 20 == 0:
                    writer.add_scalar("s2/mel_loss",  loss_mel.item(),      iters)
                    writer.add_scalar("s2/F0_loss",   loss_F0.item(),       iters)
                    writer.add_scalar("s2/dur_loss",  loss_dur.item() if hasattr(loss_dur, "item") else loss_dur, iters)
                    writer.add_scalar("s2/diff_loss", loss_diff.item(),     iters)
                    writer.add_scalar("s2/gen_loss",  loss_gen_all.item(),  iters)
                    writer.add_scalar("s2/disc_loss", d_loss.item(),        iters)

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    torch.cuda.empty_cache()
                    print(f"  [OOM] Step {i} skipped — clearing CUDA cache.")
                    continue
                raise
            finally:
                # Guaranteed cleanup on every single step to eliminate System RAM accumulation
                del waves, batch
                if "en" in locals():
                    del en
                if "p_en" in locals():
                    del p_en
                if "gt" in locals():
                    del gt
                if "wav" in locals():
                    del wav
                if (i + 1) % 50 == 0:
                    import gc
                    gc.collect()

        # ── Validation ────────────────────────────────────────────────────
        _ = [model[k].eval() for k in model]
        val_loss = 0.0
        n_val    = 0

        with torch.no_grad():
            for batch in val_dataloader:
                waves = batch[0]
                batch = [b.to(device, non_blocking=True) for b in batch[1:]]
                texts, input_lengths, _, _, mels, mel_input_length, _ = batch

                mask      = length_to_mask(mel_input_length // (2 ** n_down)).to(device)
                text_mask = length_to_mask(input_lengths).to(device)
                t_en      = model["text_encoder"](texts, input_lengths, text_mask)
                _, _, s2s_attn = model["text_aligner"](mels, mask, texts)
                s2s_attn  = s2s_attn.transpose(-1, -2)[..., 1:].transpose(-1, -2)
                asr       = t_en @ s2s_attn

                val_mel_lens = mel_input_length.tolist()
                mel_len      = min(min(val_mel_lens) // 2 - 1, max_len // 2)
                en, gt, wav_v = [], [], []
                for bib, ml_raw in enumerate(val_mel_lens):
                    ml = ml_raw // 2
                    rs = np.random.randint(0, ml - mel_len)
                    en.append(asr[bib, :, rs:rs + mel_len])
                    gt.append(mels[bib, :, rs * 2:(rs + mel_len) * 2])
                    y = waves[bib][rs * 2 * 300:(rs + mel_len) * 2 * 300]
                    wav_v.append(torch.from_numpy(y).to(device, non_blocking=True))

                en    = torch.stack(en)
                gt    = torch.stack(gt).detach()
                wav_v = torch.stack(wav_v).float().detach()

                F0_real, _, _ = model["pitch_extractor"](gt.unsqueeze(1))
                s             = model["style_encoder"](gt.unsqueeze(1))
                real_norm     = log_norm(gt.unsqueeze(1)).squeeze(1)
                y_rec         = model["decoder"](en, F0_real, real_norm, s)
                loss_mel      = stft_loss(y_rec.squeeze(), wav_v.detach())
                val_loss     += loss_mel.item()
                n_val        += 1
                del waves, batch, en, gt, wav_v

        val_loss /= max(n_val, 1)
        is_best   = val_loss < best_loss
        if is_best:
            best_loss = val_loss

        elapsed = time.time() - epoch_start
        print(
            f"  Epoch {epoch+1:03d} | val_mel={val_loss:.4f}"
            f" {'★BEST' if is_best else '     '}"
            f" | diff={'ON' if epoch >= diff_epoch else 'off'}"
            f" | joint={'ON' if epoch >= joint_epoch else 'off'}"
            f" | {elapsed:.0f}s"
        )
        writer.add_scalar("s2/val_mel_loss", val_loss, epoch + 1)

        # Write sample audio to Google Drive and TensorBoard every epoch
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
                writer.add_audio(f"s2/synth_{bib}", audio_arr, epoch + 1, sample_rate=sr)
                try:
                    import soundfile as sf
                    wav_path = os.path.join(sample_dir, f"kion_stage2_epoch_{epoch+1:03d}_sample_{bib+1}.wav")
                    sf.write(wav_path, audio_arr, sr)
                except Exception:
                    pass
        print(f"  [♫] Saved {min(3, len(en))} audio samples → {sample_dir}")

        scheduler_g.step()
        scheduler_d.step()

        # Save checkpoint every 1 epoch
        state = {
            "net":         {k: model[k].state_dict() for k in model},
            "optimizer_g": optimizer_g.state_dict(),
            "optimizer_d": optimizer_d.state_dict(),
            "iters":       iters,
            "val_loss":    val_loss,
            "epoch":       epoch,
        }
        _save_to_drive(state, epoch + 1, is_best=is_best, stage="stage2")

    # ── Final save ────────────────────────────────────────────────────────────
    print("\n[+] Stage 2 training complete!")
    final_path = os.path.join(DRIVE_CKPT_DIR, "kion_stage2_final.pth")
    torch.save({
        "net":         {k: model[k].state_dict() for k in model},
        "optimizer_g": optimizer_g.state_dict(),
        "optimizer_d": optimizer_d.state_dict(),
        "iters":       iters,
        "val_loss":    best_loss,
        "epoch":       epochs,
        "config":      config,
    }, final_path)
    print(f"  Final checkpoint → {final_path}")
    _upload_to_hf(final_path, "kion_stage2_final.pth")
    print("  Proceed to Cell 08 for inference testing.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="KionTTS Stage 2 Training & Hugging Face Checkpoint Sync")
    parser.add_argument("--upload-only", action="store_true", help="Upload existing Drive checkpoints to Hugging Face and exit")
    parser.add_argument("--repo-id", default=HF_REPO_ID, help="Target Hugging Face Model Repository ID")
    parser.add_argument("--token", default=None, help="Hugging Face write token (e.g. --token $hf_token)")
    parser.add_argument("--config", default=CONFIG_PATH, help="Path to kion_config.yml")
    args = parser.parse_args()

    if args.token:
        os.environ["HF_TOKEN"] = args.token.strip()
        HF_TOKEN = args.token.strip()

    if args.upload_only:
        sync_drive_checkpoints_to_hf(repo_id=args.repo_id)
    else:
        run_stage2_training(config_path=args.config)
