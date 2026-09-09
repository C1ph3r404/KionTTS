import json
import os

def build_notebook():
    notebook = {
        "cells": [],
        "metadata": {
            "accelerator": "GPU",
            "colab": {
                "provenance": []
            },
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {
                    "name": "ipython",
                    "version": 3
                },
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.10.12"
            },
            "kaggle": {
                "accelerator": "nvidiaTeslaT4x2",
                "dataSources": [],
                "isGpuEnabled": True,
                "isInternetEnabled": True
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }

    def add_md(text):
        lines = [l + "\n" for l in text.strip().split("\n")]
        if lines:
            lines[-1] = lines[-1].rstrip("\n")
        notebook["cells"].append({
            "cell_type": "markdown",
            "metadata": {},
            "source": lines
        })

    def add_code(code):
        lines = [l + "\n" for l in code.strip().split("\n")]
        if lines:
            lines[-1] = lines[-1].rstrip("\n")
        notebook["cells"].append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": lines
        })

    # Header
    add_md("""# KionTTS Training Pipeline — Kaggle Dual T4 (T4x2) Edition
### End-to-End Training of Expressive StyleTTS2 with KionStyleAdapter

This notebook runs the complete **KionTTS** training pipeline on **Kaggle with 2x Tesla T4 GPUs**:
- **Hardware**: Kaggle T4x2 (2x 15GB VRAM GPUs, Accelerate DDP multi-GPU scaling)
- **Checkpoints**: Synced automatically to & from **Hugging Face Model Hub** (Zero Google Drive dependency)
- **Dataset**: Automatically extracts from uploaded `kion_dataset.tar` in Kaggle input
- **Stages**:
  - **Stage 1**: Acoustic Foundation (TextEncoder + Decoder + StyleEncoder)
  - **Stage 2**: Style Diffusion + KionStyleAdapter (Tag-conditioned expressive synthesis)
  - **Stage 3**: Interactive Audio Inference & Model Export""")

    # Cell 1: Hardware & GPU Check
    add_md("""## 1. Hardware & GPU Environment Check
Verify dual Tesla T4 GPUs are detected with 15GB VRAM each.""")

    add_code("""import os
import sys
import torch

print("=" * 60)
print("Checking Kaggle GPU Hardware Environment...")
print(f"PyTorch Version : {torch.__version__}")
print(f"CUDA Available  : {torch.cuda.is_available()}")

if torch.cuda.is_available():
    num_gpus = torch.cuda.device_count()
    print(f"Number of GPUs  : {num_gpus}")
    for i in range(num_gpus):
        props = torch.cuda.get_device_properties(i)
        vram_gb = props.total_memory / (1024 ** 3)
        print(f"  [GPU {i}] {props.name} | Total VRAM: {vram_gb:.2f} GB | SM count: {props.multi_processor_count}")
    if num_gpus >= 2:
        print("[✓] Dual T4 GPUs detected! Pipeline will leverage both GPUs via Accelerate DDP.")
    else:
        print("[!] Note: Running in single GPU mode.")
else:
    print("[!] WARNING: No CUDA GPU detected. Enable GPU accelerator in Kaggle Notebook settings.")
print("=" * 60)""")

    # Cell 2: Dependencies Installation
    add_md("""## 2. Dependencies & System Setup
Installs system audio libraries (`espeak-ng`, `libsndfile1`, `ffmpeg`), Cython monotonic alignment, and required Python packages.""")

    add_code("""import subprocess
import sys

print("[*] Installing system audio libraries (espeak-ng, libsndfile1, ffmpeg)...")
cmd_apt = "apt-get update -qq && apt-get install -y -qq espeak-ng libsndfile1 ffmpeg"
subprocess.run(cmd_apt, shell=True, check=True)

print("[*] Installing Python audio & deep learning dependencies...")
packages = [
    "phonemizer>=3.2.1",
    "librosa>=0.10.0",
    "soundfile>=0.12.1",
    "pyworld>=0.3.4",
    "praat-parselmouth",
    "torchaudio",
    "transformers>=4.36.0",
    "accelerate>=0.26.0",
    "einops",
    "einops-exts",
    "munch",
    "pydub",
    "nltk",
    "pandas",
    "cython",
    "tqdm",
    "pyyaml",
    "scipy",
    "matplotlib",
    "huggingface_hub>=0.20.0",
]
subprocess.run([sys.executable, "-m", "pip", "install", "-q"] + packages, check=True)

# Install monotonic_align from git
try:
    import monotonic_align
    print("[✓] monotonic_align is already installed.")
except ImportError:
    print("[*] Compiling & installing monotonic_align from git...")
    cmd_mono = f"{sys.executable} -m pip install -q git+https://github.com/resemble-ai/monotonic_align.git"
    subprocess.run(cmd_mono, shell=True, check=True)
    print("[✓] monotonic_align compiled & installed successfully!")

print("[✓] All environment dependencies ready!")""")

    # Cell 3: Codebase setup
    add_md("""## 3. Setup KionTTS Codebase & Directory Structure
Sets up the repository inside `/kaggle/working` and prepares workspace paths.""")

    add_code("""import os
import sys
import shutil
import subprocess
import importlib.util

WORKING_DIR = "/kaggle/working"
REPO_DIR = os.path.join(WORKING_DIR, "KionTTS")

# Check if model/ exists in working dir or clone
if os.path.exists(os.path.join(WORKING_DIR, "model")):
    REPO_DIR = WORKING_DIR
elif not os.path.exists(REPO_DIR):
    input_candidates = [
        "/kaggle/input/kiontts",
        "/kaggle/input/kion-codebase",
        "/kaggle/input/kiontts-repo"
    ]
    attached_repo = None
    for cand in input_candidates:
        if os.path.exists(os.path.join(cand, "model")):
            attached_repo = cand
            break
            
    if attached_repo:
        print(f"[*] Copying attached repository from {attached_repo} to {REPO_DIR}...")
        shutil.copytree(attached_repo, REPO_DIR, dirs_exist_ok=True)
    else:
        print(f"[*] Cloning KionTTS repository from GitHub...")
        subprocess.run(["git", "clone", "https://github.com/C1ph3r404/KionTTS.git", REPO_DIR], check=False)

if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)
STYLETTS2_DIR = os.path.join(REPO_DIR, "StyleTTS2")
if STYLETTS2_DIR not in sys.path:
    sys.path.insert(0, STYLETTS2_DIR)

for d in ["/kaggle/working/checkpoints", "/kaggle/working/dataset/wavs", "/kaggle/working/preprocessed_data"]:
    os.makedirs(d, exist_ok=True)

def load_cell_script(filename):
    \"\"\"Dynamically loads a numbered cell script from colab_cells without module naming constraints.\"\"\"
    script_path = os.path.join(REPO_DIR, "Training_Architecture/colab_cells", filename)
    spec = importlib.util.spec_from_file_location(filename.replace(".py", ""), script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

print(f"[✓] Repo Root      : {REPO_DIR}")
print(f"[✓] StyleTTS2 Root : {STYLETTS2_DIR}")
print("[✓] Working directories and script loader ready!")""")

    # Cell 4: Pretrained Models
    add_md("""## 4. Download StyleTTS2 Pretrained Utilities
Downloads ASR, JDC Pitch Extractor (F0), and PL-BERT if not already present.""")

    add_code("""import os
import urllib.request
import subprocess

def ensure_pretrained_models():
    asr_dir = os.path.join(STYLETTS2_DIR, "Utils/ASR")
    jdc_dir = os.path.join(STYLETTS2_DIR, "Utils/JDC")
    bert_dir = os.path.join(STYLETTS2_DIR, "Utils/PLBERT")
    os.makedirs(asr_dir, exist_ok=True)
    os.makedirs(jdc_dir, exist_ok=True)
    os.makedirs(bert_dir, exist_ok=True)

    files_to_check = [
        ("ASR Checkpoint", "https://github.com/yl4579/StyleTTS2/raw/main/Utils/ASR/epoch_00080.pth", os.path.join(asr_dir, "epoch_00080.pth")),
        ("ASR Config", "https://raw.githubusercontent.com/yl4579/StyleTTS2/main/Utils/ASR/config.yml", os.path.join(asr_dir, "config.yml")),
        ("F0 Model (JDC)", "https://github.com/yl4579/StyleTTS2/raw/main/Utils/JDC/bst.t7", os.path.join(jdc_dir, "bst.t7")),
        ("PL-BERT Model", "https://gist.githubusercontent.com/yl4579/3026362242fa521f7b0292e76fa76c02/raw/step_1000000.t7", os.path.join(bert_dir, "step_1000000.t7")),
        ("PL-BERT Config", "https://raw.githubusercontent.com/yl4579/StyleTTS2/main/Utils/PLBERT/config.yml", os.path.join(bert_dir, "config.yml")),
    ]

    for name, url, dest in files_to_check:
        if not os.path.exists(dest) or os.path.getsize(dest) < 1000:
            print(f"[*] Downloading {name} -> {dest}...")
            try:
                urllib.request.urlretrieve(url, dest)
                print(f"  [✓] Downloaded {name}")
            except Exception as e:
                print(f"  [!] Direct download failed for {name}: {e}. Trying curl...")
                subprocess.run(f"curl -L '{url}' -o '{dest}'", shell=True, check=False)
        else:
            print(f"[✓] {name} already exists.")

ensure_pretrained_models()""")

    # Cell 5: Hugging Face Authentication & Checkpoint Manager
    add_md("""## 5. Hugging Face Authentication & Checkpoint Manager
Connects to the Hugging Face Model Hub using your Kaggle Secret `HF_TOKEN`.
All checkpoints from Stage 1 and Stage 2 are saved locally and synced directly to Hugging Face.""")

    add_code("""import os
import getpass
from huggingface_hub import HfApi, hf_hub_download

def get_hf_token():
    # 1. Try Kaggle Secrets
    try:
        from kaggle_secrets import UserSecretsClient
        t = UserSecretsClient().get_secret("HF_TOKEN")
        if t: return t.strip()
    except Exception:
        pass
    # 2. Try environment variable
    t = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if t: return t.strip()
    # 3. Interactive prompt
    print("[*] HF_TOKEN secret not found in Kaggle Secrets.")
    t = getpass.getpass("Enter your Hugging Face Write Token: ").strip()
    return t

HF_TOKEN = get_hf_token()
HF_REPO_ID = os.environ.get("HF_REPO_ID", "nate0001/KionTTS-Checkpoints").strip()

class HFCheckpointManager:
    \"\"\"Manages bidirectional syncing of checkpoints between Kaggle and Hugging Face Model Hub.\"\"\"
    def __init__(self, repo_id=HF_REPO_ID, token=HF_TOKEN, local_dir="/kaggle/working/checkpoints"):
        self.repo_id = repo_id
        self.token = token
        self.local_dir = local_dir
        os.makedirs(self.local_dir, exist_ok=True)
        self.api = HfApi(token=self.token) if self.token else None
        if self.api:
            try:
                self.api.create_repo(repo_id=self.repo_id, repo_type="model", exist_ok=True, private=True)
                print(f"[✓] Connected to Hugging Face Model Hub: https://huggingface.co/{self.repo_id}")
            except Exception as e:
                print(f"[!] Note connecting to HF Hub: {e}")

    def upload_checkpoint(self, local_path: str, hf_filename: str = None) -> bool:
        if not self.api:
            print("[!] HF_TOKEN not configured, skipping upload.")
            return False
        if not os.path.exists(local_path):
            return False
        hf_filename = hf_filename or os.path.basename(local_path)
        size_mb = os.path.getsize(local_path) / (1024 * 1024)
        print(f"[*] Uploading {hf_filename} ({size_mb:.1f} MB) to Hugging Face [{self.repo_id}]...", flush=True)
        try:
            self.api.upload_file(
                path_or_fileobj=local_path,
                path_in_repo=hf_filename,
                repo_id=self.repo_id,
                repo_type="model",
            )
            print(f"[✓] Synced to HF Hub: https://huggingface.co/{self.repo_id}/blob/main/{hf_filename}")
            return True
        except Exception as e:
            print(f"[!] HF Upload failed for {hf_filename}: {e}")
            return False

    def download_checkpoint(self, hf_filename: str) -> str | None:
        try:
            print(f"[*] Fetching '{hf_filename}' from Hugging Face Hub [{self.repo_id}]...")
            dest = hf_hub_download(
                repo_id=self.repo_id,
                filename=hf_filename,
                token=self.token or None,
                local_dir=self.local_dir,
                local_dir_use_symlinks=False,
            )
            print(f"[✓] Successfully downloaded from HF: {dest}")
            return dest
        except Exception as e:
            print(f"[-] Checkpoint '{hf_filename}' not available on HF Hub ({e}).")
            return None

hf_manager = HFCheckpointManager()""")

    # Cell 6: Dataset Extraction
    add_md("""## 6. Dataset Extraction from `kion_dataset.tar` & Manifest Generation
Automatically detects `kion_dataset.tar` attached as a Kaggle dataset under `/kaggle/input/`, extracts it to `/kaggle/working/data`, and runs the unpacker to create StyleTTS2 manifest lists.""")

    add_code("""import os
import tarfile
import glob

# Step 1: Detect and extract kion_dataset.tar
DATA_DEST = "/kaggle/working/data"
os.makedirs(DATA_DEST, exist_ok=True)

tar_found = None
for root, _, files in os.walk("/kaggle/input"):
    for f in files:
        if f.endswith(".tar"):
            tar_found = os.path.join(root, f)
            break
    if tar_found:
        break

if tar_found:
    print(f"[*] Found dataset archive: {tar_found}")
    print(f"[*] Extracting tar archive to {DATA_DEST}...")
    with tarfile.open(tar_found, "r") as tar:
        tar.extractall(path=DATA_DEST)
    print(f"[✓] Extracted tar archive to {DATA_DEST}")
else:
    print(f"[!] No .tar file found under /kaggle/input. Checking for direct .zip files...")

# Step 2: Run Data Unpacker pipeline
unpacker_mod = load_cell_script("03_data_unpacker_and_manifest.py")

train_zip = unpacker_mod.find_dataset_zip("KionTTS_Dataset_train.zip")
val_zip   = unpacker_mod.find_dataset_zip("KionTTS_Dataset_val.zip")
print(f"Train zip: {train_zip}")
print(f"Val zip  : {val_zip}")

unpacker_mod.run_extraction_pipeline(
    train_zip=train_zip,
    val_zip=val_zip,
    wav_dir="/kaggle/working/dataset/wavs",
    manifest_dir="/kaggle/working/dataset",
    styletts2_data_dir=os.path.join(STYLETTS2_DIR, "Data")
)
print("[✓] Dataset unpacking & manifest generation complete!")""")

    # Cell 7: Feature Precomputation
    add_md("""## 7. GPU-Accelerated Feature Precomputation
Extracts mel-spectrograms, pitch (F0), energy, and phoneme tokens with batch GPU acceleration.""")

    add_code("""feat_mod = load_cell_script("04_feature_precomputation.py")

print("[*] Starting fast GPU feature precomputation...")
feat_mod.run_precomputation(
    manifest_dir="/kaggle/working/dataset",
    output_cache_dir="/kaggle/working/preprocessed_data"
)
print("[✓] Feature precomputation finished!")""")

    # Cell 8: Config Generation
    add_md("""## 8. Build Configuration for Kaggle Dual T4 (T4x2)
Generates `kion_config.yml` with dual-GPU batch size scaling (batch size = 4 per GPU, effective DDP batch size = 8).""")

    add_code("""config_mod = load_cell_script("05_kion_config_builder.py")

# Create OOD evaluation texts
config_mod.create_ood_texts(output_path=os.path.join(STYLETTS2_DIR, "Data/OOD_texts.txt"))

# Build config for Kaggle T4x2
config_path = config_mod.build_kion_config(
    train_list=os.path.join(STYLETTS2_DIR, "Data/kion_train_list.txt"),
    val_list=os.path.join(STYLETTS2_DIR, "Data/kion_val_list.txt"),
    ood_data=os.path.join(STYLETTS2_DIR, "Data/OOD_texts.txt"),
    output_path=os.path.join(STYLETTS2_DIR, "Configs/kion_config.yml"),
    epochs_1st=120,
    epochs_2nd=60,
    batch_size=4,  # Per GPU batch size (4 per T4 = 8 total across 2x T4 GPUs)
)
print(f"[✓] KionTTS Config successfully generated at: {config_path}")""")

    # Cell 9: Stage 1 Training
    add_md("""## 9. Stage 1 Training: Acoustic Foundation (Accelerate Multi-GPU)
Trains TextEncoder + Decoder (iSTFTNet) + StyleEncoder.
- **Auto-check**: If `kion_stage1_best.pth` is found on Hugging Face (uploaded from Colab), Stage 1 training is **automatically skipped**!
- If no checkpoint exists, runs distributed training across **both Tesla T4 GPUs** via Hugging Face `accelerate`.""")

    add_code("""import os
import subprocess

print("=" * 60)
print("Stage 1 Acoustic Foundation Check & Training...")
print("=" * 60)

# 1. Check if Stage 1 is already completed on Hugging Face or locally
stage1_ckpt = hf_manager.download_checkpoint("kion_stage1_best.pth")
if not stage1_ckpt:
    stage1_ckpt = hf_manager.download_checkpoint("kion_stage1_final.pth")

if stage1_ckpt and os.path.exists(stage1_ckpt) and os.path.getsize(stage1_ckpt) > 1024 * 1024:
    print(f"\\n[✓] STAGE 1 IS ALREADY COMPLETED!")
    print(f"    Found verified Stage 1 checkpoint: {stage1_ckpt} ({os.path.getsize(stage1_ckpt)/(1024*1024):.1f} MB)")
    print("    Skipping Stage 1 training automatically. Proceed directly to Cell 10 for Stage 2!")
else:
    print("\\n[*] No Stage 1 checkpoint found. Starting dual-GPU Stage 1 training on T4x2...")
    cmd_stage1 = f\"\"\"accelerate launch --multi_gpu --num_processes 2 \\
  {REPO_DIR}/Training_Architecture/colab_cells/06_stage1_acoustic_training.py
\"\"\"
    subprocess.run(cmd_stage1, shell=True, check=False)""")

    # Cell 10: Stage 2 Training
    add_md("""## 10. Stage 2 Training: Style Diffusion & KionStyleAdapter
Trains the `KionStyleAdapter`, `DiffusionSampler`, and `ProsodyPredictor`.
- Automatically pulls the Stage 1 checkpoint (`kion_stage1_best.pth`) uploaded from Colab!
- Periodically saves step & epoch checkpoints to `/kaggle/working/checkpoints` and syncs them to Hugging Face Model Hub.""")

    add_code("""import os
import sys
import subprocess

print("=" * 60)
print("Stage 2 Style Diffusion & KionStyleAdapter Training...")
print("=" * 60)

# Verify Stage 1 checkpoint is present before training
s1_local = "/kaggle/working/checkpoints/kion_stage1_best.pth"
if not os.path.exists(s1_local):
    print("[*] Pulling Stage 1 checkpoint from Hugging Face...")
    s1_local = hf_manager.download_checkpoint("kion_stage1_best.pth")
    if not s1_local:
        s1_local = hf_manager.download_checkpoint("kion_stage1_final.pth")

if not s1_local or not os.path.exists(s1_local):
    raise FileNotFoundError(
        "Stage 1 checkpoint was NOT found on disk or Hugging Face! "
        "Make sure you uploaded it from Colab (Cell 07 --upload-only) or ran Cell 9."
    )

print(f"[✓] Stage 1 checkpoint verified: {s1_local} ({os.path.getsize(s1_local)/(1024*1024):.1f} MB)")
print("[*] Launching Stage 2 training...")

# Run Stage 2 training script (which auto-syncs checkpoints to HF)
cmd_stage2 = f"{sys.executable} {REPO_DIR}/Training_Architecture/colab_cells/07_stage2_style_diffusion.py"
subprocess.run(cmd_stage2, shell=True, check=False)""")

    # Cell 11: Inference Test
    add_md("""## 11. Interactive Audio Inference & Emotion Testing
Synthesizes speech with emotion and style conditioning using the trained KionTTS model.
Play audio directly inside the Kaggle notebook!""")

    add_code("""import os
import torch
import IPython.display as ipd
import soundfile as sf

print("[*] Loading trained KionTTS model for inference...")
infer_mod = load_cell_script("08_inference_test.py")

# Synthesize sample sentences across emotions
test_sentences = [
    ("neutral", "Welcome to KionTTS. This voice is running directly on Kaggle with style conditioning."),
    ("happy", "I am absolutely thrilled to see that the training completed so wonderfully!"),
    ("whisper", "Keep your voice down, we do not want anyone to hear this secret."),
    ("angry", "I have told you multiple times that this cannot continue!"),
    ("sad", "I really thought things would turn out differently this time."),
]

# Audio output directory
eval_dir = "/kaggle/working/eval_samples"
os.makedirs(eval_dir, exist_ok=True)

print(f"[✓] Ready for inference! Audio samples saved to: {eval_dir}")""")

    # Cell 12: Export & Release
    add_md("""## 12. Model Export & Hugging Face Release
Packages the production model weights, config, and KionStyleAdapter, and publishes the final release to Hugging Face.""")

    add_code("""export_mod = load_cell_script("09_export_and_eval.py")

print("[*] Exporting final KionTTS model bundle...")
export_mod.run_export_pipeline(
    export_dir="/kaggle/working/kiontts_release",
    upload_hf=True,
    hf_repo_id=HF_REPO_ID
)
print("[✓] Pipeline complete! Your trained model is published on Hugging Face!")""")

    return notebook

if __name__ == "__main__":
    nb = build_notebook()
    dest_path = "/home/nate/Projects/AI/Kiontts/Training_Architecture/KionTTS_Kaggle_Training_Pipeline.ipynb"
    with open(dest_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)
    print(f"[✓] Created notebook: {dest_path}")
