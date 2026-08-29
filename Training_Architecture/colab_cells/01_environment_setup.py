"""
Colab Cell 01: Environment Setup & Dependencies Installation
Installs required audio libraries, espeak-ng phonemizer backend,
and mounts Google Drive for persistent checkpointing.
"""

import os
import sys
import subprocess


def check_gpu():
    print("=" * 50)
    print("Checking GPU Environment...")
    try:
        import torch

        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"CUDA Available: YES")
            print(f"Device: {device_name}")
            print(f"Total VRAM: {vram_gb:.2f} GB")
        else:
            print("WARNING: CUDA is NOT available. Running on CPU.")
    except ImportError:
        print("PyTorch not installed yet.")
    print("=" * 50)


def install_dependencies():
    print("\nInstalling system & Python dependencies...")
    # System dependencies for phonemizer & audio
    cmd_apt = "apt-get update -qq && apt-get install -y -qq espeak-ng libsndfile1 ffmpeg"
    subprocess.run(cmd_apt, shell=True, check=True)

    # Python packages
    packages = [
        # ── Audio & Signal Processing ──────────────────────────────────────
        "phonemizer>=3.2.1",
        "librosa>=0.10.0",
        "soundfile>=0.12.1",
        "pyworld>=0.3.4",
        "praat-parselmouth",          # pitch extraction (Parselmouth / Praat)
        "torchaudio",
        # ── ML / Training Utilities ────────────────────────────────────────
        "transformers>=4.36.0",
        "accelerate>=0.26.0",
        "einops",                     # tensor rearrangement (used in diffusion)
        "munch",                      # dot-access dicts (required by StyleTTS2)
        # ── Alignment ─────────────────────────────────────────────────────
        "cython",                     # required to build monotonic_align
        # ── Misc ──────────────────────────────────────────────────────────
        "tqdm",
        "pyyaml",
        "scipy",
        "matplotlib",
        "tensorboard",                # TensorBoard logging
    ]
    cmd_pip = f"{sys.executable} -m pip install -q " + " ".join(packages)
    subprocess.run(cmd_pip, shell=True, check=True)

    # Build monotonic_align Cython extension (required by StyleTTS2 maximum_path)
    import subprocess as _sp
    import os as _os
    repo_root = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "../.."))
    styletts2_root = _os.path.join(repo_root, "StyleTTS2")
    if not _os.path.exists(styletts2_root):
        # Try common Colab paths
        for _p in ["/content/KionTTS/StyleTTS2", "/content/Kiontts/StyleTTS2", "/content/kiontts/StyleTTS2"]:
            if _os.path.exists(_p):
                styletts2_root = _p
                break
    monotonic_src = _os.path.join(styletts2_root, "monotonic_align")
    if _os.path.isdir(monotonic_src):
        _sp.run(
            f"cd {monotonic_src} && python setup.py build_ext --inplace -q",
            shell=True, check=False,
        )
        print(f"[+] monotonic_align Cython extension built from: {monotonic_src}")
    else:
        print(f"[!] monotonic_align source not found at {monotonic_src} — skipping build.")
    print("Dependencies installed successfully!")


def setup_drive_and_directories():
    print("\nSetting up Google Drive & working directories...")
    drive_mounted = os.path.exists("/content/drive/MyDrive")

    if not drive_mounted:
        try:
            from google.colab import drive
            drive.mount("/content/drive")
            drive_mounted = os.path.exists("/content/drive/MyDrive")
            if drive_mounted:
                print("Google Drive mounted successfully at /content/drive")
        except Exception:
            print("[*] Note: To mount Google Drive in Colab, run in an interactive cell:")
            print("    from google.colab import drive; drive.mount('/content/drive')")

    # Define standard directories
    dirs = [
        "/content/dataset",
        "/content/dataset/wavs",
        "/content/preprocessed_data",
    ]
    if os.path.exists("/content/drive"):
        dirs.extend([
            "/content/drive/MyDrive/KionTTS_Checkpoints",
            "/content/drive/MyDrive/KionTTS_Checkpoints/eval_samples",
        ])

    for d in dirs:
        try:
            os.makedirs(d, exist_ok=True)
            print(f"Directory ready: {d}")
        except Exception as e:
            print(f"Notice: Directory {d} could not be created yet: {e}")


if __name__ == "__main__":
    check_gpu()
    install_dependencies()
    setup_drive_and_directories()
    print("\n[Cell 01 Complete] Environment is ready for KionTTS.")
