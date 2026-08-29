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
        "phonemizer>=3.2.1",
        "librosa>=0.10.0",
        "soundfile>=0.12.1",
        "pyworld>=0.3.4",
        "torchaudio",
        "transformers>=4.36.0",
        "accelerate>=0.26.0",
        "tqdm",
        "pyyaml",
        "scipy",
        "matplotlib",
    ]
    cmd_pip = f"{sys.executable} -m pip install -q " + " ".join(packages)
    subprocess.run(cmd_pip, shell=True, check=True)
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
