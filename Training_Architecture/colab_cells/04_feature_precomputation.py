"""
Colab Cell 04: Feature Precomputation
Extracts phoneme sequences, 80-band 24kHz Mel-spectrograms, Pitch (F0),
and Energy contours, caching them to local disk for fast training DataLoader access.
"""

import os
import json
import torch
import librosa
import numpy as np
from tqdm import tqdm
from typing import Dict, Any, List, Tuple

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from model.data.phonemizer_util import KionPhonemizer


# Audio & Spectrogram Parameters
SAMPLE_RATE = 24000
N_FFT = 1024
HOP_LENGTH = 256
WIN_LENGTH = 1024
N_MELS = 80
F_MIN = 0.0
F_MAX = 8000.0


def extract_mel(y: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Computes 80-band log Mel-spectrogram."""
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        n_mels=N_MELS,
        fmin=F_MIN,
        fmax=F_MAX,
    )
    log_mel = np.log(np.clip(mel, a_min=1e-5, a_max=None))
    return log_mel


def extract_pitch_and_energy(y: np.ndarray, target_length: int) -> Tuple[np.ndarray, np.ndarray]:
    """Computes F0 pitch contour and frame-level energy."""
    # F0 extraction via pyin
    f0, _, _ = librosa.pyin(
        y,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C7"),
        sr=SAMPLE_RATE,
        hop_length=HOP_LENGTH,
    )
    f0 = np.nan_to_num(f0)
    # Log-F0 normalization
    nonzero_f0 = f0[f0 > 0]
    if len(nonzero_f0) > 0:
        log_f0 = np.zeros_like(f0)
        log_f0[f0 > 0] = np.log(nonzero_f0)
    else:
        log_f0 = np.zeros_like(f0)

    # Frame energy (L2 norm)
    energy = np.array(
        [
            np.sqrt(np.mean(y[i * HOP_LENGTH : i * HOP_LENGTH + WIN_LENGTH] ** 2))
            for i in range(target_length)
        ]
    )

    # Interpolate / match length
    if len(log_f0) < target_length:
        log_f0 = np.pad(log_f0, (0, target_length - len(log_f0)))
    else:
        log_f0 = log_f0[:target_length]

    return log_f0.astype(np.float32), energy.astype(np.float32)


def precompute_dataset(
    manifest_path: str,
    output_dir: str,
    output_manifest_path: str,
):
    print(f"\nPrecomputing features from: {manifest_path}")
    print(f"Output Feature Cache: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    with open(manifest_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    phonemizer = KionPhonemizer()
    preprocessed_records = []

    for item in tqdm(records, desc="Precomputing features"):
        uid = item["id"]
        wav_path = item["wav_path"]
        clean_text = item["clean_text"]
        style_vector = item["style_vector"]

        if not os.path.exists(wav_path):
            continue

        try:
            # 1. Phonemize text
            tokens = phonemizer.text_to_sequence(clean_text)
            if len(tokens) == 0:
                continue

            # 2. Load audio
            y, sr = librosa.load(wav_path, sr=SAMPLE_RATE)
            if len(y) < HOP_LENGTH:
                continue

            # 3. Extract Mel
            mel = extract_mel(y, sr)  # (80, T_frames)
            num_frames = mel.shape[1]

            # 4. Extract Pitch & Energy
            pitch, energy = extract_pitch_and_energy(y, num_frames)

            # 5. Estimate linear durations for bootstrapping
            durations = np.full(len(tokens), max(1, num_frames // len(tokens)), dtype=np.int64)
            rem = num_frames - np.sum(durations)
            if rem > 0:
                durations[-1] += rem
            elif rem < 0:
                durations[-1] = max(1, durations[-1] + rem)

            # Save feature bundle
            feat_path = os.path.join(output_dir, f"{uid}.pt")
            torch.save(
                {
                    "tokens": torch.tensor(tokens, dtype=torch.long),
                    "mel": torch.tensor(mel, dtype=torch.float32),
                    "pitch": torch.tensor(pitch, dtype=torch.float32),
                    "energy": torch.tensor(energy, dtype=torch.float32),
                    "durations": torch.tensor(durations, dtype=torch.long),
                    "style_vector": torch.tensor(style_vector, dtype=torch.float32),
                },
                feat_path,
            )

            preprocessed_records.append(
                {
                    "id": uid,
                    "feat_path": feat_path,
                    "tokens_len": len(tokens),
                    "frames_len": num_frames,
                }
            )

        except Exception as e:
            print(f"Warning: Failed to process {uid}: {e}")

    with open(output_manifest_path, "w", encoding="utf-8") as f:
        json.dump(preprocessed_records, f, indent=2)

    print(f"[+] Successfully precomputed {len(preprocessed_records)} samples to {output_manifest_path}")


def run_precomputation(
    manifest_dir: str = "/content/dataset",
    output_cache_dir: str = "/content/preprocessed_data",
):
    train_in = os.path.join(manifest_dir, "train_manifest.json")
    val_in = os.path.join(manifest_dir, "val_manifest.json")

    train_out = os.path.join(manifest_dir, "train_preprocessed.json")
    val_out = os.path.join(manifest_dir, "val_preprocessed.json")

    if os.path.exists(train_in):
        precompute_dataset(train_in, os.path.join(output_cache_dir, "train"), train_out)
    if os.path.exists(val_in):
        precompute_dataset(val_in, os.path.join(output_cache_dir, "val"), val_out)
    print("\n[Cell 04 Complete] Feature precomputation finished.")


if __name__ == "__main__":
    from typing import Tuple
    run_precomputation()
