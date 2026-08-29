# KionTTS Training Pipeline — Colab Cells

## Overview

This directory contains **numbered Colab cells** for the complete end-to-end
training pipeline of **KionStyleTTS2** — a single-speaker expressive TTS model
built on the StyleTTS2 architecture, augmented with the `KionStyleAdapter` for
tag-conditioned speech generation.

---

## Pipeline Architecture

```
Dataset (wavs + JSON metadata)
        │
        ▼
[Cell 01] Environment Setup & Google Drive Mount
        │
        ▼
[Cell 02] Drive Checkpoint Manager
        │
        ▼
[Cell 03] Data Unpacker & StyleTTS2 Manifest Builder
        │   Converts JSON → StyleTTS2 train_list.txt / val_list.txt
        │
        ▼
[Cell 04] Feature Pre-computation
        │   Mel-spectrograms, pitch (F0), energy, phoneme tokens
        │   Style vectors from KionStyleTagParser
        │
        ▼
[Cell 05] KionTTS Config Builder
        │   Generates config.yml for StyleTTS2 training
        │
        ▼
[Cell 06] Stage 1 Training — Acoustic Foundation
        │   Trains: TextEncoder + Decoder (iSTFTNet) + StyleEncoder
        │   Uses: StyleTTS2 train_first.py logic
        │   Loss: Mel + Multi-Res STFT + GAN (MPD+MSD) + Monotonic Alignment
        │   Duration: ~100 epochs / ~8–12 hours on T4 GPU
        │
        ▼
[Cell 07] Stage 2 Training — Style Diffusion + KionStyleAdapter
        │   Trains: KionStyleAdapter + DiffusionSampler + ProsodyPredictor
        │   Fine-tunes: TextEncoder, Decoder (frozen discriminators)
        │   Loss: All Stage 1 + Style KL + Diffusion Score + Duration CE
        │   Duration: ~50 epochs / ~6–8 hours on T4 GPU
        │
        ▼
[Cell 08] Inference Test & Quality Evaluation
        │   End-to-end synthesis test with emotion tags
        │   Generates audio samples for all 14 emotions + 10 styles
        │
        ▼
[Cell 09] Export & WaveRNN/HiFi-GAN Vocoder Comparison
            Exports model, runs blind A/B listening test samples
```

---

## Stage Details

### Stage 1 — Acoustic Foundation (`06_stage1_acoustic_training.py`)

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| LR | 1e-4 (text_encoder, style_encoder, decoder) |
| Epochs | 100–200 |
| Batch Size | 8–16 (T4), 24–32 (A100) |
| Losses | Mel-L1, Multi-Res STFT, GAN (MPD + MSD), WavLM-SLM, S2S Cross-Entropy, Monotonic Alignment |
| TMA Start | Epoch 50 |
| Warmup | First 10 epochs mel-only |

### Stage 2 — Style Conditioning (`07_stage2_style_diffusion.py`)

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW (separate LR groups) |
| LR Decoder/Encoder | 1e-5 (fine-tune) |
| LR KionStyleAdapter | 1e-4 (train from scratch) |
| LR Diffusion | 1e-4 |
| Epochs | 50–100 |
| Losses | All Stage 1 + Style Reconstruction, Diffusion Score Matching, Duration CE, KionAdapter Contrastive |
| Diffusion Start | Epoch 20 |
| Joint Training | Epoch 50 |

---

## Hardware Requirements

| GPU | Stage 1 ETA | Stage 2 ETA | Max Batch |
|-----|-------------|-------------|-----------|
| T4 (15GB) | 10–14 hrs | 6–8 hrs | 8 |
| V100 (16GB) | 7–10 hrs | 4–6 hrs | 12 |
| A100 (40GB) | 3–5 hrs | 2–3 hrs | 32 |

---

## Files

| Cell | Description |
|------|-------------|
| `00_README.md` | This document |
| `01_environment_setup.py` | Install deps, mount Drive |
| `02_drive_checkpoint_manager.py` | Persistent checkpoint I/O |
| `03_data_unpacker_and_manifest.py` | Dataset prep + StyleTTS2 manifest |
| `04_feature_precomputation.py` | Mel/pitch/energy/phoneme extraction |
| `05_kion_config_builder.py` | Generates `kion_config.yml` |
| `06_stage1_acoustic_training.py` | Stage 1 full training loop |
| `07_stage2_style_diffusion.py` | Stage 2 style training loop |
| `08_inference_test.py` | Inference + eval audio generation |
| `09_export_and_eval.py` | Export final model + quality metrics |
