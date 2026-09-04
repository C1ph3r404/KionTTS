"""
Colab Cell 05: KionTTS Config Builder
Generates kion_config.yml tailored for KionStyleTTS2 single-speaker expressive
training. Adapts the official StyleTTS2 config schema with Kion-specific
style conditioning, KionStyleAdapter dimensions, and Colab-optimised defaults.

Run AFTER: Cell 04 (feature precomputation)
Run BEFORE: Cell 06 (Stage 1 training)
"""

import os
import yaml

def _get_repo_root() -> str:
    rel_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    if os.path.exists(os.path.join(rel_path, "model")):
        return rel_path
    for p in ["/content/KionTTS", "/content/Kiontts", "/content/kiontts"]:
        if os.path.exists(p):
            return p
    return "/content/KionTTS"


REPO_ROOT       = _get_repo_root()
STYLETTS2_ROOT  = f"{REPO_ROOT}/StyleTTS2"
DRIVE_CKPT_DIR  = "/content/drive/MyDrive/KionTTS_Checkpoints"
DATASET_ROOT    = "/content/dataset/wavs"
LOG_DIR         = f"{DRIVE_CKPT_DIR}/logs"
STAGE1_PATH     = f"{DRIVE_CKPT_DIR}/kion_stage1.pth"


# ─── Detect GPU VRAM and set batch size ─────────────────────────────────────
def auto_batch_size():
    try:
        import torch
        if not torch.cuda.is_available():
            return 2
        vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        if vram >= 38:   return 16   # A100 40GB
        elif vram >= 22: return 8    # RTX 3090 / 4090 / A10G (24GB)
        elif vram >= 14: return 4    # T4 / V100 (15GB/16GB) — StyleTTS2 vocoder requires bs=2 on T4
        else:            return 2
    except Exception:
        return 2


def build_kion_config(
    train_list:     str  = f"{STYLETTS2_ROOT}/Data/kion_train_list.txt",
    val_list:       str  = f"{STYLETTS2_ROOT}/Data/kion_val_list.txt",
    ood_data:       str  = f"{STYLETTS2_ROOT}/Data/OOD_texts.txt",
    output_path:    str  = f"{STYLETTS2_ROOT}/Configs/kion_config.yml",
    epochs_1st:     int  = 120,
    epochs_2nd:     int  = 60,
    batch_size:     int  = None,
    num_emotions:   int  = 14,
    num_styles:     int  = 10,
    style_dim:      int  = 128,
    hidden_dim:     int  = 512,
    n_token:        int  = 178,
    use_diffusion:  bool = True,
) -> str:
    """
    Generates kion_config.yml.

    Returns:
        Path to the written config file.
    """
    bs = batch_size or auto_batch_size()

    config = {
        # ── Logging & Checkpoints ──────────────────────────────────────────
        "log_dir":               LOG_DIR,
        "first_stage_path":      "kion_stage1.pth",
        "save_freq":             1,
        "save_step_interval":    1000,
        "log_interval":          10,
        "device":                "cuda",

        # ── Training Schedule ─────────────────────────────────────────────
        "epochs_1st":            epochs_1st,
        "epochs_2nd":            epochs_2nd,
        "batch_size":            bs,
        "max_len":               200,           # max mel frames per clip (200 is optimal for T4 memory)

        # ── Checkpoint resumption ─────────────────────────────────────────
        "pretrained_model":      "",
        "second_stage_load_pretrained": True,
        "load_only_params":      False,

        # ── Pretrained Utility Models ─────────────────────────────────────
        "F0_path":      f"{STYLETTS2_ROOT}/Utils/JDC/bst.t7",
        "ASR_config":   f"{STYLETTS2_ROOT}/Utils/ASR/config.yml",
        "ASR_path":     f"{STYLETTS2_ROOT}/Utils/ASR/epoch_00080.pth",
        "PLBERT_dir":   f"{STYLETTS2_ROOT}/Utils/PLBERT/",

        # ── Data Paths ────────────────────────────────────────────────────
        "data_params": {
            "train_data":   train_list,
            "val_data":     val_list,
            "root_path":    DATASET_ROOT,
            "OOD_data":     ood_data,
            "min_length":   50,
        },

        # ── Audio Preprocessing ───────────────────────────────────────────
        "preprocess_params": {
            "sr": 24000,
            "spect_params": {
                "n_fft":      2048,
                "win_length": 1200,
                "hop_length": 300,
            },
        },

        # ── Model Architecture ────────────────────────────────────────────
        "model_params": {
            "multispeaker": False,          # single-speaker Kion

            # Text / Acoustic dims (must match KionStyleTTS2.__init__)
            "dim_in":       64,
            "hidden_dim":   hidden_dim,
            "max_conv_dim": hidden_dim,
            "n_layer":      3,
            "n_mels":       80,
            "n_token":      n_token,
            "max_dur":      50,
            "style_dim":    style_dim,
            "dropout":      0.1,

            # ── KionStyleAdapter (Kion-specific) ──────────────────────────
            "kion_style_adapter": {
                "num_emotions":     num_emotions,
                "num_styles":       num_styles,
                "tag_embed_dim":    64,
                "latent_style_dim": style_dim,
                "hidden_dim":       hidden_dim // 2,
                "dropout":          0.1,
            },

            # ── iSTFTNet Decoder ──────────────────────────────────────────
            "decoder": {
                "type":                     "istftnet",
                "resblock_kernel_sizes":    [3, 7, 11],
                "upsample_rates":           [10, 6],
                "upsample_initial_channel": 512,
                "resblock_dilation_sizes":  [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
                "upsample_kernel_sizes":    [20, 12],
                "gen_istft_n_fft":          20,
                "gen_istft_hop_size":       5,
            },

            # ── WavLM Speech Language Model ───────────────────────────────
            "slm": {
                "model":           "microsoft/wavlm-base-plus",
                "sr":              16000,
                "hidden":          768,
                "nlayers":         13,
                "initial_channel": 64,
            },

            # ── Style Diffusion Model ─────────────────────────────────────
            "diffusion": {
                "embedding_mask_proba": 0.1,
                "transformer": {
                    "num_layers":   3,
                    "num_heads":    8,
                    "head_features": 64,
                    "multiplier":   2,
                },
                "dist": {
                    "sigma_data":        0.2,
                    "estimate_sigma_data": True,
                    "mean":              -3.0,
                    "std":               1.0,
                },
            },
        },

        # ── Loss Weights ─────────────────────────────────────────────────
        "loss_params": {
            # Stage 1 losses
            "lambda_mel":   5.0,    # mel reconstruction (most important early)
            "lambda_gen":   1.0,    # GAN generator
            "lambda_slm":   1.0,    # WavLM feature match
            "lambda_mono":  1.0,    # monotonic alignment
            "lambda_s2s":   1.0,    # sequence-to-sequence cross-entropy
            "TMA_epoch":    50,     # epoch to begin TMA training (Stage 1)

            # Stage 2 losses
            "lambda_F0":    1.0,    # F0 (pitch) reconstruction
            "lambda_norm":  1.0,    # energy / norm reconstruction
            "lambda_dur":   1.0,    # duration regression
            "lambda_ce":    20.0,   # duration predictor CE loss
            "lambda_sty":   1.0,    # style reconstruction
            "lambda_diff":  1.0,    # diffusion score matching

            # KionStyleAdapter consistency loss (new for Kion)
            "lambda_kion_style": 0.5,   # tag-to-style consistency

            "diff_epoch":   20,     # epoch to activate style diffusion (Stage 2)
            "joint_epoch":  50,     # epoch to begin joint end-to-end training
        },

        # ── Optimiser ─────────────────────────────────────────────────────
        "optimizer_params": {
            "lr":      1e-4,        # default for most modules
            "bert_lr": 1e-5,        # PLBERT fine-tune LR
            "ft_lr":   1e-5,        # acoustic modules fine-tune LR in Stage 2
        },

        # ── SLM Adversarial ───────────────────────────────────────────────
        "slmadv_params": {
            "min_len":          400,
            "max_len":          500,
            "batch_percentage": 0.5,
            "iter":             10,
            "thresh":           5,
            "scale":            0.01,
            "sig":              1.5,
        },
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    print(f"[+] KionTTS config written to: {output_path}")
    print(f"    Batch size : {bs}")
    print(f"    Stage 1    : {epochs_1st} epochs")
    print(f"    Stage 2    : {epochs_2nd} epochs")
    print(f"    Style dim  : {style_dim}  | Emotions: {num_emotions} | Styles: {num_styles}")
    return output_path


def create_ood_texts(output_path: str = f"{STYLETTS2_ROOT}/Data/OOD_texts.txt"):
    """
    Creates a set of Out-of-Distribution evaluation sentences for StyleTTS2.
    These are phonetically rich sentences used during validation to test
    generalisation beyond the training set.
    """
    ood_sentences = [
        "The quick brown fox jumps over the lazy dog.",
        "She sells seashells by the seashore.",
        "How much wood would a woodchuck chuck?",
        "Peter Piper picked a peck of pickled peppers.",
        "I cannot believe this actually worked.",
        "Are you seriously telling me that right now?",
        "Everything is going to be absolutely fine.",
        "Wait, what did you just say to me?",
        "This is genuinely the most surprising thing I have ever heard.",
        "Oh, wonderful. Just what I needed today.",
        "I am so happy you are here with me.",
        "This is taking far too long and I am getting frustrated.",
        "You really thought I would not notice that?",
        "Can you please just calm down for a moment?",
        "I have been waiting for this moment my entire life.",
        "That is the most brilliant idea you have ever had.",
        "Well, I did not see that coming at all.",
        "Please stop doing that, it is incredibly annoying.",
        "I am genuinely concerned about what you just told me.",
        "Look, I understand how you feel, but we need to talk.",
    ]
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for sent in ood_sentences:
            f.write(sent + "\n")
    print(f"[+] OOD texts written: {len(ood_sentences)} sentences → {output_path}")


if __name__ == "__main__":
    create_ood_texts()
    config_path = build_kion_config()
    print("\n[Cell 05 Complete] Config ready. Proceed to Cell 06 for Stage 1 training.")
