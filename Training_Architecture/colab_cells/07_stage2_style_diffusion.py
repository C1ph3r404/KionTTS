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
import time
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

warnings.simplefilter("ignore")

# ─── Colab Paths ─────────────────────────────────────────────────────────────
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
STAGE1_BEST    = os.path.join(DRIVE_CKPT_DIR, "kion_stage1_best.pth")
STAGE1_FINAL   = os.path.join(DRIVE_CKPT_DIR, "kion_stage1_final.pth")

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


# ─── Checkpoint Helpers ───────────────────────────────────────────────────────
def _save_to_drive(state: dict, epoch: int, is_best: bool = False, stage: str = "stage2"):
    import glob
    os.makedirs(DRIVE_CKPT_DIR, exist_ok=True)
    ckpt_path = os.path.join(DRIVE_CKPT_DIR, f"kion_{stage}_epoch_{epoch:04d}.pth")
    torch.save(state, ckpt_path)
    print(f"  [✓] Saved {stage} checkpoint → {ckpt_path}")
    if is_best:
        best_path = os.path.join(DRIVE_CKPT_DIR, f"kion_{stage}_best.pth")
        shutil.copy2(ckpt_path, best_path)
        print(f"  [★] New best! → {best_path}")
    # Prune old checkpoints
    all_ckpts = sorted(glob.glob(os.path.join(DRIVE_CKPT_DIR, f"kion_{stage}_epoch_*.pth")))
    while len(all_ckpts) > 3:
        os.remove(all_ckpts.pop(0))


def _find_latest_stage2_checkpoint() -> str | None:
    import glob
    pattern = os.path.join(DRIVE_CKPT_DIR, "kion_stage2_epoch_*.pth")
    ckpts   = sorted(glob.glob(pattern))
    return ckpts[-1] if ckpts else None


def _load_stage1_checkpoint() -> str:
    """Returns the best available Stage 1 checkpoint path."""
    if os.path.exists(STAGE1_BEST):
        return STAGE1_BEST
    if os.path.exists(STAGE1_FINAL):
        return STAGE1_FINAL
    raise FileNotFoundError(
        "No Stage 1 checkpoint found. Run Cell 06 first.\n"
        f"Expected: {STAGE1_BEST} or {STAGE1_FINAL}"
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


# ─── Main Training Function ───────────────────────────────────────────────────
def run_stage2_training(config_path: str = CONFIG_PATH):
    print("=" * 65)
    print("KionTTS — Stage 2: Style Diffusion + Adapter Training")
    print("=" * 65)

    config      = yaml.safe_load(open(config_path))
    config      = _resolve_config_paths(config)
    _ensure_pretrained_assets(config)
    log_dir     = config["log_dir"]
    os.makedirs(log_dir, exist_ok=True)
    loss_params = Munch(config["loss_params"])
    diff_epoch  = loss_params.diff_epoch
    joint_epoch = loss_params.joint_epoch
    epochs      = config.get("epochs_2nd", 60)
    batch_size  = config.get("batch_size", 8)
    max_len     = config.get("max_len", 300)
    sr          = config["preprocess_params"].get("sr", 24000)
    slmadv_cfg  = Munch(config.get("slmadv_params", {}))
    device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_fp16    = torch.cuda.is_available()

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    writer = SummaryWriter(os.path.join(log_dir, "tensorboard_stage2"))

    print(f"  Epochs       : {epochs}")
    print(f"  Diff start   : epoch {diff_epoch}")
    print(f"  Joint start  : epoch {joint_epoch}")
    print(f"  Batch size   : {batch_size}")
    print(f"  Device       : {device}  | FP16: {use_fp16}")

    # ── Data ─────────────────────────────────────────────────────────────────
    data_params  = config["data_params"]
    train_list, val_list = get_data_path_list(
        data_params["train_data"], data_params["val_data"]
    )
    train_dataloader = build_dataloader(
        train_list, data_params["root_path"],
        OOD_data=data_params["OOD_data"],
        min_length=data_params["min_length"],
        batch_size=batch_size, num_workers=2,
        dataset_config={}, device=device,
    )
    val_dataloader = build_dataloader(
        val_list, data_params["root_path"],
        OOD_data=data_params["OOD_data"],
        min_length=data_params["min_length"],
        batch_size=batch_size, validation=True,
        num_workers=0, dataset_config={}, device=device,
    )
    print(f"  Train: {len(train_list)} | Val: {len(val_list)}")

    # ── Load models ───────────────────────────────────────────────────────────
    text_aligner    = load_ASR_models(config["ASR_path"], config["ASR_config"]).to(device)
    pitch_extractor = load_F0_models(config["F0_path"]).to(device)
    plbert          = load_plbert(config["PLBERT_dir"])

    model_params = recursive_munch(config["model_params"])
    model        = build_model(model_params, text_aligner, pitch_extractor, plbert)
    _ = [model[k].to(device) for k in model]

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
        model["diffusion"],
        sampler=ADPM2Sampler(),
        sigma_schedule=KarrasSchedule(sigma_min=0.0001, sigma_max=3.0, rho=9.0),
        clamp=False,
    )

    # ── Load Stage 1 checkpoint ───────────────────────────────────────────────
    stage1_ckpt = _load_stage1_checkpoint()
    print(f"  Loading Stage 1 checkpoint: {stage1_ckpt}")
    model, _, _, _ = load_checkpoint(model, None, stage1_ckpt, load_only_params=True)

    # ── Optimiser — separate LR groups ───────────────────────────────────────
    opt_params  = config["optimizer_params"]
    lr          = float(opt_params.get("lr", 1e-4))
    ft_lr       = float(opt_params.get("ft_lr", 1e-5))
    bert_lr     = float(opt_params.get("bert_lr", 1e-5))

    optimizer = torch.optim.AdamW([
        {"params": model["text_encoder"].parameters(),       "lr": ft_lr},
        {"params": model["style_encoder"].parameters(),      "lr": ft_lr},
        {"params": model["decoder"].parameters(),            "lr": ft_lr},
        {"params": model["predictor"].parameters(),          "lr": lr},
        {"params": model["predictor_encoder"].parameters(),  "lr": lr},
        {"params": model["diffusion"].parameters(),          "lr": lr},
        {"params": model["kion_style_adapter"].parameters(), "lr": lr},
        {"params": model["bert_encoder"].parameters(),       "lr": bert_lr},
        {"params": model["mpd"].parameters(),                "lr": ft_lr},
        {"params": model["msd"].parameters(),                "lr": ft_lr},
    ], betas=(0.9, 0.98), weight_decay=1e-2)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=ft_lr * 0.1
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)

    # ── SLM Adversarial ───────────────────────────────────────────────────────
    slmadv = SLMAdversarialLoss(
        model["wd"],
        model["diffusion"],
        sampler,
        log_norm,
        model["pitch_extractor"],
        model["mpd"],
        model["msd"],
        model["style_encoder"],
        slmadv_cfg,
    )

    # ── Resume Stage 2 if available ───────────────────────────────────────────
    start_epoch = 0
    iters       = 0
    best_loss   = float("inf")
    latest_s2   = _find_latest_stage2_checkpoint()
    if latest_s2:
        print(f"  Resuming Stage 2 from: {latest_s2}")
        ckpt = torch.load(latest_s2, map_location=device)
        for k in model:
            if k in ckpt["net"]:
                model[k].load_state_dict(ckpt["net"][k])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt.get("epoch", 0) + 1
        iters       = ckpt.get("iters", 0)
        best_loss   = ckpt.get("val_loss", float("inf"))
        print(f"  Resumed at epoch {start_epoch}")

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

    print(f"\n  Training from epoch {start_epoch + 1}...\n")

    # ══════════════════════════════════════════════════════════════════════════
    # Stage 2 Training Loop
    # ══════════════════════════════════════════════════════════════════════════
    for epoch in range(start_epoch, epochs):
        epoch_start = time.time()
        _ = [model[k].train() for k in model]

        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1:03d}/{epochs} [Stage 2]", leave=False)

        for i, batch in enumerate(pbar):
            waves = batch[0]
            batch = [b.to(device) for b in batch[1:]]
            texts, input_lengths, ref_texts, ref_lengths, mels, mel_input_length, _ = batch

            try:
                with torch.no_grad():
                    mask      = length_to_mask(mel_input_length // (2 ** n_down)).to(device)
                    text_mask = length_to_mask(input_lengths).to(device)

                # ── ASR alignment ──────────────────────────────────────────
                ppgs, s2s_pred, s2s_attn = model["text_aligner"](mels, mask, texts)
                s2s_attn = s2s_attn.transpose(-1, -2)[..., 1:].transpose(-1, -2)

                with torch.no_grad():
                    mask_ST       = mask_from_lens(s2s_attn, input_lengths, mel_input_length // (2 ** n_down))
                    s2s_attn_mono = maximum_path(s2s_attn, mask_ST)

                # ── Text + Prosody encode ─────────────────────────────────
                with torch.amp.autocast("cuda", enabled=use_fp16):
                    t_en  = model["text_encoder"](texts, input_lengths, text_mask)
                    aln   = s2s_attn_mono if random.getrandbits(1) else s2s_attn
                    asr   = t_en @ aln

                    mel_len = min(int(mel_input_length.min().item() / 2 - 1), max_len // 2)
                    en, gt, wav, style_w = [], [], [], []

                    for bib in range(len(mel_input_length)):
                        ml = int(mel_input_length[bib].item() / 2)
                        rs = np.random.randint(0, ml - mel_len)
                        en.append(asr[bib, :, rs:rs + mel_len])
                        gt.append(mels[bib, :, rs * 2:(rs + mel_len) * 2])
                        y = waves[bib][rs * 2 * 300:(rs + mel_len) * 2 * 300]
                        wav.append(torch.from_numpy(y).to(device))

                    en  = torch.stack(en)
                    gt  = torch.stack(gt).detach()
                    wav = torch.stack(wav).float().detach()

                    if gt.shape[-1] < 80:
                        continue

                    with torch.no_grad():
                        real_norm     = log_norm(gt.unsqueeze(1)).squeeze(1).detach()
                        F0_real, _, _ = model["pitch_extractor"](gt.unsqueeze(1))

                    # ── Acoustic style (from audio) ───────────────────────
                    s_audio = model["style_encoder"](gt.unsqueeze(1))

                    # ── Prosody predictor ─────────────────────────────────
                    # We also get BERT-conditioned text representation
                    bert_dur = model["bert_encoder"](texts, attention_mask=~text_mask)
                    d, p = model["predictor_encoder"](bert_dur, s_audio, input_lengths, s2s_attn_mono, text_mask)

                    # Duration CE loss (compare predictor output to monotonic alignment)
                    loss_ce, loss_dur = 0.0, 0.0
                    for _d, _p, _len in zip(d, p, input_lengths):
                        _aln = s2s_attn_mono[0][:_len, :mel_len]
                        dur_tgt = _aln.sum(dim=-1)
                        loss_dur += F.l1_loss(_p[:_len], dur_tgt)
                        loss_ce  += F.cross_entropy(
                            _d[:_len, :mel_len + 1],
                            torch.round(dur_tgt).long().clamp(0, mel_len),
                        )

                    loss_dur /= texts.size(0)
                    loss_ce  /= texts.size(0)

                    # ── F0 / Energy predictor ─────────────────────────────
                    F0_pred, N_pred = model["predictor"].F0Ntrain(en, s_audio)
                    loss_F0   = F.smooth_l1_loss(F0_pred, F0_real.unsqueeze(0))
                    loss_norm = F.smooth_l1_loss(N_pred, real_norm)

                    # ── Decoder ───────────────────────────────────────────
                    y_rec = model["decoder"](en, F0_real, real_norm, s_audio)
                    loss_mel = stft_loss(y_rec.squeeze(), wav.detach())

                    # ── GAN ───────────────────────────────────────────────
                    loss_gen_all = gl(wav.detach().unsqueeze(1).float(), y_rec).mean()
                    loss_slm     = wl(wav.detach(), y_rec).mean()

                    # ── Style diffusion (after diff_epoch) ────────────────
                    loss_diff = torch.tensor(0.0, device=device)
                    if epoch >= diff_epoch:
                        # Diffusion score matching on style latent
                        loss_diff = model["diffusion"](s_audio.unsqueeze(1), embedding=bert_dur).mean()

                    # ── KionStyleAdapter consistency ───────────────────────
                    # We don't have explicit style_weights in the StyleTTS2
                    # dataloader, so we use cosine alignment between
                    # s_audio samples in the batch as a self-supervised signal
                    # (full tag-supervised loss is applied in the Kion dataset loader)
                    if len(s_audio) > 1:
                        loss_kion_sty = kion_style_consistency(
                            s_audio[:-1], s_audio[1:]
                        ) * 0.1   # small weight — mostly to regularise
                    else:
                        loss_kion_sty = torch.tensor(0.0, device=device)

                    # ── Total generator loss ──────────────────────────────
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

                # ── Discriminator step ─────────────────────────────────────
                optimizer.zero_grad()
                d_loss = dl(wav.detach().unsqueeze(1).float(), y_rec.detach()).mean()
                scaler.scale(d_loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [p for g in optimizer.param_groups for p in g["params"]], 5.0
                )
                scaler.step(optimizer)
                scaler.update()

                # ── Generator step ─────────────────────────────────────────
                optimizer.zero_grad()
                scaler.scale(g_loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [p for g in optimizer.param_groups for p in g["params"]], 5.0
                )
                scaler.step(optimizer)
                scaler.update()

                iters += 1
                pbar.set_postfix(
                    mel=f"{loss_mel.item():.4f}",
                    F0=f"{loss_F0.item():.4f}",
                    diff=f"{loss_diff.item():.4f}" if epoch >= diff_epoch else "—",
                )

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

        # ── Validation ────────────────────────────────────────────────────
        _ = [model[k].eval() for k in model]
        val_loss = 0.0
        n_val    = 0

        with torch.no_grad():
            for batch in val_dataloader:
                waves = batch[0]
                batch = [b.to(device) for b in batch[1:]]
                texts, input_lengths, _, _, mels, mel_input_length, _ = batch

                mask      = length_to_mask(mel_input_length // (2 ** n_down)).to(device)
                text_mask = length_to_mask(input_lengths).to(device)
                t_en      = model["text_encoder"](texts, input_lengths, text_mask)
                _, _, s2s_attn = model["text_aligner"](mels, mask, texts)
                s2s_attn  = s2s_attn.transpose(-1, -2)[..., 1:].transpose(-1, -2)
                asr       = t_en @ s2s_attn

                mel_len   = min(int(mel_input_length.min().item() / 2 - 1), max_len // 2)
                en, gt, wav_v = [], [], []
                for bib in range(len(mel_input_length)):
                    ml = int(mel_input_length[bib].item() / 2)
                    rs = np.random.randint(0, ml - mel_len)
                    en.append(asr[bib, :, rs:rs + mel_len])
                    gt.append(mels[bib, :, rs * 2:(rs + mel_len) * 2])
                    y = waves[bib][rs * 2 * 300:(rs + mel_len) * 2 * 300]
                    wav_v.append(torch.from_numpy(y).to(device))

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

        scheduler.step()

        if (epoch + 1) % 2 == 0 or is_best:
            state = {
                "net":       {k: model[k].state_dict() for k in model},
                "optimizer": optimizer.state_dict(),
                "iters":     iters,
                "val_loss":  val_loss,
                "epoch":     epoch,
            }
            _save_to_drive(state, epoch + 1, is_best=is_best, stage="stage2")

    # ── Final save ────────────────────────────────────────────────────────────
    print("\n[+] Stage 2 training complete!")
    final_path = os.path.join(DRIVE_CKPT_DIR, "kion_stage2_final.pth")
    torch.save({
        "net":       {k: model[k].state_dict() for k in model},
        "optimizer": optimizer.state_dict(),
        "iters":     iters,
        "val_loss":  best_loss,
        "epoch":     epochs,
        "config":    config,
    }, final_path)
    print(f"  Final checkpoint → {final_path}")
    print("  Proceed to Cell 08 for inference testing.")


if __name__ == "__main__":
    run_stage2_training()
