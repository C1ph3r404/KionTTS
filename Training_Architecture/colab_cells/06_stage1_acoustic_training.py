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
def _save_to_drive(state: dict, epoch: int, is_best: bool = False):
    os.makedirs(DRIVE_CKPT_DIR, exist_ok=True)
    ckpt_path = os.path.join(DRIVE_CKPT_DIR, f"kion_stage1_epoch_{epoch:04d}.pth")
    torch.save(state, ckpt_path)
    print(f"  [✓] Saved Stage 1 checkpoint → {ckpt_path}")
    if is_best:
        best_path = os.path.join(DRIVE_CKPT_DIR, "kion_stage1_best.pth")
        shutil.copy2(ckpt_path, best_path)
        print(f"  [★] New best! Copied → {best_path}")
    # Keep only last 3 epoch checkpoints to save Drive space
    import glob
    all_ckpts = sorted(glob.glob(os.path.join(DRIVE_CKPT_DIR, "kion_stage1_epoch_*.pth")))
    while len(all_ckpts) > 3:
        os.remove(all_ckpts.pop(0))


def _find_latest_drive_checkpoint() -> str | None:
    import glob
    pattern = os.path.join(DRIVE_CKPT_DIR, "kion_stage1_epoch_*.pth")
    ckpts = sorted(glob.glob(pattern))
    return ckpts[-1] if ckpts else None


# ─── Main Training Function ───────────────────────────────────────────────────
def run_stage1_training(config_path: str = CONFIG_PATH):
    print("=" * 65)
    print("KionTTS — Stage 1: Acoustic Foundation Training")
    print("=" * 65)

    # 1. Load config
    config = yaml.safe_load(open(config_path))
    log_dir = config["log_dir"]
    os.makedirs(log_dir, exist_ok=True)
    shutil.copy(config_path, os.path.join(log_dir, os.path.basename(config_path)))

    # 2. Accelerator (single-GPU for Colab)
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        project_dir=log_dir,
        split_batches=True,
        mixed_precision="fp16",          # FP16 for Colab T4 efficiency
        kwargs_handlers=[ddp_kwargs],
    )
    device = accelerator.device

    if accelerator.is_main_process:
        writer = SummaryWriter(os.path.join(log_dir, "tensorboard"))
        file_handler = logging.FileHandler(os.path.join(log_dir, "stage1_train.log"))
        file_handler.setLevel(logging.DEBUG)
        log.addHandler(file_handler)

    # 3. Hyperparameters from config
    batch_size   = config.get("batch_size", 8)
    epochs       = config.get("epochs_1st", 120)
    save_freq    = config.get("save_freq", 2)
    log_interval = config.get("log_interval", 10)
    max_len      = config.get("max_len", 300)
    loss_params  = Munch(config["loss_params"])
    TMA_epoch    = loss_params.TMA_epoch
    sr           = config["preprocess_params"].get("sr", 24000)

    data_params = config["data_params"]
    train_list, val_list = get_data_path_list(
        data_params["train_data"], data_params["val_data"]
    )

    print(f"  Train samples : {len(train_list)}")
    print(f"  Val samples   : {len(val_list)}")
    print(f"  Batch size    : {batch_size}")
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

    # Add KionStyleAdapter to the model dict
    ksa_cfg = config["model_params"]["kion_style_adapter"]
    kion_style_adapter = KionStyleAdapter(
        num_emotions=ksa_cfg["num_emotions"],
        num_styles=ksa_cfg["num_styles"],
        tag_embed_dim=ksa_cfg["tag_embed_dim"],
        latent_style_dim=ksa_cfg["latent_style_dim"],
        hidden_dim=ksa_cfg["hidden_dim"],
        dropout=ksa_cfg["dropout"],
    ).to(device)
    model["kion_style_adapter"] = kion_style_adapter

    # 7. Scheduler + optimiser
    scheduler_params = {
        "max_lr":           float(config["optimizer_params"].get("lr", 1e-4)),
        "pct_start":        0.0,
        "epochs":           epochs,
        "steps_per_epoch":  len(train_dataloader),
    }
    optimizer = build_optimizer(
        {k: model[k].parameters() for k in model},
        scheduler_params_dict={k: scheduler_params.copy() for k in model},
        lr=float(config["optimizer_params"].get("lr", 1e-4)),
    )

    # 8. Prepare with Accelerator
    for k in model:
        model[k] = accelerator.prepare(model[k])
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

    print(f"\n  Starting training from epoch {start_epoch + 1}...")

    # ──────────────────────────────────────────────────────────────────────────
    # 11. Training Loop
    # ──────────────────────────────────────────────────────────────────────────
    for epoch in range(start_epoch, epochs):
        epoch_start = time.time()
        running_loss = 0.0
        _ = [model[k].train() for k in model]

        pbar = tqdm(
            train_dataloader,
            desc=f"Epoch {epoch+1:03d}/{epochs} [Stage 1]",
            leave=False,
        )

        for i, batch in enumerate(pbar):
            waves = batch[0]
            batch = [b.to(device) for b in batch[1:]]
            texts, input_lengths, _, _, mels, mel_input_length, _ = batch

            with torch.no_grad():
                mask      = length_to_mask(mel_input_length // (2 ** n_down)).to(device)
                text_mask = length_to_mask(input_lengths).to(device)

            # ASR aligner forward
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
                wav.append(torch.from_numpy(y).to(device))
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

            # ── Discriminator step ──────────────────────────────────────
            if epoch >= TMA_epoch:
                optimizer.zero_grad()
                d_loss = dl(wav.detach().unsqueeze(1).float(), y_rec.detach()).mean()
                accelerator.backward(d_loss)
                optimizer.step("msd")
                optimizer.step("mpd")
            else:
                d_loss = 0

            # ── Generator step ──────────────────────────────────────────
            optimizer.zero_grad()
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
            optimizer.step("kion_style_adapter")  # always train adapter

            iters += 1
            pbar.set_postfix(
                mel=f"{loss_mel:.4f}" if isinstance(loss_mel, float) else f"{loss_mel.item():.4f}",
                gen=f"{loss_gen_all:.4f}" if isinstance(loss_gen_all, float) else f"{loss_gen_all.item():.4f}",
            )

            if (i + 1) % log_interval == 0 and accelerator.is_main_process:
                avg = running_loss / log_interval
                writer.add_scalar("train/mel_loss",  avg,       iters)
                writer.add_scalar("train/gen_loss",  loss_gen_all if isinstance(loss_gen_all, float) else loss_gen_all.item(), iters)
                writer.add_scalar("train/disc_loss", d_loss if isinstance(d_loss, float) else d_loss.item(), iters)
                running_loss = 0.0

        # ── Validation ────────────────────────────────────────────────────
        _ = [model[k].eval() for k in model]
        loss_test = 0.0
        iters_test = 0

        with torch.no_grad():
            for batch in val_dataloader:
                waves = batch[0]
                batch = [b.to(device) for b in batch[1:]]
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
                    wav.append(torch.from_numpy(y).to(device))

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

            # Write sample audio to TensorBoard every 5 epochs
            if epoch % 5 == 0:
                with torch.no_grad():
                    for bib in range(min(3, len(en))):
                        ml  = int(mel_input_length[bib].item())
                        g   = mels[bib, :, :ml].unsqueeze(0)
                        e   = asr[bib, :, :ml // 2].unsqueeze(0)
                        F0r, _, _ = model["pitch_extractor"](g.unsqueeze(1))
                        s_  = model["style_encoder"](g.unsqueeze(1))
                        nr  = log_norm(g.unsqueeze(1)).squeeze(1)
                        yr  = model["decoder"](e, F0r.unsqueeze(0), nr, s_)
                        writer.add_audio(f"eval/synth_{bib}", yr.cpu().numpy().squeeze(), epoch, sample_rate=sr)

            # Save checkpoint
            if (epoch + 1) % save_freq == 0 or is_best:
                state = {
                    "net":       {k: model[k].state_dict() for k in model},
                    "optimizer": optimizer.state_dict(),
                    "iters":     iters,
                    "val_loss":  val_loss,
                    "epoch":     epoch,
                }
                _save_to_drive(state, epoch + 1, is_best=is_best)

    # Final save
    if accelerator.is_main_process:
        print("\n[+] Stage 1 training complete!")
        state = {
            "net":       {k: model[k].state_dict() for k in model},
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
