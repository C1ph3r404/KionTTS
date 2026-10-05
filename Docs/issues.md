# Training Issues & Pitfalls Guide

This document records the exact training issues and architectural pitfalls encountered, so they are not repeated in future runs.

---

## 1. Tokenizer Symbol & ID Shift (The Scrambled Audio Bug)

### What Happened
* The base StyleTTS2 model expects the official **178-symbol dictionary** defined in `text_utils.py`.
* A custom phonemizer (`model/data/phonemizer_util.py`) was used instead, which defined a custom **149-symbol list**:
  * Set `_PAD = "_"` (Official StyleTTS2 uses `_pad = "$"`)
  * Inserted `_SPECIAL = "-"` at index 1 (Official StyleTTS2 has no `_SPECIAL` token)
* Inserting `_SPECIAL = "-"` shifted every following punctuation mark, space, and letter ID by +1 to +2 positions.
* When the training manifest was generated, every phoneme was written with these shifted IDs.
* When synthesizing audio with standard text, the tokens hit the wrong embedding indices, resulting in scrambled, alien-sounding syllables.

### Rule for Future Runs
* **Use the official StyleTTS2 symbol dictionary** (`official_StyleTTS2/text_utils.py`) directly.
* Never add extra special tokens (`_SPECIAL`) or change the padding symbol (`"$"`).
* Ensure the manifest generator uses the exact same 178-token mapping that the model expects.

---

## 2. Joint Training Unfreezing & Style Encoder Collapse at `joint_epoch = 50`

### What Happened
There were two coupled causes that triggered at `joint_epoch = 50`:

1. **Unfreezing Without Grounded Gradients (`no_grad()` T4 Workaround)**:
   * From Epochs 0 to 49, both `decoder` and `style_encoder` were kept in `.eval()` mode with `requires_grad = False`, preserving the intact acoustic weights.
   * At Epoch 50 (`epoch >= joint_epoch`), `decoder` and `style_encoder` were switched to `train()` mode with `requires_grad = True`.
   * However, to avoid OOM crashes on Kaggle T4 GPUs, `y_rec` was wrapped in `torch.no_grad()`. The decoder never received proper backpropagated acoustic reconstruction gradients.
   * In `train()` mode, the decoder's AdaIN blocks activated 20% dropout, and random 1D convolution blurring was applied to F0 and energy curves (`downlist = [0, 3, 7, 15]`), degrading synthesis with no corrective gradient feedback.

2. **Faulty Adjacent-Sample Style Consistency Loss**:
   * The training loop contained:
     ```python
     if len(s_audio) > 1:
         loss_kion_sty = kion_style_consistency(s_audio[:-1], s_audio[1:]) * 0.1
     ```
   * While `style_encoder` was frozen (Epochs 0–49), this loss had no effect.
   * Once unfrozen at Epoch 50, this loss actively penalized `style_encoder` whenever two different sentences in the batch had different style representations, forcing `style_encoder` into mode collapse over epochs 50–60.
   * When this collapsed style vector was fed into the decoder's AdaIN layers, the distorted scaling ($\gamma$) and bias ($\beta$) factors produced pure static and mechanical noise.

### Rule for Future Runs
* **Never unfreeze the decoder if `y_rec` is under `torch.no_grad()`**: If the decoder is trainable, it must receive true backpropagated reconstruction gradients, not artificial noise.
* **Remove the faulty adjacent-sample consistency loss**: Do not penalize different audio samples in a batch for having different styles.
* **Do not use destructive VRAM bypasses**: If training on limited VRAM, lower batch size with gradient accumulation rather than neutering the loss computation graph.

---

## 3. DataParallel `module.` Prefix, Predictor Encoder Initialization & NaN/Inf Explosion

### What Happened
1. **Silent Pretrained Weight Loading Failure (`module.` prefix)**:
   * Official StyleTTS2 checkpoints (`epochs_2nd_00020.pth`) were saved from `DataParallel`, where all parameter dictionary keys are prefixed with `module.` (e.g., `module.shared.0.weight`).
   * When loading with `model[key].load_state_dict(pretrained_dict[key], strict=False)`, PyTorch did not find matching keys, silently ignored all weights, and threw no error.
   * As a result, the entire model architecture (`style_encoder`, `predictor_encoder`, `predictor`, `decoder`) remained completely random and uninitialized.

2. **Uninitialized `predictor_encoder` & $10^{15}$ Output Blowup**:
   * StyleTTS2's `predictor_encoder` contains 4 downsampling residual blocks with `spectral_norm`.
   * With uninitialized weights and running in default `train()` mode without backpropagation, passing mel spectrograms produced prosodic style vectors (`s_dur`) of astronomical magnitude ($\sim 10^{15}$).
   * In `compute_kion_style_loss`, squaring $s_{audio}$ exceeded float32 maximum value ($>3.4 \times 10^{38}$), exploding immediately to `style=inf` at step 0.
   * When `s_dur` was passed to `model.predictor.F0Ntrain` on odd steps, F0 and energy outputs exploded to $1.3 \times 10^{15}$.

3. **Duration Loss Computed Over Unmasked Padding**:
   * `loss_dur = F.mse_loss(duration_pred, d_gt)` computed squared errors across all 50 token slots including zero-padded phonemes, producing a steady $\approx 570.0$ loss.

### Rule for Future Runs
* **Always strip `module.` prefix when loading pretrained checkpoints**:
  ```python
  clean_sd = {k[7:] if k.startswith("module.") else k: v for k, v in state_dict.items()}
  model[key].load_state_dict(clean_sd, strict=False)
  ```
* **Sync `predictor_encoder` from `style_encoder`**: If `predictor_encoder` is not present in the checkpoint, initialize it from `model.style_encoder.state_dict()`.
* **Keep all frozen style encoders in `.eval()` mode with `requires_grad = False`**: Prevents `spectral_norm` buffers from corrupting under `torch.no_grad()`.
* **Clamp style vectors**: Clamp extracted style vectors ($[-10, 10]$) and pitch/energy predictions ($[-50, 50]$) to guarantee numerical stability.
* **Compute duration loss with masked L1 loss**: Only compute duration loss over valid unpadded phonemes `1:text_len-1` as in official StyleTTS2.

---

## 4. Stage 2 Full-Utterance Decoder Autograd OOM on 15GB T4 GPUs

### What Happened
* In Stage 1 (Epochs 0 to 2), the decoder is in `.eval()` mode and is not executed during training, keeping VRAM usage under 4 GB.
* At Stage 2 (`epoch >= joint_epoch`), the decoder (HiFi-GAN) is switched to `train()` mode with active gradient tracking.
* HiFi-GAN upsamples representations $300\times$ across 4 upsampling stages and multi-receptive field fusion resblocks with dilated convolutions.
* When passing unwindowed representations (up to 400 frames $\times 300 = 120,000$ audio samples per utterance $\times$ batch size 4), storing intermediate activation maps across all Snake1D and Conv1D layers consumed over 14.05 GiB of VRAM.
* On Kaggle 16GB T4 GPUs (14.56 GiB usable), PyTorch ran out of memory by just 86 MiB.

---

## 5. Simultaneous Multi-Graph Retention OOM & CPU RAM Offloading

### What Happened
* When `loss_total = (loss_style + loss_predictor) / accum_steps + (loss_dec * 2.0) / accum_steps` was computed together before calling `loss_total.backward()`, PyTorch was forced to retain the full autograd activation graphs of BERT (12 transformer layers), Prosody Predictor, and the HiFi-GAN Decoder simultaneously in GPU VRAM.
* Auxiliary modules (`diffusion`, `mpd`, `msd`, `wd`, and `pitch_extractor`) loaded by `build_model` were remaining on the GPU despite never being used during training, needlessly occupying >1.5 GB of VRAM.
* In interactive Jupyter/Kaggle environments, crashed executions retain references to stack frames and tensors, locking up to 13.86 GiB of VRAM unless garbage collection and cache flushing are performed.

### Rule for Future Runs
* **Decouple Stage 1 and Stage 2 backward passes**:
  1. Call `loss_total.backward()` on `(loss_style + loss_predictor) / accum_steps` first. This immediately computes predictor/BERT gradients and frees all transformer activation graphs from GPU VRAM.
  2. Execute `model.decoder` on a micro-batch (`dec_bs = 1`, `dec_window = 32`) with detached inputs (`t_en_sub`, `f0_sub`, `n_sub`).
  3. Call `loss_dec_step.backward()` independently. The backward graph now contains *only* the decoder itself, dropping decoder peak VRAM to < 50 MB.
* **Purge unused auxiliary modules to CPU RAM**:
  ```python
  for unused_k in ["diffusion", "mpd", "msd", "wd", "pitch_extractor"]:
      if unused_k in model:
          del model[unused_k]
  ```
* **Purge VRAM at cell entry**: Explicitly call `gc.collect()` and `torch.cuda.empty_cache()` before starting training pipelines in notebook cells.

---

## 6. DDP Collective Desynchronization on `is_best` & NCCL 10-Minute Timeout

### What Happened
At the end of Stage 1 Epoch 19 (Step 1720/1720, Global Step 32680), the dual-GPU training job crashed with an NCCL watchdog timeout:
```
[rank1]: Watchdog caught collective operation timeout: WorkNCCL(SeqNum=133305, OpType=ALLREDUCE, NumelIn=1, NumelOut=1, Timeout(ms)=600000) ran for 600035 milliseconds before timing out.
[rank0]: Watchdog caught collective operation timeout: WorkNCCL(SeqNum=133307, OpType=ALLREDUCE, NumelIn=46784, NumelOut=46784, Timeout(ms)=600000)
```
There were two coupled root causes:

1. **Unsynchronized `is_best` Causing Collective Desynchronization**:
   * In multi-GPU DDP training with `DistributedSampler`, each rank computes metrics only over its local subset of batches.
   * `avg_style`, `avg_pred`, and `epoch_loss` were calculated locally on each rank without cross-rank `all_reduce`.
   * At Epoch 19, `epoch_loss < best_loss` (`is_best`) evaluated differently between ranks (e.g. `True` on Rank 0, but `False` on Rank 1).
   * Rank 0 entered the checkpoint saving block and executed `torch.distributed.barrier()`, whereas Rank 1 skipped it and proceeded directly to Epoch 20, calling `torch.distributed.all_reduce(skip_step)`.
   * The barrier (an internal 1-element all-reduce) and `skip_step` all-reduce matched across ranks with mismatched semantics. Subsequent collective sequence numbers and tensor shapes diverged (`NumelIn=1` vs `NumelIn=46784`), causing collective order deadlock.

2. **Synchronous Remote Uploads Blocking the DDP Barrier**:
   * `save_checkpoint(..., upload_hf=True)` performed synchronous HTTP uploads of multiple ~500MB `.pth` files (`step`, `latest`, `best`) and sample audio on Rank 0 while other ranks waited at `torch.distributed.barrier()`.
   * When remote uploads or network latency exceeded PyTorch's default 10-minute (600,000 ms) NCCL watchdog limit, the watchdog thread aborted all worker processes.

### Rule for Future Runs
* **Always synchronize epoch metrics across distributed ranks**:
  ```python
  loss_tensor = torch.tensor([total_style_loss, total_pred_loss, float(num_batches)], device=self.device)
  torch.distributed.all_reduce(loss_tensor, op=torch.distributed.ReduceOp.SUM)
  avg_style = (loss_tensor[0] / max(1.0, loss_tensor[2])).item()
  avg_pred = (loss_tensor[1] / max(1.0, loss_tensor[2])).item()
  epoch_loss = avg_style + avg_pred
  ```
  This guarantees `is_best` and save decisions evaluate 100% identically across all ranks.
* **Make remote Hugging Face Hub checkpoint uploads asynchronous**:
  * Save checkpoints locally to disk in 1–2 seconds with `torch.save()`.
  * Offload remote Hub synchronization and audio sample uploads to a background `ThreadPoolExecutor` worker so GPUs immediately proceed with training.
* **Extend NCCL Watchdog Timeout**:
  * Set `timeout=datetime.timedelta(minutes=60)` in `torch.distributed.init_process_group` to prevent premature job failure during large I/O or evaluation phases.
