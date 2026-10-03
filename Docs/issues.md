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
