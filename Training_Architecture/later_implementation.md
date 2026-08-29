# Intra-Utterance Transition Conditioning (Future Implementation)

This document outlines the design and implementation strategy for intra-utterance style and emotion transitions to be integrated into KionTTS in a future update.

---

## 1. Overview

In multi-sentence or dynamic utterances, Kion's LLM may output text with multiple consecutive emotion/style tags:

```text
[happy=0.9] I finally fixed the issue! [concerned=0.6] Wait, why is the server still offline?
```

While the current version of KionTTS models utterance-level emotion & style combinations (and relies on the voice assistant's chunked sentence streaming to handle style switches between chunks), intra-utterance transitions allow a single synthesized waveform to smoothly shift prosodic and emotional characteristics mid-sentence.

---

## 2. Technical Architecture

### 2.1 Token/Phoneme-Aligned Style Sequences

Instead of broadcasting a single style vector $\mathbf{s} \in \mathbb{R}^{D}$ across all phoneme representations:

```text
Text:      [HH AE P IY ...] [W EY T W AY ...]
Segment:   |---- Seg 1 ----| |---- Seg 2 ----|
Style:     [ happy = 0.9 ]   [ concerned = 0.6 ]
```

1. **Alignment Expansion**:
   - The phonemizer and text parser produce phoneme tokens along with segment boundaries.
   - For each phoneme $p_t$, assign style vector:
     $$\mathbf{s}_t = \mathbf{s}_{\text{seg}(t)}$$
   - Resulting in a token-level style tensor $\mathbf{S}_{\text{seq}} \in \mathbb{R}^{T_{\text{phonemes}} \times D_{\text{style}}}$.

2. **Boundary Smoothing (Prosodic Interpolation)**:
   - Abrupt step changes at segment boundaries can cause acoustic artifacts or pitch discontinuities.
   - Apply a Hann/sigmoid cross-fade transition kernel across $k$ phonemes (e.g. $k=3-5$) around transition boundaries:
     $$\mathbf{s}_t = (1 - \lambda_t)\mathbf{s}_{\text{prev}} + \lambda_t \mathbf{s}_{\text{next}}, \quad \lambda_t = \frac{1}{1 + e^{-\gamma (t - t_{\text{boundary}})}}$$

3. **Frame-Level Projection in Prosody & Acoustic Predictor**:
   - Using the learned or predicted duration alignments, map $\mathbf{S}_{\text{seq}}$ to frame-level style tensor $\mathbf{S}_{\text{frame}} \in \mathbb{R}^{T_{\text{frames}} \times D_{\text{style}}}$.
   - Condition the Pitch ($F_0$), Energy, and Mel/Waveform decoders on the frame-aligned style sequence.

---

## 3. Dataset Requirements & Training Strategy

- **Transition Dataset Bank**: Synthesize 2-part and 3-part joined sentences with known transition timestamps.
- **Phoneme Aligner Training**: Ensure robust duration prediction when prosody dynamics shift abruptly.
- **Ablation Matrix**: Benchmark step transitions vs. smoothed sigmoid transitions on pitch contour continuity and perceptual naturalness.
