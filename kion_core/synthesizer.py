"""
Production Inference and Interactive Speech Synthesizer for KionTTS.
Supports:
- Emotion & delivery style tag conditioning (inline prompt like '[sarcasm=0.8, playful=0.5] Hello!').
- Continuous 24-dimensional style intensity vectors.
- Acoustic style cloning from reference audio WAV files.
- Full 178-token IPA phonemization pipeline.
"""

import re
import os
import torch
import numpy as np
import soundfile as sf
import librosa
from typing import Optional, Union, Tuple, Dict, List

# Official StyleTTS2 178-token dictionary
_pad = "$"
_punctuation = ';:,.!?¡¿—…"«»“” '
_letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
_letters_ipa = "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"

symbols = [_pad] + list(_punctuation) + list(_letters) + list(_letters_ipa)
dicts = {symbols[i]: i for i in range(len(symbols))}

ALL_TAGS = [
    "angry", "annoyed", "bored", "concerned", "confused",
    "curious", "disappointed", "excited", "frustrated", "happy",
    "heartbroken", "overjoyed", "sad", "surprised",
    "affectionate", "authoritative", "calm", "deadpan", "dramatic",
    "playful", "sarcasm", "serious", "soothing", "teasing"
]
TAG_TO_IDX = {tag: i for i, tag in enumerate(ALL_TAGS)}

TAG_ALIASES = {
    "suprised": "surprised",
    "sarcastic": "sarcasm",
    "heartbreak": "heartbroken",
    "dry-sarcasm": "sarcasm",
    "sarcastic-deadpan": "sarcasm",
    "sarcastic-playful": "sarcasm",
    "happy-mild": "happy",
    "happy-strong": "happy",
    "sad-mild": "sad",
    "sad-strong": "sad",
    "concerned-mild": "concerned",
    "curious-mild": "curious",
}


def parse_inline_prompt(prompt: str) -> Tuple[str, torch.Tensor]:
    """
    Parses prompts like:
      '[sarcasm=0.8] Oh, brilliant.'
      '[happy=0.7, playful=0.6] We did it!'
    Returns:
      clean_text: The stripped text without tags
      tag_vector: Tensor of shape (1, 24) with continuous float intensities in [0, 1]
    """
    tag_vector = torch.zeros((1, len(ALL_TAGS)), dtype=torch.float32)
    match = re.match(r"^\[(.*?)\]\s*(.*)$", prompt.strip())
    if match:
        tag_str = match.group(1)
        clean_text = match.group(2).strip()
        for pair in tag_str.split(","):
            if "=" in pair:
                tag, val = pair.split("=", 1)
                tag = tag.strip().lower()
                tag = TAG_ALIASES.get(tag, tag)
                try:
                    v = float(val.strip())
                    if tag in TAG_TO_IDX:
                        tag_vector[0, TAG_TO_IDX[tag]] = max(0.0, min(1.0, v))
                except ValueError:
                    pass
            else:
                tag = pair.strip().lower()
                tag = TAG_ALIASES.get(tag, tag)
                if tag in TAG_TO_IDX:
                    tag_vector[0, TAG_TO_IDX[tag]] = 1.0
    else:
        clean_text = prompt.strip()

    return clean_text, tag_vector


def length_to_mask(lengths: torch.Tensor) -> torch.Tensor:
    mask = torch.arange(lengths.max(), device=lengths.device).unsqueeze(0).expand(lengths.shape[0], -1)
    mask = torch.gt(mask + 1, lengths.unsqueeze(1))
    return mask


class KionSynthesizer:
    """
    High-level speech synthesis engine for KionTTS.
    """
    def __init__(
        self,
        model: Dict[str, Any],
        tag_encoder: torch.nn.Module,
        phonemizer_fn=None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.model = model
        self.tag_encoder = tag_encoder
        self.phonemizer_fn = phonemizer_fn
        self.device = torch.device(device)

        # Set evaluation mode
        for k, v in self.model.items():
            if v is not None and hasattr(v, "eval"):
                v.eval()
                v.to(self.device)

        if self.tag_encoder is not None:
            self.tag_encoder.eval()
            self.tag_encoder.to(self.device)

    def text_to_tokens(self, text: str) -> torch.Tensor:
        """Converts text or pre-phonemized text to 178-token IPA IDs."""
        if self.phonemizer_fn is not None:
            try:
                ps = self.phonemizer_fn([text])
                if isinstance(ps, (list, tuple)):
                    ps = ps[0]
            except Exception:
                ps = self.phonemizer_fn(text)
                if isinstance(ps, (list, tuple)):
                    ps = ps[0]
        else:
            ps = text

        indexes = []
        for char in ps:
            if char in dicts:
                indexes.append(dicts[char])

        # StyleTTS2 boundary padding
        indexes.insert(0, 0)
        indexes.append(0)
        return torch.LongTensor(indexes).unsqueeze(0).to(self.device)

    def extract_style_from_audio(self, wav_path: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extracts ground-truth acoustic and prosodic style vectors from a reference audio file."""
        wave, sr = sf.read(wav_path)
        if wave.ndim > 1:
            wave = wave[:, 0]
        if sr != 24000:
            wave = librosa.resample(wave, orig_sr=sr, target_sr=24000)

        import torchaudio
        to_mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=24000, n_mels=80, n_fft=2048, win_length=1200, hop_length=300
        )
        wave_t = torch.from_numpy(wave).float().unsqueeze(0)
        mel = to_mel(wave_t)
        mel = (torch.log(1e-5 + mel) + 4.0) / 4.0
        mel = mel.to(self.device)

        with torch.no_grad():
            ref = self.model["style_encoder"](mel.unsqueeze(1))
            s = self.model["predictor_encoder"](mel.unsqueeze(1))
        return ref, s

    @torch.no_grad()
    def synthesize(
        self,
        prompt: Optional[str] = None,
        text: Optional[str] = None,
        tag_vector: Optional[torch.Tensor] = None,
        ref_audio_path: Optional[str] = None,
        speed: float = 1.0,
    ) -> np.ndarray:
        """
        Synthesizes high-fidelity audio from text conditioned on emotion/delivery style.
        """
        if prompt is not None:
            clean_text, parsed_tag_vec = parse_inline_prompt(prompt)
            if tag_vector is None:
                tag_vector = parsed_tag_vec
        else:
            clean_text = text or ""
            if tag_vector is None:
                tag_vector = torch.zeros((1, len(ALL_TAGS)), dtype=torch.float32)

        tokens = self.text_to_tokens(clean_text)
        tag_vector = tag_vector.to(self.device)

        # 1. Style Vector Conditioning
        if ref_audio_path and os.path.exists(ref_audio_path):
            ref, s = self.extract_style_from_audio(ref_audio_path)
            # If tags are provided alongside reference audio, apply blended delta
            if tag_vector.sum() > 0.01:
                s_tag = self.tag_encoder(tag_vector)
                ref = 0.7 * ref + 0.3 * s_tag[:, :128]
                s = 0.7 * s + 0.3 * s_tag[:, 128:]
        else:
            s_tag = self.tag_encoder(tag_vector)
            ref = s_tag[:, :128]  # acoustic style for decoder AdaIN
            s = s_tag[:, 128:]    # prosodic style for predictor

        # 2. Text Encoding
        input_lengths = torch.LongTensor([tokens.shape[-1]]).to(self.device)
        text_mask = length_to_mask(input_lengths)

        t_en = self.model["text_encoder"](tokens, input_lengths, text_mask)
        bert_dur = self.model["bert"](tokens, attention_mask=(~text_mask).int())
        d_en = self.model["bert_encoder"](bert_dur).transpose(-1, -2)

        # 3. Prosody Prediction
        d = self.model["predictor"].text_encoder(d_en, s, input_lengths, text_mask)
        x, _ = self.model["predictor"].lstm(d)
        duration = self.model["predictor"].duration_proj(x)
        duration = torch.sigmoid(duration).sum(axis=-1)
        pred_dur = torch.round(duration.squeeze() / speed).clamp(min=1)
        if pred_dur.dim() == 0:
            pred_dur = pred_dur.unsqueeze(0)
        pred_dur[-1] += 5  # Small trailing pause

        total_frames = int(pred_dur.sum().item())
        pred_aln_trg = torch.zeros((input_lengths.item(), total_frames), device=self.device)
        c_frame = 0
        for i in range(pred_aln_trg.size(0)):
            d_i = int(pred_dur[i].item())
            pred_aln_trg[i, c_frame : c_frame + d_i] = 1
            c_frame += d_i

        en = d.transpose(-1, -2) @ pred_aln_trg.unsqueeze(0)
        F0_pred, N_pred = self.model["predictor"].F0Ntrain(en, s)

        # 4. Decoder Waveform Synthesis
        asr_aligned = t_en @ pred_aln_trg.unsqueeze(0)
        out = self.model["decoder"](asr_aligned, F0_pred, N_pred, ref)

        wave = out.squeeze().cpu().numpy()
        return wave
