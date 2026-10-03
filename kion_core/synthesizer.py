import re
import torch
import numpy as np

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


def parse_inline_prompt(prompt: str):
    """
    Parses prompts like:
      '[sarcasm=0.8] Oh, brilliant.'
      '[happy=0.7, playful=0.6] We did it!'
    Returns:
      clean_text, tag_vector (torch.Tensor of shape (1, 24))
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
                try:
                    val = float(val.strip())
                    if tag in TAG_TO_IDX:
                        tag_vector[0, TAG_TO_IDX[tag]] = val
                except ValueError:
                    pass
    else:
        clean_text = prompt.strip()

    return clean_text, tag_vector


class KionSynthesizer:
    def __init__(self, model, tag_encoder, phonemizer_fn=None, device="cuda"):
        self.model = model
        self.tag_encoder = tag_encoder
        self.phonemizer_fn = phonemizer_fn
        self.device = device

        self.model.eval()
        self.tag_encoder.eval()

    def text_to_tokens(self, text: str):
        if self.phonemizer_fn is not None:
            ps = self.phonemizer_fn(text)
        else:
            ps = text

        indexes = []
        for char in ps:
            if char in dicts:
                indexes.append(dicts[char])
        indexes.insert(0, 0)
        indexes.append(0)
        return torch.LongTensor(indexes).unsqueeze(0).to(self.device)

    @torch.no_grad()
    def synthesize(self, prompt: str = None, text: str = None, tag_vector: torch.Tensor = None):
        """
        Synthesizes audio conditioned on emotion/style tags.
        """
        if prompt is not None:
            clean_text, tag_vec = parse_inline_prompt(prompt)
        else:
            clean_text = text
            tag_vec = tag_vector if tag_vector is not None else torch.zeros((1, len(ALL_TAGS)))

        tokens = self.text_to_tokens(clean_text)
        tag_vec = tag_vec.to(self.device)

        # 1. Tag Conditioning
        s_tag = self.tag_encoder(tag_vec)  # (1, 256)
        ref = s_tag[:, :128]  # acoustic style for decoder AdaIN
        s = s_tag[:, 128:]    # prosodic style for predictor

        # 2. Text Encoding
        input_lengths = torch.LongTensor([tokens.shape[-1]]).to(self.device)
        text_mask = torch.arange(tokens.shape[-1]).unsqueeze(0).to(self.device) < input_lengths.unsqueeze(1)

        t_en = self.model.text_encoder(tokens, input_lengths, text_mask)
        bert_dur = self.model.bert(tokens, attention_mask=(~text_mask).int())
        d_en = self.model.bert_encoder(bert_dur).transpose(-1, -2)

        # 3. Prosody Prediction
        d = self.model.predictor.text_encoder(d_en, s, input_lengths, text_mask)
        x, _ = self.model.predictor.lstm(d)
        duration = self.model.predictor.duration_proj(x)
        duration = torch.sigmoid(duration).sum(axis=-1)
        pred_dur = torch.round(duration.squeeze()).clamp(min=1)
        if pred_dur.dim() == 0:
            pred_dur = pred_dur.unsqueeze(0)
        pred_dur[-1] += 5  # Small trailing pause

        total_frames = int(pred_dur.sum().item())
        pred_aln_trg = torch.zeros((input_lengths.item(), total_frames), device=self.device)
        c_frame = 0
        for i in range(pred_aln_trg.size(0)):
            d_i = int(pred_dur[i].item())
            pred_aln_trg[i, c_frame:c_frame + d_i] = 1
            c_frame += d_i

        en = (d.transpose(-1, -2) @ pred_aln_trg.unsqueeze(0))
        F0_pred, N_pred = self.model.predictor.F0Ntrain(en, s)

        # 4. Decoder Waveform Synthesis
        asr_aligned = (t_en @ pred_aln_trg.unsqueeze(0))
        out = self.model.decoder(asr_aligned, F0_pred, N_pred, ref)

        wave = out.squeeze().cpu().numpy()
        return wave
