import os
import os.path as osp
import json
import numpy as np
import soundfile as sf
import librosa
import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio

# Exact 178 symbols from official StyleTTS2
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

to_mel = torchaudio.transforms.MelSpectrogram(
    n_mels=80, n_fft=2048, win_length=1200, hop_length=300
)
MEL_MEAN, MEL_STD = -4.0, 4.0


def preprocess_mel(wave):
    wave_tensor = torch.from_numpy(wave).float()
    mel_tensor = to_mel(wave_tensor)
    mel_tensor = (torch.log(1e-5 + mel_tensor.unsqueeze(0)) - MEL_MEAN) / MEL_STD
    return mel_tensor


class KionManifestDataset(Dataset):
    def __init__(self, manifest_path, root_dir, phonemizer_fn=None, validation=False, max_mel_length=192):
        self.root_dir = root_dir
        self.validation = validation
        self.max_mel_length = max_mel_length
        self.phonemizer_fn = phonemizer_fn

        with open(manifest_path, "r", encoding="utf-8") as f:
            self.samples = json.load(f)

    def __len__(self):
        return len(self.samples)

    def _clean_phonemes_to_ids(self, text):
        indexes = []
        for char in text:
            if char in dicts:
                indexes.append(dicts[char])
        # Pad with 0 at boundaries
        indexes.insert(0, 0)
        indexes.append(0)
        return torch.LongTensor(indexes)

    def _get_tag_vector(self, item):
        vec = torch.zeros(len(ALL_TAGS), dtype=torch.float32)
        emotions = item.get("emotions", {})
        styles = item.get("styles", {})
        for tag, val in emotions.items():
            if tag in TAG_TO_IDX:
                vec[TAG_TO_IDX[tag]] = float(val)
        for tag, val in styles.items():
            if tag in TAG_TO_IDX:
                vec[TAG_TO_IDX[tag]] = float(val)
        return vec

    def __getitem__(self, idx):
        item = self.samples[idx]
        raw_path = item.get("wav_path") or item.get("audio_filepath", "")
        if osp.isabs(raw_path) and osp.exists(raw_path):
            wav_full_path = raw_path
        else:
            wav_full_path = osp.join(self.root_dir, raw_path)
            if not osp.exists(wav_full_path):
                for candidate in [
                    osp.join(self.root_dir, "wavs", osp.basename(raw_path)),
                    osp.join(self.root_dir, "sample_kion", raw_path),
                    osp.join(self.root_dir, osp.basename(raw_path)),
                ]:
                    if osp.exists(candidate):
                        wav_full_path = candidate
                        break

        # Load audio
        wave, sr = sf.read(wav_full_path)
        if wave.ndim > 1:
            wave = wave[:, 0].squeeze()
        if sr != 24000:
            wave_t = torch.from_numpy(wave).float()
            wave_t = torchaudio.functional.resample(wave_t, orig_freq=sr, new_freq=24000)
            wave = wave_t.numpy()

        # Prepend/append zeros like StyleTTS2
        wave = np.concatenate([np.zeros(5000), wave, np.zeros(5000)], axis=0)

        # Mel spectrogram
        mel_tensor = preprocess_mel(wave).squeeze()
        length_feature = mel_tensor.size(1)
        acoustic_feature = mel_tensor[:, :(length_feature - length_feature % 2)]

        # Phonemize or use pre-phonemized field
        raw_text = item.get("clean_text") or item.get("text", "")
        if "phonemes" in item:
            phoneme_text = item["phonemes"]
        elif self.phonemizer_fn is not None:
            phoneme_text = self.phonemizer_fn(raw_text)
        else:
            phoneme_text = raw_text

        text_tensor = self._clean_phonemes_to_ids(phoneme_text)
        tag_vector = self._get_tag_vector(item)

        # Random slice for reference mel
        mel_length = mel_tensor.size(1)
        if mel_length > self.max_mel_length:
            start = np.random.randint(0, mel_length - self.max_mel_length)
            ref_mel = mel_tensor[:, start:start + self.max_mel_length]
        else:
            ref_mel = mel_tensor

        return acoustic_feature, text_tensor, ref_mel, tag_vector, wave, wav_full_path


class KionCollater:
    def __init__(self, max_mel_length=192):
        self.max_mel_length = max_mel_length

    def __call__(self, batch):
        batch_size = len(batch)
        # Sort by mel length descending
        lengths = [b[0].shape[1] for b in batch]
        batch_indexes = np.argsort(lengths)[::-1]
        batch = [batch[bid] for bid in batch_indexes]

        nmels = batch[0][0].size(0)
        max_mel_length = max([b[0].shape[1] for b in batch])
        max_text_length = max([b[1].shape[0] for b in batch])

        mels = torch.zeros((batch_size, nmels, max_mel_length), dtype=torch.float32)
        texts = torch.zeros((batch_size, max_text_length), dtype=torch.long)
        input_lengths = torch.zeros(batch_size, dtype=torch.long)
        output_lengths = torch.zeros(batch_size, dtype=torch.long)
        ref_mels = torch.zeros((batch_size, nmels, self.max_mel_length), dtype=torch.float32)
        tag_vectors = torch.zeros((batch_size, len(ALL_TAGS)), dtype=torch.float32)
        waves = [None for _ in range(batch_size)]
        paths = ["" for _ in range(batch_size)]

        for bid, (mel, text, ref_mel, tag_vec, wave, path) in enumerate(batch):
            mel_len = mel.size(1)
            text_len = text.size(0)
            ref_mel_len = ref_mel.size(1)

            mels[bid, :, :mel_len] = mel
            texts[bid, :text_len] = text
            input_lengths[bid] = text_len
            output_lengths[bid] = mel_len
            ref_mels[bid, :, :ref_mel_len] = ref_mel
            tag_vectors[bid] = tag_vec
            waves[bid] = wave
            paths[bid] = path

        return waves, texts, input_lengths, mels, output_lengths, ref_mels, tag_vectors, paths


def build_kion_dataloader(
    manifest_path,
    root_dir,
    phonemizer_fn=None,
    batch_size=4,
    validation=False,
    num_workers=2,
    is_distributed=False,
    rank=0,
    world_size=1,
):
    dataset = KionManifestDataset(manifest_path, root_dir, phonemizer_fn=phonemizer_fn, validation=validation)
    collater = KionCollater()

    sampler = None
    if is_distributed and not validation:
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=True,
        )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(sampler is None and not validation),
        sampler=sampler,
        num_workers=num_workers,
        drop_last=(not validation),
        collate_fn=collater,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
    )
    return dataloader
