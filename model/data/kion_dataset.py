"""
KionTTS PyTorch Dataset & Dynamic Batch Collator
"""

import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Any, Tuple


class KionDataset(Dataset):
    def __init__(self, preprocessed_manifest_path: str):
        if not os.path.exists(preprocessed_manifest_path):
            raise FileNotFoundError(f"Manifest not found: {preprocessed_manifest_path}")

        with open(preprocessed_manifest_path, "r", encoding="utf-8") as f:
            self.records = json.load(f)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        feat_path = self.records[idx]["feat_path"]
        data = torch.load(feat_path, map_location="cpu")
        return data


def collate_fn_kion(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Dynamically pads tokens, Mel-spectrograms, pitch, energy, and durations.
    """
    token_lens = [item["tokens"].size(0) for item in batch]
    frame_lens = [item["mel"].size(1) for item in batch]

    max_token_len = max(token_lens)
    max_frame_len = max(frame_lens)

    batch_size = len(batch)
    mel_channels = batch[0]["mel"].size(0)
    style_dim = batch[0]["style_vector"].size(0)

    # Initialize padded tensors
    padded_tokens = torch.zeros(batch_size, max_token_len, dtype=torch.long)
    padded_mel = torch.full((batch_size, mel_channels, max_frame_len), fill_value=-11.5129, dtype=torch.float32)
    padded_pitch = torch.zeros(batch_size, max_frame_len, dtype=torch.float32)
    padded_energy = torch.zeros(batch_size, max_frame_len, dtype=torch.float32)
    padded_durations = torch.zeros(batch_size, max_token_len, dtype=torch.long)
    stacked_style = torch.zeros(batch_size, style_dim, dtype=torch.float32)

    text_mask = torch.ones(batch_size, max_token_len, dtype=torch.bool)

    for i, item in enumerate(batch):
        t_len = token_lens[i]
        f_len = frame_lens[i]

        padded_tokens[i, :t_len] = item["tokens"]
        padded_mel[i, :, :f_len] = item["mel"]
        padded_pitch[i, :f_len] = item["pitch"]
        padded_energy[i, :f_len] = item["energy"]
        padded_durations[i, :t_len] = item["durations"]
        stacked_style[i] = item["style_vector"]

        text_mask[i, :t_len] = False  # False means valid token (for Transformer mask)

    # Compute target log durations (log(dur + 1.0))
    target_log_durations = torch.log(padded_durations.float() + 1.0)

    return {
        "tokens": padded_tokens,
        "mel": padded_mel,
        "pitch": padded_pitch,
        "energy": padded_energy,
        "durations": padded_durations,
        "target_log_durations": target_log_durations,
        "style_weights": stacked_style,
        "text_mask": text_mask,
        "token_lens": torch.tensor(token_lens, dtype=torch.long),
        "frame_lens": torch.tensor(frame_lens, dtype=torch.long),
    }


def create_dataloader(
    manifest_path: str,
    batch_size: int = 16,
    shuffle: bool = True,
    num_workers: int = 2,
) -> DataLoader:
    dataset = KionDataset(manifest_path)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn_kion,
        pin_memory=True,
    )
