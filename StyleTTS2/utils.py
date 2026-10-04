try:
    from monotonic_align import maximum_path as _mp_c
    from monotonic_align import mask_from_lens
    from monotonic_align.core import maximum_path_c
    _HAS_MONOTONIC_C = True
except ImportError:
    _HAS_MONOTONIC_C = False

    def mask_from_lens(attn, input_lengths, mel_lengths):
        mask = torch.zeros_like(attn)
        for b in range(attn.size(0)):
            t_len = min(int(input_lengths[b].item()), attn.size(1))
            m_len = min(int(mel_lengths[b].item()), attn.size(2))
            mask[b, :t_len, :m_len] = 1.0
        return mask

import numpy as np
import torch
import copy
from torch import nn
import torch.nn.functional as F
import torchaudio
import librosa
try:
    import matplotlib.pyplot as plt
except (ImportError, ModuleNotFoundError):
    plt = None
from munch import Munch

def maximum_path(neg_cent, mask):
    """
    Monotonic alignment search (Viterbi).
    neg_cent: [b, t_t, t_s]
    mask: [b, t_t, t_s]
    """
    device = neg_cent.device
    dtype = neg_cent.dtype
    if _HAS_MONOTONIC_C:
        neg_cent_np = np.ascontiguousarray(neg_cent.data.cpu().numpy().astype(np.float32))
        path = np.ascontiguousarray(np.zeros(neg_cent_np.shape, dtype=np.int32))
        t_t_max = np.ascontiguousarray(mask.sum(1)[:, 0].data.cpu().numpy().astype(np.int32))
        t_s_max = np.ascontiguousarray(mask.sum(2)[:, 0].data.cpu().numpy().astype(np.int32))
        maximum_path_c(path, neg_cent_np, t_t_max, t_s_max)
        return torch.from_numpy(path).to(device=device, dtype=dtype)

    val_np = neg_cent.data.cpu().numpy()
    mask_np = mask.data.cpu().numpy()
    b, t_t, t_s = val_np.shape
    path = np.zeros((b, t_t, t_s), dtype=np.float32)

    for i in range(b):
        v = val_np[i]
        m = mask_np[i]
        t_t_max = int(m[:, 0].sum()) if m[:, 0].sum() > 0 else t_t
        t_s_max = int(m[0, :].sum()) if m[0, :].sum() > 0 else t_s

        Q = np.full((t_t_max, t_s_max), -np.inf, dtype=np.float32)
        Q[0, 0] = v[0, 0]

        for y in range(1, t_s_max):
            Q[0, y] = Q[0, y - 1] + v[0, y]

        for x in range(1, t_t_max):
            for y in range(x, t_s_max):
                prev = max(Q[x - 1, y - 1], Q[x, y - 1])
                if prev > -np.inf:
                    Q[x, y] = prev + v[x, y]

        curr_x = t_t_max - 1
        for y in range(t_s_max - 1, -1, -1):
            path[i, curr_x, y] = 1.0
            if curr_x > 0:
                if y == 0 or Q[curr_x - 1, y - 1] >= Q[curr_x, y - 1]:
                    curr_x -= 1

    return torch.from_numpy(path).to(device=device, dtype=dtype)

def get_data_path_list(train_path=None, val_path=None):
    if train_path is None:
        train_path = "Data/train_list.txt"
    if val_path is None:
        val_path = "Data/val_list.txt"

    with open(train_path, 'r', encoding='utf-8', errors='ignore') as f:
        train_list = f.readlines()
    with open(val_path, 'r', encoding='utf-8', errors='ignore') as f:
        val_list = f.readlines()

    return train_list, val_list

def length_to_mask(lengths):
    mask = torch.arange(lengths.max()).unsqueeze(0).expand(lengths.shape[0], -1).type_as(lengths)
    mask = torch.gt(mask+1, lengths.unsqueeze(1))
    return mask

# for norm consistency loss
def log_norm(x, mean=-4, std=4, dim=2):
    """
    normalized log mel -> mel -> norm -> log(norm)
    """
    x = torch.log(torch.exp(x * std + mean).norm(dim=dim))
    return x

def get_image(arrs):
    plt.switch_backend('agg')
    fig = plt.figure()
    ax = plt.gca()
    ax.imshow(arrs)

    return fig

def recursive_munch(d):
    if isinstance(d, dict):
        return Munch((k, recursive_munch(v)) for k, v in d.items())
    elif isinstance(d, list):
        return [recursive_munch(v) for v in d]
    else:
        return d
    
def log_print(message, logger):
    logger.info(message)
    print(message)
    