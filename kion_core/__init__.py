import importlib
import torch
import torch.serialization

# Cleanly reload serialization module to reset any previous recursive wrappers
importlib.reload(torch.serialization)
_real_torch_load = torch.serialization.load

def _safe_torch_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _real_torch_load(*args, **kwargs)

torch.load = _safe_torch_load

from .tag_style_encoder import KionTagStyleEncoder, compute_kion_style_loss
from .dataset import KionManifestDataset, KionCollater, build_kion_dataloader
from .synthesizer import KionSynthesizer, parse_inline_prompt
from .export import export_kion_model
