import torch

# Idempotent PyTorch 2.6+ compatibility patch for legacy StyleTTS2 checkpoints
if not hasattr(torch, "_kion_orig_load"):
    torch._kion_orig_load = torch.load

    def _safe_torch_load(*args, **kwargs):
        if "weights_only" not in kwargs:
            kwargs["weights_only"] = False
        return torch._kion_orig_load(*args, **kwargs)

    torch.load = _safe_torch_load

from .tag_style_encoder import KionTagStyleEncoder, compute_kion_style_loss
from .dataset import KionManifestDataset, KionCollater, build_kion_dataloader
from .synthesizer import KionSynthesizer, parse_inline_prompt
from .export import export_kion_model
