"""
Production Model Bundle Exporter for KionTTS.
Packages trained neural modules, tag style encoder, configuration, and tag dictionaries
into a lightweight, standalone release artifact for production inference and serving.
"""

import os
import json
import torch
import shutil
from typing import Dict, Any, Optional
from .synthesizer import ALL_TAGS, TAG_TO_IDX


def export_kion_model(
    model: Dict[str, Any],
    tag_encoder: torch.nn.Module,
    output_dir: str = "kiontts_release",
    config: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Exports a clean, self-contained KionTTS model release bundle.
    """
    os.makedirs(output_dir, exist_ok=True)
    bundle_path = os.path.join(output_dir, "kiontts_model.pth")

    def get_mod(name):
        if isinstance(model, dict) or hasattr(model, "get"):
            return model.get(name)
        elif hasattr(model, name):
            return getattr(model, name)
        return None

    def clean_sd(module):
        if module is None:
            return {}
        if hasattr(module, "module"):
            return {k: v.cpu() for k, v in module.module.state_dict().items()}
        return {k: v.cpu() for k, v in module.state_dict().items()}

    model_bundle = {
        "text_encoder": clean_sd(get_mod("text_encoder")),
        "predictor": clean_sd(get_mod("predictor")),
        "bert": clean_sd(get_mod("bert")),
        "bert_encoder": clean_sd(get_mod("bert_encoder")),
        "decoder": clean_sd(get_mod("decoder")),
        "tag_style_encoder": clean_sd(tag_encoder),
        "style_encoder": clean_sd(get_mod("style_encoder")),
        "predictor_encoder": clean_sd(get_mod("predictor_encoder")),
        "config": config or {},
        "all_tags": ALL_TAGS,
        "tag_to_idx": TAG_TO_IDX,
    }

    torch.save(model_bundle, bundle_path)
    size_mb = os.path.getsize(bundle_path) / (1024 * 1024)
    print(f"[✓] Saved production model bundle: {bundle_path} ({size_mb:.2f} MB)")

    # Save metadata & tag dictionary
    tags_path = os.path.join(output_dir, "tags.json")
    with open(tags_path, "w", encoding="utf-8") as f:
        json.dump({"tags": ALL_TAGS, "tag_to_idx": TAG_TO_IDX}, f, indent=2)

    return bundle_path
