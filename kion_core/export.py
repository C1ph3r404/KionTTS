import os
import json
import torch
import shutil

def export_kion_model(
    model,
    tag_encoder,
    config,
    tag_mapping_path,
    output_dir="/kaggle/working/kiontts_release"
):
    """
    Exports a self-contained KionTTS model release bundle.
    """
    os.makedirs(output_dir, exist_ok=True)
    bundle_path = os.path.join(output_dir, "kiontts_model.pth")
    
    # Extract clean state dicts
    def clean_sd(module):
        if hasattr(module, "module"):
            return {k: v.cpu() for k, v in module.module.state_dict().items()}
        return {k: v.cpu() for k, v in module.state_dict().items()}
    
    model_bundle = {
        "text_encoder": clean_sd(model.text_encoder),
        "predictor": clean_sd(model.predictor),
        "bert": clean_sd(model.bert),
        "bert_encoder": clean_sd(model.bert_encoder),
        "decoder": clean_sd(model.decoder),
        "tag_style_encoder": clean_sd(tag_encoder),
        "config": config,
    }
    
    torch.save(model_bundle, bundle_path)
    print(f"[✓] Saved model bundle: {bundle_path} ({os.path.getsize(bundle_path) / 1024 / 1024:.2f} MB)")
    
    # Copy tag mapping
    if os.path.exists(tag_mapping_path):
        shutil.copy(tag_mapping_path, os.path.join(output_dir, "tag_mapping.json"))
        print(f"[✓] Copied tag mapping to {output_dir}/tag_mapping.json")

    return bundle_path
