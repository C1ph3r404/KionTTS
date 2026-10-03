#!/usr/bin/env python3
"""
Extract a balanced, stratified sample dataset from kion_dataset.tar
Designed for fast Kaggle 2x T4 smoke testing and validation.
"""

import os
import io
import re
import json
import random
import tarfile
import zipfile
import argparse
from collections import defaultdict

EMOTIONS = [
    "angry", "annoyed", "bored", "concerned", "confused",
    "curious", "disappointed", "excited", "frustrated", "happy",
    "heartbroken", "overjoyed", "sad", "surprised"
]

STYLES = [
    "affectionate", "authoritative", "calm", "deadpan", "dramatic",
    "playful", "sarcasm", "serious", "soothing", "teasing"
]

ALL_TAGS = EMOTIONS + STYLES
TAG_TO_ID = {tag: i for i, tag in enumerate(ALL_TAGS)}


def clean_text_for_tts(text: str) -> str:
    # Strip any leading bracketed tags if still present
    text = re.sub(r"^\[.*?\]\s*", "", text.strip())
    # Replace curved quotes/apostrophes with standard ones
    text = text.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
    text = text.strip()
    return text


def collect_catalog(tar_path: str, max_batches: int = None):
    print(f"[*] Scanning {tar_path}...")
    catalog = {"train": [], "val": []}
    
    with tarfile.open(tar_path, "r") as tar:
        for z_name, split_key in [("KionTTS_Dataset_train.zip", "train"), ("KionTTS_Dataset_val.zip", "val")]:
            f = tar.extractfile(z_name)
            z = zipfile.ZipFile(io.BytesIO(f.read()))
            batch_names = [n for n in z.namelist() if n.endswith(".zip")]
            if max_batches:
                batch_names = batch_names[:max_batches]
                
            for b_name in batch_names:
                bz = zipfile.ZipFile(io.BytesIO(z.read(b_name)))
                if "metadata.json" in bz.namelist():
                    meta = json.loads(bz.read("metadata.json").decode("utf-8"))
                    for item in meta:
                        segs = item.get("segments", [])
                        # We strictly keep single segment utterances
                        if len(segs) != 1:
                            continue
                        
                        seg = segs[0]
                        clean_text = clean_text_for_tts(seg.get("text", ""))
                        if len(clean_text) < 5:
                            continue
                            
                        emotions_dict = seg.get("emotions", {})
                        styles_dict = seg.get("styles", {})
                        
                        # Verify tags are valid
                        all_sample_tags = list(emotions_dict.keys()) + list(styles_dict.keys())
                        if not all_sample_tags:
                            continue
                        if not all(t in TAG_TO_ID for t in all_sample_tags):
                            continue
                            
                        wav_name = f"wavs/{item['id']}.wav"
                        catalog[split_key].append({
                            "id": item["id"],
                            "z_name": z_name,
                            "b_name": b_name,
                            "wav_inner_path": wav_name,
                            "clean_text": clean_text,
                            "raw_text": item.get("text", ""),
                            "emotions": emotions_dict,
                            "styles": styles_dict,
                            "is_compound": (len(all_sample_tags) > 1),
                            "tags": all_sample_tags
                        })

    print(f"[✓] Scanned catalog: {len(catalog['train'])} train candidates, {len(catalog['val'])} val candidates.")
    return catalog


def select_stratified(candidates, target_count, rng):
    """
    Select target_count samples ensuring all 24 tags and compound blends are represented.
    """
    selected = []
    selected_ids = set()
    tag_counts = defaultdict(int)
    compound_count = 0
    
    # Shuffle candidates
    shuffled = list(candidates)
    rng.shuffle(shuffled)
    
    # Target ~25% compound blends
    target_compounds = int(target_count * 0.25)
    
    # First pass: ensure at least 5 instances of every tag
    for tag in ALL_TAGS:
        needed = 5
        for item in shuffled:
            if tag in item["tags"] and item["id"] not in selected_ids:
                selected.append(item)
                selected_ids.add(item["id"])
                for t in item["tags"]:
                    tag_counts[t] += 1
                if item["is_compound"]:
                    compound_count += 1
                needed -= 1
                if needed <= 0:
                    break
                    
    # Second pass: ensure target compound ratio if possible
    for item in shuffled:
        if len(selected) >= target_count:
            break
        if item["id"] not in selected_ids and item["is_compound"] and compound_count < target_compounds:
            selected.append(item)
            selected_ids.add(item["id"])
            for t in item["tags"]:
                tag_counts[t] += 1
            compound_count += 1

    # Third pass: fill remaining slots prioritizing underrepresented tags
    for item in shuffled:
        if len(selected) >= target_count:
            break
        if item["id"] not in selected_ids:
            selected.append(item)
            selected_ids.add(item["id"])
            for t in item["tags"]:
                tag_counts[t] += 1
            if item["is_compound"]:
                compound_count += 1

    rng.shuffle(selected)
    return selected


def extract_selected_wavs(tar_path, selected_items, out_dir):
    os.makedirs(os.path.join(out_dir, "wavs"), exist_ok=True)
    
    # Group by (z_name, b_name) for efficient extraction
    batch_map = defaultdict(list)
    for item in selected_items:
        batch_map[(item["z_name"], item["b_name"])].append(item)
        
    print(f"[*] Extracting {len(selected_items)} audio files across {len(batch_map)} batches...")
    
    extracted = 0
    with tarfile.open(tar_path, "r") as tar:
        # Cache open zip files
        open_zips = {}
        for (z_name, b_name), items in batch_map.items():
            if z_name not in open_zips:
                f = tar.extractfile(z_name)
                open_zips[z_name] = zipfile.ZipFile(io.BytesIO(f.read()))
                
            z = open_zips[z_name]
            bz = zipfile.ZipFile(io.BytesIO(z.read(b_name)))
            
            for item in items:
                out_wav_path = os.path.join(out_dir, "wavs", f"{item['id']}.wav")
                with open(out_wav_path, "wb") as wf:
                    wf.write(bz.read(item["wav_inner_path"]))
                extracted += 1
                if extracted % 100 == 0 or extracted == len(selected_items):
                    print(f"    Extracted {extracted}/{len(selected_items)} wavs...")

    print(f"[✓] Extracted all {extracted} audio files to {os.path.join(out_dir, 'wavs')}.")


def main():
    parser = argparse.ArgumentParser(description="Extract sample dataset from kion_dataset.tar")
    parser.add_argument("--tar_path", default="DataSet/data/kion_dataset.tar", help="Path to kion_dataset.tar")
    parser.add_argument("--out_dir", default="DataSet/sample_kion", help="Output directory")
    parser.add_argument("--num_train", type=int, default=500, help="Number of train samples")
    parser.add_argument("--num_val", type=int, default=50, help="Number of val samples")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--create_tar", action="store_true", default=True, help="Create sample tar.gz archive")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    catalog = collect_catalog(args.tar_path)
    train_selected = select_stratified(catalog["train"], args.num_train, rng)
    val_selected = select_stratified(catalog["val"], args.num_val, rng)

    print(f"\n[✓] Train selected: {len(train_selected)} samples")
    print(f"[✓] Val selected: {len(val_selected)} samples")

    # Extract wavs
    all_selected = train_selected + val_selected
    extract_selected_wavs(args.tar_path, all_selected, args.out_dir)

    # Save metadata manifests
    def clean_record(item):
        return {
            "id": item["id"],
            "clean_text": item["clean_text"],
            "raw_text": item["raw_text"],
            "emotions": item["emotions"],
            "styles": item["styles"],
            "wav_path": f"wavs/{item['id']}.wav"
        }

    train_manifest = [clean_record(it) for it in train_selected]
    val_manifest = [clean_record(it) for it in val_selected]

    with open(os.path.join(args.out_dir, "train_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(train_manifest, f, indent=2)

    with open(os.path.join(args.out_dir, "val_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(val_manifest, f, indent=2)

    tag_metadata = {
        "all_tags": ALL_TAGS,
        "tag_to_id": TAG_TO_ID,
        "emotions": EMOTIONS,
        "styles": STYLES,
        "total_tags": len(ALL_TAGS)
    }
    with open(os.path.join(args.out_dir, "tag_mapping.json"), "w", encoding="utf-8") as f:
        json.dump(tag_metadata, f, indent=2)

    print(f"[✓] Manifests saved to {args.out_dir}")

    # Create tar.gz for easy transport to Kaggle
    if args.create_tar:
        tar_out = os.path.join(os.path.dirname(args.out_dir), "sample_kion.tar.gz")
        print(f"[*] Packaging {tar_out} ...")
        with tarfile.open(tar_out, "w:gz") as tar:
            tar.add(args.out_dir, arcname="sample_kion")
        print(f"[✓] Packaged sample archive: {tar_out} ({os.path.getsize(tar_out) / 1024 / 1024:.2f} MB)")


if __name__ == "__main__":
    main()
