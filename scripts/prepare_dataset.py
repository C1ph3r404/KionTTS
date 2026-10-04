#!/usr/bin/env python3
"""
Production Dataset Preparation & Manifest Generator for KionTTS.
Handles:
  1. Raw .tar archives (e.g. DataSet/data/kion_dataset.tar)
  2. Auto-extracted Kaggle directories with nested batch zips
  3. Pre-extracted directories with loose metadata.json and wavs/
  4. Auto-discovery of dataset paths across /kaggle/input and local workspace
Normalizes 24 emotion & delivery style tags and outputs unified manifests.
"""

import os
import io
import re
import glob
import json
import shutil
import zipfile
import tarfile
import argparse
from typing import Dict, List, Any, Optional

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


def clean_text_for_phonemizer(text: str) -> str:
    """Removes leading bracketed prompts [tag=val] and normalizes unicode punctuation."""
    text = re.sub(r"^\[.*?\]\s*", "", text.strip())
    text = text.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
    text = text.replace("—", " - ").replace("–", " - ").replace("…", "...")
    return text.strip()


def parse_tags_to_vector(emotions_raw: Dict[str, Any], styles_raw: Dict[str, Any]) -> List[float]:
    """Converts emotion and delivery style dictionaries into a 24-d float vector."""
    vec = [0.0] * len(ALL_TAGS)

    def process_dict(d):
        for k, v in d.items():
            k_clean = TAG_ALIASES.get(k.lower().strip(), k.lower().strip())
            if k_clean in TAG_TO_IDX:
                try:
                    val = float(v)
                    vec[TAG_TO_IDX[k_clean]] = max(0.0, min(1.0, val))
                except (ValueError, TypeError):
                    vec[TAG_TO_IDX[k_clean]] = 1.0

    if isinstance(emotions_raw, dict):
        process_dict(emotions_raw)
    if isinstance(styles_raw, dict):
        process_dict(styles_raw)

    return vec


def try_phonemize(text: str, phonemizer_obj=None) -> str:
    """Runs IPA phonemization if phonemizer is available, else returns cleaned text."""
    if phonemizer_obj is not None:
        try:
            res = phonemizer_obj.phonemize([text])
            if isinstance(res, (list, tuple)) and len(res) > 0:
                return res[0].strip()
        except Exception:
            pass
    return text


def build_ood_texts(output_path: str):
    """Generates standard Out-Of-Distribution evaluation sentences covering all emotions."""
    ood_sentences = [
        '[neutral] "Antigravity voice synthesis operational on neural cluster."',
        '[happy=0.9, excited=0.8] "We did it! The full training pipeline converged with pristine acoustic fidelity!"',
        '[sarcasm=0.9] "Oh, brilliant. Another zero-division warning to brighten my morning."',
        '[angry=0.85] "I told you three times already, do not interrupt the vocoder backprop step!"',
        '[soothing=0.8, calm=0.7] "Take a deep breath. The loss curves are steadily dropping toward zero."',
        '[curious=0.85] "Wait, how did the latent style vector maintain its boundary without collapsing?"',
        '[sad=0.8, heartbroken=0.6] "I watched the whole model train for thirty epochs, only for the weights to vanish."',
        '[authoritative=0.9] "Initialize the GAN discriminators and enforce multi-resolution STFT immediately."',
        '[playful=0.8, teasing=0.6] "Did you really think a dummy amplitude loss would make a human voice speak?"',
        '[deadpan=0.85] "Affirmative. I am an emotional synthetic intelligence possessing twenty-four discrete delivery styles."',
        '[overjoyed=0.95] "Listen to that timbre! It sounds exactly like a real human in a soundproof studio!"',
        '[dramatic=0.9] "Deep in the silicon corridors, the weights aligned, and the silence gave way to speech."',
        '[frustrated=0.8] "The gradient norm exploded right as we were about to hit the validation checkpoint!"',
        '[affectionate=0.85] "You worked so hard to build this neural pipeline, and it was worth every single step."',
        '[confused=0.75] "Why is the pitch curve shifting upwards when the text has a falling cadence?"',
        '[bored=0.8] "Another five hundred thousand steps of score matching. Wake me up when the loss hits zero point one."',
        '[surprised=0.9] "Wait! The HiFi-GAN upsampling actually ran without running out of GPU memory!"'
    ]
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for s in ood_sentences:
            f.write(s + "\n")
    print(f"[✓] OOD evaluation texts saved to: {output_path}")


def discover_dataset_source(explicit_path: Optional[str] = None) -> str:
    """Auto-discovers the dataset source across Kaggle input, tar files, or directories."""
    if explicit_path and os.path.exists(explicit_path):
        return explicit_path

    # Candidate locations
    candidates = [
        # Local workspace
        "DataSet/data/kion_dataset.tar",
        "DataSet/data",
        "DataSet",
        # Kaggle input paths
        "/kaggle/input/kion-dataset",
        "/kaggle/input/kiontts",
        "/kaggle/input/kiontts-dataset",
        "/kaggle/input/kion-dataset/kion_dataset.tar",
        "/kaggle/input/kiontts/DataSet/data/kion_dataset.tar",
    ]

    for cand in candidates:
        if os.path.exists(cand):
            # Check if directory actually contains data
            if os.path.isdir(cand):
                contents = os.listdir(cand)
                if contents:
                    return cand
            else:
                return cand

    # Scan /kaggle/input recursively for any folder containing 'kion' or 'dataset'
    if os.path.exists("/kaggle/input"):
        for root, dirs, files in os.walk("/kaggle/input"):
            if "kion_dataset.tar" in files:
                return os.path.join(root, "kion_dataset.tar")
            for f in files:
                if f.endswith(".zip") and ("train" in f.lower() or "batch" in f.lower()):
                    return root
            for d in dirs:
                if "kion" in d.lower() or "dataset" in d.lower():
                    cand_dir = os.path.join(root, d)
                    return cand_dir

    raise FileNotFoundError("Could not auto-discover KionTTS dataset in workspace or /kaggle/input.")


def process_entries(
    meta_items: List[Dict[str, Any]],
    get_wav_bytes_or_path_fn,
    wav_dir: str,
    manifest_records: List[Dict[str, Any]],
    style_text_list: List[str],
    phonemizer_obj=None,
    seen_ids=None,
    max_samples=None,
) -> int:
    """Helper to parse metadata entries and record clean text, tags, and audio links."""
    if seen_ids is None:
        seen_ids = set()

    added = 0
    for entry in meta_items:
        if max_samples and len(manifest_records) >= max_samples:
            break

        uid = entry.get("id")
        if not uid or uid in seen_ids:
            continue

        segs = entry.get("segments", [])
        if segs and len(segs) == 1:
            seg = segs[0]
            clean_text = clean_text_for_phonemizer(seg.get("text", ""))
            emotions = seg.get("emotions", {})
            styles = seg.get("styles", {})
        else:
            clean_text = clean_text_for_phonemizer(entry.get("text", ""))
            emotions = entry.get("emotions", {})
            styles = entry.get("styles", {})

        if len(clean_text) < 4:
            continue

        audio_src = get_wav_bytes_or_path_fn(uid)
        if not audio_src:
            continue

        dest_wav_path = os.path.join(wav_dir, f"{uid}.wav")

        if isinstance(audio_src, (bytes, bytearray)):
            if not os.path.exists(dest_wav_path):
                with open(dest_wav_path, "wb") as f_out:
                    f_out.write(audio_src)
        elif isinstance(audio_src, str):
            # File already on disk (e.g. Kaggle uncompressed directory)
            if not os.path.exists(dest_wav_path) and not os.path.islink(dest_wav_path):
                try:
                    os.symlink(os.path.abspath(audio_src), dest_wav_path)
                except Exception:
                    try:
                        shutil.copy2(audio_src, dest_wav_path)
                    except Exception:
                        dest_wav_path = audio_src

        tag_vector = parse_tags_to_vector(emotions, styles)
        phonemes = try_phonemize(clean_text, phonemizer_obj)

        record = {
            "id": uid,
            "clean_text": clean_text,
            "phonemes": phonemes,
            "emotions": emotions,
            "styles": styles,
            "tag_vector": tag_vector,
            "wav_path": f"wavs/{uid}.wav",
            "speaker": "kion",
        }
        manifest_records.append(record)
        seen_ids.add(uid)

        rel_wav_path = os.path.join("wavs", f"{uid}.wav")
        style_text_list.append(f"{rel_wav_path}|{phonemes}|0")
        added += 1

    return added


def prepare_kion_dataset(
    source_path: Optional[str] = None,
    output_dir: str = "DataSet",
    max_samples: Optional[int] = None,
    phonemize: bool = True,
):
    """
    Unified dataset unpacker and manifest generator supporting:
    - Raw .tar archives
    - Kaggle auto-extracted directories with nested batch zips
    - Loose extracted directories containing metadata.json and wavs/
    """
    source_path = discover_dataset_source(source_path)
    print(f"[✓] Using dataset source: {source_path}")

    wav_dir = os.path.join(output_dir, "wavs")
    os.makedirs(wav_dir, exist_ok=True)

    phonemizer_obj = None
    if phonemize:
        try:
            from phonemizer.backend import EspeakBackend
            phonemizer_obj = EspeakBackend(language="en-us", preserve_punctuation=True, with_stress=True)
            print("[✓] IPA EspeakBackend initialized for pre-phonemization.")
        except Exception as e:
            print(f"[-] Phonemizer not available ({e}). Raw text will be used for on-the-fly phonemization.")

    manifests = {"train": [], "val": []}
    style_text_lists = {"train": [], "val": []}
    seen_ids = set()

    # ─────────────────────────────────────────────────────────────────────────
    # CASE 1: Source is a TAR archive (.tar, .tar.gz)
    # ─────────────────────────────────────────────────────────────────────────
    if os.path.isfile(source_path) and (".tar" in source_path.lower()):
        print(f"[*] Extracting from tar archive: {source_path}")
        with tarfile.open(source_path, "r:*") as tar:
            splits = [
                ("KionTTS_Dataset_train.zip", "train"),
                ("KionTTS_Dataset_val.zip", "val"),
            ]
            for z_name, split_key in splits:
                try:
                    f_tar = tar.extractfile(z_name)
                except KeyError:
                    # Search inside tar
                    found_member = None
                    for m in tar.getmembers():
                        if z_name.lower() in m.name.lower():
                            found_member = m
                            break
                    if found_member:
                        f_tar = tar.extractfile(found_member)
                    else:
                        print(f"[-] {z_name} not found in tar. Skipping split.")
                        continue

                master_zip = zipfile.ZipFile(io.BytesIO(f_tar.read()))
                batch_names = [n for n in master_zip.namelist() if n.endswith(".zip")]
                print(f"    Found {len(batch_names)} batch archives in {z_name}.")

                for b_name in batch_names:
                    if max_samples and len(manifests[split_key]) >= max_samples:
                        break
                    bz = zipfile.ZipFile(io.BytesIO(master_zip.read(b_name)))
                    meta_item = None
                    for fname in bz.namelist():
                        if fname.endswith("metadata.json"):
                            meta_item = json.loads(bz.read(fname).decode("utf-8"))
                            break
                    if not meta_item:
                        continue

                    audio_map = {
                        os.path.splitext(os.path.basename(f))[0]: f
                        for f in bz.namelist() if f.endswith(".wav")
                    }

                    def get_wav_fn(uid):
                        if uid in audio_map:
                            return bz.read(audio_map[uid])
                        return None

                    process_entries(
                        meta_items=meta_item,
                        get_wav_bytes_or_path_fn=get_wav_fn,
                        wav_dir=wav_dir,
                        manifest_records=manifests[split_key],
                        style_text_list=style_text_lists[split_key],
                        phonemizer_obj=phonemizer_obj,
                        seen_ids=seen_ids,
                        max_samples=max_samples,
                    )

    # ─────────────────────────────────────────────────────────────────────────
    # CASE 2: Source is a Directory (Kaggle extracted directory or local folder)
    # ─────────────────────────────────────────────────────────────────────────
    else:
        print(f"[*] Processing directory: {source_path}")

        # Sub-case 2A: Check if manifests already exist in source directory
        pre_train_manifest = os.path.join(source_path, "train_manifest.json")
        if os.path.exists(pre_train_manifest):
            print(f"[✓] Found pre-existing train_manifest.json in: {pre_train_manifest}")
            with open(pre_train_manifest, "r", encoding="utf-8") as f:
                manifests["train"] = json.load(f)
            pre_val_manifest = os.path.join(source_path, "val_manifest.json")
            if os.path.exists(pre_val_manifest):
                with open(pre_val_manifest, "r", encoding="utf-8") as f:
                    manifests["val"] = json.load(f)

            src_wavs = os.path.join(source_path, "wavs")
            if os.path.exists(src_wavs) and src_wavs != os.path.abspath(wav_dir):
                for w in os.listdir(src_wavs):
                    if w.endswith(".wav"):
                        s_f = os.path.join(src_wavs, w)
                        d_f = os.path.join(wav_dir, w)
                        if not os.path.exists(d_f):
                            try:
                                os.symlink(os.path.abspath(s_f), d_f)
                            except Exception:
                                shutil.copy2(s_f, d_f)

            # Rebuild style text lists
            for item in manifests["train"]:
                style_text_lists["train"].append(f"{item['wav_path']}|{item['phonemes']}|0")
            for item in manifests["val"]:
                style_text_lists["val"].append(f"{item['wav_path']}|{item['phonemes']}|0")

        # Sub-case 2B: Check for master zips inside directory (e.g. KionTTS_Dataset_train.zip)
        master_zips = glob.glob(os.path.join(source_path, "**", "*.zip"), recursive=True)
        train_zips = [z for z in master_zips if "train" in os.path.basename(z).lower()]
        val_zips = [z for z in master_zips if "val" in os.path.basename(z).lower()]

        # If master zips found and manifests not already loaded
        if (train_zips or val_zips) and len(manifests["train"]) == 0:
            for z_path, split_key in [(train_zips, "train"), (val_zips, "val")]:
                for master_z in z_path:
                    print(f"    Scanning zip: {master_z}")
                    with zipfile.ZipFile(master_z, "r") as mz:
                        # Check if this zip contains sub-batch zips or direct files
                        sub_zips = [n for n in mz.namelist() if n.endswith(".zip")]
                        if sub_zips:
                            for b_name in sub_zips:
                                if max_samples and len(manifests[split_key]) >= max_samples:
                                    break
                                bz = zipfile.ZipFile(io.BytesIO(mz.read(b_name)))
                                meta_item = None
                                for fname in bz.namelist():
                                    if fname.endswith("metadata.json"):
                                        meta_item = json.loads(bz.read(fname).decode("utf-8"))
                                        break
                                if not meta_item:
                                    continue
                                audio_map = {
                                    os.path.splitext(os.path.basename(f))[0]: f
                                    for f in bz.namelist() if f.endswith(".wav")
                                }
                                def get_wav_fn(uid):
                                    if uid in audio_map:
                                        return bz.read(audio_map[uid])
                                    return None

                                process_entries(
                                    meta_items=meta_item,
                                    get_wav_bytes_or_path_fn=get_wav_fn,
                                    wav_dir=wav_dir,
                                    manifest_records=manifests[split_key],
                                    style_text_list=style_text_lists[split_key],
                                    phonemizer_obj=phonemizer_obj,
                                    seen_ids=seen_ids,
                                    max_samples=max_samples,
                                )
                        else:
                            # Direct master zip with metadata.json and wavs
                            meta_item = None
                            for fname in mz.namelist():
                                if fname.endswith("metadata.json"):
                                    meta_item = json.loads(mz.read(fname).decode("utf-8"))
                                    break
                            if meta_item:
                                audio_map = {
                                    os.path.splitext(os.path.basename(f))[0]: f
                                    for f in mz.namelist() if f.endswith(".wav")
                                }
                                def get_wav_fn(uid):
                                    if uid in audio_map:
                                        return mz.read(audio_map[uid])
                                    return None

                                process_entries(
                                    meta_items=meta_item,
                                    get_wav_bytes_or_path_fn=get_wav_fn,
                                    wav_dir=wav_dir,
                                    manifest_records=manifests[split_key],
                                    style_text_list=style_text_lists[split_key],
                                    phonemizer_obj=phonemizer_obj,
                                    seen_ids=seen_ids,
                                    max_samples=max_samples,
                                )

        # Sub-case 2B: Pre-extracted directories with loose metadata.json files (Kaggle auto-unpacked)
        loose_metas = glob.glob(os.path.join(source_path, "**", "metadata.json"), recursive=True)
        if loose_metas:
            print(f"    Found {len(loose_metas)} extracted metadata.json files on disk.")
            for meta_path in loose_metas:
                split_key = "val" if "val" in meta_path.lower() else "train"
                if max_samples and len(manifests[split_key]) >= max_samples:
                    continue

                try:
                    with open(meta_path, "r", encoding="utf-8") as f:
                        meta_item = json.load(f)
                except Exception:
                    continue

                batch_dir = os.path.dirname(meta_path)
                wavs_sub = os.path.join(batch_dir, "wavs")

                def get_wav_fn(uid):
                    for cand in [
                        os.path.join(wavs_sub, f"{uid}.wav"),
                        os.path.join(batch_dir, f"{uid}.wav"),
                        os.path.join(source_path, "wavs", f"{uid}.wav"),
                    ]:
                        if os.path.exists(cand):
                            return cand
                    return None

                process_entries(
                    meta_items=meta_item,
                    get_wav_bytes_or_path_fn=get_wav_fn,
                    wav_dir=wav_dir,
                    manifest_records=manifests[split_key],
                    style_text_list=style_text_lists[split_key],
                    phonemizer_obj=phonemizer_obj,
                    seen_ids=seen_ids,
                    max_samples=max_samples,
                )

    # If val split is empty, take 5% from train
    if len(manifests["val"]) == 0 and len(manifests["train"]) > 20:
        val_count = max(5, int(len(manifests["train"]) * 0.05))
        manifests["val"] = manifests["train"][-val_count:]
        manifests["train"] = manifests["train"][:-val_count]
        style_text_lists["val"] = style_text_lists["train"][-val_count:]
        style_text_lists["train"] = style_text_lists["train"][:-val_count]

    # Save manifests
    train_json_path = os.path.join(output_dir, "train_manifest.json")
    val_json_path = os.path.join(output_dir, "val_manifest.json")
    with open(train_json_path, "w", encoding="utf-8") as f:
        json.dump(manifests["train"], f, indent=2)
    with open(val_json_path, "w", encoding="utf-8") as f:
        json.dump(manifests["val"], f, indent=2)

    # Save StyleTTS2 text lists
    train_txt_path = os.path.join(output_dir, "train_list.txt")
    val_txt_path = os.path.join(output_dir, "val_list.txt")
    with open(train_txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(style_text_lists["train"]) + "\n")
    with open(val_txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(style_text_lists["val"]) + "\n")

    # Generate OOD texts
    ood_path = os.path.join(output_dir, "OOD_texts.txt")
    build_ood_texts(ood_path)

    print("\n" + "=" * 60)
    print("Dataset Preparation Completed Successfully!")
    print(f"  Source Used              : {source_path}")
    print(f"  Total Training Samples   : {len(manifests['train'])}")
    print(f"  Total Validation Samples : {len(manifests['val'])}")
    print(f"  WAV Audio Directory      : {wav_dir}")
    print(f"  Train Manifest (JSON)    : {train_json_path}")
    print(f"  Val Manifest (JSON)      : {val_json_path}")
    print(f"  Train List (StyleTTS2)   : {train_txt_path}")
    print(f"  Val List (StyleTTS2)     : {val_txt_path}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare KionTTS production dataset.")
    parser.add_argument("--source", type=str, default=None, help="Path to kion_dataset.tar or extracted Kaggle folder")
    parser.add_argument("--tar_path", type=str, default=None, help="Alias for --source: path to kion_dataset.tar")
    parser.add_argument("--output_dir", type=str, default="DataSet", help="Output directory for manifests and audio")
    parser.add_argument("--max_samples", type=int, default=None, help="Optional sample limit for quick smoke test")
    parser.add_argument("--no_phonemize", action="store_true", help="Skip pre-phonemization")
    args = parser.parse_args()

    src = args.source or args.tar_path
    prepare_kion_dataset(
        source_path=src,
        output_dir=args.output_dir,
        max_samples=args.max_samples,
        phonemize=(not args.no_phonemize),
    )
