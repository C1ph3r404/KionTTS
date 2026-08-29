"""
Colab Cell 03: Data Unpacker and Manifest Generator
Unpacks the train and validation zip archives (including internal batch zips),
consolidates WAV audio files, extracts and normalizes emotion & style tags,
and outputs unified JSON training and validation manifests.
"""

import os
import io
import json
import zipfile
import glob
from typing import List, Dict, Any, Optional
from tqdm import tqdm

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from model.data.style_tag_parser import parse_tagged_text, create_style_vector, EMOTIONS, STYLES


def unpack_and_generate_manifest(
    zip_path: str,
    output_wav_dir: str,
    output_manifest_path: str,
    split_name: str = "train",
) -> List[Dict[str, Any]]:
    """
    Unpacks a master dataset zip file (which contains nested batch_XXXX.zip files)
    and generates a clean, structured manifest.
    """
    print(f"\n{'='*60}")
    print(f"Processing {split_name.upper()} dataset from: {zip_path}")
    print(f"Target WAV directory: {output_wav_dir}")
    print(f"Target Manifest: {output_manifest_path}")
    print(f"{'='*60}")

    os.makedirs(output_wav_dir, exist_ok=True)
    manifest_records: List[Dict[str, Any]] = []

    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"Dataset zip file not found at: {zip_path}")

    with zipfile.ZipFile(zip_path, "r") as master_zip:
        inner_zips = [f for f in master_zip.namelist() if f.endswith(".zip")]
        print(f"Found {len(inner_zips)} batch zip archives inside {os.path.basename(zip_path)}.")

        for inner_zip_name in tqdm(inner_zips, desc=f"Unpacking {split_name} batches"):
            inner_bytes = master_zip.read(inner_zip_name)
            with zipfile.ZipFile(io.BytesIO(inner_bytes), "r") as batch_zip:
                # Read batch metadata.json
                meta_content = None
                for fname in batch_zip.namelist():
                    if fname.endswith("metadata.json"):
                        meta_content = json.loads(batch_zip.read(fname).decode("utf-8"))
                        break

                if not meta_content:
                    continue

                # Extract audio files and match with metadata entries
                for entry in meta_content:
                    uid = entry.get("id")
                    raw_text = entry.get("text", "")
                    clean_text, emotions, styles = parse_tagged_text(raw_text)
                    style_vector = create_style_vector(emotions, styles).tolist()

                    # Locate WAV inside inner zip
                    wav_entry_name = None
                    for name in batch_zip.namelist():
                        if name.endswith(f"{uid}.wav") or name.endswith(f"{uid}.mp3"):
                            wav_entry_name = name
                            break

                    if wav_entry_name:
                        wav_filename = f"{uid}.wav"
                        dest_wav_path = os.path.join(output_wav_dir, wav_filename)
                        if not os.path.exists(dest_wav_path):
                            with open(dest_wav_path, "wb") as f_out:
                                f_out.write(batch_zip.read(wav_entry_name))

                        record = {
                            "id": uid,
                            "raw_text": raw_text,
                            "clean_text": clean_text,
                            "emotions": emotions,
                            "styles": styles,
                            "style_vector": style_vector,
                            "wav_path": dest_wav_path,
                            "speaker": "Kion",
                            "split": split_name,
                        }
                        manifest_records.append(record)

    print(f"[+] Total samples extracted for {split_name}: {len(manifest_records)}")

    # Save manifest
    os.makedirs(os.path.dirname(output_manifest_path), exist_ok=True)
    with open(output_manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_records, f, indent=2, ensure_ascii=False)
    print(f"[+] Manifest written to: {output_manifest_path}")

    return manifest_records


def find_dataset_zip(filename: str) -> str:
    search_paths = [
        os.path.join("/content/data", filename),
        os.path.join("/content/drive/MyDrive", filename),
        os.path.join("/content/drive/MyDrive/KionTTS_Data", filename),
        os.path.join("/content/drive/MyDrive/dataset", filename),
        os.path.join("DatasetGeneration/data", filename),
        os.path.join("../DatasetGeneration/data", filename),
    ]
    for p in search_paths:
        if os.path.exists(p):
            print(f"[+] Found {filename} at: {p}")
            return p
    return os.path.join("/content/data", filename)


def generate_styletts2_lists(
    manifest_path: str,
    output_txt_path: str,
    phonemizer_backend: str = "espeak",
) -> int:
    """
    Converts a Kion JSON manifest into the StyleTTS2 train_list.txt format:

        /path/to/audio.wav|phoneme_sequence

    This is the format StyleTTS2's meldataset.py expects.
    Phonemization is performed here so we avoid re-doing it at runtime.

    Returns:
        Number of entries written.
    """
    import sys

    # Ensure repo paths are available for phonemizer
    REPO_ROOT = "/content/KionTTS"
    if REPO_ROOT not in sys.path:
        sys.path.insert(0, REPO_ROOT)

    from model.data.phonemizer_util import phonemize_text  # noqa: E402

    with open(manifest_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    os.makedirs(os.path.dirname(output_txt_path), exist_ok=True)
    written = 0

    with open(output_txt_path, "w", encoding="utf-8") as fout:
        for rec in tqdm(records, desc=f"Generating {os.path.basename(output_txt_path)}"):
            wav_path   = rec["wav_path"]
            clean_text = rec["clean_text"]

            if not os.path.exists(wav_path):
                continue

            try:
                phoneme_ids = phonemize_text(clean_text)
                # StyleTTS2 expects space-separated integer IDs as the second column
                phoneme_str = " ".join(str(p) for p in phoneme_ids)
                fout.write(f"{wav_path}|{phoneme_str}\n")
                written += 1
            except Exception as e:
                print(f"  [WARN] Skipping {rec.get('id', '?')}: {e}")

    print(f"[+] StyleTTS2 list written: {written} entries → {output_txt_path}")
    return written


def run_extraction_pipeline(
    train_zip: Optional[str] = None,
    val_zip: Optional[str] = None,
    wav_dir: str = "/content/dataset/wavs",
    manifest_dir: str = "/content/dataset",
    styletts2_data_dir: str = "/content/KionTTS/StyleTTS2/Data",
):
    if train_zip is None or not os.path.exists(train_zip):
        train_zip = find_dataset_zip("KionTTS_Dataset_train.zip")
    if val_zip is None or not os.path.exists(val_zip):
        val_zip = find_dataset_zip("KionTTS_Dataset_val.zip")

    train_manifest = os.path.join(manifest_dir, "train_manifest.json")
    val_manifest   = os.path.join(manifest_dir, "val_manifest.json")

    # Step 1: Unpack zips → JSON manifests
    train_records = unpack_and_generate_manifest(train_zip, wav_dir, train_manifest, split_name="train")
    val_records   = unpack_and_generate_manifest(val_zip,   wav_dir, val_manifest,   split_name="val")

    # Step 2: Generate StyleTTS2-format .txt lists
    os.makedirs(styletts2_data_dir, exist_ok=True)
    train_txt = os.path.join(styletts2_data_dir, "kion_train_list.txt")
    val_txt   = os.path.join(styletts2_data_dir, "kion_val_list.txt")

    n_train = generate_styletts2_lists(train_manifest, train_txt)
    n_val   = generate_styletts2_lists(val_manifest,   val_txt)

    print("\n" + "=" * 60)
    print(f"  Train samples : {n_train}")
    print(f"  Val samples   : {n_val}")
    print(f"  StyleTTS2 lists → {styletts2_data_dir}")
    print("=" * 60)
    print("\n[Cell 03 Complete] Dataset ready. Proceed to Cell 04 for feature pre-computation.")


if __name__ == "__main__":
    run_extraction_pipeline()
