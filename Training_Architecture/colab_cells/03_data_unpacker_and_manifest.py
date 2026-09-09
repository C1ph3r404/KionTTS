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


import tarfile

def unpack_tar_if_needed(search_dirs: Optional[List[str]] = None, dest_dir: str = "/content/data") -> bool:
    """Checks for kion_dataset.tar in Kaggle input or common search paths and extracts it."""
    if search_dirs is None:
        search_dirs = [
            "/kaggle/input",
            "/kaggle/input/kion-dataset",
            "/kaggle/input/kiontts-dataset",
            "/kaggle/input/kiontts",
            "/content/data",
            "/content",
            "DatasetGeneration/data",
            "../DatasetGeneration/data",
        ]
    # Check if train zip already exists
    if os.path.exists(os.path.join(dest_dir, "KionTTS_Dataset_train.zip")):
        return True

    for sdir in search_dirs:
        if not os.path.exists(sdir):
            continue
        # Search directly or recursively for .tar files
        candidates = glob.glob(os.path.join(sdir, "**", "*.tar"), recursive=True) + glob.glob(os.path.join(sdir, "*.tar"))
        for tar_cand in candidates:
            if "kion" in os.path.basename(tar_cand).lower() or "dataset" in os.path.basename(tar_cand).lower():
                print(f"[+] Found dataset tar archive at: {tar_cand}")
                print(f"[*] Extracting tar archive to {dest_dir}...")
                os.makedirs(dest_dir, exist_ok=True)
                with tarfile.open(tar_cand, "r") as tar:
                    tar.extractall(path=dest_dir)
                print(f"[✓] Tar extraction complete: {dest_dir}")
                return True
    return False


def find_dataset_zip(filename: str) -> str:
    # First ensure any tar archives in Kaggle/Colab are extracted
    dest_data_dir = "/kaggle/working/data" if os.path.exists("/kaggle") else "/content/data"
    unpack_tar_if_needed(dest_dir=dest_data_dir)

    search_paths = [
        os.path.join(dest_data_dir, filename),
        os.path.join("/kaggle/working/data", filename),
        os.path.join("/kaggle/working", filename),
        os.path.join("/kaggle/input/kion-dataset", filename),
        os.path.join("/kaggle/input/kiontts-dataset", filename),
        os.path.join("/kaggle/input", filename),
        os.path.join("/content/data", filename),
        os.path.join("/content/drive/MyDrive", filename),
        os.path.join("/content/drive/MyDrive/KionTTS_Data", filename),
        os.path.join("/content/drive/MyDrive/dataset", filename),
        os.path.join("DatasetGeneration/data", filename),
        os.path.join("../DatasetGeneration/data", filename),
    ]
    # Also check recursive matches in /kaggle/input
    if os.path.exists("/kaggle/input"):
        for root, _, files in os.walk("/kaggle/input"):
            if filename in files:
                return os.path.join(root, filename)

    for p in search_paths:
        if os.path.exists(p):
            print(f"[+] Found {filename} at: {p}")
            return p
    return os.path.join(dest_data_dir, filename)


def _get_repo_root() -> str:
    rel_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    if os.path.exists(os.path.join(rel_path, "model")):
        return rel_path
    for p in [
        "/kaggle/working/KionTTS",
        "/kaggle/working/kiontts",
        "/kaggle/working",
        "/content/KionTTS",
        "/content/Kiontts",
        "/content/kiontts",
    ]:
        if os.path.exists(os.path.join(p, "model")):
            return p
        if os.path.exists(p):
            return p
    return "/kaggle/working" if os.path.exists("/kaggle") else "/content/KionTTS"


REPO_ROOT = _get_repo_root()
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


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
                fout.write(f"{wav_path}|{phoneme_str}|0\n")
                written += 1
            except Exception as e:
                print(f"  [WARN] Skipping {rec.get('id', '?')}: {e}")

    print(f"[+] StyleTTS2 list written: {written} entries → {output_txt_path}")
    return written


def run_extraction_pipeline(
    train_zip: Optional[str] = None,
    val_zip: Optional[str] = None,
    wav_dir: Optional[str] = None,
    manifest_dir: Optional[str] = None,
    styletts2_data_dir: Optional[str] = None,
):
    base_data = "/kaggle/working/dataset" if os.path.exists("/kaggle") else "/content/dataset"
    if wav_dir is None:
        wav_dir = os.path.join(base_data, "wavs")
    if manifest_dir is None:
        manifest_dir = base_data

    if train_zip is None or not os.path.exists(train_zip):
        train_zip = find_dataset_zip("KionTTS_Dataset_train.zip")
    if val_zip is None or not os.path.exists(val_zip):
        val_zip = find_dataset_zip("KionTTS_Dataset_val.zip")

    if styletts2_data_dir is None:
        styletts2_data_dir = os.path.join(REPO_ROOT, "StyleTTS2", "Data")

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

