"""
Colab Cell 03: Data Unpacker and Manifest Generator
Unpacks or processes the train and validation datasets (supporting nested batch zips,
pre-extracted directories from Kaggle datasets, or direct zip archives), consolidates
WAV audio files via symlinks/extraction, extracts and normalizes emotion & style tags,
and outputs unified JSON manifests and StyleTTS2-formatted text lists.
"""

import os
import io
import json
import zipfile
import glob
import shutil
import tarfile
from typing import List, Dict, Any, Optional
from tqdm import tqdm

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from model.data.style_tag_parser import parse_tagged_text, create_style_vector, EMOTIONS, STYLES


def _process_metadata_entries(
    meta_content: Any,
    output_wav_dir: str,
    manifest_records: List[Dict[str, Any]],
    split_name: str,
    seen_ids: set,
    get_wav_bytes_or_path_fn,
) -> int:
    """Helper to parse metadata entries, link/extract wav audio, and append to manifest_records."""
    if isinstance(meta_content, dict):
        meta_content = list(meta_content.values())

    added = 0
    for entry in meta_content:
        uid = entry.get("id")
        if not uid or uid in seen_ids:
            continue

        raw_text = entry.get("text", "")
        clean_text, emotions, styles = parse_tagged_text(raw_text)
        style_vector = create_style_vector(emotions, styles).tolist()

        audio_source = get_wav_bytes_or_path_fn(uid)
        if not audio_source:
            continue

        dest_wav_path = os.path.join(output_wav_dir, f"{uid}.wav")

        if isinstance(audio_source, (bytes, bytearray)):
            if not os.path.exists(dest_wav_path):
                with open(dest_wav_path, "wb") as f_out:
                    f_out.write(audio_source)
        elif isinstance(audio_source, str):
            # File already on disk (e.g. pre-extracted Kaggle dataset)
            if not os.path.islink(dest_wav_path) and not os.path.exists(dest_wav_path):
                try:
                    os.symlink(audio_source, dest_wav_path)
                except FileExistsError:
                    pass
                except Exception:
                    try:
                        shutil.copy2(audio_source, dest_wav_path)
                    except Exception as e:
                        print(f"  [WARN] Failed to link/copy {audio_source} to {dest_wav_path}: {e}")
                        dest_wav_path = audio_source

        seen_ids.add(uid)
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
        added += 1

    return added


def _process_batch_zip(
    batch_zip: zipfile.ZipFile,
    output_wav_dir: str,
    manifest_records: List[Dict[str, Any]],
    split_name: str,
    seen_ids: set,
):
    """Processes a single batch zip archive in-memory or from disk."""
    meta_content = None
    for fname in batch_zip.namelist():
        if fname.endswith("metadata.json"):
            meta_content = json.loads(batch_zip.read(fname).decode("utf-8"))
            break
    if not meta_content:
        return

    name_map = {}
    for name in batch_zip.namelist():
        if name.endswith((".wav", ".mp3", ".flac")):
            base = os.path.basename(name)
            name_map[base] = name
            name_map[os.path.splitext(base)[0]] = name

    def get_wav(uid: str):
        target = name_map.get(uid) or name_map.get(f"{uid}.wav") or name_map.get(f"{uid}.mp3")
        if target:
            return batch_zip.read(target)
        return None

    _process_metadata_entries(meta_content, output_wav_dir, manifest_records, split_name, seen_ids, get_wav)


def _process_extracted_batch_dir(
    meta_file_path: str,
    output_wav_dir: str,
    manifest_records: List[Dict[str, Any]],
    split_name: str,
    seen_ids: set,
):
    """Processes an extracted batch directory on disk (e.g. Kaggle unzipped dataset)."""
    try:
        with open(meta_file_path, "r", encoding="utf-8") as f:
            meta_content = json.load(f)
    except Exception as e:
        print(f"  [WARN] Failed reading {meta_file_path}: {e}")
        return

    batch_dir = os.path.dirname(meta_file_path)
    wavs_dir = os.path.join(batch_dir, "wavs")

    def get_wav(uid: str):
        for cand in [
            os.path.join(wavs_dir, f"{uid}.wav"),
            os.path.join(batch_dir, f"{uid}.wav"),
            os.path.join(wavs_dir, f"{uid}.mp3"),
            os.path.join(batch_dir, f"{uid}.mp3"),
        ]:
            if os.path.exists(cand):
                return cand
        return None

    _process_metadata_entries(meta_content, output_wav_dir, manifest_records, split_name, seen_ids, get_wav)


def unpack_and_generate_manifest(
    source_path: str,
    output_wav_dir: str,
    output_manifest_path: str,
    split_name: str = "train",
) -> List[Dict[str, Any]]:
    """
    Unpacks a master dataset zip or processes an extracted directory (which contains
    nested batch directories or batch zips) and generates a clean, structured manifest.
    """
    print(f"\n{'='*60}")
    print(f"Processing {split_name.upper()} dataset from: {source_path}")
    print(f"Target WAV directory: {output_wav_dir}")
    print(f"Target Manifest: {output_manifest_path}")
    print(f"{'='*60}")

    os.makedirs(output_wav_dir, exist_ok=True)
    manifest_records: List[Dict[str, Any]] = []
    seen_ids = set()

    if not os.path.exists(source_path):
        raise FileNotFoundError(f"Dataset path not found at: {source_path}")

    if os.path.isfile(source_path) and source_path.endswith(".zip"):
        with zipfile.ZipFile(source_path, "r") as master_zip:
            inner_zips = [f for f in master_zip.namelist() if f.endswith(".zip")]
            if inner_zips:
                print(f"Found {len(inner_zips)} batch zip archives inside {os.path.basename(source_path)}.")
                for inner_zip_name in tqdm(inner_zips, desc=f"Unpacking {split_name} batches"):
                    inner_bytes = master_zip.read(inner_zip_name)
                    with zipfile.ZipFile(io.BytesIO(inner_bytes), "r") as batch_zip:
                        _process_batch_zip(batch_zip, output_wav_dir, manifest_records, split_name, seen_ids)
            else:
                print(f"Processing flat zip archive: {os.path.basename(source_path)}")
                _process_batch_zip(master_zip, output_wav_dir, manifest_records, split_name, seen_ids)

    elif os.path.isdir(source_path):
        # 1. Check for batch_*.zip archives in directory
        batch_zips = []
        for root, _, files in os.walk(source_path):
            for f in files:
                if f.endswith(".zip") and ("batch" in f.lower() or "kion" in f.lower()):
                    batch_zips.append(os.path.join(root, f))

        if batch_zips:
            batch_zips.sort()
            print(f"Found {len(batch_zips)} batch zip archives inside {source_path}.")
            for bzip_path in tqdm(batch_zips, desc=f"Unpacking {split_name} batches"):
                with zipfile.ZipFile(bzip_path, "r") as batch_zip:
                    _process_batch_zip(batch_zip, output_wav_dir, manifest_records, split_name, seen_ids)
        else:
            # 2. Check for extracted metadata.json files (pre-extracted dataset like on Kaggle)
            meta_files = []
            for root, _, files in os.walk(source_path):
                if "metadata.json" in files:
                    meta_files.append(os.path.join(root, "metadata.json"))

            if meta_files:
                meta_files.sort()
                print(f"Found {len(meta_files)} extracted batch metadata files inside {source_path}.")
                for meta_file in tqdm(meta_files, desc=f"Processing {split_name} batches"):
                    _process_extracted_batch_dir(meta_file, output_wav_dir, manifest_records, split_name, seen_ids)
            else:
                raise FileNotFoundError(f"No batch zip files or metadata.json found under directory: {source_path}")
    else:
        raise ValueError(f"Unsupported dataset source: {source_path}")

    print(f"[+] Total samples extracted for {split_name}: {len(manifest_records)}")

    # Save manifest
    os.makedirs(os.path.dirname(output_manifest_path), exist_ok=True)
    with open(output_manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_records, f, indent=2, ensure_ascii=False)
    print(f"[+] Manifest written to: {output_manifest_path}")

    return manifest_records


def unpack_tar_if_needed(search_dirs: Optional[List[str]] = None, dest_dir: str = "/kaggle/working/data") -> bool:
    """Checks for dataset tar archives in search paths and extracts them if found."""
    if search_dirs is None:
        search_dirs = [
            "/kaggle/input",
            "/kaggle/input/tts_dataset",
            "/kaggle/input/kion-dataset",
            "/kaggle/input/kiontts-dataset",
            "/kaggle/input/kiontts",
            "/kaggle/working/data",
            "/content/data",
            "/content",
            "DatasetGeneration/data",
            "../DatasetGeneration/data",
        ]
    # Check if train zip or train folder already exists in dest_dir
    if os.path.exists(os.path.join(dest_dir, "KionTTS_Dataset_train.zip")) or \
       os.path.exists(os.path.join(dest_dir, "KionTTS_Dataset_train")):
        return True

    tar_patterns = ["*.tar", "*.tar.gz", "*.tgz", "*.tar.bz2"]
    for sdir in search_dirs:
        if not os.path.exists(sdir):
            continue
        candidates = []
        for pat in tar_patterns:
            candidates.extend(glob.glob(os.path.join(sdir, "**", pat), recursive=True))
            candidates.extend(glob.glob(os.path.join(sdir, pat)))
        for tar_cand in set(candidates):
            c_name = os.path.basename(tar_cand).lower()
            if "kion" in c_name or "dataset" in c_name or "tts" in c_name:
                print(f"[+] Found dataset tar archive at: {tar_cand}")
                print(f"[*] Extracting tar archive to {dest_dir}...")
                os.makedirs(dest_dir, exist_ok=True)
                try:
                    with tarfile.open(tar_cand, "r:*") as tar:
                        tar.extractall(path=dest_dir)
                    print(f"[✓] Tar extraction complete: {dest_dir}")
                    return True
                except Exception as e:
                    print(f"[!] Warning: failed extracting {tar_cand}: {e}")
    return False


def find_dataset_source(split_name: str, search_dirs: Optional[List[str]] = None) -> str:
    """
    Locates the dataset source for a given split ('train' or 'val').
    Supports:
      1. Direct .zip archive (e.g. KionTTS_Dataset_train.zip)
      2. Extracted directory (e.g. /kaggle/input/tts_dataset/KionTTS_Dataset_train/...)
      3. Tar archive that needs unpacking
    """
    split_lower = split_name.lower()
    dest_data_dir = "/kaggle/working/data" if os.path.exists("/kaggle") else "/content/data"

    if search_dirs is None:
        search_dirs = [
            "/kaggle/input",
            dest_data_dir,
            "/kaggle/working/data",
            "/kaggle/working",
            "/content/data",
            "/content",
            "DatasetGeneration/data",
            "../DatasetGeneration/data",
        ]

    # First unpack any tar files if needed
    unpack_tar_if_needed(search_dirs=search_dirs, dest_dir=dest_data_dir)

    # 1. Search for .zip archives matching split
    for sdir in search_dirs:
        if not os.path.exists(sdir):
            continue
        for root, _, files in os.walk(sdir):
            for f in files:
                if f.endswith(".zip"):
                    f_lower = f.lower()
                    if split_lower in f_lower and ("kion" in f_lower or "dataset" in f_lower or f_lower == f"{split_lower}.zip"):
                        candidate = os.path.join(root, f)
                        print(f"[+] Found {split_name} zip archive at: {candidate}")
                        return candidate

    # 2. Search for pre-extracted directory matching split
    for sdir in search_dirs:
        if not os.path.exists(sdir):
            continue
        for root, dirs, files in os.walk(sdir):
            r_lower = os.path.basename(root).lower()
            if split_lower in r_lower and ("kion" in r_lower or "dataset" in r_lower or r_lower == split_lower):
                # Verify that it contains actual data (metadata.json or batch zips)
                has_data = False
                for check_root, _, check_files in os.walk(root):
                    if "metadata.json" in check_files or any(cf.endswith(".zip") for cf in check_files):
                        has_data = True
                        break
                if has_data:
                    print(f"[+] Found extracted {split_name} dataset directory at: {root}")
                    return root

    # 3. Fallback: Search for any directory where metadata.json exists and split_lower is in path
    for sdir in search_dirs:
        if not os.path.exists(sdir):
            continue
        for root, dirs, files in os.walk(sdir):
            if "metadata.json" in files and split_lower in root.lower():
                # Walk up to find the split-level directory
                cur = root
                split_root = cur
                while cur and cur != sdir and cur != "/":
                    if split_lower in os.path.basename(cur).lower():
                        split_root = cur
                    cur = os.path.dirname(cur)
                print(f"[+] Found {split_name} directory containing metadata at: {split_root}")
                return split_root

    # If still not found, print diagnostic directory tree
    print(f"\n[!] ERROR: Could not find dataset source for split '{split_name}' in search paths.")
    if os.path.exists("/kaggle/input"):
        print("[*] Diagnostic listing of /kaggle/input:")
        for r, d, f in os.walk("/kaggle/input"):
            depth = r.replace("/kaggle/input", "").count(os.sep)
            if depth <= 3:
                print(f"    {'  '*depth}[DIR] {r}")
                sample_files = f[:3]
                if sample_files:
                    print(f"    {'  '*(depth+1)}Files: {', '.join(sample_files)}{'...' if len(f)>3 else ''}")

    raise FileNotFoundError(
        f"Dataset for split '{split_name}' not found under any search path: {search_dirs}. "
        f"Please verify your dataset is added to Kaggle input."
    )


def find_dataset_zip(filename: str) -> str:
    """Backward compatibility wrapper for find_dataset_source."""
    split = "train" if "train" in filename.lower() else "val"
    return find_dataset_source(split)


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

        /path/to/audio.wav|phoneme_sequence|0

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
    train_source: Optional[str] = None,
    val_source: Optional[str] = None,
    wav_dir: Optional[str] = None,
    manifest_dir: Optional[str] = None,
    styletts2_data_dir: Optional[str] = None,
    train_zip: Optional[str] = None,
    val_zip: Optional[str] = None,
):
    base_data = "/kaggle/working/dataset" if os.path.exists("/kaggle") else "/content/dataset"
    if wav_dir is None:
        wav_dir = os.path.join(base_data, "wavs")
    if manifest_dir is None:
        manifest_dir = base_data

    # Support train_zip / val_zip parameter names for backwards compatibility
    train_source = train_source or train_zip
    val_source   = val_source or val_zip

    if train_source is None or not os.path.exists(train_source):
        train_source = find_dataset_source("train")
    if val_source is None or not os.path.exists(val_source):
        val_source = find_dataset_source("val")

    if styletts2_data_dir is None:
        styletts2_data_dir = os.path.join(REPO_ROOT, "StyleTTS2", "Data")

    train_manifest = os.path.join(manifest_dir, "train_manifest.json")
    val_manifest   = os.path.join(manifest_dir, "val_manifest.json")

    # Step 1: Unpack / process dataset → JSON manifests
    train_records = unpack_and_generate_manifest(train_source, wav_dir, train_manifest, split_name="train")
    val_records   = unpack_and_generate_manifest(val_source,   wav_dir, val_manifest,   split_name="val")

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
