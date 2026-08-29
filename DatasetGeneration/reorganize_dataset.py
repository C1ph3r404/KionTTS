"""
reorganize_dataset.py
---------------------
Rebuilds KionTTS_Dataset.zip into a new zip with three top-level folders:

  KionTTS_Dataset_v2/
    Tier1/  <- all batch zips from Tier1_fix (regenerated files)
    Tier2/  <- filtered batches from original zip (tier2_spotcheck IDs only)
    Tier3/  <- filtered batches from original zip (tier3_clean IDs only)

Each batch zip inside Tier2/Tier3 contains only the wavs and the filtered
metadata.json for that tier — temp/ files are dropped to save space.

Run from: ~/Projects/AI/Kiontts/DatasetGeneration/
"""

import zipfile
import json
import io
import os

# ── Config ────────────────────────────────────────────────────────────────────
ORIGINAL_ZIP  = "KionTTS_Dataset.zip"
TIER1_FIX_ZIP = "Tier1_fix-20260827T125037Z-1-001.zip"
OUTPUT_ZIP    = "KionTTS_Dataset_v2.zip"

TIER2_IDS_FILE = "Analysis/tier2_spotcheck.txt"
TIER3_IDS_FILE = "Analysis/tier3_clean.txt"

# ── Load tier ID sets ─────────────────────────────────────────────────────────
with open(TIER2_IDS_FILE) as f:
    tier2_ids = set(l.strip() for l in f if l.strip())
with open(TIER3_IDS_FILE) as f:
    tier3_ids = set(l.strip() for l in f if l.strip())

print(f"Tier2 IDs: {len(tier2_ids)}")
print(f"Tier3 IDs: {len(tier3_ids)}")

# ── Helper: rebuild a batch zip keeping only the given ID set ─────────────────
def filter_batch_zip(inner_zip: zipfile.ZipFile, keep_ids: set) -> bytes | None:
    """
    Returns a new in-memory zip bytes containing only the wavs and metadata
    entries for IDs in keep_ids.  Returns None if no matching entries found.
    Drops temp/ files entirely.
    """
    if "metadata.json" not in inner_zip.namelist():
        return None

    meta = json.loads(inner_zip.read("metadata.json"))
    if not isinstance(meta, list):
        meta = list(meta.values())

    kept = [e for e in meta if e.get("id") in keep_ids]
    if not kept:
        return None

    kept_ids_set = {e["id"] for e in kept}

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as zout:
        zout.writestr("metadata.json", json.dumps(kept, ensure_ascii=False))
        zout.mkdir("wavs/")
        for name in inner_zip.namelist():
            if name.startswith("wavs/") and name != "wavs/":
                # e.g. wavs/abc12345.wav
                wav_id = os.path.splitext(os.path.basename(name))[0]
                if wav_id in kept_ids_set:
                    zout.writestr(inner_zip.getinfo(name), inner_zip.read(name))
            # skip temp/
    return buf.getvalue()


# ── Build output zip ──────────────────────────────────────────────────────────
print(f"\nBuilding {OUTPUT_ZIP}...")

tier2_written = 0
tier3_written = 0
tier1_batches = 0

with zipfile.ZipFile(OUTPUT_ZIP, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as zout:

    # ── Tier1: copy Tier1_fix batch zips verbatim ─────────────────────────────
    print("\n[Tier1] Copying Tier1_fix batch zips...")
    with zipfile.ZipFile(TIER1_FIX_ZIP, "r") as t1:
        batch_zips = sorted(f for f in t1.namelist() if f.endswith(".zip") and "batch_" in f)
        total = len(batch_zips)
        for i, src_path in enumerate(batch_zips):
            if i % 50 == 0:
                print(f"  [{i}/{total}] Tier1 batches copied so far: {i}")
            # Remap path: Tier1_fix/BatchXXX/batch_YYYY.zip -> Tier1/BatchXXX/batch_YYYY.zip
            rel = src_path.replace("Tier1_fix/", "", 1)
            dest_path = f"KionTTS_Dataset_v2/Tier1/{rel}"
            data = t1.read(src_path)
            zout.writestr(dest_path, data)
            tier1_batches += 1
    print(f"  Done. {tier1_batches} Tier1 batch zips copied.")

    # ── Tier2 & Tier3: filter original zip batches ────────────────────────────
    print("\n[Tier2/Tier3] Filtering original zip batches...")
    with zipfile.ZipFile(ORIGINAL_ZIP, "r") as orig:
        batch_zips = sorted(f for f in orig.namelist() if f.endswith(".zip") and "batch_" in f)
        total = len(batch_zips)

        for i, src_path in enumerate(batch_zips):
            if i % 100 == 0:
                print(f"  [{i}/{total}] T2={tier2_written} batches, T3={tier3_written} batches")

            try:
                with orig.open(src_path) as bz:
                    bz_data = io.BytesIO(bz.read())
                    with zipfile.ZipFile(bz_data, "r") as inner:
                        # Strip outer folder prefix e.g. KionTTS_Dataset/Batch151-200/batch_0151.zip
                        parts = src_path.split("/")
                        # parts = ['KionTTS_Dataset', 'BatchXXX-YYY', 'batch_NNNN.zip']
                        batch_group = parts[-2]  # e.g. Batch151-200
                        batch_file  = parts[-1]  # e.g. batch_0151.zip

                        # Tier2
                        t2_bytes = filter_batch_zip(inner, tier2_ids)
                        if t2_bytes:
                            dest = f"KionTTS_Dataset_v2/Tier2/{batch_group}/{batch_file}"
                            zout.writestr(dest, t2_bytes)
                            tier2_written += 1

                        # Tier3 — re-open inner since we already consumed it above
                        bz_data.seek(0)
                        with zipfile.ZipFile(bz_data, "r") as inner3:
                            t3_bytes = filter_batch_zip(inner3, tier3_ids)
                            if t3_bytes:
                                dest = f"KionTTS_Dataset_v2/Tier3/{batch_group}/{batch_file}"
                                zout.writestr(dest, t3_bytes)
                                tier3_written += 1

            except Exception as e:
                print(f"  WARNING: {src_path}: {e}")

print(f"\n{'='*60}")
print(f"Done!")
print(f"  Tier1 batches  : {tier1_batches}")
print(f"  Tier2 batches  : {tier2_written}")
print(f"  Tier3 batches  : {tier3_written}")
print(f"  Output         : {OUTPUT_ZIP}")
import os
size_gb = os.path.getsize(OUTPUT_ZIP) / 1e9
print(f"  Output size    : {size_gb:.2f} GB")
