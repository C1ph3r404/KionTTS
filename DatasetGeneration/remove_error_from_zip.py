#!/usr/bin/env python3
"""
Remove the error [concerned+affectionate] files from their source batch zips
inside KionTTS_Dataset.zip.

Strategy:
  1. Read report.json to build a map: {batch_path -> set(file_ids_to_remove)}
  2. Stream-copy the main zip to a new zip, and for each affected batch:
     - Rebuild it in-memory without the error wavs/metadata entries
  3. Output: KionTTS_Dataset_cleaned.zip
"""

import zipfile
import io
import json
import os
import sys

MAIN_ZIP = "KionTTS_Dataset.zip"
OUTPUT_ZIP = "KionTTS_Dataset_cleaned.zip"
REPORT_JSON = "Error_concerned_affectionate/report.json"

# ── Load report ──────────────────────────────────────────────────────────────
print(f"Loading report from {REPORT_JSON} ...")
with open(REPORT_JSON) as f:
    report = json.load(f)

# Build map: batch_path (as stored in zip) -> set of IDs to remove
removal_map: dict[str, set[str]] = {}
for item in report:
    bp = item["batch_path"]       # e.g. "KionTTS_Dataset/Batch151-200/batch_0159.zip"
    fid = item["entry"]["id"]     # e.g. "f89cf733"
    removal_map.setdefault(bp, set()).add(fid)

affected_batches = len(removal_map)
total_ids = sum(len(v) for v in removal_map.values())
print(f"Affected batches : {affected_batches}")
print(f"Files to remove  : {total_ids}")
print()


def rebuild_batch(batch_bytes: bytes, ids_to_remove: set[str]) -> bytes:
    """Return a new batch zip bytes with the error wav/temp files and metadata entries removed."""
    src = io.BytesIO(batch_bytes)
    dst = io.BytesIO()

    with zipfile.ZipFile(src, "r") as zin, zipfile.ZipFile(dst, "w", compression=zipfile.ZIP_DEFLATED) as zout:
        # Collect all names
        names = zin.namelist()

        # Determine which paths to skip
        skip = set()
        for fid in ids_to_remove:
            skip.add(f"wavs/{fid}.wav")
            skip.add(f"temp/{fid}_seg0.wav")   # temp segment, if present
            # Some batches may have multi-segment temp files
            for name in names:
                if name.startswith(f"temp/{fid}_"):
                    skip.add(name)

        for name in names:
            if name == "metadata.json":
                # Rewrite metadata without removed entries
                raw = zin.read("metadata.json")
                meta = json.loads(raw)
                if not isinstance(meta, list):
                    meta = [meta]
                meta_clean = [e for e in meta if e.get("id") not in ids_to_remove]
                zout.writestr("metadata.json", json.dumps(meta_clean, ensure_ascii=False))
            elif name in skip:
                print(f"      - Removing: {name}", flush=True)
            else:
                # Copy as-is
                zout.writestr(zin.getinfo(name), zin.read(name))

    return dst.getvalue()


# ── Rebuild main zip ─────────────────────────────────────────────────────────
print(f"Rebuilding main zip -> {OUTPUT_ZIP}")
print("(This will take a few minutes due to the size of the archive)\n")

processed = 0
modified = 0

with zipfile.ZipFile(MAIN_ZIP, "r") as zin, \
     zipfile.ZipFile(OUTPUT_ZIP, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as zout:

    all_names = zin.namelist()
    total = len(all_names)

    for i, name in enumerate(all_names, 1):
        if i % 100 == 0 or i == 1:
            print(f"  [{i}/{total}] Processing: {name}", flush=True)

        if name in removal_map:
            # Rebuild this batch zip without the error files
            ids = removal_map[name]
            batch_name = os.path.basename(name)
            print(f"  >> Modifying batch: {batch_name}  (removing {len(ids)} file(s))", flush=True)
            original_bytes = zin.read(name)
            cleaned_bytes = rebuild_batch(original_bytes, ids)
            zout.writestr(zin.getinfo(name), cleaned_bytes)
            modified += 1
        else:
            # Copy everything else unchanged
            zout.writestr(zin.getinfo(name), zin.read(name))

        processed += 1

print(f"\n{'='*60}")
print(f"Done!")
print(f"  Entries processed : {processed}")
print(f"  Batches modified  : {modified}")
print(f"  Output zip        : {OUTPUT_ZIP}")
print(f"\nYou can now replace the original:")
print(f"  mv {OUTPUT_ZIP} {MAIN_ZIP}")
