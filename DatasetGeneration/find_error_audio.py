#!/usr/bin/env python3
"""
Scan all batch zips inside KionTTS_Dataset.zip,
find audio files with the emotion combination [concerned=0.6][affectionate=0.5],
and copy them (wav + metadata entry) to an output folder: Error_concerned_affectionate/
"""

import zipfile
import io
import json
import os
import sys

MAIN_ZIP = "KionTTS_Dataset.zip"
OUTPUT_DIR = "Error_concerned_affectionate"
WAVS_DIR = os.path.join(OUTPUT_DIR, "wavs")
TARGET_EMOTIONS = {"concerned": 0.6, "affectionate": 0.5}
# Exclude the fix folder from scanning
EXCLUDE_PREFIX = "KionTTS_Dataset/Fix_curious_affectionate/"

os.makedirs(WAVS_DIR, exist_ok=True)

found_entries = []
total_batches = 0
matched_files = 0


def matches_target(entry: dict) -> bool:
    """Return True if this entry has both 'concerned' and 'affectionate' in the text field.
    
    Matches any intensity, e.g. [concerned=0.7, affectionate=0.6]
    """
    text = entry.get("text", "")
    if "concerned" in text and "affectionate" in text:
        return True

    # Also check segments emotions+styles in case text field differs
    has_concerned = False
    has_affectionate = False
    for seg in entry.get("segments", []):
        emotions = seg.get("emotions", {})
        styles = seg.get("styles", {})
        if "concerned" in emotions or "concerned" in styles:
            has_concerned = True
        if "affectionate" in emotions or "affectionate" in styles:
            has_affectionate = True
    return has_concerned and has_affectionate


print(f"Opening {MAIN_ZIP} ...")
with zipfile.ZipFile(MAIN_ZIP, "r") as outer:
    all_inner_zips = [
        name for name in outer.namelist()
        if name.endswith(".zip") and not name.startswith(EXCLUDE_PREFIX)
    ]
    total_batches = len(all_inner_zips)
    print(f"Found {total_batches} batch zips to scan (Fix folder excluded).\n")

    for idx, batch_path in enumerate(all_inner_zips, 1):
        batch_name = os.path.basename(batch_path)
        # Progress every 50 batches
        if idx % 50 == 0 or idx == 1:
            print(f"  [{idx}/{total_batches}] Scanning {batch_path} ...", flush=True)

        try:
            with outer.open(batch_path) as f:
                batch_data = io.BytesIO(f.read())

            with zipfile.ZipFile(batch_data, "r") as inner:
                # Read metadata
                try:
                    with inner.open("metadata.json") as mf:
                        metadata = json.load(mf)
                except KeyError:
                    # No metadata.json in this batch
                    continue

                if not isinstance(metadata, list):
                    metadata = [metadata]

                for entry in metadata:
                    if not matches_target(entry):
                        continue

                    file_id = entry.get("id", "unknown")
                    wav_name = f"{file_id}.wav"
                    wav_inner_path = f"wavs/{wav_name}"

                    # Extract the wav file
                    try:
                        wav_bytes = inner.read(wav_inner_path)
                        out_wav_path = os.path.join(WAVS_DIR, wav_name)
                        with open(out_wav_path, "wb") as wf:
                            wf.write(wav_bytes)
                        matched_files += 1
                        found_entries.append({
                            "batch": batch_name,
                            "batch_path": batch_path,
                            "entry": entry,
                        })
                        print(f"    + Found: {file_id}.wav  in {batch_name}")
                    except KeyError:
                        print(f"    WARNING: WAV not found: {wav_inner_path} (batch: {batch_name})")
                        found_entries.append({
                            "batch": batch_name,
                            "batch_path": batch_path,
                            "entry": entry,
                            "wav_missing": True,
                        })

        except Exception as e:
            print(f"  ERROR reading {batch_path}: {e}", file=sys.stderr)

# Save a summary metadata JSON
summary_path = os.path.join(OUTPUT_DIR, "metadata.json")
clean_entries = [e["entry"] for e in found_entries]
with open(summary_path, "w") as sf:
    json.dump(clean_entries, sf, indent=2)

# Save a detailed report
report_path = os.path.join(OUTPUT_DIR, "report.json")
with open(report_path, "w") as rf:
    json.dump(found_entries, rf, indent=2)

print(f"\n{'='*60}")
print(f"Scan complete!")
print(f"  Batches scanned : {total_batches}")
print(f"  Matching files  : {matched_files}")
print(f"  Output folder   : {OUTPUT_DIR}/")
print(f"  WAVs saved to   : {WAVS_DIR}/")
print(f"  Metadata saved  : {summary_path}")
print(f"  Report saved    : {report_path}")
