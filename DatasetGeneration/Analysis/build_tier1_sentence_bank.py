"""
build_tier1_sentence_bank.py
-----------------------------
Scans the dataset zip and extracts the full metadata for all Tier 1
(auto-flagged) files, producing a structured sentence bank for regeneration.

Output files:
  tier1_sentence_bank.json   — full structured data (id, text, emotions, word_count, flag_reason)
  tier1_sentence_bank.csv    — spreadsheet-friendly version
  tier1_sentence_bank.md     — human-readable Markdown table
"""

import zipfile
import json
import io
import re
import csv
from collections import Counter

# ─── Config ────────────────────────────────────────────────────────────────────
ZIP_PATH        = "KionTTS_Dataset.zip"
TIER1_IDS_FILE  = "tier1_autoflag.txt"
OUT_JSON        = "tier1_sentence_bank.json"
OUT_CSV         = "tier1_sentence_bank.csv"
OUT_MD          = "tier1_sentence_bank.md"

# ─── High-risk combos (same as flag_risky_files.py) ───────────────────────────
TIER1_SINGLE = {
    ("surprised", 0.6), ("surprised", 0.7), ("surprised", 0.8),
    ("playful",   0.6), ("playful",   0.7),
    ("happy",     0.7), ("happy",     0.8),
    ("excited",   0.7), ("excited",   0.8),
    ("authoritative", 0.6), ("authoritative", 0.7),
    ("overjoyed", 0.3), ("overjoyed", 0.4), ("overjoyed", 0.8),
    ("sarcasm",   0.8),
    ("teasing",   0.7),
    ("confused",  0.7),
    ("curious",   0.8),
    ("angry",     0.6),
    ("annoyed",   0.6),
    ("concerned", 0.8),
    ("excited",   0.3),
}
TIER1_MULTI_TRIGGER = {"playful", "surprised", "happy", "excited", "teasing", "overjoyed"}
TIER1_LONG_THRESHOLD = 50
TIER1_LONG_EXPRESSIVE = {"surprised", "playful", "happy", "excited", "authoritative",
                          "teasing", "overjoyed", "sarcasm", "curious"}

# ─── Load Tier 1 IDs ──────────────────────────────────────────────────────────
with open(TIER1_IDS_FILE) as f:
    tier1_ids = set(line.strip() for line in f if line.strip())

print(f"Loaded {len(tier1_ids)} Tier 1 IDs to look up.")

# ─── Parse helper ─────────────────────────────────────────────────────────────
def parse_entry(entry):
    text = entry.get("text", "")
    segments = entry.get("segments", [])

    emotion_tags = re.findall(r'\[(\w+)=([\d.]+)\]', text)
    seg_emotions = {}
    seg_styles   = {}
    for seg in segments:
        seg_emotions.update(seg.get("emotions", {}))
        seg_styles.update(seg.get("styles",   {}))

    all_emotions = {}
    for tag, val in emotion_tags:
        all_emotions[tag] = round(float(val), 2)
    all_emotions.update({k: round(float(v), 2) for k, v in seg_emotions.items()})
    all_emotions.update({k: round(float(v), 2) for k, v in seg_styles.items()})

    clean_text = re.sub(r'\[\w+=[\d.]+\]', '', text).strip()
    word_count = len(clean_text.split())
    char_count = len(clean_text)

    # Determine which rule flagged it
    flag_reason = "unknown"
    for em, val in all_emotions.items():
        if (em, val) in TIER1_SINGLE:
            flag_reason = f"single_high_risk:{em}={val}"
            break

    if flag_reason == "unknown" and len(all_emotions) >= 2:
        triggers = {em for em in all_emotions if em in TIER1_MULTI_TRIGGER}
        if triggers:
            flag_reason = f"multi_with_trigger:{'+'.join(sorted(triggers))}"

    if flag_reason == "unknown" and word_count > TIER1_LONG_THRESHOLD:
        for em, val in all_emotions.items():
            if em in TIER1_LONG_EXPRESSIVE and val >= 0.6:
                flag_reason = f"long_expressive:{em}={val}"
                break

    # Build tag string: "[emotion=intensity]" style
    tag_str = " ".join(f"[{em}={val}]" for em, val in sorted(all_emotions.items()))

    return {
        "id":           entry.get("id"),
        "original_text": text,
        "clean_text":   clean_text,
        "emotion_tags": tag_str,
        "emotions":     all_emotions,
        "word_count":   word_count,
        "char_count":   char_count,
        "num_emotions": len(all_emotions),
        "flag_reason":  flag_reason,
    }

# ─── Scan zip ─────────────────────────────────────────────────────────────────
found = {}
print("Scanning zip...")

with zipfile.ZipFile(ZIP_PATH, "r") as outer:
    batch_zips = sorted(f for f in outer.namelist() if f.endswith(".zip") and "batch_" in f)
    total = len(batch_zips)

    for i, batch_path in enumerate(batch_zips):
        if i % 200 == 0:
            print(f"  [{i}/{total}] Found {len(found)}/{len(tier1_ids)} so far...")
        if len(found) == len(tier1_ids):
            print("  All IDs found — stopping early.")
            break
        try:
            with outer.open(batch_path) as bz:
                bz_data = io.BytesIO(bz.read())
                with zipfile.ZipFile(bz_data, "r") as inner:
                    if "metadata.json" not in inner.namelist():
                        continue
                    meta = json.loads(inner.read("metadata.json"))
                    if not isinstance(meta, list):
                        meta = list(meta.values())
                    for entry in meta:
                        eid = entry.get("id")
                        if eid in tier1_ids and eid not in found:
                            found[eid] = parse_entry(entry)
        except Exception as e:
            print(f"  WARNING: {batch_path}: {e}")

print(f"\nDone. Matched {len(found)}/{len(tier1_ids)} Tier 1 IDs.")
missing = tier1_ids - set(found.keys())
if missing:
    print(f"  Missing (may be in fix batch): {len(missing)} IDs")

records = list(found.values())

# Sort by flag_reason then word_count for easy batching
records.sort(key=lambda r: (r["flag_reason"], -r["word_count"]))

# ─── Write JSON ───────────────────────────────────────────────────────────────
with open(OUT_JSON, "w", encoding="utf-8") as f:
    json.dump(records, f, ensure_ascii=False, indent=2)
print(f"[OK] {OUT_JSON} — {len(records)} entries")

# ─── Write CSV ────────────────────────────────────────────────────────────────
CSV_FIELDS = ["id", "flag_reason", "num_emotions", "word_count", "char_count",
              "emotion_tags", "clean_text", "original_text"]
with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(records)
print(f"[OK] {OUT_CSV} — {len(records)} rows")

# ─── Write Markdown ───────────────────────────────────────────────────────────
flag_groups = {}
for r in records:
    key = r["flag_reason"].split(":")[0]
    flag_groups.setdefault(key, []).append(r)

md_lines = []
A = md_lines.append

A("# Tier 1 Sentence Bank — Regeneration Target List")
A(f"\n**Total entries**: {len(records)}  ")
A(f"**Source**: `tier1_autoflag.txt` → matched from `KionTTS_Dataset.zip`\n")
A("Use this as your regeneration input. For each entry:")
A("- **Short texts** (≤20 words): lower the intensity by 0.1–0.2")
A("- **Medium texts** (21–50 words): lower intensity AND consider splitting at punctuation")
A("- **Long texts** (>50 words): shorten sentence to ≤30 words, then regenerate at original intensity\n")
A("---\n")

for group_key, group_records in sorted(flag_groups.items()):
    A(f"## {group_key.replace('_', ' ').title()} ({len(group_records)} files)\n")

    # Sub-group by specific flag value
    sub_groups = {}
    for r in group_records:
        sub_key = r["flag_reason"]
        sub_groups.setdefault(sub_key, []).append(r)

    for sub_key, sub_records in sorted(sub_groups.items(), key=lambda x: -len(x[1])):
        sub_label = sub_key.split(":", 1)[1] if ":" in sub_key else sub_key
        A(f"### `{sub_label}` — {len(sub_records)} files\n")
        A("| ID | Words | Emotion Tags | Text |")
        A("|----|-------|--------------|------|")
        for r in sub_records:
            # Truncate text for table readability
            txt = r["clean_text"]
            if len(txt) > 120:
                txt = txt[:117] + "..."
            txt = txt.replace("|", "\\|")
            A(f"| `{r['id']}` | {r['word_count']} | `{r['emotion_tags']}` | {txt} |")
        A("")

md_report = "\n".join(md_lines)
with open(OUT_MD, "w", encoding="utf-8") as f:
    f.write(md_report)
print(f"[OK] {OUT_MD} — {len(records)} entries in Markdown")

# ─── Stats summary ────────────────────────────────────────────────────────────
print("\n=== Sentence Bank Stats ===")
print(f"Total: {len(records)}")
print(f"By flag reason:")
for key, grp in sorted(flag_groups.items()):
    print(f"  {key}: {len(grp)}")
import statistics
wcs = [r["word_count"] for r in records]
print(f"Word count: min={min(wcs)}, max={max(wcs)}, mean={statistics.mean(wcs):.1f}, median={statistics.median(wcs)}")
short   = sum(1 for w in wcs if w <= 20)
medium  = sum(1 for w in wcs if 21 <= w <= 50)
long_   = sum(1 for w in wcs if w > 50)
print(f"  ≤20 words (lower intensity only):    {short}")
print(f"  21–50 words (lower intensity + split): {medium}")
print(f"  >50 words (shorten first):            {long_}")
