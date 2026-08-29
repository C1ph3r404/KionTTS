"""
flag_risky_files.py
-------------------
Applies the error analysis findings to the full dataset and produces:
  - tier1_autoflag.txt   : High-confidence errors — regenerate without review
  - tier2_spotcheck.txt  : Medium-risk — spot-check ~10%
  - tier3_clean.txt      : Considered clean — safe to keep
  - summary.txt          : Quick stats

Usage:
    python3 flag_risky_files.py

Thresholds derived from analyze_errors.py output on 401 confirmed errors.
"""

import zipfile
import json
import io
import re
from collections import Counter

# ─── Config ────────────────────────────────────────────────────────────────────
ZIP_PATH       = "KionTTS_Dataset.zip"
KNOWN_BAD_FILE = "fixes_id.txt"   # already-confirmed errors to skip

TIER1_OUT = "tier1_autoflag.txt"
TIER2_OUT = "tier2_spotcheck.txt"
TIER3_OUT = "tier3_clean.txt"
SUMMARY   = "flagging_summary.txt"

# ─── Tier definitions (derived from error analysis) ───────────────────────────

# Tier 1A: single emotions that are highly expressive at high intensity
# Observed error rate ≥5% at these specific levels
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

# Tier 1B: multi-emotion combos — any file with 2+ emotions AND
# the combo contains any of these expressive emotions
TIER1_MULTI_TRIGGER_EMOTIONS = {
    "playful", "surprised", "happy", "excited", "teasing", "overjoyed"
}
TIER1_MULTI_MIN_COUNT = 2   # must have ≥2 emotions

# Tier 1C: any file with word count > 50 AND any expressive emotion at ≥0.6
TIER1_LONG_THRESHOLD = 50
TIER1_LONG_EXPRESSIVE = {
    "surprised", "playful", "happy", "excited", "authoritative",
    "teasing", "overjoyed", "sarcasm", "curious"
}

# Tier 2: medium-risk emotions at moderate-to-high intensities (2.5–5% range)
TIER2_COMBOS = {
    ("sarcasm",      0.6), ("sarcasm",      0.7),
    ("teasing",      0.6), ("teasing",      0.8),
    ("frustrated",   0.5), ("frustrated",   0.6),
    ("curious",      0.7),
    ("disappointed", 0.7),
    ("calm",         0.8),
    ("dramatic",     0.6), ("dramatic",     0.7),
    ("annoyed",      0.7),
    ("authoritative", 0.5),
    ("excited",      0.4), ("excited",      0.5),
    ("surprised",    0.4), ("surprised",    0.5),
    ("playful",      0.5),
}
TIER2_WORD_MIN = 20  # only flag these if sentence is long enough to matter

# Tier 3: emotions with <1% observed error rate — assumed safe
TIER3_SAFE_EMOTIONS = {
    "calm", "serious", "affectionate", "sad", "deadpan",
    "heartbroken", "bored", "concerned"
}
TIER3_WORD_MAX = 30  # extra confidence cutoff

# ─── Load known-bad IDs to skip ───────────────────────────────────────────────
with open(KNOWN_BAD_FILE) as f:
    known_bad = set(line.strip() for line in f if line.strip())

print(f"Loaded {len(known_bad)} known-bad IDs to skip.")

# ─── Parse a metadata entry ────────────────────────────────────────────────────
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

    return {
        "id":          entry.get("id"),
        "word_count":  word_count,
        "emotions":    all_emotions,
        "num_emotions": len(all_emotions),
    }

# ─── Tier classification ───────────────────────────────────────────────────────
def classify(p):
    ems = p["emotions"]
    wc  = p["word_count"]
    n   = p["num_emotions"]

    # Tier 1A — single high-risk (emotion, intensity) pair
    for em, val in ems.items():
        if (em, val) in TIER1_SINGLE:
            return "tier1", f"single_high_risk:{em}={val}"

    # Tier 1B — multi-emotion combo containing expressive trigger emotions
    if n >= TIER1_MULTI_MIN_COUNT:
        triggers = {em for em in ems if em in TIER1_MULTI_TRIGGER_EMOTIONS}
        if triggers:
            return "tier1", f"multi_with_trigger:{'+'.join(sorted(triggers))}"

    # Tier 1C — long sentence + expressive emotion at ≥0.6
    if wc > TIER1_LONG_THRESHOLD:
        for em, val in ems.items():
            if em in TIER1_LONG_EXPRESSIVE and val >= 0.6:
                return "tier1", f"long_expressive:{em}={val},words={wc}"

    # Tier 2 — medium-risk pair AND long-enough sentence
    for em, val in ems.items():
        if (em, val) in TIER2_COMBOS and wc >= TIER2_WORD_MIN:
            return "tier2", f"medium_risk:{em}={val},words={wc}"

    # Tier 3 — all emotions are safe-category AND sentence is short
    all_safe = all(em in TIER3_SAFE_EMOTIONS for em in ems)
    if all_safe and wc <= TIER3_WORD_MAX:
        return "tier3", "safe_emotion_short"

    # Default: tier3 (no flags triggered)
    return "tier3", "no_flags"

# ─── Scan and classify ─────────────────────────────────────────────────────────
tier1_ids   = []
tier2_ids   = []
tier3_ids   = []
tier1_reasons = Counter()
tier2_reasons = Counter()
total_scanned = 0

print("Scanning zip...")

with zipfile.ZipFile(ZIP_PATH, "r") as outer:
    batch_zips = sorted(f for f in outer.namelist() if f.endswith(".zip") and "batch_" in f)
    total = len(batch_zips)

    for i, batch_path in enumerate(batch_zips):
        if i % 100 == 0:
            print(f"  [{i}/{total}] T1={len(tier1_ids)} T2={len(tier2_ids)} T3={len(tier3_ids)}")
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
                        if eid in known_bad:
                            continue  # skip already-confirmed errors
                        total_scanned += 1
                        p = parse_entry(entry)
                        tier, reason = classify(p)
                        if tier == "tier1":
                            tier1_ids.append(eid)
                            tier1_reasons[reason.split(":")[0]] += 1
                        elif tier == "tier2":
                            tier2_ids.append(eid)
                            tier2_reasons[reason.split(":")[0]] += 1
                        else:
                            tier3_ids.append(eid)

        except Exception as e:
            print(f"  WARNING: {batch_path}: {e}")

print(f"\nDone. Scanned {total_scanned} entries (excluding {len(known_bad)} known-bad).")

# ─── Write output files ────────────────────────────────────────────────────────
with open(TIER1_OUT, "w") as f:
    f.write("\n".join(tier1_ids))
print(f"[OK] {TIER1_OUT}: {len(tier1_ids)} IDs")

with open(TIER2_OUT, "w") as f:
    f.write("\n".join(tier2_ids))
print(f"[OK] {TIER2_OUT}: {len(tier2_ids)} IDs")

with open(TIER3_OUT, "w") as f:
    f.write("\n".join(tier3_ids))
print(f"[OK] {TIER3_OUT}: {len(tier3_ids)} IDs")

summary_lines = [
    "=== KionTTS Flag Summary ===\n",
    f"Total scanned (excl. known-bad): {total_scanned}",
    f"",
    f"Tier 1 — Auto-regenerate (high-risk): {len(tier1_ids)} ({len(tier1_ids)/total_scanned*100:.1f}%)",
    f"  Breakdown:",
]
for reason, cnt in tier1_reasons.most_common():
    summary_lines.append(f"    {reason}: {cnt}")

summary_lines += [
    f"",
    f"Tier 2 — Spot-check (medium-risk):    {len(tier2_ids)} ({len(tier2_ids)/total_scanned*100:.1f}%)",
    f"  Breakdown:",
]
for reason, cnt in tier2_reasons.most_common():
    summary_lines.append(f"    {reason}: {cnt}")

summary_lines += [
    f"",
    f"Tier 3 — Considered clean:            {len(tier3_ids)} ({len(tier3_ids)/total_scanned*100:.1f}%)",
]

summary = "\n".join(summary_lines)
with open(SUMMARY, "w") as f:
    f.write(summary)

print(f"\n[OK] {SUMMARY} written.\n")
print(summary)
