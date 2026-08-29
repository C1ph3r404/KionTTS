"""
Error Analysis Script for KionTTS Dataset
Analyzes the 414 manually identified error files to find patterns
in emotions, intensities, and text lengths that cause issues.
"""

import zipfile
import json
import io
import re
from collections import defaultdict, Counter
import statistics

# ─── Config ────────────────────────────────────────────────────────────────────
ZIP_PATH = "KionTTS_Dataset.zip"
ERROR_IDS_FILE = "fixes_id.txt"
OUTPUT_REPORT = "error_analysis_report.md"

# ─── Load error IDs ────────────────────────────────────────────────────────────
with open(ERROR_IDS_FILE, "r") as f:
    error_ids = set(line.strip() for line in f if line.strip())

print(f"Loaded {len(error_ids)} unique error IDs (raw: {sum(1 for _ in open(ERROR_IDS_FILE) if _.strip())} lines)")

# ─── Collect metadata for error files ─────────────────────────────────────────
error_entries = []          # list of metadata dicts for error files
all_entries = []            # list of metadata dicts for ALL files (for baseline)
found_ids = set()

print("Scanning zip file...")

with zipfile.ZipFile(ZIP_PATH, "r") as outer:
    batch_zips = sorted(f for f in outer.namelist() if f.endswith(".zip") and "batch_" in f)
    total_batches = len(batch_zips)
    
    for i, batch_path in enumerate(batch_zips):
        if i % 50 == 0:
            print(f"  [{i}/{total_batches}] Processed {len(all_entries)} entries so far, found {len(found_ids)} errors...")
        
        try:
            with outer.open(batch_path) as bz:
                bz_data = io.BytesIO(bz.read())
                with zipfile.ZipFile(bz_data, "r") as inner:
                    meta_files = [f for f in inner.namelist() if f == "metadata.json"]
                    if not meta_files:
                        continue
                    metadata = json.loads(inner.read("metadata.json"))
                    if not isinstance(metadata, list):
                        metadata = list(metadata.values())
                    
                    for entry in metadata:
                        all_entries.append(entry)
                        if entry.get("id") in error_ids:
                            error_entries.append(entry)
                            found_ids.add(entry["id"])
        except Exception as e:
            print(f"  WARNING: failed to read {batch_path}: {e}")

print(f"\nDone. Total entries scanned: {len(all_entries)}")
print(f"Error entries matched: {len(error_entries)} / {len(error_ids)} IDs")
missing = error_ids - found_ids
if missing:
    print(f"  IDs not found in current zip (may be in fix batch): {missing}")

# ─── Helper: parse a metadata entry ───────────────────────────────────────────
def parse_entry(entry):
    text = entry.get("text", "")
    segments = entry.get("segments", [])
    
    # Collect all emotion tags from the text field  [emotion=intensity]
    emotion_tags = re.findall(r'\[(\w+)=([\d.]+)\]', text)
    
    # Collect from segments too
    seg_emotions = {}
    seg_styles = {}
    for seg in segments:
        seg_emotions.update(seg.get("emotions", {}))
        seg_styles.update(seg.get("styles", {}))
    
    all_emotions = {}  # combined emotions + styles
    for tag, val in emotion_tags:
        all_emotions[tag] = float(val)
    all_emotions.update(seg_emotions)
    all_emotions.update(seg_styles)
    
    # Strip emotion tags from text to get clean text length
    clean_text = re.sub(r'\[\w+=[\d.]+\]', '', text).strip()
    word_count = len(clean_text.split())
    char_count = len(clean_text)
    
    num_emotions = len(all_emotions)
    is_multi_emotion = num_emotions > 1
    
    return {
        "id": entry.get("id"),
        "text": text,
        "clean_text": clean_text,
        "word_count": word_count,
        "char_count": char_count,
        "emotions": all_emotions,
        "num_emotions": num_emotions,
        "is_multi_emotion": is_multi_emotion,
        "emotion_tuple": tuple(sorted(all_emotions.items())),  # hashable key
        "emotion_names": frozenset(all_emotions.keys()),
        "speaker_reference": entry.get("speaker_reference", "unknown"),
    }

error_parsed = [parse_entry(e) for e in error_entries]
all_parsed   = [parse_entry(e) for e in all_entries]

# baseline counts for comparison
all_emotion_combo_counts = Counter(p["emotion_tuple"] for p in all_parsed)
all_single_emotion_counts = Counter()
for p in all_parsed:
    for em, val in p["emotions"].items():
        all_single_emotion_counts[(em, val)] += 1

# ─── Analysis 1: Emotion combination frequency in errors ──────────────────────
error_combo_counts = Counter(p["emotion_tuple"] for p in error_parsed)
error_single_emotion_counts = Counter()
for p in error_parsed:
    for em, val in p["emotions"].items():
        error_single_emotion_counts[(em, val)] += 1

# ─── Analysis 2: Individual emotion error rates ────────────────────────────────
# For each (emotion, intensity) pair -> how many errors / how many total
emotion_error_rate = {}
for key, err_cnt in error_single_emotion_counts.items():
    total_cnt = all_single_emotion_counts.get(key, 0)
    if total_cnt > 0:
        emotion_error_rate[key] = {"errors": err_cnt, "total": total_cnt, "rate": err_cnt / total_cnt}

# ─── Analysis 3: Multi-emotion vs single-emotion breakdown ────────────────────
multi_error = [p for p in error_parsed if p["is_multi_emotion"]]
single_error = [p for p in error_parsed if not p["is_multi_emotion"]]

multi_all = [p for p in all_parsed if p["is_multi_emotion"]]
single_all = [p for p in all_parsed if not p["is_multi_emotion"]]

multi_error_rate = len(multi_error) / len(multi_all) if multi_all else 0
single_error_rate = len(single_error) / len(single_all) if single_all else 0

# ─── Analysis 4: Text length distribution ─────────────────────────────────────
error_word_counts  = [p["word_count"] for p in error_parsed]
error_char_counts  = [p["char_count"] for p in error_parsed]
all_word_counts    = [p["word_count"] for p in all_parsed]
all_char_counts    = [p["char_count"] for p in all_parsed]

def stats(data):
    if not data:
        return {}
    return {
        "min": min(data),
        "max": max(data),
        "mean": round(statistics.mean(data), 1),
        "median": round(statistics.median(data), 1),
        "stdev": round(statistics.stdev(data), 1) if len(data) > 1 else 0,
    }

# ─── Analysis 5: Bucket text by length and see error rate per bucket ───────────
def bucket_by_words(parsed_list, buckets=[(0,10),(10,20),(20,30),(30,50),(50,100),(100,999)]):
    result = {}
    for lo, hi in buckets:
        label = f"{lo}-{hi} words"
        result[label] = [p for p in parsed_list if lo <= p["word_count"] < hi]
    return result

error_by_len = bucket_by_words(error_parsed)
all_by_len   = bucket_by_words(all_parsed)

length_bucket_stats = {}
for label in error_by_len:
    err_cnt = len(error_by_len[label])
    tot_cnt = len(all_by_len[label])
    rate = err_cnt / tot_cnt if tot_cnt else 0
    length_bucket_stats[label] = {"errors": err_cnt, "total": tot_cnt, "rate": rate}

# ─── Analysis 6: emotion names (ignoring intensity) ───────────────────────────
error_emotion_name_counts = Counter()
for p in error_parsed:
    for em in p["emotions"]:
        error_emotion_name_counts[em] += 1

all_emotion_name_counts = Counter()
for p in all_parsed:
    for em in p["emotions"]:
        all_emotion_name_counts[em] += 1

emotion_name_error_rate = {}
for em, err_cnt in error_emotion_name_counts.items():
    tot = all_emotion_name_counts.get(em, 0)
    emotion_name_error_rate[em] = {"errors": err_cnt, "total": tot, "rate": err_cnt/tot if tot else 0}

# ─── Compute predicted bad IDs in remaining 17k ───────────────────────────────
# Find high-risk combos (>=30% error rate)
HIGH_RATE_THRESHOLD = 0.30
high_risk_combos = {k: v for k, v in emotion_error_rate.items() if v["rate"] >= HIGH_RATE_THRESHOLD and v["total"] >= 5}
high_risk_emotion_names = {em for em, v in emotion_name_error_rate.items() if v["rate"] >= HIGH_RATE_THRESHOLD and v["total"] >= 5}

remaining_parsed = [p for p in all_parsed if p["id"] not in found_ids and p["id"] not in error_ids]
predicted_errors_by_combo = []
for p in remaining_parsed:
    for key in p["emotions"].items():
        if key in high_risk_combos:
            predicted_errors_by_combo.append(p["id"])
            break

# ─── Write report ─────────────────────────────────────────────────────────────
lines = []
A = lines.append

A("# KionTTS Dataset Error Analysis Report")
A(f"\n> **Scope**: {len(error_entries)} error entries analysed out of 2,200 manually reviewed  ")
A(f"> **Total dataset entries scanned**: {len(all_entries):,}  ")
A(f"> **Overall error rate in reviewed set**: {len(error_entries)/2200*100:.1f}%\n")

# -- Summary box
A("## Quick Summary\n")
A(f"| Metric | Value |")
A(f"|--------|-------|")
A(f"| Error files identified | {len(error_entries)} |")
A(f"| Unique emotion combos in errors | {len(error_combo_counts)} |")
A(f"| Multi-emotion error rate | {multi_error_rate*100:.1f}% ({len(multi_error)}/{len(multi_all)}) |")
A(f"| Single-emotion error rate | {single_error_rate*100:.1f}% ({len(single_error)}/{len(single_all)}) |")
A(f"| Avg word count (errors) | {stats(error_word_counts).get('mean','N/A')} |")
A(f"| Avg word count (all) | {stats(all_word_counts).get('mean','N/A')} |")
A(f"| Predicted risky files in remaining ~17k | {len(predicted_errors_by_combo)} |")
A("")

# -- Error rate by emotion name
A("---\n## 1. Error Rate by Emotion (ignoring intensity)\n")
A("Sorted by error rate (min 5 occurrences in dataset):\n")
A("| Emotion | Errors | Total in Dataset | Error Rate |")
A("|---------|--------|-----------------|------------|")
sorted_names = sorted(
    [(em, v) for em, v in emotion_name_error_rate.items() if v["total"] >= 5],
    key=lambda x: -x[1]["rate"]
)
for em, v in sorted_names:
    flag = " RED" if v["rate"] >= 0.30 else (" YELLOW" if v["rate"] >= 0.10 else "")
    A(f"| `{em}` | {v['errors']} | {v['total']} | {v['rate']*100:.1f}%{flag} |")
A("")
A("> RED = High risk (>=30%) | YELLOW = Medium risk (>=10%)\n")

# -- Error rate by emotion+intensity
A("---\n## 2. Error Rate by Emotion + Intensity\n")
A("Sorted by error rate (min 5 occurrences in dataset):\n")
A("| Emotion | Intensity | Errors | Total | Error Rate |")
A("|---------|-----------|--------|-------|------------|")
sorted_pairs = sorted(
    [(k, v) for k, v in emotion_error_rate.items() if v["total"] >= 5],
    key=lambda x: -x[1]["rate"]
)
for (em, intensity), v in sorted_pairs[:40]:  # top 40
    flag = " RED" if v["rate"] >= 0.30 else (" YELLOW" if v["rate"] >= 0.10 else "")
    A(f"| `{em}` | `{intensity}` | {v['errors']} | {v['total']} | {v['rate']*100:.1f}%{flag} |")
A("")

# -- Top error-prone emotion combos
A("---\n## 3. Top Error-Prone Emotion Combinations\n")
A("Most frequent emotion tag combos found in the 414 error files:\n")
A("| Emotion Combination | Error Count | Total in Dataset | Error Rate |")
A("|---------------------|-------------|-----------------|------------|")
sorted_combos = sorted(error_combo_counts.items(), key=lambda x: -x[1])
for combo, err_cnt in sorted_combos[:25]:
    combo_str = ", ".join(f"`{em}`={val}" for em, val in combo) if combo else "`(none/style-only)`"
    tot = all_emotion_combo_counts.get(combo, 0)
    rate = err_cnt / tot if tot else 0
    flag = " RED" if rate >= 0.30 else (" YELLOW" if rate >= 0.10 else "")
    A(f"| {combo_str} | {err_cnt} | {tot} | {rate*100:.1f}%{flag} |")
A("")

# -- Text length analysis
A("---\n## 4. Text Length Analysis\n")
A("### Word count statistics\n")
A("| Metric | Error files | All files |")
A("|--------|-------------|-----------|")
es = stats(error_word_counts)
as_ = stats(all_word_counts)
for k in ["min", "max", "mean", "median", "stdev"]:
    A(f"| {k.capitalize()} | {es.get(k)} | {as_.get(k)} |")
A("")

A("### Error rate by word count bucket\n")
A("| Word count range | Errors | Total | Error Rate |")
A("|-----------------|--------|-------|------------|")
for label, v in length_bucket_stats.items():
    flag = " RED" if v["rate"] >= 0.30 else (" YELLOW" if v["rate"] >= 0.10 else "")
    A(f"| {label} | {v['errors']} | {v['total']} | {v['rate']*100:.1f}%{flag} |")
A("")

# -- Multi vs single emotion
A("---\n## 5. Multi-Emotion vs Single-Emotion\n")
A("| Type | Error Count | Total | Error Rate |")
A("|------|-------------|-------|------------|")
A(f"| Single emotion | {len(single_error)} | {len(single_all)} | {single_error_rate*100:.1f}% |")
A(f"| Multi-emotion (2+) | {len(multi_error)} | {len(multi_all)} | {multi_error_rate*100:.1f}% |")
A("")

if multi_error:
    A("### Multi-emotion combos in errors:\n")
    multi_combos = Counter(p["emotion_tuple"] for p in multi_error)
    A("| Combination | Count |")
    A("|-------------|-------|")
    for combo, cnt in multi_combos.most_common(20):
        combo_str = " + ".join(f"`{em}`={val}" for em, val in combo)
        A(f"| {combo_str} | {cnt} |")
    A("")

# -- Predicted risky IDs
A("---\n## 6. Risk Prediction for Remaining ~17k Files\n")
A(f"Using high-risk emotion+intensity combinations (>=30% error rate, >=5 samples),  ")
A(f"**{len(predicted_errors_by_combo)}** files in the remaining dataset are flagged as likely errors.\n")
A(f"\n**High-risk (emotion, intensity) pairs identified:**\n")
for (em, intensity), v in sorted(high_risk_combos.items(), key=lambda x: -x[1]["rate"])[:20]:
    A(f"- `{em}={intensity}`: {v['rate']*100:.1f}% error rate ({v['errors']}/{v['total']})")
A("")

A("---\n## 7. Recommended Action Plan\n")
A("Based on the analysis, here is a prioritized filtering strategy:\n")
A("1. **Auto-reject high-risk combos**: Files with any of the RED emotion+intensity combinations should be regenerated without manual review.")
A("2. **Flag long files**: Files with >50 words where ANY emotion tag appears are more likely to have speaker leakage -- batch those for quick spot-checks.")
A("3. **Focus manual review on YELLOW medium-risk**: Files with medium-risk emotions but not caught by rule 1 should be spot-checked (~10% sampling).")
A("4. **Keep low-risk files as-is**: Files with emotions not appearing in this table can be assumed clean.\n")

report = "\n".join(lines)
with open(OUTPUT_REPORT, "w") as f:
    f.write(report)

print(f"\n[OK] Report written to {OUTPUT_REPORT}")
print(f"   Total error entries analysed: {len(error_entries)}")
print(f"   High-risk combos found: {len(high_risk_combos)}")
print(f"   Predicted risky files in remaining dataset: {len(predicted_errors_by_combo)}")
