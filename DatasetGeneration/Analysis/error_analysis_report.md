# KionTTS Dataset Error Analysis Report

> **Scope**: 401 error entries analysed out of 2,200 manually reviewed  
> **Total dataset entries scanned**: 19,174  
> **Overall error rate in reviewed set**: 18.2%

## Quick Summary

| Metric | Value |
|--------|-------|
| Error files identified | 401 |
| Unique emotion combos in errors | 115 |
| Multi-emotion error rate | 3.7% (117/3132) |
| Single-emotion error rate | 1.8% (284/16042) |
| Avg word count (errors) | 24.4 |
| Avg word count (all) | 18.1 |
| Predicted risky files in remaining ~17k | 0 |

---
## 1. Error Rate by Emotion (ignoring intensity)

Sorted by error rate (min 5 occurrences in dataset):

| Emotion | Errors | Total in Dataset | Error Rate |
|---------|--------|-----------------|------------|
| `surprised` | 59 | 776 | 7.6% |
| `playful` | 79 | 1406 | 5.6% |
| `excited` | 35 | 851 | 4.1% |
| `happy` | 43 | 1066 | 4.0% |
| `authoritative` | 32 | 810 | 4.0% |
| `overjoyed` | 16 | 418 | 3.8% |
| `teasing` | 36 | 959 | 3.8% |
| `sarcasm` | 28 | 876 | 3.2% |
| `curious` | 35 | 1124 | 3.1% |
| `frustrated` | 26 | 929 | 2.8% |
| `disappointed` | 25 | 960 | 2.6% |
| `annoyed` | 15 | 610 | 2.5% |
| `angry` | 15 | 686 | 2.2% |
| `dramatic` | 20 | 1021 | 2.0% |
| `confused` | 12 | 682 | 1.8% |
| `calm` | 16 | 1191 | 1.3% |
| `concerned` | 7 | 889 | 0.8% |
| `bored` | 4 | 512 | 0.8% |
| `serious` | 9 | 1178 | 0.8% |
| `deadpan` | 3 | 455 | 0.7% |
| `affectionate` | 5 | 1211 | 0.4% |
| `heartbroken` | 2 | 497 | 0.4% |
| `sad` | 1 | 902 | 0.1% |

> RED = High risk (>=30%) | YELLOW = Medium risk (>=10%)

---
## 2. Error Rate by Emotion + Intensity

Sorted by error rate (min 5 occurrences in dataset):

| Emotion | Intensity | Errors | Total | Error Rate |
|---------|-----------|--------|-------|------------|
| `surprised` | `0.7` | 14 | 109 | 12.8% YELLOW |
| `surprised` | `0.8` | 19 | 159 | 11.9% YELLOW |
| `playful` | `0.7` | 31 | 327 | 9.5% |
| `authoritative` | `0.7` | 17 | 214 | 7.9% |
| `surprised` | `0.6` | 17 | 219 | 7.8% |
| `happy` | `0.8` | 7 | 98 | 7.1% |
| `concerned` | `0.8` | 4 | 57 | 7.0% |
| `excited` | `0.3` | 5 | 72 | 6.9% |
| `sarcasm` | `0.8` | 3 | 45 | 6.7% |
| `teasing` | `0.7` | 13 | 199 | 6.5% |
| `overjoyed` | `0.3` | 3 | 46 | 6.5% |
| `playful` | `0.6` | 25 | 395 | 6.3% |
| `authoritative` | `0.6` | 4 | 66 | 6.1% |
| `happy` | `0.7` | 25 | 413 | 6.1% |
| `excited` | `0.8` | 18 | 300 | 6.0% |
| `overjoyed` | `0.8` | 3 | 52 | 5.8% |
| `overjoyed` | `0.4` | 5 | 87 | 5.7% |
| `confused` | `0.7` | 10 | 181 | 5.5% |
| `sarcasm` | `0.6` | 9 | 167 | 5.4% |
| `curious` | `0.8` | 6 | 115 | 5.2% |
| `angry` | `0.6` | 6 | 118 | 5.1% |
| `annoyed` | `0.6` | 6 | 120 | 5.0% |
| `frustrated` | `0.6` | 10 | 206 | 4.9% |
| `frustrated` | `0.5` | 6 | 130 | 4.6% |
| `playful` | `0.5` | 16 | 350 | 4.6% |
| `teasing` | `0.8` | 2 | 44 | 4.5% |
| `calm` | `0.8` | 15 | 335 | 4.5% |
| `curious` | `0.7` | 20 | 449 | 4.5% |
| `teasing` | `0.6` | 14 | 337 | 4.2% |
| `dramatic` | `0.6` | 4 | 100 | 4.0% |
| `disappointed` | `0.7` | 14 | 353 | 4.0% |
| `overjoyed` | `0.5` | 3 | 76 | 3.9% |
| `dramatic` | `0.7` | 11 | 281 | 3.9% |
| `surprised` | `0.4` | 4 | 112 | 3.6% |
| `excited` | `0.4` | 5 | 141 | 3.5% |
| `surprised` | `0.5` | 5 | 148 | 3.4% |
| `annoyed` | `0.7` | 4 | 122 | 3.3% |
| `sarcasm` | `0.7` | 14 | 435 | 3.2% |
| `authoritative` | `0.5` | 5 | 163 | 3.1% |
| `excited` | `0.5` | 3 | 100 | 3.0% |

---
## 3. Top Error-Prone Emotion Combinations

Most frequent emotion tag combos found in the 414 error files:

| Emotion Combination | Error Count | Total in Dataset | Error Rate |
|---------------------|-------------|-----------------|------------|
| `playful`=0.7 | 15 | 166 | 9.0% |
| `happy`=0.7, `playful`=0.6 | 14 | 84 | 16.7% YELLOW |
| `surprised`=0.7 | 14 | 109 | 12.8% YELLOW |
| `authoritative`=0.7 | 11 | 138 | 8.0% |
| `surprised`=0.6 | 10 | 137 | 7.3% |
| `surprised`=0.8 | 10 | 73 | 13.7% YELLOW |
| `curious`=0.7 | 10 | 199 | 5.0% |
| `excited`=0.8, `playful`=0.7 | 9 | 72 | 12.5% YELLOW |
| `dramatic`=0.7, `surprised`=0.8 | 9 | 86 | 10.5% YELLOW |
| `frustrated`=0.6, `playful`=0.5 | 8 | 86 | 9.3% |
| `disappointed`=0.7, `sarcasm`=0.6 | 8 | 83 | 9.6% |
| `curious`=0.7, `playful`=0.6 | 7 | 84 | 8.3% |
| `playful`=0.7, `teasing`=0.7 | 7 | 89 | 7.9% |
| `happy`=0.8 | 7 | 98 | 7.1% |
| `confused`=0.7, `surprised`=0.6 | 7 | 82 | 8.5% |
| `frustrated`=0.5 | 6 | 130 | 4.6% |
| `angry`=0.6 | 6 | 118 | 5.1% |
| `happy`=0.7, `teasing`=0.6 | 6 | 86 | 7.0% |
| `authoritative`=0.7, `calm`=0.8 | 6 | 76 | 7.9% |
| `playful`=0.5 | 6 | 185 | 3.2% |
| `annoyed`=0.6 | 6 | 120 | 5.0% |
| `excited`=0.8, `teasing`=0.6 | 6 | 83 | 7.2% |
| `calm`=0.8, `disappointed`=0.7, `sarcasm`=0.7 | 6 | 80 | 7.5% |
| `curious`=0.8 | 6 | 115 | 5.2% |
| `teasing`=0.7 | 6 | 110 | 5.5% |

---
## 4. Text Length Analysis

### Word count statistics

| Metric | Error files | All files |
|--------|-------------|-----------|
| Min | 1 | 1 |
| Max | 71 | 74 |
| Mean | 24.4 | 18.1 |
| Median | 23 | 15.0 |
| Stdev | 14.9 | 12.9 |

### Error rate by word count bucket

| Word count range | Errors | Total | Error Rate |
|-----------------|--------|-------|------------|
| 0-10 words | 67 | 6139 | 1.1% |
| 10-20 words | 100 | 5993 | 1.7% |
| 20-30 words | 106 | 3753 | 2.8% |
| 30-50 words | 99 | 2715 | 3.6% |
| 50-100 words | 29 | 574 | 5.1% |
| 100-999 words | 0 | 0 | 0.0% |

---
## 5. Multi-Emotion vs Single-Emotion

| Type | Error Count | Total | Error Rate |
|------|-------------|-------|------------|
| Single emotion | 284 | 16042 | 1.8% |
| Multi-emotion (2+) | 117 | 3132 | 3.7% |

### Multi-emotion combos in errors:

| Combination | Count |
|-------------|-------|
| `happy`=0.7 + `playful`=0.6 | 14 |
| `excited`=0.8 + `playful`=0.7 | 9 |
| `dramatic`=0.7 + `surprised`=0.8 | 9 |
| `frustrated`=0.6 + `playful`=0.5 | 8 |
| `disappointed`=0.7 + `sarcasm`=0.6 | 8 |
| `curious`=0.7 + `playful`=0.6 | 7 |
| `playful`=0.7 + `teasing`=0.7 | 7 |
| `confused`=0.7 + `surprised`=0.6 | 7 |
| `happy`=0.7 + `teasing`=0.6 | 6 |
| `authoritative`=0.7 + `calm`=0.8 | 6 |
| `excited`=0.8 + `teasing`=0.6 | 6 |
| `calm`=0.8 + `disappointed`=0.7 + `sarcasm`=0.7 | 6 |
| `angry`=0.7 + `authoritative`=0.8 | 3 |
| `calm`=0.8 + `serious`=0.7 | 3 |
| `frustrated`=0.7 + `playful`=0.5 + `sarcasm`=0.7 | 2 |
| `affectionate`=0.7 + `happy`=0.7 | 2 |
| `confused`=0.6 + `curious`=0.7 | 2 |
| `disappointed`=0.6 + `teasing`=0.5 | 2 |
| `dramatic`=0.7 + `serious`=0.8 | 1 |
| `concerned`=0.7 + `serious`=0.7 | 1 |

---
## 6. Risk Prediction for Remaining ~17k Files

Using high-risk emotion+intensity combinations (>=30% error rate, >=5 samples),  
**0** files in the remaining dataset are flagged as likely errors.


**High-risk (emotion, intensity) pairs identified:**


---
## 7. Recommended Action Plan

Based on the analysis, here is a prioritized filtering strategy:

1. **Auto-reject high-risk combos**: Files with any of the RED emotion+intensity combinations should be regenerated without manual review.
2. **Flag long files**: Files with >50 words where ANY emotion tag appears are more likely to have speaker leakage -- batch those for quick spot-checks.
3. **Focus manual review on YELLOW medium-risk**: Files with medium-risk emotions but not caught by rule 1 should be spot-checked (~10% sampling).
4. **Keep low-risk files as-is**: Files with emotions not appearing in this table can be assumed clean.
