import uuid
from pydub import AudioSegment
from IPython.display import Audio, display
import shutil
import os

os.makedirs("/content/dataset/wavs", exist_ok=True)
os.makedirs("/content/dataset/temp", exist_ok=True)

KION_REF = "/content/kionRefVoice/kion_reference.wav"
EMO_REFS_DIR = "/content/EmotionRefs"

EMOTIONS_SET = {"angry", "annoyed", "bored", "concerned", "confused", "curious", "disappointed", "excited", "frustrated", "happy", "heartbroken", "overjoyed", "sad", "surprised"}
STYLES_SET = {"affectionate", "authoritative", "calm", "deadpan", "dramatic", "playful", "sarcasm", "serious", "soothing", "teasing"}

# ── Alpha safety bounds ────────────────────────────────────────────────────────
# Raw emotion intensity (0.3–0.8) was being passed directly as emo_alpha, which
# caused the reference speaker's identity to bleed into the cloned voice at high
# intensities. Instead we normalise into a safe window and apply a length penalty.
ALPHA_MIN  = 0.25   # never go below this (inaudible effect)
ALPHA_MAX  = 0.58   # never go above this (reference speaker leaks above ~0.6)
ALPHA_LENGTH_PENALTY_ONSET = 30   # words — start reducing alpha above this
ALPHA_LENGTH_PENALTY_RATE  = 0.004  # alpha reduction per word above onset

def compute_emo_alpha(raw_intensity: float, word_count: int) -> float:
    """
    Map raw emotion intensity (0.0–1.0) to a safe emo_alpha, then apply a
    word-count penalty to reduce leakage on long utterances.

    Calibration notes (from error analysis of 401 confirmed bad files):
      - raw_intensity 0.3–0.5  → alpha ~0.28–0.38  (subtle, safe)
      - raw_intensity 0.6–0.7  → alpha ~0.42–0.50  (was the main problem zone)
      - raw_intensity 0.8+     → alpha ~0.52–0.58  (capped hard)
      - every word above 30 shaves off 0.004 further
    """
    # Linear normalise from [0.3, 1.0] → [ALPHA_MIN, ALPHA_MAX]
    normalised = ALPHA_MIN + (raw_intensity - 0.3) / (1.0 - 0.3) * (ALPHA_MAX - ALPHA_MIN)
    normalised = max(ALPHA_MIN, min(ALPHA_MAX, normalised))

    # Length penalty
    excess_words = max(0, word_count - ALPHA_LENGTH_PENALTY_ONSET)
    penalty = excess_words * ALPHA_LENGTH_PENALTY_RATE
    safe_alpha = max(ALPHA_MIN, normalised - penalty)

    return round(safe_alpha, 3)


def split_metadata_tags(styles_dict):
    emotions = {}
    styles = {}
    for k, v in styles_dict.items():
        if k in EMOTIONS_SET:
            emotions[k] = v
        elif k in STYLES_SET:
            styles[k] = v
        else:
            styles[k] = v
    return emotions, styles

def get_dominant_style(styles):
    if not styles: return None, 0.0
    return max(styles.items(), key=lambda x: x[1])

def get_audio_ref(base_emotion, intensity):
    """
    Choose mild vs strong reference audio based on intensity.
    Threshold kept at 0.6 — this controls WHICH reference file is used,
    not the emo_alpha (which is now handled by compute_emo_alpha separately).
    """
    mapping = {
        "angry":       [(0.6, "angry-mild"),          (1.1, "angry-strong")],
        "annoyed":     [(0.6, "annoyed-mild"),         (1.1, "annoyed-strong")],
        "concerned":   [(0.6, "concerned-mild"),       (1.1, "concerned-strong")],
        "confused":    [(0.6, "confused-mild"),        (1.1, "confused-strong")],
        "curious":     [(0.6, "curious-mild"),         (1.1, "curious-strong")],
        "disappointed":[(0.5, "disappointed-mild"),    (0.7, "disappointed-medium"), (1.1, "disappointed-strong")],
        "dramatic":    [(0.6, "dramatic-medium"),      (1.1, "dramatic-strong")],
        "excited":     [(0.6, "excited-medium"),       (1.1, "excited-strong")],
        "frustrated":  [(0.6, "frustrated-mild"),      (1.1, "frustrated-strong")],
        "happy":       [(0.6, "happy-mild"),           (1.1, "happy-strong")],
        "sad":         [(0.6, "sad-mild"),             (1.1, "sad-strong")],
        "suprised":    [(0.6, "surprised-mild"),       (1.1, "surprised-strong")],
        "surprised":   [(0.6, "surprised-mild"),       (1.1, "surprised-strong")],
        "sarcasm":     [(1.1, "dry-sarcasm")]
    }
    if base_emotion in mapping:
        for threshold, name in mapping[base_emotion]:
            if intensity < threshold:
                return f"voice_preview_{name}.wav"
    return f"voice_preview_{base_emotion}.wav"

def get_compound_audio_ref(styles_dict):
    try:
        available_refs = os.listdir(EMO_REFS_DIR)
    except:
        return None

    expected_parts = []
    for k, v in styles_dict.items():
        norm_k = 'heartbroken' if k == 'heartbreak' else 'surprised' if k == 'suprised' else 'sarcasm' if k == 'sarcastic' else k
        val_str = str(v).replace('.', '-')
        expected_parts.append(f"{norm_k}{val_str}")

    import itertools
    for file in available_refs:
        if file.startswith("voice_preview_") and file.endswith(".wav"):
            basename = file.replace("voice_preview_", "").replace(".wav", "")
            for perm in itertools.permutations(expected_parts):
                if basename == "-".join(perm):
                    return file
    return None

def generate_segment(text, styles_dict, temp_path):
    word_count = len(text.split())

    if not styles_dict:
        print(f"[neutral] '{text[:40]}...'")
        tts.infer(
            spk_audio_prompt=KION_REF,
            text=text,
            output_path=temp_path,
            verbose=False
        )
        return

    if len(styles_dict) > 1:
        compound_ref = get_compound_audio_ref(styles_dict)
        dominant_emotion, raw_intensity = get_dominant_style(styles_dict)
        alpha = compute_emo_alpha(raw_intensity, word_count)

        if compound_ref:
            expected_ref = os.path.join(EMO_REFS_DIR, compound_ref)
            print(f"[compound ref] {compound_ref} | alpha={alpha} (raw={raw_intensity}, words={word_count})")
            tts.infer(
                spk_audio_prompt=KION_REF,
                text=text,
                emo_audio_prompt=expected_ref,
                emo_alpha=alpha,
                output_path=temp_path,
                verbose=False
            )
        else:
            blend_desc = " and ".join(k for k in styles_dict.keys())
            print(f"[text blend] '{blend_desc}' | alpha={alpha} (raw={raw_intensity}, words={word_count})")
            tts.infer(
                spk_audio_prompt=KION_REF,
                text=text,
                use_emo_text=True,
                emo_text=f"{blend_desc} delivery",
                emo_alpha=alpha,
                output_path=temp_path,
                verbose=False
            )
        return

    # Single emotion/style
    dominant_emotion, raw_intensity = get_dominant_style(styles_dict)
    alpha = compute_emo_alpha(raw_intensity, word_count)
    ref_filename = get_audio_ref(dominant_emotion, raw_intensity)
    expected_ref = os.path.join(EMO_REFS_DIR, ref_filename)

    if os.path.exists(expected_ref):
        print(f"[audio ref] {ref_filename} | alpha={alpha} (raw={raw_intensity}, words={word_count})")
        tts.infer(
            spk_audio_prompt=KION_REF,
            text=text,
            emo_audio_prompt=expected_ref,
            emo_alpha=alpha,
            output_path=temp_path,
            verbose=False
        )
    else:
        print(f"[text fallback] {ref_filename} not found | alpha={alpha} (raw={raw_intensity}, words={word_count})")
        tts.infer(
            spk_audio_prompt=KION_REF,
            text=text,
            use_emo_text=True,
            emo_text=f"{dominant_emotion} delivery",
            emo_alpha=alpha,
            output_path=temp_path,
            verbose=False
        )

def process_utterance(full_text):
    segments = parse_inline_tags(full_text)
    if not segments:
        return

    sample_id = str(uuid.uuid4())[:8]
    final_wav_path = f"/content/dataset/wavs/{sample_id}.wav"

    temp_files = []
    metadata_segments = []

    for i, seg in enumerate(segments):
        temp_path = f"/content/dataset/temp/{sample_id}_seg{i}.wav"
        generate_segment(seg['text'], seg['styles'], temp_path)
        temp_files.append(temp_path)

        emotions, styles = split_metadata_tags(seg['styles'])
        metadata_segments.append({
            "text": seg['text'],
            "emotions": emotions,
            "styles": styles
        })

    if len(temp_files) == 1:
        shutil.copy(temp_files[0], final_wav_path)
    else:
        combined = AudioSegment.from_wav(temp_files[0])
        for path in temp_files[1:]:
            next_seg = AudioSegment.from_wav(path)
            combined = combined.append(next_seg, crossfade=80)
        combined.export(final_wav_path, format="wav")

    dataset_metadata.append({
        "id": sample_id,
        "text": full_text,
        "segments": metadata_segments,
        "speaker_reference": "kion_reference.wav",
        "wav_path": final_wav_path
    })

    print(f"Generated: {final_wav_path}")
    display(Audio(final_wav_path))
