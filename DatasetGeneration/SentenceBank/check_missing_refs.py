import os
import re
import itertools

EMO_REFS_DIR = "/home/nate/AI/Kiontts/DatasetGeneration/EmotionRefs/voices"
AVAILABLE_REFS = os.listdir(EMO_REFS_DIR) if os.path.exists(EMO_REFS_DIR) else []

EMOTIONS_SET = {"angry", "annoyed", "bored", "concerned", "confused", "curious", "disappointed", "excited", "frustrated", "happy", "heartbroken", "overjoyed", "sad", "surprised"}
STYLES_SET = {"affectionate", "authoritative", "calm", "deadpan", "dramatic", "playful", "sarcasm", "serious", "soothing", "teasing"}

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
    mapping = {
        "angry": [(0.6, "angry-mild"), (1.1, "angry-strong")],
        "annoyed": [(0.6, "annoyed-mild"), (1.1, "annoyed-strong")],
        "concerned": [(0.6, "concerned-mild"), (1.1, "concerned-strong")],
        "confused": [(0.6, "confused-mild"), (1.1, "confused-strong")],
        "curious": [(0.6, "curious-mild"), (1.1, "curious-strong")],
        "disappointed": [(0.5, "disappointed-mild"), (0.7, "disappointed-medium"), (1.1, "disappointed-strong")],
        "dramatic": [(0.6, "dramatic-medium"), (1.1, "dramatic-strong")],
        "excited": [(0.6, "excited-medium"), (1.1, "excited-strong")],
        "frustrated": [(0.6, "frustrated-mild"), (1.1, "frustrated-strong")],
        "happy": [(0.6, "happy-mild"), (1.1, "happy-strong")],
        "sad": [(0.6, "sad-mild"), (1.1, "sad-strong")],
        "suprised": [(0.6, "surprised-mild"), (1.1, "surprised-strong")],
        "surprised": [(0.6, "surprised-mild"), (1.1, "surprised-strong")],
        "sarcasm": [(1.1, "dry-sarcasm")]
    }
    if base_emotion in mapping:
        for threshold, name in mapping[base_emotion]:
            if intensity < threshold:
                return f"voice_preview_{name}.wav"
    return f"voice_preview_{base_emotion}.wav"

def get_compound_audio_ref(styles_dict):
    expected_parts = []
    for k, v in styles_dict.items():
        norm_k = 'heartbroken' if k == 'heartbreak' else 'surprised' if k == 'suprised' else 'sarcasm' if k == 'sarcastic' else k
        val_str = str(v).replace('.', '-')
        expected_parts.append(f"{norm_k}{val_str}")
        
    for file in AVAILABLE_REFS:
        if file.startswith("voice_preview_") and file.endswith(".wav"):
            basename = file.replace("voice_preview_", "").replace(".wav", "")
            for perm in itertools.permutations(expected_parts):
                if basename == "-".join(perm):
                    return file
    return None

missing = set()

pattern = r'\[(.*?)\]'
try:
    with open('/home/nate/AI/Kiontts/DatasetGeneration/SentenceBank/sentence_bank.txt', 'r', encoding='utf-8') as f:
        for line in f:
            matches = re.findall(pattern, line)
            for tag_str in matches:
                styles = {}
                try:
                    for pair in tag_str.split(','):
                        if '=' in pair:
                            k, v = pair.split('=')
                            styles[k.strip()] = float(v.strip())
                        elif ':' in pair:
                            k, v = pair.split(':')
                            styles[k.strip()] = float(v.strip())
                except:
                    pass # mostly skip malformed like [eighteen seventy five]
                    
                if not styles:
                    continue
                    
                if len(styles) > 1:
                    ref = get_compound_audio_ref(styles)
                    if not ref:
                        expected_parts = []
                        for k, v in styles.items():
                            norm_k = 'heartbroken' if k == 'heartbreak' else 'surprised' if k == 'suprised' else 'sarcasm' if k == 'sarcastic' else k
                            val_str = str(v).replace('.', '-')
                            expected_parts.append(f"{norm_k}{val_str}")
                        missing.add(f"{tag_str} -> Expected EXACT file: voice_preview_{'-'.join(expected_parts)}.wav (Will fallback to text)")
                else:
                    dominant, intensity = get_dominant_style(styles)
                    ref = get_audio_ref(dominant, intensity)
                    if ref not in AVAILABLE_REFS:
                        missing.add(f"{tag_str} -> Expected file: {ref}")

    print("Missing Audio References:")
    if not missing:
        print("None! All tags have exact matching audio references.")
    else:
        for m in sorted(missing):
            print(f" - [{m}]")
except Exception as e:
    print(f"Error: {e}")
