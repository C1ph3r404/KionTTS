!git clone https://github.com/index-tts/index-tts.git
%cd index-tts
!pip install -e .
!pip install pydub  # Used for audio crossfading

from huggingface_hub import snapshot_download
MODEL_DIR = snapshot_download(repo_id="IndexTeam/IndexTTS-2", local_dir="/content/index-tts/checkpoints")

# --- CELL ---

import os
from google.colab import files

# Ensure directories exist
os.makedirs('/content/kionRefVoice', exist_ok=True)
os.makedirs('/content/EmotionRefs', exist_ok=True)

print("1. Upload the main speaker reference file (e.g., kion_reference.wav):")
uploaded_spk = files.upload()
for filename in uploaded_spk.keys():
    os.rename(filename, os.path.join('/content/kionRefVoice', filename))
    print(f"Moved {filename} to /content/kionRefVoice/")

print("\n2. Upload the emotion reference files (e.g., voice_preview_dry-sarcasm.wav):")
uploaded_emo = files.upload()
for filename in uploaded_emo.keys():
    os.rename(filename, os.path.join('/content/EmotionRefs', filename))
    print(f"Moved {filename} to /content/EmotionRefs/")

# --- CELL ---

import os
import subprocess
import sys

print("Reinstalling stable torch suite to resolve C++ extension (NMS) linkage...")

# Force uninstall to clear broken links
subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "-y", "torch", "torchvision", "torchaudio"])

# Install stable versions compatible with Colab CUDA environment
# 2.4.0 is the current standard that fixes the registration errors
subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-cache-dir", "torch==2.4.0", "torchvision==0.19.0", "torchaudio==2.4.0"])

print("\nInstallation complete. RESTARTING RUNTIME to finalize operator registration...")

# --- CELL ---

# Fix Protobuf AttributeError and Numpy/Numba version conflict
!pip install "protobuf==3.20.3" "numpy==2.2.6" --force-reinstall

# --- CELL ---

import sys
import os
import torch

sys.path.append("/content/index-tts")
from indextts.infer_v2 import IndexTTS2

tts = IndexTTS2(
    cfg_path="/content/index-tts/checkpoints/config.yaml",
    model_dir="/content/index-tts/checkpoints",
    use_fp16=torch.cuda.is_available(),
    use_cuda_kernel=False,
)
print("IndexTTS2 loaded.")

# --- CELL ---

import re

def parse_inline_tags(text):
    """
    Parses: `[playful=0.7,teasing=0.5] Oh, you actually managed to do it?`
    Returns: [
        {'styles': {'playful': 0.7, 'teasing': 0.5}, 'text': 'Oh, you actually managed to do it?'}
    ]
    """
    # If there are no brackets at all, treat the entire string as neutral
    if '[' not in text and ']' not in text:
        return [{"styles": {}, "text": text.strip()}]
        
    pattern = r'\[(.*?)\]\s*(.*?)(?=\[|$)'
    matches = re.findall(pattern, text)
    
    segments = []
    for tag_str, content in matches:
        styles = {}
        for style_pair in tag_str.split(','):
            if '=' in style_pair:
                k, v = style_pair.split('=')
                styles[k.strip()] = float(v.strip())
            elif ':' in style_pair:
                k, v = style_pair.split(':')
                styles[k.strip()] = float(v.strip())
        
        segments.append({
            "styles": styles,
            "text": content.strip()
        })
        
    return segments

print("Testing parser (neutral):", parse_inline_tags("Just a normal sentence."))
print("Testing parser (tagged):", parse_inline_tags("[dry-sarcasm=0.7] Oh, fantastic. [concerned-mild=0.6] Are you okay?"))


# --- CELL ---

import uuid
from pydub import AudioSegment
from IPython.display import Audio, display
import shutil
import os

os.makedirs("/content/dataset/wavs", exist_ok=True)
os.makedirs("/content/dataset/temp", exist_ok=True)

KION_REF = "/content/kionRefVoice/kion_reference.wav"
EMO_REFS_DIR = "/content/EmotionRefs"

dataset_metadata = []

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
    if not styles_dict:
        print(f"Using neutral baseline for text: '{text[:20]}...'")
        tts.infer(
            spk_audio_prompt=KION_REF,
            text=text,
            output_path=temp_path,
            verbose=False
        )
        return

    if len(styles_dict) > 1:
        compound_ref = get_compound_audio_ref(styles_dict)
        if compound_ref:
            expected_ref = os.path.join(EMO_REFS_DIR, compound_ref)
            dominant_emotion, intensity = get_dominant_style(styles_dict)
            print(f"Using EXACT COMPOUND audio ref {expected_ref}")
            tts.infer(
                spk_audio_prompt=KION_REF,
                text=text,
                emo_audio_prompt=expected_ref,
                emo_alpha=intensity, 
                output_path=temp_path,
                verbose=False
            )
            return
        else:
            dominant_emotion, intensity = get_dominant_style(styles_dict)
            blend_desc = " and ".join([f"{k}" for k in styles_dict.keys()])
            print(f"No compound ref match, using text blend '{blend_desc}'")
            tts.infer(
                spk_audio_prompt=KION_REF,
                text=text,
                use_emo_text=True,
                emo_text=f"{blend_desc} delivery",
                emo_alpha=intensity,
                output_path=temp_path,
                verbose=False
            )
            return

    dominant_emotion, intensity = get_dominant_style(styles_dict)
    ref_filename = get_audio_ref(dominant_emotion, intensity)
    expected_ref = os.path.join(EMO_REFS_DIR, ref_filename)
    
    if os.path.exists(expected_ref):
        print(f"Using audio ref {expected_ref} for style {dominant_emotion} (intensity {intensity})")
        tts.infer(
            spk_audio_prompt=KION_REF,
            text=text,
            emo_audio_prompt=expected_ref,
            emo_alpha=intensity,
            output_path=temp_path,
            verbose=False
        )
    else:
        print(f"Audio ref {ref_filename} not found, falling back to text-based.")
        tts.infer(
            spk_audio_prompt=KION_REF,
            text=text,
            use_emo_text=True,
            emo_text=f"{dominant_emotion} delivery",
            emo_alpha=intensity,
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



# --- CELL ---

import os
bank_path = "/content/dataset/sentence_bank.txt"

if not os.path.exists(bank_path):
    print(f"Please upload your sentence bank to {bank_path}")
else:
    with open(bank_path, "r") as f:
        # Read lines and ignore empty ones
        sentence_bank = [line.strip() for line in f.readlines() if line.strip()]
    
    print(f"Loaded {len(sentence_bank)} sentences from the bank.")
    
    for text in sentence_bank:
        process_utterance(text)

    with open("/content/dataset/metadata.json", "w") as f:
        json.dump(dataset_metadata, f, indent=2)

    print("Metadata saved to /content/dataset/metadata.json")
