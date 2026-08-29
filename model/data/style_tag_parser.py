"""
KionTTS Style and Emotion Tag Parser
Handles extraction, normalization, aliasing, and continuous vector representation
for emotion and style tags.
"""

import re
from typing import Dict, List, Tuple, Optional
import numpy as np

# Canonical lists
EMOTIONS = [
    "angry",
    "annoyed",
    "bored",
    "concerned",
    "confused",
    "curious",
    "disappointed",
    "excited",
    "frustrated",
    "happy",
    "heartbroken",
    "overjoyed",
    "sad",
    "surprised",
]

STYLES = [
    "affectionate",
    "authoritative",
    "calm",
    "deadpan",
    "dramatic",
    "playful",
    "sarcasm",
    "serious",
    "soothing",
    "teasing",
]

# Tag Aliases for normalization
TAG_ALIASES = {
    "suprised": "surprised",
    "sarcastic": "sarcasm",
    "heartbreak": "heartbroken",
    "dry-sarcasm": "sarcasm",
    "sarcastic-deadpan": "sarcasm",
    "sarcastic-playful": "sarcasm",
    "sarcastic-biting": "sarcasm",
    "happy-mild": "happy",
    "happy-strong": "happy",
    "sad-mild": "sad",
    "sad-strong": "sad",
    "concerned-mild": "concerned",
    "concerned-strong": "concerned",
    "confused-mild": "confused",
    "confused-strong": "confused",
    "curious-mild": "curious",
    "curious-strong": "curious",
    "angry-mild": "angry",
    "angry-strong": "angry",
    "annoyed-mild": "annoyed",
    "annoyed-strong": "annoyed",
    "disappointed-mild": "disappointed",
    "disappointed-medium": "disappointed",
    "disappointed-strong": "disappointed",
    "dramatic-medium": "dramatic",
    "dramatic-strong": "dramatic",
    "excited-medium": "excited",
    "excited-strong": "excited",
    "frustrated-mild": "frustrated",
    "frustrated-strong": "frustrated",
    "surprised-mild": "surprised",
    "surprised-strong": "surprised",
}

TAG_REGEX = re.compile(r"\[([a-zA-Z0-9_\-,\.=\s]+)\]")
PAIR_REGEX = re.compile(r"([a-zA-Z0-9_\-]+)(?:=([0-9\.]+))?")

EMOTION_TO_IDX = {emo: i for i, emo in enumerate(EMOTIONS)}
STYLE_TO_IDX = {sty: i for i, sty in enumerate(STYLES)}
NUM_EMOTIONS = len(EMOTIONS)
NUM_STYLES = len(STYLES)
TOTAL_TAGS = NUM_EMOTIONS + NUM_STYLES


def normalize_tag(tag_name: str) -> str:
    """Normalize tag name by stripping whitespace, converting to lower case, and resolving aliases."""
    tag_clean = tag_name.strip().lower()
    if tag_clean in TAG_ALIASES:
        return TAG_ALIASES[tag_clean]
    
    # Strip common intensity qualifiers like -mild, -medium, -strong
    base_tag = re.sub(r"-(mild|medium|strong|playful|deadpan|biting)$", "", tag_clean)
    if base_tag in TAG_ALIASES:
        return TAG_ALIASES[base_tag]
    if base_tag in EMOTION_TO_IDX or base_tag in STYLE_TO_IDX:
        return base_tag
        
    return tag_clean


def parse_tag_content(content: str) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Parse the inside of a tag bracket, e.g. 'playful=0.7, teasing=0.5' or 'happy-strong=0.9'
    Returns (emotions_dict, styles_dict) with normalized names and float intensities.
    """
    emotions: Dict[str, float] = {}
    styles: Dict[str, float] = {}

    pairs = content.split(",")
    for pair in pairs:
        pair = pair.strip()
        if not pair:
            continue
        match = PAIR_REGEX.match(pair)
        if match:
            raw_tag, raw_val = match.groups()
            norm_tag = normalize_tag(raw_tag)
            intensity = float(raw_val) if raw_val is not None else 1.0
            intensity = max(0.0, min(1.0, intensity))

            if norm_tag in EMOTION_TO_IDX:
                emotions[norm_tag] = intensity
            elif norm_tag in STYLE_TO_IDX:
                styles[norm_tag] = intensity
            else:
                # Fallback: substring matching
                for emo in EMOTIONS:
                    if emo in norm_tag:
                        emotions[emo] = intensity
                        break
                else:
                    for sty in STYLES:
                        if sty in norm_tag:
                            styles[sty] = intensity
                            break

    return emotions, styles


def parse_tagged_text(raw_text: str) -> Tuple[str, Dict[str, float], Dict[str, float]]:
    """
    Extracts emotion and style controls and returns (cleaned_text, emotions, styles).
    If multiple tags are present, aggregates them for utterance-level representation.
    """
    emotions: Dict[str, float] = {}
    styles: Dict[str, float] = {}

    def _replace_match(match):
        nonlocal emotions, styles
        tag_str = match.group(1)
        sub_emos, sub_styles = parse_tag_content(tag_str)
        emotions.update(sub_emos)
        styles.update(sub_styles)
        return ""

    cleaned_text = TAG_REGEX.sub(_replace_match, raw_text).strip()
    # Normalize multiple whitespace
    cleaned_text = re.sub(r"\s+", " ", cleaned_text)
    return cleaned_text, emotions, styles


def create_style_vector(
    emotions: Dict[str, float], styles: Dict[str, float]
) -> np.ndarray:
    """
    Constructs a fixed-size 1D vector of shape (NUM_EMOTIONS + NUM_STYLES,)
    First NUM_EMOTIONS entries represent emotion intensities.
    Next NUM_STYLES entries represent style intensities.
    """
    vec = np.zeros(TOTAL_TAGS, dtype=np.float32)
    for emo, val in emotions.items():
        if emo in EMOTION_TO_IDX:
            vec[EMOTION_TO_IDX[emo]] = float(val)
    for sty, val in styles.items():
        if sty in STYLE_TO_IDX:
            vec[NUM_EMOTIONS + STYLE_TO_IDX[sty]] = float(val)
    return vec


if __name__ == "__main__":
    test_cases = [
        "[sarcastic=0.8] Oh, wonderful.",
        "[playful=0.7,teasing=0.5] You really thought I'd let you do that?",
        "[suprised-strong=0.9] I can't believe this!",
        "[disappointed=0.7,sarcasm=0.6] Oh, brilliant.",
        "Normal sentence without tags.",
    ]

    for tc in test_cases:
        txt, emos, stys = parse_tagged_text(tc)
        vec = create_style_vector(emos, stys)
        print(f"Original: {tc}")
        print(f"Cleaned:  '{txt}'")
        print(f"Emotions: {emos} | Styles: {stys}")
        print(f"Active vector indices: {np.where(vec > 0)[0]}, values: {vec[vec > 0]}\n")
