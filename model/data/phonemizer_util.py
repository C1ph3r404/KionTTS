"""
KionTTS Phonemizer and Text Tokenizer
Converts text into phoneme token ID sequences with punctuation support.
Uses espeak-ng backend with robust fallback.
"""

import re
from typing import List, Dict, Optional

# Standard punctuation and special tokens
_PAD = "_"
_PUNCTUATION = ';:,.!?¡¿—…"«»“” '
_SPECIAL = "-"
_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

# IPA symbols common in English speech synthesis
_IPA_VOWELS = "iyɨʉɯuɪʏʊeøɘɵɤoəɛœɜɞʌɔæɐaɶäɑɒᵻ"
_IPA_CONSONANTS = "pbtdʈɖcɟkɡqɢʔɴŋɲɳnɱmʙrʀⱱɾɽɸβfvθðszʃʒʂʐçʝxɣχʁħʕhɦɬɮʋɹɻjɰlɭʎʟ"
_IPA_DIACRITICS = "ˈˌːˑʼʴʰʱʲʷˠˤˁˀ"

# Combined symbols vocabulary
SYMBOLS = [_PAD] + list(_SPECIAL) + list(_PUNCTUATION) + list(_LETTERS) + list(_IPA_VOWELS) + list(_IPA_CONSONANTS) + list(_IPA_DIACRITICS)
# Deduplicate preserving order
SYMBOLS = list(dict.fromkeys(SYMBOLS))

SYMBOL_TO_ID: Dict[str, int] = {s: i for i, s in enumerate(SYMBOLS)}
ID_TO_SYMBOL: Dict[int, str] = {i: s for i, s in enumerate(SYMBOLS)}
VOCAB_SIZE = len(SYMBOLS)


class KionPhonemizer:
    def __init__(self, language: str = "en-us"):
        self.language = language
        self._backend = None
        try:
            from phonemizer.backend import EspeakBackend
            self._backend = EspeakBackend(
                language=language,
                preserve_punctuation=True,
                with_stress=True,
                words_mismatch="ignore",
            )
        except Exception as e:
            # Fallback will be used
            self._backend = None

    def phonemize_text(self, text: str) -> str:
        """Convert raw text to IPA phoneme string."""
        text = text.strip()
        if not text:
            return ""
        if self._backend is not None:
            try:
                phonemes = self._backend.phonemize([text])[0]
                return phonemes.strip()
            except Exception:
                pass
        # Fallback: simple lower-cased character sequence
        return text.lower()

    def text_to_sequence(self, text: str) -> List[int]:
        """Convert input text directly to list of token IDs."""
        phoneme_str = self.phonemize_text(text)
        sequence = []
        for char in phoneme_str:
            if char in SYMBOL_TO_ID:
                sequence.append(SYMBOL_TO_ID[char])
            elif char.lower() in SYMBOL_TO_ID:
                sequence.append(SYMBOL_TO_ID[char.lower()])
            # Unknown characters are ignored
        return sequence

    def sequence_to_text(self, sequence: List[int]) -> str:
        """Convert token ID sequence back to string."""
        return "".join([ID_TO_SYMBOL.get(idx, "") for idx in sequence])


# ─── Module-Level Convenience Functions ─────────────────────────────────────────
_default_phonemizer: Optional[KionPhonemizer] = None


def get_phonemizer() -> KionPhonemizer:
    """Returns a singleton instance of KionPhonemizer."""
    global _default_phonemizer
    if _default_phonemizer is None:
        _default_phonemizer = KionPhonemizer()
    return _default_phonemizer


def phonemize_text(text: str) -> List[int]:
    """
    Convert input text directly to a list of integer phoneme token IDs.
    Used across KionStyleTTS2 training manifests, datasets, and inference.
    """
    return get_phonemizer().text_to_sequence(text)


def phonemize_to_ipa(text: str) -> str:
    """Convert raw text to IPA phoneme string."""
    return get_phonemizer().phonemize_text(text)


if __name__ == "__main__":
    ph = KionPhonemizer()
    sample = "Hello, I am Kion! How can I help you today?"
    phonemes = ph.phonemize_text(sample)
    seq = ph.text_to_sequence(sample)
    print(f"Original: {sample}")
    print(f"Phonemes: {phonemes}")
    print(f"Sequence IDs ({len(seq)}): {seq}")
    print(f"Reconstructed: {ph.sequence_to_text(seq)}")
    print(f"Vocabulary Size: {VOCAB_SIZE}")
    print(f"Module phonemize_text test: {phonemize_text(sample)}")

