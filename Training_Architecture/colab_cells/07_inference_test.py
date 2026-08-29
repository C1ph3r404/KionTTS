"""
Colab Cell 07: Interactive Inference & Expressiveness Testing
Loads trained KionTTS model from Google Drive, synthesizes tagged text prompts
(e.g., [sarcastic=0.8], blends [happy=0.7,playful=0.6]), and generates audio outputs.
"""

import os
import sys
import torch
import librosa
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Dict

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from model.models.kion_tts_model import KionTTSModel
from model.data.style_tag_parser import parse_tagged_text, create_style_vector
from model.data.phonemizer_util import KionPhonemizer


class KionSynthesizer:
    def __init__(
        self,
        checkpoint_path: str = "/content/drive/MyDrive/KionTTS_Checkpoints/best_model.pt",
        device: Optional[str] = None,
    ):
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        print(f"Loading KionTTS synthesizer on {self.device} from: {checkpoint_path}")

        self.phonemizer = KionPhonemizer()
        self.model = KionTTSModel().to(self.device)

        if os.path.exists(checkpoint_path):
            state = torch.load(checkpoint_path, map_location=self.device)
            state_dict = state.get("model_state_dict", state)
            self.model.load_state_dict(state_dict)
            print("[+] Checkpoint loaded successfully!")
        else:
            print(f"[-] Warning: Checkpoint not found at {checkpoint_path}. Using uninitialized model.")

        self.model.eval()

    def mel_to_audio_griffin_lim(self, mel_spec: np.ndarray, sr: int = 24000) -> np.ndarray:
        """Inverts log-Mel spectrogram back to waveform via Griffin-Lim."""
        # Convert log Mel back to linear
        mel_linear = np.exp(mel_spec)
        # Approximate linear spectrogram from Mel
        inv_mel_basis = np.linalg.pinv(
            librosa.filters.mel(sr=sr, n_fft=1024, n_mels=80, fmin=0.0, fmax=8000.0)
        )
        spec = np.dot(inv_mel_basis, mel_linear)
        spec = np.maximum(spec, 0)
        # Griffin-Lim reconstruction
        wav = librosa.griffinlim(spec, n_iter=60, hop_length=256, win_length=1024)
        # Normalize
        wav = wav / (np.max(np.abs(wav)) + 1e-6)
        return wav

    def synthesize(
        self,
        tagged_text: str,
        output_wav: Optional[str] = None,
        pace: float = 1.0,
        plot: bool = False,
    ) -> np.ndarray:
        """
        Synthesizes speech from tagged text.
        e.g.: "[sarcastic=0.8] Oh, wonderful. Just what I needed."
        """
        clean_text, emotions, styles = parse_tagged_text(tagged_text)
        style_vec = create_style_vector(emotions, styles)

        tokens = self.phonemizer.text_to_sequence(clean_text)
        if len(tokens) == 0:
            print("Error: Empty phoneme token sequence.")
            return np.zeros(100)

        token_tensor = torch.tensor([tokens], dtype=torch.long, device=self.device)
        style_tensor = torch.tensor([style_vec], dtype=torch.float32, device=self.device)

        with torch.no_grad():
            outputs = self.model.inference(token_tensor, style_tensor, pace=pace)
            mel = outputs["mel"][0].cpu().numpy()  # (80, T)

        wav = self.mel_to_audio_griffin_lim(mel)

        print(f"\nPrompt: '{tagged_text}'")
        print(f"Clean Text: '{clean_text}'")
        print(f"Emotions: {emotions} | Styles: {styles}")
        print(f"Synthesized duration: {len(wav) / 24000:.2f} seconds ({mel.shape[1]} frames)")

        if output_wav:
            os.makedirs(os.path.dirname(os.path.abspath(output_wav)), exist_ok=True)
            sf.write(output_wav, wav, 24000)
            print(f"[+] Audio saved to: {output_wav}")

        if plot:
            plt.figure(figsize=(10, 4))
            plt.imshow(mel, aspect="auto", origin="lower", cmap="viridis")
            plt.colorbar(format="%+2.0f dB")
            plt.title(f"Synthesized Mel: {tagged_text}")
            plt.tight_layout()
            plt.show()

        return wav


def run_interactive_suite():
    # Attempt to load best model or fallback
    ckpt = "/content/drive/MyDrive/KionTTS_Checkpoints/best_model.pt"
    if not os.path.exists(ckpt):
        ckpt = "/content/drive/MyDrive/KionTTS_Checkpoints/prototype_kion.pt"

    synth = KionSynthesizer(checkpoint_path=ckpt)

    test_prompts = [
        "[sarcastic=0.8] Oh, wonderful. That's definitely what I asked for.",
        "[happy=0.8,playful=0.6] I just finished optimizing the entire neural pipeline!",
        "[concerned=0.7,soothing=0.6] Don't worry about the error, I've already patched it.",
        "[angry=0.7,authoritative=0.8] Access denied. That action is strictly restricted.",
        "[calm=0.8,affectionate=0.6] Sleep well, I will monitor the background servers.",
    ]

    out_dir = "/content/drive/MyDrive/KionTTS_Checkpoints/eval_samples"
    os.makedirs(out_dir, exist_ok=True)

    for i, prompt in enumerate(test_prompts):
        wav_path = os.path.join(out_dir, f"sample_{i+1}.wav")
        synth.synthesize(prompt, output_wav=wav_path)


if __name__ == "__main__":
    run_interactive_suite()
