"""
Colab Cell 07: Interactive Inference & Expressiveness Testing
Loads trained KionTTS model from Google Drive, synthesizes tagged text prompts
(e.g., [sarcastic=0.8], blends [happy=0.7,playful=0.6]), and generates audio outputs.
"""

import os
import sys
import torch
import torchaudio
import torchaudio.transforms as T
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
            val_loss = state.get("best_val_loss", state.get("val_loss", None))
            step = state.get("step", None)
            print(f"[+] Checkpoint loaded successfully! (Step: {step}, Val Loss: {val_loss})")
        else:
            print(f"[-] Warning: Checkpoint not found at {checkpoint_path}. Using uninitialized model.")

        self.model.eval()

        # Torchaudio Exact Inverse Mel Transform & Griffin-Lim (matches extraction filterbank exactly)
        self.inv_mel = T.InverseMelScale(
            n_stft=1024 // 2 + 1,
            n_mels=80,
            sample_rate=24000,
            f_min=0.0,
            f_max=8000.0,
        ).to(self.device)

        self.griffin_lim = T.GriffinLim(
            n_fft=1024,
            win_length=1024,
            hop_length=256,
            n_iter=60,
            power=1.0,
        ).to(self.device)

    def mel_to_audio(self, mel_tensor: torch.Tensor, noise_gate_db: float = 35.0) -> np.ndarray:
        """
        Inverts log-Mel spectrogram tensor (1, 80, T) into clean speech waveform
        with spectral noise-floor gating.
        """
        mel = mel_tensor.clone()

        # Print detailed spectrogram statistics for diagnostics
        mel_max = mel.max().item()
        mel_min = mel.min().item()
        mel_mean = mel.mean().item()
        print(f"[*] Mel Spectrogram Stats -> Min: {mel_min:.2f}, Max: {mel_max:.2f}, Mean: {mel_mean:.2f}")

        # Noise-floor gating: clamp background noise bins below threshold to silence (-11.5)
        gate_threshold = mel_max - (noise_gate_db / 4.3429)  # Convert dB to natural log units
        mel = torch.where(mel < gate_threshold, torch.full_like(mel, -11.5129), mel)

        # Un-log to linear magnitude Mel
        linear_mel = torch.exp(mel)

        # Invert to linear STFT magnitude spectrogram using Torchaudio basis
        spec = self.inv_mel(linear_mel)
        spec = torch.clamp(spec, min=0.0)

        # Griffin-Lim phase reconstruction
        wav = self.griffin_lim(spec)
        wav = wav.squeeze().cpu().numpy()

        # Normalize audio amplitude
        max_val = np.max(np.abs(wav)) + 1e-6
        if max_val > 0:
            wav = wav / max_val * 0.95
        return wav

    def synthesize(
        self,
        tagged_text: str,
        output_wav: Optional[str] = None,
        pace: float = 1.0,
        noise_gate_db: float = 35.0,
        plot: bool = True,
    ) -> np.ndarray:
        """
        Synthesizes speech from tagged text.
        e.g.: "[sarcastic=0.8] Oh, wonderful. That's definitely what I asked for."
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
            mel = outputs["mel"]  # (1, 80, T_frames)
            pred_dur = outputs["pred_durations"][0].cpu().numpy()

        wav = self.mel_to_audio(mel, noise_gate_db=noise_gate_db)

        print(f"\n{'='*55}")
        print(f"Prompt:     '{tagged_text}'")
        print(f"Clean Text: '{clean_text}'")
        print(f"Emotions:   {emotions} | Styles: {styles}")
        print(f"Tokens:     {len(tokens)} phonemes | Frames: {mel.shape[-1]}")
        print(f"Duration:   {len(wav) / 24000:.2f}s ({len(wav)} audio samples)")
        print(f"{'='*55}")

        if output_wav:
            os.makedirs(os.path.dirname(os.path.abspath(output_wav)), exist_ok=True)
            sf.write(output_wav, wav, 24000)
            print(f"[+] Audio saved to: {output_wav}")

        if plot:
            try:
                mel_np = mel[0].cpu().numpy()
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5))
                im = ax1.imshow(mel_np, aspect="auto", origin="lower", cmap="magma")
                fig.colorbar(im, ax=ax1, format="%+2.0f dB")
                ax1.set_title(f"Synthesized Log-Mel Spectrogram: {tagged_text[:40]}...")
                ax1.set_ylabel("Mel Bins (80)")

                ax2.plot(np.linspace(0, len(wav)/24000, len(wav)), wav, color="royalblue", lw=0.8)
                ax2.set_title("Generated Waveform")
                ax2.set_xlabel("Time (seconds)")
                ax2.set_ylabel("Amplitude")
                plt.tight_layout()
                plt.show()
            except Exception:
                pass

        return wav


def run_interactive_suite():
    ckpt = "/content/drive/MyDrive/KionTTS_Checkpoints/best_model.pt"
    if not os.path.exists(ckpt):
        ckpt = "/content/drive/MyDrive/KionTTS_Checkpoints/latest_checkpoint.pt"

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
