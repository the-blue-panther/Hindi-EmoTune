import torch
import numpy as np
from TTS.api import TTS

class EmoKnobTTS(TTS):
    def __init__(self, model_name="tts_models/multilingual/multi-dataset/xtts_v2", gpu=False):
        super().__init__(model_name=model_name, gpu=gpu)
        print("✅ EmoKnobTTS initialized with emotion control support")

    def tts_with_emotion(self, text, speaker_wav, language, file_path, emotion_vec_path=None, alpha=0.0):
        """
        Synthesizes speech using XTTS with emotion modulation.

        Args:
            text: Text to speak
            speaker_wav: Path to neutral speaker reference
            language: Language code (e.g., 'hi')
            file_path: Output file path
            emotion_vec_path: Path to .npy emotion vector
            alpha: Intensity of emotion (0.0–1.0)
        """
        # 1️⃣ Load neutral speaker embedding
        wav, sr = self.load_audio(speaker_wav)
        emb = self.get_speaker_embedding(wav, sr)

        # 2️⃣ Apply emotion direction vector (if provided)
        if emotion_vec_path:
            v = np.load(emotion_vec_path)
            v = v / np.linalg.norm(v)
            emb = emb + alpha * v
            print(f"[EmoKnob] Injected emotion vector with α={alpha}")
        else:
            print("[EmoKnob] No emotion vector provided — neutral synthesis.")

        # 3️⃣ Convert to torch tensor
        emb_tensor = torch.tensor(emb, dtype=torch.float32)

        # 4️⃣ Run synthesis (internal XTTS function supports embedding)
        wav_out = self.synthesizer.tts(
            text=text,
            speaker_embeddings=emb_tensor.unsqueeze(0),
            language=language
        )

        # 5️⃣ Save audio
        self.synthesizer.save_wav(wav_out, file_path)
        print(f"✅ Emotionally modulated audio saved: {file_path}")

    def load_audio(self, path, sr=16000):
        import librosa
        wav, s = librosa.load(path, sr=sr, mono=True)
        return wav, s

    def get_speaker_embedding(self, wav, sr=16000):
        """Extracts XTTS speaker embedding"""
        # Use the internal embedding model directly
        emb = self.synthesizer.tts_model.compute_speaker_embedding(torch.tensor(wav).unsqueeze(0))
        return emb.squeeze().cpu().numpy()
