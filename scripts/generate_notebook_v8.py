"""
generate_notebook_v8.py
----------------------------------------
This script generates the full hindi_emoknob_demo_v8.ipynb notebook
with:
- XTTS-native multi-sample emotion extraction
- Hybrid audio preprocessing
- Speaker cleaning
- Final emotional TTS synthesis
- Clean Option-B structure (production-ready)

Run this file inside your virtual environment:
    (.venv) python scripts/generate_notebook_v8.py
"""

import nbformat as nbf
from pathlib import Path

PROJECT_ROOT = Path(r"D:\Downloads\Bengali_EmoKnob")

# Output notebook path
OUT_NOTEBOOK = PROJECT_ROOT / "hindi_emoknob_demo_v8.ipynb"


# ----------------------------
# Creating notebook
# ----------------------------

nb = nbf.v4.new_notebook()
cells = []


# ----------------------------
# 1. Title Cell
# ----------------------------
cells.append(nbf.v4.new_markdown_cell(
"""
# Hindi EmoKnob Demo v8  
XTTS-native Emotion Vector Extraction • Multi-Sample Averaging • Hybrid Preprocessing  
**Clean Production Version (Option B)**  
"""
))


# ----------------------------
# 2. Project Setup Cell
# ----------------------------
cells.append(nbf.v4.new_code_cell(
"""
# Project root — change if you moved the project
PROJECT_ROOT = r"D:\\Downloads\\Bengali_EmoKnob"

from pathlib import Path
import os

DATA_DIR = Path(PROJECT_ROOT) / "data"
SPEAKERS_DIR = DATA_DIR / "speakers"
EMOTION_SAMPLES_DIR = DATA_DIR / "emotion_samples"
OUTPUT_DIR = DATA_DIR / "outputs"
OUTPUT_VECTORS_SINGLE = OUTPUT_DIR / "emotion_vectors" / "single"
OUTPUT_VECTORS_AVG = OUTPUT_DIR / "emotion_vectors" / "average"
OUTPUT_AUDIO_DIR = OUTPUT_DIR / "generated_audio"

MODELS_DIR = Path(PROJECT_ROOT) / "models"

# Create folders if they don't exist
for p in [DATA_DIR, SPEAKERS_DIR, EMOTION_SAMPLES_DIR, OUTPUT_AUDIO_DIR,
          OUTPUT_VECTORS_SINGLE, OUTPUT_VECTORS_AVG, MODELS_DIR]:
    p.mkdir(parents=True, exist_ok=True)

print("Project initialized successfully.")
print("Speakers folder:", SPEAKERS_DIR)
print("Emotion samples:", EMOTION_SAMPLES_DIR)
"""
))


# ----------------------------
# 3. Hybrid Preprocessing
# ----------------------------
cells.append(nbf.v4.new_code_cell(
"""
# Hybrid audio preprocessing utilities
import librosa, soundfile as sf
import numpy as np

def trim_silence(y, sr, top_db=30):
    yt, _ = librosa.effects.trim(y, top_db=top_db)
    return yt

def normalize_audio(y, target_rms=0.1):
    rms = np.sqrt(np.mean(y**2)) + 1e-12
    gain = target_rms / rms
    y = y * gain
    peak = np.max(np.abs(y))
    if peak > 0.999:
        y = y / peak
    return y

def denoise_audio(y, sr):
    try:
        import noisereduce as nr
    except:
        return y
    noise = y[:int(0.5 * sr)]
    try:
        return nr.reduce_noise(y=y, sr=sr, y_noise=noise)
    except:
        return y

def preprocess_audio_any(input_path, output_path, denoise=False, sr=16000):
    y, _ = librosa.load(input_path, sr=sr, mono=True)
    y = trim_silence(y, sr)
    if denoise:
        y = denoise_audio(y, sr)
    y = normalize_audio(y)
    sf.write(output_path, y, sr, subtype="PCM_16")
    return output_path

print("Hybrid preprocessing ready.")
"""
))


# ----------------------------
# 4. Load XTTS Model
# ----------------------------
cells.append(nbf.v4.new_code_cell(
"""
# Load XTTS-v2 TTS model
from TTS.api import TTS
import torch, os

# Store models locally inside project folder
os.environ["TTS_HOME"] = str(MODELS_DIR)

def load_xtts_model(gpu=torch.cuda.is_available()):
    model = "tts_models/multilingual/multi-dataset/xtts_v2"
    print("Loading XTTS model...")
    tts = TTS(model_name=model, gpu=gpu)
    print("XTTS loaded.")
    return tts

print("XTTS loader ready.")
"""
))


# ----------------------------
# 5. EmoKnob Wrapper with tts_with_emotion
# ----------------------------
cells.append(nbf.v4.new_code_cell(
"""
# EmoKnobTTS Wrapper Class — XTTS-native Embedding Injection
import numpy as np
import torch
from pathlib import Path

class EmoKnobTTSWrapper:
    def __init__(self, tts_instance):
        self.tts = tts_instance
        self.synth = tts_instance.synthesizer

    def tts_with_emotion(self, text, speaker_wav, language, file_path, emotion_vec_path=None, alpha=0.0):
        # 1. Speaker embedding using XTTS-native
        cond, emb = self.synth.tts_model.get_conditioning_latents(
            audio_path=speaker_wav,
            gpt_cond_len=3,
            max_ref_length=6,
            sound_norm_refs=True
        )
        emb = emb.squeeze().cpu().numpy()
        print("Speaker embedding:", emb.shape)

        # 2. Inject emotion vector
        if emotion_vec_path:
            v = np.load(emotion_vec_path).flatten()
            v = v / (np.linalg.norm(v) + 1e-12)
            emb = emb + alpha * v
            print(f"Injected emotion vector (alpha={alpha})")

        # 3. Shape for XTTS
        emb_tensor = torch.tensor(emb, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)

        # 4. Inference
        out = self.synth.tts_model.inference(
            text=text,
            language=language,
            gpt_cond_latent=cond,
            speaker_embedding=emb_tensor
        )
        wav = out["wav"] if isinstance(out, dict) else out

        # 5. Save (non-overwriting)
        p = Path(file_path)
        count = 1
        while p.exists():
            p = p.with_name(f"{p.stem}_{count}{p.suffix}")
            count += 1

        self.synth.save_wav(wav, str(p))
        print("Saved:", p)
        return str(p)

print("EmoKnobTTSWrapper ready.")
"""
))


# ----------------------------
# 6. Multi-sample XTTS-native Emotion Extraction
# ----------------------------
cells.append(nbf.v4.new_code_cell(
"""
# XTTS-native emotion vector extraction (single + multi-sample)
def compute_single_xtts_vector(neutral_wav, emotion_wav, tts):
    _, emb_n = tts.synthesizer.tts_model.get_conditioning_latents(neutral_wav)
    _, emb_e = tts.synthesizer.tts_model.get_conditioning_latents(emotion_wav)
    emb_n = emb_n.squeeze().cpu().numpy()
    emb_e = emb_e.squeeze().cpu().numpy()
    v = emb_e - emb_n
    return v / (np.linalg.norm(v) + 1e-12)

def compute_emotion_vector_xtts_multi(emotion_dir, tts, mode="average",
                                      save_single=None, save_avg=None):
    emotion_dir = Path(emotion_dir)
    emotion_name = emotion_dir.name
    vectors = []

    for sample in sorted(emotion_dir.iterdir()):
        if not sample.is_dir():
            continue

        neutral = None
        emotion = None

        for f in sample.glob("*.wav"):
            name = f.stem.lower()
            if "neutral" in name:
                neutral = f
            elif emotion_name.lower() in name:
                emotion = f

        if neutral and emotion:
            v = compute_single_xtts_vector(neutral, emotion, tts)
            vectors.append(v)

            if save_single:
                Path(save_single).mkdir(parents=True, exist_ok=True)
                outp = Path(save_single) / f"{emotion_name}_{sample.name}_xtts.npy"
                np.save(outp, v)
                print("Saved:", outp)

    if mode == "single":
        return vectors[0]

    avg = np.mean(vectors, axis=0)
    avg = avg / (np.linalg.norm(avg) + 1e-12)

    if save_avg:
        Path(save_avg).mkdir(parents=True, exist_ok=True)
        outp = Path(save_avg) / f"{emotion_name}_avg_xtts.npy"
        np.save(outp, avg)
        print("Saved averaged vector:", outp)

    return avg

print("XTTS-native emotion extraction ready.")
"""
))


# ----------------------------
# 7. Speaker Cleaning Utility
# ----------------------------
cells.append(nbf.v4.new_code_cell(
"""
from pathlib import Path

def preprocess_all_speakers(denoise=False):
    for f in SPEAKERS_DIR.glob("*.*"):
        if f.suffix.lower() == ".wav" and f.stem.endswith("_clean"):
            continue
        out = f.with_name(f.stem + "_clean.wav")
        print("Cleaning:", f.name, "->", out.name)
        preprocess_audio_any(f, out, denoise=denoise)

print("Speaker cleaning ready.")
"""
))


# ----------------------------
# 8. Final Synthesis Example
# ----------------------------
cells.append(nbf.v4.new_code_cell(
"""
# Example usage — emotional TTS synthesis

# 1. Load model:
# tts = load_xtts_model(gpu=False)

# 2. Wrap:
# emo = EmoKnobTTSWrapper(tts)

# 3. Preprocess speakers:
# preprocess_all_speakers()

# 4. Select files:
# speaker = str(SPEAKERS_DIR / "character_1_clean.wav")
# emotion_vec = str(OUTPUT_VECTORS_AVG / "happy_avg_xtts.npy")
# out_file = OUTPUT_AUDIO_DIR / "test_hindi_emotional.wav"

# 5. Run:
# emo.tts_with_emotion(
#     text="मैं आज बहुत खुश हूँ।",
#     speaker_wav=speaker,
#     language="hi",
#     file_path=str(out_file),
#     emotion_vec_path=emotion_vec,
#     alpha=0.7
# )

print("Final synthesis block ready — uncomment to use.")
"""
))



# ----------------------------
# Save Notebook
# ----------------------------
nb['cells'] = cells

with open(OUT_NOTEBOOK, "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print(f"Notebook generated successfully:\n{OUT_NOTEBOOK}")
