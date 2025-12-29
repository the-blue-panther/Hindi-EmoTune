import nbformat as nbf
from pathlib import Path

NOTEBOOK_NAME = "hindi_emoknob_demo_v9.ipynb"

def add_cell(cells, code):
    cells.append(nbf.v4.new_code_cell(code))

nb = nbf.v4.new_notebook()
cells = []

# ------------------ CELL 1 ------------------
add_cell(cells,
"""
# Hindi EmoKnob — v9
### Hybrid Pipeline using:  
- IndicWav2Vec (1024D) for Emotion  
- XTTS-Hindi-Finetuned (512D) for Speaker Embedding + Decoder  
- CCA Projection (1024D → 512D)
---
""")

# ------------------ CELL 2 ------------------
add_cell(cells,
r"""
import os
import torch
import librosa
import numpy as np
from pathlib import Path
from TTS.api import TTS
from sklearn.cross_decomposition import CCA
import pickle

PROJECT_DIR = Path(r"D:/Downloads/Bengali_EmoKnob")
DATA_DIR = PROJECT_DIR / "data"
EMOTION_SAMPLES_DIR = DATA_DIR / "emotion_samples"
MODELS_DIR = PROJECT_DIR / "models"
OUTPUT_DIR = DATA_DIR / "outputs"
OUTPUT_VECTORS_SINGLE = OUTPUT_DIR / "emotion_vectors" / "single"
OUTPUT_VECTORS_AVG = OUTPUT_DIR / "emotion_vectors" / "average"

print("Project paths loaded.")
""")

# ------------------ CELL 3 ------------------
add_cell(cells,
r"""
import noisereduce as nr
import soundfile as sf

def preprocess_audio(input_path, output_path):
    # Preprocess: convert to 16k mono wav + noise reduction
    wav, sr = librosa.load(input_path, sr=16000, mono=True)
    reduced = nr.reduce_noise(y=wav, sr=16000)
    sf.write(output_path, reduced, 16000)
    return output_path

print("Preprocessing ready.")
""")

# ------------------ CELL 4 ------------------
add_cell(cells,
r"""
from transformers import Wav2Vec2Processor, Wav2Vec2Model

enc_model_name = "ai4bharat/indicwav2vec-hindi"
processor = Wav2Vec2Processor.from_pretrained(enc_model_name)
encoder = Wav2Vec2Model.from_pretrained(enc_model_name).to("cpu")

print("IndicWav2Vec-Hindi loaded.")
""")

# ------------------ CELL 5 ------------------
add_cell(cells,
r"""
XTTS_HINDI_DIR = MODELS_DIR / "xtts_hindi_finetuned"

tts = TTS(
    model_path=str(XTTS_HINDI_DIR / "model.pth"),
    config_path=str(XTTS_HINDI_DIR / "config.json"),
    gpu=torch.cuda.is_available()
)

print("XTTS-Hindi-Finetuned loaded.")
""")

# ------------------ CELL 6 ------------------
add_cell(cells,
r"""
def extract_xtts_embedding(wav_path):
    wav, sr = librosa.load(wav_path, sr=16000)
    _, spk = tts.synthesizer.tts_model.get_conditioning_latents(wav)
    return spk.squeeze().cpu().numpy()

print("XTTS embedding extractor ready.")
""")

# ------------------ CELL 7 ------------------
add_cell(cells,
r"""
A_vectors = []
B_vectors = []

for emotion_folder in EMOTION_SAMPLES_DIR.iterdir():
    if not emotion_folder.is_dir():
        continue

    for sample in sorted(emotion_folder.iterdir()):
        if not sample.is_dir():
            continue

        for wav_file in sample.glob("*_clean.wav"):
            wav, sr = librosa.load(wav_file, sr=16000)
            inp = processor(wav, sampling_rate=16000, return_tensors="pt")
            with torch.no_grad():
                out = encoder(**inp).last_hidden_state
            A = out.mean(dim=1).squeeze().cpu().numpy()

            B = extract_xtts_embedding(str(wav_file))

            A_vectors.append(A)
            B_vectors.append(B)

A_vectors = np.array(A_vectors)
B_vectors = np.array(B_vectors)

print("CCA training data:", A_vectors.shape, B_vectors.shape)
""")

# ------------------ CELL 8 ------------------
add_cell(cells,
r"""
cca = CCA(n_components=512)
cca.fit(A_vectors, B_vectors)

with open(MODELS_DIR / "cca_projection.pkl", "wb") as f:
    pickle.dump(cca, f)

print("CCA trained + saved.")
""")

# ------------------ CELL 9 ------------------
add_cell(cells,
r"""
with open(MODELS_DIR / "cca_projection.pkl", "rb") as f:
    cca = pickle.load(f)

def project_emotion_vector_cca(v_1024):
    v = v_1024.reshape(1, -1)
    _, proj = cca.transform(v, np.zeros((1,512)))
    proj = proj.squeeze()
    return proj / (np.linalg.norm(proj) + 1e-12)

print("CCA projection ready.")
""")

# ------------------ CELL 10 ------------------
add_cell(cells,
r"""
def compute_emotion_vector_indic(neutral_path, emotion_path):
    # Extract emotion vector using IndicWav2Vec
    wav_n, _ = librosa.load(neutral_path, sr=16000)
    wav_e, _ = librosa.load(emotion_path, sr=16000)

    inp_n = processor(wav_n, sampling_rate=16000, return_tensors="pt")
    inp_e = processor(wav_e, sampling_rate=16000, return_tensors="pt")

    with torch.no_grad():
        emb_n = encoder(**inp_n).last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
        emb_e = encoder(**inp_e).last_hidden_state.mean(dim=1).squeeze().cpu().numpy()

    v = emb_e - emb_n
    return v / (np.linalg.norm(v) + 1e-12)

print("Emotion vector extractor ready.")
""")

# ------------------ CELL 11 ------------------
add_cell(cells,
r"""
def tts_with_emotion(text, speaker_wav, language, file_path, emotion_vec_path, alpha=0.8):
    wav, _ = librosa.load(speaker_wav, sr=16000)
    _, spk = tts.synthesizer.tts_model.get_conditioning_latents(wav)
    emb_spk = spk.squeeze().cpu().numpy()

    v_1024 = np.load(emotion_vec_path)
    v_512 = project_emotion_vector_cca(v_1024)

    emb_mod = emb_spk + alpha * v_512
    emb_mod = emb_mod / (np.linalg.norm(emb_mod) + 1e-12)
    emb_tensor = torch.tensor(emb_mod, dtype=torch.float32).unsqueeze(0)

    wav_out = tts.tts(
        text=text,
        speaker_wav=speaker_wav,
        language=language,
        emotion_embedding=emb_tensor
    )

    tts.synthesizer.save_wav(wav_out, file_path)
    print("Saved:", file_path)
""")

# ------------------ SAVE NOTEBOOK ------------------
nb['cells'] = cells
with open(NOTEBOOK_NAME, "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print(f"Notebook '{NOTEBOOK_NAME}' generated successfully.")
