# ----------------------------------------------------------
# generate_notebook_v12.py — Fully corrected version
# ----------------------------------------------------------

import nbformat
from nbformat.v4 import new_notebook, new_code_cell, new_markdown_cell
from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_NOTEBOOK = PROJECT_ROOT / "hindi_emoknob_demo_v12.ipynb"

MODELS_DIR = PROJECT_ROOT / "models"
XTTS_HINDI_DIR = MODELS_DIR / "xtts_hindi_finetuned"
XTTS_HINDI_MODEL = XTTS_HINDI_DIR / "model.pth"
XTTS_HINDI_CONFIG = XTTS_HINDI_DIR / "config.json"
INDICWV_LOCAL_DIR = MODELS_DIR / "ai4bharat_indicwav2vec_hindi"


def make_cells():
    cells = []

    # --------------------------------------------------------
    # Title Cell
    # --------------------------------------------------------
    cells.append(new_markdown_cell(
        "# Hindi EmoKnob — Demo v12\n"
        "Auto-generated notebook.\n\n"
        "**Select your venv kernel before running.**"
    ))

    # --------------------------------------------------------
    # Cell 1 — Environment + Imports
    # --------------------------------------------------------
    cells.append(new_code_cell("""
# Cell 1 — Env + Imports

import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

from pathlib import Path
import numpy as np
import librosa, soundfile as sf
import torch

PROJECT = Path.cwd()
MODELS_DIR = PROJECT / "models"
XTTS_HINDI_DIR = PROJECT / "models" / "xtts_hindi_finetuned"
XTTS_HINDI_MODEL = XTTS_HINDI_DIR / "model.pth"
XTTS_HINDI_CONFIG = XTTS_HINDI_DIR / "config.json"
INDICWV_LOCAL_DIR = PROJECT / "models" / "ai4bharat_indicwav2vec_hindi"

DATA_DIR = PROJECT / "data"
EMOTION_SAMPLES_DIR = DATA_DIR / "emotion_samples"
SPEAKERS_DIR = DATA_DIR / "speakers"
OUTPUT_DIR = DATA_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SR = 16000

torch.set_num_threads(2)
torch.set_num_interop_threads(1)

print("Environment initialized.")
"""))

    # --------------------------------------------------------
    # Cell 2 — Utilities
    # --------------------------------------------------------
    cells.append(new_code_cell("""
# Cell 2 — Utility Helpers

from pathlib import Path

def unique_path(path):
    p = Path(path)
    base = p
    counter = 1
    while base.exists():
        base = p.with_name(f"{p.stem}_{counter}{p.suffix}")
        counter += 1
    return base

def ensure_wav_mono_16k(in_path, out_path):
    y, sr = librosa.load(str(in_path), sr=None, mono=True)
    if sr != SR:
        y = librosa.resample(y, sr, SR)
    sf.write(str(out_path), y, SR)
    return out_path

print("Utilities loaded.")
"""))

    # --------------------------------------------------------
    # Cell 3 — Preprocessing
    # --------------------------------------------------------
    cells.append(new_code_cell("""
# Cell 3 — Preprocess Audio (resample + noise reduction)

import noisereduce as nr

def preprocess_audio(in_path, out_path):
    out = ensure_wav_mono_16k(in_path, out_path)
    y, _ = librosa.load(str(out), sr=SR)
    try:
        y2 = nr.reduce_noise(y=y, sr=SR)
    except:
        y2 = y
    sf.write(str(out), y2, SR)
    return out

print("Preprocessing ready.")
"""))

    # --------------------------------------------------------
    # Cell 4 — XTTS Loader
    # --------------------------------------------------------
    cells.append(new_code_cell("""
# Cell 4 — XTTS Loader

from TTS.api import TTS
import json

def patch_xtts_config_if_needed(config_path, checkpoint="model.pth"):
    cfgp = Path(config_path)
    cfg = json.loads(cfgp.read_text())
    changed = False
    ma = cfg.get("model_args", {})
    for k in ["gpt_checkpoint", "clvp_checkpoint", "decoder_checkpoint"]:
        if ma.get(k) is None:
            ma[k] = checkpoint
            changed = True
    if changed:
        cfgp.write_text(json.dumps(cfg, indent=4))
        print("[PATCH] Updated config.json for XTTS Hindi.")
    return changed

def load_xtts_model(gpu=False):
    model_path = XTTS_HINDI_MODEL
    config_path = XTTS_HINDI_CONFIG

    print("Loading XTTS from:", model_path)
    patch_xtts_config_if_needed(config_path)

    tts = TTS(
        model_path=str(model_path),
        config_path=str(config_path),
        gpu=gpu
    )
    print("XTTS loaded successfully.")
    return tts

print("XTTS loader ready.")
"""))

    # --------------------------------------------------------
    # Cell 5 — Indic Encoder
    # --------------------------------------------------------
    cells.append(new_code_cell("""
# Cell 5 — IndicWav2Vec Encoder Loader

from transformers import Wav2Vec2Processor, Wav2Vec2Model

proc = None
indic_enc = None

def load_indic_encoder():
    global proc, indic_enc
    local_dir = INDICWV_LOCAL_DIR
    print("Loading Indic model from:", local_dir)
    proc = Wav2Vec2Processor.from_pretrained(str(local_dir))
    indic_enc = Wav2Vec2Model.from_pretrained(str(local_dir))
    print("Indic Encoder loaded.")

def get_indic_embedding(wav_path):
    if proc is None:
        load_indic_encoder()
    y, _ = librosa.load(str(wav_path), sr=SR)
    inp = proc(y, sampling_rate=SR, return_tensors="pt")
    out = indic_enc(**inp).last_hidden_state
    return out.mean(dim=1).squeeze().cpu().numpy()

print("Indic encoder ready.")
"""))

    # --------------------------------------------------------
    # Cell 6 — XTTS Speaker Latent
    # --------------------------------------------------------
    cells.append(new_code_cell("""
# Cell 6 — XTTS speaker latent

def get_xtts_speaker_latent(tts, wav_path):
    y, _ = librosa.load(str(wav_path), sr=SR)
    w = torch.tensor(y).float().unsqueeze(0)
    out = tts.tts_model.get_conditioning_latents(w)
    return out[1].squeeze().cpu().numpy()

print("XTTS latent extractor OK.")
"""))

    # --------------------------------------------------------
    # Cell 7 — Vector Extraction
    # --------------------------------------------------------
    cells.append(new_code_cell("""
# Cell 7 — Compute emotion vector (CCA / PLS / PCA / XTTS-native)

from sklearn.cross_decomposition import CCA, PLSRegression
from sklearn.decomposition import PCA

def compute_emotion_vector_xtts_multi(
        emotion_dir, method="cca", n_comp=64,
        save_single_dir=None, save_avg_dir=None,
        sample_id=None):

    emotion_dir = Path(emotion_dir)
    emotion = emotion_dir.name
    sample_dirs = sorted([p for p in emotion_dir.iterdir() if p.is_dir()])

    if sample_id:
        sample_dirs = [sample_dirs[sample_id - 1]]

    X = []
    Y = []

    for sd in sample_dirs:
        files = list(sd.glob("*.*"))
        neutral = next((f for f in files if "neutral" in f.stem.lower()), None)
        emot = next((f for f in files if emotion in f.stem.lower()), None)

        if not neutral or not emot:
            continue

        n_clean = sd / (neutral.stem + "_clean.wav")
        e_clean = sd / (emot.stem + "_clean.wav")
        if not n_clean.exists(): preprocess_audio(neutral, n_clean)
        if not e_clean.exists(): preprocess_audio(emot, e_clean)

        xi = get_indic_embedding(n_clean)
        xe = get_indic_embedding(e_clean)
        yi = get_xtts_speaker_latent(tts, n_clean)
        ye = get_xtts_speaker_latent(tts, e_clean)

        X.append(xe - xi)
        Y.append(ye - yi)

    if not X:
        print("No valid sample pairs!")
        return None

    X = np.vstack(X)
    Y = np.vstack(Y)

    if method == "xtts_native":
        v = Y.mean(axis=0)

    elif method == "pca":
        p = PCA(n_components=min(n_comp, X.shape[1]))
        Xr = p.fit_transform(X)
        pls = PLSRegression(n_components=min(n_comp, Y.shape[1]))
        pls.fit(Xr, Y)
        v = pls.predict(p.transform(X.mean(axis=0).reshape(1, -1))).squeeze()

    elif method == "pls":
        pls = PLSRegression(n_components=min(n_comp, Y.shape[1]))
        pls.fit(X, Y)
        v = pls.predict(X.mean(axis=0).reshape(1, -1)).squeeze()

    elif method == "cca":
        k = min(n_comp, X.shape[0], X.shape[1], Y.shape[1])
        cca = CCA(n_components=k)
        cca.fit(X, Y)
        x0 = X.mean(axis=0).reshape(1, -1)
        x_c, y_c = cca.transform(x0, Y.mean(axis=0).reshape(1, -1))
        v = (x_c @ cca.y_loadings_.T).squeeze()

    v = v / (np.linalg.norm(v) + 1e-12)

    if save_avg_dir:
        Path(save_avg_dir).mkdir(parents=True, exist_ok=True)
        np.save(Path(save_avg_dir) / f"{emotion}_{method}_avg.npy", v)

    return v

print("Emotion vector extractor ready.")
"""))

    # --------------------------------------------------------
    # Cell 8 — Synthesis
    # --------------------------------------------------------
    cells.append(new_code_cell("""
# Cell 8 — Stylized Synthesis

import soundfile as sf

def tts_with_emotion_safe(tts, text, speaker_wav, emotion_vec, alpha=0.7):
    base_emb = get_xtts_speaker_latent(tts, speaker_wav)
    v = emotion_vec

    if v.shape[0] != base_emb.shape[0]:
        p = PCA(n_components=base_emb.shape[0])
        p.fit(v.reshape(1, -1))
        v = p.transform(v.reshape(1, -1)).squeeze()

    v = v / (np.linalg.norm(v) + 1e-12)
    mod = base_emb + alpha * v
    emb_t = torch.tensor(mod, dtype=torch.float32).unsqueeze(0)

    out = tts.tts(text=text, speaker_embedding=emb_t, language="hi")

    wav = out["wav"] if isinstance(out, dict) else np.array(out)

    outp = unique_path(Path(OUTPUT_DIR) / "test_hindi_emotional.wav")
    sf.write(str(outp), wav, SR)
    print("Saved:", outp)
    return outp

print("Synthesis ready.")
"""))

    # --------------------------------------------------------
    # Cell 9 — GUI
    # --------------------------------------------------------
    cells.append(new_code_cell("""
# Cell 9 — GUI

import ipywidgets as widgets
from IPython.display import display, clear_output

def list_speakers():
    return [str(p) for p in SPEAKERS_DIR.glob("*_clean.wav")]

emotion_dd = widgets.Dropdown(
    options=[p.name for p in EMOTION_SAMPLES_DIR.iterdir() if p.is_dir()],
    description="Emotion:"
)
method_dd = widgets.Dropdown(
    options=["cca","pls","pca","xtts_native"],
    description="Method:"
)
mode_dd = widgets.Dropdown(
    options=["average","single"],
    description="Mode:"
)
sample_id = widgets.BoundedIntText(
    value=1, min=1, max=50, description="Sample:"
)
alpha_s = widgets.FloatSlider(
    value=0.7, min=0, max=1, step=0.01, description="Alpha:"
)
speaker_dd = widgets.Dropdown(
    options=list_speakers(),
    description="Speaker:"
)
tts_text = widgets.Textarea(
    value="यह एक परीक्षण वाक्य है।",
    description="Text:"
)
run_btn = widgets.Button(
    description="Generate",
    button_style="success"
)
out_box = widgets.Output()

def on_click(b):
    with out_box:
        clear_output()

        if "tts" not in globals():
            print("Error: Load XTTS first using:  tts = load_xtts_model(gpu=False)")
            return

        emotion = emotion_dd.value
        method = method_dd.value
        mode   = mode_dd.value
        sid    = sample_id.value
        speaker= speaker_dd.value
        text   = tts_text.value

        emotion_dir = EMOTION_SAMPLES_DIR / emotion

        if mode == "average":
            v = compute_emotion_vector_xtts_multi(emotion_dir, method=method)
        else:
            v = compute_emotion_vector_xtts_multi(
                    emotion_dir, method=method, sample_id=sid)

        if v is None:
            print("Could not compute emotion vector.")
            return

        out = tts_with_emotion_safe(tts, text, speaker, v, alpha=float(alpha_s.value))
        print("Done:", out)

run_btn.on_click(on_click)

display(widgets.VBox([
    emotion_dd, method_dd, mode_dd, sample_id,
    alpha_s, speaker_dd, tts_text,
    run_btn, out_box
]))

print("GUI ready.")
"""))

    return cells


# --------------------------------------------------------
# Build notebook
# --------------------------------------------------------
nb = new_notebook(
    cells=make_cells(),
    metadata={"kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3"
    }}
)

OUT_NOTEBOOK.parent.mkdir(parents=True, exist_ok=True)
with open(OUT_NOTEBOOK, "w", encoding="utf8") as f:
    nbformat.write(nb, f)

print("Notebook generated at:", OUT_NOTEBOOK)
