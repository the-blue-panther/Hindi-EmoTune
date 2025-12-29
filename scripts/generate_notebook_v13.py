"""
generate_notebook_v13.py
Generates hindi_emoknob_demo_v13.ipynb for the Bengali_EmoKnob project.
Created to keep all models under models/ and use native XTTS-v2 + ai4bharat indicwav2vec-hindi.
"""

import json, os, textwrap, sys
from pathlib import Path
import nbformat
from nbformat.v4 import new_notebook, new_code_cell, new_markdown_cell

# ---------- USER PROJECT ROOT ----------
PROJECT_ROOT = Path(r"D:\Downloads\Bengali_EmoKnob")
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
MODELS_DIR = PROJECT_ROOT / "models"
NOTEBOOK_PATH = PROJECT_ROOT / "hindi_emoknob_demo_v13.ipynb"
# ---------------------------------------

def make_cells():
    cells = []

    # ---------------------
    # 0. Title / README
    # ---------------------
    cells.append(new_markdown_cell(
        "# Hindi EmoKnob — Demo (v13)\n\n"
        "**Native XTTS-v2 + IndicWav2Vec emotion injection**.\n\n"
        "All models stay under `models/`."
    ))

    # ---------------------
    # 1. Environment setup
    # ---------------------
    cell_env = f"""
# Cell 1 — Environment & Paths
import os, sys, json, shutil
from pathlib import Path
import torch

PROJECT_ROOT = Path(r"{PROJECT_ROOT}")
MODELS_DIR = PROJECT_ROOT / "models"
XTTS_LOCAL_DIR = MODELS_DIR / "xtts_v2"
INDIC_DIR = MODELS_DIR / "ai4bharat_indicwav2vec_hindi"

# Create folders
for p in [MODELS_DIR, XTTS_LOCAL_DIR]:
    p.mkdir(parents=True, exist_ok=True)

# Torch threads (safe version)
try:
    torch.set_num_threads(2)
except Exception as e:
    print("Warning threads:", e)

SR = 22050
print("Project root:", PROJECT_ROOT)
print("Models:", MODELS_DIR)
print("XTTS local:", XTTS_LOCAL_DIR)
print("Indic local:", INDIC_DIR)
"""
    cells.append(new_code_cell(cell_env))

    # ---------------------
    # 2. Utilities
    # ---------------------
    cell_utils = r"""
# Cell 2 — Utilities: ffmpeg audio conversion + preprocessing
import subprocess, librosa, soundfile as sf
from pathlib import Path
import numpy as np

def run_ffmpeg_convert_to_wav(in_path, out_path, sr=SR):
    '''Convert any audio file to mono WAV using ffmpeg.'''
    cmd = [
        "ffmpeg", "-y", "-i", str(in_path),
        "-ac", "1", "-ar", str(sr),
        "-vn", "-hide_banner", "-loglevel", "error",
        str(out_path)
    ]
    subprocess.run(cmd, check=True)

def preprocess_audio(in_path, out_wav_path, sr=SR, do_noise_reduce=False):
    '''Load audio → convert → normalize → save.'''
    tmp = out_wav_path.with_suffix(".tmp.wav")
    run_ffmpeg_convert_to_wav(in_path, tmp, sr=sr)

    y, _ = librosa.load(str(tmp), sr=sr, mono=True)

    peak = max(1e-9, float(np.max(np.abs(y))))
    y = 0.95 * (y / peak)
    sf.write(str(out_wav_path), y, sr)

    try: tmp.unlink()
    except: pass
    return out_wav_path

def unique_path(path: Path):
    '''Return non-conflicting path (auto _1, _2, _3…).'''
    path = Path(path)
    if not path.exists(): return path
    i = 1
    while True:
        p = path.with_name(f"{path.stem}_{i}{path.suffix}")
        if not p.exists(): return p
        i += 1
"""
    cells.append(new_code_cell(cell_utils))

    # ---------------------
    # 3. XTTS Download + Load
    # ---------------------
    cell_xtts = r"""
# Cell 3 — Download & load XTTS-v2 locally
from huggingface_hub import snapshot_download
from TTS.api import TTS
import appdirs, shutil, traceback
from pathlib import Path

def ensure_xtts_local(target_dir: Path):
    '''Ensure XTTS-v2 model exists in models/xtts_v2'''
    ck = target_dir / "model.pth"
    cfg = target_dir / "config.json"

    if ck.exists() and cfg.exists():
        print("XTTS already present locally.")
        return True

    print("Attempting snapshot_download...")
    try:
        snapshot_download(repo_id="coqui/xtts-v2",
                          repo_type="model",
                          local_dir=str(target_dir),
                          local_dir_use_symlinks=False)
    except Exception as e:
        print("snapshot_download failed:", e)

    # Fallback: allow TTS to download into user cache
    try:
        tmp = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2")
        cache_dir = Path(appdirs.user_data_dir("tts"))
        found = list(cache_dir.rglob("tts_models--multilingual--multi-dataset--xtts_v2*"))
        if found:
            shutil.copytree(found[0], target_dir, dirs_exist_ok=True)
    except Exception as e:
        print("Fallback TTS download failed:", e)

    return (target_dir/"model.pth").exists()

ensure_xtts_local(XTTS_LOCAL_DIR)

def load_xtts(gpu=False):
    '''Load XTTS from models/xtts_v2 or fallback to model_name.'''
    ck = XTTS_LOCAL_DIR / "model.pth"
    cfg = XTTS_LOCAL_DIR / "config.json"
    if ck.exists() and cfg.exists():
        try:
            print("Loading XTTS locally…")
            return TTS(model_path=str(ck), config_path=str(cfg), gpu=gpu)
        except Exception as e:
            print("Local XTTS load failed:", e)
    print("Loading XTTS via model_name (cache)…")
    return TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", gpu=gpu)

XTTS = load_xtts(gpu=False)
print("XTTS ready.")
"""
    cells.append(new_code_cell(cell_xtts))

    # ---------------------
    # 4. IndicWav2Vec Load
    # ---------------------
    cell_indic = r"""
# Cell 4 — Load ai4bharat IndicWav2Vec Hindi (local)
from transformers import Wav2Vec2Processor, Wav2Vec2Model

if not INDIC_DIR.exists():
    print("ERROR: Indic model missing:", INDIC_DIR)
else:
    print("Loading IndicWav2Vec from:", INDIC_DIR)
    processor = Wav2Vec2Processor.from_pretrained(str(INDIC_DIR))
    indic_enc = Wav2Vec2Model.from_pretrained(str(INDIC_DIR))
    indic_enc.eval()
    print("Indic encoder loaded.")
"""
    cells.append(new_code_cell(cell_indic))

    # ---------------------
    # 5. Embedding Helpers
    # ---------------------
    cell_embeddings = r"""
# Cell 5 — Embedding helpers (Indic + XTTS conditioning latents)
import numpy as np, librosa, torch

def get_indic_embedding(wav_path, sr=SR):
    '''Return Indic embedding (1024-D).'''
    if 'processor' not in globals():
        raise RuntimeError("Indic encoder not loaded.")
    y, _ = librosa.load(str(wav_path), sr=sr, mono=True)
    inp = processor(y, sampling_rate=sr, return_tensors="pt")
    with torch.no_grad():
        out = indic_enc(**inp).last_hidden_state
    return out.mean(dim=1).squeeze().detach().cpu().numpy()

def get_xtts_speaker_latent(tts, wav_path):
    '''Return XTTS speaker latent (512-D).'''
    if not hasattr(tts, "tts_model"):
        raise RuntimeError("XTTS tts_model missing — use native XTTS-v2.")
    y, _ = librosa.load(str(wav_path), sr=SR, mono=True)
    w = torch.tensor(y).float().unsqueeze(0)
    with torch.no_grad():
        lat = tts.tts_model.get_conditioning_latents(w)
        if isinstance(lat, (tuple,list)):
            return lat[1].squeeze().cpu().numpy()
        return lat.squeeze().cpu().numpy()

print("Embedding helpers ready.")
"""
    cells.append(new_code_cell(cell_embeddings))

    # ---------------------
    # 6. Projection methods (CCA, PLS)
    # ---------------------
    cell_proj = r"""
# Cell 6 — Projection: CCA, PLS, XTTS-native
from sklearn.cross_decomposition import CCA, PLSRegression

def fit_cca_pls(X, Y, method="cca", n_comp=64):
    '''Fit CCA/PLS map from Indic → XTTS space.'''
    if method == "cca":
        m = CCA(n_components=n_comp, max_iter=500)
        m.fit(X, Y)
        return m
    if method == "pls":
        m = PLSRegression(n_components=n_comp, max_iter=500)
        m.fit(X, Y)
        return m
    raise ValueError("Unknown method:", method)

def map_indic_to_xtts(model, v):
    '''Map 1024-D Indic vector → 512-D XTTS vector.'''
    v = np.asarray(v).reshape(1,-1)
    if isinstance(model, CCA):
        u, y = model.transform(v, np.zeros((1, model.n_components)))
        return y.ravel()
    if isinstance(model, PLSRegression):
        return model.predict(v).ravel()
    return v.ravel()
"""
    cells.append(new_code_cell(cell_proj))

    # ---------------------
    # 7. Emotion Vector Computation
    # ---------------------
    cell_ev = r"""
# Cell 7 — Compute emotion vector (multi-sample, average or single)
import numpy as np
from pathlib import Path

EMOTION_SAMPLES_DIR = PROJECT_ROOT/"data"/"emotion_samples"
OUT_SINGLE = PROJECT_ROOT/"data"/"outputs"/"emotion_vectors"/"single"
OUT_AVG = PROJECT_ROOT/"data"/"outputs"/"emotion_vectors"/"average"
for p in [OUT_SINGLE, OUT_AVG]: p.mkdir(parents=True, exist_ok=True)

def compute_emotion_vector_xtts_multi(emotion_dir, method="cca", mode="average", sample_id=1, n_comp=64):
    '''Compute emotion direction using Indic→XTTS mapping (CCA/PLS) or XTTS-native.'''
    emotion_dir = Path(emotion_dir)
    sample_dirs = [d for d in sorted(emotion_dir.iterdir()) if d.is_dir()]
    if not sample_dirs:
        raise RuntimeError("No samples in: "+str(emotion_dir))

    Xlist, Ylist = [], []
    pairs = []

    for sd in sample_dirs:
        # detect pair
        files = list(sd.glob("*.*"))
        n = e = None
        for f in files:
            s=f.stem.lower()
            if "neutral" in s: n=f
            elif emotion_dir.name.lower() in s: e=f
        if not n or not e:
            wavs = [x for x in files if x.suffix.lower() in [".wav",".mp3",".m4a",".flac"]]
            if len(wavs)>=2: n,e = wavs[:2]
            else: continue

        n_clean = sd/(n.stem+"_clean.wav")
        e_clean = sd/(e.stem+"_clean.wav")
        if not n_clean.exists(): preprocess_audio(n, n_clean)
        if not e_clean.exists(): preprocess_audio(e, e_clean)

        xi = get_indic_embedding(n_clean)
        xe = get_indic_embedding(e_clean)
        yi = get_xtts_speaker_latent(XTTS, n_clean)
        ye = get_xtts_speaker_latent(XTTS, e_clean)

        Xlist.append(xe - xi)
        Ylist.append(ye - yi)
        pairs.append((sd.name, xe-xi, ye-yi))

    if method == "xtts_native":
        if mode == "single":
            idx = sample_id-1
            _,_,v = pairs[idx]
            return v/np.linalg.norm(v)
        avg = np.mean([p[2] for p in pairs], axis=0)
        return avg/np.linalg.norm(avg)

    # CCA/PLS mapping
    X = np.stack(Xlist)
    Y = np.stack(Ylist)
    model = fit_cca_pls(X, Y, method=method, n_comp=min(n_comp, X.shape[1], Y.shape[1]))

    if mode == "single":
        idx=sample_id-1
        v_indic = pairs[idx][1]
    else:
        v_indic = np.mean([p[1] for p in pairs], axis=0)

    v_indic = v_indic/np.linalg.norm(v_indic)
    mapped = map_indic_to_xtts(model, v_indic)
    return mapped/np.linalg.norm(mapped)
"""
    cells.append(new_code_cell(cell_ev))

    # ---------------------
    # 8. TTS synthesis injection
    # ---------------------
    cell_synth = r"""
# Cell 8 — Inject emotion & synthesize TTS output
import numpy as np, torch, soundfile as sf

OUT_GEN = PROJECT_ROOT/"data"/"outputs"/"generated"
OUT_GEN.mkdir(parents=True, exist_ok=True)

def apply_emotion_and_synthesize(text, speaker_wav, emotion_vec, alpha=0.7, lang="hi"):
    '''Inject emotion vector & synthesize using XTTS-v2.'''
    emotion_vec = emotion_vec / (np.linalg.norm(emotion_vec)+1e-12)

    sp = get_xtts_speaker_latent(XTTS, speaker_wav)
    sp = sp.astype(float)

    if emotion_vec.shape[0] != sp.shape[0]:
        # trim/pad
        mn = min(len(sp), len(emotion_vec))
        sp = sp[:mn]; emotion_vec = emotion_vec[:mn]

    new_sp = sp + alpha*emotion_vec

    out = unique_path(OUT_GEN/"test_hindi_emotional.wav")
    try:
        XTTS.tts_to_file(text=text, speaker_wav=str(speaker_wav),
                         language=lang, speaker_embedding=torch.tensor(new_sp).unsqueeze(0),
                         file_path=str(out))
        print("Saved:", out)
        return out
    except Exception as e:
        raise RuntimeError("XTTS synthesis failed: "+str(e))
"""
    cells.append(new_code_cell(cell_synth))

    # ---------------------
    # 9. GUI
    # ---------------------
    cell_gui = r"""
# Cell 9 — Interactive GUI
import ipywidgets as widgets
from IPython.display import display

def list_speakers():
    sp = PROJECT_ROOT/"data"/"speakers"
    if not sp.exists(): sp.mkdir(parents=True)
    return [x.name for x in sp.iterdir() if x.suffix.lower() in [".wav",".mp3",".m4a",".flac"]]

emotion_dd = widgets.Dropdown(
    options=[p.name for p in (PROJECT_ROOT/"data"/"emotion_samples").iterdir() if p.is_dir()],
    description="Emotion:"
)
method_dd = widgets.Dropdown(
    options=["cca","pls","xtts_native"], description="Method:"
)
mode_dd = widgets.Dropdown(
    options=["average","single"], description="Mode:"
)
sid_in = widgets.IntText(value=1, description="Sample ID:")
alpha_sl = widgets.FloatSlider(value=0.7, min=0, max=1, step=0.01, description="alpha")
speaker_dd = widgets.Dropdown(options=list_speakers(), description="Speaker:")
text_in = widgets.Text(value="मैं आज बहुत खुश हूँ।", description="Text:")
run_btn = widgets.Button(description="RUN", button_style="success")
log = widgets.Output()

display(widgets.VBox([emotion_dd, method_dd, mode_dd, sid_in,
                      alpha_sl, speaker_dd, text_in, run_btn, log]))

def on_click(b):
    with log:
        log.clear_output()
        try:
            emo = emotion_dd.value
            method = method_dd.value
            mode = mode_dd.value
            sid = sid_in.value
            alpha = alpha_sl.value
            text = text_in.value
            sp = PROJECT_ROOT/"data"/"speakers"/speaker_dd.value

            print("Running…", emo, method, mode)

            v = compute_emotion_vector_xtts_multi(
                PROJECT_ROOT/"data"/"emotion_samples"/emo,
                method=method, mode=mode, sample_id=sid
            )
            out = apply_emotion_and_synthesize(text, sp, v, alpha)
            print("Done:", out)
        except Exception as e:
            import traceback; traceback.print_exc()

run_btn.on_click(on_click)
print("GUI ready.")
"""
    cells.append(new_code_cell(cell_gui))

    # ---------------------
    # 10. Footer
    # ---------------------
    cells.append(new_markdown_cell(
        "### ✔️ Notebook generated successfully.\n"
        "Run each cell in order."
    ))

    return cells

def write_notebook(path):
    nb = new_notebook(cells=make_cells())
    nb["metadata"] = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        }
    }
    with open(path, "w", encoding="utf8") as f:
        nbformat.write(nb, f)
    print("Notebook written to:", path)

if __name__ == "__main__":
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_notebook(NOTEBOOK_PATH)
    print("Done.")
