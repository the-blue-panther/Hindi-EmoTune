# generate_notebook_v11.py
# Creates hindi_emoknob_demo_v11.ipynb with fixes:
# - robust XTTS loader (patches config if needed)
# - uses get_conditioning_latents() for XTTS speaker latent
# - supports PCA/CCA/PLS pathways (CCA/PLS preferred for cross-space)
# - improved GUI: dropdowns, Hindi text input, alpha slider 0-1, live logs
# - unique output filenames to avoid overwrites
#
# Usage:
# (1) Put this file in scripts/ inside your project.
# (2) Activate your .venv.
# (3) Run: python scripts/generate_notebook_v11.py
# (4) Open the generated hindi_emoknob_demo_v11.ipynb and run cells.

import nbformat
from nbformat.v4 import new_notebook, new_code_cell, new_markdown_cell
import sys, os
from pathlib import Path

PROJECT_ROOT = Path.cwd().resolve()  # run script from project root
OUT_NOTEBOOK = PROJECT_ROOT / "hindi_emoknob_demo_v11.ipynb"

def make_cells():
    cells = []

    # 0 - Title / instructions
    cells.append(new_markdown_cell(
        "# Hindi EmoKnob — Demo v11\n"
        "Auto-generated notebook. Run cells in order (Kernel: the project's venv Python). "
        "Make sure you have activated the project's `.venv` before installing / running heavy cells."
    ))

    # 1 - Imports & constants
    cells.append(new_code_cell(
r'''# Cell 1 — imports & paths
import os, json, time, shutil
from pathlib import Path
import numpy as np
import soundfile as sf
import librosa
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from TTS.api import TTS
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.cross_decomposition import CCA
import nbformat

# Project paths (adjust if needed)
PROJECT = Path.cwd()
DATA_DIR = PROJECT / "data"
EMOTION_SAMPLES_DIR = DATA_DIR / "emotion_samples"
SPEAKERS_DIR = DATA_DIR / "speakers"
OUTPUT_DIR = DATA_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_SINGLE_DIR = OUTPUT_DIR / "emotion_vectors" / "single"
OUTPUT_AVG_DIR = OUTPUT_DIR / "emotion_vectors" / "average"
OUTPUT_SINGLE_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_AVG_DIR.mkdir(parents=True, exist_ok=True)

MODELS_DIR = PROJECT / "models"
XTTS_HINDI_DIR = MODELS_DIR / "xtts_hindi_finetuned"  # your downloaded model dir
INDICWV_MODEL_ID = "ai4bharat/indicwav2vec-hindi"    # if you have it locally, adjust

SR = 16000  # canonical sample rate for encoder processing (indic and xtts conditioning use 16k or specified)
print('Paths set. Project root:', PROJECT)'''
    ))

    # 2 - utility functions unique_path etc
    cells.append(new_code_cell(
r'''# Cell 2 — utility helpers
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
    """Convert input audio (any format) to mono WAV 16k using librosa. Overwrites if exists."""
    y, sr = librosa.load(str(in_path), sr=None, mono=True)
    if sr != SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=SR)
    sf.write(str(out_path), y, SR)
    return out_path

print('Utility helpers ready.')'''
    ))

    # 3 - preprocess_audio (basic)
    cells.append(new_code_cell(
r'''# Cell 3 — audio preprocessing (basic)
import noisereduce as nr  # optional, install if missing

def preprocess_audio(in_path, out_path):
    """
    Convert to WAV, mono, SR=SR and perform light noise reduction.
    If in_path is already correct and out_path exists, we skip.
    """
    inp = Path(in_path)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    # Convert/resample
    tmp = out.with_suffix('.tmp.wav')
    ensure_wav_mono_16k(inp, tmp)
    # Basic noise reduction (librosa->noisereduce)
    y, _ = librosa.load(str(tmp), sr=SR, mono=True)
    try:
        reduced = nr.reduce_noise(y=y, sr=SR)
    except Exception as e:
        # if noisereduce fails, fallback to original
        reduced = y
    sf.write(str(out), reduced, SR)
    tmp.unlink(missing_ok=True)
    return out

print("Preprocessing function ready. It supports common audio formats (m4a, mp3, wav).")'''
    ))

    # 4 - load/patch XTTS model function
    cells.append(new_code_cell(
r'''# Cell 4 — XTTS loader with config patching helper
import json
from pathlib import Path

def patch_xtts_config_if_needed(config_path, checkpoint_name="model.pth"):
    """
    If model_args.gpt_checkpoint etc are null, patch them to point to checkpoint_name.
    Returns True if patched, False if not needed.
    """
    cfgp = Path(config_path)
    if not cfgp.exists():
        raise FileNotFoundError(cfgp)
    cfg = json.loads(cfgp.read_text(encoding='utf8'))
    changed = False
    ma = cfg.get("model_args", {})
    for k in ("gpt_checkpoint", "clvp_checkpoint", "decoder_checkpoint"):
        if ma.get(k) is None:
            ma[k] = checkpoint_name
            changed = True
    if changed:
        cfg["model_args"] = ma
        cfgp.write_text(json.dumps(cfg, indent=4), encoding='utf8')
    return changed

def load_xtts_model(model_dir=None, model_path=None, config_path=None, gpu=False):
    """
    Attempt to load TTS. If loading fails due to missing config entries, try patching.
    Returns the TTS instance or raises.
    """
    if model_dir:
        model_dir = Path(model_dir)
        model_path = model_path or model_dir / "model.pth"
        config_path = config_path or model_dir / "config.json"
    model_path = Path(model_path)
    config_path = Path(config_path)
    if not model_path.exists() or not config_path.exists():
        raise FileNotFoundError("Model or config missing: ", model_path, config_path)

    # patch config if necessary
    patched = patch_xtts_config_if_needed(config_path, checkpoint_name=model_path.name)
    if patched:
        print("[XTTS loader] Patched config.json to reference:", model_path.name)

    # load TTS
    print("Loading XTTS (may take some seconds)...")
    tts = TTS(model_path=str(model_path), config_path=str(config_path), gpu=gpu)
    print("XTTS loaded.")
    return tts

print("XTTS loader helper ready.")'''
    ))

    # 5 - get_xtts_speaker_latent
    cells.append(new_code_cell(
r'''# Cell 5 — extract XTTS 512-D speaker latent via get_conditioning_latents
import torch
import numpy as np
import librosa

def get_xtts_speaker_latent(tts, wav_path, sr_local=SR):
    """
    Uses tts.tts_model.get_conditioning_latents to obtain the speaker latent.
    Returns numpy array shape (512,).
    """
    if tts is None:
        raise RuntimeError("XTTS not loaded.")
    wav, _ = librosa.load(str(wav_path), sr=sr_local, mono=True)
    wav_t = torch.tensor(wav).float().unsqueeze(0)
    # call inside no-grad
    with torch.no_grad():
        # get_conditioning_latents returns (gpt_latent, speaker_latent)
        out = tts.tts_model.get_conditioning_latents(wav_t)
    if out is None or len(out) < 2:
        raise RuntimeError("get_conditioning_latents did not return expected values.")
    spk = out[1]  # speaker latent tensor
    return spk.squeeze().cpu().numpy()

print("XTTS speaker-latent extraction ready.")'''
    ))

    # 6 - Indic wav2vec embedding helper (ai4bharat)
    cells.append(new_code_cell(
r'''# Cell 6 — IndicWav2Vec embedding (1024-D)
# NOTE: If ai4bharat/indicwav2vec-hindi is gated for you, put its local folder in models/ and set path below.

from transformers import Wav2Vec2Processor, Wav2Vec2Model

def load_indic_encoder(local_path=None):
    if local_path:
        proc = Wav2Vec2Processor.from_pretrained(str(local_path))
        enc = Wav2Vec2Model.from_pretrained(str(local_path))
    else:
        proc = Wav2Vec2Processor.from_pretrained("ai4bharat/indicwav2vec-hindi")
        enc = Wav2Vec2Model.from_pretrained("ai4bharat/indicwav2vec-hindi")
    return proc, enc

print("Load the Indic encoder with: proc, enc = load_indic_encoder(local_path_or_None)")'''
    ))

    # 7 - get_indic_embedding function
    cells.append(new_code_cell(
r'''# Cell 7 — get_indic_embedding
proc = None
enc = None

def ensure_indic_loaded(local_path=None):
    global proc, enc
    if proc is None or enc is None:
        proc, enc = load_indic_encoder(local_path)
    print("Indic encoder loaded.")

def get_indic_embedding(wav_path, sr_local=SR):
    """
    Returns mean pooling of last_hidden_state (1024-D typically).
    """
    ensure_indic_loaded(local_path=str(MODELS_DIR / "ai4bharat_indicwav2vec_hindi") if (MODELS_DIR / "ai4bharat_indicwav2vec_hindi").exists() else None)
    wav, _ = librosa.load(str(wav_path), sr=sr_local, mono=True)
    inputs = proc(wav, sampling_rate=sr_local, return_tensors="pt", padding=True)
    with torch.no_grad():
        out = enc(**inputs).last_hidden_state
    emb = out.mean(dim=1).squeeze().cpu().numpy()
    return emb

print("Indic embedding helper ready (1024-D).")'''
    ))

    # 8 - compute_emotion_vector_xtts_multi with method options
    cells.append(new_code_cell(
r'''# Cell 8 — compute_emotion_vector_xtts_multi (single or average, methods: cca, pls, pca, xtts_native)
import numpy as np
from sklearn.cross_decomposition import CCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA

def compute_emotion_vector_xtts_multi(
    emotion_dir, method="cca", n_comp=128,
    save_single_dir=None, save_avg_dir=None, sample_id=None
):
    """
    emotion_dir: path to an emotion folder that contains sample subfolders.
    method: 'cca', 'pls', 'pca', or 'xtts_native'
    sample_id: if provided and mode single, uses that sample (1-based)
    """
    emotion_dir = Path(emotion_dir)
    emotion_name = emotion_dir.name
    sample_dirs = sorted([p for p in emotion_dir.iterdir() if p.is_dir()])
    if len(sample_dirs) == 0:
        print("[WARN] No samples found under", emotion_dir)
        return None

    selected_samples = sample_dirs
    if sample_id:
        idx = sample_id - 1
        if idx < 0 or idx >= len(sample_dirs):
            raise ValueError("sample_id out of range")
        selected_samples = [sample_dirs[idx]]

    X_indic = []  # rows: samples (neutral->emotion delta)
    Y_xtts = []

    for s in selected_samples:
        # find neutral and emotion files
        wavs = list(s.glob("*.*"))
        neutral = None
        emotion_file = None
        for w in wavs:
            nm = w.stem.lower()
            if "neutral" in nm:
                neutral = w
            if emotion_name.lower() in nm:
                emotion_file = w
        if not neutral or not emotion_file:
            print("[SKIP] sample", s, "missing neutral or emotion file")
            continue

        # preprocess if necessary
        n_clean = s / (neutral.stem + "_clean.wav")
        e_clean = s / (emotion_file.stem + "_clean.wav")
        if not n_clean.exists():
            preprocess_audio(neutral, n_clean)
        if not e_clean.exists():
            preprocess_audio(emotion_file, e_clean)

        # embeddings
        xi = get_indic_embedding(str(n_clean))
        xe = get_indic_embedding(str(e_clean))
        yi = get_xtts_speaker_latent(tts, n_clean)
        ye = get_xtts_speaker_latent(tts, e_clean)

        X_indic.append(xe - xi)  # emotion delta in indic space
        Y_xtts.append(ye - yi)   # speaker delta in xtts space

        # option: save single sample vectors (indic->xtts mapping targets)
        if save_single_dir is not None:
            outdir = Path(save_single_dir)
            outdir.mkdir(parents=True, exist_ok=True)
            np.save(outdir / f"{emotion_name}_{s.name}_indic_delta.npy", (xe - xi))
            np.save(outdir / f"{emotion_name}_{s.name}_xtts_delta.npy", (ye - yi))

    if len(X_indic) == 0:
        print("[WARN] No valid sample pairs processed.")
        return None

    X = np.vstack(X_indic)  # shape (n_samples, 1024)
    Y = np.vstack(Y_xtts)   # shape (n_samples, 512)

    # Methods
    if method == "pca":
        # Project indic delta down to n_comp then map directly (not ideal cross-space)
        p = PCA(n_components=min(n_comp, X.shape[1]))
        Xr = p.fit_transform(X)
        # then learn linear regressor to Y (PLS with 1 component)
        pl = PLSRegression(n_components=min( min(n_comp, Xr.shape[1]), Y.shape[1]))
        pl.fit(Xr, Y)
        # To map an indic delta: pl.predict(p.transform(x))
        # We'll produce a 512-d average mapping vector by mapping mean(X)
        mapped = pl.predict(p.transform(np.mean(X, axis=0).reshape(1,-1))).squeeze()
        v = mapped
    elif method == "cca":
        # Use CCA to find shared directions. Fit on matched pairs X (1024) and Y (512).
        k = min(n_comp, min(X.shape[0], Y.shape[1], X.shape[1]))
        cca = CCA(n_components=min(k, Y.shape[1]))
        cca.fit(X, Y)
        # Transform mean indic delta into cca shared space and inverse-transform approximate to Y
        xmu = np.mean(X, axis=0).reshape(1,-1)
        x_c, y_c = cca.transform(xmu, np.mean(Y, axis=0).reshape(1,-1))
        # back-project x_c to Y-space (approx) via regression using cca.x_weights_ and cca.y_loadings_
        # Simpler: map xmu via cca.x_scores -> cca.y_loadings_.T (approx)
        mapped = np.dot(x_c, cca.y_loadings_.T)
        # If shape mismatch, reduce or pad
        v = mapped.squeeze()
        if v.shape[0] != Y.shape[1]:
            # reduce or pad
            v = np.resize(v, (Y.shape[1],))
    elif method == "pls":
        k = min(n_comp, min(X.shape[0], Y.shape[1], X.shape[1]))
        pls = PLSRegression(n_components=min(k, Y.shape[1]))
        pls.fit(X, Y)
        mapped = pls.predict(np.mean(X, axis=0).reshape(1,-1)).squeeze()
        v = mapped
    elif method == "xtts_native":
        # This method computes XTTS-native vector directly averaged
        v = np.mean(Y, axis=0)
    else:
        raise ValueError("Unknown method")

    # normalize
    v = v / (np.linalg.norm(v) + 1e-12)

    if save_avg_dir is not None:
        Path(save_avg_dir).mkdir(parents=True, exist_ok=True)
        outp = Path(save_avg_dir) / f"{emotion_name}_{method}_avg.npy"
        np.save(outp, v)
        print("Saved averaged vector:", outp)
    return v

print("compute_emotion_vector_xtts_multi ready (supports cca/pls/pca/xtts_native).")'''
    ))

    # 9 - tts_with_emotion (synthesize)
    cells.append(new_code_cell(
r'''# Cell 9 — tts_with_emotion (synthesis with modified embedding)
import numpy as np
import soundfile as sf

def tts_with_emotion(tts, text, speaker_wav, emotion_vec_shared, alpha=0.7, language="hi", out_dir=OUTPUT_DIR):
    """
    Synthesizes speech using XTTS: modifies the speaker latent by adding alpha * emotion_vec_shared.
    - tts: loaded TTS instance
    - speaker_wav: path to cleaned speaker wav
    - emotion_vec_shared: 512-d vector in XTTS space
    """
    # ensure speaker_wav cleaned
    sp_clean = Path(speaker_wav)
    if not sp_clean.exists():
        raise FileNotFoundError(sp_clean)
    sp_emb = get_xtts_speaker_latent(tts, sp_clean)  # 512
    print("[EmoKnob] Extracted speaker embedding shape:", sp_emb.shape)
    v = np.array(emotion_vec_shared)
    if v.shape[0] != sp_emb.shape[0]:
        # reduce/resize v to match (try PCA reduction)
        print(f"[EmoKnob] Reducing emotion vector {v.shape} -> {sp_emb.shape} using PCA")
        p = PCA(n_components=sp_emb.shape[0])
        p.fit(v.reshape(1,-1) if v.ndim==1 else v)
        v = p.transform(v.reshape(1,-1)).squeeze()
    v = v / (np.linalg.norm(v) + 1e-12)

    emb_mod = sp_emb + alpha * v
    emb_tensor = torch.tensor(emb_mod, dtype=torch.float32).unsqueeze(0)

    # try different TTS API variants
    try:
        # some TTS versions: tts.tts returns dict with 'wav'
        out = tts.tts(text, speaker_embedding=emb_tensor, language=language)
        if isinstance(out, dict) and 'wav' in out:
            wav = np.array(out['wav'])
        elif isinstance(out, np.ndarray):
            wav = out
        else:
            # tts.tts sometimes returns dict with nested structures
            try:
                wav = np.array(out[0]['wav'])
            except Exception:
                raise RuntimeError("Cannot parse TTS output.")
    except TypeError:
        # older API: synthesizer.tts_model.inference
        cond_lat = None
        wav = tts.tts_to_file(text, speaker_wav=str(sp_clean), speaker_embedding=emb_tensor)  # fallback
        wav = None

    if wav is None:
        raise RuntimeError("Synthesis did not return waveform.")

    outp = unique_path(Path(out_dir) / f"test_hindi_emotional.wav")
    sf.write(str(outp), wav, SR)
    return outp

print("tts_with_emotion ready.")'''
    ))

    # 10 - improved GUI cell
    cells.append(new_code_cell(
r'''# Cell 10 — GUI (ipywidgets) with dropdowns, text input, alpha slider, live logs.
import ipywidgets as widgets
from IPython.display import display, clear_output

def list_emotions():
    if not EMOTION_SAMPLES_DIR.exists(): return []
    return sorted([p.name for p in EMOTION_SAMPLES_DIR.iterdir() if p.is_dir()])

def list_speakers():
    if not SPEAKERS_DIR.exists(): return []
    return sorted([str(p) for p in SPEAKERS_DIR.glob("*_clean.wav")])

emotion_dd = widgets.Dropdown(options=list_emotions(), description='Emotion:')
method_dd = widgets.Dropdown(options=['cca','pls','pca','xtts_native'], description='Method:')
vec_mode_dd = widgets.Dropdown(options=['average','single'], description='Vector:')
sample_id = widgets.IntText(value=1, description='Sample ID:')
alpha_s = widgets.FloatSlider(value=0.7, min=0.0, max=1.0, step=0.01, description='Alpha:')
speaker_dd = widgets.Dropdown(options=list_speakers(), description='Speaker:')
tts_text = widgets.Textarea(value="यह एक परीक्षण वाक्य है।", description="TTS Text:", layout=widgets.Layout(width="640px",height="80px"))
run_btn = widgets.Button(description="Generate", button_style='success')
out_box = widgets.Output(layout={'border':'1px solid black'})

def on_run_clicked(b):
    with out_box:
        clear_output()
        print("⏳ Starting generation...")
        emotion = emotion_dd.value
        method = method_dd.value
        vec_mode = vec_mode_dd.value
        sid = int(sample_id.value)
        alpha = float(alpha_s.value)
        speaker = speaker_dd.value
        text_input = tts_text.value
        print(f"[RUN] emotion: {emotion} method: {method} vec: {vec_mode} sample: {sid} alpha: {alpha}")
        try:
            # check XTTS loaded
            if 'tts' not in globals() or tts is None:
                print("❌ XTTS not loaded. Load it first (Cell 4).")
                return
            # compute emotion vector
            emotion_folder = EMOTION_SAMPLES_DIR / emotion
            if vec_mode == 'average':
                v_shared = compute_emotion_vector_xtts_multi(emotion_folder, method=method, n_comp=128,
                                                           save_single_dir=OUTPUT_SINGLE_DIR / emotion,
                                                           save_avg_dir=OUTPUT_AVG_DIR)
            else:
                v_shared = compute_emotion_vector_xtts_multi(emotion_folder, method=method, n_comp=128,
                                                           save_single_dir=OUTPUT_SINGLE_DIR / emotion,
                                                           save_avg_dir=OUTPUT_AVG_DIR,
                                                           sample_id=sid)
            if v_shared is None:
                print("❌ Could not compute emotion vector.")
                return
            print("🔊 Synthesizing audio...")
            outp = tts_with_emotion(tts, text_input, speaker, v_shared, alpha=alpha, language='hi', out_dir=OUTPUT_DIR)
            print("✅ DONE. Saved:", outp)
        except Exception as e:
            print("❌ ERROR:", e)
            import traceback; traceback.print_exc()

run_btn.on_click(on_run_clicked)

display(widgets.VBox([
    widgets.HBox([emotion_dd, method_dd, vec_mode_dd, sample_id]),
    widgets.HBox([speaker_dd, alpha_s]),
    tts_text,
    run_btn,
    out_box
]))
print("GUI ready. Select emotion, method, speaker, enter text, and click Generate.")'''
    ))

    # 11 - Final notes cell
    cells.append(new_markdown_cell(
r'''## Notes & next steps
1. Make sure you run the XTTS loader cell (Cell 4) to populate `tts` before using the GUI.
2. If the Indic wav2vec model is gated, place it under `models/ai4bharat_indicwav2vec_hindi` and the notebook will load it locally.
3. If XTTS loading fails due to config mismatches, cell 4 tries to patch `config.json`. If you want me to patch automatically, allow the script to overwrite your config (it already does this).
4. After running generation, check `data/outputs` for saved vectors and generated wav files.
5. If you prefer a slightly different GUI layout or additional file pickers, tell me and I’ll adjust.
'''
    ))

    return cells


# Create notebook with proper kernelspec metadata
cells = make_cells()
nb = new_notebook(cells=cells, metadata={
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3"
    },
    "language_info": {
        "name": "python",
        "version": sys.version.split()[0]
    }
})

# Write file
OUT_NOTEBOOK.parent.mkdir(parents=True, exist_ok=True)
with open(OUT_NOTEBOOK, "w", encoding="utf-8") as f:
    nbformat.write(nb, f)

print("Notebook generated at:", OUT_NOTEBOOK)
print("Open it in VS Code or Jupyter, ensure kernel points to your project's .venv Python, then run top->bottom.")
