"""
generate_notebook_v10.py

Generates `hindi_emoknob_demo_v10.ipynb` in the project root:
D:\Downloads\Bengali_EmoKnob\hindi_emoknob_demo_v10.ipynb

Designed for Option C:
 - 3 modes: 'CCA', 'PLS', 'XTTS Native'
 - GUI controls (ipywidgets) + manual cells for advanced users

Run this script from the project root (inside your .venv):
    python scripts\generate_notebook_v10.py
"""
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell
from pathlib import Path
import textwrap, os, sys

PROJECT = Path(r"D:\Downloads\Bengali_EmoKnob")
SCRIPT_OUT = PROJECT / "hindi_emoknob_demo_v10.ipynb"

# Notebook cells list
cells = []

# Title cell
cells.append(new_markdown_cell(
"# Hindi EmoKnob — v10\n\n"
"**Modes:** CCA | PLS | XTTS-native\n\n"
"**Option C GUI**: GUI panel + manual controls. Run top→bottom after restarting kernel."
))

# Imports & paths
cells.append(new_code_cell(textwrap.dedent(f"""
# Imports & Paths
from pathlib import Path
import os, sys, time, json
import numpy as np
import joblib
import torch
import librosa
import soundfile as sf
import subprocess
from sklearn.cross_decomposition import CCA, PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from TTS.api import TTS

PROJECT = Path(r"{PROJECT}")
MODELS_DIR = PROJECT / "models"
DATA_DIR = PROJECT / "data"
EMOTION_SAMPLES_DIR = DATA_DIR / "emotion_samples"
SPEAKERS_DIR = DATA_DIR / "speakers"
OUTPUT_DIR = DATA_DIR / "outputs"
OUTPUT_VECTORS_DIR = OUTPUT_DIR / "emotion_vectors"
OUTPUT_SINGLE_DIR = OUTPUT_VECTORS_DIR / "single"
OUTPUT_AVG_DIR = OUTPUT_VECTORS_DIR / "average"
ALIGNMENTS_DIR = MODELS_DIR / "alignments"

for p in [OUTPUT_DIR, OUTPUT_SINGLE_DIR, OUTPUT_AVG_DIR, ALIGNMENTS_DIR]:
    os.makedirs(p, exist_ok=True)

print("Project:", PROJECT)
print("Models dir:", MODELS_DIR)
print("Emotion samples dir:", EMOTION_SAMPLES_DIR)
print("Speakers dir:", SPEAKERS_DIR)
print("Outputs dir:", OUTPUT_DIR)
""")))

# Model check & load
cells.append(new_code_cell(textwrap.dedent("""
# Model check & load (adjust if your model folders differ)
INDIC_DIR = MODELS_DIR / "ai4bharat_indicwav2vec_hindi"
XTTS_DIR   = MODELS_DIR / "xtts_hindi_finetuned"

print("Indic exists:", INDIC_DIR.exists())
print("XTTS exists:", XTTS_DIR.exists())

# Load Indic encoder (transformers) - expects local folder
try:
    from transformers import Wav2Vec2Processor, Wav2Vec2Model
    print("Loading IndicWav2Vec...")
    processor = Wav2Vec2Processor.from_pretrained(str(INDIC_DIR))
    indic_encoder = Wav2Vec2Model.from_pretrained(str(INDIC_DIR)).to("cpu")
    print("Indic encoder loaded.")
except Exception as e:
    print("Indic model load error:", e)
    indic_encoder = None
    processor = None

# Load XTTS (TTS API) - expects local model files inside XTTS_DIR
try:
    model_path = str(XTTS_DIR / "model.pth")
    config_path = str(XTTS_DIR / "config.json")
    print("Loading XTTS (this may be heavy)...")
    tts = TTS(model_path=model_path, config_path=config_path, gpu=torch.cuda.is_available())
    print("XTTS loaded.")
except Exception as e:
    print("XTTS load error:", e)
    tts = None
""")))

# Preprocessing utilities
cells.append(new_code_cell(textwrap.dedent("""
# Audio preprocessing utilities
import noisereduce as nr
import numpy as np
import soundfile as sf
import subprocess
from pathlib import Path

SR = 16000

def convert_to_wav(in_path, out_path, sr=SR):
    in_path = str(in_path)
    out_path = str(out_path)
    cmd = ["ffmpeg", "-y", "-i", in_path, "-ar", str(sr), "-ac", "1", out_path]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return out_path

def preprocess_audio(in_path, out_path, sr=SR, nr_reduce=True):
    in_path = Path(in_path)
    out_path = Path(out_path)
    tmp = out_path.with_suffix(".tmp.wav")
    convert_to_wav(in_path, tmp, sr=sr)
    wav, _ = librosa.load(str(tmp), sr=sr, mono=True)
    wav = wav - np.mean(wav)
    if nr_reduce:
        try:
            wav = nr.reduce_noise(y=wav, sr=sr)
        except Exception as e:
            print("noisereduce failed:", e)
    if np.max(np.abs(wav)) > 0:
        wav = wav / np.max(np.abs(wav))
    sf.write(str(out_path), wav, sr)
    try:
        tmp.unlink()
    except:
        pass
    return out_path

print("Audio preprocess utilities ready.")
""")))

# Embedding extraction
cells.append(new_code_cell(textwrap.dedent("""
# Embedding extraction helpers

def get_indic_embedding(wav_path, sr=SR):
    if processor is None or indic_encoder is None:
        raise RuntimeError("Indic encoder not loaded.")
    wav, _ = librosa.load(wav_path, sr=sr, mono=True)
    inputs = processor(wav, sampling_rate=sr, return_tensors="pt", padding=True)
    with torch.no_grad():
        out = indic_encoder(**inputs).last_hidden_state
    emb = out.mean(dim=1).squeeze().cpu().numpy()
    return emb

def get_xtts_embedding_from_wav(wav_path, sr=SR):
    if tts is None:
        raise RuntimeError("XTTS not loaded.")
    # Try common speaker extraction APIs, multiple fallbacks
    try:
        # Newer TTS versions may have speaker_manager with compute_embedding_from_clip
        emb = tts.synthesizer.tts_model.speaker_manager.compute_embedding_from_clip(str(wav_path))
        return np.asarray(emb).squeeze()
    except Exception:
        pass
    try:
        wav, _ = librosa.load(wav_path, sr=sr, mono=True)
        tensor = torch.tensor(wav).unsqueeze(0)
        emb = tts.synthesizer.tts_model.compute_speaker_embedding(tensor)
        return emb.squeeze().cpu().numpy()
    except Exception:
        pass
    raise RuntimeError("Could not obtain XTTS speaker embedding from this TTS release.")
""")))

# Alignment functions (CCA/PLS)
cells.append(new_code_cell(textwrap.dedent("""
# Alignment: fit CCA or PLS
from sklearn.cross_decomposition import CCA, PLSRegression
from sklearn.preprocessing import StandardScaler

def fit_alignment(X, Y, method='cca', n_comp=128):
    X = np.asarray(X); Y = np.asarray(Y)
    assert X.shape[0] == Y.shape[0], "X and Y must have same number of rows."
    sx = StandardScaler().fit(X)
    sy = StandardScaler().fit(Y)
    Xs = sx.transform(X); Ys = sy.transform(Y)
    if method.lower() == 'cca':
        cca = CCA(n_components=min(n_comp, Xs.shape[1], Ys.shape[1]), max_iter=1000)
        cca.fit(Xs, Ys)
        model = cca
    elif method.lower() == 'pls':
        pls = PLSRegression(n_components=min(n_comp, Xs.shape[1], Ys.shape[1]))
        pls.fit(Xs, Ys)
        model = pls
    else:
        raise ValueError("Unknown method")
    return model, sx, sy
""")))

# Compute emotion vectors multi-sample (Option A pipeline)
cells.append(new_code_cell(textwrap.dedent("""
# Compute per-emotion vectors from samples (supports averaged + single)
from pathlib import Path
import joblib

def compute_emotion_vector_xtts_multi(emotion_dir, method='cca', n_comp=128,
                                      save_single_dir=None, save_avg_dir=None, sample_id=None):
    emotion_dir = Path(emotion_dir)
    emotion_name = emotion_dir.name
    sample_dirs = sorted([p for p in emotion_dir.iterdir() if p.is_dir()])
    if len(sample_dirs) == 0:
        print("[WARN] No sample subfolders found in", emotion_dir)
        return None

    X_indic = []
    Y_xtts = []
    pairs = []

    for s in sample_dirs:
        neutral = None; emot = None
        for f in s.iterdir():
            if f.is_file() and f.suffix.lower() in ['.wav','.mp3','.flac','.m4a']:
                name = f.stem.lower()
                if 'neutral' in name:
                    neutral = f
                if emotion_name.lower() in name:
                    emot = f
        if neutral and emot:
            n_clean = s / (neutral.stem + "_clean.wav")
            e_clean = s / (emot.stem + "_clean.wav")
            if not n_clean.exists(): preprocess_audio(neutral, n_clean)
            if not e_clean.exists(): preprocess_audio(emot, e_clean)
            xi = get_indic_embedding(str(n_clean))
            xe = get_indic_embedding(str(e_clean))
            yi = get_xtts_embedding_from_wav(str(n_clean))
            ye = get_xtts_embedding_from_wav(str(e_clean))
            # store neutral & emotional rows as matched rows
            X_indic.append(xi); Y_xtts.append(yi)
            X_indic.append(xe); Y_xtts.append(ye)
            pairs.append((str(n_clean), str(e_clean)))
    X = np.vstack(X_indic); Y = np.vstack(Y_xtts)
    print("[INFO] Collected rows:", X.shape, Y.shape)

    # Fit alignment
    proj, sx, sy = fit_alignment(X, Y, method=method, n_comp=n_comp)
    joblib.dump({'proj':proj, 'scaler_x':sx, 'scaler_y':sy},
                ALIGNMENTS_DIR / f"{emotion_name}_{method}_{n_comp}.joblib")
    print("[INFO] Saved alignment:", ALIGNMENTS_DIR / f"{emotion_name}_{method}_{n_comp}.joblib")

    # compute shared-space directions per sample
    directions = []
    for (n_clean, e_clean) in pairs:
        xi = get_indic_embedding(n_clean)
        xe = get_indic_embedding(e_clean)
        xi_s = sx.transform(xi.reshape(1,-1)); xe_s = sx.transform(xe.reshape(1,-1))
        if method=='cca':
            xi_t, yi_t = proj.transform(xi_s, np.zeros_like(xi_s))
            xe_t, ye_t = proj.transform(xe_s, np.zeros_like(xe_s))
            d = xe_t.squeeze() - xi_t.squeeze()
        else:
            # for PLS use transform (returns scores)
            xt = proj.transform(xi_s)
            et = proj.transform(xe_s)
            d = et.squeeze() - xt.squeeze()
        directions.append(d)

    # save singles
    if save_single_dir:
        Path(save_single_dir).mkdir(parents=True, exist_ok=True)
        for i,d in enumerate(directions, start=1):
            outp = Path(save_single_dir) / f"{emotion_name}_sample{i:03d}_{method}.npy"
            np.save(outp, d)
            print("Saved single:", outp)

    avg = np.mean(np.vstack(directions), axis=0)
    avg = avg / (np.linalg.norm(avg) + 1e-12)
    if save_avg_dir:
        Path(save_avg_dir).mkdir(parents=True, exist_ok=True)
        outp = Path(save_avg_dir) / f"{emotion_name}_avg_{method}.npy"
        np.save(outp, avg)
        print("Saved averaged:", outp)

    # if sample_id requested, return that sample (1-indexed)
    if sample_id:
        idx = sample_id - 1
        if idx < 0 or idx >= len(directions):
            raise ValueError("sample_id out of range.")
        return directions[idx]
    return avg
""")))

# XTTS-native vector extraction
cells.append(new_code_cell(textwrap.dedent("""
# XTTS-native emotion vector: v = emb_emotion_xtts - emb_neutral_xtts
def compute_xtts_native_vector(neutral_wav, emotion_wav, save_to=None):
    # expects cleaned wavs
    emb_n = get_xtts_embedding_from_wav(neutral_wav)
    emb_e = get_xtts_embedding_from_wav(emotion_wav)
    v = emb_e - emb_n
    v = v / (np.linalg.norm(v) + 1e-12)
    if save_to:
        np.save(save_to, v)
    return v
""")))

# tts_with_emotion (map shared -> xtts approx, inject)
cells.append(new_code_cell(textwrap.dedent("""
# Inference: inject shared-space vector into XTTS speaker embedding and synthesize
import numpy as np
import soundfile as sf
from pathlib import Path

def unique_path(path):
    p = Path(path)
    if not p.exists():
        return p
    base = p.stem; suf = p.suffix; parent = p.parent
    i = 1
    while True:
        cand = parent / f"{base}_{i}{suf}"
        if not cand.exists(): return cand
        i += 1

def tts_with_emotion(text, speaker_wav, mode, emotion_vec_shared, alpha=0.7, alignment_joblib=None, out_path=None):
    # 1) prep speaker embedding
    sp_emb = get_xtts_embedding_from_wav(speaker_wav).astype(np.float32)
    # 2) load alignment object
    if alignment_joblib:
        alignment_path = Path(alignment_joblib)
    else:
        candidates = sorted(ALIGNMENTS_DIR.glob(f"*_{mode}_*.joblib"))
        if not candidates:
            raise FileNotFoundError("No alignment saved. Run compute_emotion_vector_xtts_multi first.")
        alignment_path = candidates[-1]
    data = joblib.load(alignment_path)
    proj = data['proj']; scaler_x = data['scaler_x']; scaler_y = data['scaler_y']
    # 3) project speaker into shared space
    sp_std = scaler_y.transform(sp_emb.reshape(1,-1))
    if mode=='cca':
        # proj.transform(Xs, Ys) → returns Xc, Yc
        try:
            dummy = np.zeros((1, scaler_x.mean_.shape[0]))
            x_shared, y_shared = proj.transform(dummy, sp_std)
            speaker_shared = y_shared.squeeze()
        except Exception:
            if hasattr(proj, "y_weights_"):
                speaker_shared = sp_std.dot(proj.y_weights_[:, :proj.n_components]).squeeze()
            else:
                raise RuntimeError("Cannot get speaker_shared from CCA.")
    else:
        # PLS
        try:
            speaker_shared = proj.transform(sp_std).squeeze()
        except Exception as e:
            raise RuntimeError("PLS transform failed: " + str(e))

    # 4) add emotion vector
    emotion_vec_shared = np.asarray(emotion_vec_shared).astype(np.float32)
    if emotion_vec_shared.shape[0] != speaker_shared.shape[0]:
        raise ValueError("Dimension mismatch shared dims.")
    new_shared = speaker_shared + alpha * emotion_vec_shared

    # 5) approximate inverse -> modified XTTS embedding
    if mode=='cca':
        if hasattr(proj, "y_weights_"):
            W = proj.y_weights_[:, :proj.n_components]
            pinv = np.linalg.pinv(W)
            recon = pinv.T.dot(new_shared)
            recon_orig = scaler_y.inverse_transform(recon.reshape(1,-1)).squeeze()
            modified_xtts = recon_orig.astype(np.float32)
        else:
            raise RuntimeError("CCA has no y_weights_; cannot invert.")
    else:
        if hasattr(proj, "y_loadings_"):
            Yload = proj.y_loadings_[:, :proj.n_components]
            pinvY = np.linalg.pinv(Yload)
            recon = pinvY.T.dot(new_shared)
            recon_orig = scaler_y.inverse_transform(recon.reshape(1,-1)).squeeze()
            modified_xtts = recon_orig.astype(np.float32)
        else:
            raise RuntimeError("PLS missing y_loadings_")

    # 6) synthesize using tts API (many TTS versions accept speaker_embedding)
    emb_tensor = torch.tensor(modified_xtts).unsqueeze(0)
    try:
        out = tts.tts(text=text, speaker_embedding=emb_tensor, language="hi")
        if isinstance(out, dict) and "wav" in out:
            wav_arr = np.asarray(out["wav"])
        elif isinstance(out, np.ndarray):
            wav_arr = out
        else:
            wav_arr = np.asarray(out[0])
    except Exception as e:
        raise RuntimeError("TTS generation failed: " + str(e))

    # 7) save uniquely
    if out_path is None:
        out_path = OUTPUT_DIR / "test_hindi_emotional.wav"
    out_path = unique_path(Path(out_path))
    sf.write(str(out_path), wav_arr, SR)
    print("[SAVED]", out_path)
    return out_path
""")))

# GUI cell (ipywidgets)
cells.append(new_code_cell(textwrap.dedent("""
# GUI panel (ipywidgets). Requires ipywidgets installed and notebook front-end.
import ipywidgets as widgets
from IPython.display import display, clear_output
from pathlib import Path

def list_emotions():
    if not EMOTION_SAMPLES_DIR.exists(): return []
    return sorted([p.name for p in EMOTION_SAMPLES_DIR.iterdir() if p.is_dir()])

def list_samples_for(emotion):
    d = EMOTION_SAMPLES_DIR / emotion
    if not d.exists(): return []
    return [p.name for p in sorted(d.iterdir()) if p.is_dir()]

# Widgets
emotion_dd = widgets.Dropdown(options=list_emotions(), description='Emotion:')
mode_dd = widgets.Dropdown(options=['cca','pls','xtts_native'], description='Mode:')
sample_mode_dd = widgets.Dropdown(options=['average','single'], description='Vector:')
sample_id_dd = widgets.IntText(value=1, description='Sample ID:')
alpha_s = widgets.FloatSlider(value=0.7, min=0.0, max=2.0, step=0.05, description='Alpha:')
speaker_text = widgets.Text(value=str(SPEAKERS_DIR / "character_1_clean.wav"), description='Speaker WAV:')
run_btn = widgets.Button(description='Generate', button_style='success')
out_box = widgets.Output(layout={'border':'1px solid black'})

def on_emotion_change(change):
    sample_id_dd.value = 1

def on_run_clicked(b):
    with out_box:
        clear_output()
        emotion = emotion_dd.value
        mode = mode_dd.value
        vec_mode = sample_mode_dd.value
        sid = int(sample_id_dd.value)
        alpha = float(alpha_s.value)
        speaker = Path(speaker_text.value)
        print("[RUN] emotion:", emotion, "mode:", mode, "vec:", vec_mode, "sample:", sid, "alpha:", alpha)
        if mode == 'xtts_native':
            # locate a single sample or average? xtts_native must use a sample
            if vec_mode == 'average':
                print("XTTS-native average: will compute per-sample native vectors and average.")
            # compute
            emotion_dir = EMOTION_SAMPLES_DIR / emotion
            if vec_mode == 'single':
                # find the sample folder
                sample_folders = sorted([p for p in emotion_dir.iterdir() if p.is_dir()])
                if sid-1 < 0 or sid-1 >= len(sample_folders):
                    print("Sample id out of range.")
                    return
                nf = sample_folders[sid-1] / (list(sample_folders[sid-1].glob('*neutral*'))[0].stem + "_clean.wav")
                ef = sample_folders[sid-1] / (list(sample_folders[sid-1].glob(f'*{emotion}*'))[0].stem + "_clean.wav")
                v = compute_xtts_native_vector(str(nf), str(ef))
            else:
                # average xtts-native across samples
                vs = []
                for s in sorted([p for p in emotion_dir.iterdir() if p.is_dir()]):
                    n = list(s.glob('*neutral*'))[0]; e = list(s.glob(f'*{emotion}*'))[0]
                    n_clean = s / (n.stem + "_clean.wav"); e_clean = s / (e.stem + "_clean.wav")
                    if not n_clean.exists(): preprocess_audio(n, n_clean)
                    if not e_clean.exists(): preprocess_audio(e, e_clean)
                    vs.append(compute_xtts_native_vector(str(n_clean), str(e_clean)))
                v = np.mean(np.vstack(vs), axis=0)
                v = v / (np.linalg.norm(v) + 1e-12)
            # now call tts_with_emotion with mode=cca or pls? for xtts_native we directly have 512d vector
            # We'll call a special wrapper that injects the native vector into speaker embedding directly
            try:
                # direct injection: get speaker, speaker xtts emb, do emb + alpha*v, synthesize using tts API directly
                sp_emb = get_xtts_embedding_from_wav(str(speaker))
                emb_mod = sp_emb + alpha * v
                emb_tensor = torch.tensor(emb_mod).unsqueeze(0)
                out = tts.tts(text="यह एक परीक्षण वाक्य है।", speaker_embedding=emb_tensor, language="hi")
                if isinstance(out, dict) and 'wav' in out: wav_arr = np.asarray(out['wav'])
                elif isinstance(out, np.ndarray): wav_arr = out
                else: wav_arr = np.asarray(out[0])
                target = unique_path(OUTPUT_DIR / "test_hindi_emotional_native.wav")
                sf.write(str(target), wav_arr, SR)
                print("[SAVED native]", target)
            except Exception as e:
                print("XTTS-native generation failed:", e)
            return

        # For 'cca' or 'pls' path: compute or load averaged/single shared-space vector
        emotion_dir = EMOTION_SAMPLES_DIR / emotion
        if vec_mode == 'average':
            out_avg = OUTPUT_AVG_DIR
            v_shared = compute_emotion_vector_xtts_multi(emotion_dir, method=mode, n_comp=128,
                                                       save_single_dir=OUTPUT_SINGLE_DIR / emotion,
                                                       save_avg_dir=out_avg)
        else:
            # single sample
            v_shared = compute_emotion_vector_xtts_multi(emotion_dir, method=mode, n_comp=128,
                                                         save_single_dir=OUTPUT_SINGLE_DIR / emotion,
                                                         save_avg_dir=OUTPUT_AVG_DIR,
                                                         sample_id=sid)
        if v_shared is None:
            print("Vector computation returned None.")
            return
        # Synthesize
        try:
            outp = tts_with_emotion(text="यह एक परीक्षण वाक्य है।", speaker_wav=str(speaker),
                                    mode=mode, emotion_vec_shared=v_shared, alpha=alpha)
            print("Generated:", outp)
        except Exception as e:
            print("Synthesis error:", e)

run_btn.on_click(on_run_clicked)
sample_mode_dd.observe(on_emotion_change, names='value')

ui = widgets.VBox([
    widgets.HBox([emotion_dd, mode_dd, sample_mode_dd, sample_id_dd]),
    widgets.HBox([alpha_s, speaker_text]),
    run_btn,
    out_box
])
display(ui)
print("GUI ready. Use the panel to run the pipeline interactively.")
""")))

# Manual usage example cell
cells.append(new_markdown_cell("## Manual usage (advanced)\n"
"Below are manual examples if you prefer to run pipeline steps yourself instead of using the GUI."))

cells.append(new_code_cell(textwrap.dedent("""
# EXAMPLE MANUAL STEPS (edit/execute)
# 1) compute averaged CCA vector for 'happy'
vec = compute_emotion_vector_xtts_multi(EMOTION_SAMPLES_DIR / "happy", method='cca', n_comp=128,
                                        save_single_dir=OUTPUT_SINGLE_DIR / "happy",
                                        save_avg_dir=OUTPUT_AVG_DIR)
print("vec shape:", None if vec is None else vec.shape)

# 2) synthesize using that averaged vector
outp = tts_with_emotion(text="मैं आज बहुत खुश हूँ।", speaker_wav=str(SPEAKERS_DIR / "character_1_clean.wav"),
                        mode='cca', emotion_vec_shared=vec, alpha=0.7)
print("Saved:", outp)
""")))

# Finish: write notebook
# Finish: write notebook
nb = new_notebook(
    cells=cells,
    metadata={
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": sys.version.split()[0]
        }
    }
)

os.makedirs(PROJECT, exist_ok=True)
with open(SCRIPT_OUT, "w", encoding="utf-8") as f:
    nbformat.write(nb, f)

print("Notebook generated:", SCRIPT_OUT)
print("Open it in VS Code or Jupyter, restart kernel, then run cells top → bottom.")
