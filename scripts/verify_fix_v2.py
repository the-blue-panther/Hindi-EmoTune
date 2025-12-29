import os
import sys
import torch
import numpy as np
import librosa
from pathlib import Path
from TTS.api import TTS

# Setup paths
PROJECT_ROOT = Path(r"D:\Downloads\Bengali_EmoKnob")
MODELS_DIR = PROJECT_ROOT / "models"
XTTS_LOCAL_DIR = MODELS_DIR / "xtts_v2"
SR_XTTS = 22050
SR_INDIC = 16000

# --- MOCK / COPIED FUNCTIONS ---

def resolve_xtts_internal_model(tts_obj):
    if hasattr(tts_obj, 'synthesizer') and hasattr(tts_obj.synthesizer, 'tts_model'):
        return tts_obj.synthesizer.tts_model
    if hasattr(tts_obj, 'tts_model'):
        return tts_obj.tts_model
    raise RuntimeError('Could not resolve internal XTTS model.')

def get_xtts_speaker_latent(tts_obj, wav_path, load_sr=SR_XTTS):
    # Patched version
    model = resolve_xtts_internal_model(tts_obj)
    try:
        res = model.get_conditioning_latents(str(wav_path), load_sr=load_sr)
    except TypeError:
        res = model.get_conditioning_latents(str(wav_path))
    
    if isinstance(res, (list, tuple)) and len(res) >= 1:
        gpt_cond = res[0]
    else:
        raise RuntimeError(f"Unexpected return: {type(res)}")

    try:
        sp = gpt_cond.reshape(-1)
        return sp.detach().cpu().numpy() if hasattr(sp, 'detach') else np.array(sp)
    except Exception as e:
        raise RuntimeError('Failed to convert GPT latent to numpy: ' + str(e))

def get_indic_embedding(wav_path, sr_source=SR_XTTS, sr_indic=SR_INDIC):
    # Dummy implementation for verification (we don't need actual Indic embeddings for xtts_native)
    # But the function calls it. We'll return a random vector.
    return np.random.rand(768).astype(np.float32)

def preprocess_audio(in_path, out_path, sr):
    # Dummy
    pass

def compute_emotion_vector_xtts_multi(emotion_dir, method='cca', n_comp=32, mode='average', sample_id=1,
                                      save_single_dir=None, save_avg_dir=None):
    # --- COPIED FROM PATCH V2 ---
    emotion_dir = Path(emotion_dir)
    sample_dirs = [d for d in sorted(emotion_dir.iterdir()) if d.is_dir()]
    if len(sample_dirs) == 0:
        raise ValueError('No sample subfolders found')

    X = []
    Y = []
    single_vectors = []

    for sd in sample_dirs:
        n_clean = sd / 'neutral_clean.wav'
        e_clean = sd / f'{emotion_dir.name}_clean.wav'
        
        # Assume files exist for this test
        if not n_clean.exists() or not e_clean.exists():
            print(f"Skipping {sd.name} (files not found)")
            continue

        xi = get_indic_embedding(n_clean)
        xe = get_indic_embedding(e_clean)
        yi = get_xtts_speaker_latent(tts, n_clean)
        ye = get_xtts_speaker_latent(tts, e_clean)

        X.append(xe - xi)
        Y.append(ye - yi)
        single_vectors.append((sd.name, xe - xi, ye - yi))

    if len(X) == 0:
        raise ValueError('No matched pairs extracted')

    X = np.stack(X)
    Y = np.stack(Y)

    # Check for high-dimensional latent
    if Y.shape[1] > 1024:
        if method != 'xtts_native':
            print(f'⚠️ High-dimensional latent detected ({Y.shape[1]} dims). Forcing "xtts_native" method.')
            method = 'xtts_native'

    if method == 'xtts_native':
        avg = np.mean([v for (_,_,v) in single_vectors], axis=0)
        print(f"✓ Emotion delta computed. Shape: {avg.shape}")
        return avg

    # Fallback (should not be reached if Y is high dim)
    raise RuntimeError("Should have switched to xtts_native!")

# --- MAIN ---

print("Loading XTTS...")
try:
    tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", gpu=False)
except Exception as e:
    print(f"Failed to load XTTS: {e}")
    sys.exit(1)

emotion_dir = PROJECT_ROOT / "data" / "emotion_samples" / "happy"

print("Testing compute_emotion_vector_xtts_multi with method='cca' (should force switch)...")
try:
    vec = compute_emotion_vector_xtts_multi(emotion_dir, method='cca')
    print(f"Result vector shape: {vec.shape}")
    if vec.shape[0] == 32768:
        print("SUCCESS: Vector has correct high dimension.")
    else:
        print(f"FAILURE: Vector has wrong dimension {vec.shape}")
except Exception as e:
    print(f"FAILURE: Exception occurred: {e}")
    import traceback
    traceback.print_exc()
