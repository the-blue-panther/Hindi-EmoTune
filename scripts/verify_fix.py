import os
import sys
import torch
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from TTS.api import TTS

# Setup paths
PROJECT_ROOT = Path(r"D:\Downloads\Bengali_EmoKnob")
MODELS_DIR = PROJECT_ROOT / "models"
XTTS_LOCAL_DIR = MODELS_DIR / "xtts_v2"
OUTPUT_GEN_DIR = PROJECT_ROOT / "data" / "outputs" / "generated"
OUTPUT_GEN_DIR.mkdir(parents=True, exist_ok=True)

SR_XTTS = 22050

# --- COPIED NEW FUNCTIONS ---

def resolve_xtts_internal_model(tts_obj):
    if tts_obj is None:
        raise RuntimeError('Provided tts_obj is None')
    if hasattr(tts_obj, 'synthesizer') and hasattr(tts_obj.synthesizer, 'tts_model'):
        return tts_obj.synthesizer.tts_model
    if hasattr(tts_obj, 'tts_model'):
        return tts_obj.tts_model
    raise RuntimeError('Could not resolve internal XTTS model.')

def get_xtts_speaker_latent(tts_obj, wav_path, load_sr=SR_XTTS):
    '''Extract GPT conditioning latent from XTTS (prosody/emotion).
    Returns 1D numpy array (flattened). Original shape is [1, 32, 1024].'''
    model = resolve_xtts_internal_model(tts_obj)
    try:
        res = model.get_conditioning_latents(str(wav_path), load_sr=load_sr)
    except TypeError:
        res = model.get_conditioning_latents(str(wav_path))
    
    if isinstance(res, (list, tuple)) and len(res) >= 1:
        gpt_cond = res[0]
    else:
        raise RuntimeError(f"Unexpected return from get_conditioning_latents: {type(res)}")

    try:
        sp = gpt_cond.reshape(-1)
        return sp.detach().cpu().numpy() if hasattr(sp, 'detach') else np.array(sp)
    except Exception as e:
        raise RuntimeError('Failed to convert GPT latent to numpy: ' + str(e))

def unique_path(path: Path):
    path = Path(path)
    if not path.exists(): return path
    base = path.stem
    suf = path.suffix
    parent = path.parent
    i = 1
    while True:
        candidate = parent / f"{base}_{i}{suf}"
        if not candidate.exists(): return candidate
        i += 1

def apply_emotion_and_synthesize(XTTS, text, speaker_wav, emotion_vec, alpha=0.1, out_path=None, language='hi', scale_to_speaker=True):
    if out_path is None:
        out_path = OUTPUT_GEN_DIR / 'test_hindi_emotional.wav'
    out_path = unique_path(Path(out_path))

    model = resolve_xtts_internal_model(XTTS)
    try:
        latents = model.get_conditioning_latents(str(speaker_wav), load_sr=SR_XTTS)
    except TypeError:
        latents = model.get_conditioning_latents(str(speaker_wav))
        
    gpt_cond, speaker_emb = latents[0], latents[1]
    
    ev = np.asarray(emotion_vec).astype(np.float32)
    target_size = 32 * 1024
    
    if ev.size != target_size:
        print(f"Warning: Emotion vector size {ev.size} != target {target_size}. Resizing/Padding...")
        if ev.size > target_size:
            ev = ev[:target_size]
        else:
            ev = np.pad(ev, (0, target_size - ev.size))
            
    ev_tensor = torch.tensor(ev).reshape(1, 32, 1024).to(gpt_cond.device)

    if scale_to_speaker:
        base_norm = torch.norm(gpt_cond)
        ev_norm = torch.norm(ev_tensor)
        print(f"[Emotion Scaling] Base Norm={base_norm:.4f}, Delta Norm={ev_norm:.4f}")

    new_gpt_cond = gpt_cond + alpha * ev_tensor
    
    original_get_cond = model.get_conditioning_latents
    
    def patched_get_cond(*args, **kwargs):
        return (new_gpt_cond, speaker_emb)
    
    try:
        model.get_conditioning_latents = patched_get_cond
        XTTS.tts_to_file(text=text, speaker_wav=str(speaker_wav), language=language, file_path=str(out_path))
        print(f'✓ Synthesis complete -> {out_path}')
        return out_path
    except Exception as e:
        print(f'Synthesis failed: {e}')
        raise
    finally:
        model.get_conditioning_latents = original_get_cond

# --- MAIN EXECUTION ---

print("Loading XTTS...")
try:
    tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", gpu=False)
except Exception as e:
    print(f"Failed to load XTTS: {e}")
    sys.exit(1)

# Load samples
sample_dir = PROJECT_ROOT / "data" / "emotion_samples" / "happy" / "sample001"
neutral_wav = sample_dir / "neutral_clean.wav"
happy_wav = sample_dir / "happy_clean.wav"

if not neutral_wav.exists() or not happy_wav.exists():
    print("Samples not found")
    sys.exit(1)

print("Computing emotion vector (Happy - Neutral)...")
vn = get_xtts_speaker_latent(tts, neutral_wav)
vh = get_xtts_speaker_latent(tts, happy_wav)
emotion_vec = vh - vn

print(f"Emotion vector shape: {emotion_vec.shape}")

print("Synthesizing test audio...")
text = "नमस्ते, आप कैसे हैं? मुझे बहुत खुशी हो रही है।"
out = apply_emotion_and_synthesize(tts, text, neutral_wav, emotion_vec, alpha=0.5, language='hi') # High alpha for test

if out and Path(out).exists():
    print("Verification SUCCESS!")
else:
    print("Verification FAILED!")
