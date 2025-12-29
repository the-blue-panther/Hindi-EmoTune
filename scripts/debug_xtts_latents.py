import os
import sys
import torch
import numpy as np
from pathlib import Path
from TTS.api import TTS

# Setup paths
PROJECT_ROOT = Path(r"D:\Downloads\Bengali_EmoKnob")
MODELS_DIR = PROJECT_ROOT / "models"
XTTS_LOCAL_DIR = MODELS_DIR / "xtts_v2"

# Load XTTS
print("Loading XTTS...")
try:
    # Try loading via model_name which handles cache automatically
    print("Attempting to load via model_name...")
    tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", gpu=False)
    print("XTTS loaded via model_name.")
except Exception as e:
    print(f"Failed to load XTTS via model_name: {e}")
    # Fallback to local path if model_name fails (unlikely if cache exists)
    try:
        print("Attempting to load via local path...")
        model_path = XTTS_LOCAL_DIR / "model.pth"
        config_path = XTTS_LOCAL_DIR / "config.json"
        if model_path.exists():
             tts = TTS(model_path=str(model_path), config_path=str(config_path), gpu=False)
             print("XTTS loaded via local path.")
        else:
             raise FileNotFoundError("Local model.pth not found")
    except Exception as e2:
        print(f"Failed to load XTTS via local path: {e2}")
        sys.exit(1)

model = tts.synthesizer.tts_model

# Define paths to samples
sample_dir = PROJECT_ROOT / "data" / "emotion_samples" / "happy" / "sample001"
neutral_wav = sample_dir / "neutral_clean.wav"
happy_wav = sample_dir / "happy_clean.wav"

if not neutral_wav.exists() or not happy_wav.exists():
    print("Sample files not found.")
    sys.exit(1)

print(f"Analyzing samples:\n  Neutral: {neutral_wav}\n  Happy:   {happy_wav}")

# Get latents
def get_latents(wav_path):
    # XTTS get_conditioning_latents returns (gpt_cond_latent, speaker_embedding)
    # or sometimes just speaker_embedding if gpt_cond_len is 0? No, usually tuple.
    res = model.get_conditioning_latents(str(wav_path))
    return res

lat_n = get_latents(neutral_wav)
lat_h = get_latents(happy_wav)

# Inspect results
print("\n--- Latent Inspection ---")
if isinstance(lat_n, (list, tuple)):
    print(f"Result is a tuple of length {len(lat_n)}")
    for i, item in enumerate(lat_n):
        if isinstance(item, torch.Tensor):
            print(f"  Item {i}: Tensor shape {item.shape}, dtype {item.dtype}")
            print(f"    Mean: {item.mean():.4f}, Std: {item.std():.4f}, Norm: {item.norm():.4f}")
        else:
            print(f"  Item {i}: Type {type(item)}")
else:
    print(f"Result is {type(lat_n)}")
    if isinstance(lat_n, torch.Tensor):
        print(f"  Shape {lat_n.shape}")

# Compare Neutral vs Happy
print("\n--- Comparison (Neutral vs Happy) ---")
if isinstance(lat_n, (list, tuple)) and len(lat_n) >= 2:
    gpt_n, spk_n = lat_n[0], lat_n[1]
    gpt_h, spk_h = lat_h[0], lat_h[1]
    
    # GPT Latent Difference
    if gpt_n.shape == gpt_h.shape:
        diff_gpt = gpt_h - gpt_n
        dist_gpt = torch.norm(diff_gpt).item()
        rel_dist_gpt = dist_gpt / (torch.norm(gpt_n).item() + 1e-6)
        print(f"GPT Latent (Item 0) L2 Distance: {dist_gpt:.4f} (Relative: {rel_dist_gpt:.4f})")
    else:
        print(f"GPT Latent shapes differ: {gpt_n.shape} vs {gpt_h.shape}")

    # Speaker Embedding Difference
    if spk_n.shape == spk_h.shape:
        diff_spk = spk_h - spk_n
        dist_spk = torch.norm(diff_spk).item()
        rel_dist_spk = dist_spk / (torch.norm(spk_n).item() + 1e-6)
        print(f"Speaker Emb (Item 1) L2 Distance: {dist_spk:.4f} (Relative: {rel_dist_spk:.4f})")
    else:
        print(f"Speaker Emb shapes differ: {spk_n.shape} vs {spk_h.shape}")

else:
    print("Cannot compare, unexpected structure.")
