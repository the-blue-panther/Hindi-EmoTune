
import torch
import numpy as np
from pathlib import Path
from TTS.api import TTS

PROJECT_ROOT = Path(r"D:\Downloads\Bengali_EmoKnob")
SR_XTTS = 22050

# 1. Load Model
print("Loading model...")
# Assuming model is already downloaded/available as per notebook
try:
    # Try local load
    t = TTS(model_path=str(Path('models') / 'xtts_v2' / 'model.pth'),
            config_path=str(Path('models') / 'xtts_v2' / 'config.json'),
            gpu=False)
except:
    t = TTS(model_name='tts_models/multilingual/multi-dataset/xtts_v2', gpu=False)

model = t.synthesizer.tts_model

# 2. Extract Latents from Samples
emotion = "happy"
sample_dir = PROJECT_ROOT / 'data' / 'emotion_samples' / emotion / 'sample001'
n_clean = sample_dir / 'neutral_clean.wav'
e_clean = sample_dir / f'{emotion}_clean.wav'

if not (n_clean.exists() and e_clean.exists()):
    print("Sample files not found, cannot run diagnostic.")
    exit()

print(f"Extracting latents from {n_clean} and {e_clean}...")
# Get full latents (res[0] is GPT conditioned latent)
cond_n = model.get_conditioning_latents(str(n_clean), load_sr=SR_XTTS)
cond_e = model.get_conditioning_latents(str(e_clean), load_sr=SR_XTTS)

gps_n = cond_n[0].detach().cpu().numpy().flatten()
gps_e = cond_e[0].detach().cpu().numpy().flatten()

# 3. Calculate Norms
norm_n = np.linalg.norm(gps_n)
norm_e = np.linalg.norm(gps_e)
delta = gps_e - gps_n
norm_delta = np.linalg.norm(delta)

print(f"\n--- MAGNITUDE DIAGNOSTICS ---")
print(f"Base Latent Norm (Neutral): {norm_n:.4f}")
print(f"Base Latent Norm (Happy):   {norm_e:.4f}")
print(f"Emotion Vector Norm (Diff): {norm_delta:.4f}")
print(f"Ratio (Delta / Base):       {norm_delta / norm_n:.4%}")

# 4. Check Impact of Alpha=1.0
# New = Base + 1.0 * Delta
# Verify if this shift is significant enough
print(f"\n--- INTERPRETATION ---")
if norm_delta / norm_n < 0.05:
    print(f"⚠️ Vector is VERY small ({norm_delta / norm_n:.1%} of base).")
    print(f"   Alpha=1.0 might be barely audible.")
    rec_alpha = (0.1 * norm_n) / norm_delta
    print(f"   Recommendation: Try setting alpha to ~{rec_alpha:.1f} to achieve 10% shift.")
else:
    print("Vector magnitude seems reasonable. Issue might be direction or averaging.")
