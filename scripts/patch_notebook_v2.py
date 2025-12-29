import json
import shutil
from pathlib import Path

nb_path = Path(r"d:/Downloads/Bengali_EmoKnob/hindi_emoknob_demo_v14.ipynb")
backup_path = nb_path.with_suffix(".ipynb.bak_v14_patch2")

if not nb_path.exists():
    print(f"Notebook not found: {nb_path}")
    exit(1)

# Backup
shutil.copy2(nb_path, backup_path)
print(f"Backed up to {backup_path}")

with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# New code for compute_emotion_vector_xtts_multi
new_code_compute = [
    "# Compute emotion vectors from emotion_samples folder (multi-sample average or single)\n",
    "import numpy as np\n",
    "from pathlib import Path\n",
    "\n",
    "EMOTION_SAMPLES_DIR = PROJECT_ROOT / 'data' / 'emotion_samples'\n",
    "OUTPUT_SINGLE_DIR = PROJECT_ROOT / 'data' / 'outputs' / 'emotion_vectors' / 'single'\n",
    "OUTPUT_AVG_DIR = PROJECT_ROOT / 'data' / 'outputs' / 'emotion_vectors' / 'average'\n",
    "for p in [OUTPUT_SINGLE_DIR, OUTPUT_AVG_DIR]:\n",
    "    Path(p).mkdir(parents=True, exist_ok=True)\n",
    "\n",
    "def compute_emotion_vector_xtts_multi(emotion_dir, method='cca', n_comp=32, mode='average', sample_id=1,\n",
    "                                      save_single_dir=None, save_avg_dir=None):\n",
    "    emotion_dir = Path(emotion_dir)\n",
    "    sample_dirs = [d for d in sorted(emotion_dir.iterdir()) if d.is_dir()]\n",
    "    if len(sample_dirs) == 0:\n",
    "        raise ValueError('No sample subfolders found in: ' + str(emotion_dir))\n",
    "\n",
    "    X = []\n",
    "    Y = []\n",
    "    single_vectors = []\n",
    "\n",
    "    for sd in sample_dirs:\n",
    "        emotion_name = emotion_dir.name\n",
    "        n_clean = sd / 'neutral_clean.wav'\n",
    "        e_clean = sd / f'{emotion_name}_clean.wav'\n",
    "        \n",
    "        if n_clean.exists() and e_clean.exists():\n",
    "            print(f'[{sd.name}] Found existing clean files')\n",
    "        else:\n",
    "            # (Preprocessing logic omitted for brevity, assuming files exist or user runs preprocessing)\n",
    "            # In a real patch we should keep the preprocessing logic, but for now let's assume clean files exist\n",
    "            # or just copy the logic if we want to be safe. \n",
    "            # Actually, let's copy the logic to be safe.\n",
    "            raw_files = [f for f in sorted(sd.iterdir()) if f.suffix.lower() in ['.wav', '.mp3', '.m4a', '.flac']]\n",
    "            if len(raw_files) < 2:\n",
    "                print(f'[{sd.name}] Skipping (need ≥2 raw audio files)')\n",
    "                continue\n",
    "            neutral_raw = raw_files[0]\n",
    "            emotion_raw = raw_files[1]\n",
    "            for f in raw_files:\n",
    "                if 'neutral' in f.stem.lower(): neutral_raw = f\n",
    "                elif emotion_name.lower() in f.stem.lower(): emotion_raw = f\n",
    "            if not n_clean.exists(): preprocess_audio(neutral_raw, n_clean, sr=SR_XTTS)\n",
    "            if not e_clean.exists(): preprocess_audio(emotion_raw, e_clean, sr=SR_XTTS)\n",
    "\n",
    "        xi = get_indic_embedding(n_clean, sr_source=SR_XTTS, sr_indic=SR_INDIC)\n",
    "        xe = get_indic_embedding(e_clean, sr_source=SR_XTTS, sr_indic=SR_INDIC)\n",
    "        yi = get_xtts_speaker_latent(XTTS, n_clean, load_sr=SR_XTTS)\n",
    "        ye = get_xtts_speaker_latent(XTTS, e_clean, load_sr=SR_XTTS)\n",
    "\n",
    "        X.append(xe - xi)\n",
    "        Y.append(ye - yi)\n",
    "        single_vectors.append((sd.name, xe - xi, ye - yi))\n",
    "\n",
    "    if len(X) == 0:\n",
    "        raise ValueError('No matched pairs extracted for emotion: ' + str(emotion_dir))\n",
    "\n",
    "    X = np.stack(X)\n",
    "    Y = np.stack(Y)\n",
    "\n",
    "    # Check for high-dimensional latent (GPT latent is ~32k)\n",
    "    if Y.shape[1] > 1024:\n",
    "        if method != 'xtts_native':\n",
    "            print(f'⚠️ High-dimensional latent detected ({Y.shape[1]} dims). Forcing \"xtts_native\" method.')\n",
    "            print('   CCA/PLS are not suitable for 32k dimensions with few samples.')\n",
    "            method = 'xtts_native'\n",
    "\n",
    "    if len(X) < 5 and method != 'xtts_native':\n",
    "        print(f'⚠️ Only {len(X)} samples; CCA/PLS unreliable with <5 samples. Switching to \"xtts_native\".')\n",
    "        method = 'xtts_native'\n",
    "\n",
    "    if method == 'xtts_native':\n",
    "        if mode == 'single':\n",
    "            idx = sample_id - 1\n",
    "            raw_delta = single_vectors[idx][2]\n",
    "            if save_avg_dir:\n",
    "                Path(save_avg_dir).mkdir(parents=True, exist_ok=True)\n",
    "                np.save(Path(save_avg_dir) / f\"{emotion_dir.name}_single{sample_id:03d}_xtts_raw.npy\", raw_delta)\n",
    "            return raw_delta\n",
    "        \n",
    "        avg = np.mean([v for (_,_,v) in single_vectors], axis=0)\n",
    "        if save_avg_dir:\n",
    "            Path(save_avg_dir).mkdir(parents=True, exist_ok=True)\n",
    "            np.save(Path(save_avg_dir) / f\"{emotion_dir.name}_avg_xtts_raw.npy\", avg)\n",
    "        \n",
    "        avg_norm = np.linalg.norm(avg)\n",
    "        print(f\"\\n  ✓ Emotion delta computed from {len(single_vectors)} samples\")\n",
    "        print(f\"    Shape: {avg.shape}\")\n",
    "        print(f\"    Norm: {avg_norm:.6f}\")\n",
    "        return avg\n",
    "\n",
    "    # Fallback for low-dim latents (if any) using CCA/PLS\n",
    "    max_comp = min(X.shape[0], X.shape[1], Y.shape[1])\n",
    "    actual_n_comp = min(n_comp, max_comp)\n",
    "    mapper = fit_cca_or_pls(X, Y, method=method, n_comp=actual_n_comp)\n",
    "\n",
    "    if mode == 'single':\n",
    "        idx = sample_id - 1\n",
    "        v_indic = single_vectors[idx][1]\n",
    "    else:\n",
    "        v_indic = np.mean([xi for (_,xi,_) in single_vectors], axis=0)\n",
    "\n",
    "    v_indic = v_indic / (np.linalg.norm(v_indic) + 1e-12)\n",
    "    mapped = map_indic_vector_to_xtts(mapper, v_indic)\n",
    "    mapped = mapped / (np.linalg.norm(mapped) + 1e-12)\n",
    "    return mapped\n",
    "\n",
    "print(\"✓ compute_emotion_vector_xtts_multi() updated (Forces xtts_native for high-dim latents)\")"
]

count = 0
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        src = "".join(cell['source'])
        if "def compute_emotion_vector_xtts_multi" in src:
            cell['source'] = new_code_compute
            count += 1
            print("Patched compute_emotion_vector_xtts_multi")

if count == 1:
    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
    print("Notebook patched successfully.")
else:
    print(f"Error: Expected to patch 1 cell, but patched {count}. Check cell contents.")
