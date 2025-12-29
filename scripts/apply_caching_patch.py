import json
from pathlib import Path

nb_path = Path("d:/Downloads/Bengali_EmoKnob/hindi_emoknob_demo_v15.ipynb")

with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# The new code for the cell
new_code = [
    "def compute_emotion_vector_xtts_multi(emotion_dir, method='cca', n_comp=32, mode='average', sample_id=1,\n",
    "                                      save_single_dir=None, save_avg_dir=None):\n",
    "    # FIX: Import numpy specifically at the top to avoid UnboundLocalError scopes\n",
    "    import numpy as np\n",
    "    emotion_dir = Path(emotion_dir)\n",
    "    emotion_name = emotion_dir.name\n",
    "    \n",
    "    # CACHE CHECK: If mode is average and save_avg_dir is provided\n",
    "    if mode == 'average' and save_avg_dir:\n",
    "        save_avg_dir = Path(save_avg_dir)\n",
    "        save_avg_dir.mkdir(parents=True, exist_ok=True)\n",
    "        avg_file_name = f\"{emotion_name}_avg_{method}.npy\"\n",
    "        avg_file = save_avg_dir / avg_file_name\n",
    "        \n",
    "        if avg_file.exists():\n",
    "            print(f'⚡ Loading cached average vector for {emotion_name}...')\n",
    "            return np.load(avg_file)\n",
    "            \n",
    "    sample_dirs = [d for d in sorted(emotion_dir.iterdir()) if d.is_dir()]\n",
    "    if len(sample_dirs) == 0:\n",
    "        raise ValueError('No sample subfolders found in: ' + str(emotion_dir))\n",
    "\n",
    "    X = []\n",
    "    Y = []\n",
    "    single_vectors = []\n",
    "\n",
    "    for sd in sample_dirs:\n",
    "        n_clean = sd / 'neutral_clean.wav'\n",
    "        e_clean = sd / f'{emotion_name}_clean.wav'\n",
    "        \n",
    "        if not (n_clean.exists() and e_clean.exists()):\n",
    "             continue\n",
    "\n",
    "        xi = get_indic_embedding(n_clean, sr_source=SR_XTTS, sr_indic=SR_INDIC)\n",
    "        xe = get_indic_embedding(e_clean, sr_source=SR_XTTS, sr_indic=SR_INDIC)\n",
    "        yi = get_xtts_speaker_latent(XTTS, n_clean, load_sr=SR_XTTS)\n",
    "        ye = get_xtts_speaker_latent(XTTS, e_clean, load_sr=SR_XTTS)\n",
    "        \n",
    "        # Store deltas\n",
    "        X.append(xe - xi)\n",
    "        Y.append(ye - yi)\n",
    "        single_vectors.append((sd.name, xe - xi, ye - yi))\n",
    "        \n",
    "    if len(X) == 0:\n",
    "        raise ValueError('No matched pairs extracted for emotion: ' + str(emotion_dir))\n",
    "\n",
    "    X = np.stack(X)\n",
    "    Y = np.stack(Y)\n",
    "    \n",
    "    # Check dimensions\n",
    "    dim_y = Y.shape[1] \n",
    "    if dim_y > 1024 and method != 'xtts_native':\n",
    "        print(f'⚠️ Detected high-dimensional latent ({dim_y} dims). Forcing method=\"xtts_native\".')\n",
    "        method = 'xtts_native'\n",
    "\n",
    "    if len(X) < 5 and method != 'xtts_native':\n",
    "        print(f'⚠️ Only {len(X)} samples; Switch to \"xtts_native\" for stability.')\n",
    "        method = 'xtts_native'\n",
    "\n",
    "    result_vec = None\n",
    "    \n",
    "    if method == 'xtts_native':\n",
    "        if mode == 'single':\n",
    "            idx = sample_id - 1\n",
    "            result_vec = single_vectors[idx][2]\n",
    "        else:\n",
    "            # Average raw emotion deltas\n",
    "            result_vec = np.mean([v for (_,_,v) in single_vectors], axis=0)\n",
    "            \n",
    "    else:\n",
    "        # CCA/PLS logic\n",
    "        max_comp = min(X.shape[0], X.shape[1], Y.shape[1])\n",
    "        actual_n_comp = min(n_comp, max_comp)\n",
    "        mapper = fit_cca_or_pls(X, Y, method=method, n_comp=actual_n_comp)\n",
    "        v_indic = np.mean([xi for (_,xi,_) in single_vectors], axis=0)\n",
    "        v_indic = v_indic / (np.linalg.norm(v_indic) + 1e-12)\n",
    "        result_vec = map_indic_vector_to_xtts(mapper, v_indic)\n",
    "\n",
    "    # CACHE SAVE: If mode is average and save_avg_dir is provided\n",
    "    if mode == 'average' and save_avg_dir and result_vec is not None:\n",
    "        save_avg_dir = Path(save_avg_dir)\n",
    "        save_avg_dir.mkdir(parents=True, exist_ok=True)\n",
    "        avg_file_name = f\"{emotion_name}_avg_{method}.npy\"\n",
    "        print(f'💾 Saving average vector to {save_avg_dir / avg_file_name}...')\n",
    "        np.save(save_avg_dir / avg_file_name, result_vec)\n",
    "\n",
    "    return result_vec\n",
    "\n",
    "print(\"✓ compute_emotion_vector_xtts_multi() updated (With CACHING)\")"
]

found = False
for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        source = "".join(cell["source"])
        if "def compute_emotion_vector_xtts_multi" in source:
            print("Found cell. Updating...")
            cell["source"] = new_code
            found = True
            break

if found:
    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
    print("Notebook updated successfully.")
else:
    print("Could not find the function definition to replace.")
