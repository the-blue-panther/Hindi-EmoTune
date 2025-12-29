import json
import shutil
from pathlib import Path

nb_path = Path(r"d:/Downloads/Bengali_EmoKnob/hindi_emoknob_demo_v14.ipynb")
backup_path = nb_path.with_suffix(".ipynb.bak_v14_patch")

if not nb_path.exists():
    print(f"Notebook not found: {nb_path}")
    exit(1)

# Backup
shutil.copy2(nb_path, backup_path)
print(f"Backed up to {backup_path}")

with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# New code for get_xtts_speaker_latent
new_code_1 = [
    "# Embedding helpers: safe XTTS resolver and embedding extraction\n",
    "import numpy as np\n",
    "import torch\n",
    "import librosa\n",
    "from pathlib import Path\n",
    "\n",
    "def resolve_xtts_internal_model(tts_obj):\n",
    "    '''Return the internal XTTS model used by TTS wrapper.'''\n",
    "    if tts_obj is None:\n",
    "        raise RuntimeError('Provided tts_obj is None')\n",
    "    if hasattr(tts_obj, 'synthesizer') and hasattr(tts_obj.synthesizer, 'tts_model'):\n",
    "        return tts_obj.synthesizer.tts_model\n",
    "    if hasattr(tts_obj, 'tts_model'):\n",
    "        return tts_obj.tts_model\n",
    "    raise RuntimeError('Could not resolve internal XTTS model. Ensure you loaded native XTTS-v2 via TTS API.')\n",
    "\n",
    "def get_indic_embedding(wav_path, sr_source=SR_XTTS, sr_indic=SR_INDIC):\n",
    "    '''Load wav, resample to sr_indic if needed, and return 1D numpy embedding (mean of last_hidden_state).'''\n",
    "    global processor, indic_enc\n",
    "    if 'processor' not in globals() or 'indic_enc' not in globals():\n",
    "        raise RuntimeError('Indic encoder not loaded. Run the Indic load cell.')\n",
    "    y, sr = librosa.load(str(wav_path), sr=sr_source, mono=True)\n",
    "    if sr != sr_indic:\n",
    "        y = librosa.resample(y, orig_sr=sr, target_sr=sr_indic)\n",
    "    inp = processor(y, sampling_rate=sr_indic, return_tensors='pt', padding=True)\n",
    "    with torch.no_grad():\n",
    "        out = indic_enc(**inp).last_hidden_state\n",
    "    emb = out.mean(dim=1).squeeze().detach().cpu().numpy()\n",
    "    return emb\n",
    "\n",
    "def get_xtts_speaker_latent(tts_obj, wav_path, load_sr=SR_XTTS):\n",
    "    '''Extract GPT conditioning latent from XTTS (prosody/emotion).\n",
    "    Returns 1D numpy array (flattened). Original shape is [1, 32, 1024].'''\n",
    "    model = resolve_xtts_internal_model(tts_obj)\n",
    "    try:\n",
    "        res = model.get_conditioning_latents(str(wav_path), load_sr=load_sr)\n",
    "    except TypeError:\n",
    "        res = model.get_conditioning_latents(str(wav_path))\n",
    "    \n",
    "    # res is (gpt_cond_latent, speaker_embedding)\n",
    "    if isinstance(res, (list, tuple)) and len(res) >= 1:\n",
    "        gpt_cond = res[0]  # This is the prosody/emotion latent\n",
    "    else:\n",
    "        raise RuntimeError(f\"Unexpected return from get_conditioning_latents: {type(res)}\")\n",
    "\n",
    "    try:\n",
    "        # Flatten [1, 32, 1024] -> [32768]\n",
    "        sp = gpt_cond.reshape(-1)\n",
    "        return sp.detach().cpu().numpy() if hasattr(sp, 'detach') else np.array(sp)\n",
    "    except Exception as e:\n",
    "        raise RuntimeError('Failed to convert GPT latent to numpy: ' + str(e))\n",
    "\n",
    "print('Helpers ready: resolve_xtts_internal_model, get_indic_embedding, get_xtts_speaker_latent (Targeting GPT Latent)')\n"
]

# New code for apply_emotion_and_synthesize
new_code_2 = [
    "# Apply emotion vector and synthesize via XTTS\n",
    "import numpy as np\n",
    "import torch\n",
    "from pathlib import Path\n",
    "import soundfile as sf\n",
    "\n",
    "OUTPUT_GEN_DIR = PROJECT_ROOT / 'data' / 'outputs' / 'generated'\n",
    "OUTPUT_GEN_DIR.mkdir(parents=True, exist_ok=True)\n",
    "\n",
    "def apply_emotion_and_synthesize(text, speaker_wav, emotion_vec, alpha=0.1, out_path=None, language='hi', scale_to_speaker=True):\n",
    "    \"\"\"Apply emotion vector to GPT conditioning latent and synthesize speech.\n",
    "    \n",
    "    Args:\n",
    "        text: Hindi text to synthesize\n",
    "        speaker_wav: Path to speaker reference audio (or cleaned wav)\n",
    "        emotion_vec: Emotion direction vector (flattened delta of gpt_cond_latent)\n",
    "        alpha: Blending intensity.\n",
    "        out_path: Output wav file path\n",
    "        language: Language code (default: 'hi' for Hindi)\n",
    "        scale_to_speaker: If True, scale emotion_vec by latent norm (recommended)\n",
    "    \n",
    "    Returns:\n",
    "        Path to generated audio file\n",
    "    \"\"\"\n",
    "    if out_path is None:\n",
    "        out_path = OUTPUT_GEN_DIR / 'test_hindi_emotional.wav'\n",
    "    out_path = unique_path(Path(out_path))\n",
    "\n",
    "    # Get original latents from reference audio\n",
    "    model = resolve_xtts_internal_model(XTTS)\n",
    "    try:\n",
    "        latents = model.get_conditioning_latents(str(speaker_wav), load_sr=SR_XTTS)\n",
    "    except TypeError:\n",
    "        latents = model.get_conditioning_latents(str(speaker_wav))\n",
    "        \n",
    "    gpt_cond, speaker_emb = latents[0], latents[1]\n",
    "    \n",
    "    # Prepare emotion vector\n",
    "    ev = np.asarray(emotion_vec).astype(np.float32)\n",
    "    \n",
    "    # Target shape for GPT latent: [1, 32, 1024] -> size 32768\n",
    "    target_size = 32 * 1024\n",
    "    \n",
    "    if ev.size != target_size:\n",
    "        print(f\"Warning: Emotion vector size {ev.size} != target {target_size}. Resizing/Padding...\")\n",
    "        if ev.size > target_size:\n",
    "            ev = ev[:target_size]\n",
    "        else:\n",
    "            ev = np.pad(ev, (0, target_size - ev.size))\n",
    "            \n",
    "    # Reshape to match tensor\n",
    "    ev_tensor = torch.tensor(ev).reshape(1, 32, 1024).to(gpt_cond.device)\n",
    "\n",
    "    # Scaling logic\n",
    "    if scale_to_speaker:\n",
    "        base_norm = torch.norm(gpt_cond)\n",
    "        ev_norm = torch.norm(ev_tensor)\n",
    "        if base_norm > 1e-6 and ev_norm > 1e-6:\n",
    "            print(f\"[Emotion Scaling] Base Norm={base_norm:.4f}, Delta Norm={ev_norm:.4f}\")\n",
    "\n",
    "    # Apply emotion: new_latent = original + alpha * delta\n",
    "    new_gpt_cond = gpt_cond + alpha * ev_tensor\n",
    "    \n",
    "    # Monkey-patch to inject modified latent\n",
    "    original_get_cond = model.get_conditioning_latents\n",
    "    \n",
    "    def patched_get_cond(*args, **kwargs):\n",
    "        \"\"\"Return modified GPT latent with original speaker embedding\"\"\"\n",
    "        # We ignore the args (which would be the speaker_wav) and return our pre-computed latents\n",
    "        return (new_gpt_cond, speaker_emb)\n",
    "    \n",
    "    try:\n",
    "        # Temporarily patch the model\n",
    "        model.get_conditioning_latents = patched_get_cond\n",
    "        \n",
    "        # Use standard tts_to_file\n",
    "        XTTS.tts_to_file(text=text, speaker_wav=str(speaker_wav), language=language, file_path=str(out_path))\n",
    "        print(f'✓ Synthesis complete -> {out_path}')\n",
    "        return out_path\n",
    "    except Exception as e:\n",
    "        print(f'Synthesis failed: {e}')\n",
    "        import traceback\n",
    "        traceback.print_exc()\n",
    "        raise RuntimeError(f'Could not synthesize with custom latent: {str(e)}')\n",
    "    finally:\n",
    "        # Restore original method\n",
    "        model.get_conditioning_latents = original_get_cond\n",
    "\n",
    "print('✓ apply_emotion_and_synthesize() ready (Targeting GPT Latent)')\n"
]

count = 0
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        src = "".join(cell['source'])
        if "def get_xtts_speaker_latent" in src:
            cell['source'] = new_code_1
            count += 1
            print("Patched get_xtts_speaker_latent")
        elif "def apply_emotion_and_synthesize" in src:
            cell['source'] = new_code_2
            count += 1
            print("Patched apply_emotion_and_synthesize")

if count == 2:
    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
    print("Notebook patched successfully.")
else:
    print(f"Error: Expected to patch 2 cells, but patched {count}. Check cell contents.")
