import json
import os
import shutil

nb_path = r"d:\Downloads\Bengali_EmoKnob\hindi_emoknob_demo_v15.ipynb"
backup_path = r"d:\Downloads\Bengali_EmoKnob\hindi_emoknob_demo_v15.ipynb.bak"

# Create backup
if not os.path.exists(backup_path):
    shutil.copy2(nb_path, backup_path)

with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# The new source code for the cell
# I'm commenting out the local check logic but leaving the function definition for ensure_xtts_local as requested ("keep every other functionality intact")
# actually the user just said "Update the code so that it goes straight to leading the model using model_name".
new_source_code = [
    "# XTTS local download & loader (local-first)\n",
    "from TTS.api import TTS\n",
    "from huggingface_hub import snapshot_download\n",
    "import shutil, os, traceback\n",
    "from pathlib import Path\n",
    "\n",
    "def ensure_xtts_local(target_dir: Path):\n",
    "    target_dir.mkdir(parents=True, exist_ok=True)\n",
    "    ck = target_dir / 'model.pth'\n",
    "    cfg = target_dir / 'config.json'\n",
    "    if ck.exists() and cfg.exists():\n",
    "        print('XTTS local present:', target_dir)\n",
    "        return True\n",
    "    print('Attempting snapshot_download of coqui/xtts-v2 into models folder (best-effort)...')\n",
    "    try:\n",
    "        tmp = snapshot_download(repo_id='coqui/xtts-v2', cache_dir=str(target_dir), repo_type='model', allow_patterns=['*'])\n",
    "        print('snapshot_download result:', tmp)\n",
    "    except Exception as e:\n",
    "        print('snapshot_download failed (this is OK if huggingface auth required). Error:', e)\n",
    "        traceback.print_exc()\n",
    "    ck = target_dir / 'model.pth'\n",
    "    cfg = target_dir / 'config.json'\n",
    "    if ck.exists() and cfg.exists():\n",
    "        return True\n",
    "    print('XTTS not available locally. You can allow TTS to download to cache once, then move folder to models/xtts_v2.')\n",
    "    return False\n",
    "\n",
    "def load_xtts_local_or_remote(gpu=False):\n",
    "    # Modified to skip local check and go straight to model_name loader\n",
    "    print('Loading XTTS via model_name (this will download to user cache if not present)...')\n",
    "    t = TTS(model_name='tts_models/multilingual/multi-dataset/xtts_v2', gpu=gpu)\n",
    "    print('XTTS loaded via model_name.')\n",
    "    return t\n",
    "\n",
    "# Load XTTS (CPU first)\n",
    "XTTS = None\n",
    "try:\n",
    "    XTTS = load_xtts_local_or_remote(gpu=False)\n",
    "except Exception as e:\n",
    "    print('XTTS load error:', e)\n",
    "    import traceback; traceback.print_exc()\n"
]

found = False
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        # Check if this is the target cell by looking for unique strings
        source_text = "".join(cell['source'])
        if "def load_xtts_local_or_remote(gpu=False):" in source_text and "ensure_xtts_local" in source_text:
            cell['source'] = new_source_code
            found = True
            break

if found:
    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    print("Notebook updated successfully.")
else:
    print("Target cell not found.")
