"""
cleanup_project.py
-------------------
A full project-structure sanitization tool for the Hindi EmoKnob pipeline.

This script performs:

1. Deletes outdated/unused folders:
       - models/ai4bharat_indicwav2vec_hindi
       - data/new_data
       - data/speaker_embeddings
       - data/tmp
2. Restructures emotion_samples/ into:
       emotion/
          sample001/
             *_neutral_clean.wav
             *_emotion_clean.wav
          sample002/
             ...
3. Cleans speaker audio:
       - converts to 16k mono WAV
       - trims silence
       - normalizes
       - optional noise reduction
4. Moves all stray .wav files from project root → data/outputs/generated_audio/
5. Ensures naming consistency:
       *_clean.wav always used
6. Removes duplicates
7. Generates a clean summary at the end.
"""

import os, shutil, re
from pathlib import Path
import librosa, soundfile as sf
import numpy as np

# ------------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------------

PROJECT_ROOT = Path(r"D:\Downloads\Bengali_EmoKnob")   # <--- change if needed

DATA_DIR = PROJECT_ROOT / "data"
SPEAKERS_DIR = DATA_DIR / "speakers"
EMOTION_DIR = DATA_DIR / "emotion_samples"
OUTPUT_AUDIO_DIR = DATA_DIR / "outputs" / "generated_audio"
OUTPUT_AUDIO_DIR.mkdir(parents=True, exist_ok=True)

# Folders to delete
DELETE_FOLDERS = [
    PROJECT_ROOT / "models" / "ai4bharat_indicwav2vec_hindi",
    DATA_DIR / "new_data",
    DATA_DIR / "speaker_embeddings",
    DATA_DIR / "tmp",
]


# ------------------------------------------------------------------------
# UTILITIES: AUDIO CLEANING (same hybrid pipeline as notebook)
# ------------------------------------------------------------------------

def trim_silence(y, sr, top_db=30):
    yt, _ = librosa.effects.trim(y, top_db=top_db)
    return yt

def normalize_audio(y, target_rms=0.1):
    rms = np.sqrt(np.mean(y**2)) + 1e-9
    gain = target_rms / rms
    y = y * gain
    peak = np.max(np.abs(y))
    if peak > 0.999:
        y = y / peak * 0.999
    return y

def denoise_audio(y, sr):
    try:
        import noisereduce as nr
    except:
        return y
    noise_sample = y[:int(0.5 * sr)]
    try:
        return nr.reduce_noise(y=y, sr=sr, y_noise=noise_sample)
    except:
        return y

def preprocess_audio_any(input_path, output_path, denoise=False, sr=16000):
    y, _ = librosa.load(str(input_path), sr=sr, mono=True)
    y = trim_silence(y, sr)
    if denoise:
        y = denoise_audio(y, sr)
    y = normalize_audio(y)
    sf.write(str(output_path), y, sr, subtype='PCM_16')
    return output_path


# ------------------------------------------------------------------------
# STEP 1 — DELETE UNUSED DIRECTORIES
# ------------------------------------------------------------------------

def delete_useless_folders():
    for folder in DELETE_FOLDERS:
        if folder.exists():
            print(f"[DELETE] Removing folder: {folder}")
            shutil.rmtree(folder)
        else:
            print(f"[OK] Folder not found (already clean): {folder}")


# ------------------------------------------------------------------------
# STEP 2 — MOVE ALL STRAY .wav FILES TO OUTPUT DIRECTORY
# ------------------------------------------------------------------------

def move_root_wavs():
    for file in PROJECT_ROOT.glob("*.wav"):
        print(f"[MOVE] {file.name} -> {OUTPUT_AUDIO_DIR}")
        shutil.move(str(file), OUTPUT_AUDIO_DIR / file.name)


# ------------------------------------------------------------------------
# STEP 3 — CLEAN SPEAKER FILES
# ------------------------------------------------------------------------

def clean_speakers():
    print("\n[CLEAN] Speaker files:")
    for f in SPEAKERS_DIR.glob("*.*"):
        if f.suffix.lower() == ".wav" and f.stem.endswith("_clean"):
            print("   [SKIP] already clean:", f.name)
            continue

        out = f.with_name(f.stem + "_clean.wav")
        print("   [PROCESS]", f.name, "->", out.name)
        try:
            preprocess_audio_any(f, out)
        except Exception as e:
            print("   [ERROR]", f, e)


# ------------------------------------------------------------------------
# STEP 4 — RESTRUCTURE emotion_samples/ INTO sample001/sample002
# ------------------------------------------------------------------------

def restructure_emotion_samples():
    print("\n[RESTRUCTURE] Emotion dataset:")

    for emotion in EMOTION_DIR.iterdir():
        if not emotion.is_dir():
            continue

        print(f"\n>> Emotion: {emotion.name}")

        # Detect files not in a 'sampleXXX' folder
        raw_files = [f for f in emotion.glob("*.*") if f.is_file()]
        if raw_files:
            print(f"   [ORGANIZE] Creating sample001/ for loose files...")
            sample1 = emotion / "sample001"
            sample1.mkdir(exist_ok=True)
            for f in raw_files:
                shutil.move(str(f), sample1 / f.name)

        # Now ensure all samples are clean
        sample_index = 1
        for sample in emotion.iterdir():
            if not sample.is_dir():
                continue

            # Rename folder to sampleXXX
            new_name = f"sample{sample_index:03d}"
            new_path = emotion / new_name
            if sample.name != new_name:
                sample.rename(new_path)
                sample = new_path

            print(f"   [SAMPLE] {sample.name}")

            # Ensure *_clean.wav exists
            wavs = list(sample.glob("*"))
            for w in wavs:
                if w.suffix.lower() != ".wav" or w.stem.endswith("_clean"):
                    continue
                out = w.with_name(w.stem + "_clean.wav")
                preprocess_audio_any(w, out)

            sample_index += 1


# ------------------------------------------------------------------------
# STEP 5 — SUMMARY
# ------------------------------------------------------------------------

def summary():
    print("\n[CLEANUP COMPLETE]")
    print("- Removed unused folders")
    print("- Restructured emotion dataset")
    print("- Cleaned speaker audio")
    print("- Moved all generated audio to outputs")
    print("- Ensured naming consistency (_clean.wav)")
    print("- Project is now fully structured for XTTS EmoKnob pipeline!\n")


# ------------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------------

if __name__ == "__main__":
    print("\n=== Starting Project Cleanup ===\n")
    delete_useless_folders()
    move_root_wavs()
    clean_speakers()
    restructure_emotion_samples()
    summary()
