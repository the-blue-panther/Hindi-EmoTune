# Bengali EmoKnob — Fine-Grained Emotion Control for Voice Cloning

## Overview

Bengali EmoKnob is a research and demonstration project inspired by *EmoKnob (Chen et al., 2024)* — a framework for fine-grained emotion control in speech synthesis.  
This project adapts the idea to the **Bengali language**, emphasizing authentic prosody and emotion representation.  

Key features include:
- Real recorded Bengali emotion samples (same sentence, neutral + emotion)
- Fine-grained emotion control through few-shot embeddings
- Stored reusable speaker embeddings and emotion vectors
- CPU/GPU adaptive execution
- Structured and reproducible pipeline for research and demonstrations

---

## Project Goals

- Implement emotion control for Bengali speech synthesis using embedding manipulation.
- Build a reproducible and modular architecture for academic research.
- Support both real-time and precomputed emotion control.
- Enable fast demos using stored speaker embeddings and emotion vectors.
- Allow seamless migration between CPU and GPU systems.

---

## Core Pipeline

### 1. Voice Cloning — Speaker Embedding Extraction
- **Input:** Reference speaker audio (`.wav`)
- **Output:** Speaker embedding (`.npy`)
- **Description:** Encodes a speaker’s vocal characteristics independent of text or emotion.
- **Storage:** `data/speaker_embeddings/`

### 2. Emotion Vector Extraction
- **Input:** Paired neutral and emotional clips for the same sentence
- **Output:** Normalized emotion direction vector (`.npy`)
- **Formula:**  
  $$ v_e = \frac{E(x_e) - E(x_n)}{\|E(x_e) - E(x_n)\|} $$
- **Storage:** `data/outputs/emotion_vectors/`

### 3. Emotion-Controlled Speech Synthesis
- **Input:** Stored speaker embedding, emotion vector, and text
- **Operation:**  
  $$ u_{s,e} = u_s + \alpha \cdot v_e $$
- **Output:** Emotionally controlled synthesized speech (`.wav`)
- **Storage:** `data/outputs/generated/`

---

## Extended Components

| Component | Purpose |
|------------|----------|
| Audio Preprocessing | Standardizes recordings to 24kHz mono WAV format. |
| Evaluation | Computes WER (Whisper) and speaker similarity (SpeechBrain). |
| Visualization | Embedding visualization via PCA/t-SNE for emotion direction. |
| Caching | Stores precomputed embeddings and vectors for quick reuse. |
| Live Demo | Supports on-the-fly recordings and emotional synthesis in Jupyter. |

---

## Folder Structure

```

Bengali_EmoKnob/
├─ README.md
├─ requirements.txt
├─ setup_project.py
├─ bengali_emoknob_demo.ipynb
│
├─ data/
│  ├─ emotion_samples/
│  │   ├─ happy/
│  │   │   ├─ speaker001_neutral.wav
│  │   │   ├─ speaker001_happy.wav
│  │   ├─ sad/
│  │   ├─ angry/
│  │   └─ neutral/
│  │
│  ├─ speakers/
│  ├─ speaker_embeddings/
│  ├─ outputs/
│  │   ├─ emotion_vectors/
│  │   ├─ generated/
│  │   ├─ logs/
│  │   └─ metrics.csv
│  ├─ new_data/
│  │   ├─ speakers/
│  │   └─ emotion_samples/
│  └─ tmp/
│
├─ scripts/
│  ├─ preprocess_audio.py
│  ├─ compute_emotion_vector.py
│  ├─ synthesize_with_emotion.py
│  ├─ generate_embedding.py
│  └─ utils.py
│
└─ models/
├─ README_models.txt
├─ whisper/
├─ xtts/
├─ speechbrain/
├─ bark/ (optional)
└─ fine_tuned/ (optional)

````

---

## Environment Setup

### 1. Create and Activate Virtual Environment

**Windows (PowerShell):**
```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
````

**Linux/Mac:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

If you see `(.venv)` at the start of your terminal prompt, the environment is active.

---

### 2. Install Dependencies

Upgrade pip first:

```bash
python -m pip install --upgrade pip setuptools wheel
```

Then install all project dependencies:

```bash
pip install -r requirements.txt
```

---

### 3. Installation Notes — CPU / GPU Environments

This project supports both CPU and GPU systems.

#### For CPU-only systems (default)

If you’re running on a machine without an NVIDIA GPU:

```bash
pip install -r requirements.txt
```

PyTorch will automatically install the CPU-compatible version.

#### For GPU-enabled systems (with CUDA)

When moving to a system with a GPU, install the CUDA-enabled PyTorch first:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

Replace `cu121` with your CUDA version if necessary
(see [PyTorch installation guide](https://pytorch.org/get-started/locally) for options).

Check GPU availability:

```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

Expected output on a GPU system:

```
CUDA available: True
```

---

### 4. Verify Installation

After setup, verify the core packages:

```bash
python -c "import torch, librosa, soundfile, transformers, TTS, whisper, speechbrain; print('✅ All major packages imported successfully!')"
```

Check PyTorch version and device type:

```bash
python -c "import torch; print('Torch:', torch.__version__, '| CUDA available:', torch.cuda.is_available())"
```

---

### 5. Freeze Environment

After successful setup:

```bash
pip freeze > requirements.txt
```

This saves all exact versions used in your current environment for reproducibility.

---

## Running the Project

Launch the Jupyter Notebook:

```bash
jupyter notebook bengali_emoknob_demo.ipynb
```

Follow the notebook instructions to:

* Preprocess audio files
* Compute emotion vectors
* Generate emotion-controlled speech
* Perform evaluations and visualize embeddings
* Run live audience demos

---

## Evaluation Metrics

| Metric                 | Description                                                |
| ---------------------- | ---------------------------------------------------------- |
| Word Error Rate (WER)  | Evaluates synthesized speech accuracy using Whisper ASR.   |
| Speaker Similarity     | Compares embeddings between reference and generated audio. |
| Emotion Identification | Subjective or model-based emotion classification.          |
| Execution Time         | Benchmarks CPU vs GPU synthesis speed.                     |

---

## Data Handling and Preprocessing

* Accepts `.wav`, `.mp3`, `.flac`, `.m4a`, `.ogg` formats.
* Automatically converts to standardized `.wav` at 24kHz mono.
* Normalizes amplitude for consistent embeddings.
* Saves preprocessed files to `data/emotion_samples/` or `data/tmp/`.

---

## Offline Usage

All model weights and embeddings are stored locally:

* Speaker embeddings → `data/speaker_embeddings/`
* Emotion vectors → `data/outputs/emotion_vectors/`
* Model weights → `models/`

Once downloaded, no internet connection is required.

---

## Future Enhancements

* Integration of open-ended text-based emotion prompts.
* Cross-speaker emotion transfer and blending.
* GUI-based emotion control panel.
* Incorporation of Bengali-English (Benglish) mixed speech.
* Fine-tuning XTTS for Bengali prosody.

---

## Ethical Use

This project is intended for **research and educational purposes** only.
Voice cloning should not be used for impersonation or commercial applications.
When using celebrity or fictional voices, limit usage to private or academic demonstrations only.

---

## Author

**Arnab Banerjee**
MSc in Data Science & Artificial Intelligence
Focus areas: Speech synthesis, applied mathematics, and emotion-aware AI systems.

---

## Acknowledgments

* *EmoKnob: Enhance Voice Cloning with Fine-Grained Emotion Control* (Chen et al., 2024, Columbia University)
* Coqui TTS (XTTS v2)
* OpenAI Whisper ASR
* SpeechBrain ECAPA-TDNN
* Hugging Face for open-source model distribution

---

## Citation

If you use or reference this project, please cite:

```
@misc{arnab2025bengaliemoknob,
  title={Bengali EmoKnob: Fine-Grained Emotion Control for Voice Cloning},
  author={Arnab Banerjee},
  year={2025},
  note={Inspired by EmoKnob (Chen et al., 2024)}
}
```



