"""Generate and score an unmodified Hindi XTTS baseline using local model weights."""
import argparse
import csv
import random
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import whisper
from jiwer import cer, wer
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
import unicodedata


def normalize(text):
    text = unicodedata.normalize("NFC", text)
    return " ".join("".join(" " if unicodedata.category(c).startswith("P") else c for c in text).split())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--text", default="आज सुबह मैं जल्दी उठा और खिड़की खोलकर बाहर देखा।")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    model_dir = args.project / "models" / "xtts_v2"
    reference = args.project / "data" / "speakers" / "amir_khan_clean.wav"
    config = XttsConfig()
    config.load_json(str(model_dir / "config.json"))
    model = Xtts.init_from_config(config)
    model.load_checkpoint(config, checkpoint_dir=str(model_dir), eval=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    with torch.inference_mode():
        cond, speaker = model.get_conditioning_latents(audio_path=[str(reference)])
        result = model.inference(args.text, "hi", cond, speaker, temperature=0.7,
                                 repetition_penalty=2.0, top_k=50, top_p=0.8)
    args.output.mkdir(parents=True, exist_ok=False)
    wav = args.output / "baseline.wav"
    sf.write(wav, result["wav"], config.audio.output_sample_rate)
    import librosa
    audio, _ = librosa.load(wav, sr=16000, mono=True)
    transcript = whisper.load_model("tiny", device="cpu").transcribe(audio, language="hi", fp16=False)["text"]
    ref, hyp = normalize(args.text), normalize(transcript)
    row = dict(alpha=0, seed=args.seed, speaker=reference.name, text=args.text,
               transcript=transcript, wer=wer(ref, hyp), cer=cer(ref, hyp))
    with (args.output / "baseline.csv").open("w", encoding="utf-8", newline="") as dest:
        writer = csv.DictWriter(dest, fieldnames=row)
        writer.writeheader()
        writer.writerow(row)


if __name__ == "__main__":
    main()
