"""Generate and score an unmodified Hindi XTTS baseline using local model weights."""
import argparse
import csv
import random
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
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
    parser.add_argument("--compare", action="store_true", help="Also synthesize sad, angry and happy at alpha=3")
    args = parser.parse_args()
    model_dir = args.project / "models" / "xtts_v2"
    reference = args.project / "data" / "speakers" / "amir_khan_clean.wav"
    config = XttsConfig()
    config.load_json(str(model_dir / "config.json"))
    model = Xtts.init_from_config(config)
    model.load_checkpoint(config, checkpoint_dir=str(model_dir), eval=True)
    args.output.mkdir(parents=True, exist_ok=False)
    import json
    import hashlib
    manifest = dict(text=args.text, seed=args.seed, reference=str(reference),
                    reference_sha256=hashlib.sha256(reference.read_bytes()).hexdigest(),
                    model_config_sha256=hashlib.sha256((model_dir / "config.json").read_bytes()).hexdigest(),
                    alpha=3.0, compare=args.compare)
    (args.output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    with torch.inference_mode():
        cond, speaker = model.get_conditioning_latents(audio_path=[str(reference)])
    runs = [("baseline", 0.0, torch.zeros_like(cond))]
    if args.compare:
        import librosa
        for emotion in ("sad", "angry", "happy"):
            deltas = []
            pair_root = args.project / "data" / "emotion_samples" / emotion
            for pair in sorted(pair_root.iterdir()):
                if not pair.is_dir():
                    continue
                # Explicit raw names avoid relying on potentially stale cleaned files.
                neutral = pair / "neutral.m4a"
                emotional = pair / (emotion + ".m4a")
                if emotion == "happy" and pair.name == "sample002":
                    neutral, emotional = pair / "n9.m4a", pair / "h9.m4a"
                if not neutral.is_file() or not emotional.is_file():
                    raise FileNotFoundError(f"Missing explicit raw pair in {pair}")
                paths = []
                for label, raw in (("neutral", neutral), (emotion, emotional)):
                    audio, _ = librosa.load(raw, sr=22050, mono=True)
                    if not len(audio) or not np.isfinite(audio).all():
                        raise ValueError(f"Invalid audio: {raw}")
                    peak = np.max(np.abs(audio))
                    if peak <= 0:
                        raise ValueError(f"Silent audio: {raw}")
                    cleaned = args.output / f"{emotion}_{pair.name}_{label}.wav"
                    sf.write(cleaned, audio / peak * 0.95, 22050)
                    paths.append(cleaned)
                with torch.inference_mode():
                    n = model.get_conditioning_latents(audio_path=[str(paths[0])])[0]
                    e = model.get_conditioning_latents(audio_path=[str(paths[1])])[0]
                if e.shape != cond.shape or n.shape != cond.shape:
                    raise ValueError("Pair latent shape differs from target latent")
                deltas.append(e - n)
            if not deltas:
                raise ValueError(f"No pairs for {emotion}")
            runs.append((emotion, 3.0, torch.stack(deltas).mean(dim=0)))
    rows = []
    for name, alpha, delta in runs:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        with torch.inference_mode():
            result = model.inference(args.text, "hi", cond + alpha * delta, speaker,
                                     temperature=0.7, repetition_penalty=2.0, top_k=50, top_p=0.8)
        wav = args.output / f"{name}.wav"
        sf.write(wav, result["wav"], config.audio.output_sample_rate)
        rows.append(dict(emotion=name, alpha=alpha, seed=args.seed, speaker=reference.name,
                         text=args.text, audio=wav.name, duration_s=len(result["wav"])/config.audio.output_sample_rate))
        (args.output / "synthesis.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    # Synthesis outputs are retained even if ASR initialization or scoring fails.
    import whisper
    from jiwer import cer, wer
    import librosa
    asr = whisper.load_model("tiny", device="cpu")
    with (args.output / "baseline.csv").open("w", encoding="utf-8", newline="") as dest:
        writer = csv.DictWriter(dest, fieldnames=list(rows[0]) + ["transcript", "wer", "cer"])
        writer.writeheader()
        for row in rows:
            audio, _ = librosa.load(args.output / row["audio"], sr=16000, mono=True)
            transcript = asr.transcribe(audio, language="hi", fp16=False)["text"]
            ref, hyp = normalize(args.text), normalize(transcript)
            writer.writerow(dict(row, transcript=transcript, wer=wer(ref, hyp), cer=cer(ref, hyp)))
            dest.flush()


if __name__ == "__main__":
    main()
