"""Measure the archived generated WAVs referenced by the experiment log."""
import argparse
import array
import csv
import math
import sys
import wave
from pathlib import Path


def inspect(outputs):
    with (outputs / "comprehensive_metrics_log.csv").open(encoding="utf-8-sig") as source:
        for row in csv.DictReader(source):
            path = outputs / "generated" / row["Audio_File"]
            record = {k: row[k] for k in ("Audio_File", "Emotion", "Alpha", "Mode", "Speaker")}
            try:
                with wave.open(str(path), "rb") as wav:
                    if wav.getsampwidth() != 2:
                        raise ValueError("Expected PCM16")
                    samples = array.array("h", wav.readframes(wav.getnframes()))
                    if sys.byteorder == "big":
                        samples.byteswap()
                    rms = math.sqrt(sum(x*x for x in samples) / len(samples))
                    record.update(duration_s=round(wav.getnframes()/wav.getframerate(), 4),
                                  rms_dbfs=round(20*math.log10(rms/32768), 3) if rms else "-inf",
                                  sample_rate=wav.getframerate(), status="ok")
            except (OSError, wave.Error, ValueError, EOFError, ZeroDivisionError) as error:
                record["status"] = str(error)
            yield record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("outputs", type=Path)
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    with args.report.open("w", newline="", encoding="utf-8") as report:
        writer = csv.DictWriter(report, fieldnames=["Audio_File","Emotion","Alpha","Mode","Speaker","duration_s","rms_dbfs","sample_rate","status"])
        writer.writeheader()
        writer.writerows(inspect(args.outputs))
