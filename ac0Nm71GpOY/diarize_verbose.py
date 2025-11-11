#!/usr/bin/env python3
import argparse
import datetime as dt
import threading
import time
from pathlib import Path

import torch
from pyannote.audio import Pipeline


def format_hms(seconds: float) -> str:
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(int(minutes), 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:04.1f}"


def status_line(stage: str, elapsed: float) -> str:
    gpu = "GPU not available"
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        used = torch.cuda.memory_allocated() / (1024 ** 2)
        total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)
        gpu = f"GPU mem {used:.0f}/{total:.0f} MiB"
    return f"[{dt.datetime.now():%H:%M:%S}] {stage}... elapsed {format_hms(elapsed)} ({gpu})"


def run_with_progress(fn, label: str, interval: int = 10):
    result = {"done": False, "value": None, "error": None}

    def worker():
        try:
            result["value"] = fn()
        except Exception as exc:
            result["error"] = exc
        finally:
            result["done"] = True

    thread = threading.Thread(target=worker, daemon=True)
    start = time.perf_counter()
    thread.start()

    while not result["done"]:
        print(status_line(label, time.perf_counter() - start), flush=True)
        time.sleep(interval)

    thread.join()
    elapsed = time.perf_counter() - start
    if result["error"] is not None:
        raise result["error"]
    print(f"{label} finished in {format_hms(elapsed)}", flush=True)
    return result["value"]


def main():
    parser = argparse.ArgumentParser(description="Verbose pyannote diarization runner.")
    parser.add_argument("--audio", required=True, help="Path to the WAV file to diarize.")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="Inference device.")
    parser.add_argument("--interval", type=int, default=10, help="Status update interval (seconds).")
    args = parser.parse_args()

    audio_path = Path(args.audio).expanduser().resolve()
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    print(f"Loading diarization pipeline on {args.device} ...", flush=True)
    target_device = torch.device(args.device)
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization").to(target_device)

    print(f"Starting diarization for {audio_path.name}", flush=True)
    diarization = run_with_progress(
        lambda: pipeline({"audio": str(audio_path)}),
        label="Diarization",
        interval=args.interval,
    )

    # Write outputs
    rttm_path = audio_path.with_suffix(".rttm")
    txt_path = audio_path.with_suffix(".diarization.txt")

    print(f"Writing RTTM to {rttm_path}", flush=True)
    with open(rttm_path, "w") as rttm:
        diarization.write_rttm(rttm)

    print(f"Writing readable report to {txt_path}", flush=True)
    with open(txt_path, "w") as txt:
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            txt.write(f"{turn.start:.2f}–{turn.end:.2f}s: {speaker}\n")

    print("Done. Sample lines:")
    with open(txt_path) as txt:
        for _ in range(5):
            line = txt.readline().strip()
            if not line:
                break
            print("  ", line)


if __name__ == "__main__":
    main()
