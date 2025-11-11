#!/usr/bin/env python3
"""
Transcribe audio with faster-whisper while printing periodic progress updates.
"""

import argparse
import csv
import datetime as dt
import math
import time
from pathlib import Path

from faster_whisper import WhisperModel


def format_hms(seconds: float) -> str:
    total_ms = int(round(seconds * 1000))
    hours, remainder = divmod(total_ms, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


def status(stage: str, elapsed: float, processed: float | None, total: float | None) -> str:
    pieces = [f"[{dt.datetime.now():%H:%M:%S}] {stage}", f"elapsed {format_hms(elapsed)}"]
    if processed is not None and total and total > 0:
        pct = min(100.0, processed / total * 100)
        remaining = (elapsed / processed * (total - processed)) if processed else math.nan
        eta = format_hms(remaining) if math.isfinite(remaining) and remaining >= 0 else "??:??:??.???"
        pieces.append(f"{pct:5.1f}% done (ETA {eta})")
    return " | ".join(pieces)


def transcribe_with_progress(
    audio_path: Path,
    model_size: str,
    device: str,
    compute_type: str,
    beam_size: int,
    language: str | None,
    interval: int,
    output_prefix: str,
) -> None:
    print(f"Loading faster-whisper model '{model_size}' on {device} (compute_type={compute_type})...")
    model = WhisperModel(model_size, device=device, compute_type=compute_type)

    print(f"Starting transcription of {audio_path.name}")
    start = time.perf_counter()
    last_report = start

    segments, info = model.transcribe(
        str(audio_path),
        beam_size=beam_size,
        language=language,
    )

    duration = info.duration or 0.0
    processed_audio = 0.0
    segment_count = 0

    transcript_path = audio_path.with_suffix(f".{output_prefix or 'transcript'}.txt")
    csv_path = audio_path.with_suffix(f".{output_prefix or 'segments'}.csv")

    with transcript_path.open("w") as txt, csv_path.open("w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["start", "end", "text"])

        for segment in segments:
            segment_count += 1
            processed_audio = max(processed_audio, segment.end)

            text = segment.text.strip()
            txt.write(f"[{segment.start:.2f}–{segment.end:.2f}] {text}\n")
            writer.writerow([f"{segment.start:.2f}", f"{segment.end:.2f}", text])

            now = time.perf_counter()
            if now - last_report >= interval:
                print(status("Transcribing", now - start, processed_audio, duration))
                last_report = now

    total_elapsed = time.perf_counter() - start
    print(status("Transcribing", total_elapsed, processed_audio, duration))
    print(f"Transcription complete: {segment_count} segments")
    print(f"Transcript: {transcript_path}")
    print(f"Segments CSV: {csv_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Transcribe audio with periodic progress display.")
    parser.add_argument("--audio", required=True, type=Path, help="Path to the WAV file to transcribe.")
    parser.add_argument("--model", default="base", help="faster-whisper model size or path (default: base).")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="Inference device (default: cuda).")
    parser.add_argument("--compute-type", default="float16", help="Compute type (default: float16).")
    parser.add_argument("--beam-size", type=int, default=5, help="Beam size for decoding (default: 5).")
    parser.add_argument("--language", default="en", help="Language code or leave blank for auto-detect.")
    parser.add_argument("--interval", type=int, default=10, help="Progress update interval in seconds (default: 10).")
    parser.add_argument("--output-prefix", default="", help="Optional suffix inserted before file extensions.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    audio_path = args.audio.expanduser().resolve()
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    transcribe_with_progress(
        audio_path=audio_path,
        model_size=args.model,
        device=args.device,
        compute_type=args.compute_type,
        beam_size=args.beam_size,
        language=args.language or None,
        interval=args.interval,
        output_prefix=args.output_prefix.strip(),
    )


if __name__ == "__main__":
    main()
