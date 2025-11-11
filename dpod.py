#!/usr/bin/env python3
"""
End-to-end YouTube diarized transcription pipeline (single entrypoint).

Steps (mirrors local helpers in ac0Nm71GpOY):
- Download best audio and convert to 16 kHz mono WAV via yt-dlp
- Transcribe with faster-whisper (writes audio.transcript.txt, audio.segments.csv)
- Diarize with pyannote (writes audio.rttm, audio.diarization.txt)
- Merge transcript + diarization (writes merged_transcript.txt)

The script prints verbose status and timings for each step, including periodic
progress and GPU memory stats during diarization, similar to the helper scripts.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import math
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Tuple

import torch
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline


# ---------------------------- Formatting helpers ---------------------------- #

def format_hms_ms(seconds: float) -> str:
    total_ms = int(round(seconds * 1000))
    hours, remainder = divmod(total_ms, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


def transcribe_status(stage: str, elapsed: float, processed: float | None, total: float | None) -> str:
    pieces = [f"[{dt.datetime.now():%H:%M:%S}] {stage}", f"elapsed {format_hms_ms(elapsed)}"]
    if processed is not None and total and total > 0:
        pct = min(100.0, processed / total * 100)
        remaining = (elapsed / processed * (total - processed)) if processed else math.nan
        eta = format_hms_ms(remaining) if math.isfinite(remaining) and remaining >= 0 else "??:??:??.???"
        pieces.append(f"{pct:5.1f}% done (ETA {eta})")
    return " | ".join(pieces)


def diar_status_line(stage: str, elapsed: float) -> str:
    gpu = "GPU not available"
    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        try:
            used = torch.cuda.memory_allocated() / (1024 ** 2)
            total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)
            gpu = f"GPU mem {used:.0f}/{total:.0f} MiB"
        except Exception:
            gpu = "GPU available"
    return f"[{dt.datetime.now():%H:%M:%S}] {stage}... elapsed {format_hms_ms(elapsed)} ({gpu})"


# --------------------------- External command utils ------------------------- #

def run(cmd: list[str], cwd: Path | None = None, capture: bool = False, check: bool = True) -> subprocess.CompletedProcess:
    if not capture:
        print(f"$ {' '.join(cmd)}", flush=True)
    return subprocess.run(
        cmd,
        cwd=str(cwd) if cwd is not None else None,
        check=check,
        capture_output=capture,
        text=True,
    )


def get_video_id(url: str) -> str:
    # Use yt-dlp to resolve the canonical video ID.
    try:
        res = run(["yt-dlp", "--get-id", url], capture=True)
        vid = res.stdout.strip().splitlines()[0].strip()
        if vid:
            return vid
    except Exception:
        pass
    # Fallback: parse JSON and read "id" field
    res = run(["yt-dlp", "-J", url], capture=True)
    import json

    data = json.loads(res.stdout)
    if isinstance(data, dict) and "id" in data and data["id"]:
        return str(data["id"])
    raise RuntimeError("Could not determine YouTube video ID.")


# ------------------------------- Pipeline steps ----------------------------- #

def step_download_audio(url: str, runs_root: Path, interval: int) -> tuple[Path, Path, str, float]:
    start = time.perf_counter()
    vid = get_video_id(url)
    run_dir = (runs_root / vid).expanduser().resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading audio for {vid} into {run_dir}")
    cmd = [
        "yt-dlp",
        "-f",
        "bestaudio",
        "--extract-audio",
        "--audio-format",
        "wav",
        "--postprocessor-args",
        "-ar 16000 -ac 1",
        url,
        "-o",
        "%(id)s.%(ext)s",
    ]

    last_report = start
    proc = subprocess.Popen(cmd, cwd=str(run_dir), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    assert proc.stdout is not None
    for line in proc.stdout:
        now = time.perf_counter()
        if now - last_report >= interval:
            print(diar_status_line("Downloading", now - start), flush=True)
            last_report = now
        # Stream yt-dlp output sparsely to keep context
        sys.stdout.write(line)
        sys.stdout.flush()
    ret = proc.wait()
    if ret != 0:
        raise RuntimeError(f"yt-dlp failed with exit code {ret}")

    wav_path = run_dir / f"{vid}.wav"
    if not wav_path.exists():
        # Some backends may emit m4a -> re-run convert via ffmpeg using the downloaded file
        alt = None
        for p in run_dir.glob(f"{vid}.*"):
            if p.suffix.lower() != ".wav" and p.is_file():
                alt = p
                break
        if alt is None:
            raise FileNotFoundError(f"Expected {wav_path} (or any {vid}.*) not found")
        # Convert to target WAV
        converted = run_dir / f"{vid}.wav"
        run(["ffmpeg", "-y", "-i", str(alt), "-ar", "16000", "-ac", "1", str(converted)])
        wav_path = converted

    audio_path = run_dir / "audio.wav"
    if audio_path.exists():
        audio_path.unlink()
    wav_path.rename(audio_path)

    elapsed = time.perf_counter() - start
    print(f"Download+convert finished in {format_hms_ms(elapsed)}")
    return run_dir, audio_path, vid, elapsed


def step_transcribe(
    audio_path: Path,
    model_size: str,
    device: str,
    compute_type: str,
    beam_size: int,
    language: str | None,
    interval: int,
) -> tuple[Path, Path, float]:
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

    transcript_path = audio_path.with_suffix(".transcript.txt")
    csv_path = audio_path.with_suffix(".segments.csv")

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
                print(transcribe_status("Transcribing", now - start, processed_audio, duration))
                last_report = now

    total_elapsed = time.perf_counter() - start
    print(transcribe_status("Transcribing", total_elapsed, processed_audio, duration))
    print(f"Transcription complete: {segment_count} segments")
    print(f"Transcript: {transcript_path}")
    print(f"Segments CSV: {csv_path}")
    return transcript_path, csv_path, total_elapsed


def _run_with_progress(fn, label: str, interval: int = 10):
    result: dict[str, object | None] = {"done": False, "value": None, "error": None}

    def worker():
        try:
            result["value"] = fn()
        except Exception as exc:  # noqa: BLE001
            result["error"] = exc
        finally:
            result["done"] = True

    thread = threading.Thread(target=worker, daemon=True)
    start = time.perf_counter()
    thread.start()

    while not result["done"]:
        print(diar_status_line(label, time.perf_counter() - start), flush=True)
        time.sleep(interval)

    thread.join()
    elapsed = time.perf_counter() - start
    if result["error"] is not None:
        raise result["error"]  # type: ignore[misc]
    print(f"{label} finished in {format_hms_ms(elapsed)}", flush=True)
    return result["value"], elapsed


def step_diarize(audio_path: Path, device: str, interval: int, hf_token: str | None) -> tuple[Path, Path, float]:
    print(f"Loading diarization pipeline on {device} ...", flush=True)
    target_device = torch.device(device)

    # Support HF token via arg or env (HF_TOKEN / HUGGINGFACE_TOKEN / HUGGINGFACE_HUB_TOKEN)
    env_token = (
        hf_token
        or os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGINGFACE_TOKEN")
        or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    )

    if env_token:
        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token=env_token).to(target_device)
    else:
        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization").to(target_device)

    print(f"Starting diarization for {audio_path.name}", flush=True)
    diarization, elapsed = _run_with_progress(
        lambda: pipeline({"audio": str(audio_path)}),
        label="Diarization",
        interval=interval,
    )

    # Write outputs
    rttm_path = audio_path.with_suffix(".rttm")
    txt_path = audio_path.with_suffix(".diarization.txt")

    print(f"Writing RTTM to {rttm_path}", flush=True)
    with open(rttm_path, "w") as rttm:
        diarization.write_rttm(rttm)  # type: ignore[attr-defined]

    print(f"Writing readable report to {txt_path}", flush=True)
    with open(txt_path, "w") as txt:
        for turn, _, speaker in diarization.itertracks(yield_label=True):  # type: ignore[attr-defined]
            txt.write(f"{turn.start:.2f}–{turn.end:.2f}s: {speaker}\n")

    print("Done. Sample lines:")
    with open(txt_path) as txt:
        for _ in range(5):
            line = txt.readline().strip()
            if not line:
                break
            print("  ", line)

    return rttm_path, txt_path, float(elapsed)


# ------------------------------- Merge helpers ------------------------------ #

@dataclass(frozen=True)
class Span:
    start: float
    end: float

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


def read_segments(path: Path) -> Iterable[Tuple[Span, str]]:
    with path.open() as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            start = float(row["start"])
            end = float(row["end"])
            text = row["text"].strip()
            if not text:
                continue
            yield Span(start, end), text


def read_rttm(path: Path) -> Dict[str, Iterable[Span]]:
    assignments: Dict[str, list[Span]] = {}
    with path.open() as fh:
        for line in fh:
            if not line.startswith("SPEAKER"):
                continue
            parts = line.split()
            if len(parts) < 8:
                continue
            start = float(parts[3])
            duration = float(parts[4])
            speaker = parts[7]
            assignments.setdefault(speaker, []).append(Span(start, start + duration))
    return assignments


def dominant_speaker(spans_by_speaker: Dict[str, Iterable[Span]], segment: Span) -> str:
    best_label = "UNKNOWN"
    best_overlap = 0.0
    for label, spans in spans_by_speaker.items():
        overlap = 0.0
        for span in spans:
            start = max(segment.start, span.start)
            end = min(segment.end, span.end)
            if end <= start:
                continue
            overlap += end - start
        if overlap > best_overlap:
            best_label = label
            best_overlap = overlap
    return best_label


def step_merge(transcript_csv: Path, diar_rttm: Path, output_path: Path) -> float:
    start = time.perf_counter()
    spans_by_speaker = read_rttm(diar_rttm)
    with output_path.open("w") as out:
        for span, text in read_segments(transcript_csv):
            speaker = dominant_speaker(spans_by_speaker, span)
            out.write(
                f"{format_hms_ms(span.start)} -> {format_hms_ms(span.end)} "
                f"[{speaker}] {text}\n"
            )
    elapsed = time.perf_counter() - start
    print(f"Merge finished in {format_hms_ms(elapsed)}")
    return elapsed


# ----------------------------------- CLI ----------------------------------- #

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YouTube diarized transcription: download -> transcribe -> diarize -> merge")
    parser.add_argument("--url", required=True, help="YouTube URL")
    parser.add_argument("--runs-root", default="~/runs", help="Root directory for runs (default: ~/runs)")

    # Transcription
    parser.add_argument("--model", default="base", help="faster-whisper model (default: base)")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="Inference device (default: cuda)")
    parser.add_argument("--compute-type", default="float16", help="faster-whisper compute type (default: float16)")
    parser.add_argument("--beam-size", type=int, default=5, help="Beam size (default: 5)")
    parser.add_argument("--language", default="en", help="Language (default: en)")
    parser.add_argument("--interval", type=int, default=10, help="Progress update interval seconds (default: 10)")

    # Diarization auth + options
    parser.add_argument("--hf-token", default=None, help="Hugging Face token (or set HF_TOKEN/HUGGINGFACE_TOKEN/HUGGINGFACE_HUB_TOKEN)")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    runs_root = Path(args.runs_root).expanduser()
    runs_root.mkdir(parents=True, exist_ok=True)

    # Auto-fallback to CPU if user requested CUDA but it's not available.
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("WARNING: CUDA not available; falling back to CPU.")
        device = "cpu"

    total_start = time.perf_counter()

    # 1) Download + convert
    run_dir, audio_path, vid, t_download = step_download_audio(args.url, runs_root, args.interval)

    # 2) Transcribe
    transcript_txt, segments_csv, t_trans = step_transcribe(
        audio_path=audio_path,
        model_size=args.model,
        device=device,
        compute_type=args.compute_type,
        beam_size=args.beam_size,
        language=args.language or None,
        interval=args.interval,
    )

    # 3) Diarize (pyannote)
    rttm_path, diar_txt, t_diar = step_diarize(audio_path, device=device, interval=args.interval, hf_token=args.hf_token)

    # 4) Merge
    merged_path = run_dir / "merged_transcript.txt"
    t_merge = step_merge(segments_csv, rttm_path, merged_path)

    total_elapsed = time.perf_counter() - total_start

    print("\nOutputs:")
    print(f"- {transcript_txt}")
    print(f"- {segments_csv}")
    print(f"- {rttm_path}")
    print(f"- {diar_txt}")
    print(f"- {merged_path}")

    print("\nTimings:")
    print(f"- Download+convert: {format_hms_ms(t_download)}")
    print(f"- Transcription:   {format_hms_ms(t_trans)}")
    print(f"- Diarization:     {format_hms_ms(t_diar)}")
    print(f"- Merge:           {format_hms_ms(t_merge)}")
    print(f"- Total:           {format_hms_ms(total_elapsed)}")

    # Finished
    print("\nDone.")


if __name__ == "__main__":
    main()
