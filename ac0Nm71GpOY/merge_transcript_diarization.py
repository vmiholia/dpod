#!/usr/bin/env python3
"""
Merge faster-whisper transcript segments with pyannote diarization.

Inputs
------
- segments.csv produced by faster-whisper (columns: start,end,text)
- diarization.rttm produced by pyannote (SPEAKER lines)

Output
------
- merged_transcript.txt with lines:
    HH:MM:SS.mmm → HH:MM:SS.mmm [Speaker] text
"""

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Tuple


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


def format_ts(seconds: float) -> str:
    total_ms = int(round(seconds * 1000))
    hours, remainder = divmod(total_ms, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


def merge(transcript_csv: Path, diar_rttm: Path, output_path: Path) -> None:
    spans_by_speaker = read_rttm(diar_rttm)
    with output_path.open("w") as out:
        for span, text in read_segments(transcript_csv):
            speaker = dominant_speaker(spans_by_speaker, span)
            out.write(
                f"{format_ts(span.start)} -> {format_ts(span.end)} "
                f"[{speaker}] {text}\n"
            )


def locate_default(pattern: str) -> Path | None:
    matches = sorted(Path.cwd().glob(pattern))
    return matches[0] if matches else None


def main() -> None:
    base = Path.cwd()

    parser = argparse.ArgumentParser(description="Merge transcript segments with diarization.")
    parser.add_argument("--segments", type=Path, default=base / "segments.csv",
                        help="CSV with columns start,end,text (default: segments.csv).")
    parser.add_argument("--rttm", type=Path, default=None,
                        help="RTTM diarization file (default: first *.rttm in the folder or diarization.rttm).")
    parser.add_argument("--output", type=Path, default=base / "merged_transcript.txt",
                        help="Output merged transcript file (default: merged_transcript.txt).")
    args = parser.parse_args()

    transcript_csv = args.segments.expanduser()
    if not transcript_csv.exists():
        raise FileNotFoundError(f"Transcript CSV not found: {transcript_csv}")

    if args.rttm is not None:
        diar_rttm = args.rttm.expanduser()
    else:
        diar_rttm = (base / "diarization.rttm")
        if not diar_rttm.exists():
            fallback = locate_default("*.rttm")
            if fallback is None:
                raise FileNotFoundError(
                    "Could not find an RTTM file. Pass --rttm path/to/file.rttm."
                )
            diar_rttm = fallback

    if not diar_rttm.exists():
        raise FileNotFoundError(f"Diarization RTTM not found: {diar_rttm}")

    output_path = args.output.expanduser()

    merge(transcript_csv, diar_rttm, output_path)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
