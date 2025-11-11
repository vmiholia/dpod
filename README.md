# dpod

Single-entry Python CLI to run a complete YouTube diarized transcription pipeline on a GPU VM, mirroring the helpers in `ac0Nm71GpOY`.

Pipeline steps
- Download best audio and convert to 16 kHz mono WAV via `yt-dlp`
- Transcribe with `faster-whisper` (writes `audio.transcript.txt`, `audio.segments.csv`)
- Diarize with `pyannote.audio` (writes `audio.rttm`, `audio.diarization.txt`)
- Merge transcript + diarization (writes `merged_transcript.txt`)

The script prints verbose status and per-step timings, with periodic progress during long steps.

## Requirements
- Python 3.10+
- ffmpeg (for audio conversion)
- yt-dlp
- PyTorch (CUDA build recommended on GPU instances)
- faster-whisper
- pyannote.audio (requires a Hugging Face token for `pyannote/speaker-diarization`)

Example install (GPU VM):
```
python -m venv ~/av && source ~/av/bin/activate
pip install --upgrade pip
pip install yt-dlp faster-whisper pyannote.audio torch --extra-index-url https://download.pytorch.org/whl/cu121
# ensure ffmpeg is installed via system package manager
```

## Usage
Simplest form (defaults mirror helpers):
```
python dpod.py --url https://www.youtube.com/watch?v=ac0Nm71GpOY
```

Optional flags (all have sensible defaults):
```
  --device {cuda,cpu}          # default: cuda (falls back to cpu if unavailable)
  --model base                 # faster-whisper model
  --interval 10                # progress interval seconds
  --hf-token $HF_TOKEN         # or rely on cached HF login
```

Outputs are written under `~/runs/<video_id>/`:
- `audio.wav`
- `audio.transcript.txt`
- `audio.segments.csv`
- `audio.rttm`
- `audio.diarization.txt`
- `merged_transcript.txt`

## Notes
- If `--device cuda` is set but CUDA is not available, the script falls back to CPU.
- Hugging Face token can be provided via `--hf-token` or env vars: `HF_TOKEN`, `HUGGINGFACE_TOKEN`, `HUGGINGFACE_HUB_TOKEN`.
- The logging style and file naming mirror the helper scripts in `ac0Nm71GpOY`.
