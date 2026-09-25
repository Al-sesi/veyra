#!/usr/bin/env python3
"""
Quick CLI to try the Veyra ASR module.

Usage:
    python scripts/try_asr.py path/to/audio.ogg --language ha
    python scripts/try_asr.py voice_note.opus -l ig
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Transcribe a WhatsApp / local audio file using Veyra ASR."
    )
    parser.add_argument(
        "audio_path",
        type=str,
        help="Path to the audio file (.ogg, .opus, .wav, .mp3, etc.)",
    )
    parser.add_argument(
        "-l",
        "--language",
        type=str,
        required=True,
        help="Language code or name (e.g. ha, ig, en, hausa, igbo). "
             "See LANGUAGE_MODEL_CONFIG in app/asr.py.",
    )
    args = parser.parse_args()

    # Make sure we can import from the project root even if run as script
    project_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(project_root))

    from app.asr import language_notice, transcribe  # noqa: PLC0415

    audio = Path(args.audio_path).expanduser().resolve()
    if not audio.is_file():
        parser.error(f"Audio file not found: {audio}")

    print(
        f"Transcribing {audio.name} (language={args.language}). "
        f"First run will download the model (~244M params for Whisper Small), "
        f"so it may take a while..."
    )
    notice = language_notice(args.language)
    if notice:
        print(notice)
    print("-" * 60)
    transcript = transcribe(str(audio), args.language)
    print("Transcript:")
    print(transcript)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
