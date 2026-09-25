#!/usr/bin/env python3
"""
End-to-end smoke test: voice note -> ASR -> intent routing -> ledger.

Usage:
    python scripts/try_pipeline.py path/to/audio.ogg -l yo
    python scripts/try_pipeline.py voice_note.opus -l ig --user-id 7
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    # Yoruba/Hausa/Igbo transcripts carry diacritics (ọ, ẹ, ṣ...); the default
    # Windows console encoding (cp1252) cannot print them.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(
        description="Run the full Veyra pipeline on an audio file and "
                    "print the reply, saved entries and 7-day summary."
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
        help="Language code or name (e.g. yo, ha, ig, en, pidgin). "
             "See LANGUAGE_MODEL_CONFIG in app/asr.py.",
    )
    parser.add_argument(
        "--user-id",
        type=int,
        default=None,
        help="Existing user id to book against; omit to create a new user.",
    )
    parser.add_argument(
        "--db",
        type=str,
        default=None,
        help="SQLite database path (default: ledger.db in the project root).",
    )
    args = parser.parse_args()

    # Make sure we can import from the project root even if run as script
    project_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(project_root))

    from app.asr import language_notice  # noqa: PLC0415
    from app.db import DEFAULT_DB_PATH, init_db  # noqa: PLC0415
    from app.pipeline import process_voice_note  # noqa: PLC0415

    audio = Path(args.audio_path).expanduser().resolve()
    if not audio.is_file():
        parser.error(f"Audio file not found: {audio}")

    db_path = Path(args.db).expanduser().resolve() if args.db else DEFAULT_DB_PATH
    # The FastAPI startup hook runs init_db; a bare CLI run needs it too.
    init_db(db_path)

    print(
        f"Running pipeline on {audio.name} (language={args.language}, "
        f"db={db_path.name}). First ASR run downloads the model, so it may "
        f"take a while..."
    )
    notice = language_notice(args.language)
    if notice:
        print(notice)
    print("-" * 60)
    try:
        result = process_voice_note(
            audio_path=str(audio),
            language=args.language,
            user_id=args.user_id,
            db_path=db_path,
        )
    except (ValueError, FileNotFoundError) as exc:
        print(f"Error: {exc}")
        return 1

    print(f"Transcript : {result['transcript']!r}")
    print(f"Reply      : {result['reply_text']}")
    print(f"User id    : {result['user_id']}")
    print("-" * 60)
    print(f"Entries saved ({len(result['entries'])}):")
    print(json.dumps(result["entries"], indent=2, ensure_ascii=False))
    print("-" * 60)
    print("7-day summary:")
    print(json.dumps(result["summary"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
