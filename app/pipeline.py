"""
End-to-end pipeline: voice note -> ASR transcript -> parser entries -> ledger.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from app.asr import transcribe
from app.db import DEFAULT_DB_PATH
from app.ledger import add_entries, get_summary, list_entries
from app.parser import parse_transcript


def process_voice_note(
    audio_path: str,
    language: str,
    user_id: Optional[int] = None,
    *,
    db_path=DEFAULT_DB_PATH,
) -> dict[str, Any]:
    """Run ASR -> parse -> persist to ledger -> return summary view.

    Returns:
        {
          "user_id": int,
          "transcript": str,
          "entries": list[dict],   (saved rows with ids),
          "summary": dict           7-day summary,
        }
    """
    audio_path = str(Path(audio_path).expanduser().resolve())
    if not Path(audio_path).is_file():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    transcript = transcribe(audio_path, language)
    parsed = parse_transcript(transcript)
    saved = add_entries(
        user_id=user_id,
        entries=parsed,
        transcript=transcript,
        audio_file=Path(audio_path).name,
        db_path=db_path,
    )
    # Ensure we have the new ids; now the summary over last 7 days starting today
    uid = saved[0]["user_id"] if saved else (user_id if user_id is not None else 1)
    if not saved:
        # Still make sure user row exists for later ledger() call
        from app.db import get_or_create_user
        uid = get_or_create_user(
            db_path,
            user_id,
            phone_or_name=(f"user_{user_id}" if user_id else "unknown"),
            language=language or "en",
        )

    summary = get_summary(user_id=uid, db_path=db_path)

    return {
        "user_id": uid,
        "transcript": transcript,
        "entries": saved,
        "summary": summary,
    }
