"""
End-to-end pipeline: voice note -> ASR transcript -> intent routing -> ledger.

`process_voice_note` first asks app.intents what the trader meant, then routes:

  - entry           -> parser -> ledger entries
  - correction      -> void the last entry, insert a corrected copy
  - delete_last     -> void the last entry
  - debt_owed_to_me -> open debt (person owes the trader)
  - debt_i_owe      -> open debt (trader owes the person)
  - debt_paid       -> settle the oldest matching open debt

Every path returns a `reply_text` (what to say back on WhatsApp), and the
unclear path writes NOTHING to the database.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from app.asr import transcribe
from app.db import DEFAULT_DB_PATH
from app.intents import (
    classify_message,
    extract_amount,
    extract_person,
)
from app.ledger import (
    add_debt,
    add_entries,
    correct_last_entry,
    delete_last_entry,
    get_summary,
    mark_debt_paid,
)
from app.parser import parse_transcript


# ---------------------------------------------------------------------------
# Canned replies
# ---------------------------------------------------------------------------
UNCLEAR_REPLY = (
    "I didn't quite catch that. Please say it again, for example: "
    "\"I sold 5 bags of rice for 45k\" or \"Mama Ngozi owes me 5k\"."
)
NO_USER_REPLY = (
    "I need to know whose book to update. Please send your user id with the note."
)
NO_AMOUNT_REPLY = "I couldn't find the new amount. Try: \"no, it was 4k\"."
NO_ENTRY_TO_CORRECT_REPLY = (
    "I couldn't find an earlier entry to correct. "
    "Please record the sale or expense first."
)
NO_ENTRY_TO_REMOVE_REPLY = "I couldn't find an earlier entry to remove."
DEBT_MISSING_DETAILS_REPLY = (
    "I couldn't find the person and the amount. "
    "Try: \"Mama Ngozi owes me 5k\"."
)
NO_DEBT_PERSON_REPLY = "I couldn't tell who paid. Try: \"Mama Ngozi don pay\"."


def format_naira(amount: int) -> str:
    """45000 -> "45,000"."""
    return f"{int(amount):,}"


def _empty_summary() -> dict[str, Any]:
    return {
        "total_sales": 0,
        "total_expenses": 0,
        "profit": 0,
        "top_sale_item": None,
        "top_expense_item": None,
        "total_owed_to_me": 0,
        "total_i_owe": 0,
    }


def _entry_line(entry: dict[str, Any]) -> str:
    return (
        f"{entry['item']}, {format_naira(entry['amount'])} naira, {entry['type']}"
    )


def _entries_reply(entries: list[dict[str, Any]]) -> str:
    if len(entries) == 1:
        return f"Recorded: {_entry_line(entries[0])}."
    joined = "; ".join(_entry_line(e) for e in entries)
    return f"Recorded {len(entries)} entries: {joined}."


def _response(
    transcript: str,
    uid: Optional[int],
    db_path,
    reply: str,
    entries: Optional[list[dict[str, Any]]] = None,
    *,
    saved: bool,
) -> dict[str, Any]:
    """Build the pipeline result.

    `saved=True` means the action wrote to the ledger, so we return the real
    summary; otherwise nothing was persisted and the summary is zeroed (and
    no DB reads happen, so an unclear note never even creates a user row).
    """
    if saved and uid is not None:
        summary = get_summary(user_id=uid, db_path=db_path)
    else:
        summary = _empty_summary()
    return {
        "user_id": uid,
        "transcript": transcript,
        "entries": entries or [],
        "summary": summary,
        "reply_text": reply,
    }


def process_voice_note(
    audio_path: str,
    language: str,
    user_id: Optional[int] = None,
    *,
    db_path=None,
) -> dict[str, Any]:
    """Run ASR -> classify -> route -> persist, and build the reply text.

    Returns:
        {
          "user_id": int | None,
          "transcript": str,
          "entries": list[dict],   (rows touched by this action)
          "summary": dict          (7-day summary + open-debt totals),
          "reply_text": str        (what to send back to the trader),
        }
    """
    if db_path is None:
        db_path = DEFAULT_DB_PATH
    audio_path = str(Path(audio_path).expanduser().resolve())
    if not Path(audio_path).is_file():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    transcript = transcribe(audio_path, language)
    audio_name = Path(audio_path).name
    intent = classify_message(transcript)

    # ------------------------------------------------------------------
    # Corrections / deletes: need a known user and an earlier entry
    # ------------------------------------------------------------------
    if intent == "correction":
        if user_id is None:
            return _response(transcript, None, db_path, NO_USER_REPLY, saved=False)
        new_amount = extract_amount(transcript)
        if new_amount is None:
            return _response(
                transcript, user_id, db_path, NO_AMOUNT_REPLY, saved=False
            )
        updated = correct_last_entry(user_id, new_amount, db_path=db_path)
        if updated is None:
            return _response(
                transcript, user_id, db_path, NO_ENTRY_TO_CORRECT_REPLY, saved=False
            )
        reply = (
            f"Corrected: the last entry is now "
            f"{format_naira(updated['amount'])} naira."
        )
        return _response(
            transcript, updated["user_id"], db_path, reply, [updated], saved=True
        )

    if intent == "delete_last":
        if user_id is None:
            return _response(transcript, None, db_path, NO_USER_REPLY, saved=False)
        removed = delete_last_entry(user_id, db_path=db_path)
        if removed is None:
            return _response(
                transcript, user_id, db_path, NO_ENTRY_TO_REMOVE_REPLY, saved=False
            )
        return _response(
            transcript,
            removed["user_id"],
            db_path,
            "Removed the last entry.",
            [removed],
            saved=True,
        )

    # ------------------------------------------------------------------
    # Debts
    # ------------------------------------------------------------------
    if intent in ("debt_owed_to_me", "debt_i_owe"):
        amount = extract_amount(transcript)
        person = extract_person(transcript, intent)
        if amount is None or person is None:
            return _response(
                transcript, user_id, db_path, DEBT_MISSING_DETAILS_REPLY, saved=False
            )
        direction = "owed_to_me" if intent == "debt_owed_to_me" else "i_owe"
        debt = add_debt(user_id, person, amount, direction, db_path=db_path)
        if direction == "owed_to_me":
            reply = f"Noted: {person} owes you {format_naira(amount)} naira."
        else:
            reply = f"Noted: you owe {person} {format_naira(amount)} naira."
        return _response(transcript, debt["user_id"], db_path, reply, saved=True)

    if intent == "debt_paid":
        if user_id is None:
            return _response(transcript, None, db_path, NO_USER_REPLY, saved=False)
        person = extract_person(transcript, intent)
        if person is None:
            return _response(
                transcript, user_id, db_path, NO_DEBT_PERSON_REPLY, saved=False
            )
        settled = mark_debt_paid(user_id, person, db_path=db_path)
        if settled is None:
            reply = f"I couldn't find an open debt for {person}."
            return _response(transcript, user_id, db_path, reply, saved=False)
        reply = (
            f"Noted: {person}'s debt of {format_naira(settled['amount'])} naira "
            f"is marked as paid."
        )
        return _response(
            transcript, settled["user_id"], db_path, reply, saved=True
        )

    # ------------------------------------------------------------------
    # New sale / expense (the default), or nothing understood
    # ------------------------------------------------------------------
    parsed = parse_transcript(transcript)
    if not parsed:
        return _response(transcript, user_id, db_path, UNCLEAR_REPLY, saved=False)

    saved_entries = add_entries(
        user_id=user_id,
        entries=parsed,
        transcript=transcript,
        audio_file=audio_name,
        db_path=db_path,
    )
    return _response(
        transcript,
        saved_entries[0]["user_id"],
        db_path,
        _entries_reply(saved_entries),
        saved_entries,
        saved=True,
    )
