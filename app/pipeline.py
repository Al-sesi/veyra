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

from app.asr import language_notice, transcribe
from app.db import DEFAULT_DB_PATH
from app.intents import (
    classify_message,
    extract_amount,
    extract_person,
    HELP_INTENT,
    MENU_INTENT,
    REQUEST_HISTORY_INTENT,
)
from app.ledger import (
    add_debt,
    add_entries,
    build_ledger_path,
    correct_last_entry,
    delete_last_entry,
    get_summary,
    get_user_by_id,
    mark_debt_paid,
)
from app.parser import parse_transcript


# ---------------------------------------------------------------------------
# Canned replies
# ---------------------------------------------------------------------------

MENU_REPLY = (
    "Here is what I can help with:\n"
    "  • Speak naturally to log a sale or expense, e.g. \"I sold 5 bags of rice for 45k\"\n"
    "  • Say \"my ledger\" for your permanent Excel link (always up to date)\n"
    "  • Say \"who owes me\" or \"who I owe\" to see open debts\n"
    "  • Say \"undo\" to remove your last entry\n"
    "  • Say \"help\" to see this menu again\n"
)

UNCLEAR_REPLY = (
    "I didn't quite catch that. "
    + MENU_REPLY
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

HELP_FAQ_REPLY = (
    "Here is what you need to know about Veyra right now:\n"
    "  • Veyra is your voice-note bookkeeper. Speak sales, expenses and debts "
    "to her in WhatsApp and they show up in your ledger automatically.\n"
    "  • Languages: Veyra understands Nigerian English, Nigerian Pidgin, "
    "Yorùbá, Hausa, and Igbo — Pidgin is experimental and best effort.\n"
    "  • This is a pilot project. The data you record is yours and stays "
    "tied to your phone number, but features may change as we learn.\n"
    "  • Support: there is no human customer care yet. Everything is "
    "automated — say \"help\" or \"menu\" any time and I'll answer directly "
    "with what I can do for you.\n"
)


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
    language_notice_text: Optional[str] = None,
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
        "language_notice": language_notice_text,
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

    notice_text = language_notice(language)
    transcript = transcribe(audio_path, language)
    audio_name = Path(audio_path).name
    intent = classify_message(transcript)

    # ------------------------------------------------------------------
    # Menu / help / request-history: state-read actions that never write
    # (except request_history may return saved=True because a user exists).
    # ------------------------------------------------------------------
    if intent == MENU_INTENT:
        return _response(
            transcript, user_id, db_path, MENU_REPLY, saved=False,
            language_notice_text=notice_text,
        )

    if intent == HELP_INTENT:
        return _response(
            transcript, user_id, db_path, HELP_FAQ_REPLY, saved=False,
            language_notice_text=notice_text,
        )

    # ------------------------------------------------------------------
    # Corrections / deletes: need a known user and an earlier entry
    # ------------------------------------------------------------------
    if intent == "correction":
        if user_id is None:
            return _response(transcript, None, db_path, NO_USER_REPLY, saved=False, language_notice_text=notice_text)
        new_amount = extract_amount(transcript)
        if new_amount is None:
            return _response(
                transcript, user_id, db_path, NO_AMOUNT_REPLY, saved=False, language_notice_text=notice_text
            )
        updated = correct_last_entry(user_id, new_amount, db_path=db_path)
        if updated is None:
            return _response(
                transcript, user_id, db_path, NO_ENTRY_TO_CORRECT_REPLY, saved=False, language_notice_text=notice_text
            )
        reply = (
            f"Corrected: the last entry is now "
            f"{format_naira(updated['amount'])} naira."
        )
        return _response(
            transcript, updated["user_id"], db_path, reply, [updated], saved=True, language_notice_text=notice_text
        )

    if intent == "delete_last":
        if user_id is None:
            return _response(transcript, None, db_path, NO_USER_REPLY, saved=False, language_notice_text=notice_text)
        removed = delete_last_entry(user_id, db_path=db_path)
        if removed is None:
            return _response(
                transcript, user_id, db_path, NO_ENTRY_TO_REMOVE_REPLY, saved=False, language_notice_text=notice_text
            )
        return _response(
            transcript,
            removed["user_id"],
            db_path,
            "Removed the last entry.",
            [removed],
            saved=True,
            language_notice_text=notice_text,
        )

    # ------------------------------------------------------------------
    # Debts
    # ------------------------------------------------------------------
    if intent in ("debt_owed_to_me", "debt_i_owe"):
        amount = extract_amount(transcript)
        person = extract_person(transcript, intent)
        if amount is None or person is None:
            return _response(
                transcript, user_id, db_path, DEBT_MISSING_DETAILS_REPLY, saved=False, language_notice_text=notice_text
            )
        direction = "owed_to_me" if intent == "debt_owed_to_me" else "i_owe"
        debt = add_debt(user_id, person, amount, direction, db_path=db_path)
        if direction == "owed_to_me":
            reply = f"Noted: {person} owes you {format_naira(amount)} naira."
        else:
            reply = f"Noted: you owe {person} {format_naira(amount)} naira."
        return _response(transcript, debt["user_id"], db_path, reply, saved=True, language_notice_text=notice_text)

    if intent == "debt_paid":
        if user_id is None:
            return _response(transcript, None, db_path, NO_USER_REPLY, saved=False, language_notice_text=notice_text)
        person = extract_person(transcript, intent)
        if person is None:
            return _response(
                transcript, user_id, db_path, NO_DEBT_PERSON_REPLY, saved=False, language_notice_text=notice_text
            )
        settled = mark_debt_paid(user_id, person, db_path=db_path)
        if settled is None:
            reply = f"I couldn't find an open debt for {person}."
            return _response(transcript, user_id, db_path, reply, saved=False, language_notice_text=notice_text)
        reply = (
            f"Noted: {person}'s debt of {format_naira(settled['amount'])} naira "
            f"is marked as paid."
        )
        return _response(
            transcript, settled["user_id"], db_path, reply, saved=True, language_notice_text=notice_text
        )

    # ------------------------------------------------------------------
    # Request history (persistent Excel ledger link)
    # ------------------------------------------------------------------
    if intent == REQUEST_HISTORY_INTENT:
        if user_id is None:
            no_user = (
                "I need to know whose book this is before I can share your "
                "ledger link. Please send this note along with your user id "
                "so I can give you the right one."
            )
            return _response(transcript, None, db_path, no_user, saved=False, language_notice_text=notice_text)
        user_row = get_user_by_id(user_id, db_path=db_path)
        if user_row is None or not user_row.get("phone_or_name"):
            # Fallback: if a trader has no phone_or_name set yet, fall back to
            # using "user_{id}" as a stable key so they still get a link.
            fallback_key = f"user_{user_id}"
            link_path = build_ledger_path(fallback_key)
        else:
            link_path = build_ledger_path(str(user_row["phone_or_name"]))
        reply = (
            "Here is your permanent ledger link. Bookmark it — every time "
            "you open it I regenerate a fresh Excel file with all your "
            "latest sales, expenses, and running balance. "
            f"Link: {link_path}"
        )
        return _response(transcript, user_id, db_path, reply, saved=True, language_notice_text=notice_text)

    # ------------------------------------------------------------------
    # New sale / expense (the default), or nothing understood
    # ------------------------------------------------------------------
    parsed = parse_transcript(transcript)
    if not parsed:
        return _response(transcript, user_id, db_path, UNCLEAR_REPLY, saved=False, language_notice_text=notice_text)

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
        language_notice_text=notice_text,
    )
