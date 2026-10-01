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

import logging
from pathlib import Path
from typing import Any, Optional

from app.asr import language_notice, transcribe
from app.db import DEFAULT_DB_PATH, log_llm_extraction
from app.llm_parser import extract_entries_llm_safe

logger = logging.getLogger("veyra.pipeline")
from app.intents import (
    BUSINESS_INSIGHT_INTENT,
    CHECK_STOCK_INTENT,
    classify_message,
    extract_amount,
    extract_person,
    EXPENSE_BREAKDOWN_INTENT,
    HELP_INTENT,
    MENU_INTENT,
    REQUEST_HISTORY_INTENT,
)
from app.ledger import (
    add_debt,
    add_entries,
    build_ledger_path,
    check_low_stock,
    correct_last_entry,
    delete_last_entry,
    get_expense_breakdown,
    get_stock_levels,
    get_summary,
    get_top_items,
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

# ---------------------------------------------------------------------------
# Multilingual failure notification templates
# ---------------------------------------------------------------------------
FAILURE_TEMPLATES = {
    "en": {
        "entry_not_recorded": "Entry not recorded: {reason}.",
        "ambiguous_amount": "Entry not recorded: the amount was unclear.",
        "missing_details": "Entry not recorded: missing person or amount.",
        "llm_failed": "Entry not recorded: I couldn't understand this entry.",
        "partial_success": "Recorded {count} entries. {failed_count} entries were not recorded: {details}.",
    },
    "pcm": {
        "entry_not_recorded": "Entry no record: {reason}.",
        "ambiguous_amount": "Entry no record: the amount no clear.",
        "missing_details": "Entry no record: missing person or amount.",
        "llm_failed": "Entry no record: I no understand this entry.",
        "partial_success": "Don record {count} entries. {failed_count} entries no record: {details}.",
    },
    "yo": {
        "entry_not_recorded": "Ìkọ̀wé kò gbọ́wọ́: {reason}.",
        "ambiguous_amount": "Ìkọ̀wé kò gbọ́wọ́: iye owó ò mọ̀.",
        "missing_details": "Ìkọ̀wé kò gbọ́wọ́: aìsí orúkọ tabi iye owó.",
        "llm_failed": "Ìkọ̀wé kò gbọ́wọ́: mo kò gbọ́ ìkọ̀wé yìí.",
        "partial_success": "A gbọ́wó {count} ìkọ̀wé. {failed_count} ìkọ̀wé kò gbọ́wó: {details}.",
    },
    "ha": {
        "entry_not_recorded": "Shigar ba a rubuta ba: {reason}.",
        "ambiguous_amount": "Shigar ba a rubuta ba: adadin bai bayyana ba.",
        "missing_details": "Shigar ba a rubuta ba: ba a samu suna ko adadi ba.",
        "llm_failed": "Shigar ba a rubuta ba: ina ba iya fahimtar wannan shigar ba.",
        "partial_success": "An rubuta {count} shigar. {failed_count} shigar ba a rubuta ba: {details}.",
    },
    "ig": {
        "entry_not_recorded": "Ndebanye ahụ egbughi: {reason}.",
        "ambiguous_amount": "Ndebanye ahụ egbughi: ego ahu adịghị mma.",
        "missing_details": "Ndebanye ahụ egbughi: enweghị aha ma ọ bụ ego.",
        "llm_failed": "Ndebanye ahụ egbughi: m ghọtaghị ndebanye ahu.",
        "partial_success": "Debanyere {count} ndebanye. {failed_count} ndebanye egbughi: {details}.",
    },
}

# ---------------------------------------------------------------------------
# Multilingual success response templates
# ---------------------------------------------------------------------------
SUCCESS_TEMPLATES = {
    "en": {
        "recorded": "Recorded",
        "separator": " — ",
        "expense_label": "expense",
        "sale_label": "sale",
    },
    "pcm": {
        "recorded": "Don record",
        "separator": " — ",
        "expense_label": "expense",
        "sale_label": "sale",
    },
    "yo": {
        "recorded": "A gbọ́wọ́",
        "separator": " — ",
        "expense_label": "ìṣún",
        "sale_label": "ìtájà",
    },
    "ha": {
        "recorded": "An rubuta",
        "separator": " — ",
        "expense_label": "kashe",
        "sale_label": "sayarwa",
    },
    "ig": {
        "recorded": "Debanyere",
        "separator": " — ",
        "expense_label": "mmefụ",
        "sale_label": "ire",
    },
}


def format_naira(amount: int) -> str:
    """45000 -> "45,000"."""
    return f"{int(amount):,}"


def _get_failure_message(
    message_type: str,
    language: str,
    **kwargs
) -> str:
    """Get a failure notification message in the appropriate language.
    
    Args:
        message_type: The type of failure message (e.g., "entry_not_recorded", "ambiguous_amount")
        language: Language code (en, pcm, yo, ha, ig)
        **kwargs: Additional parameters to format into the message template
    
    Returns:
        The formatted failure message in the specified language, falling back to English
    """
    templates = FAILURE_TEMPLATES.get(language, FAILURE_TEMPLATES["en"])
    template = templates.get(message_type, templates["entry_not_recorded"])
    try:
        return template.format(**kwargs)
    except (KeyError, ValueError):
        # If formatting fails, return the template with placeholders
        return template


def _format_entry_success(entry: dict[str, Any], language: str) -> str:
    """Format a single entry for the simplified success response.
    
    Args:
        entry: The entry dict with item, amount, type fields
        language: Language code for formatting
    
    Returns:
        Formatted string like "Rice — ₦5,000" or "Rice — ₦5,000 — expense"
    """
    templates = SUCCESS_TEMPLATES.get(language, SUCCESS_TEMPLATES["en"])
    item = entry.get("item", "Item")
    amount = entry.get("amount")
    entry_type = entry.get("type", "")
    
    # Format amount with naira symbol
    if amount is not None:
        amount_str = f"₦{format_naira(amount)}"
    else:
        amount_str = ""
    
    # Build the entry string
    parts = [item]
    if amount_str:
        parts.append(amount_str)
    
    # Add type label if it's an expense (sales don't need a label as they're the default)
    if entry_type == "expense":
        parts.append(templates["expense_label"])
    
    return templates["separator"].join(parts)


def _build_success_reply(entries: list[dict[str, Any]], language: str) -> str:
    """Build a simplified success response for recorded entries.
    
    Args:
        entries: List of successfully saved entries
        language: Language code for the response
    
    Returns:
        Simplified response like "Recorded: Rice — ₦5,000; Chin Chin — ₦2,000 👍"
    """
    templates = SUCCESS_TEMPLATES.get(language, SUCCESS_TEMPLATES["en"])
    
    if not entries:
        return ""
    
    # Format each entry
    entry_strings = [_format_entry_success(entry, language) for entry in entries]
    
    # Join with semicolons
    entries_text = "; ".join(entry_strings)
    
    # Build final response
    return f"{templates['recorded']}: {entries_text} 👍"


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


def _entries_reply(entries: list[dict[str, Any]], language: str = "en") -> str:
    """Simplified entry reply using the new format."""
    return _build_success_reply(entries, language)


def _response(
    transcript: str,
    uid: Optional[int],
    db_path,
    reply: str,
    entries: Optional[list[dict[str, Any]]] = None,
    *,
    saved: bool,
    language_notice_text: Optional[str] = None,
    detected_language: Optional[str] = None,
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
        "detected_language": detected_language,
    }


def process_voice_note(
    audio_path: str,
    language: str,
    user_id: Optional[int] = None,
    *,
    db_path=None,
    transcript: Optional[str] = None,
) -> dict[str, Any]:
    """Run ASR -> classify -> route -> persist, and build the reply text.

    If ``transcript`` is provided (e.g. a WhatsApp text message), the file
    check and ASR step are skipped and the given text is routed exactly like
    a transcription; ``audio_path`` is then ignored.

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

    # With a transcript override (text-message path) there is no audio file
    # at all, so the existence check and ASR step are skipped entirely.
    audio_name = ""
    if transcript is None:
        audio_path = str(Path(audio_path).expanduser().resolve())
        if not Path(audio_path).is_file():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        audio_name = Path(audio_path).name

    notice_text = language_notice(language)
    if transcript is None:
        try:
            transcript = transcribe(audio_path, language)
        except RuntimeError as e:
            # AI model loading failed - provide fallback guidance
            error_msg = str(e)
            logger.warning(f"Voice transcription failed: {error_msg}")
            fallback_reply = (
                f"Sorry, I'm having trouble with voice recognition right now. "
                f"The AI model is still loading (this can take 1-2 minutes on first use). "
                f"Please type your message instead, or try voice again in a moment. "
                f"You can say things like: \"I sold rice 5k\" or \"I spent 2k on transport\"."
            )
            return _response(
                transcript="[Voice recognition unavailable]",
                user_id=user_id,
                db_path=db_path,
                reply=fallback_reply,
                entries=[],
                saved=False,
                language_notice_text=notice_text,
                detected_language=language,
            )
    intent = classify_message(transcript)

    # ------------------------------------------------------------------
    # Menu / help / request-history: state-read actions that never write
    # (except request_history may return saved=True because a user exists).
    # ------------------------------------------------------------------
    if intent == MENU_INTENT:
        return _response(
            transcript, user_id, db_path, MENU_REPLY, saved=False,
            language_notice_text=notice_text,
            detected_language=language,
        )

    if intent == HELP_INTENT:
        return _response(
            transcript, user_id, db_path, HELP_FAQ_REPLY, saved=False,
            language_notice_text=notice_text,
            detected_language=language,
        )

    # ------------------------------------------------------------------
    # Corrections / deletes: need a known user and an earlier entry
    # ------------------------------------------------------------------
    if intent == "correction":
        if user_id is None:
            return _response(transcript, None, db_path, NO_USER_REPLY, saved=False, language_notice_text=notice_text, detected_language=language)
        new_amount = extract_amount(transcript)
        if new_amount is None:
            return _response(
                transcript, user_id, db_path, NO_AMOUNT_REPLY, saved=False, language_notice_text=notice_text, detected_language=language
            )
        updated = correct_last_entry(user_id, new_amount, db_path=db_path)
        if updated is None:
            return _response(
                transcript, user_id, db_path, NO_ENTRY_TO_CORRECT_REPLY, saved=False, language_notice_text=notice_text, detected_language=language
            )
        templates = SUCCESS_TEMPLATES.get(language, SUCCESS_TEMPLATES["en"])
        reply = f"Corrected: {format_naira(updated['amount'])} naira 👍"
        return _response(
            transcript, updated["user_id"], db_path, reply, [updated], saved=True, language_notice_text=notice_text, detected_language=language
        )

    if intent == "delete_last":
        if user_id is None:
            return _response(transcript, None, db_path, NO_USER_REPLY, saved=False, language_notice_text=notice_text)
        removed = delete_last_entry(user_id, db_path=db_path)
        if removed is None:
            return _response(
                transcript, user_id, db_path, NO_ENTRY_TO_REMOVE_REPLY, saved=False, language_notice_text=notice_text
            )
        templates = SUCCESS_TEMPLATES.get(language, SUCCESS_TEMPLATES["en"])
        return _response(
            transcript,
            removed["user_id"],
            db_path,
            "Removed 👍",
            [removed],
            saved=True,
            language_notice_text=notice_text,
            detected_language=language,
        )

    # ------------------------------------------------------------------
    # Debts
    # ------------------------------------------------------------------
    if intent in ("debt_owed_to_me", "debt_i_owe"):
        # Split transcript by debt statements to handle multiple debts in one message
        from app.parser import split_transactions
        debt_segments = split_transactions(transcript)
        
        if not debt_segments:
            debt_segments = [transcript]  # Fallback to original if splitting fails
        
        debts_added = []
        failed_entries = []  # Track failed entries with reasons
        direction = "owed_to_me" if intent == "debt_owed_to_me" else "i_owe"
        
        for segment in debt_segments:
            amount = extract_amount(segment)
            person = extract_person(segment, intent)
            
            if amount is None or person is None:
                # Track the failure instead of silently skipping
                if amount is None and person is None:
                    failed_entries.append({
                        "segment": segment,
                        "reason": _get_failure_message("missing_details", language)
                    })
                elif amount is None:
                    failed_entries.append({
                        "segment": segment,
                        "reason": _get_failure_message("ambiguous_amount", language)
                    })
                else:  # person is None
                    failed_entries.append({
                        "segment": segment,
                        "reason": _get_failure_message("missing_details", language)
                    })
                continue
            
            debt = add_debt(user_id, person, amount, direction, db_path=db_path)
            debts_added.append(debt)
        
        if not debts_added:
            # All entries failed - return detailed failure message
            if failed_entries:
                failure_details = "; ".join([f"\"{f['segment']}\" - {f['reason']}" for f in failed_entries])
                reply = _get_failure_message("entry_not_recorded", language, reason=failure_details)
            else:
                reply = DEBT_MISSING_DETAILS_REPLY
            return _response(
                transcript, user_id, db_path, reply, saved=False, language_notice_text=notice_text, detected_language=language
            )
        
        # Build reply with successful debts using simplified format
        templates = SUCCESS_TEMPLATES.get(language, SUCCESS_TEMPLATES["en"])
        debt_entries = []
        for debt in debts_added:
            person = debt['person']
            amount = format_naira(debt['amount'])
            if direction == "owed_to_me":
                debt_str = f"{person} — ₦{amount}"
            else:
                debt_str = f"{person} — ₦{amount}"
            debt_entries.append(debt_str)
        
        if len(debt_entries) == 1:
            reply = f"{templates['recorded']}: {debt_entries[0]} 👍"
        else:
            reply = f"{templates['recorded']}: {'; '.join(debt_entries)} 👍"
        
        # Append failure notifications if there were partial failures
        if failed_entries:
            failure_messages = [f['reason'] for f in failed_entries]
            reply += " " + " ".join(failure_messages)
        
        return _response(transcript, debts_added[0]["user_id"], db_path, reply, debts_added, saved=True, language_notice_text=notice_text, detected_language=language)

    if intent == "debt_paid":
        if user_id is None:
            return _response(transcript, None, db_path, NO_USER_REPLY, saved=False, language_notice_text=notice_text, detected_language=language)
        person = extract_person(transcript, intent)
        if person is None:
            return _response(
                transcript, user_id, db_path, NO_DEBT_PERSON_REPLY, saved=False, language_notice_text=notice_text, detected_language=language
            )
        settled = mark_debt_paid(user_id, person, db_path=db_path)
        if settled is None:
            reply = f"I couldn't find an open debt for {person}."
            return _response(transcript, user_id, db_path, reply, saved=False, language_notice_text=notice_text, detected_language=language)
        templates = SUCCESS_TEMPLATES.get(language, SUCCESS_TEMPLATES["en"])
        reply = f"Paid: {person} — ₦{format_naira(settled['amount'])} 👍"
        return _response(
            transcript, settled["user_id"], db_path, reply, saved=True, language_notice_text=notice_text, detected_language=language
        )

    # ------------------------------------------------------------------
    # Request history (persistent Excel ledger link)
    # ------------------------------------------------------------------
    if intent == REQUEST_HISTORY_INTENT:
        if user_id is None:
            # Multilingual no-user response
            no_user_responses = {
                "en": "I need to know whose book this is before I can share your ledger link. Please send this note along with your user id so I can give you the right one.",
                "pcm": "I need to know whose book this is before I can share your ledger link. Please send this note along with your user id so I can give you the right one.",
                "yo": "Mo nilo lati mọ tọ́wọ́ wo nínú ìwé yìí kí ṣe le pín ọna asopọ̀ ìwé rẹ. Jọ́wọ́ fi ID onírúurú pẹ̀lú nọ́tì yìí kí n lè fi ọna tọ́ fun ọ.",
                "ha": "Ina bu sanin wanda littafin take ne don kafin in i raba abin da ke. Don da wannan littafi tare da ID mai amfani don in i ba ka madaidaita.",
                "ig": "Achọrọ m ịmara onye bụ akwụkwọ a tupu m enyekwe njikọ akwụkwọ ndebi gị. Biko nye nọtị a tinyere ID onye ọrụ gị ka m enyekwe gị nke ziri ezi.",
            }
            no_user = no_user_responses.get(language, no_user_responses["en"])
            return _response(transcript, None, db_path, no_user, saved=False, language_notice_text=notice_text, detected_language=language)
        user_row = get_user_by_id(user_id, db_path=db_path)
        if user_row is None or not user_row.get("phone_or_name"):
            # Fallback: if a trader has no phone_or_name set yet, fall back to
            # using "user_{id}" as a stable key so they still get a link.
            fallback_key = f"user_{user_id}"
            link_path = build_ledger_path(fallback_key)
        else:
            link_path = build_ledger_path(str(user_row["phone_or_name"]))
        
        # Multilingual responses
        responses = {
            "en": f"Here is your permanent ledger link. Bookmark it — every time you open it I regenerate a fresh Excel file with all your latest sales, expenses, and running balance. Link: {link_path}",
            "pcm": f"Here is your permanent ledger link. Bookmark am — every time you open am I regenerate fresh Excel file with all your latest sales, expenses, and running balance. Link: {link_path}",
            "yo": f"Èyí ni ọna asopọ̀ ìwé rẹ ti ó ma ń yọ̀gbà. Fi pamọ́ — nígbà kọ̀ọ̀kọ̀ tí o bú un, mà ń ṣe àwọn faili Excel tuntun pẹ̀lú gbogbo ìtájà rẹ tuntun, ìṣún, àti ìbámupamọ́. Link: {link_path}",
            "ha": f"Wannan shine hanyar littafin take na. Aji shi — kowane lokacin da kai zai, zan saba sabunta littafin Excel tare da dukkan sayarwar kasuwancin na, farashin, da daidaitaccen. Link: {link_path}",
            "ig": f"Ná bụ njikọ akwụkwọ ndebi gị na-adịgide ide. Jide ya - n'oge ọ bụla ị megharịrị ya, m na-emepụta faịlụ Excel ọhụrụ nwere ahịa niile gị ọhụrụ, mmefụ, na nha. Link: {link_path}",
        }
        reply = responses.get(language, responses["en"])
        return _response(transcript, user_id, db_path, reply, saved=True, language_notice_text=notice_text, detected_language=language)

    # ------------------------------------------------------------------
    # Check stock levels
    # ------------------------------------------------------------------
    if intent == CHECK_STOCK_INTENT:
        if user_id is None:
            return _response(transcript, None, db_path, NO_USER_REPLY, saved=False, language_notice_text=notice_text, detected_language=language)
        stock_levels = get_stock_levels(user_id, db_path=db_path)
        if not stock_levels:
            reply = "I don't have any stock information yet. Start recording what you buy and sell, and I'll track your inventory."
        else:
            items = [f"{s['item']}: {s['quantity_remaining']}" for s in stock_levels[:5]]  # Limit to top 5 items
            reply = "Here's your stock level: " + ", ".join(items) + "."
        return _response(transcript, user_id, db_path, reply, saved=False, language_notice_text=notice_text, detected_language=language)

    # ------------------------------------------------------------------
    # Business insights
    # ------------------------------------------------------------------
    if intent == BUSINESS_INSIGHT_INTENT:
        if user_id is None:
            return _response(transcript, None, db_path, NO_USER_REPLY, saved=False, language_notice_text=notice_text, detected_language=language)
        top_items = get_top_items(user_id, metric="profit", period_days=30, db_path=db_path)
        if not top_items:
            reply = "I don't have enough data yet to tell you your best sellers. Keep recording your sales!"
        else:
            best = top_items[0]
            worst = top_items[-1] if len(top_items) > 1 else None
            reply = f"{best['item']} is your best earner this month."
            if worst and worst['item'] != best['item']:
                reply += f" {worst['item']} is moving slowly."
        return _response(transcript, user_id, db_path, reply, saved=False, language_notice_text=notice_text, detected_language=language)

    # ------------------------------------------------------------------
    # Expense breakdown
    # ------------------------------------------------------------------
    if intent == EXPENSE_BREAKDOWN_INTENT:
        if user_id is None:
            return _response(transcript, None, db_path, NO_USER_REPLY, saved=False, language_notice_text=notice_text, detected_language=language)
        breakdown = get_expense_breakdown(user_id, period_days=30, db_path=db_path)
        if not breakdown:
            reply = "I don't have any expense data for this period yet."
        else:
            items = [f"{b['category']}: {format_naira(b['total_amount'])}" for b in breakdown[:5]]
            reply = "Your expenses this month: " + ". ".join(items) + "."
        return _response(transcript, user_id, db_path, reply, saved=False, language_notice_text=notice_text, detected_language=language)

    # ------------------------------------------------------------------
    # New sale / expense (the default), or nothing understood
    # ------------------------------------------------------------------
    parsed = parse_transcript(transcript)
    failed_entries = []  # Track failed entries with reasons
    
    # LLM fallback: trigger when rule-based parser returns zero entries
    should_use_llm_fallback = False
    llm_trigger_reason = ""
    
    if not parsed:
        # Rule-based parser found nothing - try LLM
        should_use_llm_fallback = True
        llm_trigger_reason = "no entries from rule-based parser"
        logger.info("Rule-based parser returned no entries, trying LLM fallback")
    
    if should_use_llm_fallback:
        try:
            logger.info(f"Attempting LLM fallback for transcript: {transcript[:100]}...")
            llm_entries, raw_llm_output = extract_entries_llm_safe(transcript, language)
            
            # Log the LLM extraction attempt
            overall_confidence = "low" if any(e.get("confidence") == "low" for e in llm_entries) else "high"
            
            # Filter out unclear entries from LLM results and track failures
            valid_llm_entries = []
            for entry in llm_entries:
                if entry.get("type") == "unclear":
                    failed_entries.append({
                        "segment": transcript[:50] + "..." if len(transcript) > 50 else transcript,
                        "reason": _get_failure_message("llm_failed", language)
                    })
                elif entry.get("amount") is None:
                    failed_entries.append({
                        "segment": entry.get("item", "unknown") if entry.get("item") else "unknown",
                        "reason": _get_failure_message("ambiguous_amount", language)
                    })
                else:
                    valid_llm_entries.append(entry)
            
            # Log to database
            try:
                log_llm_extraction(
                    user_id=user_id,
                    transcript=transcript,
                    language=language,
                    raw_llm_output=raw_llm_output,
                    entries_extracted=llm_entries,
                    confidence=overall_confidence,
                    rule_based_parser_tried_first=True,
                    rule_based_parser_failed=True,
                    db_path=db_path
                )
            except Exception as log_error:
                logger.warning(f"Failed to log LLM extraction: {log_error}")
            
            if valid_llm_entries:
                logger.info(f"LLM fallback extracted {len(valid_llm_entries)} valid entries")
                parsed = valid_llm_entries
            else:
                logger.info("LLM fallback returned no valid entries, using unclear response")
                # If we have specific failure reasons, use them; otherwise use generic unclear
                if failed_entries:
                    failure_msg = " ".join([f['reason'] for f in failed_entries])
                    reply = failure_msg
                else:
                    reply = UNCLEAR_REPLY
                return _response(transcript, user_id, db_path, reply, saved=False, language_notice_text=notice_text, detected_language=language)
                
        except Exception as e:
            logger.warning(f"LLM fallback failed: {e}, falling back to unclear response")
            llm_failure_msg = _get_failure_message("llm_failed", language)
            return _response(transcript, user_id, db_path, llm_failure_msg, saved=False, language_notice_text=notice_text, detected_language=language)
    
    if not parsed:
        return _response(transcript, user_id, db_path, UNCLEAR_REPLY, saved=False, language_notice_text=notice_text, detected_language=language)

    saved_entries = add_entries(
        user_id=user_id,
        entries=parsed,
        transcript=transcript,
        audio_file=audio_name,
        db_path=db_path,
        detected_language=language,
    )
    
    # Check for low stock on sale entries and append warning if needed
    reply = _entries_reply(saved_entries, language)
    
    # Append failure notifications if there were any LLM failures
    if failed_entries:
        failure_messages = [f['reason'] for f in failed_entries]
        reply += " " + " ".join(failure_messages)
    
    for entry in saved_entries:
        if entry["type"] == "sale" and user_id is not None:
            stock_warning = check_low_stock(user_id, entry["item"], db_path=db_path)
            if stock_warning:  # check_low_stock returns None if no warning
                reply += f" {stock_warning}"
                break  # Only add one warning per message
    
    return _response(
        transcript,
        saved_entries[0]["user_id"],
        db_path,
        reply,
        saved_entries,
        saved=True,
        language_notice_text=notice_text,
        detected_language=language,
    )
