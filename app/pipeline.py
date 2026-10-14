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
import random
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
    GREETING_INTENT,
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

TRANSCRIPTION_CORRECTION_REPLIES = {
    "en": [
        "I see you're correcting the transcription. Let me process that instead.",
        "Got it, let me use your correction instead of the voice transcription.",
        "I understand you're fixing what I heard. Let me process your version.",
    ],
    "pcm": [
        "I see say you dey correct the transcription. Make I process your own instead.",
        "Got it, make I use your correction instead of the voice transcription.",
        "I understand say you dey fix wetin I hear. Make I process your own.",
    ],
    "yo": [
        "Mo ri pé ọ ń ṣe àtúnṣe àgbèrò. Jẹ́ kí n ṣe àyọkò àti rẹ dipo.",
        "Got it, jẹ́ kí n lo àtúnṣe rẹ dipo àgbèrò ohùn.",
        "Mo gbọ́ pé ọ ń ṣe àtúnṣe ohun tí mo gbọ́. Jẹ́ kí n ṣe àyọkò àti rẹ.",
    ],
    "ha": [
        "Ina gan cewa kana yi gyaran aiko. Zan yi tsari na maimakon.",
        "Got it, zan yi amfani da gyarewar ku maimakon aikin murya.",
        "Ina fahimtar kana gyara abin na ji na. Zan yi tsari na maimakon.",
    ],
    "ig": [
        "A hụrụ na ị na-emegharị transkripshọn. Ka m wee n'ụzọ gị kama.",
        "Got it, ka m jiri nchekota gị kama transkripshọn olu.",
        "A ghọtaghị na ị na-emezi ihe m nụrụ. Ka m wee n'ụzọ gị.",
    ],
}

GREETING_REPLIES = {
    "en": [
        "Hello! What can I do for you?",
        "Hi there! How can I help you today?",
        "Hey! What's on your mind?",
        "Good to hear from you! What can I help you with?",
    ],
    "pcm": [
        "Hello! Wetin I fit do for you?",
        "Hi there! How I fit help you today?",
        "Hey! Wetin dey your mind?",
        "Good to hear from you! Wetin I fit help you with?",
    ],
    "yo": [
        "Ẹ kú àáwọ́! Kí n ṣe eléyìí tí ó wúlò fún ọ?",
        "Kààbọ̀! Báwo ni mo lè rán ọ lọ́wọ́ láì?",
        "Hey! Kí n wà núkẹyìn?",
        "Dáadáa láti gbọ́ kí ó tó! Kí n ṣe eléyìí tí mo lè rán ọ lọ́wọ́?",
    ],
    "ha": [
        "Sannu! Menene zaka iya yi?",
        "Sannu da zuwa! Yaya zaka iya taimaka ka yau?",
        "Hey! Me ake tsammani?",
        "Yakamar da jin hadu da ku! Menene zaka iya taimaka?",
    ],
    "ig": [
        "Kedu! Kedu ihe m ga-eme gị?",
        "Ndewo! Kedu m ga-enyere gị aka taa?",
        "Hey! Ihe di n'uche gị?",
        "Oma ile gị! Kedu ihe m ga-enyere gị aka?",
    ],
}

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


def _get_greeting_reply(language: str) -> str:
    """Get a random natural greeting response in the specified language."""
    greeting_replies = GREETING_REPLIES.get(language, GREETING_REPLIES["en"])
    return random.choice(greeting_replies)


def _generate_clarification(transcript: str, language: str) -> str:
    """Generate a clarification prompt when an entry is unclear.
    
    This function tries to understand what the user might have meant
    and asks for confirmation, being tolerant of grammatical errors.
    """
    import re
    from app.parser import find_numeric_phrases
    
    # Try to extract any numbers that might be amounts
    amounts = find_numeric_phrases(transcript)
    amount_text = ""
    if amounts:
        amount_val = amounts[0]["value"]
        amount_text = f" ₦{format_naira(int(amount_val))}"
    
    # Try to extract potential item/person names (remove common words)
    words = re.findall(r"[^\W\d_][\w'\-]*", transcript, flags=re.UNICODE)
    # Filter out very short words and common stopwords
    potential_names = [w for w in words if len(w) >= 3 and w.lower() not in 
                     {"the", "and", "for", "with", "from", "have", "has", "had", 
                      "was", "were", "been", "being", "do", "does", "did", "will",
                      "would", "can", "could", "should", "may", "might", "must",
                      "this", "that", "these", "those", "what", "which", "who",
                      "when", "where", "why", "how", "all", "any", "some", "no",
                      "not", "only", "very", "just", "also", "back", "after", "before"}]
    
    name_text = ""
    if potential_names:
        # Take the first 2-3 most likely names
        name_text = " ".join(potential_names[:3])
    
    # Generate clarification based on what we found
    clarifications = {
        "en": [
            f"I'm not sure I understood. Are you trying to say something about {name_text}{amount_text}?",
            f"I didn't quite catch that. Did you mean to record {name_text}{amount_text}?",
            f"Could you clarify? I heard something about {name_text}{amount_text}.",
            f"I'm having trouble understanding. Were you talking about {name_text}{amount_text}?",
            f"If the transcription is wrong, just type what you actually said and I'll use that instead.",
        ],
        "pcm": [
            f"I no sure say I understand. You dey try to talk about {name_text}{amount_text}?",
            f"I no hear am well. You mean say make I record {name_text}{amount_text}?",
            f"Abeg explain well. I hear something about {name_text}{amount_text}.",
            f"I dey find am hard to understand. You dey talk about {name_text}{amount_text}?",
            f"If the transcription no correct, just type wetin you actually talk and I go use that instead.",
        ],
        "yo": [
            f"Mo kò gbọ́rò. Ṣe o ń sọ nǹkan nípa {name_text}{amount_text}?",
            f"Mo kò gbọ́ dáadáa. Ṣe o ń wọ́n láti kọ̀wé {name_text}{amount_text}?",
            f"Jọ́wọ́ ṣe alágbára. Mo gbọ́ nǹkan nípa {name_text}{amount_text}.",
            f"Mo ń wà nínú ìbànújẹ́. Ṣe o ń sọ̀rọ̀ nípa {name_text}{amount_text}?",
            f"Ti àgbèrò bá ṣe e, kọ nǹkan tí o sọ̣ ní kíkọ́, èmi yòó lo èyí.",
        ],
        "ha": [
            f"Ina tabbatacce. Kana son faɗa abu {name_text}{amount_text}?",
            f"Ina ji matukar. Kana son rubuta {name_text}{amount_text}?",
            f"Don yarda. Ina ji abu {name_text}{amount_text}.",
            f"Ina matsala wajen fahimta. Kana tana magana {name_text}{amount_text}?",
            f"Dan da aiko ba daidai ba, rubuta abin ka ce kuma zan yi amfani da shi.",
        ],
        "ig": [
            f"Aghọtaghị m ihe. Ọ na-ekwu maka {name_text}{amount_text}?",
            f"Aghọtaghị m mma. Ọ na-achọ ide {name_text}{amount_text}?",
            f"Biko kọwaa. M nụrụ ihe gbasara {name_text}{amount_text}.",
            f"M nwe nsogbu ịghọta. Ọ na-ekwu maka {name_text}{amount_text}?",
            f"Nr ma ọ bụrụ na transkripshọn bụ adịghị mma, dee ihe ọ kwuru n'ezie ka m jiri ya.",
        ],
    }
    
    # If we found meaningful content, use targeted clarification
    if name_text or amount_text:
        lang_clarifications = clarifications.get(language, clarifications["en"])
        return random.choice(lang_clarifications)
    
    # Otherwise, fall back to general unclear message with menu
    general_clarifications = {
        "en": [
            "I couldn't understand that voice note. Please try speaking more clearly, or type your message instead.",
            "The transcription didn't make sense. Could you try again or type what you wanted to say?",
            "I'm having trouble with voice recognition. Please type your message instead.",
        ],
        "pcm": [
            "I no understand that voice note. Abeg try talk more clear, or type your message instead.",
            "The transcription no make sense. You fit try again or type wetin you wan talk?",
            "I dey find am hard with voice recognition. Abeg type your message instead.",
        ],
        "yo": [
            "Mo kò gbọ́ ohùn náà. Jọ́wọ́ ṣe sọ̀rọ̀ rẹ̀ pọ́njú, tàbí kọ nǹkan tí o fẹ́ sọ̀ ní kíkọ́.",
            "Àgbèrò náà kò ṣe pọ́. Ṣe o lè ṣe agbára gbígbé èyí tàbí kọ nǹkan tí o fẹ́ sọ̀?",
            "Mo ń wà nínú ìṣòro pẹ̀lú ìgbọ́ ohùn. Jọ́wọ́ kọ nǹkan tí o fẹ́ sọ̀ ní kíkọ́.",
        ],
        "ha": [
            "Ina fahimtar wannan murya bayi. Don yarda, kana magana daidai, ko rubuta sakon da na.",
            "Aikin ba daidai ba. Kana sake kokari ko rubuta abin ka son faɗa.",
            "Ina da matsala wajen fahimtar murya. Don rubuta sakon da na.",
        ],
        "ig": [
            "Aghọtaghị m mụ that voice note ahụ. Biko nwaa ịkwu ka mma, ma ọ dee ozi gị n'oge.",
            "Transkripshọn ahụ adịghị mma. Ị ga-agbalị ọzọ ma ọ dee ihe ị chọrọ ikwu?",
            "M nwe nsogbu ịghọta olu. Biko dee ozi gị n'oge.",
        ],
    }
    
    lang_general = general_clarifications.get(language, general_clarifications["en"])
    return random.choice(lang_general)


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

    The summary is intentionally *not* computed here.  The demo frontend
    re-fetches it via GET /summary/{uid} immediately after receiving this
    response, and the WhatsApp webhook only reads ``reply_text``.  Computing
    it here would add 5+ cross-region DB queries for data that is thrown
    away.  Callers that need a summary should fetch it explicitly.
    """
    return {
        "user_id": uid,
        "transcript": transcript,
        "entries": entries or [],
        "summary": _empty_summary(),
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
            
            # Log the transcript for debugging
            logger.info(f"Transcribed audio in language '{language}': {transcript[:100]}...")
            
            # Check if transcription quality is poor (very short or appears to be gibberish)
            if len(transcript) < 3:
                logger.warning(f"Transcription seems too short: '{transcript}'")
                fallback_reply = (
                    f"I couldn't understand that voice note. "
                    f"Please try speaking more clearly, or type your message instead. "
                    f"You can say things like: \"I sold rice 5k\" or \"I spent 2k on transport\"."
                )
                return _response(
                    transcript=transcript,
                    user_id=user_id,
                    db_path=db_path,
                    reply=fallback_reply,
                    entries=[],
                    saved=False,
                    language_notice_text=notice_text,
                    detected_language=language,
                )
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
    # Greeting: natural conversation starter
    # ------------------------------------------------------------------
    if intent == GREETING_INTENT:
        reply = _get_greeting_reply(language)
        return _response(
            transcript, user_id, db_path, reply, saved=False,
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
            # All entries failed - use clarification prompt if both amount and person missing
            # but use detailed failure message if partial information was found
            has_partial_info = any(
                f['reason'] != _get_failure_message("missing_details", language)
                for f in failed_entries
            )
            
            if has_partial_info and failed_entries:
                failure_details = "; ".join([f"\"{f['segment']}\" - {f['reason']}" for f in failed_entries])
                reply = _get_failure_message("entry_not_recorded", language, reason=failure_details)
            else:
                # No meaningful information extracted - use clarification prompt
                reply = _generate_clarification(transcript, language)
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
    # HYBRID APPROACH: Try LLM first for better NLU, fall back to rule-based if LLM fails
    parsed = None
    failed_entries = []  # Track failed entries with reasons
    
    # Try LLM first (better natural language understanding)
    try:
        logger.info(f"Attempting LLM extraction for transcript: {transcript[:100]}...")
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
                rule_based_parser_tried_first=False,
                rule_based_parser_failed=False,
                db_path=db_path
            )
        except Exception as log_error:
            logger.warning(f"Failed to log LLM extraction: {log_error}")
        
        if valid_llm_entries and overall_confidence == "high":
            logger.info(f"LLM extracted {len(valid_llm_entries)} high-confidence entries")
            parsed = valid_llm_entries
        else:
            logger.info("LLM returned low confidence or no valid entries, trying rule-based parser")
            # Fall back to rule-based parser
            parsed = parse_transcript(transcript)
            
    except Exception as e:
        logger.warning(f"LLM extraction failed: {e}, falling back to rule-based parser")
        # Fall back to rule-based parser on any error
        parsed = parse_transcript(transcript)
    
    if not parsed:
        # Use clarification prompt instead of generic unclear message
        clarification = _generate_clarification(transcript, language)
        return _response(transcript, user_id, db_path, clarification, saved=False, language_notice_text=notice_text, detected_language=language)

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
