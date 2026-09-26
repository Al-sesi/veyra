"""
Rule-based intent classification for trader voice notes.

`classify_message(text)` maps a transcript to exactly ONE of:

  - "entry"           a new sale or expense to record
  - "correction"      fix the amount of the last entry
                      ("no, it was 4k not 40k", "sorry, make it 4k")
  - "delete_last"     void the last entry ("remove the last one", "cancel that")
  - "debt_owed_to_me" someone owes the trader ("Mama Ngozi owes me 5k")
  - "debt_i_owe"      the trader owes someone ("I owe Bisi 10k")
  - "debt_paid"       a registered debt was settled ("Mama Ngozi don pay")
  - "request_history" trader asks for their persistent Excel ledger link
                      ("show my history", "send my records", "my report")
  - "menu"            asks for the help menu or Veyra didn't understand
                      (triggers reply listing the 5 things she can do)
  - "help"            asks for FAQ-style info: what Veyra is, languages,
                      pilot status, no human support available yet

No AI model is involved: everything is keyword/phrase matching (the parser
must stay rule-based). The trigger lists below are grouped by language so
more languages (Yoruba / Hausa / Igbo) can be appended later without
touching any other code.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Dict, List, Optional, Sequence, Tuple

from app.parser import find_numeric_phrases, strip_diacritics
from app.languages.ha import HAUSA_PACK
from app.languages.ig import IGBO_PACK
from app.languages.yo import YORUBA_PACK


# ---------------------------------------------------------------------------
# Intent names
# ---------------------------------------------------------------------------
INTENTS: Tuple[str, ...] = (
    "entry",
    "correction",
    "delete_last",
    "debt_owed_to_me",
    "debt_i_owe",
    "debt_paid",
    "request_history",
    "menu",
    "help",
)

ENTRY_INTENT = "entry"
CORRECTION_INTENT = "correction"
DELETE_LAST_INTENT = "delete_last"
DEBT_OWED_TO_ME_INTENT = "debt_owed_to_me"
DEBT_I_OWE_INTENT = "debt_i_owe"
DEBT_PAID_INTENT = "debt_paid"
REQUEST_HISTORY_INTENT = "request_history"
MENU_INTENT = "menu"
HELP_INTENT = "help"

DEBT_INTENTS: Tuple[str, ...] = (
    DEBT_OWED_TO_ME_INTENT,
    DEBT_I_OWE_INTENT,
    DEBT_PAID_INTENT,
)


# ---------------------------------------------------------------------------
# Trigger phrases (lowercase; matched on word boundaries).
#
# Add new languages as new labelled blocks, e.g.:
#   YORUBA_TRIGGERS  = ["gbese mi", ...]        # he/she owes me
#   HAUSA_TRIGGERS   = ["bashi na", ...]
#   IGBO_TRIGGERS    = ["ji m ugwo", ...]
# The matcher (`find_trigger`) scans all lists in order, so appending is safe.
# ---------------------------------------------------------------------------

DEBT_OWED_TO_ME_TRIGGERS = [
    # English
    "owes me", "owe me", "owed me", "still owes me", "still owe me",
    "hasn't paid me", "has not paid me", "didn't pay me", "did not pay me",
    # Nigerian Pidgin
    "dey owe me", "still dey owe me", "never pay me", "never paid me",
    "no pay me", "im owe me",
    # Yoruba ("jẹ mi ní" = owes me; "kò san" = hasn't paid). Matched
    # diacritic-folded, so plain-ASCII ASR output works too.
    "je mi ni", "je mi lowo", "o je mi", "won je mi",
    "ko san", "ko san fun mi", "ko tii san", "ko ti san",
    # Hausa / Igbo / expanded Yoruba — from the app.languages packs. The
    # packs are always on: code-switching inside one note is the normal case.
    *HAUSA_PACK["debt_owed_to_me"],
    *IGBO_PACK["debt_owed_to_me"],
    *YORUBA_PACK["debt_owed_to_me"],
]

DEBT_I_OWE_TRIGGERS = [
    # English
    "i owe", "i still owe", "i haven't paid", "i have not paid",
    # Nigerian Pidgin
    "me owe", "i dey owe", "i no pay", "i never pay", "i never paid",
    "i still dey owe",
    # Yoruba ("mo jẹ ọ́" / "mo jẹ ẹ́" / "mo jẹ wọ́n" = I owe you/them)
    "mo je o", "mo je e", "mo je won", "mo je yin",
    # Hausa / Igbo / expanded Yoruba — from the app.languages packs.
    *HAUSA_PACK["debt_i_owe"],
    *IGBO_PACK["debt_i_owe"],
    *YORUBA_PACK["debt_i_owe"],
]

DEBT_PAID_TRIGGERS = [
    # English
    "paid me back", "paid me", "pay me back", "pay me",
    "has paid", "have paid", "has settled", "settled me", "refunded me",
    # Nigerian Pidgin
    "don pay me", "don pay", "don settle", "don clear",
    # Yoruba ("ti san" = has paid). First-person forms ("mo ti san") are
    # filtered out in classify_message: the trader paying is an entry.
    "o ti san", "ti san", "san tan", "san an", "o san an",
    # Hausa / Igbo / expanded Yoruba — from the app.languages packs.
    *HAUSA_PACK["debt_paid"],
    *IGBO_PACK["debt_paid"],
    *YORUBA_PACK["debt_paid"],
]

DELETE_LAST_TRIGGERS = [
    # English
    "remove the last", "remove last", "delete the last", "delete last",
    "cancel the last", "cancel that", "undo that", "undo",
    "take off the last", "scrap the last",
    # Nigerian Pidgin
    "comot the last", "comot last", "comot am", "remove am", "delete am",
    "cancel am",
    # Yoruba ("yọ ... kúrò" = take it out/off, "ìparí" = the end)
    "yo kuro", "yo o kuro", "ipari",
    # Hausa / Igbo / expanded Yoruba — from the app.languages packs.
    *HAUSA_PACK["delete_last"],
    *IGBO_PACK["delete_last"],
    *YORUBA_PACK["delete_last"],
]

CORRECTION_TRIGGERS = [
    # English
    "sorry", "i mean", "i meant", "make it", "change it", "change to",
    "correct it", "correction", "it should be", "should be",
    "supposed to be", "my mistake",
    # Nigerian Pidgin
    "make am", "change am", "correct am", "abeg change", "abeg make",
    # Yoruba ("kìí ṣe" = it is not, "bẹ́ẹ̀ kọ́" = not so, "àtúnṣe" = correction)
    "atunse", "mo tun se", "kii se", "kii se bee", "bee ko",
    # Hausa / Igbo / expanded Yoruba — from the app.languages packs.
    *HAUSA_PACK["correction"],
    *IGBO_PACK["correction"],
    *YORUBA_PACK["correction"],
]

REQUEST_HISTORY_TRIGGERS = [
    # English
    "show my history", "show me my history", "my history",
    "send my records", "send me my records", "my records", "my record",
    "my report", "send me my report", "show me my report", "my statement",
    "show me my statement", "ledger report", "my ledger", "show my ledger",
    "show me my ledger", "export my ledger", "download my ledger",
    "give me my report", "give me my records", "give me my history",
    "can i have my report", "can i see my records", "can i see my history",
    # Nigerian Pidgin
    "show my history", "send my report", "send my records",
    "give me my report", "give me my records", "my report",
    "my records", "show me my book", "my book", "give me my book",
    "abeg send my report", "abeg give me my records", "i need my report",
    "i need my records",
    # TODO: Yoruba triggers (e.g. "fihan iwe mi", "ranse iwe-akoso mi")
    # TODO: Hausa triggers (e.g. "nuna littafin tarihi na", "aiko rahoto na")
    # TODO: Igbo triggers (e.g. "gosi akwụkwọ m", "zipụta akwụkwọ ndebi m")
]

MENU_TRIGGERS = [
    # English — single keyword "menu" works any case, anywhere in the text.
    # Explicit menu request phrases:
    "menu", "show menu", "see menu", "what can you do",
    "what do you do", "list commands", "options", "show options",
    "main menu", "start menu", "tell me what to do", "instructions",
    # Nigerian Pidgin
    "menu", "wetin you fit do", "wetin you sabi do",
    "show me the menu", "abeg show menu", "list wetin you dey do",
    # TODO: Yoruba triggers (e.g. "akojọ ọjà kan", "kọ ohun ti o le ṣe")
    # TODO: Hausa triggers (e.g. "menu", "lissafi abin da zaka iya yi")
    # TODO: Igbo triggers (e.g. "menu", "depụta ihe ị nwere ike ime")
]

HELP_TRIGGERS = [
    # English — FAQ / customer-care-style questions
    "help", "help me", "what is veyra", "who are you", "what are you",
    "support", "customer care", "customer support", "talk to a human",
    "speak to a person", "human support", "contact support",
    "which languages", "what languages", "languages you support",
    "is this a pilot", "pilot info", "about veyra", "tell me about veyra",
    "how does veyra work", "need help", "i need help",
    # Nigerian Pidgin
    "help", "help me", "abeg help", "i need help",
    "who be veyra", "wetin be veyra", "wetin you be",
    "na pilot", "wetin languages you sabi", "customer care",
    "abeg send person help me",
    # TODO: Yoruba triggers (e.g. "e ran mi lọwọ", "kini Veyra")
    # TODO: Hausa triggers (e.g. "taya ni", "menene Veyra")
    # TODO: Igbo triggers (e.g. "nyere m aka", "kedu ihe Veyra bụ")
]

# "no, it was 4k not 40k" — a bare sentence-initial "no" is a correction
# signal. Matched against the diacritic-folded text so Yoruba "rárá" works too.
CORRECTION_OPENER_RE = re.compile(r"^\s*(no|rara)\b", re.IGNORECASE)

# Words that are never part of a person's name when extracting debt names.
PERSON_STOPWORDS = {
    "i", "me", "my", "mine", "we", "us", "our", "you", "your",
    "he", "him", "his", "she", "her", "it", "they", "them",
    "a", "an", "the", "of", "for", "to", "from", "on", "in", "into", "at",
    "and", "or", "but", "that", "this", "these", "those", "with", "about",
    "am", "is", "are", "was", "were", "be", "been", "being",
    "do", "does", "did", "don", "dey", "go", "still", "never", "no", "not",
    "has", "have", "had", "will", "would", "can", "cannot", "cant",
    "owe", "owes", "owed", "owing",
    "pay", "pays", "paid", "paying", "settle", "settled", "clear", "cleared",
    "refund", "refunded", "back", "debt", "debts", "money", "naira",
    "say", "said", "tell", "told", "when", "what", "which", "who", "abeg",
    "so", "just", "now", "today", "yesterday", "tomorrow", "o", "oh", "na",
    # Yoruba particles (compared diacritic-folded)
    "mo", "mi", "ti", "tii", "san", "je", "lowo", "owo", "kuro", "yo",
    "fun", "ko", "kii", "se", "bee", "rara", "tun", "ni", "yen", "naa",
    "gan", "pa", "ipare", "atunse", "wa", "lo",
}
# Particles from the Hausa/Igbo/Yoruba packs (app.languages), compared
# diacritic-folded just like the lists above.
LANGUAGE_PERSON_STOPWORDS = (
    HAUSA_PACK["person_stopwords"]
    + IGBO_PACK["person_stopwords"]
    + YORUBA_PACK["person_stopwords"]
)
PERSON_STOPWORDS = PERSON_STOPWORDS | set(LANGUAGE_PERSON_STOPWORDS)


# ---------------------------------------------------------------------------
# Matching helpers
# ---------------------------------------------------------------------------

def _fold_with_index_map(text: str) -> Tuple[str, List[int]]:
    """Diacritic-fold `text` and remember where each folded char came from.

    Returns (folded, starts) with the same length; `folded[i]` is the base
    letter of the original character at `text[starts[i]]`. Combining tone
    marks disappear while folding, so offsets are mapped back through this
    table to keep trigger positions valid for the ORIGINAL text.
    """
    folded_chars: List[str] = []
    starts: List[int] = []
    for idx, ch in enumerate(text):
        for f in unicodedata.normalize("NFKD", ch):
            if unicodedata.combining(f):
                continue
            folded_chars.append(f)
            starts.append(idx)
    return "".join(folded_chars), starts


def find_trigger(
    text: str,
    triggers: Sequence[str],
) -> Optional[Tuple[str, int, int]]:
    """Find the earliest whole-word occurrence of any trigger in `text`.

    Matching is diacritic-insensitive ("tà" also matches "ta") but the
    returned (trigger, start, end) positions always refer to `text` itself.
    On a tie (same start position) the longest trigger wins, so
    "never pay me" beats "pay me".
    """
    if not text:
        return None
    folded, starts = _fold_with_index_map(text)
    lowered = folded.lower()
    best: Optional[Tuple[str, int, int]] = None
    for trig in triggers:
        start = lowered.find(trig)
        while start != -1:
            end = start + len(trig)
            before_ok = start == 0 or not lowered[start - 1].isalnum()
            after_ok = end == len(lowered) or not lowered[end].isalnum()
            if before_ok and after_ok:
                o_start = starts[start]
                o_end = starts[end - 1] + 1
                # Include any tone marks that follow the last matched letter.
                while o_end < len(text) and unicodedata.combining(text[o_end]):
                    o_end += 1
                if (
                    best is None
                    or o_start < best[1]
                    or (o_start == best[1] and len(trig) > len(best[0]))
                ):
                    best = (trig, o_start, o_end)
                break
            start = lowered.find(trig, start + 1)
    return best


def _first_person_before(text: str, position: int) -> bool:
    """True if a first-person subject ("I", "we", "me", Yoruba "mo") stands
    before `position`.

    "I don pay transport 2k" / "mo ti san transport 2k" means the trader paid
    an expense — it must NOT be read as someone paying the trader back
    ("don pay" / "ti san" alone would suggest that).
    """
    return re.search(r"\b(i|we|me|mo)\b", text[:position], re.IGNORECASE) is not None


def classify_message(text: str) -> str:
    """Rule-based classifier; returns exactly one of `INTENTS`.

    Order matters:
      - `request_history` first so "show my history" is never mistaken for
        a debt/expense.
      - `help` next so FAQ phrases like "what is Veyra" are answered with
        the FAQ reply instead of falling into the generic menu.
      - `menu` next so "what can you do" / "menu" are recognised before
        any debt/delete/correction matching.
      - Debt statements before payment statements
        ("Musa never pay me 3k" is a debt, not a payment).
      - Deletes before corrections.
      - Anything unrecognised falls through to "entry" (the pipeline later
        decides whether the parser actually extracted a transaction; if not
        the menu is shown as a fallback).
    """
    text = text or ""
    if find_trigger(text, REQUEST_HISTORY_TRIGGERS):
        return REQUEST_HISTORY_INTENT
    if find_trigger(text, HELP_TRIGGERS):
        return HELP_INTENT
    if find_trigger(text, MENU_TRIGGERS):
        return MENU_INTENT
    if find_trigger(text, DEBT_OWED_TO_ME_TRIGGERS):
        return DEBT_OWED_TO_ME_INTENT
    if find_trigger(text, DEBT_I_OWE_TRIGGERS):
        return DEBT_I_OWE_INTENT
    paid = find_trigger(text, DEBT_PAID_TRIGGERS)
    if paid is not None and not _first_person_before(text, paid[1]):
        return DEBT_PAID_INTENT
    if find_trigger(text, DELETE_LAST_TRIGGERS):
        return DELETE_LAST_INTENT
    if CORRECTION_OPENER_RE.match(strip_diacritics(text)) or find_trigger(
        text, CORRECTION_TRIGGERS
    ):
        return CORRECTION_INTENT
    return ENTRY_INTENT


# ---------------------------------------------------------------------------
# Value extraction
# ---------------------------------------------------------------------------

def extract_amount(text: str) -> Optional[int]:
    """Return the amount the message refers to, or None.

    The FIRST numeric phrase wins: for corrections the corrected value comes
    before the old one ("no, it was 4k not 40k" -> 4000).
    """
    phrases = find_numeric_phrases(text or "")
    if not phrases:
        return None
    return int(phrases[0]["value"])


_DEBT_TRIGGERS_BY_INTENT: Dict[str, Sequence[str]] = {
    DEBT_OWED_TO_ME_INTENT: DEBT_OWED_TO_ME_TRIGGERS,
    DEBT_I_OWE_INTENT: DEBT_I_OWE_TRIGGERS,
    DEBT_PAID_INTENT: DEBT_PAID_TRIGGERS,
}

# For these intents the person's name sits BEFORE the trigger phrase.
_NAME_BEFORE_TRIGGER = {DEBT_OWED_TO_ME_INTENT, DEBT_PAID_INTENT}


def extract_person(text: str, intent: str) -> Optional[str]:
    """Best-effort extraction of the person's name for a debt intent.

    Strategy: delete the amount phrase and the trigger phrase from the text,
    then keep the meaningful words nearest the trigger, title-cased.
    Returns None when nothing usable is left.

    Examples:
      "Mama Ngozi owes me 5k"       -> "Mama Ngozi"
      "I owe Bisi 10k"              -> "Bisi"
      "I never pay the supplier 20k"-> "Supplier"
    """
    triggers = _DEBT_TRIGGERS_BY_INTENT.get(intent)
    if triggers is None:
        return None

    work = text or ""
    phrases = find_numeric_phrases(work)
    if phrases:
        first = phrases[0]
        work = work[: first["start"]] + " " + work[first["end"] :]

    trigger = find_trigger(work, triggers)
    if trigger is None:
        scope = work
    else:
        _, start, end = trigger
        scope = work[:start] if intent in _NAME_BEFORE_TRIGGER else work[end:]

    words = re.findall(r"[^\W\d_][\w'\-]*", scope, flags=re.UNICODE)
    kept = [w for w in words if strip_diacritics(w.lower()) not in PERSON_STOPWORDS]
    if not kept:
        return None
    if intent in _NAME_BEFORE_TRIGGER:
        kept = kept[-3:]  # closest words to the trigger
    else:
        kept = kept[:3]
    return " ".join(w if w[:1].isupper() else w.capitalize() for w in kept)
