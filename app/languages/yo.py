"""Yoruba vocabulary pack (native-speaker-confirmed, expanded).

This pack is the single source of truth for Yoruba number words (moved here
from app.parser) and the expanded, now-confirmed transaction-verb phrases.

Phrase categories:
  buy_verbs / sale_verbs / pay_verbs  -> feed detect_transaction_type in
      app.parser (buy and pay count as expense, sale as sale).
  debt_owed_to_me / debt_i_owe / debt_paid / correction / delete_last
      -> appended to the trigger lists in app.intents (the short words
      "ra"/"ta" stay in app.parser as whole-word single-token lists).
  number_words -> merged into the parser's written-number lookup. Yoruba
      puts the scale word BEFORE the multiplier ("ẹgbẹ̀rún márùn-ún" =
      5000), handled by the parser's scale-first branch.
  number_tails -> hyphen-glued filler syllables ("márùn-ún" -> head "marun"
      + tail "un") that carry no value of their own.
  pronouns -> stripped during item extraction ("Mo ta iresi fun 10k" ->
      item "Iresi").
  person_stopwords -> dropped during debt-name extraction.

All phrases are folded (diacritics stripped, lowercased) at import, because
ASR output may or may not carry Yoruba tone marks.
"""

from __future__ import annotations

import unicodedata
from typing import Any, Dict, List


def _fold(text: str) -> str:
    decomposed = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch)).lower()


YORUBA_PACK: Dict[str, Any] = {
    "language": "yo",
    "buy_verbs": [_fold(p) for p in [
        "mo ra", "mo rà", "mo ti ra",
    ]],
    "sale_verbs": [_fold(p) for p in [
        "mo ta", "mo tà", "mo ti ta",
    ]],
    "pay_verbs": [_fold(p) for p in [
        "mo san", "mo na", "mo ná",
    ]],
    "debt_owed_to_me": [_fold(p) for p in [
        "o je mi", "o si je mi", "ko tii san mi",
    ]],
    "debt_i_owe": [_fold(p) for p in [
        "mo je", "mo si je",
    ]],
    "debt_paid": [_fold(p) for p in [
        "o ti san mi", "gbese naa ti tan", "a ti san an",
    ]],
    "correction": [_fold(p) for p in [
        "rara", "mo tumo si", "e ma binu",
    ]],
    "delete_last": [_fold(p) for p in [
        "yo eyi to kehin", "pa a re", "fagilee eyi",
    ]],
    "number_words": {
        "okan": 1, "kan": 1,
        "meji": 2, "eji": 2,
        "meta": 3, "eta": 3,
        "merin": 4, "erin": 4,
        "marun": 5, "arun": 5,
        "mefa": 6, "efa": 6,
        "meje": 7, "eje": 7,
        "mejo": 8, "ejo": 8,
        "mesan": 9, "esan": 9,
        "mewa": 10, "ewa": 10,
        "ogun": 20,
        "ogoji": 40,
        "egberun": 1000,
        "milionu": 1000000,
    },
    "number_tails": {"un", "an", "aa"},
    "pronouns": [
        "fun",
    ],
    "person_stopwords": [
        "si", "gbese", "tan", "kehin", "eyi", "tumo", "binu", "ma", "re",
        "fagilee", "fun",
    ],
}
