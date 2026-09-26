"""Hausa vocabulary pack (native-speaker-confirmed).

Phrase categories:
  buy_verbs / sale_verbs / pay_verbs  -> feed detect_transaction_type in
      app.parser (buy and pay count as expense, sale as sale).
  debt_owed_to_me / debt_i_owe / debt_paid / correction / delete_last
      -> appended to the trigger lists in app.intents.
  number_words -> merged into the parser's written-number lookup. Hausa
      puts the scale word BEFORE the multiplier ("dubu biyu" = 2000), which
      the parser's scale-first branch already handles.
  pronouns -> stripped during item extraction so English items like
      "rice" survive ("Na sayi rice 5k" -> item "Rice").
  person_stopwords -> dropped during debt-name extraction.

All phrases are folded (diacritics stripped, lowercased) at import, because
ASR output may or may not carry Hausa tone marks.
"""

from __future__ import annotations

import unicodedata
from typing import Any, Dict, List


def _fold(text: str) -> str:
    decomposed = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch)).lower()


HAUSA_PACK: Dict[str, Any] = {
    "language": "ha",
    "buy_verbs": [_fold(p) for p in [
        "na sayi", "na saya", "na sayi shi", "na sayi wannan",
    ]],
    "sale_verbs": [_fold(p) for p in [
        "na sayar", "na sayar da", "na sayar da shi",
    ]],
    "pay_verbs": [_fold(p) for p in [
        "na biya", "na kashe kudi", "na kashe",
    ]],
    "debt_owed_to_me": [_fold(p) for p in [
        "yana bina", "yana bin ni", "bai biya ni ba",
    ]],
    "debt_i_owe": [_fold(p) for p in [
        "ina bin", "ina da bashi", "ina bin bashi",
    ]],
    "debt_paid": [_fold(p) for p in [
        "ya biya ni", "ta biya ni", "ya biya", "bashin ya kare", "an biya",
    ]],
    "correction": [_fold(p) for p in [
        "a'a", "ina nufin", "yi hakuri",
    ]],
    "delete_last": [_fold(p) for p in [
        "cire na karshe", "goge na karshe", "soke na karshe",
    ]],
    "number_words": {
        "daya": 1, "biyu": 2, "uku": 3, "hudu": 4, "biyar": 5,
        "shida": 6, "bakwai": 7, "takwas": 8, "tara": 9, "goma": 10,
        "ashirin": 20, "talatin": 30, "arbain": 40, "hamsin": 50,
        "sittin": 60, "sabain": 70, "tamanin": 80, "tisain": 90,
        "dari": 100, "dubu": 1000, "miliyan": 1000000,
    },
    "pronouns": [
        "ni", "ma",
    ],
    "person_stopwords": [
        "na", "ni", "ma", "ba", "bai", "yana", "bina", "bin", "ina", "da",
        "bashi", "biya", "ya", "ta", "an", "kare", "cire", "goge", "soke",
        "karshe", "hakuri", "nufin", "yi", "wannan", "kudi", "kashe",
        "sayi", "saya", "sayar", "shi",
    ],
}
