"""Igbo vocabulary pack (native-speaker-confirmed).

Phrase categories:
  buy_verbs / sale_verbs / pay_verbs  -> feed detect_transaction_type in
      app.parser (buy and pay count as expense, sale as sale).
  debt_owed_to_me / debt_i_owe / debt_paid / correction / delete_last
      -> appended to the trigger lists in app.intents.
  number_words -> merged into the parser's written-number lookup. Like
      Hausa and Yoruba, Igbo puts the scale word BEFORE the multiplier
      ("puku abuo" = 2000).
  pronouns -> stripped during item extraction ("Azụrụ m rice 5k" ->
      item "Rice").
  person_stopwords -> dropped during debt-name extraction.

All phrases are folded (diacritics stripped, lowercased) at import, because
ASR output may or may not carry Igbo tone marks.
"""

from __future__ import annotations

import unicodedata
from typing import Any, Dict, List


def _fold(text: str) -> str:
    decomposed = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch)).lower()


IGBO_PACK: Dict[str, Any] = {
    "language": "ig",
    "buy_verbs": [_fold(p) for p in [
        "azụrụ m", "azụrụ m ya", "zụrụ m",
    ]],
    "sale_verbs": [_fold(p) for p in [
        "e rere m", "e rere m ya", "rere m",
    ]],
    "pay_verbs": [_fold(p) for p in [
        "akwụụrụ m", "eji m ego kwuo",
    ]],
    "debt_owed_to_me": [_fold(p) for p in [
        "ọ ji m ụgwọ", "ọ ji m ego", "ọ akwụbeghị m",
    ]],
    "debt_i_owe": [_fold(p) for p in [
        "m ji", "m ji ya ụgwọ",
    ]],
    "debt_paid": [_fold(p) for p in [
        "ọ kwụọla m", "akwụọla", "ụgwọ ahụ akwụchaala",
    ]],
    "correction": [_fold(p) for p in [
        "mba", "ihe m pụtara bụ", "ndo",
    ]],
    "delete_last": [_fold(p) for p in [
        "hichapụ nke ikpeazụ", "wepụ nke ikpeazụ", "kagbuo nke ikpeazụ",
    ]],
    "number_words": {
        "otu": 1, "abuo": 2, "ato": 3, "ano": 4, "ise": 5,
        "isii": 6, "asaa": 7, "asato": 8, "itoolu": 9, "iri": 10,
        "puku": 1000,
    },
    "pronouns": [],
    "person_stopwords": [
        "m", "ji", "ya", "ugwo", "ego", "kwuo", "kwuola", "akwuola",
        "akwubeghi", "rere", "azuru", "zuru", "akwuuru", "eji", "ahu",
        "mba", "ndo", "ihe", "putara", "bu", "nke", "hichapu", "wepu",
        "kagbuo", "ikpeazu", "akwuchala",
    ],
}
