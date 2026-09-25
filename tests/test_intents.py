"""
Unit tests for the rule-based intent classifier (app/intents.py).

No DB, no ASR: pure text in, intent / amount / person out.
"""

from __future__ import annotations

import pytest

from app.intents import INTENTS, classify_message, extract_amount, extract_person


# ---------------------------------------------------------------------------
# classify_message
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # plain entries
        ("I sold 5 bags of rice for 45k", "entry"),
        ("I don sell akara two-fifty", "entry"),
        ("Don buy stock 150 thousand naira", "entry"),
        # corrections
        ("no, it was 4k not 40k", "correction"),
        ("No it was 4k", "correction"),
        ("sorry, make it 4k", "correction"),
        ("I mean 4k", "correction"),
        ("abeg change am to 4k", "correction"),
        # delete last
        ("remove the last one", "delete_last"),
        ("cancel that", "delete_last"),
        ("comot the last one", "delete_last"),
        # debts
        ("Mama Ngozi owes me 5k", "debt_owed_to_me"),
        ("Musa never pay me 3k", "debt_owed_to_me"),
        ("I owe Bisi 10k", "debt_i_owe"),
        ("I never pay the supplier 20k", "debt_i_owe"),
        ("Mama Ngozi don pay", "debt_paid"),
        ("Musa paid me 3k", "debt_paid"),
        # "I don pay ..." is the trader paying an expense, not someone paying back
        ("I don pay transport 2k", "entry"),
    ],
)
def test_classify_message(text: str, expected: str) -> None:
    assert classify_message(text) == expected


@pytest.mark.parametrize(
    "text",
    ["", "   ", "hello", "asdkjh qwerty", "I sold rice 5k", "Ngozi"],
)
def test_classify_message_always_returns_a_known_intent(text: str) -> None:
    assert classify_message(text) in INTENTS


# ---------------------------------------------------------------------------
# extract_amount
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("no, it was 4k not 40k", 4000),  # the corrected (first) value wins
        ("sorry, make it 4k", 4000),
        ("I mean 4k", 4000),
        ("abeg change am to 4k", 4000),
        ("Mama Ngozi owes me 5k", 5000),
        ("I owe Bisi 10k", 10000),
        ("I never pay the supplier 20k", 20000),
        ("Mama Ngozi don pay", None),  # debt_paid needs no amount
        # "one" is a word-number in the frozen parser, so this returns 1 even
        # though it reads like "delete_last" — harmless because pipeline never
        # calls extract_amount for delete_last/entry intents.
        ("remove the last one", 1),
    ],
)
def test_extract_amount(text: str, expected) -> None:
    assert extract_amount(text) == expected


# ---------------------------------------------------------------------------
# extract_person
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("text", "intent", "expected"),
    [
        ("Mama Ngozi owes me 5k", "debt_owed_to_me", "Mama Ngozi"),
        ("Musa never pay me 3k", "debt_owed_to_me", "Musa"),
        ("I owe Bisi 10k", "debt_i_owe", "Bisi"),
        ("I never pay the supplier 20k", "debt_i_owe", "Supplier"),
        ("Mama Ngozi don pay", "debt_paid", "Mama Ngozi"),
        ("Musa paid me 3k", "debt_paid", "Musa"),
        # not a debt intent -> no person
        ("remove the last one", "delete_last", None),
    ],
)
def test_extract_person(text: str, intent: str, expected) -> None:
    assert extract_person(text, intent) == expected
