"""Hausa / Igbo / expanded-Yoruba vocabulary packs (app/languages/).

The packs are always applied alongside the English/Pidgin baseline, because
code-switching inside a single note ("Na sayi rice 5k" — Hausa verb, English
item, universal "5k" shorthand) is the normal case. Every phrase is matched
diacritic-folded, so both "Azụrụ m" and "Azuru m" reach the same trigger.
"""

import pytest

from app.intents import classify_message, extract_amount, extract_person
from app.parser import (
    detect_transaction_type,
    parse_transcript,
    parse_written_number_words,
    split_transactions,
    strip_diacritics,
)


# ---------------------------------------------------------------------------
# Required scenarios
# ---------------------------------------------------------------------------

def test_hausa_buy_entry():
    # "Na sayi rice 5k" -> Hausa buy, item rice, 5000.
    result = parse_transcript("Na sayi rice 5k")
    assert len(result) == 1
    assert result[0]["type"] == "expense"
    assert result[0]["amount"] == 5000
    assert result[0]["item"].lower() == "rice"
    assert classify_message("Na sayi rice 5k") == "entry"


def test_igbo_buy_entry_with_and_without_diacritics():
    # "Azụrụ m rice 5k" -> Igbo buy, item rice, 5000 (both orthographies).
    for text in ["Azụrụ m rice 5k", "Azuru m rice 5k"]:
        result = parse_transcript(text)
        assert len(result) == 1, text
        assert result[0]["type"] == "expense"
        assert result[0]["amount"] == 5000
        assert result[0]["item"].lower() == "rice"
        assert classify_message(text) == "entry"


def test_hausa_debt_owed_to_me():
    # "Musa yana bina 20k" -> debt, person Musa, 20000.
    text = "Musa yana bina 20k"
    assert classify_message(text) == "debt_owed_to_me"
    assert extract_amount(text) == 20000
    assert extract_person(text, "debt_owed_to_me") == "Musa"


def test_yoruba_sell_entry_with_and_without_diacritics():
    # "Mo ta iresi fun 10k" -> Yoruba sell, item iresi, 10000.
    for text in ["Mo ta iresi fun 10k", "Mò tá ìrèsí fún 10k"]:
        result = parse_transcript(text)
        assert len(result) == 1, text
        assert result[0]["type"] == "sale"
        assert result[0]["amount"] == 10000
        assert strip_diacritics(result[0]["item"].lower()) == "iresi"
        assert classify_message(text) == "entry"


# ---------------------------------------------------------------------------
# Number words (scale word BEFORE the multiplier, like Yoruba)
# ---------------------------------------------------------------------------

def test_hausa_number_words():
    assert parse_written_number_words(["dubu"]) == 1000
    assert parse_written_number_words(["dubu", "biyu"]) == 2000
    assert parse_written_number_words(["dubu", "biyar"]) == 5000


def test_igbo_number_words():
    # "puku" = thousand (standard Igbo orthography).
    assert parse_written_number_words(["puku"]) == 1000
    assert parse_written_number_words(["puku", "abuo"]) == 2000
    assert parse_written_number_words(["puku", "ise"]) == 5000


def test_written_thousands_parse_end_to_end():
    result = parse_transcript("Na sayi rice dubu biyu")
    assert len(result) == 1
    assert result[0]["amount"] == 2000
    assert result[0]["type"] == "expense"


def test_k_shorthand_is_language_agnostic():
    for text in ["Na sayi rice 5k", "Azuru m rice 5k", "Mo ta iresi 10k"]:
        amounts = [e["amount"] for e in parse_transcript(text)]
        assert amounts, text


# ---------------------------------------------------------------------------
# Pack transaction verbs (via detect_transaction_type)
# ---------------------------------------------------------------------------

def test_pack_pay_verbs_are_expenses():
    assert detect_transaction_type("na biya transport 3k") == "expense"
    assert detect_transaction_type("mo na transport 2k") == "expense"


def test_pack_sale_verbs_are_sales():
    assert detect_transaction_type("na sayar da akara 2k") == "sale"
    assert detect_transaction_type("e rere m akara 2k") == "sale"


def test_first_person_yoruba_payment_still_an_entry():
    # "mo na transport 2k" = "I paid for transport" -> the trader's own
    # expense; classify must not route it to a debt intent.
    assert classify_message("mo na transport 2k") == "entry"


# ---------------------------------------------------------------------------
# Hausa intent triggers
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text", [
    "Musa yana bin ni 5k",
    "Musa bai biya ni ba 5k",
])
def test_hausa_debt_owed_to_me_triggers(text):
    assert classify_message(text) == "debt_owed_to_me"


@pytest.mark.parametrize("text", [
    "ina bin bashi 5k",
    "ina da bashi 5k",
])
def test_hausa_debt_i_owe_triggers(text):
    assert classify_message(text) == "debt_i_owe"


def test_hausa_debt_paid():
    text = "Musa ya biya ni 5k"
    assert classify_message(text) == "debt_paid"
    assert extract_person(text, "debt_paid") == "Musa"
    assert classify_message("an biya") == "debt_paid"


@pytest.mark.parametrize("text", [
    "cire na karshe",
    "goge na karshe",
    "soke na karshe",
])
def test_hausa_delete_last_triggers(text):
    assert classify_message(text) == "delete_last"


def test_hausa_correction_triggers():
    assert classify_message("a'a, 4k ne") == "correction"
    assert extract_amount("a'a, 4k ne") == 4000
    assert classify_message("yi hakuri, 5k") == "correction"


def test_hausa_new_buy_verbs():
    """Test newly added Hausa buy verb variants confirmed by native speaker."""
    # "siya" and "sai" should be recognized as buy verbs
    assert classify_message("na siya rice 5k") == "entry"
    assert classify_message("na sai alewa 10k") == "entry"
    
    # Parse should recognize these as expenses
    from app.parser import parse_transcript
    entries = parse_transcript("na siya biskit dubu biyar")
    assert len(entries) == 1
    assert entries[0]["type"] == "expense"
    assert entries[0]["amount"] == 5000


def test_hausa_compound_numbers_with_sha():
    """Test Hausa compound numbers using 'sha' (and/plus) connector."""
    from app.parser import parse_transcript
    # "dubu goma sha biyu" = 10000 + 2000 = 12000 (contextual: biyu = 2, but after sha with thousands it's 2000)
    entries = parse_transcript("na siya maggi dubu goma sha biyu")
    assert len(entries) == 1
    assert entries[0]["amount"] == 12000  # 10000 + 2000
    assert entries[0]["type"] == "expense"
    
    # Test compound with thousands: "dubu goma sha dubu biyar" = 10000 + 5000 = 15000
    entries = parse_transcript("na sai alewa dubu goma sha dubu biyar")
    assert len(entries) == 1
    assert entries[0]["amount"] == 15000  # 10000 + 5000


def test_hausa_multiple_debt_statements():
    """Test that multiple debt statements in one transcript are captured separately."""
    from app.parser import split_transactions
    
    # Test with simpler debt statements (debt portion of testtt.ogg)
    transcript = "ina bin shi dubu biyar, ina bi baban musa dubu goma sha biyu, ina bi isma'iyi dubu takwas"
    
    # Split into segments and check each can be parsed
    segments = split_transactions(transcript)
    
    # Should have 3 debt statements
    assert len(segments) == 3
    
    # Each should have debt_i_owe intent
    for seg in segments:
        assert classify_message(seg) == "debt_i_owe"
        person = extract_person(seg, "debt_i_owe")
        # Should extract names or fallback to someone
        assert person is not None


def test_hausa_person_name_fallback():
    """Test that unclear person names fall back to 'someone' instead of number words."""
    # "ina bi dubu biyar" - "dubu biyar" are number words, "bi" is a verb fragment
    person = extract_person("ina bi dubu biyar", "debt_i_owe")
    # Should return "someone" since no clear name is present
    assert person == "someone"
    
    # "ina bi abu maggi" - "abu" is a name, "maggi" is an item
    person = extract_person("ina bi abu maggi dubu biyar", "debt_i_owe")
    # Should extract "Abu" as the name
    assert person == "Abu"


def test_testtt_ogg_transcript():
    """Test the exact testtt.ogg transcript with all entries and debts."""
    from app.parser import split_transactions
    
    # Exact transcript from testtt.ogg
    transcript = "na siya biskit dubu biyar, na sai alewa na dubu goma, na sai abin gatimati gongoni na dubu takwas, na sai maggi na dubu goma sha biyu, tsohama abu maggi ina bin shi dubu biyar, ina bi baban musa dubu goma sha biyu, ina bi isma'iyi dubu takwas, ina bi abubuƙa dubu uku."
    
    # Split into segments
    segments = split_transactions(transcript)
    
    # Should have 8 segments (4 expenses + 4 debts)
    assert len(segments) == 8
    
    # First 4 should be entry intents (expenses)
    for i in range(4):
        assert classify_message(segments[i]) == "entry"
    
    # Last 4 should be debt_i_owe intents
    for i in range(4, 8):
        assert classify_message(segments[i]) == "debt_i_owe"
    
    # Test expense parsing
    expense_entries = parse_transcript(segments[0])
    assert len(expense_entries) == 1
    assert expense_entries[0]["type"] == "expense"
    assert expense_entries[0]["amount"] == 5000  # dubu biyar = 5000
    
    # Test compound number with sha
    sha_entry = parse_transcript(segments[3])
    assert len(sha_entry) == 1
    assert sha_entry[0]["amount"] == 12000  # dubu goma sha biyu = 10000 + 2000
    
    # Test debt name extraction
    # "tsohama abu maggi ina bin shi dubu biyar" - should extract "Abu" as name (maggi is item/stopword)
    person = extract_person(segments[4], "debt_i_owe")
    assert person == "Abu"
    
    # "ina bi baban musa dubu goma sha biyu" - should extract "Musa" (baban = father of, is a title)
    person = extract_person(segments[5], "debt_i_owe")
    assert person == "Musa"
    
    # "ina bi isma'iyi dubu takwas" - should extract "Isma'iyi"
    person = extract_person(segments[6], "debt_i_owe")
    assert person == "Isma'iyi"
    
    # "ina bi abubuƙa dubu uku" - should extract "Abubuƙa"
    person = extract_person(segments[7], "debt_i_owe")
    assert person == "Abubuƙa"


# ---------------------------------------------------------------------------
# Igbo intent triggers
# ---------------------------------------------------------------------------

def test_igbo_debt_owed_to_me():
    for text in ["ọ ji m ụgwọ 5k", "o ji m ugwo 5k", "ọ akwụbeghị m 5k"]:
        assert classify_message(text) == "debt_owed_to_me", text
    named = "Bisi ọ ji m ụgwọ 5k"
    assert extract_person(named, "debt_owed_to_me") == "Bisi"


def test_igbo_debt_i_owe():
    assert classify_message("m ji ya ugwo 5k") == "debt_i_owe"


def test_igbo_debt_paid():
    assert classify_message("ọ kwụọla m") == "debt_paid"
    assert classify_message("akwụọla") == "debt_paid"


@pytest.mark.parametrize("text", [
    "hichapụ nke ikpeazụ",
    "wepu nke ikpeazu",
    "kagbuo nke ikpeazu",
])
def test_igbo_delete_last_triggers(text):
    assert classify_message(text) == "delete_last"


def test_igbo_correction_triggers():
    assert classify_message("mba, ihe m putara bu 4k") == "correction"
    assert extract_amount("mba, ihe m putara bu 4k") == 4000


# ---------------------------------------------------------------------------
# Expanded Yoruba intent triggers
# ---------------------------------------------------------------------------

def test_yoruba_expanded_debt_triggers():
    assert classify_message("o si je mi 5k") == "debt_owed_to_me"
    assert classify_message("Musa ko tii san mi 5k") == "debt_owed_to_me"
    assert classify_message("mo si je 5k") == "debt_i_owe"
    assert classify_message("gbese naa ti tan") == "debt_paid"
    assert classify_message("a ti san an") == "debt_paid"


@pytest.mark.parametrize("text", [
    "yo eyi to kehin",
    "fagilee eyi",
])
def test_yoruba_expanded_delete_last_triggers(text):
    assert classify_message(text) == "delete_last"


def test_yoruba_expanded_correction_triggers():
    assert classify_message("rara, mo tumo si 4k") == "correction"
    assert extract_amount("rara, mo tumo si 4k") == 4000


# ---------------------------------------------------------------------------
# Code-switching stays an entry (regression guard for always-on packs)
# ---------------------------------------------------------------------------

def test_code_switched_entries_are_entries():
    for text in [
        "Na sayi rice 5k",
        "Azụrụ m rice 5k",
        "Mo ta iresi fun 10k",
        "Na sayar da rice 10k",
    ]:
        assert classify_message(text) == "entry", text
