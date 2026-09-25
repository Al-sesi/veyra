"""Yoruba support: buy/sell verbs, number words, thousand separators, intent triggers.

The three TRANSCRIPT_* constants are byte-exact ASR output from the real
recordings run through `scripts/try_pipeline.py <file> -l yo`
(NCAIR1 N-ATLaS Whisper). They pin end-to-end behaviour on real speech.
"""

from app.intents import classify_message, extract_amount, extract_person
from app.parser import (
    detect_transaction_type,
    find_numeric_phrases,
    parse_transcript,
    parse_written_number_words,
    strip_diacritics,
)


# Real recordings (test1.ogg.ogg 17.4s, test2.ogg.ogg 25.6s, test3.ogg 36.8s).
TRANSCRIPT_TEST1 = (
    "mo ra rice 45,000 naira lọja, mo dẹwọ mọto 2,000 naira, "
    "mo dẹra chinchin 5,000 naira, mo dẹra ounjẹ 4,000 naira."
)
TRANSCRIPT_TEST2 = (
    "mo lọ sí ọjà lónìí, mo dẹ ra baguette rice, tẹ́milion, "
    "mo wá ra chinchin five-million, mo tó wá ra bisikiti tirike, "
    "mo tó n ra twitififuke, mo wá wọ mọto sevenk."
)
TRANSCRIPT_TEST3 = (
    "mo lọ sọ́jà lónìí, mo dàrá chinchin 7000 náírà pẹ̀lú rice 25000 náírà, "
    "mo tún ra bisikíìtì 18000 náírà, mo bá ra bọ́kẹ́tì 10000 náírà, "
    "ìyá abílíkí, ìyá owó lọ́wọ́ mi 5.000 naira, "
    "osmanura ẹhùn lọ́wọ́ mi 2.000 naira biskid lóra."
)


# ---------------------------------------------------------------------------
# Diacritic folding
# ---------------------------------------------------------------------------

def test_strip_diacritics_folds_yoruba_marks():
    assert strip_diacritics("Ẹ̀tà") == "Eta"
    assert strip_diacritics("ọ̀kàn-ún") == "okan-un"
    assert strip_diacritics("dàrá") == "dara"


# ---------------------------------------------------------------------------
# Yoruba number words
# ---------------------------------------------------------------------------

def test_yoruba_scale_word_comes_first():
    # "ẹgbẹ̀rún márùn-ún" = 1000 x 5 (English would be "five thousand").
    assert parse_written_number_words(["egberun", "marun"]) == 5000
    assert parse_written_number_words(["egberun", "meji"]) == 2000
    assert parse_written_number_words(["egberun"]) == 1000
    assert parse_written_number_words(["milionu", "meta"]) == 3000000


def test_yoruba_number_words_with_diacritics():
    assert parse_written_number_words(["ẹ̀gbẹ̀rún"]) == 1000
    assert parse_written_number_words(["mẹ́ta"]) == 3
    assert parse_written_number_words(["ọgọ́ji"]) == 40
    assert parse_written_number_words(["mílíọ̀nù"]) == 1000000


def test_yoruba_hyphen_filler_tail_carries_no_value():
    # "márùn-ún" (= 5) and "ọ̀kàn-ún" (= 1) glue a filler tail onto the head.
    assert parse_written_number_words(["márùn-ún"]) == 5
    assert parse_written_number_words(["ọ̀kàn-ún"]) == 1


def test_unknown_word_fails_whole_phrase():
    assert parse_written_number_words(["mefa", "lelogun"]) is None


def test_english_word_numbers_unchanged():
    assert parse_written_number_words(["five", "thousand"]) == 5000
    assert parse_written_number_words(["twenty", "five"]) == 25


# ---------------------------------------------------------------------------
# Yoruba verbs: rà = buy (expense), tà = sell (sale)
# ---------------------------------------------------------------------------

def test_yoruba_buy_verb_is_expense():
    result = parse_transcript("mo ra rice 45,000 naira")
    assert len(result) == 1
    assert result[0]["type"] == "expense"
    assert result[0]["amount"] == 45000
    assert "rice" in result[0]["item"].lower()


def test_yoruba_sell_verb_is_sale():
    result = parse_transcript("mo ta akara 2k")
    assert len(result) == 1
    assert result[0]["type"] == "sale"
    assert result[0]["amount"] == 2000
    assert "akara" in result[0]["item"].lower()


def test_yoruba_verb_with_diacritics_and_item():
    result = parse_transcript("mo ta ọgẹ̀dẹ̀ 2k")
    assert result[0]["type"] == "sale"
    assert result[0]["amount"] == 2000
    assert result[0]["item"].lower() == "ọgẹ̀dẹ̀"


def test_moto_is_an_expense_item():
    result = parse_transcript("mo ra moto 4k")
    assert result[0]["type"] == "expense"
    assert result[0]["amount"] == 4000


def test_yoruba_verbs_match_whole_words_only():
    # "ra"/"ta" must not fire inside English words ("extra", "data").
    assert detect_transaction_type("extra 5k") is None
    assert detect_transaction_type("data 5k") is None
    assert detect_transaction_type("mo ra moto 2k") == "expense"
    assert detect_transaction_type("mo ta akara 2k") == "sale"


# ---------------------------------------------------------------------------
# Dot thousand separators and ASR number artifacts
# ---------------------------------------------------------------------------

def test_dot_thousand_separator():
    assert find_numeric_phrases("5.000 naira")[0]["value"] == 5000
    assert find_numeric_phrases("1.234.567 naira")[0]["value"] == 1234567


def test_dot_thousand_separator_in_transcript():
    result = parse_transcript("mo ra rice 5.000 naira")
    assert result[0]["amount"] == 5000


def test_decimal_is_not_a_thousand_separator():
    assert find_numeric_phrases("2.5k")[0]["value"] == 2500


def test_spoken_k_glued_to_word_number():
    # ASR renders spoken "seven k" as "sevenk".
    result = parse_transcript("mo wá wọ mọto sevenk")
    assert len(result) == 1
    assert result[0]["type"] == "expense"
    assert result[0]["amount"] == 7000


# ---------------------------------------------------------------------------
# Real recordings: end-to-end pins
# ---------------------------------------------------------------------------

def test_transcript_test1_recording():
    result = parse_transcript(TRANSCRIPT_TEST1)
    assert len(result) == 4
    assert all(e["type"] == "expense" for e in result)
    assert sorted(e["amount"] for e in result) == [2000, 4000, 5000, 45000]
    items = " ".join(e["item"].lower() for e in result)
    assert "rice" in items and "chinchin" in items and "mọto" in items
    assert classify_message(TRANSCRIPT_TEST1) == "entry"


def test_transcript_test2_recording():
    # "five-million" is an ASR artifact of spoken number shorthand; the
    # remaining garbled tokens ("tirike", "twitififuke") yield no amount.
    result = parse_transcript(TRANSCRIPT_TEST2)
    assert len(result) == 2
    assert all(e["type"] == "expense" for e in result)
    assert sorted(e["amount"] for e in result) == [7000, 5000000]
    assert classify_message(TRANSCRIPT_TEST2) == "entry"


def test_transcript_test3_recording():
    result = parse_transcript(TRANSCRIPT_TEST3)
    assert len(result) == 3
    assert all(e["type"] == "expense" for e in result)
    assert sorted(e["amount"] for e in result) == [10000, 18000, 25000]
    assert classify_message(TRANSCRIPT_TEST3) == "entry"


def test_transcript_test3_dropped_amounts_are_stable():
    """Documented limitations: 7000 is the 2nd amount in one comma segment
    (one amount per segment), and the "lọ́wọ́ mi" (in my hand) segments carry
    no transaction verb, so 5000 / 2000 are not booked."""
    amounts = [e["amount"] for e in parse_transcript(TRANSCRIPT_TEST3)]
    assert 7000 not in amounts
    assert 5000 not in amounts
    assert 2000 not in amounts


# ---------------------------------------------------------------------------
# Yoruba intent triggers
# ---------------------------------------------------------------------------

def test_yoruba_debt_owed_to_me():
    for text in ["Mama Ngozi jẹ mi ni 5k", "Mama Ngozi je mi ni 5k",
                 "o je mi 5k", "Musa kò tíì san", "Musa ko tii san"]:
        assert classify_message(text) == "debt_owed_to_me", text
    assert extract_amount("Mama Ngozi jẹ mi ni 5k") == 5000
    assert extract_person("Mama Ngozi jẹ mi ni 5k", "debt_owed_to_me") == "Mama Ngozi"
    assert extract_person("Musa kò tíì san", "debt_owed_to_me") == "Musa"


def test_yoruba_debt_i_owe():
    for text in ["mo jẹ ọ 10k", "mo je o 10k", "mo jẹ ẹ 10k", "mo jẹ wọ́n 10k"]:
        assert classify_message(text) == "debt_i_owe", text
    assert extract_amount("mo jẹ ọ 10k") == 10000
    # "ọ" is the pronoun "you", not a name -> pipeline asks who.
    assert extract_person("mo jẹ ọ 10k", "debt_i_owe") is None


def test_yoruba_debt_paid():
    assert classify_message("Bisi ti san 10k") == "debt_paid"
    assert classify_message("Bisi san tan") == "debt_paid"
    assert extract_person("Bisi ti san 10k", "debt_paid") == "Bisi"


def test_yoruba_delete_last():
    for text in ["yọ ọ kúrò", "yo o kuro", "ìparí"]:
        assert classify_message(text) == "delete_last", text


def test_yoruba_correction():
    for text in ["rárá, mo sọ 4k", "mo tún ṣe é, 4k", "kìí ṣe bẹ́ẹ̀"]:
        assert classify_message(text) == "correction", text
    assert extract_amount("rárá, mo sọ 4k") == 4000
    assert extract_amount("mo tún ṣe é, 4k") == 4000


def test_yoruba_first_person_payment_is_an_entry():
    # "mo ti san transport 2k" = "I have paid for transport" -> the trader's
    # own expense, NOT someone paying back a debt.
    assert classify_message("mo ti san transport 2k") == "entry"


def test_yoruba_purchase_is_an_entry():
    assert classify_message("mo tún ra bisikíìtì 18000 náírà") == "entry"
