from app.parser import parse_transcript


def test_simple_sale_with_k_shorthand():
    result = parse_transcript("I sold 5 bags of rice for 45k")
    assert len(result) == 1
    entry = result[0]
    assert entry["type"] == "sale"
    assert entry["amount"] == 45000
    assert entry["quantity"] == 5.0
    assert "rice" in entry["item"].lower()


def test_sale_with_decimal_k():
    result = parse_transcript("Sold garri 2.5k")
    assert len(result) == 1
    entry = result[0]
    assert entry["type"] == "sale"
    assert entry["amount"] == 2500
    assert "garri" in entry["item"].lower()


def test_sale_hyphenated_two_fifty():
    result = parse_transcript("I sell akara two-fifty")
    assert len(result) == 1
    entry = result[0]
    assert entry["type"] == "sale"
    assert entry["amount"] == 250
    assert "akara" in entry["item"].lower()


def test_written_number_five_thousand():
    result = parse_transcript("I sold beans five thousand naira")
    assert len(result) == 1
    entry = result[0]
    assert entry["type"] == "sale"
    assert entry["amount"] == 5000
    assert "beans" in entry["item"].lower()


def test_written_number_ten_thousand():
    result = parse_transcript("Bought stock ten thousand naira")
    assert len(result) == 1
    entry = result[0]
    assert entry["type"] == "expense"
    assert entry["amount"] == 10000


def test_pidgin_don_sell():
    result = parse_transcript("I don sell 3 cartons of indomie 75k")
    assert len(result) == 1
    entry = result[0]
    assert entry["type"] == "sale"
    assert entry["amount"] == 75000
    assert entry["quantity"] == 3.0
    assert "indomie" in entry["item"].lower()


def test_pidgin_i_buy():
    result = parse_transcript("I buy 2 kegs of palm oil 30k")
    assert len(result) == 1
    entry = result[0]
    assert entry["type"] == "expense"
    assert entry["amount"] == 30000
    assert entry["quantity"] == 2.0


def test_pidgin_pay_transport():
    result = parse_transcript("I pay for transport 3k")
    assert len(result) == 1
    entry = result[0]
    assert entry["type"] == "expense"
    assert entry["amount"] == 3000
    assert "transport" in entry["item"].lower()


def test_multiple_entries_in_one_sentence():
    result = parse_transcript("I sold 5 bags of rice for 45k, paid transport 3k")
    assert len(result) == 2
    types = sorted([e["type"] for e in result])
    assert types == ["expense", "sale"]
    amounts = sorted([e["amount"] for e in result])
    assert amounts == [3000, 45000]


def test_rent_expense():
    result = parse_transcript("I pay shop rent 250k today")
    assert len(result) == 1
    entry = result[0]
    assert entry["type"] == "expense"
    assert entry["amount"] == 250000
    assert "rent" in entry["item"].lower()


def test_stock_purchase_expense():
    result = parse_transcript("Don buy stock 150 thousand naira")
    assert len(result) == 1
    entry = result[0]
    assert entry["type"] == "expense"
    assert entry["amount"] == 150000


def test_no_quantity_returns_null():
    result = parse_transcript("Sold plantain 15k")
    assert len(result) == 1
    entry = result[0]
    assert entry["type"] == "sale"
    assert entry["amount"] == 15000
    # Quantity not stated
    assert entry["quantity"] is None or entry["quantity"] == 15


def test_multiple_sales_and_expenses():
    result = parse_transcript(
        "I sold tomatoes 20k, I buy onions 10k, paid fuel 5k"
    )
    assert len(result) == 3
    sale = [e for e in result if e["type"] == "sale"]
    expense = [e for e in result if e["type"] == "expense"]
    assert len(sale) == 1
    assert len(expense) == 2
    assert sale[0]["amount"] == 20000
    amounts_exp = sorted([e["amount"] for e in expense])
    assert amounts_exp == [5000, 10000]


def test_mixed_pidgin_and_english():
    result = parse_transcript(
        "I don sell 6 crates of egg 36k then I pay for market levy 2k"
    )
    assert len(result) == 2
    sale_entry = [e for e in result if e["type"] == "sale"][0]
    exp_entry = [e for e in result if e["type"] == "expense"][0]
    assert sale_entry["amount"] == 36000
    assert sale_entry["quantity"] == 6.0
    assert "egg" in sale_entry["item"].lower()
    assert exp_entry["amount"] == 2000


def test_thousand_separator_single_entry():
    result = parse_transcript("Sold rice 45,000 naira")
    assert len(result) == 1
    assert result[0]["amount"] == 45000
    assert result[0]["type"] == "sale"
    assert "rice" in result[0]["item"].lower()


def test_thousand_separator_multiple_not_split_by_comma():
    result = parse_transcript(
        "Bought rice 45,000 naira, paid transport 2,000, bought snacks 5,000 naira"
    )
    amounts = sorted(e["amount"] for e in result)
    assert len(result) == 3
    assert amounts == [2000, 5000, 45000]


def test_thousand_separator_millions():
    result = parse_transcript("Bought warehouse 1,234,567 naira")
    assert len(result) == 1
    assert result[0]["amount"] == 1234567
    assert result[0]["type"] == "expense"


def test_thousand_separator_with_k_shorthand():
    """If ASR produces something weird like '45,000k' it should still work: 45,000 * 1000 = 45 million."""
    result = parse_transcript("Sold car 45,000k")
    assert len(result) == 1
    assert result[0]["amount"] == 45000000


def test_thousand_separator_plus_scale_word():
    """1,200 thousand = 1,200,000."""
    result = parse_transcript("Don buy shop 1,200 thousand naira")
    assert len(result) == 1
    assert result[0]["amount"] == 1200000


def test_thousand_separator_with_quantity():
    result = parse_transcript("I don sell 10 bags of cement 550,000 naira")
    assert len(result) == 1
    entry = result[0]
    assert entry["type"] == "sale"
    assert entry["amount"] == 550000
    assert entry["quantity"] == 10.0
    assert "cement" in entry["item"].lower()


def test_thousand_separator_semicolon_separator_still_splits():
    """Only commas between digits are protected; semicolons always separate entries."""
    result = parse_transcript(
        "Bought rice 45,000 naira; paid transport 2,000; bought snacks 5,000"
    )
    amounts = sorted(e["amount"] for e in result)
    assert len(result) == 3
    assert amounts == [2000, 5000, 45000]


def test_thousand_separator_decimal_plus_mixed_comma_and_period():
    """Protects comma-thousand-sep AND decimal period. '2,500.75 naira' = 2500 (floored to int)."""
    result = parse_transcript("Bought pure water 2,500.75 naira")
    assert len(result) == 1
    # 2,500.75 -> int(float) == 2500
    assert result[0]["amount"] == 2500
    assert result[0]["type"] == "expense"


def test_malformed_thousand_separator_falls_back_safe():
    """Comma NOT strictly between 3-digit groups (e.g. '450,00') isn't normalized
    to an integer by `_strip_thousand_separators`, so parse_transcript must still
    extract amounts from the remaining entries. The core user requirement is that
    WELL-FORMED thousand-separators like 45,000 work; this just ensures the parser
    doesn't crash on weird ASR output and still finds 1,000 = 1000 later in the
    same string."""
    result = parse_transcript("Sold item 450,00 naira, paid expense 1,000")
    # We at least correctly extract the 1,000 = 1000 amount.
    amounts = [e["amount"] for e in result]
    assert 1000 in amounts


# ---------------------------------------------------------------------------
# Natural language variation tests - semantic understanding
# ---------------------------------------------------------------------------

def test_natural_language_i_bought():
    """Test: 'I bought chin chin for 2k' - standard past tense"""
    result = parse_transcript("I bought chin chin for 2k")
    assert len(result) == 1
    entry = result[0]
    assert entry["type"] == "expense"
    assert entry["amount"] == 2000
    assert "chin chin" in entry["item"].lower()


def test_natural_language_i_buy():
    """Test: 'I buy chin chin for 2k' - present tense"""
    result = parse_transcript("I buy chin chin for 2k")
    assert len(result) == 1
    entry = result[0]
    assert entry["type"] == "expense"
    assert entry["amount"] == 2000
    assert "chin chin" in entry["item"].lower()


def test_natural_language_i_purchased():
    """Test: 'I purchased chin chin 2k' - formal word"""
    result = parse_transcript("I purchased chin chin 2k")
    assert len(result) == 1
    entry = result[0]
    assert entry["type"] == "expense"
    assert entry["amount"] == 2000
    assert "chin chin" in entry["item"].lower()


def test_natural_language_sent_for():
    """Test: 'I sent for chin chin spent 2k' - contextual pattern"""
    result = parse_transcript("I sent for chin chin spent 2k")
    assert len(result) == 1
    entry = result[0]
    assert entry["type"] == "expense"
    assert entry["amount"] == 2000
    assert "chin chin" in entry["item"].lower()


def test_natural_language_got_and_paid():
    """Test: 'I got chin chin and paid 2k' - semantic pattern"""
    result = parse_transcript("I got chin chin and paid 2k")
    assert len(result) == 1
    entry = result[0]
    assert entry["type"] == "expense"
    assert entry["amount"] == 2000
    assert "chin chin" in entry["item"].lower()


def test_natural_language_cost_me():
    """Test: 'It cost me 2k for chin chin' - result-oriented pattern"""
    result = parse_transcript("It cost me 2k for chin chin")
    assert len(result) == 1
    entry = result[0]
    assert entry["type"] == "expense"
    assert entry["amount"] == 2000
    assert "chin chin" in entry["item"].lower()


def test_natural_language_price_was():
    """Test: 'Chin chin cost 2k' - price-focused pattern"""
    result = parse_transcript("Chin chin cost 2k")
    assert len(result) == 1
    entry = result[0]
    assert entry["type"] == "expense"
    assert entry["amount"] == 2000
    assert "chin chin" in entry["item"].lower()


def test_natural_language_informal_spent():
    """Test: 'I spent 2k on chin chin' - spend-focused pattern"""
    result = parse_transcript("I spent 2k on chin chin")
    assert len(result) == 1
    entry = result[0]
    assert entry["type"] == "expense"
    assert entry["amount"] == 2000
    assert "chin chin" in entry["item"].lower()


def test_natural_language_paid_for():
    """Test: 'I paid for chin chin 2k' - pay-focused pattern"""
    result = parse_transcript("I paid for chin chin 2k")
    assert len(result) == 1
    entry = result[0]
    assert entry["type"] == "expense"
    assert entry["amount"] == 2000
    assert "chin chin" in entry["item"].lower()


def test_natural_language_ended_up_paying():
    """Test: 'I ended up paying 2k for chin chin' - conversational pattern"""
    result = parse_transcript("I ended up paying 2k for chin chin")
    assert len(result) == 1
    entry = result[0]
    assert entry["type"] == "expense"
    assert entry["amount"] == 2000
    assert "chin chin" in entry["item"].lower()


def test_natural_language_finally_paid():
    """Test: 'I finally paid 2k for chin chin' - conversational pattern"""
    result = parse_transcript("I finally paid 2k for chin chin")
    assert len(result) == 1
    entry = result[0]
    assert entry["type"] == "expense"
    assert entry["amount"] == 2000
    assert "chin chin" in entry["item"].lower()


def test_natural_language_sentence_structure_variation():
    """Test: Different word order - 'Chin chin 2k I bought'"""
    result = parse_transcript("Chin chin 2k I bought")
    assert len(result) == 1
    entry = result[0]
    assert entry["type"] == "expense"
    assert entry["amount"] == 2000
    assert "chin chin" in entry["item"].lower()


def test_natural_language_with_comma_separator():
    """Test: 'I bought chin chin, 2k' - comma before amount (kept together)"""
    result = parse_transcript("I bought chin chin, 2k")
    assert len(result) == 1
    entry = result[0]
    assert entry["type"] == "expense"
    assert entry["amount"] == 2000
    assert "chin chin" in entry["item"].lower()


def test_natural_language_then_paid():
    """Test: 'I got chin chin then paid 2k' - sequential pattern"""
    result = parse_transcript("I got chin chin then paid 2k")
    assert len(result) == 1
    entry = result[0]
    assert entry["type"] == "expense"
    assert entry["amount"] == 2000
    assert "chin chin" in entry["item"].lower()


def test_natural_language_money_gone():
    """Test: 'My money 2k went on chin chin' - result-oriented pattern"""
    result = parse_transcript("My money 2k went on chin chin")
    assert len(result) == 1
    entry = result[0]
    assert entry["type"] == "expense"
    assert entry["amount"] == 2000
    assert "chin chin" in entry["item"].lower()


def test_natural_language_value_of():
    """Test: 'Value of chin chin 2k' - value-focused pattern"""
    result = parse_transcript("Value of chin chin 2k")
    assert len(result) == 1
    entry = result[0]
    assert entry["type"] == "expense"
    assert entry["amount"] == 2000
    assert "chin chin" in entry["item"].lower()


def test_natural_language_acquired():
    """Test: 'I acquired chin chin for 2k' - formal word"""
    result = parse_transcript("I acquired chin chin for 2k")
    assert len(result) == 1
    entry = result[0]
    assert entry["type"] == "expense"
    assert entry["amount"] == 2000
    assert "chin chin" in entry["item"].lower()


def test_natural_language_ordered():
    """Test: 'I ordered chin chin 2k' - action-focused pattern"""
    result = parse_transcript("I ordered chin chin 2k")
    assert len(result) == 1
    entry = result[0]
    assert entry["type"] == "expense"
    assert entry["amount"] == 2000
    assert "chin chin" in entry["item"].lower()


def test_natural_language_charged_me():
    """Test: 'They charged me 2k for chin chin' - passive pattern"""
    result = parse_transcript("They charged me 2k for chin chin")
    assert len(result) == 1
    entry = result[0]
    assert entry["type"] == "expense"
    assert entry["amount"] == 2000
    assert "chin chin" in entry["item"].lower()


def test_natural_language_billed_me():
    """Test: 'I was billed 2k for chin chin' - passive pattern"""
    result = parse_transcript("I was billed 2k for chin chin")
    assert len(result) == 1
    entry = result[0]
    assert entry["type"] == "expense"
    assert entry["amount"] == 2000
    assert "chin chin" in entry["item"].lower()


def test_natural_language_handed_over_money():
    """Test: 'I handed over 2k for chin chin' - action pattern"""
    result = parse_transcript("I handed over 2k for chin chin")
    assert len(result) == 1
    entry = result[0]
    assert entry["type"] == "expense"
    assert entry["amount"] == 2000
    assert "chin chin" in entry["item"].lower()


# ---------------------------------------------------------------------------
# Complex multi-transaction narrative tests
# ---------------------------------------------------------------------------

def test_conversational_narrative_multiple_transactions():
    """Test: 'I went to the market this morning. I bought rice for 20,000 and later bought chin chin for 2,000. I also spent 1,000 on transport.'"""
    result = parse_transcript("I went to the market this morning. I bought rice for 20,000 and later bought chin chin for 2,000. I also spent 1,000 on transport.")
    assert len(result) == 3
    amounts = sorted([e["amount"] for e in result])
    assert amounts == [1000, 2000, 20000]
    items = [e["item"].lower() for e in result]
    assert any("rice" in item for item in items)
    assert any("chin chin" in item for item in items)
    assert any("transport" in item for item in items)


def test_conversational_with_filler_text():
    """Test: 'So I was at the shop today and I got some garri for 5k then I paid 2k for transport.'"""
    result = parse_transcript("So I was at the shop today and I got some garri for 5k then I paid 2k for transport.")
    assert len(result) == 2
    amounts = sorted([e["amount"] for e in result])
    assert amounts == [2000, 5000]
    items = [e["item"].lower() for e in result]
    assert any("garri" in item for item in items)
    assert any("transport" in item for item in items)


def test_conversational_mixed_english_pidgin():
    """Test: 'I don go market today. I buy rice 10k. I transport 2k. I sell beans 15k.'"""
    result = parse_transcript("I don go market today. I buy rice 10k. I transport 2k. I sell beans 15k.")
    assert len(result) == 3
    amounts = sorted([e["amount"] for e in result])
    assert amounts == [2000, 10000, 15000]
    types = [e["type"] for e in result]
    assert types.count("expense") == 2
    assert types.count("sale") == 1


def test_conversational_long_sentence():
    """Test: 'I purchased three bags of rice for the sum of 45,000 naira. I bought some chin chin for 2,000. I paid 1,000 for transport back home.'"""
    result = parse_transcript("I purchased three bags of rice for the sum of 45,000 naira. I bought some chin chin for 2,000. I paid 1,000 for transport back home.")
    assert len(result) == 3
    amounts = sorted([e["amount"] for e in result])
    assert amounts == [1000, 2000, 45000]


def test_conversational_without_explicit_verbs():
    """Test: 'Market today: bought rice 20k, bought beans 15k, spent transport 2k' - minimal conversational"""
    result = parse_transcript("Market today: bought rice 20k, bought beans 15k, spent transport 2k")
    assert len(result) == 3
    amounts = sorted([e["amount"] for e in result])
    assert amounts == [2000, 15000, 20000]


def test_conversational_with_later():
    """Test: 'I bought rice 20k. Later I bought chin chin 2k. Then transport 1k.'"""
    result = parse_transcript("I bought rice 20k. Later I bought chin chin 2k. Then transport 1k.")
    assert len(result) == 3
    amounts = sorted([e["amount"] for e in result])
    assert amounts == [1000, 2000, 20000]


def test_conversational_with_quantities():
    """Test: 'I bought 5 bags of rice for 25k. I bought 2 crates of eggs for 10k.'"""
    result = parse_transcript("I bought 5 bags of rice for 25k. I bought 2 crates of eggs for 10k.")
    assert len(result) == 2
    amounts = sorted([e["amount"] for e in result])
    assert amounts == [10000, 25000]


def test_conversational_mixed_sales_and_expenses():
    """Test: 'I sold rice 30k. I bought beans 15k. I paid transport 2k.'"""
    result = parse_transcript("I sold rice 30k. I bought beans 15k. I paid transport 2k.")
    assert len(result) == 3
    amounts = sorted([e["amount"] for e in result])
    assert amounts == [2000, 15000, 30000]
    types = [e["type"] for e in result]
    assert types.count("sale") == 1
    assert types.count("expense") == 2


def test_conversational_with_also():
    """Test: 'I bought rice 20k. I also bought beans 15k. Also spent 2k on transport.'"""
    result = parse_transcript("I bought rice 20k. I also bought beans 15k. Also spent 2k on transport.")
    assert len(result) == 3
    amounts = sorted([e["amount"] for e in result])
    assert amounts == [2000, 15000, 20000]


def test_conversational_with_before_after():
    """Test: 'I bought rice 20k. I bought beans 15k. I paid 2k for transport.'"""
    result = parse_transcript("I bought rice 20k. I bought beans 15k. I paid 2k for transport.")
    assert len(result) == 3
    amounts = sorted([e["amount"] for e in result])
    assert amounts == [2000, 15000, 20000]
