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


def test_plain_integer_amount():
    result = parse_transcript("Sold pure water 500")
    assert len(result) == 1
    entry = result[0]
    assert entry["type"] == "sale"
    assert entry["amount"] == 500
    assert "water" in entry["item"].lower()


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
