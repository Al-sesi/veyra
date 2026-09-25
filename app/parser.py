import re
import unicodedata
from typing import List, Optional, Dict, Any, Tuple


# ---------------------------------------------------------------------------
# Number parsing: convert words and shorthand into integers.
# WORD_NUMBERS maps English number words to their base values.
# ---------------------------------------------------------------------------
WORD_NUMBERS: Dict[str, int] = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20,
    "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70,
    "eighty": 80, "ninety": 90,
    "hundred": 100, "thousand": 1000, "million": 1000000,
}

TENS_WORDS = {"twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"}
ONES_WORDS = {"one", "two", "three", "four", "five", "six", "seven", "eight", "nine"}
SCALE_WORDS = {"hundred", "thousand", "million"}

# ---------------------------------------------------------------------------
# Yoruba number words, stored diacritic-folded ("ẹgbẹ̀rún" -> "egberun").
# Note the order difference: Yoruba puts the scale word BEFORE its multiplier
# ("ẹgbẹ̀rún márùn-ún" = 1000 x 5 = 5,000) while English puts it after
# ("five thousand" = 5 x 1000).
# ---------------------------------------------------------------------------
YORUBA_NUMBER_WORDS: Dict[str, int] = {
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
}

# Short tails glued onto Yoruba number words with a hyphen, e.g. "márùn-ún"
# (= 5) or "ọ̀kàn-ún" (= 1). They carry no numeric value of their own.
YORUBA_NUMBER_TAILS = {"un", "an", "aa"}


def strip_diacritics(text: str) -> str:
    """Fold diacritics to base letters: "Ẹ̀tà" -> "Eta", "ò" -> "o", "ṣ" -> "s".

    ASR output may or may not emit Yoruba tone marks, so both sides of a match
    are folded. Callers that need character offsets (app/intents) must map
    indices back themselves; this helper only transforms the text.
    """
    if not text:
        return text
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def _lookup_number_word(word: str) -> Optional[int]:
    """Value of an English or (diacritic-folded) Yoruba number word, else None.

    Yoruba filler tails are handled too: "márùn-ún" and "marun-un" -> 5.
    """
    w = word.lower().strip(".,!?;:")
    if w in WORD_NUMBERS:
        return WORD_NUMBERS[w]
    folded = strip_diacritics(w)
    if folded in YORUBA_NUMBER_WORDS:
        return YORUBA_NUMBER_WORDS[folded]
    if "-" in folded:
        head, *tails = folded.split("-")
        if tails and all(t in YORUBA_NUMBER_TAILS for t in tails):
            return YORUBA_NUMBER_WORDS.get(head)
    return None


def parse_written_number_words(words: List[str]) -> Optional[int]:
    """
    Parse a list of pure word-only number tokens (no Arabic numerals, no k)
    into an integer. Accepts English and Yoruba number words.
    Example: ["five", "thousand"] -> 5000, ["twenty", "five"] -> 25,
             ["egberun", "marun"] -> 5000 (Yoruba word order).
    Returns None if the tokens don't form a valid number phrase.
    """
    if not words:
        return None

    # Lowercase and strip punctuation for comparison
    clean = [w.lower().strip(".,!?;:") for w in words]
    values: List[int] = []
    for w in clean:
        v = _lookup_number_word(w)
        if v is None:
            return None
        values.append(v)

    # Yoruba puts the scale word first: "ẹgbẹ̀rún márùn-ún" = 1000 * 5.
    # English ("five thousand") starts with the multiplier, so it never
    # enters this branch and keeps the original accumulation logic below.
    if values[0] >= 1000:
        multiplier = sum(values[1:])
        if len(values) == 1:
            return values[0]
        return values[0] * multiplier if multiplier > 0 else None

    total = 0
    current = 0
    for v in values:
        if v >= 1000:  # thousand / million
            current *= v
            total += current
            current = 0
        elif v == 100:  # hundred
            if current == 0:
                current = 1
            current *= v
        else:  # 0-99
            current += v
    total += current
    return total if total > 0 else None


def parse_slang_hyphenated(token: str) -> Optional[int]:
    """
    Parse Nigerian-market hyphenated slang number tokens.
    "two-fifty"  -> 250  (two hundred + fifty)
    "three-twenty" -> 320 (three hundred + twenty)
    "twenty-five" -> 25  (normal, non-slang: twenty + five)
    Returns None if the token is not a valid hyphenated number.
    """
    if "-" not in token:
        return None
    parts = token.lower().strip(".,!?;:").split("-")
    if not all(p in WORD_NUMBERS for p in parts):
        # Yoruba number words can carry a filler tail after a hyphen:
        # "márùn-ún" (= 5), "ọ̀kàn-ún" (= 1), "mẹ́sàn-án" (= 9).
        if len(parts) > 1 and all(
            strip_diacritics(p) in YORUBA_NUMBER_TAILS for p in parts[1:]
        ):
            return _lookup_number_word(parts[0])
        return None

    if len(parts) == 2 and parts[0] in ONES_WORDS and parts[1] in TENS_WORDS:
        # Slang pattern: ones-tens => hundreds-tens. E.g. two-fifty = 250.
        return WORD_NUMBERS[parts[0]] * 100 + WORD_NUMBERS[parts[1]]

    # Normal compound: e.g. twenty-five, forty-two
    return parse_written_number_words(parts)


def _strip_thousand_separators(token: str) -> str:
    """
    Remove thousands separators from a numeric-looking token.

    Comma separators (English/Arabic style, exactly 3 digits per group):
        "45,000"      -> "45000"
        "1,234,567"   -> "1234567"
        "45,000k"     -> "45000k"       (k-shorthand preserved)
        "45,000naira" -> "45000naira"   (currency suffix preserved)
        "2,500.75"    -> "2500.75"      (decimal mixed with thousand-sep)

    Dot separators (seen in Yoruba ASR output, e.g. "5.000 naira" = 5,000):
        "5.000"       -> "5000"
        "1.234.567"   -> "1234567"

    Anything that doesn't cleanly match a separator structure is returned
    untouched (so hyphenated slang like "two-fifty" and decimals like "2.5k"
    keep their punctuation).
    """
    if "," in token:
        # digits, then repeating (comma + 3 digits), optional decimal + digits,
        # optional k/currency suffix.
        m = re.match(
            r"^(\d{1,3}(?:,\d{3})+)(\.\d+)?(k|naira|n|#)?$",
            token,
            flags=re.IGNORECASE,
        )
        if m:
            core = m.group(1).replace(",", "")
            decimal = m.group(2) or ""
            suffix = m.group(3) or ""
            return core + decimal + suffix
        return token
    if "." in token:
        # Exactly 3 digits per dot-group, so "2.5k" stays a decimal.
        m = re.match(
            r"^(\d{1,3}(?:\.\d{3})+)(k|naira|n|#)?$",
            token,
            flags=re.IGNORECASE,
        )
        if m:
            return m.group(1).replace(".", "") + (m.group(2) or "")
        return token
    return token


def parse_tokens_as_number(tokens: List[str]) -> Tuple[Optional[int], Optional[str]]:
    """
    Try to parse a sequence of raw tokens (could include Arabic numerals,
    written words, "k" shorthand, hyphenated slang, or currency markers) as
    a single numeric amount.

    Returns (value, matched_text) or (None, None) if not a number phrase.
    This is the workhorse that combines:
      - Arabic numerals + scale word (e.g. "150 thousand naira" -> 150000)
      - Pure word numbers (e.g. "five thousand" -> 5000)
      - k-shorthand (e.g. "45k" -> 45000, "2.5k" -> 2500)
      - hyphenated slang (e.g. "two-fifty" -> 250)
      - thousand separators (e.g. "45,000" -> 45000, "1,234,567" -> 1234567)
      - Plain numbers with optional currency suffix
    """
    if not tokens:
        return None, None

    raw_text = " ".join(tokens)

    # ------------------------------------------------------------------
    # Case 1: single token with k-shorthand or slang
    # ------------------------------------------------------------------
    if len(tokens) == 1:
        tok = tokens[0].strip().lower().strip(".,!?;:₦$")
        # Strip trailing currency markers: "naira", "n", "#"
        for sfx in ["naira"]:
            if tok.endswith(sfx):
                tok = tok[: -len(sfx)]
        # Normalize thousand separators ("45,000" -> "45000", "2,500k" -> "2500k")
        tok = _strip_thousand_separators(tok)
        # Pattern: 45k, 2.5k
        m = re.match(r"^(\d+(?:\.\d+)?)k$", tok)
        if m:
            return int(float(m.group(1)) * 1000), raw_text
        # Pattern: 45000n or 45000#
        m2 = re.match(r"^(\d+(?:\.\d+)?)[n#]$", tok)
        if m2:
            val = float(m2.group(1))
            return int(val) if val == int(val) else int(val), raw_text
        # Pattern: plain number
        m3 = re.match(r"^(\d+(?:\.\d+)?)$", tok)
        if m3:
            val = float(m3.group(1))
            return int(val) if val == int(val) else int(val), raw_text
        # Pattern: hyphenated slang / compound words
        slang = parse_slang_hyphenated(tok)
        if slang is not None:
            return slang, raw_text
        # Pattern: spelled-out value with a trailing k, e.g. "sevenk" -> 7000
        # (ASR sometimes glues the k-shorthand onto the number word).
        m4 = re.match(r"^([a-z]+)k$", tok)
        if m4:
            base = _lookup_number_word(m4.group(1))
            if base is not None:
                return base * 1000, raw_text
        # Pattern: single word number (English or Yoruba)
        if _lookup_number_word(tok) is not None:
            return parse_written_number_words([tok]), raw_text
        return None, None

    # ------------------------------------------------------------------
    # Case 2: multi-token number phrase
    # Normalize tokens to lowercase/stripped forms preserving order
    # ------------------------------------------------------------------
    stripped = [t.strip().lower().strip(".,!?;:₦$") for t in tokens]
    # Drop trailing currency markers from consideration ("náírà" only differs
    # from "naira" by diacritics, so compare folded forms)
    while stripped and strip_diacritics(stripped[-1]) in {"naira", "n", "#"}:
        stripped.pop()
    if not stripped:
        return None, None
    # Normalize thousand separators on the first (Arabic) token if present.
    # E.g. ["45,000", "thousand"] -> ["45000", "thousand"]
    stripped = [_strip_thousand_separators(s) for s in stripped]

    # Sub-case A: mixed Arabic + scale word(s). E.g. ["150", "thousand"] -> 150000
    if re.match(r"^\d+(?:\.\d+)?$", stripped[0]):
        # First token is Arabic numeral
        try:
            base = float(stripped[0])
            # If only one number word is left
            rest = stripped[1:]
            if len(rest) == 0:
                val = int(base) if base == int(base) else int(base)
                return val, raw_text
            # Check the rest are valid scale words
            total = base
            valid = True
            for r in rest:
                if r in SCALE_WORDS:
                    total *= WORD_NUMBERS[r]
                else:
                    valid = False
                    break
            if valid:
                return int(total), raw_text
        except ValueError:
            pass

    # Sub-case B: pure word-number phrase. E.g. ["five", "thousand"] -> 5000
    val = parse_written_number_words(stripped)
    if val is not None:
        return val, raw_text

    return None, None


def find_numeric_phrases(segment: str) -> List[Dict[str, Any]]:
    """
    Walk the segment and identify every candidate numeric phrase (amount or
    quantity). Returns a list of dicts sorted by position:
      { "value": int, "text": str, "start": int, "end": int,
        "has_k": bool, "has_currency": bool }
    """
    results: List[Dict[str, Any]] = []
    # Split into tokens (whitespace-separated), keeping original positions
    tokens: List[Tuple[int, int, str]] = []  # (start, end, text)
    for m in re.finditer(r"\S+", segment):
        tokens.append((m.start(), m.end(), m.group()))

    n = len(tokens)
    i = 0
    while i < n:
        matched = False
        # Try longest match first (up to 6 tokens = "150 thousand naira" etc.)
        for length in range(min(6, n - i), 0, -1):
            sub_tokens = [tokens[i + j][2] for j in range(length)]
            val, _ = parse_tokens_as_number(sub_tokens)
            if val is not None:
                s = tokens[i][0]
                e = tokens[i + length - 1][1]
                text = segment[s:e]
                # Detect k-shorthand / currency
                lowered = text.lower()
                has_k = bool(re.search(r"\d+(?:\.\d+)?k", lowered))
                has_currency = bool(
                    re.search(r"(naira|[#₦$])", strip_diacritics(lowered))
                ) or (
                    any(t[2].lower().strip(".,!?;:") == "n" and i + length - 1 == j
                        for j, t in enumerate(tokens[i:i+length]))
                )
                results.append({
                    "value": val,
                    "text": text,
                    "start": s,
                    "end": e,
                    "has_k": has_k,
                    "has_currency": has_currency,
                })
                i += length
                matched = True
                break
        if not matched:
            i += 1
    return results


# ---------------------------------------------------------------------------
# Transaction-type detection (sale vs expense)
# ---------------------------------------------------------------------------
SALE_WORDS = [
    "sell", "sold", "sells", "selling",
    "don sell", "i don sell", "i sell", "me sell", "sold am",
    "sale", "sales",
]

EXPENSE_WORDS = [
    "buy", "bought", "buys", "buying",
    "don buy", "i don buy", "i buy", "me buy", "buy am",
    "pay", "paid", "pays", "paying",
    "don pay", "i don pay", "i pay", "me pay", "pay am",
    "purchase", "purchased",
    "spend", "spent", "spending",
    "rent", "transport", "fare", "fuel", "petrol", "diesel",
    "stock", "restock",
]

EXPENSE_ITEMS = {"transport", "rent", "fare", "fuel", "petrol", "diesel", "light", "electricity"}

# Yoruba transaction words, stored diacritic-folded ("ra" = buy, "tà" = sell,
# "mọ́tò" = car). The verbs are only two letters, so they MUST be matched on
# word boundaries: substring matching would find "ra" inside "transport".
YORUBA_EXPENSE_WORDS = ["ra", "raa", "dera", "dara"]
YORUBA_SALE_WORDS = ["ta", "taa", "tita"]
YORUBA_EXPENSE_ITEMS = ["moto", "okada"]


def _contains_word(text: str, word: str) -> bool:
    """Whole-word match, used for the short Yoruba words after folding."""
    return re.search(rf"\b{re.escape(word)}\b", text) is not None


def detect_transaction_type(segment: str) -> Optional[str]:
    """
    Return "sale", "expense", or None based on keywords in the segment.
    Handles English, common Nigerian Pidgin phrasing, and Yoruba verbs
    ("ra" = buy, "tà" = sell), matched diacritic-insensitively.
    """
    lowered = segment.lower()
    folded = strip_diacritics(lowered)

    # Strong expense-item signals (transport / rent) override
    for item in EXPENSE_ITEMS:
        if item in lowered:
            return "expense"

    expense_hits = sum(1 for w in EXPENSE_WORDS if w in lowered)
    sale_hits = sum(1 for w in SALE_WORDS if w in lowered)
    expense_hits += sum(1 for w in YORUBA_EXPENSE_WORDS if _contains_word(folded, w))
    expense_hits += sum(1 for w in YORUBA_EXPENSE_ITEMS if _contains_word(folded, w))
    sale_hits += sum(1 for w in YORUBA_SALE_WORDS if _contains_word(folded, w))

    if expense_hits > sale_hits:
        return "expense"
    if sale_hits > expense_hits:
        return "sale"
    return None


# ---------------------------------------------------------------------------
# Item / quantity extraction helpers
# ---------------------------------------------------------------------------
UNIT_WORDS = {
    "bag", "bags", "packet", "packets", "pack", "packs",
    "crate", "crates", "carton", "cartons",
    "piece", "pieces", "pcs", "dozen", "bottle", "bottles",
    "keg", "kegs", "roll", "rolls", "tin", "tins", "can", "cans",
    "cup", "cups", "bowl", "bowls", "paint", "gallon", "gallons",
    "litre", "litres", "liter", "liters", "kg", "kilo", "kilos",
    "meter", "meters", "yard", "yards",
}


def select_amount_and_quantity(numerics: List[Dict[str, Any]]) -> Tuple[Optional[Dict[str, Any]], Optional[float]]:
    """
    Given all numeric phrases found in a segment, pick the one that's the
    *amount* (naira) and optionally a separate *quantity* (count of items).

    Heuristics:
      - Amount strongly prefers phrases with k-shorthand or a currency marker.
      - If ties, the largest value is the amount.
      - If a smaller numeric phrase exists and is not the amount, it is the
        quantity (floored to float if it could be unit count).
    """
    if not numerics:
        return None, None

    # Rank by (has_k or has_currency, value) descending.
    def rank(n: Dict[str, Any]) -> Tuple[int, int]:
        score = 2 if (n["has_k"] or n["has_currency"]) else 0
        return (score, n["value"])

    sorted_n = sorted(numerics, key=rank, reverse=True)
    amount = sorted_n[0]
    amount_val = amount["value"]

    quantity: Optional[float] = None
    # Quantity is the next non-overlapping numeric phrase whose value is
    # substantially smaller than the amount (or amount < 1000, then a
    # different phrase). Limit quantity to <= 1000 range.
    for cand in sorted_n[1:]:
        if cand["start"] >= amount["end"] or cand["end"] <= amount["start"]:
            qv = cand["value"]
            if 0 < qv <= 2000 and qv < max(amount_val, 100):
                quantity = float(qv)
                break
    # Also: if only one numeric phrase but it has k/currency, treat as
    # amount only (no quantity).
    return amount, quantity


def _is_numeric_token(clean: str) -> bool:
    """True if a cleaned token is an amount/quantity: Arabic numerals,
    k-shorthand, hyphenated slang, or an English/Yoruba number word."""
    if re.match(r"^\d+(?:\.\d+)?(k)?$", clean):
        return True
    if _lookup_number_word(clean) is not None:
        return True
    if "-" in clean:
        parts = clean.split("-")
        if all(p in WORD_NUMBERS for p in parts):
            return True
        if len(parts) > 1 and all(
            strip_diacritics(p) in YORUBA_NUMBER_TAILS for p in parts[1:]
        ):
            return _lookup_number_word(parts[0]) is not None
    return False


def extract_item(segment: str, amount_text: str, quantity_text: Optional[str], tx_type: str) -> str:
    """
    Build a short item description by stripping amount, quantity, transaction
    verbs, unit words, and common stopwords from the segment.
    """
    text = segment
    if amount_text:
        text = text.replace(amount_text, " ")
    if quantity_text:
        text = text.replace(quantity_text, " ")

    stopwords = set(UNIT_WORDS) | {
        "i", "me", "my", "for", "the", "a", "an", "of", "and", "or",
        "is", "was", "be", "been", "being", "am", "are",
        "sold", "sell", "sells", "selling", "don",
        "bought", "buy", "buys", "buying",
        "paid", "pay", "pays", "paying",
        "spent", "spend", "spending",
        "purchased", "purchase",
        "at", "price", "cost", "worth", "naira", "n", "#", "₦", "$",
        "with", "to", "from", "on", "in", "into", "then",
        "today", "yesterday", "tomorrow", "now", "just",
        "am", "dem", "dey", "been", "market", "shop", "store",
        "levy", "fee",
        # Yoruba particles/verbs (compared diacritic-folded)
        "mo", "lo", "si", "ni", "ra", "ta", "tun", "ba", "wa", "wo", "mi",
        "dewo", "dera", "dara", "pelu", "ati", "lowo", "owo", "loja",
        "soja", "lonii",
    }

    words = text.split()
    filtered: List[str] = []
    for w in words:
        clean = w.strip(".,!?;:").lower()
        if _is_numeric_token(clean):
            continue
        if strip_diacritics(clean) in stopwords:
            continue
        filtered.append(w.strip(".,!?;:"))

    result = " ".join(w for w in filtered if w).strip()

    if not result and tx_type == "expense":
        lowered = segment.lower()
        for item in EXPENSE_ITEMS:
            if item in lowered:
                return item.capitalize()
        if "levy" in lowered:
            return "Levy"
        if "stock" in lowered:
            return "Stock"

    if result:
        return result[0].upper() + result[1:]
    return "Item"


def find_quantity_text(segment: str, amount: Dict[str, Any], numerics: List[Dict[str, Any]]) -> Optional[str]:
    """Find the raw text of the quantity phrase (if any) to strip it from item."""
    for n in numerics:
        if n["start"] >= amount["end"] or n["end"] <= amount["start"]:
            if n["value"] != amount["value"]:
                return n["text"]
    return None


# ---------------------------------------------------------------------------
# Transcript splitting (one sentence -> multiple transactions)
# ---------------------------------------------------------------------------
def split_transactions(text: str) -> List[str]:
    """
    Split a transcript into individual transaction segments by:
      - commas that are NOT between digits (i.e. NOT thousand separators like 45,000)
      - semicolons
      - periods that are NOT part of decimal numbers (e.g. 2.5k) and NOT part of
        thousand separators (although thousand separators use commas, not periods)
      - connector words: " and ", " then "

    Examples:
      "45,000 naira, transport 2,000"  -> ["45,000 naira", "transport 2,000"]
      "Sold rice 2.5k. Bought beans 15k." -> ["Sold rice 2.5k", "Bought beans 15k"]
    """
    normalized = re.sub(r"\s+(and|then)\s+", " , ", text, flags=re.IGNORECASE)
    # Split rules (split on the match, then drop the separator):
    #   1. Semicolon `;`         -> always split.
    #   2. Comma `,`             -> split ONLY if it is NOT flanked by digits on BOTH sides.
    #                               (A comma between digits is a thousand separator like 45,000.)
    #   3. Period `.`            -> split ONLY if it is NOT flanked by digits on BOTH sides.
    #                               (A period between digits is a decimal like 2.5k.)
    #
    # Regex explanation:
    #   `;`                     -> any semicolon
    #   `,(?!(?<=\d,)\d)`       -> comma NOT followed by digit that itself follows digit-comma
    #      Equivalent: split on comma unless the comma is strictly between two digits.
    parts = re.split(
        r";|,(?!(?<=\d,)\d)|(?<!\d)\.(?!\d)",
        normalized,
    )
    return [p.strip() for p in parts if p.strip()]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def parse_transcript(text: str) -> List[Dict[str, Any]]:
    """
    Parse a trader's voice-note transcript into structured bookkeeping entries.

    Parameters
    ----------
    text : str
        What the trader said (English, Pidgin, or mixed). Examples:
        - "I sold 5 bags of rice for 45k, paid transport 3k"
        - "I don sell akara two-fifty"
        - "Don buy stock 150 thousand naira"

    Returns
    -------
    list[dict]
        Each entry has:
          - item (str): short description of the item
          - quantity (float | None): how many of the item, or None if unstated
          - amount (int): amount in naira
          - type (str): "sale" or "expense"
    """
    entries: List[Dict[str, Any]] = []
    segments = split_transactions(text)

    for seg in segments:
        tx_type = detect_transaction_type(seg)
        if tx_type is None:
            continue

        numerics = find_numeric_phrases(seg)
        if not numerics:
            continue

        amount, quantity = select_amount_and_quantity(numerics)
        if amount is None:
            continue

        qty_text = find_quantity_text(seg, amount, numerics)
        item = extract_item(seg, amount["text"], qty_text, tx_type)

        entries.append({
            "item": item,
            "quantity": quantity,
            "amount": amount["value"],
            "type": tx_type,
        })

    return entries
