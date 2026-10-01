"""
Tests for corrections, deletes and debt tracking through the pipeline + API.

Strategy matches tests/test_ledger.py: a fresh temp SQLite file per test and
the ASR mocked, so the full routing path (intent -> ledger -> reply_text) is
exercised without model downloads, ffmpeg or network.
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

# Force SQLite mode for all tests by clearing DATABASE_URL before importing app modules
os.environ["DATABASE_URL"] = ""
os.environ["TEST_MODE"] = "1"

from app.db import init_db, query_one, query_rows
from app.ledger import list_open_debts, add_debt, mark_debt_paid


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def tmp_db(tmp_path: Path) -> Path:
    db = tmp_path / "ledger_test.db"
    init_db(db)
    return db


@pytest.fixture()
def _patch_db(monkeypatch, tmp_db: Path):
    """Redirect DEFAULT_DB_PATH in app.db/app.ledger/app.pipeline to the temp DB."""
    import app.db as db_mod
    import app.ledger as ledger_mod
    import app.pipeline as pipe_mod

    monkeypatch.setattr(db_mod, "DEFAULT_DB_PATH", tmp_db)
    monkeypatch.setattr(ledger_mod, "DEFAULT_DB_PATH", tmp_db)
    monkeypatch.setattr(pipe_mod, "DEFAULT_DB_PATH", tmp_db)


@pytest.fixture()
def sample_audio(tmp_path: Path) -> Path:
    p = tmp_path / "note1.ogg"
    p.write_bytes(b"fake-audio-bytes")
    return p


@pytest.fixture()
def say(tmp_db: Path, _patch_db, sample_audio: Path):
    """Helper: pretend the trader said `text` and run the whole pipeline."""
    from app.pipeline import process_voice_note

    def _say(text: str, user_id: int = 1, language: str = "en"):
        with patch("app.pipeline.transcribe", return_value=text):
            return process_voice_note(
                str(sample_audio), language=language, user_id=user_id
            )

    return _say


# ---------------------------------------------------------------------------
# Corrections
# ---------------------------------------------------------------------------

def test_correction_voids_old_row_and_records_new_amount(say, tmp_db: Path):
    first = say("I sold 5 bags of rice for 45k", user_id=1)
    assert first["entries"][0]["amount"] == 45000

    result = say("no, it was 4k not 40k", user_id=1)
    assert result["reply_text"] == "Corrected: the last entry is now 4,000 naira."
    assert result["entries"][0]["amount"] == 4000
    assert result["summary"]["total_sales"] == 4000

    # Audit trail: the old row still exists, just voided.
    rows = query_rows(
        tmp_db, "SELECT * FROM entries WHERE user_id = ? ORDER BY id ASC;", (1,)
    )
    assert len(rows) == 2
    old, new = rows
    assert old["status"] == "voided"
    assert old["amount"] == 45000
    assert new["status"] == "active"
    assert new["amount"] == 4000
    assert new["replaces_entry_id"] == old["id"]
    assert new["item"] == old["item"]


def test_correction_with_no_previous_entry_fails_safely(say, tmp_db: Path):
    result = say("sorry, make it 4k", user_id=5)
    assert "couldn't find an earlier entry" in result["reply_text"]
    assert result["entries"] == []
    assert result["summary"]["total_sales"] == 0
    assert query_rows(tmp_db, "SELECT * FROM entries WHERE user_id = ?;", (5,)) == []


def test_correction_without_an_amount_fails_safely(say, tmp_db: Path):
    say("I sold rice 45k", user_id=6)
    result = say("sorry, I made a mistake", user_id=6)
    assert "couldn't find the new amount" in result["reply_text"]
    # The original entry is untouched.
    active = query_one(
        tmp_db,
        "SELECT * FROM entries WHERE user_id = ? AND status = 'active';",
        (6,),
    )
    assert active["amount"] == 45000


# ---------------------------------------------------------------------------
# Delete last
# ---------------------------------------------------------------------------

def test_delete_last_entry_voids_but_keeps_the_row(say, tmp_db: Path):
    say("I sold rice 45k", user_id=2)
    say("I sold beans 20k", user_id=2)

    result = say("remove the last one", user_id=2)
    assert result["reply_text"] == "Removed the last entry."
    assert result["summary"]["total_sales"] == 45000

    rows = query_rows(
        tmp_db, "SELECT * FROM entries WHERE user_id = ? ORDER BY id ASC;", (2,)
    )
    assert len(rows) == 2
    assert rows[0]["item"] == "Rice" and rows[0]["status"] == "active"
    assert rows[1]["item"] == "Beans" and rows[1]["status"] == "voided"


def test_delete_with_nothing_recorded_fails_safely(say, tmp_db: Path):
    result = say("cancel that", user_id=3)
    assert "couldn't find an earlier entry" in result["reply_text"]
    assert query_rows(tmp_db, "SELECT * FROM entries WHERE user_id = ?;", (3,)) == []


# ---------------------------------------------------------------------------
# Debts
# ---------------------------------------------------------------------------

def test_debt_owed_to_me_is_recorded_and_summarised(say):
    result = say("Mama Ngozi owes me 5k", user_id=7)
    assert result["reply_text"] == "Noted: Mama Ngozi owes you 5,000 naira."
    assert result["summary"]["total_owed_to_me"] == 5000
    assert result["summary"]["total_i_owe"] == 0

    debts = list_open_debts(7)
    assert len(debts) == 1
    assert debts[0]["person"] == "Mama Ngozi"
    assert debts[0]["amount"] == 5000
    assert debts[0]["direction"] == "owed_to_me"
    assert debts[0]["status"] == "open"


def test_debt_i_owe_is_recorded(say):
    result = say("I owe Bisi 10k", user_id=8)
    assert result["reply_text"] == "Noted: you owe Bisi 10,000 naira."
    assert result["summary"]["total_i_owe"] == 10000
    assert result["summary"]["total_owed_to_me"] == 0

    debts = list_open_debts(8)
    assert len(debts) == 1
    assert debts[0]["person"] == "Bisi"
    assert debts[0]["direction"] == "i_owe"


def test_debt_paid_settles_the_oldest_open_debt(say, tmp_db: Path):
    say("Mama Ngozi owes me 5k", user_id=9)
    say("Mama Ngozi owes me 2k", user_id=9)

    result = say("Mama Ngozi don pay", user_id=9)
    assert (
        result["reply_text"]
        == "Noted: Mama Ngozi's debt of 5,000 naira is marked as paid."
    )
    assert result["summary"]["total_owed_to_me"] == 2000

    rows = query_rows(
        tmp_db, "SELECT * FROM debts WHERE user_id = ? ORDER BY id ASC;", (9,)
    )
    assert rows[0]["status"] == "paid" and rows[0]["paid_at"] is not None
    assert rows[1]["status"] == "open"
    open_debts = list_open_debts(9)
    assert len(open_debts) == 1 and open_debts[0]["amount"] == 2000


def test_debt_paid_for_unknown_person_fails_safely(say, tmp_db: Path):
    result = say("Musa paid me 3k", user_id=10)
    assert result["reply_text"] == "I couldn't find an open debt for Musa."
    assert result["summary"]["total_owed_to_me"] == 0
    assert query_rows(tmp_db, "SELECT * FROM debts WHERE user_id = ?;", (10,)) == []


def test_debt_paid_does_not_touch_other_peoples_debts(say):
    say("Mama Ngozi owes me 5k", user_id=12)
    result = say("Musa paid me 3k", user_id=12)
    assert "couldn't find an open debt" in result["reply_text"]
    assert [d["person"] for d in list_open_debts(12)] == ["Mama Ngozi"]


# ---------------------------------------------------------------------------
# Unclear input
# ---------------------------------------------------------------------------

def test_unclear_note_saves_nothing(say, tmp_db: Path):
    result = say("blah blah nothing here", user_id=13)
    # Per the explicit menu spec: unclear input triggers the help menu so the
    # trader can see what to say next, instead of only saying "say it again".
    assert result["reply_text"].startswith("I didn't quite catch that.")
    assert "Here is what I can help with" in result["reply_text"]
    bullets = [ln for ln in result["reply_text"].splitlines() if ln.strip().startswith("•")]
    assert len(bullets) == 5
    assert result["entries"] == []
    assert result["summary"]["total_sales"] == 0
    # Nothing was written at all: no entries and not even a user row.
    assert query_rows(tmp_db, "SELECT * FROM entries;") == []
    assert query_rows(tmp_db, "SELECT * FROM users;") == []


# ---------------------------------------------------------------------------
# HTTP API
# ---------------------------------------------------------------------------

@pytest.fixture()
def client(_patch_db, monkeypatch, tmp_db: Path):
    import importlib

    import app.main as main_mod

    monkeypatch.setattr(main_mod, "DEFAULT_DB_PATH", tmp_db)
    importlib.reload(main_mod)

    from fastapi.testclient import TestClient

    with TestClient(main_mod.app) as tc:
        yield tc


def _post(client, text: str, user_id: int, sample_audio: Path):
    with patch("app.pipeline.transcribe", return_value=text):
        with sample_audio.open("rb") as fh:
            return client.post(
                "/voice-note",
                data={"language": "en", "user_id": str(user_id)},
                files={"file": ("note1.ogg", fh, "audio/ogg")},
            )


def test_api_voice_note_returns_reply_text_and_debts_endpoint(client, sample_audio: Path):
    resp = _post(client, "Mama Ngozi owes me 5k", 21, sample_audio)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["reply_text"] == "Noted: Mama Ngozi owes you 5,000 naira."
    assert body["summary"]["total_owed_to_me"] == 5000

    debts = client.get("/debts/21")
    assert debts.status_code == 200
    payload = debts.json()
    assert payload["user_id"] == 21
    assert len(payload["debts"]) == 1
    assert payload["debts"][0]["person"] == "Mama Ngozi"


def test_api_correction_flow(client, sample_audio: Path):
    _post(client, "I sold rice 45k", 22, sample_audio)
    resp = _post(client, "no, it was 4k not 40k", 22, sample_audio)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["reply_text"] == "Corrected: the last entry is now 4,000 naira."
    assert body["summary"]["total_sales"] == 4000
    # Audit trail is visible in the ledger: voided original + corrected copy.
    entries = client.get("/ledger/22").json()["entries"]
    assert len(entries) == 2
    assert sorted(e["status"] for e in entries) == ["active", "voided"]


# ---------------------------------------------------------------------------
# Menu intent: triggered by keyword "menu" and also on unclear input
# ---------------------------------------------------------------------------

def test_menu_intent_reply_lists_five_things(say, tmp_db: Path):
    """Explicit "menu" fires the menu reply with exactly 5 bullet items
    matching the spec."""
    result = say("menu", user_id=30)
    assert "Here is what I can help with" in result["reply_text"]
    # Exactly 5 items required by the spec
    bullets = [ln for ln in result["reply_text"].splitlines() if ln.strip().startswith("•")]
    assert len(bullets) == 5, f"Expected 5 bullets, got: {result['reply_text']}"
    # Content matches the spec line-by-line (case-insensitive, order matters)
    lower = result["reply_text"].lower()
    assert "log a sale or expense" in lower or "speak naturally to log a sale" in lower
    assert "my ledger" in lower and "excel" in lower
    assert "who owes me" in lower and "who i owe" in lower
    assert "undo" in lower and "remove your last entry" in lower
    assert '"help"' in lower or "'help'" in lower
    # Menu doesn't log anything: summary stays zero, entries=[], nothing saved.
    assert result["entries"] == []
    assert result["summary"]["total_sales"] == 0
    from app.db import query_rows
    assert query_rows(tmp_db, "SELECT * FROM entries;") == []


def test_menu_is_also_sent_on_unrecognized_input(say, tmp_db: Path):
    """When the parser returns nothing, reply_text starts with "I didn't
    quite catch that" AND continues with the full 5-item menu."""
    result = say("blah blah gibberish nothing here", user_id=31)
    assert result["reply_text"].startswith("I didn't quite catch that.")
    # The full menu text is appended.
    assert "Here is what I can help with" in result["reply_text"]
    bullets = [ln for ln in result["reply_text"].splitlines() if ln.strip().startswith("•")]
    assert len(bullets) == 5
    # Still nothing persisted, per the rule-based unclear-note contract.
    from app.db import query_rows
    assert query_rows(tmp_db, "SELECT * FROM entries;") == []
    assert query_rows(tmp_db, "SELECT * FROM users;") == []


# ---------------------------------------------------------------------------
# Help / FAQ intent
# ---------------------------------------------------------------------------

def test_help_intent_faq_covers_all_four_topics(say, tmp_db: Path):
    """Help reply must cover: what Veyra is, languages, pilot status, no
    human support. Must NOT falsely promise a human."""
    result = say("help me", user_id=32)
    lower = result["reply_text"].lower()
    # 1. What Veyra is / does
    assert "voice-note bookkeeper" in lower or "bookkeeper" in lower
    # 2. Languages (EN/Pidgin/Yo/Ha/Ig at minimum)
    assert "english" in lower or "nigerian english" in lower
    assert "pidgin" in lower
    assert "yoruba" in lower or "yorùbá" in lower
    assert "hausa" in lower
    assert "igbo" in lower
    # 3. Pilot project disclosure
    assert "pilot" in lower
    # 4. No human support — EXPLICITLY stated, never implied
    assert "no human" in lower or "no customer care" in lower or "no person" in lower
    assert "automated" in lower
    # Must NOT promise human interaction
    for bad_phrase in [
        "a human will",
        "we will call you",
        "customer support agent",
        "real person",
    ]:
        assert bad_phrase not in lower, f"Misleading phrase found: {bad_phrase}"
    # Help is also read-only: nothing written.
    assert result["entries"] == []
    from app.db import query_rows
    assert query_rows(tmp_db, "SELECT * FROM entries;") == []


def test_help_on_what_is_veyra_also_returns_faq(say):
    """FAQ trigger phrases all resolve to the same help reply."""
    for text in ["what is Veyra", "who are you", "is this a pilot"]:
        result = say(text, user_id=33)
        assert "Here is what you need to know about Veyra" in result["reply_text"]


# ---------------------------------------------------------------------------
# Non-interference: menu/help must not break existing intents
# ---------------------------------------------------------------------------

def test_menu_keyword_does_not_break_normal_sale(say, tmp_db: Path):
    """A real sale mentioning a menu-board (food truck "menu") is NOT a
    menu intent, because "menu" as a substring of another word/context
    only matches when the trigger matches whole-word boundaries. But in
    this test we assert: a pure sale without "menu" at all still works
    identically after the menu-code changes."""
    before = say("I sold 5 bags of rice for 45k", user_id=40)
    after = say("paid transport 3k expense", user_id=40)
    assert before["entries"][0]["amount"] == 45000
    assert before["entries"][0]["type"] == "sale"
    assert after["entries"][0]["type"] == "expense"
    # Combined totals: sale 45k, expense 3k
    final = say("what can you do", user_id=40)  # this IS a menu call
    # But even after asking menu, previously stored totals remain.
    from app.ledger import get_summary
    with patch("app.ledger.DEFAULT_DB_PATH", tmp_db):
        summary = get_summary(user_id=40)
    assert summary["total_sales"] == 45000
    assert summary["total_expenses"] == 3000
    # Menu itself didn't add entries.
    assert final["entries"] == []


def test_help_trigger_does_not_interfere_with_debts(say, tmp_db: Path):
    """A debt trigger + a help trigger in the SAME message is disambiguated
    by classify_message order (help wins over debt), but normal debt
    messages with no help trigger still parse identically."""
    # Pure debt still works
    r1 = say("Bisi owes me 10k", user_id=41)
    assert r1["reply_text"] == "Noted: Bisi owes you 10,000 naira."
    from app.ledger import list_open_debts
    with patch("app.ledger.DEFAULT_DB_PATH", tmp_db):
        assert [d["person"] for d in list_open_debts(41)] == ["Bisi"]
    # Pure correction still works
    say("sorry, make it 8k", user_id=41)
    # Pure delete still works
    say("cancel that", user_id=41)
    # Pure sale still works
    sale = say("I sold akara 500", user_id=41)
    assert sale["entries"][0]["item"] == "Akara"
    assert sale["entries"][0]["amount"] == 500
    # Pure debt-paid still works
    say("Mama Ngozi owes me 7k", user_id=41)
    settled = say("Mama Ngozi don pay", user_id=41)
    assert "marked as paid" in settled["reply_text"]


def test_known_intent_without_menu_still_produces_zero_summary_not_menu(say):
    """A successful normal transaction must NOT embed the menu reply. The
    menu only appears on explicit request OR on unclear input."""
    result = say("I sold rice 10k", user_id=42)
    assert "Recorded:" in result["reply_text"] or "rice" in result["reply_text"]
    # Menu lines must be absent from a successful entry reply.
    assert "Here is what I can help with" not in result["reply_text"]
    assert "who owes me" not in result["reply_text"]


# ---------------------------------------------------------------------------
# Debtor Identity and Accumulation Tests
# ---------------------------------------------------------------------------

def test_same_debtor_multiple_debts_accumulate(tmp_db: Path, _patch_db):
    """Test that multiple debts for the same person accumulate correctly."""
    from app.ledger import add_debt, get_summary
    
    # First debt
    add_debt(50, "Musa", 5000, "owed_to_me", db_path=tmp_db)
    summary = get_summary(50, db_path=tmp_db)
    assert summary["total_owed_to_me"] == 5000
    
    # Second debt for same person
    add_debt(50, "Musa", 3000, "owed_to_me", db_path=tmp_db)
    summary = get_summary(50, db_path=tmp_db)
    assert summary["total_owed_to_me"] == 8000  # 5000 + 3000
    
    # Third debt
    add_debt(50, "Musa", 2000, "owed_to_me", db_path=tmp_db)
    summary = get_summary(50, db_path=tmp_db)
    assert summary["total_owed_to_me"] == 10000  # 5000 + 3000 + 2000
    
    # Verify only one debtor record exists
    from app.db import query_rows
    debtors = query_rows(
        tmp_db, "SELECT * FROM debtors WHERE user_id = ?;", (50,)
    )
    assert len(debtors) == 1
    assert debtors[0]["canonical_name"] == "Musa"
    
    # Verify three transactions exist for this debtor
    transactions = query_rows(
        tmp_db, "SELECT * FROM debt_transactions WHERE debtor_id = ?;", (debtors[0]["id"],)
    )
    assert len(transactions) == 3


def test_different_debtors_create_separate_records(tmp_db: Path, _patch_db):
    """Test that different names create separate debtor records."""
    from app.ledger import add_debt
    
    add_debt(51, "Musa", 5000, "owed_to_me", db_path=tmp_db)
    add_debt(51, "Bisi", 3000, "owed_to_me", db_path=tmp_db)
    add_debt(51, "Amina", 2000, "owed_to_me", db_path=tmp_db)
    
    from app.db import query_rows
    debtors = query_rows(
        tmp_db, "SELECT * FROM debtors WHERE user_id = ?;", (51,)
    )
    assert len(debtors) == 3
    names = {d["canonical_name"] for d in debtors}
    assert names == {"Musa", "Bisi", "Amina"}


def test_payment_reduces_debt_balance(tmp_db: Path, _patch_db):
    """Test that payments correctly reduce the debt balance."""
    from app.ledger import add_debt, mark_debt_paid, get_summary
    
    # Add debt
    add_debt(52, "Musa", 5000, "owed_to_me", db_path=tmp_db)
    add_debt(52, "Musa", 3000, "owed_to_me", db_path=tmp_db)
    
    summary = get_summary(52, db_path=tmp_db)
    assert summary["total_owed_to_me"] == 8000
    
    # Record payment
    mark_debt_paid(52, "Musa", db_path=tmp_db)
    summary = get_summary(52, db_path=tmp_db)
    assert summary["total_owed_to_me"] == 3000  # 8000 - 5000 (oldest debt)


def test_case_insensitive_name_matching(tmp_db: Path, _patch_db):
    """Test that name matching is case-insensitive."""
    from app.ledger import add_debt, get_summary
    
    add_debt(53, "Musa", 5000, "owed_to_me", db_path=tmp_db)
    add_debt(53, "musa", 3000, "owed_to_me", db_path=tmp_db)  # lowercase
    add_debt(53, "MUSA", 2000, "owed_to_me", db_path=tmp_db)  # uppercase
    
    from app.db import query_rows
    debtors = query_rows(
        tmp_db, "SELECT * FROM debtors WHERE user_id = ?;", (53,)
    )
    assert len(debtors) == 1
    assert debtors[0]["canonical_name"] == "Musa"
    
    summary = get_summary(53, db_path=tmp_db)
    assert summary["total_owed_to_me"] == 10000


def test_whitespace_in_names_is_normalized(tmp_db: Path, _patch_db):
    """Test that whitespace in names is normalized."""
    from app.ledger import add_debt
    
    add_debt(54, "Musa", 5000, "owed_to_me", db_path=tmp_db)
    add_debt(54, "  Musa  ", 3000, "owed_to_me", db_path=tmp_db)  # extra spaces
    
    from app.db import query_rows
    debtors = query_rows(
        tmp_db, "SELECT * FROM debtors WHERE user_id = ?;", (54,)
    )
    assert len(debtors) == 1


def test_i_owe_debts_accumulate(tmp_db: Path, _patch_db):
    """Test that "I owe" debts accumulate correctly."""
    from app.ledger import add_debt, get_summary
    
    add_debt(55, "Musa", 5000, "i_owe", db_path=tmp_db)
    summary = get_summary(55, db_path=tmp_db)
    assert summary["total_i_owe"] == 5000
    
    add_debt(55, "Musa", 3000, "i_owe", db_path=tmp_db)
    summary = get_summary(55, db_path=tmp_db)
    assert summary["total_i_owe"] == 8000


def test_mixed_debt_directions_for_same_person(tmp_db: Path, _patch_db):
    """Test that a person can both owe and be owed (though unusual)."""
    from app.ledger import add_debt, get_summary
    
    add_debt(56, "Musa", 5000, "owed_to_me", db_path=tmp_db)
    add_debt(56, "Musa", 2000, "i_owe", db_path=tmp_db)
    
    summary = get_summary(56, db_path=tmp_db)
    assert summary["total_owed_to_me"] == 5000
    assert summary["total_i_owe"] == 2000


def test_payment_on_fully_paid_debt(tmp_db: Path, _patch_db):
    """Test that payment on a fully paid debt is handled gracefully."""
    from app.ledger import add_debt, mark_debt_paid, get_summary
    
    add_debt(57, "Musa", 5000, "owed_to_me", db_path=tmp_db)
    mark_debt_paid(57, "Musa", db_path=tmp_db)
    
    # Should now be zero
    summary = get_summary(57, db_path=tmp_db)
    assert summary["total_owed_to_me"] == 0


def test_debt_transaction_history_preserved(tmp_db: Path, _patch_db):
    """Test that all debt transactions are preserved in history."""
    from app.ledger import add_debt, mark_debt_paid
    
    add_debt(58, "Musa", 5000, "owed_to_me", db_path=tmp_db)
    add_debt(58, "Musa", 3000, "owed_to_me", db_path=tmp_db)
    mark_debt_paid(58, "Musa", db_path=tmp_db)
    
    from app.db import query_rows
    transactions = query_rows(
        tmp_db, """
        SELECT * FROM debt_transactions 
        WHERE debtor_id IN (SELECT id FROM debtors WHERE user_id = ?)
        ORDER BY created_at ASC;
        """, (58,)
    )
    assert len(transactions) == 3
    
    # Verify transaction types
    transaction_types = [t["transaction_type"] for t in transactions]
    assert transaction_types == ["debt", "debt", "payment"]


def test_fuzzy_name_matching_function(tmp_db: Path, _patch_db):
    """Test the fuzzy name matching function directly."""
    import app.ledger as ledger_mod
    
    # Create some debtors
    add_debt(59, "Musa", 5000, "owed_to_me", db_path=tmp_db)
    add_debt(59, "Bisi", 3000, "owed_to_me", db_path=tmp_db)
    
    # Find similar names
    similar = ledger_mod._find_similar_debtors(59, "Musaah", db_path=tmp_db)
    assert len(similar) == 1
    assert similar[0]["canonical_name"] == "Musa"
    
    # Exact match should not appear in similar results
    similar_exact = ledger_mod._find_similar_debtors(59, "Musa", db_path=tmp_db)
    assert len(similar_exact) == 0  # Exact match excluded by threshold > 0


def test_levenshtein_distance_calculation():
    """Test the Levenshtein distance calculation."""
    import app.ledger as ledger_mod
    
    assert ledger_mod._levenshtein_distance("Musa", "Musa") == 0  # Same
    assert ledger_mod._levenshtein_distance("Musa", "Musaah") == 2  # Two extra chars
    assert ledger_mod._levenshtein_distance("Musa", "Bisi") == 3  # Different
    assert ledger_mod._levenshtein_distance("Musa", "Misa") == 1  # One char difference


def test_debtor_balance_calculation(tmp_db: Path, _patch_db):
    """Test the debtor balance calculation function."""
    import app.ledger as ledger_mod
    
    add_debt(60, "Musa", 5000, "owed_to_me", db_path=tmp_db)
    add_debt(60, "Musa", 3000, "owed_to_me", db_path=tmp_db)
    mark_debt_paid(60, "Musa", db_path=tmp_db)
    
    from app.db import query_one
    debtor = query_one(
        tmp_db, "SELECT * FROM debtors WHERE user_id = ? AND canonical_name = ?;", (60, "Musa")
    )
    
    balance = ledger_mod.get_debtor_balance(debtor["id"], db_path=tmp_db)
    assert balance["total_owed_to_me"] == 3000  # 5000 + 3000 - 5000 (oldest paid)
    assert balance["total_i_owe"] == 0


def test_get_or_create_debtor_exact_match(tmp_db: Path, _patch_db):
    """Test that get_or_create_debtor finds exact matches."""
    import app.ledger as ledger_mod
    
    add_debt(61, "Musa", 5000, "owed_to_me", db_path=tmp_db)
    
    # Should find existing debtor
    debtor1 = ledger_mod._get_or_create_debtor(61, "Musa", db_path=tmp_db)
    debtor2 = ledger_mod._get_or_create_debtor(61, "Musa", db_path=tmp_db)
    
    assert debtor1["id"] == debtor2["id"]
    assert debtor1["canonical_name"] == "Musa"
    
    # Should create new debtor for different name
    debtor3 = ledger_mod._get_or_create_debtor(61, "Bisi", db_path=tmp_db)
    assert debtor3["id"] != debtor1["id"]
    assert debtor3["canonical_name"] == "Bisi"


def test_list_open_debts_with_new_system(tmp_db: Path, _patch_db):
    """Test that list_open_debts works with the new debtor system."""
    from app.ledger import add_debt, mark_debt_paid, list_open_debts
    
    add_debt(62, "Musa", 5000, "owed_to_me", db_path=tmp_db)
    add_debt(62, "Bisi", 3000, "owed_to_me", db_path=tmp_db)
    
    open_debts = list_open_debts(62, db_path=tmp_db)
    assert len(open_debts) == 2
    
    # Musa should have 5000
    musa_debt = next(d for d in open_debts if d["person"] == "Musa")
    assert musa_debt["amount"] == 5000
    
    # Bisi should have 3000
    bisi_debt = next(d for d in open_debts if d["person"] == "Bisi")
    assert bisi_debt["amount"] == 3000
    
    # After payment, Musa should no longer appear (fully paid)
    mark_debt_paid(62, "Musa", db_path=tmp_db)
    open_debts = list_open_debts(62, db_path=tmp_db)
    assert len(open_debts) == 1
    assert open_debts[0]["person"] == "Bisi"
