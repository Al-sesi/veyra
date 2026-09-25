"""
Tests for corrections, deletes and debt tracking through the pipeline + API.

Strategy matches tests/test_ledger.py: a fresh temp SQLite file per test and
the ASR mocked, so the full routing path (intent -> ledger -> reply_text) is
exercised without model downloads, ffmpeg or network.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from app.db import init_db, query_one, query_rows
from app.ledger import list_open_debts


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
    assert "say it again" in result["reply_text"]
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
