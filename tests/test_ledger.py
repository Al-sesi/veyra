"""
Unit tests for the ledger/SQLite + pipeline + FastAPI endpoints.

Strategy for speed and reliability:
  - Each test uses a temporary SQLite file (no leftover state across tests).
  - ASR pipeline is mocked with a deterministic return value so tests don't
    need network, ffmpeg, or model downloads.
  - FastAPI TestClient is used to exercise the HTTP endpoints in-process.
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from app.db import init_db, query_one, query_rows
from app.ledger import add_entries, get_summary, list_entries


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def tmp_db(tmp_path: Path) -> Path:
    """Create a fresh sqlite3 db file in a temp dir and return its path."""
    db = tmp_path / "ledger_test.db"
    init_db(db)
    return db


@pytest.fixture()
def _patch_db(monkeypatch, tmp_db: Path):
    """Redirect DEFAULT_DB_PATH in app.db/app.ledger/app.pipeline to the
    temp DB so all ledger/pipeline tests write to the test file instead of
    the real ledger.db. FastAPI's DEFAULT_DB_PATH is patched separately in
    the `client` fixture so non-FastAPI tests don't import app.main at all.
    """
    import app.db as db_mod
    import app.ledger as ledger_mod
    import app.pipeline as pipe_mod

    monkeypatch.setattr(db_mod, "DEFAULT_DB_PATH", tmp_db)
    monkeypatch.setattr(ledger_mod, "DEFAULT_DB_PATH", tmp_db)
    monkeypatch.setattr(pipe_mod, "DEFAULT_DB_PATH", tmp_db)


# ---------------------------------------------------------------------------
# app.ledger + app.db tests
# ---------------------------------------------------------------------------

def test_add_entries_creates_user_and_rows(tmp_db: Path, _patch_db):
    saved = add_entries(
        user_id=1,
        entries=[
            {"item": "rice", "quantity": 5, "amount": 45000, "type": "sale"},
            {"item": "transport", "quantity": None, "amount": 3000, "type": "expense"},
        ],
        transcript="Sold rice 45k and paid transport 3k",
        audio_file="note1.ogg",
    )
    assert len(saved) == 2
    assert all(r["status"] == "active" for r in saved)
    assert all(r["user_id"] == 1 for r in saved)
    assert [r["amount"] for r in saved] == [45000, 3000]
    user = query_one(tmp_db, "SELECT * FROM users WHERE id = ?;", (1,))
    assert user["language"] == "en"


def test_list_entries_orders_newest_first(tmp_db: Path, _patch_db):
    add_entries(1, [{"item": "rice", "amount": 1000, "type": "sale"}], "t1", "")
    add_entries(1, [{"item": "beans", "amount": 2000, "type": "sale"}], "t2", "")
    items = [r["item"] for r in list_entries(1)]
    assert items == ["beans", "rice"]


def test_get_summary_ignores_voided_and_computes_profit(tmp_db: Path, _patch_db):
    saved = add_entries(
        1,
        [
            {"item": "rice",    "amount": 50000, "type": "sale"},
            {"item": "tomatoes","amount": 20000, "type": "sale"},
            {"item": "transport","amount": 5000,  "type": "expense"},
            {"item": "rent",    "amount": 15000, "type": "expense"},
        ],
        transcript="t",
        audio_file="",
    )
    # Void one sale: rice 50000 removed -> sales should be only 20000
    rice_id = next(r["id"] for r in saved if r["item"] == "rice")
    with patch("app.ledger.DEFAULT_DB_PATH", tmp_db):
        from app.db import get_connection
        with get_connection(tmp_db) as conn:
            conn.execute("UPDATE entries SET status='voided' WHERE id = ?;", (rice_id,))

    summary = get_summary(1, days=7)
    assert summary["total_sales"] == 20000
    assert summary["total_expenses"] == 20000
    assert summary["profit"] == 0
    # Top sale: tomatoes (the only remaining active sale)
    assert summary["top_sale_item"] == {"item": "tomatoes", "total_amount": 20000, "count": 1}
    # Top expense: rent (15000)
    assert summary["top_expense_item"] == {"item": "rent", "total_amount": 15000, "count": 1}


def test_get_summary_empty_user_returns_zeros(tmp_db: Path, _patch_db):
    summary = get_summary(user_id=99, days=7)
    assert summary["total_sales"] == 0
    assert summary["total_expenses"] == 0
    assert summary["profit"] == 0
    assert summary["top_sale_item"] is None
    assert summary["top_expense_item"] is None


def test_add_entries_accepts_empty_list(tmp_db: Path, _patch_db):
    assert add_entries(1, [], "", "") == []


# ---------------------------------------------------------------------------
# app.pipeline + FastAPI endpoint tests (all use mocked ASR)
# ---------------------------------------------------------------------------

TRANSCRIPT_RETURN = "I sold 5 bags of rice for 45,000 naira, paid transport 2,000, bought snacks 5,000 naira."


@pytest.fixture()
def _mock_asr():
    """Make `app.asr.transcribe` return a fixed transcript instead of running
    the real pipeline. This lets the full pipeline end-to-end run on any
    dummy file without model/ffmpeg/network access."""
    with patch("app.pipeline.transcribe", return_value=TRANSCRIPT_RETURN) as m:
        yield m


@pytest.fixture()
def sample_audio(tmp_path: Path) -> Path:
    """Create a tiny dummy binary pretending to be an audio file."""
    p = tmp_path / "note1.ogg"
    p.write_bytes(b"fake-audio-bytes")
    return p


def test_pipeline_process_saves_and_returns_summary(
    sample_audio: Path, tmp_db: Path, _patch_db, _mock_asr
):
    from app.pipeline import process_voice_note

    result = process_voice_note(str(sample_audio), language="en", user_id=42)
    assert result["transcript"] == TRANSCRIPT_RETURN
    assert result["user_id"] == 42
    # 3 entries from the mocked transcript: rice/transport/snacks
    amounts = sorted(e["amount"] for e in result["entries"])
    assert amounts == [2000, 5000, 45000]
    # Summary now reflects these additions
    assert result["summary"]["total_sales"] == 45000
    assert result["summary"]["total_expenses"] == 2000 + 5000
    assert result["summary"]["profit"] == 45000 - 7000
    # Saved rows have transcript + audio_file filled in
    rice = next(e for e in result["entries"] if e["amount"] == 45000)
    assert rice["transcript"] == TRANSCRIPT_RETURN
    assert rice["audio_file"] == sample_audio.name


def test_pipeline_missing_file_raises(tmp_db: Path, _patch_db):
    from app.pipeline import process_voice_note

    with pytest.raises(FileNotFoundError):
        process_voice_note("/tmp/no-such-file.flac", "en", 1)


@pytest.fixture()
def client(_patch_db, _mock_asr, monkeypatch, tmp_db: Path):
    # Start the TestClient after DB path is patched so init_db uses temp file.
    import app.main as main_mod

    monkeypatch.setattr(main_mod, "DEFAULT_DB_PATH", tmp_db)
    # Reload the module so the decorator re-registers routes against the patched
    # DEFAULT_DB_PATH. (Re-import works here because `import` caches by path.)
    import importlib
    importlib.reload(main_mod)

    from fastapi.testclient import TestClient
    with TestClient(main_mod.app) as tc:
        yield tc


def test_api_post_voice_note_full_flow(client, sample_audio: Path, tmp_db: Path):
    with sample_audio.open("rb") as fh:
        resp = client.post(
            "/voice-note",
            data={"language": "en", "user_id": "7"},
            files={"file": ("note1.ogg", fh, "audio/ogg")},
        )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["user_id"] == 7
    assert body["transcript"] == TRANSCRIPT_RETURN
    amounts = sorted(e["amount"] for e in body["entries"])
    assert amounts == [2000, 5000, 45000]
    assert body["summary"]["profit"] == 45000 - (2000 + 5000)


def test_api_get_ledger(client, sample_audio: Path):
    # First: seed entries via the pipeline endpoint
    with sample_audio.open("rb") as fh:
        client.post(
            "/voice-note",
            data={"language": "en", "user_id": "9"},
            files={"file": ("note1.ogg", fh, "audio/ogg")},
        )
    resp = client.get("/ledger/9")
    assert resp.status_code == 200
    data = resp.json()
    assert data["user_id"] == 9
    assert len(data["entries"]) == 3


def test_api_get_summary(client, sample_audio: Path):
    with sample_audio.open("rb") as fh:
        client.post(
            "/voice-note",
            data={"language": "en", "user_id": "11"},
            files={"file": ("note1.ogg", fh, "audio/ogg")},
        )
    resp = client.get("/summary/11?days=7")
    assert resp.status_code == 200
    s = resp.json()
    assert s["user_id"] == 11
    assert s["days"] == 7
    assert s["total_sales"] == 45000
    assert s["total_expenses"] == 2000 + 5000
    assert s["top_sale_item"]["item"] == "Rice"  # parser capitalises item names
    assert s["top_expense_item"]["total_amount"] == 5000  # snacks > transport


def test_api_voice_note_rejects_missing_language(client, sample_audio: Path):
    with sample_audio.open("rb") as fh:
        resp = client.post(
            "/voice-note",
            data={"user_id": "3"},  # language omitted
            files={"file": ("note1.ogg", fh, "audio/ogg")},
        )
    assert resp.status_code == 422  # FastAPI missing-field validation
