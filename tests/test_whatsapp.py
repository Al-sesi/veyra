"""
Tests for the WhatsApp Cloud API messaging layer (app/whatsapp.py + /webhook).

Nothing here hits the network:
  - app/whatsapp.py unit tests mock its ``_request_json`` / ``_request_bytes``
    HTTP helpers.
  - Webhook tests run the real FastAPI app against a temp SQLite DB (same
    fixture pattern as tests/test_corrections_debts.py) and monkeypatch
    ``app.whatsapp.send_message`` / ``app.whatsapp.download_media`` and
    ``app.pipeline.transcribe``.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from unittest.mock import patch

import pytest

from app.db import get_or_create_user, init_db, query_rows
from app.whatsapp import (
    WhatsAppConfigError,
    download_media,
    parse_webhook_payload,
    send_message,
)

PHONE = "2348012345678"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def tmp_db(tmp_path: Path) -> Path:
    db = tmp_path / "whatsapp_test.db"
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
def client(_patch_db, monkeypatch, tmp_db: Path):
    import app.main as main_mod

    monkeypatch.setattr(main_mod, "DEFAULT_DB_PATH", tmp_db)
    importlib.reload(main_mod)

    from fastapi.testclient import TestClient

    with TestClient(main_mod.app) as tc:
        yield tc


@pytest.fixture()
def wa_env(monkeypatch):
    monkeypatch.setenv("WHATSAPP_ACCESS_TOKEN", "test-access-token")
    monkeypatch.setenv("WHATSAPP_PHONE_NUMBER_ID", "123456789")
    monkeypatch.setenv("WHATSAPP_VERIFY_TOKEN", "veyra-verify-123")


@pytest.fixture()
def sent(monkeypatch):
    """Capture outgoing WhatsApp replies as [(to, text), ...]."""
    calls: list[tuple[str, str]] = []

    def fake_send(to: str, text: str):
        calls.append((to, text))
        return {}

    monkeypatch.setattr("app.whatsapp.send_message", fake_send)
    return calls


def _seed_user(tmp_db: Path, phone: str = PHONE, language: str = "en") -> int:
    return get_or_create_user(tmp_db, None, phone_or_name=phone, language=language)


def _wa_payload(
    phone: str = PHONE,
    msg_type: str = "text",
    body: str = "",
    media_id: str | None = None,
):
    """Build a Meta WhatsApp Cloud API notification payload."""
    message = {
        "from": phone,
        "id": "wamid.HBgLNjIzNDgwMTIzNDU2OBUCABIYFDNBRjRB",
        "timestamp": "1700000000",
        "type": msg_type,
    }
    if msg_type == "text":
        message["text"] = {"body": body}
    else:
        message[msg_type] = {
            "id": media_id or "MEDIA_ID_1",
            "mime_type": "audio/ogg" if msg_type == "audio" else "image/jpeg",
        }
    return {
        "object": "whatsapp_business_account",
        "entry": [
            {
                "id": "WABA_ID",
                "changes": [
                    {
                        "field": "messages",
                        "value": {
                            "messaging_product": "whatsapp",
                            "metadata": {
                                "display_phone_number": "15550001122",
                                "phone_number_id": "123456789",
                            },
                            "contacts": [
                                {"profile": {"name": "Test Trader"}, "wa_id": phone}
                            ],
                            "messages": [message],
                        },
                    }
                ],
            }
        ],
    }


# ---------------------------------------------------------------------------
# app/whatsapp.py unit tests
# ---------------------------------------------------------------------------

def test_send_message_builds_correct_request(monkeypatch):
    captured: dict = {}

    def fake_json(method, url, payload=None):
        captured.update(method=method, url=url, payload=payload)
        return {"messages": [{"id": "wamid.out.1"}]}

    monkeypatch.setattr("app.whatsapp._request_json", fake_json)
    monkeypatch.setenv("WHATSAPP_PHONE_NUMBER_ID", "PHONE_ID_1")
    monkeypatch.setenv("WHATSAPP_ACCESS_TOKEN", "TOKEN_1")

    out = send_message(PHONE, "hello trader")

    assert out == {"messages": [{"id": "wamid.out.1"}]}
    assert captured["method"] == "POST"
    assert captured["url"] == (
        "https://graph.facebook.com/v21.0/PHONE_ID_1/messages"
    )
    body = captured["payload"]
    assert body["messaging_product"] == "whatsapp"
    assert body["recipient_type"] == "individual"
    assert body["to"] == PHONE
    assert body["type"] == "text"
    assert body["text"]["body"] == "hello trader"


def test_send_message_requires_phone_number_id(monkeypatch):
    monkeypatch.delenv("WHATSAPP_PHONE_NUMBER_ID", raising=False)
    with pytest.raises(WhatsAppConfigError):
        send_message(PHONE, "hi")


def test_send_message_requires_access_token(monkeypatch):
    monkeypatch.delenv("WHATSAPP_ACCESS_TOKEN", raising=False)
    monkeypatch.setenv("WHATSAPP_PHONE_NUMBER_ID", "PHONE_ID_1")
    with pytest.raises(WhatsAppConfigError):
        send_message(PHONE, "hi")


def test_download_media_is_two_step(monkeypatch):
    calls: list[tuple[str, str, str]] = []

    def fake_json(method, url, payload=None):
        calls.append(("json", method, url))
        return {"url": "https://cdn.fbsbx.com/whatsapp_bsp/attachment XYZ"}

    def fake_bytes(method, url):
        calls.append(("bytes", method, url))
        return b"fake-audio-bytes"

    monkeypatch.setattr("app.whatsapp._request_json", fake_json)
    monkeypatch.setattr("app.whatsapp._request_bytes", fake_bytes)

    data = download_media("MEDIA_ID_9")

    assert data == b"fake-audio-bytes"
    assert calls == [
        ("json", "GET", "https://graph.facebook.com/v21.0/MEDIA_ID_9"),
        ("bytes", "GET", "https://cdn.fbsbx.com/whatsapp_bsp/attachment XYZ"),
    ]


def test_download_media_raises_when_url_missing(monkeypatch):
    monkeypatch.setattr("app.whatsapp._request_json", lambda m, u, p=None: {})
    with pytest.raises(RuntimeError):
        download_media("MEDIA_X")


def test_parse_webhook_payload_text_message():
    items = parse_webhook_payload(_wa_payload(body="I sold rice 45k"))
    assert items == [
        {
            "phone": PHONE,
            "name": "Test Trader",
            "type": "text",
            "text": "I sold rice 45k",
            "media_id": None,
        }
    ]


def test_parse_webhook_payload_audio_message():
    items = parse_webhook_payload(_wa_payload(msg_type="audio", media_id="M1"))
    assert len(items) == 1
    assert items[0]["type"] == "audio"
    assert items[0]["media_id"] == "M1"
    assert items[0]["text"] is None


def test_parse_webhook_payload_ignores_status_receipts():
    payload = {
        "object": "whatsapp_business_account",
        "entry": [
            {
                "id": "WABA_ID",
                "changes": [
                    {
                        "field": "messages",
                        "value": {
                            "messaging_product": "whatsapp",
                            "statuses": [
                                {"id": "wamid.1", "status": "delivered"},
                                {"id": "wamid.1", "status": "read"},
                            ],
                        },
                    }
                ],
            }
        ],
    }
    assert parse_webhook_payload(payload) == []


# ---------------------------------------------------------------------------
# GET /webhook verification
# ---------------------------------------------------------------------------

def test_webhook_verification_success(client, wa_env):
    resp = client.get(
        "/webhook",
        params={
            "hub.mode": "subscribe",
            "hub.verify_token": "veyra-verify-123",
            "hub.challenge": "CHALLENGE_CODE_XYZ",
        },
    )
    assert resp.status_code == 200
    assert resp.text == "CHALLENGE_CODE_XYZ"
    assert resp.headers["content-type"].startswith("text/plain")


def test_webhook_verification_wrong_token_rejected(client, wa_env):
    resp = client.get(
        "/webhook",
        params={
            "hub.mode": "subscribe",
            "hub.verify_token": "not-the-token",
            "hub.challenge": "CHALLENGE_CODE_XYZ",
        },
    )
    assert resp.status_code == 403


def test_webhook_verification_missing_env_token_rejected(client, monkeypatch):
    monkeypatch.delenv("WHATSAPP_VERIFY_TOKEN", raising=False)
    resp = client.get(
        "/webhook",
        params={
            "hub.mode": "subscribe",
            "hub.verify_token": "veyra-verify-123",
            "hub.challenge": "X",
        },
    )
    assert resp.status_code == 403


# ---------------------------------------------------------------------------
# POST /webhook flows
# ---------------------------------------------------------------------------

def test_text_message_is_processed_and_replied(client, wa_env, sent, tmp_db):
    _seed_user(tmp_db, PHONE, "en")
    resp = client.post("/webhook", json=_wa_payload(body="I sold rice 45k"))
    assert resp.status_code == 200
    assert resp.json()["messages_received"] == 1
    assert len(sent) == 1
    to, text = sent[0]
    assert to == PHONE
    assert "Recorded:" in text
    assert "45,000" in text
    rows = query_rows(tmp_db, "SELECT * FROM entries;")
    assert len(rows) == 1
    assert rows[0]["amount"] == 45000
    assert rows[0]["item"] == "Rice"


def test_audio_message_flow_end_to_end(client, wa_env, sent, monkeypatch, tmp_db):
    _seed_user(tmp_db, PHONE, "en")
    monkeypatch.setattr(
        "app.whatsapp.download_media", lambda media_id: b"fake-ogg-bytes"
    )
    with patch("app.pipeline.transcribe", return_value="I sold beans 20k"):
        resp = client.post(
            "/webhook", json=_wa_payload(msg_type="audio", media_id="MEDIA_42")
        )
    assert resp.status_code == 200
    assert len(sent) == 1
    assert sent[0][0] == PHONE
    assert "Beans" in sent[0][1]
    assert "20,000" in sent[0][1]
    rows = query_rows(tmp_db, "SELECT * FROM entries;")
    assert len(rows) == 1
    assert rows[0]["item"] == "Beans"


def test_unknown_user_is_asked_for_language(client, wa_env, sent, tmp_db):
    resp = client.post("/webhook", json=_wa_payload(body="I sold rice 45k"))
    assert resp.status_code == 200
    assert len(sent) == 1
    reply = sent[0][1]
    assert "which language" in reply.lower()
    for language_name in ("yoruba", "hausa", "igbo", "english"):
        assert language_name in reply.lower()
    # Nothing was persisted: no user row, no entries.
    assert query_rows(tmp_db, "SELECT * FROM users;") == []
    assert query_rows(tmp_db, "SELECT * FROM entries;") == []


@pytest.mark.parametrize(
    ("reply_text", "expected_code", "expected_display"),
    [
        ("Yoruba", "yo", "Yorùbá"),
        ("yorùbá", "yo", "Yorùbá"),
        ("Hausa", "ha", "Hausa"),
        ("Igbo", "ig", "Igbo"),
        ("English", "en", "English"),
    ],
)
def test_language_choice_creates_user_and_continues(
    client, wa_env, sent, tmp_db, reply_text, expected_code, expected_display
):
    resp = client.post("/webhook", json=_wa_payload(body=reply_text))
    assert resp.status_code == 200
    assert expected_display in sent[0][1]

    users = query_rows(tmp_db, "SELECT * FROM users;")
    assert len(users) == 1
    assert users[0]["phone_or_name"] == PHONE
    assert users[0]["language"] == expected_code

    # Follow-up message now goes straight into the pipeline in that language.
    resp = client.post("/webhook", json=_wa_payload(body="I sold rice 10k"))
    assert resp.status_code == 200
    assert "Recorded:" in sent[-1][1]
    assert "10,000" in sent[-1][1]


def test_download_failure_sends_friendly_reply(client, wa_env, sent, monkeypatch, tmp_db):
    _seed_user(tmp_db, PHONE, "en")

    def boom(media_id):
        raise RuntimeError("network down")

    monkeypatch.setattr("app.whatsapp.download_media", boom)
    resp = client.post("/webhook", json=_wa_payload(msg_type="audio"))
    assert resp.status_code == 200
    assert sent == [(PHONE, "Sorry, I couldn't process that, please try again.")]


def test_unsupported_message_type_gets_friendly_reply(client, wa_env, sent, tmp_db):
    _seed_user(tmp_db, PHONE, "en")
    resp = client.post("/webhook", json=_wa_payload(msg_type="image"))
    assert resp.status_code == 200
    assert len(sent) == 1
    assert "voice notes and text messages" in sent[0][1]


def test_status_receipt_is_acked_without_reply(client, wa_env, sent):
    payload = {
        "object": "whatsapp_business_account",
        "entry": [
            {
                "id": "WABA_ID",
                "changes": [
                    {
                        "field": "messages",
                        "value": {
                            "messaging_product": "whatsapp",
                            "statuses": [{"id": "wamid.1", "status": "delivered"}],
                        },
                    }
                ],
            }
        ],
    }
    resp = client.post("/webhook", json=payload)
    assert resp.status_code == 200
    assert resp.json()["messages_received"] == 0
    assert sent == []
