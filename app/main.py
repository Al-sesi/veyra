"""
FastAPI application exposing four endpoints:

  - POST /voice-note     Upload an audio file and language/user_id; run
                          ASR -> intent routing -> ledger persistence, and
                          return the transcript + saved entries + 7-day
                          summary + reply_text (what to say back).
  - GET  /ledger/{user_id}?limit=50
                         Recent entries for a user (newest first).
  - GET  /summary/{user_id}?days=7
                         Aggregated totals + top items over a rolling window,
                         plus open-debt totals.
  - GET  /debts/{user_id}
                         Open (unpaid) debts for a user.

Uses sqlite3 via app.db/app.ledger/app.pipeline. No third-party DB libraries.
"""

from __future__ import annotations

import logging
import os
import secrets
import shutil
import tempfile
from pathlib import Path
from typing import Any, Optional

from io import BytesIO

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse

from app.db import DEFAULT_DB_PATH, get_or_create_user, init_db, lookup_user_by_phone
from app.ledger import build_ledger_path, generate_xlsx_ledger_bytes, get_summary, list_entries, list_open_debts

logger = logging.getLogger("veyra.webhook")


def _parse_extra_origins() -> list[str]:
    raw = os.environ.get("VEYRA_CORS_ORIGINS", "")
    if not raw:
        return []
    return [o.strip() for o in raw.split(",") if o.strip()]


HF_HOME = os.environ.get("HF_HOME")
if HF_HOME:
    os.environ["HF_HOME"] = HF_HOME
TRANSFORMERS_CACHE = os.environ.get("TRANSFORMERS_CACHE")
if TRANSFORMERS_CACHE:
    os.environ["TRANSFORMERS_CACHE"] = TRANSFORMERS_CACHE


app = FastAPI(title="Veyra", version="0.1.0")

_local_origins = [
    "http://127.0.0.1:8123",
    "http://localhost:8123",
    "http://127.0.0.1:8000",
    "http://localhost:8000",
    "http://127.0.0.1:3000",
    "http://localhost:3000",
    "http://127.0.0.1:5500",
    "http://localhost:5500",
    "http://127.0.0.1:5173",
    "http://localhost:5173",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_local_origins + _parse_extra_origins(),
    allow_origin_regex=r"^https?://(127\.0\.0\.1|localhost)(:\d+)?$",
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _ensure_db() -> None:  # pragma: no cover - trivial side effect
    try:
        DEFAULT_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    for env_key in ("HF_HOME", "TRANSFORMERS_CACHE"):
        env_val = os.environ.get(env_key)
        if env_val:
            try:
                Path(env_val).mkdir(parents=True, exist_ok=True)
            except Exception:
                pass
    init_db(DEFAULT_DB_PATH)


@app.get("/")
def root():
    return {
        "service": "Veyra",
        "version": "0.1.0",
        "status": "ok",
        "db_path": str(DEFAULT_DB_PATH),
        "endpoints": [
            "POST /voice-note",
            "GET  /ledger/{user_id}",
            "GET  /ledger/{phone_number}.xlsx",
            "GET  /summary/{user_id}",
            "GET  /debts/{user_id}",
            "GET  /webhook",
            "POST /webhook",
        ],
    }


# ---------------------------------------------------------------------------
# Request body helpers
# ---------------------------------------------------------------------------

def _safe_int_form(value: Optional[str], *, field: str) -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"{field} must be integer") from exc


def _required_form(value: Optional[str], *, field: str) -> str:
    if value is None or value == "":
        raise HTTPException(status_code=400, detail=f"{field} is required")
    return value


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/voice-note")
async def voice_note(
    file: UploadFile = File(...),
    language: str = Form(...),
    user_id: Optional[str] = Form(None),
):
    """Upload an audio file, get transcript + entries + 7-day summary."""
    try:
        from app.pipeline import process_voice_note
    except Exception as exc:  # pragma: no cover - environment dependent
        raise HTTPException(
            status_code=503,
            detail=f"Voice-note ASR pipeline is not available in this deployment ({exc.__class__.__name__}: {exc}). Use the text-only endpoints or deploy with ASR dependencies.",
        ) from exc
    lang = _required_form(language, field="language")
    uid = _safe_int_form(user_id, field="user_id")

    # Stream the upload into a temp file so `process_voice_note` (which expects
    # a filesystem path, needed for ffmpeg) can use it.
    suffix = Path(file.filename or "").suffix or ".bin"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tf:
        try:
            shutil.copyfileobj(file.file, tf)
            tmp_path = Path(tf.name)
        finally:
            file.file.close()

    try:
        result = process_voice_note(
            audio_path=str(tmp_path),
            language=lang,
            user_id=uid,
        )
        return JSONResponse(
            {
                "user_id": result["user_id"],
                "transcript": result["transcript"],
                "entries": result["entries"],
                "summary": result["summary"],
                "reply_text": result["reply_text"],
                "language_notice": result.get("language_notice"),
            }
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ValueError as exc:
        # Unknown language code, etc.
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


@app.get("/ledger/{phone_number}.xlsx")
def ledger_xlsx(phone_number: str):
    """Persistent ledger link: regenerate a fresh .xlsx every call.

    Privacy: only the entries for the exact matched phone number are ever
    returned; unknown numbers get a 404 so links can't be enumerated to
    discover which traders have accounts.
    """
    user = lookup_user_by_phone(DEFAULT_DB_PATH, phone_number)
    if user is None:
        raise HTTPException(status_code=404, detail="No ledger for this phone number")
    payload = generate_xlsx_ledger_bytes(
        int(user["id"]),
        db_path=DEFAULT_DB_PATH,
        phone_label=user["phone_or_name"],
    )
    filename = f"{user['phone_or_name']}.xlsx"
    headers = {
        "Content-Disposition": f"attachment; filename=\"{filename}\"",
        "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
        "Pragma": "no-cache",
        "Expires": "0",
    }
    return StreamingResponse(
        BytesIO(payload),
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers=headers,
    )


@app.get("/ledger/{user_id}")
def ledger(user_id: int, limit: int = 50):
    if limit <= 0 or limit > 500:
        raise HTTPException(status_code=400, detail="limit must be between 1 and 500")
    return {"user_id": user_id, "entries": list_entries(user_id=user_id, limit=limit)}


@app.get("/summary/{user_id}")
def summary(user_id: int, days: int = 7):
    if days <= 0 or days > 365 * 10:
        raise HTTPException(status_code=400, detail="days out of range")
    payload = get_summary(user_id=user_id, days=days)
    return {"user_id": user_id, "days": days, **payload}


@app.get("/debts/{user_id}")
def debts(user_id: int):
    """Open (unpaid) debts for a user, newest first."""
    return {"user_id": user_id, "debts": list_open_debts(user_id=user_id)}


# ---------------------------------------------------------------------------
# WhatsApp Cloud API webhook
# ---------------------------------------------------------------------------
#
# GET /webhook  — Meta's one-time verification (hub.mode/hub.verify_token/
#                 hub.challenge query params).
# POST /webhook — incoming WhatsApp messages (voice notes + text), routed
#                 through the same pipeline as POST /voice-note.

WHATSAPP_FALLBACK_REPLY = "Sorry, I couldn't process that, please try again."
WHATSAPP_UNSUPPORTED_REPLY = (
    "Sorry, I can only work with voice notes and text messages for now. "
    "Send one of those and I'll update your books."
)
WHATSAPP_LANGUAGE_ASK_REPLY = (
    "Welcome to Veyra, your voice-note bookkeeper!\n"
    "Which language would you like to use? Reply with one of:\n"
    "Yoruba, Hausa, Igbo, or English."
)

_LANGUAGE_DISPLAY_NAMES = {
    "en": "English",
    "yo": "Yorùbá",
    "ha": "Hausa",
    "ig": "Igbo",
    "pcm": "Nigerian Pidgin",
}


def _detect_language_choice(text: Optional[str]) -> Optional[str]:
    """Map a language-picking reply ("Yoruba", "yorùbá", "English please")
    to an ASR language code, or None if it doesn't name a supported language."""
    if not text:
        return None
    words = text.strip().lower().split()
    if not words or len(words) > 6:
        return None
    from app.asr import _resolve_language  # noqa: PLC0415 (single source of language codes)
    from app.parser import strip_diacritics  # noqa: PLC0415

    folded = strip_diacritics(" ".join(words))
    for candidate in (folded, *folded.split()):
        try:
            return _resolve_language(candidate)
        except ValueError:
            continue
    return None


def _handle_whatsapp_message(item: dict[str, Any]) -> None:
    """Process one parsed WhatsApp message and reply via the Cloud API."""
    from app import whatsapp  # noqa: PLC0415 (kept import-time cheap, patchable in tests)
    from app.pipeline import process_voice_note  # noqa: PLC0415

    phone = item["phone"]
    try:
        user = lookup_user_by_phone(DEFAULT_DB_PATH, phone)

        # First contact: no user row yet -> ask which language to use, unless
        # this very message is the language choice (then store it and confirm).
        if user is None:
            language = _detect_language_choice(item.get("text"))
            if language is None:
                whatsapp.send_message(phone, WHATSAPP_LANGUAGE_ASK_REPLY)
                return
            get_or_create_user(
                DEFAULT_DB_PATH, None,
                phone_or_name=phone,
                language=language,
            )
            display = _LANGUAGE_DISPLAY_NAMES.get(language, language)
            whatsapp.send_message(
                phone,
                f"Thank you! Veyra will use {display} for your account. "
                "Send a voice note or a text to log a sale or expense — "
                'for example "I sold rice 5k".',
            )
            return

        user_id = int(user["id"])
        language = user.get("language") or "en"

        if item.get("type") in whatsapp.SUPPORTED_MEDIA_TYPES:
            audio_bytes = whatsapp.download_media(item["media_id"])
            with tempfile.NamedTemporaryFile(delete=False, suffix=".ogg") as tf:
                tf.write(audio_bytes)
                tmp_path = Path(tf.name)
            try:
                result = process_voice_note(
                    str(tmp_path), language=language, user_id=user_id
                )
            finally:
                tmp_path.unlink(missing_ok=True)
        elif item.get("type") == "text":
            result = process_voice_note(
                "", language=language, user_id=user_id,
                transcript=item.get("text") or "",
            )
        else:
            whatsapp.send_message(phone, WHATSAPP_UNSUPPORTED_REPLY)
            return

        whatsapp.send_message(phone, result["reply_text"])
    except Exception:
        logger.exception("Failed to process WhatsApp message from %s", phone)
        try:
            whatsapp.send_message(phone, WHATSAPP_FALLBACK_REPLY)
        except Exception:
            logger.exception("Failed to send fallback WhatsApp reply to %s", phone)


@app.get("/webhook")
def webhook_verify(request: Request):
    """Meta webhook verification: echo hub.challenge when the verify token matches."""
    params = request.query_params
    expected = (os.environ.get("WHATSAPP_VERIFY_TOKEN") or "").strip()
    token = params.get("hub.verify_token") or ""
    if (
        params.get("hub.mode") == "subscribe"
        and expected
        and secrets.compare_digest(token, expected)
    ):
        return PlainTextResponse(params.get("hub.challenge") or "")
    return PlainTextResponse("Webhook verification failed", status_code=403)


@app.post("/webhook")
def webhook_receive(payload: dict[str, Any]):
    """Receive WhatsApp messages; always ack 200 so Meta doesn't retry."""
    from app import whatsapp  # noqa: PLC0415

    items = whatsapp.parse_webhook_payload(payload)
    for item in items:
        _handle_whatsapp_message(item)
    return {"status": "ok", "messages_received": len(items)}
