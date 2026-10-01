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

from app.db import DEFAULT_DB_PATH, USE_POSTGRES, DATABASE_URL, get_or_create_user, init_db, lookup_user_by_phone
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
    # Initialize database - init_db handles both SQLite and PostgreSQL based on DATABASE_URL
    init_db()

    # Log which ffmpeg is being used for audio conversion
    try:
        from app.asr import _get_ffmpeg_path
        ffmpeg_path = _get_ffmpeg_path()
        if ffmpeg_path:
            logger.info(f"Using ffmpeg: {ffmpeg_path}")
        else:
            logger.warning("No ffmpeg available - audio processing will fail")
    except Exception as e:
        logger.warning(f"Could not check ffmpeg availability: {e}")

    # Skip automatic model loading during startup to prevent 502 errors on free tier
    # Models will load on-demand when users make requests
    # Call /warmup endpoint manually if you want to preload models
    logger.info("Skipping automatic model loading to prevent startup timeout")


@app.get("/")
@app.get("/health")
def root():
    """Lightweight health check endpoint - returns status without loading models or heavy processing."""
    return {
        "service": "Veyra",
        "version": "0.1.0",
        "status": "ok",
        "db_type": "PostgreSQL" if USE_POSTGRES else "SQLite",
        "db_path": str(DEFAULT_DB_PATH) if not USE_POSTGRES else DATABASE_URL[:20] + "...",  # Partial URL for security
        "endpoints": [
            "GET  / (health check)",
            "GET  /health (health check)",
            "POST /voice-note",
            "POST /text-message",
            "GET  /ledger/{user_id}",
            "GET  /ledger/{phone_number}.xlsx",
            "GET  /summary/{user_id}",
            "GET  /debts/{user_id}",
            "GET  /webhook",
            "POST /webhook",
        ],
    }


@app.get("/health/deep")
def deep_health():
    """Deep health check that tests model loading - for monitoring and warmup."""
    try:
        from app.asr import get_model_status
        model_status = get_model_status()
        return {
            "service": "Veyra",
            "status": "ok",
            **model_status,
        }
    except Exception as e:
        return {
            "service": "Veyra",
            "status": "degraded",
            "error": str(e),
        }


@app.get("/warmup")
def warmup():
    """Warmup endpoint to preload English model - call this after deployment to avoid cold starts."""
    try:
        from app.asr import load_essential_models, get_model_status

        # Load English model with more retries for first-time loading
        load_status = load_essential_models(max_retries=3, retry_delay=2.0)
        model_status = get_model_status()

        return {
            "service": "Veyra",
            "status": "warmed_up",
            "load_status": load_status,
            **model_status,
        }
    except Exception as e:
        return {
            "service": "Veyra",
            "status": "warmup_failed",
            "error": str(e),
        }


@app.get("/model-status")
def model_status():
    """Check if AI models are loaded without triggering downloads."""
    try:
        from app.asr import get_model_status
        return {
            "service": "Veyra",
            "status": "ok",
            **get_model_status(),
        }
    except Exception as e:
        return {
            "service": "Veyra",
            "status": "error",
            "error": str(e),
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
    file: Optional[UploadFile] = File(None),
    transcript: Optional[str] = Form(None),
    language: Optional[str] = Form(None),
    user_id: Optional[str] = Form(None),
):
    """Upload an audio file or transcript, get transcript + entries + 7-day summary.
    
    Language is optional - if not provided, it will be auto-detected from the transcript
    (for text input) or from the transcribed audio (for voice input).
    """
    try:
        from app.pipeline import process_voice_note
        from app.asr import detect_language_from_text  # noqa: PLC0415
    except Exception as exc:  # pragma: no cover - environment dependent
        raise HTTPException(
            status_code=503,
            detail=f"Voice-note ASR pipeline is not available in this deployment ({exc.__class__.__name__}: {exc}). Use the text-only endpoints or deploy with ASR dependencies.",
        ) from exc
    
    uid = _safe_int_form(user_id, field="user_id")
    
    # If transcript is provided, skip audio processing
    if transcript:
        # Auto-detect language if not provided
        if language is None:
            lang = detect_language_from_text(transcript)
        else:
            lang = language
        
        try:
            result = process_voice_note(
                audio_path="",
                language=lang,
                user_id=uid,
                transcript=transcript,
            )
            return JSONResponse(
                {
                    "user_id": result["user_id"],
                    "transcript": result["transcript"],
                    "entries": result["entries"],
                    "summary": result["summary"],
                    "reply_text": result["reply_text"],
                    "language_notice": result.get("language_notice"),
                    "detected_language": lang,
                }
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    # Otherwise process audio file
    if not file:
        raise HTTPException(status_code=400, detail="Either file or transcript is required")

    # For audio, use provided language or default to English
    # The ASR will transcribe in that language, and we can re-detect after if needed
    lang = language if language else "en"

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
                "detected_language": lang,
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


@app.post("/text-message")
async def text_message(
    text: str = Form(...),
    language: Optional[str] = Form(None),
    user_id: Optional[str] = Form(None),
):
    """Submit a text message directly (no audio), get entries + 7-day summary.
    
    This is a dedicated endpoint for text-only input, supporting:
    - English, Nigerian Pidgin, Yoruba, Hausa, Igbo
    
    Language is optional - if not provided, it will be auto-detected from the text.
    
    Example requests:
    - English: text="I bought rice for 5000", language="en" (or omit language)
    - Pidgin: text="I buy rice 5k", language="pcm" (or omit language)
    - Yoruba: text="Mo ra iresi fun 5000", language="yo" (or omit language)
    - Hausa: text="Na saya shinkafa 5k", language="ha" (or omit language)
    - Igbo: text="M zụrọ osikapa 5k", language="ig" (or omit language)
    
    The text goes through the same extraction and transaction pipeline as voice input.
    """
    try:
        from app.pipeline import process_voice_note
        from app.asr import detect_language_from_text  # noqa: PLC0415
    except Exception as exc:  # pragma: no cover - environment dependent
        raise HTTPException(
            status_code=503,
            detail=f"Pipeline is not available in this deployment ({exc.__class__.__name__}: {exc}).",
        ) from exc
    
    uid = _safe_int_form(user_id, field="user_id")
    
    # Auto-detect language if not provided
    if language is None:
        lang = detect_language_from_text(text)
    else:
        lang = language
    
    try:
        result = process_voice_note(
            audio_path="",
            language=lang,
            user_id=uid,
            transcript=text,
        )
        return JSONResponse(
            {
                "user_id": result["user_id"],
                "transcript": result["transcript"],
                "entries": result["entries"],
                "summary": result["summary"],
                "reply_text": result["reply_text"],
                "language_notice": result.get("language_notice"),
                "detected_language": lang,
            }
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


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
    from app.asr import detect_language_from_text  # noqa: PLC0415

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
                "You can switch languages anytime — just send messages in Yoruba, "
                "Hausa, Igbo, Pidgin, or English and I'll understand. "
                "Send a voice note or a text to log a sale or expense — "
                'for example "I sold rice 5k".',
            )
            return

        user_id = int(user["id"])
        
        # Auto-detect language from the message text/transcript
        # For audio, we'll detect after transcription, but for text messages we can detect now
        if item.get("type") == "text":
            message_text = item.get("text") or ""
            language = detect_language_from_text(message_text)
        else:
            # For audio, use user's stored language as initial hint
            # The ASR model will use this, but we'll re-detect after transcription
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
