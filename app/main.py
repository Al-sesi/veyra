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

import shutil
import tempfile
from pathlib import Path
from typing import Optional

from io import BytesIO

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from app.db import DEFAULT_DB_PATH, init_db, lookup_user_by_phone
from app.ledger import build_ledger_path, generate_xlsx_ledger_bytes, get_summary, list_entries, list_open_debts
from app.pipeline import process_voice_note


# Make sure tables exist before the first request hits. FastAPI calls this on
# startup exactly once per process (so DB is initialised both under `uvicorn`
# and in the TestClient used by unit tests).
app = FastAPI(title="Veyra", version="0.1.0")

# The static demo site (site/, served by `python -m http.server 8123`) calls
# these endpoints from the browser; without CORS the browser blocks the calls.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
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
    ],
    allow_origin_regex=r"^https?://(127\.0\.0\.1|localhost)(:\d+)?$",
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _ensure_db() -> None:  # pragma: no cover - trivial side effect
    init_db(DEFAULT_DB_PATH)


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
