"""
WhatsApp Cloud API client: receiving messages, downloading media, sending replies.

Configuration comes exclusively from environment variables — tokens must never
be hardcoded or committed:

  WHATSAPP_ACCESS_TOKEN      Permanent access token (Meta Business Settings >
                             System Users, or the temporary token from
                             WhatsApp > API Setup while testing).
  WHATSAPP_PHONE_NUMBER_ID   The phone-number id Veyra sends messages *from*
                             (WhatsApp > API Setup shows it; it is a numeric id,
                             not the display number).
  WHATSAPP_VERIFY_TOKEN      Random string you invent; entered in the Meta
                             developer dashboard when configuring the webhook,
                             and checked by GET /webhook.
  WHATSAPP_GRAPH_API_VERSION Optional Graph API version, default "v21.0".

Endpoints used (WhatsApp Cloud API / Graph API):
  - Send text message : POST https://graph.facebook.com/{version}/{phone_number_id}/messages
  - Resolve media URL : GET  https://graph.facebook.com/{version}/{media_id}
                        -> {"url": "<temporary CDN url>", ...}
  - Download media    : GET  <temporary CDN url> (Authorization header required)
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any, Optional

DEFAULT_GRAPH_API_VERSION = "v21.0"
_REQUEST_TIMEOUT = 30

# Message types that carry a downloadable media payload. Meta currently sends
# voice notes as type "audio"; "voice" is accepted defensively.
SUPPORTED_MEDIA_TYPES = ("audio", "voice")


class WhatsAppConfigError(RuntimeError):
    """A required WHATSAPP_* environment variable is missing or empty."""


def _require_env(name: str) -> str:
    value = (os.environ.get(name) or "").strip()
    if not value:
        raise WhatsAppConfigError(f"Missing required environment variable {name}")
    return value


def _graph_api_base() -> str:
    version = (os.environ.get("WHATSAPP_GRAPH_API_VERSION") or "").strip()
    return f"https://graph.facebook.com/{version or DEFAULT_GRAPH_API_VERSION}"


def _auth_headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {_require_env('WHATSAPP_ACCESS_TOKEN')}"}


def _request_json(
    method: str,
    url: str,
    payload: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    data: Optional[bytes] = None
    headers = _auth_headers()
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=_REQUEST_TIMEOUT) as resp:
            body = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"WhatsApp API HTTP {exc.code} for {method} {url}: {detail}"
        ) from exc
    return json.loads(body) if body else {}


def _request_bytes(method: str, url: str) -> bytes:
    request = urllib.request.Request(url, headers=_auth_headers(), method=method)
    try:
        with urllib.request.urlopen(request, timeout=_REQUEST_TIMEOUT) as resp:
            return resp.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"WhatsApp media download HTTP {exc.code} for {url}: {detail}"
        ) from exc


# ---------------------------------------------------------------------------
# Outgoing / media API
# ---------------------------------------------------------------------------

def send_message(to_phone_number: str, text: str) -> dict[str, Any]:
    """Send a plain text WhatsApp message to ``to_phone_number`` (digits only)."""
    phone_number_id = _require_env("WHATSAPP_PHONE_NUMBER_ID")
    url = f"{_graph_api_base()}/{phone_number_id}/messages"
    payload = {
        "messaging_product": "whatsapp",
        "recipient_type": "individual",
        "to": to_phone_number,
        "type": "text",
        "text": {"preview_url": False, "body": text},
    }
    return _request_json("POST", url, payload)


def get_media_url(media_id: str) -> str:
    """Step 1 of the two-step media download: resolve the temporary CDN URL."""
    info = _request_json("GET", f"{_graph_api_base()}/{media_id}")
    url = info.get("url")
    if not url:
        raise RuntimeError(f"WhatsApp media info response had no 'url': {info}")
    return url


def download_media(media_id: str) -> bytes:
    """Download a voice note / media blob: media id -> temporary URL -> bytes."""
    return _request_bytes("GET", get_media_url(media_id))


# ---------------------------------------------------------------------------
# Incoming webhook payload parsing
# ---------------------------------------------------------------------------

def parse_webhook_payload(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract incoming user messages from a Meta webhook notification.

    Returns one dict per message::

        {"phone": str, "name": str | None, "type": str,
         "text": str | None, "media_id": str | None}

    Delivery/read receipts (``statuses``) and non-message changes produce no
    items — Meta expects a 200 for those too, which the endpoint always sends.
    """
    items: list[dict[str, Any]] = []
    for entry in payload.get("entry") or []:
        for change in entry.get("changes") or []:
            value = change.get("value") or {}
            names = {
                c.get("wa_id"): (c.get("profile") or {}).get("name")
                for c in value.get("contacts") or []
            }
            for message in value.get("messages") or []:
                phone = message.get("from")
                if not phone:
                    continue
                msg_type = message.get("type")
                item: dict[str, Any] = {
                    "phone": phone,
                    "name": names.get(phone),
                    "type": msg_type,
                    "text": None,
                    "media_id": None,
                }
                if msg_type == "text":
                    item["text"] = (message.get("text") or {}).get("body") or ""
                elif msg_type in SUPPORTED_MEDIA_TYPES:
                    item["media_id"] = (message.get(msg_type) or {}).get("id")
                items.append(item)
    return items
