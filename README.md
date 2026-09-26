---
title: Veyra API
emoji: 🇳🇬
colorFrom: green
colorTo: teal
sdk: docker
app_port: 7860
license: mit
tags:
  - fastapi
  - uvicorn
  - whisper
  - nigerian-languages
  - asr
  - bookkeeping
---

# Veyra

A WhatsApp voice-note bookkeeper for Nigerian market traders.

## Features

### 1. Transcript parser (`app/parser.py`)

Converts trader voice-note transcripts (English, Pidgin, or mixed) into structured bookkeeping entries.

```python
from app.parser import parse_transcript

entries = parse_transcript("I sold 5 bags of rice for 45k, paid transport 3k")
# Returns:
# [
#   {"item": "Rice", "quantity": 5.0, "amount": 45000, "type": "sale"},
#   {"item": "Transport", "quantity": None, "amount": 3000, "type": "expense"},
# ]
```

Entry fields:
- `item`: short string description of the item
- `quantity`: numeric quantity, or `None` if the trader didn't state one
- `amount`: integer naira amount
- `type`: `"sale"` (money in) or `"expense"` (money out)

Supported amount formats:
- Plain numbers: `45000`, `500`
- `k` shorthand: `45k` → 45000, `2.5k` → 2500
- Word numbers: `five thousand naira`, `ten thousand`
- Slang hyphenated: `two-fifty` → 250

### 2. Speech-to-text / ASR (`app/asr.py`)

Transcribes audio (WhatsApp `.ogg` Opus voice notes, or any format ffmpeg
can decode) into text using the N-ATLaS Nigerian-language Whisper Small
models on Hugging Face.

```python
from app.asr import transcribe

text = transcribe("voice_note.ogg", language="ha")      # Hausa
text = transcribe("voice_note.ogg", language="ig")      # Igbo
text = transcribe("voice_note.ogg", language="yo")      # Yoruba
text = transcribe("voice_note.ogg", language="en")      # Nigerian English (Accented)
text = transcribe("voice_note.ogg", language="pcm")     # Pidgin (experimental, untested)
text = transcribe("voice_note.ogg", language="Naija")   # aliases also work
```

Supported languages (add more by adding one entry to `LANGUAGE_MODEL_CONFIG`
in [asr.py](app/asr.py)):

| Code    | Name                               | Model repo                       | Status                 |
|---------|------------------------------------|----------------------------------|------------------------|
| `ha`    | Hausa                              | `NCAIR1/Hausa-ASR`               | supported              |
| `ig`    | Igbo                               | `NCAIR1/Igbo-ASR`                | supported              |
| `yo`    | Yoruba                             | `NCAIR1/Yoruba-ASR`              | supported              |
| `en`    | Nigerian English (Accented)        | `NCAIR1/NigerianAccentedEnglish` | supported              |
| `pcm`   | Pidgin (Nigerian Accented English) | `NCAIR1/NigerianAccentedEnglish` | experimental, untested |

All 5 entries above point to official **N-ATLaS** / NCAIR1 / Awarri checkpoints on
Hugging Face — there is intentionally no generic `openai/whisper-small` fallback,
so any English or Pidgin voice-note transcription still counts as N-ATLaS
integration evidence.

Notes on the `pcm` + `en` rows: NCAIR1 don't yet publish standalone
"Nigerian Pidgin" or "Nigerian English" named repos.  Their
**NigerianAccentedEnglish** Whisper-Small fine-tune is trained on speakers
across Nigeria's 6 geopolitical zones, with Nigerian English conventions
and Pidgin phrases in the training set (per its model card), so it's the
best official N-ATLaS match.  `pcm` therefore **routes to the same model
as `en`** and is marked **experimental, untested**:
`app.asr.language_notice("pcm")` returns the warning both CLI scripts
print before running.  If Awarri/NCAIR later releases a dedicated
`NCAIR1/Pidgin-ASR` or `NCAIR1/English-ASR`, swap only the `model` string
for the relevant code — everything else stays the same.

Accepted `language=` aliases (case-insensitive):
- `ha`/`Hausa`, `ig`/`Igbo`, `yo`/`Yoruba`
- `en`/`English`, `Nigerian English`, `en-ng`
- `pcm`, `Pidgin`, `Naija`, `Nigerian Pidgin` (experimental, untested)

The function follows the **exact** "Basic Usage" snippet on each model
card: load via `transformers.pipeline("automatic-speech-recognition",
model=...)`, load audio at 16 kHz with `librosa.load(..., sr=16000)`,
then call `result = asr(audio)` and return `result["text"]`.

Before inference, any input audio is **always** converted to 16 kHz mono
PCM WAV with `ffmpeg` (this is what WhatsApp's `.ogg` voice notes need).
Each model is loaded once and cached in-process; subsequent calls reuse
the same pipeline.

#### Quick CLI try-it

```
python scripts/try_asr.py path/to/voice_note.ogg -l ha
python scripts/try_asr.py path/to/voice_note.opus -l yo
python scripts/try_asr.py path/to/voice_note.opus -l pcm
```

First run will download the Whisper Small weight file (~500 MB / 244M
params per language).

## Setup

### 1. Install ffmpeg (required for audio conversion)

WhatsApp voice notes are `.ogg` (Opus). The ASR step re-encodes them to
16 kHz mono WAV before inference, so ffmpeg **must** be on your PATH.

| OS            | How to install                                            |
|---------------|-----------------------------------------------------------|
| **Windows**   | `winget install Gyan.FFmpeg`  (or download from [ffmpeg.org](https://ffmpeg.org/download.html) and add `bin/` to PATH) |
| **macOS**     | `brew install ffmpeg`                                     |
| **Linux**     | `sudo apt update && sudo apt install ffmpeg`              |

Verify with `ffmpeg -version` before continuing.

### 2. Install Python deps

```
python -m pip install -r requirements-full.txt
```

`requirements-full.txt` pins a **CPU-only** torch wheel (small download).  If
you have an NVIDIA GPU and want CUDA acceleration, remove the
`--index-url https://download.pytorch.org/whl/cpu` line before installing.

### 3. (Optional) Hugging Face auth

The NCAIR1 models are public, so no token is required for download.  Some
cards (e.g. `NCAIR1/Yoruba-ASR`) ask you to accept conditions before the
files unlock — if a download is refused, run `huggingface-cli login` and
accept the terms on the model page.  If you hit rate limits, set
`HF_TOKEN` in your shell.

## Running tests

```
pytest
```

- `tests/test_parser.py` — 15 tests against the parser (no network, no model).
- `tests/test_asr.py` — mocked tests for the ASR wrapper (no network, no
  model download, no ffmpeg required). The pipeline, audio conversion,
  and cache behaviour are all exercised via monkeypatch.

## Running locally

### Backend (FastAPI, port 8000)

```
python -m uvicorn app.main:app --port 8000
```

Endpoints:
- `GET  /` — health check
- `POST /voice-note` — upload audio + language + optional user_id, get transcript + entries + reply
- `GET  /ledger/{user_id}?limit=50` — recent entries
- `GET  /ledger/{phone_number}.xlsx` — download full Excel ledger for a phone number
- `GET  /summary/{user_id}?days=7` — totals, profit, top items + debt summary
- `GET  /debts/{user_id}` — open (unpaid) debts
- `GET  /webhook` / `POST /webhook` — WhatsApp Cloud API verification + incoming messages (see "WhatsApp Setup")

### Frontend (static demo site, port 8123)

```
cd site && python -m http.server 8123
```

Then open `http://localhost:8123/demo/` in the browser. The demo page auto-detects
the API origin (same host/port or localhost:8000 fallback) or accepts
`?api=https://your-backend.onrender.com` to point at a deployed instance.

## Deploying to Render

Veyra ships with `render.yaml` (Blueprint / Infrastructure-as-Code) that
provisions two services:

| Service       | Type     | Purpose                                         |
|---------------|----------|-------------------------------------------------|
| `veyra-api`   | Web      | FastAPI backend (ASR, intents, ledger, SQLite)  |
| `veyra-site`  | Static   | Marketing + live demo pages from `site/`        |

### Prerequisites

1. A Render account (render.com) — paid **Starter** plan or higher recommended
   for the backend because the **Free** plan has no persistent disk (your
   SQLite data and cached ASR models are wiped on every deploy/redeploy, and
   cold starts re-download ~500 MB of Hugging Face weights).
2. This repository pushed to GitHub/GitLab and connected to Render.

### Option A — Deploy from Blueprint (recommended)

1. In the Render dashboard go to **Blueprints → New Blueprint Instance**.
2. Select your connected repository (the one containing `render.yaml`).
3. On the "Environment Groups" step you will be prompted for
   `VEYRA_CORS_ORIGINS` (marked `sync: false` so Render asks for the value).
   Set it to the HTTPS URL of the `veyra-site` static service once you know
   it, or to a comma-separated list of origins that may call the API, e.g.:
   ```
   https://veyra-site.onrender.com,https://example.com
   ```
   You can change this later under **Environment** for the `veyra-api`
   service and then redeploy.
4. Click **Apply**. Render builds both services and assigns `.onrender.com`
   hostnames.

### Option B — Manual service setup

If you don't want to use Blueprints, create the two services manually from
the Render dashboard:

**Backend — Web Service (Python):**

- **Branch:** `main`
- **Root Directory:** (repo root)
- **Runtime:** Python 3.11
- **Build Command:** `chmod +x render-build.sh && ./render-build.sh`
- **Start Command:** `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
- **Plan:** Starter (minimum for persistent disk)
- **Advanced → Health Check Path:** `/`
- **Advanced → Auto-Deploy:** Yes
- **Environment Variables:**
  - `VEYRA_DB_PATH=/var/data/ledger.db`
  - `HF_HOME=/var/data/hf-cache`
  - `TRANSFORMERS_CACHE=/var/data/hf-cache`
  - `VEYRA_CORS_ORIGINS=https://<your-static-site>.onrender.com`
  - `PYTHON_VERSION=3.11.9`
- **Disks:** Add disk named `veyra-data`, mount path `/var/data`, size 10 GB.

**Frontend — Static Site:**

- **Branch:** `main`
- **Root Directory:** (repo root)
- **Build Command:** *(leave empty)*
- **Publish Directory:** `./site`

### Using the deployed demo

Once both services are live, open:

```
https://<veyra-site>.onrender.com/demo/?api=https://<veyra-api>.onrender.com
```

The `?api=` query parameter tells the demo `demo.js` client which backend to
call (the static site has no server-side rendering, so auto-detection of the
origin would be wrong if you deployed site + api separately).

### First cold start note

The **first** request to `/voice-note` in each language downloads the
relevant N-ATLaS Whisper Small checkpoint from Hugging Face (~500 MB per
unique model). The `HF_HOME` / `TRANSFORMERS_CACHE` env vars point at the
persistent disk, so subsequent cold starts after a restart reuse the
already-downloaded files.

### Plan limitations & trade-offs

| Render plan      | Disk       | Typical cold start | Data survives deploy? |
|------------------|------------|--------------------|-----------------------|
| **Free (Web)**   | none       | 30–90 s (models redownload every time) | **No** — ephemeral FS only |
| **Starter ($7)** | 10 GB disk | 10–20 s (cached models) | **Yes** — via /var/data |
| **Pro ($20+)**   | 10 GB+ disk| 5–15 s             | **Yes**               |

For the pilot/demo phase a **Starter** web service + free Static Site is the
minimum sensible configuration.

## WhatsApp Setup

Veyra receives and answers WhatsApp messages through the **WhatsApp Cloud API**
(Meta). Two endpoints in `app/main.py` implement the integration:

- `GET /webhook` — Meta calls this once to verify the webhook (checks
  `hub.verify_token`, echoes `hub.challenge`).
- `POST /webhook` — receives incoming messages. Voice notes are downloaded
  and run through `process_voice_note()`; text messages run through the same
  pipeline with ASR skipped. Replies are sent back with the Send Message API.

### 1. Environment variables

| Variable                     | What it is |
|------------------------------|------------|
| `WHATSAPP_ACCESS_TOKEN`      | Permanent token (Meta Business Settings → System Users → token with `whatsapp_business_messaging` permission), or the temporary token from WhatsApp → API Setup while testing. **Never commit it.** |
| `WHATSAPP_PHONE_NUMBER_ID`   | Numeric id of the phone number Veyra sends *from* — shown at the top of WhatsApp → API Setup. This is **not** the display phone number. |
| `WHATSAPP_VERIFY_TOKEN`      | Any random string you invent (e.g. `openssl rand -hex 16`). Must match what you type into the Meta dashboard. |
| `WHATSAPP_BUSINESS_ACCOUNT_ID` | Your WABA id. The app itself doesn't call it, but Meta's setup flow asks for it when subscribing to webhooks via the API. |
| `WHATSAPP_GRAPH_API_VERSION` | Optional. Defaults to `v21.0`; bump when Meta deprecates a version. |

Set these in the Render dashboard under the `veyra-api` service →
**Environment** (same place as `VEYRA_DB_PATH`), then redeploy. For local
testing, put them in your shell or a gitignored `.env` loader.

### 2. Configure the webhook in Meta

1. In the [Meta developer dashboard](https://developers.facebook.com/apps),
   open your app → **WhatsApp** → **Configuration**.
2. Under **Webhook**, click **Edit** and enter:
   - **Callback URL:** `https://<your-veyra-api-host>/webhook`
     (e.g. `https://veyra-api.onrender.com/webhook`)
   - **Verify token:** the same value as `WHATSAPP_VERIFY_TOKEN`.
3. Meta immediately calls `GET /webhook` to verify; a correct verify token
   returns the challenge and the dashboard saves the subscription.
4. Under **Webhook fields**, make sure **messages** is subscribed.

### 3. First contact flow

Traders are identified by their WhatsApp number. On a trader's **first**
message Veyra asks which language to use (Yoruba, Hausa, Igbo, or English);
the choice is stored on the user's profile (`users.language`) and every later
voice note is transcribed with the matching N-ATLaS model. Anything the app
can't handle (images, download/ASR failures) gets a friendly WhatsApp reply,
and the real error is logged server-side.

## Planned (not yet built)

- Rich WhatsApp replies (buttons, ledger-link templates) beyond plain text
