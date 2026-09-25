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
python -m pip install -r requirements.txt
```

`requirements.txt` pins a **CPU-only** torch wheel (small download).  If
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

## Planned (not yet built)

- WhatsApp voice-note ingestion (Twilio / Meta Graph API)
- Database persistence (entries stored with timestamps + speaker)
- FastAPI endpoints wiring `transcribe` + `parse_transcript` together
- Frontend UI: per-trader day-books, search, summaries, debt tracking
