# N-ATLaS Integration Notes

## ASR Backend

Veyra uses the **N-ATLaS (Nigerian Accent and Language Speech)** family of Whisper Small
models from the `NCAIR1` organisation on Hugging Face:

| Language  | Model                          | `language` code |
|-----------|--------------------------------|-----------------|
| English   | `NCAIR1/NigerianAccentedEnglish` | `en`            |
| Yorùbá    | `NCAIR1/Yoruba-ASR`            | `yo`            |
| Hausa     | `NCAIR1/Hausa-ASR`             | `ha`            |
| Igbo      | `NCAIR1/Igbo-ASR`              | `ig`            |
| Pidgin    | `NCAIR1/NigerianAccentedEnglish` with a "best-effort Pidgin" notice | `pcm` |

The integration path is `app/asr.py:transcribe` / `app/asr.py:language_notice`. The
models are loaded via `transformers.pipeline` on first use and cached in a module-level
dict so subsequent requests are cheap (no re-download, no re-initialisation).

## Voice Note Pre-processing

The live WhatsApp demo currently sends `.ogg` Opus files. Before passing to the N-ATLaS
pipeline, `app/asr.py:_convert_to_wav` uses `ffmpeg` to:

1. Downsample to 16 kHz mono PCM WAV — exactly what the Hugging Face model card's
   "Basic Usage" example uses with `librosa.load(..., sr=16000, mono=True)`.
2. Write the result to a temp file that `soundfile.read` can open.

This mirrors the exact N-ATLaS card usage pattern.

## Ledger Export Formats

The Veyra pipeline currently provides the following export options for a trader's
book-keeping records:

- **Excel (.xlsx) — LIVE and always current.** `GET /ledger/{phone_number}.xlsx`
  (see `app/main.py:ledger_xlsx`) regenerates a fresh workbook from the SQLite
  entries table on every request. Nothing is cached on disk; the response headers
  (`Cache-Control: no-store`, `Pragma: no-cache`) force browsers and proxies to
  re-fetch on every click. Columns: **Date**, **Item**, **Amount**, **Type**,
  **Running Balance**, plus a summary footer. Privacy: only that phone number's
  own entries are ever returned; unknown numbers get a 404.

- **Google Sheets export — PLANNED, not built yet.** Exporting the same column
  layout directly into a trader's Google Drive (via Sheets API + OAuth) and
  optionally keeping it in sync on every new voice note is a planned future
  feature. It is intentionally deferred until WhatsApp integration lands, since
  the Google Sheets flow requires:
  1. Per-user OAuth consent (not available in the standalone demo).
  2. A production Supabase/Postgres-backed user table (currently using
     per-process SQLite for the local demo).
  3. A background job or webhook to refresh the sheet row-by-row as the trader
     adds entries, instead of the current one-shot Excel download.

  When implemented, Google Sheets export will live alongside the existing
  `/ledger/{phone}.xlsx` endpoint — the Excel link is and will remain the
  "always available" zero-config export path that works for any trader without
  any external account setup.

## Accuracy Validation

The `scripts/validate_accuracy.py` helper computes Word Error Rate (WER) on a
ground-truth CSV of local-language voice notes and writes the per-language
breakdown into the project root as `wer_*.json`. The WER files are tracked so
regressions in the ASR preprocessing (sampling rate, ffmpeg flags) are visible
before changes ship.
