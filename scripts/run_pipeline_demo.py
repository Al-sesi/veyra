#!/usr/bin/env python3
"""
Run the full Veyra pipeline on a single audio file and display results
in a clean format for screenshot/submission purposes.
"""

import os
import sys
from pathlib import Path

# Set UTF-8 encoding for stdout to handle Yoruba diacritics
if sys.platform == "win32":
    import codecs
    sys.stdout = codecs.getwriter("utf-8")(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter("utf-8")(sys.stderr.buffer, 'strict')

# Load environment variables from .env
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.asr import transcribe, LANGUAGE_MODEL_CONFIG, _load_pipeline
from app.pipeline import process_voice_note
from app.db import USE_POSTGRES, DATABASE_URL


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/run_pipeline_demo.py <audio_file> [language]")
        print("Example: python scripts/run_pipeline_demo.py test1.ogg yo")
        sys.exit(1)

    audio_path = sys.argv[1]
    language = sys.argv[2] if len(sys.argv) > 2 else "yo"

    if not Path(audio_path).exists():
        print(f"Error: Audio file not found: {audio_path}")
        sys.exit(1)

    print("=" * 70)
    print("VEYRA PIPELINE DEMO")
    print("=" * 70)
    print()

    # 1. Show N-ATLaS model used
    lang_code = language.lower()
    if lang_code not in LANGUAGE_MODEL_CONFIG:
        print(f"Error: Unknown language code: {lang_code}")
        print(f"Supported: {list(LANGUAGE_MODEL_CONFIG.keys())}")
        sys.exit(1)

    model_config = LANGUAGE_MODEL_CONFIG[lang_code]
    model_repo = model_config["model"]
    display_name = model_config["display_name"]

    print("1. N-ATLaS MODEL USED")
    print("-" * 70)
    print(f"   Repository: {model_repo}")
    print(f"   Language:   {display_name}")
    print(f"   Code:       {lang_code}")
    print()

    # 2. Transcribe
    print("2. TRANSCRIBING AUDIO...")
    print("-" * 70)
    try:
        transcript = transcribe(audio_path, language)
        print(f"   Raw Transcript: {transcript}")
    except Exception as e:
        print(f"   Error during transcription: {e}")
        sys.exit(1)
    print()

    # 3. Run full pipeline
    print("3. RUNNING FULL PIPELINE...")
    print("-" * 70)
    try:
        result = process_voice_note(
            audio_path=audio_path,
            language=language,
            user_id=1,  # Use user_id 1 for demo
        )
    except Exception as e:
        print(f"   Error during pipeline: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # 4. Show entries extracted
    print("4. ENTRIES EXTRACTED")
    print("-" * 70)
    entries = result.get("entries", [])
    if not entries:
        print("   No entries extracted")
    else:
        for i, entry in enumerate(entries, 1):
            print(f"   Entry {i}:")
            print(f"     Item:     {entry.get('item')}")
            print(f"     Quantity: {entry.get('quantity')}")
            print(f"     Amount:   {entry.get('amount')} naira")
            print(f"     Type:     {entry.get('type')}")
    print()

    # 5. Show reply text
    print("5. REPLY TEXT")
    print("-" * 70)
    reply_text = result.get("reply_text", "")
    print(f"   {reply_text}")
    print()

    # 6. Database confirmation
    print("6. DATABASE CONFIRMATION")
    print("-" * 70)
    if USE_POSTGRES:
        try:
            from urllib.parse import urlparse
            parsed = urlparse(DATABASE_URL)
            safe_display = f"{parsed.scheme}://[hidden]@{parsed.hostname}:{parsed.port}{parsed.path}"
            print(f"   Database: PostgreSQL (Supabase)")
            print(f"   Connection: {safe_display}")
        except:
            print(f"   Database: PostgreSQL (Supabase)")
            print(f"   Connection: {DATABASE_URL[:30]}...{DATABASE_URL[-10:]}")
    else:
        from app.db import DEFAULT_DB_PATH
        print(f"   Database: SQLite")
        print(f"   Path: {DEFAULT_DB_PATH}")

    if entries:
        print(f"   Status: ✅ SAVED ({len(entries)} entry/entries)")
    else:
        print(f"   Status: ⚠️  No entries saved (unclear transcript)")
    print()

    print("=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
