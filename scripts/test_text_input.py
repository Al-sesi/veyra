#!/usr/bin/env python3
"""
Test text input support across all supported languages.
Demonstrates that text input produces the same structured entry format as voice input.
"""

import os
import sys
from pathlib import Path

# Set UTF-8 encoding for stdout to handle non-ASCII characters
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

from app.pipeline import process_voice_note
from app.db import USE_POSTGRES, DATABASE_URL


def test_text_input(text: str, language: str, user_id: int = 1):
    """Test text input through the pipeline."""
    print(f"\nTesting: '{text}' (language: {language})")
    print("-" * 70)
    
    try:
        result = process_voice_note(
            audio_path="",
            language=language,
            user_id=user_id,
            transcript=text,
        )
        
        print(f"✓ Transcript: {result['transcript']}")
        print(f"✓ Entries extracted: {len(result['entries'])}")
        
        for i, entry in enumerate(result['entries'], 1):
            print(f"  Entry {i}: {entry['item']}, {entry['amount']} naira, {entry['type']}")
        
        print(f"✓ Reply: {result['reply_text']}")
        print(f"✓ Summary - Sales: {result['summary']['total_sales']}, "
              f"Expenses: {result['summary']['total_expenses']}, "
              f"Profit: {result['summary']['profit']}")
        
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("=" * 70)
    print("VEYRA TEXT INPUT DEMO")
    print("=" * 70)
    
    # Database info
    print("\nDatabase Configuration:")
    print("-" * 70)
    if USE_POSTGRES:
        try:
            from urllib.parse import urlparse
            parsed = urlparse(DATABASE_URL)
            safe_display = f"{parsed.scheme}://[hidden]@{parsed.hostname}:{parsed.port}{parsed.path}"
            print(f"PostgreSQL (Supabase): {safe_display}")
        except:
            print(f"PostgreSQL (Supabase): {DATABASE_URL[:30]}...{DATABASE_URL[-10:]}")
    else:
        from app.db import DEFAULT_DB_PATH
        print(f"SQLite: {DEFAULT_DB_PATH}")
    
    # Test all supported languages
    test_cases = [
        # English
        ("I bought rice for 5000", "en", 1),
        ("I sold rice 45k and paid transport 3k", "en", 2),
        
        # Nigerian Pidgin
        ("I buy rice 5k", "pcm", 3),
        ("I don sell beans 20k", "pcm", 4),
        
        # Yoruba
        ("Mo ra iresi fun 5000", "yo", 5),
        ("Mo ta akara 2k", "yo", 6),
        
        # Hausa
        ("Na saya shinkafa 5k", "ha", 7),
        ("Na saya shinkafa, biyu", "ha", 8),
        
        # Igbo
        ("M zụrọ osikapa 5k", "ig", 9),
    ]
    
    print("\n" + "=" * 70)
    print("TESTING TEXT INPUT ACROSS LANGUAGES")
    print("=" * 70)
    
    passed = 0
    failed = 0
    
    for text, language, user_id in test_cases:
        if test_text_input(text, language, user_id):
            passed += 1
        else:
            failed += 1
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total tests: {len(test_cases)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    
    if failed == 0:
        print("\n✓ All text input tests passed!")
    else:
        print(f"\n✗ {failed} test(s) failed")
    
    print("=" * 70)


if __name__ == "__main__":
    main()
