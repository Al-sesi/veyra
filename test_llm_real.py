"""
Test the LLM parser with real API calls to verify functionality.
"""

import os
import sys
from pathlib import Path

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# Add app directory to path
sys.path.insert(0, str(Path(__file__).parent))

from app.llm_parser import extract_entries_llm

print("=== Testing LLM Parser with Real API ===\n")

# Test with a simple English transaction
test_transcripts = [
    ("I sold rice for 5k", "en", "Simple English sale"),
    ("I bought chin chin for 2k", "en", "Simple English expense"),
    ("Na saya shinkafa 2k", "ha", "Hausa expense"),
    ("Mo ra chin chin 2k", "yo", "Yoruba expense"),
]

for transcript, language, description in test_transcripts:
    print(f"Testing: {description}")
    print(f"Transcript: {transcript}")
    print(f"Language: {language}")
    
    try:
        entries, raw_output = extract_entries_llm(transcript, language)
        print(f"Success! Extracted {len(entries)} entries:")
        for i, entry in enumerate(entries):
            print(f"  Entry {i+1}: {entry}")
        print(f"Raw output: {raw_output[:100]}...")
    except Exception as e:
        print(f"Error: {e}")
    
    print("\n" + "="*60 + "\n")

print("=== LLM API Test Complete ===")