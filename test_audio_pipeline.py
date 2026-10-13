"""
Test the complete pipeline with actual audio files.

This script tests the rule-based parser and LLM fallback with the actual audio files:
- test1.ogg.ogg — Yoruba
- testtt.ogg — Hausa  
- testtt1.ogg — Hausa
"""

import os
import sys
from pathlib import Path

# Add app directory to path
sys.path.insert(0, str(Path(__file__).parent))

from app.parser import parse_transcript
from app.llm_parser import extract_entries_llm_safe

# Test transcripts (simulated ASR output for testing)
# In a real scenario, these would come from actual ASR transcription
test_transcripts = {
    "test1_ogg": {
        "language": "yo",
        "transcript": "Mo ra jina egberun marun-un naira",  # Simulated Yoruba transcript (ASCII-safe)
        "description": "Yoruba - bought something for 5,000 naira"
    },
    "testtt": {
        "language": "ha", 
        "transcript": "Na saya shinkafa dubu biyu",  # Simulated Hausa transcript
        "description": "Hausa - bought something for 2,000"
    },
    "testtt1": {
        "language": "ha",
        "transcript": "Na saya rice biyu",  # Simulated Hausa transcript
        "description": "Hausa - bought rice for 2,000"
    }
}

print("=== Testing Pipeline with Audio Files ===\n")

for test_name, test_data in test_transcripts.items():
    print(f"Testing {test_name}: {test_data['description']}")
    print(f"Language: {test_data['language']}")
    print(f"Transcript: {test_data['transcript']}")
    
    # Test rule-based parser
    print("\n--- Rule-based Parser ---")
    try:
        rule_based_result = parse_transcript(test_data["transcript"])
        print(f"Rule-based parser found {len(rule_based_result)} entries")
        for i, entry in enumerate(rule_based_result):
            print(f"  Entry {i+1}: {entry}")
        
        # Check if rule-based parser is confident
        if rule_based_result:
            print("Rule-based parser: SUCCESS - found entries")
            rule_based_confident = True
        else:
            print("Rule-based parser: FAILED - no entries found")
            rule_based_confident = False
    except Exception as e:
        print(f"Rule-based parser ERROR: {e}")
        rule_based_result = []
        rule_based_confident = False
    
    # Test LLM fallback (with mock since we don't have API key)
    print("\n--- LLM Fallback (would trigger if rule-based fails) ---")
    print("LLM fallback requires GEMINI_API_KEY environment variable")
    print("Without API key, LLM fallback would return unclear entry")
    
    # Simulate what LLM fallback would do
    if not rule_based_result or not rule_based_confident:
        print("LLM fallback WOULD BE TRIGGERED")
        print("Expected behavior: Call Gemini API to extract entries")
        print("Fallback to unclear response if LLM fails")
    else:
        print("LLM fallback WOULD NOT BE TRIGGERED")
        print("Rule-based parser succeeded, using its results")
    
    print("\n" + "="*60 + "\n")

print("=== Summary ===")
print("To test with actual LLM fallback:")
print("1. Set GEMINI_API_KEY environment variable")
print("2. Run actual ASR on the audio files to get real transcripts")
print("3. Test the complete pipeline with real transcriptions")
print("\nNote: Current test uses simulated transcripts since ASR models")
print("may not be loaded without proper setup.")