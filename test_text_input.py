"""Test the Veyra pipeline with text input (no AI models needed)"""
from app.pipeline import process_voice_note

# Test with text input instead of audio - this bypasses the AI model download
result = process_voice_note(
    audio_path="",  # Empty since we're using text
    language="en", 
    user_id=None,
    transcript="I sold 5 bags of rice for 45,000 naira"
)

print("Transcript:", result["transcript"])
print("Reply:", result["reply_text"])
print("Entries:", result["entries"])
print("Summary:", result["summary"])
