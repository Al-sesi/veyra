"""
LLM-based fallback extraction layer for Veyra using Google Gemini 2.5 Flash-Lite.

This module provides a fallback extraction mechanism when the rule-based parser
fails or returns low-confidence results. It uses Google Gemini 2.5 Flash-Lite to
extract structured bookkeeping entries from transcripts.

The LLM is designed as a fallback only - it does not replace the existing
rule-based parser and all entries must pass through the same validation pipeline.
"""

import logging
import os
from typing import List, Dict, Any, Optional
import json

logger = logging.getLogger("veyra.llm_parser")

# Language names for the LLM instruction
LANGUAGE_NAMES = {
    "en": "Nigerian English",
    "ha": "Hausa", 
    "ig": "Igbo",
    "yo": "Yoruba",
    "pcm": "Nigerian Pidgin"
}

# System instruction for Gemini
SYSTEM_INSTRUCTION = """
You are a bookkeeping extraction assistant for Nigerian market traders. 
Extract structured bookkeeping entries from the transcript and return ONLY valid JSON.

STRICT RULES:
1. Return ONLY a JSON array of entries. No markdown, no explanations, no extra text.
2. Each entry must have exactly these fields:
   - "item": string - the item/service being transacted
   - "quantity": number or null - if quantity is mentioned, otherwise null
   - "amount": number in Nigerian naira, or null if unclear
   - "type": one of: "sale", "expense", "debt_owed_to_me", "debt_i_owe", "debt_paid", "correction", "delete_last", "unclear"
   - "confidence": "high" or "low"

3. NEVER guess an amount. If the amount is not clearly stated, return null for "amount".
4. If the meaning or amount is ambiguous, use "confidence": "low".
5. Preserve the exact meaning of the user's language.
6. Support English, Hausa, Yoruba, and Nigerian English (Pidgin).
7. Do NOT invent quantities, amounts, items, or transaction types.
8. Do NOT make assumptions based on common prices.
9. This is an extraction fallback only - you are NOT replacing any existing validation or safety logic.
10. If you cannot extract a valid entry, return a single entry with type "unclear" and confidence "low".

NATURAL LANGUAGE UNDERSTANDING:
- Understand the SEMANTIC MEANING, not just exact keywords
- Focus on what the user intends, not how they phrase it
- All these variations mean the same thing (expense/purchase):
  * "I bought chin chin for 2k"
  * "I buy chin chin for 2k"
  * "I purchased chin chin, 2k"
  * "I sent for chin chin, spent 2k"
  * "I got chin chin and paid 2,000"
  * "It cost me 2k for chin chin"
  * "Chin chin was 2k"
- Understand tense variations (bought, buy, buying, paid, pay, paying)
- Understand informal expressions and conversational phrasing
- Understand code-switching (mixing languages in one sentence)
- If the context clearly indicates a purchase/expense with an amount, extract it as an expense

TRANSACTION TYPES:
- "sale": Selling items
- "expense": Buying items or paying for services (transport, rent, etc.)
- "debt_owed_to_me": Someone owes the speaker money
- "debt_i_owe": Speaker owes someone money  
- "debt_paid": A debt was paid
- "correction": Correcting a previous entry
- "delete_last": Removing the last entry
- "unclear": Cannot determine the transaction type

AMOUNT RULES:
- Only extract amounts that are explicitly stated in the transcript
- If no amount is mentioned, set "amount": null
- Do not infer amounts from context or market prices
- Handle Nigerian shorthand: "5k" = 5000, "10k" = 10000, etc.

EXAMPLES:
Input: "I sold rice for 5k"
Output: [{"item": "rice", "quantity": null, "amount": 5000, "type": "sale", "confidence": "high"}]

Input: "I bought chin chin, spent 2k"  
Output: [{"item": "chin chin", "quantity": null, "amount": 2000, "type": "expense", "confidence": "high"}]

Input: "I got chin chin and paid 2,000"
Output: [{"item": "chin chin", "quantity": null, "amount": 2000, "type": "expense", "confidence": "high"}]

Input: "I sent for chin chin, spent 2k"
Output: [{"item": "chin chin", "quantity": null, "amount": 2000, "type": "expense", "confidence": "high"}]

Input: "Mama Ngozi owes me 5k"
Output: [{"item": "Mama Ngozi", "quantity": null, "amount": 5000, "type": "debt_owed_to_me", "confidence": "high"}]

Input: "something unclear"
Output: [{"item": null, "quantity": null, "amount": null, "type": "unclear", "confidence": "low"}]
"""


def _get_gemini_client():
    """
    Initialize and return the Gemini client.
    Reads API key from GEMINI_API_KEY environment variable.
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable is not set")
    
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        return genai.GenerativeModel(
            model_name="gemini-1.5-flash",  # Using Flash model for speed and cost
            generation_config={
                "temperature": 0.1,  # Low temperature for consistent extraction
                "max_output_tokens": 1024,
                "response_mime_type": "application/json",
            }
        )
    except ImportError:
        raise ImportError("google-generativeai package is not installed. Add it to requirements.")
    except Exception as e:
        raise RuntimeError(f"Failed to initialize Gemini client: {e}")


def extract_entries_llm(transcript: str, language: str) -> tuple[List[Dict[str, Any]], str]:
    """
    Extract bookkeeping entries from a transcript using Gemini Flash.
    
    This is a fallback extraction mechanism that should only be called when:
    - The rule-based parser returns zero entries, OR
    - The rule-based parser returns entries but all are flagged as low-confidence
    
    Parameters
    ----------
    transcript : str
        The text transcript from ASR (speech-to-text)
    language : str
        Language code: "en", "ha", "ig", "yo", or "pcm"
    
    Returns
    -------
    tuple[list[dict], str]
        (entries, raw_output) where entries is a list of extracted entries with fields:
        - item (str | null): item/service description
        - quantity (float | null): quantity if mentioned
        - amount (int | null): amount in naira, or null if unclear
        - type (str): transaction type
        - confidence (str): "high" or "low"
        and raw_output is the raw JSON string from the LLM
    
    Raises
    ------
    ValueError
        If GEMINI_API_KEY is not set
    RuntimeError
        If the Gemini API call fails for any reason
    """
    if not transcript or not transcript.strip():
        logger.warning("Empty transcript provided to LLM parser")
        return [{
            "item": None,
            "quantity": None, 
            "amount": None,
            "type": "unclear",
            "confidence": "low"
        }], "Empty transcript"
    
    language_name = LANGUAGE_NAMES.get(language, language)
    
    try:
        model = _get_gemini_client()
        
        prompt = f"""
Transcript: "{transcript}"
Language: {language_name}

Extract the bookkeeping entries according to the system instructions.
Return ONLY a JSON array of entries.
"""
        
        logger.info(f"Calling Gemini API for transcript in {language_name}")
        
        response = model.generate_content(
            prompt,
            generation_config={
                "temperature": 0.1,
                "max_output_tokens": 1024,
                "response_mime_type": "application/json",
            }
        )
        
        if not response or not response.text:
            logger.error("Gemini API returned empty response")
            raise RuntimeError("Gemini API returned empty response")
        
        raw_output = response.text.strip()
        logger.debug(f"Raw LLM output: {raw_output}")
        
        # Parse JSON response
        try:
            entries = json.loads(raw_output)
            if not isinstance(entries, list):
                logger.error(f"LLM did not return a list: {type(entries)}")
                raise ValueError("LLM did not return a JSON array")
            
            # Validate entry structure
            validated_entries = []
            for entry in entries:
                if not isinstance(entry, dict):
                    logger.warning(f"Skipping non-dict entry: {entry}")
                    continue
                
                # Ensure required fields exist with proper types
                validated = {
                    "item": entry.get("item"),
                    "quantity": entry.get("quantity"),
                    "amount": entry.get("amount"),
                    "type": entry.get("type", "unclear"),
                    "confidence": entry.get("confidence", "low")
                }
                
                # Validate type field
                valid_types = {"sale", "expense", "debt_owed_to_me", "debt_i_owe", 
                               "debt_paid", "correction", "delete_last", "unclear"}
                if validated["type"] not in valid_types:
                    logger.warning(f"Invalid type '{validated['type']}', defaulting to 'unclear'")
                    validated["type"] = "unclear"
                
                # Validate confidence field
                if validated["confidence"] not in {"high", "low"}:
                    logger.warning(f"Invalid confidence '{validated['confidence']}', defaulting to 'low'")
                    validated["confidence"] = "low"
                
                # Ensure amount is null if unclear
                if validated["amount"] is not None and not isinstance(validated["amount"], (int, float)):
                    logger.warning(f"Invalid amount type {type(validated['amount'])}, setting to null")
                    validated["amount"] = None
                
                validated_entries.append(validated)
            
            logger.info(f"Successfully extracted {len(validated_entries)} entries from LLM")
            return validated_entries, raw_output
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM JSON response: {e}")
            logger.error(f"Raw output was: {raw_output}")
            raise RuntimeError(f"Invalid JSON from LLM: {e}")
        
    except Exception as e:
        logger.error(f"LLM extraction failed: {e}")
        # Re-raise the original exception type if it's ValueError, otherwise wrap in RuntimeError
        if isinstance(e, ValueError):
            raise
        raise RuntimeError(f"LLM extraction failed: {e}") from e


def extract_entries_llm_safe(transcript: str, language: str) -> tuple[List[Dict[str, Any]], str]:
    """
    Safe wrapper for extract_entries_llm that handles all errors gracefully.
    
    If the LLM call fails for any reason (timeout, API error, rate limit, network error,
    malformed response, invalid JSON, unexpected structure), this function returns
    an "unclear" entry with low confidence instead of crashing.
    
    Parameters
    ----------
    transcript : str
        The text transcript from ASR
    language : str
        Language code
    
    Returns
    -------
    tuple[list[dict], str]
        Returns (entries, raw_output) where entries is always a list with at least one entry.
        On error, returns a single entry with type "unclear" and confidence "low",
        and raw_output contains the error message.
    """
    try:
        return extract_entries_llm(transcript, language)
    except Exception as e:
        logger.warning(f"LLM extraction failed, returning unclear entry: {e}")
        return [{
            "item": None,
            "quantity": None,
            "amount": None,
            "type": "unclear",
            "confidence": "low"
        }], str(e)