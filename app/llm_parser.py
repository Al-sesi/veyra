"""
LLM-based fallback extraction layer for Veyra using Azure OpenAI / OpenAI.

This module provides a fallback extraction mechanism when the rule-based parser
fails or returns low-confidence results. It uses OpenAI (gpt-4o-mini by default,
or any Azure OpenAI deployment via AZURE_OPENAI_ENDPOINT + AZURE_OPENAI_DEPLOYMENT)
to extract structured bookkeeping entries from transcripts.

The LLM is designed as a fallback only - it does not replace the existing
rule-based parser and all entries must pass through the same validation pipeline.

Environment variables:
  - AZURE_OPENAI_API_KEY  : API key for Azure OpenAI Service (preferred)
  - AZURE_OPENAI_ENDPOINT : https://<name>.openai.azure.com/ (required for Azure)
  - AZURE_OPENAI_DEPLOYMENT : deployment name e.g. gpt-4o-mini (required for Azure)
  - AZURE_OPENAI_API_VERSION : optional, defaults to 2024-08-01-preview

  Fallback (direct OpenAI):
  - OPENAI_API_KEY        : direct OpenAI key (used only if AZURE_OPENAI_API_KEY is unset)
  - OPENAI_MODEL          : optional model name, defaults to gpt-4o-mini
"""

import logging
import os
from typing import List, Dict, Any, Optional
import json

from dotenv import load_dotenv
import time

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger("veyra.llm_parser")

# Language names for the LLM instruction
LANGUAGE_NAMES = {
    "en": "Nigerian English",
    "ha": "Hausa",
    "ig": "Igbo",
    "yo": "Yoruba",
    "pcm": "Nigerian Pidgin"
}

# System instruction — identical structure to the previous prompt; only the
# provider/client code below changes so behaviour stays equivalent.
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

CRITICAL - MULTIPLE TRANSACTIONS:
11. When the transcript contains MULTIPLE transactions (separated by commas, "and", "then", or periods), extract ALL of them as separate entries in the array.
12. Examples of multiple transactions:
    - "I bought rice 4,000, I bought beans 5,000, I bought Indomie 7,000" -> 3 expense entries
    - "I sold rice 5k and beans 3k" -> 2 sale entries
    - "I buy rice 4k, I buy beans 5k" -> 2 expense entries
    - "I bought rice 4,000. I bought beans 5,000." -> 2 expense entries
13. Do NOT stop after the first transaction - extract EVERY transaction mentioned in the transcript.

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

NIGERIAN MARKET TERMINOLOGY:
- "clear" = sell/sold (e.g., "I clear 5 bags rice")
- "move" = sell/sold (e.g., "I move goods for 45k")
- "bag" = sell in bags (e.g., "bagged rice 5k per bag")
- "settle" = pay (e.g., "settle bike man 500")
- "bring money" = payment received (e.g., "customer bring money 10k")
- "balance" = remaining payment (e.g., "balance 5k")
- "on credit" = debt/owing (e.g., "take on credit")
- "now now" = immediately/cash payment
- "later" = credit/debt
- "dash" = give for free/discount
- "carry" = buy/take (e.g., "carry 2 bags")
- "offload" = sell large quantity
- "load" = stock/inventory
- "market women" = female customers
- "mallam" = Hausa trader/customer
- "mama" = older female customer
- "baba" = older male customer
- "brother" / "sister" = customer (respectful)

SALES VARIATIONS (all mean SALE):
- "I sold", "I clear", "I move", "I offload", "I dispose"
- "Customer pay me", "bring money", "cash for", "money for"
- "5 bags gone", "10 cartons clear", "stock finished"
- "Make sale", "deal done", "transaction complete"

EXPENSE VARIATIONS (all mean EXPENSE):
- "I bought", "I purchase", "I acquire", "I get"
- "I spent", "I pay", "I settle", "I dash out"
- "Cost me", "price was", "amount is", "value"
- "Transport fare", "bike money", "keke fee", "bus ticket"
- "Shop rent", "market fee", "union dues", "loading charge"

DEBT VARIATIONS:
- "owe me", "still owe", "never pay", "balance on me"
- "I owe", "dey owe", "credit", "take now pay later"
- "don pay", "settled", "cleared", "balance paid"

CODE-SWITCHING EXAMPLES:
- "I sold rice for 5k, the customer happy" (English)
- "I clear 5 bags rice, the customer dey happy" (Pidgin)
- "Mo ta 5 apo irin, onibara n fara dara" (Yoruba)
- "A sayar da guda 5, abokin ciniki ya fara cikin farin ciki" (Hausa)
- "M ritere 5 sacks osikapa, onye ahia ji mma" (Igbo)

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

Input: "I clear 5 bags rice for 45k"
Output: [{"item": "rice", "quantity": 5, "amount": 45000, "type": "sale", "confidence": "high"}]

Input: "I move goods, customer pay me 10k"
Output: [{"item": "goods", "quantity": null, "amount": 10000, "type": "sale", "confidence": "high"}]

Input: "I bought chin chin, spent 2k"
Output: [{"item": "chin chin", "quantity": null, "amount": 2000, "type": "expense", "confidence": "high"}]

Input: "I settle bike man 500"
Output: [{"item": "transport", "quantity": null, "amount": 500, "type": "expense", "confidence": "high"}]

Input: "I got chin chin and paid 2,000"
Output: [{"item": "chin chin", "quantity": null, "amount": 2000, "type": "expense", "confidence": "high"}]

Input: "I sent for chin chin, spent 2k"
Output: [{"item": "chin chin", "quantity": null, "amount": 2000, "type": "expense", "confidence": "high"}]

Input: "I bought rice 4,000, I bought beans 5,000, I bought Indomie 7,000"
Output: [{"item": "rice", "quantity": null, "amount": 4000, "type": "expense", "confidence": "high"}, {"item": "beans", "quantity": null, "amount": 5000, "type": "expense", "confidence": "high"}, {"item": "Indomie", "quantity": null, "amount": 7000, "type": "expense", "confidence": "high"}]

Input: "I sold rice 5k and beans 3k"
Output: [{"item": "rice", "quantity": null, "amount": 5000, "type": "sale", "confidence": "high"}, {"item": "beans", "quantity": null, "amount": 3000, "type": "sale", "confidence": "high"}]

Input: "I buy rice 4k, I buy beans 5k"
Output: [{"item": "rice", "quantity": null, "amount": 4000, "type": "expense", "confidence": "high"}, {"item": "beans", "quantity": null, "amount": 5000, "type": "expense", "confidence": "high"}]

Input: "I bought rice 4,000. I bought beans 5,000."
Output: [{"item": "rice", "quantity": null, "amount": 4000, "type": "expense", "confidence": "high"}, {"item": "beans", "quantity": null, "amount": 5000, "type": "expense", "confidence": "high"}]

Input: "Mama Ngozi owes me 5k"
Output: [{"item": "Mama Ngozi", "quantity": null, "amount": 5000, "type": "debt_owed_to_me", "confidence": "high"}]

Input: "Mallam still dey owe me 15k"
Output: [{"item": "Mallam", "quantity": null, "amount": 15000, "type": "debt_owed_to_me", "confidence": "high"}]

Input: "I owe Baba Chukwu 10k"
Output: [{"item": "Baba Chukwu", "quantity": null, "amount": 10000, "type": "debt_i_owe", "confidence": "high"}]

Input: "Mama Ngozi don pay"
Output: [{"item": "Mama Ngozi", "quantity": null, "amount": null, "type": "debt_paid", "confidence": "high"}]

Input: "something unclear"
Output: [{"item": null, "quantity": null, "amount": null, "type": "unclear", "confidence": "low"}]
"""


def _get_llm_client_and_model():
    """
    Initialize and return a tuple of (openai_client, model_or_deployment_name).

    Priority:
      1. Azure OpenAI Service  (AZURE_OPENAI_API_KEY + AZURE_OPENAI_ENDPOINT + AZURE_OPENAI_DEPLOYMENT)
      2. Direct OpenAI API     (OPENAI_API_KEY, optional OPENAI_MODEL)
    """
    azure_api_key = os.environ.get("AZURE_OPENAI_API_KEY")
    azure_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
    azure_deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT")
    azure_api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")

    try:
        import openai
    except ImportError:
        raise ImportError(
            "openai package is not installed. Add 'openai>=1.0' to requirements."
        )

    # --- Azure OpenAI Service ---
    if azure_api_key and azure_endpoint and azure_deployment:
        try:
            client = openai.AzureOpenAI(
                api_key=azure_api_key,
                api_version=azure_api_version,
                azure_endpoint=azure_endpoint,
            )
            logger.info("Using Azure OpenAI Service endpoint: %s (deployment: %s)",
                        azure_endpoint, azure_deployment)
            return client, azure_deployment
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Azure OpenAI client: {e}")

    # --- Fallback: direct OpenAI ---
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if openai_api_key:
        try:
            client = openai.OpenAI(api_key=openai_api_key)
            model_name = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
            logger.info("Using direct OpenAI API model: %s", model_name)
            return client, model_name
        except Exception as e:
            raise RuntimeError(f"Failed to initialize OpenAI client: {e}")

    raise ValueError(
        "No LLM credentials provided. Set either "
        "(AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_DEPLOYMENT) for Azure OpenAI, "
        "or OPENAI_API_KEY for the direct OpenAI API."
    )


def _build_user_prompt(transcript: str, language_name: str) -> str:
    return (
        f'Transcript: "{transcript}"\n'
        f"Language: {language_name}\n\n"
        "Extract the bookkeeping entries according to the system instructions.\n"
        "Return ONLY a JSON array of entries."
    )


def warmup_llm() -> dict:
    """
    Warmup the LLM by making a simple test call during startup.
    This ensures the first user message doesn't have the cold start delay.
    
    Returns a dict with warmup status.
    """
    try:
        logger.info("Warming up LLM connection...")
        start_time = time.time()
        
        # Make a simple test call with a known input
        test_transcript = "I sold rice for 5000 naira"
        test_language = "en"
        
        entries, raw_output = extract_entries_llm(test_transcript, test_language)
        
        elapsed = time.time() - start_time
        
        logger.info(f"LLM warmup completed in {elapsed:.2f}s")
        
        return {
            "status": "success",
            "warmup_time_seconds": elapsed,
            "test_entries_count": len(entries),
        }
    except Exception as e:
        logger.warning(f"LLM warmup failed: {e}")
        return {
            "status": "failed",
            "error": str(e),
        }


def extract_entries_llm(transcript: str, language: str) -> tuple[List[Dict[str, Any]], str]:
    """
    Extract bookkeeping entries from a transcript using Azure OpenAI / OpenAI.

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
        If neither Azure OpenAI nor OpenAI credentials are set
    RuntimeError
        If the LLM API call fails for any reason
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
        client, model = _get_llm_client_and_model()

        user_prompt = _build_user_prompt(transcript, language_name)

        logger.info(f"Calling LLM (Azure/OpenAI) for transcript in {language_name}")

        response = client.chat.completions.create(
            model=model,
            temperature=0.1,
            max_tokens=1024,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SYSTEM_INSTRUCTION.strip()},
                {"role": "user", "content": user_prompt},
            ],
        )

        if (not response
                or not response.choices
                or not response.choices[0].message
                or not response.choices[0].message.content):
            logger.error("LLM API returned empty response")
            raise RuntimeError("LLM API returned empty response")

        raw_output = response.choices[0].message.content.strip()
        logger.debug(f"Raw LLM output: {raw_output}")

        # Parse JSON response
        try:
            parsed = json.loads(raw_output)
            # The system prompt asks for a JSON *array*, but because we used
            # `response_format: json_object` some providers/models may wrap
            # it in an object like {"entries": [...]}.  Handle both shapes.
            if isinstance(parsed, dict):
                # Pick the first list-valued key, fallback to empty list
                entries = []
                for _key, val in parsed.items():
                    if isinstance(val, list):
                        entries = val
                        break
            elif isinstance(parsed, list):
                entries = parsed
            else:
                logger.error(f"LLM did not return a list or wrapper object: {type(parsed)}")
                raise ValueError("LLM did not return a JSON array")

            validated_entries = []
            for entry in entries:
                if not isinstance(entry, dict):
                    logger.warning(f"Skipping non-dict entry: {entry}")
                    continue

                validated = {
                    "item": entry.get("item"),
                    "quantity": entry.get("quantity"),
                    "amount": entry.get("amount"),
                    "type": entry.get("type", "unclear"),
                    "confidence": entry.get("confidence", "low")
                }

                valid_types = {"sale", "expense", "debt_owed_to_me", "debt_i_owe",
                               "debt_paid", "correction", "delete_last", "unclear"}
                if validated["type"] not in valid_types:
                    logger.warning(f"Invalid type '{validated['type']}', defaulting to 'unclear'")
                    validated["type"] = "unclear"

                if validated["confidence"] not in {"high", "low"}:
                    logger.warning(f"Invalid confidence '{validated['confidence']}', defaulting to 'low'")
                    validated["confidence"] = "low"

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
