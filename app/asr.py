"""
Speech-to-text module for sabi-books.

Wraps Hugging Face N-ATLaS Whisper models -- NCAIR1/Yoruba-ASR,
NCAIR1/Hausa-ASR, NCAIR1/Igbo-ASR and NCAIR1/NigerianAccentedEnglish
(which also backs the experimental ``pcm`` route) -- and exposes a simple
``transcribe(audio_path, language)`` function.  WhatsApp voice notes
arrive as .ogg (Opus); we normalise any input to 16 kHz mono WAV via
ffmpeg before passing it to the model.

Usage patterns follow each model card:
    https://huggingface.co/NCAIR1/Yoruba-ASR
    https://huggingface.co/NCAIR1/Hausa-ASR
    https://huggingface.co/NCAIR1/Igbo-ASR
    https://huggingface.co/NCAIR1/NigerianAccentedEnglish

We use the ``transformers`` ``pipeline`` helper exactly as the model cards
recommend in their "Basic Usage" example:

    from transformers import pipeline
    import librosa
    asr = pipeline("automatic-speech-recognition", model="NCAIR1/Hausa-ASR")
    audio, sr = librosa.load(path, sr=16000)
    result = asr(audio)
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any, Dict, Optional

logger = logging.getLogger("veyra.asr")

# ---------------------------------------------------------------------------
# Heavy imports (librosa, numpy, transformers, torch) are deferred until the
# functions that need them are called.  This keeps `import app.asr` cheap and
# lets tests run without these packages installed (they can monkeypatch the
# internal helpers instead).
#
# We expose module-level aliases `librosa` and `np`. Each alias is a plain
# object whose attributes can be *replaced* via `setattr` BEFORE the first access
# to anything else on the alias.  The first attribute access that isn't setting an
# override triggers the real package import and swaps out the proxy dict with the real
# module.  Tests use:
#
#     monkeypatch.setattr(asr_module.librosa, "load", _fake_load)
#
# On first access the real packages are imported lazily.
# ---------------------------------------------------------------------------

class _LazyModuleProxy:
    """
    Lazy stand-in for a Python module.  Allows setting overrides before the real
    module is imported (for monkeypatch).  On any other attribute access,
    import the real module once and permanently switch over.
    """

    __slots__ = ("_modname", "_overrides", "_real_module")

    def __init__(self, modname: str):
        object.__setattr__(self, "_modname", modname)
        object.__setattr__(self, "_overrides", {})
        object.__setattr__(self, "_real_module", None)

    def _import_real(self):
        mod = object.__getattribute__(self, "_real_module")
        if mod is None:
            import importlib
            modname = object.__getattribute__(self, "_modname")
            mod = importlib.import_module(modname)
            # Apply any overrides that tests set via setattr before import
            overrides = object.__getattribute__(self, "_overrides")
            for attr, val in overrides.items():
                setattr(mod, attr, val)
            object.__setattr__(self, "_real_module", mod)
        return mod

    def __setattr__(self, name, value):
        overrides = object.__getattribute__(self, "_overrides")
        overrides[name] = value
        mod = object.__getattribute__(self, "_real_module")
        if mod is not None:
            setattr(mod, name, value)

    def __getattr__(self, name):
        mod = self._import_real()
        return getattr(mod, name)


librosa = _LazyModuleProxy("librosa")
"""Lazy proxy for the `librosa` module; actually imports on first attribute access."""

np = _LazyModuleProxy("numpy")
"""Lazy proxy for the `numpy` module; actually imports on first attribute access."""

# ---------------------------------------------------------------------------
# Language-to-model mapping.  Every language here uses an official model
# from the N-ATLaS programme (NCAIR1 / NITDA / Awarri Technologies) on
# Hugging Face — no generic OpenAI Whisper fallback is used, so any
# transcription is backed by a Nigerian N-ATLaS checkpoint and counts as
# evidence for the integration.
#
# To add a new language drop one entry below (e.g.
# ``"ebira": {"model": "<repo_id>", "display_name": "Ebira"}``) and add
# any aliases in _resolve_language.  Everything else (caching, ffmpeg
# conversion, pipeline setup) works automatically.
# ---------------------------------------------------------------------------
LANGUAGE_MODEL_CONFIG: Dict[str, Dict[str, Any]] = {
    "ha": {
        "model": "NCAIR1/Hausa-ASR",
        "display_name": "Hausa",
    },
    "ig": {
        "model": "NCAIR1/Igbo-ASR",
        "display_name": "Igbo",
    },
    "yo": {
        "model": "NCAIR1/Yoruba-ASR",
        "display_name": "Yoruba",
    },
    # Nigerian-accented English / Pidgin.
    #
    # NCAIR1 do not (yet) publish a standalone Pidgin-ASR checkpoint, so
    # `pcm` routes to the SAME NCAIR1/NigerianAccentedEnglish checkpoint
    # as `en` (their Whisper-Small fine-tune on 6-zone Nigerian speech,
    # which includes Pidgin phrases and Nigerian English patterns per its
    # model card) and is marked "experimental, untested".  If a dedicated
    # NCAIR1/Pidgin-ASR checkpoint is released later, swap only the
    # `model` string for `pcm`.
    "pcm": {
        "model": "NCAIR1/NigerianAccentedEnglish",
        "display_name": "Pidgin (Nigerian Accented English)",
        "experimental": True,
    },
    "en": {
        "model": "NCAIR1/NigerianAccentedEnglish",
        "display_name": "Nigerian English (Accented)",
    },
}

# ---------------------------------------------------------------------------
# Model cache (pipeline per checkpoint).  Loaded lazily on first use and
# retained for subsequent calls.  Keys are model repo ids -- NOT language
# codes -- so languages that share a checkpoint (`en` and `pcm` both use
# NCAIR1/NigerianAccentedEnglish) reuse a single loaded pipeline.
# ---------------------------------------------------------------------------
_MODEL_CACHE: Dict[str, Any] = {}


def _resolve_language(language: str) -> str:
    """Return the canonical language code, raising ValueError if unknown."""
    lang = language.strip().lower()
    # Accept a few common aliases.  Keep this list in sync with the keys
    # of LANGUAGE_MODEL_CONFIG above.
    aliases = {
        "hausa": "ha",
        "igbo": "ig",
        "yoruba": "yo",
        "english": "en",
        "pidgin": "pcm",
        "naija": "pcm",
        "nigerian pidgin": "pcm",
        # Nigerian English is a fully supported language (same checkpoint
        # as `pcm`, but without the experimental/untested flag).
        "nigerian english": "en",
        "nigerianaccentedenglish": "en",
        "en-ng": "en",
    }
    lang = aliases.get(lang, lang)
    if lang not in LANGUAGE_MODEL_CONFIG:
        raise ValueError(
            f"Unknown language '{language}'. Supported codes: "
            f"{sorted(LANGUAGE_MODEL_CONFIG.keys())}"
        )
    return lang


def language_notice(language: str) -> Optional[str]:
    """Return a user-facing caveat for ``language``, or None if it is stable.

    ``pcm`` (Nigerian Pidgin) is routed to the NCAIR1/NigerianAccentedEnglish
    checkpoint and shipped as "experimental, untested"; callers (CLI, API)
    should show this notice so nobody mistakes Pidgin transcripts for
    validated output.
    """
    cfg = LANGUAGE_MODEL_CONFIG[_resolve_language(language)]
    if not cfg.get("experimental"):
        return None
    return (
        f"Warning: {cfg['display_name']} support is experimental and untested "
        f"(routed to the {cfg['model']} checkpoint)."
    )


def _get_ffmpeg_path() -> Optional[str]:
    """
    Find ffmpeg executable in order of preference:
    1. System ffmpeg on PATH
    2. Bundled binary from imageio-ffmpeg package
    Returns None if neither is available.
    """
    # First check system PATH
    system_ffmpeg = shutil.which("ffmpeg")
    if system_ffmpeg:
        logger.info(f"Using system ffmpeg: {system_ffmpeg}")
        return system_ffmpeg
    
    # Fall back to bundled ffmpeg from imageio-ffmpeg
    try:
        import imageio_ffmpeg
        bundled_ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        if bundled_ffmpeg and os.path.exists(bundled_ffmpeg):
            logger.info(f"Using bundled ffmpeg from imageio-ffmpeg: {bundled_ffmpeg}")
            return bundled_ffmpeg
    except ImportError:
        logger.warning("imageio-ffmpeg not installed, no bundled ffmpeg available")
    except Exception as e:
        logger.warning(f"Failed to get bundled ffmpeg: {e}")
    
    return None


def _convert_to_wav_16k_mono(audio_path: str) -> str:
    """
    Convert any audio file to 16 kHz mono PCM WAV via ffmpeg.

    WhatsApp voice notes arrive as .ogg (Opus).  The ASR model was trained
    on 16 kHz mono audio (per model card: "16kHz recommended"), so
    standardising here keeps results consistent regardless of input format.

    Returns the path to the temporary WAV file (caller is responsible for
    cleaning it up).
    """
    if not os.path.isfile(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    ffmpeg_path = _get_ffmpeg_path()
    if not ffmpeg_path:
        raise RuntimeError(
            "ffmpeg was not found on PATH and no bundled ffmpeg is available. "
            "It is required to convert WhatsApp .ogg (Opus) voice notes to 16 kHz mono WAV.\n"
            "Install: https://ffmpeg.org/download.html  or  "
            "`winget install Gyan.FFmpeg`  on Windows / "
            "`brew install ffmpeg`  on macOS / "
            "`sudo apt install ffmpeg`  on Linux."
        )

    tmp_fd, wav_path = tempfile.mkstemp(
        prefix=f"sabibooks_asr_{os.getpid()}_", suffix=".wav"
    )
    os.close(tmp_fd)

    # ffmpeg flags:
    #   -y             overwrite output without asking
    #   -i <input>     input file
    #   -ar 16000      16 kHz sample rate
    #   -ac 1          mono (1 audio channel)
    #   -acodec pcm_s16le   16-bit little-endian PCM (standard WAV)
    cmd = [
        ffmpeg_path, "-y", "-i", audio_path,
        "-ar", "16000",
        "-ac", "1",
        "-acodec", "pcm_s16le",
        wav_path,
    ]
    try:
        subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
    except subprocess.CalledProcessError as exc:
        try:
            os.unlink(wav_path)
        except OSError:
            pass
        stderr_msg = exc.stderr.decode("utf-8", errors="replace") if exc.stderr else ""
        raise RuntimeError(
            f"ffmpeg failed to convert audio file. ffmpeg output:\n{stderr_msg}"
        ) from exc

    return wav_path


def _load_pipeline(lang_code: str, max_retries: int = 3, retry_delay: float = 2.0):
    """
    Load the ASR pipeline for ``lang_code`` or return a cached copy.

    Importing transformers/torch is deferred to here so that importing
    ``app.asr`` doesn't force 2+ GB of model libraries to load eagerly;
    the cold start cost is only paid on the first call to ``transcribe``.

    Args:
        lang_code: Language code to load model for
        max_retries: Maximum number of retry attempts for model loading
        retry_delay: Initial delay between retries in seconds (exponential backoff)
    """
    cfg = LANGUAGE_MODEL_CONFIG[lang_code]
    model_name = cfg["model"]

    if model_name in _MODEL_CACHE:
        return _MODEL_CACHE[model_name]

    # Imported lazily -- they are heavy.
    from transformers import pipeline  # noqa: PLC0415

    logger.info(f"Loading ASR model {model_name} for language {lang_code}...")

    # Exact usage pattern from all four NCAIR1 model cards ("Basic Usage"
    # section), e.g.:
    #   asr = pipeline("automatic-speech-recognition", model="NCAIR1/Hausa-ASR")
    # Every card caps inference at 30 s of audio, so chunk_length_s=30 both
    # follows the card and lets long voice notes run without raising
    # "more than 3000 mel input features".

    last_error = None
    for attempt in range(max_retries):
        try:
            pipe = pipeline(
                task="automatic-speech-recognition",
                model=model_name,
                chunk_length_s=30,
            )
            _MODEL_CACHE[model_name] = pipe
            logger.info(f"Successfully loaded ASR model {model_name}")
            return pipe
        except Exception as e:
            last_error = e
            logger.warning(
                f"Attempt {attempt + 1}/{max_retries} failed to load model {model_name}: {e}"
            )
            if attempt < max_retries - 1:
                wait_time = retry_delay * (2 ** attempt)  # Exponential backoff
                logger.info(f"Retrying in {wait_time:.1f} seconds...")
                time.sleep(wait_time)

    # All retries failed
    logger.error(f"Failed to load ASR model {model_name} after {max_retries} attempts")
    raise RuntimeError(
        f"Failed to load AI model for {cfg['display_name']}. "
        f"This may be due to network issues or limited resources. "
        f"Please try again or use text input instead."
    ) from last_error


def get_model_status() -> dict:
    """
    Return current model loading status for monitoring and debugging.
    Useful for health checks and determining if models are ready.
    """
    return {
        "cached_models": list(_MODEL_CACHE.keys()),
        "total_languages": len(LANGUAGE_MODEL_CONFIG),
        "total_unique_models": len(set(cfg["model"] for cfg in LANGUAGE_MODEL_CONFIG.values())),
    }


def load_essential_models(max_retries: int = 2, retry_delay: float = 1.0) -> dict:
    """
    Load the most commonly used models during startup to reduce cold start latency.
    Returns status of loaded models.

    Args:
        max_retries: Maximum retry attempts for model loading during startup
        retry_delay: Initial delay between retries in seconds
    """
    # Preload only English initially to avoid crashing free tier
    # Other languages will load on-demand when needed
    essential_languages = ["en"]
    loaded = []
    failed = []

    for lang in essential_languages:
        try:
            _load_pipeline(lang, max_retries=max_retries, retry_delay=retry_delay)
            loaded.append(lang)
            logger.info(f"Preloaded essential model for {lang}")
        except Exception as e:
            failed.append({"language": lang, "error": str(e)})
            logger.warning(f"Failed to preload essential model for {lang}: {e}")

    return {
        "loaded": loaded,
        "failed": failed,
        "total_cached": len(_MODEL_CACHE),
    }


def _load_audio_array_16k(wav_path: str):
    """
    Load a WAV file into a 16 kHz mono float array via librosa.

    This is exactly the model-card snippet:
        audio, sr = librosa.load("your_hausa_audio.wav", sr=16000)

    Pulled out as its own function so tests can monkeypatch it in ONE call
    to ``monkeypatch.setattr`` without ever importing librosa (the lazy
    librosa module has tricky interaction with pytest's monkeypatch because
    pytest reads the old value before writing a new one).
    """
    # librosa is a lazy proxy; attribute access triggers the real import.
    audio_array, sample_rate = librosa.load(wav_path, sr=16000)
    return audio_array, sample_rate


def transcribe(audio_path: str, language: str) -> str:
    """
    Transcribe an audio file (e.g. a WhatsApp .ogg voice note) to text.

    Parameters
    ----------
    audio_path : str
        Path to the input audio.  Any format supported by ffmpeg works;
        WhatsApp's default .ogg (Opus) is converted automatically.
    language : str
        Language code or name: ``"yo"``, ``"ha"``, ``"ig"``, ``"en"`` or
        ``"pcm"`` (aliases such as ``"Yoruba"``, ``"Hausa"``, ``"Naija"``
        also work).  Must match an entry in ``LANGUAGE_MODEL_CONFIG``
        (add a new entry there to support more languages).  ``pcm`` is
        marked experimental and untested -- see ``language_notice``.

    Returns
    -------
    str
        The transcription text.  Whitespace-normalised and stripped.
    """
    lang_code = _resolve_language(language)

    # Step 1: convert input to 16 kHz mono WAV.  This is the model's
    # expected format per the model cards ("16kHz recommended").
    wav_path: Optional[str] = None
    try:
        wav_path = _convert_to_wav_16k_mono(audio_path)

        # Step 2: load audio array with librosa @ 16 kHz -- same call as
        # the model-card examples (see docstring of _load_audio_array_16k).
        audio_array, _sample_rate = _load_audio_array_16k(wav_path)
    finally:
        if wav_path is not None:
            try:
                os.unlink(wav_path)
            except OSError:
                pass

    # Step 3: get/cache the pipeline and run inference.
    pipe = _load_pipeline(lang_code)
    result = pipe(audio_array)

    # pipeline() returns dict like {"text": "..."}
    text = (result or {}).get("text", "") or ""
    return text.strip()


def clear_model_cache() -> None:
    """Drop all cached ASR pipelines (useful in tests or to free RAM)."""
    _MODEL_CACHE.clear()
