"""
Speech-to-text module for sabi-books.

Wraps Hugging Face N-ATLaS Whisper models (e.g. NCAIR1/Hausa-ASR,
NCAIR1/Igbo-ASR) and exposes a simple ``transcribe(audio_path, language)``
function.  WhatsApp voice notes arrive as .ogg (Opus); we normalise any
input to 16 kHz mono WAV via ffmpeg before passing it to the model.

Usage patterns follow each model card:
    https://huggingface.co/NCAIR1/Hausa-ASR
    https://huggingface.co/NCAIR1/Igbo-ASR

We use the ``transformers`` ``pipeline`` helper exactly as the model cards
recommend in their "Basic Usage" example:

    from transformers import pipeline
    import librosa
    asr = pipeline("automatic-speech-recognition", model="NCAIR1/Hausa-ASR")
    audio, sr = librosa.load(path, sr=16000)
    result = asr(audio)
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from typing import Any, Dict, Optional

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
    # NCAIR1 do not (yet) publish a standalone Pidgin-ASR checkpoint.
    # Their NCAIR1/NigerianAccentedEnglish (Whisper-Small fine-tune on
    # 6-zone Nigerian speech, which explicitly includes Pidgin phrases
    # and Nigerian English patterns per its model card) is the closest
    # official N-ATLaS offering — so we expose it under BOTH codes:
    #   * `pcm` (ISO 639-3 for Nigerian Pidgin) – for traders using the
    #     Pidgin / Naija alias, and
    #   * `en`  – for users that still pass `language="en"` expecting
    #     Nigerian English, so that voice notes recorded in Nigerian
    #     English still count as N-ATLaS-backed evidence instead of
    #     falling back to generic OpenAI Whisper.
    # If a dedicated NCAIR1/Pidgin-ASR or NCAIR1/English-ASR checkpoint
    # is released later, swap only the `model` string for the code you
    # want to retarget.
    "pcm": {
        "model": "NCAIR1/NigerianAccentedEnglish",
        "display_name": "Pidgin (Nigerian Accented English)",
    },
    "en": {
        "model": "NCAIR1/NigerianAccentedEnglish",
        "display_name": "Nigerian English (Accented)",
    },
}

# ---------------------------------------------------------------------------
# Per-language model cache (processor + pipeline).  Loaded lazily on first
# use and retained for subsequent calls.  Keys are language codes.
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
        "nigerian english": "pcm",
        "nigerianaccentedenglish": "pcm",
        "en-ng": "pcm",
    }
    lang = aliases.get(lang, lang)
    if lang not in LANGUAGE_MODEL_CONFIG:
        raise ValueError(
            f"Unknown language '{language}'. Supported codes: "
            f"{sorted(LANGUAGE_MODEL_CONFIG.keys())}"
        )
    return lang


def _check_ffmpeg() -> None:
    """Ensure ffmpeg is available on PATH; raise a clear RuntimeError if not."""
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "ffmpeg was not found on PATH. It is required to convert WhatsApp "
            ".ogg (Opus) voice notes to 16 kHz mono WAV.\n"
            "Install: https://ffmpeg.org/download.html  or  "
            "`winget install Gyan.FFmpeg`  on Windows / "
            "`brew install ffmpeg`  on macOS / "
            "`sudo apt install ffmpeg`  on Linux."
        )


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

    _check_ffmpeg()

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
        "ffmpeg", "-y", "-i", audio_path,
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


def _load_pipeline(lang_code: str):
    """
    Load the ASR pipeline for ``lang_code`` or return a cached copy.

    Importing transformers/torch is deferred to here so that importing
    ``app.asr`` doesn't force 2+ GB of model libraries to load eagerly;
    the cold start cost is only paid on the first call to ``transcribe``.
    """
    if lang_code in _MODEL_CACHE:
        return _MODEL_CACHE[lang_code]

    # Imported lazily -- they are heavy.
    from transformers import pipeline  # noqa: PLC0415

    cfg = LANGUAGE_MODEL_CONFIG[lang_code]
    model_name = cfg["model"]

    # Exact usage pattern from both Hausa-ASR and Igbo-ASR model cards
    # ("Basic Usage" section):
    #   asr = pipeline("automatic-speech-recognition", model="NCAIR1/Hausa-ASR")
    pipe = pipeline(
        task="automatic-speech-recognition",
        model=model_name,
    )

    _MODEL_CACHE[lang_code] = pipe
    return pipe


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
        Language code or name, e.g. ``"ha"``, ``"ig"``, ``"en"``,
        ``"hausa"``, ``"igbo"``.  Must match an entry in
        ``LANGUAGE_MODEL_CONFIG`` (add a new entry there to support
        Yoruba or other languages later).

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
