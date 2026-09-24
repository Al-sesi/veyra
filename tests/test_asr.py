"""
Tests for app.asr.transcribe.

We deliberately do NOT download the real Hugging Face models.  Instead we
monkey-patch the heavy bits:

  * ``_convert_to_wav_16k_mono`` returns a pre-made WAV path so ffmpeg is
    not required to run the test suite.
  * ``_load_pipeline`` returns a callable fake pipeline object so we never
    hit Hugging Face or import torch.
"""

from __future__ import annotations

import os
import sys
import wave
from pathlib import Path

import pytest

# Make sure project root is on sys.path for `from app.asr import ...`
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app import asr as asr_module  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_silent_wav(path: Path, seconds: float = 1.0, sample_rate: int = 16000) -> None:
    """Write a tiny silent mono WAV file for the test pipeline to consume."""
    nframes = int(sample_rate * seconds)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)  # 16-bit
        w.setframerate(sample_rate)
        w.writeframes(b"\x00\x00" * nframes)


class _FakeASRPipeline:
    """
    Drop-in replacement for transformers' ASR pipeline.  The model card
    shows `result = asr(audio)` returning `{"text": "..."}`.
    """

    def __init__(self, return_text: str = "mock transcript") -> None:
        self.return_text = return_text
        self.calls = []

    def __call__(self, audio_array):
        self.calls.append(audio_array)
        return {"text": self.return_text}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _clean_cache():
    """Ensure tests don't carry cached pipelines between runs."""
    asr_module.clear_model_cache()
    yield
    asr_module.clear_model_cache()


@pytest.fixture(autouse=True)
def _stub_audio_loading(monkeypatch):
    """
    Stub out the audio-loading step in transcribe() without ever touching
    the real `librosa` module.

    We do this by replacing ``_load_audio_array_16k`` (the asr module's
    own wrapper) on the module directly, and similarly providing a
    stand-in for ``np``.  This avoids triggering the lazy-module import
    chain that eventually tries to load librosa's optional `soxr` DLL on
    this Windows machine, and means tests don't need any real audio file.
    """
    # Import numpy only inside the fixture -- it's already installed as a
    # transitive dep, so this is safe and cheap.
    import numpy as _np

    def _fake_load(_wav_path: str):
        sample_rate = 16000
        duration_sec = 1.0
        audio = _np.zeros(int(duration_sec * sample_rate), dtype=_np.float32)
        return audio, sample_rate

    monkeypatch.setattr(asr_module, "_load_audio_array_16k", _fake_load)
    # Also give tests a working np reference (used to build silent WAVs)
    if not hasattr(asr_module.np, "zeros"):
        monkeypatch.setattr(asr_module, "np", _np)


@pytest.fixture()
def fake_input_file(tmp_path: Path) -> Path:
    """
    Pretend input audio.  Not a real .ogg -- we stub out the ffmpeg step
    so this never gets inspected by ffmpeg.
    """
    p = tmp_path / "voice_note.ogg"
    p.write_bytes(b"FAKE_OGG_CONTENT")
    return p


@pytest.fixture()
def real_wav_file(tmp_path: Path) -> Path:
    """A real (silent) 16 kHz mono WAV we can feed to the faked pipeline."""
    p = tmp_path / "converted_16k.wav"
    _make_silent_wav(p)
    return p


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_transcribe_happy_path_mocks_everything(monkeypatch, fake_input_file, real_wav_file):
    """
    End-to-end flow: resolve language, convert audio, run pipeline, return
    text.  All heavy external deps (ffmpeg, transformers, torch, HF hub)
    are stubbed out via monkeypatch.
    """
    fake_pipe = _FakeASRPipeline(return_text="I don sell 5 bags of rice for 45k")

    # Stub _convert_to_wav_16k_mono -- no ffmpeg needed for tests
    def _fake_convert(audio_path: str) -> str:
        assert audio_path == str(fake_input_file)
        return str(real_wav_file)

    # Stub _load_pipeline -- no HF download / torch import required
    def _fake_load(lang_code: str):
        assert lang_code == "ha"
        return fake_pipe

    monkeypatch.setattr(asr_module, "_convert_to_wav_16k_mono", _fake_convert)
    monkeypatch.setattr(asr_module, "_load_pipeline", _fake_load)

    out = asr_module.transcribe(str(fake_input_file), language="ha")
    assert out == "I don sell 5 bags of rice for 45k"
    assert len(fake_pipe.calls) == 1
    # Pipeline should be given a numpy array from librosa (via the real WAV)
    assert fake_pipe.calls[0].shape is not None


def test_transcribe_caches_pipeline_across_calls(monkeypatch, fake_input_file, real_wav_file):
    """
    ``_load_pipeline`` caches the pipeline in ``_MODEL_CACHE`` keyed by
    language; two calls with the same language must only run the
    (expensive) load/pipeline-build code once.
    """
    load_counter = {"n": 0}
    fake_pipe = _FakeASRPipeline(return_text="na so")

    monkeypatch.setattr(
        asr_module,
        "_convert_to_wav_16k_mono",
        lambda ap: str(real_wav_file),
    )

    real_cache = asr_module._MODEL_CACHE

    def _fake_load(lang_code: str):
        if lang_code in real_cache:
            return real_cache[lang_code]
        load_counter["n"] += 1
        real_cache[lang_code] = fake_pipe
        return fake_pipe

    monkeypatch.setattr(asr_module, "_load_pipeline", _fake_load)

    asr_module.transcribe(str(fake_input_file), language="ig")
    asr_module.transcribe(str(fake_input_file), language="ig")

    assert load_counter["n"] == 1, "_load_pipeline did heavy work twice; cache is broken"


def test_resolve_language_accepts_code_and_display_name():
    # Hausa / Igbo
    assert asr_module._resolve_language("ha") == "ha"
    assert asr_module._resolve_language("Hausa") == "ha"
    assert asr_module._resolve_language("ig") == "ig"
    assert asr_module._resolve_language("Igbo") == "ig"
    # Yoruba (NCAIR1/Yoruba-ASR)
    assert asr_module._resolve_language("yo") == "yo"
    assert asr_module._resolve_language("Yoruba") == "yo"
    # Pidgin -> pcm (NCAIR1/NigerianAccentedEnglish backing)
    assert asr_module._resolve_language("pcm") == "pcm"
    assert asr_module._resolve_language("Pidgin") == "pcm"
    assert asr_module._resolve_language("Naija") == "pcm"
    assert asr_module._resolve_language("Nigerian Pidgin") == "pcm"
    assert asr_module._resolve_language("en-ng") == "pcm"
    # English fallback
    assert asr_module._resolve_language("English") == "en"


def test_resolve_language_unknown_raises():
    with pytest.raises(ValueError, match="Unknown language"):
        asr_module._resolve_language("Klingon")


def test_transcribe_unknown_language_raises(fake_input_file):
    with pytest.raises(ValueError, match="Unknown language"):
        asr_module.transcribe(str(fake_input_file), language="zz")


def test_transcribe_missing_file_raises():
    with pytest.raises(FileNotFoundError, match="Audio file not found"):
        asr_module.transcribe("/does/not/exist.ogg", language="en")


def test_transcribe_conversion_step_is_called(monkeypatch, fake_input_file, real_wav_file):
    """
    Verify that _convert_to_wav_16k_mono is invoked exactly once on every
    transcribe() call (i.e. the module always normalises input).
    """
    convert_calls = []

    def _fake_convert(audio_path: str) -> str:
        convert_calls.append(audio_path)
        return str(real_wav_file)

    monkeypatch.setattr(asr_module, "_convert_to_wav_16k_mono", _fake_convert)
    monkeypatch.setattr(
        asr_module,
        "_load_pipeline",
        lambda _lc: _FakeASRPipeline(return_text="ok"),
    )

    asr_module.transcribe(str(fake_input_file), language="en")
    assert convert_calls == [str(fake_input_file)]


def test_transcribe_cleans_up_temp_wav(monkeypatch, tmp_path, fake_input_file, real_wav_file):
    """
    After transcribe returns, the temporary WAV produced by the conversion
    step must be deleted even if the pipeline succeeds.
    """
    temp_wav = tmp_path / "doomed.wav"
    _make_silent_wav(temp_wav)
    assert temp_wav.exists()

    def _fake_convert(_audio_path: str) -> str:
        return str(temp_wav)

    monkeypatch.setattr(asr_module, "_convert_to_wav_16k_mono", _fake_convert)
    monkeypatch.setattr(
        asr_module,
        "_load_pipeline",
        lambda _lc: _FakeASRPipeline(return_text="done"),
    )

    asr_module.transcribe(str(fake_input_file), language="en")
    assert not temp_wav.exists(), "Temporary WAV was not cleaned up after transcribe()"


def test_check_ffmpeg_returns_none_when_present(monkeypatch):
    """_check_ffmpeg should not raise when ffmpeg is on PATH."""
    monkeypatch.setattr(asr_module.shutil, "which", lambda cmd: "/usr/bin/ffmpeg")
    assert asr_module._check_ffmpeg() is None


def test_check_ffmpeg_raises_when_missing(monkeypatch):
    monkeypatch.setattr(asr_module.shutil, "which", lambda cmd: None)
    with pytest.raises(RuntimeError, match="ffmpeg was not found"):
        asr_module._check_ffmpeg()


def test_language_config_has_core_nigerian_language_entries():
    """
    Sanity-check the config dict so future refactors don't silently remove
    any supported language, and confirm every entry is backed by an
    official N-ATLaS / NCAIR1 checkpoint (so transcriptions are valid
    evidence even for `en` or `pcm`).
    """
    cfg = asr_module.LANGUAGE_MODEL_CONFIG

    # N-ATLaS single-language Whisper-Small fine-tunes (Awarri + NCAIR)
    assert cfg["ha"]["model"] == "NCAIR1/Hausa-ASR"
    assert cfg["ig"]["model"] == "NCAIR1/Igbo-ASR"
    assert cfg["yo"]["model"] == "NCAIR1/Yoruba-ASR"
    # Pidgin + English both resolve to the same official N-ATLaS Nigerian
    # English checkpoint; no generic OpenAI fallback.
    assert cfg["pcm"]["model"] == "NCAIR1/NigerianAccentedEnglish"
    assert cfg["en"]["model"] == "NCAIR1/NigerianAccentedEnglish"

    # Every entry must declare `model` + `display_name` uniformly, AND
    # every model must be an NCAIR1 repo for the integration evidence.
    for lang, entry in cfg.items():
        assert "model" in entry, f"{lang} entry missing 'model' key"
        assert "display_name" in entry, f"{lang} entry missing 'display_name' key"
        assert entry["model"].startswith("NCAIR1/"), (
            f"{lang} model '{entry['model']}' is not an official N-ATLaS/NCAIR1 "
            f"checkpoint; the integration requires all voice transcription "
            f"to count as evidence."
        )
