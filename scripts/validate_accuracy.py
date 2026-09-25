#!/usr/bin/env python3
"""
Word Error Rate (WER) validation of Veyra ASR against real speech.

For each language (ha, ig, yo, en) the script takes N clips from the FLEURS
test split (google/fleurs on the HF hub -- real recorded studio speech with
reference transcripts), transcribes them with ``app.asr.transcribe`` using
the matching NCAIR1/N-ATLaS checkpoint, and computes word-level WER against
the reference transcript.

`pcm` (Nigerian Pidgin) has NO public audio corpus in FLEURS or Common
Voice, so it cannot be validated here -- it is reported as N/A. Real Pidgin
voice notes recorded by a speaker are needed to measure it.

Prerequisites:
    - the four NCAIR1 checkpoints already in the HF cache (see
      scripts/try_asr.py / `hf download NCAIR1/<model>`)
    - FLEURS test.tsv + audio/test.tar.gz cached, e.g.:
        python -c "from huggingface_hub import hf_hub_download; \\
          [hf_hub_download('google/fleurs', f) for f in \\
           ['data/ha_ng/test.tsv','data/ha_ng/audio/test.tar.gz']]"

Usage:
    python scripts/validate_accuracy.py                 # 10 clips per language
    python scripts/validate_accuracy.py --clips 25 --langs ha ig
    python scripts/validate_accuracy.py --out wer_results.json
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import tarfile
import tempfile
import unicodedata
from pathlib import Path

# FLEURS config name on the hub -> Veyra / app.asr language code.
# `pcm` is intentionally absent: no Pidgin config exists in FLEURS.
FLEURS_LANGS = {
    "ha": "ha_ng",
    "ig": "ig_ng",
    "yo": "yo_ng",
    "en": "en_us",
}

_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)


def fleurs_snapshot() -> Path:
    """Return the local snapshot dir of the cached google/fleurs dataset."""
    cache = Path.home() / ".cache" / "huggingface" / "hub" / "datasets--google--fleurs" / "snapshots"
    revs = sorted(cache.iterdir()) if cache.is_dir() else []
    if not revs:
        raise SystemExit(
            "FLEURS is not in the HF cache. Download the test split first, e.g.:\n"
            "  python -c \"from huggingface_hub import hf_hub_download; \"\n"
            "  \"[hf_hub_download('google/fleurs', f) for f in \"\n"
            "  \"['data/ha_ng/test.tsv', 'data/ha_ng/audio/test.tar.gz']]\""
        )
    return revs[0]


def normalize(text: str) -> list[str]:
    """Lowercase, drop diacritics (NFD + strip combining marks) and punctuation."""
    decomposed = unicodedata.normalize("NFD", text)
    plain = "".join(c for c in decomposed if unicodedata.category(c) != "Mn")
    plain = _PUNCT_RE.sub(" ", plain.lower())
    return plain.split()


def word_errors(ref: list[str], hyp: list[str]) -> tuple[int, int, int, int]:
    """Return (substitutions, deletions, insertions, ref_words) via DP backtrace."""
    n, m = len(ref), len(hyp)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = min(
                    dp[i - 1][j - 1] + 1,  # substitution
                    dp[i - 1][j] + 1,      # deletion
                    dp[i][j - 1] + 1,      # insertion
                )
    subs = dels = ins = 0
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref[i - 1] == hyp[j - 1] and dp[i][j] == dp[i - 1][j - 1]:
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            subs += 1
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            dels += 1
            i -= 1
        else:
            ins += 1
            j -= 1
    return subs, dels, ins, n


def load_ground_truth(tsv_path: Path) -> dict[str, list[str]]:
    """Map wav filename -> reference word list from a FLEURS test.tsv.

    Columns (no header): id, file_name, raw_transcription, transcription,
    chars, num_samples, gender.  We use the normalized `transcription`
    column and re-normalize it ourselves for a diacritic-insensitive match.
    """
    truth: dict[str, list[str]] = {}
    with tsv_path.open(encoding="utf-8", newline="") as fh:
        for row in csv.reader(fh, delimiter="\t"):
            if len(row) < 4:
                continue
            truth[row[1]] = normalize(row[3])
    return truth


def extract_clips(tar_path: Path, wanted: set[str], dest: Path) -> dict[str, Path]:
    """Extract the wav files whose basename is in `wanted` from the tarball."""
    found: dict[str, Path] = {}
    with tarfile.open(tar_path, "r:gz") as tar:
        for member in tar:
            if not member.isfile():
                continue
            base = os.path.basename(member.name)
            if base in wanted and base not in found:
                member.name = base  # flatten path inside dest
                tar.extract(member, dest, filter="data")
                found[base] = dest / base
    return found


def validate_language(lang: str, cfg: str, clips: int, snapshot: Path, dest: Path) -> dict:
    from app.asr import transcribe  # noqa: PLC0415  (heavy imports deferred inside)

    tsv = snapshot / "data" / cfg / "test.tsv"
    tar_path = snapshot / "data" / cfg / "audio" / "test.tar.gz"
    if not tsv.is_file() or not tar_path.is_file():
        print(f"[{lang}] SKIP -- {tsv if not tsv.is_file() else tar_path} not cached", flush=True)
        return {"skipped": True}

    truth = load_ground_truth(tsv)
    # Take the first `clips` filenames that appear in the TSV (stable order).
    wanted = dict(list(truth.items())[:clips])
    wavs = extract_clips(tar_path, set(wanted), dest)

    results = []
    total_s = total_d = total_i = total_n = 0
    for fname, ref in wanted.items():
        wav = wavs.get(fname)
        if wav is None:
            print(f"[{lang}] {fname}: MISSING in tarball", flush=True)
            continue
        hyp_text = transcribe(str(wav), lang)
        hyp = normalize(hyp_text)
        s, d, ins, n = word_errors(ref, hyp)
        wer = (s + d + ins) / n if n else 0.0
        total_s += s
        total_d += d
        total_i += ins
        total_n += n
        print(f"[{lang}] {fname}: WER={wer:.3f} ({s}S {d}D {ins}I / {n}w)", flush=True)
        results.append({
            "file": fname,
            "wer": round(wer, 4),
            "ref": " ".join(ref),
            "hyp": hyp_text,
        })

    corpus_wer = (total_s + total_d + total_i) / total_n if total_n else None
    print(
        f"[{lang}] corpus WER over {len(results)} clips "
        f"({total_n} ref words): {corpus_wer:.3f}" if corpus_wer is not None
        else f"[{lang}] no clips scored",
        flush=True,
    )
    return {
        "fleurs_config": cfg,
        "clips_scored": len(results),
        "ref_words": total_n,
        "corpus_wer": round(corpus_wer, 4) if corpus_wer is not None else None,
        "clips": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate Veyra ASR word error rate on FLEURS test audio."
    )
    parser.add_argument(
        "--clips", type=int, default=10,
        help="Number of FLEURS test clips per language (default: 10).",
    )
    parser.add_argument(
        "--langs", nargs="+", default=list(FLEURS_LANGS),
        choices=sorted(FLEURS_LANGS), metavar="LANG",
        help="Languages to validate (default: ha ig yo en).",
    )
    parser.add_argument(
        "--out", type=str, default=None,
        help="Optional path to write full per-clip JSON results.",
    )
    args = parser.parse_args()

    # Hausa/Igbo/Yoruba transcripts contain chars outside cp1252; Windows
    # consoles default to cp1252 and crash on print without this.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    project_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(project_root))

    snapshot = fleurs_snapshot()
    print(f"FLEURS snapshot: {snapshot}", flush=True)
    print(
        "NOTE: `pcm` (Pidgin) is reported N/A -- FLEURS/Common Voice ship no "
        "Pidgin audio; it needs real recorded Pidgin voice notes.\n", flush=True,
    )

    summary = {}
    with tempfile.TemporaryDirectory(prefix="veyra_wer_") as tmp:
        dest = Path(tmp)
        for lang in args.langs:
            print(f"\n=== {lang} ({FLEURS_LANGS[lang]}) ===", flush=True)
            summary[lang] = validate_language(lang, FLEURS_LANGS[lang], args.clips, snapshot, dest)

    summary["pcm"] = {
        "skipped": True,
        "reason": "No Pidgin config in FLEURS or Common Voice; the -l pcm route is "
                  "backed by NCAIR1/NigerianAccentedEnglish and must be validated "
                  "with real recorded Pidgin voice notes.",
    }

    print("\n===== WER SUMMARY (FLEURS test, first {} clips/lang) =====".format(args.clips))
    for lang, res in summary.items():
        if res.get("skipped"):
            why = res.get("reason", "data not cached")
            print(f"  {lang:>4}: N/A -- {why}")
        else:
            print(f"  {lang:>4}: corpus WER {res['corpus_wer']:.3f} "
                  f"({res['clips_scored']} clips, {res['ref_words']} ref words)")

    if args.out:
        out_path = Path(args.out).resolve()
        out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nFull results written to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
