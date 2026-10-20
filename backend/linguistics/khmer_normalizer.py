from __future__ import annotations

import re
import unicodedata

# Preserve \u200b (Zero-Width Space) as a word boundary delimiter
_ZERO_WIDTH_RE = re.compile(r"[\u200c\u200d\ufeff]")
_VOWEL_BEFORE_COENG_RE = re.compile(r"([\u17B6-\u17CA]+)((?:\u17D2[\u1780-\u17A2])+)")
_VOWEL_BEFORE_SHIFTER_RE = re.compile(r"([\u17B6-\u17C8]+)([\u17C9\u17CA])")

# Canonical mapping of Khmer independent vowels (\u17A3..\u17B3).
#
# Every independent vowel is canonicalized to the glottal onset \u17A2 plus a
# dependent vowel sign. For U+17A3..\u17AA and U+17AF..\u17B3 that is a plain
# consonant-plus-vowel spelling; for U+17AB..\u17AE (RY / RYY / LY / LYY) it is
# the consonant-plus-vowel pair \u179A+\u17B9, \u179A+\u17BA, \u179B+\u17B9 and
# \u179B+\u17BA.
#
# v10.7 P1-A: U+17AB..\u17AE previously mapped to THEMSELVES, and
# _SYLLABLE_SCAN_RE in syllable_decomposer.py begins at U+1780, so a
# self-mapping made the decomposer skip these four vowels entirely. A word such
# as "\u17AB\u179F\u17D2\u179F" lost its first syllable, and a standalone
# "\u17AE" fell back to the default onset id 1 (\u1780). Each of the four is
# now canonicalized to a consonant plus a dependent vowel, which is both
# phonologically faithful and reachable by the scanner.
#
# U+17A8 (QUK) additionally carried the CLOSED coda \u1782 instead of the open
# (vowel-final) \u1780, which mis-derived its coda_id.
_INDEPENDENT_VOWEL_MAP = {
    "\u17A3": "\u17A2",
    "\u17A4": "\u17A2\u17B6",
    "\u17A5": "\u17A2\u17B7",
    "\u17A6": "\u17A2\u17B8",
    "\u17A7": "\u17A2\u17BB",
    "\u17A8": "\u17A2\u17BB\u1780",
    "\u17A9": "\u17A2\u17BC",
    "\u17AA": "\u17A2\u17BC\u179C",
    "\u17AB": "\u179A\u17B9",
    "\u17AC": "\u179A\u17BA",
    "\u17AD": "\u179B\u17B9",
    "\u17AE": "\u179B\u17BA",
    "\u17AF": "\u17A2\u17C2",
    "\u17B0": "\u17A2\u17C3",
    "\u17B1": "\u17A2\u17C4",
    "\u17B2": "\u17A2\u17C4\u179C",
    "\u17B3": "\u17A2\u17C5",
}

try:
    import khmer_linguistics_rs as _rs
    _HAS_RUST = hasattr(_rs, "normalize_khmer")
except Exception:
    _rs = None
    _HAS_RUST = False


def normalize_khmer(text: str) -> str:
    """NFC normalize, canonicalize independent vowels, preserve \u200b, and enforce Base->Coeng->Shifter->Vowel order."""
    if _HAS_RUST:
        return _rs.normalize_khmer(text)
    out = unicodedata.normalize("NFC", text)
    out = _ZERO_WIDTH_RE.sub("", out)
    for k, v in _INDEPENDENT_VOWEL_MAP.items():
        if k in out:
            out = out.replace(k, v)
    for _ in range(3):
        out = _VOWEL_BEFORE_COENG_RE.sub(r"\2\1", out)
        out = _VOWEL_BEFORE_SHIFTER_RE.sub(r"\2\1", out)
    return out
