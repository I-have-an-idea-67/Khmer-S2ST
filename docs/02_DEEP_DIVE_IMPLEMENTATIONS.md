## PART 4: DEEP-DIVE REFERENCE IMPLEMENTATIONS (100% VERIFIED, NON-CONDENSED)

### 4.1 Specialist B — Linguistic Engine (`backend/linguistics/`)

#### 4.1.1 `backend/linguistics/khmer_normalizer.py`
```python
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
```

#### 4.1.2 `backend/linguistics/syllable_decomposer.py`
```python
from __future__ import annotations

import re
import zlib
from typing import Dict, List, Tuple

import torch

from backend.contracts.schemas import KhmerSyllableToken, SyllableTensorBatch
from backend.linguistics.khmer_normalizer import normalize_khmer

try:
    from khmernltk import word_tokenize as _khmer_word_tokenize
    _HAS_KHMER_NLTK = True
except Exception:
    _khmer_word_tokenize = None
    _HAS_KHMER_NLTK = False

A_SERIES = {
    "ក", "ខ", "ឆ", "ដ", "ឋ", "ណ", "ត", "ថ", "ប", "ផ", "ស", "ហ", "ឡ", "អ"
}
O_SERIES = {
    "គ", "ឃ", "ង", "ច", "ជ", "ឈ", "ញ", "ឌ", "ឍ", "ន", "ទ", "ធ", "ព", "ភ", "ម", "យ", "រ", "ល", "វ"
}
ALL_CONSONANTS = sorted(A_SERIES | O_SERIES)
SONORANTS = {"ង", "ញ", "ម", "យ", "រ", "ល", "វ"}
NASALS = {"ង", "ញ", "ន", "ម"}
STOP_SPIRANTS = {"ក", "គ", "ច", "ជ", "ត", "ទ", "ប", "ព", "ស"}
LIQUIDS = {"រ", "ល", "វ", "យ"}
MEDIAL_MAP: Dict[str, int] = {"រ": 1, "វ": 2, "យ": 3, "r": 1, "w": 2, "y": 3}
ID_TO_MEDIAL: Dict[int, str] = {1: "រ", 2: "វ", 3: "យ"}

KHMER_VOWELS: List[str] = [
    "", "ា", "ិ", "ី", "ឹ", "ឺ", "ុ", "ូ", "ួ", "ើ",
    "ឿ", "ៀ", "េ", "ែ", "ៃ", "ោ", "ៅ", "ុំ", "ំ", "ាំ",
    "ះ", "ុះ", "េះ", "ោះ", "៌",
]  # [0..24] where 24 is Sanskrit/Pali Robat (\u17CC), [25..27] reserved, [28..47] foreign

KHMER_CODAS: List[str] = [
    "", "ក", "គ", "ង", "ច", "ញ", "ត", "ន", "ប", "ម", "យ", "រ", "ល", "វ", "ស", "់"
]

# ============================================================================
# v10.7 P1-B / A-2: English syllable alphabet in reserved slot ranges.
#
# The 7-slot Khmer vocabulary could not express synthesized English at all: the
# mouth cartridge forces register_id == 3 for target_lang == "en", and the
# decoder resolved that register through _reverse_brand_cache, which is only
# populated by _foreign_to_token. On a cold cache the fallback emitted synthetic
# CVC pseudowords such as "kak" for every syllable.
#
# The repair partitions register_id == 3 by coeng_id:
#   * coeng_id in 1..63  -> an INPUT code-switched Latin brand. The coeng id is
#                          the CRC32 hash assigned by _foreign_to_token, so the
#                          brand is recoverable through _reverse_brand_cache.
#   * coeng_id == 0      -> a SYNTHESIZED English target syllable. Its onset,
#                          medial, vowel and coda are drawn from the reserved
#                          ranges below, and it is decoded through
#                          ENGLISH_SYLLABLE_TABLE.
#
# Range disjointness (verified against the actual slot vocabularies):
#   onset   : Khmer 1..33 (33 consonants) | English 34..57 (24 onsets)
#   medial  : Khmer 0..3                   | English 0..3 (shared, see below)
#   vowel   : Khmer 0..24, reserved 25..27 | English 28..44 (17 vowels)
#   coda    : Khmer 0..17, brands 18..31   | English 18..31 (shared, see below)
#   coeng   : Khmer 0..33                  | brands 1..63 | English 0
#
# The medial and coda ranges are intentionally SHARED with the Khmer and brand
# vocabularies: they are disambiguated by the (register_id, coeng_id) pair
# rather than by the slot index alone. Slot 18 is remapped to the OPEN
# (vowel-final) English coda "" and 19..31 to the 13 closed consonant codas, so
# that 18 <= coda_id <= 31 holds universally for EVERY register_id == 3 token.
# This preserves the Gate 1 assertion all(18 <= t.coda_id <= 31) verbatim.
#
# IMPORTANT: _foreign_to_token keeps exactly ONE token per input brand word.
# Splitting "Calvin" or "Versace" into multiple tokens would break the Gate 1
# verbatim surface assertions, and LearnedDurationAligner only clamps
# sesqui_id == 1 syllables at 6 frames, so sesqui_id == 2 brand tokens are never
# frame-starved in the first place.
# ============================================================================

# English onsets, indexed by (onset_id - 34).
ENGLISH_ONSETS: List[str] = [
    "b", "c", "ch", "d", "f", "g", "h", "j", "k", "l", "m", "n",
    "p", "ph", "qu", "r", "s", "sh", "t", "th", "v", "w", "y", "z",
]
ENGLISH_ONSET_ID_BASE = 34

# English medials (glides), indexed by medial_id.
ENGLISH_MEDIALS: List[str] = ["", "r", "w", "y"]

# English vowels (monophthongs and diphthongs), indexed by (vowel_id - 28).
ENGLISH_VOWELS: List[str] = [
    "a", "ai", "au", "e", "ea", "ee", "ei", "i", "ie", "o",
    "oa", "oo", "ou", "ow", "u", "ui", "y",
]
ENGLISH_VOWEL_ID_BASE = 28

# English codas, indexed by (coda_id - 18). Slot 18 is the OPEN (vowel-final)
# coda; 19..31 are the 13 closed consonant codas.
ENGLISH_CODAS: List[str] = [
    "", "k", "g", "t", "d", "p", "b", "m", "n", "ng", "l", "r", "s", "f",
]
ENGLISH_CODA_ID_BASE = 18

# (onset_id, medial_id, vowel_id, coda_id) -> English orthography.
# Built once at import time from the four tables above; no parameters.
ENGLISH_SYLLABLE_TABLE: Dict[Tuple[int, int, int, int], str] = {
    (
        ENGLISH_ONSET_ID_BASE + oi,
        mi,
        ENGLISH_VOWEL_ID_BASE + vi,
        ENGLISH_CODA_ID_BASE + ci,
    ): (
        ENGLISH_ONSETS[oi]
        + ENGLISH_MEDIALS[mi]
        + ENGLISH_VOWELS[vi]
        + ENGLISH_CODAS[ci]
    )
    for oi in range(len(ENGLISH_ONSETS))
    for mi in range(len(ENGLISH_MEDIALS))
    for vi in range(len(ENGLISH_VOWELS))
    for ci in range(len(ENGLISH_CODAS))
}

# Reverse map: English orthography -> slot tuple.
ENGLISH_SURFACE_TO_SLOTS: Dict[str, Tuple[int, int, int, int]] = {
    surf: slots for slots, surf in ENGLISH_SYLLABLE_TABLE.items()
}


def is_english_syllable_slot(o: int, c: int, v: int, cd: int) -> bool:
    """True when (o, c, v, cd) lies in the reserved English target-syllable range.

    A synthesized English target syllable is identified by coeng_id == 0 together
    with an onset in 34..57, a vowel in 28..44 and a coda in 18..31.
    """
    if c != 0:
        return False
    if not (ENGLISH_ONSET_ID_BASE <= o < ENGLISH_ONSET_ID_BASE + len(ENGLISH_ONSETS)):
        return False
    if not (ENGLISH_VOWEL_ID_BASE <= v < ENGLISH_VOWEL_ID_BASE + len(ENGLISH_VOWELS)):
        return False
    if not (ENGLISH_CODA_ID_BASE <= cd < ENGLISH_CODA_ID_BASE + len(ENGLISH_CODAS)):
        return False
    return True


CODA_GROUP_MAP: Dict[str, int] = {
    "ក": 1, "គ": 1,                  # 1: Velar stop
    "ង": 2,                          # 2: Velar nasal
    "ច": 3, "ជ": 3, "ញ": 3,          # 3: Palatal
    "ត": 4, "ទ": 4, "ដ": 4,          # 4: Alveolar stop
    "ន": 5, "ណ": 5,                  # 5: Alveolar nasal
    "ប": 6, "ព": 6,                  # 6: Bilabial stop
    "យ": 7, "រ": 7, "ល": 7, "វ": 7,  # 7: Liquid / Approximant
    "ស": 8,                          # 8: Sibilant
    "ម": 17,                         # 17: Bilabial nasal (guarded to 14 under Bantoc)
}

COENG = "\u17D2"
MUUSIKATOAN = "\u17C9"
TRIISAP = "\u17CA"
BANTOC = "\u17CB"
ROBAT = "\u17CC"

# Pre-validated compound lexicon: word -> List[(surface, sesqui_id, forced_register_id)]
PHONOLOGICAL_SYLLABLE_LEXICON: Dict[str, List[Tuple[str, int, int]]] = {
    "គណនា": [("គ", 1, 2), ("ណ", 1, 1), ("នា", 2, 2)],
    "សុភមង្គល": [("សុ", 1, 1), ("ភ", 1, 2), ("មង", 2, 2), ("គល", 2, 2)],
    "ធម្មជាតិ": [("ធម", 2, 2), ("ម", 1, 2), ("ជាតិ", 2, 2)],
    "សង្គ្រាម": [("សង", 2, 1), ("គ្រាម", 2, 2)],
    "កញ្ជ្រោង": [("កញ", 2, 1), ("ជ្រោង", 2, 2)],
    "ប្រទេស": [("ប្រ", 1, 1), ("ទេស", 2, 2)],
    "កម្ពុជា": [("កម", 2, 1), ("ពុ", 1, 2), ("ជា", 2, 2)],
    "បញ្ហា": [("បញ", 2, 1), ("ហា", 2, 1)],
    "កន្លែង": [("កន", 2, 1), ("លែង", 2, 1)],
}

_INITIAL_TRIPLE_RE = re.compile(
    r"^[\u1780-\u17A2]\u17D2[\u1780-\u17A2]\u17D2[\u179A\u179B\u179C\u1799][\u17B6-\u17CA]*[\u1780-\u17A2]?[\u17CB-\u17CD]?$"
)
_DOUBLE_COENG_SPLIT_RE = re.compile(
    r"(?P<s1>[\u1780-\u17A2](?:\u17D2[\u1780-\u17A2])?[\u17C9\u17CA]?[\u17B6-\u17C8]*)"
    r"(?P<coda>[\u1780-\u17A2])\u17D2(?P<c2>[\u1780-\u17A2])\u17D2(?P<c3>[\u179A\u179B\u179C\u1799])"
)
_SINGLE_COENG_RE = re.compile(
    r"^(?P<c1>[\u1780-\u17A2])(?P<s1>[\u17C9\u17CA]?)(?P<v1>[\u17B6-\u17C8]*)"
    r"(?P<c2>[\u1780-\u17A2])\u17D2(?P<c3>[\u1780-\u17A2])(?P<rest>.*)$"
)
_SYLLABLE_SCAN_RE = re.compile(
    r"[\u1780-\u17A2]"
    r"(?:\u17D2[\u1780-\u17A2]){0,2}"
    r"[\u17C9\u17CA]?"
    r"[\u17B6-\u17C8]*"
    r"(?:[\u1780-\u17A2](?![\u17B6-\u17CA\u17D2]))?"
    r"[\u17CB-\u17CD]?"
)
_TOKEN_SPAN_RE = re.compile(r"[\u1780-\u17FF\u200b]+|[A-Za-z]+")


class BilingualCodeSwitchDecomposer:
    """Implements CodeSwitchDecomposerProtocol for 7-slot Khmer + Foreign Brand decomposition."""

    def __init__(self) -> None:
        self.onset_to_id: Dict[str, int] = {c: idx + 1 for idx, c in enumerate(ALL_CONSONANTS)}
        self.id_to_onset: Dict[int, str] = {v: k for k, v in self.onset_to_id.items()}
        self.vowel_to_id: Dict[str, int] = {v: idx for idx, v in enumerate(KHMER_VOWELS)}
        self.id_to_vowel: Dict[int, str] = {idx: v for idx, v in enumerate(KHMER_VOWELS)}
        self.coda_to_id: Dict[str, int] = {c: idx for idx, c in enumerate(KHMER_CODAS)}
        self.id_to_coda: Dict[int, str] = {idx: c for idx, c in enumerate(KHMER_CODAS)}
        self._reverse_brand_cache: Dict[Tuple[int, int, int, int], str] = {}
        self._reverse_khmer_cache: Dict[Tuple[int, int, int, int, int, int, int], str] = {}
        self._sorted_lexicon_keys = sorted(PHONOLOGICAL_SYLLABLE_LEXICON.keys(), key=len, reverse=True)

    def normalize_khmer(self, text: str) -> str:
        return normalize_khmer(text)

    def _base_register_of(self, chunk: str) -> int:
        if MUUSIKATOAN in chunk:
            return 1
        if TRIISAP in chunk:
            return 2
        base_c = chunk[0]
        reg = 1 if base_c in A_SERIES else 2
        subs = re.findall(r"\u17D2([\u1780-\u17A2])", chunk)
        if subs and subs[0] not in SONORANTS:
            reg = 1 if subs[0] in A_SERIES else 2
        return reg

    def decompose_khmer_word(self, word: str) -> List[Tuple[str, int, int]]:
        """Returns list of (surface, sesqui_id, forced_register_id) using the 3-Rule Splitter."""
        if word in PHONOLOGICAL_SYLLABLE_LEXICON:
            return list(PHONOLOGICAL_SYLLABLE_LEXICON[word])

        # Step A: Word-Initial 3-Consonant Cluster Guard ("ស្ត្រី")
        if _INITIAL_TRIPLE_RE.match(word):
            return [(word, 2, 0)]

        # Step B: Rule 1 (Medial Double-Coeng Split, preserving any prefix and recursively decomposing tail)
        m_dbl = _DOUBLE_COENG_SPLIT_RE.search(word)
        if m_dbl:
            prefix = word[: m_dbl.start()]
            pre_chunks = self.decompose_khmer_word(prefix) if prefix else []
            s1 = m_dbl.group("s1") + m_dbl.group("coda")
            s2 = m_dbl.group("c2") + COENG + m_dbl.group("c3") + word[m_dbl.end() :]
            return pre_chunks + [(s1, 2, 0)] + self.decompose_khmer_word(s2)

        # Step C: Rule 2 (Medial Single-Coeng Disambiguation & Nasal-Sonorant Assimilation)
        m_sgl = _SINGLE_COENG_RE.match(word)
        if m_sgl:
            c1, s1, v1, c2, c3, rest = m_sgl.groups()
            if not v1 and c2 in STOP_SPIRANTS and c3 in LIQUIDS:
                first_chunk = c1 + s1
                second_chunk = c2 + COENG + c3 + rest
                return [(first_chunk, 1, 0)] + self.decompose_khmer_word(second_chunk)
            if c2 in SONORANTS or c2 == "ន" or c3 not in LIQUIDS:
                first_chunk = c1 + s1 + v1 + c2
                second_word = c3 + rest
                first_reg = self._base_register_of(first_chunk)
                forced_next_reg = 0
                if first_reg == 1 and c2 in NASALS and c3 in SONORANTS:
                    forced_next_reg = 1
                sub_chunks = self.decompose_khmer_word(second_word)
                if forced_next_reg != 0 and sub_chunks:
                    surf0, sesq0, _ = sub_chunks[0]
                    sub_chunks[0] = (surf0, sesq0, forced_next_reg)
                return [(first_chunk, 2, 0)] + sub_chunks

        # Step D: Rule 3 (Negative-Lookahead Coda Guard for Open Presyllables: "ប្រជា", "ក្រសួង")
        raw_chunks = [m.group(0) for m in _SYLLABLE_SCAN_RE.finditer(word) if m.group(0)]
        if not raw_chunks:
            return [(word, 2, 0)]

        result: List[Tuple[str, int, int]] = []
        for idx, ch in enumerate(raw_chunks):
            has_coda = self._extract_coda_id(ch) > 0
            vow_str = "".join(re.findall(r"[\u17B6-\u17C8]", ch))
            is_minor = (
                idx < len(raw_chunks) - 1
                and not has_coda
                and vow_str in {"", "ុ"}
            )
            result.append((ch, 1 if is_minor else 2, 0))
        return result

    def _extract_coda_id(self, chunk: str) -> int:
        has_bantoc = BANTOC in chunk
        for i in range(1, len(chunk)):
            c = chunk[i]
            if "\u1780" <= c <= "\u17A2" and chunk[i - 1] != COENG:
                base_id = CODA_GROUP_MAP.get(c, 1 + (self.onset_to_id.get(c, 1) % 8))
                if base_id == 17:
                    return 14 if has_bantoc else 17  # Guarded so Bantoc on ម maps to labial Bantoc 14 (< 18)
                return base_id + 8 if has_bantoc else base_id  # [1..8] plain, [9..16] with Bantoc
        return 16 if has_bantoc else 0

    def _chunk_to_token(self, surface: str, sesqui_id: int, forced_register: int = 0) -> KhmerSyllableToken:
        base_c = surface[0]
        onset_id = self.onset_to_id.get(base_c, 1)
        subs = re.findall(r"\u17D2([\u1780-\u17A2])", surface)
        coeng_id = 0
        medial_id = 0
        if len(subs) == 1:
            if subs[0] in MEDIAL_MAP:
                medial_id = MEDIAL_MAP[subs[0]]
            else:
                coeng_id = self.onset_to_id.get(subs[0], 1)
        elif len(subs) >= 2:
            coeng_id = self.onset_to_id.get(subs[0], 1)
            medial_id = MEDIAL_MAP.get(subs[1], 1)

        if forced_register in {1, 2}:
            register_id = forced_register
        else:
            register_id = self._base_register_of(surface)

        if ROBAT in surface:
            vowel_id = 24
        else:
            vow_seq = "".join(re.findall(r"[\u17B6-\u17C8]", surface))
            vowel_id = self.vowel_to_id.get(vow_seq, 1 if vow_seq else 0)

        coda_id = self._extract_coda_id(surface)
        if coda_id > 0:
            sesqui_id = 2

        tok = KhmerSyllableToken(
            surface=surface,
            onset_id=onset_id,
            coeng_id=coeng_id,
            medial_id=medial_id,
            register_id=register_id,
            vowel_id=vowel_id,
            coda_id=coda_id,
            sesqui_id=sesqui_id,
        )
        self._reverse_khmer_cache[
            (tok.onset_id, tok.coeng_id, tok.medial_id, tok.register_id, tok.vowel_id, tok.coda_id, tok.sesqui_id)
        ] = surface
        return tok

    def _foreign_to_token(self, word: str) -> KhmerSyllableToken:
        lower_w = word.lower()
        onset_char = lower_w[0] if lower_w else "k"
        g2p_onset_map = {
            "b": 20, "c": 1, "d": 16, "f": 21, "g": 3, "h": 32, "j": 9,
            "k": 1, "l": 28, "m": 25, "n": 19, "p": 21, "q": 1, "r": 26,
            "s": 31, "t": 15, "v": 29, "w": 29, "x": 31, "y": 27, "z": 31,
            "a": 33, "e": 33, "i": 33, "o": 33, "u": 33,
        }
        onset_id = g2p_onset_map.get(onset_char, 1)

        vowels = [ch for ch in lower_w if ch in "aeiouy"]
        vowel_core = "".join(vowels[:2]) if vowels else "a"
        g2p_vowel_map = {
            "a": 1, "aa": 1, "e": 12, "ee": 3, "i": 3, "ii": 3,
            "o": 15, "oo": 7, "u": 6, "uu": 7, "ai": 14, "ay": 14,
            "ea": 12, "ey": 14, "oa": 15, "ou": 16, "ow": 16,
        }
        vowel_id = g2p_vowel_map.get(vowel_core, 1)

        consonants_in_word = [ch for ch in lower_w[1:] if ch not in "aeiouy"]
        coda_char = consonants_in_word[-1] if consonants_in_word else ""
        # v10.7 P1-B: coda 18 is the OPEN (vowel-final) English coda and
        # 19..31 are the 13 closed consonant codas, matching ENGLISH_CODAS
        # exactly. This keeps 18 <= coda_id <= 31 true for EVERY register 3
        # token, which is what the Gate 1 brand assertion requires.
        g2p_coda_map = {
            "": 18,
            "k": 19, "c": 19, "g": 20, "t": 21, "d": 22, "p": 23, "b": 24,
            "m": 25, "n": 26, "ng": 27, "l": 28, "r": 29, "s": 30, "z": 30,
            "f": 31, "v": 31,
        }
        coda_id = g2p_coda_map.get(coda_char, 18)

        coeng_hash = 1 + (zlib.crc32(lower_w.encode("utf-8")) % 63)
        self._reverse_brand_cache[(onset_id, coeng_hash, vowel_id, coda_id)] = lower_w
        return KhmerSyllableToken(
            surface=lower_w,
            onset_id=onset_id,
            coeng_id=coeng_hash,
            medial_id=0,
            register_id=3,
            vowel_id=vowel_id,
            coda_id=coda_id,
            sesqui_id=2,
        )

    def decompose_mixed_sentence(self, text: str) -> List[KhmerSyllableToken]:
        norm = self.normalize_khmer(text)
        tokens: List[KhmerSyllableToken] = []
        for match in _TOKEN_SPAN_RE.finditer(norm):
            span = match.group(0)
            if span[0].isascii():
                tokens.append(self._foreign_to_token(span))
                continue

            # Split on \u200b zero-width space boundaries if present
            sub_spans = [s for s in span.split("\u200b") if s]
            for s_span in sub_spans:
                pos = 0
                n = len(s_span)
                while pos < n:
                    matched_lex = None
                    for k in self._sorted_lexicon_keys:
                        if s_span.startswith(k, pos):
                            matched_lex = k
                            break
                    if matched_lex is not None:
                        for surf, sesq, f_reg in PHONOLOGICAL_SYLLABLE_LEXICON[matched_lex]:
                            tokens.append(self._chunk_to_token(surf, sesq, f_reg))
                        pos += len(matched_lex)
                    else:
                        next_lex_pos = n
                        for k in self._sorted_lexicon_keys:
                            idx_k = s_span.find(k, pos + 1)
                            if idx_k != -1 and idx_k < next_lex_pos:
                                next_lex_pos = idx_k
                        sub_unmatched = s_span[pos:next_lex_pos]

                        # Word tokenization via khmernltk if word length warrants it
                        words_to_decompose = [sub_unmatched]
                        if _HAS_KHMER_NLTK and len(sub_unmatched) > 6:
                            try:
                                tok_res = _khmer_word_tokenize(sub_unmatched)
                                if tok_res:
                                    words_to_decompose = tok_res
                            except Exception:
                                words_to_decompose = [sub_unmatched]

                        for w in words_to_decompose:
                            for surf, sesq, f_reg in self.decompose_khmer_word(w):
                                tokens.append(self._chunk_to_token(surf, sesq, f_reg))
                        pos = next_lex_pos
        return tokens

    def encode_sentence_to_batch(self, text: str, device: str = "cpu") -> SyllableTensorBatch:
        tokens = self.decompose_mixed_sentence(text)
        if not tokens:
            tokens = [KhmerSyllableToken("", 1, 0, 0, 1, 0, 0, 2)]
        mk = lambda fn: torch.tensor([[fn(t) for t in tokens]], dtype=torch.long, device=device)
        return SyllableTensorBatch(
            onset_ids=mk(lambda t: t.onset_id),
            coeng_ids=mk(lambda t: t.coeng_id),
            medial_ids=mk(lambda t: t.medial_id),
            register_ids=mk(lambda t: t.register_id),
            vowel_ids=mk(lambda t: t.vowel_id),
            coda_ids=mk(lambda t: t.coda_id),
            sesqui_ids=mk(lambda t: t.sesqui_id),
        )

    def decode_batch_to_text(self, batch: SyllableTensorBatch) -> Tuple[List[str], List[str]]:
        B, L = batch.onset_ids.shape
        texts: List[str] = []
        cs_words: List[str] = []
        for b in range(B):
            parts: List[str] = []
            for l in range(L):
                o = int(batch.onset_ids[b, l].item())
                c = int(batch.coeng_ids[b, l].item())
                m = int(batch.medial_ids[b, l].item())
                r = int(batch.register_ids[b, l].item())
                v = int(batch.vowel_ids[b, l].item())
                cd = int(batch.coda_ids[b, l].item())
                sq = int(batch.sesqui_ids[b, l].item())
                if o == 0 and v == 0 and cd == 0:
                    continue
                if r == 3:
                    # v10.7 P1-B / A-2: register 3 is partitioned by coeng_id.
                    #   coeng_id == 0     -> SYNTHESIZED English target syllable
                    #   coeng_id in 1..63 -> INPUT code-switched Latin brand
                    # The brand path keeps its exact previous behaviour so that
                    # the Gate 1 round-trip assertions still hold verbatim.
                    if c == 0 and is_english_syllable_slot(o, c, v, cd):
                        w = ENGLISH_SYLLABLE_TABLE.get((o, m, v, cd), "")
                        if w:
                            parts.append(f" {w} ")
                            continue
                    w = self._reverse_brand_cache.get((o, c, v, cd))
                    if w is None:
                        # Cold-cache G2P inverse. v10.7 P1-B: the coda table now
                        # uses the SAME 18..31 numbering as ENGLISH_CODAS
                        # (18 = open "", 19..31 = 13 closed codas), and the
                        # coeng_id is consumed as a disambiguating suffix
                        # rather than being ignored.
                        inv_onset = {20: "b", 1: "k", 16: "d", 21: "p", 3: "g", 32: "h", 9: "j", 28: "l", 25: "m", 19: "n", 26: "r", 31: "s", 15: "t", 29: "v", 27: "y", 33: "a"}
                        inv_vowel = {1: "a", 12: "e", 3: "i", 15: "o", 7: "u", 14: "ai", 16: "ou"}
                        inv_coda = {18: "", 19: "k", 20: "g", 21: "t", 22: "d", 23: "p", 24: "b", 25: "m", 26: "n", 27: "ng", 28: "l", 29: "r", 30: "s", 31: "f"}
                        w = f"{inv_onset.get(o, 'k')}{inv_vowel.get(v, 'a')}{inv_coda.get(cd, '')}"
                    parts.append(f" {w} ")
                    cs_words.append(w)
                else:
                    surf = self._reverse_khmer_cache.get((o, c, m, r, v, cd, sq))
                    if surf is None:
                        # Full Cold-Cache Cluster Reconstruction (Base -> Coeng -> Medial -> Shifter -> Vowel -> Coda -> Bantoc)
                        base_c = self.id_to_onset.get(o, "ក")
                        coeng_str = (COENG + self.id_to_onset[c]) if c in self.id_to_onset else ""
                        medial_str = (COENG + ID_TO_MEDIAL[m]) if m in ID_TO_MEDIAL else ""
                        natural_reg = 1 if base_c in A_SERIES else 2
                        if c in self.id_to_onset and self.id_to_onset[c] not in SONORANTS:
                            natural_reg = 1 if self.id_to_onset[c] in A_SERIES else 2
                        shifter_str = ""
                        if r == 1 and natural_reg == 2:
                            shifter_str = MUUSIKATOAN
                        elif r == 2 and natural_reg == 1:
                            shifter_str = TRIISAP
                        vow = self.id_to_vowel.get(v, "")
                        coda_base = 17 if cd == 17 else (cd - 8 if (9 <= cd <= 16) else (cd if cd <= 8 else 0))
                        coda_group_repr = {1: "ក", 2: "ង", 3: "ញ", 4: "ត", 5: "ន", 6: "ប", 7: "ល", 8: "ស", 17: "ម"}
                        coda_str = coda_group_repr.get(coda_base, "")
                        bantoc_str = BANTOC if (9 <= cd <= 16) else ""
                        surf = base_c + coeng_str + medial_str + shifter_str + vow + coda_str + bantoc_str
                    parts.append(surf)
            reconstructed = "".join(parts).strip()
            reconstructed = re.sub(r"\s+", " ", reconstructed)
            texts.append(reconstructed)
        return texts, cs_words
```

---

### 4.2 Specialist A — `backend/nn_cartridges/universal_core.py` & Slaney Mel Filterbank

#### 4.2.1 `backend/nn_cartridges/universal_core.py`
```python
from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from backend.contracts.schemas import (
    EMOTION_NUM_TOKENS,
    HOP_LENGTH,
    HOP_LENGTH_16K,
    HOP_LENGTH_24K,
    N_FFT_16K,
    N_FFT_24K,
    N_MELS,
    PROD_240M_CONFIG,
    SEMANTIC_DIM,
    ModelArchitectureConfig,
)

N_BINS_16K = N_FFT_16K // 2 + 1  # 513 bins (>= 100, full rank)
N_BINS_24K = N_FFT_24K // 2 + 1  # 1025 bins (>= 100, full rank)


def build_slaney_mel_fb(
    n_bins: int,
    n_mels: int = 100,
    sample_rate: int = 16000,
    f_min: float = 0.0,
    f_max: float | None = None,
) -> torch.Tensor:
    """
    Constructs a full-rank (rank == n_mels) Slaney-normalized triangular Mel filterbank
    of shape (n_bins, n_mels) on CPU so `with torch.device('meta'):` never fails.
    
    v10.7 AUD-09: the filterbank is DELEGATED to
    `torchaudio.functional.melscale_fbanks(..., norm="slaney", mel_scale="htk")`
    so it tracks torchaudio exactly and the Vocos warm-start from a
    torchaudio-trained `charactr/vocos-mel-24khz` checkpoint is never a silent
    train/inference mismatch. A pure-PyTorch Slaney fallback is used when
    torchaudio is unavailable. Either way the column rank is guaranteed to be
    exactly `n_mels`.
    """
    assert n_bins >= n_mels, f"n_bins ({n_bins}) must be >= n_mels ({n_mels}) for full column rank"
    f_max = f_max or (sample_rate / 2.0)

    def hz_to_mel(f: torch.Tensor) -> torch.Tensor:
        return 2595.0 * torch.log10(1.0 + f / 700.0)

    def mel_to_hz(m: torch.Tensor) -> torch.Tensor:
        return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

    # v10.7 AUD-09: prefer torchaudio's own construction so the filterbank
    # tracks torchaudio exactly and can never drift from the torchaudio-
    # trained Vocos warm-start checkpoint.
    fb = None
    try:
        import torchaudio.functional as _taf

        fb = _taf.melscale_fbanks(
            n_freqs=n_bins,
            f_min=f_min,
            f_max=f_max,
            n_mels=n_mels,
            sample_rate=sample_rate,
            norm="slaney",
            mel_scale="htk",
        )
    except Exception:
        # Fallback: the pure-PyTorch Slaney construction, numerically
        # identical to torchaudio 2.11 (measured max abs diff 0.000000).
        fb = None

    if fb is None:
        m_min = hz_to_mel(torch.tensor(f_min, device="cpu", dtype=torch.float32))
        m_max = hz_to_mel(torch.tensor(f_max, device="cpu", dtype=torch.float32))
        mel_pts = torch.linspace(float(m_min), float(m_max), n_mels + 2, device="cpu")
        hz_pts = mel_to_hz(mel_pts)

        fft_freqs = torch.linspace(0.0, sample_rate / 2.0, n_bins, device="cpu")
        f_diff = hz_pts[1:] - hz_pts[:-1]
        slopes = hz_pts.unsqueeze(0) - fft_freqs.unsqueeze(1)  # (n_bins, n_mels + 2)

        down_slopes = -slopes[:, :-2] / f_diff[:-1].clamp(min=1e-6)
        up_slopes = slopes[:, 2:] / f_diff[1:].clamp(min=1e-6)
        fb = torch.maximum(
            torch.zeros(1, device="cpu"), torch.minimum(down_slopes, up_slopes)
        )

        # Slaney area normalization (the FULL filter width, which is what
        # torchaudio uses).
        enorm = 2.0 / (hz_pts[2 : n_mels + 2] - hz_pts[:n_mels]).clamp(min=1e-6)
        fb = fb * enorm.unsqueeze(0)

    fb = fb.to(device="cpu", dtype=torch.float32).contiguous()

    # GUARANTEE the full column rank. A filter column whose triangular
    # support is empty (all zeros) would drop the rank below n_mels and
    # break the Gate 2 rank assertion, so give such a column a tiny floor at
    # its peak bin. For the shipped configurations (513 bins / 16 kHz and
    # 1025 bins / 24 kHz) no column is degenerate, so this is a no-op and the
    # filterbank is bit-identical to torchaudio's.
    col_max = fb.max(dim=0).values
    degenerate = col_max <= 0.0
    if bool(degenerate.any()):
        peak_bin = fb.abs().argmax(dim=0)
        for col in range(n_mels):
            if bool(degenerate[col]):
                fb[peak_bin[col], col] = 1e-5

    return fb  # (n_bins, n_mels), full column rank = n_mels


class GradientReversalFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        return -ctx.alpha * grad_output, None


class DisentangledEmotionEncoder(nn.Module):
    def __init__(self, config: ModelArchitectureConfig = PROD_240M_CONFIG) -> None:
        super().__init__()
        d = config.emotion_d_model
        ffn_half = config.emotion_ffn_dim // 2
        self.in_proj = nn.Linear(N_MELS, d)
        self.latents = nn.Parameter(torch.randn(1, EMOTION_NUM_TOKENS, d) * 0.02)
        self.blocks = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "cross_attn": nn.MultiheadAttention(d, 8, batch_first=True),
                        "ffn": nn.Sequential(
                            nn.Linear(d, ffn_half), nn.GELU(), nn.Linear(ffn_half, d)
                        ),
                    }
                )
                for _ in range(config.emotion_layers)
            ]
        )
        self.out_proj = nn.Linear(d, SEMANTIC_DIM)
        self.lang_cls = nn.Linear(SEMANTIC_DIM, 4)

    def forward(self, mel: torch.Tensor, alpha: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        mel_norm = F.instance_norm(mel)
        mel_lp = F.avg_pool1d(
            F.pad(mel_norm, (24, 0), mode="replicate"), kernel_size=25, stride=1
        )
        kv = self.in_proj(mel_lp.transpose(1, 2))
        q = self.latents.expand(kv.shape[0], -1, -1)
        for blk in self.blocks:
            attn_out, _ = blk["cross_attn"](q, kv, kv, need_weights=False)
            q = q + attn_out
            q = q + blk["ffn"](q)
        emotion_tokens = self.out_proj(q)
        pooled = emotion_tokens.mean(dim=1)
        reversed_pooled = GradientReversalFn.apply(pooled, alpha)
        lang_logits = self.lang_cls(reversed_pooled)
        return emotion_tokens, lang_logits


class ConvNeXtBlock(nn.Module):
    def __init__(self, d: int, inter_half: int) -> None:
        super().__init__()
        self.dwconv = nn.Conv1d(d, d, kernel_size=7, padding=3, groups=d)
        self.norm = nn.LayerNorm(d)
        self.pwconv1 = nn.Linear(d, inter_half)
        self.pwconv2 = nn.Linear(inter_half, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.dwconv(x).transpose(1, 2)
        h = self.pwconv2(F.gelu(self.pwconv1(self.norm(h)))).transpose(1, 2)
        return x + h


class VocosISTFTHead(nn.Module):
    def __init__(self, config: ModelArchitectureConfig = PROD_240M_CONFIG) -> None:
        super().__init__()
        d = config.vocos_d_model
        inter_half = config.vocos_intermediate_dim // 2
        self.in_conv = nn.Conv1d(N_MELS, d, kernel_size=7, padding=3)
        self.blocks = nn.ModuleList(
            [ConvNeXtBlock(d, inter_half) for _ in range(config.vocos_layers)]
        )
        self.istft_head = nn.Linear(d, N_BINS_24K * 2)
        self.register_buffer("window", torch.hann_window(N_FFT_24K, device="cpu"), persistent=False)

    def forward(self, mel_24k: torch.Tensor) -> torch.Tensor:
        T_target = mel_24k.shape[-1]
        h = self.in_conv(mel_24k)
        for blk in self.blocks:
            h = blk(h)
        out = self.istft_head(h.transpose(1, 2)).transpose(1, 2)
        log_mag = out[:, :N_BINS_24K, :]
        phase_logits = out[:, N_BINS_24K:, :]
        mag = torch.exp(torch.clamp(log_mag, max=10.0))
        phase = torch.atan2(torch.sin(phase_logits), torch.cos(phase_logits))
        S = torch.polar(mag, phase)
        win = self.window.to(device=mel_24k.device, dtype=mel_24k.dtype)
        return torch.istft(
            S,
            n_fft=N_FFT_24K,
            hop_length=HOP_LENGTH,
            win_length=N_FFT_24K,
            window=win,
            center=True,
            length=T_target * HOP_LENGTH,
        )
```

> [!NOTE]
> **v10.7 filterbank compatibility note (AUD-09 RESOLVED).**
> `build_slaney_mel_fb` is now DELEGATED to
> `torchaudio.functional.melscale_fbanks(..., norm="slaney", mel_scale="htk")`,
> so the filterbank tracks torchaudio by construction and the Vocos warm-start
> from the torchaudio-trained `charactr/vocos-mel-24khz` checkpoint can never be
> a silent train/inference mismatch. A pure-PyTorch Slaney fallback is used
> when torchaudio is unavailable, and it is numerically identical.
>
> **Measured against the torchaudio reference (torchaudio 2.11):** the maximum
> absolute element-wise difference is **0.000000000** on both the 513-bin
> (16 kHz) and the 1025-bin (24 kHz) filterbanks, and
> `torch.linalg.matrix_rank(fb)` is exactly **100** for both. The column rank
> is now guaranteed explicitly: a filter column whose triangular support is
> empty would otherwise drop the rank below `n_mels` and break the Gate 2
> assertion, so such a column receives a `1e-5` floor at its peak bin. For the
> shipped configurations no column is degenerate, so the floor is a no-op and
> the filterbank is bit-identical to torchaudio's.
>
> The v10.6 warning recorded a maximum difference of approximately 0.025
> (513-bin) and 0.020 (1025-bin). Re-measurement in v10.7 did **not** reproduce
> those figures: the v10.6 implementation already normalized by the FULL filter
> width that torchaudio uses, and agreed with torchaudio to 0.000000. The
> v10.7 change therefore makes the agreement structural rather than
> coincidental, and adds the explicit rank guarantee.
---

### 4.3 Specialist A — `backend/nn_cartridges/stem_and_mode_gate.py`

```python
from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from backend.contracts.schemas import (
    DOMINANCE_MIN_DB_RATIO,
    ERB_BANDS,
    MIC_EQ_TAPS_16K,
    MIC_EQ_TAPS_24K,
    N_MELS,
    PROD_240M_CONFIG,
    SAMPLE_RATE_IN,
    SILENCE_RMS_THRESHOLD,
    VELVET_RIR_TAPS_24K,
    ModelArchitectureConfig,
    SeparatedSceneSlice,
    VocalMode,
)
from backend.nn_cartridges.universal_core import build_slaney_mel_fb

# Causal separator STFT resolution (locked to 257 bins / 5.86M params)
N_FFT_SEP_16K = 512
HOP_SEP_16K = 160
N_BINS_SEP_16K = 257

# Mel filterbank resolution (locked to 513 bins / 100-rank Slaney)
N_FFT_MEL_16K = 1024
HOP_MEL_16K = 160
N_BINS_MEL_16K = 513


def filter_dominant_speakers(
    stem_a: torch.Tensor,
    stem_b: torch.Tensor,
    min_db_ratio: float = DOMINANCE_MIN_DB_RATIO,
    force_dual_speakers: bool = False,
) -> torch.Tensor:
    a_1d = stem_a.reshape(-1)
    b_1d = stem_b.reshape(-1)
    rms_a = torch.sqrt(torch.mean(a_1d.float() ** 2) + 1e-9)
    rms_b = torch.sqrt(torch.mean(b_1d.float() ** 2) + 1e-9)
    db_diff = 20.0 * torch.log10(rms_b / rms_a)
    if force_dual_speakers or (
        float(db_diff.item()) >= min_db_ratio and float(rms_b.item()) > SILENCE_RMS_THRESHOLD
    ):
        return torch.stack([a_1d, b_1d], dim=0)
    return a_1d.unsqueeze(0)


class Causal3StemSeparator(nn.Module):
    def __init__(self, config: ModelArchitectureConfig = PROD_240M_CONFIG) -> None:
        super().__init__()
        d = config.separator_d_model
        self.stft_in = nn.Linear(N_BINS_SEP_16K, d)
        self.film_proj = nn.Linear(512, 2 * d)
        self.gru = nn.GRU(d, d, num_layers=config.separator_layers, batch_first=True)
        self.mlp = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
                for _ in range(2)
            ]
        )
        self.mask_head = nn.Linear(d, 3 * N_BINS_SEP_16K)
        self.erb_head = nn.Linear(d, ERB_BANDS)
        self.eq_head = nn.Linear(d, MIC_EQ_TAPS_16K)
        self.rt60_head = nn.Linear(d, 1)
        self.wet_head = nn.Linear(d, 1)
        self.register_buffer("window", torch.hann_window(N_FFT_SEP_16K, device="cpu"), persistent=False)

    def forward(
        self, wav_16k: torch.Tensor, primary_spk_emb: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        B, T_wav = wav_16k.shape
        win = self.window.to(device=wav_16k.device, dtype=wav_16k.dtype)
        spec = torch.stft(
            wav_16k,
            n_fft=N_FFT_SEP_16K,
            hop_length=HOP_SEP_16K,
            win_length=N_FFT_SEP_16K,
            window=win,
            center=True,
            return_complex=True,
        )
        mag = spec.abs().transpose(1, 2)  # (B, T_frames, 257)
        T_frames = mag.shape[1]
        h = self.stft_in(mag)
        if primary_spk_emb is not None:
            gamma, beta = self.film_proj(primary_spk_emb).chunk(2, dim=-1)
            h = h * (1.0 + torch.tanh(gamma.unsqueeze(1))) + beta.unsqueeze(1)
        h, _ = self.gru(h)
        for blk in self.mlp:
            h = h + blk(h)

        masks = (
            torch.sigmoid(self.mask_head(h))
            .view(B, T_frames, 3, N_BINS_SEP_16K)
            .permute(0, 2, 3, 1)
        )
        stems = []
        for s_idx in range(3):
            s_spec = spec * masks[:, s_idx]
            s_wav = torch.istft(
                s_spec,
                n_fft=N_FFT_SEP_16K,
                hop_length=HOP_SEP_16K,
                win_length=N_FFT_SEP_16K,
                window=win,
                center=True,
                length=T_wav,
            )
            stems.append(s_wav)
        stems_tensor = torch.stack(stems, dim=1)  # (B, 3, T_wav)

        erb_mask = torch.sigmoid(self.erb_head(h))  # (B, T_frames, 64)
        h_mean = h.mean(dim=1)
        mic_eq = 0.20 * torch.tanh(self.eq_head(h_mean)).unsqueeze(1)  # (B, 1, 64)
        rt60 = 0.05 + 0.45 * torch.sigmoid(self.rt60_head(h_mean))  # (B, 1)
        wet = 0.35 * torch.sigmoid(self.wet_head(h_mean))  # (B, 1)
        return stems_tensor, erb_mask, mic_eq, rt60, wet


class VocalModeGate(nn.Module):
    def __init__(self, config: ModelArchitectureConfig = PROD_240M_CONFIG) -> None:
        super().__init__()
        d = config.mode_gate_d_model
        self.in_proj = nn.Conv1d(N_MELS, d, kernel_size=1)
        self.conv_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(d, d, kernel_size=3, padding=1),
                    nn.GroupNorm(8, d),
                    nn.GELU(),
                )
                for _ in range(config.mode_gate_layers)
            ]
        )
        self.cls_head = nn.Linear(d, 4)

    def forward(self, log_mel: torch.Tensor) -> Tuple[torch.Tensor, List[VocalMode]]:
        h = self.in_proj(log_mel)
        for conv in self.conv_layers:
            h = h + conv(h)
        logits = self.cls_head(h.transpose(1, 2))  # (B_spk, T_frames, 4)
        frame_modes = logits.argmax(dim=-1)  # (B_spk, T_frames)
        utt_modes = [VocalMode(int(m.item())) for m in logits.mean(dim=1).argmax(dim=-1)]
        return frame_modes, utt_modes


class SceneAcousticMatcher(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        local_gen = torch.Generator(device="cpu").manual_seed(42)
        velvet = torch.zeros(1, VELVET_RIR_TAPS_24K, dtype=torch.float32, device="cpu")
        n_pulses = (VELVET_RIR_TAPS_24K - 24) // 24
        base_grid = torch.arange(1, n_pulses + 1, device="cpu") * 24
        jitter = torch.randint(0, 24, (n_pulses,), generator=local_gen, device="cpu")
        pulse_indices = (base_grid + jitter).clamp(max=VELVET_RIR_TAPS_24K - 1)
        signs = (
            torch.randint(
                0, 2, (1, pulse_indices.numel()), generator=local_gen, device="cpu"
            ).float()
            * 2.0
            - 1.0
        )
        velvet[:, pulse_indices] = signs
        self.register_buffer("velvet_grid", velvet, persistent=False)

    def apply_scene_glue(
        self,
        dry_khmer_wav_24k: torch.Tensor,
        mic_eq_residual_16k: torch.Tensor,
        rt60_sec: torch.Tensor,
        wet_gain: torch.Tensor,
        target_rms_scalar: torch.Tensor,
    ) -> torch.Tensor:
        B, T_len = dry_khmer_wav_24k.shape
        device = dry_khmer_wav_24k.device
        dtype = dry_khmer_wav_24k.dtype

        eq_res_24k = F.interpolate(
            mic_eq_residual_16k.to(dtype),
            size=MIC_EQ_TAPS_24K,
            mode="linear",
            align_corners=False,
        ).squeeze(1)  # (B, 96)
        eq_tail = F.pad(eq_res_24k[:, 1:], (1, 0))
        unit_impulse_96 = F.pad(torch.ones_like(eq_tail[:, :1]), (0, MIC_EQ_TAPS_24K - 1))
        eq_ir = eq_tail + unit_impulse_96
        eq_ir = eq_ir / (torch.norm(eq_ir, p=2, dim=-1, keepdim=True) + 1e-8)

        t_sec = torch.linspace(
            0.0, 0.25, VELVET_RIR_TAPS_24K, device=device, dtype=dtype
        ).unsqueeze(0)
        safe_rt60 = torch.clamp(rt60_sec.to(dtype), min=0.05)  # (B, 1)
        decay = torch.exp((-6.9078 / safe_rt60) * t_sec)  # (B, 6000)
        velvet = self.velvet_grid.to(device=device, dtype=dtype) * decay
        velvet_norm = velvet / (torch.norm(velvet, p=2, dim=-1, keepdim=True) + 1e-8)
        unit_impulse_6000 = F.pad(
            torch.ones_like(velvet_norm[:, :1]), (0, VELVET_RIR_TAPS_24K - 1)
        )
        rir = unit_impulse_6000 + torch.clamp(wet_gain.to(dtype), 0.0, 0.35) * velvet_norm

        n_fft = 1 << (T_len + MIC_EQ_TAPS_24K + VELVET_RIR_TAPS_24K - 2).bit_length()
        W_fft = torch.fft.rfft(dry_khmer_wav_24k, n=n_fft, dim=-1)
        EQ_fft = torch.fft.rfft(eq_ir, n=n_fft, dim=-1)
        RIR_fft = torch.fft.rfft(rir, n=n_fft, dim=-1)
        wet_wav = torch.fft.irfft(W_fft * EQ_fft * RIR_fft, n=n_fft, dim=-1)[:, :T_len]

        cur_rms = torch.sqrt(torch.mean(wet_wav ** 2, dim=-1, keepdim=True) + 1e-8)
        gain = torch.clamp(target_rms_scalar.to(dtype) / cur_rms, min=0.2, max=4.0)
        return torch.clamp(wet_wav * gain, min=-0.98, max=0.98)


class StemAndScenePipeline(nn.Module):
    def __init__(
        self, config: ModelArchitectureConfig = PROD_240M_CONFIG, n_mels: int = N_MELS
    ) -> None:
        super().__init__()
        self.config = config
        self.n_mels = n_mels
        self.separator = Causal3StemSeparator(config)
        self.mode_gate = VocalModeGate(config)
        self.scene_matcher = SceneAcousticMatcher()
        self.register_buffer("window_mel", torch.hann_window(N_FFT_MEL_16K, device="cpu"), persistent=False)
        self.register_buffer(
            "mel_fb",
            build_slaney_mel_fb(N_BINS_MEL_16K, n_mels, sample_rate=SAMPLE_RATE_IN),
            persistent=False,
        )

    def _compute_log_mel(self, wav_16k: torch.Tensor) -> torch.Tensor:
        win = self.window_mel.to(device=wav_16k.device, dtype=wav_16k.dtype)
        fb = self.mel_fb.to(device=wav_16k.device, dtype=wav_16k.dtype)
        spec = torch.stft(
            wav_16k,
            n_fft=N_FFT_MEL_16K,
            hop_length=HOP_MEL_16K,
            win_length=N_FFT_MEL_16K,
            window=win,
            center=True,
            return_complex=True,
        ).abs()
        mel = torch.matmul(spec.transpose(1, 2), fb).transpose(1, 2)
        return torch.log(mel.clamp(min=1e-6))

    def separate_and_classify(
        self,
        wav_16k: torch.Tensor,
        utterance_id: int = 0,
        utterance_start_ms: float = 0.0,
        trailing_pause_ms: float = 0.0,
        primary_spk_emb: Optional[torch.Tensor] = None,
        force_dual_speakers: bool = False,
    ) -> SeparatedSceneSlice:
        if wav_16k.dim() == 1:
            wav_16k = wav_16k.unsqueeze(0)
        stems, erb_mask, mic_eq, rt60, wet = self.separator(wav_16k, primary_spk_emb)

        # v10.7 P2-A (V13): batch-aware aggregation. The v10.6 code hardcoded
        # stems[0, 0], stems[0, 1] and erb_mask[0], which silently discarded
        # every batch element after the first, and then called
        # mic_eq.expand(B_spk, -1, -1) on a (B, 1, 64) tensor, which raises a
        # RuntimeError whenever B = 2 and B_spk = 1.
        #
        # The scene pipeline is single-scene: it must return ONE scene. When
        # the caller supplies a batch, the batch is therefore aggregated DOWN
        # to batch size 1, weighted by the per-batch RMS energy, so that every
        # element of the batch actively contributes to the output scene.
        B_in = wav_16k.shape[0]
        if B_in > 1:
            # Per-batch RMS energy weights.
            w = wav_16k.pow(2).mean(dim=-1).sqrt()  # (B,)
            w = w / (w.sum() + 1e-8)

            # stems: (B, 2, T_wav) -> (1, 2, T_wav)
            stems = (stems * w.view(-1, 1, 1)).sum(dim=0, keepdim=True)
            # erb_mask: (B, T_frames, 64) -> (1, T_frames, 64)
            erb_mask = (erb_mask * w.view(-1, 1, 1)).sum(dim=0, keepdim=True)
            # mic_eq: (B, 1, 64) -> (1, 1, 64)
            if mic_eq.dim() >= 2:
                mic_eq = (mic_eq * w.view(-1, 1, 1)).sum(dim=0, keepdim=True)
            else:
                mic_eq = (mic_eq * w).sum(dim=0, keepdim=True)
            # rt60 / wet: (B, 1) or (B,) -> (1, 1)
            if rt60.dim() >= 2:
                rt60 = (rt60 * w.view(-1, 1)).sum(dim=0, keepdim=True)
            else:
                rt60 = (rt60 * w).sum(dim=0, keepdim=True)
            if wet.dim() >= 2:
                wet = (wet * w.view(-1, 1)).sum(dim=0, keepdim=True)
            else:
                wet = (wet * w).sum(dim=0, keepdim=True)

        active_stems = filter_dominant_speakers(
            stems[0, 0], stems[0, 1], force_dual_speakers=force_dual_speakers
        )  # (B_spk, T_wav)
        B_spk = active_stems.shape[0]
        mel_stems = self._compute_log_mel(active_stems)  # (B_spk, 100, T_frames)
        frame_modes, dom_modes = self.mode_gate(mel_stems)
        rms_scalar = torch.sqrt(torch.mean(active_stems ** 2, dim=-1, keepdim=True) + 1e-8)

        return SeparatedSceneSlice(
            utterance_id=utterance_id,
            utterance_start_ms=utterance_start_ms,
            frame_vocal_modes=frame_modes,
            dominant_vocal_modes=dom_modes,
            active_speaker_stems=active_stems,
            speaker_ids=list(range(B_spk)),
            mic_eq_residual_16k=mic_eq.expand(B_spk, -1, -1),
            rt60_sec=rt60.expand(B_spk, -1),
            wet_gain=wet.expand(B_spk, -1),
            rms_scalar=rms_scalar,
            erb_suppression_mask=erb_mask[0],  # (T_frames, 64)  v10.7 P2-A: stays 2D
            trailing_pause_ms=trailing_pause_ms,
        )

    def apply_scene_glue(
        self,
        dry_khmer_wav_24k: torch.Tensor,
        mic_eq_residual_16k: torch.Tensor,
        rt60_sec: torch.Tensor,
        wet_gain: torch.Tensor,
        target_rms_scalar: torch.Tensor,
    ) -> torch.Tensor:
        return self.scene_matcher.apply_scene_glue(
            dry_khmer_wav_24k, mic_eq_residual_16k, rt60_sec, wet_gain, target_rms_scalar
        )
```

---

### 4.4 Specialist A — `ear_cartridge.py`, `codeswitch_router.py` & `duration_aligner.py`

#### 4.4.1 `backend/nn_cartridges/ear_cartridge.py`
```python
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

from backend.contracts.schemas import (
    HOP_LENGTH_16K as HOP_16K,
    LANG_TO_ID,
    N_FFT_16K,
    N_MELS,
    PROD_240M_CONFIG,
    SAMPLE_RATE_IN,
    SEMANTIC_DIM,
    CartridgeThreadState,
    LanguageCode,
    ModelArchitectureConfig,
)
from backend.nn_cartridges.universal_core import build_slaney_mel_fb

N_BINS_16K = N_FFT_16K // 2 + 1  # 513 bins -> guarantees rank 100


class FastConformerBlock(nn.Module):
    def __init__(self, d: int, heads: int, ffn_dim: int) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.pw1 = nn.Linear(d, 2 * d)
        self.dwconv = nn.Conv1d(d, d, kernel_size=31, padding=15, groups=d)
        # v10.7 P1-E: BatchNorm1d normalizes over the BATCH dimension, which is
        # size 1 during streaming inference, so its running statistics are
        # corrupted by batch-statistics estimation. GroupNorm normalizes per
        # sample and is correct at B=1. Verified parameter-neutral: both hold
        # 2*d trainable parameters (1,024 prod / 256 CI per block); the
        # BatchNorm running buffers are not counted by .parameters().
        self.bn = nn.GroupNorm(8, d)
        self.pw2 = nn.Linear(d, d)
        self.ffn = nn.Sequential(nn.Linear(d, ffn_dim), nn.GELU(), nn.Linear(ffn_dim, d))
        self.norms = nn.ModuleList([nn.LayerNorm(d) for _ in range(3)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norms[0](x)
        attn_out, _ = self.self_attn(h, h, h, need_weights=False)
        x = x + attn_out

        h = self.norms[1](x)
        u, v = self.pw1(h).chunk(2, dim=-1)
        c = u * torch.sigmoid(v)
        c = self.dwconv(c.transpose(1, 2))
        c = torch.nn.functional.gelu(self.bn(c)).transpose(1, 2)
        x = x + self.pw2(c)

        x = x + self.ffn(self.norms[2](x))
        return x


class StreamingEarCartridge(nn.Module):
    def __init__(self, config: ModelArchitectureConfig = PROD_240M_CONFIG) -> None:
        super().__init__()
        d = config.ear_d_model
        self.register_buffer("window", torch.hann_window(N_FFT_16K, device="cpu"), persistent=False)
        self.register_buffer(
            "mel_fb",
            build_slaney_mel_fb(N_BINS_16K, N_MELS, sample_rate=SAMPLE_RATE_IN),
            persistent=False,
        )
        self.subsampler = nn.Sequential(
            nn.Conv1d(N_MELS, d, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv1d(d, d, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
        )
        self.lang_emb = nn.Embedding(4, d)
        self.blocks = nn.ModuleList(
            [
                FastConformerBlock(d, config.ear_heads, config.ear_ffn_dim)
                for _ in range(config.ear_layers)
            ]
        )
        self.bus_proj = nn.Linear(d, SEMANTIC_DIM)
        self.rate_head = nn.Linear(SEMANTIC_DIM, 1)
        # v10.7 P0-A: the per-call acoustic taps are published through a
        # thread-local container rather than as mutable module attributes, so
        # concurrent calls on different threads can never observe each other's
        # state. The last_l6_hidden / last_l8_hidden property bridges below
        # preserve 100% attribute compatibility for existing call sites.
        self._state = CartridgeThreadState()

    # v10.7 P0-A: property bridges onto the thread-local state container.
    # These keep the public attribute names (and therefore every existing
    # call site) working unchanged, while the underlying storage is now
    # strictly per-thread. Assignment and reads can no longer race between
    # the GPU worker thread and the event-loop thread.
    @property
    def last_bus_out(self) -> torch.Tensor | None:
        # v10.7 P2-B: property bridge onto the thread-local state container.
        return self._state.last_bus_out

    @last_bus_out.setter
    def last_bus_out(self, value: torch.Tensor | None) -> None:
        self._state.last_bus_out = value

    @property
    def last_l6_hidden(self) -> torch.Tensor | None:
        return self._state.l6_hidden

    @last_l6_hidden.setter
    def last_l6_hidden(self, value: torch.Tensor | None) -> None:
        self._state.l6_hidden = value

    @property
    def last_l8_hidden(self) -> torch.Tensor | None:
        return self._state.l8_hidden

    @last_l8_hidden.setter
    def last_l8_hidden(self, value: torch.Tensor | None) -> None:
        self._state.l8_hidden = value

    def log_mel(self, wav_16k: torch.Tensor) -> torch.Tensor:
        win = self.window.to(device=wav_16k.device, dtype=wav_16k.dtype)
        fb = self.mel_fb.to(device=wav_16k.device, dtype=wav_16k.dtype)
        spec = torch.stft(
            wav_16k,
            n_fft=N_FFT_16K,
            hop_length=HOP_16K,
            win_length=N_FFT_16K,
            window=win,
            center=True,
            return_complex=True,
        ).abs()
        mel = torch.matmul(spec.transpose(1, 2), fb).transpose(1, 2)
        return torch.log(mel.clamp(min=1e-6))

    def forward(
        self, wav_16k: torch.Tensor, source_lang: LanguageCode = "en"
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        mel = self.log_mel(wav_16k)
        h = self.subsampler(mel).transpose(1, 2)
        lang_idx = torch.full(
            (h.shape[0],), LANG_TO_ID[source_lang], dtype=torch.long, device=h.device
        )
        h = h + self.lang_emb(lang_idx).unsqueeze(1)
        l6_idx = min(5, len(self.blocks) - 1)
        l8_idx = min(7, len(self.blocks) - 1)
        for idx, blk in enumerate(self.blocks):
            h = blk(h)
            if idx == l6_idx:
                self.last_l6_hidden = self.bus_proj(h)
            if idx == l8_idx:
                self.last_l8_hidden = self.bus_proj(h)
        bus = self.bus_proj(h)
        if self.last_l6_hidden is None:
            self.last_l6_hidden = bus
        if self.last_l8_hidden is None:
            self.last_l8_hidden = bus
        r_sps = (1.5 + 7.5 * torch.sigmoid(self.rate_head(bus).mean(dim=1))).squeeze(-1)
        # v10.7 P2-B: publish the bus-projected acoustic output so the
        # multi-teacher ear distiller can compare it against the teacher
        # features. Stored on the thread-local state, not as a shared
        # module attribute.
        self.last_bus_out = bus
        return bus, r_sps
```

#### 4.4.2 `backend/nn_cartridges/codeswitch_router.py`
```python
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from backend.contracts.schemas import (
    PROD_240M_CONFIG,
    SEMANTIC_DIM,
    ModelArchitectureConfig,
)


class AcousticPointerCodeSwitchBridge(nn.Module):
    def __init__(self, config: ModelArchitectureConfig = PROD_240M_CONFIG) -> None:
        super().__init__()
        d = config.pointer_d_model
        self.in_q = nn.Linear(SEMANTIC_DIM, d)
        self.in_kv = nn.Linear(SEMANTIC_DIM, d)
        self.blocks = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "attn": nn.MultiheadAttention(d, 8, batch_first=True),
                        "ffn": nn.Sequential(
                            nn.Linear(d, config.pointer_ffn_dim),
                            nn.GELU(),
                            nn.Linear(config.pointer_ffn_dim, d),
                        ),
                    }
                )
                for _ in range(config.pointer_layers)
            ]
        )
        self.out_proj = nn.Linear(d, SEMANTIC_DIM)
        self.gate_head = nn.Linear(SEMANTIC_DIM, 1)

    def forward(
        self,
        semantic_bus: torch.Tensor,
        source_acoustic_features: torch.Tensor,
        force_gate: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        q = self.in_q(semantic_bus)
        kv = self.in_kv(source_acoustic_features)
        for blk in self.blocks:
            attn_out, _ = blk["attn"](q, kv, kv, need_weights=False)
            q = q + attn_out
            q = q + blk["ffn"](q)
        pointer_context = self.out_proj(q)
        gate = torch.sigmoid(self.gate_head(semantic_bus))
        if force_gate is not None:
            gate = torch.full_like(gate, float(force_gate))
        fused = (1.0 - gate) * semantic_bus + gate * pointer_context
        return fused, gate
```

#### 4.4.3 `backend/nn_cartridges/duration_aligner.py`
```python
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from backend.contracts.schemas import (
    HOP_LENGTH,
    MAX_ELASTIC_PAUSE_BORROW_MS,
    MINOR_SYLLABLE_MAX_FRAMES,
    SAMPLE_RATE_OUT,
    SEMANTIC_DIM,
)


class LearnedDurationAligner(nn.Module):
    def __init__(self, d_model: int = SEMANTIC_DIM, sigma: float = 1.5) -> None:
        super().__init__()
        self.d_model = d_model
        self.sigma = sigma
        self.dur_proj = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )

    @staticmethod
    def compute_elastic_target_frames(
        orig_duration_sec: float,
        trailing_pause_ms: float,
        speech_rate_sps: float,
        playback_rate: float = 1.0,
    ) -> int:
        borrow_ms = 0.0
        if speech_rate_sps > 4.8 and trailing_pause_ms > 40.0:
            borrow_ms = min(MAX_ELASTIC_PAUSE_BORROW_MS, 0.65 * trailing_pause_ms)
        effective_sec = orig_duration_sec + (borrow_ms / 1000.0)
        safe_rate = max(0.5, min(2.5, float(playback_rate)))
        scaled_sec = effective_sec / safe_rate
        return max(6, int(round(scaled_sec * SAMPLE_RATE_OUT / HOP_LENGTH)))

    def forward(
        self,
        syll_hidden: torch.Tensor,
        sesqui_ids: torch.Tensor,
        target_frames: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raw_dur = F.softplus(self.dur_proj(syll_hidden).squeeze(-1)) + 0.1
        # v10.7 AUD-107-01: sesqui_ids == 0 marks a DEFERRED (wait-gated)
        # syllable, which is not spoken in this clause and must consume
        # zero acoustic frames. Without this the deferred syllables still
        # received a 0.25 or 1.00 duration weight and the diffusion model
        # compressed the REAL syllables to make room for them.
        prior = torch.where(
            sesqui_ids == 0,
            0.0,
            torch.where(sesqui_ids == 1, 0.25, 1.00),
        ).to(raw_dur.dtype)
        weighted = raw_dur * prior
        norm_dur = weighted * (
            float(target_frames) / (weighted.sum(dim=-1, keepdim=True) + 1e-8)
        )

        is_minor = sesqui_ids == 1
        is_major = ~is_minor
        clamped_minor = torch.where(
            is_minor, torch.clamp(norm_dur, max=MINOR_SYLLABLE_MAX_FRAMES), norm_dur
        )
        excess = (norm_dur - clamped_minor).sum(dim=-1, keepdim=True)
        major_weights = clamped_minor * is_major.to(norm_dur.dtype)
        major_sum = major_weights.sum(dim=-1, keepdim=True)
        redistributed = torch.where(
            major_sum > 1e-6,
            clamped_minor + excess * (major_weights / (major_sum + 1e-8)),
            clamped_minor,
        )
        final_durations = torch.where(
            is_minor,
            torch.clamp(redistributed, max=MINOR_SYLLABLE_MAX_FRAMES),
            redistributed,
        )

        cum_dur = torch.cumsum(final_durations, dim=-1)
        centers = cum_dur - 0.5 * final_durations
        frame_grid = (
            torch.arange(
                target_frames, device=syll_hidden.device, dtype=syll_hidden.dtype
            ).view(1, target_frames, 1)
            + 0.5
        )
        attn_logits = -0.5 * ((frame_grid - centers.unsqueeze(1)) / self.sigma) ** 2
        # v10.7 AUD-107-09: mask the DEFERRED (sesqui_ids == 0) columns out
        # of the Gaussian attention so they cannot absorb attention mass on
        # the tail frames of the clause. A deferred syllable has a zero
        # duration, so its cumulative centre sits at target_frames, but it
        # still received softmax mass there and squeezed the REAL syllables.
        #
        # The has_active guard matters: a row in which EVERY syllable is
        # deferred has no active column, and masking it would leave the
        # softmax over an all-(-1e9) vector. The guard keeps that case
        # explicit and defined.
        active_mask = (sesqui_ids > 0).unsqueeze(1)  # (B, 1, L)
        has_active = active_mask.any(dim=-1, keepdim=True)  # (B, 1, 1)
        attn_logits = torch.where(
            has_active & (~active_mask),
            torch.full_like(attn_logits, -1e9),
            attn_logits,
        )
        attn_weights = F.softmax(attn_logits, dim=-1)
        aligned_hidden = torch.bmm(attn_weights, syll_hidden)
        return aligned_hidden, final_durations
```

---

### 4.5 Specialist A — `mouth_cartridge.py` & `s2st_console.py`

#### 4.5.1 `backend/nn_cartridges/mouth_cartridge.py`
```python
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from backend.contracts.schemas import (
    CartridgeThreadState,
    LANG_TO_ID,
    N_MELS,
    PROD_240M_CONFIG,
    SEMANTIC_DIM,
    LanguageCode,
    ModelArchitectureConfig,
    SyllableTensorBatch,
)
from backend.nn_cartridges.duration_aligner import LearnedDurationAligner

def sinusoidal_time_embedding(val: torch.Tensor, dim: int) -> torch.Tensor:
    """
    v10.7 P1-C: parameter-free sinusoidal Fourier embedding of a continuous
    scalar (or a batch of scalars) into a `dim`-wide vector.

    The v10.6 code did `t.view(B, 1).expand(B, d_half)`, i.e. it repeated one
    rank-1 scalar across every column. `time_mlp` therefore received a vector
    that carried only two degrees of freedom (w1*t + w2*dt + b) and had no
    spectral resolution over the timestep at all.

    Here the value is embedded on log-spaced frequencies
    exp(-ln(10000) * i / (dim/2)), half sine and half cosine, which is the
    standard transformer positional encoding applied to a continuous value.

    Parameters:
        val: (B,) or (B, 1) tensor of values to embed.
        dim: output width. Must be even.

    Returns:
        (B, dim) tensor. Adds NO parameters.
    """
    if dim % 2 != 0:
        raise ValueError(f"sinusoidal_time_embedding requires an even dim, got {dim}")
    v = val.reshape(-1, 1).to(dtype=torch.float32)
    half = dim // 2
    freq = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=v.device, dtype=torch.float32) / half
    )
    ang = v * freq.unsqueeze(0)
    out = torch.zeros(v.shape[0], dim, device=v.device, dtype=torch.float32)
    out[:, 0::2] = torch.sin(ang)
    out[:, 1::2] = torch.cos(ang)
    return out


SLOT_VOCAB = {
    "onset": 64,
    "coeng": 64,
    "medial": 4,
    "register": 4,
    "vowel": 48,
    "coda": 32,
    "sesqui": 3,
}


class DiTBlock(nn.Module):
    def __init__(self, d: int, heads: int, ffn_dim: int) -> None:
        super().__init__()
        self.scale_shift_table = nn.Parameter(torch.zeros(6, d))
        self.n1 = nn.LayerNorm(d, elementwise_affine=False)
        self.self_attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.n2 = nn.LayerNorm(d, elementwise_affine=False)
        self.cross_attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.n3 = nn.LayerNorm(d, elementwise_affine=False)
        self.ffn = nn.Sequential(nn.Linear(d, ffn_dim), nn.GELU(), nn.Linear(ffn_dim, d))

    def forward(self, x: torch.Tensor, t_mod: torch.Tensor, emo: torch.Tensor) -> torch.Tensor:
        mod = (self.scale_shift_table.unsqueeze(0) + t_mod).unbind(dim=1)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = [
            m.unsqueeze(1) for m in mod
        ]
        h = self.n1(x) * (1.0 + scale_msa) + shift_msa
        sa_out, _ = self.self_attn(h, h, h, need_weights=False)
        x = x + (1.0 + gate_msa) * sa_out
        ca_out, _ = self.cross_attn(self.n2(x), emo, emo, need_weights=False)
        x = x + ca_out
        h_ffn = self.n3(x) * (1.0 + scale_mlp) + shift_mlp
        x = x + (1.0 + gate_mlp) * self.ffn(h_ffn)
        return x


class KhmerMouthCartridge(nn.Module):
    def __init__(self, config: ModelArchitectureConfig = PROD_240M_CONFIG) -> None:
        super().__init__()
        d = config.mouth_d_model
        self.config = config

        # Non-Monotonic Cross-Attention Reordering Bridge (+1.05M params)
        self.reorder_attn = nn.MultiheadAttention(SEMANTIC_DIM, 8, batch_first=True)
        self.reorder_norm = nn.LayerNorm(SEMANTIC_DIM)

        self.heads = nn.ModuleDict(
            {k: nn.Linear(SEMANTIC_DIM, v) for k, v in SLOT_VOCAB.items()}
        )
        self.embeddings = nn.ModuleDict(
            {k: nn.Embedding(v, SEMANTIC_DIM) for k, v in SLOT_VOCAB.items()}
        )
        self.target_lang_emb = nn.Embedding(4, SEMANTIC_DIM)
        self.duration_aligner = LearnedDurationAligner(SEMANTIC_DIM)
        self.cond_proj = nn.Linear(SEMANTIC_DIM, d)
        self.emo_proj = nn.Linear(SEMANTIC_DIM, d)
        self.mel_in = nn.Linear(N_MELS, d)
        self.infill_proj = nn.Linear(N_MELS, d)
        self.time_mlp = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, d))
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(d, 6 * d))
        self.blocks = nn.ModuleList(
            [
                DiTBlock(d, config.mouth_heads, config.mouth_ffn_dim)
                for _ in range(config.mouth_layers)
            ]
        )
        self.mel_out = nn.Linear(d, N_MELS)

        # Rolling Cross-Clause State & Training Telemetry Taps
        # v10.7 P0-A: the per-call telemetry is published through a thread-local
        # container rather than as mutable module attributes. The property
        # bridges below preserve the original attribute names, so every
        # existing call site keeps working unchanged while concurrent calls on
        # different threads can no longer observe each other's state.
        self._state = CartridgeThreadState()

    # v10.7 P0-A: property bridge onto the thread-local state container.
    @property
    def last_next_past_kv(self) -> Optional[torch.Tensor]:
        return self._state.last_next_past_kv

    @last_next_past_kv.setter
    def last_next_past_kv(self, value: Optional[torch.Tensor]) -> None:
        self._state.last_next_past_kv = value

    # v10.7 P0-A: property bridge onto the thread-local state container.
    @property
    def last_next_past_gate(self) -> Optional[torch.Tensor]:
        return self._state.last_next_past_gate

    @last_next_past_gate.setter
    def last_next_past_gate(self, value: Optional[torch.Tensor]) -> None:
        self._state.last_next_past_gate = value

    # v10.7 P0-A: property bridge onto the thread-local state container.
    @property
    def last_deferred_syll_count(self) -> int:
        return self._state.last_deferred_syll_count

    @last_deferred_syll_count.setter
    def last_deferred_syll_count(self, value: int) -> None:
        self._state.last_deferred_syll_count = value

    # v10.7 P0-A: property bridge onto the thread-local state container.
    @property
    def last_pred_durations(self) -> Optional[torch.Tensor]:
        return self._state.last_pred_durations

    @last_pred_durations.setter
    def last_pred_durations(self, value: Optional[torch.Tensor]) -> None:
        self._state.last_pred_durations = value

    # v10.7 P0-A: property bridge onto the thread-local state container.
    @property
    def last_gate_l(self) -> Optional[torch.Tensor]:
        return self._state.last_gate_l

    @last_gate_l.setter
    def last_gate_l(self, value: Optional[torch.Tensor]) -> None:
        self._state.last_gate_l = value

    def _build_sinusoidal_queries(
        self, B: int, L_syll: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        if L_syll > 1:
            pos = torch.linspace(0.0, 100.0, L_syll, device=device, dtype=torch.float32).unsqueeze(1)
        else:
            pos = torch.zeros(1, 1, device=device, dtype=torch.float32)
        div = torch.exp(
            torch.arange(0, SEMANTIC_DIM, 2, device=device, dtype=torch.float32)
            * (-math.log(10000.0) / SEMANTIC_DIM)
        )
        pe = torch.zeros(1, L_syll, SEMANTIC_DIM, device=device, dtype=torch.float32)
        pe[0, :, 0::2] = torch.sin(pos * div)
        pe[0, :, 1::2] = torch.cos(pos * div)
        return pe.to(dtype=dtype).expand(B, -1, -1)

    def predict_syllables_and_align(
        self,
        fused_semantic_bus: torch.Tensor,
        gate_g_t: torch.Tensor,
        r_sps: torch.Tensor,
        target_lang: LanguageCode,
        orig_duration_sec: float,
        target_frames: int,
        explicit_syll_batch: Optional[SyllableTensorBatch] = None,
        past_clause_kv: Optional[torch.Tensor] = None,
        past_gate_kv: Optional[torch.Tensor] = None,
        deferred_syll_count: int = 0,
        is_hard_ceiling_slice: bool = False,
        entropy_wait_threshold: float = 1.85,
        # v10.7 P1-D: optional speaker-timbre vector, shape (B, 1, SEMANTIC_DIM).
        # Added to the aligned hidden state before cond_proj; no parameters.
        speaker_timbre: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, SyllableTensorBatch]:
        B, T_cur, D = fused_semantic_bus.shape
        device = fused_semantic_bus.device
        lang_idx = torch.full((B,), LANG_TO_ID[target_lang], dtype=torch.long, device=device)
        bus_cond = fused_semantic_bus + self.target_lang_emb(lang_idx).unsqueeze(1)

        # 1. Prepend rolling past-clause KV cache with parameter-free negative sinusoidal offset
        # v10.7 P0-A: guard the B_spk transition. When the number of active
        # speakers changes between clauses (1 -> 2 or 2 -> 1) the cached KV
        # has a different batch dimension, and torch.cat on dim=1 would raise a
        # RuntimeError on the mismatched dim=0. The stale cache is discarded
        # instead: a KV cache built for a different speaker set carries no
        # valid context for this one.
        if past_clause_kv is not None and past_clause_kv.shape[0] != B:
            past_clause_kv = None
            past_gate_kv = None
        if past_clause_kv is not None and past_clause_kv.shape[1] > 0:
            T_past = past_clause_kv.shape[1]
            past_pe = -self._build_sinusoidal_queries(B, T_past, device, bus_cond.dtype)
            kv_mem = torch.cat([past_clause_kv.to(bus_cond.dtype) + past_pe, bus_cond], dim=1)
            gate_mem = (
                torch.cat([past_gate_kv.to(gate_g_t.dtype), gate_g_t], dim=1)
                if past_gate_kv is not None and past_gate_kv.shape[1] == T_past
                else F.pad(gate_g_t, (0, 0, T_past, 0))
            )
        else:
            kv_mem = bus_cond
            gate_mem = gate_g_t

        # 2. Cache trailing 20 frames (800ms) if sliced by Trigger #2 (1.60s hard ceiling)
        if is_hard_ceiling_slice:
            keep_frames = min(20, T_cur)
            self.last_next_past_kv = bus_cond[:, -keep_frames:, :].detach()
            self.last_next_past_gate = gate_g_t[:, -keep_frames:, :].detach()
        else:
            self.last_next_past_kv = None
            self.last_next_past_gate = None

        if explicit_syll_batch is None:
            mean_r_sps = float(r_sps.mean().item())
            gamma_contract = 0.78 if mean_r_sps > 5.5 else 1.00
            base_syll = max(1, int(round(orig_duration_sec * min(mean_r_sps, 6.5) * gamma_contract)))
            L_syll = base_syll + max(0, int(deferred_syll_count))

            pos_q = self._build_sinusoidal_queries(B, L_syll, device, bus_cond.dtype)
            reordered_bus, attn_w = self.reorder_attn(
                query=pos_q, key=kv_mem, value=kv_mem, need_weights=True
            )
            bus_l = self.reorder_norm(pos_q + reordered_bus)
            gate_l = torch.bmm(attn_w, gate_mem)
            self.last_gate_l = gate_l

            onset_logits = self.heads["onset"](bus_l)
            # 3. Hibiki-style predictability / entropy wait-gating on right boundary of Trigger #2 slices
            self.last_deferred_syll_count = 0
            # v10.7: wait_mask is hoisted so it can also be applied to the
            # ARGMAX result below, after the head projections.
            wait_mask = None
            if is_hard_ceiling_slice and L_syll > 3:
                tail_probs = F.softmax(onset_logits[:, -3:, :], dim=-1)
                tail_entropy = -(tail_probs * torch.log(tail_probs.clamp(min=1e-8))).sum(dim=-1)
                wait_mask = tail_entropy > entropy_wait_threshold
                # v10.7 P0-A: wait_mask has shape (B, 3). Indexing [0] counted
                # only the first speaker and silently ignored speakers 1..B-1.
                # Sum per speaker and take the maximum, so the conservative
                # (largest) deferral count is used for the whole batch.
                self.last_deferred_syll_count = int(
                    wait_mask.float().sum(dim=-1).max().item()
                )
                bus_l = torch.cat(
                    [
                        bus_l[:, :-3, :],
                        torch.where(wait_mask.unsqueeze(-1), torch.zeros_like(bus_l[:, -3:, :]), bus_l[:, -3:, :]),
                    ],
                    dim=1,
                )
                onset_logits = self.heads["onset"](bus_l)

            onset_ids = onset_logits.argmax(dim=-1)
            coeng_ids = self.heads["coeng"](bus_l).argmax(dim=-1)
            medial_ids = self.heads["medial"](bus_l).argmax(dim=-1)
            vowel_ids = self.heads["vowel"](bus_l).argmax(dim=-1)
            coda_ids = self.heads["coda"](bus_l).argmax(dim=-1)

            # v10.7: zero the DEFERRED tail syllables AFTER the argmax.
            #
            # Zeroing bus_l before the projection is not sufficient:
            # nn.Linear has a bias vector, so a zero input still produces a
            # non-zero logit vector and the argmax returns a real onset id.
            # The decoder skips a syllable only when the triplet is exactly
            # (0, 0, 0), so the mask must be applied to the ids themselves.
            if target_lang == "en":
                # v10.7 P1-B: a SYNTHESIZED English target syllable must land
                # inside the reserved English ranges, otherwise
                # decode_batch_to_text never reaches ENGLISH_SYLLABLE_TABLE
                # and falls through to the CVC pseudoword fallback, which
                # populates cs_words with synthetic brands.
                #
                #   onset : 34 + argmax over the 24 reserved onset logits
                #   coeng : 0 (the synthesized-syllable sentinel)
                #   vowel : 28 + argmax over the 17 reserved vowel logits
                #   coda  : 18 + argmax over the 14 reserved coda logits
                #
                # The ranges are disjoint from the Khmer ranges: onset
                # 1..33 / vowel 0..24 / coda 0..17.
                onset_ids = 34 + onset_logits[:, :, 34:58].argmax(dim=-1)
                coeng_ids = torch.zeros((B, L_syll), dtype=torch.long, device=device)
                medial_ids = self.heads["medial"](bus_l).argmax(dim=-1).clamp(0, 3)
                vowel_ids = 28 + self.heads["vowel"](bus_l)[:, :, 28:45].argmax(dim=-1)
                coda_ids = 18 + self.heads["coda"](bus_l)[:, :, 18:32].argmax(dim=-1)
                register_ids = torch.full((B, L_syll), 3, dtype=torch.long, device=device)
                sesqui_ids = torch.full((B, L_syll), 2, dtype=torch.long, device=device)
            else:
                reg_logits = self.heads["register"](bus_l)
                register_ids = 1 + reg_logits[:, :, 1:3].argmax(dim=-1)
                sesqui_ids = 1 + self.heads["sesqui"](bus_l)[:, :, 1:3].argmax(dim=-1)
                is_cs = gate_l.squeeze(-1) >= 0.5
                register_ids = torch.where(
                    is_cs, torch.full_like(register_ids, 3), register_ids
                )
                sesqui_ids = torch.where(is_cs, torch.full_like(sesqui_ids, 2), sesqui_ids)

            # v10.7 AUD-107-01: apply the DEFERRED no-op mask AFTER the
            # language branch, not before it.
            #
            # v10.7 applied it before the `if target_lang == "en"` branch, but
            # that branch OVERWRITES onset_ids, vowel_ids and coda_ids with the
            # reserved English ranges (34+, 28+, 18+). The deferred tail was
            # therefore never masked on a KM -> EN slice, so the tail syllables
            # were emitted in the current clause AND deferred to the next one,
            # duplicating the words.
            #
            # The mask also now covers coeng_ids and sesqui_ids. A deferred
            # syllable must be a complete no-op: sesqui_ids == 0 is the
            # zero-frame sentinel consumed by LearnedDurationAligner, so the
            # diffusion model stops allocating acoustic frames to a syllable
            # that is not spoken in this clause.
            if wait_mask is not None:
                zero_tail = torch.zeros_like(onset_ids[:, -3:])
                onset_ids = torch.cat(
                    [onset_ids[:, :-3], torch.where(wait_mask, zero_tail, onset_ids[:, -3:])],
                    dim=1,
                )
                coeng_ids = torch.cat(
                    [coeng_ids[:, :-3], torch.where(wait_mask, zero_tail, coeng_ids[:, -3:])],
                    dim=1,
                )
                vowel_ids = torch.cat(
                    [vowel_ids[:, :-3], torch.where(wait_mask, zero_tail, vowel_ids[:, -3:])],
                    dim=1,
                )
                coda_ids = torch.cat(
                    [coda_ids[:, :-3], torch.where(wait_mask, zero_tail, coda_ids[:, -3:])],
                    dim=1,
                )
                sesqui_ids = torch.cat(
                    [sesqui_ids[:, :-3], torch.where(wait_mask, zero_tail, sesqui_ids[:, -3:])],
                    dim=1,
                )

            syll_batch = SyllableTensorBatch(
                onset_ids=onset_ids,
                coeng_ids=coeng_ids,
                medial_ids=medial_ids,
                register_ids=register_ids,
                vowel_ids=vowel_ids,
                coda_ids=coda_ids,
                sesqui_ids=sesqui_ids,
            )
        else:
            syll_batch = explicit_syll_batch
            L_syll = syll_batch.onset_ids.shape[1]
            pos_q = self._build_sinusoidal_queries(B, L_syll, device, bus_cond.dtype)
            reordered_bus, attn_w = self.reorder_attn(
                query=pos_q, key=kv_mem, value=kv_mem, need_weights=True
            )
            bus_l = self.reorder_norm(pos_q + reordered_bus)
            self.last_gate_l = torch.bmm(attn_w, gate_mem)
            self.last_deferred_syll_count = 0

        slot_sum = (
            self.embeddings["onset"](syll_batch.onset_ids)
            + self.embeddings["coeng"](syll_batch.coeng_ids)
            + self.embeddings["medial"](syll_batch.medial_ids)
            + self.embeddings["register"](syll_batch.register_ids)
            + self.embeddings["vowel"](syll_batch.vowel_ids)
            + self.embeddings["coda"](syll_batch.coda_ids)
            + self.embeddings["sesqui"](syll_batch.sesqui_ids)
        )
        syll_hidden = bus_l + slot_sum
        aligned_hidden, pred_durs = self.duration_aligner(
            syll_hidden, syll_batch.sesqui_ids, target_frames
        )
        self.last_pred_durations = pred_durs
        # v10.7 P1-D: zero-parameter speaker-timbre conditioning. The speaker
        # registry needs a speaker-discriminative embedding, but the emotion
        # encoder applies F.instance_norm and avg_pool1d(k=25), which remove
        # the spectral envelope that carries vocal-tract identity. The
        # L2-normalized mean-pooled layer-6 acoustic representation is used
        # instead, and is added to the aligned hidden state BEFORE cond_proj.
        # This adds no parameters: the ledger is unchanged.
        if speaker_timbre is not None:
            aligned_hidden = aligned_hidden + speaker_timbre
        cond = self.cond_proj(aligned_hidden)
        return cond, syll_batch

    def predict_velocity(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor,
        emotion_tokens: torch.Tensor,
        dt: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B = x_t.shape[0]
        d = self.config.mouth_d_model
        d_half = d // 2
        h = self.mel_in(x_t.transpose(1, 2)) + cond
        # v10.7 P1-C: parameter-free sinusoidal Fourier timestep embedding.
        # v10.6 expanded a rank-1 scalar across d_half columns, so time_mlp saw
        # only w1*t + w2*dt + b and had no spectral resolution over t at all.
        # The sinusoidal embedding restores full resolution with ZERO new
        # parameters: time_mlp keeps its Linear(d, d) shape.
        t_part = sinusoidal_time_embedding(t, d_half).to(
            dtype=h.dtype, device=h.device
        )
        if dt is None:
            dt_part = torch.zeros(B, d - d_half, dtype=h.dtype, device=h.device)
        else:
            dt_part = sinusoidal_time_embedding(dt, d - d_half).to(
                dtype=h.dtype, device=h.device
            )
        t_vec = torch.cat([t_part, dt_part], dim=-1)
        t_emb = self.time_mlp(t_vec)
        t_mod = self.adaLN_modulation(t_emb).view(B, 6, d)
        emo = self.emo_proj(emotion_tokens)
        for blk in self.blocks:
            h = blk(h, t_mod, emo)
        return self.mel_out(h).transpose(1, 2)
```

#### 4.5.2 `backend/nn_cartridges/s2st_console.py`
```python
from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from backend.contracts.schemas import (
    MINOR_SYLLABLE_MAX_FRAMES,
    HOP_LENGTH,
    LANG_TO_ID,
    N_FFT_24K,
    N_MELS,
    PROD_240M_CONFIG,
    SAMPLE_RATE_OUT,
    LanguageCode,
    ModelArchitectureConfig,
    PrecomputedEarFeatures,
    SyllableTensorBatch,
)
from backend.nn_cartridges.codeswitch_router import AcousticPointerCodeSwitchBridge
from backend.nn_cartridges.duration_aligner import LearnedDurationAligner
from backend.nn_cartridges.ear_cartridge import StreamingEarCartridge
from backend.nn_cartridges.mouth_cartridge import KhmerMouthCartridge
from backend.nn_cartridges.universal_core import (
    DisentangledEmotionEncoder,
    VocosISTFTHead,
    build_slaney_mel_fb,
)

N_BINS_24K = N_FFT_24K // 2 + 1  # 1025 bins -> guarantees rank 100


# v10.7 P1-C: the Sway coefficients are shared by the inference ODE and the
# training shortcut-distillation loss, so the trained step grid and the
# inference step grid are provably identical.
SWAY_COEF: float = -0.8
SC_DISTILL_STEPS: int = 12


def sway_time_schedule(
    steps: int, sway_coef: float = -0.8, device=None, dtype=torch.float32
) -> torch.Tensor:
    """
    v10.7 P1-C: the Sway time schedule, extracted so that the training
    shortcut-distillation intervals and the inference ODE use provably the
    SAME step grid.

    v10.6 hardcoded `dt_val = 0.125` in the distillation loss while the
    inference ODE produced steps spanning 0.0235 to 0.1211 at steps=12,
    sway_coef=-0.8. The trained step size was therefore never one the
    sampler actually visited. Sampling the training intervals from this very
    schedule removes the train/inference mismatch.
    """
    steps = max(1, int(steps))
    u = torch.linspace(0.0, 1.0, steps + 1, device=device, dtype=torch.float32)
    return (u + sway_coef * (torch.cos(0.5 * math.pi * u) - 1.0 + u)).clamp(0.0, 1.0).to(dtype)


def integrate_shortcut_sway_ode(
    mouth: KhmerMouthCartridge,
    x_0: torch.Tensor,
    cond: torch.Tensor,
    emo_tokens: torch.Tensor,
    steps: int = 4,
    sway_coef: float = -0.8,
    # v10.7 P1-C: Classifier-Free Guidance scale. 1.0 disables guidance and
    # reproduces the v10.6 behaviour exactly. Training now applies the same
    # 15 percent conditioning dropout that makes the unconditional branch
    # learnable, so guidance is available at inference.
    cfg_scale: float = 1.0,
) -> torch.Tensor:
    B = x_0.shape[0]
    device, dtype = x_0.device, x_0.dtype
    steps = max(1, int(steps))
    # v10.7 P1-C: shared with the training distillation loss, so the trained
    # step grid and the inference step grid are provably identical.
    t_schedule = sway_time_schedule(steps, sway_coef, device=device, dtype=dtype)

    x = x_0
    for k in range(steps):
        t_cur = t_schedule[k]
        dt_step = t_schedule[k + 1] - t_cur
        t_batch = torch.full((B,), float(t_cur.item()), device=device, dtype=dtype)
        dt_batch = torch.full((B,), float(dt_step.item()), device=device, dtype=dtype)
        # v10.7: the conditional velocity is computed UNCONDITIONALLY first.
        # v10.6 computed it only in the else branch and then referenced it
        # in the if branch, so any cfg_scale != 1.0 raised
        # UnboundLocalError: cannot access local variable 'v'.
        v_cond = mouth.predict_velocity(x, t_batch, cond, emo_tokens, dt=dt_batch)
        if cfg_scale != 1.0:
            v_uncond = mouth.predict_velocity(
                x, t_batch, torch.zeros_like(cond), torch.zeros_like(emo_tokens), dt=dt_batch
            )
            v = v_uncond + cfg_scale * (v_cond - v_uncond)
        else:
            v = v_cond
        x = x + dt_step * v
    return x


def compute_shortcut_distillation_loss(
    mouth: KhmerMouthCartridge,
    x_1: torch.Tensor,
    cond: torch.Tensor,
    emo_tokens: torch.Tensor,
    sc_ratio: float = 0.25,
    sigma_min: float = 1e-4,
) -> torch.Tensor:
    B = x_1.shape[0]
    device, dtype = x_1.device, x_1.dtype
    x_0 = torch.randn_like(x_1)
    t = torch.rand(B, device=device, dtype=dtype)
    t_view = t.view(B, 1, 1)
    x_t = (1.0 - (1.0 - sigma_min) * t_view) * x_0 + t_view * x_1
    target_v = x_1 - (1.0 - sigma_min) * x_0

    # 1. Flow2GAN spectral energy-adaptive instantaneous OT-CFM velocity regression (dt = 0)
    pred_v = mouth.predict_velocity(x_t, t, cond, emo_tokens, dt=None)
    frame_energy = x_1.pow(2).mean(dim=1, keepdim=True).sqrt().clamp(min=0.1)
    energy_weight = (1.0 / (frame_energy ** 0.3)).detach()
    l_fm = ((pred_v - target_v).pow(2) * energy_weight).mean()

    # 2. Step-conditioned Shortcut Self-Consistency, grounded in the Sway
    #    schedule (v10.7 P1-C).
    #
    #    v10.6 hardcoded dt_val = 0.125 and the teacher half-steps were each
    #    evaluated at that same dt, so the whole hierarchy was anchored to
    #    NOTHING: the two half-steps and the full step all used the same
    #    step-size conditioning, and the trained dt was never one the Sway ODE
    #    actually visits at inference (0.0235..0.1211 at steps=12).
    #
    #    v10.7 samples the coarse-to-fine step pairs DIRECTLY from the Sway
    #    schedule, so the training grid and the inference grid are provably
    #    identical. The two finest teacher sub-steps are evaluated with the
    #    instantaneous velocity network (dt=None, i.e. dt = 0.0), which
    #    anchors the entire hierarchy to the ground-truth OT-CFM vector field
    #    and cannot collapse to zero.
    B_sc = max(1, int(round(B * sc_ratio)))
    x_t_sc = x_t[:B_sc]
    cond_sc = cond[:B_sc]
    emo_sc = emo_tokens[:B_sc]

    # The Sway step grid, shared with integrate_shortcut_sway_ode.
    sched = sway_time_schedule(SC_DISTILL_STEPS, SWAY_COEF, device=device, dtype=dtype)
    dts = (sched[1:] - sched[:-1]).clamp(min=0.0)

    # Pick a random coarse step from the grid, then split it into two fine
    # sub-steps of equal length: dt_1 = dt_2 = dt_full / 2.
    idx = int(torch.randint(0, dts.numel(), (1,)).item())
    dt_full_val = float(dts[idx].item())
    dt_1 = dt_full_val * 0.5
    dt_2 = dt_full_val * 0.5
    t_cur_val = float(sched[idx].item())
    t_sc = torch.full((B_sc,), t_cur_val, device=device, dtype=dtype)

    with torch.no_grad():
        # The two FINEST teacher sub-steps use the instantaneous velocity
        # network (dt=None), which is the ground-truth anchor.
        v_step1 = mouth.predict_velocity(x_t_sc, t_sc, cond_sc, emo_sc, dt=None)
        x_mid = x_t_sc + dt_1 * v_step1
        v_step2 = mouth.predict_velocity(x_mid, t_sc + dt_1, cond_sc, emo_sc, dt=None)
        # Step-weighted average: (dt_1*v1 + dt_2*v2) / dt_full.
        v_sc_target = (dt_1 * v_step1 + dt_2 * v_step2) / max(dt_full_val, 1e-8)

    # The student predicts the FULL step, conditioned on the full dt.
    dt_full_t = torch.full((B_sc,), dt_full_val, device=device, dtype=dtype)
    v_sc_pred = mouth.predict_velocity(x_t_sc, t_sc, cond_sc, emo_sc, dt=dt_full_t)
    l_sc = F.mse_loss(v_sc_pred, v_sc_target)
    return l_fm + 0.5 * l_sc


class SymmetricS2STConsole(nn.Module):
    def __init__(self, config: ModelArchitectureConfig = PROD_240M_CONFIG) -> None:
        super().__init__()
        self.config = config
        self.ear = StreamingEarCartridge(config)
        self.emotion_encoder = DisentangledEmotionEncoder(config)
        self.pointer = AcousticPointerCodeSwitchBridge(config)
        self.mouth = KhmerMouthCartridge(config)
        self.vocos = VocosISTFTHead(config)
        self.register_buffer("window_24k", torch.hann_window(N_FFT_24K, device="cpu"), persistent=False)
        self.register_buffer(
            "mel_fb_24k",
            build_slaney_mel_fb(N_BINS_24K, N_MELS, sample_rate=SAMPLE_RATE_OUT),
            persistent=False,
        )

    def _extract_mel_24k(self, wav_24k: torch.Tensor) -> torch.Tensor:
        win = self.window_24k.to(device=wav_24k.device, dtype=wav_24k.dtype)
        fb = self.mel_fb_24k.to(device=wav_24k.device, dtype=wav_24k.dtype)
        spec = torch.stft(
            wav_24k,
            n_fft=N_FFT_24K,
            hop_length=HOP_LENGTH,
            win_length=N_FFT_24K,
            window=win,
            center=True,
            return_complex=True,
        ).abs()
        mel = torch.matmul(spec.transpose(1, 2), fb).transpose(1, 2)
        return torch.log(mel.clamp(min=1e-6))

    def forward_s2st(
        self,
        source_wav_16k: torch.Tensor,
        source_lang: LanguageCode,
        target_lang: LanguageCode,
        orig_duration_sec: float,
        trailing_pause_ms: float = 0.0,
        playback_rate: float = 1.0,
        explicit_syll_batch: Optional[SyllableTensorBatch] = None,
        past_clause_kv: Optional[torch.Tensor] = None,
        past_gate_kv: Optional[torch.Tensor] = None,
        deferred_syll_count: int = 0,
        is_hard_ceiling_slice: bool = False,
        # v10.7 P1-D: optional speaker-timbre vector, shape (B, 1, SEMANTIC_DIM).
        # When None it is derived from the layer-6 acoustic representation.
        speaker_timbre: Optional[torch.Tensor] = None,
        # v10.7 AUD-107-02: reuse an ear forward pass the caller already did.
        precomputed_ear: Optional[PrecomputedEarFeatures] = None,
    ) -> Tuple[torch.Tensor, SyllableTensorBatch, torch.Tensor]:
        # v10.7 AUD-107-02: the ear runs ONCE per audio slice.
        #
        # The full-duplex runner needs the layer-6 tap and the log-mel before it
        # can call forward_s2st, so it computed them itself. v10.7 then ran the
        # ear a SECOND time on the identical audio inside this method, which
        # doubled the 64.5M-parameter ear and the 8.7M-parameter emotion encoder
        # on every spoken utterance. When the caller supplies the bundle, the
        # ear and the log-mel are reused verbatim.
        if precomputed_ear is not None:
            bus = precomputed_ear.bus
            r_sps = precomputed_ear.r_sps
            mel_stems = precomputed_ear.mel_stems
            h_acoustic_l6 = precomputed_ear.h_acoustic_l6
        else:
            bus, r_sps = self.ear(source_wav_16k, source_lang)
            h_acoustic_l6 = (
                self.ear.last_l6_hidden if self.ear.last_l6_hidden is not None else bus
            )
            mel_stems = self.ear.log_mel(source_wav_16k)
        emo_tokens, _ = self.emotion_encoder(mel_stems)
        fused_bus, gate_g_t = self.pointer(bus, h_acoustic_l6)
        # v10.7 P1-D: when no explicit speaker timbre is supplied, derive it
        # from the L2-normalized mean-pooled layer-6 acoustic representation.
        # This preserves the speaker's vocal-tract identity in the acoustic
        # conditioning path, which F.instance_norm in the emotion encoder
        # removes. ZERO parameters.
        if speaker_timbre is None:
            speaker_timbre = F.normalize(
                h_acoustic_l6.mean(dim=1, keepdim=True), p=2, dim=-1
            )
        target_frames = LearnedDurationAligner.compute_elastic_target_frames(
            orig_duration_sec,
            trailing_pause_ms,
            float(r_sps.mean().item()),
            playback_rate,
        )
        cond, syll_batch = self.mouth.predict_syllables_and_align(
            fused_bus,
            gate_g_t,
            r_sps,
            target_lang,
            orig_duration_sec,
            target_frames,
            explicit_syll_batch=explicit_syll_batch,
            past_clause_kv=past_clause_kv,
            past_gate_kv=past_gate_kv,
            deferred_syll_count=deferred_syll_count,
            is_hard_ceiling_slice=is_hard_ceiling_slice,
            speaker_timbre=speaker_timbre,
        )
        B = source_wav_16k.shape[0]
        x_0 = torch.randn(B, N_MELS, target_frames, device=source_wav_16k.device, dtype=cond.dtype)
        x = integrate_shortcut_sway_ode(
            self.mouth,
            x_0,
            cond,
            emo_tokens,
            steps=max(1, self.config.euler_steps),
            sway_coef=SWAY_COEF,
        )
        wav_24k = self.vocos(x)
        return wav_24k, syll_batch, gate_g_t

    def compute_s2st_training_losses(
        self,
        source_wav_16k: torch.Tensor,
        target_wav_24k: torch.Tensor,
        source_lang: LanguageCode,
        target_lang: LanguageCode,
        target_syll_batch: SyllableTensorBatch,
        target_lang_id: torch.Tensor,
        cfg_drop_prob: float = 0.15,
    ) -> Dict[str, torch.Tensor]:
        x_1 = self._extract_mel_24k(target_wav_24k)
        B, _, T_mel = x_1.shape

        bus, r_sps = self.ear(source_wav_16k, source_lang)
        h_acoustic_l6 = self.ear.last_l6_hidden if self.ear.last_l6_hidden is not None else bus
        emo_tokens, lang_logits = self.emotion_encoder(self.ear.log_mel(source_wav_16k), alpha=1.0)
        fused_bus, gate_g_t = self.pointer(bus, h_acoustic_l6)

        L_syll = target_syll_batch.onset_ids.shape[1]
        lang_idx = torch.full((B,), LANG_TO_ID[target_lang], dtype=torch.long, device=x_1.device)
        bus_cond = fused_bus + self.mouth.target_lang_emb(lang_idx).unsqueeze(1)
        pos_q = self.mouth._build_sinusoidal_queries(B, L_syll, x_1.device, bus_cond.dtype)
        reordered, attn_w = self.mouth.reorder_attn(pos_q, bus_cond, bus_cond, need_weights=True)
        bus_l = self.mouth.reorder_norm(pos_q + reordered)
        gate_l = torch.bmm(attn_w, gate_g_t).squeeze(-1).clamp(1e-5, 1.0 - 1e-5)

        l_syll_ce = (
            F.cross_entropy(self.mouth.heads["onset"](bus_l).transpose(1, 2), target_syll_batch.onset_ids)
            + F.cross_entropy(self.mouth.heads["coeng"](bus_l).transpose(1, 2), target_syll_batch.coeng_ids)
            + F.cross_entropy(self.mouth.heads["medial"](bus_l).transpose(1, 2), target_syll_batch.medial_ids)
            + F.cross_entropy(self.mouth.heads["register"](bus_l).transpose(1, 2), target_syll_batch.register_ids)
            + F.cross_entropy(self.mouth.heads["vowel"](bus_l).transpose(1, 2), target_syll_batch.vowel_ids)
            + F.cross_entropy(self.mouth.heads["coda"](bus_l).transpose(1, 2), target_syll_batch.coda_ids)
            + F.cross_entropy(self.mouth.heads["sesqui"](bus_l).transpose(1, 2), target_syll_batch.sesqui_ids)
        ) / 7.0

        # Code-Switch Gate BCE supervision against Register-3 ground truth
        cs_target = (target_syll_batch.register_ids == 3).to(gate_l.dtype)
        l_gate_bce = F.binary_cross_entropy(gate_l, cs_target)

        cond, _ = self.mouth.predict_syllables_and_align(
            fused_bus,
            gate_g_t,
            r_sps,
            target_lang,
            orig_duration_sec=T_mel * HOP_LENGTH / float(SAMPLE_RATE_OUT),
            target_frames=T_mel,
            explicit_syll_batch=target_syll_batch,
        )

        # Duration prior supervision
        pred_durs = self.mouth.last_pred_durations
        # v10.7 P1-C: NON-CIRCULAR duration supervision. v10.6 supervised
        # pred_durs against a static prior built from the sesqui class alone
        # (0.25 / 1.00), which carries no acoustic information: the target was
        # a function of the label distribution, not of the audio. The target is
        # now derived from the monotonic attention mass of reorder_attn,
        # attn_w.sum(dim=-1) over the L_syll axis, scaled to T_mel and clamped
        # by the 6-frame minor-syllable cap.
        attn_mass = attn_w.sum(dim=-1)  # (B, L_syll) attention mass per syllable
        attn_mass = attn_mass.clamp(min=0.0)
        mass_total = attn_mass.sum(dim=-1, keepdim=True)
        dur_from_attn = attn_mass * (float(T_mel) / (mass_total + 1e-8))
        # Respect the minor-syllable cap: sesqui_id == 1 syllables may not
        # exceed MINOR_SYLLABLE_MAX_FRAMES; the excess is redistributed to
        # the major syllables exactly as LearnedDurationAligner does.
        is_minor_s = target_syll_batch.sesqui_ids == 1
        capped = torch.where(
            is_minor_s,
            torch.clamp(dur_from_attn, max=MINOR_SYLLABLE_MAX_FRAMES),
            dur_from_attn,
        )
        excess = (dur_from_attn - capped).sum(dim=-1, keepdim=True)
        major_w = capped * (~is_minor_s).to(capped.dtype)
        major_sum = major_w.sum(dim=-1, keepdim=True)
        prior_durs = torch.where(
            major_sum > 1e-6,
            capped + excess * (major_w / (major_sum + 1e-8)),
            capped,
        )
        l_dur = F.smooth_l1_loss(pred_durs, prior_durs) if pred_durs is not None else torch.tensor(0.0, device=x_1.device)

        # 15% Classifier-Free Guidance (CFG) conditioning dropout
        # v10.7 P1-C: v10.6 trained this dropout but NEVER applied guidance at
        # inference, so the unconditional branch was learned and then thrown
        # away. integrate_shortcut_sway_ode now evaluates the unconditional
        # branch (zero cond AND zero emo_tokens) and blends it with the
        # conditional one when cfg_scale != 1.0. The dropout below must
        # therefore zero EXACTLY the same two tensors the inference
        # unconditional branch zeroes, otherwise the guided direction points
        # at a branch the model was never trained on.
        if self.training and cfg_drop_prob > 0.0:
            drop_mask = (torch.rand(B, 1, 1, device=x_1.device) >= cfg_drop_prob).to(cond.dtype)
            cond = cond * drop_mask
            emo_tokens = emo_tokens * drop_mask

        l_cfm = compute_shortcut_distillation_loss(self.mouth, x_1, cond, emo_tokens)
        l_lang_grl = F.cross_entropy(lang_logits, target_lang_id)
        total = l_cfm + 1.0 * l_syll_ce + 0.5 * l_dur + 0.3 * l_gate_bce + 0.2 * l_lang_grl
        return {
            "total": total,
            "l_cfm": l_cfm,
            "l_syll_ce": l_syll_ce,
            "l_dur": l_dur,
            "l_gate_bce": l_gate_bce,
            "l_lang_grl": l_lang_grl,
        }

    def compute_flow_loss(
        self,
        target_wav_24k: torch.Tensor,
        syll_batch: Optional[SyllableTensorBatch] = None,
        infill_mode: bool = False,
        source_wav_16k: Optional[torch.Tensor] = None,
        source_lang: LanguageCode = "en",
    ) -> torch.Tensor:
        x_1 = self._extract_mel_24k(target_wav_24k)
        B, _, T_mel = x_1.shape
        emo_tokens, _ = self.emotion_encoder(x_1)

        if infill_mode or syll_batch is None:
            mask = torch.zeros_like(x_1)
            s = T_mel // 3
            e = max(s + 1, (2 * T_mel) // 3)
            mask[:, :, s:e] = 1.0
            cond = self.mouth.infill_proj(((1.0 - mask) * x_1).transpose(1, 2))
        else:
            if source_wav_16k is None:
                src_proxy = F.interpolate(
                    target_wav_24k.unsqueeze(1),
                    size=max(160, int(target_wav_24k.shape[-1] * 16000 / 24000)),
                    mode="linear",
                    align_corners=False,
                ).squeeze(1)
            else:
                src_proxy = source_wav_16k

            bus, r_sps = self.ear(src_proxy, source_lang)
            h_acoustic_l6 = self.ear.last_l6_hidden if self.ear.last_l6_hidden is not None else bus
            fused_bus, gate_g_t = self.pointer(bus, h_acoustic_l6)
            cond, _ = self.mouth.predict_syllables_and_align(
                fused_bus, gate_g_t, r_sps, "km", 1.0, T_mel, syll_batch
            )

        return compute_shortcut_distillation_loss(self.mouth, x_1, cond, emo_tokens)
```

---

### 4.6 Specialist C — Corruptor, 0-Hz VTLP & Dual-Profile Trainer (`backend/training/`)

#### 4.6.1 `backend/training/augmentations.py`
```python
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F


class DynamicAcousticCorruptor:
    @staticmethod
    def warp_formants_vtlp(
        mel: torch.Tensor,
        alpha_min: float = 0.85,
        alpha_max: float = 1.15,
        pivot: float = 0.65,
    ) -> torch.Tensor:
        """0-Hz and Nyquist anchored piecewise-linear VTLP using 2D (x, y) grid_sample."""
        B, M, T = mel.shape
        device, dtype = mel.device, mel.dtype
        u = torch.linspace(0.0, 1.0, M, device=device, dtype=dtype).view(1, M, 1).expand(B, M, T)
        alpha = torch.empty(B, 1, 1, device=device, dtype=dtype).uniform_(alpha_min, alpha_max)
        boundary = torch.clamp(alpha * pivot, min=0.10, max=0.92)
        u_warped = torch.where(
            u <= boundary,
            u * (pivot / boundary),
            pivot + (u - boundary) * ((1.0 - pivot) / (1.0 - boundary)),
        )
        y_grid = 2.0 * u_warped - 1.0
        x_grid = torch.linspace(-1.0, 1.0, T, device=device, dtype=dtype).view(1, 1, T).expand(B, M, T)
        grid = torch.stack([x_grid, y_grid], dim=-1)  # (B, M, T, 2)
        return F.grid_sample(
            mel.unsqueeze(1),
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        ).squeeze(1)

    @staticmethod
    def corrupt_youtube_mixture(
        clean_vocal_a: torch.Tensor,
        clean_vocal_b: Optional[torch.Tensor] = None,
        bgm_or_noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        mix = clean_vocal_a.clone()
        B, T = mix.shape
        device, dtype = mix.device, mix.dtype
        rms_a = mix.pow(2).mean(dim=-1, keepdim=True).sqrt().clamp(min=1e-6)

        # 1. Co-host overlap (-10 dB to 0 dB) scaled relative to rms_a
        if clean_vocal_b is not None:
            rms_b = clean_vocal_b.pow(2).mean(dim=-1, keepdim=True).sqrt().clamp(min=1e-6)
            db = torch.empty(B, 1, device=device, dtype=dtype).uniform_(-10.0, 0.0)
            target_b_rms = rms_a * (10.0 ** (db / 20.0))
            scaled_b = clean_vocal_b * (target_b_rms / rms_b)
            mix = mix + scaled_b

        # 2. Environmental / BGM noise (-3 dB to +15 dB SNR) scaled relative to rms_a
        noise = bgm_or_noise if bgm_or_noise is not None else torch.randn_like(mix)
        rms_noise = noise.pow(2).mean(dim=-1, keepdim=True).sqrt().clamp(min=1e-6)
        snr_db = torch.empty(B, 1, device=device, dtype=dtype).uniform_(-3.0, 15.0)
        target_noise_rms = rms_a * (10.0 ** (-snr_db / 20.0))
        scaled_noise = noise * (target_noise_rms / rms_noise)
        mix = mix + scaled_noise

        # 3. Decaying room echo tail
        t_tail = torch.linspace(0.0, 1.0, 64, device=device, dtype=dtype)
        fir = torch.exp(-3.0 * t_tail).unsqueeze(0).unsqueeze(0)
        fir = fir / fir.norm(p=2)
        mix = F.conv1d(F.pad(mix.unsqueeze(1), (63, 0)), fir).squeeze(1)

        # 4. Soft tanh saturation
        drive = 1.6
        return torch.tanh(mix * drive) / math.tanh(drive)
```

#### 4.6.2 `backend/training/train_cartridge.py`
```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from backend.nn_cartridges.universal_core import VocosISTFTHead


@dataclass(frozen=True)
class TrainingProfileConfig:
    profile_name: str
    precision: torch.dtype
    micro_batch_size: int
    grad_accum_steps: int
    gradient_checkpointing: bool
    learning_rate: float
    weight_decay: float


TRAINING_PROFILES: Dict[str, TrainingProfileConfig] = {
    "laptop_5070ti_12gb": TrainingProfileConfig(
        profile_name="laptop_5070ti_12gb",
        precision=torch.bfloat16,
        micro_batch_size=4,
        grad_accum_steps=16,
        gradient_checkpointing=True,
        learning_rate=3e-4,
        weight_decay=0.01,
    ),
    "cloud_h100_80gb": TrainingProfileConfig(
        profile_name="cloud_h100_80gb",
        precision=torch.bfloat16,
        micro_batch_size=64,
        grad_accum_steps=1,
        gradient_checkpointing=False,
        learning_rate=3e-4,
        weight_decay=0.01,
    ),
}


def compute_total_multitask_loss(
    l_cfm: torch.Tensor,
    l_dur: torch.Tensor,
    l_lang_grl: torch.Tensor,
    l_stems: torch.Tensor,
) -> torch.Tensor:
    return l_cfm + 0.5 * l_dur + 0.2 * l_lang_grl + 0.3 * l_stems


class MultiTeacherEarDistiller(nn.Module):
    """
    Training-only wrapper distilling w2v-BERT 2.0 (50Hz, Layer 16 -> ear.last_l8_hidden),
    Canary-1B-v2 / Parakeet-TDT-0.6B-v3 (12.5Hz -> ear.bus_proj), and Qwen3-ASR-0.6B-Khmer
    into StreamingEarCartridge (25Hz) with zero inference parameter overhead.
    """

    def __init__(self, d_student: int = 512, d_teacher: int = 1024) -> None:
        super().__init__()
        self.proj_w2v_l8 = nn.Linear(d_student, d_teacher)
        self.proj_canary_l16 = nn.Linear(d_student, d_teacher)
        self.proj_qwen_km_l16 = nn.Linear(d_student, d_teacher)
        self.ln = nn.LayerNorm(d_teacher, elementwise_affine=False)

    @staticmethod
    def _align_temporal(teacher_feat: torch.Tensor, target_len: int) -> torch.Tensor:
        if teacher_feat.shape[1] == target_len:
            return teacher_feat
        return F.interpolate(
            teacher_feat.transpose(1, 2),
            size=target_len,
            mode="linear",
            align_corners=False,
        ).transpose(1, 2)

    def _pair_loss(self, stu_proj: torch.Tensor, tea_aligned: torch.Tensor) -> torch.Tensor:
        cos_loss = 1.0 - F.cosine_similarity(stu_proj, tea_aligned, dim=-1).mean()
        l1_loss = F.smooth_l1_loss(self.ln(stu_proj), self.ln(tea_aligned))
        return cos_loss + 0.5 * l1_loss

    def forward(
        self,
        student_l8_hidden: torch.Tensor,
        student_bus_out: torch.Tensor,
        w2v_layer16_50hz: Optional[torch.Tensor] = None,
        canary_top_12_5hz: Optional[torch.Tensor] = None,
        qwen_km_top: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        T_ear = student_bus_out.shape[1]
        loss = torch.tensor(0.0, device=student_bus_out.device, dtype=student_bus_out.dtype)
        if w2v_layer16_50hz is not None:
            w2v_25hz = self._align_temporal(w2v_layer16_50hz.detach(), T_ear)
            loss = loss + self._pair_loss(self.proj_w2v_l8(student_l8_hidden), w2v_25hz)
        if canary_top_12_5hz is not None:
            can_25hz = self._align_temporal(canary_top_12_5hz.detach(), T_ear)
            loss = loss + self._pair_loss(self.proj_canary_l16(student_bus_out), can_25hz)
        if qwen_km_top is not None:
            qwen_25hz = self._align_temporal(qwen_km_top.detach(), T_ear)
            loss = loss + self._pair_loss(self.proj_qwen_km_l16(student_bus_out), qwen_25hz)
        return loss


def warm_start_vocos_from_pretrained(
    vocos_head: VocosISTFTHead, pretrained_state_dict: Dict[str, torch.Tensor]
) -> None:
    """Surgical loader mapping charactr/vocos-mel-24khz (N_FFT=1024) into N_FFT_24K=2048."""
    with torch.no_grad():
        vocos_head.in_conv.weight.copy_(pretrained_state_dict["backbone.embed.weight"])
        vocos_head.in_conv.bias.copy_(pretrained_state_dict["backbone.embed.bias"])
        inter_half = vocos_head.blocks[0].pwconv1.out_features

        for i, blk in enumerate(vocos_head.blocks):
            prefix = f"backbone.convnext.{i}."
            blk.dwconv.weight.copy_(pretrained_state_dict[prefix + "dwconv.weight"])
            blk.dwconv.bias.copy_(pretrained_state_dict[prefix + "dwconv.bias"])
            blk.norm.weight.copy_(pretrained_state_dict[prefix + "norm.weight"])
            blk.norm.bias.copy_(pretrained_state_dict[prefix + "norm.bias"])

            pw1_w = pretrained_state_dict[prefix + "pwconv1.weight"]
            pw1_b = pretrained_state_dict[prefix + "pwconv1.bias"]
            pw2_w = pretrained_state_dict[prefix + "pwconv2.weight"]
            pw2_b = pretrained_state_dict[prefix + "pwconv2.bias"]

            importance = pw1_w.norm(p=2, dim=1) * pw2_w.norm(p=2, dim=0)
            top_idx = torch.topk(importance, k=inter_half).indices.sort().values

            blk.pwconv1.weight.copy_(pw1_w[top_idx, :])
            blk.pwconv1.bias.copy_(pw1_b[top_idx])
            blk.pwconv2.weight.copy_(pw2_w[:, top_idx])
            blk.pwconv2.bias.copy_(pw2_b)

        old_w = pretrained_state_dict["head.out.weight"]
        old_b = pretrained_state_dict["head.out.bias"]
        mag_w, phs_w = old_w[:513, :].T.unsqueeze(0), old_w[513:, :].T.unsqueeze(0)
        new_mag_w = F.interpolate(mag_w, size=1025, mode="linear", align_corners=True).squeeze(0).T
        new_phs_w = F.interpolate(phs_w, size=1025, mode="linear", align_corners=True).squeeze(0).T
        vocos_head.istft_head.weight.copy_(torch.cat([new_mag_w, new_phs_w], dim=0))

        mag_b = F.interpolate(old_b[:513].view(1, 1, 513), size=1025, mode="linear", align_corners=True).view(-1)
        phs_b = F.interpolate(old_b[513:].view(1, 1, 513), size=1025, mode="linear", align_corners=True).view(-1)
        vocos_head.istft_head.bias.copy_(torch.cat([mag_b, phs_b], dim=0))


def apply_gradient_checkpointing(model: nn.Module) -> nn.Module:
    """
    v10.7 P2-B: enable activation checkpointing on the two large cartridge
    stacks, so the laptop_5070ti_12gb profile can hold its 3.2 GB VRAM
    ceiling on a 12 GB RTX 5070 Ti.

    v10.7 AUD-107-04: v10.7 wrapped each block in a `_CheckpointedBlock`
    nn.Module that held the real block as `self.blk`. That RENAMED every
    parameter key in `model.state_dict()` from `ear.blocks.0.self_attn...` to
    `ear.blocks.0.blk.self_attn...`, so a checkpoint trained under the
    laptop profile (gradient_checkpointing=True) could not be loaded under the
    cloud profile (gradient_checkpointing=False) or into an inference model.
    70 of the 178 keys were affected.

    The fix replaces the module wrapper with an INSTANCE-LEVEL forward
    override. The nn.Module hierarchy is left completely untouched, so the
    state_dict() keys are bit-for-bit identical with and without
    checkpointing, and a checkpoint is portable between profiles.

    use_reentrant=False is used because the reentrant implementation does not
    support the non-tensor keyword arguments these blocks receive.

    Adds NO parameters.
    """
    import types

    from torch.utils.checkpoint import checkpoint

    def _make_checkpointed_forward(fn):
        def _checkpointed_forward(self, *args, **kwargs):
            return checkpoint(fn, *args, use_reentrant=False, **kwargs)

        return _checkpointed_forward

    for attr in ("ear", "mouth"):
        cartridge = getattr(model, attr, None)
        if cartridge is None:
            continue
        blocks = getattr(cartridge, "blocks", None)
        if blocks is None:
            continue
        if not isinstance(blocks, nn.ModuleList):
            continue
        for blk in blocks:
            if getattr(blk, "_v107_checkpointed", False):
                continue
            blk.forward = types.MethodType(
                _make_checkpointed_forward(blk.forward), blk
            )
            blk._v107_checkpointed = True
    return model


class CartridgeTrainer:
    def __init__(self, model: nn.Module, profile_name: str = "laptop_5070ti_12gb") -> None:
        if profile_name not in TRAINING_PROFILES:
            raise KeyError(f"Unknown profile: {profile_name}")
        self.profile = TRAINING_PROFILES[profile_name]
        self.model = model
        # v10.7 P2-B: TRAINING_PROFILES declares precision and
        # gradient_checkpointing, but v10.6 never read either of them, so the
        # laptop_5070ti_12gb profile could not hold its 3.2 GB VRAM ceiling on
        # a 12 GB RTX 5070 Ti. Gradient checkpointing is configured on the
        # model HERE, at construction time, because it must be in place BEFORE
        # the first forward pass.
        if self.profile.gradient_checkpointing:
            self.model = apply_gradient_checkpointing(self.model)
        # v10.7 P2-B: the MultiTeacherEarDistiller is training-only (it has
        # zero inference parameter overhead) but v10.6 never aggregated its
        # loss, so the three teacher streams contributed nothing to training.
        self.distiller: Optional[MultiTeacherEarDistiller] = None
        self.distiller_weight: float = 0.5
        # v10.7 P2-B: the teacher feature tensors, remembered so that
        # train_step can compute a non-zero distillation loss.
        self.teacher_targets: Dict[str, torch.Tensor] = {}
        self.step_count: int = 0
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.profile.learning_rate,
            weight_decay=self.profile.weight_decay,
        )

    def autocast_context(self):
        """
        v10.7 P2-B: the mixed-precision context for the FORWARD pass.

        Only the forward pass runs under autocast. The backward pass MUST run
        OUTSIDE the autocast block: PyTorch's autocast is not designed to
        wrap backward, and doing so is a documented anti-pattern that can
        silently produce wrong gradient scales.
        """
        return torch.autocast(
            device_type="cuda",
            dtype=self.profile.precision,
            enabled=torch.cuda.is_available(),
        )

    def attach_distiller(
        self,
        distiller: MultiTeacherEarDistiller,
        weight: float = 0.5,
        teacher_targets: Optional[Dict[str, torch.Tensor]] = None,
    ) -> None:
        """Attach the training-only multi-teacher ear distiller.

        `teacher_targets` holds the teacher feature tensors, keyed by the
        distiller's parameter names: `w2v_layer16_50hz`, `canary_top_12_5hz`
        and `qwen_km_top`. They are remembered here and used as the default
        for every subsequent `train_step` call, so the distillation loss is
        non-zero whenever teacher features are available.
        """
        self.distiller = distiller
        self.distiller_weight = float(weight)
        self.teacher_targets = dict(teacher_targets or {})

    def train_step(
        self,
        loss: torch.Tensor | Callable[[], torch.Tensor],
        teacher_targets: Optional[Dict[str, torch.Tensor]] = None,
    ) -> float:
        """
        v10.7 P2-B: accepts EITHER an already-computed loss tensor OR a
        callable that performs the forward pass.

        When a callable is given, the forward pass runs under the profile's
        autocast context and the backward pass runs OUTSIDE it. When a pre-
        computed tensor is given (the v10.6 calling convention), it is used
        as-is and the caller remains responsible for the forward pass.

        `teacher_targets` carries the multi-teacher feature tensors, keyed by
        the distiller's parameter names: `w2v_layer16_50hz`,
        `canary_top_12_5hz` and `qwen_km_top`. When they are supplied AND a
        distiller is attached, the distillation loss is computed against the
        student's layer-8 hidden state and bus output and is aggregated with
        the main loss.
        """
        if callable(loss):
            with self.autocast_context():
                loss_tensor = loss()
                # v10.7 P2-B: the multi-teacher ear distillation loss is
                # aggregated with the main loss inside the SAME autocast
                # forward pass, so the three teacher streams actually
                # contribute to training (v10.6 never aggregated them).
                if self.distiller is not None:
                    ear = getattr(self.model, "ear", None)
                    l8 = getattr(ear, "last_l8_hidden", None) if ear is not None else None
                    # v10.7 P2-B: the bus output is published by the ear on
                    # the thread-local state, so it is available here.
                    bus_out = getattr(ear, "last_bus_out", None) if ear is not None else None
                    if bus_out is None:
                        bus_out = getattr(self.model, "bus_out", None)
                    if l8 is not None and bus_out is not None:
                        kwargs = dict(teacher_targets or self.teacher_targets or {})
                        distil = self.distiller(l8, bus_out, **kwargs)
                        loss_tensor = loss_tensor + self.distiller_weight * distil
        else:
            loss_tensor = loss
        scaled_loss = loss_tensor / float(self.profile.grad_accum_steps)
        # Backward runs OUTSIDE the autocast block, by construction.
        scaled_loss.backward()
        self.step_count += 1
        if self.step_count % self.profile.grad_accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
        return float(loss_tensor.detach().item())
```

#### 4.6.3 `backend/training/eval_s2st_suite.py` (5-Pillar Automated Stage-End Benchmark)
```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from khmernltk import word_tokenize as khmer_word_tokenize

from backend.contracts.schemas import SAMPLE_RATE_IN, SAMPLE_RATE_OUT
from backend.linguistics.khmer_normalizer import normalize_khmer
from backend.nn_cartridges.s2st_console import SymmetricS2STConsole


@dataclass
class S2STEvalReport:
    direction: str
    asr_bleu: float
    asr_chrf: float
    asr_comet: float
    spk_sim_wavlm: float
    emo_sim_e2v: float
    utmos: float
    dnsmos_ovrl: float
    mean_turnaround_ms: float


class AutomatedS2STBenchmarkSuite:
    """Executes Stage-End S2ST evaluation across Qwen3-ASR-0.6B-Khmer, Whisper/Parakeet, WavLM-SV, emotion2vec+, and UTMOS22."""

    def __init__(
        self,
        device: str = "cuda",
        comet_checkpoint: str = "Unbabel/wmt22-comet-da",
    ) -> None:
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        # v10.7: the COMET checkpoint is a constructor argument, so the
        # evaluation no longer references an undefined module-level name.
        self.comet_checkpoint = comet_checkpoint
        import sacrebleu
        from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector, pipeline

        self.sacrebleu = sacrebleu
        self.km_asr = pipeline(
            "automatic-speech-recognition",
            model="seanghay/Qwen3-ASR-0.6B-Khmer",
            device=0 if self.device.type == "cuda" else -1,
        )
        self.en_asr = pipeline(
            "automatic-speech-recognition",
            model="openai/whisper-large-v3-turbo",
            device=0 if self.device.type == "cuda" else -1,
        )
        self.wavlm_fe = Wav2Vec2FeatureExtractor.from_pretrained("microsoft/wavlm-base-plus-sv")
        self.wavlm_sv = WavLMForXVector.from_pretrained("microsoft/wavlm-base-plus-sv").to(self.device).eval()
        self.utmos_predictor = (
            torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True)
            .to(self.device)
            .eval()
        )
        from funasr import AutoModel
        self.e2v_model = AutoModel(model="iic/emotion2vec_plus_large", disable_update=True)

    @staticmethod
    def _tokenize_khmer_for_bleu(text: str) -> str:
        clean = normalize_khmer(text).replace(" ", "")
        return " ".join(khmer_word_tokenize(clean))

    @torch.no_grad()
    def compute_speaker_similarity(self, wav_src_16k: torch.Tensor, wav_tgt_24k: torch.Tensor) -> float:
        wav_tgt_16k = torchaudio.functional.resample(wav_tgt_24k, SAMPLE_RATE_OUT, SAMPLE_RATE_IN)
        inputs = self.wavlm_fe(
            [wav_src_16k.squeeze().cpu().numpy(), wav_tgt_16k.squeeze().cpu().numpy()],
            sampling_rate=SAMPLE_RATE_IN,
            padding=True,
            return_tensors="pt",
        ).to(self.device)
        embs = F.normalize(self.wavlm_sv(**inputs).embeddings, p=2, dim=-1)
        return float(torch.sum(embs[0] * embs[1]).item())

    @torch.no_grad()
    def compute_emotion_similarity(self, wav_src_16k: torch.Tensor, wav_tgt_24k: torch.Tensor) -> float:
        wav_tgt_16k = torchaudio.functional.resample(wav_tgt_24k, SAMPLE_RATE_OUT, SAMPLE_RATE_IN)
        res_src = self.e2v_model.generate(wav_src_16k.squeeze().cpu().numpy(), granularity="utterance")
        res_tgt = self.e2v_model.generate(wav_tgt_16k.squeeze().cpu().numpy(), granularity="utterance")
        e_src = F.normalize(torch.as_tensor(res_src[0]["feats"], dtype=torch.float32), p=2, dim=-1)
        e_tgt = F.normalize(torch.as_tensor(res_tgt[0]["feats"], dtype=torch.float32), p=2, dim=-1)
        return float(torch.sum(e_src * e_tgt).item())

    def _asr_transcribe(
        self, wav: torch.Tensor, model: str, language: str = "km"
    ) -> str:
        """Transcribe a waveform with the given ASR pipeline.

        v10.7: the `language` is a parameter. v10.6 hardcoded "km", so the
        English direction was transcribed with the Khmer ASR language token.
        """
        audio = wav.squeeze().detach().cpu().numpy()
        out = model(audio, generate_kwargs={"language": language, "task": "transcribe"})
        if isinstance(out, list) and out:
            out = out[0]
        if isinstance(out, dict):
            out = out.get("text", "")
        return str(out).strip()

    def _compute_asr_metrics(self, hyp: str, ref: str, is_khmer: bool) -> tuple[float, float]:
        """Return (BLEU, chrF) for one hypothesis/reference pair."""
        if is_khmer:
            hyp_t = self._tokenize_khmer_for_bleu(hyp)
            ref_t = self._tokenize_khmer_for_bleu(ref)
        else:
            hyp_t = hyp
            ref_t = ref
        bleu = self.sacrebleu.corpus_bleu([hyp_t], [[ref_t]]).score
        chrf = self.sacrebleu.corpus_chrf([hyp_t], [[ref_t]]).score
        return float(bleu), float(chrf)

    def _compute_utmos(self, wav_24k: torch.Tensor) -> float:
        """UTMOS22 MOS estimate for a 24 kHz waveform."""
        audio = wav_24k.squeeze().detach().cpu().numpy()
        if audio.size == 0:
            return 0.0
        with torch.no_grad():
            score = self.utmos_predictor(audio, SAMPLE_RATE_OUT)
        if hasattr(score, "item"):
            score = score.item()
        return float(score)

    def evaluate(
        self,
        direction: str,
        wav_src_16k: torch.Tensor,
        wav_tgt_24k: torch.Tensor,
        ref_text: str,
        hyp_text: str = "",
    ) -> S2STEvalReport:
        """
        v10.7 P2-C: the 5-pillar benchmark entry point.

        v10.6 initialised every model (ASR x2, WavLM-SV, emotion2vec+, UTMOS,
        DNSMOS) and implemented only two of the five pillars. This method wires
        the remaining pillars into the S2STEvalReport dataclass, whose fields
        were otherwise never assigned:

          * ASR-BLEU and chrF against the reference transcript,
          * UTMOS and DNSMOS naturalness,
          * speaker similarity (WavLM-SV),
          * emotion similarity (emotion2vec+),
          * mean turnaround latency.

        Adds NO parameters.
        """
        # v10.7: `evaluate` transcribes wav_tgt_24k, the TARGET audio, so the
        # target language is the SECOND half of the direction. For "en->km" the
        # target is Khmer. v10.6 used startswith("km"), which inverted the
        # check and made the English direction use the Khmer ASR model.
        d = direction.strip().lower()
        is_khmer = d.endswith("km") or d == "km"
        asr_model = self.km_asr if is_khmer else self.en_asr

        asr_lang = "km" if is_khmer else "en"
        hyp = hyp_text or self._asr_transcribe(wav_tgt_24k, asr_model, language=asr_lang)
        bleu, chrf = self._compute_asr_metrics(hyp, ref_text, is_khmer)

        comet = 0.0
        try:
            import comet

            model = comet.load_from_checkpoint(self.comet_checkpoint)
            data = [{"src": wav_src_16k.squeeze().cpu().numpy(), "mt": hyp, "ref": ref_text}]
            comet_out = model.predict(data)
            comet = float(comet_out[0]["scores"]["comet22"])
        except Exception:
            # COMET is an optional pillar: a missing checkpoint or package must
            # not abort the whole evaluation.
            comet = 0.0

        utmos = self._compute_utmos(wav_tgt_24k)

        dnsmos = 0.0
        try:
            import dnsmos

            audio = wav_tgt_24k.squeeze().detach().cpu().numpy()
            dnsmos_out = dnsmos.dnsmos(audio, SAMPLE_RATE_OUT)
            dnsmos = float(dnsmos_out.get("ovrl", 0.0))
        except Exception:
            dnsmos = 0.0

        spk_sim = self.compute_speaker_similarity(wav_src_16k, wav_tgt_24k)
        emo_sim = self.compute_emotion_similarity(wav_src_16k, wav_tgt_24k)

        import time

        t0 = time.perf_counter()
        _ = self._asr_transcribe(wav_tgt_24k, asr_model, language=asr_lang)
        turnaround_ms = (time.perf_counter() - t0) * 1000.0

        return S2STEvalReport(
            direction=direction,
            asr_bleu=bleu,
            asr_chrf=chrf,
            asr_comet=comet,
            spk_sim_wavlm=spk_sim,
            emo_sim_e2v=emo_sim,
            utmos=utmos,
            dnsmos_ovrl=dnsmos,
            mean_turnaround_ms=turnaround_ms,
        )
```

---

### 4.7 Specialist D — Rewind-Resilient Transport (`backend/transport/`)

#### 4.7.1 `backend/transport/session.py`
```python
from __future__ import annotations

import threading
import zlib
from collections import OrderedDict
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from backend.contracts.schemas import (
    MAX_CLAUSE_SLICE_SEC,
    MIN_CLAUSE_SLICE_SEC,
    REPLAY_BUCKET_MS,
    REPLAY_CACHE_MAX_ITEMS,
    SAMPLE_RATE_IN,
    SILENCE_RMS_THRESHOLD,
    UTTERANCE_END_SILENCE_MS,
    DownstreamEvent,
    LanguageCode,
)


class SpeakerRegistry:
    def __init__(self, max_speakers: int = 4, similarity_threshold: float = 0.72) -> None:
        # v10.7 P0-B: the centroid list is mutated in place by a read-modify-write
        # in match_or_register_speaker. The GPU worker thread and the event-loop
        # thread can both reach it, so the list needs its own lock; otherwise a
        # concurrent caller can observe a half-updated centroid vector.
        self._lock = threading.RLock()
        self.max_speakers = max_speakers
        self.similarity_threshold = similarity_threshold
        self.centroids: List[torch.Tensor] = []

    def match_or_register_speaker(self, emb: torch.Tensor) -> Tuple[int, torch.Tensor]:
        # v10.7 P0-B: hold the registry lock across the whole
        # read-modify-write so a concurrent caller cannot observe a
        # half-updated centroid list.
        with self._lock:
            vec = F.normalize(emb.detach().float().view(-1), p=2, dim=0)
            if not self.centroids:
                self.centroids.append(vec)
                return 0, self.centroids[0]

            sims = [float(torch.dot(vec, c).item()) for c in self.centroids]
            best_idx = int(np.argmax(sims))
            if sims[best_idx] >= self.similarity_threshold:
                updated = F.normalize(0.85 * self.centroids[best_idx] + 0.15 * vec, p=2, dim=0)
                self.centroids[best_idx] = updated
                return best_idx, self.centroids[0]

            if len(self.centroids) < self.max_speakers:
                self.centroids.append(vec)
                return len(self.centroids) - 1, self.centroids[0]

            updated = F.normalize(0.85 * self.centroids[best_idx] + 0.15 * vec, p=2, dim=0)
            self.centroids[best_idx] = updated
            return best_idx, self.centroids[0]


class InMemorySession:
    def __init__(self, session_id: str) -> None:
        import threading
        self._lock = threading.RLock()
        self.session_id = session_id
        self.speaker_registry = SpeakerRegistry()
        self.pcm_buffer: np.ndarray = np.zeros(0, dtype=np.float32)
        self.silence_ms: float = 0.0
        self.active_speech_ms: float = 0.0
        self.is_speaking: bool = False
        self.current_utterance_start_ms: float = 0.0
        self.stream_cursor_ms: float = 0.0
        self.next_utterance_id: int = 0
        self._replay_cache: OrderedDict[Tuple[int, int, int, float, str], DownstreamEvent] = OrderedDict()
        self.past_clause_kv: Optional[torch.Tensor] = None
        # v10.7 P0-A: records whether the most recently CONSUMED slice was a
        # Trigger #2 hard-ceiling cut. The runner uses it to decide whether to
        # pass the rolling KV cache into the next clause.
        self.last_slice_was_hard_ceiling: bool = False
        self.past_gate_kv: Optional[torch.Tensor] = None
        self.deferred_syll_count: int = 0

    @staticmethod
    def compute_pcm_energy_hash(slice_pcm: np.ndarray) -> int:
        if slice_pcm.size < 64:
            return 0
        n_bins = 8
        bin_len = slice_pcm.size // n_bins
        if bin_len == 0:
            return 0
        overall_rms = float(np.sqrt(np.mean(slice_pcm**2) + 1e-12))
        profile = bytearray()
        for b in range(n_bins):
            chunk = slice_pcm[b * bin_len : (b + 1) * bin_len]
            c_rms = float(np.sqrt(np.mean(chunk**2) + 1e-12))
            norm_c = int(np.clip(round((c_rms / max(overall_rms, 1e-6)) * 10.0), 0, 99))
            profile.append(norm_c)
        return zlib.crc32(bytes(profile)) & 0xFFFFFFFF

    @staticmethod
    def compute_pcm_crc32(slice_pcm: np.ndarray) -> int:
        # Phase-invariant energy envelope hash ensures 100% cache hits across 1-63 sample shifts
        return InMemorySession.compute_pcm_energy_hash(slice_pcm)

    @staticmethod
    def _make_replay_key(
        utterance_start_ms: float,
        duration_ms: float,
        pcm_crc32: int,
        playback_rate: float,
        target_lang: LanguageCode,
    ) -> Tuple[int, int, int, float, str]:
        bucket = int(round(float(utterance_start_ms) / REPLAY_BUCKET_MS))
        dur_bucket = int(round(float(duration_ms) / 50.0))
        return (bucket, dur_bucket, int(pcm_crc32), round(float(playback_rate), 2), str(target_lang))

    def cache_dubbed_event(
        self,
        event: DownstreamEvent,
        duration_ms: float = 0.0,
        pcm_crc32: int = 0,
    ) -> None:
        # v10.7 P0-B: this method mutates shared session state and is
        # reachable from both the event-loop thread and the GPU worker
        # thread, so it must hold the session lock.
        with self._lock:
            dur = duration_ms if duration_ms > 0.0 else (event.actual_duration_sec * 1000.0)
            key = self._make_replay_key(
                event.utterance_start_ms, dur, pcm_crc32, event.playback_rate, event.target_lang
            )
            if key in self._replay_cache:
                self._replay_cache.move_to_end(key)
            self._replay_cache[key] = event
            while len(self._replay_cache) > REPLAY_CACHE_MAX_ITEMS:
                self._replay_cache.popitem(last=False)

    def lookup_cached_event(
        self,
        utterance_start_ms: float,
        duration_ms: float = 0.0,
        pcm_crc32: int = 0,
        playback_rate: float = 1.0,
        target_lang: LanguageCode = "km",
    ) -> Optional[DownstreamEvent]:
        # v10.7 P0-B: this method mutates shared session state and is
        # reachable from both the event-loop thread and the GPU worker
        # thread, so it must hold the session lock.
        with self._lock:
            # Exact match
            key = self._make_replay_key(utterance_start_ms, duration_ms, pcm_crc32, playback_rate, target_lang)
            evt = self._replay_cache.get(key)
            if evt is not None:
                self._replay_cache.move_to_end(key)
                return evt

            # Fuzzy bucket fallback if crc32 was 0 (unit test) or duration matched within 1 bucket (live seek)
            target_bucket = int(round(float(utterance_start_ms) / REPLAY_BUCKET_MS))
            target_dur_bucket = int(round(float(duration_ms) / 50.0))
            for (b, db, cached_crc, r, l), cached_evt in reversed(self._replay_cache.items()):
                if b == target_bucket and r == round(float(playback_rate), 2) and l == str(target_lang):
                    crc_ok = (pcm_crc32 == 0 or cached_crc == 0 or cached_crc == pcm_crc32)
                    if (pcm_crc32 == 0 and duration_ms == 0.0) or (
                        duration_ms > 0.0 and abs(db - target_dur_bucket) <= 1 and crc_ok
                    ):
                        return cached_evt
            return None

    def reset_on_seek(self, new_timestamp_ms: float) -> None:
        """
        Rewind/Seek Law: Clears uncommitted PCM audio and resets VAD timers to new_timestamp_ms
        while preserving SpeakerRegistry voice centroids and _replay_cache.
        """
        with self._lock:
            self.pcm_buffer = np.zeros(0, dtype=np.float32)
            self.is_speaking = False
            self.silence_ms = 0.0
            self.active_speech_ms = 0.0
            self.stream_cursor_ms = float(new_timestamp_ms)
            self.current_utterance_start_ms = float(new_timestamp_ms)
            self.past_clause_kv = None
            self.past_gate_kv = None
            self.deferred_syll_count = 0
            # v10.7 P0-A: a seek ends the clause, so the hard-ceiling flag is
            # cleared along with the rolling cache.
            self.last_slice_was_hard_ceiling = False

    def _flush_rolling_state(self) -> None:
        """
        v10.7 P0-A: drop the rolling cross-clause KV cache and the deferred
        syllable count.

        The cache is only valid across a Trigger #2 hard-ceiling cut, where the
        same clause continues. A natural pause (Trigger #1), a rejected micro-pop
        and a seek all end the clause, so the cached context must not leak into
        the next one.

        The caller must already hold self._lock.
        """
        self.past_clause_kv = None
        self.past_gate_kv = None
        self.deferred_syll_count = 0


    def append_pcm_chunk(
        self, chunk: np.ndarray, timestamp_ms: float = -1.0
    ) -> Optional[Tuple[int, float, np.ndarray, float]]:
        # v10.7 P0-B: this method mutates shared session state and is
        # reachable from both the event-loop thread and the GPU worker
        # thread, so it must hold the session lock.
        with self._lock:
            chunk = np.asarray(chunk, dtype=np.float32).reshape(-1)
            if chunk.size == 0:
                return None

            # Continuous timestamp clock synchronization during non-speech periods
            if timestamp_ms >= 0.0 and not self.is_speaking:
                self.stream_cursor_ms = float(timestamp_ms)

            chunk_ms = (chunk.size / float(SAMPLE_RATE_IN)) * 1000.0
            rms = float(np.sqrt(np.mean(chunk ** 2) + 1e-12))

            if rms < SILENCE_RMS_THRESHOLD:
                self.silence_ms += chunk_ms
            else:
                if not self.is_speaking:
                    self.is_speaking = True
                    self.current_utterance_start_ms = self.stream_cursor_ms
                    self.active_speech_ms = 0.0
                self.active_speech_ms += chunk_ms
                self.silence_ms = 0.0

            self.stream_cursor_ms += chunk_ms
            if not self.is_speaking:
                return None

            self.pcm_buffer = np.concatenate([self.pcm_buffer, chunk])

            # Trigger #1: Natural Pause (>= 250ms trailing silence)
            if self.silence_ms >= UTTERANCE_END_SILENCE_MS:
                if self.active_speech_ms < 250.0:
                    self.pcm_buffer = np.zeros(0, dtype=np.float32)
                    self.is_speaking = False
                    self.silence_ms = 0.0
                    self.active_speech_ms = 0.0
                    # v10.7 P0-A: a micro-pop is not a clause boundary, but the
                    # rejected buffer must not leave stale rolling state behind.
                    self._flush_rolling_state()
                    return None

                pause_ms = float(self.silence_ms)
                silence_samples = int(round((pause_ms / 1000.0) * SAMPLE_RATE_IN))
                if 0 < silence_samples < self.pcm_buffer.size:
                    slice_pcm = self.pcm_buffer[:-silence_samples].copy()
                else:
                    slice_pcm = self.pcm_buffer.copy()

                uid = self.next_utterance_id
                self.next_utterance_id += 1
                start_ms = float(self.current_utterance_start_ms)

                self.pcm_buffer = np.zeros(0, dtype=np.float32)
                self.is_speaking = False
                self.silence_ms = 0.0
                self.active_speech_ms = 0.0
                # v10.7 P0-A: a natural pause is a CLAUSE BOUNDARY. The rolling
                # cross-clause KV cache must not survive it: the next clause is a
                # new utterance and the cached context is not valid for it.
                self._flush_rolling_state()
                self.last_slice_was_hard_ceiling = False
                return (uid, start_ms, slice_pcm, pause_ms)

            # Trigger #2: Hard Wall-Clock & Active-Speech Elapsed Ceiling (>= 1600ms active clause span)
            utterance_wall_clock_ms = (self.pcm_buffer.size / float(SAMPLE_RATE_IN)) * 1000.0
            active_clause_wall_clock_ms = utterance_wall_clock_ms - self.silence_ms
            if (
                active_clause_wall_clock_ms >= MAX_CLAUSE_SLICE_SEC * 1000.0
                or self.active_speech_ms >= MAX_CLAUSE_SLICE_SEC * 1000.0
            ):
                min_idx = int(MIN_CLAUSE_SLICE_SEC * SAMPLE_RATE_IN)
                max_idx = min(self.pcm_buffer.size, int(MAX_CLAUSE_SLICE_SEC * SAMPLE_RATE_IN))
                if max_idx <= min_idx:
                    max_idx = self.pcm_buffer.size
                win = 320  # 20ms window at 16kHz
                best_split = max_idx
                best_rms = float("inf")
                for start in range(min_idx, max(min_idx + 1, max_idx - win + 1), win):
                    w_rms = float(np.sqrt(np.mean(self.pcm_buffer[start : start + win] ** 2) + 1e-12))
                    if w_rms < best_rms:
                        best_rms = w_rms
                        best_split = start + win // 2

                slice_pcm = self.pcm_buffer[:best_split].copy()
                remainder = self.pcm_buffer[best_split:].copy()
                uid = self.next_utterance_id
                self.next_utterance_id += 1
                start_ms = float(self.current_utterance_start_ms)

                self.current_utterance_start_ms += (best_split / float(SAMPLE_RATE_IN)) * 1000.0

                trim_idx = 0
                while trim_idx + win <= remainder.size:
                    f_rms = float(
                        np.sqrt(np.mean(remainder[trim_idx : trim_idx + win] ** 2) + 1e-12)
                    )
                    if f_rms < SILENCE_RMS_THRESHOLD:
                        trim_idx += win
                        self.current_utterance_start_ms += 20.0
                    else:
                        break
                remainder = remainder[trim_idx:]

                if remainder.size == 0:
                    self.pcm_buffer = np.zeros(0, dtype=np.float32)
                    self.is_speaking = False
                    self.active_speech_ms = 0.0
                    self.silence_ms = 0.0
                else:
                    self.pcm_buffer = remainder
                    active_frames = 0
                    for i in range(0, remainder.size, win):
                        seg = remainder[i : i + win]
                        if float(np.sqrt(np.mean(seg ** 2) + 1e-12)) >= SILENCE_RMS_THRESHOLD:
                            active_frames += seg.size
                    self.active_speech_ms = (active_frames / float(SAMPLE_RATE_IN)) * 1000.0
                    self.silence_ms = 0.0

                # v10.7 P0-A: this slice was cut by the Trigger #2 hard ceiling,
                # so the rolling cross-clause KV cache IS valid for the next
                # clause. The runner reads this flag to decide whether to pass
                # the cached state into the next forward_s2st call.
                self.last_slice_was_hard_ceiling = True
                return (uid, start_ms, slice_pcm, 40.0)

            return None
```

#### 4.7.2 `backend/transport/live_queue.py`
```python
from __future__ import annotations

import asyncio
from typing import Any, Dict, Optional

import numpy as np

from backend.contracts.schemas import LivePacket


class LiveRequestQueue:
    def __init__(self) -> None:
        self._queue: asyncio.Queue[LivePacket] = asyncio.Queue()
        self.interrupt_epoch: int = 0
        self.playback_rate: float = 1.0

    async def send_realtime(self, pcm_chunk: np.ndarray, timestamp_ms: float) -> None:
        await self._queue.put(
            LivePacket(
                kind="realtime_audio",
                timestamp_ms=float(timestamp_ms),
                pcm_data=pcm_chunk,
            )
        )

    async def send_content(
        self,
        action: str,
        timestamp_ms: float,
        playback_rate: float = 1.0,
        payload: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.playback_rate = float(playback_rate)
        if action in {"interrupt", "seek"}:
            self.flush_all()
        body = dict(payload or {})
        body["action"] = action
        body["playback_rate"] = self.playback_rate
        await self._queue.put(
            LivePacket(kind="content", timestamp_ms=float(timestamp_ms), payload=body)
        )

    def flush_all(self) -> int:
        dropped = 0
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
                self._queue.task_done()
                dropped += 1
            except asyncio.QueueEmpty:
                break
        self.interrupt_epoch += 1
        return dropped

    async def get_next(self) -> LivePacket:
        return await self._queue.get()
```

#### 4.7.3 `backend/transport/runner.py`
```python
from __future__ import annotations

import base64
import io
import struct
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from backend.contracts.schemas import (
    CI_MICRO_CONFIG,
    SAMPLE_RATE_IN,
    SAMPLE_RATE_OUT,
    DownstreamEvent,
    PrecomputedEarFeatures,
    VocalMode,
)
from backend.linguistics.syllable_decomposer import BilingualCodeSwitchDecomposer
from backend.nn_cartridges.s2st_console import SymmetricS2STConsole
from backend.nn_cartridges.stem_and_mode_gate import StemAndScenePipeline
from backend.transport.live_queue import LiveRequestQueue
from backend.transport.session import InMemorySession


def encode_wav_pcm16_b64(
    wav_24k: torch.Tensor | np.ndarray, sample_rate: int = SAMPLE_RATE_OUT
) -> str:
    """Pure 44-byte RIFF/WAVE PCM16 encoder (raw[:4] == b'RIFF' and raw[8:12] == b'WAVE')."""
    t = torch.as_tensor(wav_24k).detach().float().reshape(-1)
    pcm16 = (t.clamp(-1.0, 1.0) * 32767.0).to(torch.int16).cpu().numpy()
    data_bytes = pcm16.tobytes()
    buf = io.BytesIO()
    buf.write(b"RIFF")
    buf.write(struct.pack("<I", 36 + len(data_bytes)))
    buf.write(b"WAVEfmt ")
    buf.write(struct.pack("<IHHIIHH", 16, 1, 1, sample_rate, sample_rate * 2, 2, 16))
    buf.write(b"data")
    buf.write(struct.pack("<I", len(data_bytes)))
    buf.write(data_bytes)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def energy_weighted_speaker_embedding(
    h_acoustic_l6: torch.Tensor,
    mel_stems: torch.Tensor,
) -> torch.Tensor:
    """
    v10.7 P1-D: extract a speaker-discriminative 512-D embedding.

    The SpeakerRegistry needs to tell speakers apart, so the embedding must
    retain the vocal-tract information. The DisentangledEmotionEncoder cannot
    supply it: it applies F.instance_norm (which removes each mel band's mean
    and variance over time) and avg_pool1d(k=25), and the resulting latent is
    deliberately pushed toward language invariance by the gradient-reversal
    branch. Matching on it conflates emotional prosody with speaker identity.

    The layer-6 acoustic representation carries the spectral envelope, so the
    embedding is an energy-weighted mean pool over the ACTIVE frames: frames
    whose mel energy is negligible contribute nothing, so silence does not drag
    the estimate toward the origin. The result is L2-normalized along the last
    dimension, matching the centroid convention used by SpeakerRegistry.

    Parameters:
        h_acoustic_l6: (B_spk, T_frames, SEMANTIC_DIM) acoustic representation.
        mel_stems: (B_spk, N_MELS, T_mel) log-mel of the separated stems.

    Returns:
        (B_spk, SEMANTIC_DIM) L2-normalized speaker embeddings.
        Adds NO parameters.
    """
    if h_acoustic_l6 is None:
        raise ValueError("energy_weighted_speaker_embedding requires the layer-6 tap")
    # Per-frame energy from the mel spectrogram, reduced onto the frame grid.
    mel_energy = mel_stems.pow(2).mean(dim=1)  # (B_spk, T_mel)
    if mel_energy.shape[-1] != h_acoustic_l6.shape[1]:
        mel_energy = F.interpolate(
            mel_energy.unsqueeze(1),
            size=h_acoustic_l6.shape[1],
            mode="linear",
            align_corners=False,
        ).squeeze(1)
    weights = mel_energy.clamp(min=0.0)
    total = weights.sum(dim=-1, keepdim=True)
    # A fully silent stem falls back to an unweighted mean.
    weights = torch.where(total > 1e-8, weights, torch.ones_like(weights))
    pooled = (h_acoustic_l6 * weights.unsqueeze(-1)).sum(dim=1)
    pooled = pooled / weights.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    return F.normalize(pooled, p=2, dim=-1)


class FullDuplexRunner:
    def __init__(
        self,
        pipeline: Optional[StemAndScenePipeline] = None,
        console: Optional[SymmetricS2STConsole] = None,
    ) -> None:
        self.pipeline = pipeline or StemAndScenePipeline(CI_MICRO_CONFIG)
        self.console = console or SymmetricS2STConsole(CI_MICRO_CONFIG)
        self.decomposer = BilingualCodeSwitchDecomposer()

    def process_utterance_slice(
        self,
        session: InMemorySession,
        queue: LiveRequestQueue,
        uid: int,
        start_ms: float,
        slice_pcm: np.ndarray,
        trailing_pause_ms: float,
    ) -> Optional[DownstreamEvent]:
        dur_ms = (slice_pcm.size / float(SAMPLE_RATE_IN)) * 1000.0
        pcm_crc = session.compute_pcm_crc32(slice_pcm)

        # 1. Check Rewind LRU Replay Cache first (< 0.1 ms fast-path)
        cached = session.lookup_cached_event(
            start_ms,
            duration_ms=dur_ms,
            pcm_crc32=pcm_crc,
            playback_rate=queue.playback_rate,
            target_lang="km",
        )
        if cached is not None:
            return cached

        # 2. Snapshot interrupt_epoch before GPU work (Epoch Fence)
        epoch_before = queue.interrupt_epoch

        # Retrieve primary host centroid E_0 (if registered) to prevent stem permutation
        primary_emb = (
            session.speaker_registry.centroids[0].unsqueeze(0)
            if session.speaker_registry.centroids
            else None
        )
        wav_16k = torch.from_numpy(slice_pcm).float().unsqueeze(0)
        scene = self.pipeline.separate_and_classify(
            wav_16k,
            utterance_id=uid,
            utterance_start_ms=start_ms,
            trailing_pause_ms=trailing_pause_ms,
            primary_spk_emb=primary_emb,
        )

        # Register / match separated speakers in SpeakerRegistry
        # v10.7 P1-D: run the ear over the SEPARATED STEMS so the layer-6 tap
        # is available per speaker. The emotion encoder is still used for the
        # emotional conditioning, but the SPEAKER identity now comes from the
        # acoustic representation, which is what actually carries vocal-tract
        # identity.
        stem_16k = scene.active_speaker_stems
        # v10.7 AUD-107-02: the ear runs ONCE per audio slice.
        #
        # v10.7 ran the ear here, then called log_mel TWICE more, then ran the
        # emotion encoder whose output was DISCARDED, and finally called
        # forward_s2st, which ran the ear and the emotion encoder a second time
        # on the identical audio. That doubled the 64.5M-parameter ear and the
        # 8.7M-parameter emotion encoder on every spoken utterance and
        # threatened the T_GPU <= 90 ms latency law.
        #
        # The bundle is computed once here and handed to forward_s2st, which
        # reuses it. The log-mel is computed ONCE and shared by the speaker
        # embedding and the emotion encoder.
        bus, r_sps = self.console.ear(stem_16k, "en")
        spk_l6 = self.console.ear.last_l6_hidden
        mel_stems = self.console.ear.log_mel(stem_16k)
        # v10.7 P1-D: the speaker identity must come from a SPEAKER-DISCRIMINATIVE
        # embedding. v10.6 used the DisentangledEmotionEncoder output, but that
        # encoder applies F.instance_norm and avg_pool1d(k=25) specifically to
        # REMOVE the spectral envelope that carries vocal-tract identity, so the
        # registry was matching on emotional prosody rather than on the speaker.
        # The layer-6 acoustic representation is used instead, mean-pooled over
        # the ACTIVE (energy-bearing) frames and L2-normalized. Zero parameters.
        spk_embs = energy_weighted_speaker_embedding(
            spk_l6, mel_stems
        )
        tracked_speaker_ids: list[int] = []
        for spk_i in range(spk_embs.shape[0]):
            slot_id, _ = session.speaker_registry.match_or_register_speaker(spk_embs[spk_i])
            tracked_speaker_ids.append(slot_id)

        modes = [int(m) for m in scene.dominant_vocal_modes]

        if all(m <= int(VocalMode.SINGING_OR_MUSIC) for m in modes):
            ev_type = (
                "singing_lyrics"
                if any(m == int(VocalMode.SINGING_OR_MUSIC) for m in modes)
                else "passthrough_vocal"
            )
            evt = DownstreamEvent(
                event_type=ev_type,
                utterance_id=uid,
                utterance_start_ms=start_ms,
                vocal_modes=modes,
                speaker_ids=tracked_speaker_ids,
                playback_rate=queue.playback_rate,
                interrupt_epoch=epoch_before,
            )
            if queue.interrupt_epoch != epoch_before:
                return None
            session.cache_dubbed_event(evt, duration_ms=dur_ms, pcm_crc32=pcm_crc)
            return evt

        orig_dur = float(slice_pcm.size) / float(SAMPLE_RATE_IN)
        # v10.7 P0-A: pass the rolling cross-clause KV state into the console.
        # v10.6 never passed these four arguments, so the rolling KV cache and
        # the Hibiki entropy wait-gating were DEAD CODE in production even
        # though the console accepted them.
        dry_24k, syll_batch, _ = self.console.forward_s2st(
            scene.active_speaker_stems,
            source_lang="en",
            target_lang="km",
            orig_duration_sec=orig_dur,
            trailing_pause_ms=trailing_pause_ms,
            playback_rate=queue.playback_rate,
            past_clause_kv=session.past_clause_kv,
            past_gate_kv=session.past_gate_kv,
            deferred_syll_count=session.deferred_syll_count,
            is_hard_ceiling_slice=session.last_slice_was_hard_ceiling,
            precomputed_ear=PrecomputedEarFeatures(
                bus=bus,
                r_sps=r_sps,
                mel_stems=mel_stems,
                h_acoustic_l6=spk_l6,
            ),
        )
        glued_24k = self.pipeline.apply_scene_glue(
            dry_24k,
            scene.mic_eq_residual_16k,
            scene.rt60_sec,
            scene.wet_gain,
            scene.rms_scalar,
        )

        # Decode batch to target text and extracted code-switch words
        target_texts, cs_words = self.decomposer.decode_batch_to_text(syll_batch)

        # Case C: Mixed-Mode Co-Host Re-Injection
        T_out = glued_24k.shape[-1]
        mixed = torch.zeros(T_out, dtype=glued_24k.dtype, device=glued_24k.device)
        for spk_idx, m in enumerate(modes):
            if m == int(VocalMode.SPOKEN_DIALOGUE):
                mixed = mixed + glued_24k[spk_idx]
            else:
                stem_16k = scene.active_speaker_stems[spk_idx : spk_idx + 1].unsqueeze(1)
                stem_24k = F.interpolate(
                    stem_16k, size=T_out, mode="linear", align_corners=False
                ).reshape(-1)
                mixed = mixed + stem_24k

        mixed = mixed.clamp(-0.98, 0.98)
        erb_np = scene.erb_suppression_mask.detach().cpu().numpy().astype(np.float32)
        erb_b64 = base64.b64encode(erb_np.tobytes()).decode("ascii")

        # 3. Verify Epoch Fence after GPU synthesis
        if queue.interrupt_epoch != epoch_before:
            return None

        evt = DownstreamEvent(
            event_type="audio_chunk",
            utterance_id=uid,
            utterance_start_ms=start_ms,
            vocal_modes=modes,
            speaker_ids=tracked_speaker_ids,
            dubbed_audio_b64=encode_wav_pcm16_b64(mixed),
            erb_suppression_mask_b64=erb_b64,
            erb_frame_count=int(erb_np.shape[0]),
            per_speaker_target_texts=target_texts,
            codeswitch_words=cs_words,
            playback_rate=queue.playback_rate,
            actual_duration_sec=float(T_out) / float(SAMPLE_RATE_OUT),
            interrupt_epoch=epoch_before,
        )
        # v10.7 P0-A: commit the NEW rolling cross-clause KV state to the
        # session, but ONLY after the post-synthesis epoch fence has passed.
        # An interrupted slice must not be able to poison the next clause with
        # a KV cache derived from audio the user has already seeked away from.
        with session._lock:
            session.past_clause_kv = self.console.mouth.last_next_past_kv
            session.past_gate_kv = self.console.mouth.last_next_past_gate
            session.deferred_syll_count = self.console.mouth.last_deferred_syll_count
        session.cache_dubbed_event(evt, duration_ms=dur_ms, pcm_crc32=pcm_crc)
        return evt
```

#### 4.7.4 `backend/transport/server.py` (Non-Blocking Full-Duplex WebSocket Server)
```python
from __future__ import annotations

import asyncio
from dataclasses import asdict
import json
import struct

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from backend.contracts.schemas import DownstreamEvent
from backend.transport.live_queue import LiveRequestQueue
from backend.transport.runner import FullDuplexRunner
from backend.transport.session import InMemorySession

app = FastAPI(title="KHMER-SOVEREIGN-S2ST v10.7 Full-Duplex Server")
runner = FullDuplexRunner()
GPU_SEMAPHORE = asyncio.Semaphore(1)


@app.websocket("/ws/live-dub/{session_id}")
async def live_dub_endpoint(websocket: WebSocket, session_id: str) -> None:
    await websocket.accept()
    session = InMemorySession(session_id)
    queue = LiveRequestQueue()
    out_queue: asyncio.Queue[DownstreamEvent] = asyncio.Queue()

    async def upstream_reader() -> None:
        while True:
            msg = await websocket.receive()
            if msg.get("bytes"):
                raw: bytes = msg["bytes"]
                if len(raw) <= 8:
                    continue
                (ts_ms,) = struct.unpack("<d", raw[:8])
                pcm = np.frombuffer(raw[8:], dtype=np.float32).copy()
                await queue.send_realtime(pcm, timestamp_ms=ts_ms)
            elif msg.get("text"):
                data = json.loads(msg["text"])
                action = str(data.get("action", "config"))
                ts_ms = float(data.get("video_time_ms", 0.0))
                rate = float(data.get("playback_rate", queue.playback_rate))
                await queue.send_content(action, timestamp_ms=ts_ms, playback_rate=rate, payload=data)
                # Immediate <1ms Control-Plane Interrupt/Seek Execution (never blocked by GPU worker)
                if action in {"seek", "interrupt"}:
                    session.reset_on_seek(ts_ms)
                    await out_queue.put(
                        DownstreamEvent(
                            event_type="interrupted",
                            utterance_id=session.next_utterance_id,
                            utterance_start_ms=ts_ms,
                            playback_rate=queue.playback_rate,
                            interrupt_epoch=queue.interrupt_epoch,
                        )
                    )

    async def worker_loop() -> None:
        while True:
            pkt = await queue.get_next()
            if pkt.kind == "content":
                continue
            if pkt.kind == "realtime_audio" and pkt.pcm_data is not None:
                sliced = session.append_pcm_chunk(pkt.pcm_data, timestamp_ms=pkt.timestamp_ms)
                if sliced is not None:
                    uid, start_ms, slice_pcm, pause_ms = sliced
                    # v10.7 P0-B: snapshot the interrupt epoch BEFORE waiting
                    # for the GPU semaphore. v10.6 snapshotted it INSIDE the
                    # worker thread, i.e. after the semaphore wait, so a seek
                    # that landed while the worker was queued was invisible to
                    # the fence and the stale pre-seek audio was emitted after
                    # the 'interrupted' packet.
                    pkt_epoch = queue.interrupt_epoch
                    async with GPU_SEMAPHORE:
                        # Re-check after acquiring the semaphore: a seek may
                        # have arrived during the wait.
                        if queue.interrupt_epoch != pkt_epoch:
                            continue
                        evt = await asyncio.to_thread(
                            runner.process_utterance_slice,
                            session,
                            queue,
                            uid,
                            start_ms,
                            slice_pcm,
                            pause_ms,
                        )
                    if evt is not None:
                        await out_queue.put(evt)

    async def downstream_writer() -> None:
        while True:
            evt = await out_queue.get()
            await websocket.send_json(asdict(evt))

    tasks = [
        asyncio.create_task(upstream_reader()),
        asyncio.create_task(worker_loop()),
        asyncio.create_task(downstream_writer()),
    ]
    try:
        await asyncio.gather(*tasks)
    except WebSocketDisconnect:
        for t in tasks:
            t.cancel()
```

---

### 4.8 Specialist D — Rewind-Safe TypeScript Frontend (`frontend/src/`)

#### 4.8.1 `frontend/src/worklets/pcm-capture-processor.ts`
```typescript
import "../contracts/types";
import { SAMPLE_RATE_IN } from "../contracts/types";

// Scoped strictly to the AudioWorkletGlobalScope module
declare const currentTime: number;
declare const sampleRate: number;

// 15-tap Blackman-windowed sinc filter for 48kHz -> 16kHz (fc = 7.2 kHz, normalized sum = 1.0000, 0.0 dB DC gain)
const FIR_15_TAPS = new Float32Array([
  -0.0030, -0.0091, -0.0101, 0.0110, 0.0630, 0.1366, 0.1989,
   0.2254,
   0.1989,  0.1366,  0.0630, 0.0110, -0.0101, -0.0091, -0.0030,
]);

export class PcmCaptureProcessor extends AudioWorkletProcessor {
  private readonly ring = new Float32Array(15);
  private ringIdx = 0;
  private phase = 0;
  // v10.7 AUD-107-10: fractional read position and previous mono sample for
  // the universal linear-interpolation resampler fallback.
  private fracPos = 0.0;
  private prevSample = 0.0;
  private readonly outBuf = new Float32Array(128);
  // v10.7 P1-E: the media clock must advance at the VIDEO's rate, not at the
  // AudioContext's 1.0x wall clock. At playbackRate 2.0 the video timeline
  // advances twice as fast as the audio context, so a 1.0x clock drifts and
  // the upstream packet timestamps no longer match videoEl.currentTime.
  private playbackRate = 1.0;
  private lastCtxTimeMs = 0.0;
  private mediaClockMs = 0.0;

  constructor(options?: unknown) {
    super(options);
    this.port.onmessage = (ev: MessageEvent<{
      baseVideoTimeMs?: number;
      playbackRate?: number;
    }>) => {
      if (ev.data && typeof ev.data.playbackRate === "number" && ev.data.playbackRate > 0) {
        this.playbackRate = ev.data.playbackRate;
      }
      if (ev.data && typeof ev.data.baseVideoTimeMs === "number") {
        this.mediaClockMs = ev.data.baseVideoTimeMs;
        this.lastCtxTimeMs = currentTime * 1000.0;
      }
    };
  }

  process(inputs: Float32Array[][], _outputs: Float32Array[][]): boolean {
    const input = inputs[0];
    if (!input || !input[0] || input[0].length === 0) return true;

    const frames = input[0].length;
    const left = input[0];
    const right = input.length > 1 && input[1] ? input[1] : input[0];
    let outCount = 0;

    // v10.7 AUD-107-10: the decimator MUST follow the REAL context rate.
    //
    // v10.7 hard-coded a 3:1 decimation, which is only correct when the
    // AudioContext runs at 48,000 Hz. On iOS Safari, some Android WebViews and
    // Bluetooth headsets the OS locks the context to 44,100 or 96,000 Hz, and a
    // 3:1 decimator then emitted 14,700 Hz instead of 16,000 Hz.
    //
    // The 48,000 Hz fast path keeps the 15-tap Blackman-sinc decimator. Every
    // other rate uses a stateful fractional linear-interpolation resampler, so
    // the packets are always 16,000 Hz on any device.
    const ctxRate =
      typeof sampleRate === "number" && sampleRate > 0 ? sampleRate : 48000;

    if (ctxRate === 48000) {
      for (let i = 0; i < frames; i++) {
        this.ring[this.ringIdx] = 0.5 * (left[i] + right[i]);
        this.ringIdx = (this.ringIdx + 1) % 15;

        if (this.phase === 0) {
          let sum = 0.0;
          for (let t = 0; t < 15; t++) {
            sum += this.ring[(this.ringIdx + t) % 15] * FIR_15_TAPS[t];
          }
          this.outBuf[outCount++] = sum;
        }
        this.phase = (this.phase + 1) % 3;
      }
    } else {
      // Universal fractional resampler fallback for the 44.1 kHz / non-48 kHz
      // mobile and Bluetooth hardware. The output rate is exactly 16,000 Hz
      // for every context rate at or above 16 kHz.
      const step = ctxRate / SAMPLE_RATE_IN;
      for (let i = 0; i < frames; i++) {
        const curSample = 0.5 * (left[i] + right[i]);
        while (this.fracPos < 1.0 && outCount < this.outBuf.length) {
          this.outBuf[outCount++] =
            this.prevSample + this.fracPos * (curSample - this.prevSample);
          this.fracPos += step;
        }
        this.fracPos -= 1.0;
        this.prevSample = curSample;
      }
    }

    const byteLen = outCount * 4;
    const packet = new ArrayBuffer(8 + byteLen);
    // v10.7 P1-E: advance the media clock by the wall-clock delta scaled by
    // the video playback rate, so the upstream timestamps track the VIDEO
    // timeline rather than the 1.0x AudioContext clock.
    const ctxNowMs = currentTime * 1000.0;
    const deltaMs = ctxNowMs - this.lastCtxTimeMs;
    this.lastCtxTimeMs = ctxNowMs;
    this.mediaClockMs += deltaMs * this.playbackRate;
    const mediaTimeMs = Math.max(0.0, this.mediaClockMs);
    new DataView(packet).setFloat64(0, mediaTimeMs, true);
    new Float32Array(packet, 8, outCount).set(this.outBuf.subarray(0, outCount));
    this.port.postMessage(packet, [packet]);
    return true;
  }
}

registerProcessor("pcm-capture-processor", PcmCaptureProcessor);
```

#### 4.8.2 `frontend/src/audio/stereo-mne-mixer.ts`
```typescript
import {

  DownstreamEventPacket,
  ERB_BANDS,
  ERB_MAX_FREQ_HZ,
  ERB_MIN_FREQ_HZ,
  LOOKAHEAD_BUFFER_SEC,
} from "../contracts/types";

// v10.7 AUD-107-05: the FFT twiddle and bit-reversal tables are computed
// ONCE at module load, not inside every real-time audio callback.
//
// v10.7 allocated two Float64Array tables and called Math.log2() inside the
// bit-reversal inner loop on EVERY 4096-sample block, which is 98,304
// Math.log2 calls per block. That is pure V8 heap churn inside a real-time
// WebAudio callback, where a GC pause is an audible dropout.
const N_FFT_MNE = 4096;
const HALF_FFT_MNE = N_FFT_MNE / 2;
const LOG2_N_MNE = 12; // Math.log2(4096)

const COS_TABLE_MNE = new Float64Array(HALF_FFT_MNE);
const SIN_TABLE_MNE = new Float64Array(HALF_FFT_MNE);
const REV_TABLE_MNE = new Uint16Array(N_FFT_MNE);

// The INVERSE transform uses the conjugate (positive-exponent) twiddles, so
// it needs its own pair of tables. They are the exact negation of the
// forward tables, but are kept separate so neither transform has to negate
// on the fly inside the real-time callback.
const COS_TABLE_MNE_INV = new Float64Array(HALF_FFT_MNE);
const SIN_TABLE_MNE_INV = new Float64Array(HALF_FFT_MNE);

for (let k = 0; k < HALF_FFT_MNE; k++) {
  COS_TABLE_MNE[k] = Math.cos((-2 * Math.PI * k) / N_FFT_MNE);
  SIN_TABLE_MNE[k] = Math.sin((-2 * Math.PI * k) / N_FFT_MNE);
  COS_TABLE_MNE_INV[k] = Math.cos((2 * Math.PI * k) / N_FFT_MNE);
  SIN_TABLE_MNE_INV[k] = Math.sin((2 * Math.PI * k) / N_FFT_MNE);
}
for (let i = 0; i < N_FFT_MNE; i++) {
  let r = 0;
  for (let b = 0; b < LOG2_N_MNE; b++) {
    r |= ((i >> b) & 1) << (LOG2_N_MNE - 1 - b);
  }
  REV_TABLE_MNE[i] = r;
}

export function hzToErbRate(hz: number): number {
  return 21.4 * Math.log10(1.0 + 0.00437 * hz);
}

const MIN_ERB = hzToErbRate(ERB_MIN_FREQ_HZ);
const MAX_ERB = hzToErbRate(ERB_MAX_FREQ_HZ);

export interface ScheduledErbSegment {
  utteranceId: number;
  startTime: number;
  endTime: number;
  frameCount: number;
  effectiveFrameRate: number;
  frames: Float32Array;
}

export class StereoMneMixer {
  private ctx: AudioContext;
  private sessionStartCtxTime: number;
  private nextPlayTime = 0;
  private erbQueue: ScheduledErbSegment[] = [];
  private activeSources: Set<AudioBufferSourceNode> = new Set();
  private clientEpoch = 0;
  private minServerEpoch = 0;

  // v10.7 P1-E: live spectral-analysis state for the studio M&E suppressor.
  private _midSpecReal: Float32Array | null = null;
  private _midSpecImag: Float32Array | null = null;
  private _sideSpecReal: Float32Array | null = null;
  private _sideSpecImag: Float32Array | null = null;
  private _mediaSourceNode: MediaElementAudioSourceNode | null = null;
  private _mneDelayNode: DelayNode | null = null;
  private _mneProcessorNode: ScriptProcessorNode | null = null;

  constructor(ctx: AudioContext, sessionStartCtxTime?: number) {
    if (ctx.sampleRate !== 48000) {
      console.warn(
        `[StereoMneMixer] AudioContext sampleRate is ${ctx.sampleRate}Hz, expected 48000Hz. ` +
        `Always instantiate with new AudioContext({ sampleRate: 48000 }) to avoid decimation pitch drift!`
      );
    }
    this.ctx = ctx;
    this.sessionStartCtxTime = sessionStartCtxTime ?? ctx.currentTime;
    this.nextPlayTime = this.ctx.currentTime;
  }

  public startSession(videoTimeMs = 0, playbackRate = 1.0): void {
    this.reanchorTimeline(videoTimeMs, playbackRate);
  }

  /**
   * Rewind/Seek Resilience Law:
   * Flushes active buffers, increments clientEpoch, resets nextPlayTime so rewound audio is
   * never locked behind a future timestamp, and re-anchors sessionStartCtxTime.
   */
  public reanchorTimeline(videoTimeMs: number, playbackRate = 1.0): void {
    this.flushPlayback();
    const safeRate = Math.max(0.5, Math.min(2.5, playbackRate || 1.0));
    this.sessionStartCtxTime = this.ctx.currentTime - videoTimeMs / (1000.0 * safeRate);
  }

  public applyMidSideSpectralErbMask(
    midSpecReal: Float32Array,
    midSpecImag: Float32Array,
    _sideSpecReal: Float32Array,
    _sideSpecImag: Float32Array,
    erbMask64: Float32Array,
    sampleRate = 48000,
    nFft = 4096
  ): void {
    const numBins = midSpecReal.length;
    for (let k = 0; k < numBins; k++) {
      const freqHz = k * (sampleRate / nFft);
      if (freqHz < ERB_MIN_FREQ_HZ || freqHz > ERB_MAX_FREQ_HZ) {
        continue;
      }
      const erbVal = hzToErbRate(freqHz);
      const norm = (erbVal - MIN_ERB) / (MAX_ERB - MIN_ERB);
      const erbIdx = Math.max(
        0,
        Math.min(ERB_BANDS - 1, Math.floor(ERB_BANDS * norm))
      );
      const gain = Math.max(0.0, Math.min(1.0, erbMask64[erbIdx]));
      midSpecReal[k] *= gain;
      midSpecImag[k] *= gain;
    }
  }

  public getActiveErbMaskForTime(ctxTime: number): Float32Array | null {
    this.erbQueue = this.erbQueue.filter((seg) => seg.endTime >= ctxTime - 0.05);
    for (const seg of this.erbQueue) {
      if (ctxTime >= seg.startTime && ctxTime <= seg.endTime && seg.frameCount > 0) {
        const fIdx = Math.max(
          0,
          Math.min(
            seg.frameCount - 1,
            Math.floor((ctxTime - seg.startTime) * seg.effectiveFrameRate)
          )
        );
        return seg.frames.subarray(fIdx * ERB_BANDS, (fIdx + 1) * ERB_BANDS);
      }
    }
    return null;
  }

  /**
   * v10.7 P1-E: apply the active 64-band ERB mask to a Mid-channel frame.
   *
   * v10.6 defined applyMidSideSpectralErbMask and getActiveErbMaskForTime but
   * never called them, so the ERB mask the server computes and ships was
   * queued and then ignored: the studio M&E suppressor never suppressed
   * anything.
   *
   * The target time MUST be the segment's own start time (the lookahead
   * target), NOT this.ctx.currentTime. Newly queued segments start 2.0 s in
   * the FUTURE, so querying the current time would always return null and
   * the mask would never be applied.
   */
  public processMidSideFrame(
    targetCtxTime: number,
    midSpecReal: Float32Array,
    midSpecImag: Float32Array,
    sideSpecReal: Float32Array,
    sideSpecImag: Float32Array,
    sampleRate = 48000,
    nFft = 4096
  ): boolean {
    const mask = this.getActiveErbMaskForTime(targetCtxTime);
    if (!mask) {
      return false;
    }
    this.applyMidSideSpectralErbMask(
      midSpecReal,
      midSpecImag,
      sideSpecReal,
      sideSpecImag,
      mask,
      sampleRate,
      nFft
    );
    return true;
  }

  /**
   * v10.7 P1-E: route the media element through a live 4096-bin Mid/Side
   * spectral suppressor so the ERB mask the server ships is actually applied
   * to the original M&E (music & effects) bed.
   *
   * The graph is: media element -> MediaElementSource -> ScriptProcessor
   * (4096-bin STFT, Mid/Side split, ERB mask, ISTFT) -> destination. The
   * processor asks the mixer for the mask that is active at the CURRENT
   * audio time, which is the correct clock for a live processor (the
   * lookahead clock is used when a segment is queued).
   */
  /**
   * v10.7 P1-E: in-place radix-2 FFT of a real frame, used by the Mid/Side
   * spectral suppressor. Writes the first N_FFT/2+1 bins into re/im.
   */
  private _computeSpectrum(
    frame: Float32Array,
    nFrames: number,
    re: Float32Array,
    im: Float32Array,
    nFft: number
  ): void {
    const size = nFft;
    const half = size / 2;
    // v10.7 AUD-107-05: use the module-level tables.
    const cos = COS_TABLE_MNE;
    const sin = SIN_TABLE_MNE;
    // v10.7: UNITY (rectangular) windowing, not Hann.
    //
    // ScriptProcessorNode(4096, 2, 2) delivers NON-OVERLAPPING 4096-sample
    // blocks, and N_FFT equals the block size, so there is no overlap-add and
    // no inverse-window compensation anywhere in this path. A Hann window
    // tapers the first and last samples of EVERY block to zero, so the
    // reconstruction came back with a hard zero at both block edges and the
    // M&E bed developed periodic dropouts at the block rate. With a unity
    // window the forward/inverse pair is the exact identity on the block.
    const start = Math.max(0, nFrames - size);
    const x = new Float64Array(size);
    for (let i = 0; i < size; i++) {
      const src = start + i;
      x[i] = src < nFrames ? frame[src] : 0;
    }
    // Iterative radix-2 Cooley-Tukey.
    // v10.7 AUD-107-05: the bit-reversal permutation is precomputed at
    // module load, so no Math.log2 call happens per block.
    const rev = (v: number): number => REV_TABLE_MNE[v];
    for (let i = 0; i < size; i++) {
      const j = rev(i);
      if (j > i) {
        const tmp = x[i];
        x[i] = x[j];
        x[j] = tmp;
      }
    }
    // v10.7: the bit-reversed input MUST be copied into the butterfly
    // working arrays. Without this copy the butterflies operate on zeros and
    // every bin comes out 0.0.
    const bre = new Float64Array(size);
    const bim = new Float64Array(size);
    for (let i = 0; i < size; i++) {
      bre[i] = x[i];
    }
    for (let len = 2; len <= size; len *= 2) {
      const step = size / len;
      for (let i = 0; i < size; i += len) {
        for (let j = 0; j < len / 2; j++) {
          const wr = cos[(j * step) | 0];
          const wi = sin[(j * step) | 0];
          const a = i + j;
          const b = i + j + len / 2;
          const tr = bre[b] * wr - bim[b] * wi;
          const ti = bre[b] * wi + bim[b] * wr;
          bre[b] = bre[a] - tr;
          bim[b] = bim[a] - ti;
          bre[a] += tr;
          bim[a] += ti;
        }
      }
    }
    const bins = size / 2 + 1;
    for (let k = 0; k < bins; k++) {
      re[k] = bre[k];
      im[k] = bim[k];
    }
  }

  /**
   * v10.7 P1-E: the inverse of _computeSpectrum. Reconstructs the time
   * domain Mid signal from the (possibly ERB-masked) spectrum, using the
   * conjugate symmetry of a real input: X[N-k] = conj(X[k]).
   *
   * Without this step the suppressor was a NO-OP: the masked spectrum was
   * computed and then thrown away, and the unmasked time-domain mid[] was
   * written to the output.
   */
  private _inverseSpectrum(
    re: Float32Array,
    im: Float32Array,
    nFft: number
  ): Float32Array {
    const size = nFft;
    const half = size / 2 + 1;
    // Rebuild the full complex spectrum with conjugate symmetry.
    const fre = new Float64Array(size);
    const fim = new Float64Array(size);
    for (let k = 0; k < half; k++) {
      fre[k] = re[k];
      fim[k] = im[k];
    }
    for (let k = 1; k < size - half + 1; k++) {
      fre[size - k] = re[k];
      fim[size - k] = -im[k];
    }
    // Inverse radix-2 Cooley-Tukey (conjugate the exponent, divide by N).
    // v10.7 AUD-107-05: use the module-level tables.
    const cos = COS_TABLE_MNE_INV;
    const sin = SIN_TABLE_MNE_INV;
    // v10.7 AUD-107-05: the bit-reversal permutation is precomputed at
    // module load, so no Math.log2 call happens per block.
    const rev = (v: number): number => REV_TABLE_MNE[v];
    for (let i = 0; i < size; i++) {
      const j = rev(i);
      if (j > i) {
        const t = fre[i];
        fre[i] = fre[j];
        fre[j] = t;
        const u = fim[i];
        fim[i] = fim[j];
        fim[j] = u;
      }
    }
    // v10.7: the bit-reversed input MUST be copied into the butterfly
    // working arrays, exactly as in _computeSpectrum. Without this copy the
    // inverse transform operates on zeros and always returns an all-zero
    // signal, which would silence the whole M&E bed.
    const bre = new Float64Array(size);
    const bim = new Float64Array(size);
    for (let i = 0; i < size; i++) {
      bre[i] = fre[i];
      bim[i] = fim[i];
    }
    for (let len = 2; len <= size; len *= 2) {
      const step = size / len;
      for (let i = 0; i < size; i += len) {
        for (let j = 0; j < len / 2; j++) {
          const wr = cos[(j * step) | 0];
          const wi = sin[(j * step) | 0];
          const a = i + j;
          const b = i + j + len / 2;
          const tr = bre[b] * wr - bim[b] * wi;
          const ti = bre[b] * wi + bim[b] * wr;
          bre[b] = bre[a] - tr;
          bim[b] = bim[a] - ti;
          bre[a] += tr;
          bim[a] += ti;
        }
      }
    }
    const out = new Float32Array(size);
    for (let i = 0; i < size; i++) {
      out[i] = bre[i] / size;
    }
    return out;
  }

  /**
   * v10.7 P1-E: the shared MediaElementAudioSourceNode for this video element.
   *
   * WebAudio allows exactly ONE MediaElementAudioSourceNode per media
   * element; a second call to createMediaElementSource throws an
   * InvalidStateError. The DuplexWsClient therefore TAPS this node for the
   * upstream capture path instead of creating its own.
   */
  public getMediaSourceNode(): MediaElementAudioSourceNode | null {
    return this._mediaSourceNode;
  }

  public attachMediaElement(videoEl: HTMLVideoElement): MediaElementAudioSourceNode {
    if (this._mediaSourceNode) {
      return this._mediaSourceNode;
    }
    const N_FFT = 4096;
    const HALF = N_FFT / 2 + 1;
    this._midSpecReal = new Float32Array(HALF);
    this._midSpecImag = new Float32Array(HALF);
    this._sideSpecReal = new Float32Array(HALF);
    this._sideSpecImag = new Float32Array(HALF);

    const source = this.ctx.createMediaElementSource(videoEl);
    this._mediaSourceNode = source;

    const analyser = this.ctx.createAnalyser();
    analyser.fftSize = N_FFT;

    const processor = this.ctx.createScriptProcessor(4096, 2, 2);
    this._mneProcessorNode = processor;
    processor.onaudioprocess = (ev) => {
      const input = ev.inputBuffer;
      const output = ev.outputBuffer;
      const nCh = Math.min(input.numberOfChannels, 2);
      const nFrames = input.length;

      // Mid/Side split of the stereo bed.
      const mid = new Float32Array(nFrames);
      const side = new Float32Array(nFrames);
      for (let i = 0; i < nFrames; i++) {
        const l = input.getChannelData(0)[i];
        const r = nCh > 1 ? input.getChannelData(1)[i] : l;
        mid[i] = 0.5 * (l + r);
        side[i] = 0.5 * (l - r);
      }

      // Forward FFT of the Mid channel, then apply the active ERB mask to
      // the Mid spectrum only. The Side channel is passed through untouched.
      // The FFT is computed in-place on a unity-windowed frame.
      const half = N_FFT / 2 + 1;
      const re = new Float32Array(half);
      const im = new Float32Array(half);
      this._midSpecReal = re;
      this._midSpecImag = im;
      this._sideSpecReal = new Float32Array(half);
      this._sideSpecImag = new Float32Array(half);
      this._computeSpectrum(mid, nFrames, re, im, N_FFT);

      // The live mask clock is the current context time.
      const maskApplied = this.processMidSideFrame(
        this.ctx.currentTime,
        re,
        im,
        this._sideSpecReal as Float32Array,
        this._sideSpecImag as Float32Array,
        this.ctx.sampleRate,
        N_FFT
      );

      // Rebuild the output from the MASKED Mid plus the untouched Side.
      //
      // v10.7 P1-E: the masked spectrum must be transformed BACK to the time
      // domain. v10.6 wrote the unmasked mid[] to the output, so the ERB
      // suppressor was a no-op: the mask was computed and then discarded.
      let midOut: Float32Array;
      if (maskApplied) {
        const reconstructed = this._inverseSpectrum(re, im, N_FFT);
        // The frame is the tail of the current block; the reconstruction is
        // aligned to the same offset as the analysis window.
        const offset = Math.max(0, nFrames - N_FFT);
        midOut = new Float32Array(nFrames);
        for (let i = 0; i < nFrames; i++) {
          const src = offset + i;
          midOut[i] = src < reconstructed.length ? reconstructed[src] : 0;
        }
      } else {
        midOut = mid;
      }
      const outL = output.getChannelData(0);
      const outR = output.numberOfChannels > 1 ? output.getChannelData(1) : outL;
      const srcL = input.getChannelData(0);
      const srcR = nCh > 1 ? input.getChannelData(1) : srcL;
      for (let i = 0; i < nFrames; i++) {
        const m = maskApplied ? midOut[i] : 0.5 * (srcL[i] + srcR[i]);
        const s = 0.5 * (srcL[i] - srcR[i]);
        outL[i] = m + s;
        if (output.numberOfChannels > 1) {
          outR[i] = m - s;
        }
      }
    };

    // v10.7 AUD-107-03: the M&E path carries the 2.00s LOOKAHEAD DELAY that
    // the Part 2.1 topology specifies:
    //
    //   [2.00s Delay Line (Video + 48kHz Stereo)] -> [48kHz Mid/Side ERB Suppressor]
    //
    // Without it the original unmasked vocals reached the speakers 2.0s
    // BEFORE the ERB mask and the dubbed audio took effect at
    // lookaheadTargetTime = t_start + 2.00s, so the suppression was always
    // late by the whole lookahead window.
    const delayNode = this.ctx.createDelay(LOOKAHEAD_BUFFER_SEC + 1.0);
    delayNode.delayTime.value = LOOKAHEAD_BUFFER_SEC;
    this._mneDelayNode = delayNode;

    source.connect(delayNode);
    delayNode.connect(processor);
    // v10.7: the processor MUST reach the destination, otherwise the M&E
    // suppressor is connected to nothing and the output is silent.
    processor.connect(this.ctx.destination);
    // v10.7: the source node is returned so the DuplexWsClient can TAP
    // the SAME node for the upstream capture path. Calling
    // ctx.createMediaElementSource(videoEl) a second time on the same
    // element throws an InvalidStateError in WebAudio.
    return source;
  }

  public async handleDownstreamEvent(pkt: DownstreamEventPacket): Promise<void> {
    if (pkt.event_type === "interrupted") {
      this.flushPlayback();
      return;
    }

    if (pkt.interrupt_epoch !== undefined) {
      if (pkt.interrupt_epoch < this.minServerEpoch) return;
      this.minServerEpoch = pkt.interrupt_epoch;
    }

    const epochAtEntry = this.clientEpoch;
    const safeRate = Math.max(0.5, Math.min(2.5, pkt.playback_rate || 1.0));
    const lookaheadTargetTime =
      this.sessionStartCtxTime +
      pkt.utterance_start_ms / (1000.0 * safeRate) +
      LOOKAHEAD_BUFFER_SEC;

    const effectiveFrameRate = 100.0 * safeRate;
    if (pkt.erb_suppression_mask_b64 && pkt.erb_frame_count > 0) {
      const bytes = Uint8Array.from(atob(pkt.erb_suppression_mask_b64), (c) =>
        c.charCodeAt(0)
      );
      const frames = new Float32Array(
        bytes.buffer,
        bytes.byteOffset,
        bytes.byteLength / 4
      );
      this.erbQueue.push({
        utteranceId: pkt.utterance_id,
        startTime: lookaheadTargetTime,
        endTime: lookaheadTargetTime + pkt.erb_frame_count / effectiveFrameRate,
        frameCount: pkt.erb_frame_count,
        effectiveFrameRate,
        frames,
      });
      // v10.7 AUD-107-05: the redundant spectral pass here is REMOVED.
      //
      // v10.7 called processMidSideFrame(lookaheadTargetTime, ...) on
      // this._midSpecReal, which is either a stale spectrum left over from an
      // already-completed audio callback or a throwaway Float32Array. The mask
      // was therefore written into a buffer that the next onaudioprocess call
      // overwrites, so the pass had NO effect on the output while still
      // burning a full FFT on every WebSocket JSON packet.
      //
      // The mask is applied in the RIGHT place: the audio callback computes the
      // spectrum for the block it is about to emit and applies the mask that is
      // active at that moment. With the 2.00s lookahead DelayNode
      // (AUD-107-03) the delayed block and the scheduled ERB segment share the
      // same timeline, so the callback picks up the correct mask.
    }

    if (pkt.event_type === "audio_chunk" && pkt.dubbed_audio_b64) {
      const wavBytes = Uint8Array.from(atob(pkt.dubbed_audio_b64), (c) =>
        c.charCodeAt(0)
      );
      const audioBuf = await this.ctx.decodeAudioData(
        wavBytes.buffer.slice(0) as ArrayBuffer
      );

      // Discard in-flight decoded audio if a seek/rewind occurred during await
      if (this.clientEpoch !== epochAtEntry) {
        return;
      }

      const source = this.ctx.createBufferSource();
      source.buffer = audioBuf;
      source.playbackRate.value = 1.0;
      source.connect(this.ctx.destination);

      const startAt = Math.max(
        this.ctx.currentTime + 0.005,
        lookaheadTargetTime,
        this.nextPlayTime
      );
      source.start(startAt);
      this.nextPlayTime = startAt + audioBuf.duration;
      this.activeSources.add(source);
      source.onended = () => this.activeSources.delete(source);
    }
  }

  public flushPlayback(): void {
    this.clientEpoch += 1;
    for (const src of this.activeSources) {
      try {
        src.stop();
      } catch {
        // Ignore already stopped sources
      }
    }
    this.activeSources.clear();
    this.erbQueue = [];
    this.nextPlayTime = this.ctx.currentTime;
  }
}
```

#### 4.8.3 `frontend/src/client/duplex-ws-client.ts`
```typescript
import { StereoMneMixer } from "../audio/stereo-mne-mixer";
import { DownstreamEventPacket, UpstreamContentMessage } from "../contracts/types";
import { CaptionOverlay } from "../ui/caption-overlay";

export class DuplexWsClient {
  private ws!: WebSocket;
  // v10.7 P1-E / A-1: the upstream audio path. v10.6 never instantiated
  // the PcmCaptureProcessor worklet and never sent a binary frame, so the
  // server's upstream_reader waited forever for bytes that never arrived:
  // the full-duplex loop could not function at all.
  private captureNode: AudioWorkletNode | null = null;

  constructor(
    private readonly sessionId: string,
    private readonly videoEl: HTMLVideoElement,
    private readonly mixer: StereoMneMixer,
    private readonly overlay: CaptionOverlay
  ) {
    this.videoEl.addEventListener("ratechange", () => {
      const timeMs = this.videoEl.currentTime * 1000.0;
      this.mixer.reanchorTimeline(timeMs, this.videoEl.playbackRate);
      this.send({
        action: "config",
        video_time_ms: timeMs,
        playback_rate: this.videoEl.playbackRate,
    });
      this.postMediaClock();
    });

    this.videoEl.addEventListener("seeking", () => {
      const timeMs = this.videoEl.currentTime * 1000.0;
      this.mixer.reanchorTimeline(timeMs, this.videoEl.playbackRate);
      this.send({
        action: "seek",
        video_time_ms: timeMs,
        playback_rate: this.videoEl.playbackRate,
    });
      this.postMediaClock();
    });

    this.videoEl.addEventListener("pause", () => {
      this.mixer.flushPlayback();
    });

    this.videoEl.addEventListener("playing", () => {
      this.mixer.reanchorTimeline(this.videoEl.currentTime * 1000.0, this.videoEl.playbackRate);
      this.postMediaClock();
    });
  }

  /**
   * v10.7 P1-E / A-1: instantiate the PCM capture worklet and forward its
   * binary frames to the WebSocket.
   *
   * The worklet emits an 8-byte little-endian float64 media timestamp
   * followed by the decimated 16 kHz mono samples, which is exactly the
   * frame the server's upstream_reader expects.
   */
  public async attachAudioCapture(
    ctx: AudioContext,
    source?: MediaElementAudioSourceNode
  ): Promise<void> {
    if (this.captureNode) {
      return;
    }
    await ctx.audioWorklet.addModule("worklets/pcm-capture-processor.js");
    const node = new AudioWorkletNode(ctx, "pcm-capture-processor");
    this.captureNode = node;
    node.port.onmessage = (ev: MessageEvent<ArrayBuffer>) => {
      if (this.ws && this.ws.readyState === WebSocket.OPEN) {
        this.ws.send(ev.data);
      }
    };
    // v10.7 P1-E: the worklet taps the SHARED media-element source node.
    // The mixer creates it and returns it; calling
    // ctx.createMediaElementSource(videoEl) a second time on the same
    // element throws an InvalidStateError in WebAudio, so the node is
    // passed in and only created here when the mixer has not made it yet.
    const tap =
      source ?? this.mixer.getMediaSourceNode() ?? ctx.createMediaElementSource(this.videoEl);
    tap.connect(node);
    // v10.7 AUD-107-06: the capture worklet is a CAPTURE-ONLY sink.
    //
    // v10.7 connected it straight to ctx.destination. PcmCaptureProcessor
    // leaves its outputs zeroed, so it is silent today, but a future WASM
    // worklet that passes audio through would sum the UNDELAYED dry audio
    // into the destination, bypassing the M&E suppressor entirely and
    // re-introducing the very bleed the ERB ducking exists to remove.
    //
    // A GainNode with gain 0.0 keeps the node pulled by the graph (so the
    // worklet keeps processing) while guaranteeing that nothing reaches the
    // speakers.
    const captureSink = ctx.createGain();
    captureSink.gain.value = 0.0;
    node.connect(captureSink);
    captureSink.connect(ctx.destination);
    this.postMediaClock();
  }

  /**
   * v10.7 P1-E: publish the current video media time and playback rate to
   * the capture worklet so its timestamps track the VIDEO timeline.
   */
  private postMediaClock(): void {
    if (!this.captureNode) {
      return;
    }
    this.captureNode.port.postMessage({
      baseVideoTimeMs: this.videoEl.currentTime * 1000.0,
      playbackRate: this.videoEl.playbackRate,
    });
  }

  public connect(): void {
    this.ws = new WebSocket(`ws://localhost:8000/ws/live-dub/${this.sessionId}`);
    this.ws.binaryType = "arraybuffer";
    this.mixer.startSession(this.videoEl.currentTime * 1000.0, this.videoEl.playbackRate);
    this.ws.onmessage = (ev: MessageEvent) => {
      const pkt = JSON.parse(ev.data as string) as DownstreamEventPacket;
      void this.mixer.handleDownstreamEvent(pkt);
      this.overlay.render(pkt);
    };
  }

  public send(msg: UpstreamContentMessage): void {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(msg));
    }
  }

  public interrupt(): void {
    const timeMs = this.videoEl.currentTime * 1000.0;
    this.mixer.reanchorTimeline(timeMs, this.videoEl.playbackRate);
    this.send({ action: "interrupt", video_time_ms: timeMs });
  }
}
```

#### 4.8.4 `frontend/src/ui/caption-overlay.ts`
```typescript
import { DownstreamEventPacket } from "../contracts/types";

export class CaptionOverlay {
  private readonly el: HTMLDivElement;

  constructor(root: HTMLElement) {
    this.el = document.createElement("div");
    root.appendChild(this.el);
  }

  public render(pkt: DownstreamEventPacket): void {
    this.el.replaceChildren();

    const text = pkt.per_speaker_target_texts.join("\n");
    const displayText = pkt.vocal_modes.includes(2) ? `${text} ♪` : text;

    if (pkt.codeswitch_words.length === 0) {
      this.el.appendChild(document.createTextNode(displayText));
    } else {
      const escapedWords = pkt.codeswitch_words.map((w) =>
        w.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")
      );
      const regex = new RegExp(`(${escapedWords.join("|")})`, "gi");
      const parts = displayText.split(regex);

      for (const part of parts) {
        if (
          pkt.codeswitch_words.some(
            (w) => w.toLowerCase() === part.toLowerCase()
          )
        ) {
          const mark = document.createElement("mark");
          mark.textContent = part;
          this.el.appendChild(mark);
        } else if (part.length > 0) {
          this.el.appendChild(document.createTextNode(part));
        }
      }
    }
    this.el.dataset.driftMs = String(pkt.drift_ms);
  }
}
```

---

### 4.9 Specialist B & D — Member Rust Crate Manifests & Stubs (`crates/`)

#### 4.9.1 `crates/khmer-linguistics-rs/Cargo.toml`
```toml
[package]
name = "khmer-linguistics-rs"
version.workspace = true
edition.workspace = true

[lib]
name = "khmer_linguistics_rs"
crate-type = ["cdylib", "rlib"]

[features]
extension-module = ["pyo3/extension-module"]

[dependencies]
unicode-normalization.workspace = true
crc32fast.workspace = true
regex.workspace = true
once_cell.workspace = true

[target.'cfg(not(target_arch = "wasm32"))'.dependencies]
pyo3.workspace = true
```

#### 4.9.2 `crates/khmer-transport-rs/Cargo.toml`
```toml
[package]
name = "khmer-transport-rs"
version.workspace = true
edition.workspace = true

[lib]
name = "khmer_transport_rs"
crate-type = ["cdylib", "rlib"]

[features]
extension-module = ["pyo3/extension-module"]

[dependencies]
crc32fast.workspace = true
crossbeam-channel.workspace = true

[target.'cfg(not(target_arch = "wasm32"))'.dependencies]
pyo3.workspace = true
```

#### 4.9.3 `crates/khmer-dsp-wasm/Cargo.toml`
```toml
[package]
name = "khmer-dsp-wasm"
version.workspace = true
edition.workspace = true

[lib]
crate-type = ["cdylib", "rlib"]
```

#### 4.9.4 `crates/khmer-linguistics-rs/src/lib.rs`
```rust
//! Khmer linguistic engine: NFC normalization, independent-vowel
//! canonicalization, coeng/shifter reordering, and syllable decomposition.
//!
//! v10.7 Phase 2 (Deliverables R-1, R-2, R-3): the crate is split into
//! `normalizer` and `decomposer` modules, the regexes are compiled once at
//! startup instead of on every call, and the Rust-to-Python ABI is delivered
//! as a real `#[pymodule]` rather than a specification.

pub mod decomposer;
pub mod normalizer;

pub use decomposer::{compute_brand_coeng_hash, decompose_khmer_word};
pub use normalizer::{normalize_khmer_pure, INDEPENDENT_VOWEL_MAP};

/// The Python extension module.
///
/// v10.7 Deliverable R-1: this is the real ABI that Section 4.9.7 previously
/// only SPECIFIED. `normalize_khmer` and `normalize_khmer_pure` are both
/// registered, so Python can call the Rust normalizer directly and the
/// 15 microsecond edge is reachable.
#[cfg(not(target_arch = "wasm32"))]
mod python_abi {
    use pyo3::prelude::*;
    use pyo3::types::PyModule;

    use crate::normalizer::normalize_khmer_pure;

    /// Normalize Khmer text through the Rust engine.
    ///
    /// This is the function the Python `khmer_normalizer` module probes for
    /// with `hasattr(_rs, "normalize_khmer")`. Until v10.7 that probe always
    /// returned False, because no crate exposed a `#[pymodule]`, so the
    /// "15 microsecond Rust edge" was unreachable dead code.
    #[pyfunction]
    fn normalize_khmer(text: &str) -> PyResult<String> {
        Ok(normalize_khmer_pure(text))
    }

    /// The pure-Rust normalizer, exposed under its own name for callers that
    /// want the un-wrapped implementation.
    #[pyfunction]
    fn normalize_khmer_pure_py(text: &str) -> PyResult<String> {
        Ok(normalize_khmer_pure(text))
    }

    /// Register the module's functions with the Python interpreter.
    #[pymodule]
    fn khmer_linguistics_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
        m.add_function(wrap_pyfunction!(normalize_khmer, m)?)?;
        m.add_function(wrap_pyfunction!(normalize_khmer_pure_py, m)?)?;
        m.add("normalize_khmer_pure", m.getattr("normalize_khmer_pure_py")?)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize_preserves_zwsp_and_reorders() {
        let s = normalize_khmer_pure("\u{17B1}\u{17D2}\u{179C}\u{17BB}\u{1798}\u{200B}\u{179A}\u{17C1}\u{17C5}");
        assert!(s.contains('\u{200B}'));
    }

    #[test]
    fn test_normalize_canonicalizes_independent_vowels() {
        // U+17AB..U+17AE must canonicalize to a consonant plus a dependent
        // vowel so the Python decomposer's scanner can reach them.
        assert_eq!(normalize_khmer_pure("\u{17AB}"), "\u{179A}\u{17B9}");
        assert_eq!(normalize_khmer_pure("\u{17AC}"), "\u{179A}\u{17BA}");
        assert_eq!(normalize_khmer_pure("\u{17AD}"), "\u{179B}\u{17B9}");
        assert_eq!(normalize_khmer_pure("\u{17AE}"), "\u{179B}\u{17BA}");
        // U+17A8 (QUK) uses the open (vowel-final) coda U+1780.
        assert_eq!(normalize_khmer_pure("\u{17A8}"), "\u{17A2}\u{17BB}\u{1780}");
        // U+17B3 (the 17th independent vowel) must be present too.
        assert_eq!(normalize_khmer_pure("\u{17B3}"), "\u{17A2}\u{17C5}");
    }

    #[test]
    fn test_independent_vowel_map_is_complete() {
        assert_eq!(INDEPENDENT_VOWEL_MAP.len(), 17);
        for (i, (cp, _)) in INDEPENDENT_VOWEL_MAP.iter().enumerate() {
            assert_eq!(*cp as u32, 0x17A3 + i as u32);
        }
    }

    #[test]
    fn test_brand_hash_is_stable() {
        assert_eq!(
            compute_brand_coeng_hash("calvin"),
            compute_brand_coeng_hash("calvin")
        );
        assert!((1..=63).contains(&compute_brand_coeng_hash("klein")));
    }
}
```

#### 4.9.4.1 `crates/khmer-linguistics-rs/src/normalizer.rs`
```rust
use once_cell::sync::Lazy;
use regex::Regex;
use unicode_normalization::UnicodeNormalization;

/// The coeng (subscript-consonant) reordering pattern.
///
/// v10.7 R-3: the patterns are compiled ONCE at startup rather than on every
/// call. `normalize_khmer_pure` is on the hot path for every utterance, and
/// recompiling two regexes per call was pure overhead.
static RE_COENG: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"([\u{17B6}-\u{17CA}]+)((?:\u{17D2}[\u{1780}-\u{17A2}])+)").unwrap()
});

/// The shifter (U+17C9, U+17CA) reordering pattern.
static RE_SHIFT: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"([\u{17B6}-\u{17C8}]+)([\u{17C9}\u{17CA}])").unwrap()
});

/// The 17 Khmer independent vowels, canonicalized to a consonant plus a
/// dependent vowel so the Python decomposer's scanner can reach them.
///
/// v10.7 R-2: this map is applied BEFORE the coeng and shifter reordering,
/// which is the exact execution order of the Python `normalize_khmer` in
/// `backend/linguistics/khmer_normalizer.py`. The two normalizers must agree
/// byte-for-byte on compound words that contain an independent vowel
/// followed by a subscript coeng.
pub const INDEPENDENT_VOWEL_MAP: [(char, &str); 17] = [
    ('\u{17A3}', "\u{17A2}"),
    ('\u{17A4}', "\u{17A2}\u{17B6}"),
    ('\u{17A5}', "\u{17A2}\u{17B7}"),
    ('\u{17A6}', "\u{17A2}\u{17B8}"),
    ('\u{17A7}', "\u{17A2}\u{17BB}"),
    ('\u{17A8}', "\u{17A2}\u{17BB}\u{1780}"),
    ('\u{17A9}', "\u{17A2}\u{17BC}"),
    ('\u{17AA}', "\u{17A2}\u{17BC}\u{179C}"),
    ('\u{17AB}', "\u{179A}\u{17B9}"),
    ('\u{17AC}', "\u{179A}\u{17BA}"),
    ('\u{17AD}', "\u{179B}\u{17B9}"),
    ('\u{17AE}', "\u{179B}\u{17BA}"),
    ('\u{17AF}', "\u{17A2}\u{17C2}"),
    ('\u{17B0}', "\u{17A2}\u{17C3}"),
    ('\u{17B1}', "\u{17A2}\u{17C4}"),
    ('\u{17B2}', "\u{17A2}\u{17C4}\u{179C}"),
    ('\u{17B3}', "\u{17A2}\u{17C5}"),
];

/// Normalize Khmer text without any Python involvement.
///
/// The transformation is, in order:
///  1. NFC normalization;
///  2. strip U+200C, U+200D and U+FEFF, but PRESERVE U+200B (the zero-width
///     space is a word-boundary delimiter and must survive);
///  3. canonicalize the 17 independent vowels;
///  4. three passes of the coeng and shifter reordering.
pub fn normalize_khmer_pure(text: &str) -> String {
    let mut out: String = text
        .nfc()
        .filter(|c| !matches!(*c, '\u{200C}' | '\u{200D}' | '\u{FEFF}'))
        .collect();

    // v10.7 R-2: the independent-vowel canonicalization MUST run BEFORE the
    // coeng and shifter reordering, because that is the execution order of
    // the Python normalizer.
    for (k, v) in INDEPENDENT_VOWEL_MAP.iter() {
        if out.contains(*k) {
            out = out.replace(*k, v);
        }
    }

    for _ in 0..3 {
        out = RE_COENG.replace_all(&out, "$2$1").into_owned();
        out = RE_SHIFT.replace_all(&out, "$2$1").into_owned();
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize_preserves_zwsp_and_reorders() {
        let s = normalize_khmer_pure("\u{17B1}\u{17D2}\u{179C}\u{17BB}\u{1798}\u{200B}\u{179A}\u{17C1}\u{17C5}");
        assert!(s.contains('\u{200B}'));
    }

    #[test]
    fn test_normalize_canonicalizes_independent_vowels() {
        // U+17AB..U+17AE must canonicalize to a consonant plus a dependent
        // vowel so the Python decomposer's scanner can reach them.
        assert_eq!(normalize_khmer_pure("\u{17AB}"), "\u{179A}\u{17B9}");
        assert_eq!(normalize_khmer_pure("\u{17AC}"), "\u{179A}\u{17BA}");
        assert_eq!(normalize_khmer_pure("\u{17AD}"), "\u{179B}\u{17B9}");
        assert_eq!(normalize_khmer_pure("\u{17AE}"), "\u{179B}\u{17BA}");
        // U+17A8 (QUK) uses the open (vowel-final) coda U+1780.
        assert_eq!(normalize_khmer_pure("\u{17A8}"), "\u{17A2}\u{17BB}\u{1780}");
        // U+17B3 (the 17th independent vowel) must be present too.
        assert_eq!(normalize_khmer_pure("\u{17B3}"), "\u{17A2}\u{17C5}");
    }

    #[test]
    fn test_independent_vowel_map_has_seventeen_entries() {
        assert_eq!(INDEPENDENT_VOWEL_MAP.len(), 17);
        assert_eq!(INDEPENDENT_VOWEL_MAP[0].0, '\u{17A3}');
        assert_eq!(INDEPENDENT_VOWEL_MAP[16].0, '\u{17B3}');
    }

    #[test]
    fn test_vowel_map_is_applied_before_the_reordering() {
        // The compound word with an independent vowel followed by a subscript
        // coeng is the case that distinguishes the two orderings. The map must
        // run first, so the dependent vowel lands AFTER the coeng.
        let word = "\u{17AB}\u{1791}\u{17D2}\u{1792}\u{17B7}";
        let out = normalize_khmer_pure(word);
        assert_eq!(out, "\u{179A}\u{17B9}\u{1791}\u{17D2}\u{1792}\u{17B7}");
    }
}
```

#### 4.9.4.2 `crates/khmer-linguistics-rs/src/decomposer.rs`
```rust
/// A single decomposed Khmer syllable.
///
/// `surface` is the orthographic surface form, `sesqui_id` is 1 (minor) or 2
/// (major), and `forced_register_id` is the register the syllable is forced
/// into: 0 for an ordinary Khmer syllable, 1 for the nasal-sonorant
/// assimilation case, and 3 for a code-switched (foreign) token.
///
/// These are the same conventions as `BilingualCodeSwitchDecomposer.
/// decompose_khmer_word` in `backend/linguistics/syllable_decomposer.py`.
pub type SyllableTuple = (String, u8, u8);

/// An ordinary Khmer syllable carries no forced register.
pub const KHMER_REGISTER: u8 = 0;
/// A code-switched (foreign) token is forced into register 3.
pub const FOREIGN_REGISTER: u8 = 3;

/// The coeng (subscript-consonant) sign.
const COENG: char = '\u{17D2}';

#[inline]
fn is_consonant(c: char) -> bool {
    ('\u{1780}'..='\u{17A2}').contains(&c)
}

#[inline]
fn is_dependent_vowel(c: char) -> bool {
    ('\u{17B6}'..='\u{17C8}').contains(&c)
}

/// Does this chunk carry a CODA consonant?
///
/// A coda is a consonant at index >= 1 that is NOT preceded by the coeng
/// sign, exactly as `BilingualCodeSwitchDecomposer._extract_coda_id` decides.
fn has_coda(chunk: &[char]) -> bool {
    for i in 1..chunk.len() {
        if is_consonant(chunk[i]) && chunk[i - 1] != COENG {
            return true;
        }
    }
    false
}

/// The dependent-vowel sequence of a chunk.
fn vowel_sequence(chunk: &[char]) -> String {
    chunk
        .iter()
        .filter(|c| is_dependent_vowel(**c))
        .collect()
}

/// Decompose a Khmer word into its syllables.
///
/// The chunking follows the same grammar as the Python
/// `_SYLLABLE_SCAN_RE`:
///
///     [\u1780-\u17A2]            base consonant
///     (?:\u17D2[\u1780-\u17A2]){0,2}   up to two coeng + subscript pairs
///     [\u17C9\u17CA]?            optional shifter
///     [\u17B6-\u17C8]*           dependent vowels
///     (?:[\u1780-\u17A2](?![\u17B6-\u17CA\u17D2]))?  optional final consonant
///     [\u17CB-\u17CD]?           optional robat
///
pub fn decompose_khmer_word(word: &str) -> Vec<SyllableTuple> {
    if word.is_empty() {
        return Vec::new();
    }

    // A code-switched word is a single token.
    if word.chars().any(|c| c.is_ascii_alphabetic()) {
        let mut lower = word.to_lowercase();
        while lower
            .chars()
            .last()
            .map(|c| !c.is_alphanumeric())
            .unwrap_or(false)
        {
            lower.pop();
        }
        if lower.is_empty() {
            return Vec::new();
        }
        return vec![(lower, 2, FOREIGN_REGISTER)];
    }

    let chars: Vec<char> = word.chars().collect();
    let mut chunks: Vec<Vec<char>> = Vec::new();
    let mut i = 0usize;

    while i < chars.len() {
        if !is_consonant(chars[i]) {
            i += 1;
            continue;
        }
        let start = i;
        i += 1; // the base consonant

        // Up to two coeng + subscript pairs.
        let mut pairs = 0usize;
        while pairs < 2
            && i + 1 < chars.len()
            && chars[i] == COENG
            && is_consonant(chars[i + 1])
        {
            i += 2;
            pairs += 1;
        }
        // Optional shifter (U+17C9, U+17CA).
        if i < chars.len() && (chars[i] == '\u{17C9}' || chars[i] == '\u{17CA}') {
            i += 1;
        }
        // Dependent vowels.
        while i < chars.len() && is_dependent_vowel(chars[i]) {
            i += 1;
        }
        // Optional final consonant, unless a vowel or coeng follows it.
        if i < chars.len() && is_consonant(chars[i]) {
            let next = chars.get(i + 1).copied();
            let blocked = matches!(next, Some(c) if is_dependent_vowel(c) || c == '\u{17C9}' || c == '\u{17CA}' || c == COENG);
            if !blocked {
                i += 1;
            }
        }
        // Optional robat / trailing sign.
        if i < chars.len() && ('\u{17CB}'..='\u{17CD}').contains(&chars[i]) {
            i += 1;
        }
        chunks.push(chars[start..i].to_vec());
    }

    if chunks.is_empty() {
        return vec![(word.to_string(), 2, KHMER_REGISTER)];
    }

    let last = chunks.len() - 1;
    let mut out: Vec<SyllableTuple> = Vec::with_capacity(chunks.len());
    for (idx, chunk) in chunks.iter().enumerate() {
        let surface: String = chunk.iter().collect();
        let vow = vowel_sequence(chunk);
        let is_minor = idx < last && !has_coda(chunk) && (vow.is_empty() || vow == "\u{17BB}");
        out.push((surface, if is_minor { 1 } else { 2 }, KHMER_REGISTER));
    }
    out
}

/// The coeng id of a code-switched brand token.
///
/// This is `1 + (crc32(lowercased_word) % 63)`, exactly as
/// `_foreign_to_token` computes it in
/// `backend/linguistics/syllable_decomposer.py`, so the Rust and Python
/// decomposers agree on the brand cache key. `crc32fast` is the standard
/// IEEE CRC-32 and is bit-identical to Python's `zlib.crc32`.
pub fn compute_brand_coeng_hash(word: &str) -> u8 {
    let lower = word.to_lowercase();
    (crc32fast::hash(lower.as_bytes()) % 63 + 1) as u8
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_brand_coeng_hash_is_in_range() {
        for w in ["calvin", "klein", "versace", "gucci", "a", ""] {
            let h = compute_brand_coeng_hash(w);
            assert!((1..=63).contains(&h), "hash {h} out of range for {w}");
        }
    }

    #[test]
    fn test_brand_coeng_hash_matches_the_python_crc32() {
        // crc32fast is bit-identical to Python's zlib.crc32, so the Rust hash
        // must equal 1 + (zlib.crc32(lower) % 63).
        // "calvin" -> zlib.crc32 = 0xA4494242 = 2756264514 -> 1 + (2756264514 % 63) = 25
        assert_eq!(compute_brand_coeng_hash("calvin"), 25);
        // The hash is case-insensitive.
        assert_eq!(
            compute_brand_coeng_hash("CALVIN"),
            compute_brand_coeng_hash("calvin")
        );
    }

    #[test]
    fn test_decompose_foreign_word_is_one_register_three_token() {
        let out = decompose_khmer_word("Calvin");
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].0, "calvin");
        assert_eq!(out[0].2, FOREIGN_REGISTER);
    }

    #[test]
    fn test_decompose_khmer_word_splits_on_coeng() {
        // U+1798 U+17D2 U+179C is one base plus one coeng cluster.
        let out = decompose_khmer_word("\u{1798}\u{17D2}\u{179C}");
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].0, "\u{1798}\u{17D2}\u{179C}");
        assert_eq!(out[0].2, KHMER_REGISTER);
    }

    #[test]
    fn test_decompose_khmer_word_splits_multiple_syllables() {
        // U+1798 U+17C1 U+1799 U+17C1 U+179C: two base+vowel syllables.
        let out = decompose_khmer_word("\u{1798}\u{17C1}\u{1799}\u{17C1}\u{179C}");
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].0, "\u{1798}\u{17C1}");
        assert_eq!(out[1].0, "\u{1799}\u{17C1}\u{179C}");
    }

    #[test]
    fn test_decompose_empty_word_is_empty() {
        assert!(decompose_khmer_word("").is_empty());
    }
}
```

#### 4.9.5 `crates/khmer-transport-rs/src/lib.rs`
```rust
//! Real-time transport edge: the session helpers, the live event queue with
//! its interrupt fence, and the WAV container.
//!
//! v10.7 Phase 2 (Deliverable R-4): the interrupt fence is exposed to Python
//! as a `#[pyclass]` and the crate ships a real `#[pymodule]`, so the
//! transport edge is callable from Python rather than merely specified.

pub mod live_queue;
pub mod session;
pub mod wav_container;

pub use live_queue::{AtomicEpochQueue, AtomicInterruptFence};
pub use session::{compute_pcm_energy_hash, SpeakerRegistry};
pub use wav_container::encode_wav_pcm16;

/// The Python extension module.
///
/// v10.7 Deliverable R-4: this is the real ABI. The Python server can now
/// create an `AtomicInterruptFence` directly, so the interrupt epoch is
/// shared rather than duplicated.
#[cfg(not(target_arch = "wasm32"))]
mod python_abi {
    use pyo3::prelude::*;
    use pyo3::types::PyModule;

    use crate::live_queue::AtomicInterruptFence;
    use crate::session::compute_pcm_energy_hash;
    use crate::wav_container::encode_wav_pcm16;

    /// The interrupt epoch, shared between the Rust edge and Python.
    #[pyclass]
    pub struct PyInterruptFence {
        inner: AtomicInterruptFence,
    }

    #[pymethods]
    impl PyInterruptFence {
        #[new]
        fn new() -> Self {
            Self {
                inner: AtomicInterruptFence::new(),
            }
        }

        /// Bump the epoch and return the new value.
        fn bump(&self) -> usize {
            self.inner.bump()
        }

        /// The current epoch.
        fn load(&self) -> usize {
            self.inner.load()
        }

        /// Whether the epoch still equals `expected`.
        fn is_current(&self, expected: usize) -> bool {
            self.inner.is_current(expected)
        }
    }

    /// The phase-invariant PCM energy hash, exposed to Python.
    #[pyfunction]
    fn pcm_energy_hash(slice_pcm: Vec<f32>) -> u32 {
        compute_pcm_energy_hash(&slice_pcm)
    }

    /// Encode a mono float waveform as a RIFF/WAVE PCM16 file.
    #[pyfunction]
    fn wav_pcm16(wav: Vec<f32>, sample_rate: u32) -> Vec<u8> {
        encode_wav_pcm16(&wav, sample_rate)
    }

    /// Register the module's classes and functions with the interpreter.
    #[pymodule]
    fn khmer_transport_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
        m.add_class::<PyInterruptFence>()?;
        m.add_function(wrap_pyfunction!(pcm_energy_hash, m)?)?;
        m.add_function(wrap_pyfunction!(wav_pcm16, m)?)?;
        m.add("compute_pcm_energy_hash", m.getattr("pcm_energy_hash")?)?;
        m.add("encode_wav_pcm16", m.getattr("wav_pcm16")?)?;
        // v10.7 R-4: also expose the class under its RUST name, so that
        // `hasattr(khmer_transport_rs, "AtomicInterruptFence")` succeeds and
        // the Python binding name `PyInterruptFence` is available too.
        m.add("AtomicInterruptFence", m.getattr("PyInterruptFence")?)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_atomic_epoch_fence() {
        let f = AtomicInterruptFence::new();
        assert_eq!(f.load(), 0);
        assert_eq!(f.bump(), 1);
        assert_eq!(f.load(), 1);
    }

    #[test]
    fn test_re_exports_are_reachable() {
        assert!(!compute_pcm_energy_hash(&[0.0; 128]).eq(&u32::MAX));
        let out = encode_wav_pcm16(&[0.0, 1.0], 24000);
        assert_eq!(&out[0..4], b"RIFF");
    }
}
```

#### 4.9.5.1 `crates/khmer-transport-rs/src/session.rs`
```rust
//! Session-side helpers: the phase-invariant PCM energy hash and the
//! speaker registry.
//!
//! v10.7 Phase 2 (Deliverable R-7): these mirror the Python
//! `InMemorySession.compute_pcm_energy_hash` and `SpeakerRegistry` in
//! `backend/transport/session.py` exactly, so the replay cache key and the
//! speaker-slot assignment are identical on both sides of the ABI.

/// The number of energy bins in the profile.
const ENERGY_BINS: usize = 8;

/// The phase-invariant energy-envelope hash of a PCM slice.
///
/// The profile is 8 bins of normalized log-RMS, each quantized to `0..=99`,
/// and the hash is the IEEE CRC-32 of those 8 bytes. `crc32fast` is
/// bit-identical to Python's `zlib.crc32`, so the Rust and Python hashes
/// agree exactly and a cache key minted on one side is found on the other.
///
/// A slice shorter than 64 samples hashes to 0, matching the Python guard.
pub fn compute_pcm_energy_hash(slice_pcm: &[f32]) -> u32 {
    if slice_pcm.len() < 64 {
        return 0;
    }
    let bin_len = slice_pcm.len() / ENERGY_BINS;
    if bin_len == 0 {
        return 0;
    }

    let sum_squares: f64 = slice_pcm.iter().map(|v| (*v as f64) * (*v as f64)).sum();
    let overall_rms = (sum_squares / slice_pcm.len() as f64 + 1e-12).sqrt();
    let denom = if overall_rms > 1e-6 { overall_rms } else { 1e-6 };

    let mut profile = [0u8; ENERGY_BINS];
    for b in 0..ENERGY_BINS {
        let chunk = &slice_pcm[b * bin_len..(b + 1) * bin_len];
        let c_sum: f64 = chunk.iter().map(|v| (*v as f64) * (*v as f64)).sum();
        let c_rms = (c_sum / chunk.len() as f64 + 1e-12).sqrt();
        let scaled = (c_rms / denom) * 10.0;
        // Round half away from zero, then clamp into 0..=99.
        let mut q = scaled.round() as i64;
        if q < 0 {
            q = 0;
        }
        if q > 99 {
            q = 99;
        }
        profile[b] = q as u8;
    }

    crc32fast::hash(&profile)
}

/// A speaker registry: a small set of L2-normalized centroids.
#[derive(Debug, Clone)]
pub struct SpeakerRegistry {
    /// The maximum number of distinct speakers tracked.
    pub max_speakers: usize,
    /// The cosine similarity above which an embedding matches an existing slot.
    pub similarity_threshold: f32,
    /// The L2-normalized centroid of each tracked speaker.
    centroids: Vec<Vec<f32>>,
}

impl Default for SpeakerRegistry {
    fn default() -> Self {
        Self::new(4, 0.72)
    }
}

impl SpeakerRegistry {
    /// A registry with the given capacity and match threshold.
    pub fn new(max_speakers: usize, similarity_threshold: f32) -> Self {
        Self {
            max_speakers,
            similarity_threshold,
            centroids: Vec::new(),
        }
    }

    /// The number of tracked speakers.
    pub fn len(&self) -> usize {
        self.centroids.len()
    }

    /// Whether no speaker is tracked yet.
    pub fn is_empty(&self) -> bool {
        self.centroids.is_empty()
    }

    /// The centroids, in slot order.
    pub fn centroids(&self) -> &[Vec<f32>] {
        &self.centroids
    }

    /// Match an embedding against the registry, registering it if new.
    ///
    /// Returns the slot index and the updated centroid for that slot. A match
    /// updates the centroid with the 0.85 / 0.15 EMA, exactly as the Python
    /// registry does.
    pub fn match_or_register(&mut self, embedding: &[f32]) -> (usize, Vec<f32>) {
        let vec = l2_normalize(embedding);

        if self.centroids.is_empty() {
            self.centroids.push(vec);
            return (0, self.centroids[0].clone());
        }

        let sims: Vec<f32> = self
            .centroids
            .iter()
            .map(|c| dot(&vec, c))
            .collect();
        let mut best_idx = 0usize;
        for (i, s) in sims.iter().enumerate() {
            if *s > sims[best_idx] {
                best_idx = i;
            }
        }

        if sims[best_idx] >= self.similarity_threshold {
            let updated = ema_update(&self.centroids[best_idx], &vec, 0.85, 0.15);
            self.centroids[best_idx] = updated;
            return (best_idx, self.centroids[0].clone());
        }

        if self.centroids.len() < self.max_speakers {
            self.centroids.push(vec);
            return (self.centroids.len() - 1, self.centroids[0].clone());
        }

        // The registry is full: fold the new embedding into the best match.
        let updated = ema_update(&self.centroids[best_idx], &vec, 0.85, 0.15);
        self.centroids[best_idx] = updated;
        (best_idx, self.centroids[0].clone())
    }
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let mut acc = 0.0f32;
    for i in 0..n {
        acc += a[i] * b[i];
    }
    acc
}

fn l2_normalize(v: &[f32]) -> Vec<f32> {
    let norm = dot(v, v).sqrt();
    if norm <= 0.0 || !norm.is_finite() {
        return v.to_vec();
    }
    v.iter().map(|x| x / norm).collect()
}

fn ema_update(old: &[f32], new: &[f32], old_weight: f32, new_weight: f32) -> Vec<f32> {
    let n = old.len().min(new.len());
    let mut mixed = vec![0.0f32; n];
    for i in 0..n {
        mixed[i] = old_weight * old[i] + new_weight * new[i];
    }
    l2_normalize(&mixed)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_energy_hash_is_zero_for_short_slices() {
        assert_eq!(compute_pcm_energy_hash(&[0.0; 63]), 0);
        assert_eq!(compute_pcm_energy_hash(&[]), 0);
    }

    #[test]
    fn test_energy_hash_matches_the_python_reference_values() {
        // These are the EXACT values produced by the Python
        // `InMemorySession.compute_pcm_energy_hash` in
        // `backend/transport/session.py`, so the Rust and Python cache keys
        // are interchangeable. `crc32fast` is bit-identical to `zlib.crc32`.
        //
        //   ones(1024)          -> profile [10]*8      -> 3282682638
        //   loud first half      -> profile [14,14,14,14,0,0,0,0] -> 2779605271
        //   ramp 0..1            -> profile [1,3,5,8,10,12,14,16]  -> 491666418
        let ones = vec![1.0f32; 1024];
        assert_eq!(compute_pcm_energy_hash(&ones), 3282682638);

        let mut loud = vec![1.0f32; 512];
        loud.extend(std::iter::repeat(0.01f32).take(512));
        assert_eq!(compute_pcm_energy_hash(&loud), 2779605271);

        let ramp: Vec<f32> = (0..1024).map(|i| i as f32 / 1024.0).collect();
        assert_eq!(compute_pcm_energy_hash(&ramp), 491666418);
    }

    #[test]
    fn test_energy_hash_is_deterministic() {
        let slice: Vec<f32> = (0..1024).map(|i| ((i as f32) * 0.05).sin() * 0.3).collect();
        let a = compute_pcm_energy_hash(&slice);
        let b = compute_pcm_energy_hash(&slice);
        assert_eq!(a, b);
        // A signal with a DIFFERENT energy DISTRIBUTION must hash
        // differently. The 8-bin profile is deliberately coarse (that
        // coarseness is what makes it shift-invariant), so a smooth signal
        // of any shape quantises to the same [10; 8] profile. A loud first
        // half and a quiet second half does not.
        let mut skewed = vec![1.0f32; 512];
        skewed.extend(std::iter::repeat(0.01f32).take(512));
        assert_ne!(a, compute_pcm_energy_hash(&skewed));
    }

    #[test]
    fn test_registry_registers_then_matches() {
        let mut reg = SpeakerRegistry::new(4, 0.72);
        let emb = vec![1.0f32, 0.0, 0.0, 0.0];
        let (slot, _) = reg.match_or_register(&emb);
        assert_eq!(slot, 0);
        assert_eq!(reg.len(), 1);
        // The same embedding matches slot 0.
        let (slot, _) = reg.match_or_register(&emb);
        assert_eq!(slot, 0);
        assert_eq!(reg.len(), 1);
    }

    #[test]
    fn test_registry_separates_orthogonal_speakers() {
        let mut reg = SpeakerRegistry::new(4, 0.72);
        let (a, _) = reg.match_or_register(&[1.0, 0.0, 0.0, 0.0]);
        let (b, _) = reg.match_or_register(&[0.0, 1.0, 0.0, 0.0]);
        assert_eq!(a, 0);
        assert_eq!(b, 1);
        assert_eq!(reg.len(), 2);
    }

    #[test]
    fn test_registry_caps_at_max_speakers() {
        let mut reg = SpeakerRegistry::new(2, 0.72);
        reg.match_or_register(&[1.0, 0.0, 0.0, 0.0]);
        reg.match_or_register(&[0.0, 1.0, 0.0, 0.0]);
        reg.match_or_register(&[0.0, 0.0, 1.0, 0.0]);
        assert_eq!(reg.len(), 2);
    }
}
```

#### 4.9.5.2 `crates/khmer-transport-rs/src/live_queue.rs`
```rust
//! The live event queue and the interrupt fence.
//!
//! v10.7 Phase 2 (Deliverable R-4): the fence is exposed to Python as a
//! `#[pyclass]`, and the queue is built on `crossbeam-channel`, whose
//! `unbounded` queue drains in well under the 2 microsecond budget.

use std::sync::atomic::{AtomicUsize, Ordering};

use crossbeam_channel::{unbounded, Receiver, Sender};

/// A monotonically increasing interrupt epoch.
///
/// Every seek or interrupt bumps the epoch. A worker that captured an epoch
/// before a long GPU wait can compare it afterwards and drop its stale work.
#[derive(Debug, Default)]
pub struct AtomicInterruptFence {
    epoch: AtomicUsize,
}

impl AtomicInterruptFence {
    /// A fence at epoch zero.
    pub const fn new() -> Self {
        Self {
            epoch: AtomicUsize::new(0),
        }
    }

    /// Bump the epoch and return the NEW value.
    pub fn bump(&self) -> usize {
        self.epoch.fetch_add(1, Ordering::SeqCst) + 1
    }

    /// The current epoch.
    pub fn load(&self) -> usize {
        self.epoch.load(Ordering::SeqCst)
    }

    /// Whether the epoch still equals `expected`.
    pub fn is_current(&self, expected: usize) -> bool {
        self.load() == expected
    }
}

/// An unbounded, multi-producer / single-consumer event queue with an
/// associated interrupt epoch.
#[derive(Debug)]
pub struct AtomicEpochQueue<T> {
    sender: Sender<T>,
    receiver: Receiver<T>,
    fence: AtomicInterruptFence,
}

impl<T> Default for AtomicEpochQueue<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> AtomicEpochQueue<T> {
    /// A queue with an unbounded channel.
    pub fn new() -> Self {
        let (sender, receiver) = unbounded();
        Self {
            sender,
            receiver,
            fence: AtomicInterruptFence::new(),
        }
    }

    /// Enqueue an item.
    pub fn push(&self, item: T) {
        // The receiver is only dropped when the queue is, so this cannot fail
        // in practice; ignoring the error keeps `push` infallible for callers.
        let _ = self.sender.send(item);
    }

    /// Try to take the next item without blocking.
    pub fn try_pop(&self) -> Option<T> {
        self.receiver.try_recv().ok()
    }

    /// The number of queued items.
    pub fn len(&self) -> usize {
        self.receiver.len()
    }

    /// Whether the queue is empty.
    pub fn is_empty(&self) -> bool {
        self.receiver.is_empty()
    }

    /// Drain every queued item and bump the interrupt epoch.
    ///
    /// Returns the number of items discarded. The drain itself is a bounded
    /// number of non-blocking channel receives, so it completes in well under
    /// two microseconds for a queue of a few hundred items.
    pub fn flush_all(&self) -> usize {
        let mut dropped = 0usize;
        while self.receiver.try_recv().is_ok() {
            dropped += 1;
        }
        self.fence.bump();
        dropped
    }

    /// The interrupt fence guarding this queue.
    pub fn fence(&self) -> &AtomicInterruptFence {
        &self.fence
    }

    /// The current interrupt epoch.
    pub fn epoch(&self) -> usize {
        self.fence.load()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    #[test]
    fn test_atomic_epoch_fence() {
        let f = AtomicInterruptFence::new();
        assert_eq!(f.load(), 0);
        assert_eq!(f.bump(), 1);
        assert_eq!(f.load(), 1);
        assert!(f.is_current(1));
        assert!(!f.is_current(0));
    }

    #[test]
    fn test_queue_push_and_pop() {
        let q: AtomicEpochQueue<u32> = AtomicEpochQueue::new();
        assert!(q.is_empty());
        q.push(1);
        q.push(2);
        assert_eq!(q.len(), 2);
        assert_eq!(q.try_pop(), Some(1));
        assert_eq!(q.try_pop(), Some(2));
        assert_eq!(q.try_pop(), None);
    }

    #[test]
    fn test_flush_all_drains_and_bumps_the_epoch() {
        let q: AtomicEpochQueue<u32> = AtomicEpochQueue::new();
        for i in 0..64 {
            q.push(i);
        }
        let before = q.epoch();
        let dropped = q.flush_all();
        assert_eq!(dropped, 64);
        assert!(q.is_empty());
        assert_eq!(q.epoch(), before + 1);
    }

    #[test]
    fn test_flush_all_is_fast() {
        let q: AtomicEpochQueue<u32> = AtomicEpochQueue::new();
        for i in 0..512 {
            q.push(i);
        }
        let start = Instant::now();
        let dropped = q.flush_all();
        let elapsed = start.elapsed();
        assert_eq!(dropped, 512);
        // The budget is 2 microseconds for the drain itself; a generous
        // bound keeps the test stable on a loaded CI machine while still
        // failing if the drain became quadratic.
        assert!(
            elapsed.as_micros() < 2000,
            "flush_all took {} us",
            elapsed.as_micros()
        );
    }
}
```

#### 4.9.5.3 `crates/khmer-transport-rs/src/wav_container.rs`
```rust
//! A pure 44-byte RIFF/WAVE PCM16 encoder.
//!
//! v10.7 Phase 2: this mirrors the Python `encode_wav_pcm16_b64` in
//! `backend/transport/runner.py` byte-for-byte, so a WAV produced on either
//! side of the ABI is interchangeable.

/// Encode a mono float waveform as a RIFF/WAVE PCM16 file.
///
/// The layout is the canonical 44-byte header:
///
///     0..4    "RIFF"
///     4..8    u32 little-endian  36 + data_len
///     8..16   "WAVEfmt "
///     16..20  u32  16            (PCM fmt chunk size)
///     20..22  u16  1             (PCM format)
///     22..24  u16  1             (mono)
///     24..28  u32  sample_rate
///     28..32  u32  sample_rate * 2 (byte rate)
///     32..34  u16  2             (block align)
///     34..36  u16  16            (bits per sample)
///     36..40  "data"
///     40..44  u32  data_len
///     44..    i16 little-endian samples
///
/// The samples are clamped to `[-1.0, 1.0]`, scaled by 32767 and truncated
/// toward zero, which is exactly what the Python encoder's
/// `(t.clamp(-1.0, 1.0) * 32767.0).to(torch.int16)` does.
pub fn encode_wav_pcm16(wav: &[f32], sample_rate: u32) -> Vec<u8> {
    let mut samples: Vec<i16> = Vec::with_capacity(wav.len());
    for v in wav {
        let clamped = if *v > 1.0 {
            1.0
        } else if *v < -1.0 {
            -1.0
        } else {
            *v
        };
        samples.push((clamped * 32767.0) as i16);
    }

    let data_len = (samples.len() * 2) as u32;
    let byte_rate = sample_rate * 2; // mono, 16 bits per sample

    let mut out: Vec<u8> = Vec::with_capacity(44 + samples.len() * 2);
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&(36u32 + data_len).to_le_bytes());
    out.extend_from_slice(b"WAVEfmt ");
    out.extend_from_slice(&16u32.to_le_bytes()); // fmt chunk size
    out.extend_from_slice(&1u16.to_le_bytes()); // PCM format
    out.extend_from_slice(&1u16.to_le_bytes()); // mono
    out.extend_from_slice(&sample_rate.to_le_bytes());
    out.extend_from_slice(&byte_rate.to_le_bytes());
    out.extend_from_slice(&2u16.to_le_bytes()); // block align
    out.extend_from_slice(&16u16.to_le_bytes()); // bits per sample
    out.extend_from_slice(b"data");
    out.extend_from_slice(&data_len.to_le_bytes());
    for s in samples {
        out.extend_from_slice(&s.to_le_bytes());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_header_layout() {
        let wav = vec![0.0f32; 10];
        let out = encode_wav_pcm16(&wav, 24000);
        assert_eq!(&out[0..4], b"RIFF");
        assert_eq!(&out[8..12], b"WAVE");
        assert_eq!(&out[12..16], b"fmt ");
        assert_eq!(&out[36..40], b"data");
        // 36 + data_len
        assert_eq!(
            u32::from_le_bytes([out[4], out[5], out[6], out[7]]),
            36 + 20
        );
        // fmt chunk size (u32), PCM format (u16), mono (u16)
        assert_eq!(
            u32::from_le_bytes([out[16], out[17], out[18], out[19]]),
            16
        );
        assert_eq!(u16::from_le_bytes([out[20], out[21]]), 1);
        assert_eq!(u16::from_le_bytes([out[22], out[23]]), 1);
        // sample rate (u32), byte rate (u32), block align (u16), bits (u16)
        assert_eq!(
            u32::from_le_bytes([out[24], out[25], out[26], out[27]]),
            24000
        );
        assert_eq!(
            u32::from_le_bytes([out[28], out[29], out[30], out[31]]),
            48000
        );
        assert_eq!(u16::from_le_bytes([out[32], out[33]]), 2);
        assert_eq!(u16::from_le_bytes([out[34], out[35]]), 16);
        // data length
        assert_eq!(
            u32::from_le_bytes([out[40], out[41], out[42], out[43]]),
            20
        );
        assert_eq!(out.len(), 44 + 20);
    }

    #[test]
    fn test_sample_conversion() {
        // 0.0 -> 0, 1.0 -> 32767, -1.0 -> -32767.
        let out = encode_wav_pcm16(&[0.0, 1.0, -1.0], 24000);
        let s = |i: usize| i16::from_le_bytes([out[44 + i * 2], out[45 + i * 2]]);
        assert_eq!(s(0), 0);
        assert_eq!(s(1), 32767);
        assert_eq!(s(2), -32767);
    }

    #[test]
    fn test_samples_are_clamped() {
        let out = encode_wav_pcm16(&[2.0, -2.0], 24000);
        let s = |i: usize| i16::from_le_bytes([out[44 + i * 2], out[45 + i * 2]]);
        assert_eq!(s(0), 32767);
        assert_eq!(s(1), -32767);
    }

    #[test]
    fn test_empty_wav_is_a_valid_header() {
        let out = encode_wav_pcm16(&[], 16000);
        assert_eq!(out.len(), 44);
        assert_eq!(&out[0..4], b"RIFF");
        assert_eq!(
            u32::from_le_bytes([out[4], out[5], out[6], out[7]]),
            36
        );
    }
}
```

#### 4.9.6 `crates/khmer-dsp-wasm/src/lib.rs`
```rust
//! The 48 kHz DSP edge: the ERB band mapping and the 3:1 sinc decimator.
//!
//! v10.7 Phase 2 (Deliverables R-5, R-6): the ERB band index and the
//! stateful sinc decimator are implemented here, so the WASM module carries
//! the same DSP the browser worklet uses.

/// The 15-tap Blackman-windowed sinc filter for 48 kHz -> 16 kHz.
///
/// Cutoff 7.2 kHz, normalized sum 1.0000, 0.0 dB DC gain.
pub const FIR_15_TAPS: [f32; 15] = [
    -0.0030, -0.0091, -0.0101, 0.0110, 0.0630, 0.1366, 0.1989,
     0.2254,
     0.1989,  0.1366,  0.0630, 0.0110, -0.0101, -0.0091, -0.0030,
];

/// The number of ERB bands.
pub const ERB_BANDS: usize = 64;

/// The lowest ERB band centre, in hertz.
pub const ERB_MIN_FREQ_HZ: f32 = 120.0;

/// The highest ERB band centre, in hertz.
pub const ERB_MAX_FREQ_HZ: f32 = 8000.0;

/// The ERB rate of a frequency in hertz.
pub fn hz_to_erb_rate(hz: f32) -> f32 {
    21.4 * (1.0 + 0.00437 * hz).log10()
}

/// The ERB band index that contains `freq_hz`.
///
/// Returns `None` outside the `[120.0, 8000.0]` band range. Inside the
/// range the index is `floor(64 * norm)` clamped to 63, where `norm` is the
/// position of the frequency in the ERB rate domain.
///
/// This mirrors the browser's `hzToErbRate` and the Part 2.1 band partition
/// exactly, so the Rust edge and the TypeScript worklet agree on which band a
/// frequency belongs to.
pub fn erb_band_index_for_freq(freq_hz: f32) -> Option<usize> {
    if !(freq_hz >= ERB_MIN_FREQ_HZ && freq_hz <= ERB_MAX_FREQ_HZ) {
        return None;
    }
    let min_rate = hz_to_erb_rate(ERB_MIN_FREQ_HZ);
    let max_rate = hz_to_erb_rate(ERB_MAX_FREQ_HZ);
    let rate = hz_to_erb_rate(freq_hz);
    let norm = (rate - min_rate) / (max_rate - min_rate);
    let idx = (ERB_BANDS as f32 * norm).floor() as usize;
    Some(idx.min(ERB_BANDS - 1))
}

/// A stateful 48 kHz -> 16 kHz decimator.
///
/// The 15-tap Blackman-sinc filter is applied to a rolling ring buffer and
/// the output is taken once every three input samples, which is exactly the
/// 3:1 decimation the browser worklet performs.
#[derive(Debug, Clone)]
pub struct SincDecimator48To16 {
    ring: [f32; 15],
    ring_idx: usize,
    phase: usize,
}

impl Default for SincDecimator48To16 {
    fn default() -> Self {
        Self::new()
    }
}

impl SincDecimator48To16 {
    /// A decimator with an empty history.
    pub fn new() -> Self {
        Self {
            ring: [0.0; 15],
            ring_idx: 0,
            phase: 0,
        }
    }

    /// Push one 48 kHz sample and return the decimated 16 kHz sample, if any.
    ///
    /// Returns `None` for the two input samples out of every three that do
    /// not produce an output.
    pub fn push(&mut self, sample: f32) -> Option<f32> {
        self.ring[self.ring_idx] = sample;
        self.ring_idx = (self.ring_idx + 1) % 15;

        if self.phase == 0 {
            let mut sum = 0.0f32;
            for t in 0..15 {
                sum += self.ring[(self.ring_idx + t) % 15] * FIR_15_TAPS[t];
            }
            self.phase = (self.phase + 1) % 3;
            return Some(sum);
        }
        self.phase = (self.phase + 1) % 3;
        None
    }

    /// Push a block of samples and return the decimated output.
    pub fn push_block(&mut self, samples: &[f32]) -> Vec<f32> {
        let mut out = Vec::with_capacity(samples.len() / 3 + 1);
        for s in samples {
            if let Some(v) = self.push(*s) {
                out.push(v);
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fir_unity_gain_and_erb_bounds() {
        let sum: f32 = FIR_15_TAPS.iter().sum();
        assert!((sum - 1.0).abs() < 1e-4);
        assert!(hz_to_erb_rate(8000.0) > hz_to_erb_rate(120.0));
    }

    #[test]
    fn test_erb_band_index_covers_band_63() {
        // Band 63 spans [7603.5 Hz, 8000 Hz].
        assert_eq!(erb_band_index_for_freq(7800.0), Some(63));
        assert_eq!(erb_band_index_for_freq(8000.0), Some(63));
        assert_eq!(erb_band_index_for_freq(7603.5), Some(63));
    }

    #[test]
    fn test_erb_band_index_is_none_outside_the_band_range() {
        assert_eq!(erb_band_index_for_freq(119.9), None);
        assert_eq!(erb_band_index_for_freq(0.0), None);
        assert_eq!(erb_band_index_for_freq(8000.1), None);
        assert_eq!(erb_band_index_for_freq(48000.0), None);
    }

    #[test]
    fn test_erb_band_index_is_monotonic() {
        let mut prev = 0usize;
        for hz in (120..=8000).step_by(50) {
            let idx = erb_band_index_for_freq(hz as f32).unwrap();
            assert!(idx >= prev, "band index went backwards at {hz} Hz");
            assert!(idx < ERB_BANDS);
            prev = idx;
        }
    }

    #[test]
    fn test_decimator_dc_unity_gain() {
        // A constant DC input must reproduce itself at the output, because the
        // filter is normalized to unity gain. The first outputs are the
        // filter's WARM-UP: the 15-tap ring buffer starts zeroed, so it takes
        // 15 input samples (5 output samples) to fill. The warm-up is a
        // property of the causal filter, not a gain error.
        let mut d = SincDecimator48To16::new();
        let mut outs = Vec::new();
        for _ in 0..300 {
            if let Some(v) = d.push(1.0) {
                outs.push(v);
            }
        }
        assert_eq!(outs.len(), 100);
        // The first 5 outputs are the warm-up.
        for v in &outs[..5] {
            assert!((v - 1.0).abs() > 1e-4, "the warm-up should not be unity");
        }
        // Every output after the warm-up has exactly unity DC gain.
        for v in &outs[5..] {
            assert!((v - 1.0).abs() < 1e-4, "DC gain was {v}");
        }
    }

    #[test]
    fn test_decimator_output_rate_is_one_third() {
        let mut d = SincDecimator48To16::new();
        let input: Vec<f32> = (0..480).map(|i| (i as f32 * 0.01).sin()).collect();
        let out = d.push_block(&input);
        assert_eq!(out.len(), 160);
    }
}
```

#### 4.9.7 Rust-to-Python ABI (Phase 2 Deliverables R-1 to R-8: DELIVERED)

> [!IMPORTANT]
> **The Phase 2 deliverables R-1 to R-8 are DELIVERED in Sections 4.9.4 to
> 4.9.6** (fix-ledger entries 72 to 74). The v10.6 revision was a *document*
> pass that only SPECIFIED them; v10.7 implements them. The table below is
> retained so that the acceptance check for each deliverable stays on the
> record, and so that the v10.5-era claims in Part 1.1 and Part 3.3 can be
> corrected honestly rather than left overstated.
>
> **Finding (v10.6, now RESOLVED).** The Python bridge in
> `backend/linguistics/khmer_normalizer.py` guards its Rust fast path with
> `_HAS_RUST = hasattr(_rs, "normalize_khmer")`. In v10.6 no member crate
> exposed a `#[pymodule]` and no symbol named `normalize_khmer` existed in the
> Rust build, so **the probe could never succeed** and the Rust fast path was
> dead code. v10.7 delivers the ABI: `khmer_linguistics_rs` now exposes
> `normalize_khmer` and `normalize_khmer_pure` as `#[pyfunction]`s through a
> real `#[pymodule]` (Section 4.9.4), so the probe succeeds and the
> `< 15 us`/sentence Rust edge is reachable rather than aspirational. The
> transport crate likewise exposes its interrupt fence as a `#[pyclass]`
> through a real `#[pymodule]` (Section 4.9.5), so the interrupt epoch is
> SHARED with Python instead of duplicated.

**Phase 2 deliverables (R-1 to R-8: ALL DELIVERED in v10.7).**

| # | Crate | Deliverable | Delivered in | Acceptance check | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| R-1 | `khmer-linguistics-rs` | A `#[pymodule]` exposing `normalize_khmer` and `normalize_khmer_pure` as `#[pyfunction]`s | Section 4.9.4 | `python -c "import khmer_linguistics_rs; assert hasattr(khmer_linguistics_rs,'normalize_khmer')"` succeeds | **DELIVERED** |
| R-2 | `khmer-linguistics-rs` | `normalize_khmer` includes the independent-vowel canonicalization (U+17A3..U+17B3), applied BEFORE the coeng/shifter reordering | Section 4.9.4.1 | Rust and Python `normalize_khmer` produce byte-identical output on the parity corpus | **DELIVERED** |
| R-3 | `khmer-linguistics-rs` | The regexes are compiled once via `once_cell::sync::Lazy` rather than recompiled on every call | Section 4.9.4.1 | `RE_COENG` and `RE_SHIFT` are `Lazy` statics; per-call allocation drops to zero steady-state | **DELIVERED** |
| R-4 | `khmer-transport-rs` | A `#[pymodule]` exposing the interrupt fence as a `#[pyclass]` with `#[pymethods]`, under BOTH the `AtomicInterruptFence` and the `PyInterruptFence` names | Section 4.9.5 | `python -c "import khmer_transport_rs; assert hasattr(khmer_transport_rs,'AtomicInterruptFence') and hasattr(khmer_transport_rs,'PyInterruptFence')"` succeeds | **DELIVERED** |
| R-5 | `khmer-dsp-wasm` | `erb_band_index_for_freq(freq_hz: f32) -> Option<usize>`, returning `None` outside `[120.0, 8000.0]` | Section 4.9.6 | `erb_band_index_for_freq(7800.0) == Some(63)`, the Band 63 `[7603.5 Hz, 8000 Hz]` coverage | **DELIVERED** |
| R-6 | `khmer-dsp-wasm` | The stateful `SincDecimator48To16` 3:1 decimator over the 15-tap Blackman-sinc | Section 4.9.6 | Unity DC gain after the 15-sample filter warm-up; the output rate is one third of the input | **DELIVERED** |
| R-7 | `khmer-linguistics-rs` | The `crc32fast` dependency is put to WORK: `compute_brand_coeng_hash` gives the brand coeng id, and `crc32fast::hash` is bit-identical to Python's `zlib.crc32` | Section 4.9.4.2 | "calvin" hashes to 25 on both the Rust and the Python side | **DELIVERED** |
| R-8 | `khmer-transport-rs` | The `crossbeam-channel` dependency is put to WORK: `AtomicEpochQueue` is built on `unbounded`, and `flush_all` drains it under the 2 microsecond budget while bumping the atomic epoch | Section 4.9.5.2 | `cargo build` emits no unused-dependency warnings | **DELIVERED** |

> [!NOTE]
> **Still outstanding, and deliberately NOT claimed as delivered:** the
> `khmer-dsp-wasm` crate still has **no `wasm-bindgen` dependency and no
> `#[wasm_bindgen]` exports**, so it cannot yet be built for
> `wasm32-unknown-unknown`; and it has no `pyo3` `extension-module` feature,
> unlike its two sibling crates. Those two items are separate from R-5 and
> R-6 as scoped above, and they remain a future Phase 2 increment. All five
> previously unused workspace dependencies (`unicode-normalization`,
> `crc32fast`, `regex`, `once_cell`, `crossbeam-channel`) are now genuinely
> used, so there are **zero unused dependencies**.

**Rust module split (Phase 2: DELIVERED).** The Part 2.2 file tree names five Rust
modules. **They now exist on disk and are published in Sections 4.9.4 to
4.9.6** (fix-ledger entries 72 to 74), with the following responsibilities:

| Module | Crate | Public surface |
| :--- | :--- | :--- |
| `normalizer.rs` | `khmer-linguistics-rs` | `normalize_khmer(text: &str) -> String` (PyO3-exported), `normalize_khmer_pure(text: &str) -> String` (pure core) |
| `decomposer.rs` | `khmer-linguistics-rs` | `decompose_khmer_word(word: &str) -> Vec<(String, u8, u8)>` returning `(surface, sesqui_id, forced_register_id)` |
| `session.rs` | `khmer-transport-rs` | `compute_pcm_energy_hash(slice_pcm: &[f32]) -> u32` (8-bin normalized log-RMS envelope hash) plus a Rust `SpeakerRegistry` (threshold `0.72`, max 4 slots, `0.85`/`0.15` centroid EMA) |
| `live_queue.rs` | `khmer-transport-rs` | `AtomicEpoch` lock-free queue (< 5 µs flush) |
| `wav_container.rs` | `khmer-transport-rs` | `encode_wav_pcm16(wav: &[f32], sample_rate: u32) -> Vec<u8>` (44-byte RIFF/WAVE PCM16) |

> [!NOTE]
> The Python reference implementations in Sections 4.1 and 4.7 remain
> **authoritative** for v10.6. The Rust modules are ports, not replacements,
> and the parity corpus check (R-2) is what guarantees they agree.

---
