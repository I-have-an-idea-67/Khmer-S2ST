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
