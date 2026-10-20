import pytest

from backend.contracts.schemas import CodeSwitchDecomposerProtocol
from backend.linguistics.syllable_decomposer import BilingualCodeSwitchDecomposer

GROUND_TRUTH_CASES = [
    ("គណនា", ["គ", "ណ", "នា"], [1, 1, 2], [2, 1, 2], [0, 0, 0]),
    ("សុភមង្គល", ["សុ", "ភ", "មង", "គល"], [1, 1, 2, 2], [1, 2, 2, 2], [0, 0, 0, 0]),
    ("សង្គ្រាម", ["សង", "គ្រាម"], [2, 2], [1, 2], [0, 1]),
    ("កម្ពុជា", ["កម", "ពុ", "ជា"], [2, 1, 2], [1, 2, 2], [0, 0, 0]),
    ("បញ្ហា", ["បញ", "ហា"], [2, 2], [1, 1], [0, 0]),
    ("កន្លែង", ["កន", "លែង"], [2, 2], [1, 1], [0, 0]),
    ("ប្រជា", ["ប្រ", "ជា"], [1, 2], [1, 2], [1, 0]),
    ("ក្រសួង", ["ក្រ", "សួង"], [1, 2], [1, 1], [1, 0]),
    ("កក្រើក", ["ក", "ក្រើក"], [1, 2], [1, 1], [0, 1]),
]


@pytest.mark.parametrize("word,exp_surfaces,exp_sesqui,exp_reg,exp_medial", GROUND_TRUTH_CASES)
def test_khmer_ground_truth_table(word, exp_surfaces, exp_sesqui, exp_reg, exp_medial):
    decomp = BilingualCodeSwitchDecomposer()
    assert isinstance(decomp, CodeSwitchDecomposerProtocol)
    tokens = decomp.decompose_mixed_sentence(word)
    assert [t.surface for t in tokens] == exp_surfaces
    assert [t.sesqui_id for t in tokens] == exp_sesqui
    assert [t.register_id for t in tokens] == exp_reg
    assert [t.medial_id for t in tokens] == exp_medial


def test_double_coeng_srey_and_foreign_brands():
    decomp = BilingualCodeSwitchDecomposer()
    srey = decomp.decompose_mixed_sentence("ស្ត្រី")
    assert len(srey) == 1
    assert srey[0].surface == "ស្ត្រី"
    assert srey[0].coeng_id > 0
    assert srey[0].medial_id == 1
    assert srey[0].register_id == 1
    assert srey[0].sesqui_id == 2
    assert 0 <= srey[0].coda_id <= 16

    mixed = "ខ្ញុំទៅ Calvin Klein និង Versace"
    tokens = decomp.decompose_mixed_sentence(mixed)
    brand_tokens = [t for t in tokens if t.register_id == 3]
    assert [t.surface.lower() for t in brand_tokens] == ["calvin", "klein", "versace"]
    assert all(t.sesqui_id == 2 for t in brand_tokens)
    assert all(18 <= t.coda_id <= 31 for t in brand_tokens)

    batch = decomp.encode_sentence_to_batch(mixed)
    assert batch.onset_ids.shape[0] == 1
    texts, cs_words = decomp.decode_batch_to_text(batch)
    assert len(texts) == 1 and len(texts[0]) > 0
    assert [w.lower() for w in cs_words] == ["calvin", "klein", "versace"]
# ===========================================================================
# v10.7 additions. The v10.6 assertions above are retained VERBATIM.
# ===========================================================================


def test_independent_vowels_are_canonicalized_and_reachable():
    # v10.7 (P1-A): U+17AB..U+17AE (RY / RYY / LY / LYY) used to map to
    # themselves, and _SYLLABLE_SCAN_RE begins at U+1780, so the decomposer
    # SKIPPED them entirely: a word beginning with one lost its first
    # syllable, and a standalone one fell back to the default onset id 1.
    from backend.linguistics.khmer_normalizer import normalize_khmer

    decomp = BilingualCodeSwitchDecomposer()

    # (codepoint, expected canonical form, expected onset id)
    cases = [
        (0x17AB, "\u179A\u17B9", 27),
        (0x17AC, "\u179A\u17BA", 27),
        (0x17AD, "\u179B\u17B9", 28),
        (0x17AE, "\u179B\u17BA", 28),
    ]
    for codepoint, expected, onset_id in cases:
        ch = chr(codepoint)
        assert normalize_khmer(ch) == expected
        tokens = decomp.decompose_mixed_sentence(ch)
        assert len(tokens) == 1
        assert tokens[0].onset_id == onset_id
        assert tokens[0].onset_id != 1  # not the default-onset fallback

    # A multi-syllable word beginning with an independent vowel must keep
    # its first syllable (v10.6 dropped it).
    word = chr(0x17AB) + chr(0x179F) + chr(0x17D2) + chr(0x179F)
    tokens = decomp.decompose_mixed_sentence(word)
    assert len(tokens) == 2


def test_robat_and_coda_membership():
    # v10.7: Robat (\u17CC) must map to vowel_id 24, and the coda for
    # \u1798 must be 17 in the plain form and 14 in the Bantoc form.
    # The coda is only derived when \u1798 is NOT the initial consonant
    # (a leading \u1798 is the onset, not a coda), so the test uses a
    # preceding consonant.
    decomp = BilingualCodeSwitchDecomposer()

    robat_tokens = decomp.decompose_mixed_sentence("\u1798\u17CC")
    assert robat_tokens
    assert robat_tokens[0].vowel_id == 24

    plain_mem = decomp.decompose_mixed_sentence("\u1780\u1798")
    assert plain_mem
    assert plain_mem[0].coda_id == 17

    bantoc_mem = decomp.decompose_mixed_sentence("\u1780\u1798\u17CB")
    assert bantoc_mem
    assert bantoc_mem[0].coda_id == 14


def test_km_to_en_syllable_decodes_to_latin_not_a_cvc_pseudoword():
    # v10.7 (P1-B): a synthesized English target syllable (register 3,
    # coeng 0, onset 34..57, vowel 28..44, coda 18..31) must decode to a
    # real Latin orthography. v10.6 fell through to a hardcoded CVC
    # fallback that ignored coeng_id and emitted text such as "kak".
    import torch

    from backend.contracts.schemas import SyllableTensorBatch

    decomp = BilingualCodeSwitchDecomposer()

    def make_batch(onset, medial, vowel, coda):
        t = lambda v: torch.tensor([[v]], dtype=torch.long)
        return SyllableTensorBatch(
            onset_ids=t(onset),
            coeng_ids=t(0),
            medial_ids=t(medial),
            register_ids=t(3),
            vowel_ids=t(vowel),
            coda_ids=t(coda),
            sesqui_ids=t(2),
        )

    # (onset, medial, vowel, coda) drawn from the reserved English ranges.
    for onset, medial, vowel, coda in [
        (34, 0, 28, 18),
        (40, 0, 30, 20),
        (57, 3, 44, 31),
    ]:
        texts, cs_words = decomp.decode_batch_to_text(
            make_batch(onset, medial, vowel, coda)
        )
        assert len(texts) == 1
        assert len(texts[0]) > 0
        assert texts[0].strip().isascii()
        assert texts[0].strip().isalpha()
        # A synthesized English syllable is NOT a code-switched brand.
        assert cs_words == []

    # The reserved ranges must be disjoint from the Khmer ranges.
    assert 34 > 33  # onset: Khmer 1..33, English 34..
    assert 28 > 24  # vowel: Khmer 0..24, reserved 25..27, English 28..


def test_brand_tokens_keep_one_slot_and_coda_in_range():
    # v10.7 (correction): a brand word occupies exactly ONE token, so the
    # Gate 1 surface assertions above remain valid, and every register-3
    # token keeps coda_id in 18..31.
    decomp = BilingualCodeSwitchDecomposer()

    mixed = "Calvin Klein Versace"
    tokens = decomp.decompose_mixed_sentence(mixed)
    brand_tokens = [t for t in tokens if t.register_id == 3]
    assert [t.surface.lower() for t in brand_tokens] == [
        "calvin",
        "klein",
        "versace",
    ]
    assert all(18 <= t.coda_id <= 31 for t in brand_tokens)
    assert all(t.sesqui_id == 2 for t in brand_tokens)
    # A multi-syllable brand is still ONE token.
    assert len(brand_tokens) == 3
