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
