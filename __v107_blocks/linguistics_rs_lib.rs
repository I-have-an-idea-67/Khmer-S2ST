
use regex::Regex;
use unicode_normalization::UnicodeNormalization;

pub fn normalize_khmer_pure(text: &str) -> String {
    let mut out: String = text
        .nfc()
        .filter(|c| !matches!(*c, '\u{200C}' | '\u{200D}' | '\u{FEFF}'))
        .collect();
    let re_coeng = Regex::new(r"([\u{17B6}-\u{17CA}]+)((?:\u{17D2}[\u{1780}-\u{17A2}])+)").unwrap();
    let re_shift = Regex::new(r"([\u{17B6}-\u{17C8}]+)([\u{17C9}\u{17CA}])").unwrap();
    // v10.7 P1-A: independent-vowel canonicalization, identical to the Python
    // path in backend/linguistics/khmer_normalizer.py. U+17AB..U+17AE map to a
    // consonant + dependent-vowel pair and U+17A8 is corrected to the open
    // (vowel-final) spelling, so the Rust and Python normalizers agree.
    const INDEPENDENT_VOWEL_MAP: [(char, &str); 17] = [
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
    // v10.7: the independent-vowel canonicalization MUST run BEFORE the coeng
    // and shifter reordering, because that is the execution order of the
    // Python normalize_khmer in backend/linguistics/khmer_normalizer.py. The
    // two normalizers must agree byte-for-byte on compound words that contain
    // an independent vowel followed by a subscript coeng.
    for (k, v) in INDEPENDENT_VOWEL_MAP.iter() {
        if out.contains(*k) {
            out = out.replace(*k, v);
        }
    }
    for _ in 0..3 {
        out = re_coeng.replace_all(&out, "$2$1").into_owned();
        out = re_shift.replace_all(&out, "$2$1").into_owned();
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_normalize_preserves_zwsp_and_reorders() {
        let s = normalize_khmer_pure("ខ្ញុំ\u{200B}ទៅ");
        assert!(s.contains('\u{200B}'));
    }

    #[test]
    fn test_normalize_canonicalizes_independent_vowels() {
        // v10.7 P1-A: U+17AB..U+17AE must canonicalize to a consonant plus a
        // dependent vowel so the Python decomposer's scanner can reach them.
        assert_eq!(normalize_khmer_pure("\u{17AB}"), "\u{179A}\u{17B9}");
        assert_eq!(normalize_khmer_pure("\u{17AC}"), "\u{179A}\u{17BA}");
        assert_eq!(normalize_khmer_pure("\u{17AD}"), "\u{179B}\u{17B9}");
        assert_eq!(normalize_khmer_pure("\u{17AE}"), "\u{179B}\u{17BA}");
        // U+17A8 (QUK) uses the open (vowel-final) coda U+1780.
        assert_eq!(normalize_khmer_pure("\u{17A8}"), "\u{17A2}\u{17BB}\u{1780}");
        // U+17B3 (the 17th independent vowel) must be present too.
        assert_eq!(normalize_khmer_pure("\u{17B3}"), "\u{17A2}\u{17C5}");
    }
}
