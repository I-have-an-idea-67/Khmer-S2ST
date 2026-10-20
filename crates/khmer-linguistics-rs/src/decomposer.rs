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
