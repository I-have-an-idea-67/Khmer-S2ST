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
