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
