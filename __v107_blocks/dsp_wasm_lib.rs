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
