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
