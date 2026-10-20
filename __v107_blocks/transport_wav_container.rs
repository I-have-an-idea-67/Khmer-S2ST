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
