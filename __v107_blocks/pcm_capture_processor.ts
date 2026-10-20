import "../contracts/types";
import { SAMPLE_RATE_IN } from "../contracts/types";

// Scoped strictly to the AudioWorkletGlobalScope module
declare const currentTime: number;
declare const sampleRate: number;

// 15-tap Blackman-windowed sinc filter for 48kHz -> 16kHz (fc = 7.2 kHz, normalized sum = 1.0000, 0.0 dB DC gain)
const FIR_15_TAPS = new Float32Array([
  -0.0030, -0.0091, -0.0101, 0.0110, 0.0630, 0.1366, 0.1989,
   0.2254,
   0.1989,  0.1366,  0.0630, 0.0110, -0.0101, -0.0091, -0.0030,
]);

export class PcmCaptureProcessor extends AudioWorkletProcessor {
  private readonly ring = new Float32Array(15);
  private ringIdx = 0;
  private phase = 0;
  // v10.7 AUD-107-10: fractional read position and previous mono sample for
  // the universal linear-interpolation resampler fallback.
  private fracPos = 0.0;
  private prevSample = 0.0;
  private readonly outBuf = new Float32Array(128);
  // v10.7 P1-E: the media clock must advance at the VIDEO's rate, not at the
  // AudioContext's 1.0x wall clock. At playbackRate 2.0 the video timeline
  // advances twice as fast as the audio context, so a 1.0x clock drifts and
  // the upstream packet timestamps no longer match videoEl.currentTime.
  private playbackRate = 1.0;
  private lastCtxTimeMs = 0.0;
  private mediaClockMs = 0.0;

  constructor(options?: unknown) {
    super(options);
    this.port.onmessage = (ev: MessageEvent<{
      baseVideoTimeMs?: number;
      playbackRate?: number;
    }>) => {
      if (ev.data && typeof ev.data.playbackRate === "number" && ev.data.playbackRate > 0) {
        this.playbackRate = ev.data.playbackRate;
      }
      if (ev.data && typeof ev.data.baseVideoTimeMs === "number") {
        this.mediaClockMs = ev.data.baseVideoTimeMs;
        this.lastCtxTimeMs = currentTime * 1000.0;
      }
    };
  }

  process(inputs: Float32Array[][], _outputs: Float32Array[][]): boolean {
    const input = inputs[0];
    if (!input || !input[0] || input[0].length === 0) return true;

    const frames = input[0].length;
    const left = input[0];
    const right = input.length > 1 && input[1] ? input[1] : input[0];
    let outCount = 0;

    // v10.7 AUD-107-10: the decimator MUST follow the REAL context rate.
    //
    // v10.7 hard-coded a 3:1 decimation, which is only correct when the
    // AudioContext runs at 48,000 Hz. On iOS Safari, some Android WebViews and
    // Bluetooth headsets the OS locks the context to 44,100 or 96,000 Hz, and a
    // 3:1 decimator then emitted 14,700 Hz instead of 16,000 Hz.
    //
    // The 48,000 Hz fast path keeps the 15-tap Blackman-sinc decimator. Every
    // other rate uses a stateful fractional linear-interpolation resampler, so
    // the packets are always 16,000 Hz on any device.
    const ctxRate =
      typeof sampleRate === "number" && sampleRate > 0 ? sampleRate : 48000;

    if (ctxRate === 48000) {
      for (let i = 0; i < frames; i++) {
        this.ring[this.ringIdx] = 0.5 * (left[i] + right[i]);
        this.ringIdx = (this.ringIdx + 1) % 15;

        if (this.phase === 0) {
          let sum = 0.0;
          for (let t = 0; t < 15; t++) {
            sum += this.ring[(this.ringIdx + t) % 15] * FIR_15_TAPS[t];
          }
          this.outBuf[outCount++] = sum;
        }
        this.phase = (this.phase + 1) % 3;
      }
    } else {
      // Universal fractional resampler fallback for the 44.1 kHz / non-48 kHz
      // mobile and Bluetooth hardware. The output rate is exactly 16,000 Hz
      // for every context rate at or above 16 kHz.
      const step = ctxRate / SAMPLE_RATE_IN;
      for (let i = 0; i < frames; i++) {
        const curSample = 0.5 * (left[i] + right[i]);
        while (this.fracPos < 1.0 && outCount < this.outBuf.length) {
          this.outBuf[outCount++] =
            this.prevSample + this.fracPos * (curSample - this.prevSample);
          this.fracPos += step;
        }
        this.fracPos -= 1.0;
        this.prevSample = curSample;
      }
    }

    const byteLen = outCount * 4;
    const packet = new ArrayBuffer(8 + byteLen);
    // v10.7 P1-E: advance the media clock by the wall-clock delta scaled by
    // the video playback rate, so the upstream timestamps track the VIDEO
    // timeline rather than the 1.0x AudioContext clock.
    const ctxNowMs = currentTime * 1000.0;
    const deltaMs = ctxNowMs - this.lastCtxTimeMs;
    this.lastCtxTimeMs = ctxNowMs;
    this.mediaClockMs += deltaMs * this.playbackRate;
    const mediaTimeMs = Math.max(0.0, this.mediaClockMs);
    new DataView(packet).setFloat64(0, mediaTimeMs, true);
    new Float32Array(packet, 8, outCount).set(this.outBuf.subarray(0, outCount));
    this.port.postMessage(packet, [packet]);
    return true;
  }
}

registerProcessor("pcm-capture-processor", PcmCaptureProcessor);
