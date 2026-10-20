import {

  DownstreamEventPacket,
  ERB_BANDS,
  ERB_MAX_FREQ_HZ,
  ERB_MIN_FREQ_HZ,
  LOOKAHEAD_BUFFER_SEC,
} from "../contracts/types";

// v10.7 AUD-107-05: the FFT twiddle and bit-reversal tables are computed
// ONCE at module load, not inside every real-time audio callback.
//
// v10.7 allocated two Float64Array tables and called Math.log2() inside the
// bit-reversal inner loop on EVERY 4096-sample block, which is 98,304
// Math.log2 calls per block. That is pure V8 heap churn inside a real-time
// WebAudio callback, where a GC pause is an audible dropout.
const N_FFT_MNE = 4096;
const HALF_FFT_MNE = N_FFT_MNE / 2;
const LOG2_N_MNE = 12; // Math.log2(4096)

const COS_TABLE_MNE = new Float64Array(HALF_FFT_MNE);
const SIN_TABLE_MNE = new Float64Array(HALF_FFT_MNE);
const REV_TABLE_MNE = new Uint16Array(N_FFT_MNE);

// The INVERSE transform uses the conjugate (positive-exponent) twiddles, so
// it needs its own pair of tables. They are the exact negation of the
// forward tables, but are kept separate so neither transform has to negate
// on the fly inside the real-time callback.
const COS_TABLE_MNE_INV = new Float64Array(HALF_FFT_MNE);
const SIN_TABLE_MNE_INV = new Float64Array(HALF_FFT_MNE);

for (let k = 0; k < HALF_FFT_MNE; k++) {
  COS_TABLE_MNE[k] = Math.cos((-2 * Math.PI * k) / N_FFT_MNE);
  SIN_TABLE_MNE[k] = Math.sin((-2 * Math.PI * k) / N_FFT_MNE);
  COS_TABLE_MNE_INV[k] = Math.cos((2 * Math.PI * k) / N_FFT_MNE);
  SIN_TABLE_MNE_INV[k] = Math.sin((2 * Math.PI * k) / N_FFT_MNE);
}
for (let i = 0; i < N_FFT_MNE; i++) {
  let r = 0;
  for (let b = 0; b < LOG2_N_MNE; b++) {
    r |= ((i >> b) & 1) << (LOG2_N_MNE - 1 - b);
  }
  REV_TABLE_MNE[i] = r;
}

export function hzToErbRate(hz: number): number {
  return 21.4 * Math.log10(1.0 + 0.00437 * hz);
}

const MIN_ERB = hzToErbRate(ERB_MIN_FREQ_HZ);
const MAX_ERB = hzToErbRate(ERB_MAX_FREQ_HZ);

export interface ScheduledErbSegment {
  utteranceId: number;
  startTime: number;
  endTime: number;
  frameCount: number;
  effectiveFrameRate: number;
  frames: Float32Array;
}

export class StereoMneMixer {
  private ctx: AudioContext;
  private sessionStartCtxTime: number;
  private nextPlayTime = 0;
  private erbQueue: ScheduledErbSegment[] = [];
  private activeSources: Set<AudioBufferSourceNode> = new Set();
  private clientEpoch = 0;
  private minServerEpoch = 0;

  // v10.7 P1-E: live spectral-analysis state for the studio M&E suppressor.
  private _midSpecReal: Float32Array | null = null;
  private _midSpecImag: Float32Array | null = null;
  private _sideSpecReal: Float32Array | null = null;
  private _sideSpecImag: Float32Array | null = null;
  private _mediaSourceNode: MediaElementAudioSourceNode | null = null;
  private _mneDelayNode: DelayNode | null = null;
  private _mneProcessorNode: ScriptProcessorNode | null = null;

  constructor(ctx: AudioContext, sessionStartCtxTime?: number) {
    if (ctx.sampleRate !== 48000) {
      console.warn(
        `[StereoMneMixer] AudioContext sampleRate is ${ctx.sampleRate}Hz, expected 48000Hz. ` +
        `Always instantiate with new AudioContext({ sampleRate: 48000 }) to avoid decimation pitch drift!`
      );
    }
    this.ctx = ctx;
    this.sessionStartCtxTime = sessionStartCtxTime ?? ctx.currentTime;
    this.nextPlayTime = this.ctx.currentTime;
  }

  public startSession(videoTimeMs = 0, playbackRate = 1.0): void {
    this.reanchorTimeline(videoTimeMs, playbackRate);
  }

  /**
   * Rewind/Seek Resilience Law:
   * Flushes active buffers, increments clientEpoch, resets nextPlayTime so rewound audio is
   * never locked behind a future timestamp, and re-anchors sessionStartCtxTime.
   */
  public reanchorTimeline(videoTimeMs: number, playbackRate = 1.0): void {
    this.flushPlayback();
    const safeRate = Math.max(0.5, Math.min(2.5, playbackRate || 1.0));
    this.sessionStartCtxTime = this.ctx.currentTime - videoTimeMs / (1000.0 * safeRate);
  }

  public applyMidSideSpectralErbMask(
    midSpecReal: Float32Array,
    midSpecImag: Float32Array,
    _sideSpecReal: Float32Array,
    _sideSpecImag: Float32Array,
    erbMask64: Float32Array,
    sampleRate = 48000,
    nFft = 4096
  ): void {
    const numBins = midSpecReal.length;
    for (let k = 0; k < numBins; k++) {
      const freqHz = k * (sampleRate / nFft);
      if (freqHz < ERB_MIN_FREQ_HZ || freqHz > ERB_MAX_FREQ_HZ) {
        continue;
      }
      const erbVal = hzToErbRate(freqHz);
      const norm = (erbVal - MIN_ERB) / (MAX_ERB - MIN_ERB);
      const erbIdx = Math.max(
        0,
        Math.min(ERB_BANDS - 1, Math.floor(ERB_BANDS * norm))
      );
      const gain = Math.max(0.0, Math.min(1.0, erbMask64[erbIdx]));
      midSpecReal[k] *= gain;
      midSpecImag[k] *= gain;
    }
  }

  public getActiveErbMaskForTime(ctxTime: number): Float32Array | null {
    this.erbQueue = this.erbQueue.filter((seg) => seg.endTime >= ctxTime - 0.05);
    for (const seg of this.erbQueue) {
      if (ctxTime >= seg.startTime && ctxTime <= seg.endTime && seg.frameCount > 0) {
        const fIdx = Math.max(
          0,
          Math.min(
            seg.frameCount - 1,
            Math.floor((ctxTime - seg.startTime) * seg.effectiveFrameRate)
          )
        );
        return seg.frames.subarray(fIdx * ERB_BANDS, (fIdx + 1) * ERB_BANDS);
      }
    }
    return null;
  }

  /**
   * v10.7 P1-E: apply the active 64-band ERB mask to a Mid-channel frame.
   *
   * v10.6 defined applyMidSideSpectralErbMask and getActiveErbMaskForTime but
   * never called them, so the ERB mask the server computes and ships was
   * queued and then ignored: the studio M&E suppressor never suppressed
   * anything.
   *
   * The target time MUST be the segment's own start time (the lookahead
   * target), NOT this.ctx.currentTime. Newly queued segments start 2.0 s in
   * the FUTURE, so querying the current time would always return null and
   * the mask would never be applied.
   */
  public processMidSideFrame(
    targetCtxTime: number,
    midSpecReal: Float32Array,
    midSpecImag: Float32Array,
    sideSpecReal: Float32Array,
    sideSpecImag: Float32Array,
    sampleRate = 48000,
    nFft = 4096
  ): boolean {
    const mask = this.getActiveErbMaskForTime(targetCtxTime);
    if (!mask) {
      return false;
    }
    this.applyMidSideSpectralErbMask(
      midSpecReal,
      midSpecImag,
      sideSpecReal,
      sideSpecImag,
      mask,
      sampleRate,
      nFft
    );
    return true;
  }

  /**
   * v10.7 P1-E: route the media element through a live 4096-bin Mid/Side
   * spectral suppressor so the ERB mask the server ships is actually applied
   * to the original M&E (music & effects) bed.
   *
   * The graph is: media element -> MediaElementSource -> ScriptProcessor
   * (4096-bin STFT, Mid/Side split, ERB mask, ISTFT) -> destination. The
   * processor asks the mixer for the mask that is active at the CURRENT
   * audio time, which is the correct clock for a live processor (the
   * lookahead clock is used when a segment is queued).
   */
  /**
   * v10.7 P1-E: in-place radix-2 FFT of a real frame, used by the Mid/Side
   * spectral suppressor. Writes the first N_FFT/2+1 bins into re/im.
   */
  private _computeSpectrum(
    frame: Float32Array,
    nFrames: number,
    re: Float32Array,
    im: Float32Array,
    nFft: number
  ): void {
    const size = nFft;
    const half = size / 2;
    // v10.7 AUD-107-05: use the module-level tables.
    const cos = COS_TABLE_MNE;
    const sin = SIN_TABLE_MNE;
    // v10.7: UNITY (rectangular) windowing, not Hann.
    //
    // ScriptProcessorNode(4096, 2, 2) delivers NON-OVERLAPPING 4096-sample
    // blocks, and N_FFT equals the block size, so there is no overlap-add and
    // no inverse-window compensation anywhere in this path. A Hann window
    // tapers the first and last samples of EVERY block to zero, so the
    // reconstruction came back with a hard zero at both block edges and the
    // M&E bed developed periodic dropouts at the block rate. With a unity
    // window the forward/inverse pair is the exact identity on the block.
    const start = Math.max(0, nFrames - size);
    const x = new Float64Array(size);
    for (let i = 0; i < size; i++) {
      const src = start + i;
      x[i] = src < nFrames ? frame[src] : 0;
    }
    // Iterative radix-2 Cooley-Tukey.
    // v10.7 AUD-107-05: the bit-reversal permutation is precomputed at
    // module load, so no Math.log2 call happens per block.
    const rev = (v: number): number => REV_TABLE_MNE[v];
    for (let i = 0; i < size; i++) {
      const j = rev(i);
      if (j > i) {
        const tmp = x[i];
        x[i] = x[j];
        x[j] = tmp;
      }
    }
    // v10.7: the bit-reversed input MUST be copied into the butterfly
    // working arrays. Without this copy the butterflies operate on zeros and
    // every bin comes out 0.0.
    const bre = new Float64Array(size);
    const bim = new Float64Array(size);
    for (let i = 0; i < size; i++) {
      bre[i] = x[i];
    }
    for (let len = 2; len <= size; len *= 2) {
      const step = size / len;
      for (let i = 0; i < size; i += len) {
        for (let j = 0; j < len / 2; j++) {
          const wr = cos[(j * step) | 0];
          const wi = sin[(j * step) | 0];
          const a = i + j;
          const b = i + j + len / 2;
          const tr = bre[b] * wr - bim[b] * wi;
          const ti = bre[b] * wi + bim[b] * wr;
          bre[b] = bre[a] - tr;
          bim[b] = bim[a] - ti;
          bre[a] += tr;
          bim[a] += ti;
        }
      }
    }
    const bins = size / 2 + 1;
    for (let k = 0; k < bins; k++) {
      re[k] = bre[k];
      im[k] = bim[k];
    }
  }

  /**
   * v10.7 P1-E: the inverse of _computeSpectrum. Reconstructs the time
   * domain Mid signal from the (possibly ERB-masked) spectrum, using the
   * conjugate symmetry of a real input: X[N-k] = conj(X[k]).
   *
   * Without this step the suppressor was a NO-OP: the masked spectrum was
   * computed and then thrown away, and the unmasked time-domain mid[] was
   * written to the output.
   */
  private _inverseSpectrum(
    re: Float32Array,
    im: Float32Array,
    nFft: number
  ): Float32Array {
    const size = nFft;
    const half = size / 2 + 1;
    // Rebuild the full complex spectrum with conjugate symmetry.
    const fre = new Float64Array(size);
    const fim = new Float64Array(size);
    for (let k = 0; k < half; k++) {
      fre[k] = re[k];
      fim[k] = im[k];
    }
    for (let k = 1; k < size - half + 1; k++) {
      fre[size - k] = re[k];
      fim[size - k] = -im[k];
    }
    // Inverse radix-2 Cooley-Tukey (conjugate the exponent, divide by N).
    // v10.7 AUD-107-05: use the module-level tables.
    const cos = COS_TABLE_MNE_INV;
    const sin = SIN_TABLE_MNE_INV;
    // v10.7 AUD-107-05: the bit-reversal permutation is precomputed at
    // module load, so no Math.log2 call happens per block.
    const rev = (v: number): number => REV_TABLE_MNE[v];
    for (let i = 0; i < size; i++) {
      const j = rev(i);
      if (j > i) {
        const t = fre[i];
        fre[i] = fre[j];
        fre[j] = t;
        const u = fim[i];
        fim[i] = fim[j];
        fim[j] = u;
      }
    }
    // v10.7: the bit-reversed input MUST be copied into the butterfly
    // working arrays, exactly as in _computeSpectrum. Without this copy the
    // inverse transform operates on zeros and always returns an all-zero
    // signal, which would silence the whole M&E bed.
    const bre = new Float64Array(size);
    const bim = new Float64Array(size);
    for (let i = 0; i < size; i++) {
      bre[i] = fre[i];
      bim[i] = fim[i];
    }
    for (let len = 2; len <= size; len *= 2) {
      const step = size / len;
      for (let i = 0; i < size; i += len) {
        for (let j = 0; j < len / 2; j++) {
          const wr = cos[(j * step) | 0];
          const wi = sin[(j * step) | 0];
          const a = i + j;
          const b = i + j + len / 2;
          const tr = bre[b] * wr - bim[b] * wi;
          const ti = bre[b] * wi + bim[b] * wr;
          bre[b] = bre[a] - tr;
          bim[b] = bim[a] - ti;
          bre[a] += tr;
          bim[a] += ti;
        }
      }
    }
    const out = new Float32Array(size);
    for (let i = 0; i < size; i++) {
      out[i] = bre[i] / size;
    }
    return out;
  }

  /**
   * v10.7 P1-E: the shared MediaElementAudioSourceNode for this video element.
   *
   * WebAudio allows exactly ONE MediaElementAudioSourceNode per media
   * element; a second call to createMediaElementSource throws an
   * InvalidStateError. The DuplexWsClient therefore TAPS this node for the
   * upstream capture path instead of creating its own.
   */
  public getMediaSourceNode(): MediaElementAudioSourceNode | null {
    return this._mediaSourceNode;
  }

  public attachMediaElement(videoEl: HTMLVideoElement): MediaElementAudioSourceNode {
    if (this._mediaSourceNode) {
      return this._mediaSourceNode;
    }
    const N_FFT = 4096;
    const HALF = N_FFT / 2 + 1;
    this._midSpecReal = new Float32Array(HALF);
    this._midSpecImag = new Float32Array(HALF);
    this._sideSpecReal = new Float32Array(HALF);
    this._sideSpecImag = new Float32Array(HALF);

    const source = this.ctx.createMediaElementSource(videoEl);
    this._mediaSourceNode = source;

    const analyser = this.ctx.createAnalyser();
    analyser.fftSize = N_FFT;

    const processor = this.ctx.createScriptProcessor(4096, 2, 2);
    this._mneProcessorNode = processor;
    processor.onaudioprocess = (ev) => {
      const input = ev.inputBuffer;
      const output = ev.outputBuffer;
      const nCh = Math.min(input.numberOfChannels, 2);
      const nFrames = input.length;

      // Mid/Side split of the stereo bed.
      const mid = new Float32Array(nFrames);
      const side = new Float32Array(nFrames);
      for (let i = 0; i < nFrames; i++) {
        const l = input.getChannelData(0)[i];
        const r = nCh > 1 ? input.getChannelData(1)[i] : l;
        mid[i] = 0.5 * (l + r);
        side[i] = 0.5 * (l - r);
      }

      // Forward FFT of the Mid channel, then apply the active ERB mask to
      // the Mid spectrum only. The Side channel is passed through untouched.
      // The FFT is computed in-place on a unity-windowed frame.
      const half = N_FFT / 2 + 1;
      const re = new Float32Array(half);
      const im = new Float32Array(half);
      this._midSpecReal = re;
      this._midSpecImag = im;
      this._sideSpecReal = new Float32Array(half);
      this._sideSpecImag = new Float32Array(half);
      this._computeSpectrum(mid, nFrames, re, im, N_FFT);

      // The live mask clock is the current context time.
      const maskApplied = this.processMidSideFrame(
        this.ctx.currentTime,
        re,
        im,
        this._sideSpecReal as Float32Array,
        this._sideSpecImag as Float32Array,
        this.ctx.sampleRate,
        N_FFT
      );

      // Rebuild the output from the MASKED Mid plus the untouched Side.
      //
      // v10.7 P1-E: the masked spectrum must be transformed BACK to the time
      // domain. v10.6 wrote the unmasked mid[] to the output, so the ERB
      // suppressor was a no-op: the mask was computed and then discarded.
      let midOut: Float32Array;
      if (maskApplied) {
        const reconstructed = this._inverseSpectrum(re, im, N_FFT);
        // The frame is the tail of the current block; the reconstruction is
        // aligned to the same offset as the analysis window.
        const offset = Math.max(0, nFrames - N_FFT);
        midOut = new Float32Array(nFrames);
        for (let i = 0; i < nFrames; i++) {
          const src = offset + i;
          midOut[i] = src < reconstructed.length ? reconstructed[src] : 0;
        }
      } else {
        midOut = mid;
      }
      const outL = output.getChannelData(0);
      const outR = output.numberOfChannels > 1 ? output.getChannelData(1) : outL;
      const srcL = input.getChannelData(0);
      const srcR = nCh > 1 ? input.getChannelData(1) : srcL;
      for (let i = 0; i < nFrames; i++) {
        const m = maskApplied ? midOut[i] : 0.5 * (srcL[i] + srcR[i]);
        const s = 0.5 * (srcL[i] - srcR[i]);
        outL[i] = m + s;
        if (output.numberOfChannels > 1) {
          outR[i] = m - s;
        }
      }
    };

    // v10.7 AUD-107-03: the M&E path carries the 2.00s LOOKAHEAD DELAY that
    // the Part 2.1 topology specifies:
    //
    //   [2.00s Delay Line (Video + 48kHz Stereo)] -> [48kHz Mid/Side ERB Suppressor]
    //
    // Without it the original unmasked vocals reached the speakers 2.0s
    // BEFORE the ERB mask and the dubbed audio took effect at
    // lookaheadTargetTime = t_start + 2.00s, so the suppression was always
    // late by the whole lookahead window.
    const delayNode = this.ctx.createDelay(LOOKAHEAD_BUFFER_SEC + 1.0);
    delayNode.delayTime.value = LOOKAHEAD_BUFFER_SEC;
    this._mneDelayNode = delayNode;

    source.connect(delayNode);
    delayNode.connect(processor);
    // v10.7: the processor MUST reach the destination, otherwise the M&E
    // suppressor is connected to nothing and the output is silent.
    processor.connect(this.ctx.destination);
    // v10.7: the source node is returned so the DuplexWsClient can TAP
    // the SAME node for the upstream capture path. Calling
    // ctx.createMediaElementSource(videoEl) a second time on the same
    // element throws an InvalidStateError in WebAudio.
    return source;
  }

  public async handleDownstreamEvent(pkt: DownstreamEventPacket): Promise<void> {
    if (pkt.event_type === "interrupted") {
      this.flushPlayback();
      return;
    }

    if (pkt.interrupt_epoch !== undefined) {
      if (pkt.interrupt_epoch < this.minServerEpoch) return;
      this.minServerEpoch = pkt.interrupt_epoch;
    }

    const epochAtEntry = this.clientEpoch;
    const safeRate = Math.max(0.5, Math.min(2.5, pkt.playback_rate || 1.0));
    const lookaheadTargetTime =
      this.sessionStartCtxTime +
      pkt.utterance_start_ms / (1000.0 * safeRate) +
      LOOKAHEAD_BUFFER_SEC;

    const effectiveFrameRate = 100.0 * safeRate;
    if (pkt.erb_suppression_mask_b64 && pkt.erb_frame_count > 0) {
      const bytes = Uint8Array.from(atob(pkt.erb_suppression_mask_b64), (c) =>
        c.charCodeAt(0)
      );
      const frames = new Float32Array(
        bytes.buffer,
        bytes.byteOffset,
        bytes.byteLength / 4
      );
      this.erbQueue.push({
        utteranceId: pkt.utterance_id,
        startTime: lookaheadTargetTime,
        endTime: lookaheadTargetTime + pkt.erb_frame_count / effectiveFrameRate,
        frameCount: pkt.erb_frame_count,
        effectiveFrameRate,
        frames,
      });
      // v10.7 AUD-107-05: the redundant spectral pass here is REMOVED.
      //
      // v10.7 called processMidSideFrame(lookaheadTargetTime, ...) on
      // this._midSpecReal, which is either a stale spectrum left over from an
      // already-completed audio callback or a throwaway Float32Array. The mask
      // was therefore written into a buffer that the next onaudioprocess call
      // overwrites, so the pass had NO effect on the output while still
      // burning a full FFT on every WebSocket JSON packet.
      //
      // The mask is applied in the RIGHT place: the audio callback computes the
      // spectrum for the block it is about to emit and applies the mask that is
      // active at that moment. With the 2.00s lookahead DelayNode
      // (AUD-107-03) the delayed block and the scheduled ERB segment share the
      // same timeline, so the callback picks up the correct mask.
    }

    if (pkt.event_type === "audio_chunk" && pkt.dubbed_audio_b64) {
      const wavBytes = Uint8Array.from(atob(pkt.dubbed_audio_b64), (c) =>
        c.charCodeAt(0)
      );
      const audioBuf = await this.ctx.decodeAudioData(
        wavBytes.buffer.slice(0) as ArrayBuffer
      );

      // Discard in-flight decoded audio if a seek/rewind occurred during await
      if (this.clientEpoch !== epochAtEntry) {
        return;
      }

      const source = this.ctx.createBufferSource();
      source.buffer = audioBuf;
      source.playbackRate.value = 1.0;
      source.connect(this.ctx.destination);

      const startAt = Math.max(
        this.ctx.currentTime + 0.005,
        lookaheadTargetTime,
        this.nextPlayTime
      );
      source.start(startAt);
      this.nextPlayTime = startAt + audioBuf.duration;
      this.activeSources.add(source);
      source.onended = () => this.activeSources.delete(source);
    }
  }

  public flushPlayback(): void {
    this.clientEpoch += 1;
    for (const src of this.activeSources) {
      try {
        src.stop();
      } catch {
        // Ignore already stopped sources
      }
    }
    this.activeSources.clear();
    this.erbQueue = [];
    this.nextPlayTime = this.ctx.currentTime;
  }
}
