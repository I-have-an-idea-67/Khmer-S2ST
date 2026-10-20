export const SAMPLE_RATE_IN = 16000;
export const SAMPLE_RATE_OUT = 24000;
export const SAMPLE_RATE_MNE = 48000;
export const N_FFT_16K = 1024;
export const HOP_LENGTH_16K = 160;
export const N_FFT_24K = 2048;
export const HOP_LENGTH_24K = 240;
export const N_FFT_48K = 4096;
export const HOP_LENGTH_48K = 480;
export const HOP_LENGTH = 240;
export const N_MELS = 100;
export const ERB_BANDS = 64;
export const ERB_MIN_FREQ_HZ = 120.0;
export const ERB_MAX_FREQ_HZ = 8000.0;
export const SEMANTIC_DIM = 512;
export const EMOTION_NUM_TOKENS = 32;
export const MIC_EQ_TAPS_16K = 64;
export const MIC_EQ_TAPS_24K = 96;
export const VELVET_RIR_TAPS_24K = 6000;
export const MINOR_SYLLABLE_MAX_FRAMES = 6.0;
export const MAX_ELASTIC_PAUSE_BORROW_MS = 150.0;
export const DOMINANCE_MIN_DB_RATIO = -12.0;
export const LOOKAHEAD_BUFFER_SEC = 2.0;
export const MAX_CLAUSE_SLICE_SEC = 1.6;
export const MIN_CLAUSE_SLICE_SEC = 1.2;
export const SILENCE_RMS_THRESHOLD = 0.008;
export const UTTERANCE_END_SILENCE_MS = 250.0;
export const REPLAY_CACHE_MAX_ITEMS = 32;
export const REPLAY_BUCKET_MS = 250.0;

export type LanguageCode = "en" | "km" | "th" | "vi";

export enum VocalMode {
  AMBIENT_OR_CLAP = 0,
  NON_VERBAL_HUMAN = 1,
  SINGING_OR_MUSIC = 2,
  SPOKEN_DIALOGUE = 3,
}

export interface DownstreamEventPacket {
  event_type:
    | "audio_chunk"
    | "passthrough_vocal"
    | "singing_lyrics"
    | "caption"
    | "interrupted";
  utterance_id: number;
  utterance_start_ms: number;
  vocal_modes: Array<0 | 1 | 2 | 3>;
  speaker_ids: number[];
  source_lang: LanguageCode;
  target_lang: LanguageCode;
  dubbed_audio_b64?: string | null;
  erb_suppression_mask_b64?: string | null;
  erb_frame_count: number;
  per_speaker_source_texts: string[];
  per_speaker_target_texts: string[];
  codeswitch_words: string[];
  playback_rate: number;
  actual_duration_sec: number;
  drift_ms: number;
  interrupt_epoch?: number;
}

export interface UpstreamContentMessage {
  action: "interrupt" | "seek" | "config";
  video_time_ms: number;
  playback_rate?: number;
  payload?: Record<string, unknown>;
}

// Ambient WebAudio Worklet types so `tsc --noEmit` passes under strict DOM lib
declare global {
  abstract class AudioWorkletProcessor {
    readonly port: MessagePort;
    constructor(options?: unknown);
    abstract process(
      inputs: Float32Array[][],
      outputs: Float32Array[][],
      parameters: Record<string, Float32Array>
    ): boolean;
  }
  function registerProcessor(
    name: string,
    processorCtor: new (options?: unknown) => AudioWorkletProcessor
  ): void;
}
