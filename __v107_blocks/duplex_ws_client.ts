import { StereoMneMixer } from "../audio/stereo-mne-mixer";
import { DownstreamEventPacket, UpstreamContentMessage } from "../contracts/types";
import { CaptionOverlay } from "../ui/caption-overlay";

export class DuplexWsClient {
  private ws!: WebSocket;
  // v10.7 P1-E / A-1: the upstream audio path. v10.6 never instantiated
  // the PcmCaptureProcessor worklet and never sent a binary frame, so the
  // server's upstream_reader waited forever for bytes that never arrived:
  // the full-duplex loop could not function at all.
  private captureNode: AudioWorkletNode | null = null;

  constructor(
    private readonly sessionId: string,
    private readonly videoEl: HTMLVideoElement,
    private readonly mixer: StereoMneMixer,
    private readonly overlay: CaptionOverlay
  ) {
    this.videoEl.addEventListener("ratechange", () => {
      const timeMs = this.videoEl.currentTime * 1000.0;
      this.mixer.reanchorTimeline(timeMs, this.videoEl.playbackRate);
      this.send({
        action: "config",
        video_time_ms: timeMs,
        playback_rate: this.videoEl.playbackRate,
    });
      this.postMediaClock();
    });

    this.videoEl.addEventListener("seeking", () => {
      const timeMs = this.videoEl.currentTime * 1000.0;
      this.mixer.reanchorTimeline(timeMs, this.videoEl.playbackRate);
      this.send({
        action: "seek",
        video_time_ms: timeMs,
        playback_rate: this.videoEl.playbackRate,
    });
      this.postMediaClock();
    });

    this.videoEl.addEventListener("pause", () => {
      this.mixer.flushPlayback();
    });

    this.videoEl.addEventListener("playing", () => {
      this.mixer.reanchorTimeline(this.videoEl.currentTime * 1000.0, this.videoEl.playbackRate);
      this.postMediaClock();
    });
  }

  /**
   * v10.7 P1-E / A-1: instantiate the PCM capture worklet and forward its
   * binary frames to the WebSocket.
   *
   * The worklet emits an 8-byte little-endian float64 media timestamp
   * followed by the decimated 16 kHz mono samples, which is exactly the
   * frame the server's upstream_reader expects.
   */
  public async attachAudioCapture(
    ctx: AudioContext,
    source?: MediaElementAudioSourceNode
  ): Promise<void> {
    if (this.captureNode) {
      return;
    }
    await ctx.audioWorklet.addModule("worklets/pcm-capture-processor.js");
    const node = new AudioWorkletNode(ctx, "pcm-capture-processor");
    this.captureNode = node;
    node.port.onmessage = (ev: MessageEvent<ArrayBuffer>) => {
      if (this.ws && this.ws.readyState === WebSocket.OPEN) {
        this.ws.send(ev.data);
      }
    };
    // v10.7 P1-E: the worklet taps the SHARED media-element source node.
    // The mixer creates it and returns it; calling
    // ctx.createMediaElementSource(videoEl) a second time on the same
    // element throws an InvalidStateError in WebAudio, so the node is
    // passed in and only created here when the mixer has not made it yet.
    const tap =
      source ?? this.mixer.getMediaSourceNode() ?? ctx.createMediaElementSource(this.videoEl);
    tap.connect(node);
    // v10.7 AUD-107-06: the capture worklet is a CAPTURE-ONLY sink.
    //
    // v10.7 connected it straight to ctx.destination. PcmCaptureProcessor
    // leaves its outputs zeroed, so it is silent today, but a future WASM
    // worklet that passes audio through would sum the UNDELAYED dry audio
    // into the destination, bypassing the M&E suppressor entirely and
    // re-introducing the very bleed the ERB ducking exists to remove.
    //
    // A GainNode with gain 0.0 keeps the node pulled by the graph (so the
    // worklet keeps processing) while guaranteeing that nothing reaches the
    // speakers.
    const captureSink = ctx.createGain();
    captureSink.gain.value = 0.0;
    node.connect(captureSink);
    captureSink.connect(ctx.destination);
    this.postMediaClock();
  }

  /**
   * v10.7 P1-E: publish the current video media time and playback rate to
   * the capture worklet so its timestamps track the VIDEO timeline.
   */
  private postMediaClock(): void {
    if (!this.captureNode) {
      return;
    }
    this.captureNode.port.postMessage({
      baseVideoTimeMs: this.videoEl.currentTime * 1000.0,
      playbackRate: this.videoEl.playbackRate,
    });
  }

  public connect(): void {
    this.ws = new WebSocket(`ws://localhost:8000/ws/live-dub/${this.sessionId}`);
    this.ws.binaryType = "arraybuffer";
    this.mixer.startSession(this.videoEl.currentTime * 1000.0, this.videoEl.playbackRate);
    this.ws.onmessage = (ev: MessageEvent) => {
      const pkt = JSON.parse(ev.data as string) as DownstreamEventPacket;
      void this.mixer.handleDownstreamEvent(pkt);
      this.overlay.render(pkt);
    };
  }

  public send(msg: UpstreamContentMessage): void {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(msg));
    }
  }

  public interrupt(): void {
    const timeMs = this.videoEl.currentTime * 1000.0;
    this.mixer.reanchorTimeline(timeMs, this.videoEl.playbackRate);
    this.send({ action: "interrupt", video_time_ms: timeMs });
  }
}
