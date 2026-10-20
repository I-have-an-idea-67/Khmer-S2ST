# KHMER-SOVEREIGN-S2ST: Master Architectural & Engineering Plans (v10.7)

Symmetric, Bidirectional (`EN <-> KM`), Full-Duplex Direct Speech-to-Speech Translation (S2ST) engine with FiLM-conditioned multi-speaker separation, 7-slot Khmer phonological decomposition, non-monotonic syntax reordering, and 48kHz Mid/Side Log-ERB studio M&E reconstruction (`241.637M` active parameters in `PROD_240M_CONFIG`; `3.838M` parameters in `CI_MICRO_CONFIG`).

> [!NOTE]
> **v10.7 is a code-level remediation revision.** A line-by-line inspection of the
> Part 4 reference implementations found eleven defects that the document-level
> v10.6 audit could not see, including two that prevented the live server from
> working at all: the rolling cross-clause KV cache was **dead code** (the runner
> never passed it to the console), and the frontend **never sent upstream audio**
> (the client instantiated no worklet and no binary frame), so the full-duplex
> loop could not function. v10.7 repairs all of them with **zero new trainable
> parameters** - the ledger is unchanged at `241,637,457` / `3,838,033`.
>
> * Code-level findings: [`docs/05_CODE_LEVEL_AUDIT_AND_REMEDIATION.md`](./docs/05_CODE_LEVEL_AUDIT_AND_REMEDIATION.md)
> * Document-level findings (v10.5 -> v10.6): [`docs/04_AUDIT_FINDINGS_AND_REMEDIATION.md`](./docs/04_AUDIT_FINDINGS_AND_REMEDIATION.md)
> * Complete work record (all phases, method, decisions, and known limitations): [`WORKLOG.md`](./WORKLOG.md)

---

## Architectural Plans & Specification Structure

The v10.7 master blueprint ([`KHMER-SOVEREIGN-S2ST-BLUEPRINT-v10.7.txt`](./KHMER-SOVEREIGN-S2ST-BLUEPRINT-v10.7.txt)) is organized across four structured engineering volumes in [`docs/`](./docs/):

1. **[`docs/01_ARCHITECTURE_AND_CONTRACTS.md`](./docs/01_ARCHITECTURE_AND_CONTRACTS.md)** (Parts 1-3)
   * **Audit Ledger & System Overview (Part 1):** The full v10.5 fix ledger with the v10.6 corrections appended, the hybrid Rust/WASM edge vs. PyTorch autograd core division, and the deterministic turnaround law (`<= 1.94s < 2.00s` lookahead).
   * **Topology & Immutable Contracts (Part 2):** Full-duplex lookahead loop topology, workspace file tree, v10.7.0 toolchain configurations (`Cargo.toml`, `pyproject.toml`, `package.json`, `tsconfig.json`), **Section 2.3.1 Version Manifest**, declarative contracts (`schemas.py` & `types.ts` with locked 100Hz STFT/Hop constants), `@runtime_checkable` protocols, and locked helper signatures.
   * **Subsystem Specifications & Parameter Ledger (Part 3):** 7-slot Khmer linguistics, 8-group phonological coda mapping (`CODA_GROUP_MAP` `[1..8]`), Bantoc codas `[9..16]`, G2P phonetic brand mapping `[18..31]` with vowel-final fallback, sonorant register inheritance, the 4 rewind/seek resilience laws, and the **corrected `241,637,457` / `3,838,033` parameter ledger** with exact per-module integers.
2. **[`docs/02_DEEP_DIVE_IMPLEMENTATIONS.md`](./docs/02_DEEP_DIVE_IMPLEMENTATIONS.md)** (Part 4: Sections 4.1-4.9)
   * **4.1 Specialist B (Linguistic Engine):** `khmer_normalizer.py` (preserving `\u200b`) and `syllable_decomposer.py` (`BilingualCodeSwitchDecomposer` with `CODA_GROUP_MAP` `[1..8]`, Bantoc `[9..16]`, and 63-way `coeng_id` brand hash disambiguation).
   * **4.2 Specialist A (Universal Core & Slaney Mel):** `universal_core.py` (`build_slaney_mel_fb` with full column rank `100` on `513`/`1025` bins, `GradientReversalFn`, `DisentangledEmotionEncoder`, and `VocosISTFTHead`), plus the **torchaudio filterbank-compatibility caveat**.
   * **4.3 Specialist A (Stem Separation, Mode Gate & Scene Glue):** `stem_and_mode_gate.py` (`filter_dominant_speakers`, `Causal3StemSeparator` at `N_FFT=512`, `VocalModeGate`, `SceneAcousticMatcher` with RNG-isolated jittered Velvet-Noise RIR, and `StemAndScenePipeline` at `N_FFT=1024`).
   * **4.4 Specialist A (Streaming Ear, Pointer Bridge & Duration Aligner):** `ear_cartridge.py` (`StreamingEarCartridge` at `N_FFT=1024`), `codeswitch_router.py` (`AcousticPointerCodeSwitchBridge`), and `duration_aligner.py` (`LearnedDurationAligner` with 6-frame minor presyllable cap).
   * **4.5 Specialist A (Khmer Mouth & S2ST Console):** `mouth_cartridge.py` (`KhmerMouthCartridge` with float32 length-normalized sinusoidal queries, `reorder_attn` MHA syntax bridge, and 15-layer AdaLN-Zero DiT) and `s2st_console.py` (`SymmetricS2STConsole` with `compute_s2st_training_losses` and `compute_flow_loss`).
   * **4.6 Specialist C (Corruptor, 0-Hz VTLP & Dual-Profile Trainer):** `augmentations.py` (`DynamicAcousticCorruptor` with 0-Hz/Nyquist-invariant 2D bilinear VTLP and RMS-scaled YouTube mixture corruption), `train_cartridge.py` (`CartridgeTrainer`), and `eval_s2st_suite.py` (`AutomatedS2STBenchmarkSuite`).
   * **4.7 Specialist D (Rewind-Resilient Transport):** `session.py` (`SpeakerRegistry` and `InMemorySession` with `active_clause_wall_clock_ms` Trigger #2 guard, `reset_on_seek`, and 8-bin energy-envelope replay cache with `+/- 1` duration-bucket live-seek fallback), `live_queue.py` (`LiveRequestQueue`), `runner.py` (`FullDuplexRunner` and 44-byte RIFF/WAVE `encode_wav_pcm16_b64`), and `server.py` (FastAPI WebSocket server with 8-byte Float64 timestamp unpacking and `asyncio.to_thread` non-blocking GPU offload).
   * **4.8 Specialist D (TypeScript Frontend & WebAudio Worklets):** `pcm-capture-processor.ts` (unity-gain 15-tap Blackman-sinc decimator, sum = 1.0000), `stereo-mne-mixer.ts` (4096-bin 48kHz Mid/Side Log-ERB suppressor indexing all 64 bands via `Math.floor(ERB_BANDS * norm)` + `decodeAudioData` epoch fencing and 48kHz AudioContext guard), `duplex-ws-client.ts`, and XSS-safe `caption-overlay.ts`.
   * **4.9 Specialist B & D (Member Rust Crate Manifests & ABI Spec):** `Cargo.toml` manifests for `crates/khmer-linguistics-rs`, `crates/khmer-transport-rs`, and `crates/khmer-dsp-wasm`, and **Section 4.9.7: the Rust-to-Python ABI specification (Phase 2 deliverables R-1 through R-8)**.
3. **[`docs/03_TDD_GATES_AND_ORCHESTRATION.md`](./docs/03_TDD_GATES_AND_ORCHESTRATION.md)** (Parts 5-8)
   * **Part 5 (6 Adversarial TDD Test Gates):** Complete pytest suite across `test_codeswitch_decomposer.py` (Gate 1), `test_symmetric_s2st.py` (Gate 2), `test_multispeaker_and_modes.py` (Gate 3), `test_fast_talker_and_glue.py` (Gate 4), `test_corruptor_and_infill.py` (Gate 5), and `test_live_queue.py` (Gate 6).
   * **Parts 6-8 (Acceptance Matrix & Multi-Agent Orchestration):** Ground-truth acceptance table, zero-file-overlap specialist ownership plan (now including `eval_s2st_suite.py` and the five Rust modules), and the copy-pasteable v10.7 Master Orchestrator prompt with the new STAGE 06 document-truth validation gate.
4. **[`docs/04_AUDIT_FINDINGS_AND_REMEDIATION.md`](./docs/04_AUDIT_FINDINGS_AND_REMEDIATION.md)** (Audit Ledger)
5. **[`docs/05_CODE_LEVEL_AUDIT_AND_REMEDIATION.md`](./docs/05_CODE_LEVEL_AUDIT_AND_REMEDIATION.md)** (Code-Level Audit Ledger): the v10.6 to v10.7 code-level audit - the audit method with every re-derived numeric claim, the eleven Part 4 reference-implementation defects (P0-A, P0-B, P1-A..P1-E, P2-A..P2-D) plus three additional findings (A-1 the absent upstream audio path, A-2 the brand hash collision, A-3 the dead telemetry), the **unchanged** parameter ledger with the parameter-neutrality proofs, the V1-V24 validation suite, the environment caveats, and the regeneration procedure.
   * All 15 v10.5 audit findings (AUD-01 through AUD-15) with evidence, root cause, and remediation; the audit method; the V1-V10 validation suite and v10.6 results; and the regeneration procedure.

---

## Quick Verification Commands

```bash
# 1. Run all Python Red-to-Green TDD verification gates (Gates 1-6)
pytest backend/tests/ -v

# 2. Run strict frontend TypeScript typecheck
npm --prefix frontend run typecheck

# 3. Run Rust workspace unit tests
cargo test --workspace
```

> [!IMPORTANT]
> These commands describe the **target** workspace, which is fully specified in
> Part 2.2. The repository root is the **blueprint**: the authoritative
> artifacts are `KHMER-SOVEREIGN-S2ST-BLUEPRINT-v10.7.txt` and the `docs/`
> volumes.
>
> `backend/`, `frontend/src/` and the three Rust crates ARE materialised on
> disk, but they are **generated**, not hand-written. `__v107_materialize.py`
> writes them from the Part 4 reference blocks, and `__gen_v107.py` writes
> the master and the `docs/` volumes. Regenerating either replaces the
> materialised tree, so never edit a file under `backend/` or `frontend/src/`
> directly — edit the blueprint block it came from and re-run the
> generator.
>
> This project is still in the **planning and specification** phase: the
> materialised tree is the reference implementation the blueprint specifies,
> not a trained or deployed engine.
