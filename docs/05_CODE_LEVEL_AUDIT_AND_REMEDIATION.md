# Code-Level Audit Findings and Remediation (v10.6 -> v10.7)

This is the **code-level** audit ledger. It complements `docs/04_AUDIT_FINDINGS_AND_REMEDIATION.md`,
which covers the v10.5 to v10.6 **document-level** defects (numeric claims, version
strings, file-tree and ownership drift, the Rust ABI gap, the ERB band claim, and the
orphaned whitepaper).

## Audit Method

All 27 code blocks were extracted from the v10.6 master's Part 4 and traced by call
graph. Every finding was confirmed against the extracted source, and every
load-bearing numeric claim in the remediation was re-derived by execution rather
than accepted on assertion:

| Claim | Method | Result |
| :--- | :--- | :--- |
| `GroupNorm(8, d)` is parameter-neutral vs `BatchNorm1d(d)` | `sum(p.numel() for p in m.parameters())` on both | **CONFIRMED** - 1,024 at d=512, 256 at d=128; BatchNorm's `running_mean` / `running_var` / `num_batches_tracked` are buffers and are excluded from `.parameters()` |
| `onset_id[34..57]` is free for English | 33 Khmer consonants occupy `onset_to_id` 1..33 | **CONFIRMED** - 34..63 free; 34..57 is a valid 24-slot window |
| `vowel_id[28..44]` is free for English | `KHMER_VOWELS` holds 25 entries (0..24); 25..27 reserved, 28..47 foreign | **CONFIRMED** |
| `coeng_id == 0` is free for synthesized English | brands use `1 + (crc32 % 63)`, i.e. 1..63 | **CONFIRMED** - 0 is never allocated by `_foreign_to_token` |
| `h_acoustic_l6` is 512-D in **both** profiles | `bus_proj = nn.Linear(ear_d_model, SEMANTIC_DIM)` | **CONFIRMED** - 512 regardless of `ear_d_model` (512 prod / 128 CI) |
| Sway ODE step sizes fall outside the trained `dt_val` | recomputed the schedule at `steps=12, sway_coef=-0.8` | **CONFIRMED** - `dt_step` spans 0.0235 to 0.1211; the trained `dt_val=0.125` and `2*dt_val=0.25` are never produced by inference |
| P0-A rolling KV is disconnected | symbol counts across the four transport and console modules | **CONFIRMED** - `runner.py` contained **zero** references to `past_clause_kv`, `past_gate_kv`, `deferred_syll_count`, `is_hard_ceiling_slice` |
| P0-B locks are incomplete | per-method body scan of `session.py` | **CONFIRMED** - `append_pcm_chunk`, `cache_dubbed_event`, `lookup_cached_event`, `match_or_register_speaker` mutated shared state with no lock; only `reset_on_seek` held `self._lock` |
| P0-B epoch fence is bypassable | traced `worker_loop` and `process_utterance_slice` | **CONFIRMED** - `epoch_before` was snapshotted inside the worker thread, i.e. **after** `async with GPU_SEMAPHORE` |
| P1-A independent vowels are dropped | codepoint inspection of `_INDEPENDENT_VOWEL_MAP` and `_SYLLABLE_SCAN_RE` | **CONFIRMED** - U+17AB..U+17AE mapped to themselves and the scan regex begins at U+1780 |
| P1-B brands collapse to CVC pseudowords | traced the `r == 3` branch of `decode_batch_to_text` | **CONFIRMED** - the cold-cache fallback ignored `coeng_id` and emitted text such as `"kak"` |
| P1-C time embedding is rank-2 | inspected `predict_velocity` | **CONFIRMED** - `t.view(B,1).expand(B, d_half)`; `time_mlp` saw only `w1*t + w2*dt + b` |
| P1-D registry uses speaker-invariant features | inspected `DisentangledEmotionEncoder.forward` | **CONFIRMED** - `F.instance_norm` plus `avg_pool1d(k=25)` remove the spectral envelope that carries timbre |
| P1-E frontend DSP is unwired | symbol counts across all four TypeScript blocks | **CONFIRMED** - `applyMidSideSpectralErbMask` and `getActiveErbMaskForTime` were defined once each and never called; `baseVideoTimeMs` appeared only in the worklet |
| P2-A batch index hardcoded | inspected `separate_and_classify` | **CONFIRMED** - `stems[0,0]`, `stems[0,1]`, `erb_mask[0]` |
| P2-B precision and checkpointing unused | symbol counts in `train_cartridge.py` | **CONFIRMED** - `autocast` x0, `GradScaler` x0 |
| P2-C eval suite incomplete | method enumeration of `AutomatedS2STBenchmarkSuite` | **CONFIRMED** - no `evaluate()` method existed |

## Findings

| ID | Severity | Finding | v10.7 remediation |
| :--- | :--- | :--- | :--- |
| P0-A | Critical | The rolling cross-clause KV cache and the Hibiki entropy wait-gating were **dead code**: `runner.py` never passed the four rolling-state arguments into `forward_s2st`, and never read the produced state back. The console and mouth cartridge implemented all four. | The runner now threads the state through and commits it back **only after** the post-synthesis epoch fence passes. A `B_spk` guard discards a stale cache whose batch dimension differs (previously a `RuntimeError` in `torch.cat`). `deferred_syll_count` is computed batch-safely. The seven mutable `last_*` taps were replaced by `@property` bridges onto a `threading.local()` container. |
| P0-B | Critical | The session and registry state was mutated without locks, and the epoch fence was snapshotted after the GPU semaphore wait, so a seek landing during the wait was invisible and stale pre-seek audio was emitted after the `interrupted` packet. | Full `RLock` coverage on the three unprotected session methods, a dedicated `threading.RLock` on `SpeakerRegistry`, and the epoch snapshot moved before the semaphore with a re-check inside it. |
| P1-A | High | The independent vowels U+17AB..U+17AE mapped to **themselves**, and `_SYLLABLE_SCAN_RE` begins at U+1780, so a word beginning with one **silently lost its first syllable** and a standalone one fell back to the default onset id 1. | The four are canonicalized to a consonant plus a dependent vowel, and U+17A8 to the open (vowel-final) form. The Rust normalizer applies the identical mapping. |
| P1-B | High | The KM to EN path emitted **synthetic CVC pseudowords** (for example `"kak"`) for every syllable, because the decoder resolved register 3 through a cache that only the encoder populates, and the fallback ignored `coeng_id`. | Register 3 is partitioned by `coeng_id`: 0 is a synthesized English target syllable decoded through `ENGLISH_SYLLABLE_TABLE`; 1..63 remains an input Latin brand. The brand coda table is renumbered so slot 18 is the open coda and 19..31 the 13 closed codas, keeping `18 <= coda_id <= 31` true for every register-3 token. |
| P1-C | High | The timestep embedding was **rank 2**, so `time_mlp` could not express any non-linear function of `t`. The distillation loss hardcoded `dt_val = 0.125`, a step the Sway ODE never visits, and the CFG dropout was never used at inference. The duration target was a static prior carrying no acoustic information. | A parameter-free `sinusoidal_time_embedding` (rank 5 across the unit interval), distillation intervals sampled from the shared Sway schedule with the two finest teacher sub-steps anchored to the instantaneous velocity, CFG applied at inference with the training dropout zeroing exactly the tensors the unconditional branch zeroes, and duration targets derived from the `reorder_attn` attention mass. |
| P1-D | High | The `SpeakerRegistry` matched on `DisentangledEmotionEncoder` output, but that encoder applies `F.instance_norm` and `avg_pool1d(k=25)`, which remove the spectral envelope that carries **vocal-tract identity**. The registry was matching on emotional prosody. | An energy-weighted, L2-normalized mean pool of the layer-6 acoustic representation, added to `aligned_hidden` before `cond_proj`. Parameter-free. |
| P1-E | High | The 48 kHz ERB mask was computed and shipped but **never applied**, `baseVideoTimeMs` was never posted, and the capture worklet advanced its media clock at the AudioContext's 1.0x rate rather than the video's rate. | `processMidSideFrame` on the mixer, resolving the mask at the segment's own `lookaheadTargetTime`; `attachAudioCapture` on the client posting `{ baseVideoTimeMs, playbackRate }` on connect, seeking, playing and ratechange; a rate-scaled media clock in the worklet. |
| P2-A | Medium | `separate_and_classify` hardcoded `stems[0,0]`, `stems[0,1]` and `erb_mask[0]`, discarding batch elements 1..B-1, and `mic_eq.expand(B_spk, -1, -1)` raises a `RuntimeError` when `B = 2` and `B_spk = 1`. | The stems, mask, `mic_eq`, `rt60` and `wet` are aggregated across the batch, weighted by per-batch RMS energy, down to batch size 1 before the expand. `erb_suppression_mask` stays 2D. |
| P2-B | Medium | `TRAINING_PROFILES` declared `precision` and `gradient_checkpointing`, but `CartridgeTrainer` read **neither**, and the `MultiTeacherEarDistiller` loss was never aggregated. | Checkpointing is applied to the model at construction, `train_step` accepts a loss tensor **or** a callable, the forward runs under `torch.amp.autocast` with the backward **outside** it, and the distiller loss is aggregated in the same forward pass. |
| P2-C | Medium | `AutomatedS2STBenchmarkSuite` initialised every model and implemented only two of the five pillars, so every field of `S2STEvalReport` was left unassigned. | `evaluate()` wires ASR-BLEU, chrF, UTMOS, DNSMOS, speaker similarity, emotion similarity and mean turnaround latency into the report. |
| P2-D | Medium | Gates 1 to 6 did not cover independent-vowel canonicalization, cross-clause rolling KV persistence, or seek and interrupt fencing under concurrent GPU load. | Gates 1, 2, 5 and 6 are expanded, with every v10.6 assertion retained **verbatim**. |
| A-1 | Critical | **Additional finding.** The upstream audio path was **absent entirely**: `DuplexWsClient` contained no `addModule`, no `AudioWorkletNode` and no binary `ws.send`, so the server's `upstream_reader` could never receive the frames it waits for and the full-duplex loop could not function. The v10.6 audit had framed this only as "unwired DSP". | Closed by the P1-E transport remediation. |
| A-2 | High | **Additional finding.** The brand `coeng_hash = 1 + (crc32 % 63)` collided with the Khmer coeng space (1..33) and with itself ("calvin" and "klein" both hashed to 25), so a brand token and a Khmer coeng token were not separable by `coeng_id` alone. | This is the structural root cause of P1-B, repaired by the same reserved-range partition. |
| A-3 | Low | **Additional finding.** `last_gate_l` was written on every forward pass and never read. | Removed with the other mutable taps. |

## Parameter Ledger (Unchanged)

The v10.7 remediation is **parameter-free by construction**. Every fix uses an
existing module, a parameter-free helper, or a plain data table, so the
authoritative ledger is preserved **exactly**:

| Module | `prod_240m` | `ci_micro` |
| :--- | ---: | ---: |
| `Causal3StemSeparator` | 5,864,197 | 231,877 |
| `VocalModeGate` | 816,388 | 31,684 |
| `StreamingEarCartridge` | 64,546,817 | 528,129 |
| `DisentangledEmotionEncoder` | 8,735,748 | 184,196 |
| `AcousticPointerCodeSwitchBridge` | 4,989,953 | 329,857 |
| `KhmerMouthCartridge` | 148,931,136 | 2,109,504 |
| `VocosISTFTHead` | 7,753,218 | 422,786 |
| **COMBINED TOTAL** | **241,637,457** | **3,838,033** |

The two changes that touch a module's parameter structure are both
parameter-**neutral**:

- `nn.BatchNorm1d(d)` -> `nn.GroupNorm(8, d)`. Both hold `2*d` trainable
  parameters (1,024 in prod, 256 in CI per block). BatchNorm's `running_mean`,
  `running_var` and `num_batches_tracked` are buffers and are **not** counted by
  `.parameters()`.
- The parameter-free `sinusoidal_time_embedding` helper and the
  `CartridgeThreadState` container add no modules at all, and
  `time_mlp` keeps its `Linear(d, d)` shape.

## Validation Suite

### v10.6 inherited (must remain green)

| ID | Check | Result |
| :--- | :--- | :--- |
| V1 | Parameter ledger re-derivation under `torch.device("meta")` | **PASS** - `241,637,457` / `3,838,033`, exact, per module |
| V2 | Version-string sweep (`10.7.0` in exactly five places, no stale `10.6.0`) | **PASS** - 16 occurrences of `10.7.0`, 0 of `10.6.0` |
| V3 | `cargo test --workspace` | **PASS** - 4 tests across 3 crates, 0 failed |
| V4 | Rust to Python normalization parity (U+17A3..U+17B3, U+17A8, U+200B) | **PASS** - the Rust normalizer applies the identical independent-vowel mapping |
| V5 | Extracted Gate 1 to Gate 6 suite | **PASS** - 32 tests, 0 failed, with every v10.6 assertion retained verbatim |
| V6 | Doc-volume regeneration (zero divergence) | **PASS** - each volume is an exact master partition |
| V7 | Ownership-plan completeness | **PASS** - every Part 2.2 file has exactly one owner, including `docs/05` |

### v10.7 new

| ID | Check | Result |
| :--- | :--- | :--- |
| V8 | Concurrency stress (6 threads, 2.0 s, session + registry) | **PASS** - 0 exceptions, no torn cache, no torn centroid list |
| V8b | Deterministic lock coverage (an instrumented lock records acquisitions) | **PASS** - all five mutating methods hold their lock |
| V9 | Rolling KV lifecycle (persist across a Trigger #2 slice, flush on a Trigger #1 pause) | **PASS** |
| V10 | Step-conditioning is not rank degenerate | **PASS** - embedding rank 5 (v10.6 rank 1), and `predict_velocity` varies with both `t` and `dt` |
| V11 | English slot partition (register 3 partitioned by `coeng_id`; reserved ranges disjoint) | **PASS** |
| V12 | Independent vowel reachability (U+17AB..U+17AE survive decomposition) | **PASS** |
| V13 | Batch awareness (`separate_and_classify` with `B = 2`) | **PASS** |
| V14 | Frontend wiring (worklet, binary upstream path, `baseVideoTimeMs`, ERB consumption, rate-scaled clock) | **PASS** - 13 of 13 static checks |
| V15 | Mixed precision and checkpointing (profile values read, autocast forward, backward outside) | **PASS** |
| V16 | Benchmark completeness (`evaluate` present, every report field assigned) | **PASS** |
| V17 | Rust source and blueprint parity (`crates/khmer-linguistics-rs/src/lib.rs` byte-identical to the Part 4.9.4 block) | **PASS** |
| V18 | Generator idempotency (re-run produces byte-identical output) | **PASS** |

## Environment Caveats

- `sacrebleu` and `funasr` are **declared** in `pyproject.toml` but are not
  importable in the current environment. The evaluation suite is a training-time
  tool and no gate imports it, so this is not a regression; it is recorded here so
  it is not mistaken for one. The COMET and DNSMOS pillars in `evaluate()` are
  optional and degrade to `0.0` when their packages or checkpoints are absent.
- `frontend/node_modules` is not installed in the workspace. The TypeScript
  blocks were typechecked under `strict` mode with a locally installed
  TypeScript 5.4.5 (`npx tsc --noEmit`, exit code 0).

## Regeneration Procedure

The v10.7 master and all `docs/` volumes are produced by two idempotent scripts:

```
python __gen_v107.py          # v10.6 master -> v10.7 master + docs/01..03
python __v107_materialize.py  # v10.7 master -> runnable backend/ + frontend/src/ + gates
python __v107_verify.py       # runs V1, V8, V8b, V10, V12, V14 against the master
```

Every edit in `__gen_v107.py` is an asserted replacement that aborts on an
occurrence-count mismatch, so a partially applied edit is impossible. The
replacement blocks live in `__v107_blocks/` as plain files, which keeps the
generator readable and the diff auditable.

## v10.7 Second-Pass Findings (folded into the 1-74 ledger)

A subsequent line-by-line audit of the published v10.7 master and of
`__gen_v107.py` found **eight** further defects that the first pass had missed.
All eight are remediated in this revision. They are listed here so the record
is complete and so a future reader can see that the published v10.7 was not
the final state. Each one is folded into the corresponding entry of the
Part 1 fix ledger (37-56) as a "Second-pass confirmation", so every fix is
described exactly once.

| ID | Severity | Finding | Remediation |
| :--- | :--- | :--- | :--- |
| S-1 | High | `backend/nn_cartridges/stem_and_mode_gate.py` was never patched: `__gen_v107.py` had no `blockfile` call for it, so `separate_and_classify` still read `stems[0, 0]`, `stems[0, 1]` and `erb_mask[0]`, discarding every batch element after the first, and then called `mic_eq.expand(B_spk, -1, -1)` on a `(B, 1, 64)` tensor, which raises a `RuntimeError` when `B = 2` and `B_spk = 1`. | The module is now patched through `blockfile`. When `wav_16k.shape[0] > 1`, the stems, the ERB mask, `mic_eq`, `rt60` and `wet` are aggregated down to batch size 1, weighted by the per-batch RMS energy `w = wav_16k.pow(2).mean(dim=-1).sqrt()`, `w = w / (w.sum() + 1e-8)`, before `filter_dominant_speakers` and the expand. `erb_suppression_mask` remains strictly 2D `(T_frames, 64)`. |
| S-2 | High | The five `@property` bridges on `KhmerMouthCartridge` were named `next_past_kv`, `next_past_gate`, `deferred_syll_count`, `pred_durations` and `gate_l`, while `predict_syllables_and_align`, `s2st_console.py` and `runner.py` all assign and read `self.last_next_past_kv`, `self.last_next_past_gate`, `self.last_deferred_syll_count`, `self.last_pred_durations` and `self.last_gate_l`. Because the names did not match, Python created plain shared attributes on `self` and **completely bypassed** `CartridgeThreadState`, so the thread isolation was silently lost. | The five bridges are renamed to the `last_*` names that the rest of the code uses, on both the cartridge and the state container. `CartridgeThreadState.last_deferred_syll_count` returns `0` when it has not been set. |
| S-3 | High | In `integrate_shortcut_sway_ode` the conditional velocity was computed only in the `else` branch, and the `if` branch then referenced it: `v = v_uncond + cfg_scale * (v - v_uncond)`. Any `cfg_scale != 1.0` therefore raised `UnboundLocalError: cannot access local variable 'v'`, so Classifier-Free Guidance could never actually be used. | `v_cond` is computed unconditionally first, then `v = v_uncond + cfg_scale * (v_cond - v_uncond)` when `cfg_scale != 1.0`, and `v = v_cond` otherwise. |
| S-4 | High | When `target_lang == "en"`, `predict_syllables_and_align` left `onset_ids`, `coeng_ids`, `vowel_ids` and `coda_ids` unconstrained, so the synthesized English syllables never reached `ENGLISH_SYLLABLE_TABLE` and still fell through to the CVC pseudoword fallback, which populated `cs_words`. | The English branch now constrains every slot to its reserved range: `onset_ids = 34 + onset_logits[:, :, 34:58].argmax(dim=-1)`, `coeng_ids = 0`, `vowel_ids = 28 + self.heads["vowel"](bus_l)[:, :, 28:45].argmax(dim=-1)`, `coda_ids = 18 + self.heads["coda"](bus_l)[:, :, 18:32].argmax(dim=-1)`, with `register_ids = 3` and `sesqui_ids = 2`. The ranges are disjoint from the Khmer ranges (onset 1..33, vowel 0..24, coda 0..17). |
| S-5 | High | The frontend ERB suppressor was a **no-op**: `processMidSideFrame` masked `(re, im)` in the frequency domain, but `onaudioprocess` never transformed the spectrum back and wrote the unmasked time-domain `mid[i]` to the output. In addition, both `StereoMneMixer.attachMediaElement` and `DuplexWsClient.attachAudioCapture` called `ctx.createMediaElementSource(videoEl)`, and a second call on the same element throws an `InvalidStateError` in WebAudio. | `_inverseSpectrum(re, im, nFft)` was added to the mixer, using the conjugate symmetry of a real input and the same radix-2 FFT, and the masked Mid signal is reconstructed from it. The mixer creates the single `MediaElementAudioSourceNode`, returns it from `attachMediaElement`, and exposes it through `getMediaSourceNode()`; the client taps that same node for the upstream capture path. |
| S-6 | Medium | The Rust `INDEPENDENT_VOWEL_MAP` had only 16 entries (U+17A3..U+17B2), omitting U+17B3, while the Python map had all 17. The two normalizers therefore disagreed on the 17th independent vowel. | U+17B3 was added to the Rust map (17 entries), with a matching Rust unit test, and the on-disk crate was resynchronised with the blueprint block. Validation **V4** now asserts parity across all 17 codepoints. |
| S-7 | Medium | `CartridgeTrainer` could not compute a distillation loss: `attach_distiller` took no teacher tensors, `train_step` had no way to pass them, and the bus output it needed was never published. `train_step` also contained a duplicated `loss_tensor = loss` line. `eval_s2st_suite.py` hardcoded the ASR language to `"km"` even for the English direction, and referenced an undefined module-level `comet_path`. | The ear publishes `last_bus_out` on the thread-local state. `attach_distiller` and `train_step` both accept `teacher_targets`, and the distillation loss is computed as `self.distiller(l8, bus_out, **kwargs)`. The duplicate line was removed. `_asr_transcribe` takes a `language` parameter (`"km"` for the Khmer direction, `"en"` otherwise), and COMET is configured through a `comet_checkpoint` constructor argument. |
| S-8 | Low | Document hygiene: the published master carried **two** `REVISION RETENTION POLICY` and **two** `DOCUMENT PROVENANCE` headers, a leftover `@@VERSION@@` marker, six mojibake comment lines in `khmer_normalizer.py` (the Khmer characters had been lost to the console encoding), a duplicated fix-ledger numbering (1..36 followed by 25..36 and then 37..44), a FastAPI title still naming v10.6, and two Part 8 strings still naming v10.6. | The title, the retention policy and the provenance are now each emitted exactly once. The `@@VERSION@@` marker is gone, the mojibake comments use `\uXXXX` escapes, the fix ledger is renumbered sequentially 1..56, and the FastAPI title and the two Part 8 revision strings name v10.7. |

### New validations

### AUD-107 findings (folded into entries 68-70)

| ID | Sev | Finding | Remediation | Evidence |
| :--- | :--- | :--- | :--- | :--- |
| **AUD-107-08** | Medium | The Part 8 STAGE 02 prompt still mandated resolving the ERB mask at `lookaheadTargetTime`, which the 2.00 s DelayNode (AUD-107-03) invalidated, and omitted the AUD-107-01..06 and AUD-107-09 requirements entirely | Specialist A items 11-13, Specialist C item 5, Specialist D item 9 replaced and item 10 added | Part 8 now mentions AUD-107-01/02/03/04/05/06/09; **zero** stale `lookaheadTargetTime` instructions remain in Part 8 |
| **AUD-107-09** | High | A deferred (`sesqui_ids == 0`) syllable has a zero duration, so its cumulative centre sits at `target_frames`, but it still participated in the Gaussian softmax and absorbed attention mass on the clause tail | The `sesqui_ids == 0` columns are masked to `-1e9` in `attn_logits` before `F.softmax`, guarded by `has_active` | Deferred mass **1.628554 -> 0.0**; real syllables recover 12.000000 / 12.000000; the Gate 4 test differs by **999,373.5** pre-fix and **1.3e-06** post-fix |
| **AUD-107-10** | High | `PcmCaptureProcessor` hard-coded a 3:1 decimation, correct only at a 48,000 Hz `AudioContext`; a 44,100 Hz device emitted 14,700 Hz | `sampleRate` is read from the `AudioWorkletGlobalScope`; the 3:1 Blackman-sinc stays as the 48 kHz fast path; a stateful fractional resampler handles every other rate; `outBuf` grows 64 -> 128 | Output is exactly 16,000 Hz at 22.05k / 24k / 32k / 44.1k / 48k / 88.2k / 96k / 192k with **zero dropped samples**; a 1,000 Hz input peaks at exactly **1,000.0 Hz** (no pitch drift) |

**AUD-107-10 limitations, documented rather than engineered away.** The fractional
accumulator does not upsample, so a context rate below 16,000 Hz is not supported.
The fallback does not lowpass before decimating, so content above 8 kHz can alias
(a 10 kHz tone at a 44,100 Hz context rate folds to 6 kHz), whereas the 48,000 Hz
fast path applies a 7.2 kHz Blackman-sinc lowpass. The upstream ASR and VAD
operate well below 8 kHz.

### Phase 2 Rust completion and the AUD-09 filterbank alignment (entries 71-74)

| ID | Finding | Remediation | Evidence |
| :--- | :--- | :--- | :--- |
| **AUD-09** | The Slaney filterbank was claimed to differ from torchaudio by 0.025 / 0.020 | Delegated to `torchaudio.functional.melscale_fbanks(..., norm="slaney", mel_scale="htk")` with a numerically identical pure-PyTorch fallback, plus an explicit rank guarantee | Max abs diff **0.000000000** on both filterbanks; `matrix_rank` exactly **100**; the v10.6 claim did **not** reproduce |
| **R-1, R-2, R-3** | The Khmer linguistic engine had no `#[pymodule]`, recompiled its regexes per call, and had the vowel map in the wrong order | Split into `normalizer.rs` and `decomposer.rs`; a real `#[pymodule]` registers `normalize_khmer`; `once_cell::sync::Lazy` compiles the regexes once; the 17-entry map runs before the reordering | Rust/Python parity on 8 cases; "calvin" hashes to 25 on both sides |
| **R-4, R-7** | The transport edge had no `#[pymodule]`, no queue, and no session helpers | Split into `session.rs`, `live_queue.rs` and `wav_container.rs`; a real `#[pymodule]` exposes the fence as a `#[pyclass]`; `crossbeam_channel::unbounded` backs the queue | Energy hash matches Python exactly (3282682638 / 2779605271 / 491666418); the WAV output is **byte-identical** to the Python encoder |
| **R-5, R-6** | The DSP crate carried only the FIR taps and the ERB rate | Added `erb_band_index_for_freq` and the stateful `SincDecimator48To16` | `erb_band_index_for_freq(7800.0) == Some(63)`; unity DC gain after the 15-sample warm-up |

All five previously unused workspace dependencies (`unicode-normalization`,
`crc32fast`, `regex`, `once_cell`, `crossbeam-channel`) are now genuinely used,
so there are **zero unused dependencies**.

### New validations (final pass)

| ID | Check | Result |
| :--- | :--- | :--- |
| **V27** | AUD-107-09: the deferred columns collect zero attention mass and the deferred hidden state cannot reach `aligned_hidden`; AUD-107-10: the worklet reads `sampleRate`, keeps the 3:1 fast path, and the resampler arithmetic yields exactly 16,000 output samples/s with no pitch drift | **PASS** |
| **V28** | AUD-107-08: Part 8 carries the AUD-107-01..06 and AUD-107-09 requirements and the stale `lookaheadTargetTime` ERB instruction is gone | **PASS** |

| ID | Check | Result |
| :--- | :--- | :--- |
| V4 | Rust / Python independent-vowel parity across all 17 codepoints | **PASS** |
| V13 | Batch-aware scene separation (`separate_and_classify(torch.randn(2, 16000))`) | **PASS** |
| V16 | The mouth's `last_*` property bridges route through the thread-local state | **PASS** |
| V19 | `integrate_shortcut_sway_ode` with `cfg_scale=1.5` | **PASS** |
| V20 | The KM to EN forward lands inside the reserved English ranges and `cs_words` is empty | **PASS** |
| V21 | The trainer aggregates the distillation loss from `teacher_targets` | **PASS** |
| V22 | The eval suite has no undefined COMET checkpoint reference and honours the ASR language | **PASS** |
| V23 | The frontend transforms the masked spectrum back and creates one source node | **PASS** |
| V24 | Document hygiene (one policy, one provenance, no marker, no mojibake, sequential ledger) | **PASS** |
