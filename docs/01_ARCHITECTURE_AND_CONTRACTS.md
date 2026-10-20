# KHMER-SOVEREIGN-S2ST: Rust + PyTorch Hybrid Master Blueprint (v10.7 - Code-Level Audit Remediation Edition)

**System Type:** Symmetric, Bidirectional (`EN <-> KM`), Full-Duplex Direct Speech-to-Speech Translation (S2ST), Rust Zero-Alloc Phonological Engine (`PyO3` + `WASM`), Lock-Free `Tokio`/`Axum` Transport with Epoch-Fenced Seek & Phase-Invariant Utterance Replay Cache, Rust SIMD WebAudio `48kHz` Mid/Side Log-ERB Studio M&E Suppressor & PyTorch `241.637M` Flow-Matching Neural Cartridges
**Active Neural Footprint:** `241.637M` Active Parameters (`PROD_240M_CONFIG`, strictly inside `[230.0M, 250.0M]`) | `3.838M` Parameters (`CI_MICRO_CONFIG`)
**Hardware Target:** Local Dev & Live Inference on 12GB RTX 5070 Ti Laptop GPU (`<= 3.2 GB` VRAM, PyTorch >= 2.7.0 CUDA 12.8+ sm_120) | Full Pre-Training on 80GB H100
**Deterministic Sync & Latency Law:**
`Total_Turnaround = T_active_clause (<= 1.60s wall-clock) + T_pause_detect (0.25s) + T_Rust_VAD_Queue (<= 0.0001s) + T_GPU (<= 0.09s) <= 1.94s < T_lookahead (2.00s)`

---

**REVISION RETENTION POLICY:** The workspace root retains only the **current** revision (v10.7) and the **immediately preceding** one (v10.6). Revisions v10.1 through v10.5 are archived under `_archive_blueprints/` for provenance only and are not authoritative. The authoritative specification is this file; the `docs/` volumes are generated from it.


**DOCUMENT PROVENANCE & v10.7 AUDITED UPGRADE LEDGER:**
This file is the **v10.7 (Code-Level Audit Remediation) Edition** of the KHMER-SOVEREIGN-S2ST blueprint. It supersedes v10.6, carries forward every v10.6 correction, and adds the code-level remediation catalogued in `docs/05_CODE_LEVEL_AUDIT_AND_REMEDIATION.md`:

1. **Fix P0-1 (Full-Rank Mel Constant Propagation):** Fixed `ear_cartridge.py` importing `N_FFT_16K = 1024` (`N_BINS_16K = 513`), guaranteeing full column rank 100 and passing Gate 2. Preserved $N=512$ exclusively for `Causal3StemSeparator` mask to maintain $5.86\text{M}$ parameter ledger invariant while `StemAndScenePipeline.mel_fb` uses 513 bins. Updated `s2st_console.py` to `N_FFT_24K = 2048` (`N_BINS_24K = 1025`) and `HOP_LENGTH = 240`.
2. **Fix P0-2 (Open/Vowel-Final Brand Coda Support):** Fixed `_foreign_to_token` extracting consonant from `lower_w[1:]` or defaulting to slot 18 for vowel-final foreign brands (e.g., "Versace"), ensuring $18 \le \text{coda\_id} \le 31$ always holds and passing Gate 1.
3. **Fix P1-1 (8-Group Phonological Khmer Coda Mapping):** Replaced `min(8, base_id)` clamping with `CODA_GROUP_MAP` partitioning all 14 Khmer codas into 8 phonological place/manner classes `[1..8]` (and `[9..16]` with Bantoc), preserving distinct distinctions for "ម", "ល", "ស", "ក", "ង", etc.
4. **Fix P1-2 (ERB Band 63 Full Passband Partition):** Updated `StereoMneMixer.applyMidSideSpectralErbMask` to `Math.floor(ERB_BANDS * norm)` instead of `ERB_BANDS - 1`, ensuring Band 63 $[7603.5\text{ Hz}, 8000\text{ Hz}]$ is indexed across high-frequency bins. *(v10.6 correction: the v10.5 text stated $7353\text{ Hz}$, which is the Band 62 edge; recomputing $21.4\log_{10}(1+0.00437 f)$ at the Band 63 boundary yields $7603.5\text{ Hz}$.)*
5. **Fix P1-3 (VocosISTFTHead Parameter Ledger Alignment):** Updated parameter ledger for $N_{\text{FFT\_24K}} = 2048$ (`istft_head: Linear(512, 2050)` with $1.05\text{M}$ params; VocosISTFTHead = $7.753\text{M}$; combined engine = **241.637M**, strictly inside `[230M, 250M]`).
6. **Fix P2-1 (Guarded Replay Cache Fallback):** Guarded fuzzy bucket fallback in `InMemorySession.lookup_cached_event` to execute only when `pcm_crc32 == 0` or within 1 duration bucket.
7. **Fix P2-2 (Toolchain & Orchestration Spec Synchronization):** Bumped toolchain versions to `10.4.0` in `Cargo.toml`, `pyproject.toml`, and `frontend/package.json`. Synchronized Part 2.2, 2.3, and Part 8 master orchestration prompt to v10.4 specifications. *(v10.6 correction: this synchronization was never actually applied to the shipped `docs/` volumes or to the Part 8 prompt, which still carried `10.4.0`; v10.6 applies it for real at `10.7.0`.)*
8. **Fix P0-3 (Trigger #2 Trailing Silence Pre-emption in Gate 6):** Subtracted `self.silence_ms` from `utterance_wall_clock_ms` (`active_clause_wall_clock_ms = utterance_wall_clock_ms - self.silence_ms`) in `InMemorySession.append_pcm_chunk`. This prevents trailing pause accumulation from prematurely tripping Trigger #2 during natural sentence boundary pauses, ensuring Gate 6 test (b) cleanly fires Trigger #1 with `pause_ms >= 250.0`.
9. **Fix P2-3 (Live Seek Duration-Bucket Fallback & 241.55M Label Polish):** Enabled $\pm 1$ duration-bucket fuzzy matching on live seeks with non-zero `pcm_crc32` in `lookup_cached_event`. Reconciled all remaining `241.02M` prose and table labels across Part 1.1, Part 2.1, and Part 2.2 to `241.55M`. *(v10.6 correction: the target label was itself inaccurate; the measured value is `241.637M`, see Part 3.3.)*
10. **Fix P0-4 (Cargo Virtual Workspace Syntax Error):** Moved `pyo3 = { version = "0.22.2", features = ["auto-initialize"] }` into `[workspace.dependencies]` in root `Cargo.toml`, delegating `[target.'cfg(...)'.dependencies]` exclusively to member crates (`khmer-linguistics-rs`, `khmer-transport-rs`) to eliminate Cargo manifest parse errors.
11. **Fix P2-4 (Replay Cache Live-Seek CRC Check & G2P 63-way Brand Disambiguation):** Enforced `cached_crc == pcm_crc32` check on $\pm 1$ duration fallback in `InMemorySession.lookup_cached_event`. Seeded `coeng_id = 1 + (crc32 % 63)` for Latin loanwords in `_foreign_to_token` to eliminate reverse brand cache shell collisions across homophonic loanwords.
12. **Fix P0-5 (WebSocket Server & GPU Event Loop Non-Blocking Architecture):** Added `backend/transport/server.py` (Section 4.7.4) unpacking the 8-byte Float64 `timestamp_ms` prefix and offloading synchronous GPU inference via `await asyncio.to_thread(runner.process_utterance_slice, ...)` to ensure atomic interrupt and seek epoch fencing is never blocked.
13. **Fix P1-4 (Member Rust Crate Manifests & Hardware Guards):** Added Section 4.9 detailing `Cargo.toml` manifests for member crates `khmer-linguistics-rs`, `khmer-transport-rs`, and `khmer-dsp-wasm`. Added explicit 48000Hz `AudioContext` guard to `StereoMneMixer` constructor to eliminate 44.1kHz hardware decimation pitch drift.
14. **Fix P0-6 (Immediate Control-Plane Interrupts & Thread-Safe Session/GPU Locks):** Moved `"seek"` and `"interrupt"` control handling directly into `upstream_reader()` in `backend/transport/server.py` so `session.reset_on_seek()` and downstream `"interrupted"` events fire in `< 1 ms` without waiting for in-flight `asyncio.to_thread` GPU work. Guarded `runner` GPU inference with `GPU_SEMAPHORE = asyncio.Semaphore(1)` and protected `InMemorySession` caches with `threading.RLock()`.
15. **Fix P0-7 (Layer-6 Acoustic Pointer Source Routing & Layer-8 Distillation Tap):** Updated `StreamingEarCartridge` to expose intermediate Layer-6 (`last_l6_hidden`) and Layer-8 (`last_l8_hidden`) representations projected to `SEMANTIC_DIM = 512`. `SymmetricS2STConsole` now passes `(bus, self.ear.last_l6_hidden)` into `AcousticPointerCodeSwitchBridge` instead of `(bus, bus)` so brand passthrough copies genuine early formant structure.
16. **Fix P0-8 (Cross-Clause Rolling KV-Cache & Hibiki Entropy Wait-Gating):** Integrated `past_clause_kv`, `past_gate_kv`, `deferred_syll_count`, and Shannon entropy wait-gating (`entropy_wait_threshold = 1.85`) directly into `KhmerMouthCartridge.predict_syllables_and_align` and `InMemorySession`. When Trigger #2 slices at `1.60s`, trailing 20 frames (`800ms`) and deferred modifier syllable counts roll into the next clause; Trigger #1 natural pauses and `reset_on_seek` flush the cache.
17. **Fix P0-9 (Step-Conditioned Shortcut Flow Matching, Sway ODE, Flow2GAN & CFG):** Extended `KhmerMouthCartridge.predict_velocity` with step-size conditioning `dt`, added Sway Sampling (`sway_coef = -0.8`), binary step-halving Shortcut Distillation (`compute_shortcut_distillation_loss`), Flow2GAN spectral energy-adaptive loss weighting, 15% CFG conditioning dropout, explicit gate BCE (`l_gate_bce`), and duration MSE (`l_dur`) supervision.
18. **Fix P1-5 (Cold-Cache Cluster Detokenizer, Independent Vowels, Robat & Coda 17 Guard):** Updated `khmer_normalizer.py` to canonicalize independent vowels (`\u17A3..\u17B3`) into `អ` + dependent vowel. Updated `syllable_decomposer.py` to preserve prefixes on Step B double-coeng splits, map Sanskrit/Pali Robat (`៌`, `\u17CC`) to vowel slot `24`, assign plain bilabial nasal `ម` to coda slot `17` (while guarding Bantoc `ម់` -> `14` to avoid colliding with foreign codas `[18..31]`), and reconstruct full `coeng_id`, `medial_id`, and register shifters in cold-cache `decode_batch_to_text`.
19. **Fix P1-6 (Multi-Teacher Ear Distiller, Vocos Surgical Warm-Start & 5-Stage Curriculum):** Added `MultiTeacherEarDistiller` (`w2v-BERT 2.0`, `Canary-1B-v2`, `Qwen3-ASR-0.6B-Khmer`), `warm_start_vocos_from_pretrained` (`charactr/vocos-mel-24khz` 1024 -> 2048 surgical loader), complete `CartridgeTrainer.train_step`, and `backend/training/eval_s2st_suite.py` (`AutomatedS2STBenchmarkSuite`).
20. **Fix P1-7 (Unified Frontend Media Clock & 2.0x Playback-Scaled ERB Masking):** Synchronized `baseVideoTimeMs` into `PcmCaptureProcessor` so audio and seek events share the media clock domain, and scaled `StereoMneMixer` ERB mask duration and frame indexing by `100.0 * safeRate`.
21. **Fix P1-8 (Compilable Rust Crate Sources for Phase 1 Workspace Tests):** Added complete compilable Rust source files (`src/lib.rs`) for `khmer-linguistics-rs`, `khmer-transport-rs`, and `khmer-dsp-wasm` in Section 4.9 so `cargo test --workspace` passes out of the box.

22. **Fix AUD-01 (CI_MICRO Parameter Ledger Understated by 60%):** Re-derived the entire Part 3.3 ledger by instantiating every Part 4 reference module under `torch.device("meta")` and summing `p.numel()`, exactly as Gate 2 does. The `ci_micro` column was never computed against `CI_MICRO_CONFIG` and claimed `2.40M`; the measured total is **`3,838,033` (`3.838M`)**. Root cause documented in Part 3.3: `KhmerMouthCartridge` holds 1,409,756 parameters pinned to `SEMANTIC_DIM = 512` that do not scale with `mouth_d_model`, accounting for 36.7% of the CI budget. Gate 2's `[230M, 250M]` assertion is unaffected.
23. **Fix AUD-02 (PROD Parameter Ledger Precision Defect):** Same re-derivation. The `prod_240m` column claimed `241.55M`; the measured total is **`241,637,457` (`241.637M`)** - an 87,457 overstatement of precision. All seven per-module rows and the combined total now carry exact integers alongside their rounded forms. Gate 2's `[230M, 250M]` assertion is unaffected.
24. **Fix AUD-03 (v10.4 / v10.5 Master Files Were Byte-Identical):** Verified by SHA-256 comparison: `KHMER-SOVEREIGN-S2ST-BLUEPRINT-v10.4.txt` and `KHMER-SOVEREIGN-S2ST-BLUEPRINT-v10.5.txt` shared the identical digest `0DC1A212EB76D1AD06ECF7A319FC33F06E3D8F35DD63E7F04BE0152BAF95618C`. The "v10.5" revision was a copy, not a delta. Only the **current** revision (v10.6) and the **immediately preceding** one (v10.5) are retained in the workspace root; v10.1 through v10.4 were removed after the v10.6 sign-off and are archived under `_archive_blueprints/`. This v10.6 file is a genuine, independently generated revision.
25. **Fix AUD-04 (Version-String Drift Across Five Files):** The v10.5 "Fix P2-2" entry claimed the toolchain was synchronized to `10.4.0`, but the shipped `docs/` volumes and the Part 8 prompt still carried `10.4.0` while the root manifests carried `10.5.0` - the synchronization was never applied. v10.6 applies it for real at `10.7.0` and adds **Section 2.3.1 Version Manifest** enumerating the five authoritative locations, so the next drift is caught by reading one block. Validation **V2** enforces it.
26. **Fix AUD-05 (Rust-to-Python ABI Is Dead Code):** The Python bridge in `backend/linguistics/khmer_normalizer.py` gates its fast path on `hasattr(_rs, "normalize_khmer")`, but no member crate exposes a `#[pymodule]` and no symbol of that name exists in the Rust build - the probe can never succeed, and the Part 1.1 "`< 15 µs`/sentence, zero-GC Rust edge" claim is unsupported. Added **Section 4.9.7** specifying the exact Phase 2 deliverables (R-1 through R-8) and their acceptance checks, including the mandatory independent-vowel canonicalization (R-2) that the Rust path currently omits, and the `Lazy<Regex>` compile-once requirement (R-3) that removes the per-call regex recompilation contradicting the "zero-allocation" claim.
27. **Fix AUD-06 (Rust File Tree Named Modules That Do Not Exist):** The Part 2.2 tree named `normalizer.rs`, `decomposer.rs`, `session.rs`, `live_queue.rs`, and `wav_container.rs`, none of which exist on disk. Section 4.9.7 now specifies their responsibilities and public surfaces as Phase 2 deliverables, and the Part 2.2 tree annotates them as specified-but-not-yet-implemented so the tree is no longer silently wrong.
28. **Fix AUD-07 (`eval_s2st_suite.py` Was Orphaned):** `backend/training/eval_s2st_suite.py` was specified in Section 4.6.3 and referenced in the v10.5 fix ledger, but appeared in **neither** the Part 2.2 file tree **nor** the Part 7 ownership list - no specialist would ever have been assigned it. It is now present in the Part 2.2 tree and explicitly owned by Specialist C in Part 7.
29. **Fix AUD-08 (ERB Band 63 Edge Frequency Misstated):** The v10.5 fix ledger claimed Band 63 spans `[7353 Hz, 8000 Hz]`. Recomputing `21.4 * log10(1 + 0.00437 f)` at the Band 63 boundary of the `Math.floor(ERB_BANDS * norm)` indexer yields **7603.5 Hz**; 7353 Hz is the **Band 62** edge. The indexing code was always correct - only the prose was wrong. Corrected in the Fix P1-2 entry above, in Section 4.8.2, and in the Part 8 validation checklist.

30. **Fix AUD-09 (Slaney Filterbank Is Not torchaudio-Compatible):** `build_slaney_mel_fb` reimplements torchaudio's Slaney construction. The v10.6 audit recorded a maximum absolute element-wise difference against `torchaudio.functional.melscale_fbanks` of approximately 0.025 (513-bin) and 0.020 (1025-bin), attributed to normalizing each triangular filter by its own half-width rather than by the full filter width torchaudio uses. **Re-measurement in v10.7 did not reproduce those figures:** the implementation has normalized by the FULL filter width in every revision from v10.4 onward, and it agrees with torchaudio 2.11 to a maximum absolute difference of **0.000000000** on both the 513-bin and the 1025-bin filterbank, with `torch.linalg.matrix_rank(fb)` exactly **100** for both. The v10.7 remediation therefore makes the agreement STRUCTURAL rather than coincidental: `build_slaney_mel_fb` is DELEGATED to `torchaudio.functional.melscale_fbanks(..., norm="slaney", mel_scale="htk")`, so a future torchaudio change to the Slaney construction can no longer silently desynchronise the filterbank from the `charactr/vocos-mel-24khz` warm-start checkpoint. A pure-PyTorch Slaney fallback is retained for environments without torchaudio and is numerically identical (verified by blocking the torchaudio import). The column rank is now guaranteed explicitly: a filter column with an empty triangular support would drop the rank below `n_mels` and break the Gate 2 assertion, so it receives a `1e-5` floor at its peak bin; no shipped configuration has a degenerate column, so the floor is a no-op and the filterbank stays bit-identical to torchaudio's. PARAMETER-FREE: the filterbank is a buffer, never a parameter.
31. **Fix AUD-10 (Undeclared Evaluation-Suite Dependencies):** `backend/training/eval_s2st_suite.py` imports `sacrebleu`, `transformers`, and `funasr`, none of which appear in `pyproject.toml`; `sacrebleu` and `funasr` are also not importable in the audited environment, so the suite would raise `ImportError` on first use. Added to the Part 2.3 `pyproject.toml` block and to the Part 8 orchestrator prompt.
32. **Fix AUD-11 (Orphan Training Whitepaper):** `Training, Distillation, and Dataset.txt` (8,617 words) was referenced by **nothing** in `README.md`, `docs/`, or the master blueprint. It is now cross-referenced from the Part 7 and Part 8 sections of this blueprint and from `docs/README.md`, and its `241.55M` / `2.40M` figures are aligned with the corrected ledger.
33. **Fix AUD-12 (Doc-Volume Line Counts and Alignment Claim Were False):** `docs/README.md` claimed "100% bitwise & token aligned" and published line counts of 691 / 2,764 / 407. The actual counts were 699 / 3,273 / 407, and a 12-line version-token divergence existed between the master blueprint and the concatenated volumes. v10.6 regenerates the volumes from this master with zero divergence, re-derives the published counts from the regenerated files, and replaces the alignment claim with a verifiable statement plus a documented regeneration procedure.
34. **Fix AUD-13 (`frontend/tsconfig.json` `include` Diverged From Contract):** The on-disk file declared `"include": ["src/**/*"]` while Part 2.3 specifies `["src/**/*.ts"]`. The Part 2.3 block is authoritative and the on-disk file is aligned to it.
35. **Fix AUD-14 (Unused Rust Dependencies Declared):** `khmer-linguistics-rs` declared `crc32fast` and `khmer-transport-rs` declared `crossbeam-channel`; neither is imported anywhere. Removal is specified as R-7 / R-8 in Section 4.9.7. `once_cell` is also currently unused, but is **retained** because R-3 puts it to work for compile-once regexes.
36. **Fix AUD-15 (FastAPI App Title Carried a Stale Version):** `backend/transport/server.py` Section 4.7.4 hard-codes `FastAPI(title="KHMER-SOVEREIGN-S2ST v10.5 Full-Duplex Server")`; updated to `v10.6`.
37. **Fix P0-A (Rolling Cross-Clause KV Cache Was Dead Code):** `FullDuplexRunner.process_utterance_slice` never passed `past_clause_kv`, `past_gate_kv`, `deferred_syll_count` or `is_hard_ceiling_slice` into `console.forward_s2st`, and never read the produced state back. The console and mouth cartridge accepted and implemented all four arguments, so the entire cross-clause feature was unreachable in production. v10.7 wires the state through the runner, guards the `B_spk` transition in `predict_syllables_and_align` (a stale cache with a different batch dimension previously raised a `RuntimeError` in `torch.cat`), and computes `deferred_syll_count` batch-safely as `int(wait_mask.float().sum(dim=-1).max().item())` instead of indexing `wait_mask[0]`, which silently ignored speakers 1..B-1.
38. **Fix P0-A (Mutable Module Attributes Replaced by Thread-Local State):** The per-call cartridge taps (`last_l6_hidden`, `last_l8_hidden`, `last_next_past_kv`, `last_next_past_gate`, `last_deferred_syll_count`, `last_pred_durations`, `last_gate_l`) were mutable attributes on `StreamingEarCartridge` and `KhmerMouthCartridge`. The GPU worker runs in `asyncio.to_thread` while the event loop owns the control plane, so one thread could observe another's half-written state. v10.7 introduces `CartridgeThreadState` in the Part 2.4 contract, backed by `threading.local()`, and exposes every tap as a `@property` bridge onto it. Attribute compatibility is 100 percent preserved, so no existing call site changed, and the storage is now strictly per-thread. No parameters are added. Second-pass confirmation (S-2): The five `@property` bridges on `KhmerMouthCartridge` were named without the `last_` prefix that `predict_syllables_and_align`, `s2st_console.py` and `runner.py` use. Python therefore created plain shared attributes on `self` and bypassed `CartridgeThreadState` entirely, so the thread isolation was silently lost. v10.7 renames all five bridges, on both the cartridge and the state container, and `last_deferred_syll_count` now returns `0` when unset.
39. **Fix P0-B (Full Lock Coverage):** `InMemorySession.append_pcm_chunk`, `cache_dubbed_event` and `lookup_cached_event` mutated shared state without holding `self._lock`, and `SpeakerRegistry.match_or_register_speaker` performed an unsynchronised read-modify-write on `self.centroids`. The `OrderedDict` resize could occur during the `reversed(self._replay_cache.items())` iteration in `lookup_cached_event`, and a concurrent caller could observe a half-updated centroid vector. v10.7 holds the session lock across all three method bodies and gives `SpeakerRegistry` its own `threading.RLock`. No parameters are added.
40. **Fix P0-B (Epoch Fence Snapshot Moved Before the GPU Semaphore):** `worker_loop` snapshotted `epoch_before` inside the worker thread, that is, after `async with GPU_SEMAPHORE`. A seek that arrived while the worker was queued therefore incremented `queue.interrupt_epoch` before the snapshot was taken, so the fence saw no change and the stale pre-seek audio was emitted after the `interrupted` packet. v10.7 snapshots `pkt_epoch` before the semaphore is awaited and re-checks it inside the semaphore, aborting the slice when it changed.
41. **Fix P1-A (Independent Vowels Were Silently Dropped):** `_INDEPENDENT_VOWEL_MAP` mapped U+17AB..U+17AE (RY / RYY / LY / LYY) to THEMSELVES. `_SYLLABLE_SCAN_RE` begins at U+1780, so a self-mapped codepoint is unreachable by the scanner: a word beginning with one lost its first syllable, and a standalone one fell back to the default onset id 1. v10.7 canonicalizes the four to a consonant plus a dependent vowel (U+179A U+17B9, U+179A U+17BA, U+179B U+17B9, U+179B U+17BA) and corrects U+17A8 to the open (vowel-final) form U+17A2 U+17BB U+1780, which was carrying the closed coda U+1782. The Rust `normalize_khmer_pure` applies the identical mapping, so the two paths agree once the Phase 2 ABI lands. Second-pass confirmation (S-6): The Rust `INDEPENDENT_VOWEL_MAP` held 16 entries while the Python map held 17, so the two normalizers disagreed on the 17th independent vowel. v10.7 adds U+17B3 to the Rust map, with a matching unit test, and Validation V4 asserts parity across all 17 codepoints.
42. **Fix P1-B (KM to EN Emitted Synthetic CVC Pseudowords):** The mouth cartridge forces `register_id == 3` for `target_lang == "en"`, and the decoder resolved that register through `_reverse_brand_cache`, which is populated only by `_foreign_to_token`. On a cold cache the fallback ignored `coeng_id` entirely and emitted text such as "kak" for every syllable. v10.7 partitions register 3 by `coeng_id`: `coeng_id == 0` is a synthesized English target syllable, decoded through the new `ENGLISH_SYLLABLE_TABLE` (onset ids 34..57, medial ids 0..3, vowel ids 28..44, coda ids 18..31, disjoint from the Khmer ranges), while `coeng_id` in 1..63 remains an input code-switched Latin brand. The brand G2P coda table is renumbered so that slot 18 is the open coda and 19..31 are the 13 closed codas, which keeps `18 <= coda_id <= 31` true for EVERY register-3 token, and `_foreign_to_token` keeps exactly one token per brand word. Second-pass confirmation (S-4): With `target_lang == "en"` the onset, coeng, vowel and coda ids were left unconstrained, so the synthesized English syllables never reached `ENGLISH_SYLLABLE_TABLE` and still fell through to the CVC pseudoword fallback, populating `cs_words`. v10.7 constrains every slot to its reserved range, which is disjoint from the Khmer ranges.
43. **Fix P1-C (Rank-2 Timestep Embedding):** `predict_velocity` expanded a rank-1 scalar across `d_half` columns, so `time_mlp` only ever saw `w1*t + w2*dt + b` and could not express any non-linear function of the timestep. v10.7 replaces the expansion with a parameter-free `sinusoidal_time_embedding` helper using log-spaced frequencies `exp(-ln(10000) * i / (dim/2))`. `time_mlp` keeps its `Linear(d, d)` shape, so the ledger is unchanged, and the embedding rank rises from 1 to 5 across the unit interval.
44. **Fix P1-C (Shortcut Distillation Anchored to the Sway Schedule):** The distillation loss hardcoded `dt_val = 0.125` and evaluated the teacher half-steps at that same step size, so the hierarchy was anchored to nothing, and the trained step size was never one the Sway ODE visits at inference (0.0235 to 0.1211 at `steps=12`, `sway_coef=-0.8`). v10.7 samples the coarse-to-fine step pairs directly from the Sway schedule through a shared `sway_time_schedule` helper, evaluates the two finest teacher sub-steps with the instantaneous velocity network (`dt=None`, i.e. `dt = 0.0`) so the whole hierarchy is anchored to the ground-truth OT-CFM vector field, and combines them with the step-weighted average `(dt_1 * v_step1 + dt_2 * v_step2) / dt_full`.
45. **Fix P1-C (Classifier-Free Guidance Was Never Applied):** The 15 percent conditioning dropout was applied in training only, so the unconditional branch was learned and then discarded, and the inference side applied no guidance at all. v10.7 gives `integrate_shortcut_sway_ode` a `cfg_scale` and blends the conditional and unconditional velocities, and the training-side dropout now zeroes exactly the tensors the inference unconditional branch zeroes (`cond` and `emo_tokens`), so the guided direction points at a branch the model was actually trained on. `cfg_scale = 1.0` reproduces the v10.6 behaviour exactly. Second-pass confirmation (S-3): `integrate_shortcut_sway_ode` computed the conditional velocity only in the `else` branch and referenced it in the `if` branch, so any `cfg_scale != 1.0` raised `UnboundLocalError: cannot access local variable 'v'`. v10.7 computes `v_cond` unconditionally and blends from it.
46. **Fix P1-C (Non-Circular Duration Supervision):** `compute_s2st_training_losses` supervised `pred_durs` against a static 0.25 / 1.00 prior built from the `sesqui` class alone, a target that is a function of the label distribution rather than of the audio. v10.7 derives the target from the monotonic attention mass of `reorder_attn` (`attn_w.sum(dim=-1)`, scaled to `T_mel`, with the 6-frame minor-syllable cap and the excess redistributed to the major syllables).
47. **Fix P1-D (Speaker Registry Matched on Speaker-Invariant Features):** `process_utterance_slice` registered speakers using the `DisentangledEmotionEncoder` output, but that encoder applies `F.instance_norm` and `avg_pool1d(k=25)`, which remove the spectral envelope that carries vocal-tract identity, so the registry was matching on emotional prosody. v10.7 uses an energy-weighted, L2-normalized mean pool of the layer-6 acoustic representation, which is speaker-discriminative, and adds the same vector to `aligned_hidden` before `self.cond_proj` so the acoustic conditioning path preserves speaker identity. Both changes are parameter-free.
48. **Fix P1-E (Frontend DSP and Transport Were Unwired):** `applyMidSideSpectralErbMask` and `getActiveErbMaskForTime` were defined once each and never called, so the 64-band ERB mask the server computes and shipped was queued and then ignored. `DuplexWsClient` never instantiated `PcmCaptureProcessor` and never sent a binary frame, so the server's `upstream_reader` waited for bytes that never arrived and the full-duplex loop could not function. `baseVideoTimeMs` was never posted, and the capture worklet advanced its media clock at the AudioContext's 1.0x rate rather than at the video's rate. v10.7 adds `attachAudioCapture(ctx)` to the client, posts `{ baseVideoTimeMs, playbackRate }` on connect, seeking, playing and ratechange, advances the worklet media clock by the wall-clock delta scaled by `playbackRate`, and adds `processMidSideFrame` to the mixer. The ERB mask MUST be resolved at the segment's own `lookaheadTargetTime`, NOT at `this.ctx.currentTime`: a newly queued segment starts 2.0 s in the future, so a current-time query always returned null. Second-pass confirmation (S-5): The masked spectrum was computed and then discarded, and the unmasked time-domain Mid was written to the output. Both the mixer and the client also called `ctx.createMediaElementSource` on the same element, which throws an `InvalidStateError`. v10.7 adds `_inverseSpectrum` to the mixer, reconstructs the masked Mid from it, and has the mixer own the single source node, which it returns and exposes through `getMediaSourceNode()` for the client to tap.
49. **Fix P1-E (BatchNorm Replaced by GroupNorm):** `FastConformerBlock` used `nn.BatchNorm1d(d)`, which normalizes over the batch axis. The batch axis has size 1 during streaming inference, so the running statistics are corrupted by batch-statistics estimation. v10.7 uses `nn.GroupNorm(8, d)`, matching `VocalModeGate`. The swap is PARAMETER-NEUTRAL: both hold 2*d trainable parameters (1,024 in prod, 256 in CI per block), and BatchNorm's running buffers are not counted by `.parameters()`.
50. **Fix P2-A (Batch Dimension Was Hardcoded):** `separate_and_classify` read `stems[0, 0]`, `stems[0, 1]` and `erb_mask[0]`, silently discarding elements 1..B-1 when called with a batch, and `mic_eq.expand(B_spk, -1, -1)` on a `(B, 1, 64)` tensor raises a `RuntimeError` when `B = 2` and `B_spk = 1`. v10.7 aggregates stems, `erb_mask`, `mic_eq`, `rt60` and `wet` across the batch dimension, weighted by per-batch RMS energy, down to batch size 1 before calling `filter_dominant_speakers` and expanding to `B_spk`. `erb_suppression_mask` remains strictly 2D `(T_frames, 64)` as Gate 3 requires. Second-pass confirmation (S-1): `separate_and_classify` hardcoded `stems[0, 0]`, `stems[0, 1]` and `erb_mask[0]`, discarding every batch element after the first, and `mic_eq.expand(B_spk, -1, -1)` on a `(B, 1, 64)` tensor raises a `RuntimeError` when `B = 2` and `B_spk = 1`. v10.7 aggregates the stems, the ERB mask, `mic_eq`, `rt60` and `wet` down to batch size 1, weighted by the per-batch RMS energy, before `filter_dominant_speakers` and the expand. `erb_suppression_mask` stays strictly 2D.
51. **Fix P2-B (Mixed Precision and Checkpointing Were Declared but Unused):** `TRAINING_PROFILES["laptop_5070ti_12gb"]` declares `precision=torch.bfloat16` and `gradient_checkpointing=True`, but `CartridgeTrainer` read neither, so the profile could not hold its 3.2 GB VRAM ceiling on a 12 GB RTX 5070 Ti. v10.7 applies the checkpointing configuration to the model at construction time, and `train_step` accepts EITHER a pre-computed loss tensor OR a callable that performs the forward pass. The forward pass runs under `torch.amp.autocast`; the backward pass runs OUTSIDE the autocast block, because wrapping `.backward()` in autocast is a documented PyTorch anti-pattern. The `MultiTeacherEarDistiller` loss is also aggregated, which the v10.6 trainer never did. Second-pass confirmation (S-7): The ear never published its bus output, so the multi-teacher distiller could not compute a loss; `attach_distiller` and `train_step` had no way to receive the teacher tensors; and `train_step` carried a duplicated `loss_tensor = loss` line.
52. **Fix P2-C (Benchmark Suite Had No Entry Point):** `AutomatedS2STBenchmarkSuite` initialised every model (ASR x2, WavLM-SV, emotion2vec+, UTMOS, DNSMOS) and implemented only two of the five pillars, so every field of `S2STEvalReport` was left unassigned. v10.7 implements `evaluate()`, wiring ASR-BLEU, chrF, UTMOS, DNSMOS, speaker similarity, emotion similarity and mean turnaround latency into the report. Second-pass confirmation (S-7): `eval_s2st_suite.py` hardcoded the ASR language to `"km"` even for the English direction and referenced an undefined `comet_path`. v10.7 parameterises the ASR language so the English direction uses the English ASR model, and configures COMET through a `comet_checkpoint` constructor argument instead of an undefined module-level name.
53. **Fix P2-D (Gate Coverage Extended):** Gates 1, 2, 5 and 6 are expanded to cover the repaired behaviour, with every v10.6 assertion retained VERBATIM. Gate 1 adds independent-vowel canonicalization and reachability, Robat and coda membership, the KM to EN English decode, and the brand one-slot and coda-range invariants. Gate 2 adds the exact integer ledger equality, the non-rank-degenerate step conditioning, and the KM to EN forward decode. Gate 5 adds the multi-teacher distillation loss and the AMP and checkpointing contract. Gate 6 adds the rolling KV lifecycle, the `B_spk` transition, the thread-safety stress and the epoch-fence ordering. Second-pass confirmation (S-8): The published master carried two revision-retention headers, two document-provenance headers, a leftover version marker token, six mojibake comment lines in `khmer_normalizer.py`, a duplicated fix-ledger numbering, a FastAPI title naming v10.6, and two Part 8 strings naming v10.6. v10.7 emits each header exactly once, removes the marker, uses `\uXXXX` escapes in the affected comments, renumbers the ledger sequentially from 1 to 56, and names v10.7 throughout.
54. **Fix A-1 (Additional Finding, Upstream Audio Path Was Absent):** The v10.6 audit treated the missing frontend DSP as a P1 defect. Direct inspection showed the larger problem: `DuplexWsClient` contained no `addModule`, no `AudioWorkletNode`, no `ArrayBuffer` and no binary `ws.send`, so the server's `upstream_reader` could never receive the frames it waits for and the full-duplex loop could not function at all. This is the reason the frontend remediation in entry 48 is a transport fix and not only a DSP fix.
55. **Fix A-2 (Additional Finding, Brand Hash Collided With the Khmer Coeng Space):** `coeng_hash = 1 + (crc32 % 63)` lands in 1..63 while Khmer coeng ids occupy 1..33, so a brand token and a Khmer coeng token were not separable by `coeng_id` alone ("calvin" and "klein" both hashed to 25). This is the structural root cause of P1-B, and the reserved-range partition in entry 42 repairs it.
56. **Fix A-3 (Additional Finding, Dead Telemetry):** `last_gate_l` was written on every forward pass and never read by anything. It is removed with the other mutable taps in entry 38.
57. **Fix P1-E (Hann Window Tapered Every Audio Block to Zero):** `ScriptProcessorNode(4096, 2, 2)` delivers NON-OVERLAPPING 4096-sample blocks and `N_FFT` equals the block size, so the frontend has no overlap-add and no inverse-window compensation anywhere in the path. `_computeSpectrum` nevertheless applied a Hann window, which tapers the first and last samples of every block to zero. Under an active ERB mask the reconstruction therefore came back with a hard zero at both block edges, producing periodic dropouts in the M&E bed at the block rate. v10.7 uses UNITY (rectangular) windowing, so the forward/inverse pair is the exact identity on the block, and the unused `const HOP = 480` declaration was removed. Validation V23a executes the pair under node and asserts both the round-trip energy ratio (1.000000000) and per-sample equality (max abs error 1.6e-16) against the unwindowed input.
58. **Fix P0-A (Deferred Tail Syllables Were Not True No-Ops):** The Hibiki wait-gating zeroed `bus_l` BEFORE the head projections, but `nn.Linear` has a bias vector, so a zero input still produces a non-zero logit vector and the argmax returns a real onset id. `decode_batch_to_text` skips a syllable only when the triplet is exactly `(0, 0, 0)`, so the deferred tail syllables were emitted as real content. v10.7 applies `wait_mask` to the ARGMAXED `onset_ids`, `vowel_ids` and `coda_ids` directly, so a deferred syllable becomes a true no-op and the decoder skips it. Validation V25 and a new Gate 6 test assert the `(0, 0, 0)` triplet and the shortened decode.
59. **Fix S-8 (Document Hygiene, Second Pass):** The Part 1 ledger entry for the second pass still claimed a sixty-four entry ledger after the redundant entries had been folded, and three mojibake `?` characters remained in `WORKLOG.md` sections 13.1, 13.3 and 13.5. The ledger entry now states the final ledger length, the three characters are em-dashes, and the worklog header, sections 5, 7 and 12 consistently report the current check count, the current gate count and the final ledger length.
60. **Fix S-8 (Workspace State Aligned With README):** The README note claimed `backend/` and `frontend/src/` did not exist, but both ARE materialised on disk by `__v107_materialize.py`. The note now states that the repository root is the blueprint, that the materialised tree is generated rather than hand-written, and that files under `backend/` and `frontend/src/` must never be edited directly.
61. **Fix AUD-107-01 (KM to EN Wait-Gating Was Defeated and Deferred Syllables Consumed Frames):** The v10.7 no-op mask was applied BEFORE the `if target_lang == "en"` branch, but that branch overwrites `onset_ids`, `vowel_ids` and `coda_ids` with the reserved English ranges, so on a KM to EN Trigger #2 slice the deferred tail was never masked: the tail syllables were emitted in the current clause AND deferred to the next one, duplicating the words. Separately, `sesqui_ids` was never zeroed, so `LearnedDurationAligner` still assigned a 0.25 or 1.00 duration weight to a deferred `(0, 0, 0)` syllable and the diffusion model spent acoustic frames on a syllable that is not spoken. v10.7 applies the mask AFTER the language branch, extends it to `coeng_ids` and `sesqui_ids`, and gives `sesqui_ids == 0` a zero duration weight in the aligner. Measured: the KM to EN path now yields three true `(0, 0, 0)` no-ops (was zero) and the deferred durations are `[0.0, 0.0, 0.0]` (was `0.25`/`1.00` weighted).

62. **Fix AUD-107-02 (The Ear and the Emotion Encoder Ran Twice on Every Utterance):** `FullDuplexRunner.process_utterance_slice` ran `self.console.ear(stem_16k, "en")`, then `self.console.ear.log_mel(stem_16k)` twice more, then `self.console.emotion_encoder(mel_stems)` whose output was DISCARDED, and finally called `forward_s2st`, which ran the ear and the emotion encoder a second time on the identical audio. That doubled the 64.5M-parameter `StreamingEarCartridge` and the 8.7M-parameter `DisentangledEmotionEncoder` on every spoken utterance and threatened the `T_GPU <= 90 ms` latency law. v10.7 introduces `PrecomputedEarFeatures`, computes the bundle ONCE in the runner, and passes it to `forward_s2st`, which reuses it instead of re-running the ear. Measured: the extra ear calls drop from 1 to 0 and the extra log-mel calls from 2 to 0.

63. **Fix AUD-107-03 (The M&E Graph Had No 2.00s Lookahead Delay):** Part 2.1 specifies a `[2.00s Delay Line (Video + 48kHz Stereo)]` before the `[48kHz Mid/Side ERB Suppressor]`, but `StereoMneMixer.attachMediaElement` connected the source straight into the processor with zero delay. The original unmasked vocals therefore reached the speakers 2.0s BEFORE the ERB mask and the dubbed audio took effect at `lookaheadTargetTime = t_start + 2.00s`, so the suppression was always late by the whole lookahead window. v10.7 inserts a `DelayNode` of `LOOKAHEAD_BUFFER_SEC` between the source and the processor, and still returns the UNDELAYED source node so `DuplexWsClient.attachAudioCapture` taps the audio ahead of playback.

64. **Fix AUD-107-04 (Gradient Checkpointing Renamed Every state_dict Key):** `apply_gradient_checkpointing` wrapped each block in a `_CheckpointedBlock` nn.Module holding the real block as `self.blk`, which renamed every parameter key from `ear.blocks.0.self_attn...` to `ear.blocks.0.blk.self_attn...`. A checkpoint trained under `laptop_5070ti_12gb` (`gradient_checkpointing=True`) therefore could not be loaded under `cloud_h100_80gb` or into an inference model. Measured: 70 of the 178 keys were renamed. v10.7 replaces the module wrapper with an instance-level `forward` override, leaving the nn.Module hierarchy untouched. Measured after the fix: 178 keys before, 178 after, and the key sets are IDENTICAL; the forward output is bit-identical (`max abs diff = 0.0`).

65. **Fix AUD-107-05 (Per-Block FFT Table Allocation and a Dead Spectral Pass):** `_computeSpectrum` and `_inverseSpectrum` allocated their twiddle tables and called `Math.log2(size)` inside the bit-reversal inner loop on EVERY 4096-sample block, which is 98,304 `Math.log2` calls per block, inside a real-time WebAudio callback where a GC pause is an audible dropout. `handleDownstreamEvent` additionally called `processMidSideFrame(lookaheadTargetTime, ...)` on `this._midSpecReal`, which is either a stale spectrum from a completed callback or a throwaway array, so the pass had no effect on the output while still burning a full FFT on every JSON packet. v10.7 precomputes the twiddle and bit-reversal tables at module load (`COS_TABLE_MNE`, `SIN_TABLE_MNE`, `COS_TABLE_MNE_INV`, `SIN_TABLE_MNE_INV`, `REV_TABLE_MNE`) and removes the redundant pass. The mask is applied in the audio callback, which computes the spectrum for the block it is about to emit.

66. **Fix AUD-107-06 (The Capture Worklet Was Wired Straight to the Destination):** `DuplexWsClient.attachAudioCapture` did `node.connect(ctx.destination)`. `PcmCaptureProcessor` leaves its outputs zeroed, so it is silent today, but a future WASM worklet that passes audio through would sum the UNDELAYED dry audio into the destination, bypassing the M&E suppressor and reintroducing the bleed the ERB ducking exists to remove. v10.7 routes the worklet through a `GainNode` with `gain = 0.0`, which keeps the node pulled by the graph while guaranteeing that nothing reaches the speakers.

67. **Fix AUD-107-07 (Split Provenance Ledger and Minor Artifacts):** The audit ledger was SPLIT: entries 1..36 sat above Part 1 while entries 37..60 were pasted inside Part 1 after Section 1.1, so the upgrade ledger had two separate homes. v10.7 keeps the entire ledger in ONE place, immediately under the document-provenance header, and Part 1 begins with the architectural overview it is supposed to contain. Entry 59's claim about the ledger length is corrected to the final count. The duplicated `assert not errors, errors[0]` in Gate 6 is removed, and the unused `clockOffsetMs` field in `PcmCaptureProcessor` is deleted.
68. **Fix AUD-107-08 (Part 8 Master Orchestration Prompt Was Out of Sync With Section 4):** The Part 8 STAGE 02 specialist instructions still carried the v10.7 AUD-107-01 instruction that the ERB mask "MUST be resolved at the segment's own `lookaheadTargetTime`, NOT at `this.ctx.currentTime`". The 2.00s lookahead DelayNode (AUD-107-03) invalidated that instruction: with the M&E audio stream delayed by 2.0s, the mask MUST be resolved at the audio callback's own clock. Part 8 also omitted the AUD-107-01, AUD-107-02, AUD-107-03, AUD-107-04, AUD-107-05, AUD-107-06 and AUD-107-09 requirements entirely, so a downstream implementer following the copy-pasteable prompt would have reproduced five of the defects the audit had just closed. v10.7 rewrites Specialist A items 11-13 (AUD-107-01, AUD-107-09, AUD-107-02), adds Specialist C item 5 (AUD-107-04), and replaces Specialist D item 9 with the AUD-107-03 / AUD-107-05 / AUD-107-06 requirements, adding item 10 (AUD-107-02). Every pre-existing Part 8 item is retained verbatim.

69. **Fix AUD-107-09 (Deferred Syllables Stole Attention Mass From the Clause Tail):** A deferred (wait-gated) syllable has a zero duration, so its cumulative centre sits at `target_frames`, but it still participated in the Gaussian `F.softmax(attn_logits, dim=-1)` and absorbed attention mass on the final frames of the clause. Measured with `sesqui_ids = [[2, 2, 0, 0]]`, `sigma = 1.5` and `target_frames = 24`: the two deferred columns collected **1.628554** of the 24-frame mass each, squeezing the two REAL syllables from 12.000000 / 12.000000 down to 12.000000 / 8.742891. v10.7 masks the `sesqui_ids == 0` columns to `-1e9` in `attn_logits` BEFORE the softmax, and applies that mask only where the row has at least one active syllable (`has_active`), so a fully deferred row stays well defined. Measured after the fix: the deferred columns collect **0.0** and the real syllables recover 12.000000 / 12.000000. The regression test is in Gate 4, and it is discriminating: against the pre-fix aligner the aligned output differs by **999,373.5**, and against the fixed aligner by **1.3e-06**. PARAMETER-FREE.

70. **Fix AUD-107-10 (The Capture Worklet Assumed a 48,000 Hz AudioContext):** `PcmCaptureProcessor` hard-coded a 3:1 decimation, which is only correct when the `AudioContext` runs at 48,000 Hz. On iOS Safari, certain Android WebViews and Bluetooth headsets the OS locks the context to 44,100 or 96,000 Hz, and the 3:1 decimator then emitted 14,700 Hz instead of 16,000 Hz, so every upstream packet was timestamped against the wrong sample rate. v10.7 declares `sampleRate` from the `AudioWorkletGlobalScope`, keeps the 15-tap Blackman-sinc 3:1 decimator as the 48,000 Hz fast path, and adds a stateful fractional linear-interpolation resampler for every other rate, with the output buffer raised from 64 to 128 entries because the fallback needs up to 128 outputs per 128-frame quantum. Measured: the output rate is exactly 16,000 Hz at 22,050 / 24,000 / 32,000 / 44,100 / 48,000 / 88,200 / 96,000 and 192,000 Hz, with **zero dropped samples**, and a 1,000 Hz input peaks at exactly 1,000.0 Hz in the output, so there is no pitch or speed drift. Two limitations are documented rather than engineered away: the fractional accumulator does not upsample, so a context rate below 16,000 Hz is not supported, and the fallback does not lowpass before decimating, so content above 8 kHz can alias (a 10 kHz tone at a 44,100 Hz context rate folds to 6 kHz), whereas the 48,000 Hz fast path applies a 7.2 kHz Blackman-sinc lowpass. PARAMETER-FREE.
71. **Fix AUD-09 (Slaney Filterbank Aligned With torchaudio):** `build_slaney_mel_fb` in `backend/nn_cartridges/universal_core.py` (Section 4.2.1) is now DELEGATED to `torchaudio.functional.melscale_fbanks(..., norm="slaney", mel_scale="htk")`, with the pure-PyTorch Slaney construction retained as a fallback for environments without torchaudio. Measured against the torchaudio reference (torchaudio 2.11): the maximum absolute element-wise difference is **0.000000000** on BOTH the 513-bin (16 kHz) and the 1025-bin (24 kHz) filterbank, and `torch.linalg.matrix_rank(fb)` is exactly **100** for both. The fallback was verified to be numerically identical by blocking the torchaudio import. The column rank is now GUARANTEED explicitly: a filter column with an empty triangular support would drop the rank below `n_mels` and break the Gate 2 assertion, so it receives a `1e-5` floor at its peak bin; no shipped configuration has a degenerate column, so the floor is a no-op and the filterbank stays bit-identical to torchaudio's. The v10.6 WARNING that recorded the discrepancy is replaced by a resolved note. PARAMETER-FREE: the filterbank is a buffer, never a parameter.

72. **Fix Phase 2 R-1, R-2, R-3 (The Khmer Linguistic Engine Is Complete):** `crates/khmer-linguistics-rs` is now split into `src/normalizer.rs` (Section 4.9.4.1) and `src/decomposer.rs` (Section 4.9.4.2) with `src/lib.rs` (Section 4.9.4) as the crate root. **R-1:** a real `#[pymodule] fn khmer_linguistics_rs` is delivered, registering `normalize_khmer` and `normalize_khmer_pure` as `#[pyfunction]`s, so the Python `hasattr(_rs, "normalize_khmer")` probe now succeeds and the "15 microsecond Rust edge" is reachable rather than dead code. **R-2:** the 17-entry `INDEPENDENT_VOWEL_MAP` is applied BEFORE the three-pass coeng/shifter reordering, matching the Python execution order exactly; measured Rust/Python parity on 8 cases including the compound word U+17AB U+1791 U+17D2 U+1792 U+17B7. **R-3:** `RE_COENG` and `RE_SHIFT` are compiled ONCE at startup through `once_cell::sync::Lazy` instead of being recompiled on every call. `decompose_khmer_word` returns `(surface, sesqui_id, forced_register_id)` using the same chunk grammar as the Python `_SYLLABLE_SCAN_RE`, and `compute_brand_coeng_hash` uses `crc32fast::hash`, which is bit-identical to Python's `zlib.crc32` (verified: "calvin" -> 25 on both sides).

73. **Fix Phase 2 R-4, R-7 (The Real-Time Transport Edge Is Complete):** `crates/khmer-transport-rs` is now split into `src/session.rs` (Section 4.9.5.1), `src/live_queue.rs` (Section 4.9.5.2) and `src/wav_container.rs` (Section 4.9.5.3) with `src/lib.rs` (Section 4.9.5) as the crate root. **R-4:** a real `#[pymodule] fn khmer_transport_rs` is delivered, exposing the interrupt fence as a `#[pyclass]` so the epoch is SHARED with Python rather than duplicated, and the queue is built on `crossbeam_channel::unbounded`, whose `flush_all` drains the channel in well under the 2 microsecond budget and bumps the atomic epoch. **R-7:** `compute_pcm_energy_hash` reproduces the Python `InMemorySession.compute_pcm_energy_hash` exactly, including the 8-bin normalized log-RMS profile quantized to 0..=99 and the IEEE CRC-32; verified against the Python values 3282682638 (ones), 2779605271 (loud first half) and 491666418 (ramp). The `SpeakerRegistry` mirrors the Python registry (max 4 speakers, 0.72 cosine threshold, 0.85/0.15 L2-normalized EMA centroid update), and `encode_wav_pcm16` produces a WAV byte-for-byte identical to the Python `encode_wav_pcm16_b64`.

74. **Fix Phase 2 R-5, R-6 (The 48 kHz DSP Edge Carries the ERB Mapping and the Decimator):** `crates/khmer-dsp-wasm/src/lib.rs` (Section 4.9.6) keeps `FIR_15_TAPS` and `hz_to_erb_rate` and adds `erb_band_index_for_freq(freq_hz: f32) -> Option<usize>`, which returns `None` outside `[120.0, 8000.0]` and `Some(floor(64 * norm).min(63))` inside it, mirroring the browser's `hzToErbRate` and the Part 2.1 band partition. The unit tests assert that `erb_band_index_for_freq(7800.0) == Some(63)`, which is the Band 63 `[7603.5 Hz, 8000 Hz]` coverage, that the index is monotonic across the band, and that the stateful `SincDecimator48To16` has unity DC gain after its 15-sample filter warm-up. PARAMETER-FREE throughout: no trainable parameter is added, so the ledger remains exactly 241,637,457 (`prod_240m`) and 3,838,033 (`ci_micro`), and all five previously unused workspace dependencies (`unicode-normalization`, `crc32fast`, `regex`, `once_cell`, `crossbeam-channel`) are now genuinely used.

---

## PART 1: HYBRID ARCHITECTURAL OVERVIEW (WHY RUST + PYTORCH)

### 1.1 Subsystem Division: Rust Real-Time Edge vs. PyTorch Autograd Core

| Track & Subsystem | Execution Engine | Target Latency / Footprint | Why This Engine Wins |
| :--- | :--- | :--- | :--- |
| **Specialist B: 7-Slot Khmer & Code-Switch Engine** (`crates/khmer-linguistics-rs` + `backend/linguistics/`) | **Rust (`PyO3` + `WASM`)** / **Python Reference** | **`< 15 µs` / sentence** (`0` heap GC pauses) | Replaces Python regex loops with deterministic UTF-8 state machines, G2P phonetic loanword approximation, and compile-time phonological lexicon lookup. Phase 1 provides pure Python reference; Phase 2 compiles to Python (`maturin`) and Browser/Mobile (`wasm32`). |
| **Specialist D (Backend): VAD, Seek-Safe Session, Queue, Speaker Registry & WS** (`crates/khmer-transport-rs` + `backend/transport/`) | **Rust (`tokio` + `axum` + `crossbeam` + `PyO3`)** / **Python Async** | **`< 10 µs` VAD slice / `< 2 µs` interrupt flush / `< 0.1 ms` rewind replay** | Enforces wall-clock elapsed ceiling ($\le 1.60\text{s}$) on Trigger #2. Atomic `interrupt_epoch` fences in-flight GPU work and phase-invariant 8-bin energy envelope replay cache serves rewound utterances instantly. |
| **Specialist D (Frontend): 48kHz Decimator & Mid/Side Log-ERB Suppressor** (`crates/khmer-dsp-wasm` + `frontend/src/`) | **Rust `wasm32` SIMD + TypeScript** | **`< 0.18 ms` per 128-sample WebAudio quantum** | Runs unity-gain 15-tap sinc decimation ($48\text{kHz} \to 16\text{kHz}$) and 4096-bin STFT/ISTFT 64-band `[120Hz..8kHz]` Mid-channel masking in pre-allocated memory so V8 garbage collection never causes audio clicks. |
| **Specialist A: 241.637M Neural Cartridges & Differentiable Scene Glue** (`backend/nn_cartridges/`) | **PyTorch (`CUDA` / `bfloat16` / `torch.fft`)** | **`<= 90 ms` (`B_spk=2`)** / **`241.637M` params** | Retains native complex STFT/ISTFT autograd, full-rank Slaney Mel filterbank, batched `torch.fft.rfft`/`irfft` room/EQ convolution, `GradientReversalFn`, and `torch.device("meta")` instant verification. |
| **Specialist C: Corruptor, 0-Hz VTLP & Dual-Profile Trainer** (`backend/training/`) | **PyTorch (`bfloat16` + `F.grid_sample`)** | **`laptop_5070ti_12gb` & `cloud_h100_80gb`** | Retains 2D bilinear `align_corners=True` 0-Hz/Nyquist-invariant VTLP formant warping and text-free OT-CFM flow infilling (`infill_mode=True`). |

> [!NOTE]
> **Dual-Phase Implementation Architecture:**
> - **Phase 1 (Active TDD & Live Reference):** High-velocity Python (`backend/`) and TypeScript (`frontend/`) modules pass all 6 adversarial TDD gates, run natively in standard PyTorch and WebAudio runtimes, and establish exact behavioral contracts.
> - **Phase 2 (Zero-GIL Production Compilation Target):** Standalone Rust crates (`crates/khmer-*`) provide identical high-throughput, zero-allocation drop-in replacements via Maturin and WASM bindings without modifying any neural cartridge or protocol contracts.

---

## PART 2: STAGE 00 & 01 SINGLE-SOURCE-OF-TRUTH BLUEPRINT

### 2.1 System Topology Map (Rust + PyTorch + WASM Full-Duplex Loop)

```text
+----------------------------------------------------------------------------------------------------------+
| BROWSER / MEDIA CLIENT (Specialist D — TypeScript + Rust WebAssembly AudioWorklet)                       |
|                                                                                                          |
|  [48kHz Stereo Media / Stream Element]                                                                   |
|        |--+-> [Lookahead Tap (t + 2.00s)] --> [khmer-dsp-wasm: Stateful 15-Tap 16kHz Sinc Decimator]     |
|        |                                                       | (8B Float64 Timestamp + Float32 PCM WS) |
|        v                                                       v                                         |
|  [2.00s Delay Line (Video + 48kHz Stereo)]        [khmer-transport-rs: Lock-Free Queue & Monotonic VAD]  |
|        |                                           * <250ms Micro-Pop Filter | 250ms Natural Pause       |
|        |                                           * 1.60s Active Ceiling ([1.20s..1.60s] Min-RMS Split) |
|        |                                           * <2us Atomic Epoch Flush + 32-Slot CRC32 Replay Cache|
|        |                                                       |                                         |
|        |                                                       v                                         |
|        |                                          [PyTorch 241.637M S2ST & Scene Glue + khmer-linguistics]|
|        |                                           * Causal3StemSeparator + VocalModeGate (B_spk <= 2)   |
|        |                                           * StreamingEar + DisentangledEmotion (GRL) + Pointer  |
|        |                                           * ReorderAttn (MHA) + KhmerMouth (15-DiT) + Vocos     |
|        v                                                       | (44B RIFF/WAVE + 64-Band ERB + t_start) |
|  [khmer-dsp-wasm: 48kHz Mid/Side ERB Suppressor] <-------------+--> [Epoch-Fenced 1.0x Timestamp Player] |
|  (Suppresses Mid [120Hz..8kHz] at 100Hz; Keeps <120Hz, >8kHz & Side)  (Re-anchors on Seek; Discards Old) |
+----------------------------------------------------------------------------------------------------------+
```

---

### 2.2 Complete Workspace File Tree & Specialist Ownership Blueprint

```text
khmer-sovereign-s2st/
├── Cargo.toml                                 # [Stage 00 Orchestrator] Rust Workspace Root (WASM-gated pyo3)
├── pyproject.toml                             # [Stage 00 Orchestrator] Python + Maturin packaging (torch>=2.7.0)
├── README.md                                  # [Stage 00 Orchestrator] Hybrid Rust + PyTorch Runbook
├── crates/
│   ├── khmer-linguistics-rs/                  # [Specialist B - Rust Zero-Alloc Khmer + Brand Engine]
│   │   ├── Cargo.toml                         # crate-type = ["cdylib", "rlib"]
│   │   └── src/
│   │       ├── lib.rs                         # PyO3 + Rust exports
│   │       ├── normalizer.rs                  # Canonical Unicode reorderer (preserves \u{200B}) + #[pyfunction] normalize_khmer
│   │       └── decomposer.rs                  # Lexicon + 3-Rule Splitter + Bantoc [9..16] + G2P Brands
│   ├── khmer-transport-rs/                    # [Specialist D - Rust Lock-Free VAD, Phase-Invariant Cache, Queue & WAV]
│   │   ├── Cargo.toml                         # crate-type = ["cdylib", "rlib"]
│   │   └── src/
│   │       ├── lib.rs                         # PyO3 + Rust exports
│   │       ├── session.rs                     # Monotonic VAD (Wall-Clock Ceiling) + reset_on_seek + SpeakerRegistry
│   │       ├── live_queue.rs                  # AtomicEpoch Lock-Free Queue (<5us flush)
│   │       └── wav_container.rs               # 44-Byte RIFF/WAVE PCM16 Encoder
│   │                                          # [v10.6: all 5 Rust modules are SPECIFIED in Section 4.9.7;
│   │                                          #  implementation is a Phase 2 deliverable, NOT a v10.6 change]
│   └── khmer-dsp-wasm/                        # [Specialist D - Rust WebAssembly 48kHz Audio DSP]
│       ├── Cargo.toml                         # crate-type = ["cdylib", "rlib"]
│       └── src/
│           └── lib.rs                         # Unity-Gain Decimator + 4096-bin Mid/Side Log-ERB [120Hz..8kHz] Mask
│                                              # [v10.6: crate requires wasm-bindgen (Section 4.9.7) to build for wasm32]
├── backend/
│   ├── __init__.py
│   ├── contracts/                             # [Stage 01 Orchestrator - IMMUTABLE SOURCE OF TRUTH]
│   │   ├── __init__.py
│   │   └── schemas.py                         # Constants, Configs, Dataclasses & @runtime_checkable Protocols
│   ├── linguistics/                           # [Specialist B - Python Bridge to khmer-linguistics-rs]
│   │   ├── __init__.py
│   │   ├── khmer_normalizer.py                # Normalizer preserving \u200b
│   │   └── syllable_decomposer.py             # BilingualCodeSwitchDecomposer (\u200b + khmer_nltk + Bantoc [9..16] + G2P)
│   ├── nn_cartridges/                         # [Specialist A - 241.637M PyTorch S2ST & FFT Scene Glue]
│   │   ├── __init__.py
│   │   ├── stem_and_mode_gate.py              # filter_dominant_speakers, Causal3StemSeparator, VocalModeGate,
│   │   │                                      # SceneAcousticMatcher (Jittered Velvet RIR), StemAndScenePipeline
│   │   ├── ear_cartridge.py                   # StreamingEarCartridge (Slaney Mel FB -> 512D Bus)
│   │   ├── universal_core.py                  # build_slaney_mel_fb, GradientReversalFn, DisentangledEmotion, Vocos
│   │   ├── codeswitch_router.py               # AcousticPointerCodeSwitchBridge (Brand/Loanword Passthrough)
│   │   ├── duration_aligner.py                # LearnedDurationAligner (6-Frame Minor Cap + Playback Scale)
│   │   ├── mouth_cartridge.py                 # KhmerMouthCartridge (ReorderAttn MHA + 15-DiT + Infill)
│   │   └── s2st_console.py                    # SymmetricS2STConsole (compute_flow_loss + compute_s2st_losses)
│   ├── training/                              # [Specialist C - Corruptor, 0-Hz VTLP & Dual-Profile Trainer]
│   │   ├── __init__.py
│   │   ├── augmentations.py                   # DynamicAcousticCorruptor (RMS-relative SNR + 0-Hz/Nyquist VTLP)
│   │   ├── train_cartridge.py                 # CartridgeTrainer ("laptop_5070ti_12gb" & "cloud_h100_80gb")
│   │   └── eval_s2st_suite.py                 # [v10.6 ADDED] AutomatedS2STBenchmarkSuite (5-Pillar Stage-End Benchmark)
│   ├── transport/                             # [Specialist D - Full-Duplex Transport & Rewind-Safe Runner]
│   │   ├── __init__.py
│   │   ├── session.py                         # InMemorySession (VAD + CRC32 Cache + reset_on_seek) + Registry
│   │   ├── live_queue.py                      # LiveRequestQueue (<5ms interrupt flush + interrupt_epoch)
│   │   ├── runner.py                          # FullDuplexRunner (Epoch Fence + SpeakerRegistry + WAV RIFF)
│   │   └── server.py                          # FastAPI/Rust WebSocket endpoint (/ws/live-dub/{session_id})
│   └── tests/                                 # [Stage 01 Critic Agent - RED-TO-GREEN TDD VERIFICATION GATES]
│       ├── test_codeswitch_decomposer.py      # Gate 1: 11-row Khmer/Brand truth table + reverse detokenizer
│       ├── test_symmetric_s2st.py             # Gate 2: Meta 241.637M check, Rank 100 Slaney, Protocols, GRL
│       ├── test_multispeaker_and_modes.py     # Gate 3: DominanceGate, (B,T) VocalModeGate & ERB mask bounds
│       ├── test_fast_talker_and_glue.py       # Gate 4: Autograd backward through apply_scene_glue & 6-frame cap
│       ├── test_corruptor_and_infill.py       # Gate 5: 0-Hz/Nyquist VTLP invariance & compute_flow_loss
│       └── test_live_queue.py                 # Gate 6: Pop filter, 1.35s pause, 1.60s slice, Epoch, CRC32, RIFF
└── frontend/                                  # [Specialist D - 48kHz Mid/Side Log-ERB & Rewind-Safe Player]
    ├── package.json                           # [Stage 00 Orchestrator] Includes "typecheck": "tsc --noEmit"
    ├── tsconfig.json                          # [Stage 00 Orchestrator] Strict ES2022 TypeScript config
    └── src/
        ├── contracts/types.ts                 # [Stage 01 Orchestrator] Locked TypeScript Mirror of schemas.py
        ├── worklets/pcm-capture-processor.ts  # Stateful 15-Tap 3:1 Blackman-Sinc Decimator + 8B Timestamp
        ├── audio/stereo-mne-mixer.ts          # StereoMneMixer (Log-ERB + Epoch Fence across decodeAudioData)
        ├── client/duplex-ws-client.ts         # Non-destructive addEventListener WS Client + Re-Anchor
        └── ui/caption-overlay.ts              # XSS-Safe DOM node creation with <mark> + Telemetry badge
```

---

### 2.3 Root Workspace Toolchain Configs (`Cargo.toml`, `pyproject.toml`, `package.json`, `tsconfig.json`)

1. **Root `Cargo.toml` (Linker-Safe & WASM-Gated):**
   ```toml
   [workspace]
   resolver = "2"
   members = [
       "crates/khmer-linguistics-rs",
       "crates/khmer-transport-rs",
       "crates/khmer-dsp-wasm",
   ]

   [workspace.package]
   version = "10.7.0"
   edition = "2021"
   license = "Apache-2.0"

   [workspace.dependencies]
   unicode-normalization = "0.1.23"
   crc32fast = "1.4.2"
   regex = "1.10.6"
   once_cell = "1.19.0"
   crossbeam-channel = "0.5.13"
   pyo3 = { version = "0.22.2", features = ["auto-initialize"] }
   # Note: Member crates (`khmer-linguistics-rs`, `khmer-transport-rs`) pull `pyo3 = { workspace = true }`
   # under `[target.'cfg(not(target_arch = "wasm32"))'.dependencies]` in their own Cargo.toml.
   ```

2. **Root `pyproject.toml` (Blackwell sm_120 Supported):**
   ```toml
   [build-system]
   requires = ["setuptools>=68.0", "wheel"]
   build-backend = "setuptools.build_meta"

   [project]
   name = "khmer-sovereign-s2st"
   version = "10.7.0"
   description = "Hybrid Rust + PyTorch Symmetric Bidirectional (EN <-> KM) Full-Duplex Direct S2ST Engine"
   readme = "README.md"
   requires-python = ">=3.10"
   dependencies = [
       "torch>=2.7.0",
       "torchaudio>=2.7.0",
       "numpy>=1.26.0",
       "scipy>=1.11.0",
       "soundfile>=0.12.1",
       "datasets>=2.18.0",
       "khmer-nltk>=1.6",
       "fastapi>=0.110.0",
       "uvicorn[standard]>=0.28.0",
       "websockets>=12.0",
       "pytest>=8.0.0",
       "pytest-asyncio>=0.23.5",
       "sacrebleu>=2.4.0",
       "transformers>=4.44.0",
       "funasr>=1.1.0",
   ]

   [tool.pytest.ini_options]
   asyncio_mode = "auto"
   testpaths = ["backend/tests"]
   pythonpath = ["."]
   ```

3. **`frontend/package.json` & `frontend/tsconfig.json`:**
   * `frontend/package.json`:
     ```json
     {
       "name": "khmer-sovereign-s2st-frontend",
       "version": "10.7.0",
       "private": true,
       "scripts": {
         "typecheck": "tsc --noEmit"
       },
       "devDependencies": {
         "typescript": "^5.4.0"
       }
     }
     ```
   * `frontend/tsconfig.json`:
     ```json
     {
       "compilerOptions": {
         "target": "ES2022",
         "module": "ESNext",
         "moduleResolution": "bundler",
         "lib": ["ES2022", "DOM", "DOM.Iterable"],
         "strict": true,
         "noEmit": true,
         "skipLibCheck": true,
         "isolatedModules": true
       },
       "include": ["src/**/*.ts"]
     }
     ```

### 2.3.1 Version Manifest (Single Authoritative Source)

The project version is **`10.7.0`**. Exactly five files carry it; every other
file must reference, never redefine, this value.

| File | Manifest key | Value |
| :--- | :--- | :--- |
| `Cargo.toml` | `[workspace.package] version` | `10.7.0` |
| `pyproject.toml` | `[project] version` | `10.7.0` |
| `frontend/package.json` | `version` | `10.7.0` |
| Master blueprint, Part 2.3 | `Cargo.toml` / `pyproject.toml` / `package.json` blocks | `10.7.0` |
| Master blueprint, Part 8 | Master Orchestrator prompt, STAGE 00 step 2 | `10.7.0` |

> [!IMPORTANT]
> **v10.7 audit note.** The v10.6 code-level audit found eleven defects in the
> Part 4 reference implementations: the rolling cross-clause KV cache was
> never passed to the console (so the feature was dead code), the session and
> registry state was mutated without locks, the epoch fence was snapshotted
> after the GPU semaphore wait, the timestep embedding was rank-2, the speaker
> registry matched on speaker-invariant features, the KM to EN path emitted
> CVC pseudowords, the frontend never transmitted upstream audio, and the
> 48 kHz ERB mask was computed but never applied. All eleven are remediated
> in v10.7 with **zero new trainable parameters**: the parameter ledger is
> unchanged at `241,637,457` (`prod_240m`) and `3,838,033` (`ci_micro`). Any
> future revision must re-run this sweep (Validation **V2**) before sign-off.

> [!IMPORTANT]
> **v10.6 audit note.** The v10.5 "Fix P2-2" entry claimed this
> synchronization was performed at `10.4.0`, but the shipped `docs/` volumes
> and the Part 8 prompt still carried `10.4.0` while the root manifests
> carried `10.5.0`. v10.6 applies the synchronization for real. Any future
> revision must re-run this sweep (Validation **V2**) before sign-off.

---

### 2.4 Stage 01 Immutable Contracts (`backend/contracts/schemas.py` & `frontend/src/contracts/types.ts`)

#### 1. `backend/contracts/schemas.py`
```python
from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Dict, List, Literal, Optional, Protocol, Tuple, runtime_checkable

import numpy as np
import torch

# ============================================================================
# 1. LOCKED GLOBAL CONSTANTS
# ============================================================================
SAMPLE_RATE_IN: int = 16000
SAMPLE_RATE_OUT: int = 24000
SAMPLE_RATE_MNE: int = 48000
N_FFT_16K: int = 1024
HOP_LENGTH_16K: int = 160          # 10.0ms (100 Hz frame rate)
N_FFT_24K: int = 2048
HOP_LENGTH_24K: int = 240          # 10.0ms (100 Hz frame rate)
N_FFT_48K: int = 4096
HOP_LENGTH_48K: int = 480          # 10.0ms (100 Hz frame rate)
HOP_LENGTH: int = 240              # Backward-compatible 24k hop alias (100 Hz)
N_MELS: int = 100
ERB_BANDS: int = 64
ERB_MIN_FREQ_HZ: float = 120.0
ERB_MAX_FREQ_HZ: float = 8000.0
SEMANTIC_DIM: int = 512
EMOTION_NUM_TOKENS: int = 32
MIC_EQ_TAPS_16K: int = 64
MIC_EQ_TAPS_24K: int = 96
VELVET_RIR_TAPS_24K: int = 6000
MINOR_SYLLABLE_MAX_FRAMES: float = 6.0
MAX_ELASTIC_PAUSE_BORROW_MS: float = 150.0
DOMINANCE_MIN_DB_RATIO: float = -12.0
LOOKAHEAD_BUFFER_SEC: float = 2.0
MAX_CLAUSE_SLICE_SEC: float = 1.6
MIN_CLAUSE_SLICE_SEC: float = 1.2
SILENCE_RMS_THRESHOLD: float = 0.008
UTTERANCE_END_SILENCE_MS: float = 250.0
REPLAY_CACHE_MAX_ITEMS: int = 32
REPLAY_BUCKET_MS: float = 250.0

LanguageCode = Literal["en", "km", "th", "vi"]
LANG_TO_ID: Dict[str, int] = {"en": 0, "km": 1, "th": 2, "vi": 3}


class VocalMode(IntEnum):
    AMBIENT_OR_CLAP = 0
    NON_VERBAL_HUMAN = 1
    SINGING_OR_MUSIC = 2
    SPOKEN_DIALOGUE = 3


# ============================================================================
# 2. DUAL-PROFILE ARCHITECTURE CONFIGURATION
# ============================================================================
@dataclass(frozen=True)
class ModelArchitectureConfig:
    profile_name: Literal["prod_240m", "ci_micro"]
    separator_d_model: int
    separator_layers: int
    mode_gate_d_model: int
    mode_gate_layers: int
    ear_d_model: int
    ear_layers: int
    ear_heads: int
    ear_ffn_dim: int
    emotion_d_model: int
    emotion_layers: int
    emotion_ffn_dim: int
    pointer_d_model: int
    pointer_layers: int
    pointer_ffn_dim: int
    mouth_d_model: int
    mouth_layers: int
    mouth_heads: int
    mouth_ffn_dim: int
    euler_steps: int
    vocos_d_model: int
    vocos_layers: int
    vocos_intermediate_dim: int


PROD_240M_CONFIG = ModelArchitectureConfig(
    profile_name="prod_240m",
    separator_d_model=384,
    separator_layers=3,
    mode_gate_d_model=256,
    mode_gate_layers=4,
    ear_d_model=512,
    ear_layers=16,
    ear_heads=8,
    ear_ffn_dim=2048,
    emotion_d_model=512,
    emotion_layers=4,
    emotion_ffn_dim=2048,
    pointer_d_model=512,
    pointer_layers=2,
    pointer_ffn_dim=1024,
    mouth_d_model=768,
    mouth_layers=15,
    mouth_heads=12,
    mouth_ffn_dim=3072,
    euler_steps=12,
    vocos_d_model=512,
    vocos_layers=8,
    vocos_intermediate_dim=1536,
)

CI_MICRO_CONFIG = ModelArchitectureConfig(
    profile_name="ci_micro",
    separator_d_model=64,
    separator_layers=1,
    mode_gate_d_model=64,
    mode_gate_layers=2,
    ear_d_model=128,
    ear_layers=2,
    ear_heads=4,
    ear_ffn_dim=256,
    emotion_d_model=128,
    emotion_layers=1,
    emotion_ffn_dim=256,
    pointer_d_model=128,
    pointer_layers=1,
    pointer_ffn_dim=256,
    mouth_d_model=128,
    mouth_layers=2,
    mouth_heads=4,
    mouth_ffn_dim=256,
    euler_steps=2,
    vocos_d_model=128,
    vocos_layers=2,
    vocos_intermediate_dim=256,
)


# ============================================================================
# 3. FROZEN DATACLASSES & TENSOR SCHEMAS
# ============================================================================
@dataclass(frozen=True)
class KhmerSyllableToken:
    surface: str
    onset_id: int
    coeng_id: int
    medial_id: int
    register_id: int
    vowel_id: int
    coda_id: int
    sesqui_id: int


@dataclass(frozen=True)
class SyllableTensorBatch:
    onset_ids: torch.Tensor
    coeng_ids: torch.Tensor
    medial_ids: torch.Tensor
    register_ids: torch.Tensor
    vowel_ids: torch.Tensor
    coda_ids: torch.Tensor
    sesqui_ids: torch.Tensor


@dataclass
class CartridgeThreadState:
    """
    v10.7 P0-A: thread-local carrier for the per-call cartridge telemetry that
    v10.6 stored as mutable module attributes (last_l6_hidden, last_l8_hidden,
    last_bus_out, last_next_past_kv, last_next_past_gate,
    last_deferred_syll_count,
    last_pred_durations, last_gate_l).

    The GPU worker runs in asyncio.to_thread while the event loop thread owns
    the control plane, so a shared module attribute could be observed
    half-updated by the other thread. Each field is therefore stored in a
    threading.local() slot, and the owning module exposes property bridges
    under the original attribute names so every existing call site keeps
    working unchanged.

    This class holds NO parameters and no trainable state.
    """

    def __post_init__(self) -> None:
        import threading

        self._tls = threading.local()

    def _get(self, name: str):
        return getattr(self._tls, name, None)

    def _set(self, name: str, value) -> None:
        setattr(self._tls, name, value)

    @property
    def l6_hidden(self):
        return self._get("l6_hidden")

    @l6_hidden.setter
    def l6_hidden(self, value) -> None:
        self._set("l6_hidden", value)

    @property
    def l8_hidden(self):
        return self._get("l8_hidden")

    @l8_hidden.setter
    def l8_hidden(self, value) -> None:
        self._set("l8_hidden", value)

    @property
    def last_bus_out(self):
        """v10.7 P2-B: the bus-projected acoustic output of the ear.

        The multi-teacher distiller needs the student bus representation to
        compare against the teacher features, so it is published here.
        """
        return self._get("last_bus_out")

    @last_bus_out.setter
    def last_bus_out(self, value) -> None:
        self._set("last_bus_out", value)

    @property
    def last_next_past_kv(self):
        return self._get("last_next_past_kv")

    @last_next_past_kv.setter
    def last_next_past_kv(self, value) -> None:
        self._set("last_next_past_kv", value)

    @property
    def last_next_past_gate(self):
        return self._get("last_next_past_gate")

    @last_next_past_gate.setter
    def last_next_past_gate(self, value) -> None:
        self._set("last_next_past_gate", value)

    @property
    def last_deferred_syll_count(self):
        # v10.7: the deferred syllable count defaults to 0 when it has not
        # been set yet, so a reader never sees None.
        value = self._get("last_deferred_syll_count")
        return 0 if value is None else value

    @last_deferred_syll_count.setter
    def last_deferred_syll_count(self, value) -> None:
        self._set("last_deferred_syll_count", value)

    @property
    def last_pred_durations(self):
        return self._get("last_pred_durations")

    @last_pred_durations.setter
    def last_pred_durations(self, value) -> None:
        self._set("last_pred_durations", value)

    @property
    def last_gate_l(self):
        return self._get("last_gate_l")

    @last_gate_l.setter
    def last_gate_l(self, value) -> None:
        self._set("last_gate_l", value)


@dataclass(frozen=True)
class SeparatedSceneSlice:
    utterance_id: int
    utterance_start_ms: float
    frame_vocal_modes: torch.Tensor
    dominant_vocal_modes: List[VocalMode]
    active_speaker_stems: torch.Tensor
    speaker_ids: List[int]
    mic_eq_residual_16k: torch.Tensor
    rt60_sec: torch.Tensor
    wet_gain: torch.Tensor
    rms_scalar: torch.Tensor
    erb_suppression_mask: torch.Tensor
    trailing_pause_ms: float


@dataclass(frozen=True)
class LivePacket:
    kind: Literal["realtime_audio", "content"]
    timestamp_ms: float
    pcm_data: Optional[np.ndarray] = None
    payload: Optional[Dict[str, Any]] = None


@dataclass(frozen=True)
class DownstreamEvent:
    event_type: Literal[
        "audio_chunk", "passthrough_vocal", "singing_lyrics", "caption", "interrupted"
    ]
    utterance_id: int
    utterance_start_ms: float = 0.0
    vocal_modes: List[int] = field(default_factory=lambda: [3])
    speaker_ids: List[int] = field(default_factory=lambda: [0])
    source_lang: LanguageCode = "en"
    target_lang: LanguageCode = "km"
    dubbed_audio_b64: Optional[str] = None
    erb_suppression_mask_b64: Optional[str] = None
    erb_frame_count: int = 0
    per_speaker_source_texts: List[str] = field(default_factory=list)
    per_speaker_target_texts: List[str] = field(default_factory=list)
    codeswitch_words: List[str] = field(default_factory=list)
    playback_rate: float = 1.0
    actual_duration_sec: float = 0.0
    drift_ms: float = 0.0
    interrupt_epoch: int = 0


# ============================================================================
# 4. RUNTIME-CHECKABLE MULTI-AGENT PROTOCOLS
# ============================================================================
@runtime_checkable
class CodeSwitchDecomposerProtocol(Protocol):
    def normalize_khmer(self, text: str) -> str: ...
    def decompose_mixed_sentence(self, text: str) -> List[KhmerSyllableToken]: ...
    def encode_sentence_to_batch(
        self, text: str, device: str = "cpu"
    ) -> SyllableTensorBatch: ...
    def decode_batch_to_text(
        self, batch: SyllableTensorBatch
    ) -> Tuple[List[str], List[str]]: ...


@runtime_checkable
class StemAndSceneProtocol(Protocol):
    def separate_and_classify(
        self,
        wav_16k: torch.Tensor,
        utterance_id: int = 0,
        utterance_start_ms: float = 0.0,
        trailing_pause_ms: float = 0.0,
        primary_spk_emb: Optional[torch.Tensor] = None,
        force_dual_speakers: bool = False,
    ) -> SeparatedSceneSlice: ...

    def apply_scene_glue(
        self,
        dry_khmer_wav_24k: torch.Tensor,
        mic_eq_residual_16k: torch.Tensor,
        rt60_sec: torch.Tensor,
        wet_gain: torch.Tensor,
        target_rms_scalar: torch.Tensor,
    ) -> torch.Tensor: ...


@dataclass(frozen=True)
class PrecomputedEarFeatures:
    """
    v10.7 AUD-107-02: the results of ONE StreamingEarCartridge forward pass.

    The full-duplex runner needs the layer-6 acoustic tap (for the speaker
    registry) and the log-mel (for the energy weighting) BEFORE it can call
    forward_s2st. v10.7 ran the ear twice and the log-mel three times on the
    same audio slice, doubling the 64.5M-parameter ear and the 8.7M-parameter
    emotion encoder on every spoken utterance and threatening the
    T_GPU <= 90 ms latency law.

    The runner computes this bundle ONCE and passes it to forward_s2st, which
    reuses it instead of re-running the ear. Adds NO parameters.
    """

    bus: torch.Tensor
    r_sps: torch.Tensor
    mel_stems: torch.Tensor
    h_acoustic_l6: torch.Tensor


@runtime_checkable
class S2STConsoleProtocol(Protocol):
    def forward_s2st(
        self,
        source_wav_16k: torch.Tensor,
        source_lang: LanguageCode,
        target_lang: LanguageCode,
        orig_duration_sec: float,
        trailing_pause_ms: float = 0.0,
        playback_rate: float = 1.0,
        explicit_syll_batch: Optional[SyllableTensorBatch] = None,
        precomputed_ear: Optional[PrecomputedEarFeatures] = None,
    ) -> Tuple[torch.Tensor, SyllableTensorBatch, torch.Tensor]: ...


    def compute_flow_loss(
        self,
        target_wav_24k: torch.Tensor,
        syll_batch: Optional[SyllableTensorBatch] = None,
        infill_mode: bool = False,
        source_wav_16k: Optional[torch.Tensor] = None,
        source_lang: LanguageCode = "en",
    ) -> torch.Tensor: ...
```

#### 2. `frontend/src/contracts/types.ts`
```typescript
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
```

---

### 2.5 Locked Submodule & Helper Method Signatures (Required by Gates 1–6)

| Class / Function | Module Path | Locked Method Signature | Return Contract |
| :--- | :--- | :--- | :--- |
| **`build_slaney_mel_fb`** | `backend/nn_cartridges/universal_core.py` | `build_slaney_mel_fb(n_bins: int, n_mels: int = 100, sample_rate: int = 16000, f_min: float = 0.0, f_max: float \| None = None)` | `torch.Tensor` `(n_bins, n_mels)`, column rank = 100 on CPU |
| **`filter_dominant_speakers`** | `backend/nn_cartridges/stem_and_mode_gate.py` | `filter_dominant_speakers(stem_a: torch.Tensor, stem_b: torch.Tensor, min_db_ratio: float = DOMINANCE_MIN_DB_RATIO, force_dual_speakers: bool = False)` | `torch.Tensor` `(2, T)` if dual active, else `(1, T)` |
| **`SceneAcousticMatcher`** | `backend/nn_cartridges/stem_and_mode_gate.py` | `__init__(self) -> None`<br>`apply_scene_glue(self, dry_khmer_wav_24k: torch.Tensor, mic_eq_residual_16k: torch.Tensor, rt60_sec: torch.Tensor, wet_gain: torch.Tensor, target_rms_scalar: torch.Tensor)` | `torch.Tensor` `(B_spk, T_samples)` |
| **`LearnedDurationAligner`** | `backend/nn_cartridges/duration_aligner.py` | `__init__(self, d_model: int = 512, sigma: float = 1.5) -> None`<br>`@staticmethod compute_elastic_target_frames(orig_duration_sec: float, trailing_pause_ms: float, speech_rate_sps: float, playback_rate: float = 1.0) -> int`<br>`forward(self, syll_hidden: torch.Tensor, sesqui_ids: torch.Tensor, target_frames: int)` | `Tuple[torch.Tensor, torch.Tensor]` *`(aligned_hidden (B, target_frames, D), final_durations (B, L))`* |
| **`AcousticPointerCodeSwitchBridge`** | `backend/nn_cartridges/codeswitch_router.py` | `__init__(self, config: ModelArchitectureConfig = PROD_240M_CONFIG) -> None`<br>`forward(self, semantic_bus: torch.Tensor, source_acoustic_features: torch.Tensor, force_gate: Optional[float] = None)` | `Tuple[torch.Tensor, torch.Tensor]` *`(fused_features (B, T, 512), gate (B, T, 1))`* |
| **`GradientReversalFn`** | `backend/nn_cartridges/universal_core.py` | `GradientReversalFn.apply(x: torch.Tensor, alpha: float)` | `torch.Tensor` *(identity forward; `-alpha * grad` backward)* |
| **`DynamicAcousticCorruptor`** | `backend/training/augmentations.py` | `@staticmethod warp_formants_vtlp(mel: torch.Tensor, alpha_min: float = 0.85, alpha_max: float = 1.15, pivot: float = 0.65) -> torch.Tensor`<br>`@staticmethod corrupt_youtube_mixture(clean_vocal_a: torch.Tensor, clean_vocal_b: Optional[torch.Tensor] = None, bgm_or_noise: Optional[torch.Tensor] = None) -> torch.Tensor` | `torch.Tensor` `(B, M, T)`<br>`torch.Tensor` `(B, T)` |
| **`InMemorySession`** | `backend/transport/session.py` | `__init__(self, session_id: str) -> None`<br>`append_pcm_chunk(self, chunk: np.ndarray, timestamp_ms: float = -1.0)`<br>`reset_on_seek(self, new_timestamp_ms: float) -> None`<br>`cache_dubbed_event(self, event: DownstreamEvent, duration_ms: float = 0.0, pcm_crc32: int = 0) -> None`<br>`lookup_cached_event(self, utterance_start_ms: float, duration_ms: float = 0.0, pcm_crc32: int = 0, playback_rate: float = 1.0, target_lang: LanguageCode = "km") -> Optional[DownstreamEvent]` | `Optional[Tuple[int, float, np.ndarray, float]]` *`(uid, start_ms, slice_pcm, pause_ms)`* |
| **`SpeakerRegistry`** | `backend/transport/session.py` | `__init__(self, max_speakers: int = 4, similarity_threshold: float = 0.72) -> None`<br>`match_or_register_speaker(self, emb: torch.Tensor)` | `Tuple[int, torch.Tensor]` *`(slot_id, primary_centroid_E0)`* |
| **`LiveRequestQueue`** | `backend/transport/live_queue.py` | `__init__(self) -> None` (exposes `.interrupt_epoch: int`)<br>`async send_realtime(self, pcm_chunk: np.ndarray, timestamp_ms: float) -> None`<br>`async send_content(self, action: str, timestamp_ms: float, playback_rate: float = 1.0, payload: Optional[Dict[str, Any]] = None) -> None`<br>`flush_all(self) -> int`<br>`async get_next(self) -> LivePacket` | `LivePacket` |
| **`encode_wav_pcm16_b64`** | `backend/transport/runner.py` | `encode_wav_pcm16_b64(wav_24k: torch.Tensor \| np.ndarray, sample_rate: int = SAMPLE_RATE_OUT) -> str` | Base64 WAV string (`raw[:4] == b"RIFF"`, `raw[8:12] == b"WAVE"`) |

---

## PART 3: RUST CRATE & PYTORCH CARTRIDGE SPECIFICATIONS

### 3.1 Specialist B: `crates/khmer-linguistics-rs` & `backend/linguistics/`

1. **Crate Configuration (`crates/khmer-linguistics-rs/Cargo.toml`):** Must declare `[lib] crate-type = ["cdylib", "rlib"]` and `[features] extension-module = ["pyo3/extension-module"]` so both `cargo test -p khmer-linguistics-rs` and Python `maturin` builds succeed.
2. **Canonical Cluster Reordering:** Normalize input to NFC, strip zero-width format chars (`\u{200C}`, `\u{200D}`, `\u{FEFF}`) while preserving `\u{200B}` as a primary word delimiter, and reorder any misplaced register shifter (`\u{17C9}`, `\u{17CA}`) or dependent vowel (`\u{17B6}`..`\u{17C5}`) that precedes subscript `Coeng` sequences (`\u{17D2}[\u{1780}-\u{17A2}]`) into: **Base Consonant $\rightarrow$ Subscript Coeng(s) $\rightarrow$ Register Shifter $\rightarrow$ Dependent Vowel $\rightarrow$ Diacritics / Coda**.
3. **Phonological Syllable Lexicon Priority (`PHONOLOGICAL_SYLLABLE_LEXICON`):**
   Scan contiguous Khmer spans using longest-prefix matching before statistical/rule splitting:
   * `"គណនា"` $\rightarrow$ `[("គ", 1, 2), ("ណ", 1, 1), ("នា", 2, 2)]`
   * `"សុភមង្គល"` $\rightarrow$ `[("សុ", 1, 1), ("ភ", 1, 2), ("មង", 2, 2), ("គល", 2, 2)]`
   * `"ធម្មជាតិ"` $\rightarrow$ `[("ធម", 2, 2), ("ម", 1, 2), ("ជាតិ", 2, 2)]`
   * `"សង្គ្រាម"` $\rightarrow$ `[("សង", 2, 1), ("គ្រាម", 2, 2)]`
   * `"កញ្ជ្រោង"` $\rightarrow$ `[("កញ", 2, 1), ("ជ្រោង", 2, 2)]`
   * `"ប្រទេស"` $\rightarrow$ `[("ប្រ", 1, 1), ("ទេស", 2, 2)]`
   * `"កម្ពុជា"` $\rightarrow$ `[("កម", 2, 1), ("ពុ", 1, 2), ("ជា", 2, 2)]`
   * `"បញ្ហា"` $\rightarrow$ `[("បញ", 2, 1), ("ហា", 2, 1)]`
   * `"កន្លែង"` $\rightarrow$ `[("កន", 2, 1), ("លែង", 2, 1)]`
4. **General 3-Rule Orthographic Resyllabification, `khmer-nltk` & Bantoc Encoding Law:**
   * Split on `\u200b` first. If sub-span is unspaced and not in lexicon, call `khmernltk.word_tokenize` before `decompose_khmer_word`.
   * **Step A (`"ស្ត្រី"` Rule):** Word-initial 3-consonant cluster `C1 + \u17D2 + C2 + \u17D2 + C3` with no preceding vowel stays a single syllable (`surface="ស្ត្រី"`, `coeng_id > 0`, `medial_id = 1`, `register_id = 1`, `sesqui_id = 2`).
   * **Step B (Medial Double-Coeng Split):** Split between `coda` and `c2\u17D2c3`.
   * **Step C (Medial Single-Coeng Disambiguation & Sonorant Register Inheritance):**
     * **Case 2a (`"កក្រើក"` $\rightarrow$ `["ក", "ក្រើក"]`):** When `C1` has no dependent vowel, `C2 in {'ក','គ','ច','ជ','ត','ទ','ប','ព','ស'}`, and `C3 in {'រ','ល','វ','យ'}`, split *before* `C2` so `C1` is an open minor presyllable (`sesqui_id = 1`, `coda_id = 0`) and `C2\u17D2C3` starts the major syllable (`medial_id = 1`).
     * **Case 2b (Nasal/Final Coda Split & Assimilation):** When `C2 in {'ង','ញ','ន','ម','យ','រ','ល','វ'}` or `C3 not in {'រ','ល','វ','យ'}`, `C2` closes the first syllable (`sesqui_id = 2`, `coda_id > 0`). If `C1` is A-series (`register_id == 1`), `C2 in {'ង','ញ','ន','ម'}`, and `C3 in {'ង','ញ','ម','យ','រ','ល','វ'}`, lock `forced_register = 1` on the `C3` syllable. For `"កន្លែង"` $\rightarrow$ `["កន", "លែង"]`, the sonorant liquid `ល` explicitly inherits A-Series register (`register_id = 1`) from base consonant `ក`.
   * **Step D (Negative-Lookahead Coda Guard for `"ប្រជា"`, `"ក្រសួង"`):** Never attach a trailing consonant as a coda if it is followed by `[\u17B6-\u17CA\u17D2]`.
   * **Bantoc (`់`) Offset Law:** Encode Bantoc (`់`, `\u17CB`) in `_extract_coda_id` such that plain codas map to `[1..8]` and Bantoc-marked codas map strictly to `[9..16]` (8 IDs matching plain codas `[1..8]` 1:1, leaving `[17..31]` for foreign/loanword codas).

---

### 3.2 Specialist D: `crates/khmer-transport-rs`, `crates/khmer-dsp-wasm` & Rewind-Resilience Laws

1. **Rewind & Seek Resilience Law #1 — `InMemorySession.reset_on_seek(new_timestamp_ms)`:**
   * Clear uncommitted buffer: `self.pcm_buffer = np.zeros(0, dtype=np.float32)`.
   * Reset VAD timers: `self.is_speaking = False`, `self.silence_ms = 0.0`, `self.active_speech_ms = 0.0`.
   * Re-anchor cursor: `self.stream_cursor_ms = float(new_timestamp_ms)`, `self.current_utterance_start_ms = float(new_timestamp_ms)`.
   * **Preserve `self.speaker_registry.centroids`** and **preserve `self._replay_cache`**.
2. **Rewind & Seek Resilience Law #2 — 32-Entry Phase-Invariant Replay Cache (`cache_dubbed_event` & `lookup_cached_event`):**
   * Key computation: `(int(round(start_ms / 250.0)), int(round(dur_ms / 50.0)), pcm_env_hash, round(rate, 2), lang)`.
   * Phase-invariant coarse energy envelope hash: 8-bin normalized log-RMS profile across the clause quantized to integers `compute_pcm_energy_hash(slice_pcm)`, guaranteeing 100% cache hit rate across sub-millisecond 1–63 sample WebAudio scrubbing phase shifts.
3. **Rewind & Seek Resilience Law #3 — In-Flight GPU Epoch Fencing (`FullDuplexRunner`):**
   * Snapshot `epoch_before = queue.interrupt_epoch` prior to `separate_and_classify` or `forward_s2st`. If `queue.interrupt_epoch != epoch_before` post-synthesis, discard output immediately.
4. **Rewind & Seek Resilience Law #4 — Frontend `StereoMneMixer.reanchorTimeline` & Epoch Fencing:**
   * Stop active nodes, clear `this.erbQueue`, reset `this.nextPlayTime = this.ctx.currentTime`, re-anchor `this.sessionStartCtxTime`, and discard decoded audio if `this.clientEpoch` changed while awaiting `ctx.decodeAudioData`.

---

### 3.3 Specialist A & C: Authoritative `241.637M` PyTorch Parameter Ledger (`PROD_240M_CONFIG` vs. `CI_MICRO_CONFIG`)

| Module & File | Exact Layer Blueprint (`prod_240m`) | `prod_240m` Params | `ci_micro` Params |
| :--- | :--- | :--- | :--- |
| **1. `Causal3StemSeparator`** (`stem_and_mode_gate.py`) | • `stft_in`: `Linear(257, d)` (`99K`)<br>• `film_proj`: `Linear(512, 2 * d)` (`394K`)<br>• `gru`: `nn.GRU(d, d, num_layers=layers, batch_first=True)` (`2.66M`)<br>• `mlp`: 2 blocks of `Linear(d, 4*d) -> GELU -> Linear(4*d, d)` (`2.36M`)<br>• `mask_head`: `Linear(d, 3 * 257)` (`297K`)<br>• `erb_head`: `Linear(d, 64)` (`25K`)<br>• `eq_head`: `Linear(d, 64)`, `rt60_head`: `Linear(d, 1)`, `wet_head`: `Linear(d, 1)` (`25K`) | **5.864M** (`5,864,197`) | **0.232M** (`231,877`) |
| **2. `VocalModeGate`** (`stem_and_mode_gate.py`) | • `in_proj`: `Conv1d(100, d, kernel_size=1)` (`26K`)<br>• `conv_layers`: `layers` × (`Conv1d(d, d, kernel_size=3, padding=1)` + `GroupNorm(8, d)`) (`788K`)<br>• `cls_head`: `Linear(d, 4)` (`1K`) | **0.816M** (`816,388`) | **0.032M** (`31,684`) |
| **3. `StreamingEarCartridge`** (`ear_cartridge.py`) | • `subsampler` (`d=512`): `Conv1d(100, 512, k=3, s=2, p=1)` + `Conv1d(512, 512, k=3, s=2, p=1)` (`941K`)<br>• `lang_emb`: `Embedding(4, 512)` (`2K`)<br>• `blocks`: `16` FastConformer blocks (single FFN `Linear(d, ffn_dim) -> GELU -> Linear(ffn_dim, d)` = `3.95M/block` → `63.26M`)<br>• `bus_proj`: `Linear(512, 512)` (`263K`) + `rate_head`: `Linear(512, 1)` (`0.5K`) | **64.547M** (`64,546,817`) | **0.528M** (`528,129`) |
| **4. `DisentangledEmotionEncoder`** (`universal_core.py`) | • `in_proj`: `Linear(100, d)` (`52K`) + `latents`: `Parameter(1, 32, d)` (`16K`)<br>• `blocks`: `4` Perceiver blocks (`MultiheadAttention(d, 8)` + `Linear(d, ffn_dim // 2) -> GELU -> Linear(ffn_dim // 2, d)` = `8.40M`)<br>• `out_proj`: `Linear(d, 512)` (`263K`) + `lang_cls`: `Linear(512, 4)` (`2K`) | **8.736M** (`8,735,748`) | **0.184M** (`184,196`) |
| **5. `AcousticPointerCodeSwitchBridge`** (`codeswitch_router.py`) | • `in_q`, `in_kv`: `Linear(512, d)` × 2 (`525K`)<br>• `blocks`: `2` blocks of `MultiheadAttention(d, 8)` + `Linear(d, ffn_dim) -> GELU -> Linear(ffn_dim, d)` (`4.20M`)<br>• `out_proj`: `Linear(d, 512)` (`263K`) + `gate_head`: `Linear(512, 1)` (`0.5K`) | **4.990M** (`4,989,953`) | **0.330M** (`329,857`) |
| **6. `KhmerMouthCartridge`** (`mouth_cartridge.py` + `duration_aligner.py`) | • `reorder_attn`: `MultiheadAttention(512, 8)` (`1.05M`) + `reorder_norm`: `LayerNorm(512)` (`1K`)<br>• 7-slot heads & embeddings (`0.23M`) + `target_lang_emb`: `Embedding(4, 512)` (`2K`)<br>• `LearnedDurationAligner` (`512 -> 256 -> 1`) (`0.13M`)<br>• `cond_proj`, `emo_proj`, `mel_in`, `infill_proj` (`0.94M`) + `time_mlp` (`1.18M`) + shared `adaLN_modulation` (`3.54M`)<br>• `blocks`: `15` DiT blocks (`scale_shift_table` + `self_attn` + `cross_attn` + `ffn` = `141.78M`) + `mel_out` (`77K`) | **148.931M** (`148,931,136`) | **2.110M** (`2,109,504`) |
| **7. `VocosISTFTHead`** (`universal_core.py`) | • `in_conv`: `Conv1d(100, d, k=7, p=3)` (`359K`)<br>• `blocks`: `8` ConvNeXt blocks (`dwconv k=7` + `LayerNorm(d)` + `Linear(d, intermediate_dim // 2) -> GELU -> Linear(intermediate_dim // 2, d)` = `6.34M`)<br>• `istft_head`: `Linear(d, 2050)` (`1.05M`) | **7.753M** (`7,753,218`) | **0.423M** (`422,786`) |
| **COMBINED ENGINE TOTAL** | **`StemAndScenePipeline` + `SymmetricS2STConsole`** | **241.637M** (`241,637,457`) *(Strictly inside `[230M, 250M]`)* | **3.838M** (`3,838,033`) *(Runs in `< 2s`)* |

> [!IMPORTANT]
> **v10.6 ledger correction.** Every figure in this table was re-derived by
> instantiating the Part 4 reference modules under `torch.device("meta")` and
> summing `p.numel()`, exactly as Gate 2 does. The v10.5 table claimed
> `241.55M` / `2.40M`; the measured values are **`241,637,457`** and
> **`3,838,033`**. The prod figure is a precision defect (87,457 too low in
> the label); the CI figure was a **60% understatement**, because the CI
> column had never been computed against `CI_MICRO_CONFIG`.
>
> **Root cause of the CI gap.** `KhmerMouthCartridge` contains 1,409,756
> parameters that are pinned to `SEMANTIC_DIM = 512` and therefore do **not**
> scale with `mouth_d_model` (which is `128` under `CI_MICRO_CONFIG`):
> `reorder_attn`, `reorder_norm`, the seven slot `heads`, the seven slot
> `embeddings`, `target_lang_emb`, and `LearnedDurationAligner`. That is
> **36.7%** of the entire CI budget. This is a **documented property of the
> current design**, not a defect to be silently patched: the v10.6 revision
> publishes the true number and states the cause, so that any future
> optimisation to shrink the CI profile is an explicit, measured decision
> rather than an accident.
>
> **Validation.** Re-derive this table on every revision (Validation **V1**).
> Gate 2's `[230M, 250M]` assertion remains satisfied by the corrected prod
> value, so no gate changes as a result of this correction.

---
