# Implementation Plan

## Overview

Apply the final four post-audit remediations - **AUD-107-09** (mask deferred `sesqui_ids == 0` tokens out of the Gaussian attention in `LearnedDurationAligner`), **AUD-107-10** (universal cross-device sample-rate support in the PCM capture worklet), **AUD-107-08** (synchronize the Part 8 master-orchestration prompt with the AUD-107-01..09 code changes), and the provenance-ledger extension to entries 68-70 - to the v10.7 blueprint, the `docs/` volumes, and the materialized workspace, with the parameter ledger held **exactly** at `241,637,457` (`prod_240m`) and `3,838,033` (`ci_micro`).

This is a **blueprint-revision task**. The authoritative artifact remains `KHMER-SOVEREIGN-S2ST-BLUEPRINT-v10.7.txt`, which is regenerated in place from the v10.6 source by `__gen_v107.py`; the `docs/` volumes are then re-partitioned from that master, and `__v107_materialize.py` re-materialises `backend/`, `frontend/src/` and the six gates from the master's code blocks. **No new master filename is created and no toolchain script changes its hard-coded filename** (confirmed with the user: the existing v10.7 master is regenerated in place, the ledger grows to 1..70, and the project version stays `10.7.0`).

**Every remediation is parameter-free.** The ledger is the blueprint's most load-bearing claim, so V1 re-derives it from the materialised modules and must remain green byte-for-byte. None of the three code changes adds, removes, or reshapes a single `nn.Parameter`.

### Audit method (independently reproduced, not accepted on assertion)

Each finding was confirmed against the actual files, and every load-bearing numeric claim in the remediation specification was re-derived **by execution** before being planned:

| Claim | Method | Result |
| :--- | :--- | :--- |
| AUD-107-09: deferred tokens steal attention mass | recomputed the aligner's `attn_logits`/`softmax` with `sesqui_ids = [[2,2,0,0]]`, `sigma = 1.5`, `target_frames = 24` | **CONFIRMED** - the two deferred columns collect **1.628554** attention mass each, and the real syllables are correspondingly squeezed to 12.000000 / 8.742891 instead of 12.000000 / 12.000000 |
| AUD-107-09: the proposed `-1e9` mask works | applied the spec's `active_mask` / `has_active` mask to the same logits | **CONFIRMED** - deferred mass drops to **0.0** and the real syllables recover 12.000000 / 12.000000 |
| AUD-107-09: the proposed test form | ran the spec's literal test (independent `randn` tensors) against a fixed aligner | **FAILS for the wrong reason** - the two calls use different random hidden states, so `allclose` is False even with a correct implementation (max diff 3.67) |
| AUD-107-09: the corrected test form | shared the first two token rows between the 4-token and 2-token calls | **PASSES** - durations identical `[13.4532, 10.5468]` and `allclose` True with max diff **3.9e-06** |
| AUD-107-10: the resampler produces 16 kHz | simulated the spec's `fracPos` loop with 128-frame quanta at eleven context rates | **CONFIRMED for every rate >= 16 kHz** - output is exactly 16,000 samples/s at 16 k, 22.05 k, 24 k, 32 k, 44.1 k, 48 k, 88.2 k, 96 k and 192 k |
| AUD-107-10: no pitch drift | Goertzel frequency estimate on the resampled output | **CONFIRMED** - a 1,000 Hz input peaks at exactly **1000.0 Hz** at 44.1 k, 96 k, 24 k and 32 k context rates (amplitude 0.635, the expected linear-interpolation droop) |
| AUD-107-10: rates below 16 kHz | same simulation at 8 k and 11.025 k | **NOT SUPPORTED** - the output is 8,064 and 11,136 samples/s because `step = ctxRate / 16000` is below 1 and the `fracPos` accumulator cannot upsample. Documented as a limitation, not fixed. |
| AUD-107-10: the current 3:1 decimator is 48 kHz-only | read `pcm-capture-processor.ts` | **CONFIRMED** - the worklet hard-codes a 3:1 decimation with no reference to the context rate, so a 44.1 kHz device emits 14,700 Hz |
| AUD-107-08: Specialist D item 9 is stale | read `stage02.txt` line 168-174 | **CONFIRMED** - it still mandates resolving the ERB mask at `lookaheadTargetTime`, which the 2.00 s DelayNode (AUD-107-03) made wrong, and it is silent on the FFT tables, the DelayNode, the capture sink and the single-ear-pass requirement |
| AUD-107-10: `SAMPLE_RATE_IN` is importable from the worklet | read `frontend/src/contracts/types.ts` and the worklet import path | **CONFIRMED** - `export const SAMPLE_RATE_IN = 16000;` exists and `../contracts/types` is the correct relative path from `frontend/src/worklets/` |
| The `outBuf` must grow from 64 to 128 | counted outputs produced per 128-frame quantum at each context rate | **CONFIRMED** - the fallback needs up to **128** outputs per quantum at a 16 kHz context rate; the current 64-slot buffer truncates and yields 8,000 Hz |

### Two defects in the incoming specification, found by execution and corrected in this plan

1. **The AUD-107-09 test as literally specified cannot pass.** It compares a 4-token call against a 2-token call using **independently drawn** random hidden states, so the two calls differ regardless of the attention mask (measured max diff 3.67 even against a correct implementation). The plan specifies the corrected form: the 2-token reference **reuses the first two rows of the same hidden tensor**. This is the same class of mistake caught earlier in this project, where a supplied test case did not discriminate the defect it was meant to catch.
2. **The AUD-107-10 buffer size is stated but not carried through.** The spec's replacement class raises `outBuf` from 64 to 128, which is necessary, but the plan also makes the worklet **drop nothing**: the plan records that a context rate below 16 kHz is not supported by the fractional accumulator, and the `has_active` guard in AUD-107-09 likewise records the all-deferred edge case.

### Scope boundaries

The Rust crates, the parameter ledger, the six-gate structure, the gate file names, the Part 7 ownership boundaries, the ERB band partition, the 48 kHz frontend rate, and the 1.94 s turnaround law are all **unchanged**. The `docs/05` audit ledger is extended with the three new findings. The `docs/04` ledger is untouched. No dependency is added, removed, or version-bumped. The project version stays `10.7.0` in all five declared locations.

## Types

**No new types are introduced anywhere.** All three code remediations are parameter-free and type-compatible: the existing dataclasses, Protocols, and enums keep their current shapes, so the six gates continue to type-check and the `S2STConsoleProtocol` runtime check remains valid.

### `backend/contracts/schemas.py` (Part 2.4) - unchanged

- `SyllableTensorBatch` - **unchanged.** `sesqui_ids: torch.Tensor` keeps its shape and dtype. The AUD-107-09 fix reads `sesqui_ids == 0` as a sentinel; it does not redefine the field.
- `S2STConsoleProtocol` - **unchanged.** The AUD-107-02 `precomputed_ear: Optional[PrecomputedEarFeatures] = None` keyword was added in the previous pass and is untouched.
- `PrecomputedEarFeatures` - **unchanged.** Already present from the previous pass.

### `frontend/src/contracts/types.ts` (Part 2.4, section 2) - unchanged

- `SAMPLE_RATE_IN` - **unchanged at `16000`.** The worklet now *imports* this constant rather than hard-coding a 3:1 ratio, but the constant's value and its role as the single source of truth for the 16 kHz contract are unchanged.
- `SAMPLE_RATE_MNE` (`48000`) - **unchanged.** The 48 kHz fast path remains gated on the *actual* `AudioWorkletGlobalScope.sampleRate`, not on this constant.

### New sentinel semantics (not a new type)

`sesqui_ids == 0` becomes the **zero-frame, zero-attention sentinel** for a deferred syllable. This is a semantic clarification of an existing value, not a new field:

| Value | Meaning | Duration prior | Attention column |
| :--- | :--- | :--- | :--- |
| `0` | **deferred** - wait-gated at a Trigger #2 boundary, not spoken in this clause | `0.0` | masked to `-1e9` (no mass) |
| `1` | minor (sesqui) syllable | `0.25` | normal |
| `2` | major (full) syllable | `1.00` | normal |

The sentinel is **unambiguous**: `sesqui_ids` is produced by `1 + argmax` over logits `[1:3]`, so the model can only emit `1` or `2`; `0` is written **only** by the AUD-107-01 deferred-tail mask in `KhmerMouthCartridge.predict_syllables_and_align`. Verified: the only assignments to `sesqui_ids` in the whole codebase are `torch.full(..., 2)`, `1 + argmax`, `torch.where(is_cs, 2, ...)`, and the deferred-tail `torch.cat` mask.

### `frontend/src/worklets/pcm-capture-processor.ts` - no new exported types

`PcmCaptureProcessor` keeps its public surface. The change adds two **private** instance fields:

| Field | Type | Purpose |
| :--- | :--- | :--- |
| `fracPos` | `number` | Fractional read position for the resampler fallback. Initialized `0.0`. |
| `prevSample` | `number` | The previous mono input sample, for linear interpolation. Initialized `0.0`. |

A module-scope `declare const sampleRate: number;` is added next to the existing `declare const currentTime: number;`, matching the `AudioWorkletGlobalScope` convention already used in the file.

## Files

### New files

**None.** All four remediations are applied to existing artifacts. No new blueprint file, no new test module, no new block file.

### Modified files

| Path | Change |
| :--- | :--- |
| `__v107_blocks/pcm_capture_processor.ts` | **AUD-107-10 (universal cross-device sample rate).** Replace the file body with the spec's implementation: add `declare const sampleRate: number;` next to the existing `declare const currentTime: number;`; add the `import { SAMPLE_RATE_IN } from "../contracts/types";` (keeping the existing side-effect `import "../contracts/types";`); add the private fields `fracPos = 0.0` and `prevSample = 0.0`; raise `outBuf` from `new Float32Array(64)` to `new Float32Array(128)`; and split `process()` into the 48 kHz fast path and the fractional-resampler fallback exactly as the spec writes them. The 48 kHz branch keeps the 15-tap Blackman-sinc 3:1 decimator verbatim. |
| `__v107_blocks/stage02.txt` | **AUD-107-08 (Part 8 synchronization).** Under Specialist A, add item 11 (AUD-107-01 + AUD-107-09) and item 12 (AUD-107-02). Under Specialist C, add item 5 (AUD-107-04). Under Specialist D, **replace item 9** (the stale `lookaheadTargetTime` ERB instruction) with the four AUD-107-03 / AUD-107-05 / AUD-107-06 / AUD-107-02 requirements exactly as the spec states them. Every existing item is retained verbatim. |
| `__v107_blocks/fix_ledger.txt` | **Provenance ledger entries 68, 69, 70.** Append the three new entries (AUD-107-08, AUD-107-09, AUD-107-10), each with its measured evidence, its root cause, and the exact remediation. The file currently holds entries **37..67** (31 entries); after the append it holds **37..70** (34 entries). Entries **1..36** come from the v10.6 source, so the published master carries a contiguous **1..70** ledger with no duplicate homes. |
| `__v107_blocks/docs05_part3.txt` | Extend the second-pass findings section with the three new findings and add the new validation names to the "New validations" list, so `docs/05` does not lag the master. |
| `__gen_v107.py` | **Three new asserted edits**, each aborting on an occurrence-count mismatch exactly like the existing `rep()` / `blockfile()` helpers: (a) the `duration_aligner.py` attention-mask replacement, (b) the `pcm-capture-processor.ts` block replacement, (c) the `stage02.txt` text replacement. The `fix_ledger.txt` is already read and inserted by the existing 9.2 step, so the new entries flow through with no new call. |
| `__v107_verify.py` | **Two new checks** (V27 for the AUD-107-09 attention mask and the AUD-107-10 resampler, V28 for the AUD-107-08 Part 8 synchronization) plus the V24 ledger-length expectation updated from `1..67` to `1..70`. |
| `KHMER-SOVEREIGN-S2ST-BLUEPRINT-v10.7.txt` | **Regenerated in place** by `__gen_v107.py`. Not hand-edited. |
| `docs/01_ARCHITECTURE_AND_CONTRACTS.md` | Regenerated - an exact partition of the new master. |
| `docs/02_DEEP_DIVE_IMPLEMENTATIONS.md` | Regenerated - carries the new `duration_aligner.py` and `pcm-capture-processor.ts` blocks. |
| `docs/03_TDD_GATES_AND_ORCHESTRATION.md` | Regenerated - carries the new Gate 4 assertions and the updated Part 8. |
| `docs/05_CODE_LEVEL_AUDIT_AND_REMEDIATION.md` | Regenerated - extended with the three new findings. |
| `backend/nn_cartridges/duration_aligner.py` | **Materialised.** The AUD-107-09 attention mask is applied inside the master's `#### 4.4.3` block, so the materialised file changes as a consequence. **Not** edited directly. |
| `backend/tests/test_fast_talker_and_glue.py` | **Materialised** (Gate 4). The new AUD-107-09 test is added to the master's `### 5.4 Gate 4` block, so the materialised file changes as a consequence. **Not** edited directly. |
| `frontend/src/worklets/pcm-capture-processor.ts` | **Materialised.** Changes as a consequence of the master's `#### 4.8.1` block. **Not** edited directly. |
| `frontend/src/contracts/types.ts` | **Unchanged.** `SAMPLE_RATE_IN` already exists and is already exported; the worklet only starts importing it. |

### Files not modified

`crates/khmer-linguistics-rs/src/lib.rs`, `crates/khmer-transport-rs/src/lib.rs`, `crates/khmer-dsp-wasm/src/lib.rs`, `Cargo.toml`, `Cargo.lock`, `pyproject.toml`, `frontend/package.json`, `frontend/tsconfig.json`, `README.md`, `docs/README.md`, `docs/04_AUDIT_FINDINGS_AND_REMEDIATION.md`, `KHMER-SOVEREIGN-S2ST-BLUEPRINT-v10.6.txt`, and the `_archive_blueprints/` folder are all untouched. The version stays `10.7.0` in all five declared locations.

### A note on `duration_aligner.py` and `test_fast_talker_and_glue.py`

Neither currently exists as a `__v107_blocks/` file. Both are **inline** in the v10.6 source and flow into the master through the generator's plain-text pass-through. The plan therefore edits them with `rep()` (an asserted, counted string replacement) rather than `blockfile()`. `rep()` is the correct tool here because the two files are small and their edits are single, well-anchored hunks; promoting them to block files would be a larger refactor with no benefit to this pass. **The AUD-107-09 test must be added to the master's `### 5.4 Gate 4` fenced block**, which the materializer extracts by heading - so the test reaches `backend/tests/test_fast_talker_and_glue.py` automatically.

## Functions

### New functions

**None.** All three code changes are edits to existing function bodies.

### Modified functions

| Function | File | Required change |
| :--- | :--- | :--- |
| `LearnedDurationAligner.forward` | `backend/nn_cartridges/duration_aligner.py` (Part 4.4.3) | **AUD-107-09.** After `attn_logits = -0.5 * ((frame_grid - centers.unsqueeze(1)) / self.sigma) ** 2` and **before** `attn_weights = F.softmax(attn_logits, dim=-1)`, mask the deferred columns out of the attention: build `active_mask = (sesqui_ids > 0).unsqueeze(1)` (shape `(B, 1, L)`), compute `has_active = active_mask.any(dim=-1, keepdim=True)` (shape `(B, 1, 1)`), and replace the inactive columns with `-1e9` **only where `has_active` is true**: `attn_logits = torch.where(has_active & (~active_mask), torch.full_like(attn_logits, -1e9), attn_logits)`. The `has_active` guard is required: a row in which **every** syllable is deferred has no active column, and masking it would leave `softmax` over an all-`-1e9` vector, which is uniform rather than undefined - but the guard keeps the behaviour defined and explicit. **No parameters are added**; `self.sigma` and `self.dur_proj` are untouched. |
| `PcmCaptureProcessor.process` | `frontend/src/worklets/pcm-capture-processor.ts` (Part 4.8.1) | **AUD-107-10.** Read the real context rate once per call: `const ctxRate = typeof sampleRate === "number" && sampleRate > 0 ? sampleRate : 48000;`. If `ctxRate === 48000`, run the existing 15-tap Blackman-sinc 3:1 decimator unchanged (`this.phase = (this.phase + 1) % 3` still drives the 1-in-3 decimation). Otherwise run the fractional linear-interpolation resampler: `const step = ctxRate / SAMPLE_RATE_IN;` and, per input frame, `while (this.fracPos < 1.0 && outCount < this.outBuf.length) { this.outBuf[outCount++] = this.prevSample + this.fracPos * (curSample - this.prevSample); this.fracPos += step; }` followed by `this.fracPos -= 1.0;` and `this.prevSample = curSample;`. The packet-packing tail (the 8-byte float64 media timestamp, the `Float32Array` payload, the `postMessage` transfer list) is **unchanged**. |
| `PcmCaptureProcessor` constructor / class body | `frontend/src/worklets/pcm-capture-processor.ts` (Part 4.8.1) | **AUD-107-10.** Add `private fracPos = 0.0;` and `private prevSample = 0.0;`; raise `outBuf` from `Float32Array(64)` to `Float32Array(128)`; add `import { SAMPLE_RATE_IN } from "../contracts/types";` alongside the existing side-effect import; add `declare const sampleRate: number;` alongside the existing `declare const currentTime: number;`. The `port.onmessage` handler (media clock and `playbackRate`) is **unchanged**. |

### Removed functions

**None.**

### The AUD-107-09 test - corrected form

The specification's literal test compares a 4-token call against a 2-token call that uses **independently drawn** random hidden states, so the two calls differ regardless of the attention mask (measured max diff 3.67 against a correct implementation). The test must share the first two token rows:

```python
def test_deferred_syllables_are_excluded_from_gaussian_attention():
    # v10.7 AUD-107-09: a deferred (sesqui_ids == 0) syllable has a zero
    # duration, so its cumulative centre sits at target_frames. Before the
    # fix it still participated in the Gaussian softmax and stole attention
    # mass from the tail frames of the clause (measured: 1.628554 of the
    # 24-frame mass, per deferred token).
    import torch

    from backend.nn_cartridges.duration_aligner import LearnedDurationAligner

    torch.manual_seed(0)
    aligner = LearnedDurationAligner(d_model=64)

    B, L, D, T = 1, 4, 64, 24

    # The hidden state for the four-token call. The first two rows are the
    # REAL syllables; the last two are the deferred ones, given an extreme
    # value so that ANY attention mass they receive is glaringly visible.
    hidden = torch.randn(B, L, D)
    hidden[0, 2:, :] = 1e6

    # The two-token reference REUSES the first two rows of the same tensor.
    # Using an independent randn here would compare two different signals and
    # would fail regardless of whether the attention mask is correct.
    hidden_ref = hidden[:, :2, :].contiguous()

    sesqui_four = torch.tensor([[2, 2, 0, 0]], dtype=torch.long)
    sesqui_two = torch.tensor([[2, 2]], dtype=torch.long)

    aligned_four, durs_four = aligner(hidden, sesqui_four, target_frames=T)
    aligned_two, durs_two = aligner(hidden_ref, sesqui_two, target_frames=T)

    # The deferred syllables consume no frames.
    assert float(durs_four[0, 2]) == 0.0
    assert float(durs_four[0, 3]) == 0.0

    # The real syllables are unaffected by the deferred ones.
    assert torch.allclose(durs_four[0, :2], durs_two[0, :], atol=1e-4)

    # THE REGRESSION ASSERTION: the deferred tokens' hidden state (1e6) must
    # not reach the aligned output. Before the fix this differs by ~1e6.
    assert torch.allclose(aligned_four, aligned_two, atol=1e-4)
```

**Why this is discriminating.** Against the current v10.7 code the deferred columns collect 1.628554 of attention mass each, so `aligned_four` picks up roughly `1.63e6` of the `1e6` hidden values and the final assertion fails by five orders of magnitude. Against the fixed code the measured max difference is `3.9e-06`, inside the `1e-4` tolerance.

### The AUD-107-10 resampler - verified behaviour

The plan records the measured behaviour rather than asserting it:

| Context rate | Output rate | Pitch | Buffer |
| :--- | ---: | :--- | :--- |
| 48,000 Hz | 16,000 Hz | exact | 42.67 out/quantum, fits in 128 |
| 44,100 Hz | 16,000 Hz | exact (peak at 1000.0 Hz for a 1,000 Hz input) | 46.45 out/quantum, fits in 128 |
| 96,000 Hz | 16,000 Hz | exact | 21.33 out/quantum, fits in 128 |
| 24,000 Hz | 16,000 Hz | exact | 85.33 out/quantum, fits in 128 |
| 22,050 Hz | 16,000 Hz | exact | 92.88 out/quantum, fits in 128 |
| 16,000 Hz | 16,000 Hz | exact | 128 out/quantum, exactly fits |
| **below 16,000 Hz** | **not supported** | - | the accumulator cannot upsample; documented as a limitation |

**Documented limitation:** the fractional accumulator supports context rates **at or above 16 kHz**. A device locked below 16 kHz (which no current mobile browser or Bluetooth profile does) would need a different, interpolating formulation. The plan records this in `docs/05` and in the Part 8 requirement so it is not mistaken for a regression.

**Documented quality difference:** the 48 kHz fast path lowpasses with the 15-tap Blackman-sinc (cutoff 7.2 kHz) before decimating; the fallback does not lowpass, so content above 8 kHz can alias (measured: a 10 kHz tone at a 44.1 kHz context rate folds to 6 kHz). The fallback also carries the ~0.635 linear-interpolation passband droop at 1 kHz. This is accepted: the fallback exists to keep the pipeline functional on non-48 kHz devices, and the upstream ASR and VAD operate well below 8 kHz. The plan records it as a known difference between device paths.

## Classes

### Modified classes

| Class | File | Specific modification |
| :--- | :--- | :--- |
| `LearnedDurationAligner` | `backend/nn_cartridges/duration_aligner.py` (Part 4.4.3) | `forward` gains the deferred-column attention mask described above. **`__init__` is unchanged**: `self.sigma` and `self.dur_proj` keep their shapes, so the class contributes exactly the same number of parameters. **No parameter change.** |
| `PcmCaptureProcessor` | `frontend/src/worklets/pcm-capture-processor.ts` (Part 4.8.1) | Gains the private fields `fracPos` and `prevSample`, and `outBuf` grows from 64 to 128 entries. `process` gains the rate branch. The class keeps `extends AudioWorkletProcessor` and the `registerProcessor("pcm-capture-processor", PcmCaptureProcessor)` registration is unchanged. **No parameter change** (TypeScript, not a model). |

### New classes

**None.**

### Removed classes

**None.**

---

## Dependencies

### New packages

**None.** The entire pass is parameter-free and adds no third-party dependency. This is a hard constraint: the ledger `241,637,457` / `3,838,033` must be preserved exactly, and both code remediations are therefore implemented with existing modules (`torch.nn.functional` for the attention mask) or plain arithmetic (the resampler).

### Version changes

**None.** The project version stays `10.7.0` in all five declared locations (`Cargo.toml`, `pyproject.toml`, `frontend/package.json`, the master's Part 2.3 blocks, and the master's Part 8 prompt). The Part 8 wording is **replaced**, not re-versioned.

### Dependency removals

None.

### Integration requirements

- The `pcm-capture-processor.ts` import of `SAMPLE_RATE_IN` resolves to `frontend/src/contracts/types.ts`, which is materialised from the master's `#### 2. frontend/src/contracts/types.ts` block and already exports `export const SAMPLE_RATE_IN = 16000;`. The relative path `../contracts/types` from `frontend/src/worklets/` is correct. The `frontend/tsconfig.json` `include` is `["src/**/*.ts"]`, so the worklet and the contracts file are in the same program and the import typechecks.
- `declare const sampleRate: number;` follows the pattern already used for `currentTime` in the same file. `AudioWorkletGlobalScope` provides both.
- The `pyo3` version and the Rust dependency set are untouched. The Rust crates keep their byte-for-byte parity with their Part 4.9.x blueprint blocks (V17).
- `sacrebleu` and `funasr` remain declared but not importable in this environment. That gap is unchanged and does not block the pass, because the eval suite is a training-time tool and no gate imports it.

---

## Testing

### Test file requirements

**No new test file is created.** The AUD-107-09 regression test is added to the **existing** Gate 4 suite (`backend/tests/test_fast_talker_and_glue.py`, master Section 5.4), which is the natural home: Gate 4 already exercises `LearnedDurationAligner` directly, and it already imports `LearnedDurationAligner` and `MINOR_SYLLABLE_MAX_FRAMES`. The six-gate structure, the gate file names, and the Part 7 ownership boundaries are all preserved.

The AUD-107-10 and AUD-107-08 requirements are verified by the **static** verification harness (`__v107_verify.py`), because the resampler is a TypeScript AudioWorklet that cannot be instantiated in a Node test, and the Part 8 prompt is prose.

### Existing test modifications

**Gates are extended, never weakened.** The three existing Gate 4 assertions are retained verbatim:

- `test_scene_matcher_rng_isolation_and_autograd_backward` - unchanged
- `test_duration_aligner_minor_cap_and_playback_scaling` - unchanged. **This one is important:** it asserts `durs.sum() == 400` for `sesqui_ids = [[1, 1, 2, 2]]`, and the new attention mask must not change any duration. The mask is applied to `attn_logits` only, **after** `final_durations` is computed, so the durations are provably untouched. This assertion is a live regression guard for the fix.
- The new `test_deferred_syllables_are_excluded_from_gaussian_attention` is **added alongside** them.

### Validation strategies

The inherited checks V1 through V26 must remain green, including V1 (the parameter ledger, re-derived under `torch.device("meta")` from the materialised modules) and V24 (document hygiene). Two new checks are added.

**V1 - Parameter ledger re-derivation (mandatory, blocking).**
Re-derive every module count from the materialised tree exactly as Gate 2 does. The expected values are **unchanged**:

| Module | `prod_240m` | `ci_micro` |
| :--- | ---: | ---: |
| `Causal3StemSeparator` | 5,864,197 | 231,877 |
| `VocalModeGate` | 816,388 | 31,684 |
| `StreamingEarCartridge` | 64,546,817 | 528,129 |
| `DisentangledEmotionEncoder` | 8,735,748 | 184,196 |
| `AcousticPointerCodeSwitchBridge` | 4,989,953 | 329,857 |
| `KhmerMouthCartridge` | 148,931,136 | 2,109,504 |
| `VocosISTFTHead` | 7,753,218 | 422,786 |
| **Combined total** | **241,637,457** | **3,838,033** |

**V24 - Document hygiene (mandatory, blocking), updated.**
The ledger-length expectation changes from `1..67` to **`1..70`**, and the "no stale `from 1 to 64`" style checks continue to pass. The single-policy / single-provenance / no-marker / no-mojibake checks are unchanged.

**V27 - AUD-107-09 and AUD-107-10 (mandatory, blocking). NEW.**
Two parts, both executed rather than pattern-matched where execution is possible:

*Part 1 - the attention mask is real.* Build a `LearnedDurationAligner`, run a 4-token call with `sesqui_ids = [[2, 2, 0, 0]]` and a 2-token reference that **reuses the first two hidden rows**, and assert the deferred syllables' `1e6` hidden values do not reach the aligned output. This is the same assertion as the Gate 4 test, run in the harness so a failure is reported with the harness's own diagnostics. Additionally assert that the deferred columns collect **zero** attention mass, by recomputing `softmax(attn_logits)` and summing per syllable.

*Part 2 - the resampler is wired and the rate is read.* Static assertions on the materialised worklet block: `declare const sampleRate: number;` is present; `SAMPLE_RATE_IN` is imported; the 48 kHz fast path and the fallback branch both exist; `fracPos` and `prevSample` are declared; `outBuf` is at least 128 entries; and the fixed 3:1 decimator is retained for the 48 kHz path. In addition, the harness **re-implements the spec's resampler arithmetic in Python and measures it**, asserting that a 44.1 kHz context produces exactly 16,000 output samples per second and that a 1,000 Hz input peaks at 1,000 Hz in the output (Goertzel), i.e. no pitch drift. This turns the two device claims into executable assertions.

**V28 - AUD-107-08 Part 8 synchronization (mandatory, blocking). NEW.**
Static assertions on the Part 8 segment of the master: Specialist A carries the AUD-107-01 / AUD-107-09 requirement and the AUD-107-02 requirement; Specialist C carries the AUD-107-04 requirement; Specialist D carries the AUD-107-03, AUD-107-05, AUD-107-06 and AUD-107-02 requirements; and the **stale** `lookaheadTargetTime` ERB instruction is **gone**. The last assertion is the discriminating one: it fails against the current v10.7 Part 8, which still mandates the instruction that the 2.00 s DelayNode invalidated.

### Independent corroboration

| Suite | Expected |
| :--- | :--- |
| The six TDD gates (`__v107_tests/`) | **38 passed** (37 today, plus the new Gate 4 test) |
| `cargo test --workspace` | **6 test binaries green** |
| `npx tsc --noEmit` (strict, ES2022, DOM) | **exit 0** |
| Generator idempotency | byte-identical re-run (SHA-256 match) |
| Rust <-> blueprint parity (V17) | all 3 crates byte-identical to their Part 4.9.x blocks |
| Doc-volume zero divergence | each of `docs/01`..`docs/03` is an exact master partition |
| V1 ledger re-derivation | `241,637,457` / `3,838,033`, per module and in total |

## Implementation Order

The steps are ordered so that each is independently verifiable, so the parameter ledger is re-derived **after** the code change that could plausibly move it, and so the prose is written against the final code.

**Step 1 - Establish the baseline.**
Record the current state before any edit: the master SHA-256, the current V-check results (V1..V26 green, 19/19), the current gate count (37), the current ledger length (1..67), and the current `docs/` volume hashes. This makes the pass auditable: every later number can be attributed to a specific step.

*Rationale for first:* the whole pass is a set of asserted replacements, and the generator's guarantee is only meaningful if the pre-state is known.

**Step 2 - AUD-107-09: the deferred-column attention mask.**
Add the `active_mask` / `has_active` mask to `LearnedDurationAligner.forward` in the master's `#### 4.4.3` block, via a `rep()` call in `__gen_v107.py` that anchors on the exact `attn_logits = -0.5 * ((frame_grid - centers.unsqueeze(1)) / self.sigma) ** 2` line and the following `attn_weights = F.softmax(attn_logits, dim=-1)` line. Regenerate, re-materialise, and confirm the deferred attention mass drops from 1.628554 to 0.0 and that the real syllables recover 12.000000 / 12.000000.

*Rationale for second:* this is the only change that touches a model-internal numeric path, so it is verified in isolation before anything else lands.

**Step 3 - Re-derive the parameter ledger (V1, mandatory gate).**
Recompute all seven module counts under `torch.device("meta")` from the materialised tree. The expected values are unchanged. This is repeated at the end of the pass as well.

*Rationale for third:* the ledger is the blueprint's most load-bearing claim, and verifying it immediately after the one numeric change converts a silent regression into a localised failure.

**Step 4 - AUD-107-10: the universal cross-device resampler.**
Replace the `__v107_blocks/pcm_capture_processor.ts` body with the spec's implementation via a `blockfile()` call (the file already has a `blockfile()` entry, so the call is edited rather than added). Regenerate, re-materialise, and confirm `tsc --noEmit` is still exit 0. Then run the Python simulation of the spec's arithmetic to confirm 16,000 output samples/s at every context rate from 16 kHz to 192 kHz and an exact 1,000 Hz output peak for a 1,000 Hz input.

*Rationale for fourth:* the worklet is independent of the Python model, so it can land once the aligner change is verified.

**Step 5 - AUD-107-09's regression test in Gate 4.**
Add `test_deferred_syllables_are_excluded_from_gaussian_attention` to the master's `### 5.4 Gate 4` fenced block, using the **corrected** form with the shared hidden rows. Re-materialise and run the gate suite. Confirm the new test **fails against the pre-fix aligner** and **passes against the fixed one** - a test that cannot fail is not a regression test.

*Rationale for fifth:* the test is written after the implementation is verified, so it encodes the measured contract rather than an assumed one.

**Step 6 - AUD-107-08: the Part 8 synchronization.**
Edit `__v107_blocks/stage02.txt`: add the Specialist A items 11 and 12, the Specialist C item 5, and replace the Specialist D item 9 with the four new requirements. The `stage02.txt` file is already wired into the generator through a `textfile()` call anchored on `STAGE 02: PARALLEL SPECIALIST EXECUTION (GREEN PHASE`, so the edit is to the block file plus the `until_marker` if it needs adjusting. Regenerate and confirm the stale `lookaheadTargetTime` ERB instruction is gone from Part 8.

*Rationale for sixth:* the orchestration prompt is the artefact a downstream implementer reads, so it is written last, against the final code.

**Step 7 - The provenance ledger, entries 68 to 70.**
Append the three entries to `__v107_blocks/fix_ledger.txt`, each recording the measured evidence, the root cause, and the exact remediation. The existing 9.2 generator step already reads that file and inserts it after entry 30, so the master becomes a contiguous 1..70 ledger with no duplicate homes. Regenerate and confirm the ledger is 1..70, sequential, and entirely before Part 1.

*Rationale for seventh:* the ledger is the record of the pass, so it is written once the fixes are final and their measurements are known.

**Step 8 - Extend `docs/05` and the verification harness.**
Add the three new findings to `__v107_blocks/docs05_part3.txt` and the new validation names to its "New validations" list. Add **V27** and **V28** to `__v107_verify.py` and update the V24 ledger-length expectation to `1..70`. Confirm the harness reports 21/21.

*Rationale for eighth:* the harness and the audit ledger are the project's regression guards, so they are updated to cover exactly what this pass changed.

**Step 9 - Full validation sweep.**
Run the complete gate suite, the Rust workspace tests, the strict TypeScript typecheck, the V1 ledger re-derivation, the V17 Rust parity check, and the doc-volume zero-divergence check. Then re-run `__gen_v107.py` and confirm the output is byte-identical (idempotency).

*Rationale for ninth:* idempotency is the property that guarantees the master and the volumes can never silently drift apart, and it is the last thing that can regress.

---

## Explicitly Out of Scope

To keep this pass bounded and the diff auditable, the following are **not** done:

- **No new trainable parameters.** The ledger `241,637,457` / `3,838,033` is preserved exactly. The attention mask is pure tensor arithmetic; the resampler is pure arithmetic.
- **No new master file.** The existing `KHMER-SOVEREIGN-S2ST-BLUEPRINT-v10.7.txt` is regenerated in place. The version stays `10.7.0`.
- **No support for context rates below 16 kHz** in the resampler. The fractional accumulator cannot upsample; the limitation is documented rather than engineered around, because no current mobile browser or Bluetooth profile uses a sub-16 kHz `AudioContext`.
- **No anti-alias filter on the resampler fallback.** The 48 kHz fast path keeps its 15-tap Blackman-sinc; the fallback does not lowpass, and the resulting aliasing difference between device paths is documented, not engineered away.
- **No change to the ERB band partition, the 48 kHz frontend rate, the 1.94 s turnaround law, or the `[230M, 250M]` Gate 2 assertion.**
- **No change to the six-gate structure, the gate file names, or the Part 7 ownership boundaries.**
- **No change to the Rust crates, the dependency sets, or the training dataset.**
- **No implementation of the Phase 2 Rust ABI.** It remains exactly as Section 4.9.7 specifies.
- **No change to `docs/04`.** It remains the v10.5 to v10.6 document-level ledger.

---

## Acceptance Criteria

This pass is accepted when all of the following hold:

1. `KHMER-SOVEREIGN-S2ST-BLUEPRINT-v10.7.txt` is regenerated in place, and `docs/01` through `docs/03` regenerate from it with **zero** divergence.
2. The parameter ledger equals `241,637,457` (`prod_240m`) and `3,838,033` (`ci_micro`), per module and in total, with no offsetting error.
3. The provenance ledger runs **1..70**, contiguous and unique, entirely before Part 1, and the three new entries (68, 69, 70) describe AUD-107-08, AUD-107-09 and AUD-107-10.
4. `LearnedDurationAligner.forward` masks `sesqui_ids == 0` columns out of `attn_logits` before the softmax, and the deferred columns collect **zero** attention mass (measured: 1.628554 before, 0.0 after).
5. The deferred syllables' hidden values cannot reach `aligned_hidden`; the new Gate 4 test **fails against the pre-fix aligner** and **passes against the fixed one**.
6. The Gate 4 durations assertion (`durs.sum() == 400`) still passes, proving the mask changed no duration.
7. `pcm-capture-processor.ts` reads the real `AudioWorkletGlobalScope.sampleRate`, keeps the 3:1 Blackman-sinc fast path at 48 kHz, and has the fractional-resampler fallback for every other rate, with an `outBuf` of at least 128 entries.
8. The resampler produces exactly 16,000 output samples per second at every context rate from 16 kHz to 192 kHz, with no pitch drift (a 1,000 Hz input peaks at 1,000.0 Hz).
9. Part 8 carries the AUD-107-01, AUD-107-02, AUD-107-03, AUD-107-04, AUD-107-05, AUD-107-06 and AUD-107-09 requirements, and the **stale** `lookaheadTargetTime` ERB instruction is gone.
10. The six gates are fully green at **38 passed**, with every pre-existing assertion retained verbatim.
11. `cargo test --workspace` passes and the three Rust crates remain byte-identical to their Part 4.9.x blueprint blocks.
12. `npx tsc --noEmit` exits 0 across all five TypeScript files.
13. The verification harness reports **21/21** checks green, including the new V27 and V28.
14. The generator is idempotent: a re-run produces byte-identical output.
15. `docs/05_CODE_LEVEL_AUDIT_AND_REMEDIATION.md` records the three new findings, the measured evidence, the resampler's supported rate range, and its documented aliasing difference from the 48 kHz fast path.
