# Audit Findings & Remediation (v10.5 -> v10.6)

This document is the **regression ledger** for the v10.6 truthfulness pass. It records
every defect found while auditing the v10.5 blueprint, the exact evidence for each,
the root cause, and how v10.6 resolves it. Any future revision must re-run the
validation suite at the end of this file before sign-off.

---

## Audit Method

Findings were established by **executing the blueprint's own reference code**, not by
reading it:

| Technique | What it proved |
| :--- | :--- |
| SHA-256 comparison of the v10.4 / v10.5 master files | The two revisions were byte-identical (AUD-03) |
| Instantiating every Part 4 module under `torch.device("meta")` and summing `p.numel()` | The true parameter ledger (AUD-01, AUD-02) |
| Rebuilding `build_slaney_mel_fb` and computing `torch.linalg.matrix_rank` | Rank is exactly 100 on both 513 and 1025 bins (Gate 2 holds) |
| Running all Gate 1-6 assertions against code extracted from the markdown | 10 of 10 assertions green (the design is sound) |
| Recomputing ERB band edges from the shipped `hzToErbRate` | Band 63 starts at 7603.5 Hz, not 7353 Hz (AUD-08) |
| `cargo test --workspace` | 3 of 3 Rust unit tests pass |
| Line-by-line diff of the master against the concatenated `docs/` volumes | 12-line version-token divergence and wrong line counts (AUD-12) |

**Headline conclusion:** the v10.5 *design* is sound and gate-passing. The v10.5
*bookkeeping* was not. All 15 findings are documentation-truth defects, not
architecture defects.

---

## Findings

| ID | Severity | Finding | Status in v10.6 |
| :--- | :--- | :--- | :--- |
| AUD-01 | Critical | `CI_MICRO` parameter ledger understated by 60% | Corrected |
| AUD-02 | High | `PROD` parameter ledger off by 87,457 | Corrected |
| AUD-03 | High | v10.4 and v10.5 master files byte-identical | Superseded, banner added |
| AUD-04 | High | Version-string drift across five files | Corrected + manifest added |
| AUD-05 | Critical | Rust-to-Python ABI is dead code | Specified (Phase 2) |
| AUD-06 | Medium | File tree names Rust modules that do not exist | Specified (Phase 2) |
| AUD-07 | Medium | `eval_s2st_suite.py` orphaned from tree and ownership | Corrected |
| AUD-08 | Medium | ERB Band 63 edge frequency misstated | Corrected |
| AUD-09 | Medium | Slaney filterbank not torchaudio-compatible | Documented (Phase 2) |
| AUD-10 | Medium | Eval-suite dependencies undeclared | Corrected |
| AUD-11 | Low | Training whitepaper orphaned | Cross-referenced |
| AUD-12 | Medium | Doc-volume counts and alignment claim false | Corrected + regenerated |
| AUD-13 | Low | `tsconfig.json` `include` diverged from contract | Corrected |
| AUD-14 | Low | Unused Rust dependencies declared | Specified (Phase 2) |
| AUD-15 | Low | FastAPI app title carried a stale version | Corrected |

---

## Detail

### AUD-01 - `CI_MICRO` parameter ledger understated by 60% (Critical)

**Claim.** The v10.5 Part 3.3 ledger stated `2.40M` parameters for
`CI_MICRO_CONFIG`.

**Evidence.** Re-deriving the count by instantiating every Part 4 module under
`torch.device("meta")` and summing `p.numel()`:

| Module | Claimed | Measured |
| :--- | ---: | ---: |
| `Causal3StemSeparator` | 0.19M | 231,877 |
| `VocalModeGate` | 0.04M | 31,684 |
| `StreamingEarCartridge` | 0.68M | 528,129 |
| `DisentangledEmotionEncoder` | 0.18M | 184,196 |
| `AcousticPointerCodeSwitchBridge` | 0.26M | 329,857 |
| `KhmerMouthCartridge` | 0.83M | 2,109,504 |
| `VocosISTFTHead` | 0.22M | 422,786 |
| **Total** | **2.40M** | **3,838,033** |

**Root cause.** The CI column had never been computed against
`CI_MICRO_CONFIG`. `KhmerMouthCartridge` contains **1,409,756** parameters
that are hardcoded to `SEMANTIC_DIM = 512` and therefore do **not** scale with
`mouth_d_model` (which is `128` under the CI profile): `reorder_attn`,
`reorder_norm`, the seven slot `heads`, the seven slot `embeddings`,
`target_lang_emb`, and `LearnedDurationAligner`. That is **36.7%** of the
entire CI budget.

**Impact.** No gate breaks - Gate 2 only asserts the prod range. But the
published CI figure was wrong by 1,438,033 parameters, and the "CI profile"
label overstated how small the CI build actually is.

**Remediation (v10.6).** Part 3.3 now publishes the measured value with exact
integers and states the root cause inline, so that any future attempt to
shrink the CI profile is an explicit, measured decision rather than an
accident.

---

### AUD-02 - `PROD` parameter ledger off by 87,457 (High)

**Claim.** The v10.5 ledger stated `241.55M` for `PROD_240M_CONFIG`.

**Evidence.** Measured total: **241,637,457** (`241.637M`).

| Module | Claimed | Measured |
| :--- | ---: | ---: |
| `Causal3StemSeparator` | 5.86M | 5,864,197 |
| `VocalModeGate` | 0.82M | 816,388 |
| `StreamingEarCartridge` | 64.46M | 64,546,817 |
| `DisentangledEmotionEncoder` | 8.73M | 8,735,748 |
| `AcousticPointerCodeSwitchBridge` | 4.99M | 4,989,953 |
| `KhmerMouthCartridge` | 148.93M | 148,931,136 |
| `VocosISTFTHead` | 7.76M | 7,753,218 |
| **Total** | **241.55M** | **241,637,457** |

**Root cause.** The per-row figures were estimates, not measurements, and the
rounded row values were then summed to produce the headline. The largest
single row error was `StreamingEarCartridge` (64.46M claimed vs 64.55M
measured), driven by the 16 `FastConformerBlock` sub-components not being
itemised in the ledger.

**Impact.** No gate breaks - `241,637,457` is still strictly inside
`[230M, 250M]`. The defect is that 16 locations across the blueprint, the
`docs/` volumes, the `README.md`, and the training whitepaper published a
figure the code did not produce.

**Remediation (v10.6).** All seven rows and the total now carry exact
integers alongside their rounded forms. Validation **V1** re-derives the table
on every revision.

---

### AUD-03 - v10.4 and v10.5 master files were byte-identical (High)

**Evidence.** `Get-FileHash` on both files returned the identical digest
`0DC1A212EB76D1AD06ECF7A319FC33F06E3D8F35DD63E7F04BE0152BAF95618C`, and a
line-by-line `Compare-Object` produced zero differences.

**Root cause.** The "v10.5" revision was a copy of v10.4, not a delta. The
21-entry fix ledger in the v10.5 header describes work that was applied to
the *text* of the file, but there is no diff between the two revisions to
audit - the header claims a history the file does not have.

**Remediation (v10.6).** Both files are retained as historical records with an
explicit supersession banner. `KHMER-SOVEREIGN-S2ST-BLUEPRINT-v10.6.txt` is
generated by a documented transformation script, so it is a genuine,
independently produced revision.

---

### AUD-04 - Version-string drift across five files (High)

**Evidence.** At audit time:

| Location | Value found | Should have been |
| :--- | :--- | :--- |
| `Cargo.toml` | `10.5.0` | consistent set |
| `pyproject.toml` | `10.5.0` | consistent set |
| `frontend/package.json` | `10.5.0` | consistent set |
| `docs/01` lines 173, 196, 226 | `10.4.0` | consistent set |
| `docs/03` line 380 | `10.4.0` | consistent set |
| Master blueprint Part 8 prompt | `10.4.0` | consistent set |

**Root cause.** The v10.5 "Fix P2-2" entry *claimed* the toolchain had been
synchronized to `10.4.0`, but the synchronization was never actually applied
to the shipped `docs/` volumes or to the Part 8 orchestrator prompt. The
ledger recorded an intention as though it were a completed action.

**Remediation (v10.6).** All five locations now carry `10.6.0`, and Part 2.3
gains a new **Section 2.3.1 Version Manifest** enumerating the authoritative
locations, so the next drift is caught by reading a single block rather than
by diffing five files. Validation **V2** enforces it.

---

### AUD-05 - The Rust-to-Python ABI is dead code (Critical)

**Claim.** Part 1.1 advertises a "Rust Zero-Alloc Phonological Engine" with a
"`< 15 ?s`/sentence, zero-GC" profile, and `backend/linguistics/khmer_normalizer.py`
contains a fast path guarded by:

```python
try:
    import khmer_linguistics_rs as _rs
    _HAS_RUST = hasattr(_rs, "normalize_khmer")
except Exception:
    _rs = None
    _HAS_RUST = False
```

**Evidence.** `Select-String` for `pymodule|pyfunction` across
`crates/*/src/*.rs` returned **zero matches**. The crate exports a plain Rust
`pub fn normalize_khmer_pure`, not a `#[pyfunction]`, and there is **no symbol
named `normalize_khmer`** anywhere in the Rust build. The probe therefore
**can never succeed**, and the Rust path is unreachable dead code.

A second, subtler defect: even once exported, the Rust normalizer omits the
independent-vowel canonicalization (`\u17A3`..`\u17B3` -> glottal onset +
dependent vowel) that the Python reference performs, so the two paths would
silently disagree.

**Impact.** The headline Rust-performance claim is currently unsupported. The
design is sound; the ABI was simply never wired.

**Remediation (v10.6 - specification only).** v10.6 is a document pass and
does **not** create or modify Rust source. Master blueprint **Section 4.9.7**
now specifies the exact Phase 2 deliverables (R-1 through R-8), each with its
acceptance check, including the mandatory independent-vowel canonicalization
and the `Lazy<Regex>` compile-once requirement that removes the per-call
regex recompilation (the current `lib.rs` rebuilds two `Regex` values on
*every* call, contradicting the "zero-allocation" claim).

---

### AUD-06 - File tree names Rust modules that do not exist (Medium)

**Evidence.** The Part 2.2 tree lists `normalizer.rs`, `decomposer.rs`,
`session.rs`, `live_queue.rs`, and `wav_container.rs`. `Get-ChildItem` over
`crates` found only three files: one `lib.rs` per crate.

**Root cause.** Section 4.9 (the crate listings) and the Part 2.2 tree
described different module structures, and neither matched disk.

**Remediation (v10.6 - specification only).** Section 4.9.7 specifies each
module's responsibility and public surface as a Phase 2 deliverable, and the
Part 2.2 tree annotates them as specified-but-not-yet-implemented so the tree
is no longer silently wrong.

---

### AUD-07 - `eval_s2st_suite.py` was orphaned (Medium)

**Evidence.** `backend/training/eval_s2st_suite.py` is specified in Section
4.6.3 and referenced in the v10.5 fix ledger, but a search across the Part 2.2
tree and the Part 7 ownership list found **no occurrence of it in either**.

**Root cause.** The file was added to the spec in a later patch without being
added to the tree or to the ownership plan, so no specialist would ever have
been assigned it.

**Remediation (v10.6).** Present in the Part 2.2 tree and explicitly owned by
Specialist C in Part 7.

---

### AUD-08 - ERB Band 63 edge frequency misstated (Medium)

**Claim.** The v10.5 fix ledger states that the `Math.floor(ERB_BANDS * norm)`
change "ensur[es] Band 63 `$[7353\text{ Hz}, 8000\text{ Hz}]$ is indexed".

**Evidence.** Recomputing from the shipped `hzToErbRate` (`21.4 * log10(1 +
0.00437 f)`) with `ERB_BANDS = 64` over `[120 Hz, 8000 Hz]`:

| Band | Computed lower edge |
| :--- | ---: |
| 62 | 7,226.0 Hz |
| 63 | **7,603.5 Hz** |
| 63 upper | 8,000.0 Hz |

`7353 Hz` maps to `norm = 0.9741`, so `floor(64 * 0.9741) = 62` - it is the
**Band 62** edge, not Band 63.

**Root cause.** The indexing **code** is correct and does reach Band 63 for
`7603.5 Hz` and above. Only the **prose** was miscalculated.

**Remediation (v10.6).** Corrected in the Fix P1-2 entry, in Section 4.8.2,
and in the Part 8 validation checklist. Validation **V7** re-derives it.

---

### AUD-09 - Slaney filterbank is not torchaudio-compatible (Medium)

**Evidence.** `build_slaney_mel_fb` normalizes each triangular filter by its
own half-width (`f_diff`) rather than by the full filter width used by
`torchaudio.functional.melscale_fbanks`. Measured against a torchaudio-faithful
reimplementation:

| Filterbank | Max abs difference | Mean abs difference |
| :--- | ---: | ---: |
| 513-bin (16 kHz) | 0.0250 | 0.000062 |
| 1025-bin (24 kHz) | 0.0204 | 0.000042 |

The lowest bands differ by roughly a factor of two in peak gain.

**Impact.** `warm_start_vocos_from_pretrained` loads `charactr/vocos-mel-24khz`
weights, a checkpoint trained on *torchaudio* filterbanks. Loading it into
this filterbank is a silent train/inference mismatch. Gate 2 is **not**
affected: the column rank is exactly `100` on both filterbanks, as re-verified.

**Remediation (v10.6 - specification only).** Documented as a WARNING at the
end of Section 4.2.1 with the two Phase 2 resolutions: (a) switch to the exact
torchaudio construction and re-verify Gate 2, or (b) keep the current
construction and re-train the Vocos head, abandoning the warm-start. The
filterbank is deliberately **not** changed, because (a) alters a Gate-2-critical
constant and (b) alters the training recipe - both are design decisions, not
documentation fixes.

---

### AUD-10 - Eval-suite dependencies undeclared (Medium)

**Evidence.** `backend/training/eval_s2st_suite.py` imports `sacrebleu`
(line 2860 of the master), `transformers`, and `funasr` (line 2881). None
appear in `pyproject.toml`. An environment probe confirmed `sacrebleu` and
`funasr` are **not importable**, so the suite would raise `ImportError` on
first use.

**Remediation (v10.6).** Added to the Part 2.3 `pyproject.toml` block, to the
live `pyproject.toml`, and to the Part 8 orchestrator prompt.

---

### AUD-11 - The training whitepaper was orphaned (Low)

**Evidence.** `Training, Distillation, and Dataset.txt` (8,617 words) was
referenced by **nothing** in `README.md`, `docs/`, or the master blueprint. A
search for `Training, Distillation` across all three returned no hits. It
also carried the stale `241.55M` / `2.40M` figures.

**Remediation (v10.6).** Cross-referenced from Part 7, Part 8, and
`docs/README.md`; its parameter figures are aligned with the corrected
ledger.

---

### AUD-12 - Doc-volume counts and the alignment claim were false (Medium)

**Claim.** `docs/README.md` stated the volumes were "100% bitwise & token
aligned across all volumes with zero compaction and zero omitted code", and
published line counts of 691 / 2,764 / 407.

**Evidence.**

| Volume | Claimed lines | Actual lines |
| :--- | ---: | ---: |
| `01_ARCHITECTURE_AND_CONTRACTS.md` | 691 | 699 |
| `02_DEEP_DIVE_IMPLEMENTATIONS.md` | 2,764 | 3,273 |
| `03_TDD_GATES_AND_ORCHESTRATION.md` | 407 | 407 |

A `Compare-Object` of the master against the concatenated volumes returned
**12 diff entries (6 divergent lines)**, all version tokens - so the
"bitwise aligned" claim was false.

**Remediation (v10.6).** The three volumes are now *generated* from the v10.6
master by the transformation script, partitioned at the Part 4 / Part 5
boundaries, so divergence is structurally impossible. The published counts are
re-derived from the regenerated files. Validation **V8** re-checks this.

---

### AUD-13 - `tsconfig.json` `include` diverged from the contract (Low)

**Evidence.** The on-disk `frontend/tsconfig.json` declared
`"include": ["src/**/*"]`; Part 2.3 specifies `["src/**/*.ts"]`.

**Remediation (v10.6).** The Part 2.3 block is authoritative; the on-disk file
is aligned to it.

---

### AUD-14 - Unused Rust dependencies declared (Low)

**Evidence.** `khmer-linguistics-rs` declares `crc32fast`; `khmer-transport-rs`
declares `crossbeam-channel`. Neither appears in any `src/*.rs` file.
`once_cell` is also declared but unused.

**Remediation (v10.6 - specification only).** Removal is specified as R-7 /
R-8 in Section 4.9.7. `once_cell` is **retained** because R-3 puts it to work
for compile-once regexes.

---

### AUD-15 - FastAPI app title carried a stale version (Low)

**Evidence.** Section 4.7.4 hard-codes
`FastAPI(title="KHMER-SOVEREIGN-S2ST v10.5 Full-Duplex Server")`.

**Remediation (v10.6).** Updated to `v10.6`.

---

## Validation Suite

v10.6 is accepted only when every check below passes. All are runnable against
the documents alone; none requires an implementation.

| ID | Check | Blocking |
| :--- | :--- | :--- |
| **V1** | Re-derive the Part 3.3 ledger under `torch.device("meta")`; totals must equal `241,637,457` and `3,838,033` **exactly** | Yes |
| **V2** | `10.6.0` appears in exactly the five Part 2.3.1 locations; no `10.4.0` / `10.5.0` version token survives | Yes |
| **V3** | `cargo test --workspace` passes; `khmer-dsp-wasm` builds for `wasm32` | Yes |
| **V4** | `import khmer_linguistics_rs` exposes `normalize_khmer` | Yes (Phase 2) |
| **V5** | Rust and Python normalizers agree on the parity corpus | Yes (Phase 2) |
| **V6** | The extracted Gate 1-6 suite remains 10 of 10 green and byte-identical to the v10.5 gate code | Yes |
| **V7** | The ERB Band 63 lower edge is stated as `7603.5 Hz` | Yes |
| **V8** | The `docs/` volumes regenerate from the master with zero divergence; published counts are re-derived | Yes |
| **V9** | Every file in the Part 2.2 tree has exactly one owner in Part 7 | Advisory |
| **V10** | The Slaney/torchaudio deviation and the Vocos warm-start mismatch are documented | Advisory |

### v10.6 Results

| ID | Result | Note |
| :--- | :--- | :--- |
| V1 | **PASS** | Ledger re-derived; totals exact |
| V2 | **PASS** | Five locations carry `10.6.0`; remaining `10.4.0` / `10.5.0` mentions are inside historical correction notes only |
| V3 | **PASS** | `cargo test --workspace` - 3 of 3 pass (crates unmodified by v10.6) |
| V4 | **DEFERRED** | Phase 2 deliverable - specified in Section 4.9.7 (R-1) |
| V5 | **DEFERRED** | Phase 2 deliverable - specified in Section 4.9.7 (R-2) |
| V6 | **PASS** | All 10 gate assertions re-run green; gate code byte-identical |
| V7 | **PASS** | `7603.5 Hz` |
| V8 | **PASS** | Volumes regenerated; zero divergence |
| V9 | **PASS** | `eval_s2st_suite.py` and the five Rust modules now owned |
| V10 | **PASS** | Documented in Section 4.2.1 |

---

## Regeneration Procedure

The v10.6 documents are produced by `__gen_v106.py` in the workspace root. It
performs every edit as an **asserted** string replacement: each replacement
declares the expected occurrence count, and the script aborts if the source
text does not match exactly. This makes silent drift impossible - a future
edit that fails to apply is a hard error, not a partial write.

```bash
cd "KHMER S2ST"
python __gen_v106.py      # regenerates v10.6 master + the three docs/ volumes
```

After running, re-derive the line and token counts published in
`docs/README.md` from the regenerated files.

---

# STAGE 07 - Self-Auditing Bootstrap (Rust ABI, Vocos Warm-Start, Pilot Shard, RTX 5070 Ti)

This section records the findings of the STAGE 07 execution pass. Every number
below was **measured by running the shipped code**, not inferred. The three
new scripts are:

| Script | Purpose |
| :--- | :--- |
| `scripts/bootstrap_vocos_warmstart.py` | STEP 2 - surgical Vocos warm-start + mel-to-wav round-trip audit |
| `scripts/prepare_khmer_pilot_shard.py` | STEP 3 - streaming pilot Khmer shard + 7-slot pre-tokenization |
| `scripts/smoke_train_stage2_infill.py` | STEP 4 - 25-step Stage-2 / 5A smoke test on the RTX 5070 Ti |

## STAGE-07-01: The Rust PyO3 ABI was never linked into the Python environment

**Severity:** CRITICAL (the v10.7 "15 microsecond Rust edge" was unreachable).

**Evidence.** The crates compiled to `target/release/khmer_linguistics_rs.dll`
and `khmer_transport_rs.dll`, but neither was importable from the active
interpreter, so `backend/linguistics/khmer_normalizer.py` silently took its
pure-Python fallback and `_HAS_RUST` was `False`.

**Root cause.** The workspace has no `maturin` in the toolchain and the
`.pyd` files were never produced or installed. The `#[pymodule]` ABIs were
correct - they were simply not deployed.

**Remediation.** Build both crates with the `extension-module` feature and
install the `cdylib` into `site-packages` as an importable `.pyd`:

```bash
cargo build --release -p khmer-linguistics-rs --features extension-module
cargo build --release -p khmer-transport-rs  --features extension-module
cp target/release/khmer_linguistics_rs.dll <site-packages>/khmer_linguistics_rs.pyd
cp target/release/khmer_transport_rs.dll  <site-packages>/khmer_transport_rs.pyd
```

**Verification.** `_HAS_RUST is True`; `normalize_khmer('\u17AB\u1791\u17D2\u1792\u17B7')`
returns `'\u179A\u17B9\u1791\u17D2\u1792\u17B7'`; `khmer_transport_rs` exports
`AtomicInterruptFence`, `pcm_energy_hash` and `wav_pcm16`. All 38 gates in
`backend/tests/` remain green **with the Rust ABI active**.

## STAGE-07-02: `VocosISTFTHead` ISTFT emits one hop more than the source

**Severity:** MEDIUM (a latent off-by-one in every mel-to-wav round trip).

**Evidence.** `VocosISTFTHead.forward` runs `torch.istft(..., center=True,
length=T * HOP_LENGTH)`. A forward `center=True` STFT of `N` samples yields
`ceil(N / HOP) + 1` mel frames, so the inverse transform emits
`ceil(N / HOP) * HOP` samples - for a 1.0 s signal at 24 kHz, **24 240**
samples instead of 24 000. The trailing hop is the zero-padded centre frame
and carries no signal.

**Impact.** Any consumer that assumes `wav.shape[-1] == T * HOP` is off by one
hop. The upstream Vocos reference implementation trims this hop; this
repository did not.

**Remediation.** `scripts/bootstrap_vocos_warmstart.py::_align_length` performs
the trim explicitly and the audit gate asserts the reconstruction is exactly
sample-aligned with the source.

## STAGE-07-03: A fully GPU-resident console exceeds the 3.20 GiB VRAM ceiling

**Severity:** MEDIUM (the advertised laptop-profile headroom does not hold as
written).

**Evidence.** Measured on the RTX 5070 Ti with the `laptop_5070ti_12gb`
profile, batch 4 x 2.0 s:

| Configuration | Peak `max_memory_allocated` |
| :--- | ---: |
| Fully resident console | **3.2867 GiB** (exceeds the 3.20 GiB ceiling) |
| Stage-2 infill configuration | **3.0229 GiB** (within the ceiling) |

**Root cause.** `compute_flow_loss(..., infill_mode=True)` takes the
`mouth.infill_proj` branch and **never calls** `console.ear` (64.5M params) or
`console.pointer` (5.0M params). Those 69.5M parameters receive no gradient on
the infill path, so they are pure VRAM ballast: 0.2638 GiB of the peak.

**Remediation.** The two modules are parked on the CPU for the training phase
and moved back to the GPU for the end-to-end inference check. The trained
weights are **identical** either way, because neither module receives a
gradient on the infill branch. The fully-resident figure is measured in an
isolated subprocess and is **disclosed** in the STEP 4 report rather than
hidden.

**Audit-method note.** A same-process probe cannot be used to measure this:
`apply_gradient_checkpointing` binds `blk.forward = types.MethodType(...)`,
which creates a reference CYCLE (block -> bound method -> block). The cycle
keeps the probe's parameters and gradients alive in VRAM long after `del`, so
the probe silently contaminates the real measurement by ~1.3 GiB. The
subprocess isolation is required for a trustworthy number.

## STAGE-07-04: The flow-matching loss is dominated by sampler noise

**Severity:** MEDIUM (a loss curve measured this way does not measure learning).

**Evidence.** `compute_shortcut_distillation_loss` draws `x_0 ~ N(0, I)` and
`t ~ U(0, 1)` internally. With a per-step-varying seed, the observed loss
trajectory is dominated by that noise: in the first STEP 4 probe the loss read
4.12, 3.87, 4.05, ... 11.99, 9.40, 10.61 - i.e. the *sign* of the change
between step 1 and step 25 was an artefact of the sampler.

**Remediation.** The flow-matching noise is re-seeded identically at every
step, so the step-1 vs step-25 comparison measures the optimizer update and
nothing else. The measured result is `22.906364 -> 10.470715`.

## STAGE-07-05: 25 backward passes contain only ONE AdamW update

**Severity:** LOW (correct production behaviour, but a misleading smoke test).

**Evidence.** The `laptop_5070ti_12gb` profile sets `grad_accum_steps = 16`,
and `CartridgeTrainer.train_step` only calls `optimizer.step()` on
`step_count % grad_accum_steps == 0`. 25 forward/backward passes therefore
yield exactly **1** optimizer update, and the loss trajectory is a step
function rather than a descent curve.

**Remediation.** The gate still reports the mandated step-1 vs step-25 figures,
and a **supplementary diagnostic** re-runs the identical loop with
`grad_accum_steps = 1` so that all 25 steps are genuine optimizer updates. The
measured descent is `22.906364 -> 0.900560`, which demonstrates that the
optimizer genuinely reduces the objective.

## STAGE-07-06: `torchcodec`/FFmpeg is a hard dependency of the default audio decoder

**Severity:** LOW (environmental, but it aborts a pilot run on the first row).

**Evidence.** `datasets` decodes `Audio` columns through `torchcodec`, which
requires the FFmpeg shared libraries. Without them the first `next()` raises
`RuntimeError: Could not load libtorchcodec`.

**Remediation.** `scripts/prepare_khmer_pilot_shard.py` casts the audio column
to `Audio(decode=False)` and decodes the raw bytes with `soundfile`, removing
the FFmpeg dependency entirely.

## STAGE-07-07: `HF_HUB_OFFLINE` is a MODULE CONSTANT, not an environment lookup

**Severity:** LOW (an environment-only offline switch silently does nothing).

**Evidence.** `huggingface_hub` reads `HF_HUB_OFFLINE` into a module constant
at import time. Setting `os.environ["HF_HUB_OFFLINE"] = "1"` after the import
leaves `huggingface_hub.constants.HF_HUB_OFFLINE == False`, and the streaming
iterator silently goes to the network anyway.

**Remediation.** `scripts/prepare_khmer_pilot_shard.py::_set_offline` patches the
constants of BOTH `huggingface_hub.constants` and `datasets.config`, and keeps
the patch active for the whole lifetime of the dataset (the Parquet shards are
resolved lazily on the first `next()`, so restoring the flag too early sends
the iterator back to the network).

## STAGE-07-08: The primary pilot corpus is gated; the fallback needs a bounded probe

**Severity:** LOW (dataset availability).

**Evidence.** `phonsobon/openslr42-khmer-male` is a **gated** dataset and
raises `DatasetNotFoundError` without a token. `seanghay/km-speech-corpus`
opens but its first row group requires a ~475 MB shard download. The
`google/fleurs` `km_kh` fallback is the one that works unauthenticated.

**Remediation.** The candidate resolver probes each candidate for its FIRST
row under a wall-clock timeout on a daemon thread, so a stalled HTTP read can
never wedge the process, and it prefers the local cache when the data is
already present.

## STAGE 07 - Executed Gate Summary

| Gate | Metric | Measured | Status |
| :--- | :--- | :--- | :--- |
| STEP 1 | `_HAS_RUST` | `True` | PASS |
| STEP 1 | `backend/tests/` | 38 passed | PASS |
| STEP 2 | Vocos parameters | `7,753,218` | PASS |
| STEP 2 | reconstruction shape | `(1, 24000)` | PASS |
| STEP 2 | `max|wav_recon|` | `60.227093` (> 0.05) | PASS |
| STEP 2 | Log-Mel L1 warm vs random | `2.271349` vs `6.904077` (3.040x) | PASS |
| STEP 3 | clips saved | `200` (>= 50) | PASS |
| STEP 3 | manifest size | `260.43 MiB` (< 500 MB) | PASS |
| STEP 3 | clips with NaN/Inf | `0` | PASS |
| STEP 3 | max 16k/24k duration skew | `0.0000 ms` (<= 1.0 ms) | PASS |
| STEP 3 | slot bounds | all 7 slots within `SLOT_VOCAB` | PASS |
| STEP 3 | round-trip decode | 5/5 valid non-empty UTF-8 Khmer | PASS |
| STEP 4 | loss step 1 -> step 25 | `22.906364 -> 10.470715` | PASS |
| STEP 4 | peak VRAM (infill) | `3.0229 GiB` (<= 3.20) | PASS |
| STEP 4 | `strict=True` reload | 735 == 735 keys | PASS |
| STEP 4 | `.blk.` key pollution | `0` keys (AUD-107-04 holds) | PASS |
| STEP 4 | WAV header | RIFF/WAVE, 1ch, 24000 Hz, 16 bit, 72 044 B | PASS |
| STEP 4 | disclosed: fully-resident peak | `3.2867 GiB` | FINDING (STAGE-07-03) |
