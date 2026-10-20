# Worklog — KHMER-SOVEREIGN-S2ST Blueprint Remediation

**Workspace:** `c:\Users\Chhean\Downloads\KHMER S2ST`
**Scope:** Blueprint documents only. No engine implementation was written or modified.
**Outcome:** v10.5 → v10.6 → v10.7, all 8 second-pass, 5 further, 7 AUD-107 and 3 further AUD-107 defect groups closed, 21/21 validations green.

---

## 0. What this worklog covers

This is the complete record of the remediation work performed on the
KHMER-SOVEREIGN-S2ST blueprint in this session. It spans four phases:

| Phase | From → To | Nature of the work |
| :--- | :--- | :--- |
| **A** | v10.5 → v10.6 | Document-level audit: numeric claims, version strings, file tree, ownership, ERB band claim, orphaned assets |
| **B** | — | Blueprint retention policy: archive old revisions, keep current + preceding |
| **C** | v10.6 → v10.7 | Code-level audit of the Part 4 reference implementations: 11 defects + 3 additional findings |
| **D** | v10.7 (2nd pass) | Line-by-line re-audit of the published v10.7: 8 further defects |

Every validation result quoted below is reproducible with the three scripts
listed in §6.

---

## 1. Method

Two principles governed the whole session.

### 1.1 Verify by execution, not by assertion

No finding was accepted from prose. Every claim was checked against the
extracted code and, where numeric, re-derived by running it:

- All 27 code blocks were extracted from the master and traced by call graph.
- The parameter ledger was re-derived under `torch.device("meta")`, per module.
- Numeric claims that looked plausible were recomputed (ERB band edges, Sway
  step sizes, `GroupNorm` vs `BatchNorm1d` parameter counts, brand hash
  collisions, slot-range disjointness).
- The Rust↔Python normalizer maps were compared **codepoint by codepoint**.

### 1.2 Tests that can fail

A validation was only accepted if it *discriminates* — i.e. it fails against
the defective version. Two of my own tests initially passed against v10.6 and
were rewritten to be genuinely discriminating (see §8, Defects 1 and 2).

---

## 2. Phase A — v10.5 → v10.6 (document-level audit)

### 2.1 Findings and remediations

| ID | Finding | Remediation |
| :--- | :--- | :--- | 
| AUD-01 | `241.55M` label was off by 87,457 | Replaced with the exact integer `241,637,457` plus per-module integers |
| AUD-02 | CI ledger claimed `2.40M`, actually `3,838,033` | Published exactly; root cause documented (1,409,756 params pinned to `SEMANTIC_DIM = 512`, 36.7 % of the CI budget) |
| AUD-03 | v10.4 and v10.5 were **byte-identical** (same SHA-256) | "v10.5" was a copy, not a delta — recorded as such rather than silently rewritten |
| AUD-04 | Version-string drift across five files | Applied a real sweep; added §2.3.1 Version Manifest as the single authoritative source |
| AUD-05 | Rust↔Python bridge was dead code | `hasattr(_rs, "normalize_khmer")` probed a symbol in **no** crate; no `#[pymodule]` existed anywhere |
| AUD-06 | ERB Band 63 passband claim was wrong | Stated `[7353, 8000]`, actual lower edge is **7603.5 Hz** (7353 Hz is the Band 62 edge) — the *code* was right, the *prose* was wrong |
| AUD-07 | `docs/README.md` line counts wrong (691/2,764/407 vs 699/3,273/407) | Counts re-derived from the regenerated files |
| AUD-08 | `docs/README.md` claimed "100 % bitwise & token aligned" — false | Replaced with a verifiable statement; a 12-line version-token divergence had existed |
| AUD-09 | `eval_s2st_suite.py` was in §4.6.3 but in neither the file tree nor the ownership plan | Added to both, with an owner |
| AUD-10 | `Training, Distillation, and Dataset.txt` was referenced by nothing | Cross-referenced from Part 7 and Part 8 |
| AUD-11 | `sacrebleu` / `funasr` used but undeclared | Added to `pyproject.toml` |
| AUD-12 | `khmer-dsp-wasm` could not compile to WASM | No `wasm-bindgen` in `Cargo.toml` or `Cargo.lock` — recorded as a Phase 2 deliverable |
| AUD-13 | Rust ABI had no specification | Added §4.9.7 with requirements R-1…R-8, explicitly Phase 2 |
| AUD-14 | Ownership plan had gaps | Every Part 2.2 file given exactly one owner |
| AUD-15 | FastAPI title named the wrong revision | Corrected |

### 2.2 Artefacts produced

- `KHMER-SOVEREIGN-S2ST-BLUEPRINT-v10.6.txt` (216,306 bytes)
- `docs/04_AUDIT_FINDINGS_AND_REMEDIATION.md` — the AUD-01…AUD-15 ledger

---

## 3. Phase B — revision retention

Policy adopted and enforced: **the workspace root retains only the current
revision and the immediately preceding one**; everything older goes to
`_archive_blueprints/` for provenance and is explicitly not authoritative.

Final state:

```
KHMER-SOVEREIGN-S2ST-BLUEPRINT-v10.7.txt   current
KHMER-SOVEREIGN-S2ST-BLUEPRINT-v10.6.txt   preceding (banner: SUPERSEDED -> v10.7)
_archive_blueprints/                        v10.1 .. v10.5
```

Deliberate choice: revisions were **archived, not hard-deleted**. The workspace
has no git history and no other backup, so deletion would have been
irrecoverable.

---

## 4. Phase C — v10.6 → v10.7 (code-level audit)

The v10.6 audit was document-level. This phase audited the **Part 4 reference
implementations** themselves, which turned out to harbour defects that no
document review could see.

### 4.1 The two that broke the product

**P0-A — the rolling cross-clause KV cache was dead code.**
`runner.py` contained **zero** references to `past_clause_kv`,
`past_gate_kv`, `deferred_syll_count` or `is_hard_ceiling_slice`. The console
and mouth cartridge accepted and implemented all four. The entire cross-clause
feature was unreachable in production.

**A-1 — the upstream audio path was absent entirely.**
`DuplexWsClient` contained no `addModule`, no `AudioWorkletNode`, no
`ArrayBuffer` and no binary `ws.send`. The server's `upstream_reader` waits for
frames that could never arrive, so the full-duplex loop could not function at
all. The v10.6 audit had framed this only as "unwired DSP"; it was a transport
failure.

### 4.2 Full finding table

| ID | Sev | Finding | Remediation |
| :--- | :--- | :--- | :--- |
| P0-A | Crit | Rolling KV cache and Hibiki wait-gating were dead code | Runner threads the state through and commits it back **only after** the epoch fence; `B_spk` guard; batch-safe deferral count; the 7 mutable `last_*` taps replaced by `@property` bridges onto a `threading.local()` container |
| P0-B | Crit | Session and registry state mutated without locks; epoch fence snapshotted **after** the GPU semaphore wait | Full `RLock` coverage on the three unprotected session methods; a dedicated lock on `SpeakerRegistry`; the epoch snapshot moved before the semaphore with a re-check inside it |
| P1-A | High | U+17AB..U+17AE mapped to **themselves**, and `_SYLLABLE_SCAN_RE` begins at U+1780 — a word starting with one **silently lost its first syllable** | Canonicalized to consonant + dependent vowel; U+17A8 corrected to the open (vowel-final) form; Rust parity applied |
| P1-B | High | KM→EN emitted **synthetic CVC pseudowords** (`"kak"`) because the cold-cache fallback ignored `coeng_id` | Register 3 partitioned by `coeng_id`: 0 = synthesized English syllable, 1..63 = input brand; brand coda table renumbered so slot 18 is the open coda and 19..31 the 13 closed codas |
| P1-C | High | Timestep embedding was **rank 2**; the distillation loss hardcoded a step the Sway ODE never visits; CFG never applied at inference; the duration target was a static, information-free prior | Parameter-free `sinusoidal_time_embedding`; intervals sampled from the shared Sway schedule with the finest teacher sub-steps anchored to the instantaneous velocity; CFG applied with the training dropout zeroing exactly the tensors the unconditional branch zeroes; duration targets derived from the `reorder_attn` attention mass |
| P1-D | High | The `SpeakerRegistry` matched on `DisentangledEmotionEncoder` output — which applies `F.instance_norm` and `avg_pool1d(k=25)`, removing the spectral envelope that carries **vocal-tract identity** | Energy-weighted, L2-normalized mean pool of the layer-6 acoustic representation, also added to `aligned_hidden` before `cond_proj` |
| P1-E | High | The 48 kHz ERB mask was computed and shipped but **never applied**; `baseVideoTimeMs` never posted; the capture worklet clocked at 1.0× instead of the video rate | `processMidSideFrame` on the mixer resolving the mask at `lookaheadTargetTime`; `attachAudioCapture` on the client; a rate-scaled media clock |
| P2-A | Med | `separate_and_classify` hardcoded `stems[0,0]`, `stems[0,1]`, `erb_mask[0]`; `mic_eq.expand(B_spk,-1,-1)` raises on `B=2, B_spk=1` | Batch aggregation, RMS-energy-weighted, down to batch 1; `erb_suppression_mask` stays 2D |
| P2-B | Med | `TRAINING_PROFILES` declared `precision` and `gradient_checkpointing`; `CartridgeTrainer` read **neither**; the distiller loss was never aggregated | Checkpointing applied at construction; `train_step` accepts a loss tensor **or** a callable; forward under `autocast`, **backward outside it**; distiller loss aggregated |
| P2-C | Med | `AutomatedS2STBenchmarkSuite` implemented only two of five pillars; every `S2STEvalReport` field was unassigned | `evaluate()` wiring ASR-BLEU, chrF, UTMOS, DNSMOS, speaker and emotion similarity, turnaround latency |
| P2-D | Med | Gates 1–6 covered none of the new behaviour | Gates 1, 2, 5, 6 expanded, **every v10.6 assertion retained verbatim** |
| A-1 | Crit | Upstream audio path absent entirely (see §4.1) | Closed by the P1-E transport remediation |
| A-2 | High | Brand `coeng_hash = 1 + (crc32 % 63)` collides with the Khmer coeng space **and with itself** (`"calvin"` and `"klein"` both → 25) | Structural root cause of P1-B, repaired by the same reserved-range partition |
| A-3 | Low | `last_gate_l` written every forward pass, never read | Removed with the other mutable taps |

### 4.3 The binding constraint: zero new parameters

Every fix is parameter-free by construction — an existing module, a
parameter-free helper, or a plain data table. The two changes that touch a
module's parameter structure are both **parameter-neutral**, and both were
proved by execution:

- `nn.BatchNorm1d(d)` → `nn.GroupNorm(8, d)`. Both hold `2*d` trainable
  parameters (1,024 prod / 256 CI per block). BatchNorm's `running_mean`,
  `running_var` and `num_batches_tracked` are **buffers** and are excluded
  from `.parameters()`. BatchNorm also normalizes over the batch axis, which
  has size 1 during streaming inference, so its running statistics were
  corrupted by batch-statistics estimation.
- The sinusoidal timestep embedding and the `CartridgeThreadState` container
  add no modules at all; `time_mlp` keeps its `Linear(d, d)` shape.

**Ledger preserved exactly: `241,637,457` (`prod_240m`) / `3,838,033` (`ci_micro`).**

### 4.4 Artefacts produced

- `KHMER-SOVEREIGN-S2ST-BLUEPRINT-v10.7.txt` (323,865 bytes)
- `docs/05_CODE_LEVEL_AUDIT_AND_REMEDIATION.md` — the P0-A…A-3 ledger
- Fix-ledger entries **37…56** appended to Part 1

---

## 5. Phase D — v10.7 second pass (8 further defects)

A line-by-line re-audit of the *published* v10.7 and of `__gen_v107.py` found
eight defects the first pass had missed. All were confirmed before fixing.

| ID | Sev | Finding | Remediation |
| :--- | :--- | :--- | :--- |
| S-1 | High | `stem_and_mode_gate.py` was **never patched** — `__gen_v107.py` had no `blockfile` call for it at all, so `separate_and_classify` still read `stems[0,0]`, `stems[0,1]`, `erb_mask[0]` and could raise on `B=2, B_spk=1` | Added the missing `blockfile` call; batch aggregation, RMS-energy-weighted, down to batch 1 |
| S-2 | High | The five `@property` bridges on `KhmerMouthCartridge` were named `next_past_kv`… while every call site uses `self.last_next_past_kv`… — Python therefore created **plain shared attributes** and **completely bypassed** `CartridgeThreadState`, silently voiding the thread isolation | All five bridges renamed on both the cartridge and the state container; `last_deferred_syll_count` returns `0` when unset |
| S-3 | High | `integrate_shortcut_sway_ode` computed `v` only in the `else` branch and referenced it in the `if` branch → `UnboundLocalError` for any `cfg_scale != 1.0`; CFG could never be used | `v_cond` computed unconditionally, then blended |
| S-4 | High | With `target_lang == "en"` the onset/coeng/vowel/coda ids were unconstrained, so `ENGLISH_SYLLABLE_TABLE` was never reached and `cs_words` filled with pseudowords | All four slots constrained to the reserved ranges, disjoint from the Khmer ranges |
| S-5 | High | The ERB suppressor was a **no-op**: the masked spectrum was computed then discarded and the unmasked `mid[]` written out. Plus **two** `createMediaElementSource` calls on one element → `InvalidStateError` | `_inverseSpectrum` added; masked Mid reconstructed; the mixer owns the single source node, returns it and exposes `getMediaSourceNode()`; the client taps that node |
| S-6 | Medium | The Rust `INDEPENDENT_VOWEL_MAP` had 16 entries (U+17A3..U+17B2); Python had 17 — the two disagreed on U+17B3 | U+17B3 added with a matching Rust unit test; the on-disk crate resynced; V4 asserts parity across all 17 |
| S-7 | Medium | The ear never published its bus output, so the distiller could not compute a loss; no `teacher_targets` path; a duplicated `loss_tensor = loss` line; ASR language hardcoded `"km"`; an undefined `comet_path` | `last_bus_out` published on the thread-local state; `teacher_targets` accepted by both attach and step; duplicate removed; language parameterised; `comet_checkpoint` constructor argument |
| S-8 | Low | Two retention-policy headers, two provenance headers, a leftover version-marker token, six mojibake comment lines, a duplicated ledger numbering (1..36 → 25..36 → 37..44), a v10.6 FastAPI title, two v10.6 Part 8 strings | Each header emitted once; marker removed; `\uXXXX` escapes used; ledger renumbered sequentially **1…64**; v10.7 named throughout |

S-2 deserves emphasis: it was a **silent** failure. The thread-local isolation
deliberately built in Phase C was present in the code but unreachable, because
the property names did not match their call sites. Validation V16 now asserts
`"last_next_past_kv" not in mouth.__dict__` after a forward pass — the property
that actually matters.

Fix-ledger entries **57…64** were appended in the first pass, giving a single
sequential **1…64** ledger with no duplicates. They were subsequently FOLDED
into entries 37…56 as "Second-pass confirmation" prose, so the published ledger
is now **1…56** with every fix described exactly once.

---

## 6. Tooling created

Three idempotent scripts, plus a directory of replacement blocks. They exist so
the blueprint can be regenerated and re-verified from scratch, and so the
documents can never silently drift from the master.

| Path | Purpose |
| :--- | :--- |
| `__gen_v107.py` (15,104 B) | v10.6 master → v10.7 master + `docs/01`…`docs/05`. Every edit is an **asserted** replacement that aborts on an occurrence-count mismatch, so a partially applied edit is impossible. Ends with a zero-divergence check that each volume is an exact master partition. |
| `__v107_materialize.py` (3,297 B) | v10.7 master → a runnable tree: `backend/` (28 modules), `frontend/src/` (5 files) and the six gates in `__v107_tests/`. This is what makes the reference implementations executable. |
| `__v107_verify.py` (36,755 B) | The 16-check validation harness (§7). |
| `__v107_blocks/` (31 files) | The replacement blocks and prose fragments the generator applies. Kept as plain files so the diff is auditable and the generator stays readable. |

A key design decision: the generator stores replacement blocks as **files**
rather than inline strings. Inline heredoc strings repeatedly introduced
encoding damage (§8, Defect 4) and quoting hazards; external files eliminated
both.

---

## 7. Validation suite

All checks are reproducible with:

```
python __gen_v107.py           # regenerate the master and the docs volumes
python __v107_materialize.py   # materialise a runnable tree
python __v107_verify.py        # run the 21 checks
```

| ID | Check | Result |
| :--- | :--- | :--- |
| **V1** | Parameter ledger re-derived under `torch.device("meta")`, per module | **PASS** — `241,637,457` / `3,838,033`, exact |
| **V2** | Version synchronization (5 declarations, 0 stale `10.6.0`) | **PASS** |
| **V4** | Rust ↔ Python independent-vowel parity, all 17 codepoints | **PASS** |
| **V8** | Concurrency stress — 6 threads × 2.0 s over the session and registry | **PASS** — 0 exceptions |
| **V8b** | Deterministic lock coverage (instrumented lock records acquisitions) | **PASS** — all 5 mutating methods hold their lock |
| **V10** | Step conditioning is not rank-degenerate | **PASS** — embedding rank 5 (v10.6 rank 1); varies with both `t` and `dt` |
| **V12** | Independent vowel reachability (U+17AB..U+17AE) | **PASS** |
| **V13** | Batch-aware scene separation, `separate_and_classify(torch.randn(2, 16000))` | **PASS** |
| **V14** | Frontend wiring (13 static checks) | **PASS** |
| **V16** | Mouth `last_*` property bridges route through the thread-local state | **PASS** |
| **V19** | `integrate_shortcut_sway_ode` with `cfg_scale=1.5` | **PASS** |
| **V20** | KM→EN slots land in the reserved English ranges; `cs_words` empty | **PASS** |
| **V21** | Trainer aggregates the distillation loss from `teacher_targets` | **PASS** |
| **V22** | Eval suite has no undefined COMET reference; honours the ASR language | **PASS** |
| **V23** | Frontend inverse FFT and a single shared source node | **PASS** |
| **V23a** | FFT/IFFT round trip executed under node — energy **and** sample equality against the unwindowed block | **PASS** — `ratio=1.000000000`, `maxAbsErr=1.6e-16` |
| **V24** | Document hygiene (one policy, one provenance, no marker, no mojibake, sequential ledger 1..56) | **PASS** |
| **V25** | Deferred (wait-gated) tail syllables are true `(0,0,0)` no-ops and are skipped by the decoder | **PASS** |
| **V26** | The seven AUD-107 code findings are closed (state_dict keys, single ear pass, delay node, zero-frame sentinel, FFT tables, muted capture sink) | **PASS** |
| **V27** | AUD-107-09 deferred columns collect zero attention mass; AUD-107-10 the resampler yields exactly 16 kHz at every supported rate with no pitch drift | **PASS** |
| **V28** | AUD-107-08 the Part 8 prompt carries AUD-107-01..06 and AUD-107-09, and the stale ERB instruction is gone | **PASS** ||

### Independent corroboration

| Suite | Result |
| :--- | :--- |
| The six TDD gates (`__v107_tests/`) | **38 passed, 0 failed** — every v10.6 assertion retained verbatim |
| `cargo test --workspace` | **6 test binaries green** (3 crates, 4 assertions) |
| `npx tsc --noEmit` (strict, ES2022, DOM) | **exit 0** across all 5 TypeScript files |
| Generator idempotency | `66e8a1043f6d1ac3` → `66e8a1043f6d1ac3`, byte-identical |
| Rust ↔ blueprint parity (V17) | All 3 crates byte-identical to their Part 4.9.x blocks |
| Doc-volume zero divergence | Each of `docs/01`…`docs/03` is an exact master partition |

### Discriminating power

A validation is only worth having if it **fails against the defective code**.
Each new check was confirmed to discriminate:

| Check | Against the defective version |
| :--- | :--- |
| V8b | v10.6 fails 4 of 5 methods (only `reset_on_seek` held the lock) |
| V10 | v10.6 embedding rank is 1, not 5 |
| V16 | v10.6's mismatched names put `last_next_past_kv` into `mouth.__dict__` |
| V19 | v10.6 raises `UnboundLocalError` at `cfg_scale != 1.0` |
| V20 | v10.6's unconstrained slots put `cs_words` outside the English ranges |
| V4 | v10.6's Rust map has 16 entries and disagrees on U+17B3 |
| V24 | v10.6 had 2 policies, 2 provenances, the marker and 6 mojibake lines |

---

## 8. Defects in my own work (honest record)

Four errors were made and caught during this session. All are recorded because
a worklog that only lists successes is not a worklog.

**1. A test that could not fail.** My first V10 asserted only that the
velocity *changed* with `t`. It passed against the defective v10.6 code,
because the learned weights alone made the output vary. The actual defect was
**rank**, not sensitivity. Rewritten to assert `rank > 1`, which v10.6 fails.

**2. A false "failure" in V2.** V2 initially reported FAIL. The cause was my
own filter, which excluded a legitimate `| Master blueprint` manifest row. The
document was correct; the test was wrong. The manifest lists exactly 5 rows.

**3. A false-positive parity comparator.** The Rust↔Python map comparison
reported 17 mismatches. The comparator compared Rust's `\u{17A2}` escape *text*
against Python's literal Khmer characters. The underlying code was correct
throughout; the comparator was rewritten to decode both sides.

**4. Encoding damage from heredocs.** Six comment lines in
`khmer_normalizer.py` lost their Khmer characters to the console encoding and
became `?`. This is why the replacement blocks are now **external files** — the
failure mode is structurally eliminated rather than merely avoided.

Two further near-misses are worth noting: a fence-handling bug in the generator
that silently produced empty code blocks (caught because the gate suite
imported them and failed loudly), and a materialization slicer that matched the
wrong heading and produced a stale test file (caught because the gate hung
rather than passing quietly).

---

## 9. Design decisions and their rationale

| Decision | Rationale |
| :--- | :--- |
| Archive rather than delete old revisions | The workspace has no git history and no other backup; deletion is unrecoverable |
| Regenerate `docs/` from the master rather than hand-editing | Makes drift structurally impossible; every volume is provably an exact partition |
| Asserted replacements only | A count mismatch aborts the generator, so a partially applied edit cannot be published |
| Replacement blocks as external files | Eliminated the encoding damage and quoting hazards that inline heredocs caused |
| Parameter-free fixes only | The ledger is a hard published invariant; it had to stay exactly `241,637,457` / `3,838,033` |
| Expand the gates rather than replace them | Every v10.6 assertion is retained **verbatim**, so the original guarantees cannot be lost |
| Reserve slot ranges instead of adding vocabulary | Adding English onsets/vowels/codas as *new* slots would have moved the ledger; reserved ranges keep it fixed |
| Thread-local state with `@property` bridges | Preserves 100 % attribute compatibility for existing call sites while making the storage per-thread |
| Materialize the reference implementations for testing | The blueprint's code was previously prose; making it executable is what allowed the code-level audit to be real |

---

## 10. Known limitations and outstanding items

These remain **open** and are not claimed as fixed.

1. **Rust↔Python ABI is still Phase 2.** No crate exposes a `#[pymodule]`, so
   the Python-side Rust fast path remains unreachable. The requirement is
   specified in §4.9.7 (R-1…R-8) but not implemented — a deliberate deferral
   carried forward from v10.6.
2. **`khmer-dsp-wasm` cannot compile to WASM.** `wasm-bindgen` is absent from
   both `Cargo.toml` and `Cargo.lock`.
3. **`sacrebleu` and `funasr` are declared but not importable** in this
   environment. The evaluation suite is a training-time tool and no gate
   imports it. COMET and DNSMOS are optional pillars and degrade to `0.0` when
   their packages or checkpoints are absent.
4. **`frontend/node_modules` is not installed.** The TypeScript was typechecked
   with a temporary local TypeScript 5.4.5, which was then removed.
5. **The ERB ducking path is verified statically, not audibly.** V23 asserts the
   inverse transform exists and is wired; it does not measure the attenuation a
   browser would produce.
6. **The concurrency stress is bounded.** V8 exercises 6 threads for 2 s and V8b
   proves lock *acquisition* deterministically, but neither is a formal proof of
   the absence of a race.

---

## 11. File inventory (final state)

### Workspace root
```
Cargo.lock                         7,185   KHMER-SOVEREIGN-S2ST-BLUEPRINT-v10.6.txt   216,747
Cargo.toml                           426   KHMER-SOVEREIGN-S2ST-BLUEPRINT-v10.7.txt   323,865
README.md                          7,543   Training, Distillation, and Dataset.txt   81,035
WORKLOG.md                             —   implementation_plan.md                    55,091
__gen_v107.py                     15,104   pyproject.toml                             790
__v107_materialize.py              3,297
__v107_verify.py                  36,755
```

### Directories
| Path | Contents |
| :--- | :--- |
| `docs/` | `01_ARCHITECTURE_AND_CONTRACTS.md` (78,949 B), `02_DEEP_DIVE_IMPLEMENTATIONS.md` (191,018 B), `03_TDD_GATES_AND_ORCHESTRATION.md` (53,896 B), `04_AUDIT_FINDINGS_AND_REMEDIATION.md` (18,434 B), `05_CODE_LEVEL_AUDIT_AND_REMEDIATION.md` (22,061 B), `README.md` (6,195 B) |
| `crates/` | 3 Rust crates, byte-identical to their Part 4.9.x blueprint blocks |
| `backend/` | 28 Python modules plus `__init__.py` files, materialized from the master |
| `frontend/src/` | 5 TypeScript files, materialized from the master |
| `__v107_blocks/` | 31 replacement blocks and prose fragments |
| `__v107_tests/` | The six materialized gates |
| `_archive_blueprints/` | v10.1 (152,718 B), v10.2 (158,919 B), v10.3 (167,303 B), v10.4 (198,004 B), v10.5 (198,037 B) |

### Cross-references
| Document | Contains |
| :--- | :--- |
| `docs/04_AUDIT_FINDINGS_AND_REMEDIATION.md` | 27 763 B |
| `docs/05_CODE_LEVEL_AUDIT_AND_REMEDIATION.md` | The v10.6 → v10.7 code-level ledger (P0-A…A-3) and the v10.7 second-pass ledger (S-1…S-8) |
| `implementation_plan.md` | The plan this work executed |
| Part 1 of the master | The full fix ledger, entries **1…64**, sequential and unique |

---

## 12. Reproducing this work

```
cd "c:\Users\Chhean\Downloads\KHMER S2ST"

python __gen_v107.py          # rebuild the master and docs/ from the v10.6 source
python __v107_materialize.py  # materialise backend/, frontend/src/ and the gates
python __v107_verify.py       # 21/21 checks
python -m pytest __v107_tests -q
cargo test --workspace
cd frontend && npx tsc --noEmit
```

Expected output at the time of writing: `ALL: PASS (21/21 checks)`,
`38 passed`, 6 green Rust test binaries, and `tsc` exit code 0.

---

*End of worklog. Blueprint documents only — no engine implementation was
written or modified at any point.*


---

## 13. Session: making the worklog an always-on skill

**Trigger.** The user pointed out that the worklog had been produced *because
they asked for it*, and that the intent was for it to happen automatically. The
worklog became a **skill** so the behaviour persists across sessions rather
than depending on a request.

### 13.1 What was created

| Path | Purpose |
| :--- | :--- |
| `~/.agents/skills/worklog/SKILL.md` (8,111 B) | The skill definition — the live copy the `skills` tool loads |
| `~/.claude/skills/worklog/SKILL.md` (8,111 B) | Byte-identical copy for Claude Code |
| `~/.agents/skills/worklog/.worklog_version` (`1.0.0`) | Version marker, following the `graphify` convention |
| `~/.claude/CLAUDE.md` | A `# worklog` section registering the skill as **always-on**, with the explicit instruction to invoke it *without the user asking* |
| `~/.clinerules` (1,558 B) | A global Cline rule carrying the same always-on instruction and the non-negotiable honesty constraints |

### 13.2 Why the description is the load-bearing part

The skill is only automatic if its **description** matches the work, because
that is what is shown when deciding whether an available skill applies. It
therefore reads:

> "maintain a durable, auditable worklog of the session - use PROACTIVELY and
> WITHOUT being asked at the end of ANY substantive task: audits, bug hunts,
> refactors, migrations, investigations, multi-file changes, or anything that
> produced findings, decisions, or verification results"

The `trigger: /worklog` line additionally makes it explicitly invocable.

### 13.3 What the skill encodes

The lessons from this session were generalised into the skill rather than left
implicit:

- **Run it unprompted**, but **skip trivial sessions** — a worklog that always
  fires becomes noise, and noise destroys trust in the audit trail.
- **Never claim something is fixed that is not.** Distinguish implemented and
  verified / implemented but unverified / specified but not implemented /
  deliberately deferred.
- **A test must be able to fail.** Confirm a new validation *discriminates* by
  running it against the defective code.
- **Distinguish "the code was wrong" from "my test was wrong."**
- **Record your own mistakes** — the mechanism, not just the apology.
- **Prefer structural fixes over vigilance.**
- **Write UTF-8 without a BOM, LF endings**, and verify non-ASCII survived.
- **Update an existing worklog** rather than overwriting it.

### 13.4 Verification

- The skill loads through the `skills` tool and renders its full instructions.
- Both `SKILL.md` copies are byte-identical.
- The YAML frontmatter parses, with `name`, `description` and `trigger`.
- All four files are valid UTF-8, no BOM, no `U+FFFD`.
- One "mojibake" hit was a false positive: the template's rhetorical
  "execution? reading? both?".

### 13.5 Defect in my own work

The first placement of the skill body failed — the editor rejected an 8,111-byte
payload against its 6,000-byte limit, and the file was left empty. It was
detected immediately because the file was read back and checked rather than
assumed, then split into two edits.


---

## 14. Session: second-pass fixes to the published v10.7 (5 defect groups)

**Trigger.** A line-by-line re-audit of the published v10.7 and of
`__gen_v107.py` found 5 further defect groups, two of them P0, introduced
*during* the previous session's patch. All were confirmed against the files
before fixing.

### 14.1 G1 - two P0 bugs in `stereo-mne-mixer.ts`

**The zero-output FFT/IFFT.** `_computeSpectrum` bit-reversed its input `x`
in place and then ran the butterflies on two freshly allocated **zero** arrays
`bre`/`bim`; the bit-reversed `x` was never copied in, so every bin came out
`0.0`. `_inverseSpectrum` had the identical defect and always returned an
all-zero signal. The ERB suppressor was therefore not merely a no-op, it also
**silenced the M&E bed**.

**The disconnected processor.** `source.connect(processor)` appeared twice and
`processor.connect(this.ctx.destination)` had been deleted, so the processor
reached nothing.

Both were fixed. The round trip was then **executed under node** rather than
inspected:

| | forward non-zero bins | round-trip energy ratio |
| :--- | ---: | ---: |
| before the fix | 0 | 0.000000 |
| after the fix | 513 | 1.000000 |

The ratio is measured against the **windowed** input, because the Hann window
is applied in the forward transform, so the exact inverse of the pair is the
windowed signal. The new validation **V23a** runs this under node and asserts
`nonzeroBins > 0`, `outPeak > 1e-9` and `|ratio - 1| < 1e-3`, so it fails
against the defective version.

### 14.2 G2 - inverted target language in `eval_s2st_suite.py`

`evaluate` transcribes `wav_tgt_24k`, the **target** audio, so for
`"en->km"` the target is Khmer. The check used `startswith("km")`, which
inverted it and made the English direction use the Khmer ASR model. Now
`d.endswith("km") or d == "km"`.

### 14.3 G3 - Rust normalization order

The Rust `normalize_khmer_pure` applied the coeng/shifter reordering **before**
the independent-vowel map, while the Python `normalize_khmer` does the
opposite. The map loop was moved ahead of the reordering loop and the on-disk
crate was resynced.

The user-supplied compound word (`\u17AB\u1791\u17D2\u1792\u17B7`) was
verified, and it was found **not to discriminate the ordering** - both orders
give the same result for it. A second case that *does* discriminate
(`\u17AB\u17B7\u17D2\u1792`: `179A 17D2 1792 17B9 17B7` after the fix,
`179A 17B9 17D2 1792 17B7` before) was added alongside it, so the ordering fix
is actually tested rather than merely asserted.

### 14.4 G4 - V19 and V20 moved into Gate 2

`test_v107_km_to_en_forward_decodes_to_latin` now unpacks
`texts, cs_words = decomp.decode_batch_to_text(batch_en)`, asserts
`cs_words == []`, and asserts that every predicted token satisfies
`is_english_syllable_slot`. A new `test_v107_classifier_free_guidance_path_is_usable`
exercises `integrate_shortcut_sway_ode(..., cfg_scale=1.5)` directly. The
blueprint's own test suite now enforces both, rather than leaving them to an
external harness only.

### 14.5 G5 - remaining code-block and prose glitches

| Item | Fix |
| :--- | :--- |
| `session.py` | Removed a duplicated nested `if self.active_speech_ms < 250.0:` |
| `khmer_normalizer.py` | Rewrote the scrambled, truncated comment block above `_INDEPENDENT_VOWEL_MAP` |
| Part 1 ledger | Folded the redundant S-1..S-8 entries (57..64) into entries 37..56 as "Second-pass confirmation", so each fix is described once; the ledger is now **1..56**, unique |
| Ledger cross-references | Fix A-1: entry 36 -> **48**; Fix A-2: entry 30 -> **42**; Fix A-3: entry 26 -> **38** |
| Part 8 STAGE 00 step 2 | "Part 2.3 of the v10.6 Blueprint" -> **v10.7** |
| `README.md` | "copy-pasteable v10.6 Master Orchestrator prompt" -> **v10.7**; added **item 5** for `docs/05_CODE_LEVEL_AUDIT_AND_REMEDIATION.md` |

### 14.6 Validation after the fixes

```
V1  V2  V4  V8  V8b  V10  V12  V13  V14  V16  V19  V20  V21  V22  V23  V23a  V24
ALL: PASS (17/17 checks)
Gates: 33 passed (Gate 2 gained the CFG test)
Rust: 6 binaries green;  tsc --noEmit: exit 0
```

### 14.7 Defects in my own work this session

1. **A botched brace edit in `session.py`.** The first attempt to remove the
   duplicated `if` dedented the wrong range and left the file with an
   `IndentationError` and a duplicated body. It was caught by parsing the
   result and reading the region back, then repaired by replacing the whole
   damaged range with the correct eight lines.
2. **A broken Rust edit.** The first attempt to reorder the Rust function
   extracted the map block with a naive `index("    }\n")`, which stopped at
   the inner `if` block's closing brace and left the outer `for` unclosed. The
   file was rewritten wholesale from a known-good reconstruction and
   re-verified with `cargo test`.
3. **A lost label during the ledger fold.** The S-7 prose was split between
   entries 51 and 52, and the split dropped the "Second-pass confirmation
   (S-7):" label from entry 51 and left trainer-specific sentences in entry
   52. Both were corrected by inspecting the rendered text rather than the
   diff.
4. **A stale V24 expectation.** After the fold the ledger is 1..56, but V24
   still expected 1..64 and reported FAIL. The expectation was updated.


---

## 15. Session: final polish pass on v10.7 (4 points)

**Trigger.** A fourth audit of the published v10.7 found four remaining
polish items, two of them audible DSP defects. All four were confirmed
against the files before fixing.

### 15.1 P1 - the Hann window was tapering every audio block to zero

`ScriptProcessorNode(4096, 2, 2)` delivers **non-overlapping** 4096-sample
blocks, and `N_FFT` equals the block size, so the frontend has no
overlap-add and no inverse-window compensation anywhere in the path.
`_computeSpectrum` nevertheless applied a Hann window, which tapers the
first and last sample of **every** block to zero. Under an active ERB mask
the reconstruction came back with a hard zero at both block edges.

The fix uses UNITY (rectangular) windowing, making the forward/inverse pair
the exact identity on the block, and removes the unused `const HOP = 480`.

The round trip was executed under node on a 4096-sample sine:

| | max abs error | energy ratio | last sample out | last sample in |
| :--- | ---: | ---: | ---: | ---: |
| Hann (before) | 4.881e-1 | 0.374908499 | **0.000000000** | -0.003834914 |
| unity (after) | **1.625e-16** | **1.000000000** | **-0.003834914** | -0.003834914 |

The `LAST` column is the defect in one line: with the Hann window the last
sample of the block was silently zeroed. **V23a** now asserts both the
energy ratio (1.000000000) and per-sample equality (max abs error
< 1e-9) against the unwindowed input, so it fails against the defective
version.

### 15.2 P0 - deferred tail syllables were not true no-ops

The Hibiki wait-gating zeroed `bus_l` **before** the head projections.
That is not sufficient: `nn.Linear` has a bias vector, so a zero input
still produces a non-zero logit vector and the argmax returns a real onset
id. `decode_batch_to_text` skips a syllable only when the triplet is
exactly `(0, 0, 0)`, so the deferred tail syllables were emitted as real
content.

v10.7 applies `wait_mask` to the **argmaxed** `onset_ids`, `vowel_ids` and
`coda_ids` directly. Measured on the real model with the wait threshold
forced negative:

| | triplets | no-op count | decoded |
| :--- | :--- | ---: | :--- |
| before | `[(5,37,2), (61,19,21), (19,29,17), (19,29,17), (19,29,17)]` | 0 | 5 syllables (3 hallucinated) |
| after | `[(61,19,21), (0,0,0), (0,0,0), (0,0,0)]` | **3** | 2 syllables |

**V25** and a new Gate 6 test assert the `(0, 0, 0)` triplet, the deferred
count, and the shortened decode.

### 15.3 P2 - prose contradictions

- Part 1 entry 53 claimed the ledger was "renumbered sequentially from 1
  to 64" after the redundant entries had been folded. It now says **1 to
  56**.
- `WORKLOG.md` sections 13.1, 13.3 and 13.5 carried three mojibake `?`
  characters where em-dashes belonged. All three are now em-dashes.
- The worklog header, section 5, section 7 and section 12 now consistently
  report **18/18** checks, **34 passed**, and a **1..56** ledger.

### 15.4 P2 - workspace state aligned with the README

The README note claimed `backend/` and `frontend/src/` did not exist, but
both **are** materialised on disk by `__v107_materialize.py`. The note now
states that the repository root is the blueprint, that the materialised tree
is **generated rather than hand-written**, and that files under `backend/`
and `frontend/src/` must never be edited directly - the blueprint block
they came from is the source of truth.

### 15.5 Validation after the polish pass

```
ALL: PASS (18/18 checks)
Gates: 34 passed (Gate 6 gained the deferred no-op test)
Rust: 6 binaries green;  tsc --noEmit: exit 0
Idempotent: b2ab3bb4bd82e648 -> b2ab3bb4bd82e648
```

### 15.6 Defects in my own work this session

1. **A dropped closing brace in the mixer.** The windowing replacement
   removed one `}` too many, which left `_computeSpectrum` truncated. It
   was caught because the extracted method stopped at the wrong line and the
   node round trip returned all zeros. The orphan brace was removed and the
   round trip then measured `maxAbsErr=1.6e-16`.
2. **Two mis-targeted edits.** The first two attempts to replace the
   windowing block used off-by-one and off-by-two line offsets and both
   aborted on their own assertions rather than writing a wrong file. The
   offsets were re-derived from the actual file content.
3. **A stale expectation in V24.** After the polish pass the ledger is
   1..60, but V24 still expected 1..56 and reported FAIL. It was updated.
4. **A self-referential string.** The new ledger entry 59 quoted the stale
   phrase "from 1 to 64" verbatim, which made the V24 hygiene check report
   a residual occurrence. The entry was rephrased so the check stays strict
   rather than being weakened with an exception.


---

## 16. Session: the seven AUD-107 code-level findings (3 High, 3 Medium, 1 Low)

**Trigger.** A line-by-line code and architectural audit of the v10.7
reference implementations found seven further defects across the neural
cartridges, the full-duplex runner, the WebAudio lookahead topology and the
checkpointing wrapper. All seven were confirmed against the files - and
several were confirmed by EXECUTION rather than by reading - before fixing.

### 16.1 AUD-107-01 (High) - wait-gating defeated on KM -> EN, and deferred syllables consumed frames

The v10.7 no-op mask was applied **before** the `if target_lang == "en"`
branch, but that branch overwrites `onset_ids`, `vowel_ids` and `coda_ids`
with the reserved English ranges, so the deferred tail was never masked on a
KM -> EN slice. Separately, `sesqui_ids` was never zeroed, so the aligner
still gave a deferred `(0, 0, 0)` syllable a duration weight.

Measured before the fix:

| path | triplets | no-ops |
| :--- | :--- | ---: |
| KM | `[(5,37,2), (61,19,21), (0,0,0), (0,0,0), (0,0,0)]` | 3 |
| EN | `[(49,37,31), (37,37,31), (49,29,18), (49,29,18), (49,29,18)]` | **0** |

After the fix, both paths produce three true `(0, 0, 0)` no-ops, and the
deferred durations are `[20.238, 3.762, 0.0, 0.0, 0.0]` - the deferred
syllables consume **zero** acoustic frames.

The mask is now applied after the language branch, covers `coeng_ids` and
`sesqui_ids` as well, and `LearnedDurationAligner` gives `sesqui_ids == 0` a
zero duration weight.

### 16.2 AUD-107-02 (High) - the ear and the emotion encoder ran twice

The runner ran the ear, then `log_mel` twice more, then the emotion encoder
whose output was DISCARDED, and then called `forward_s2st`, which ran the
ear and the emotion encoder again on the identical audio.

v10.7 introduces `PrecomputedEarFeatures`, computes the bundle once, and
passes it to `forward_s2st`. Measured: the extra ear calls drop from 1 to 0
and the extra log-mel calls from 2 to 0. The emotion encoder still runs once,
which is correct - `emo_tokens` is genuinely needed for the CFG branch.

### 16.3 AUD-107-03 (High) - the 2.00s lookahead DelayNode was missing

Part 2.1 specifies a `[2.00s Delay Line]` before the ERB suppressor, but the
source was connected straight to the processor, so the suppression was
always late by the whole lookahead window. v10.7 inserts the `DelayNode` and
still returns the **undelayed** source so the capture tap stays ahead of
playback.

### 16.4 AUD-107-04 (Medium) - gradient checkpointing renamed 70 of 178 state_dict keys

This was confirmed by execution, not inspection: wrapping each block in a
`_CheckpointedBlock` renamed every key from `ear.blocks.0.self_attn...` to
`ear.blocks.0.blk.self_attn...`. Measured: **70 of 178 keys renamed**, so a
laptop-profile checkpoint could not load under the cloud profile.

v10.7 replaces the module wrapper with an instance-level `forward` override.
Measured after the fix: 178 keys before, 178 after, **identical sets**, and
the forward output is bit-identical (`max abs diff = 0.0`).

### 16.5 AUD-107-05 (Medium) - per-block FFT allocations and a dead spectral pass

The twiddle tables were allocated per block with `Math.log2(size)` called
98,304 times per block. They are now precomputed at module load, and the
dead `processMidSideFrame` call in `handleDownstreamEvent` - which wrote a
mask into a buffer the next audio callback overwrites - is removed.

### 16.6 AUD-107-06 (Medium) - the capture worklet was wired to the destination

The worklet is now routed through a `GainNode` with `gain = 0.0`, so a
future pass-through WASM worklet cannot bleed dry audio past the suppressor.

### 16.7 AUD-107-07 (Low) - the provenance ledger was split

Entries 1..30 sat above Part 1 while entries 31..67 were pasted INSIDE Part
1 after Section 1.1. The whole ledger now lives in one place, immediately
under the document-provenance header, and Part 1 begins with the
architectural overview it is supposed to contain. The duplicated
`assert not errors, errors[0]` in Gate 6 and the unused `clockOffsetMs`
field in `PcmCaptureProcessor` are removed.

### 16.8 Validation after the seven fixes

```
ALL: PASS (19/19 checks)      (V26 added: all seven findings, executed)
Gates: 37 passed (Gate 5 +1, Gate 6 +2)
Rust: 6 binaries green;  tsc --noEmit: exit 0
Idempotent: 790f2d10da2a1e0f -> 790f2d10da2a1e0f
Ledger: 1..67, contiguous, entirely before Part 1
```

### 16.9 Defects in my own work this session

1. **The `@runtime_checkable` decorator was orphaned.** Inserting the
   `PrecomputedEarFeatures` dataclass between the decorator and its protocol
   class made every gate fail to import with
   `TypeError: @runtime_checkable can be only applied to protocol classes`.
   It was caught by running the gates, and the dataclass was moved above the
   decorator.
2. **A table block was inserted inside an import statement.** The FFT tables
   were placed between `import {` and the names it imported. The file was
   repaired by moving the whole block after the import, and the placement was
   then checked by eye.
3. **Two off-by-one anchors.** The windowing and table edits each missed
   their target by one or two lines and aborted on their own assertions
   rather than writing a wrong file. The offsets were re-derived from the
   actual file content.
4. **A first-draft Gate 5 test compared two different model instances**, so
   it compared different random weights and failed. It was rewritten to
   compare the SAME instance before and after checkpointing.
5. **That same test polluted the global RNG**, which made the pre-existing
   VTLP test fail when run in the same file. The test now saves and restores
   the RNG state, and the VTLP test passes again.


---

## 17. Session: the final four post-audit remediations (AUD-107-08/09/10)

**Trigger.** A post-audit of the published v10.7 required three further code
changes plus a Part 8 synchronization, and the provenance ledger to grow to
entries 68-70. The master was regenerated **in place** (no new filename), per
the user's decision.

### 17.1 AUD-107-09 - deferred syllables stole attention mass

A deferred syllable has a zero duration, so its cumulative centre sits at
`target_frames`, but it still participated in the Gaussian softmax. Measured
before the fix, with `sesqui_ids = [[2,2,0,0]]`, `sigma = 1.5`,
`target_frames = 24`:

| | syllable 0 | syllable 1 | deferred 2 | deferred 3 |
| :--- | ---: | ---: | ---: | ---: |
| before | 12.000000 | 8.742891 | **1.628554** | **1.628554** |
| after | 12.000000 | 12.000000 | **0.0** | **0.0** |

The fix masks the `sesqui_ids == 0` columns to `-1e9` before the softmax,
guarded by `has_active` so a fully deferred row stays well defined.

### 17.2 AUD-107-10 - the capture worklet assumed a 48 kHz context

The worklet hard-coded a 3:1 decimation, so a 44.1 kHz device emitted
14,700 Hz. It now reads the real `AudioWorkletGlobalScope.sampleRate`, keeps
the 15-tap Blackman-sinc as the 48 kHz fast path, and uses a fractional
linear-interpolation resampler otherwise, with `outBuf` raised from 64 to
128 entries (the fallback needs up to 128 outputs per 128-frame quantum).

Measured: exactly 16,000 output samples/s at 22.05k / 24k / 32k / 44.1k / 48k /
88.2k / 96k / 192k, **zero dropped samples** at every rate, and a 1,000 Hz
input peaking at exactly **1,000.0 Hz**, so there is no pitch or speed drift.

### 17.3 AUD-107-08 - Part 8 was out of sync with Section 4

Specialist D item 9 still mandated resolving the ERB mask at
`lookaheadTargetTime`, which the 2.00 s DelayNode (AUD-107-03) invalidated,
and Part 8 omitted AUD-107-01 through AUD-107-06 and AUD-107-09 entirely -
so an implementer following the copy-pasteable prompt would have reproduced
five of the defects the audit had just closed. Specialist A gained items 11-13,
Specialist C gained item 5, and Specialist D item 9 was replaced with the
AUD-107-03/05/06 requirements plus a new item 10.

### 17.4 Two defects in the incoming specification, found by execution

1. **The AUD-107-09 test as literally specified could not pass.** It compared a
   4-token call against a 2-token call using **independently drawn** random
   hidden states, so the two calls differed regardless of the attention mask
   (measured max diff 3.67 even against a correct implementation). The plan
   specified the corrected form: the 2-token reference **reuses the first two
   rows of the same hidden tensor**. This is the same class of mistake caught
   earlier in this project, where a supplied test case did not discriminate the
   defect it was meant to catch.
2. **A measurement-window artefact in my own V27 check.** The first version
   measured the resampler rate over a fixed 800-quantum window, which is only
   0.53 s at 192 kHz, and reported 16001.25 Hz. That is a window artefact -
   the accumulator emits a whole number of samples per quantum - and the rate
   converges to 16000.0125 Hz over 53 s. The check was corrected to size the
   window to a constant 5 s at every rate.

### 17.5 Validation after the four remediations

```
ALL: PASS (21/21 checks)      (V27 and V28 added)
Gates: 38 passed (Gate 4 gained the deferred-attention test)
Rust: 6 binaries green;  tsc --noEmit: exit 0
Idempotent: e14e6bfb1b9c346d -> e14e6bfb1b9c346d
Ledger: 1..70, contiguous, entirely before Part 1
```

**Discriminating power confirmed by re-running against the pre-fix code:**
the Gate 4 test differs by **999,373.5** against the pre-fix aligner and
**1.3e-06** against the fixed one, and every one of V27's six static
worklet assertions is false against the true pre-fix worklet.

### 17.6 Defects in my own work this session

1. **A plan error caught by cross-checking against the real files.** The plan
   stated that `fix_ledger.txt` would hold entries 31..70; it actually holds
   37..67, with 1..36 supplied by the v10.6 source. The plan was corrected
   before any code was written.
2. **A dropped closing brace in the worklet.** The first replacement of the
   `process()` loop matched an inner `}` and left the block malformed. It was
   caught by the assertion in the edit script, and the replacement was
   redone against the exact line boundaries.
3. **A mis-targeted edit in `stage02.txt`.** The first attempt to replace
   Specialist D item 9 asserted on text that included a trailing blank line
   and aborted without writing. The bounds were re-derived from the file.


---

## 18. Session: Phase 2 Rust completion and the AUD-09 filterbank alignment

**Trigger.** A post-audit required the AUD-09 filterbank alignment, the
completion of all three Rust crates (Deliverables R-1..R-8), and the
provenance ledger to grow to entries 71-74. The version stayed `10.7.0` and
the parameter ledger stayed exactly `241,637,457` / `3,838,033`.

### 18.1 AUD-09 - the filterbank was ALREADY torchaudio-compatible

The v10.6 note claimed a 0.025 / 0.020 element-wise divergence and a
half-width normalization root cause. **Neither reproduces.** Every revision
from v10.4 onward normalized by the FULL filter width that torchaudio uses,
and the measured divergence against torchaudio 2.11 is **0.000000000** on
both the 513-bin and the 1025-bin filterbank, with `matrix_rank` exactly
**100**. A sweep of five candidate root causes (half-width, mel_scale
slaney, integer bin half, and their combinations) reproduced 0.049 / 0.041 at
best, never 0.025 / 0.020.

The remediation was still worth making, for a different reason: the
implementation REIMPLEMENTED torchaudio's algorithm, so the agreement was
coincidental rather than structural. It is now DELEGATED to
`torchaudio.functional.melscale_fbanks`, with a numerically identical
pure-PyTorch fallback (verified by blocking the import) and an explicit
rank guarantee.

### 18.2 The three crates

| Deliverable | What was delivered |
| :--- | :--- |
| R-1, R-2, R-3 | `khmer-linguistics-rs` split into `normalizer.rs` and `decomposer.rs`; a real `#[pymodule]`; `once_cell::sync::Lazy` compiles the regexes once; the 17-entry vowel map runs before the reordering |
| R-4, R-7 | `khmer-transport-rs` split into `session.rs`, `live_queue.rs` and `wav_container.rs`; a real `#[pymodule]` with the fence as a `#[pyclass]`; `crossbeam_channel::unbounded` backs the queue; the energy hash and WAV encoder match Python exactly |
| R-5, R-6 | `khmer-dsp-wasm` gains `erb_band_index_for_freq` and the stateful `SincDecimator48To16` |

**Parity was measured, not assumed.** The Rust normalizer agrees with the
Python one on 8 cases; the energy hash matches the Python values
3282682638 / 2779605271 / 491666418; and the Rust WAV output is
**byte-identical** to the Python encoder. `crc32fast::hash` was confirmed
bit-identical to Python's `zlib.crc32` before it was relied on.

All five previously unused workspace dependencies are now genuinely used:
**zero unused dependencies**.

### 18.3 Defects in my own work this session

1. **The AUD-09 claim was wrong, and I said so.** The plan to "fix" a 0.025
   divergence would have been built on a false premise. It was measured
   first, found to be 0.000000000, and the ledger entry was rewritten to
   state the measured truth.
2. **A wrong WAV header layout.** My first Rust encoder put "fmt " at
   offset 12 and the chunk size as a `u16`, instead of the Python's
   "WAVEfmt " at offset 8 and a `u32`. The Rust unit test caught it; the
   header is now byte-identical to the Python output.
3. **A wrong CRC constant in a test.** I hand-computed `zlib.crc32` as
   2755087938 instead of 2756264514. The Rust was right and the test was
   wrong. The test now uses values measured from Python.
4. **An invalid phase-invariance assertion.** I asserted that two different
   smooth signals hash differently. They do not: the 8-bin profile is
   deliberately coarse, which is exactly WHY it is shift-invariant. The test
   now compares a skewed energy distribution.
5. **A DC-gain test that ignored the filter warm-up.** The first 5 outputs
   are the 15-sample ring buffer filling. The test now asserts the warm-up
   explicitly and unity gain after it.
6. **A ```` ``` ```` fence inside a block file.** The Rust doc-comments used
   a fenced code block, which terminated the master's code fence early and
   silently truncated the published source. The `section()` helper caught it.
   A guard now rejects any block file containing a fence, so this cannot
   recur.
7. **A stale V4 parser.** The vowel map moved from `lib.rs` to
   `normalizer.rs`, so V4 read the wrong section and reported a false
   failure. It now reads Section 4.9.4.1.
8. **A too-broad mojibake check.** V24 flagged any comment containing a
   single `?`, which is legitimate in Rust doc-comments and TypeScript
   optional properties. It now targets LOST characters (runs of `?`).

### 18.4 Validation after the Phase 2 completion

```
ALL: PASS (22/22 checks)      (V29 added: crate parity + deliverables)
Gates: 38 passed
Rust: 4 test binaries green, no unused-dependency warnings
tsc --noEmit: exit 0
Idempotent: a11f941a161fbb68 -> a11f941a161fbb68
Ledger: 241,637,457 / 3,838,033 (UNCHANGED);  version 10.7.0 (UNCHANGED)
Fix ledger: 1..74, contiguous, entirely before Part 1
```


---

## 19. Session: three-item polish pass on the Phase 2 Rust delivery

**Trigger.** A final polish pass on the published v10.7: expose the interrupt
fence under its Rust name, remove two duplicated doc-comment blocks, correct a
stale byte-offset comment, and bring the Phase 2 status prose up to date.

### 19.1 The `AtomicInterruptFence` alias (R-4)

`khmer_transport_rs` exposed the fence only as `PyInterruptFence`. The
`#[pymodule]` now also registers it under its Rust name, so
`hasattr(khmer_transport_rs, "AtomicInterruptFence")` and
`hasattr(khmer_transport_rs, "PyInterruptFence")` both succeed.

**Verified by actually importing the built cdylib from Python** (copied to a
`.pyd` so the interpreter can load it):

```
hasattr AtomicInterruptFence   : True
hasattr PyInterruptFence      : True
hasattr compute_pcm_energy_hash: True
hasattr encode_wav_pcm16      : True
fence load/bump               : 0 1 1
energy hash (ones)            : 3282682638
wav header                    : b'RIFF(\x00\x00\x00WAVEfmt '
```

The linguistics crate was checked the same way: `normalize_khmer` and
`normalize_khmer_pure` both resolve, and `normalize_khmer` on the compound
word U+17AB U+1791 U+17D2 U+1792 U+17B7 returns
`179A 17B9 1791 17D2 1792 17B7`, identical to the Python normalizer.

### 19.2 The duplicated doc-comments and the byte offsets

Both `decomposer.rs` and `wav_container.rs` carried a triplicated doc
comment, left over from the fence-guard repair in session 18. Each is now a
single copy.

The `wav_container.rs` byte-offset comment was also wrong: it described the
pre-correction layout (`12..16 "fmt "`, `16..18 u16 16`, `38..42 data_len`),
while the code writes the Python-compatible layout. The corrected comment
was verified by PARSING the `extend_from_slice` calls and recomputing every
offset from the code:

```
 0..4   b"RIFF"          4..8   data_len         8..16  b"WAVEfmt "
16..20  16u32            20..22 1u16             22..24 1u16
24..28  sample_rate      28..32 byte_rate       32..34 2u16
34..36  16u16            36..40 b"data"          40..44 data_len
total header bytes: 44
```

### 19.3 The Phase 2 status prose

Section 4.9.7, Part 2.2 and Part 7 now state that R-1 to R-8 are DELIVERED
in Sections 4.9.4 to 4.9.6 (fix-ledger entries 72 to 74). The **Finding**
paragraph was rewritten in the same operation, because it still claimed that
"no member crate currently exposes a `#[pymodule]`" and that "this probe can
never succeed", which the delivered ABI makes false.

The R-1 to R-8 table was rewritten to record what was ACTUALLY delivered
rather than the stale v10.6 wording (which asked for a `wasm-bindgen`
binding, a `pyo3` `extension-module` feature for the DSP crate, and the
REMOVAL of the `crc32fast` and `crossbeam-channel` dependencies - the last
two being the opposite of what was done). The items that remain genuinely
outstanding, namely the `wasm-bindgen` binding and the DSP crate's
`extension-module` feature, are recorded explicitly as NOT delivered.

### 19.4 Validation after the polish pass

```
ALL: PASS (22/22 checks)
Gates: 38 passed
Rust: 4 test binaries green, no warnings
tsc --noEmit: exit 0
Idempotent: 9601b6cf29e74a3b -> 9601b6cf29e74a3b
```

### 19.5 Defects in my own work this session

1. **A backslash-escaping bug in the generator.** The R-1 to R-8 table
   contains the literal text `\u17A3`, and embedding it into the
   generator's Python string literal made Python interpret it as a unicode
   escape, so the `old` text no longer matched the file. The generator now
   DOUBLES the backslashes when embedding blueprint text.
2. **Six piecemeal note replacements** were replaced by a single operation,
   because the piecewise version left a duplicated trailing line and could
   not keep the note and the Finding paragraph consistent.
3. **The `old` text for a `rep()` must come from the v10.6 SOURCE**, not
   from the already-modified master, or the replacement never matches. This
   was diagnosed after the first attempt failed with "found 0".


---

# Session â€” STAGE 07 Self-Auditing Bootstrap (Rust ABI, Vocos Warm-Start, Pilot Shard, RTX 5070 Ti)

**Scope:** Executed the four mandated STAGE 07 steps end-to-end, built three
new self-auditing scripts, and recorded the findings. This session did NOT
modify any locked contract (`backend/contracts/schemas.py`,
`frontend/src/contracts/types.ts`) and did NOT weaken or alter any of the 38
tests in `backend/tests/`.

**Outcome:** All four step gates PASS. 8 findings recorded (1 CRITICAL,
3 MEDIUM, 4 LOW), of which 3 were latent defects in the *shipped* code and
2 were defects in the *new* scripts written during this session.

---

## 0. Overview

| Phase | Nature |
| :--- | :--- |
| Survey of available agent skills | Tooling discovery (GitHub + local marketplace) |
| STEP 1 | Git hygiene + native Rust PyO3 ABI activation |
| STEP 2 | Surgical `VocosISTFTHead` warm-start + mel-to-wav round-trip audit |
| STEP 3 | Streaming pilot Khmer shard + 7-slot pre-tokenization |
| STEP 4 | 25-step Stage-2 / 5A smoke test on the RTX 5070 Ti |
| Adversarial review | Fresh-context mutation testing of the three new gates |
| Worklog | This document |

## 1. Method

- **Findings were established by EXECUTION, not by reading.** Every number
  quoted below was re-read from the artifact on disk or from the run's own
  output.
- **Binding constraints honoured:**
  - No modification to the locked contracts or the 38 tests -> their file
    mtimes (10/21/2026 1:48:48 AM) still predate this session, and
    `pytest backend/tests/` reports `38 passed` with the Rust ABI active.
  - No bulk dataset download -> `datasets.load_dataset(..., streaming=True)`
    with a strict `--max-clips` cap; peak local disk usage for the pilot is
    the 260.43 MiB manifest.
  - Step-by-step self-audit gates, each executed and printed before the next
    step began.
- **Discriminating checks.** Per the "a test must be able to fail" rule, the
  six principal gates were mutation-tested: each gate expression was
  evaluated against a known-correct value both as written and as a stricter
  mutant. All six discriminate.

## 2. Phase-by-phase

### STEP 1 â€” Git hygiene + Rust ABI activation

| Finding | Remediation |
| :--- | :--- |
| `STAGE-07-01` (CRITICAL): the Rust PyO3 ABI was **never deployed**. The crates compiled to `target/release/*.dll` but neither was importable, so `_HAS_RUST` was `False` and the "15 microsecond Rust edge" was unreachable dead code. | Built both crates with `--features extension-module` and installed the `cdylib`s into site-packages as `khmer_linguistics_rs.pyd` / `khmer_transport_rs.pyd`. |

Created `.gitignore` at the workspace root covering `.venv/`, `__pycache__/`,
`*.pyc`, `target/`, `node_modules/`, `dist/`, `.pytest_cache/`,
`checkpoints/`, `data/`, `*.pt`, `*.pth`, `*.safetensors`, `*.onnx`, `*.wav`,
`*.Parquet` (plus `*.pyd`/`*.so` for the locally built extensions).

### STEP 2 â€” Surgical Vocos warm-start

| Finding | Remediation |
| :--- | :--- |
| `STAGE-07-02` (MEDIUM): `VocosISTFTHead.forward` runs `torch.istft(..., center=True, length=T*HOP)`. A forward `center=True` STFT of `N` samples yields `ceil(N/HOP)+1` frames, so the inverse emits `ceil(N/HOP)*HOP` samples â€” **24 240** for a 1.0 s signal, not 24 000. The trailing hop is the zero-padded centre frame. | Added `_align_length()` in the new script, which trims the hop the same way the upstream Vocos reference implementation does. The gate now asserts exact sample alignment. |

### STEP 3 â€” Streaming pilot Khmer shard

| Finding | Remediation |
| :--- | :--- |
| `STAGE-07-06` (LOW): `datasets` decodes `Audio` columns through `torchcodec`, which needs the FFmpeg shared libraries. Without them the first `next()` raises `RuntimeError: Could not load libtorchcodec`. | Cast the audio column to `Audio(decode=False)` and decode the raw bytes with `soundfile`. FFmpeg is no longer a dependency. |
| `STAGE-07-07` (LOW): `huggingface_hub` reads `HF_HUB_OFFLINE` into a **module constant** at import time. Setting `os.environ["HF_HUB_OFFLINE"]="1"` after the import leaves the constant `False` and the iterator silently goes to the network. | `_set_offline()` patches the constants of BOTH `huggingface_hub.constants` and `datasets.config`, and keeps the patch alive for the whole lifetime of the dataset (Parquet shards resolve lazily on the first `next()`). |
| `STAGE-07-08` (LOW): `phonsobon/openslr42-khmer-male` is **gated**; `seanghay/km-speech-corpus` needs a ~475 MB first row group. Only `google/fleurs` `km_kh` works unauthenticated. | Candidate resolver probes each candidate for its FIRST row under a wall-clock timeout on a daemon thread, and prefers the local cache when the data is already present. |

### STEP 4 â€” 25-step Stage-2 smoke test

| Finding | Remediation |
| :--- | :--- |
| `STAGE-07-03` (MEDIUM): a **fully GPU-resident** console peaks at **3.2867 GiB**, exceeding the 3.20 GiB ceiling. `compute_flow_loss(..., infill_mode=True)` never calls `console.ear` (64.5M params) or `console.pointer` (5.0M params); those 69.5M parameters receive no gradient and are pure VRAM ballast (0.2638 GiB of the peak). | Parked the two modules on the CPU for the training phase, moved back for the end-to-end inference check. Trained weights are **identical** either way. The fully-resident figure is measured in an isolated subprocess and **disclosed** in the report, not hidden. |
| `STAGE-07-04` (MEDIUM): the flow-matching loss is **dominated by sampler noise**. `compute_shortcut_distillation_loss` draws `x_0 ~ N(0,I)` and `t ~ U(0,1)` internally. With a per-step-varying seed the first probe read 4.12, 3.87, 4.05, ... 11.99 â€” the *sign* of the step-1-to-step-25 change was an artefact. | Re-seeded the flow-matching noise identically at every step, so the comparison measures the optimizer update and nothing else. |
| `STAGE-07-05` (LOW): the `laptop_5070ti_12gb` profile sets `grad_accum_steps=16`, so 25 forward/backward passes contain exactly **ONE** AdamW update and the loss trajectory is a step function, not a descent curve. | Kept the mandated step-1 vs step-25 figures AND added a supplementary `grad_accum=1` diagnostic in an isolated subprocess, which shows a genuine descent. |

## 3. Tooling

| Script | Purpose | Re-run |
| :--- | :--- | :--- |
| `scripts/bootstrap_vocos_warmstart.py` | 14 013 B | `python scripts/bootstrap_vocos_warmstart.py` |
| `scripts/prepare_khmer_pilot_shard.py` | 23 001 B | `python scripts/prepare_khmer_pilot_shard.py --max-clips 200` |
| `scripts/smoke_train_stage2_infill.py` | 30 835 B | `python scripts/smoke_train_stage2_infill.py` |

Generated artefacts: `checkpoints/vocos_24k_warmstart.pt` (29.60 MiB, 68
tensors), `checkpoints/stage2_smoke_test.pt` (896.5 MiB, 735 tensors),
`checkpoints/smoke_sample_out.wav` (72 044 B),
`data/pilot_khmer_shard/pilot_manifest.pt` (260.43 MiB, 200 clips).

## 4. Verification

Every check below was executed in this session.

| Check | Command | Result |
| :--- | :--- | :--- |
| Rust ABI live | `python -c "import khmer_linguistics_rs, khmer_transport_rs; ... _HAS_RUST ..."` | `STEP 1 PASS: _HAS_RUST=True, Rust PyO3 ABI active` |
| Test suite | `python -m pytest backend/tests/ -q` | `38 passed, 1 warning in 5.26s` |
| Vocos params | STEP 2 gate (a) | `7,753,218` (expected `7,753,218`) |
| Reconstruction shape | STEP 2 gate (b) | `(1, 24000)` == `(1, 24000)` |
| Reconstruction amplitude | STEP 2 gate (c) | `max|wav_recon| = 60.227093` (> 0.05), all finite |
| Round-trip error | STEP 2 gate (d) | warm `2.271349` vs random `6.904077` = **3.040x** |
| Vocos strict reload | STEP 2 | 68 tensors, bit-identical |
| Clip count | STEP 3 gate (a) | `200` (>= 50) |
| Disk size | STEP 3 gate (a) | `260.43 MiB` (< 500 MB) |
| NaN/Inf | STEP 3 gate (b) | `0` clips |
| 16k/24k duration skew | STEP 3 gate (b) | `0.0000 ms` (<= 1.0 ms) |
| Slot bounds | STEP 3 gate (c) | `onset=[1,33] coeng=[0,63] medial=[0,3] register=[1,3] vowel=[0,24] coda=[0,31] sesqui=[1,2]` |
| Khmer round-trip | STEP 3 gate (d) | 5/5 valid non-empty UTF-8 Khmer |
| Loss descent | STEP 4 gate (a) | `22.906364 -> 10.471284` |
| All losses finite | STEP 4 gate (a) | `True` (25 values) |
| Peak VRAM | STEP 4 gate (b) | `3.0229 GB` (<= 3.20) |
| Strict reload | STEP 4 gate (c) | `735 == 735` keys |
| `.blk.` pollution | STEP 4 gate (c) | `0` keys (AUD-107-04 holds) |
| WAV header | STEP 4 gate (d) | RIFF/WAVE, 1ch, 24000 Hz, 16 bit, 72 044 B |
| Gate discrimination | mutation harness | 6/6 gates accept correct input, reject mutants |

Corroborating checks: `soundfile` independently re-read
`smoke_sample_out.wav` (shape `(36000,)`, 24000 Hz, all finite, peak
0.999969); `git check-ignore` confirmed every heavy path is ignored while
120 source files remain trackable.

## 5. Decisions and rationale

- **Park `ear`/`pointer` on the CPU during training rather than shrinking the
  batch or the model.** The infill path never calls them, so this changes no
  gradient and no trained weight. Alternatives rejected: reducing the batch
  below the mandated 4, and reporting the 3.2867 GiB figure as the gate value
  (which would fail the mandated ceiling and hide a real finding).
- **Measure the fully-resident peak in a subprocess.** A same-process probe
  is invalid: `apply_gradient_checkpointing` binds
  `blk.forward = types.MethodType(...)`, creating a reference CYCLE
  (block -> bound method -> block) that keeps the probe's VRAM alive after
  `del`. The first attempt at this measurement was contaminated by ~1.3 GiB
  and produced a meaningless 4.3182 GB; the subprocess isolation fixed it.
- **Re-seed the flow-matching noise per step.** This is a measurement
  decision, not a training decision: the mandated 25 steps still use the
  production profile unchanged.
- **Report the `grad_accum=1` descent as a DIAGNOSTIC, not as the gate
  value.** The gate reports what the profile actually does (1 real update);
  the diagnostic shows the optimizer works.
- **Do NOT touch the locked contracts or the 38 tests**, even to add a
  regression test for the findings. The findings are recorded in
  `docs/04_AUDIT_FINDINGS_AND_REMEDIATION.md` and in this worklog instead.

## 6. Defects in my own work

1. **The offline flag leaked across candidate attempts** (found by the
   adversarial review, fixed). `_iter_candidates` set `HF_HUB_OFFLINE` for
   the offline attempt but only restored it on the success path, so the
   subsequent "online" attempt ran with the flag still ON and failed for
   the wrong reason. *How it happened:* the restore lived in a `finally`
   attached to the `yield`, which is only reached on success. *How it was
   caught:* a red-green probe that mocks `datasets.load_dataset` and asserts
   the offline flag is inactive on the second attempt. The probe first
   "passed" for the wrong reason because it replaced `_open_stream`
   wholesale; re-pointing it at the real `_open_stream` exposed the bug.
2. **My first mutation harness was wrong, not the code.** It compared assert
   COUNTS, but mutating a condition does not change the count, so all three
   mutations were misreported as "deleted a gate". It also mutated gates in
   the LOOSER direction, where the mutant still accepts a correct value â€”
   which proves nothing. The harness was rewritten to make the mutant
   STRICTER and to execute both versions against the real value. The
   underlying scripts were never at fault; the check was.
3. **A slot-bounds check in STEP 3 was wrong.** It used `bound[0]`/`bound[1]`
   on a 3-element tuple `(1, 2, 3)`, so `register_ids` maxing at 3 was
   rejected as "violating (1, 2, 3)". This was the CODE's fault, caught
   because the gate fired on real data. Fixed to test set membership.
4. **A stale background process polluted a log**, making a killed run look
   like it had reached the FLEURS candidate. Resolved by killing
   recently-started processes before each timed run and using unique log
   names.
5. **`torchcodec` was installed but unusable** (no FFmpeg DLLs). Rather than
   leave a broken dependency, the audio path was switched to
   `Audio(decode=False)` + `soundfile`.
6. **This worklog itself shipped with two errors, both corrected.** (a) The
   file-inventory table quoted sizes taken from memory rather than from disk
   (`13 907`, `22 619`, `30 806`, `78 137` B) and five of the nine rows were
   wrong. Corrected against `stat()` on the artifacts: `14 013`, `23 001`,
   `30 835`, `27 763` B. (b) The appended section was written with CRLF
   line endings (245 CRLF) because it was appended through a PowerShell
   heredoc, which also risks silent non-ASCII corruption. The file is now
   UTF-8, no BOM, LF throughout (`CRLF=0`, `LF=1321`), and was re-read to
   confirm there is no `U+FFFD` and that the heading order is intact.

## 7. Known limitations / still open

- The pilot corpus resolves to `google/fleurs` `km_kh` because the primary
  OpenSLR-42 repo is **gated** and the Seanghay corpus needs a ~475 MB
  first row group. With an HF token the blueprint's intended corpus becomes
  available. This is a **dataset-availability** limitation, not a code one.
- `STAGE-07-03` is **disclosed, not fixed**: a fully GPU-resident console
  still exceeds 3.20 GiB. The fix belongs in the training recipe (or in the
  `laptop_5070ti_12gb` profile), which is a blueprint-level decision and was
  explicitly out of scope for this session.
- The `docs/04_AUDIT_FINDINGS_AND_REMEDIATION.md` ledger was appended to.
  The `WORKLOG.md` is **already** cross-referenced from `README.md` (line 17,
  `README.md` mtime 10/20/2026 12:17 PM, which predates this session), so
  no README change was needed or made.
- `pyright` / `ruff` / `mypy` are not installed on this machine, so the
  three new scripts were checked with `python -m py_compile` and the
  mutation harness only. No static type-check result is claimed.
- The `graphify` skill (installed locally) was **not** run: it would build a
  knowledge graph over the whole workspace, which is a large, separate
  deliverable from the four mandated steps.

## 8. File inventory

Verified against disk in this session:

| Path | Size | Status |
| :--- | ---: | :--- |
| `.gitignore` | 1 459 B | new |
| `scripts/bootstrap_vocos_warmstart.py` | 14 013 B | new |
| `scripts/prepare_khmer_pilot_shard.py` | 23 001 B | new |
| `scripts/smoke_train_stage2_infill.py` | 30 835 B | new |
| `docs/04_AUDIT_FINDINGS_AND_REMEDIATION.md` | 27 763 B | appended (STAGE 07 section) |
| `WORKLOG.md` | this file | appended |
| `checkpoints/vocos_24k_warmstart.pt` | 29.60 MiB | generated |
| `checkpoints/stage2_smoke_test.pt` | 896.54 MiB | generated |
| `checkpoints/smoke_sample_out.wav` | 72 044 B | generated |
| `data/pilot_khmer_shard/pilot_manifest.pt` | 260.43 MiB | generated |

Unchanged (locked): `backend/contracts/schemas.py`,
`frontend/src/contracts/types.ts`, all 6 files in `backend/tests/`.

## 9. Reproduction

```bash
cd "KHMER S2ST"

# STEP 0: build + install the Rust ABIs
cargo build --release -p khmer-linguistics-rs --features extension-module
cargo build --release -p khmer-transport-rs  --features extension-module
SP=$(python -c "import site; print(site.getsitepackages()[0])")
cp target/release/khmer_linguistics_rs.dll "$SP/khmer_linguistics_rs.pyd"
cp target/release/khmer_transport_rs.dll  "$SP/khmer_transport_rs.pyd"

# STEP 1 gate
python -c "import khmer_linguistics_rs, khmer_transport_rs; from backend.linguistics.khmer_normalizer import _HAS_RUST, normalize_khmer; assert _HAS_RUST is True; assert normalize_khmer('\u17AB\u1791\u17D2\u1792\u17B7') == '\u179A\u17B9\u1791\u17D2\u1792\u17B7'; print('STEP 1 PASS')"
python -m pytest backend/tests/ -q

# STEP 2 / 3 / 4
python scripts/bootstrap_vocos_warmstart.py
python scripts/prepare_khmer_pilot_shard.py --max-clips 200
python scripts/smoke_train_stage2_infill.py
```

Expected: `STEP 1 PASS`, `38 passed`, `STEP 2 PASS`, `STEP 3 PASS`,
`STEP 4 PASS`, with the STEP 4 table showing `3.0229 GB` peak VRAM and
`22.906364 -> 10.471284` loss.

---

## 10. STAGE 08 — Git Remote Wiring, Resumable Overnight Sharder & Stage 2 Pre-Training

**Date:** 2026-10-21  
**Scope:** Stage 08 execution on top of verified `v10.7.0` workspace.  
**Hardware Profile:** NVIDIA GeForce RTX 5070 Ti Laptop GPU (12 GB VRAM ceiling target: <= 3.20 GB).

### 10.1 Step 1: Git Remote Wiring & Verification
1. `.gitignore` audited: Verified exclusion of `.venv/`, `target/`, `node_modules/`, `data/`, `checkpoints/`, `*.pt`, `*.pth`, `*.safetensors`, `*.onnx`, `*.wav`, `__pycache__/`.
2. Git repository initialized at workspace root, branch set to `main`, remote wired:
   ```bash
   git remote add origin https://github.com/I-have-an-idea-67/Khmer-S2ST.git
   ```
3. Self-Audit Gate:
   - `git status --ignored --short` confirmed `checkpoints/`, `data/`, `target/`, `.pytest_cache/`, `*__pycache__/` are ignored (`!!`).
   - Confirmed zero `.pt` or `.wav` files staged.
   - Committed clean `v10.7.0` tree: `feat(v10.7.0): locked 241.637M S2ST architecture, Rust PyO3 ABI, and Stage 07 bootstrap` (120 files changed, 64,033 insertions).
   - Pushed successfully to `origin/main` (`git push -u origin main`).

### 10.2 Step 2: Crash-Proof, Resumable Overnight Khmer Sharder
1. Created `scripts/overnight_khmer_ingester.py` with multi-corpus streaming priority, PCM16 quantization (50% storage footprint reduction), 7-slot Rust decomposition, atomic shard flushing, and free disk space enforcement (`--min-free-gb 15.0`).
2. Self-Audit Gate (Pilot Dry-Run):
   - First run: `python scripts/overnight_khmer_ingester.py --shard-size 50 --max-clips 100`
     - Created `shard_00000.pt` (50 clips, 38.17 MiB, < 40 MB ceiling).
     - Created `shard_00001.pt` (50 clips, 38.57 MiB, < 40 MB ceiling).
     - Total clips saved: 100.
     - Audio tensors verified: `wav_16k` and `wav_24k` dtype `torch.int16`, 1-D mono.
     - Linguistic slot tensors verified: 7 slots (`onset_ids`, `coeng_ids`, `medial_ids`, `register_ids`, `vowel_ids`, `coda_ids`, `sesqui_ids`), dtype `torch.int16`, non-negative vocab IDs, non-empty.
   - Second run (Resume Safety):
     - `Measure-Command { python scripts/overnight_khmer_ingester.py --shard-size 50 --max-clips 100 }`
     - Elapsed: `1.952 seconds` (< 2.0s requirement).
     - Output: `[RESUME] Target of 100 clips already reached (100 clips on disk). 0 duplicate shards written.`
     - Self-Audit Gate: `PASSED [100% OK]`.

### 10.3 Step 3: Overnight Runner & Stage 2 Pre-Training Loop
1. Created `scripts/overnight_runner.py` chaining Phase A (sharding up to 10,000 clips in 500-clip shards) and Phase B (Stage 2 infill pre-training on RTX 5070 Ti).
   - Dynamic batching: 4 random 2.0s segments (`wav_24k.float() / 32768.0`).
   - Augmentation: `DynamicAcousticCorruptor.warp_formants_vtlp` with Vocos resynthesis.
   - VRAM protection: `console.ear` and `console.pointer` parked on CPU during training.
   - Rolling checkpoints: `stage2_latest.pt` and `stage2_best.pt` every 250 optimizer updates.
2. Self-Audit Gate (10-Step Smoke Test):
   - `python scripts/overnight_runner.py --smoke-test --smoke-steps 10`
   - Initial loss: `67.004318` -> Final loss: `65.777145` (Best loss recorded: `65.777145`).
   - Peak VRAM measured: `1.7702 GB` (Strictly within `<= 3.20 GB` ceiling).
   - Rolling checkpoints verified saved to `checkpoints/stage2_latest.pt` and `checkpoints/stage2_best.pt`.
   - Logging verified written to `checkpoints/overnight_train.log`.
3. Background Overnight Job Launched:
   - **Process ID (PID):** `56488`
   - **Command Line:**
     ```bash
     python -u scripts/overnight_runner.py --phase all --shard-size 500 --max-clips 10000 --min-free-gb 15.0 --train-steps 10000 --save-every 250
     ```
   - **Active Log Paths:**
     - Ingestion log: `data/khmer_overnight_shards/ingestion.log`
     - Pre-training log: `checkpoints/overnight_train.log`
     - Rolling checkpoint: `checkpoints/stage2_latest.pt`
     - Best checkpoint: `checkpoints/stage2_best.pt`

