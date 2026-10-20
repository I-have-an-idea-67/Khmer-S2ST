# KHMER-SOVEREIGN-S2ST: Master Architectural & Engineering Documentation (v10.7)

This directory contains the complete, non-condensed, audited v10.7 architectural
specifications and planning documentation, partitioned into three structured
volumes plus two audit ledgers.

| Document | Lines | Words | Purpose & Core Contents |
| :--- | ---: | ---: | :--- |
| **[`01_ARCHITECTURE_AND_CONTRACTS.md`](./01_ARCHITECTURE_AND_CONTRACTS.md)** | **894** | **8,690** | **Header, Audit Ledger, Architecture & Contracts:** Master header, the v10.5 / v10.6 fix ledger with the **v10.7 code-level remediation entries 25-44** appended, Part 1 (Hybrid Overview, latency & the corrected 241.637M parameter budget), Part 2 (File tree, toolchains, **Section 2.3.1 Version Manifest** at `10.7.0**, `schemas.py` & `types.ts` contracts with locked FFT/Hop constants, the **`CartridgeThreadState` container**), submodule signatures, and Part 3 (Linguistics, transport, the **corrected authoritative parameter ledger** with exact per-module integers). |
| **[`02_DEEP_DIVE_IMPLEMENTATIONS.md`](./02_DEEP_DIVE_IMPLEMENTATIONS.md)** | **4,448** | **17,514** | **Verbatim Implementation Reference (Part 4):** Full deep-dive reference implementations (4.1-4.9) for all neural cartridges with rank-100 Slaney mel filterbanks, the 241.637M Vocos ISTFT head, 8-class phonological Khmer codas (`CODA_GROUP_MAP`), the **reserved-range English syllable alphabet** with 63-way `coeng_id` brand hash disambiguation, unity-gain sinc decimator, 64-band ERB suppressor, active clause wall-clock VAD ceiling, the **thread-local cartridge state**, the **sinusoidal timestep embedding**, the **Sway-sampled shortcut distillation**, and the **Section 4.9.7 Rust-to-Python ABI specification (Phase 2)**. |
| **[`03_TDD_GATES_AND_ORCHESTRATION.md`](./03_TDD_GATES_AND_ORCHESTRATION.md)** | **1,006** | **5,666** | **Verification, Ground-Truth & Multi-Agent Plan:** Part 5 (The 6 complete adversarial TDD test gates, **expanded in v10.7** for independent-vowel canonicalization, the exact integer ledger, step-conditioning rank, the multi-teacher distillation loss, the AMP / checkpointing contract, the rolling KV lifecycle, the `B_spk` transition, thread safety, and the epoch-fence ordering - with every v10.6 assertion retained **verbatim**), Part 6 (Ground-Truth targets & acoustic criteria), Part 7 (Multi-Agent execution roadmap with the corrected ownership plan), and Part 8 (Master orchestration prompt, **STAGE 02 rewritten** for v10.7, including the new STAGE 06 document-truth validation gate). |
| **[`04_AUDIT_FINDINGS_AND_REMEDIATION.md`](./04_AUDIT_FINDINGS_AND_REMEDIATION.md)** | **425** | **2,834** | The v10.5 -> v10.6 **document-level** audit ledger: all 15 findings (AUD-01 through AUD-15) with their evidence, root cause, and remediation; the audit method; the V1-V10 validation suite and v10.6 results; and the regeneration procedure. |
| **[`05_CODE_LEVEL_AUDIT_AND_REMEDIATION.md`](./05_CODE_LEVEL_AUDIT_AND_REMEDIATION.md)** | **137** | **2,255** | **[NEW in v10.7]** The **code-level** audit ledger: the audit method with the re-derived numeric claims, all 11 v10.6 findings plus 3 additional ones (A-1 the absent upstream audio path, A-2 the brand hash collision, A-3 the dead telemetry), the **unchanged** parameter ledger with the parameter-neutrality proofs, the V1-V18 validation suite, the environment caveats, and the regeneration procedure. |

---

## Related Documents (outside this directory)

| Document | Purpose |
| :--- | :--- |
| [`../KHMER-SOVEREIGN-S2ST-BLUEPRINT-v10.7.txt`](../KHMER-SOVEREIGN-S2ST-BLUEPRINT-v10.7.txt) | **Authoritative master blueprint.** The three volumes above are generated from it. |
| [`../KHMER-SOVEREIGN-S2ST-BLUEPRINT-v10.6.txt`](../KHMER-SOVEREIGN-S2ST-BLUEPRINT-v10.6.txt) | The immediately preceding revision. **Superseded** by v10.7. |
| [`../_archive_blueprints/`](../_archive_blueprints/) | Archived revisions v10.1 - v10.5, retained for provenance only. Not authoritative. |
| [`../Training, Distillation, and Dataset.txt`](../Training,%20Distillation,%20and%20Dataset.txt) | **Training, distillation & dataset engineering whitepaper** (8,617 words): 5-stage curriculum, multi-teacher distillation design, dataset construction, and the 40,000-hour parallel-speech pipeline. Cross-referenced from Part 7 and Part 8 of the master blueprint (AUD-11). |

---

## Integrity Status

The three volumes are **generated** from `KHMER-SOVEREIGN-S2ST-BLUEPRINT-v10.7.txt`
by the transformation script in the workspace root, partitioned at the Part 4 /
Part 5 boundaries. Every edit in that script is an **asserted** replacement that
aborts on a count mismatch, so a partial or silently-drifted write is impossible.

To verify zero divergence yourself:

```bash
cd "KHMER S2ST"
python __gen_v107.py
```

The script regenerates the master and all three volumes and prints a
`zero-divergence check: OK` line. The line and word counts published in the
table above are re-derived from the regenerated files - see Validation **V6**
in [`04_AUDIT_FINDINGS_AND_REMEDIATION.md`](./04_AUDIT_FINDINGS_AND_REMEDIATION.md).

> [!NOTE]
> The v10.5 `docs/README.md` claimed the volumes were "100% bitwise & token
> aligned" and published line counts of 691 / 2,764 / 407. Both claims were
> false: a 12-line version-token divergence existed and the counts were off by
> 8 and 509. See **AUD-12**.

---

## Verification Commands

```bash
# Run all Python TDD verification gates (Gates 1-6)
pytest backend/tests/ -v

# Run strict frontend TypeScript typecheck
npm --prefix frontend run typecheck

# Run Rust workspace unit tests
cargo test --workspace
```

> [!IMPORTANT]
> These commands describe the **target** workspace, which is specified in full
> in Part 2.2. The v10.7 materialisation scripts make the reference
> implementations runnable for validation:
>
> ```bash
> python __v107_materialize.py   # v10.7 master -> backend/ + frontend/src/ + gates
> python __v107_verify.py        # V1, V8, V8b, V10, V12, V14 against the master
> ```
>
> The Rust ABI deliverables required to make the Python-side Rust fast path
> reachable are specified in Section 4.9.7 as **Phase 2** work.
