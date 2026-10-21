# AUDIT REPORT: STAGE 09 SYSTEM & ML INVESTIGATION

**To:** System Orchestrator / Lead ML Architect  
**From:** Principal Systems & ML Auditor  
**Date:** 2026-10-21  
**Workspace:** `c:\Users\Chhean\Downloads\KHMER S2ST`  
**Target Codebase:** `v10.7.0` (241.637M S2ST Console)  
**Hardware Profile:** NVIDIA GeForce RTX 5070 Ti Laptop GPU (12 GB VRAM, 3.20 GB ceiling enforced)  
**Status:** Verification Passed Under Synthetic Contracts | **Audio Quality Status: UNACCEPTABLE / UNINTELLIGIBLE ("SHIT")**

---

## 1. Executive Summary & Raw Verdict

Stage 09 completed all contractual ML pipeline milestones:
1. Ingested 3,000 multi-source Khmer audio clips across 8 shards (`shard_00000.pt` .. `shard_00007.pt`).
2. Ran 50-step self-audit and 3,000 training steps on Stage 2B (`checkpoints/stage2b_tts_best.pt`, loss reduced from `7.30` to `0.8908`).
3. Synthesized 3 benchmark clips (`tts_test_1_greeting.wav`, `tts_test_2_sentence.wav`, `tts_test_3_codeswitch.wav`).

**Empirical Finding on Audio Output:**
Despite passing mathematical loss thresholds and gradient checks, **the synthesized audio sounds terrible / robotic / unintelligible ("shit")**. It resembles a low-frequency muffled drone/hum with metallic buzzing rather than recognizable human speech.

This report documents **what was measured and proven by execution**, why the audio is degraded, and what architectural flaws in the vocoder and training setup cause this behavior.

---

## 2. Empirical Findings: Step-by-Step

### 2.1 Step 1 — Corpus Expansion & Ingestion Findings

| Metric | Measured Value | Requirement | Status |
| :--- | :--- | :--- | :--- |
| **Total Shards on Disk** | 8 shards (`shard_00000.pt` .. `shard_00007.pt`) | $\ge 6$ shards | **PASS** |
| **Total Audio Clips** | 3,000 clips | $\ge 2,500$ clips | **PASS** |
| **Audio Format** | 1-D mono `torch.int16` @ 16 kHz & 24 kHz | `int16` mono | **PASS** |
| **Linguistic Slots** | 7 slots (`onset`, `coeng`, `medial`, `register`, `vowel`, `coda`, `sesqui`) | All non-empty, valid token IDs | **PASS** |
| **Average Shard Size** | 0.38 – 0.77 MiB per clip | $< 2.0\text{ MiB/clip}$ | **PASS** |
| **Streaming Throughput** | ~5.6 clips / second from `KrorngAI/fleurs_openslr42_mpwt` | Live streaming | **PASS** |

**Observations:**
* `phonsobon/openslr42-khmer-male` is gated and unreachable without an HF token.
* `seanghay/km-speech-corpus` contains 475 MB parquet row groups which time out on standard probes.
* `KrorngAI/fleurs_openslr42_mpwt` is ungated, fast, and contains combined OpenSLR-42, FLEURS, and MPWT clips. It provided all 2,425 required clips cleanly.

---

### 2.2 Step 2 — Stage 2B Training Trajectory & Memory Metrics

| Metric | Measured Value | Target Ceiling | Status |
| :--- | :--- | :--- | :--- |
| **Peak CUDA VRAM** | **3.0637 GB** | $\le 3.20\text{ GB}$ | **PASS** |
| **Initial Loss (Step 1)** | `7.3026` ($\mathcal{L}_{\text{CFM}} = 4.56, \mathcal{L}_{\text{dur}} = 2.51$) | Baseline | — |
| **Mid Loss (Step 1,500)** | `1.0423` ($\mathcal{L}_{\text{CFM}} = 0.94, \mathcal{L}_{\text{dur}} = 0.10$) | Progress | — |
| **Final Loss (Step 3,000)** | **`0.890884`** ($\mathcal{L}_{\text{CFM}} = 1.3187, \mathcal{L}_{\text{dur}} = 0.0489$) | Convergence | **PASS** |
| **Elapsed Training Time** | 2,777.8 seconds (~46.3 minutes) on RTX 5070 Ti | $\approx 25\text{--}50\text{ min}$ | **PASS** |
| **50-Step Gradient Check** | `onset`: 0.708 $\to$ 0.132, `dur`: 71.1 $\to$ 9.48, `cond`: 9.45 $\to$ 4.72 | Norm $> 0$, finite | **PASS** |
| **Token-0 Padded Duration** | Exact `0.0000` frames across all `sesqui_ids == 0` | Exact 0.0 | **PASS** |

**Observations:**
* VRAM remained strictly constrained at 3.06 GB because `ear` and `pointer` were parked on CPU and `vocos` was frozen.
* The duration aligner loss converged rapidly (`0.0489`), meaning the model learned the sesqui-ratio heuristic.
* The CFM flow loss stabilized around `1.1 – 1.3`, indicating the DiT learned the coarse macro-structure of the mel-spectrogram, but did not resolve fine harmonic partials.

---

### 2.3 Step 3 — Speech Synthesis & Audio Quality Audit (The "Shit Audio" Breakdown)

Three benchmark sentences were synthesized and audited:

| Benchmark File | Text | Syllables | Duration | RMS Energy | Spectral Centroid | Human Speech Standard |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| `tts_test_1_greeting.wav` | សួស្តីកម្ពុជា ខ្ញុំឈ្មោះសុភមង្គល | 11 | 2.64s | 0.0545 | **216.8 Hz** | 1,500 – 3,500 Hz |
| `tts_test_2_sentence.wav` | ប្រទេសកម្ពុជាមានធម្មជាតិដ៏ស្រស់ស្អាត | 12 | 2.88s | 0.0548 | **240.5 Hz** | 1,500 – 3,500 Hz |
| `tts_test_3_codeswitch.wav` | ខ្ញុំទៅភ្នំពេញទិញ Calvin Klein និង Versace | 9 | 2.16s | 0.0555 | **512.2 Hz** | 1,500 – 3,500 Hz |

---

## 3. Root Cause Analysis: Why the Audio Sounds Horrible

An empirical investigation of the generated audio signals and the neural modules revealed **five distinct engineering root causes**:

### Cause 1: Severe High-Frequency Collapse (Spectral Centroid ~216 Hz vs Normal ~2,500 Hz)
* **Measurement:** Human speech energy is distributed across vowels ($500\text{--}2,500\text{ Hz}$) and consonants/fricatives ($3,000\text{--}8,000\text{ Hz}$).
* **Measured reality:** All 3 clips have spectral centroids between **216 Hz and 512 Hz**.
* **Result:** There are almost no audible high frequencies or formant definitions. Consonants (like /s/, /t/, /k/) are completely muffled or absent, leaving only a low-frequency pitch buzz.

### Cause 2: Vocos Surgical Warm-Start Defect (Even Ground Truth Mel Sounds Broken)
* **Test executed:** Reconstructed a real ground-truth reference audio clip directly through the warm-started `VocosISTFTHead` (`console.vocos(real_mel)`).
* **Finding:** The reconstruction of **real ground-truth speech** through the current Vocos checkpoint yielded a spectral centroid of only **123.8 Hz** and an extreme uncalibrated peak amplitude of **68.0**!
* **Root Cause in `backend/training/train_cartridge.py` (`warm_start_vocos_from_pretrained`):**
  1. The pretrained `charactr/vocos-mel-24khz` was designed for $N_{\text{fft}} = 1024, \text{hop} = 256$.
  2. This repository changed the architecture to $N_{\text{fft}} = 2048, \text{hop} = 240$.
  3. The 1,536-wide ConvNeXt MLP was surgically down-sampled by 50% to 768 (`top_k` channel slicing), destroying half the network weights.
  4. The 513-bin ISTFT linear projection was **naively linearly interpolated** to 1025 bins. Linearly interpolating complex phase and magnitude across STFT bins destroys phase coherence and produces massive destructive interference.
  5. Vocos was **frozen** throughout Stage 2 and Stage 2B. It has never been fine-tuned on this repository's mel representation.

### Cause 3: Vocos Amplitude Explosion & Previous Rail Clipping
* **Measurement:** `VocosISTFTHead` outputs linear waveforms with raw peak amplitudes ranging from `10.4` to `102.1` (instead of unit range $[-1.0, 1.0]$).
* **Previous Bug:** In the first iteration of `scripts/synthesize_khmer_tts.py`, raw waveforms were clamped via `clamp(-1.0, 1.0)`. Clamping a signal with peak 10.4 chopped over 90% of the active waveform into flat rails at $+32,767$ and $-32,767$, creating square-wave DC saturation that triggered browser audio muting.
* **Remediation:** Added `wav / max(peak, 1e-6) * 0.92` peak normalization, eliminating rail clipping. However, while clipping is resolved, the underlying vocoder phase incoherence (Cause 2) remains.

### Cause 4: Zero Semantic Bus Input (`zero_bus`)
* During Stage 2B training and inference, the semantic bus input was set to all zeros:
  ```python
  zero_bus = torch.zeros(B, L_max, 512)
  zero_gate = torch.zeros(B, L_max, 1)
  ```
* In `KhmerMouthCartridge.predict_syllables_and_align`:
  ```python
  reordered_bus, attn_w = self.reorder_attn(query=pos_q, key=kv_mem, value=kv_mem)
  bus_l = self.reorder_norm(pos_q + reordered_bus)
  syll_hidden = bus_l + slot_sum
  ```
* Because `kv_mem` received all zeros, the cross-attention reordering bridge had zero semantic guidance and operated purely off sinusoidal position queries and static slot embeddings.

### Cause 5: Insufficient Step Budget (3,000 Steps on 241M Model)
* Stage 2B ran for 3,000 steps with $B = 4$, representing only **12,000 clip exposures**.
* `mouth.embeddings` (7 dictionaries), `mouth.cond_proj`, and `mouth.duration_aligner` started from random initialization.
* 12,000 clips is sufficient to learn macro loss minimization ($\mathcal{L} \approx 0.89$), but diffusion-based CFM decoders require $\ge 50,000\text{--}100,000$ steps to synthesize clear phonetic formants.

---

## 4. UI "1 Second" Display Explanation

In the browser player interface:
* The audio was originally synthesized at 180ms/syllable (18 frames @ 100 Hz), resulting in a 1.62-second clip for 9 syllables.
* Standard HTML5 audio elements format time as integer seconds (`mm:ss`). Any duration under 2.0s displays as `0:01 / 0:01`.
* **Fix applied:** Pacing was adjusted to 240ms/syllable (24 frames @ 100 Hz), producing durations of 2.16s, 2.64s, and 2.88s (displaying `0:02` and `0:03`).

---

## 5. Concrete Action Plan for Orchestrator

If the goal is to achieve human-intelligible, broadcast-quality Khmer voice cloning, the following steps must be taken:

1. **Re-Train / Fine-Tune `VocosISTFTHead` Separately:**
   * Do NOT use linear bin interpolation for the ISTFT head without retraining.
   * Run a standalone 20,000-step training job on `VocosISTFTHead` using multi-resolution STFT loss and Mel L1 loss on the 3,000 Khmer audio shards.
   * Verify that Vocos can reconstruct *ground-truth* Khmer audio with a spectral centroid $> 2,000\text{ Hz}$ before connecting it to the DiT.

2. **Calibrate Vocos Amplitude:**
   * Add a gain scaling parameter or LayerNorm before ISTFT so Vocos natively outputs in $[-1.0, 1.0]$.

3. **Train with Real Semantic Bus (Precomputed Ear Representations):**
   * Pass real `ear.last_l8_hidden` and `pointer` fused bus representations into Stage 2B instead of `zero_bus = 0`.

4. **Increase Training Budget for Stage 2B:**
   * Scale training from 3,000 steps to at least 25,000–50,000 steps on the RTX 5070 Ti.

---

## 6. Repository Integrity & Commit Verification

* **Modified files:**
  - `scripts/overnight_khmer_ingester.py` (candidate list and multi-corpus streaming)
  - `scripts/train_stage2b_khmer_tts.py` (Stage 2B training pipeline)
  - `scripts/synthesize_khmer_tts.py` (synthesis script with peak normalization)
  - `WORKLOG.md` (Stage 09 log record)
* **Locked files:** `schemas.py`, `types.ts`, `backend/tests/test_*.py` — untouched.
* **Unit Tests:** `pytest backend/tests -q` $\to$ **38 passed, 0 failed**.
* **Git Commit:** Pushed to `origin/main` (`commit 78147a1`).
