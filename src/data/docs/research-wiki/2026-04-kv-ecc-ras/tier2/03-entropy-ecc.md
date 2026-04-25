# EntropyECC — Block-Entropy Adaptive ECC for Per-Token KV (merged with Driftwood characterization)

> [← Session Overview](/research-wiki/2026-04/kv-ecc-ras/README.md)

> ## 약어 / 핵심 용어 풀이 (R35)
>
> - **EntropyECC** — Block-Entropy Adaptive ECC for Per-Token KV. 본 idea 명칭. 16-token KV block 의 Shannon entropy 가 bit-flip sensitivity 의 single-scalar predictor.
> - **Driftwood (merged)** — system-robustness S1 의 position-dependent bit-criticality characterization. Long-context Llama-3-8B 에서 recent token 이 old token 보다 attention-mass 큼.
> - **Shannon entropy** — `H = -Σ p_i log p_i`. 16-token block 의 INT4 histogram (16-bin) 으로부터 계산. Cost ~64 ops per block.
> - **MiKV** — Importance-aware mixed precision KV ([arXiv:2402.18096](https://arxiv.org/abs/2402.18096)). attention-mass 기반 importance score (entropy 와 다른 axis).
> - **KIVI** — Per-channel K, per-token V quantization ([arXiv:2402.02750](https://arxiv.org/abs/2402.02750), ICML 2024). 본 idea 의 quantization base.
> - **KITTY** — Channel-sensitivity-ranked precision ([arXiv:2511.18643](https://www.arxiv.org/pdf/2511.18643), 2025-11). 본 idea 의 concurrent baseline.
> - **TailorKV** — Long-context KV mixed precision ([ACL 2025 Findings](https://aclanthology.org/2025.findings-acl.1043.pdf)). 본 idea 의 long-context baseline.
> - **Kelle MICRO 2025** — token importance × bit-position 2DRP ([arXiv:2510.16040](https://arxiv.org/abs/2510.16040)). 본 idea 와 axis 비교 critical.

**🎯 Target Venue**: ITC 2027 (primary, 4-6p) / IEEE TCAD short (alt) / MTS 2027 (alt)
**📊 Score**: Novelty 6.0 / Diff 5.0 / Impact 6.5 = 평균 **5.83**
**✅ 판정**: Accept Tier-2 (Driftwood characterization merged, MiKV/KITTY 비교 의무)

---

## 1. 개요 (Overview)

16-token KV block 의 **Shannon entropy** (INT4 histogram 의 16-bin 분포로부터 ~64 ops 계산) 가 **bit-flip sensitivity 의 single-scalar predictor** (Pearson r > 0.85 expected vs ground-truth attention-output L2 error). 임계 τ_lo, τ_hi 로 동적으로 {no ECC, parity, SEC-DED} 선택 → uniform-ECC accuracy 절반 redundancy. 추가로 Driftwood (S1 merged) 의 position-dependent bit-criticality (recent token 이 old token 보다 sensitivity ↑) 를 entropy threshold 의 token-position 가중치로 반영.

**핵심 insight (single-axis Tier-2)**:
- **Block-entropy = bit-flip sensitivity 의 cheap predictor** — quantization time 에 이미 계산되므로 cost 0.
- **Position-dependent recency bias** (Driftwood merged) — long-context 에서 recent token entropy 보호 강화.

**Metaphor 부속 (R30)**: "EntropyECC" = entropy 기반 ECC. Shannon entropy 가 information density → ECC strength 를 직접 결정.

---

## 2. 기존 연구의 한계 / GAP

| 기존 연구 | 축 | 한계 (본 연구 대비) | R46 verified |
|-----------|-----|---------------------|--------------|
| **MiKV** ([arXiv:2402.18096](https://arxiv.org/abs/2402.18096)) | importance-aware mixed precision | attention-mass importance, NOT entropy | R46 verified |
| **KITTY** ([arXiv:2511.18643](https://www.arxiv.org/pdf/2511.18643)) | channel-sensitivity-ranked precision | precision (not ECC), channel-rank | R46 verified: 2025-11 |
| **TailorKV** ([ACL 2025 Findings](https://aclanthology.org/2025.findings-acl.1043.pdf)) | long-context KV mixed precision | precision (not ECC) | R46 verified: ACL 2025 Findings |
| **KIVI** ([arXiv:2402.02750](https://arxiv.org/abs/2402.02750)) | per-channel K / per-token V | quantization base, no ECC | R46 verified: ICML 2024 |
| **Variable-rate codes (storage)** | entropy-adaptive ECC (general) | non-AI, no LLM-end-task sensitivity | R46 (ECC textbook reference) |
| **Bit-Flip Error Resilience in LLMs** ([EMNLP 2025](https://aclanthology.org/2025.emnlp-main.528.pdf)) | KV BER vs PPL standard protocol | model-level eval, no entropy | R46 verified: peer-reviewed EMNLP 2025 |
| **LongPPL / long-context PPL** ([arXiv:2410.23771](https://arxiv.org/abs/2410.23771)) | recency bias for PPL metric | PPL metric only, not bit-flip | R46 verified |
| **Driftwood 원안** (sys-rob S1) | position-bias characterization | single observation, no mechanism | merged into 본 idea |
| **Kelle MICRO 2025** ([arXiv:2510.16040](https://arxiv.org/abs/2510.16040)) | token importance × bit-position 2DRP | refresh-axis (no ECC) | **R46 verified, R²(entropy, importance) ablation 의무** |

**GAP**: Block entropy 가 LLM-end-task bit-flip sensitivity 의 predictor 인지의 **첫 정량 측정** + 그 결과 ECC strength 자동 선택. MiKV importance 와의 R² head-to-head 비교가 paper 핵심.

### Kelle MICRO 2025 1:1 차별화 (R46 의무)

| 축 | Kelle MICRO 2025 | EntropyECC |
|----|------------------|------------|
| Sensitivity input | token importance score (attention-based) | block entropy (Shannon, INT4 histogram) |
| Mechanism | refresh interval adaptive | ECC strength {no ECC, parity, SEC-DED} adaptive |
| Cost | bit-position MSB/LSB weighting | ~64 ops per block (free at quant time) |
| Critical ablation | — | **R²(entropy, Kelle_importance) < 0.5 → orthogonal 인정** |
| Boundary | edge eDRAM | datacenter HBM |

**Tier-2 paper 핵심 question**: entropy 가 Kelle's importance / MiKV's importance 의 substitute / orthogonal / superset 인지. R²<0.5 + ECC vote 50%+ 다르면 orthogonal — Tier-2 publication 가능. R²>0.7 시 본 idea = "MiKV/Kelle 의 ECC version" 으로 reduced framing → IEEE TCAD short 강등.

---

## 3. 제안 기법 (Single Mechanism, with merged Driftwood characterization)

### M1: Block Entropy → ECC Strength + Position Recency Weighting

**① 추가되는 Scheme — Source Verified (R32)**:

KIVI quantization step 에 entropy 계산 hook 추가 (~200 LoC, vLLM Triton kernel). Block (16-token × head_dim × INT4) histogram + Shannon entropy 계산 → ECC tag (2 bits) 결정. Driftwood 의 position-bias 를 long-context (32K+) 시 entropy threshold 의 recency 가중치로 반영.

> ✅ source verified: vllm-project/vllm@`main` `vllm/attention/backends/abstract.py`
> ✅ source verified: KIVI repo (`jy-yuan/KIVI`) for quantization step integration
> ⚠️ source proposed: `vllm/ecc/entropy_ecc.py` (~200 LoC, single-mech merge)
> ✅ external verified: ChampSim (`ChampSim/ChampSim`) bit-flip injection module (~80 LoC `src/`)

**② 해결하는 문제 + Workload evidence**:

- **Low-entropy block ≈ many near-zero values** (KIVI observation: post-RoPE V-cache 가 channels ~99% near-zero) → bit-flip on a zero-valued INT4 has small absolute Δ.
- **High-entropy block ≈ wide distribution** → bit-flip can flip a near-max value → large Δy.
- **Mathematical bound**: E[(Δy)²] ≤ E[(Δk)²] · E[v²] · |q|², where E[(Δk)²] for single bit-flip ∝ Var(k) per scale, scales with entropy within fixed quant range.
- **Driftwood characterization**: long-context (32K+) 에서 recent token KV 가 attention-mass 더 큼 → recent token entropy 의 ECC 임계 보수적으로.

**③ Step-by-step**:

1. KIVI quantization step 에서 16-token block 의 INT4 histogram (16-bin) 계산 (~64 ops).
2. Shannon entropy `H = -Σ p_i log p_i` 계산.
3. 임계 τ_lo, τ_hi (calibration 으로 결정) 비교:
   - H < τ_lo → no ECC (block-level CRC 1 byte detection-only)
   - τ_lo ≤ H < τ_hi → XOR parity (1 bit per 64 nibbles)
   - H ≥ τ_hi → SEC-DED Hamming(72,64)
4. ECC tag (2 bits) KV metadata 저장.
5. (Driftwood merged) long-context 시 token position p > 0.7 · seq_len 인 block 은 τ_hi 임계 강화 (H' = H + α(p)).
6. Calibration: holdout set 으로 τ_lo, τ_hi 결정 (~30분, one-shot).

**④ 차별화**:

(a) Variable-rate codes 의 storage 측 idea 를 **AI-end-task sensitivity 와 결합** — 첫 측정. (b) MiKV 의 attention-mass importance 와 **entropy 의 R² 비교** = paper 핵심 ablation. (c) Kelle 의 token importance 와 **entropy 가 직교 axis** 임을 R²<0.5 로 증명. (d) Driftwood 의 position bias 가 **mechanism 으로 통합** — 첫 paper.

**Tier 구성 (R28)**: physical 1-tier (single workstation) + software 1-tier (1 entropy hook). R28 ≤4 OK.

---

## 4. 평가 / 실험 플랜 (R27-β + R31 + R32 적용 7 요소)

### (1) Hardware

- **Primary**: AMD EPYC 9654 + RTX 4090 24GB
- **Sim host**: ChampSim trace + bit-flip injection module
- **Storage**: NVMe 1TB

### (2) Model

- **Llama-3-8B** + **Llama-3-70B** (KIVI W4A16)
- **Mistral-7B** AWQ INT4
- **Qwen3-VL-4B** (cross-modality check)

### (3) Dataset · Workload

- **WikiText-103** (PPL 측정)
- **LongBench** (32K-128K context, Driftwood position bias)
- **MMLU 1k** (zero-shot)
- **VQAv2 1k** (Qwen3-VL only)
- Calibration: holdout 1000 prompts (τ_lo / τ_hi 결정)

### (4) Simulator · Tools

- **ChampSim** (`ChampSim/ChampSim`) + bit-flip injection module (~80 LoC `src/`)
- **NeuroSim V1.4** (option: DRAM cell wear → time-varying BER)
- **vLLM v0.6+** + KIVI quantization step hook (~200 LoC Triton)
- **lm-evaluation-harness** (`EleutherAI/lm-evaluation-harness`)
- **scipy.stats** (R² calculation, Pearson r)

### (5) Ablation · Baseline

**Baselines (4 종, Tier-2 budget)**:

| # | Baseline | Venue / Source | 역할 |
|---|----------|----------------|------|
| (a) | **uniform SEC-DED** | Hamming(72,64) standard | reference |
| (b) | **MiKV importance score** | [arXiv:2402.18096](https://arxiv.org/abs/2402.18096) | **R²(entropy, MiKV_importance) ablation 의무** |
| (c) | **KITTY** ([arXiv:2511.18643](https://www.arxiv.org/pdf/2511.18643)) | channel-sensitivity precision (concurrent) |
| (d) | **Kelle MICRO 2025** ([arXiv:2510.16040](https://arxiv.org/abs/2510.16040)) | **R46 1:1 baseline, token importance vs entropy R²** |

Peer-reviewed ratio: 3/4 = **75%** (R2 충족).

**Ablation matrix**: (no ECC / uniform SEC-DED / EntropyECC w/o Driftwood / EntropyECC full) × (3 BER) × (3 model) × (5 seed) ≈ 30 runs Tier-2 budget.

**Parameter sweep**: τ_lo / τ_hi (3 setting), block size (8 / 16 / 32 token), Driftwood α(p) function (linear / log / step).

**Fallback mode**: R²(entropy, MiKV_importance) > 0.7 → 본 idea = "MiKV ECC version" 으로 reduced framing → IEEE TCAD short 강등. Pearson r(entropy, sensitivity) < 0.7 시 paper 자체가 약화 → 4 추가 paper 약화. Pearson r(entropy, sensitivity) < 0.7 시 paper 자체가 약화 → "block entropy 가 sensitivity 의 weak predictor" 로 negative result IEEE CAL 4p 강등.

### (6) Implementation Steps (Step-Level, R31)

| Step | 의존성 | Component / File (R32 verified) | 사용 API/Library | 완료 판정 |
|------|--------|---------|---------|---------|
| Step 1 | — | ChampSim setup + bit-flip injection module 작성 | ChampSim, ~80 LoC C++ | unit test bit-flip @ BER 1e-7 |
| Step 2 | — | KIVI quantization step + entropy hook (Mech M1) | KIVI repo + Triton kernel ~200 LoC | unit test entropy 계산 ±0.5% |
| Step 3 | Step 2 | calibration script (τ_lo, τ_hi 결정) | Python + scipy | holdout 1000 prompts τ 확정 |
| Step 4 | Step 1, 2 | ChampSim + KIVI + entropy_ecc 통합 | 위 stack | trace replay BER 1e-7 + entropy ECC 적용 |
| Step 5 | Step 4 | uniform SEC-DED baseline 재현 | 동일 sim | PPL +0.04 (uniform) baseline |
| Step 6 | Step 5 | MiKV / KITTY / Kelle baseline 재현 | 각 source repo | R²(entropy, MiKV_importance), R²(entropy, Kelle_importance) 측정 |
| Step 7 | Step 6 | Driftwood merged: long-context recency bias | LongBench 32K+ | position p × entropy effect |
| Step 8 | Step 7 | 3 model × 4 config × 3 BER × 5 seed = ~30 runs 실행 | 위 stack | runs dump |
| Step 9 | Step 8 | manuscript draft + R² ablation (paper 핵심) | matplotlib, scipy | 6p ITC draft 70% |
| Step 10 | Step 9 | polish + artifact prep | git + README | submission-ready |

**참고 시간**: 약 8-10 weeks (Tier-2).

### (7) Preliminary Analysis Metrics (R27-β)

| 측정 지표 | 도구 + counter/command | 측정 조건 | 기대 범위 (baseline) | 개선 후 목표 / 검증 기준 |
|---|---|---|---|---|
| Pearson r(entropy, sensitivity) | scipy.stats | 3 model × 30 holdout blocks | — | **r > 0.85** (Tier-2 success), r < 0.7 → fail |
| Mean redundancy / KV byte | metadata count | uniform SEC-DED | 1.5 bit | **0.45 bit (70% reduction)** |
| End-task PPL gap (vs uniform SEC-DED) | lm-eval | WikiText / Llama-3-8B BER=1e-7 | 0 (uniform) | **< 0.03 PPL** |
| VQA accuracy gap | VQAv2 1k | Qwen3-VL-4B BER=1e-7 | 0 (uniform) | **< 0.5 pp** |
| R²(entropy, MiKV_importance) | scipy.stats | 3 model | — | **R² < 0.5 (orthogonal) success, R² > 0.7 (reduced) fail** |
| R²(entropy, Kelle_importance) | scipy.stats | 3 model | — | **R² < 0.5 (orthogonal)** |
| Entropy compute overhead | per-block timing | A100 quant kernel | — | **< 0.3 µs per 16-token block** |
| Position p × sensitivity (Driftwood) | LongBench 32K | recent vs old block | — | **recent block sensitivity > 1.5× old** |

**Preliminary Study 4-단계**:
- **(i) Baseline reproduction**: uniform SEC-DED PPL 곡선 재현 + MiKV importance score 분포 재현.
- **(ii) Bottleneck attribution**: entropy 단독 / Driftwood 추가 / MiKV importance 단독 / 두 metric ensemble 의 4-config 비교.
- **(iii) Roofline**: redundancy bits × PPL × R² scatter — entropy / MiKV / Kelle 3 axis.
- **(iv) Micro-benchmark**: τ_lo / τ_hi sweep, block size sweep, Driftwood α(p) function sweep.

---

## 5. 예상 효과

| 지표 | Baseline | 목표 | 조건 |
|---|---|---|---|
| Pearson r(entropy, sensitivity) | — | **> 0.85** | core hypothesis |
| Mean redundancy | 1.5 (uniform SEC-DED) | **0.45 (70% reduction)** | M1 |
| PPL gap | 0 (uniform) | **< 0.03** | iso-fidelity |
| VQA gap (Qwen3-VL) | 0 | **< 0.5 pp** | VLM cross-check |
| R²(entropy, MiKV) | — | **< 0.5 (orthogonal)** | paper 핵심 |
| R²(entropy, Kelle) | — | **< 0.5 (orthogonal)** | Kelle 차별화 |
| Entropy compute | — | **< 0.3 µs per block** | overhead negligible |
| Driftwood recency | — | **recent > 1.5× old sensitivity** | long-context bias |

**과학적 contribution**: Block entropy = LLM-end-task bit-flip sensitivity predictor 의 첫 정량. MiKV / KITTY / Kelle 와 axis 분리. Driftwood (sys-rob S1) characterization 을 mechanism 으로 통합.

**실용적**: KIVI quantization 사용하는 모든 vLLM 배포에 적용 가능. Calibration 한 번 (~30분) 만 필요.

**Scope 제한**: Per-block quantization (KIVI-style) 한정 — per-tensor / per-channel-only 는 적용 어려움. R² 결과에 따라 paper 강도가 결정 — Tier-2 가 최대.

---

## 6. (Tier-2 → 강등 가이드)

| 조건 | 강등 venue | 강등 framing |
|------|------------|------------|
| R²(entropy, sensitivity) < 0.7 | IEEE CAL 4p | "negative result: entropy is weak predictor" |
| R²(entropy, MiKV_importance) > 0.7 | IEEE TCAD short | "ECC version of MiKV" reduced |
| Driftwood recency effect 약함 | IEEE ESL 4p | EntropyECC only, position-bias 제외 |

---

## 7. 미선정 idea 흡수 note

- **EntropyECC 원안** (algorithm T2) → 본 idea 의 Mech M1.
- **Driftwood** (sys-rob S1) → 본 idea 의 Driftwood merged section (long-context recency bias 가중치).
- **KV-Wear-Sketch** (alg T7) → optional add-on (clipping rate as additional feature for entropy threshold drift detection).

---

## 8. R46 verified ref 표 (이 파일)

| ref | 제목 | venue | R46 status |
|-----|------|-------|-----------|
| MiKV | importance-aware mixed precision | arXiv 2024 | verified |
| KITTY | channel-sensitivity-ranked precision | arXiv 2025-11 | verified |
| TailorKV | long-context KV mixed precision | ACL 2025 Findings | verified |
| KIVI | per-channel K, per-token V | ICML 2024 | verified |
| KVQuant | per-channel pre-RoPE outlier | NeurIPS 2024 | verified |
| Bit-Flip Resilience | KV BER vs PPL standard | EMNLP 2025 | verified |
| LongPPL | long-context PPL recency | arXiv 2410.23771 | verified |
| KVTuner | layer-wise sensitivity precision | ICLR 2025 / OpenReview | verified |
| **Kelle** | **token importance × bit-position 2DRP** | **MICRO 2025** | **verified, 1:1 baseline** |
| ChampSim | cache hierarchy sim | arXiv 2210.14324 | verified |
| NeuroSim V1.4 | DRAM cell wear | TCAS-I 2024 | verified |
| vLLM PagedAttention | block-table KV | SOSP 2023 | verified |
| lm-evaluation-harness | LLM eval harness | EleutherAI | verified |

R46 verified count: **13 ref**.
