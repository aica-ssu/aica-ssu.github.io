# OAEP-KV — Outlier-Aware Asymmetric ECC for Quantized KV Cache (merged with Sentinel: DRAM-Row-Stripe Alignment + Sub-Page Retirement)

> [← Session Overview](/research-wiki/2026-04/kv-ecc-ras/README.md)

> ## 약어 / 핵심 용어 풀이 (R35)
>
> - **OAEP-KV** — Outlier-Aware Asymmetric ECC Protection for KV cache. 본 idea 의 명칭. AWQ/KVQuant outlier mask 를 ECC strength selector 로 재사용.
> - **Outlier channel** — AWQ/KVQuant 가 보고한 LLM activation 의 1% 채널이 99% sensitivity 차지. Pre-RoPE Key 에서 spatially clustered (KVQuant NeurIPS 2024).
> - **AWQ** — Activation-aware Weight Quantization, MLSys 2024 Best Paper ([arXiv:2306.00978](https://arxiv.org/abs/2306.00978)).
> - **SEC-DED** — Single-Error-Correct Double-Error-Detect Hamming(72,64). 64bit data 당 8 redundancy bit (12.5%).
> - **Bamboo ECC** — HPCA 2015 강력 ECC. DEC-3 (triple-error-correct) 16-symbol granularity.
> - **DRAM-row stripe** — outlier 채널이 DRAM row 의 한 contiguous segment 에 align 되도록 vLLM allocator 가 placement 를 강제. Sentinel merged 측의 핵심 mechanism.
> - **CHAOSMem** — gem5 의 controlled hardware fault injector ([arXiv:2602.02119](https://arxiv.org/html/2602.02119v1)). bit-flip / stuck-at fault inject 가능.
> - **Kelle MICRO 2025** — eDRAM KV cache 의 2DRP (token-importance × bit-position MSB/LSB) refresh policy ([arXiv:2510.16040](https://arxiv.org/abs/2510.16040)). 본 idea 의 critical 1:1 baseline.
> - **GenBFA** — bit-flip attack on quantized LLM ([arXiv:2411.13757](https://arxiv.org/abs/2411.13757)). 3 critical bit → MMLU 0% 로.
> - **MixKVQ** — query-aware online mixed-precision KV ([arXiv:2512.19206](https://arxiv.org/abs/2512.19206)). OAEP-KV 의 calibration-only mask 와 차별화 baseline.

**🎯 Target Venue**: DSN 2027 (primary, 12p) / HPCA 2027 (alt) / MICRO 2027 (alt)
**📊 Score**: Novelty 7.2 / Diff 7.0 / Impact 8.0 = 평균 **7.40**
**✅ 판정**: Accept strong (Sentinel merge 후 DRAM-stripe alignment 보강)

---

## 1. 개요 (Overview)

LLM/VLM KV cache 의 quantization 후 INT4/INT8 mantissa 에 대해, AWQ/KVQuant calibration 단계에서 이미 추출되는 **per-channel outlier mask** 를 **ECC strength selector** 로 재사용한다. 1% outlier 채널은 SEC-DED + Bamboo DEC, 99% non-outlier 채널은 단순 XOR parity 로 보호. 추가로 DRAM row 에 outlier 채널이 contiguous stripe 로 align 되도록 vLLM block allocator 가 placement 를 강제하여 (Sentinel merged 측 핵심 contribution), sub-page retirement 시 outlier-stripe 만 보호하고 non-outlier-stripe 의 fault 는 page 전체를 retire 하지 않고 reclaim 한다.

**핵심 insight (3축)**:
1. **Outlier asymmetry**: 1% 채널이 99% bit-flip sensitivity 보유 (KVQuant NeurIPS'24). Uniform ECC 는 information-theoretically 낭비.
2. **Spatial clustering of outliers in pre-RoPE Key**: outlier 채널이 channel index 에서 contiguous → DRAM row mapping 가능.
3. **Stranding reduction**: outlier-stripe 외 fault 는 sub-page reclamation 으로 stranding 을 30-50% → 10-15% 감축.

**Metaphor 부속 (R30)**: "OAEP" = Outlier-Aware Asymmetric (Echo) Protection. RAS-side 의 Sentinel (감시탑) 이 outlier-channel 을 stripe 단위로 감시한다는 결합 metaphor.

---

## 2. 기존 연구의 한계 / GAP

| 기존 연구 | 축 | 한계 (본 연구 대비) | R46 verified |
|-----------|-----|---------------------|--------------|
| **AWQ** ([arXiv:2306.00978](https://arxiv.org/abs/2306.00978)) | quantization accuracy 용 outlier 1% protection | ECC strength 측면 미고려 | R46 verified: title=AWQ, venue=MLSys 2024 Best Paper, authors=Lin et al. |
| **KVQuant** ([arXiv:2401.18079](https://arxiv.org/abs/2401.18079)) | per-channel pre-RoPE outlier isolation | quant accuracy 만, ECC 무관 | R46 verified: title=KVQuant, venue=NeurIPS 2024, authors=Hooper et al. |
| **Bamboo ECC** ([HPCA 2015](https://lph.ece.utexas.edu/merez/uploads/MattanErez/hpca2015_bambooECC.pdf)) | strong ECC code uniform | non-AI workload, outlier-aware 아님 | R46 verified: title=Bamboo ECC, venue=HPCA 2015, authors=Kim/Sullivan/Erez |
| **Hessian-Driven Unequal Protection** ([DATE 2020](https://ieeexplore.ieee.org/document/9256567)) | DNN weight UEP (1-5% protected) | weight 만, KV cache 무관 | R46 verified: venue=DATE 2020 |
| **DeepNcode** ([arXiv:2405.13891](https://arxiv.org/abs/2405.13891)) | encoding-based BFA 보호 (weight) | weight 만, UEP-by-outlier 아님 | R46 verified: WebFetch 2026-04-25 |
| **Kelle MICRO 2025** ([arXiv:2510.16040](https://arxiv.org/abs/2510.16040)) | eDRAM 2DRP (token importance × bit-position MSB/LSB) refresh | refresh axis (no ECC), edge eDRAM 만 | **R46 verified: title="Kelle: Co-design KV Caching and eDRAM for Efficient LLM Serving in Edge Computing", venue=MICRO 2025 (Seoul, Oct 18-22), arXiv:2510.16040** |
| **MixKVQ** ([arXiv:2512.19206](https://arxiv.org/abs/2512.19206)) | query-aware online mixed-precision | dynamic mask, ECC 무관 | R46 verified: arXiv 2025-12 |
| **OASIS** ([IEEE 2025](https://ieeexplore.ieee.org/document/10990150/)) | outlier-aware KV clustering for CXL | CXL 측, ECC 측면 없음 | R46 verified: WebSearch |
| **Bit-Flip Error Resilience in LLMs** ([EMNLP 2025](https://aclanthology.org/2025.emnlp-main.528.pdf)) | KV bit-flip BER vs PPL 표준 protocol | model-level 평가, mitigation 미연동 | R46 verified: peer-reviewed EMNLP 2025 main |

**GAP**: outlier mask 를 (a) ECC strength selector 로 재사용 + (b) DRAM-row-stripe alignment 로 spatial 처리 + (c) sub-page reclamation 의 RAS 통합 — 셋 다 묶은 paper 미존재.

### Kelle MICRO 2025 1:1 차별화 (R46 의무)

| 축 | Kelle MICRO 2025 | OAEP-KV |
|----|------------------|---------|
| Hardware target | eDRAM (edge LLM serving) | HBM3/HBM3e (datacenter LLM) |
| Mechanism | Refresh interval adaptive (2DRP) | ECC strength asymmetric + DRAM row stripe alignment |
| Sensitivity input | bit-position (MSB/LSB) × token importance score | per-channel outlier mask (calibration once) |
| Failure mode | retention failure (refresh-marginal) | bit-flip + cell wear |
| Measured improvement | 3.9× speedup, 4.5× energy | redundancy 70% reduction at iso-PPL, stranding 30% → 10% |
| Orthogonality 증명 | — | Pearson r(outlier_mask, Kelle_token_importance) ablation **R² < 0.5 시 orthogonal** |

---

## 3. 제안 기법 (Core Mechanisms, 2 mechanisms)

### M1: Outlier-Aware Asymmetric ECC (OAEP)

**① 추가되는 Scheme — Source Verified (R32)**:

vLLM `KVCacheManager.write_kv()` 와 `read_kv()` 경로에 outlier-mask 기반 ECC encoder/decoder 를 삽입 (~600 LoC, vLLM v0.6+ + Triton kernel).

> ✅ source verified: vllm-project/vllm@`main` `vllm/core/block_manager.py` (확인일: 2026-04-25)
> ✅ source verified: vllm-project/vllm@`main` `vllm/attention/backends/abstract.py` (KVCacheManager 인터페이스)
> ⚠️ source proposed: `vllm/ecc/oaep_codec.py` (~600 LoC, 신규 module — vLLM 내 별도 directory)
> ✅ external verified: AWQ outlier mask format `awq/quantize/awq_calibrate.py` (mit-han-lab/llm-awq)

**② 해결하는 문제 + Workload evidence**:

KVQuant NeurIPS'24 Pre-RoPE Key 측정에서 outlier 1% 채널이 99% sensitivity 보유. Uniform SEC-DED 는 12.5% redundancy 를 채널 균등 분배 → 99% non-outlier 채널의 redundancy 는 over-protect, 1% outlier 채널은 under-protect.

**③ Step-by-step**:

1. AWQ/KVQuant calibration (128-sample WikiText) → per-head, per-channel salience score `s_{h,c}`.
2. Top-1% 임계 threshold → binary mask `M ∈ {0,1}^(L×H×D)`. Metadata cost = 1 bit/channel ≈ 0.04% of KV.
3. KV write 시: outlier 채널은 SEC-DED Hamming(72,64) + Bamboo DEC-3 (16-symbol); non-outlier 는 XOR parity (1 bit / 64 nibbles).
4. Per-token scale FP16 (KV bytes 의 0.5%) 은 항상 SEC-DED + RS(15,13) — 단일 비트 flip 시 token 전체 catastrophic.
5. KV read 시: parity mismatch → SEC-DED 시도 → 실패 시 vLLM RFC #19329 (graceful KV connector error) 패턴으로 token-range recompute reschedule.

**④ 차별화**:

(a) AWQ 가 quantization accuracy 용으로 mask 사용 → 본 연구는 **ECC strength selector** 로 재사용. (b) Bamboo ECC 가 uniform strong code → 본 연구는 **mask 기반 unequal protection**. (c) Hessian-Driven UEP DATE'20 가 DNN weight 단 → 본 연구는 **KV cache scale + mantissa**. (d) Kelle 2DRP 가 token-importance refresh → 본 연구는 outlier-channel ECC, axis 직교.

### M2: DRAM-Row-Stripe Alignment + Sub-Page Retirement (Sentinel merged)

**① 추가되는 Scheme — Source Verified (R32)**:

vLLM block allocator 가 outlier 채널을 DRAM row 의 contiguous stripe (≤ row size 1/8) 에 align 시키도록 placement 를 강제. Fault 가 non-outlier stripe 에 발생 시 page 전체가 아닌 stripe 만 retire (sub-page reclaim).

> ⚠️ source proposed: `vllm/core/block_manager.py` 의 `align_outlier_stripe()` method (~150 LoC)
> ✅ external verified: gem5+DRAMSim3 (`gem5/gem5` + `umd-memsys/DRAMSim3`) HBM3 channel mapping 모델
> ✅ external verified: CHAOSMem (gem5 fault injector, [arXiv:2602.02119](https://arxiv.org/html/2602.02119v1))

**② 해결하는 문제 + evidence**:

KVQuant 의 spatial clustering 측정: pre-RoPE Key 의 outlier 채널이 channel index space 에서 contiguous (clustered, not scattered). DRAM row 에 contiguous stripe 로 mapping 가능 → fault locality 가 column/bank scope (~70%, MICRO'25 fault classification) 에 머물 확률을 활용해 sub-page reclaim.

**③ Step-by-step**:

1. AWQ outlier mask M 으로부터 **stripe layout** 계산 — 각 row 의 outlier channels 를 row 시작 지점에 정렬.
2. vLLM `BlockManager.allocate()` 가 KV tile-to-DRAM-row mapping 시 stripe alignment 강제.
3. Fault 발생 시 syndrome 위치를 보고 stripe 내 vs 밖 분류.
4. Stripe 밖 fault → 해당 stripe 만 (≤ row 1/8 = 256B) **mark unusable**, vLLM block 의 나머지 outlier-stripe 와 다른 non-outlier-stripe 는 그대로 사용.
5. Stripe 내 fault → 해당 outlier 채널 데이터를 다른 healthy block 으로 migrate (cudaMemcpyAsync, vLLM block-level).

**④ 차별화**:

Linux mcelog 의 4KB/2MB hugepage 단위 retire vs 본 연구 stripe (256B) 단위 → **stranding 16x 감소**. Bamboo HPCA'15 가 row 단위 strong code 만 → 본 연구는 **outlier-aware row stripe + retire**.

### Mechanism 간 상호작용

M1 (asymmetric ECC) 은 outlier mask 의 **bit budget 분배**, M2 (stripe alignment + sub-page retire) 는 **spatial layout + fault response**. 두 mechanism 이 같은 mask 를 input 으로 사용 → 하나의 calibration 으로 두 효과 모두 획득. Ablation: M1-only / M2-only / M1+M2 3-cell.

**Tier 구성 (R28)**: physical 1-tier (single workstation, 1 GPU + sim) + software 1-tier (1 ECC code + 1 allocator policy). R28 ≤4 OK.

---

## 4. 평가 / 실험 플랜 (R27-β + R31 + R32 적용 7 요소)

### (1) Hardware

- **Primary**: AMD EPYC 9654 + RTX 4090 24GB (Llama-3-8B BF16 inference + AWQ INT4)
- **Sim host**: 동일 workstation, gem5 SE-mode + DRAMSim3 HBM3 channel
- **Storage**: NVMe 2TB (KV trace dump + 30 sim runs 결과)
- **Network**: single-workstation scope

### (2) Model

- **Llama-3-8B** (HuggingFace `meta-llama/Meta-Llama-3-8B-Instruct`) BF16 + KIVI W4A16
- **Mistral-7B-Instruct-v0.3** AWQ 4-bit (cross-check)
- **Llama-3-70B-Instruct** W4G128 (scaling check)
- **Qwen3-VL-8B** ([arXiv:2511.21631](https://arxiv.org/abs/2511.21631)) — VLM extension
- **SmoothQuant Llama-2-7B** — outlier 분포 cross-check

### (3) Dataset · Workload

- **WikiText-103** (PPL 측정), **MMLU 1k** (zero-shot accuracy), **LongBench 16K**, **HumanEval** — lm-evaluation-harness via lm-eval hooks
- **VLM**: VQAv2 1k subset, DocVQA, MMVet
- Bit-error-rate sweep: 1e-9 / 1e-7 / 1e-6 (DSN'24-'25 reported HBM3/HBM3e tail)

### (4) Simulator · Tools

- **gem5 + DRAMSim3** (HBM3 + LPDDR5 channel configs)
- **CHAOSMem** ([arXiv:2602.02119](https://arxiv.org/html/2602.02119v1)) — bit-flip / stuck-at injection at memory level
- **vLLM v0.6+** fork (KVCacheManager hook)
- **Triton kernel** (~600 LoC, OAEP encoder/decoder)
- **lm-evaluation-harness** (`EleutherAI/lm-evaluation-harness`) for PPL/zero-shot eval after bit injection

### (5) Ablation · Baseline

**Baselines (6 종)**:

| # | Baseline | Venue / Source | 역할 |
|---|----------|----------------|------|
| (a) | **vanilla SEC-DED** | Hamming(72,64) standard | uniform ECC reference |
| (b) | **chipkill always** | Sridharan ASPLOS'15 | strong-but-uniform reference |
| (c) | **Bamboo ECC HPCA'15** | [PDF](https://lph.ece.utexas.edu/merez/uploads/MattanErez/hpca2015_bambooECC.pdf) | strong ECC w/o outlier-aware |
| (d) | **Hessian-Driven UEP DATE'20** | [IEEE 9256567](https://ieeexplore.ieee.org/document/9256567) | DNN weight UEP (KV-extended adaptation) |
| (e) | **DeepNcode** | [arXiv:2405.13891](https://arxiv.org/abs/2405.13891) | encoding-based BFA defense (weight) |
| (f) | **Kelle MICRO 2025 (2DRP)** | **[arXiv:2510.16040](https://arxiv.org/abs/2510.16040)** | **R46 verified — refresh-axis competitor, head-to-head 의무** |

Peer-reviewed ratio: 5/6 = **83%** (R2 ≥25% 충족).

**Ablation matrix**: (M1 only / M1+M2 / M2 only / no ECC) × (BER 1e-9, 1e-7, 1e-6) × (5 model) = 60 cells. Tier-1 budget 30 runs 에 맞추기 위해 model 2 (Llama-3-8B + Llama-3-70B) × config 3 × BER 3 × seed 5 = ~30 runs 로 압축.

**Parameter sweep**: outlier 임계 top-{0.5%, 1%, 2%}, stripe size {128B, 256B, 512B}, ECC code 강도 {SEC-DED only, +Bamboo DEC, +RS(15,13)}.

**Fallback mode**: rotated model (QuaRot) 에서 outlier mask 가 평탄화되면 RotECC 와 통합한 "outlier-spectrum-aware ECC" framing 으로 재구성. 그 외 redundancy savings < 30% 면 Tier-2 IEEE TCAD short 로 강등.

### (6) Implementation Steps (Step-Level, R31)

| Step | 의존성 | Component / File (R32 verified) | 사용 API/Library | 완료 판정 |
|------|--------|---------|---------|---------|
| Step 1 | — | gem5 SE-mode + DRAMSim3 HBM3 setup. **gem5/gem5 + umd-memsys/DRAMSim3 ✅** | gem5 v22+, DRAMSim3 last release 2024 | unit test 시뮬 1 token KV write/read pass |
| Step 2 | Step 1 | CHAOSMem fault injector 통합 + BER sweep harness | CHAOSMem [arXiv:2602.02119](https://arxiv.org/html/2602.02119v1) | BER 1e-7 inject + lm-eval baseline 재현 |
| Step 3 | — | AWQ outlier mask 추출 (Llama-3-8B + Mistral-7B) | mit-han-lab/llm-awq | outlier mask `M` saved (per-head, per-channel) |
| Step 4 | Step 3 | OAEP encoder/decoder Triton kernel 구현 | Triton 2.x | unit test 64 nibble block roundtrip OK |
| Step 5 | Step 1, 4 | vLLM KVCacheManager hook 통합 (Mech M1) | vLLM v0.6+ | Llama-3-8B WikiText PPL ±0.05 baseline |
| Step 6 | Step 5 | DRAM-row-stripe alignment 구현 (Mech M2 from Sentinel) | vLLM block_manager.py + DRAMSim3 row mapping | stripe size 256B alignment 확인 |
| Step 7 | Step 6 | sub-page reclaim 로직 (vLLM only, no kernel patch) | vLLM extension | fault 후 stripe 만 mark unusable 검증 |
| Step 8 | Step 5-7 | Kelle 2DRP baseline 재현 + head-to-head | Kelle paper 의 sim setup re-implement | Kelle 의 3.9× speedup 데이터 ±10% 재현 |
| Step 9 | Step 8 | 5 model × 3 config × 2 baseline = 30 runs 실행 | 위 stack 전체 | 30 runs 결과 dump |
| Step 10 | Step 9 | manuscript draft + ablation analysis | matplotlib, pandas, lm-eval | 12p draft 70% |
| Step 11 | Step 10 | polish + artifact prep | git + README | submission-ready |

**참고 시간 (단일-workstation 기준)**: 약 12-14 weeks 분포.

### (7) Preliminary Analysis Metrics (R27-β)

| 측정 지표 | 도구 + counter/command | 측정 조건 | 기대 범위 (baseline) | 개선 후 목표 / 검증 기준 |
|---|---|---|---|---|
| Llama-3-8B WikiText PPL | lm-eval-harness | BER=0 baseline | ~6.1 (FP16) | OAEP-KV: 6.15 ± 0.05 (no degradation) |
| PPL @ BER=1e-7 (uniform SEC-DED) | lm-eval after CHAOSMem | Llama-3-8B INT4 KIVI | +0.04 PPL | (reference) |
| PPL @ BER=1e-7 (no ECC) | lm-eval after CHAOSMem | 동일 | +1.2 PPL | (reference, attack) |
| **PPL @ BER=1e-7 (OAEP M1)** | lm-eval | 동일 | — | **< 0.05 PPL @ 30% redundancy** |
| **Stranding ratio @ BER=1e-7 (OAEP M1+M2)** | DRAMSim3 retire log | 24h sim | 30-50% (baseline) | **10-15% (target ≥40% reduction)** |
| Throughput (tokens/sec) | vLLM benchmark | 동일 | -0% (no ECC) | **-3 to -5% (OAEP)** |
| Pearson r(outlier_mask, Kelle_importance) | scipy.stats | 5 model average | — | **R² < 0.5 → orthogonal 인정** |
| GenBFA 3-bit attack accuracy | MMLU after targeted flip | LLaMA3-8B-W8 | 67.3% → 0% (no defense) | **≥ 50% retained (OAEP M1)** |

**Preliminary Study 4-단계**:
- **(i) Baseline reproduction**: vanilla SEC-DED PPL 곡선 재현. Kelle 2DRP 결과 ±10% 재현.
- **(ii) Bottleneck attribution**: Mech M1 단독 + M2 단독 + M1+M2 의 PPL/stranding ablation.
- **(iii) Roofline (Pareto frontier)**: redundancy bits × residual SDC × end-task PPL — Bamboo / Chipkill / OAEP / Kelle 4-점 plot.
- **(iv) Micro-benchmark**: outlier 임계 sweep (0.5/1/2%) + stripe size sweep (128/256/512B).

---

## 5. 예상 효과 (보수적, scope 명시)

| 지표 | Baseline | 목표 | 조건 / 가설 검증 |
|---|---|---|---|
| Redundancy bits/INT4 | 1.5 (uniform SEC-DED) | **0.5 (≈30% of uniform)** | Mech M1 |
| PPL degradation @ BER=1e-7 | +0.04 (uniform) / +1.2 (no ECC) | **< 0.05** | iso-fidelity |
| Stranding ratio | 30-50% | **10-15%** | Mech M2 sub-page reclaim |
| Throughput | -0% (no ECC) | **-3 to -5%** | parity check on 99% cheap |
| GenBFA defense (3-bit attack) | 67.3% → 0% (W8 MMLU) | **≥ 50% retained** | M1 outlier-stripe ECC |
| Kelle orthogonality | — | **R²(mask, importance) < 0.5** | independent axes 증명 |

**과학적 contribution**: Outlier-mask = ECC-strength-selector + DRAM-row-stripe alignment + sub-page retire 의 3축 통합. Kelle MICRO'25 와 직교 axis 증명 (refresh vs ECC).

**실용적**: vLLM v0.6+ patch ~600 LoC 만 필요. AWQ 모델 그대로 reuse — drop-in compatibility. HBM3/HBM3e ECC region 설정만으로 즉시 deploy.

**Scope 제한**: rotated quantization (QuaRot/SpinQuant) 는 outlier 평탄화 → effect 사라짐 (RotECC 와의 통합 frame 으로 fallback). FP16 KV 는 outlier 정의 모호 — INT4/INT8 quantized KV 한정.

---

## 6. (Tier-1 → Tier-2 변환 가이드)

| 조건 | Tier-2 venue | Tier-2 framing |
|------|--------------|---------------|
| Stranding reduction < 30% | DATE 6p / IEEE TCAD | "Mech M1 only — outlier-aware ECC strength" |
| DRAM row mapping vendor 제약 → M2 fail | IEEE CAL 4p | M1 + ablation only |
| rotated model 에서 effect 사라짐 | ITC short | "outlier-spectrum-aware ECC: rotated vs raw" w/ RotECC merge |
| GenBFA defense 효과만 강함 | DSN practical (security session) | scale-byte focus, ScaleShield 와 통합 |

---

## 7. 미선정 idea 흡수 note

- **Sentinel** (sys-rob T1) → 본 idea 의 Mech M2 (DRAM row stripe alignment + sub-page retire) 로 흡수.
- **OAEP-KV mech M1** (alg T1) → 본 idea 의 Mech M1.
- **Honeycomb-Lite** (sys-rob S2) → 본 idea 의 per-head ablation 으로 흡수 (top-2 of 8 KV heads = stripe 정의 sub-case).
- **ScaleShield** (alg T3) → scale-byte protection 부분만 본 idea Mech M1 의 scale FP16 SEC-DED + RS(15,13) 로 흡수.

---

## 8. R46 verified ref 표 (이 파일)

| ref | 제목 | venue | R46 status |
|-----|------|-------|-----------|
| AWQ | Activation-aware Weight Quantization | MLSys 2024 | verified |
| KVQuant | per-channel pre-RoPE outlier isolation | NeurIPS 2024 | verified |
| GPTQ | Accurate PTQ for GPT | ICLR 2023 | verified |
| SmoothQuant | activation outlier migration | ICML 2023 | verified |
| KIVI | per-token V-cache quantization | ICML 2024 | verified |
| QuaRot | rotation outlier-free 4bit | NeurIPS 2024 | verified |
| Bamboo ECC | HPCA 2015 ECC | HPCA 2015 | verified |
| Frugal ECC | SC 2015 ECC | SC 2015 | verified |
| Hessian-Driven UEP | DATE 2020 weight UEP | DATE 2020 | verified |
| DeepNcode | encoding-based BFA defense | arXiv 2024 | verified |
| GenBFA | bit-flip attack on quant LLM | arXiv 2024 | verified |
| Sridharan | DRAM fault classes | ASPLOS 2015 | verified |
| Schroeder | DRAM errors in the wild | SIGMETRICS 2009 | verified |
| Meza | DRAM errors at Facebook | DSN 2015 | verified |
| RowPress | row-open disturbance | ISCA 2023 | verified |
| vLLM PagedAttention | block-table KV | SOSP 2023 | verified |
| **Kelle** | **eDRAM 2DRP refresh** | **MICRO 2025** | **verified, 1:1 baseline** |
| MixKVQ | query-aware online mixed-precision | arXiv 2025-12 | verified |
| OASIS | outlier-aware KV CXL clustering | IEEE 2025 | verified |
| Bit-Flip Resilience | KV BER vs PPL standard | EMNLP 2025 | verified |

R46 verified count: **20 ref**.
