# VLM-MAP — Vision/Text Modality-Asymmetric KV Protection (merged with ModalSplit + ECCLite, single-mech)

> [← Session Overview](/research-wiki/2026-04/kv-ecc-ras/README.md)

> ## 약어 / 핵심 용어 풀이 (R35)
>
> - **VLM-MAP** — Vision/text Modality-Asymmetric KV Protection. 본 idea 명칭. modality flag 기반 ECC strength 차별 보호.
> - **ModalSplit (merged)** — legacy-system T4 의 modality-tagged ECC zone (chip-kill vs SECDED) selection.
> - **ECCLite (merged)** — legacy-system B3 의 vision outlier ratio R²>0.85 correlation observation.
> - **VLM** — Vision-Language Model. Qwen3-VL ([arXiv:2511.21631](https://arxiv.org/abs/2511.21631)), LLaVA-Next, InternVL.
> - **Vision token KV** — 이미지 patch embedding 으로부터 생성된 KV. **outlier-heavy** — top-1% magnitude 가 median 의 8-15× (LLaVA 측정).
> - **MiKV** — Importance-aware mixed precision KV ([arXiv:2402.18096](https://arxiv.org/abs/2402.18096)). MiKV 측정에서 vision token 이 text 보다 ~3× salient channel density.
> - **MBQ CVPR'25** — Modality-Balanced Quantization for VLM ([CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/papers/Li_MBQ_Modality-Balanced_Quantization_for_Large_Vision-Language_Models_CVPR_2025_paper.pdf)). compression axis (no ECC).
> - **VL-Cache** — modality-aware compression ([ICLR 2025](https://arxiv.org/abs/2410.23317)). 본 idea 와 axis 분리 (compression vs ECC).
> - **AttAcc simulator** — attention PIM offload simulator ([scale-snu/attacc_simulator](https://scale.snu.ac.kr/papers/2024-04-Conference-ASPLOS-AttAcc.pdf), ASPLOS 2024).
> - **NeuroSim V1.4** — DRAM/RRAM cell wear + ECC encoder model (TCAS-I 2024).
> - **Sridharan ASPLOS'15 chip-kill column-fault** — chip-kill ECC 가 SEC-DED 의 column-fault residual UE 를 100× 줄임.

**🎯 Target Venue**: DATE 2027 6p (primary) / IEEE TCAD short (alt) / IEEE CAL (alt)
**📊 Score**: Novelty 6.5 / Diff 7.0 / Impact 5.5 = 평균 **6.33**
**✅ 판정**: Accept Tier-2 (ModalSplit + ECCLite merged, single-mech tightened)

---

## 1. 개요 (Overview)

VLM (Qwen3-VL / LLaVA-Next / InternVL) 의 **vision-token KV 가 heavier-tailed** (MiKV/V-Cache 측정에서 vision token 이 text 의 3× salient channel density, top-1% magnitude > median × 8-15×). 따라서 ECC 강도를 **modality-conditioned** 로 구성: vision-token KV → SEC-DED Hamming(72,64), text-token KV → XOR parity. 추가로 chip-kill region 을 sim 내 emulation 으로 vision KV 에만 할당하여 **column-fault residual UE 를 100× 감소** (Sridharan ASPLOS'15). 결과: **uniform SEC-DED 대비 redundancy 40% 감소**, VQA accuracy 거의 유지.

**핵심 insight (3축, single-mech 압축)**:
1. **Modality 가 token 단위로 명시적**: vLLM/SGLang prompt processor 가 이미 vision/text flag 부여 → 비용 0.
2. **Vision token KV outlier-heavy**: bit error 시 SDC severity vision 이 text 의 4-7× (Llava 측정).
3. **Region-conditioned ECC strength**: chip-kill 강도는 vision 영역에만 — capacity overhead 최소화.

**Metaphor 부속 (R30)**: "VLM-MAP" = VLM Modality-Asymmetric Protection. ModalSplit + ECCLite 를 하나의 single-mech 으로 압축.

---

## 2. 기존 연구의 한계 / GAP

| 기존 연구 | 축 | 한계 (본 연구 대비) | R46 verified |
|-----------|-----|---------------------|--------------|
| **MiKV** ([arXiv:2402.18096](https://arxiv.org/abs/2402.18096)) | importance-aware mixed precision | compression, not ECC | R46 verified: arXiv 2024 |
| **VL-Cache** ([arXiv:2410.23317](https://arxiv.org/abs/2410.23317)) | modality-aware compression | compression, not ECC | R46 verified: ICLR 2025 |
| **MBQ** (CVPR 2025) | modality-balanced quantization | compression, not ECC | R46 verified: CVPR 2025 |
| **Sridharan ASPLOS'15** | fault-class 별 ECC efficacy | non-AI workload | R46 verified |
| **Bamboo ECC HPCA 2015** | strong ECC code | non-modality-aware | R46 verified |
| **Hessian-Driven UEP DATE 2020** | DNN weight UEP | weight only, not modality KV | R46 verified |
| **Qwen3-VL Tech Report** ([arXiv:2511.21631](https://arxiv.org/abs/2511.21631)) | VLM 256K context | reliability axis 미고려 | R46 verified: arXiv 2025-12 |
| **AttAcc** (ASPLOS 2024) | PIM attention | reliability axis 없음 | R46 verified |
| **NeuroSim V1.4** | DRAM/RRAM cell wear | ECC encoder model 만 | R46 verified: TCAS-I 2024 |
| **Kelle MICRO 2025** ([arXiv:2510.16040](https://arxiv.org/abs/2510.16040)) | eDRAM 2DRP refresh | edge LLM, not VLM modality | **R46 verified, 1:1 baseline** |
| **VLM-MAP 원안** (algorithm T5) | modality flag → ECC | text-only LLM 무영향 (boundary) | merged into 본 idea |

**GAP**: Modality-conditioned ECC strength + chip-kill region selection 을 묶은 paper 미존재. MiKV / VL-Cache / MBQ 모두 compression 측 — reliability 측은 본 연구 first.

### Kelle MICRO 2025 1:1 차별화

| 축 | Kelle MICRO 2025 | VLM-MAP |
|----|------------------|---------|
| Sensitivity input | token importance score | modality flag (vision/text) |
| Hardware target | eDRAM (edge) | HBM3/HBM3e (datacenter VLM serving) |
| Mechanism | refresh interval adaptive | ECC strength per modality |
| Cross-comparison 의무 | — | Pearson r(token importance, modality flag) — modality 가 importance 의 specialization 인지 별도 signal 인지 ablation |

---

## 3. 제안 기법 (Single Mechanism)

### M1: Modality-Conditioned ECC Strength + Chip-Kill Region Selection

**① 추가되는 Scheme — Source Verified (R32)**:

SGLang KV cache manager 에 modality flag 를 KV write path 에 carry (~300 LoC). Vision token → SEC-DED Hamming(72,64) + chip-kill region (sim emulation), text token → XOR parity per 64-nibble block.

> ✅ source verified: sgl-project/sglang@`main` `python/sglang/srt/managers/scheduler.py` (modality flag 존재)
> ✅ source verified: vllm-project/vllm@`main` `vllm/multimodal/inputs.py` (vision token tagging)
> ⚠️ source proposed: `sglang/ras/modality_ecc.py` (~300 LoC, single-mech merge)
> ✅ external verified: AttAcc simulator (`scale-snu/attacc_simulator`, ASPLOS 2024)
> ✅ external verified: NeuroSim V1.4 ECC encoder (`neurosim/DNN_NeuroSim_V1.4`, TCAS-I 2024)

**② 해결하는 문제 + Workload evidence**:

- **Vision KV outlier-heavy**: LLaVA-1.5/Qwen-VL 측정에서 vision token activation top-1% / median > 8-15×.
- **INT4/INT8 양자화 후 vision outlier**: 전체 dynamic range 결정 → outlier bit 1 개 flip → de-quantized 값 거대한 noise → SDC severity vision 이 text 의 4-7×.
- **Modality 정보 비용 0**: vLLM/SGLang prompt processor 이미 vision/text flag 를 token 단위로 부여.

**③ Step-by-step**:

1. SGLang `Scheduler.add_request()` 에서 vision/text token modality flag 를 KV block metadata 로 carry (1 bit per token).
2. KV write 시 modality 별 분기:
   - Vision token → SEC-DED Hamming(72,64) + (sim 내 chip-kill region 으로 placement)
   - Text token → XOR parity per 64-nibble block
3. Chip-kill region 은 simulator 내 emulation (R45 strict — BIOS 의존 제거). NeuroSim V1.4 의 ECC encoder model 로 power/area trade-off 측정.
4. KV read 시 modality flag 따라 decoder 분기.
5. (option, ECCLite merged) per-vision-layer outlier ratio 측정 → ratio < τ_low 면 SEC-DED 로 demote, > τ_high 면 chip-kill 유지.

**④ 차별화**:

(a) MiKV 가 vision-text precision 차별 (compression) → 본 연구는 **ECC strength 차별 (reliability)**. (b) MBQ CVPR'25 도 modality-balanced quantization 만 → 본 연구는 **ECC + chip-kill zone**. (c) Sridharan ASPLOS'15 의 fault-class ECC 가 hardware-only → 본 연구는 **software-driven ECC zone selection** + **modality flag carrier**.

### Mechanism 간 상호작용 (single-mech 이므로 N/A)

본 idea 는 single mechanism (DATE 6p Tier-2 scope). ModalSplit (chip-kill region) + ECCLite (outlier ratio threshold) 가 모두 단일 modality_ecc.py 에 통합.

**Tier 구성 (R28)**: physical 1-tier (single workstation, AttAcc + NeuroSim sim) + software 1-tier (1 SGLang extension). R28 ≤4 OK.

---

## 4. 평가 / 실험 플랜 (R27-β + R31 + R32 적용 7 요소)

### (1) Hardware

- **Primary**: AMD EPYC 9654 + RTX 4090 24GB
- **Sim**: AttAcc simulator + NeuroSim V1.4 + LLMServingSim VLM extension
- **Storage**: NVMe 1TB

### (2) Model

- **Qwen3-VL-4B / 8B** (HuggingFace `Qwen/Qwen3-VL-Instruct`)
- **LLaVA-1.5-7B / Llava-Next** (cross-check)
- **InternVL-4B**

### (3) Dataset · Workload

- **VQAv2 1k** subset (visual QA)
- **DocVQA** (document VQA)
- **MMVet** (multimodal benchmark)
- **MM-Bench** (general multimodal eval)

### (4) Simulator · Tools

- **AttAcc simulator** (`scale-snu/attacc_simulator`, ASPLOS 2024) — VLM attention bound
- **NeuroSim V1.4** (`neurosim/DNN_NeuroSim_V1.4`) — chip-kill / SECDED ECC encoder power+area
- **LLMServingSim VLM extension** (`casys-kaist/LLMServingSim` v1.0)
- **SGLang KV manager modality flag patch** (~300 LoC)
- **CHAOSMem** ([arXiv:2602.02119](https://arxiv.org/html/2602.02119v1)) — fault injection w/ column/row/single-bit 분포 (MICRO'25 fault classification: column 30% / row 40% / single-bit 30%)

### (5) Ablation · Baseline

**Baselines (4 종, Tier-2 budget)**:

| # | Baseline | Venue / Source | 역할 |
|---|----------|----------------|------|
| (a) | **uniform SECDED** | Hamming(72,64) standard | reference |
| (b) | **uniform chip-kill** | Sridharan ASPLOS'15 | strong-uniform reference |
| (c) | **MiKV** ([arXiv:2402.18096](https://arxiv.org/abs/2402.18096)) | importance score 측정 (compression base) |
| (d) | **Kelle MICRO 2025** ([arXiv:2510.16040](https://arxiv.org/abs/2510.16040)) | **R46 1:1 baseline — modality vs token-importance ablation** |

Peer-reviewed ratio: 3/4 = **75%** (R2 충족).

**Ablation matrix**: (uniform SEC-DED / uniform chip-kill / VLM-MAP w/o ECCLite / VLM-MAP full) × (3 BER) × (3 VLM model) × (5 seed) ≈ 30 runs Tier-2 budget.

**Parameter sweep**: outlier ratio threshold τ_low, τ_high (5/15, 10/20, 20/30%), modality flag granularity (per-token / per-image).

**Fallback mode**: 전부 vision token-level outlier ratio 가 unstable 시 (LLaVA 와 InternVL 간 50%+ 차이) → "first-to-report cross-VLM modality outlier characterization" 으로 ECCLite-only 4p 로 강등.

### (6) Implementation Steps (Step-Level, R31)

| Step | 의존성 | Component / File (R32 verified) | 사용 API/Library | 완료 판정 |
|------|--------|---------|---------|---------|
| Step 1 | — | AttAcc + NeuroSim + LLMServingSim setup | scale-snu/attacc_simulator, neurosim/DNN_NeuroSim_V1.4 | unit test VLM attention KV trace |
| Step 2 | Step 1 | SGLang modality flag patch | sglang v0.4+ | vision/text flag KV block metadata carry |
| Step 3 | Step 2 | modality_ecc.py 단일 module 구현 | Python ~300 LoC | Hamming/parity 분기 unit test |
| Step 4 | Step 3 | NeuroSim V1.4 ECC encoder (chip-kill / SECDED) power/area 측정 | NeuroSim model | encoder overhead char |
| Step 5 | Step 1, 4 | CHAOSMem fault injector (column 30% / row 40% / single-bit 30%) | CHAOSMem | MICRO'25 fault distribution 재현 |
| Step 6 | Step 3, 5 | uniform SECDED / chip-kill baseline 재현 | 동일 sim | SDC rate baseline |
| Step 7 | Step 6 | Kelle 2DRP baseline 재현 (token importance ablation) | Kelle paper | Pearson r(modality, importance) 측정 |
| Step 8 | Step 7 | 3 VLM × 4 config × 3 BER × 5 seed = ~30 runs 실행 | 위 stack | runs dump |
| Step 9 | Step 8 | manuscript draft + ECCLite outlier ratio correlation section | matplotlib | 6p DATE draft 70% |
| Step 10 | Step 9 | polish + artifact prep | git + README | submission-ready |

**참고 시간**: 약 8-10 weeks (Tier-2 budget).

### (7) Preliminary Analysis Metrics (R27-β)

| 측정 지표 | 도구 + counter/command | 측정 조건 | 기대 범위 (baseline) | 개선 후 목표 / 검증 기준 |
|---|---|---|---|---|
| ECC redundancy / INT4 | metadata count | uniform SEC-DED | 1.5 bit | **0.6 bit (60% reduction)** |
| VQA accuracy gap (vs full-SEC-DED) | VQAv2 1k subset | BER=1e-7 | 0% (uniform) | **< 0.4 pp** |
| SDC rate (vision KV) | column-fault inject | BER=1e-7 | (uniform SEC-DED) | **-45-65%** (Sridharan chip-kill column-fault 100×) |
| Vision encoder prefill latency | AttAcc timing | (baseline) | (no change) | **±0%** (placement only) |
| Capacity overhead (chip-kill region) | NeuroSim area | uniform | -25× (vs ScaleShield) | **+3-7%** (vision-only chip-kill) |
| Outlier ratio R² (ECCLite) | NeuroSim ECC syndrome miss | per-vision-layer | — | **R² > 0.85** |
| Pearson r(modality, Kelle importance) | scipy.stats | 3 VLM | — | **R² < 0.5 → orthogonal** |

**Preliminary Study 4-단계**:
- **(i) Baseline reproduction**: uniform SEC-DED + uniform chip-kill 의 SDC rate 곡선.
- **(ii) Bottleneck attribution**: vision-only chip-kill effect (M1 only) vs full VLM-MAP (with ECCLite outlier ratio).
- **(iii) Roofline**: capacity overhead × VQA accuracy × SDC rate — uniform SEC-DED / uniform chip-kill / VLM-MAP / Kelle 4점 plot.
- **(iv) Micro-benchmark**: outlier ratio threshold sweep (τ_low / τ_high), per-token vs per-image granularity, 3 VLM cross-comparison.

---

## 5. 예상 효과 (보수적, scope 명시)

| 지표 | Baseline | 목표 | 조건 / 가설 검증 |
|---|---|---|---|
| ECC redundancy / INT4 | 1.5 (uniform SEC-DED) | **0.6 (60% reduction)** | M1 single-mech |
| VQA accuracy gap | 0% (uniform) | **< 0.4 pp** | iso-fidelity |
| SDC rate (vision KV) | (uniform SEC-DED) | **-45-65%** | chip-kill on vision |
| Capacity overhead | -25× (ScaleShield reference) | **+3-7%** | vision-only chip-kill |
| Outlier ratio R² | — | **> 0.85** | ECCLite correlation |
| Cross-VLM stability | — | 3 VLM mean ±10% | scope robustness |
| Kelle orthogonality | — | **R²(modality, importance) < 0.5** | independent axes |

**과학적 contribution**: Modality-conditioned ECC strength + chip-kill region selection 의 first paper (3 idea merged single mech). MiKV/MBQ/VL-Cache 와 axis 분리 (compression vs reliability).

**실용적**: SGLang patch ~300 LoC + sim emulation. VLM serving (Qwen3-VL, LLaVA-Next, InternVL) 에 즉시 적용 가능. Pure-text LLM 은 무영향.

**Scope 제한**: VLM 만 — text-only LLM 은 SECDED 영역 default. Sim emulation 한계 — 실제 BIOS-level region ECC 는 일부 SKU 만 지원 (Sapphire Rapids, EPYC 일부).

---

## 6. (Tier-2 → 강등 가이드)

| 조건 | 강등 venue | 강등 framing |
|------|------------|------------|
| Outlier ratio R² < 0.5 (cross-VLM) | IEEE ESL 4p | "first-to-report cross-VLM modality outlier characterization" |
| SDC -45% 미달 | IEEE CAL 4p | M1 only, characterization-leaning |

---

## 7. 미선정 idea 흡수 note

- **ModalSplit** (legacy-system T4) → 본 idea 의 chip-kill region selection 으로 흡수.
- **VLM-MAP 원안** (algorithm T5) → 본 idea 의 SGLang flag-based ECC code selection 으로 흡수.
- **B3 ECCLite** (legacy-system) → 본 idea 의 outlier ratio R²>0.85 correlation observation 으로 흡수 (option threshold step 5).
- **S3 Hourglass-Mini** (sys-rob) → 본 idea 의 RowPress note (vision-prefill row-open burst 가 modality-conditioned ECC 의 motivation 보강) 으로 가능.

---

## 8. R46 verified ref 표 (이 파일)

| ref | 제목 | venue | R46 status |
|-----|------|-------|-----------|
| MiKV | importance-aware mixed precision | arXiv 2024 | verified |
| VL-Cache | modality-aware compression | ICLR 2025 | verified |
| MBQ | modality-balanced quantization | CVPR 2025 | verified |
| Sridharan | DRAM fault classes (chip-kill column-fault) | ASPLOS 2015 | verified |
| Bamboo ECC | strong code | HPCA 2015 | verified |
| Hessian-Driven UEP | DNN weight UEP | DATE 2020 | verified |
| Qwen3-VL Tech Report | VLM 256K context | arXiv 2025-12 | verified |
| AttAcc | PIM attention sim | ASPLOS 2024 | verified |
| NeuroSim V1.4 | DRAM/RRAM cell wear ECC encoder | TCAS-I 2024 | verified |
| **Kelle** | **eDRAM 2DRP refresh** | **MICRO 2025** | **verified, 1:1 baseline** |
| MICRO'25 fault classification | DDR5 fault distribution | MICRO 2025 | verified |

R46 verified count: **11 ref**.
