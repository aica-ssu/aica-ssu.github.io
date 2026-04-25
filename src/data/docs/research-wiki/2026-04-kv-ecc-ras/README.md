# KV Cache Memory ECC + RAS for Quantization-Aware AI Serving (Session #2 Overview)

> **Date**: 2026-04-25 (#2 of the day) · **Mode**: 1 (sentence input) · **Lab**: SSU AICA · **Student**: 학생 (KV cache memory reliability track)
>
> **Bundle 형식**: 2026-04-25 R28-R36 + R44-R46 hierarchical 적용. 학생 / AI 가 본 README + 진행하려는 idea 파일 1-2개만 읽어도 (a) 전체 ideation flow 이해, (b) 시작점 결정, (c) Tier 분기 판단 자율 수행 가능.
>
> **Phase 1 staging (3 expert)**: system-robustness · algorithm · legacy-system
>
> **Phase 2 staging (3 reviewer)**: novelty · diff · impact

---

## 약어 / 핵심 용어 풀이 (R35)

본 세션 전체에서 등장하는 핵심 약어 + 알고리즘 + 도메인 용어. 각 idea 파일도 자기에게 필요한 것을 다시 풀이 (학생 self-sufficient 위해 중복 OK).

- **KV cache** — Transformer decoder 의 attention 에서 매 토큰의 Key (K) / Value (V) 를 재계산 없이 재사용하기 위해 layer 별로 저장하는 tensor pool. Llama-3-8B 32K 컨텍스트 ≈ 5GB, 256K ≈ 20GB 이상으로 LLM serving 의 가장 큰 메모리 면적이자 reliability 노출 표면.
- **ECC** (Error Correcting Code) — 메모리 비트 오류를 검출/정정하는 redundancy 코드. 본 세션 핵심 패턴은 SEC-DED (single-error-correct, double-error-detect Hamming(72,64)) / Bamboo / Frugal / Chipkill / Reed-Solomon.
- **SEC-DED** — Single-Error-Correct, Double-Error-Detect Hamming code. 표준 ECC DRAM 의 default. 1bit 정정 / 2bit 검출. 64bit data 당 8 redundancy bit (12.5%).
- **Chipkill ECC** — DRAM chip 1개 전체 fail 도 정정 가능한 강력 ECC. 데이터센터급 표준. SEC-DED 대비 column-fault residual 100배 감소 (Sridharan ASPLOS'15).
- **Bamboo ECC** — HPCA 2015 (Kim/Sullivan/Erez) 의 강력 ECC code family. DEC-3 (triple-error-correct) 까지 16-symbol 단위 정정. 본 세션의 outlier-stripe 강결합 ECC backbone.
- **DRAM SDC vs DUE** — SDC (Silent Data Corruption) = ECC가 검출 못한 채 잘못된 값 반환. DUE (Detected Uncorrectable Error) = 검출했지만 정정 못 한 경우. SDC 는 silent 라 가장 위험; HBM3 on-die ECC 가 SDC 를 "조용히" 키우는 부작용 (MICRO 2023).
- **Page Retirement** — Linux memory_failure() / mcelog 의 default RAS 정책. CE (correctable error) 누적 또는 UE (uncorrectable error) 발생 시 4KB page 또는 2MB hugepage 단위로 OS 가 영구 격리. **Memory stranding** 의 주 원인.
- **Memory Stranding** — fault-affected bytes 대비 retired bytes 의 초과량 (`(retired − fault_affected)/fault_affected`). vLLM 16-token block (512B-2KB) 손상 시 4KB/2MB hugepage 단위 retire → 30-50% stranding 발생.
- **Outlier Channel** — AWQ/KVQuant/SmoothQuant 가 보고한 LLM activation 의 1% (heavy-tail) 채널이 99% 의 quantization error 와 bit-flip sensitivity 를 차지하는 비대칭. Pre-RoPE Key 에서 channel 단위로 spatially clustered (KVQuant NeurIPS 2024).
- **Per-token vs Per-channel quantization** — Per-token (KIVI V-cache) = 토큰마다 별도 scale FP16; Per-channel (KIVI K-cache) = 채널마다 별도 scale FP16. 두 경우 모두 scale 이 KV bytes 의 0.5-2% 에 불과하나 한 비트 flip 시 전체 token/channel 손상.
- **Scale Tensor** — 양자화 후 INT4/INT8 mantissa 와 함께 저장되는 FP16 floating-point scaling factor. mantissa bit-flip 영향은 scale × 8/16 로 bounded; scale FP16 mantissa bit-7 flip 은 2× shift → 전체 token catastrophic.
- **RowPress** — ISCA 2023 (ETH Zurich, Mutlu group). Row-open-time 이 길수록 인접 row 의 read disturbance가 18-160배 증폭. **Prefill row-open burst** = LLM serving 의 RowPress 위험 phase.
- **vLLM PagedAttention** — SOSP 2023 (Kwon, UC Berkeley). KV cache 를 16-token block (≈ 512B-4KB) 단위 contiguous physical block 으로 저장, block table 로 logical→physical 매핑. **본 세션 모든 idea 의 핵심 granularity unit**.
- **Llama-3** — Meta Llama-3-8B/70B-Instruct (2024-04). 본 세션의 main LLM workload. 32K-128K context, GQA (Grouped-Query Attention) 8 KV heads.
- **Qwen3-VL** — Alibaba Qwen3-VL (2025-12, [arXiv:2511.21631](https://arxiv.org/abs/2511.21631)). 256K context 까지 지원하는 VLM, 2B/4B/8B/32B dense + MoE. Vision-token KV 가 outlier-heavy → 본 세션 VLM idea 의 직접 대상.
- **GenBFA** — [arXiv:2411.13757](https://arxiv.org/abs/2411.13757) (Sandia/Purdue). Bit-flip attack on quantized LLM. 3 critical bit flip → LLaMA3-8B-W8 MMLU 67.3% → 0%. 본 세션 Chameleon/ScaleShield 의 직접 attack baseline.
- **Kelle MICRO 2025** — [arXiv:2510.16040](https://arxiv.org/abs/2510.16040). eDRAM KV cache 의 retention 을 (bit-position MSB/LSB × token importance score) 2DRP (2D adaptive refresh policy) 로 제어. **본 세션 모든 affected idea 의 1:1 closest competitor**, R46 baseline 의무 반영.
- **AWQ / GPTQ / SmoothQuant / QuaRot** — LLM weight/activation quantization 표준. AWQ = Activation-aware Weight Quantization (1% outlier protect, MLSys 2024 Best Paper); QuaRot = Hadamard rotation 으로 outlier 평탄화 (NeurIPS 2024).
- **Simulator stack (R45.9 active)** — gem5 + DRAMSim3 (cycle-accurate DRAM), ChampSim (cache+memory trace), AttAcc (PIM attention), NeuPIMs (NPU+PIM 이종), NeuroSim V1.4 (DRAM/RRAM cell wear), LLMServingSim (KAIST CASYS, MIT license, ISPASS 2026 v1.0). 모두 GitHub active maintenance 확인됨.
- **Tier-1 vs Tier-2 (venue)** — Tier-1 = ASPLOS / OSDI / SOSP / DSN / HPCA / MICRO / MLSys / NeurIPS / ICLR (long paper). Tier-2 = DATE / ITC / IOLTS / IEEE TCAD / IEEE CAL / IEEE ESL (short 4-8p / journal short).
- **Self-scoop pair** — 본 세션 내 다른 expert 간 mechanism 60-70%+ 겹침. R5 differentiation-critique 규칙으로 merge 또는 venue 분리 결정. 본 세션 6쌍: BlockShard↔Lattice / OAEP-KV↔Sentinel / ScaleShield↔Chameleon / ModalSplit↔VLM-MAP↔ECCLite / EntropyECC↔Driftwood / ScaleShield M2↔Hourglass M2.
- **R45 strict** — Implementation surface = application-level (vLLM/SGLang KV manager + PyTorch/Triton kernel + Linux soft-offline ABI + Python harness). Evaluation surface = simulator-only (gem5/ChampSim/AttAcc/NeuPIMs/NeuroSim V1.4/LLMServingSim). No FPGA, no real-DRAM rowhammer rig, no kernel patch.
- **R46 verified** — 모든 인용에 WebFetch/WebSearch 결과 (title + author + venue) 검증. 본 세션 50+ 핵심 ref 모두 R46 통과.

---

## 0. Executive Summary

### 0.1 사용자 입력 정리

사용자 쿼리 핵심: **"LLM/VLM prefill-stage KV cache memory ECC + RAS (page retirement / migration / stranding) — quantization-aware serving 환경에서 outlier 와 modality 와 layer access pattern 을 ECC 와 retire 정책에 활용하는 unique novelty 탐색"**.

세 expert 가 axis 를 분담:
- **algorithm-expert** — outlier mask / scale tensor / block-entropy / rotation 의 quantization-side novelty 7편
- **system-robustness-expert** — DRAM-row-stripe alignment / lifecycle-phase / vLLM-block ECC / metadata-defense 의 robustness-side 7편
- **legacy-system-expert** — block-granular page retirement / layer-tier migration / ML predictor / VLM modality split 의 RAS-side 7편

총 21편 → Phase 1' improve-first 정제 + Phase 2 (novelty/diff/impact) 3-reviewer 평가 + Kelle MICRO 2025 baseline 보강 의무 반영 → Phase 1'' 6편 최종 선정 (Tier-1 Top 3 + Tier-2 독립 Top 3) + 미선정 15편.

### 0.2 Tier-1 Top 3 (DSN/HPCA/MICRO/ASPLOS/OSDI 급)

| Rank | Title | 출처 | Score (N/D/I) | R45 risk | Target venue | 링크 |
|------|-------|------|--------------:|----------|--------------|------|
| 🥇 | **OAEP-KV (merged with Sentinel)** — Outlier-Aware Asymmetric ECC for Quantized KV Cache, with DRAM-Row-Stripe Alignment + Sub-Page Retirement | algorithm + system-robustness | **7.2 / 7.0 / 8.0 = 7.40** | 낮음 (vLLM-only) | DSN 2027 / HPCA 2027 / MICRO 2027 | [tier1/01-oaep-kv.md](/research-wiki/2026-04/kv-ecc-ras/tier1/01-oaep-kv.md) |
| 🥈 | **BlockShard (merged with Lattice)** — Block-Granular KV Page Retirement + vLLM-RFC-#19329-aware Recompute Reschedule | legacy-system + system-robustness | **8.0 / 7.0 / 9.0 = 8.00** | 중간 (Linux ABI 제안) | ASPLOS 2027 / OSDI 2027 / DSN 2027 | [tier1/02-blockshard.md](/research-wiki/2026-04/kv-ecc-ras/tier1/02-blockshard.md) |
| 🥉 | **LayerTier** — Layer-Wise KV Page Migration with Access-Frequency-Awareness + Reliability Zone | legacy-system | **6.8 / 6.0 / 7.5 = 6.77** | 중간 (zone abstraction) | MICRO 2027 / DSN 2027 / OSDI 2027 | [tier1/03-layertier.md](/research-wiki/2026-04/kv-ecc-ras/tier1/03-layertier.md) |

### 0.3 Tier-2 독립 Top 3 (DATE/ITC/IEEE TCAD 4-8p)

| Rank | Title | 출처 | Score (N/D/I) | R45 risk | Target venue | 링크 |
|------|-------|------|--------------:|----------|--------------|------|
| T1 | **VLM-MAP (merged with ModalSplit + ECCLite)** — Vision/Text Modality-Asymmetric KV Protection (single-mech) | algorithm + legacy-system | **6.5 / 7.0 / 5.5 = 6.33** | 낮음 (sim only) | DATE 2027 6p / IEEE TCAD short | [tier2/01-vlm-map.md](/research-wiki/2026-04/kv-ecc-ras/tier2/01-vlm-map.md) |
| T2 | **B1 FrostFloor** — Layer-Wise Frequency-Tier KV Migration (single-config DATE 6p) | legacy-system | **5.0 / 5.5 / 4.5 = 5.00** | 낮음 (Jetson sim) | DATE 2027 6p | [tier2/02-frostfloor.md](/research-wiki/2026-04/kv-ecc-ras/tier2/02-frostfloor.md) |
| T3 | **EntropyECC (merged with Driftwood characterization)** — Block-Entropy Adaptive ECC for Per-Token KV | algorithm + system-robustness | **6.0 / 5.0 / 6.5 = 5.83** | 낮음 (ChampSim) | ITC 2027 / MTS / IEEE TCAD | [tier2/03-entropy-ecc.md](/research-wiki/2026-04/kv-ecc-ras/tier2/03-entropy-ecc.md) |

### 0.4 미선정 15편 + Kelle 보강

원안 21 → 최종 6 (Tier-1 3 + Tier-2 3) → 미선정 15. 상세는 [unselected.md](/research-wiki/2026-04/kv-ecc-ras/unselected.md). 핵심 사유:
- **Self-scoop merge** 6 (Sentinel/Lattice/Chameleon/ModalSplit/ECCLite/Driftwood 등)
- **R45 borderline** 2 (Linux kernel patch 또는 BIOS region 의존)
- **Triviality / 단일 관찰** 4 (FrostFloor B1 자체는 Tier-2 로 채택; Honeycomb-Lite / MigGate / B3 ECCLite drop)
- **Architecture-fragile** 2 (DeepStackECC / KV-Wear-Sketch — Tier-2 add-on 가능)
- **Pipeline-conditional** 1 (RotECC — QuaRot 한정)

**Kelle MICRO 2025 baseline 보강 적용 idea (6개)**: Tier-1 OAEP-KV / Tier-1 BlockShard / Tier-2 EntropyECC + 미선정 Sentinel(merged) / Chameleon(merged) / ModalSplit(merged). 각 affected idea 의 1:1 차별화 표 포함.

---

## 1. 학생 연구 실행 흐름 (Post-Ideation Decision Tree, R29/R44)

> 학생 / AI 가 어떤 시뮬레이터 환경에서 시작하고, 결과에 따라 Tier-1 vs Tier-2 분기 결정을 내릴 수 있도록.

<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 880 1280" width="880" height="1280" font-family="Inter, system-ui, sans-serif" font-size="13">
<rect x="0" y="0" width="880" height="1280" fill="#fafafa"/>
<rect x="280" y="20" width="320" height="60" rx="8" fill="#dbeafe" stroke="#2563eb" stroke-width="2"/>
<text x="440" y="44" text-anchor="middle" font-weight="bold">Step 1 — Simulator 환경 selection</text>
<text x="440" y="64" text-anchor="middle">gem5+DRAMSim3 / ChampSim / NeuPIMs / AttAcc / LLMServingSim</text>
<rect x="40" y="110" width="240" height="56" rx="6" fill="#fef3c7" stroke="#d97706"/>
<text x="160" y="132" text-anchor="middle" font-weight="bold">Tier-1 OAEP-KV</text>
<text x="160" y="152" text-anchor="middle">gem5+DRAMSim3 (HBM3 BER inj.)</text>
<rect x="320" y="110" width="240" height="56" rx="6" fill="#fef3c7" stroke="#d97706"/>
<text x="440" y="132" text-anchor="middle" font-weight="bold">Tier-1 BlockShard</text>
<text x="440" y="152" text-anchor="middle">gem5+DRAMSim3 + Linux MCE wrapper</text>
<rect x="600" y="110" width="240" height="56" rx="6" fill="#fef3c7" stroke="#d97706"/>
<text x="720" y="132" text-anchor="middle" font-weight="bold">Tier-1 LayerTier</text>
<text x="720" y="152" text-anchor="middle">ChampSim + LLMServingSim</text>
<line x1="440" y1="80" x2="160" y2="110" stroke="#94a3b8" stroke-width="1"/>
<line x1="440" y1="80" x2="440" y2="110" stroke="#94a3b8" stroke-width="1"/>
<line x1="440" y1="80" x2="720" y2="110" stroke="#94a3b8" stroke-width="1"/>
<rect x="280" y="200" width="320" height="60" rx="8" fill="#dcfce7" stroke="#16a34a" stroke-width="2"/>
<text x="440" y="224" text-anchor="middle" font-weight="bold">Step 2 — Baseline ECC 재현</text>
<text x="440" y="244" text-anchor="middle">vanilla SEC-DED + Linux page retirement (±5% match)</text>
<line x1="160" y1="166" x2="160" y2="195" stroke="#94a3b8" stroke-width="1"/>
<line x1="160" y1="195" x2="440" y2="200" stroke="#94a3b8" stroke-width="1"/>
<line x1="440" y1="166" x2="440" y2="200" stroke="#94a3b8" stroke-width="1"/>
<line x1="720" y1="166" x2="720" y2="195" stroke="#94a3b8" stroke-width="1"/>
<line x1="720" y1="195" x2="440" y2="200" stroke="#94a3b8" stroke-width="1"/>
<rect x="280" y="290" width="320" height="60" rx="8" fill="#dcfce7" stroke="#16a34a" stroke-width="2"/>
<text x="440" y="314" text-anchor="middle" font-weight="bold">Step 3 — Mech #1 단독 구현 (Wk 3-6)</text>
<text x="440" y="334" text-anchor="middle">single mechanism solo eval (BER 1e-9~1e-6, 5 seeds)</text>
<line x1="440" y1="260" x2="440" y2="290" stroke="#94a3b8" stroke-width="1"/>
<rect x="280" y="380" width="320" height="60" rx="8" fill="#dcfce7" stroke="#16a34a" stroke-width="2"/>
<text x="440" y="404" text-anchor="middle" font-weight="bold">Step 4 — Mech #2 통합 + ablation (Wk 7-12)</text>
<text x="440" y="424" text-anchor="middle">5 workload × 3 config × 2 baseline = 30 runs</text>
<line x1="440" y1="350" x2="440" y2="380" stroke="#94a3b8" stroke-width="1"/>
<rect x="240" y="470" width="400" height="80" rx="8" fill="#fef9c3" stroke="#ca8a04" stroke-width="2"/>
<text x="440" y="498" text-anchor="middle" font-weight="bold">Decision — Tier-1 cut</text>
<text x="440" y="518" text-anchor="middle">Memory stranding ≥ 40% reduction</text>
<text x="440" y="536" text-anchor="middle">OR accuracy preservation ≥ 98% under BER 1e-7</text>
<line x1="440" y1="440" x2="440" y2="470" stroke="#94a3b8" stroke-width="1"/>
<rect x="60" y="600" width="320" height="84" rx="8" fill="#fee2e2" stroke="#dc2626" stroke-width="2"/>
<text x="220" y="628" text-anchor="middle" font-weight="bold">Tier-1 (조건 충족)</text>
<text x="220" y="650" text-anchor="middle">DSN 2027 / HPCA 2027 / MICRO 2027</text>
<text x="220" y="668" text-anchor="middle">ASPLOS 2027 / OSDI 2027 (12-13p)</text>
<rect x="500" y="600" width="320" height="84" rx="8" fill="#e0e7ff" stroke="#4338ca" stroke-width="2"/>
<text x="660" y="628" text-anchor="middle" font-weight="bold">Tier-2 (cut 미달)</text>
<text x="660" y="650" text-anchor="middle">DATE 2027 6p / ITC 2027 / IOLTS</text>
<text x="660" y="668" text-anchor="middle">IEEE TCAD short / IEEE CAL / IEEE ESL</text>
<line x1="340" y1="550" x2="220" y2="600" stroke="#94a3b8" stroke-width="1"/>
<line x1="540" y1="550" x2="660" y2="600" stroke="#94a3b8" stroke-width="1"/>
<rect x="280" y="730" width="320" height="60" rx="8" fill="#f3e8ff" stroke="#9333ea" stroke-width="2"/>
<text x="440" y="754" text-anchor="middle" font-weight="bold">Step 5 — Kelle baseline 1:1 보강</text>
<text x="440" y="774" text-anchor="middle">arXiv:2510.16040 R46 verified — 모든 affected idea</text>
<line x1="220" y1="684" x2="440" y2="730" stroke="#94a3b8" stroke-width="1"/>
<line x1="660" y1="684" x2="440" y2="730" stroke="#94a3b8" stroke-width="1"/>
<rect x="280" y="820" width="320" height="60" rx="8" fill="#f3e8ff" stroke="#9333ea" stroke-width="2"/>
<text x="440" y="844" text-anchor="middle" font-weight="bold">Step 6 — 7 element experiment plan (R27-β)</text>
<text x="440" y="864" text-anchor="middle">HW / Model / Dataset / Sim / Baseline / Steps / Metric</text>
<line x1="440" y1="790" x2="440" y2="820" stroke="#94a3b8" stroke-width="1"/>
<rect x="280" y="910" width="320" height="60" rx="8" fill="#fce7f3" stroke="#db2777" stroke-width="2"/>
<text x="440" y="934" text-anchor="middle" font-weight="bold">Step 7 — 시뮬레이션 budget</text>
<text x="440" y="954" text-anchor="middle">5 workload × 3 config × 2 baseline = 30 runs / 4-8주</text>
<line x1="440" y1="880" x2="440" y2="910" stroke="#94a3b8" stroke-width="1"/>
<rect x="40" y="1000" width="240" height="100" rx="8" fill="#fed7aa" stroke="#ea580c"/>
<text x="160" y="1024" text-anchor="middle" font-weight="bold">Workload (5)</text>
<text x="160" y="1046" text-anchor="middle">Llama-3-8B (FP16 + INT4 KIVI)</text>
<text x="160" y="1064" text-anchor="middle">Llama-3-70B (W4G128)</text>
<text x="160" y="1082" text-anchor="middle">Qwen3-VL-8B / Mistral-7B</text>
<rect x="320" y="1000" width="240" height="100" rx="8" fill="#fed7aa" stroke="#ea580c"/>
<text x="440" y="1024" text-anchor="middle" font-weight="bold">Config (3)</text>
<text x="440" y="1046" text-anchor="middle">No-ECC / SEC-DED / Proposed</text>
<text x="440" y="1064" text-anchor="middle">BER 1e-9 / 1e-7 / 1e-6</text>
<text x="440" y="1082" text-anchor="middle">+ Kelle 2DRP comparison</text>
<rect x="600" y="1000" width="240" height="100" rx="8" fill="#fed7aa" stroke="#ea580c"/>
<text x="720" y="1024" text-anchor="middle" font-weight="bold">Baseline (2)</text>
<text x="720" y="1046" text-anchor="middle">Linux mcelog page retire</text>
<text x="720" y="1064" text-anchor="middle">Chipkill always</text>
<text x="720" y="1082" text-anchor="middle">+ R45 simulator-only</text>
<line x1="440" y1="970" x2="160" y2="1000" stroke="#94a3b8" stroke-width="1"/>
<line x1="440" y1="970" x2="440" y2="1000" stroke="#94a3b8" stroke-width="1"/>
<line x1="440" y1="970" x2="720" y2="1000" stroke="#94a3b8" stroke-width="1"/>
<rect x="180" y="1140" width="520" height="120" rx="8" fill="#cffafe" stroke="#0891b2" stroke-width="2"/>
<text x="440" y="1166" text-anchor="middle" font-weight="bold">Final Submission</text>
<text x="440" y="1188" text-anchor="middle">Tier-1: DSN 2027 (Feb deadline) / HPCA 2027 / MICRO 2027 / ASPLOS 2027 / OSDI 2027</text>
<text x="440" y="1210" text-anchor="middle">Tier-2: DATE 2027 (Sep) / ITC 2027 / IEEE TCAD short / IEEE CAL / IOLTS</text>
<text x="440" y="1232" text-anchor="middle">Manuscript draft 70% by Wk 13 → polish + artifact eval Wk 14-16</text>
<line x1="440" y1="1100" x2="440" y2="1140" stroke="#94a3b8" stroke-width="1"/>
</svg>

### Tier-1 Track 가이드 (가장 유망한 ideation 부터 step-by-step)

**Path A — BlockShard Tier-1 (가장 유망, 측정 가능 contribution 가장 강함)**:
- **시작점**: Step 1 gem5+DRAMSim3 setup → Step 2 Linux mcelog 4KB hugepage retirement baseline 재현 → Step 3 Mech #1 (vLLM block manager hook + sub-page reclamation 구현).
- **Tier-1 진입 조건**: 5 workload 평균 stranding 30-50% → 8-12% 측정 + prefill batch capacity +14-22% 확인.
- **추가 의무 (R46 Kelle)**: arXiv:2510.16040 인용 + retirement axis 직교성 정량 비교 plot (Kelle = refresh adaptive / 본 idea = block retire).
- **분기 (Tier-1 → Tier-2)**: stranding ratio 만약 30% → 25% (감소 < 40%) 면 ABI 제안만 분리하여 Tier-2 IEEE TCAD position paper 로 강등.

**Path B — OAEP-KV Tier-1**:
- **시작점**: Step 1 gem5+DRAMSim3 + AWQ outlier mask 추출 → Step 2 vanilla SEC-DED Hamming(72,64) baseline.
- **Tier-1 진입 조건**: BER 1e-7 에서 redundancy bits 30% 수준에서 PPL 가 0.05 이하로 유지 (uniform SEC-DED 와 동등).
- **추가 의무**: Sentinel merge (DRAM-row-stripe alignment) 의 vendor-specific row mapping ablation. Kelle 의 token-importance-score 와 outlier-mask 의 R² 비교 (orthogonality 증명).
- **분기**: rotated-model (QuaRot) 에서 effect 사라지면 RotECC 와 통합한 "outlier-spectrum-aware ECC" framing 으로 재구성.

**Path C — LayerTier Tier-1 (long-context decode-bound 시 가장 효과)**:
- **시작점**: Step 1 ChampSim + LLMServingSim setup → Step 2 LRU baseline 재현.
- **Tier-1 진입 조건**: decode p99 -12-18% + error-induced refetch -40-55%.
- **추가 의무**: KVTuner ICLR'25 (layer-wise sensitivity) 와의 R² 비교 / TTKV 2025 와의 axis 분리 (capacity vs reliability).

### Tier-2 Path 가이드

**Tier-2 Path A — VLM-MAP**:
- **진입 조건**: Path A/B/C Tier-1 cut 미달 시; VLM modality 특화 single-mech.
- **시작점**: AttAcc + LLMServingSim VLM extension.

**Tier-2 Path B — FrostFloor**:
- **진입 조건**: Edge LLM (Jetson Orin LPDDR5) 시나리오; Tier-1 GPU sim 환경 unavailable 시.
- **시작점**: vLLM-edge fork + bitmap allocator.

**Tier-2 Path C — EntropyECC**:
- **진입 조건**: ChampSim trace replay 만 가능 시; entropy predictor 단일 axis 충분.
- **시작점**: ChampSim trace + KIVI quantization + lm-eval-harness.

---

## 2. 연구 진행 Meta

### 2.1 Input

- **사용자 쿼리 핵심**: LLM/VLM prefill-stage KV cache memory ECC + RAS (page retirement / migration / stranding) for quantization-aware AI serving. Outlier 와 modality 와 layer access pattern 의 unique novelty.
- **Mode**: 1 (sentence input)
- **실행 일시**: 2026-04-25 (#2)
- **관련 이전 세션**: 2026-04-23 (Edge VLM Energy), 2026-04-24 (MoE Fingerprint Security+Serving)

### 2.2 접근 방식 (주요 키워드)

- **도메인**: LLM/VLM prefill stage, KV cache HBM3/HBM3e/LPDDR5 memory, vLLM PagedAttention, SGLang RadixAttention
- **관찰**: outlier-1% concentrated sensitivity (AWQ/KVQuant) + scale tensor 0.5% 100% semantic risk + layer access 8-12× asymmetry + vision/text modality outlier 분포 차이 + vLLM 16-token block ≪ 4KB OS page (8-64× stranding factor)
- **제안 기법**: (a) outlier-channel-aware unequal ECC (b) block-granular KV page retirement (c) layer access frequency × reliability zone migration (d) modality-asymmetric ECC (e) block-entropy adaptive ECC
- **타겟 지표**: memory stranding %, prefill batch capacity, decode p99 latency, accuracy under BER, redundancy bits/INT4, SDC rate, page-retirement decision F1

### 2.3 중점적으로 고려한 축

1. **R45 strict (no kernel patch / simulator only)** — 학생 1인 12-16주 feasibility.
2. **R46 strict (verified ref)** — 모든 인용 WebFetch/WebSearch 검증; Kelle MICRO 2025 critical baseline 보강 의무.
3. **Quantization-aware KV cache 의 unique novelty axis** — outlier / scale / modality / layer / lifecycle phase 5축.
4. **Self-scoop 6쌍 명시 처리** — merge 또는 venue 분리.
5. **Tier-1 + Tier-2 paper-pair 전략** — 동일 core idea 의 Top-tier 13p + Tier-2 4-6p 분리 제출 가능성.

### 2.4 의도적으로 제외한 축 + 이유

| 제외 축 | 이유 |
|---------|------|
| **Real DRAM rowhammer rig / FPGA prototype** | R45 strict — 학생 1인 simulator-only 범위. |
| **Hardware ECC encoder ASIC tape-out** | 자원 미가용; NeuroSim V1.4 의 power/area model 로 대체. |
| **Linux kernel mainline patch upstream merge** | R45 borderline — BlockShard 의 ABI 제안만 position paper 로 분리. |
| **Adversarial bit-flip attack ML pipeline** | 본 세션은 reliability axis. GenBFA 는 baseline 으로만 인용. |
| **Cross-system generality (TensorRT-LLM 등)** | 본 세션 vLLM/SGLang 한정. |
| **Distributed/multi-node KV serving (LMCache 2025)** | 본 세션 single-node sim. |

### 2.5 외부 탐색 범위

- Phase 1: 약 50+ 핵심 ref 수집 (peer-reviewed 25 / preprint 25)
- Reference Integrity R1: WebSearch + WebFetch fallback 검증, fabricated 0
- OpenReview verified: KVTuner ICLR'25 ([zDwipF6h06](https://openreview.net/forum?id=zDwipF6h06))
- Phase 2 fresh search 12 쿼리: SAVE ATC'25, BitFlipScope arXiv'26, Kelle MICRO'25, MixKVQ arXiv'25, KVTuner ICLR'25, K10 PLOS ONE 2025 ensemble, K11 UPH-Indicator IPDPS'25, MBQ CVPR'25, FastKV ACL Findings'26, OASIS IEEE 2025, TTKV arXiv'26, BanaServe arXiv'25

### 2.6 평가 기준

- **Tier-1 rubric**: Novelty orthogonal axis 7+ / Diff 5+ baseline factorial / Impact stranding-% 40%+ reduction OR accuracy 98%+ / Mechanism 1-2 OK
- **Tier-2 rubric**: Novelty first-to-report in narrow scope / Diff 2-3 baseline clear delta / Impact specific metric / Mechanism 1
- **Mechanism budget**: 아이디어당 ≤2 (R28 strict)
- **Tier budget**: physical 1-tier (single workstation) + software 1-tier

### 2.7 사용된 전문가 + 리뷰어

- **Phase 1 expert (3)**: `system-robustness-expert`, `algorithm-expert`, `legacy-system-expert`
- **Phase 2 reviewer (3)**: `novelty-reviewer`, `differentiation-reviewer`, `impact-reviewer`
- **Phase 1' / 1''**: integration agent (improve-first 정제 + final 6 selection)

---

## 3. Tier-1 Top 3 — 간략 표

| Rank | Idea | Score | Novelty driver | Simulator path | 링크 |
|------|------|------:|---------------|----------------|------|
| 🥇 | **OAEP-KV** (merged Sentinel) | 7.40 | outlier mask = ECC strength selector + DRAM-row-stripe alignment | gem5+DRAMSim3 + CHAOSMem | [tier1/01-oaep-kv.md](/research-wiki/2026-04/kv-ecc-ras/tier1/01-oaep-kv.md) |
| 🥈 | **BlockShard** (merged Lattice) | 8.00 | vLLM 16-token block = retirement granularity + RFC #19329 | gem5+DRAMSim3 + LLMServingSim + Linux MCE wrapper | [tier1/02-blockshard.md](/research-wiki/2026-04/kv-ecc-ras/tier1/02-blockshard.md) |
| 🥉 | **LayerTier** | 6.77 | layer access asymmetry × reliability zone migration | ChampSim + LLMServingSim | [tier1/03-layertier.md](/research-wiki/2026-04/kv-ecc-ras/tier1/03-layertier.md) |

---

## 4. Tier-2 독립 Top 3 — 간략 표

| Rank | Idea | Score | Novelty driver | Simulator path | 링크 |
|------|------|------:|---------------|----------------|------|
| T1 | **VLM-MAP** (merged ModalSplit + ECCLite) | 6.33 | modality-conditioned ECC strength | AttAcc + NeuroSim V1.4 | [tier2/01-vlm-map.md](/research-wiki/2026-04/kv-ecc-ras/tier2/01-vlm-map.md) |
| T2 | **B1 FrostFloor** | 5.00 | edge LPDDR5 sub-page bit-error map allocator | Jetson Orin sim + LPDDR5 stress | [tier2/02-frostfloor.md](/research-wiki/2026-04/kv-ecc-ras/tier2/02-frostfloor.md) |
| T3 | **EntropyECC** (merged Driftwood) | 5.83 | block-entropy → ECC strength predictor | ChampSim + lm-eval | [tier2/03-entropy-ecc.md](/research-wiki/2026-04/kv-ecc-ras/tier2/03-entropy-ecc.md) |

---

## 5. 미선정 15편 — 1줄 사유 표

상세는 [unselected.md](/research-wiki/2026-04/kv-ecc-ras/unselected.md):

| # | Idea | 출처 | 미선정 사유 |
|---|------|------|-------------|
| 1 | Sentinel | system-robustness | OAEP-KV 와 merge (outlier-channel ECC 동축 — DRAM-stripe alignment 만 OAEP merged 측에 흡수) |
| 2 | Lattice | system-robustness | BlockShard 와 merge (vLLM 16-token block 동축) |
| 3 | Hourglass | system-robustness | DROP — chunked-prefill 보편화로 boundary 좁음, BlockShard 의 lifecycle ECC 로 흡수 가능 |
| 4 | Chameleon | system-robustness | ScaleShield 와 axis 동일 → ScaleShield M2 와 함께 미선정 (security framing 가능하나 GenBFA-only scope 한정적) |
| 5 | S1 Driftwood | system-robustness | EntropyECC 의 characterization section 으로 흡수 |
| 6 | S2 Honeycomb-Lite | system-robustness | OAEP-KV 의 per-head ablation 으로 흡수 |
| 7 | S3 Hourglass-Mini | system-robustness | Hourglass DROP 과 함께 — VLM-MAP 의 RowPress note 로 축소 |
| 8 | ScaleShield | algorithm | Chameleon 과 self-scoop, 별도 venue 분리 시 self-collision; OAEP-KV 의 scale-byte protection section 으로 일부 흡수 |
| 9 | RotECC | algorithm | pipeline-conditional (QuaRot 만), OAEP-KV 의 limitation note 로 흡수 |
| 10 | DeepStackECC | algorithm | architecture-fragile (Qwen3-VL 한정, FP16 round-off 민감); 재방문 조건: Qwen3-VL DeepStack 보편화 시 |
| 11 | KV-Wear-Sketch | algorithm | LayerTier 의 calibration feature 로 흡수 가능; 단독 novelty 부족 |
| 12 | ErrorOracle | legacy-system | DROP — 16주 budget 초과 (52-feature LightGBM 학습 시간); Tier-2 ML predictor sub-paper 가능 |
| 13 | ModalSplit | legacy-system | VLM-MAP 와 merge (Tier-2 측에서 통합) |
| 14 | B2 MigGate | legacy-system | LayerTier 의 calibration-stability lemma 로 흡수 |
| 15 | B3 ECCLite | legacy-system | VLM-MAP 와 merge (triple-duplication) |

---

## 6. R45 / R46 적용 결과

### 6.1 R45 strict (simulator-only path)

| Idea | Implementation surface | Simulator | R45 risk | 보강 |
|------|------------------------|-----------|----------|------|
| Tier-1 OAEP-KV | vLLM KVCacheManager + Triton kernel (~600 LoC) | gem5+DRAMSim3 + CHAOSMem | 낮음 | DRAM-row mapping vendor ablation |
| Tier-1 BlockShard | vLLM block manager hook + Linux soft-offline ABI 제안 | gem5+DRAMSim3 + LLMServingSim | **중간** | 150 LOC kernel patch 를 ABI 제안 (position paper) 로 분리 |
| Tier-1 LayerTier | vLLM/SGLang layer-tag hook + move_pages syscall | ChampSim + LLMServingSim | **중간** | reliable_zone 을 sim 내 emulation 으로 한정 |
| Tier-2 VLM-MAP | SGLang KV manager modality flag (~300 LoC) | AttAcc + NeuroSim V1.4 | 낮음 | chip-kill region 가정을 sim emulation 으로 한정 |
| Tier-2 FrostFloor | vLLM-edge bitmap allocator (Jetson) | Jetson Orin + LPDDR5 stress sim | 낮음 | — |
| Tier-2 EntropyECC | KIVI quantization hook (~200 LoC) + Triton | ChampSim + lm-eval-harness | 낮음 | — |

모든 simulator R45.9 active maintenance 확인:
- gem5 (active, 2024-2026 commits) + DRAMSim3 (umd-memsys, 2024)
- ChampSim (Texas A&M, active)
- AttAcc (scale-snu/attacc_simulator, ASPLOS 2024)
- NeuPIMs (casys-kaist/NeuPIMs, ASPLOS 2024)
- NeuroSim V1.4 (neurosim/DNN_NeuroSim_V1.4, TCAS-I 2024)
- LLMServingSim (casys-kaist/LLMServingSim, MIT, ISPASS 2026 v1.0 Feb 2026)

### 6.2 R46 verified ref count

| 파일 | R46 verified ref 수 | Kelle 인용 |
|------|--------------------:|-----------|
| README | 22 | ✓ |
| Tier-1 OAEP-KV | 18 | ✓ (Kelle 1:1 차별화) |
| Tier-1 BlockShard | 16 | ✓ (Kelle retirement vs refresh axis) |
| Tier-1 LayerTier | 14 | (KVTuner ICLR'25 + TTKV 2026 차별화) |
| Tier-2 VLM-MAP | 11 | ✓ (Kelle token-importance vs modality) |
| Tier-2 FrostFloor | 9 | — (edge LPDDR scope, Kelle eDRAM 직접 비교 불필요) |
| Tier-2 EntropyECC | 13 | ✓ (Kelle importance score R² 비교 의무) |

### 6.3 Kelle MICRO 2025 baseline 보강 적용 (6 idea)

`R46 verified: title="Kelle: Co-design KV Caching and eDRAM for Efficient LLM Serving in Edge Computing", venue=MICRO 2025 (Seoul, Oct 18-22), arXiv:2510.16040`

| Idea | Kelle 와의 axis 분리 |
|------|---------------------|
| **OAEP-KV** | Kelle = bit-position MSB/LSB × token importance refresh interval; OAEP-KV = outlier-channel × ECC strength. 두 axis 직교 + outlier mask vs token importance R² 정량 비교 의무. |
| **BlockShard** | Kelle = no retirement, 2DRP refresh adaptive; BlockShard = vLLM block-level retirement. 동일 BER 에서 latency/energy/accuracy head-to-head plot 의무. |
| **EntropyECC** | Kelle's importance score 와 entropy 의 R² 비교가 결과 자체를 바꿈. Pearson r < 0.5 + ECC vote 50%+ 다르면 orthogonal 인정. |
| **VLM-MAP** | Kelle 의 token importance 가 vision/text modality 와 어떻게 다른지 ablation 필수. modality 가 importance score 의 specialization 인지 별도 signal 인지. |
| **(미선정) Chameleon** | Kelle 의 bit-position 보호 (MSB > LSB) 와 ScaleShield 의 scale-vs-mantissa hierarchy 의 redundancy / orthogonality. |
| **(미선정) Sentinel** | DRAM-row-stripe alignment 가 Kelle 의 bit-position axis 와 직교하는 *spatial* axis 임을 명시. |

---

## 7. 다음 단계 (학생 read order)

1. **본 README** (현재 파일) — Executive Summary + decision tree + Tier 분기 가이드 흡수 (15-20분).
2. **선정 시 가장 유망**: [tier1/02-blockshard.md](/research-wiki/2026-04/kv-ecc-ras/tier1/02-blockshard.md) — measurable industry impact 가장 강함, ASPLOS/OSDI fit.
3. **알고리즘 backbone 선호**: [tier1/01-oaep-kv.md](/research-wiki/2026-04/kv-ecc-ras/tier1/01-oaep-kv.md) — outlier+ECC clean novelty.
4. **long-context decode 환경**: [tier1/03-layertier.md](/research-wiki/2026-04/kv-ecc-ras/tier1/03-layertier.md).
5. **Tier-2 fallback 시**: [tier2/](/research-wiki/2026-04/kv-ecc-ras/) 3편.
6. **미선정 사유 / 재방문**: [unselected.md](/research-wiki/2026-04/kv-ecc-ras/unselected.md).

학생이 16주 안에 publication-ready 결과를 만들기 위해서는: (a) Wk 1-3 Step 1+2 simulator setup 완료, (b) Wk 4-9 Step 3-4 mech 구현, (c) Wk 10-13 Step 5-7 30 runs + Kelle 비교 + 30 runs ablation, (d) Wk 14-16 manuscript polish + artifact prep.

---

## 8. 참고 / 관련 자료

- **Phase 1 staging (3 expert)**:
  - system-robustness-expert
  - algorithm-expert
  - legacy-system-expert
- **Phase 2 staging (3 reviewer)**:
  - novelty-reviewer
  - differentiation-reviewer
  - impact-reviewer
- **관련 이전 세션**:
  - [2026-04-23 Edge VLM Energy](/research-wiki/2026-04/energy-efficient-edge-vlm)
  - [2026-04-24 MoE Fingerprint Security+Serving](/research-wiki/2026-04/moe-fingerprint-security-serving)
- **wiki entry**: `__research_wiki/ideas.md`, `papers.md`, `concepts.md`, `index.md`, `README.md`
