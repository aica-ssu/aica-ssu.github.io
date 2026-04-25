# CacheVeil-Sim — gem5 + ChampSim Cache Partition Simulator for VLM Visual-Token KV Pinning (Tier-2 Simulator-Path Spinoff)

> [← Session Overview](/research-wiki/2026-04/vlm-context-edge-jetson) · **Tier-2 독립 #3** (CacheVeil 원안 R45.1 위반의 simulator-path 변환)

> ## 약어 / 핵심 용어 풀이 (R35)
>
> - **VLM** — Vision-Language Model.
> - **SLC** (System Level Cache) — Jetson Orin AGX 4MB / Thor 추정 16MB.
> - **ARM CMN `por_hnf_pwpr`** — ARM CoreLink CMN HNF Partition-Wise Power/Allocation Policy Register. 본 simulator-path 에서는 register write 대신 ChampSim cache replacement policy modification 으로 partition 효과 modeling.
> - **gem5** — open-source full-system / syscall-mode simulator. ARMv8-A + AArch64 지원. [gem5.org](https://www.gem5.org/)
> - **ChampSim** — cache replacement policy simulator. SPEC CPU 2017 trace 표준. [ChampSim repo](https://github.com/ChampSim/ChampSim)
> - **DRAMSim3** — DRAM model. LPDDR5X 모델링 (LPDDR5 모델 기반 + scaling).
> - **Aladdin** — accelerator simulator. DLA-like compute unit modeling 가능 (옵션).
> - **MLPerf Inference Edge** — VLM inference benchmark. Qwen3-VL-4B / InternVL3-2B trace.
> - **Visual-Token KV** — VLM prefix 중 vision encoder 출력 의 KV cache.
> - **CacheVeil 원안** — Tier-1 Top 1 ★ lead 원안. R45.1 위반 (undocumented BPMP IOCTL) 으로 demotion. [tier1/01-cacheveil](/research-wiki/2026-04/vlm-context-edge-jetson/tier1/01-cacheveil.md) 참조.
> - **R45.2 simulator-path exception** — novelty 명확 시 범용 simulator path 로 변환 허용.
> - **R45.3 simulation time feasibility** — 한 학기 12-16주 내 5+ workload × 3+ config × 2+ baseline = 30 runs 가능.

**Target Venue**: ISLPED 2026 6p (primary) / DAC 2026 6p (alternative)
**Score** (Tier-1 demotion → Tier-2 보정): Novelty **6.5** / Diff **6.0** / Impact **7.0** = 평균 **6.5** (CacheVeil 원안 8.00 에서 R45.4 적용 -1.5 = 6.5)
**판정**: Accept Tier-2 (simulator-path Tier-2 spinoff, R45.2 exception 적용)
**R45 risk**: **7/10 MED-HIGH** — simulator-only validation, real HW 검증 부재. 그러나 reproducible eval + R45.3 feasibility 확보로 단일 학기 fit.

---

## 1. 개요 (Overview)

본 연구는 [CacheVeil 원안 (Tier-1 demotion)](/research-wiki/2026-04/vlm-context-edge-jetson/tier1/01-cacheveil.md) 의 R45.1 위반 (ARM CMN `por_hnf_pwpr` partition register 의 undocumented BPMP IOCTL `tegra_bpmp_transfer(MRQ_CMN_SLC_PARTITION)` 의존) 을 **R45.2 simulator-path exception** 적용으로 reframe — 실제 register write 대신 **gem5 syscall-mode + ChampSim cache replacement policy modification** 으로 SLC way-partition 효과를 reproducible simulator path 로 정량화한다.

**가설**: Jetson Orin AGX 의 SLC 4MB 8-way 가정 환경을 gem5 (ARMv8-A AArch64) + ChampSim (LLC partition policy custom) + DRAMSim3 (LPDDR5X model) 으로 modeling, visual KV 4-way pin + CPU workload 4-way isolation 의 partition 효과를 5 workload × 3 config × 2 baseline = 30 runs 으로 측정 시 (a) LLC conflict miss rate 28-45% ↓, (b) memory subsystem stall cycle 15-22% ↓, (c) simulated TPOT p99 latency 12-18% ↓.

**Metaphor 부속**: "CacheVeil-Sim" = CacheVeil 원안의 simulator twin. 후보: SimVeil / GemPin.

---

## 2. CacheVeil 원안 vs CacheVeil-Sim — 변환 의의

| 항목 | CacheVeil 원안 (Tier-1, R45 demoted) | CacheVeil-Sim (Tier-2, R45.2 exception) |
|------|----------------------------------------|------------------------------------------|
| **Mechanism #1 (way-partition)** | ARM CMN `por_hnf_pwpr` register write via undocumented BPMP IOCTL | ChampSim LLC replacement policy 의 partition-aware variant 구현 |
| **Mechanism #2 (CPU isolation)** | sysfs `/sys/devices/system/cpu/cmn/...` BPMP IOCTL | gem5 syscall-mode 의 affinity-aware memory access trace |
| **Mechanism #3 (dynamic resize)** | Tegra BPMP firmware IOCTL runtime 조정 | ChampSim runtime parameter sweep (offline, 4-4 / 6-2 / 2-6 / dynamic) |
| **Mechanism #4 (DC CIVAC flush)** | ARMv8-A inline asm `__builtin___clear_cache` | gem5 의 cache flush event injection |
| **HW 의존** | Jetson Orin AGX 64GB physical device + JetPack 7.1 fixed-target | 범용 x86 server 4-core (gem5 시뮬레이션) |
| **R45 status** | R45.1 위반 (undocumented register manipulation) | R45.2 exception 적용 (reproducible simulator) |
| **Reproducibility** | JetPack version-specific, BPMP firmware revision-specific | Open-source simulator + open trace = fully reproducible |
| **Venue** | MICRO/HPCA 2027 (13p) → 진입 불가 | ISLPED 6p / DAC 6p (simulator-only validation venue 적합) |

**의의**: 실제 register write 가 R45.1 위반이지만, simulator-driven cache partition study 자체는 systems research 에서 ISLPED / DAC 등에서 수용되는 path. Tier-1 진입 불가하나 Tier-2 venue 에서 reproducible eval 으로 publish 가능.

---

## 3. 제안 기법 (Single Mechanism — Tier-2 scope)

### 3.1 ChampSim LLC Partition Policy Modification

**① Scheme**:

ChampSim 의 LLC replacement policy (LRU 기본) 을 partition-aware variant 로 수정 — way 8개 중 way 0-3 (4-way) 은 visual KV access 만 allocate, way 4-7 (4-way) 은 CPU workload (camera ISP DMA simulation, OS/preprocessor) 만 allocate. Replacement decision 시 partition group 위반 access 는 fallback way 만 사용.

> ✅ source verified: [ChampSim repo](https://github.com/ChampSim/ChampSim) — `replacement/lru/lru.cc` 수정 path.
> ✅ source verified: ChampSim 의 SPEC CPU 2017 trace 형식 — public dataset.

### 3.2 gem5 + DRAMSim3 Integration

**① Scheme**:

gem5 syscall-mode 에서 ARMv8-A AArch64 CPU 모델 (Cortex-A78AE 근사) + 4MB LLC + DRAMSim3 LPDDR5X model 사용. Visual-token KV access pattern 을 trace 로 추출 (Qwen3-VL-4B / InternVL3-2B / MiniCPM-V-2.6 / LLaVA-NeXT-7B / video MileBench-Long) → ChampSim LLC partition policy 와 통합.

> ✅ source verified: [gem5.org](https://www.gem5.org/) — ARMv8-A 모델 공식 지원.
> ✅ source verified: [DRAMSim3 repo](https://github.com/umd-memsys/DRAMsim3) — LPDDR5 모델 (LPDDR5X 는 frequency scaling 으로 근사).

### 3.3 Workload Trace Generation

**① Scheme**:

5 workload trace = SPEC CPU 2017 rate-1 fast input (CPU side workload, baseline) + MLPerf Inference Edge (VLM trace, 4 model). Trace 추출 path: stock vLLM v1 빌드 → `nsys` trace + custom memory access logger → ChampSim trace 변환.

> ✅ source verified: [SPEC CPU 2017](https://www.spec.org/cpu2017/) — 학교 라이센스 가용.
> ✅ source verified: [MLPerf Inference Edge](https://mlcommons.org/benchmarks/inference-edge/) — public benchmark.

### 3.4 R45 적용 — Simulator-Path Implementation 검증

본 idea 는 R45.2 simulator-path exception 으로 적용. R45.1 의 4 금지 카테고리 (kernel patch / kernel module / undocumented register / closed-source firmware) 모두 해당 없음 — 모든 변경은 ChampSim user-space C++ source modification + gem5 user-space Python config 만.

**R45.3 한 학기 feasibility**:
- gem5 syscall mode 1 workload simulation 24-72 시간.
- 5 workload × 3 config (4-4 / 6-2 / 2-6 partition) × 2 baseline (LRU vanilla / partition-aware) = 30 runs.
- Sequential: 360-1080 시간 (15-45일).
- **Parallel 4 servers** (학교 라이센스 또는 cloud GPU node): 90-270 시간 = **4-11일**.
- **단일 학기 12-16주 내 fit** ✅. 추가 2-3주 trace generation + 2-3주 paper writing.

**R45.4 reviewer scoring**:
- Risk 7/10 MED-HIGH (simulator-only validation, real HW 검증 부재).
- Tier-1 진입 불가 (real HW 측정 부재) — Tier-2 venue (ISLPED / DAC) 적합.
- Novelty 6.5 (CacheVeil 원안 8.5 에서 -2.0) — register-level 정밀도 부재, 그러나 partition study 자체는 first-to-report on edge VLM trace.
- Reproducible eval = ISLPED / DAC reviewer 가 인정하는 evaluation path.

---

## 4. 평가 / 실험 플랜 (Tier-2 7 elements R27-β)

### 4.1 Hardware (simulation host)

- Linux x86_64 server, 4-core 16GB RAM, gem5 + ChampSim + DRAMSim3 build.
- 4 server parallel 권장 (학교 cluster 또는 cloud).

### 4.2 Software Stack

- gem5 v23.0+ (ARMv8-A AArch64 syscall mode).
- ChampSim 최신 master (LLC partition policy 수정 fork).
- DRAMSim3 (LPDDR5X model — LPDDR5 모델 + frequency scaling).
- nsys + custom trace logger (Jetson Orin AGX physical device 1대 — trace 추출만, 실측 부재).

### 4.3 Workload (5 workload × 3 config × 2 baseline = 30 runs)

| Workload | trace 길이 | 설명 |
|----------|----------|------|
| Qwen3-VL-4B | 1B instructions | VLM 일반 |
| InternVL3-2B | 1B instructions | edge VLM |
| MiniCPM-V-2.6 | 1B instructions | 8B 미만 VLM |
| LLaVA-NeXT-7B | 1B instructions | CLIP-ViT-L/14 + Vicuna |
| MileBench long-context | 2B instructions | long-context VLM |

### 4.4 Metric (7 elements)

1. LLC conflict miss rate (ChampSim PMU stat).
2. LLC partition occupancy (way 0-3 vs way 4-7).
3. DRAMSim3 row buffer hit rate.
4. Simulated CPU memory subsystem stall cycle.
5. Simulated TPOT (token-level decode latency derived from LLC + DRAM stall).
6. Visual KV residency (way 0-3 hit rate over time).
7. Simulator runtime (R45.3 feasibility 추적).

### 4.5 Variant

- Partition split: {4-4, 6-2, 2-6, dynamic}.
- Pinning region: {layer 0 only, 0-3, 0-7, all visual KV}.
- Concurrent CPU workload: {none, ISP-DMA simulated, ISP+OS preprocessor}.

### 4.6 Baseline (2편 + simulator vanilla)

- LRU vanilla replacement (ChampSim default).
- LRU + way-partition aware (논문 기존: [Energy-Efficient GPU L2 Cache Design ACM TACO 2020](https://dl.acm.org/doi/fullHtml/10.1145/3408060)).
- Hardware Compute Partitioning (libsmctrl) [ECRTS 2025] — 비교 baseline (compute side, simulator path 외).
- VL-Cache [ICLR 2025 OpenReview HMrcv7Q4Ub](https://openreview.net/forum?id=HMrcv7Q4Ub) — KV compression baseline (orthogonal axis).

### 4.7 Fallback Scope

- 실 Jetson Orin AGX trace 가용 시 ChampSim trace 변환만, 실 register write 불필요 — R45.1 위반 회피.
- DRAMSim3 LPDDR5X model 부재 시 LPDDR5 model 사용 + frequency scaling 보정.

---

## 5. 예상 효과 (보수적, simulator-only scope)

| 지표 | Baseline (LRU vanilla) | 목표 (partition-aware) | 적용 조건 |
|------|----------------------|----------------------|---------|
| LLC conflict miss rate | base | -28~-45% | 5 workload sustained simulation |
| Memory subsystem stall cycle | base | -15~-22% | concurrent CPU workload (ISP-DMA simulated) |
| Simulated TPOT p99 | base | -12~-18% | sustained VLM trace |
| Visual KV residency rate | 변동 | 90%+ (way 0-3 pinned) | 4-4 partition |

**Workload 특이성**:
- 적용: **simulator-driven SLC partition study**, **edge VLM cache hierarchy analysis**.
- 미적용: **real HW measurement** (R45.2 exception scope), **real-time perception loop** (simulator timing 부정확).
- 불확실: LPDDR5X 정확 model — DRAMSim3 LPDDR5 + frequency scaling 보정만.

---

## 6. Tier-2 Venue 적합성

**ISLPED / DAC 가 simulator-only validation 을 수용하는 이유**:
- ISLPED (International Symposium on Low Power Electronics and Design) 6p — power/cache study 의 simulator-driven evaluation 일반적.
- DAC (Design Automation Conference) 6p — design space exploration + simulator-based first-to-report 수용.
- 둘 다 reproducible eval (open-source simulator + open trace) 우대.

**진입 조건**:
- LLC conflict miss rate -28% 이상 simulator 측정.
- 5 workload × 3 config × 2 baseline = 30 runs 완료 (R45.3 feasibility).
- ChampSim partition policy fork 공개 + trace 공개 (reproducibility).

**Tier-1 재진입 조건** (CacheVeil 원안과 동일):
- Thor JetPack 7.x 에서 BPMP cache partition IOCTL 공식화 시.
- 또는 NVIDIA partner channel 통한 BPMP firmware extension 공개 시.

---

## 7. Source Verification (R32 통합)

| Component | Path / Function | 상태 |
|---|---|---|
| gem5 ARMv8-A syscall mode | [gem5.org](https://www.gem5.org/) | ✅ open-source |
| ChampSim LLC replacement | [repo](https://github.com/ChampSim/ChampSim), `replacement/lru/lru.cc` | ✅ fork 변경 |
| DRAMSim3 LPDDR5 model | [repo](https://github.com/umd-memsys/DRAMsim3) | ✅ |
| MLPerf Inference Edge | [link](https://mlcommons.org/benchmarks/inference-edge/) | ✅ |
| SPEC CPU 2017 | [link](https://www.spec.org/cpu2017/) | ✅ 학교 라이센스 |
| ARM CMN `por_hnf_pwpr` 원리 | [ARM KA005251](https://developer.arm.com/documentation/ka005251/latest/) | ✅ docs only — register write 부재 (R45.1 회피) |
| Real Jetson trace 추출 | nsys + custom logger | ✅ trace 만, register write 없음 |

---

## 8. Reference 목록 (이 idea 핵심)

- **CacheVeil 원안** (Tier-1 demotion 사례): [tier1/01-cacheveil](/research-wiki/2026-04/vlm-context-edge-jetson/tier1/01-cacheveil.md)
- **gem5 simulator**: [gem5.org](https://www.gem5.org/), [paper SIMUTOOLS 2018](https://dl.acm.org/doi/10.1145/3232540.3232550)
- **ChampSim**: [repo](https://github.com/ChampSim/ChampSim), [JWAC paper](https://github.com/ChampSim/ChampSim/wiki)
- **DRAMSim3**: [repo](https://github.com/umd-memsys/DRAMsim3)
- **Hardware Compute Partitioning (libsmctrl)** [ECRTS 2025](https://drops.dagstuhl.de/storage/00lipics/lipics-vol335-ecrts2025/LIPIcs.ECRTS.2025.21/LIPIcs.ECRTS.2025.21.pdf)
- **Energy-Efficient GPU L2 Cache Design** [ACM TACO 2020](https://dl.acm.org/doi/fullHtml/10.1145/3408060)
- **VL-Cache** [ICLR 2025 OpenReview HMrcv7Q4Ub](https://openreview.net/forum?id=HMrcv7Q4Ub)
- **Mind the Memory Gap** [arXiv:2503.08311](https://arxiv.org/abs/2503.08311)
- **ARM CMN `por_hnf_pwpr`** [ARM KA005251](https://developer.arm.com/documentation/ka005251/latest/)
- **MLPerf Inference Edge**: [link](https://mlcommons.org/benchmarks/inference-edge/)

---

## 9. 차별화 (CacheVeil 원안과 같은 effect 영역에서의 simulator-driven reproducible eval)

본 idea 는 CacheVeil 원안 (R45.1 위반) 의 effect 영역 (SLC way-partition 으로 visual KV 보호) 을 동일 다루나, **register write 대신 simulator-driven evaluation** 으로 reframe. 이 path 는:

- **Tier-1 진입 불가**: real HW measurement 부재 → Tier-2 venue 만.
- **Reproducibility 우위**: ChampSim fork + trace 공개로 다른 lab 가 재현 가능 (CacheVeil 원안은 JetPack/BPMP firmware 의존).
- **R45 compliance**: R45.2 simulator-path exception 으로 학생이 R45.1 위반 idea 를 어떻게 reframe 하는지 학습 사례.
- **Future Tier-1 path**: Thor JetPack 7.x BPMP IOCTL 공식화 시 본 simulator-driven eval 결과를 motivation 으로 CacheVeil 원안 재진입 가능.

---

## 10. Source 추적

본 idea 는 [CacheVeil 원안 (Tier-1 demotion)](/research-wiki/2026-04/vlm-context-edge-jetson/tier1/01-cacheveil.md) 의 R45.2 simulator-path exception 변환. 1차 publish (2026-04-25 오전) 시점에는 CacheVeil 가 ★ Tier-1 lead (8.00) 로 진입했으나, R45 (Implementation Difficulty + Risk Discipline) 추가 도입 후 mechanism #1 의 ARM CMN `por_hnf_pwpr` undocumented BPMP IOCTL 의존이 R45.1 위반으로 demotion. simulator-path Tier-2 spinoff 으로 reposition.
