# Thermal-Envelope-Driven UMA Zone Migration of Visual KV on Jetson AGX/Thor (ShelfSwap) — Tier-2 Single Mechanism

> [← Session Overview](/research-wiki/2026-04/vlm-context-edge-jetson) · **Tier-2 독립 #2**

> ## 약어 풀이 (R35, 핵심만)
>
> - **UMA** (Unified Memory Architecture) — Jetson 의 GPU+CPU 가 동일 LPDDR5X physical pool 공유.
> - **UMA zone** — single physical LPDDR pool 의 logical 구분: GPU-affine zone (LPDDR 의 GPU 측 controller) / CPU-affine zone (CPU 측). cuMemAdvise 의 SetPreferredLocation 으로 hint.
> - **Skin temperature** — Jetson SoC 외부 표면 온도. `tegrastats` 의 thermal_zone 측정.
> - **Junction temperature** — SoC 내부 die 온도. throttle 진입 trigger.
> - **cuMemAdvise(SetPreferredLocation)** — CUDA Unified Memory hint API. ([NVIDIA docs](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html))
> - **DVFS** — Dynamic Voltage and Frequency Scaling.
> - **Visual KV cold layer** — VLM decoder 의 layer 0-2 visual KV (재사용 빈도 낮은 prefix).
> - **Thermal Envelope** — sustained workload 의 thermal budget. burst vs sustained 구분.
> - **SparseDVFS** — sparsity-aware DVFS, baseline (adjacent).
> - **Four Over Six** — 4-6 mode adaptive frequency, baseline (adjacent).
> - **A Thermal-Aware Workload Scheduler** [ACM SIGEnergy 2025] — thermal scheduling adjacent.
> - **ThermalLoom** (legacy-sys, 미선정) — Thermal-Throttle BW degradation + NVPModel hide. 본 idea 와 thermal axis 중복 — 분리 명시.
> - **STELE** (ai-opt, 미선정) — Thermal-budget-aware speculative decoding. 본 idea 와 axis 다름 (specdec vs zone migration).

**Target Venue**: ISLPED 2026 6p (primary) / DATE 2026 6p (alternative)
**Score** (Tier-2 rubric): Novelty **5.5** / Diff **6.0** / Impact **6.7** = 평균 **6.07**
**판정**: Accept Tier-2 (single-mechanism, novelty 5.5 — adjacent baseline 명시 의무)
**Phase 1' diff**: ΔM = 0 — SparseDVFS, Four Over Six adjacent 명시 baseline 강화.

---

## 1. 개요

본 연구는 **Jetson AGX Orin 64GB / Thor 128GB** (single HW class) 에서 sustained robot perception workload (continuous 30 FPS, 10 min) 시 GPU 발열이 throttle entry 를 trigger 하는 현상을 정량화하고, **single mechanism** — skin temperature > 78°C 시 visual KV cold layer (layer 0-2) 를 cuMemAdvise(SetPreferredLocation=CPU) 으로 CPU-affine UMA zone 으로 migration → GPU 측 LPDDR access 감소 → GPU 발열 5-9°C ↓ → throttle entry 30-50% 지연 — 을 제안한다.

**Metaphor 부속 (R30)**: "ShelfSwap" = 선반 교체. 후보: HeatRelay / ZoneTilt.

---

## 2. 기존 연구의 한계 / GAP

| 기존 | 본 ShelfSwap 와 차별 |
|------|---------------------|
| Linux thermal governor | clock 만, zone migration 부재 |
| nvpmodel | static mode preset, runtime migration 부재 |
| SparseDVFS [adjacent] | sparsity-aware DVFS, zone 부재 |
| Four Over Six [adjacent] | 4-6 mode adaptive freq, zone 부재 |
| ThermalLoom (legacy-sys 미선정) | BW kernel re-tile, zone migration 미공략 |
| STELE (ai-opt 미선정) | specdec 활성/비활성, zone 미공략 |
| **A Thermal-Aware Workload Scheduler** [ACM SIGEnergy 2025] | task scheduling, KV migration 부재 |

**GAP**: thermal-driven UMA zone migration of visual KV — first-to-report on Jetson edge.

### Trivialty 회피

- cuMemAdvise(SetPreferredLocation) 자체는 known API. 본 idea 의 novelty 는 **thermal trigger + visual KV cold layer 선별 + 30-50% throttle entry 지연 정량화**.
- ThermalLoom 과 axis 분리: ThermalLoom = sustained workload BW kernel re-tile / ShelfSwap = zone migration. STELE 와 axis 분리: STELE = burst workload specdec / ShelfSwap = zone migration.

---

## 3. 제안 기법 (Single mechanism)

### M1: Thermal-Triggered Visual KV Zone Migration

**① Scheme — Source Verified (R32)**:

UMA 내부 zone 을 **GPU-affine zone** (LPDDR 의 GPU controller 측) 와 **CPU-affine zone** (CPU 측) 으로 logical 구분. tegrastats skin temp polling — > 78°C 시 visual KV cold layer (decoder layer 0-2) 를 cuMemAdvise(SetPreferredLocation=CPU) 으로 migration. GPU 발열 감소.

> ✅ source verified: cuMemAdvise(SetPreferredLocation) ([NVIDIA docs](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html))
> ✅ source verified: tegrastats skin temperature ([NVIDIA Jetson docs](https://docs.nvidia.com/jetson/archives/r35.4.1/DeveloperGuide/text/AT/JetsonLinuxDevelopmentTools/AppendixTegraStats.html))
> ✅ source verified: vLLM `KVCacheManager.allocate` (zone hint 추가 가능)

**② 문제 + evidence**:

[Jetson NVIDIA Tech Blog](https://developer.nvidia.com/blog/nvidia-jetson-orin-the-superchip-for-edge-ai/): MAXN_SUPER 130W 와 Orin Nano 7W token/s 차이 4.1×. Sustained 10min workload 에서 throttle entry 가 평균 throughput 의 main contributor. Visual KV cold layer (재사용 빈도 낮음) 는 GPU residence 가 필수 아님.

**③ Step-by-step**:

1. tegrastats 1s cadence 로 skin temp polling.
2. skin temp > 78°C 시 visual KV cold layer (layer 0-2) block list 추출.
3. 해당 block 의 backing memory 에 cuMemAdvise(SetPreferredLocation=CPU).
4. Decode time GPU access 시 PCIe-style penalty 없음 (UMA), but GPU controller 측 LPDDR 발열 감소.
5. skin temp < 73°C 시 hysteresis — 다시 GPU-affine 으로 (cuMemAdvise(SetPreferredLocation=GPU)).

**④ 차별화**:

- **vs nvpmodel**: static mode, runtime migration 부재.
- **vs Linux thermal governor**: clock 만.
- **vs vLLM CPU offload**: 전체 KV 이동, layer-cold 선별 부재.

---

## 4. 평가 (R27-β, scope 축소)

### (1) Hardware

- **Single HW**: Jetson AGX Orin 64GB or Thor 128GB.
- 비교: Orin NX 25W (zone 효과 약함).

### (2) Model

- **Qwen2.5-VL-7B** AWQ 4-bit
- **VLM agent loop** (continuous 30 FPS robot perception)

### (3) Dataset / Workload

- Robot perception loop (continuous 30 FPS, 10 min sustained)
- Multi-turn LLaVA agent (1 hr session)

### (4) Tools

- **tegrastats** (skin/junction temp, throttle entry detection)
- **vLLM v1 fork** (zone hint patch ~50 LOC)
- **NvArgus** (camera workload generator)

### (5) Baseline

| # | Baseline | Venue / Source | 역할 |
|---|----------|----------------|------|
| (a) | **Vanilla GPU-pinned KV** | vLLM stock | upper bound thermal |
| (b) | **nvpmodel static mode** | NVIDIA standard | mode preset |
| (c) | **SparseDVFS** (Phase 1' 추가) | adjacent baseline | sparsity DVFS |
| (d) | **Four Over Six** (Phase 1' 추가) | adjacent baseline | 4-6 mode |
| (e) | **A Thermal-Aware Workload Scheduler** (Phase 1' 추가) | ACM SIGEnergy 2025 | thermal scheduling |
| (f) | **throttLL'eM** (Phase 1' 추가) | [arXiv:2408.05235](https://arxiv.org/abs/2408.05235) | predictive throttling |

Peer-reviewed ratio: 1-2/6 (Phase 1' 보강 후) — Tier-2 권장 25% 충족.

### (6) Implementation Steps

| Step | 의존성 | Component | 완료 판정 |
|------|--------|---------|---------|
| Step 1 | — | Jetson AGX 64GB sustained workload 측정 | throttle entry 시간 baseline |
| Step 2 | Step 1 | tegrastats polling integration + 78°C threshold logic | thermal trigger 검증 |
| Step 3 | Step 2 | vLLM `KVCacheManager` zone hint patch | layer 0-2 zone migration 동작 |
| Step 4 | Step 3 | hysteresis 73°C 회복 로직 | oscillation 0 |
| Step 5 | Step 4 | 6 baseline 재현 + 10min sustained 평가 | manuscript draft |

**참고 소요**: 약 6 weeks.

### (7) Preliminary Analysis

| 지표 | Baseline | 목표 | 조건 |
|---|---|---|---|
| GPU thermal | base | **-5~-9°C** | sustained 10min |
| GPU thermal throttle entry | base | **+30~+50% delay** | continuous 30 FPS |
| Average throughput (10min sustained) | base | **+10~+18%** | robot perception |
| Skin temp oscillation | — | hysteresis 안정 | no flip-flop |
| Accuracy drop | 0pp | 0pp | KV layer-cold migration |

---

## 5. 예상 효과

- GPU thermal 5-9°C ↓, throttle entry 30-50% 지연, average throughput +10-18% over 10min sustained.
- **single-mechanism + Orin AGX 64GB / Thor scope**.

**Scope 제한**:
- Orin NX 25W 환경은 thermal headroom 충분 — 효과 약함.
- Sustained < 60s burst 환경 효과 0% (thermal trigger 미발생).

---

## 6. 미선정 baseline 보강 (Phase 1' 정제 사항)

Phase 1' improve-only:

- **SparseDVFS** (adjacent) — sparsity-aware DVFS. 본 idea 와 axis 다름 (DVFS vs zone migration) — 직접 비교 명시 의무.
- **Four Over Six** (adjacent) — 4-6 mode adaptive freq. zone 부재 — 명시 baseline.
- **throttLL'eM** [arXiv:2408.05235](https://arxiv.org/abs/2408.05235) — predictive throttling. zone migration 부재 — 명시 baseline.
- **ThermalLoom 과 axis 분리** 강화 — 본 idea single mechanism scope 만 유지.

---

## 7. Source Verification (R32)

| Component | Path / Function | 상태 |
|---|---|---|
| cuMemAdvise(SetPreferredLocation) | [NVIDIA docs](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html) | ✅ |
| tegrastats skin temp | [Jetson docs](https://docs.nvidia.com/jetson/archives/r35.4.1/DeveloperGuide/text/AT/JetsonLinuxDevelopmentTools/AppendixTegraStats.html) | ✅ |
| vLLM KVCacheManager | `vllm/v1/core/kv_cache_manager.py` | ✅ zone hint 추가 가능 |
| jtop Python | [jetson_stats](https://github.com/rbonghi/jetson_stats) | ✅ |
| NvArgus camera | [NVIDIA Multimedia API](https://docs.nvidia.com/jetson/l4t-multimedia/) | ✅ |

---

## 8. Reference

- **NVIDIA Maximizing Memory Efficiency**: [Tech Blog](https://developer.nvidia.com/blog/maximizing-memory-efficiency-to-run-bigger-models-on-nvidia-jetson/)
- **NVIDIA Jetson Orin Power and Performance**: [Tech Blog](https://developer.nvidia.com/blog/nvidia-jetson-orin-the-superchip-for-edge-ai/)
- **throttLL'eM**: [arXiv:2408.05235](https://arxiv.org/abs/2408.05235)
- **EdgeReasoning** [IISWC 2025]: [accepted-papers](https://iiswc.org/iiswc2025/accepted-papers.html)
- **CLONE** [USENIX ATC 2025]: [pdf](https://www.usenix.org/system/files/atc25-tian.pdf)
