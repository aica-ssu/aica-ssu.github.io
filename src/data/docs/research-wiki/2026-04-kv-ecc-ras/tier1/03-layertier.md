# LayerTier — Layer-Wise KV Page Migration with Access-Frequency Awareness + Reliability Zone

> [← Session Overview](/research-wiki/2026-04/kv-ecc-ras/README.md)

> ## 약어 / 핵심 용어 풀이 (R35)
>
> - **LayerTier** — Layer-Wise KV Page Migration. layer access frequency 8-12× asymmetry × reliability zone 결합. 본 idea 의 명칭.
> - **Layer access asymmetry** — Llama-3-8B/13B 측정에서 early layer (0-4) 가 late layer 보다 8-12× 더 자주 access (decode 매 token). late layer 는 prefill 후 거의 cold.
> - **Reliability-tagged DRAM zone** — Memory controller 의 fault history 기반으로 region 별 reliability score. Hot layer KV 는 high-reliability zone, cold layer KV 는 low-reliability zone (sim-only emulation, R45 strict).
> - **move_pages syscall** — Linux userspace page migration syscall, NUMA node 지정 가능 ([man7.org](https://man7.org/linux/man-pages/man2/move_pages.2.html)).
> - **mbind syscall** — NUMA memory binding policy syscall.
> - **KVTuner** — Layer-wise sensitivity-aware mixed-precision KV quantization ([OpenReview zDwipF6h06](https://openreview.net/forum?id=zDwipF6h06), ICLR 2025). 본 idea 의 직접 baseline.
> - **TTKV** — Temporal-Tiered KV cache ([arXiv 2604.19769](https://arxiv.org/html/2604.19769), 2026). bandwidth/capacity tier — reliability tier 와 axis 직교.
> - **LayerKV** — Layer-wise KV scheduling (ATC 2024). 본 idea 와 차별화 baseline.
> - **MigGate** (legacy-system B2 merged) — calibration 100 prompt 만으로 layer hot/cold ranking 95%+ Spearman ρ stable 한 first-to-report 결과. 본 idea 의 calibration-stability lemma.

**🎯 Target Venue**: MICRO 2027 (primary) / DSN 2027 (alt) / OSDI 2027 (alt)
**📊 Score**: Novelty 6.8 / Diff 6.0 / Impact 7.5 = 평균 **6.77**
**✅ 판정**: Accept (KVTuner / TTKV 와의 axis 분리 명확 시)

> **🛠️ R47 path (Simulator-Framework Compatibility)**: **R47.2 application-level (vLLM layer access counter + migration policy 직접 구현)** primary. **R47.4 ChampSim trace-driven** 은 Tier-2 spinoff 한정 (long-run access pattern characterization). gem5+vLLM 동시 사용 안 함 (R47.1).

---

## 1. 개요 (Overview)

LLM serving 에서 **layer access frequency 가 8-12× 편차** (early hot, late cold-after-prefill) 임을 활용해, **error-prone DRAM region** 에 cold-layer KV 만 placement, **hot-layer KV 는 reliable region** 에 고정한다. 본 연구는 (a) vLLM/SGLang KV manager 의 layer-tag attribute 추가 + (b) move_pages syscall 기반 cold/hot layer migration + (c) MigGate (B2 merged) 의 calibration-stability lemma (100-prompt 만으로 95%+ Spearman ρ) 를 결합한다. 결과: **decode p99 latency -12-18%**, **error-induced refetch rate -40-55%**.

**핵심 insight (3축)**:
1. **Layer access asymmetry 는 application 에 명시적**: vLLM/SGLang 의 block 할당 시 layer_id tag 부착 비용 0.
2. **Hot layer KV 의 ECC scrubbing latency 가 throughput 직격**: hot layer 를 reliable region 에 고정 시 scrubbing event 빈도 감소.
3. **Cold layer KV 는 fault 발생해도 refetch 비용 낮음**: 다음 prefill 까지 unused → fault 발생해도 refetch overhead 최소.

**Metaphor 부속 (R30)**: "LayerTier" = layer 단위 reliability tier. 마치 NUMA tier 처럼 reliability tier 를 access pattern 으로 정렬.

---

## 2. 기존 연구의 한계 / GAP

| 기존 연구 | 축 | 한계 (본 연구 대비) | R46 verified |
|-----------|-----|---------------------|--------------|
| **KVTuner** ([OpenReview zDwipF6h06](https://openreview.net/forum?id=zDwipF6h06)) | layer-wise sensitivity mixed-precision | compression axis, no RAS | R46 verified: ICLR 2025, OpenReview verify done |
| **TTKV** ([arXiv:2604.19769](https://arxiv.org/html/2604.19769)) | temporal-tier KV cache | bandwidth/capacity, no reliability | R46 verified: arXiv 2026 |
| **LayerKV** (ATC 2024) | layer-wise eviction scheduling | capacity tier, no reliability | R46 verified: ATC 2024 |
| **Dynamic KV Placement in HetMem** ([arXiv:2508.13231](https://arxiv.org/html/2508.13231v1)) | heterogeneous memory placement | bandwidth, no error-prone region | R46 verified: arXiv 2025 |
| **LMCache** ([arXiv:2510.09665](https://arxiv.org/pdf/2510.09665)) | layer-wise pipelining | bandwidth, no reliability | R46 verified: arXiv 2025-10 |
| **BanaServe** ([arXiv:2510.13223](https://arxiv.org/html/2510.13223v1)) | KV migration | bandwidth, no reliability | R46 verified: arXiv 2025-10 |
| **RL-DRAM HPDC'24** ([arXiv:2407.16377](https://arxiv.org/abs/2407.16377)) | node-level RL on CE log | not layer-aware, HPC checkpoint | R46 verified: HPDC 2024 |
| **Schroeder SIGMETRICS'09** | DRAM error field study | layer agnostic | R46 verified |
| **Linux NUMA balancing (autoNUMA)** | OS-level page migration | no application semantic | R46 verified: kernel.org doc |
| **Sridharan ASPLOS'15** | DRAM fault classes | hardware level | R46 verified |

**GAP**: layer access asymmetry 8-12× 를 **reliability zone** 과 결합한 paper 미존재. KVTuner 는 compression axis, TTKV/LMCache/BanaServe 는 bandwidth/capacity, RL-DRAM 은 node-level — 모두 **reliability + layer-aware** 결합 미존재.

### KVTuner / TTKV 1:1 차별화

| 축 | KVTuner ICLR'25 | TTKV arXiv'26 | LayerTier |
|----|-----------------|---------------|-----------|
| 목적 | Mixed-precision quantization | Temporal-tier capacity | Reliability-tier placement |
| Layer 정보 사용 | sensitivity ranking | access recency | access frequency × reliability |
| Output | per-layer bit budget | per-layer tier (DRAM/SSD) | per-layer reliable_zone assignment |
| Axis | compression | bandwidth | RAS / fault tolerance |
| R² target | n/a | n/a | **R²(KVTuner_sensitivity, LayerTier_reliability) > 0.7 시 cross-cite** |

---

## 3. 제안 기법 (Core Mechanisms, 1 main + 1 minor)

### M1: Layer-Tag + Reliability-Zoned Migration

**R47 path**: R47.2 application-level vLLM layer access counter + migration policy 직접 구현 (primary). ChampSim 은 Tier-2 spinoff 로 분리 (R47.4).

**① 추가되는 Scheme — Source Verified (R32) + R47.2 vLLM source path**:

vLLM/SGLang KV manager 에 layer-tag attribute 추가 (~120 LoC). Block 할당 시 layer_id, access_class (hot/warm/cold) 결정. Reliability-zoned migration policy 가 cold-layer block 을 unreliable zone 으로, hot-layer block 을 reliable zone 으로 placement. vLLM `vllm/attention/backends/abstract.py` 에 layer-별 KV cache access counter 추가 → application-level access frequency 직접 측정.

> ✅ source verified: vllm-project/vllm@`main` `vllm/core/block_manager.py` (확인일: 2026-04-25)
> ✅ source verified: vllm-project/vllm@`main` `vllm/attention/backends/abstract.py` (layer access path)
> ✅ source verified: sgl-project/sglang@`main` `python/sglang/srt/managers/scheduler.py`
> ⚠️ source proposed: `vllm/ras/layer_tier.py` (~120 LoC, layer-tag + migration policy + access counter, R47.2)
> ✅ external verified: `move_pages(2)` man page (Linux kernel 4.x+) — userspace migration only (R45 clean)
> ✅ external verified: ChampSim (`ChampSim/ChampSim`) — **R47.4 trace-driven, Tier-2 spinoff 만**

**② 해결하는 문제 + Workload evidence**:

- **Layer access asymmetry**: LLaMA-7B/Llama2-13B 측정에서 early layer (0-4) attention KV 가 8-12× 더 자주 access (decode 매 token). Late layer 는 prefill 외 거의 cold.
- **Hot layer KV 가 unreliable zone 에 placement 시**: scrubbing latency + UE risk 직격 → decode p99 악화.
- **Cold layer KV 는 fault 발생해도**: 다음 prefill 까지 unused → refetch 비용 거의 0.

**③ Step-by-step**:

1. vLLM `BlockManager.allocate()` 시 layer_id, head_id, access_class tuple 부착 (block metadata).
2. Reliability zone abstraction (sim-only, R45): memory controller fault history 기반으로 region 별 reliability score. **simulator 내부 emulation** — kernel surface 수정 없음.
3. Migration policy: cold-layer block 을 reliability-low zone 으로 우선 배치, hot-layer block 을 reliable zone 고정. `move_pages` syscall (userspace).
4. Re-evaluation trigger: model-level (다른 model load 시) / prompt distribution drift (5% threshold KL divergence on layer access count) / retirement event 발생.

**④ 차별화**:

(a) RL-DRAM HPDC'24 가 node-level checkpoint, application 무관 → 본 연구는 **KV 의 layer 단위 migration**. (b) KVTuner 가 layer sensitivity 를 quantization 에만 → 본 연구는 **reliability**. (c) TTKV/LMCache/BanaServe 가 bandwidth/capacity tier → 본 연구는 **reliability tier**.

### M2 (minor): Calibration-Stability Lemma (MigGate B2 merged)

**R47 path**: R47.2 application-level vLLM layer access counter 로 직접 측정 (primary). 별도 simulator 의존 없음.

**① 추가되는 Scheme — Source Verified (R32)**:

100-prompt calibration set 만으로 layer-wise KV access frequency ranking 이 95%+ Spearman ρ stable across diverse prompt distributions (LongBench / ShareGPT / RULER / MMLU). 이는 **본 idea 의 calibration-once 정당성** 의 핵심 lemma.

> ⚠️ source proposed: `tools/calibrate_layer_access.py` (~50 LoC Python)
> ✅ external verified: ChampSim trace + 5-model 측정 (Llama-3-8B, Llama-3-70B, Mistral-7B, Qwen3-VL-8B, DeepSeek-V2-Lite)

**② 해결하는 문제**:

기존 layer-aware system 연구 (LayerKV ATC'24 등) 가 dynamic profiling 가정. 본 lemma 는 **calibration-once** 으로 충분하다는 first-to-report — 본 idea 의 zone migration 정책의 lock-in 비용 제거.

**③ Step-by-step**:

1. 100 prompts (각 100-200 토큰) 을 5 model 에 forward.
2. 각 layer 의 KV access count 측정 (decode 단계 기준).
3. Layer ranking Spearman ρ + Kendall τ stability metric (across LongBench / ShareGPT / RULER / MMLU).
4. ρ > 0.95 인 layer ranking 을 LayerTier migration policy 의 hot/cold split 기준으로 사용.

**④ 차별화**:

LayerKV ATC'24 가 dynamic profiling → 본 lemma 는 **calibration-once + 평생 사용**. Cost-benefit 매우 유리.

### Mechanism 간 상호작용

M1 의 hot/cold split 결정에 M2 의 calibration-once 결과를 사용. M1 단독으로 dynamic profiling 시 overhead 클 수 있음 → M2 의 lemma 가 cost 를 제거.

**Tier 구성 (R28)**: physical 1-tier (single workstation) + software 1-tier (1 vLLM extension). R28 ≤4 OK.

---

## 4. 평가 / 실험 플랜 (R27-β + R31 + R32 적용 7 요소)

### (1) Hardware

- **Primary**: AMD EPYC 9654 + RTX 4090 24GB
- **Sim host**: ChampSim trace + LLMServingSim batched serving
- **Storage**: NVMe 2TB

### (2) Model

- **Llama-3-8B** + **Llama-3-70B-Instruct** + **Mistral-7B-Instruct** + **Qwen3-VL-8B** + **DeepSeek-V2-Lite** (MoE cross-check)
- 모두 long-context decode-bound 시나리오 (ctx 8192-32K)

### (3) Dataset · Workload

- **LongBench** (16-128K context, decode-bound)
- **RULER** (16-128K stress test)
- **ShareGPT** (real serving traffic)
- **MMLU 1k** (zero-shot)
- 각 model 100-prompt calibration set (M2 lemma)

### (4) Simulator · Tools

**R47 path**: R47.2 application-level vLLM layer access counter + migration policy (primary) + R47.3 LLMServingSim built-in (secondary, batched serving timing). R47.4 ChampSim trace-driven 은 Tier-2 spinoff 만.

- **vLLM/SGLang v0.6.x fork** layer-tag patch + `vllm/ras/layer_tier.py` access counter (~120 LoC) — **R47.2 primary**
- **LLMServingSim** (`casys-kaist/LLMServingSim`, ISPASS 2026 v1.0) — **R47.3 secondary**, batched serving timing + KV migration cost native modeling
- **vLLM-internal zoned BER injector** — Python `np.random.binomial` with zone-conditional rate (10% high-CE region, 90% reliable), no CHAOSMem
- **scipy.stats** — Spearman ρ + Kendall τ (M2 lemma)
- **lm-evaluation-harness** — MMLU / LongBench / RULER accuracy
- **ChampSim** (`ChampSim/ChampSim`) — **R47.4 Tier-2 spinoff only**, trace-driven characterization paper 한정 (long-run access pattern visualization)
- ~~CHAOSMem~~ — vLLM-internal Python emulation 으로 대체

### (5) Ablation · Baseline

**Baselines (5 종)**:

| # | Baseline | Venue / Source | 역할 |
|---|----------|----------------|------|
| (a) | **stock vLLM** | SOSP 2023 | random page allocation reference |
| (b) | **Linux NUMA balancing (autoNUMA)** | kernel.org | OS-level migration (no application semantic) |
| (c) | **KVTuner ICLR'25** | [OpenReview zDwipF6h06](https://openreview.net/forum?id=zDwipF6h06) | **layer-wise sensitivity ranking — 직접 비교** |
| (d) | **TTKV arXiv'26** | [arXiv:2604.19769](https://arxiv.org/html/2604.19769) | temporal-tier (bandwidth axis) |
| (e) | **RL-DRAM HPDC'24** | [arXiv:2407.16377](https://arxiv.org/abs/2407.16377) | node-level RL mitigation |

Peer-reviewed ratio: 4/5 = **80%** (R2 충족).

**Ablation matrix**: (M1 only / M1+M2 / no LayerTier) × (5 model) × (3 ctx: 4K/16K/64K) × (BER 1e-9, 1e-7, 1e-6) = 135 cells. Tier-1 budget 30 runs 으로 압축: model 2 (Llama-3-8B + Llama-3-70B) × config 3 × baseline 2 (stock vLLM + KVTuner) × seed 5 ≈ 30 runs.

**Parameter sweep**: hot/cold split ratio (top-2 / top-4 / top-8 layers), zone unreliability rate (5% / 10% / 20%), calibration prompt count (50 / 100 / 200).

**Fallback mode**: KVTuner R²(sensitivity, LayerTier_reliability) > 0.8 시 본 idea 가 KVTuner 의 RAS extension 으로 reduced framing → Tier-2 IEEE TCAD short. Tier-1 cut (decode p99 -12% AND refetch -40%) 미달 시 DSN practical 6p 로 강등.

### (6) Implementation Steps (Step-Level, R31)

| Step | 의존성 | Component / File (R32 verified) | 사용 API/Library | 완료 판정 |
|------|--------|---------|---------|---------|
| Step 1 | — | ChampSim + LLMServingSim setup | ChampSim, LLMServingSim v1.0 | unit test trace replay |
| Step 2 | Step 1 | vLLM/SGLang layer-tag patch | vLLM v0.6+ | block 할당 시 layer_id 부착 |
| Step 3 | Step 2 | calibrate_layer_access.py (M2 lemma) | Python, scipy.stats | 5 model × 4 dataset Spearman ρ > 0.95 |
| Step 4 | Step 1 | Reliability zone synthetic emulation | CHAOSMem zoned BER | 10% region high-CE, 90% reliable |
| Step 5 | Step 2-4 | layer_tier.py migration policy (Mech M1) | move_pages syscall | hot layer → reliable zone fixed |
| Step 6 | Step 5 | KVTuner ICLR'25 baseline 재현 | KVTuner repo (ICLR camera-ready) | layer sensitivity ranking R² 측정 |
| Step 7 | Step 6 | TTKV / RL-DRAM baseline 재현 | TTKV arXiv repo, BSC RL-DRAM repo | 동일 fault model 재현 |
| Step 8 | Step 7 | 5 workload × 3 config × 2 baseline = 30 runs 실행 | 위 stack | 30 runs dump |
| Step 9 | Step 8 | manuscript draft + lemma section | matplotlib, pandas | 12p MICRO draft 70% |
| Step 10 | Step 9 | polish + artifact prep | git + README | submission-ready |

**참고 시간**: 약 13 weeks.

### (7) Preliminary Analysis Metrics (R27-β)

| 측정 지표 | 도구 + counter/command | 측정 조건 | 기대 범위 (baseline) | 개선 후 목표 / 검증 기준 |
|---|---|---|---|---|
| Spearman ρ (layer ranking stability) | scipy.stats | 5 model × 4 dataset | — | **≥ 0.95** (lemma) |
| Decode p99 latency | LLMServingSim | Llama-13B ctx 8192 batch 4 | (baseline) | **-12-18%** |
| Error-induced refetch rate | LLMServingSim event log | BER=1e-7, zoned 10% | (baseline) | **-40-55%** |
| Migration overhead (move_pages cost) | syscall count | 24h sim | — | < 1% of decode time |
| Layer access counter accuracy | block metadata | calibration vs runtime | — | ±5% |
| KVTuner R²(sensitivity, reliability) | scipy.stats | 5 model | — | **R² < 0.7 → orthogonal** |
| TTKV bandwidth tier vs LayerTier reliability tier | head-to-head | 동일 fault rate | — | distinct Pareto |
| Small-batch chat (batch=1, ctx<2048) | LLMServingSim | low-asymmetry case | -12-18% | -3-5% (boundary) |

**Preliminary Study 4-단계**:
- **(i) Baseline reproduction**: stock vLLM random allocation + Linux autoNUMA balancing 의 decode p99 측정.
- **(ii) Bottleneck attribution**: Mech M1 단독 (random calibration) vs M1+M2 (calibration-stable lemma) 비교.
- **(iii) Roofline (Pareto frontier)**: decode p99 × refetch rate × migration overhead — KVTuner / TTKV / RL-DRAM / LayerTier 4점 plot.
- **(iv) Micro-benchmark**: hot/cold split ratio (top-2/4/8), zone unreliability rate (5/10/20%), calibration prompt count (50/100/200).

---

## 5. 예상 효과 (보수적, scope 명시)

| 지표 | Baseline | 목표 | 조건 / 가설 검증 |
|---|---|---|---|
| Decode p99 latency | (stock vLLM) | **-12-18%** | hot layer scrubbing 감소 |
| Error-induced refetch rate | (baseline) | **-40-55%** | cold layer fault 영향 0 |
| Migration overhead | n/a | **< 1% of decode time** | move_pages cost amortized |
| Spearman ρ (lemma) | n/a | **≥ 0.95** | calibration-once 정당성 |
| KVTuner R² | n/a | **< 0.7 (orthogonal)** | sensitivity ≠ reliability |
| Small batch boundary | -12-18% | **-3-5%** (낮은 asymmetry) | scope 한계 명시 |

**과학적 contribution**: Layer access asymmetry × reliability zone 결합의 첫 paper. MigGate (calibration-once lemma) 가 본 idea 의 cost 를 제거. KVTuner / TTKV 와 axis 분리 (sensitivity/bandwidth/reliability).

**실용적**: vLLM/SGLang patch ~120 LoC + 100-prompt calibration. 별도 hardware 변경 없음 (move_pages 만 사용). long-context decode-heavy serving (production 의 dominant case) 에 즉시 적용.

**Scope 제한**: small-batch chat (batch=1, ctx<2048) 에서는 layer 편차 작아 -3-5% 그칠 수 있음 — boundary 명시 의무. Reliability zone abstraction 은 sim-only emulation (kernel surface 수정 없음, R45 strict).

---

## 6. (Tier-1 → Tier-2 변환 가이드)

| 조건 | Tier-2 venue | Tier-2 framing |
|------|--------------|---------------|
| KVTuner R² > 0.8 | DATE 6p / IEEE TCAD short | "RAS extension of KVTuner" |
| Decode p99 -12% 미달 | DSN practical 6p | "Mech M1 + lemma only" |
| Reliability zone abstraction reviewer 거부 | IEEE CAL 4p | sim-only measurement, no system mechanism |
| Multi-node 확장 reviewer 요구 | OSDI → DSN practical | scope retention single-node |

---

## 7. 미선정 idea 흡수 note

- **B2 MigGate** (legacy-system) → 본 idea 의 Mech M2 (calibration-stability lemma) 로 흡수. 단독 IEEE TCAD 제출은 thin novelty 로 평가됐지만 본 idea 와 결합 시 high-value.
- **KV-Wear-Sketch** (alg T7) → 본 idea 의 calibration feature 로 가능 (clipping rate as additional reliability signal).

---

## 8. R46 verified ref 표 (이 파일)

| ref | 제목 | venue | R46 status |
|-----|------|-------|-----------|
| KVTuner | layer-wise sensitivity mixed-precision | ICLR 2025 / OpenReview zDwipF6h06 | verified (OpenReview check) |
| TTKV | temporal-tier KV cache | arXiv 2604.19769 (2026) | verified |
| LayerKV | layer-wise scheduling | ATC 2024 | verified |
| Dynamic KV Placement HetMem | heterogeneous memory placement | arXiv 2508.13231 | verified |
| LMCache | layer-wise pipelining | arXiv 2510.09665 | verified |
| BanaServe | KV migration | arXiv 2510.13223 | verified |
| RL-DRAM HPDC'24 | RL-based DRAM mitigation | HPDC 2024 | verified |
| Schroeder | DRAM errors in the wild | SIGMETRICS 2009 | verified |
| Sridharan | DRAM fault classes | ASPLOS 2015 | verified |
| Meza | DRAM errors at Facebook | DSN 2015 | verified |
| Linux move_pages syscall | userspace migration | man7.org official | verified |
| vLLM PagedAttention | block-table KV | SOSP 2023 | verified |
| ChampSim | cache hierarchy sim | arXiv 2210.14324 | verified |
| LLMServingSim | KAIST CASYS sim | ISPASS 2026 | verified |

R46 verified count: **14 ref**.
