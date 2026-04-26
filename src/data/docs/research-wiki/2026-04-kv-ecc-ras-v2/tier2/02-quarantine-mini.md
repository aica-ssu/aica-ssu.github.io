# V3 Quarantine-Mini — Single-Agent CXL DPA Poison Detection-to-Recompute Latency Profile for vLLM Token-Range Recovery

> [← Session Overview](../README.md)
> **Paper pair**: V3 ↔ [P3 Quarantine](../tier1/02-quarantine.md) (P3 의 M2 만 분리)

> ## 약어 / 핵심 용어 풀이 (R35)
>
> - **Quarantine-Mini** — 본 idea 의 명칭. P3 Quarantine 의 M2 (CXL ECS mailbox query → token-range recompute) 만 분리.
> - **CXL ECS mailbox + DPA poison + Memory Event Record** — CXL 3.x 의 RAS feature (P3 와 동일).
> - **vLLM RFC #19329** — graceful KV connector error → token-range recompute pattern ([github.com/vllm-project/vllm/issues/19329](https://github.com/vllm-project/vllm/issues/19329)).
> - **vLLM-ascend RFC #5067** — token-level re-inference, mainstream Q2 2026 upstream — V3 의 production-ready supplement.
> - **vLLM RFC #27774** — expert parallelism FT.
> - **TPOT (Time Per Output Token)** — MLPerf v5.1 interactive 의 tight bound 40ms.
> - **TraCT** — CXL shared memory KV ([arXiv:2512.18194](https://arxiv.org/abs/2512.18194)).

**🎯 Target Venue**: DAC 2027 6p (primary) / IEEE TCAD short / IEEE CAL 4p
**📊 Score**: Novelty 7.0 / Differentiation 7.0 / Impact 7.0 / Feasibility 9.0 = 평균 **7.5**
**✅ 판정**: Accept (paper pair V3 ↔ P3, vLLM-ascend RFC #5067 mainstream supplement)

---

## 1. 개요 (Overview)

P3 Quarantine 의 M2 (CXL ECS mailbox query → vLLM RFC #19329 token-range recompute) 만 독립 contribution. **Single-agent** 환경에서 CXL DPA poison event detect 후 vLLM RFC #19329 의 affected token-range recompute path 의 **latency profile** 측정. vLLM-ascend RFC #5067 (token-level re-inference, mainstream upstream Q2 2026) 의 production-ready measurement supplement.

**Metaphor noun ↔ mechanism**: "Quarantine-Mini" = single-agent 격리. **(M1 single-agent ECS poison detect-to-recompute path = single mechanism profile)**.

**사용 simulator**: vLLM 0.7+ source mod (R47.2 only). gem5 미사용 (R47.1).

---

## 2. Workload Evidence (R23)

| # | Source | Year | 측정 숫자 | 발생 조건 |
|---|--------|------|-----------|-----------|
| F2 | KVFlow ([arXiv:2507.07400](https://arxiv.org/abs/2507.07400)) | 2025-07 | single-agent workflow trace | agentic single-turn |
| α7 | MLPerf v5.1 ([mlcommons.org](https://mlcommons.org/2025/09/mlperf-inference-v5-1-results/)) | 2025-09 | TPOT 40ms × 5 budget = 200ms | interactive |
| vLLM-ascend RFC #5067 | mainstream upstream Q2 2026 | 2026 | token-level re-inference path | vLLM mainstream |

---

## 3. Modern Memory Standard 활용 (R50.2)

**핵심 mechanism integration 3 feature** — CXL 3.x 의 RAS feature 가 mechanism 의 핵심:

- **CXL 3.1 ECS mailbox**: M1 의 poison event source.
- **CXL 3.x DPA poison**: M1 의 corrupted DPA range 식별의 핵심.
- **CXL Memory Event Record**: AER 한계 보완, M1 의 corrupted address resolution.

LPDDR5x / HBM3 미사용.

---

## 4. Mechanism 동작 원리 (1 mechanism, scope-shrink)

### M1: ECS Mailbox Query → Poison Event → Token-Range Recompute Latency Profile

**R47 path**: R47.2 application-level vLLM source mod only.

**① 추가되는 Scheme — Source Verified (R32)**:

ECS mailbox query → poison event → affected DPA range → vLLM RFC #19329 recompute path. **Detection-to-recompute latency profile** 만 측정 (P3 의 multi-agent table 와 cluster sim 제외).

> ✅ source verified: vllm-project/vllm RFC #19329 (graceful KV connector error)
> ✅ source verified: vllm-ascend RFC #5067 (token-level re-inference) — Q2 2026 mainstream upstream
> ✅ source verified: CXL 3.1 §8.2.9.9.11.2 ECS + Memory Event Record
> ⚠️ source proposed: `vllm/ras/single_agent_poison_handler.py` (~80 LoC Python, single-agent path, R47.2)
> 🔧 R32: vLLM-ascend RFC #5067 의 token-level re-inference path 가 mainstream upstream — vLLM RFC #19329 과 일치 mechanism

**② 해결하려는 문제 + Workload 1:1 대응**:

- F2 KVFlow single-agent: single-agent workflow trace 의 RFC #19329 path latency 부재.
- α7 MLPerf v5.1: TPOT 40ms × 5 budget = 200ms — 8K token + 32 layer recompute 가 200ms 내 가능한지 측정.
- vLLM-ascend RFC #5067: mainstream production-ready measurement supplement.

**③ Step-by-step**:

1. CXL ECS mailbox sysfs poll (`/sys/bus/edac/devices/<cxl_dev>/scrub0/persistent`) — 5s interval.
2. poison event 감지 시 즉시 (on-demand) Memory Event Record query — corrupted DPA address 획득.
3. vLLM `BlockManager` 의 DPA → KV block hash 역참조 — affected token range 식별.
4. vLLM scheduler 가 RFC #19329 / RFC #5067 pattern 으로 affected token-range 만 recompute (전체 sequence 가 아님).
5. **(detect_timestamp - recompute_done_timestamp)** latency 를 token range 길이 × layer 수에 대해 profile.

**④ 차별화**: P3 가 multi-agent isolation framework — V3 는 **single-agent recompute latency profile** 만. vLLM-ascend RFC #5067 mainstream supplement 로 production-ready measurement.

---

## 5. 실험 플랜 (R27-β 7 element)

### (1) Hardware

H100 1 GPU + CXL Type-3 pool (sim emulated).

### (2) Model

| Role | Model | Precision | Checkpoint |
|------|-------|-----------|------------|
| Primary | Llama-3.1-8B | FP16 | `meta-llama/Llama-3.1-8B-Instruct` |

### (3) Dataset / Workload

- KVFlow single-agent trace (single).
- ShareGPT subset (decode-heavy).
- **Metric**: detect-to-recompute latency, throughput drop, recompute overhead.

### (4) Simulator / Tools (R49 cross-check)

- **vLLM 0.7+** ([github.com/vllm-project/vllm](https://github.com/vllm-project/vllm)) — R47.2 only.
- gem5 미사용. ChaosMem 미사용. LLMServingSim 미사용 (Tier-2 scope-shrink).

### (5) Ablation / Protocol

- **Factorial**: token range (1K/4K/8K/16K) × layer count (8/16/32) × poison injection rate (1/min, 1/hr).
- **Sweep**: ECS poll interval (1s/5s).
- **Baseline**: vanilla vLLM + RFC #19329 path / TraCT (CXL shared KV) / vLLM-ascend RFC #5067.
- **Runtime**: 6-8주 + 24 runs.

### (6) Implementation Steps

| Wk | Step | 파일 경로 | 도구 | 완료 판정 |
|----|------|----------|------|-----------|
| 1-2 | vLLM 0.7+ + ECS sysfs sim setup | `vllm/ras/` | vLLM build | ECS mailbox sim emulated 가능 |
| 2-4 | M1 single_agent_poison_handler.py | `vllm/ras/single_agent_poison_handler.py` | Python + pytest | poison event detect 검증 |
| 4-6 | RFC #19329 recompute latency profile | vLLM scheduler | Python | 8K token × 32 layer profile |
| 6-8 | Ablation + RFC #5067 비교 | M1 통합 | vLLM | factorial 24 runs |
| 8-10 | Manuscript polish | 전체 | git, paper draft | DAC 2027 6p submission |

### (7) Preliminary Analysis Metrics

| 지표 | 측정 도구 | Target |
|------|----------|--------|
| Detect-to-recompute latency | vLLM scheduler timestamp | 8K token × 32 layer < 200ms |
| Throughput drop | vLLM benchmark | <3% |
| Recompute overhead (% TPOT) | vLLM scheduler stat | <5× TPOT |
| ECS poll overhead | Python thread profiler | <0.05% |

**Preliminary Study (4 단계)**: (1) baseline RFC #19329 path 재현, (2) ECS poll overhead 측정, (3) token range × layer linear scaling 검증, (4) RFC #5067 mainstream supplement 비교.

---

## 6. 관련 연구 / 차별점 / Risk / Baseline 표

| # | Paper | Year | Axis | V3 차별화 | what / why / how |
|---|-------|------|------|-----------|---------------------|
| C1 | vLLM RFC #19329 (internal) | 2025 | Graceful connector error | in-memory CE/UE 미고려 | what: connector failure handling; why: graceful; how: token-range reschedule |
| C2 | vLLM-ascend RFC #5067 (mainstream Q2 2026) | 2026 | token-level re-inference | latency profile 부재 | what: token-level re-inference; why: mainstream upstream; how: scheduler |
| C3 | TraCT ([arXiv:2512.18194](https://arxiv.org/abs/2512.18194)) | 2025-12 | CXL shared KV (latency) | poison/DPA tracking 미언급 | what: CXL shared memory; why: TTFT 9.8×; how: cacheline flush |
| C4 | P3 Quarantine (sister paper) | 2026-04-26 | Multi-agent CXL DPA isolation | multi-agent table + cluster sim 포함 | V3 = P3 의 M2 분리 |

**Risk + 완화**:
- (a) Single-agent scope 가 incremental → vLLM RFC #5067 mainstream supplement 강조.
- (b) CXL device mailbox vendor-specific → sim emulated 명시.

---

## 7. R49 Cross-check 자동 점검

| 항목 | 내용 | 일관 |
|------|------|------|
| (a) 개요 simulator | vLLM | ✅ |
| (b) M1 R47 path | R47.2 only | ✅ |
| (c) (4) Simulator/Tools | vLLM 0.7+ | ✅ |
| (d) (6) Implementation Steps | vLLM source mod only | ✅ |

**4/4 항목 일관 ✅** — ChaosMem / gem5 / LLMServingSim 미사용 (scope-shrink 의도).

---

## 8. R45 Self-check

| R45 항목 | 내용 | 통과 |
|----------|------|------|
| R45.1 | sysfs read-only polling | ✅ |
| R45.7 | vLLM source mod path | ✅ |
| R45.9 | active vLLM 0.7+ only | ✅ |
| R47.1 | gem5 미사용 | ✅ |

---

## 9. Reviewer Scoring 표

| Phase | Novelty | Differentiation | Impact | Feasibility | 평균 | 핵심 critique |
|-------|--------:|----------------:|-------:|------------:|-----:|----------------|
| Phase 1' (신규 variant) | 7 | 7 | 7 | 9 | **7.5** | scope-shrink design |
| Phase 1'' | 7 | 7 | 7 | 9 | **7.5** | polish — vLLM-ascend RFC #5067 supplement narrative |

**3 expert value Y 만장일치** — paper pair (V3 ↔ P3) 1 쌍.

---

## 10. OpenReview Check

DAC 2027 6p fit. vLLM-ascend RFC #5067 의 production-ready measurement supplement 로 reposition — first profile work.
