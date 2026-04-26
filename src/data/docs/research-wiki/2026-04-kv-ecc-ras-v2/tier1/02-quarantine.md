# P3 Quarantine — Per-Agent DPA-Level KV Cache Poison Isolation for Agentic Multi-Turn LLM Serving on CXL Pools

> [← Session Overview](../README.md)

> ## 약어 / 핵심 용어 풀이 (R35)
>
> - **Quarantine** — 본 idea 의 명칭. 한 agent 의 corrupted KV 가 다른 agent 로 전파되지 않도록 DPA-level isolate.
> - **DPA (Device Physical Address)** — CXL device 내부 주소 공간. HPA→DPA translation 후 device 가 자체 관리.
> - **Poison** — CXL spec 의 corrupted data 격리 mechanism. process/VM 레벨 격리.
> - **Memory Event Record** — CXL device 가 host 에 보내는 memory error report. AER 의 "memory error address 미log" 한계 보완.
> - **CXL ECS mailbox** — CXL 3.1 §8.2.9.9.11.2 Error Check and Scrub mailbox query.
> - **vLLM RFC #19329** — graceful KV connector error handling pattern, affected token-range 만 reschedule ([github.com/vllm-project/vllm/issues/19329](https://github.com/vllm-project/vllm/issues/19329)).
> - **vLLM-ascend RFC #5067** — token-level re-inference, mainstream upstream Q2 2026.
> - **Targeted BFA on Agents** — agent-level bit-flip attack ([arXiv:2603.10042](https://arxiv.org/abs/2603.10042)).
> - **CacheSolidarity** — timing side-channel defense via selective isolation ([arXiv:2603.10726](https://arxiv.org/abs/2603.10726)).
> - **SafeKV** — system-level 권한 enforcement + cross-tenant privacy ([arXiv:2508.08438](https://arxiv.org/abs/2508.08438)).
> - **Oneiros** — multi-tenant parameter remapping ([arXiv:2507.11507](https://arxiv.org/abs/2507.11507)).
> - **KVFlow** — multi-agent prefix cache ([arXiv:2507.07400](https://arxiv.org/abs/2507.07400)).
> - **LLMServingSim** — KAIST CASYS, MIT license, ISPASS 2026.

**🎯 Target Venue**: USENIX Security 2027 (primary, 13p) / OSDI 2027 (alt) / DSN 2027 (alt)
**📊 Score**: Novelty 9.0 / Differentiation 8.5 / Impact 8.5 / Feasibility 8.0 = 평균 **8.5**
**✅ 판정**: Conditional Accept (CacheSolidarity unified framework paragraph + agent_id table footprint 분석 추가)

---

## 1. 개요 (Overview)

KVFlow (F2, [arXiv:2507.07400](https://arxiv.org/abs/2507.07400)) 의 multi-agent prefix cache 공유 환경에서 한 agent 의 KV block CE 가 다른 agent 로 silent propagate. Two-Decade-Old Prophecy ([arXiv:2510.00490](https://arxiv.org/abs/2510.00490)) 의 단 1 bit flip → 73.5% → 0% (31.7s) + Targeted BFA on Agents ([arXiv:2603.10042](https://arxiv.org/abs/2603.10042)) agent-level threat → multi-tenant CXL pool 의 cross-tenant Rowhammer 가 KV 오염. **CXL 3.x DPA tracking + poison + ECS mailbox + Memory Event Record 4 feature** 를 vLLM agent isolation 과 결합 — cross-agent silent corruption 100% 차단 + multi-agent throughput drop <5%.

**Metaphor noun ↔ mechanism**: "Quarantine" = 격리. **(M1 agent_id × DPA table = 격리 방 배정 / M2 ECS mailbox query → token-range recompute = 격리 발견 시 회복 / M3 multi-agent cluster = 격리 효과 cluster scale 검증)**.

**사용 simulator**: vLLM 0.7+ source mod (R47.2 primary) + LLMServingSim 1.x multi-agent (R47.3). gem5 미사용 (R47.1).

---

## 2. Workload Evidence (R23)

| # | Source | Year | 측정 숫자 | 발생 조건 |
|---|--------|------|-----------|-----------|
| F2 | KVFlow ([arXiv:2507.07400](https://arxiv.org/abs/2507.07400)) | 2025-07 | multi-agent workflow prefix reuse 가 throughput driver | agentic multi-turn |
| α4 | KVCache-in-Wild USENIX ATC'25 ([arXiv:2506.02634](https://arxiv.org/abs/2506.02634)) | 2025-06 | multi-turn KV reuse 가 single-turn 만큼 significant | Aliyun production |
| B2 | Two-Decade-Old Prophecy ([arXiv:2510.00490](https://arxiv.org/abs/2510.00490)) | 2025-10 | **단 1 bit flip → 73.5% → 0% (31.7s @464.3 flips/s)** | .gguf quantized LLM |
| BFA-Agents | Targeted BFA on Agents ([arXiv:2603.10042](https://arxiv.org/abs/2603.10042)) | 2026-03 | agent-level BFA threat | multi-agent serving |
| C1 | Phoenix Rowhammer ([comsec.ethz.ch/phoenix](https://comsec.ethz.ch/research/dram/phoenix/)) | 2026 (S&P) | DDR5 on-die ECC 우회 109s | CXL-attached DDR5 |
| α7 | MLPerf Inference v5.1 ([mlcommons.org](https://mlcommons.org/2025/09/mlperf-inference-v5-1-results/)) | 2025-09 | TPOT 40ms / TTFT 450ms tight bound | interactive |

---

## 3. Modern Memory Standard 활용 (R50.2)

**핵심 mechanism integration 4 feature** — CXL 3.x 신규 RAS feature 가 mechanism 의 핵심:

- **CXL 3.x DPA tracking + poison**: M1 의 agent_id × DPA range 분리, M2 의 poison detect 시 affected agent 식별의 핵심.
- **CXL 3.1 ECS mailbox (§8.2.9.9.11.2)**: M2 의 5s poll + on-demand poison event source.
- **CXL Memory Event Record**: AER 의 한계 보완 — M2 의 corrupted DPA address 식별의 핵심 (AER 만으로는 불가능).
- **CXL Linux 6.16 upstream ([Phoronix 2025](https://www.phoronix.com/news/Linux-6.16-CXL))**: Memory Event Record / scrub_subsystem mainstream 진입 reference.

LPDDR5x / HBM3 미사용 — CXL pool 만 핵심 mechanism.

---

## 4. Mechanism 동작 원리 (3 mechanism)

### M1: Agent_id × DPA_Range 2-Level Sparse Hash Table

**R47 path**: R47.2 application-level vLLM source mod (primary).

**① 추가되는 Scheme — Source Verified (R32)**:

`vllm/distributed/kv_transfer/` 에 **agent_id × DPA_range 2-level sparse hash table** 추가. PagedAttention 16-token block 이 어떤 agent 에 속한지 mark.
- Top: agent_id → DPA-block-list (8 byte pointer per agent).
- Leaf: DPA → KV block hash (16 byte per block).
- Worst-case 16 agent × 1M block = 16M entries (128MB metadata).
- Sparse hash 의 average 1/8 fill rate 가정 — average 4-16MB.

> ✅ source verified: vllm-project/vllm@`main` `vllm/distributed/kv_transfer/` (확인일: 2026-04-26)
> ⚠️ source proposed: `vllm/distributed/kv_transfer/agent_dpa_table.py` (~150 LoC Python, 2-level sparse hash, R47.2)
> 🔧 R32: 2-level sparse hash 의 worst-case 128MB → average 4-16MB 가정 검증 — Phase 2 ai-optimization-expert 비판 후 sharpen

**② 해결하려는 문제 + Workload 1:1 대응**:

- F2 KVFlow: multi-agent prefix reuse → agent 별 KV block 분리 + cross-agent reuse 정책 필요.
- α4 KVCache-in-Wild: multi-turn KV reuse 가 single-turn 만큼 significant → trusted/untrusted zoning.

**③ Step-by-step**:

1. agent 첫 spawn 시 agent_id 발급 (UUID v4, 32-bit hash).
2. 해당 agent 가 첫 KV block 할당 시 BlockManager.allocate() 가 agent_id 를 hash table top level 에 등록.
3. 매 prefix block reuse 시 leaf level 에 DPA → block hash mapping 추가.
4. **Trusted zone (within-tenant)**: agent_id 끼리 prefix reuse 허용 (KVFlow 패턴).
5. **Untrusted zone (cross-tenant CXL pool)**: prefix reuse 차단 — DPA range 격리.

**④ 차별화**: KVFlow 가 multi-agent prefix cache 의 **reliability axis 부재**. CacheSolidarity ([arXiv:2603.10726](https://arxiv.org/abs/2603.10726)) 가 timing side-channel (security attack model) — P3 와 trigger source 직교 (hardware fault vs attacker). SafeKV ([arXiv:2508.08438](https://arxiv.org/abs/2508.08438)) 가 cross-tenant privacy — **hardware-induced poison axis 부재**. P3 = first hardware-induced KV poison isolation framework.

### M2: CXL ECS Mailbox Query → Token-Range Recompute (RFC #19329 pattern)

**R47 path**: R47.2 application-level vLLM source mod (primary).

**① 추가되는 Scheme — Source Verified (R32)**:

CXL ECS mailbox query (sysfs/ioctl emulated) → poison event 반환 시 그 DPA range 의 모든 block 을 quarantine + affected agent 의 token-range 만 recompute. **5s poll + on-demand poison event** 결합 (interrupt-driven). vLLM RFC #19329 pattern 활용.

> ✅ source verified: CXL 3.1 §8.2.9.9.11.2 ECS + CXL Memory Event Record ([CXL 3.1 RAS Whitepaper](https://computeexpresslink.org/wp-content/uploads/2024/08/An-Overview-of-RAS-for-Compute-Express-Link-3.1-Whitepaper.pdf))
> ✅ source verified: vllm-project/vllm RFC #19329 (graceful KV connector error)
> ✅ source verified: vllm-ascend RFC #5067 (token-level re-inference, mainstream Q2 2026 upstream)
> ⚠️ source proposed: `vllm/distributed/kv_transfer/poison_handler.py` (~200 LoC Python, ECS mailbox poll + recompute trigger, R47.2)
> 🔧 R32: CXL Type-3 device mailbox interface 가 vendor-specific — Linux 6.16 EDAC scrub_subsystem 의 mailbox protocol 만 사용 (sim emulated)

**② 해결하려는 문제 + Workload 1:1 대응**:

- B2 Prophecy: 단 1 bit flip → 31.7s collapse → sub-second corruption detect 필수 (5s poll + on-demand).
- BFA-Agents: agent-level threat → DPA-level isolate.
- α7 MLPerf v5.1: TPOT 40ms × 5 budget = 200ms — recompute overhead < TPOT budget 이어야.

**③ Step-by-step**:

1. `poison_handler.py` 가 별도 thread 로 시작.
2. 매 5초마다 sysfs `/sys/bus/edac/devices/<cxl_dev>/scrub0/persistent` polling, ECS history mailbox query.
3. ECS history 가 poison event 반환 시 **즉시 (on-demand)** 처리: corrupted DPA range 식별.
4. agent_dpa_table 역참조 → affected agent_id 와 KV block hash list 획득.
5. vLLM scheduler 가 RFC #19329 pattern 로 affected token-range 만 reschedule (전체 sequence 가 아님).
6. agent 가 untrusted zone 인 경우 cross-agent prefix reuse 도 invalidate.

**④ 차별화**: vLLM RFC #19329 가 **connector failure** 만 graceful — **in-memory CE/UE** 까지 graceful 한 work 0건. Targeted BFA on Agents ([arXiv:2603.10042](https://arxiv.org/abs/2603.10042)) attack paper — defense 측면 first.

### M3: LLMServingSim 4-16 Multi-Agent Cluster

**R47 path**: R47.3 LLMServingSim built-in.

**① 추가되는 Scheme**:

multi-agent (4/8/16 agent) workflow 에서 corruption injection × isolation effectiveness 측정. CXL pool poison rate sweep.

> ✅ source verified: LLMServingSim v1.0 ([github.com/casys-kaist/LLMServingSim](https://github.com/casys-kaist/LLMServingSim)) — built-in multi-agent workflow

**② 해결 / 대응**: KVFlow 의 multi-agent throughput driver 환경에서 P3 isolation 의 throughput-reliability trade-off 측정.

---

## 5. 실험 플랜 (R27-β 7 element)

### (1) Hardware

H100/H200 cluster (4-16 GPU) + CXL Type-3 pool 256GB (sim emulated).

### (2) Model

| Role | Model | Precision | Checkpoint |
|------|-------|-----------|------------|
| Primary | Llama-3.1-70B-Instruct | W4G128 | `meta-llama/Llama-3.1-70B-Instruct` |
| Secondary | Qwen3-30B-A3B-MoE | INT4 | `Qwen/Qwen3-30B-A3B` |
| Robustness | Llama-3.1-8B (multi-agent) | INT4 | `meta-llama/Llama-3.1-8B-Instruct` |

### (3) Dataset / Workload

- AgentBench (multi-agent benchmark).
- multi-agent workflow trace (KVFlow synthetic).
- ShareGPT multi-turn.
- **Metric**: cross-agent corruption rate, recompute overhead %, throughput, TPOT (agent-level).

### (4) Simulator / Tools (R49 cross-check)

- **vLLM 0.7+** ([github.com/vllm-project/vllm](https://github.com/vllm-project/vllm)) — R47.2 primary.
- **LLMServingSim 1.x** ([github.com/casys-kaist/LLMServingSim](https://github.com/casys-kaist/LLMServingSim)) — R47.3 multi-agent cluster.
- gem5 미사용. ChaosMem 미사용.

### (5) Ablation / Protocol

- **Factorial**: agent count (4/8/16) × isolation granularity (DPA-level / 4KB / 16-token block) × poison injection rate (1/min, 1/hr) × trusted/untrusted zoning (on/off).
- **Sweep**: ECS poll interval (1s/5s/30s) × on-demand event (on/off).
- **Baseline**: vanilla vLLM + KVFlow / TraCT (CXL latency baseline) / CacheSolidarity (timing side-channel).
- **Runtime**: 5-9주 + 30 runs.
- **Fallback**: 2-level sparse hash 의 average footprint > 16MB 시 1-level flat hash 로 simplify.

### (6) Implementation Steps

| Wk | Step | 파일 경로 | 도구 | 완료 판정 |
|----|------|----------|------|-----------|
| 1-2 | vLLM 0.7+ + KVFlow trace setup | `vllm/distributed/kv_transfer/` | vLLM build | KVFlow synthetic trace replay 가능 |
| 3-5 | M1 agent_dpa_table 구현 | `vllm/distributed/kv_transfer/agent_dpa_table.py` | Python + pytest | 16 agent × 1M block stress 완료 |
| 5-7 | M2 poison_handler 구현 | `vllm/distributed/kv_transfer/poison_handler.py` | Python + EDAC sysfs | sub-second poison detect 검증 |
| 7-9 | RFC #19329 token-range recompute | vLLM scheduler | Python | 8K token + 32 layer recompute < 200ms |
| 9-12 | M3 LLMServingSim multi-agent | `LLMServingSim/configs/multi_agent.yaml` | LLMServingSim 1.x | 4/8/16 agent 30 runs |
| 12-14 | Ablation + CacheSolidarity 비교 | M1+M2+M3 통합 | vLLM + LLMServingSim | factorial 30 runs |
| 14-16 | Manuscript polish | 전체 | git, paper draft | USENIX Security 2027 submission |

### (7) Preliminary Analysis Metrics

| 지표 | 측정 도구 | Target |
|------|----------|--------|
| Cross-agent corruption rate | LLMServingSim multi-agent counter | 100% 차단 |
| Recompute overhead (% TPOT) | vLLM scheduler stat | <5× TPOT (200ms) |
| Throughput (multi-agent) | LLMServingSim throughput | drop <5% |
| Agent-level TPOT | vLLM benchmark | 40ms tight bound 충족 |
| agent_dpa_table footprint (MB) | Python `tracemalloc` | average <16MB |

**Preliminary Study (4 단계)**: (1) baseline KVFlow throughput 재현, (2) M1 단독 agent_id table footprint 측정, (3) M2 단독 poison detect-to-recompute latency, (4) M3 통합 multi-agent corruption rate.

---

## 6. 관련 연구 / 차별점 / Risk / Baseline 표

| # | Paper | Year | Axis | P3 차별화 | what / why / how |
|---|-------|------|------|-----------|---------------------|
| C1 | KVFlow ([arXiv:2507.07400](https://arxiv.org/abs/2507.07400)) | 2025-07 | Multi-agent prefix reuse | reliability axis 부재 | what: multi-agent prefix cache; why: throughput driver; how: vLLM extension |
| C2 | CacheSolidarity ([arXiv:2603.10726](https://arxiv.org/abs/2603.10726)) | 2026-03 | Timing side-channel | hardware fault axis 직교 | what: timing side-channel defense; why: 70% reuse + 30% latency; how: selective isolation |
| C3 | SafeKV ([arXiv:2508.08438](https://arxiv.org/abs/2508.08438)) | 2025-08 | Cross-tenant privacy | hardware-induced poison 직교 | what: privacy enforcement; why: cross-tenant; how: system-level |
| C4 | Oneiros ([arXiv:2507.11507](https://arxiv.org/abs/2507.11507)) | 2025-07 | Multi-tenant parameter remapping | isolation axis 직교 | what: parameter remap; why: memory efficiency; how: parameter sharing |
| C5 | Targeted BFA on Agents ([arXiv:2603.10042](https://arxiv.org/abs/2603.10042)) | 2026-03 | Agent BFA attack | P3 의 defense target 강화 | what: agent-level BFA; why: threat model; how: bit-flip attack |
| C6 | TraCT ([arXiv:2512.18194](https://arxiv.org/abs/2512.18194)) | 2025-12 | CXL shared KV (latency) | DPA poison/ECS 미언급 | what: CXL shared memory; why: TTFT 9.8×; how: cacheline flush |

**Risk + 완화**:
- (a) agent_id × DPA_range table footprint 가 worst-case 128MB → 2-level sparse hash 로 average 4-16MB.
- (b) cross-agent prefix reuse 깨짐 → trusted/untrusted zoning 으로 within-tenant 는 reuse 허용.
- (c) CXL device mailbox interface 가 vendor-specific → Linux 6.16 EDAC scrub_subsystem 의 mailbox protocol 만 사용 (sim emulated).

---

## 7. R49 Cross-check 자동 점검

| 항목 | 내용 | 일관 |
|------|------|------|
| (a) 개요 simulator | vLLM + LLMServingSim | ✅ |
| (b) M1/M2/M3 R47 path | R47.2 / R47.2 / R47.3 | ✅ |
| (c) (4) Simulator/Tools | vLLM 0.7+ + LLMServingSim 1.x | ✅ |
| (d) (6) Implementation Steps | vLLM source mod + LLMServingSim multi-agent | ✅ |

**4/4 항목 일관 ✅** — ChaosMem 잔존 0건.

---

## 8. R45 Self-check

| R45 항목 | 내용 | 통과 |
|----------|------|------|
| R45.1 | CXL device 명령은 emulated (sysfs read-only) | ✅ |
| R45.7 | vLLM source mod path | ✅ |
| R45.9 | LLMServingSim 1.x active | ✅ |
| R47.1 | gem5 미사용 | ✅ |

---

## 9. Reviewer Scoring 표

| Phase | Novelty | Differentiation | Impact | Feasibility | 평균 | 핵심 critique |
|-------|--------:|----------------:|-------:|------------:|-----:|----------------|
| Phase 1 | 9 | 8 | 8 | 8 | **8.4** | initial |
| Phase 2 | 9 | 8 | 8 | 8 | **8.4** | "CacheSolidarity unified framework paragraph" / "agent_id × DPA_range footprint 분석" / "agent 격리 시 cross-agent prefix reuse 깨짐" |
| Phase 1' | 9 | **8.5** | **8.5** | 8 | **8.5** | Phase 2 critique 100% 해소 (SafeKV/Oneiros/Targeted BFA on Agents baseline 추가, 2-level sparse hash 명시, trusted/untrusted zoning) |
| Phase 1'' | 9 | 8.5 | 8.5 | 8 | **8.5** | polish — vLLM-ascend RFC #5067 mainstream supplement narrative |

**3 expert sponsor Y 만장일치**.

---

## 10. OpenReview Check

USENIX Security 2027 fit 의 published peer-reviewed work 부재 (CXL DPA × multi-agent KV poison axis). **CacheSolidarity ([arXiv:2603.10726](https://arxiv.org/abs/2603.10726))** 의 timing side-channel defense framework 와 unified framework paragraph 추가는 reviewer simulation 결과 — Phase 1' 에서 반영됨.
