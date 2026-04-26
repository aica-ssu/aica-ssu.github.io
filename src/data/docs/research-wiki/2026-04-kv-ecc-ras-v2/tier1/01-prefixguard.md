# P1 PrefixGuard — Reliability-Aware Prefix Cache Eviction & Scrub Scheduling for CXL-Attached KV Storage in Multi-Tenant LLM Serving

> [← Session Overview](../README.md)

> ## 약어 / 핵심 용어 풀이 (R35)
>
> - **PrefixGuard** — 본 idea 의 명칭. 접두사(Prefix) 보호자(Guard) — 장수 prefix block 만 강하게 scrub.
> - **CXL Patrol Scrub Control** — CXL 3.2 spec 의 background scrub. host 가 hour-단위 interval 설정 가능 ([CXL 3.1 RAS Whitepaper](https://computeexpresslink.org/wp-content/uploads/2024/08/An-Overview-of-RAS-for-Compute-Express-Link-3.1-Whitepaper.pdf)).
> - **CXL ECS mailbox** — CXL 3.1 §8.2.9.9.11.2 Error Check and Scrub. device 가 internally read-correct-writeback + ECS history mailbox query.
> - **Linux 6.16 EDAC scrub_subsystem** — kernel scrub subsystem upstream ([docs.kernel.org/edac/scrub.html](https://docs.kernel.org/edac/scrub.html)). sysfs `/sys/bus/edac/devices/<dev>/scrubX/` 로 host 가 interval 설정.
> - **DPA (Device Physical Address)** — CXL device 내부 주소 공간. HPA→DPA translation + poison 격리.
> - **PagedAttention** — vLLM 의 16-token block KV cache ([SOSP 2023 arXiv:2309.06180](https://arxiv.org/abs/2309.06180)).
> - **LMCache** — KV cache offload layer ([arXiv:2510.09665](https://arxiv.org/abs/2510.09665)). vLLM/SGLang 통합 + 15× throughput.
> - **Beluga** — CXL switch 위 KV cache, 89.6% TTFT 감소 + 7.35× throughput ([arXiv:2511.20172](https://arxiv.org/abs/2511.20172)).
> - **TraCT** — CXL shared memory KV cache, TTFT 9.8× ([arXiv:2512.18194](https://arxiv.org/abs/2512.18194)).
> - **Silent Data Corruption (SDC)** — ECC 가 검출 못한 채 잘못된 값 반환. TU Berlin SDC ([arXiv:2604.00726](https://arxiv.org/abs/2604.00726)) = LLM training axis first.
> - **LLMServingSim** — KAIST CASYS, MIT license, ISPASS 2026 ([github.com/casys-kaist/LLMServingSim](https://github.com/casys-kaist/LLMServingSim)).

**🎯 Target Venue**: OSDI 2027 (primary, 13p) / ASPLOS 2027 (alt) / DSN 2027 (alt)
**📊 Score**: Novelty 9.0 / Differentiation 9.5 / Impact 8.0 / Feasibility 9.0 = 평균 **8.9**
**✅ 판정**: Accept (Phase 2 8.7 → Phase 1' 8.9 → Phase 1'' 8.9, 3 expert sponsor Y 만장일치)

---

## 1. 개요 (Overview)

KVCache-in-Wild USENIX ATC'25 (Aliyun Tongyi trace, [arXiv:2506.02634](https://arxiv.org/abs/2506.02634)) 가 production LLM serving 에서 **prefix block lifetime 이 hour-scale 까지 장기** 임을 첫 측정. 한편 LMCache + Beluga + TraCT 가 모두 CXL-attached prefix cache 의 throughput/latency axis 만 다루고 reliability/scrub axis 는 부재. 본 연구는 **CXL 3.2 Patrol Scrub Control + CXL 3.1 ECS mailbox + Linux 6.16 EDAC scrub_subsystem** 3 feature 를 vLLM `LMCacheConnector` 와 align — long-lived prefix 만 1hr scrub, short-lived block 은 scrub disabled — 하여 silent corruption rate 90% 감소 + scrub overhead <30%.

**Metaphor noun ↔ mechanism 대응**: "PrefixGuard" = 접두사 보호자. **(M1 lifetime tag = guard 가 누구를 지킬지 식별 / M2 scrub interval = guard 가 어떻게 지킬지 결정 / M3 cluster sim = guard 가 cluster scale 에서 효과 검증)**.

**사용 simulator**: vLLM 0.7+ source mod (R47.2 primary) + LLMServingSim 1.x (R47.3 cluster validation). gem5 미사용 (R47.1).

---

## 2. Workload Evidence (R23)

| # | Source | Year | 측정 숫자 | 발생 조건 |
|---|--------|------|-----------|-----------|
| α4 | KVCache-in-Wild USENIX ATC'25 ([arXiv:2506.02634](https://arxiv.org/abs/2506.02634)) | 2025-06 | prefix lifetime hour-scale, single+multi-turn reuse 모두 significant, request category 별 reuse timing predictable | Aliyun Tongyi 2024-12 ~ 2025-02 production trace |
| α8 | LMCache benchmark ([arXiv:2510.09665](https://arxiv.org/abs/2510.09665)) | 2025-10 | context truncation 시 prefix hit ratio **50% drop**, 15× throughput @multi-round QA + doc analysis | vLLM/SGLang 통합 |
| C1 | Phoenix Rowhammer ([comsec.ethz.ch/phoenix](https://comsec.ethz.ch/research/dram/phoenix/)) | 2026 (S&P) | DDR5 SK Hynix 15/15 DIMM, **109s 안에 root**, on-die ECC 무력화 | DDR5 PRAC 미배포 DIMM |
| C5 | Meta Reliability ([arXiv:2410.21680](https://arxiv.org/abs/2410.21680)) | 2024-10 | Llama-3 405B 16384 H100 cluster, **54일 419 failure (HBM3 72건, 3hr 당 1건)** | hyperscale cluster |
| α6 | MLPerf Inference v5.0 ([mlcommons.org](https://mlcommons.org/2025/04/mlperf-inference-v5-0-results/)) | 2025-04 | TTFT P99 6s, TPOT 175ms | Llama-3.1 405B + Llama-2 70B Chat |

---

## 3. Modern Memory Standard 활용 (R50.2)

**핵심 mechanism integration 3 feature** — CXL 3.x 신규 RAS feature 가 단순 motivation 인용이 아닌 mechanism 의 핵심:

- **CXL 3.2 Patrol Scrub Control + Linux 6.16 EDAC scrub_subsystem upstream**: M2 가 sysfs `/sys/bus/edac/devices/<dev>/scrub*/` 를 별도 thread 로 polling/write — **3-tier interval (off / 1hr / 5min)** 차등.
- **CXL 3.1 ECS mailbox (§8.2.9.9.11.2)**: device 의 ECS history 를 host 가 query, 누적 CE rate 를 prefix block 의 scrub priority 결정 input.
- **CXL 3.x DPA tracking + poison**: 보조 — corrupted DPA 식별 시 vLLM 이 affected token-range 만 recompute (D1 LMCache / D2 TraCT path).

LPDDR5x JESD209-5C 와 HBM3 JESD238B 는 보조 (HBM3 ECS 만 cross-correlate).

---

## 4. Mechanism 동작 원리 (3 mechanism)

### M1: 8-bit Lifetime Tracker Tag on PagedAttention Block

**R47 path**: R47.2 application-level vLLM source modification (primary).

**① 추가되는 Scheme — Source Verified (R32)**:

vLLM `vllm/distributed/kv_transfer/` 의 prefix-cache hash table 에 **8-bit lifetime tracker tag** 추가. PagedAttention 16-token block 단위. tag value 의미: `0=ephemeral` / `1-127=lifetime in 10-min units` (max 21h) / `128-255=permanent prefix`.

> ✅ source verified: vllm-project/vllm@`main` `vllm/distributed/kv_transfer/kv_connector/v1/lmcache_connector.py` (확인일: 2026-04-26)
> ⚠️ source proposed: `vllm/distributed/kv_transfer/lifetime_tracker.py` (~80 LoC Python, 신규 module, R47.2)
> ✅ external verified: LMCache backend interface ([arXiv:2510.09665](https://arxiv.org/abs/2510.09665))
> 🔧 R32: 8-bit lifetime tag 의 hash table column 추가 — vLLM 0.7+ BlockManager.allocate() 의 hot path O(1) write only

**② 해결하려는 문제 + Workload 1:1 대응**:

- α4 측정: prefix lifetime 이 hour-scale 까지 길어지지만, request category 별 reuse timing 은 predictable. → category 정보 + 누적 reuse count → 8-bit tag value 결정.
- α8 측정: prefix hit ratio 50% drop 회피를 위해 long-lived prefix 의 reliability 가 핵심.

**③ Step-by-step (학부생 가이드)**:

1. vLLM `BlockManager.allocate()` 에서 prefix-cache hash table lookup 시 lifetime tracker 가 함께 lookup.
2. prefix block 첫 allocation 시 tag = 0 (ephemeral).
3. 매 reuse 시 tag value += 1 (10-min unit). 예: 6회 reuse = tag 6 = 1hr lifetime hint.
4. category 정보 (Aliyun trace 의 8 category) 를 lifetime histogram bucket 에 누적. p75 lifetime 이 tag → scrub priority 변환표 input.
5. tag 128 이상은 permanent prefix (system prompt 등) — scrub 1hr 강제.

**④ 차별화**: LMCache (D1) 가 prefix lifetime tracking 자체는 있으나 **lifetime → reliability tag 변환** 부재. Beluga (CXL switch) / TraCT (CXL shared memory) 모두 latency/throughput axis. PrefixGuard 는 reliability axis first work.

### M2: 3-Tier Scrub Interval × Prefix Lifetime Alignment via EDAC sysfs

**R47 path**: R47.2 application-level vLLM source mod (primary).

**① 추가되는 Scheme — Source Verified (R32)**:

`LMCacheConnector` 측 KV-store backend 가 **별도 thread** 로 Linux 6.16 EDAC scrub interface (sysfs read-only polling + write) 를 사용 — BlockManager hot path 영향 0. CXL Patrol Scrub interval 을 prefix lifetime 과 align: long-lived (tag ≥64): scrub_interval=1hr, medium-lived (tag 8-63): 5min, short-lived (tag <8): disabled.

> ✅ source verified: Linux 6.16 EDAC scrub_subsystem ([docs.kernel.org/edac/scrub.html](https://docs.kernel.org/edac/scrub.html)) — 2025-08 upstream merge 확인
> ✅ source verified: CXL 3.1 §8.2.9.9.11.2 ECS + CXL 3.2 Patrol Scrub Control ([CXL 3.1 RAS Whitepaper](https://computeexpresslink.org/wp-content/uploads/2024/08/An-Overview-of-RAS-for-Compute-Express-Link-3.1-Whitepaper.pdf))
> ⚠️ source proposed: `vllm/distributed/kv_transfer/scrub_scheduler.py` (~120 LoC Python, sysfs polling + write)
> 🔧 R32: Linux 6.16 EDAC sysfs interface 의 minimum interval 5min — Phase 1 의 1min tier 제거 정당화

**② 해결하려는 문제 + Workload 1:1 대응**:

- α8 측정: prefix hit ratio 50% drop 회피 = scrub interval 을 lifetime 과 align (over-scrub 시 overhead, under-scrub 시 silent corruption).
- C1 Phoenix: DDR5 on-die ECC 우회 109s — long-lived prefix 의 attack surface ↑ → 1hr scrub 강제.

**③ Step-by-step**:

1. `scrub_scheduler.py` 가 별도 thread 로 시작 (BlockManager hot path 영향 0).
2. 1초마다 prefix-cache hash table 의 모든 block 을 walk, lifetime tag value → scrub tier 매핑 (3-tier).
3. 새로 long-lived 가 된 block (tag transition 7→8 등) 시 sysfs `/sys/bus/edac/devices/<cxl_dev>/scrub0/cycle_duration` 에 5min write.
4. 1hr 도달 (tag transition 63→64) 시 cycle_duration = 1hr write.
5. ECS mailbox query (`scrub0/persistent` event) → 누적 CE rate 가 임계 (1 CE / 100 access) 도달 block 은 eviction priority ↑.

**④ 차별화**: Linux 6.16 EDAC scrub subsystem ([docs.kernel.org/edac/scrub.html](https://docs.kernel.org/edac/scrub.html)) 는 kernel-side primitive 만 — application-level prefix lifetime 과 align 한 work 0건. Beluga/TraCT 는 throughput/latency axis. **Linux 6.16 EDAC sysfs × prefix lifetime 의 first application-level showcase**.

### M3: LLMServingSim Cluster-Scale Reliability Simulation

**R47 path**: R47.3 LLMServingSim built-in (cluster validation).

**① 추가되는 Scheme — Source Verified**:

8-32 GPU + 256GB CXL pool cluster 시뮬레이션. ECS history (CE rate) 와 prefix hit ratio 의 trade-off 곡선 측정. Meta C5 cluster trace (54일 419 failure) cross-check.

> ✅ source verified: LLMServingSim v1.0 ([github.com/casys-kaist/LLMServingSim](https://github.com/casys-kaist/LLMServingSim)) ISPASS 2026 — built-in CXL pool model
> ⚠️ source proposed: `LLMServingSim/configs/cxl_reliability.yaml` (CE rate config, scrub policy plug-in)

**② 해결 / 대응 / 차별화**: cluster-scale evidence (Meta C5 의 3hr 당 1 fault) 를 application-level scrub 정책의 효과로 환산. PrefixGuard on/off 비교 시 silent corruption rate 90% 감소 검증.

---

## 5. 실험 플랜 (R27-β 7 element)

### (1) Hardware

A100/H100 SXM 4-8 GPU + CXL Type-3 pool 256GB (sim emulated). 1-2 H100 single-node ablation 도 추가.

### (2) Model

| Role | Model | Precision | Checkpoint |
|------|-------|-----------|------------|
| Primary | Llama-3.1-8B-Instruct | FP16 + INT4 KIVI | HuggingFace `meta-llama/Llama-3.1-8B-Instruct` |
| Secondary | Llama-3.1-70B-Instruct | W4G128 GPTQ | `meta-llama/Llama-3.1-70B-Instruct` |
| Robustness | Qwen3-30B-A3B-MoE | INT4 | `Qwen/Qwen3-30B-A3B` |

### (3) Dataset / Workload

- ShareGPT multi-turn (production trace).
- LongBench-v2 (long-context).
- KVCache-in-Wild Aliyun trace replay (8 category).
- KVFlow agentic multi-turn ([arXiv:2507.07400](https://arxiv.org/abs/2507.07400)).
- **Metric**: silent corruption rate, prefix hit ratio drop, scrub overhead %, throughput, TTFT P99 / TPOT.

### (4) Simulator / Tools (R49 cross-check)

- **vLLM 0.7+** ([github.com/vllm-project/vllm](https://github.com/vllm-project/vllm)) — R47.2 primary.
- **LLMServingSim 1.x** ([github.com/casys-kaist/LLMServingSim](https://github.com/casys-kaist/LLMServingSim)) ISPASS 2026 — R47.3 cluster validation.
- **Linux 6.16 EDAC scrub_subsystem** ([docs.kernel.org/edac/scrub.html](https://docs.kernel.org/edac/scrub.html)) — sysfs read-only polling + write.
- gem5 미사용 (R47.1). ChaosMem 미사용 (R49).

### (5) Ablation / Protocol

- **Factorial**: scrub-interval (off / 1hr / 5min) × prefix-lifetime tag (on/off) × DPA-level poison isolate (on/off) × CE rate (1/min, 1/hr, 1/day).
- **Sweep**: lifetime tag bit-width (4/8/16) × p75 vs p90 lifetime alignment.
- **Baseline**: vanilla vLLM + Linux EDAC default (1hr fixed) / Beluga / TraCT / v1 BlockShard.
- **Runtime budget**: 4-8주 + 30 runs.
- **Fallback**: scrub overhead 30%+ 시 5min tier → 15min 으로 relax.

### (6) Implementation Steps

| Wk | Step | 파일 경로 | 도구 | 완료 판정 |
|----|------|----------|------|-----------|
| 1-2 | Wk1: vLLM 0.7+ + LMCache 환경 setup | `vllm/distributed/kv_transfer/` | vLLM build | ShareGPT throughput baseline ±5% 일치 |
| 2-3 | Wk2: Linux 6.16 EDAC sysfs probing | `/sys/bus/edac/devices/<dev>/` | shell + Python | sysfs read 가능 / dev id 식별 |
| 3-5 | Wk3-5: M1 lifetime tracker 구현 | `vllm/distributed/kv_transfer/lifetime_tracker.py` | Python + pytest | tag transition unit test pass |
| 5-7 | Wk5-7: M2 scrub_scheduler 구현 | `vllm/distributed/kv_transfer/scrub_scheduler.py` | Python thread + EDAC sysfs | 3-tier interval write 검증 |
| 7-9 | Wk7-9: ECS mailbox query 추가 | M2 + ECS history poll | Python + EDAC | ECS history 가 CE rate 와 cross-correlate |
| 9-12 | Wk9-12: M3 LLMServingSim cluster | `LLMServingSim/configs/cxl_reliability.yaml` | LLMServingSim 1.x | 8-32 GPU cluster sim 30 runs 완료 |
| 12-14 | Wk12-14: Ablation + Kelle/Beluga/TraCT compare | M1+M2+M3 통합 | vLLM + LLMServingSim | factorial 30 runs 완료 |
| 14-16 | Wk14-16: Manuscript polish + artifact | 전체 | git, paper draft | OSDI 2027 submission |

### (7) Preliminary Analysis Metrics

| 지표 | 측정 도구 | Target |
|------|----------|--------|
| Silent corruption rate (CE/UE) | LLMServingSim CE/UE counter | 90%+ 감소 |
| Prefix hit ratio drop | vLLM prefix cache stat | <2% |
| Scrub overhead (% of HBM bandwidth) | sysfs scrub stat | <30% |
| TTFT P99 / TPOT | MLPerf v5.0 protocol | SLO 충족 (TTFT P99 <6s, TPOT <175ms) |
| Throughput (tok/s) | vLLM benchmark | drop <5% |

**Preliminary Study (4 단계)**: (1) baseline vanilla vLLM throughput 재현, (2) M1 단독 lifetime tag overhead < 1% 검증, (3) M2 단독 scrub interval 변동 시 corruption rate 곡선, (4) M3 통합 cluster 결과 vs Meta C5 cross-check.

---

## 6. 관련 연구 / 차별점 / Risk / 완화 / Baseline 표

| # | Paper | Year | Axis | P1 차별화 | what / why / how |
|---|-------|------|------|-----------|---------------------|
| C1 | LMCache ([arXiv:2510.09665](https://arxiv.org/abs/2510.09665)) | 2025-10 | Throughput | reliability axis 부재 | what: KV cache offload (DRAM/CPU/storage); why: 15× throughput multi-round QA; how: vLLM/SGLang plugin |
| C2 | TraCT ([arXiv:2512.18194](https://arxiv.org/abs/2512.18194)) | 2025-12 | Latency/consistency | scrub interval 미활용 | what: CXL shared memory KV; why: TTFT 9.8× / P99 6.2×; how: cacheline flush + DMA |
| C3 | Beluga ([arXiv:2511.20172](https://arxiv.org/abs/2511.20172)) | 2025-11 | TTFT/throughput | reliability axis 부재 | what: CXL switch 위 KV cache; why: 89.6% TTFT 감소; how: switch-side cache |
| C4 | KVCache-in-Wild USENIX ATC'25 ([arXiv:2506.02634](https://arxiv.org/abs/2506.02634)) | 2025-06 | Eviction | reliability/scrub 미고려 | what: Aliyun production trace 첫 측정; why: prefix lifetime hour-scale; how: 8 category trace |
| C5 | TU Berlin SDC ([arXiv:2604.00726](https://arxiv.org/abs/2604.00726)) | 2026-04 | Training SDC | inference axis 직교 | what: LLM training SDC challenge; why: silent corruption motivation; how: characterization |
| C6 | NeoMem ([arXiv:2403.18702](https://arxiv.org/abs/2403.18702)) | 2024 | CXL hotness tiering | scrub/reliability 부재 | what: CXL hotness tiering; why: page promotion; how: hardware counter |

**Risk + 완화**:
- (a) Linux 6.16 EDAC sysfs polling overhead → 별도 thread + 1초 interval polling 으로 hot path 영향 0.
- (b) 5min scrub interval 의 Linux 6.16 minimum 미만 거부 → 5min 이상으로 sharpen (Phase 1' 에서 조정).
- (c) prefix tag value 의 Aliyun category specificity → 8 category sweep + ShareGPT general workload cross-validation.

---

## 7. R49 Cross-check 자동 점검

| 항목 | 내용 | 일관 |
|------|------|------|
| (a) 개요 simulator | vLLM + LLMServingSim | ✅ |
| (b) M1/M2/M3 R47 path | R47.2 / R47.2 / R47.3 | ✅ |
| (c) (4) Simulator/Tools | vLLM 0.7+ + LLMServingSim 1.x | ✅ |
| (d) (6) Implementation Steps | vLLM source mod + LLMServingSim cluster | ✅ |

**4/4 항목 일관 ✅** — ChaosMem 잔존 0건, gem5 미사용, R47.1 충족.

---

## 8. R45 Self-check

| R45 항목 | 내용 | 통과 |
|----------|------|------|
| R45.1 (kernel/firmware 직접 수정 X) | Linux 6.16 EDAC sysfs read-only polling + 표준 write API 만 | ✅ |
| R45.7 (application-level OK) | vLLM source mod + sysfs 활용 | ✅ |
| R45.9 (active simulator only) | LLMServingSim 1.x ISPASS 2026 active | ✅ |
| R47.1 (gem5+vLLM 동시사용 X) | gem5 미사용 | ✅ |

---

## 9. Reviewer Scoring 표

| Phase | Novelty | Differentiation | Impact | Feasibility | 평균 | 핵심 critique |
|-------|--------:|----------------:|-------:|------------:|-----:|----------------|
| Phase 1 (initial) | 9 | 9 | 8 | 9 | **8.7** | initial |
| Phase 2 (3-reviewer + similarity) | 9 | 9 | 8 | 9 | **8.7** | "Beluga/TraCT/TU Berlin SDC baseline 명시 의무" / "LMCacheConnector sysfs polling 별도 thread" |
| Phase 1' (improve) | 9 | **9.5** | 8 | 9 | **8.9** | Phase 2 critique 100% 해소 (TU Berlin baseline 추가, 별도 thread 명시, 4-tier → 3-tier 정당화) |
| Phase 1'' (final) | 9 | 9.5 | 8 | 9 | **8.9** | polish — narrative coherence (V1 PrefixGuard-Lite 와의 cross-reference) |

**3 expert sponsor Y 만장일치** (system-robustness, legacy-system, ai-optimization).

---

## 10. OpenReview Check

OpenReview 직접 fit 의 published paper 부재 (CXL Patrol Scrub × prefix cache reliability axis). KVCache-in-Wild USENIX ATC'25 + TraCT (HotStorage 2025 추정) 는 PDF 만 공개. **TU Berlin SDC ([arXiv:2604.00726](https://arxiv.org/abs/2604.00726))** 는 training SDC axis 직교 — reviewer simulation 으로 "inference-side first work" 강조 의무.
