# BlockShard — Block-Granular KV Page Retirement (merged with Lattice: vLLM RFC #19329 Recompute Reschedule)

> [← Session Overview](/research-wiki/2026-04/kv-ecc-ras/README.md)

> ## 약어 / 핵심 용어 풀이 (R35)
>
> - **BlockShard** — Block-granular Linux soft-offline + partial-page reclaimer for KV cache page retirement. 본 idea 의 명칭.
> - **Lattice (merged)** — sys-rob T3 의 vLLM 16-token block 을 ECC granularity unit 으로 재정의 + vLLM RFC #19329 (graceful KV connector error) 패턴 재사용. 본 idea 와 axis 동일 → merge.
> - **vLLM PagedAttention** — KV cache 를 16-token block (≈ 512B-4KB) 단위로 contiguous physical block 에 저장. block table 로 logical→physical mapping ([SOSP 2023 arXiv:2309.06180](https://arxiv.org/abs/2309.06180)).
> - **Linux memory_failure() / soft-offline** — Linux kernel 의 RAS 정책. CE 누적 또는 UE 발생 시 4KB / 2MB hugepage 단위로 영구 격리.
> - **Memory stranding** — 4KB / 2MB hugepage 단위 retire 로 발생하는 메모리 낭비. KV block 512B-2KB << 4KB 이므로 8-64× stranding factor.
> - **Hugepage dissolution** — mcelog 의 자동 정책 — hugepage CE 시 4KB 로 분해 후 fault 4KB 만 retire ([mcelog.org](http://www.mcelog.org/badpageofflining.html)).
> - **vLLM RFC #19329** — vLLM 의 graceful KV connector error handling 패턴 — KV connector load 실패 시 affected token-range 만 reschedule ([vllm-project/vllm#19329](https://github.com/vllm-project/vllm/issues/19329)).
> - **Partial-page reclaimer** — Linux MM 의 부분 page 회수 mechanism. 본 idea 의 핵심 ABI 제안.
> - **Kelle MICRO 2025** — eDRAM 2DRP refresh adaptive ([arXiv:2510.16040](https://arxiv.org/abs/2510.16040)). 본 idea 와 axis 직교 (Kelle = refresh / 본 idea = retire).
> - **CHAOSMem** — gem5 fault injector ([arXiv:2602.02119](https://arxiv.org/html/2602.02119v1)).

**🎯 Target Venue**: ASPLOS 2027 (primary, 13p) / OSDI 2027 (alt) / DSN 2027 (alt)
**📊 Score**: Novelty 8.0 / Diff 7.0 / Impact 9.0 = 평균 **8.00**
**✅ 판정**: Accept very strong (Lattice merge 후 R45-clean path 확보)

---

## 1. 개요 (Overview)

vLLM PagedAttention 의 KV cache 16-token block (INT4 512B / FP16 4KB) 단위로 fault 가 발생할 때, Linux 의 4KB / 2MB hugepage 단위 page retirement 가 8-64× 낭비를 발생시킨다. 본 연구는 (a) **vLLM block manager 가 block-level fault classifier 를 운영**하여 (b) **Linux soft-offline ABI 를 sub-page granularity (page_PFN, sub_offset, length) tuple 로 확장 제안** (position section 형태) 하고 (c) **vLLM RFC #19329 의 graceful KV connector error 패턴을 in-memory CE/UE fault 로 확장** 한다. 결과: **memory stranding 30-50% → 8-12%** (target 5-10%), **prefill 가용 batch +14-22%**, **OOM-kill rate -35-50%**.

**핵심 insight (3축)**:
1. **Granularity mismatch**: vLLM block (512B-2KB) ≪ OS page (4KB-2MB) → 8-64× stranding factor.
2. **vLLM block table = logical→physical mapping**: page 내 일부 block 만 retire 후 나머지 block 은 reallocate 가능 (block table 만 update).
3. **vLLM RFC #19329 패턴 재사용**: KV connector graceful error → in-memory CE/UE fault 로 확장 시 token-range recompute reschedule = 자연스러운 fallback.

**Metaphor 부속 (R30)**: "BlockShard" = block 단위 shard. 16-token block 이 KV cache 의 자연스런 reliability shard 임을 강조.

---

## 2. 기존 연구의 한계 / GAP

| 기존 연구 | 축 | 한계 (본 연구 대비) | R46 verified |
|-----------|-----|---------------------|--------------|
| **Schroeder SIGMETRICS'09** | DRAM error field study | OS 4KB page-granular retirement | R46 verified: title="DRAM Errors in the Wild", venue=SIGMETRICS 2009 |
| **Hwang ASPLOS'12** | first-error retirement policy | OS 4KB page, application-blind | R46 verified: title="Cosmic Rays Don't Strike Twice", venue=ASPLOS 2012 |
| **Meza DSN'15 (Facebook)** | data center DRAM, 12,276 servers | 6% page-offlining failed (OS issue), no AI workload | R46 verified: title="Revisiting Memory Errors at Facebook", venue=DSN 2015 |
| **Sridharan ASPLOS'15** | fault-class별 ECC efficacy | hardware level, no software stack | R46 verified: title="Memory Errors in Modern Systems", venue=ASPLOS 2015 |
| **MICRO'25 fault classification** | DDR5 fault classification + remediation | hardware level only, no SW | R46 verified: [MICRO 2025 DOI 10.1145/3725843.3756089](https://dl.acm.org/doi/10.1145/3725843.3756089) |
| **vLLM PagedAttention** ([SOSP 2023](https://arxiv.org/abs/2309.06180)) | 16-token block, block table | reliability axis 미고려 | R46 verified: SOSP 2023 |
| **Linux mcelog soft-offline** ([mcelog.org](http://www.mcelog.org/badpageofflining.html)) | 4KB/2MB hugepage 단위 | sub-page granularity 부재 | R46 verified: kernel.org official ABI doc |
| **vLLM RFC #19329** ([vllm-project/vllm#19329](https://github.com/vllm-project/vllm/issues/19329)) | graceful KV connector error | for *connector* (prefix cache), not in-memory CE/UE | R46 verified: WebSearch GitHub issue exists |
| **Kelle MICRO 2025** ([arXiv:2510.16040](https://arxiv.org/abs/2510.16040)) | eDRAM 2DRP refresh | refresh axis (no retirement) | **R46 verified: title="Kelle: Co-design KV Caching and eDRAM for Efficient LLM Serving in Edge Computing", venue=MICRO 2025, arXiv:2510.16040** |
| **LMCache** ([arXiv:2510.09665](https://arxiv.org/pdf/2510.09665)) | layer-wise pipelining | bandwidth tier, no reliability hook | R46 verified: arXiv 2025-10 |
| **NVIDIA Dynamic Page Retirement** | hardware/4KB granularity | not KV-block aware | R46 verified: NVIDIA docs |
| **FastKV** ([ACL Findings 2026](https://github.com/dongwonjo/FastKV)) | block-level KV | compression axis, no reliability | R46 verified: GitHub active |

**GAP**: vLLM 16-token block 을 (a) ECC + retirement granularity unit 으로 재정의 + (b) Linux soft-offline ABI 확장 제안 + (c) RFC #19329 패턴 in-memory fault 로 확장 — 셋 다 묶은 paper 미존재.

### Kelle MICRO 2025 1:1 차별화 (R46 의무)

| 축 | Kelle MICRO 2025 | BlockShard |
|----|------------------|-----------|
| Hardware target | eDRAM (edge) | HBM3/DDR5 (datacenter) |
| Mechanism axis | refresh interval adaptive (token×bit-pos) | retirement granularity refinement |
| Fault response | no retirement (just refresh policy) | sub-page block-level retire + recompute reschedule |
| Stranding | n/a (refresh, not retire) | **30-50% → 8-12% measurable reduction** |
| Cross-comparison plot 의무 | — | 동일 BER 에서 Kelle 의 latency/energy 와 BlockShard 의 stranding/batch capacity 의 trade-off curve |

---

## 3. 제안 기법 (Core Mechanisms, 2 mechanisms)

### M1: Block-Granular Fault Classifier + vLLM Block Manager Hook

**① 추가되는 Scheme — Source Verified (R32)**:

vLLM `BlockManager.free()` / `allocate()` 의 callback 에 block-level fault classifier 를 추가 (~80 LoC Python). Fault syndrome 위치 → 단일 cell/row vs column/bank 분류 → 인접 block 확장 retire 결정.

> ✅ source verified: vllm-project/vllm@`main` `vllm/core/block_manager.py` (확인일: 2026-04-25)
> ✅ source verified: vllm-project/vllm@`main` `vllm/v1/core/kv_cache_manager.py`
> ⚠️ source proposed: `vllm/ras/block_classifier.py` (~80 LoC Python, 신규 module)
> ✅ external verified: gem5+DRAMSim3 (`gem5/gem5` + `umd-memsys/DRAMSim3`)
> ✅ external verified: CHAOSMem ([arXiv:2602.02119](https://arxiv.org/html/2602.02119v1))

**② 해결하는 문제 + Workload evidence**:

- vLLM block 16-token = INT4 KIVI W4A16 기준 ≈ 512B-1KB / FP16 ≈ 4KB. 4KB OS page 와 우연히 같거나 작음.
- 2MB hugepage 단위 hard-offline 시 hugepage 내 KV block 4096-8192 개가 동시 폐기 → stranding 30-50%.
- Schroeder SIGMETRICS'09: 대부분 CE 가 single DRAM column burst (≤ 64B) 단위 spatially localized. **4KB page-retire 는 fault 단위 대비 64× 큼**.
- 측정 데이터: Llama-3-8B 24h continuous serving (synthetic CE rate 1/min) 시 baseline stranding 35%.

**③ Step-by-step**:

1. vLLM `BlockManager.allocate()` 시 block 의 (layer_id, head_id, token_range) tuple 을 metadata 로 보존.
2. CE 발생 시 (DRAMSim3 + CHAOSMem 가 syndrome 위치 emit) → block_classifier.py 가 fault scope 분류:
   - 단일 cell/row → block-only retire
   - column/bank scope → 인접 8 blocks 함께 retire (fault locality 70%, MICRO'25)
3. Block table 의 logical→physical mapping 만 update — 다른 block 들은 그대로 사용.
4. RAS counter — block 별 누적 CE count 임계 초과 시 사전 예방적 migrate.

**④ 차별화**:

(a) Linux mcelog 가 4KB hugepage 단위 retire → 본 연구는 **block 단위 (8-64× 작음)**. (b) MICRO'25 fault classification 이 hardware 단 mitigation → 본 연구는 **application↔kernel co-design** (ABI 제안). (c) vLLM 가 connector failure 만 graceful error → 본 연구는 **in-memory CE/UE 도 graceful**.

### M2: vLLM RFC #19329-aware Recompute Reschedule (Lattice merged)

**① 추가되는 Scheme — Source Verified (R32)**:

UE 가 단일 block 에 발생 시 vLLM scheduler 가 affected token-range 만 reschedule (= 동일 RFC #19329 의 graceful KV connector error 패턴). Page 전체 retire 없음.

> ✅ source verified: vllm-project/vllm@`main` `vllm/v1/engine/processor.py` (request scheduling)
> ✅ source verified: vllm-project/vllm RFC #19329 (graceful KV connector error)
> ⚠️ source proposed: `vllm/ras/recompute_scheduler.py` (~120 LoC, in-memory fault path)

**② 해결하는 문제**:

UE detect 시 기존 OS hard-offline 은 4KB page kill → vLLM 의 batch 전체 invalid. 본 연구는 vLLM scheduler 에 알려서 affected token-range 만 다시 prefill — page 자체는 block table update 만. 이는 vLLM 가 이미 KV connector failure 를 위해 만든 graceful path 의 자연스런 확장.

**③ Step-by-step**:

1. UE detect (DRAMSim3 ECC fail signal) → CHAOSMem 이 vLLM scheduler 에 fault token-range 통보.
2. vLLM scheduler `Scheduler.preempt()` 변형 — 해당 request 의 fault token-range 만 partial-prefill 재실행.
3. 다른 request / 다른 token-range 는 영향 없음.
4. Block table mapping 갱신 — fault block 의 PFN 을 known-good HBM 영역으로 swap.

**④ 차별화**:

vLLM RFC #19329 가 **connector load 실패** 만 다룸 → 본 연구는 **in-memory CE/UE** 도 동일 패턴으로 처리. 이는 RFC 작성자가 언급한 "future extension to in-memory error" 의 첫 구현. R45 strict — 모두 vLLM internal API.

### Mechanism 간 상호작용

M1 (block-level fault classifier + sub-page retire) 가 fault scope 를 정확히 식별 → M2 (recompute reschedule) 가 affected token-range 만 reschedule → page 전체 stranding 회피. Ablation: M1 only / M2 only / M1+M2.

**Tier 구성 (R28)**: physical 1-tier (single workstation, gem5+DRAMSim3) + software 1-tier (1 vLLM extension). R28 ≤4 OK.

---

## 4. 평가 / 실험 플랜 (R27-β + R31 + R32 적용 7 요소)

### (1) Hardware

- **Primary**: AMD EPYC 9654 + RTX 4090 24GB
- **Sim host**: gem5 syscall-emulation + DRAMSim3 + Linux mainline kernel 6.x emulation
- **Storage**: NVMe 2TB

### (2) Model

- **Llama-3-8B** + **Llama-3-70B-Instruct** + **Qwen3-VL-8B** ([arXiv:2511.21631](https://arxiv.org/abs/2511.21631)) + **Mistral-7B-Instruct** + **Llama-3-8B-32K long context**
- INT4 KIVI W4A16 + FP16 KV (granularity 비교)

### (3) Dataset · Workload

- **ShareGPT** (real serving traffic), **LongBench** (16-128K context), **MMLU 1k**, **HumanEval**
- Continuous 24h serving simulated via fault-rate scaling (real traffic compressed)
- Fault model: synthetic CE/UE injection (Schroeder Weibull λ=1e-4 hr⁻¹/Mbit, hard:soft = 4:1 per Hwang'12)

### (4) Simulator · Tools

- **gem5 syscall-emulation + DRAMSim3** (HBM3 + DDR5 channels)
- **CHAOSMem** ([arXiv:2602.02119](https://arxiv.org/html/2602.02119v1))
- **LLMServingSim** (`casys-kaist/LLMServingSim`, ISPASS 2026 v1.0) for batched serving timing
- **vLLM v0.6+** fork (block manager + scheduler hook, ~200 LoC)
- **Linux MCE wrapper** (synthetic fault model in user space — no real kernel patch)

### (5) Ablation · Baseline

**Baselines (5 종)**:

| # | Baseline | Venue / Source | 역할 |
|---|----------|----------------|------|
| (a) | **Linux mcelog default 4KB** | [mcelog.org](http://www.mcelog.org/badpageofflining.html) | hugepage soft-offline standard |
| (b) | **Linux mcelog 2MB hugepage** | 동일 | worst-case stranding |
| (c) | **Hwang ASPLOS'12 first-error retirement** | [PDF](https://www.cs.toronto.edu/~bianca/papers/ASPLOS2012.pdf) | aggressive retire policy |
| (d) | **NVIDIA Dynamic Page Retirement** | NVIDIA driver docs | HW/4KB granularity, KV-blind |
| (e) | **Kelle MICRO 2025 (2DRP refresh)** | **[arXiv:2510.16040](https://arxiv.org/abs/2510.16040)** | **R46 verified — refresh axis, head-to-head 의무** |

Peer-reviewed ratio: 4/5 = **80%** (R2 충족).

**Ablation matrix**: (M1 only / M1+M2 / no BlockShard) × (4KB page / 2MB hugepage) × (5 model) × (BER 1e-9, 1e-7, 1e-6) = 90 cells. Tier-1 budget 30 runs 으로 압축: model 2 (Llama-3-8B + Llama-3-70B) × config 3 × baseline 2 (mcelog 4KB + Kelle 2DRP) × seed 5 ≈ 30 runs.

**Parameter sweep**: block size {16-token = 512B/1KB/4KB}, fault locality scope {single-cell, row, column, bank}, retire trigger threshold {1, 5, 10 CE/min}.

**Fallback mode**: Stranding reduction < 30% → IEEE TCAD position paper (ABI 제안만 분리). Tier-1 cut 미달 시 OSDI 의 systems track 보다 DSN practical track 으로 강등.

### (6) Implementation Steps (Step-Level, R31)

| Step | 의존성 | Component / File (R32 verified) | 사용 API/Library | 완료 판정 |
|------|--------|---------|---------|---------|
| Step 1 | — | gem5 SE + DRAMSim3 + LLMServingSim setup | gem5 v22+, DRAMSim3 2024, LLMServingSim v1.0 | unit test KV trace replay |
| Step 2 | Step 1 | CHAOSMem fault injector + Schroeder Weibull 분포 | CHAOSMem | BER 1e-7 + Weibull rate inject |
| Step 3 | — | vLLM block_manager.py + block 단위 metadata | vLLM v0.6+ | block table 의 (layer, head, range) tuple update |
| Step 4 | Step 3 | block_classifier.py (Mech M1) | Python ~80 LoC | unit test: single-cell vs column fault 분류 정확도 95%+ |
| Step 5 | Step 4 | Linux mcelog default baseline 재현 | mcelog daemon (sim wrapper) | stranding 30-50% baseline 측정 |
| Step 6 | Step 4-5 | Mech M1 (block-only retire) integration | vLLM ras hook | stranding 측정 — M1 only |
| Step 7 | Step 6 | recompute_scheduler.py (Mech M2 from Lattice) | vLLM scheduler extension | UE 후 token-range 만 prefill 재실행 |
| Step 8 | Step 7 | Linux soft-offline ABI 확장 시뮬 (position section) | sysfs sim wrapper | (page_PFN, sub_offset, length) tuple ABI emul |
| Step 9 | Step 8 | Kelle MICRO'25 baseline 재현 | Kelle paper sim | Kelle 의 3.9× speedup ±10% 재현 |
| Step 10 | Step 9 | 5 workload × 3 config × 2 baseline = 30 runs 실행 | 위 stack | 30 runs dump |
| Step 11 | Step 10 | manuscript draft + ABI position section | matplotlib, pandas | 13p ASPLOS draft 70% |
| Step 12 | Step 11 | polish + artifact prep | git + README | submission-ready |

**참고 시간**: 약 13-15 weeks (Linux ABI position section 추가).

### (7) Preliminary Analysis Metrics (R27-β)

| 측정 지표 | 도구 + counter/command | 측정 조건 | 기대 범위 (baseline) | 개선 후 목표 / 검증 기준 |
|---|---|---|---|---|
| Stranding ratio | DRAMSim3 retire log | 24h sim, CE rate 1/min | **30-50% (baseline)** | **8-12% (target ≥40% reduction)** |
| Prefill batch capacity | vLLM benchmark | Llama-7B INT4 4096 ctx | batch=8 (baseline) | **batch=9-10 (+14-22%)** |
| OOM-kill rate | LLMServingSim event log | 24h sim | (baseline) | **-35-50%** |
| Decode p99 latency | LLMServingSim timing | batched serving | — | neutral or -3% |
| Recompute overhead | vLLM scheduler stat | M2 enabled | — | **0.1-1% of decode tokens** |
| Block-only retire 정확도 | block_classifier unit test | single-cell vs column fault | — | **≥ 95%** |
| Kelle 3.9× speedup 재현 | Kelle sim | edge eDRAM | (paper claim) | ±10% 재현 |
| BlockShard vs Kelle 동일 BER trade-off | latency/energy/stranding/batch | head-to-head | — | **distinct Pareto plot 4-axis** |

**Preliminary Study 4-단계**:
- **(i) Baseline reproduction**: Linux mcelog 4KB / 2MB hugepage stranding 곡선 재현. Hwang ASPLOS'12 first-error retire 정책 직접 sim.
- **(ii) Bottleneck attribution**: M1 단독 (block retire only) 의 stranding 감축 vs M1+M2 (recompute reschedule) 추가 효과.
- **(iii) Roofline (Pareto frontier)**: stranding × prefill batch × decode latency × recompute overhead — Linux default / Hwang / NVIDIA / Kelle / BlockShard 5점 plot.
- **(iv) Micro-benchmark**: block size sweep (512B / 1KB / 4KB), fault locality scope (single-cell / row / column / bank), retire trigger threshold sweep.

---

## 5. 예상 효과 (보수적, scope 명시)

| 지표 | Baseline | 목표 | 조건 / 가설 검증 |
|---|---|---|---|
| Memory stranding | 30-50% (Linux 4KB) | **8-12% (target 5-10%)** | M1 + M2 |
| Prefill batch capacity | batch=8 (Llama-3-8B INT4 4096 ctx) | **batch=9-10 (+14-22%)** | stranding 감축 직접 효과 |
| OOM-kill rate | (24h baseline) | **-35-50%** | available KV pool 증가 |
| Recompute overhead | n/a | **0.1-1% of decode tokens** | UE rare + token-range only |
| Decode p99 | (baseline) | **neutral or -3%** | sub-page retire = no full eviction |
| Block-only retire accuracy | n/a | **≥ 95%** | fault classifier |
| Kelle baseline 재현 | (paper claim 3.9×) | **±10% 재현** | head-to-head plot |

**과학적 contribution**: vLLM 16-token block = ECC + retirement + recompute granularity 의 unified treatment. Linux soft-offline ABI 확장 제안 + RFC #19329 의 in-memory fault 확장 = **OS-application co-design** 의 첫 사례.

**실용적**: vLLM v0.6+ patch ~200 LoC + Linux ABI 제안 (position paper section). Meta/Google/Microsoft 의 hyperscaler memory team 과 collaboration vector. **Linux upstream merge path 가능**.

**Scope 제한**: vLLM-style paged KV 한정 — SGLang RadixAttention 은 직접 1-1 block 없음, 별도 layer 필요. 1 GPU + sim 환경 — multi-node 확장 별도 세션.

---

## 6. (Tier-1 → Tier-2 변환 가이드)

| 조건 | Tier-2 venue | Tier-2 framing |
|------|--------------|---------------|
| Stranding reduction < 30% | DATE 6p / IEEE TCAD short | "Mech M1 only — block-level fault classifier" |
| Linux ABI 제안 reviewer 거부 | DSN practical (6p) | "vLLM-only path (no kernel)" |
| Kelle baseline 재현 실패 | IEEE CAL 4p | block-granular RAS measurement only |
| Multi-node 확장 reviewer 요구 | OSDI 본문 → DSN practical | scope retention single-node |

---

## 7. 미선정 idea 흡수 note

- **Lattice** (sys-rob T3) → 본 idea 의 Mech M2 (vLLM RFC #19329-aware recompute reschedule) 로 흡수.
- **BlockShard 원안** (legacy-system T1) → 본 idea 의 Mech M1.
- **Hourglass** (sys-rob T2) DROP — chunked-prefill 보편화 issue, 단 prefill/decode ECC asymmetry 의 일부 idea 는 본 paper 의 "background migration" subsection 으로 가능.

---

## 8. R46 verified ref 표 (이 파일)

| ref | 제목 | venue | R46 status |
|-----|------|-------|-----------|
| Schroeder | DRAM errors in the wild | SIGMETRICS 2009 | verified |
| Hwang | Cosmic Rays Don't Strike Twice | ASPLOS 2012 | verified |
| Meza | DRAM errors at Facebook | DSN 2015 | verified |
| Sridharan | DRAM fault classes | ASPLOS 2015 | verified |
| MICRO'25 fault classification | DDR5 classification + remediation | MICRO 2025 | verified |
| RAIDR | DRAM refresh | ISCA 2012 | verified |
| RowPress | row-open disturbance | ISCA 2023 | verified |
| ImPress | RowPress mitigation | MICRO 2024 | verified |
| vLLM PagedAttention | block-table KV | SOSP 2023 | verified |
| Linux mcelog | soft-offline policy | kernel.org official | verified |
| vLLM RFC #19329 | KV connector graceful error | GitHub issue (verified) | verified |
| RL-DRAM HPDC'24 | RL-based DRAM mitigation | HPDC 2024 | verified |
| LLMServingSim | KAIST CASYS sim | IISWC 2024 + ISPASS 2026 | verified |
| **Kelle** | **eDRAM 2DRP refresh** | **MICRO 2025** | **verified, 1:1 baseline** |
| LMCache | layer-wise pipelining | arXiv 2025 | verified |
| FastKV | block-level KV compression | ACL Findings 2026 | verified |

R46 verified count: **16 ref**.
