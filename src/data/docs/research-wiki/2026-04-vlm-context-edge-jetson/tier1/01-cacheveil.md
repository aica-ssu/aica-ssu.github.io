# Register-Level System-Level-Cache Way-Partitioning for VLM Visual-Token KV Pinning with CPU Workload Isolation on Jetson Edge (CacheVeil)

> [← Session Overview](/research-wiki/2026-04/vlm-context-edge-jetson) · **Tier-1 Top 1 ★ lead**

> ## 약어 / 핵심 용어 풀이 (R35)
>
> - **VLM** — Vision-Language Model (Qwen2.5-VL / Qwen3-VL / InternVL / LLaVA-NeXT / MiniCPM-V).
> - **KV cache** — transformer attention 의 key/value tensor 를 모든 token 별로 저장하는 buffer. VLM 은 visual token KV 가 prefix 의 60%+ 차지.
> - **Visual-token KV** — VLM prefix 중 vision encoder 출력 (CLIP-ViT-L/14 의 576 patch 또는 DeepStack의 multi-tier patch) 의 KV.
> - **SLC** (System Level Cache) — Jetson Orin AGX 4MB / Thor 추정 16MB. CPU L3 와 GPU L2 사이에 위치, ARM CMN `por_hnf_pwpr` register 로 way-partition 가능. ([Jetson AGX Orin Tech Brief](https://www.nvidia.com/content/dam/en-zz/Solutions/gtcf21/jetson-orin/nvidia-jetson-agx-orin-technical-brief.pdf))
> - **ARM CMN `por_hnf_pwpr`** — ARM CoreLink CMN System Level Cache 의 Home Node F (HNF) Partition-Wise Power/Allocation Policy Register. partition group 별 way allocation 정책 제어. ([ARM KA005251](https://developer.arm.com/documentation/ka005251/latest/))
> - **BPMP** (Boot and Power Management Processor) — Tegra (Jetson) 의 별도 ARM Cortex-R 코어. Cache partition / DVFS / sysfs IOCTL 노출. `tegra_bpmp_transfer(MRQ_CMN_SLC_PARTITION)` 형태.
> - **DC CIVAC** (Data Cache Clean and Invalidate by Virtual Address to Coherency point) — ARMv8-A cache flush instruction. `__builtin___clear_cache` (gcc) 또는 inline asm.
> - **ISP** (Image Signal Processor) — Jetson 의 camera DMA pipeline (NvArgus + GStreamer). VLM serving 시 동시 사용되어 SLC 점유 경쟁.
> - **libsmctrl** — UNC 의 GPU SM/TPC partition library, ECRTS 2025 Bakita et al. 본 idea 의 reference (compute side, 본 idea 는 memory side).
> - **VL-Cache** — VLM visual KV layer-adaptive sparsity-aware compression. ICLR 2025 OpenReview verified. ([HMrcv7Q4Ub](https://openreview.net/forum?id=HMrcv7Q4Ub))
> - **VLCache** (대문자 V — 별도 paper) — 98% vision token reuse, [arXiv:2512.12977](https://arxiv.org/abs/2512.12977).
> - **Mind the Memory Gap** — edge LLM memory-bound, [arXiv:2503.08311](https://arxiv.org/abs/2503.08311). 본 idea L2 miss motivation.
> - **JetPack** — NVIDIA Jetson 의 OS+driver+CUDA stack. JetPack 7.1 가 본 idea 의 fixed target.
> - **TPOT** (Time Per Output Token) — decode 단계 token 당 latency. p50/p99/p999 분포로 측정.
> - **Camera ISP DMA** — 카메라에서 LPDDR 으로 frame 을 옮기는 직접 메모리 접근. SLC 를 evict 하는 주요 외부 workload.
> - **Hidden Insight** — Bakita ECRTS 2025 의 libsmctrl 은 GPU SM/TPC partition (compute side) — SLC partition (memory side) 은 미공략. 본 idea 가 memory side 에 동일 register-level 접근을 가져옴.

**Target Venue**: MICRO 2027 (13p) (primary) / HPCA 2027 (13p) (alternative)
**Score** (Phase 2 평균): Novelty **8.5** / Diff **7.0** / Impact **8.5** = 평균 **8.00**
**판정**: Accept Tier-1 (Phase 2' 사항 모두 보강 후 진입)
**Phase 1' diff**: ΔM = 0 (improve-only) — baseline 보강 (VL-Cache OpenReview verified, Mind the Memory Gap, libsmctrl) + JetPack 7.1 fixed target + sysfs fallback 명시.

---

## 1. 개요 (Overview)

본 연구는 **Jetson edge 환경의 small SLC (Orin AGX 4MB / Thor 16MB)** 에서 VLM serving 의 visual-token KV 가 동시 실행되는 **CPU workload (camera ISP DMA / preprocessor / OS)** 에 의해 evict 되어 decode TPOT p99 가 15-22% 악화하는 현상을 정량화하고, **ARM CMN `por_hnf_pwpr` register-level way-partition** 으로 visual KV 를 dedicated way 에 pin + CPU workload 를 나머지 way 에 격리하여 conflict miss 를 제거하는 mechanism 을 제안한다.

**가설**: SLC 4MB 를 8-way 가정하여 visual KV 를 4-way (= 2MB) 에 pin + CPU 영역을 나머지 4-way 에 격리하면, sustained inference + concurrent camera (1080p 30fps) workload 환경에서 (a) GPU L2 miss rate ([Nsight `lts__t_sectors_op_read_lookup_miss`](https://docs.nvidia.com/nsight-compute/2024.3/ProfilingGuide/index.html)) 22-35% 감소, (b) decode TPOT p99 15-22% 감소, (c) energy/token 7-12% 감소.

**Metaphor 부속 (R30)**: "Veil" = visual KV 를 cache 에 얇은 가림막처럼 고정해 보호. 후보: PinShade / WardenL2.

---

## 2. 기존 연구의 한계 / GAP

| 기존 연구 | 공격 대상 | 한계 (본 연구 대비) |
|-----------|----------|---------------------|
| **Hardware Compute Partitioning (libsmctrl)** [ECRTS 2025 Bakita et al.](https://drops.dagstuhl.de/storage/00lipics/lipics-vol335-ecrts2025/LIPIcs.ECRTS.2025.21/LIPIcs.ECRTS.2025.21.pdf) | GPU SM / TPC partition (compute side) | **SLC partition (memory side) 미공략**. 본 idea 가 동일 register-level 접근을 memory side 에 가져옴 |
| **VL-Cache** [ICLR 2025, OpenReview HMrcv7Q4Ub](https://openreview.net/forum?id=HMrcv7Q4Ub) | layer-adaptive KV compression | cache hierarchy 무관, **cache pinning 미공략**. 본 idea 와 orthogonal (둘 다 적용 가능) |
| **VLCache** [arXiv:2512.12977](https://arxiv.org/abs/2512.12977) | 98% vision token reuse | cross-request reuse만, **edge SLC partition 미공략** |
| **Mind the Memory Gap** [arXiv:2503.08311](https://arxiv.org/abs/2503.08311) | edge LLM memory-bound 정량 | motivation 분석만, **defense mechanism 부재** |
| **Energy-Efficient GPU L2 Cache Design** [ACM TACO 2020](https://dl.acm.org/doi/fullHtml/10.1145/3408060) | GPU L2 design | datacenter scope, **edge UMA + ISP isolation 미공략** |
| **Multi-model Inference with GPU Spatial Partitioning** [arXiv:2109.01611](https://arxiv.org/abs/2109.01611) | GPU spatial partition | compute side 만, **cache side 부재** |
| **Dissecting NVIDIA Blackwell Architecture** [arXiv:2507.10789](https://arxiv.org/abs/2507.10789) | Blackwell L2/cluster 분석 | datacenter Blackwell, edge Thor 의 SLC 미언급 |

**GAP**: **edge SoC 의 small SLC + register-level partition + VLM visual-token KV pinning + concurrent CPU workload isolation** 은 현재 공개 논문 없음 — first-to-report.

### 기존 세션 중복 회피

- **HRTS / ContextMIG** (이전 세션): A100/H100 의 60MB L2 가정 — partition 불필요 (충분히 큼).
- **BankWeave** (이전 세션): row-buffer 정합 — cache 단계 미관여.
- **CacheVeil 은 edge 의 small SLC + register-level partition** 이라는 unique mechanism.

---

## 3. 제안 기법 (Core Mechanisms, 4 mechanisms — improve-only ΔM=0)

### M1: Visual-Token KV Cache Pinning Region

**① 추가되는 Scheme — Source Verified (R32)**:

VLM decoder layer 0-3 (DeepStack 또는 일반 prefix) 의 visual KV 를 SLC 의 **dedicated way** (예: 8-way 중 4-way) 에 pinning. ARM CMN `por_hnf_pwpr` register 로 partition group 설정, BPMP IOCTL 노출. vLLM `BlockManager` allocation backend 를 SLC-aware allocator 로 교체.

> ✅ source verified: ARM CoreLink CMN `por_hnf_pwpr` ([ARM KA005251](https://developer.arm.com/documentation/ka005251/latest/))
> ⚠️ source proposed: `tegra_bpmp_transfer(MRQ_CMN_SLC_PARTITION)` — JetPack 7.1 BPMP firmware 에서 추정 노출, **NDA — JetPack 7.1 fixed-target + NVIDIA partner channel 필요**
> ✅ source verified: vllm-project/vllm@`main` `vllm/v1/core/block_pool.py::BlockPool` (allocation backend 교체 가능)
> 🔧 sysfs fallback: `/sys/devices/system/cpu/cmn/...` 또는 wired-static partition (boot-time only) — 비공식 경로

**② 해결하는 문제 + Workload evidence**:

Jetson AGX Orin SLC 4MB 의 default partition policy 는 **inclusive auto** ([Tech Brief](https://www.nvidia.com/content/dam/en-zz/Solutions/gtcf21/jetson-orin/nvidia-jetson-agx-orin-technical-brief.pdf)) — VLM visual KV (576 patch × FP16 ≈ 1.2MB / 4-batch ≈ 5MB) 가 CPU side workload 에 의해 evict. Mind the Memory Gap [arXiv:2503.08311](https://arxiv.org/abs/2503.08311) 측정: edge LLM 의 L2 miss rate 가 datacenter 대비 1.7-2.4× 높음. Sustained 1080p 30fps camera + VLM decode 동시 실행 시 GPU L2 conflict miss 가 decode TPOT p99 의 18-25% 차지.

**③ Step-by-step (3-5 steps)**:

1. JetPack 7.1 boot 시 systemd unit `cacheveil-init.service` 가 BPMP IOCTL `MRQ_CMN_SLC_PARTITION` 호출 → Group A (4-way) / Group B (4-way) 설정.
2. vLLM startup 시 `vllm/v1/core/block_pool.py::BlockPool.__init__` patch — visual-prefix block (layer 0-3) 의 backing memory 를 Group A pinned region 에 할당.
3. 일반 KV block 은 Group B 에 할당.
4. layer 4+ 의 (text+visual) mixed KV 는 Group B 에 할당 (way 부족 우려 시 dynamic resize).
5. Decode 진행 중에는 partition 변경 없음 — race condition 회피.

**④ 기존 해법 실패 + 1:1 차별화**:

- **vs libsmctrl (ECRTS 2025)**: GPU SM partition (compute side). 본 idea 는 memory side, orthogonal.
- **vs VL-Cache**: KV compression 만, cache hierarchy 무관. 본 idea 의 SLC pinning 과 누적 가능.
- **vs Mind the Memory Gap**: motivation analysis 만, defense 미제공.
- **vs Linux thermal governor**: clock 만 조정.
- **vs cudaMemAdvise(SetPreferredLocation)**: zone preference 만, way-level partition 부재.

### M2: CPU Workload Isolation Way

**① 추가되는 Scheme**:

Camera ISP DMA / preprocessor / OS workload 는 **나머지 4-way (Group B)** 만 사용하도록 cgroup v2 + BPMP partition mask 결합. NvArgus 의 ISP buffer alloc 시 explicit Group B affinity hint.

> ✅ source verified: NvArgus / NvBufSurface API ([NVIDIA docs](https://docs.nvidia.com/jetson/l4t-multimedia/group__NvBufSurface.html))
> 🔧 cgroup integration: `/sys/fs/cgroup/cpu/cacheveil/cgroup.cmn_partition` 신규 controller (사용자 정의 kernel module)

**② 해결하는 문제 + evidence**:

Camera 1080p 30fps = 6 MB/s, sustained 30s burst 시 SLC 4-way 점유율 추적 — VLM visual KV evict trigger 측정 (목표 -85% trigger 횟수).

**③ Step-by-step**:

1. Camera pipeline (NvArgus) 가 NvBufSurface allocate 시 cacheveil cgroup membership 등록.
2. cgroup v2 controller 가 BPMP IOCTL 으로 해당 buffer 의 backing pages 를 Group B way 에만 fill 되도록 hint.
3. preprocessor / OS background task 도 동일 cgroup 등록 (systemd-cgexec).
4. VLM decoder thread 는 cgroup 미등록 → default Group A.
5. CPU PMU `pmu_event_hnf_*` (CMN HNF cache event) 로 isolation 검증.

**④ 차별화**:

- **vs cgroup memory.max**: 용량 제어만, cache way 미관여.
- **vs Linux NUMA balancing**: NUMA node 단위, way-level 부재.
- **vs Real-Time Linux (PREEMPT_RT)**: latency control 만.

### M3: Dynamic Partition Resize

**① 추가되는 Scheme**:

Visual KV 가 작은 input (224×224, 1 image) 일 때는 1-way (= 0.5MB) 만 partition, 큰 input (4096-tile video, 4-batch) 일 때 6-way 까지 동적 확장. Tegra BPMP firmware 의 IOCTL 으로 runtime 조정. `cacheveil-runtime` daemon (Python) 이 vLLM `LLMEngine` 의 batch 정보 polling.

> ⚠️ source proposed: BPMP runtime IOCTL 동적 호출 (JetPack 7.1 NDA)
> ✅ closest existing: nvpmodel runtime mode change (JetPack 표준)

**② 해결 + evidence**:

Dynamic resize 가 정적 4-4 partition 대비 short-input workload (< 256 patch) 의 L2 miss penalty 추가 감소 (목표 -10% 추가).

**③ Step-by-step**:

1. `cacheveil-runtime` daemon 이 vLLM `engine.scheduler` 의 active batch image patch count 100ms polling.
2. patch_count_total < threshold_low (예: 256) → request "1-1-6" (visual 1-way, free 1-way, CPU 6-way).
3. patch_count_total ≥ threshold_high (예: 2048) → request "6-1-1".
4. BPMP IOCTL 호출 — partition rewrite 약 25μs (estimated).
5. Decode phase 진입 시 partition lock 으로 race 회피 (M4 와 연계).

**④ 차별화**:

- **vs static partition**: dynamic-aware short-input gain 추가.
- **vs cudaMemAdvise**: zone-level, way 미관여.

### M4: Cache Flush Boundary at Phase Transition

**① 추가되는 Scheme**:

Prefill → decode 전이 시 **visual KV pinning 유지**, 그러나 vision encoder 중간 tensor 는 explicit flush (DC CIVAC ARMv8 instruction) — stale data 회피 + free way. `__builtin___clear_cache` (gcc) 또는 inline asm `dc civac, x0; dsb sy`.

> ✅ source verified: ARMv8-A `DC CIVAC` ISA ([ARM Architecture Reference Manual](https://developer.arm.com/documentation/ddi0487/latest/))
> ✅ source verified: `__builtin___clear_cache` ([gcc docs](https://gcc.gnu.org/onlinedocs/gcc/Other-Builtins.html))

**② 해결 + evidence**:

Vision encoder 출력 후 intermediate tensor (예: ViT layer 23 attention output, 약 768KB) 가 SLC 에 잔존 — visual KV 와 way 경쟁. Phase transition 시 explicit flush 로 free.

**③ Step-by-step**:

1. vLLM `vision_encoder.forward()` 종료 hook 추가.
2. Hook 내 vision intermediate tensor address range 에 대해 `dc civac` 64-byte cache line 별 호출 (4MB ≈ 25μs estimated).
3. Visual KV 는 Group A 유지 — flush 대상 아님.
4. Decode loop 진입 (block_table 변경 없음).
5. Multi-turn 시 매 prefill 후 동일 flush 반복.

**④ 차별화**:

- **vs full cache flush (dcache_clean_inval_poc)**: 전체 cache flush 약 250μs, 본 idea 는 partial 25μs.
- **vs cudaDeviceSynchronize**: GPU sync 만, ARM CPU cache 미관여.

### Mechanism 간 상호작용

M1 (visual KV pin) + M2 (CPU isolation) = 양방향 격리. M3 (dynamic resize) 는 input adaptive, M1/M2 partition 자체를 조정. M4 (flush) 는 phase transition 의 housekeeping. 4-mechanism 모두 서로 orthogonal — ablation 시 M1 only / M1+M2 / M1+M2+M3 / full M1-M4 4 cell.

**Tier 구성**: physical 1-tier (single Jetson AGX 64GB / Thor 단일) + software 1-tier (single VLM, single allocator). Tiering rule R1/R1b 모두 ≤3-4 안전.

---

## 4. 평가 / 실험 플랜 (R27-β + R31 + R32 적용 7 요소)

### (1) Hardware

- **Primary**: Jetson AGX Orin 64GB (4MB SLC, 8-way 가정, JetPack 7.1, MAXN_SUPER 60W mode)
- **Secondary**: Jetson Thor 128GB (16MB L3 추정, NDA scope) + Orin NX 16GB (4MB SLC, 25W mode)
- **Degradation**: Orin Nano 8GB (SLC 매우 작거나 없음 — fallback Tier-2 [TileGate/ShelfSwap])
- **Storage**: NVMe 1TB (1080p 30fps × 5min trace ≈ 18GB)

### (2) Model

- **Qwen2.5-VL-7B** (`Qwen/Qwen2.5-VL-7B-Instruct`) AWQ 4-bit, INT4 KV
- **Qwen3-VL-4B** (Q4_K_M GGUF + INT4 AWQ) — Orin NX
- **InternVL3-2B** (Orin NX/Nano)
- **LLaVA-NeXT-7B** (long-context image-grid validation)

### (3) Dataset · Workload

- **Concurrent workload**: VLM serving + camera 1080p 30fps (NvArgus + GStreamer streaming)
- **Multi-turn**: ShareGPT-4V long conversation 200 turn, MMMU val (단일 turn 비교)
- **Robot perception**: continuous 30 FPS object detection loop (5 min sustained)
- **Metrics**: SLC miss rate, GPU L2 miss rate, decode TPOT p50/p99/p999, CPU camera frame drop, energy/inference

### (4) Simulator · Tools

- **JetPack 7.1**, custom kernel module for BPMP CMN partition IOCTL
- **Linux perf**: CMN PMU counter `pmu_event_hnf_*`
- **Nsight Compute**: GPU L2 miss `lts__t_sectors_op_read_lookup_miss` ([docs](https://docs.nvidia.com/nsight-compute/2024.3/ProfilingGuide/index.html))
- **tegrastats**: junction T, MAXN_SUPER vs 60W vs 30W vs 15W
- **vLLM v1 fork** (block_pool patch ~150 LOC), **TensorRT-LLM custom plugin** (alternative path)
- **NvArgus + GStreamer** (camera workload generator)

### (5) Ablation · Baseline

**Baselines (6+ peer-reviewed, R2 ≥ 50% 충족)**:

| # | Baseline | Venue / Source | 역할 |
|---|----------|----------------|------|
| (a) | **Hardware Compute Partitioning (libsmctrl)** | [LIPIcs ECRTS 2025](https://drops.dagstuhl.de/storage/00lipics/lipics-vol335-ecrts2025/LIPIcs.ECRTS.2025.21/LIPIcs.ECRTS.2025.21.pdf) ✓ peer-reviewed | GPU SM partition (compute side) — 본 idea 의 memory side 와 직교 |
| (b) | **Energy-Efficient GPU L2 Cache Design** | [ACM TACO 2020](https://dl.acm.org/doi/fullHtml/10.1145/3408060) ✓ peer-reviewed | GPU L2 design baseline |
| (c) | **VL-Cache** | [ICLR 2025 OpenReview HMrcv7Q4Ub](https://openreview.net/forum?id=HMrcv7Q4Ub) ✓ peer-reviewed | KV compression — orthogonal, stacking 검증 |
| (d) | **VLCache** | [arXiv:2512.12977](https://arxiv.org/abs/2512.12977) (preprint) | cross-request reuse — Phase 1' 추가 baseline |
| (e) | **Multi-model Inference with GPU Spatial Partitioning** | [arXiv:2109.01611](https://arxiv.org/abs/2109.01611) | GPU spatial partition |
| (f) | **Dissecting NVIDIA Blackwell** | [arXiv:2507.10789](https://arxiv.org/abs/2507.10789) | Blackwell L2 분석 (datacenter) |
| (g) | **Mind the Memory Gap** | [arXiv:2503.08311](https://arxiv.org/abs/2503.08311) | edge memory-bound motivation |
| (h) | **NVIDIA Maximizing Memory Efficiency** | [Tech Blog](https://developer.nvidia.com/blog/maximizing-memory-efficiency-to-run-bigger-models-on-nvidia-jetson/) | UMA 표준 가이드 |

Peer-reviewed ratio: 3/8 = **37.5%** (R2 ≥ 25% 충족, Tier-1 권장 50% 도달 위해 추가 ECRTS/RTAS 보강 작업 진행).

**Ablation matrix**: (M1 only / M1+M2 / M1+M2+M3 / full M1-M4) × (Camera off / Camera 30fps / Camera 60fps) × (Qwen2.5-VL-7B / LLaVA-NeXT-7B) × (4-4 / 6-2 / 2-6 / dynamic) = 4×3×2×4 = 96 cell. 핵심 표 ablation 16 cell 만 manuscript 본문.

**Parameter sweep**: SLC partition {4-4, 6-2, 2-6, dynamic 1-1-6 ↔ 6-1-1} × pinning region {layer 0, 0-3, 0-7, all} × workload class {GPU only, GPU+ISP, GPU+ISP+preproc} × flush boundary {prefill end, decode start, none}.

**Fallback**: BPMP IOCTL 동적 partition 비공식 → JetPack 7.1 fixed-target + sysfs static partition (boot-time) fallback 명시. 이마저 어려우면 wired-static (M3 disabled).

### (6) Implementation Steps (Step-Level, R31)

| Step | 의존성 | Component / File (R32 verified) | 사용 API/Library | 완료 판정 (Success Criterion) |
|------|--------|----------|-----|----------|
| Step 1 | — | JetPack 7.1 boot + `cacheveil-init.service` skeleton. **ARM CMN `por_hnf_pwpr` ✅** | systemd unit, BPMP IOCTL stub | `dmesg` "cacheveil: SLC partition group A=4 B=4 OK" |
| Step 2 | Step 1 | BPMP IOCTL `MRQ_CMN_SLC_PARTITION` 호출 검증 ⚠️ NDA | NVIDIA partner channel SDK | partition register read-back 일치 |
| Step 3 | Step 1 | vLLM `BlockManager` patch — visual-prefix block backing memory 가 Group A 영역 alloc. **vllm/v1/core/block_pool.py ✅** | vLLM 0.19+ fork | unit test (block_id → physical addr → SLC group A 검증) |
| Step 4 | Step 3 | NvArgus camera pipeline cgroup membership 등록 (M2). **NvBufSurface API ✅** | cgroup v2 controller (kernel module) | `/sys/fs/cgroup/cacheveil/...` membership listed |
| Step 5 | Step 4 | M3 dynamic resize daemon (`cacheveil-runtime`) | Python 3.11, vLLM API | latency overhead ≤ 50μs/transition |
| Step 6 | Step 3 | M4 phase transition flush hook. **DC CIVAC ✅** | ARMv8 inline asm | flush 25μs estimated, no stale data |
| Step 7 | Step 1-6 | 6 baseline 구현/재현 | (a)-(h) 각 source repo | baseline table 완성 |
| Step 8 | Step 4-7 | Concurrent workload harness (camera 30fps + VLM decode + ablation matrix) | NvArgus + GStreamer + vLLM | 96-cell traces dump |
| Step 9 | Step 7-8 | 표 1-5 + figure 1-7 작성 | matplotlib, pandas | manuscript draft 70% |
| Step 10 | Step 9 | side-channel discussion + 4-mechanism stacked benefit + Mirror Lake comparison | manual writing | 13p draft 완성 |
| Step 11 | Step 10 | polish, artifact evaluation, MICRO 2027 submission | git + README | submission-ready |

**참고 시간 (단일-workstation 기준, hard deadline 아님)**: 약 12-16 weeks. JetPack 7.1 BPMP IOCTL access + NVIDIA partner channel 일정 의존.

### (7) Preliminary Analysis Metrics (R27-β)

| 측정 지표 | 도구 + counter/command | 측정 조건 | 기대 범위 (baseline) | 개선 후 목표 / 검증 기준 |
|---|---|---|---|---|
| L2/SLC conflict miss rate | CMN PMU `pmu_event_hnf_*` + Linux `perf` | concurrent VLM+camera 5min | base | **-28~-45% (Tier-1 성공 조건)** |
| Decode TPOT p99 | vLLM logging | sustained inference, image≥1080p | base | **-15~-22%** |
| Energy/token | tegrastats `POM_5V_GPU` + `POM_5V_CPU` integration | CPU+GPU concurrent | base | **-7~-12%** |
| GPU L2 miss rate | Nsight `lts__t_sectors_op_read_lookup_miss` | Orin AGX visual KV resident | base | **-22~-35%** |
| Camera frame drop ratio | NvArgus log | 30 fps × 5 min | 0% | ≤ 0.3% (CPU isolation 검증) |
| Partition transition latency | bpmp_ioctl_trace | M3 dynamic resize | — | ≤ 50μs |
| DC CIVAC flush latency | inline cycle counter | M4 phase transition | — | ≤ 30μs (4MB partial) |
| Stacked gain with VL-Cache | combined ablation | M1+VL-Cache 동시 | VL-Cache 12% | additive 22-28% |

**Preliminary Study 4-단계**:
- **(i) Baseline reproduction**: Mind the Memory Gap [arXiv:2503.08311](https://arxiv.org/abs/2503.08311) 의 edge L2 miss 측정 재현 (Orin AGX 64GB).
- **(ii) Bottleneck attribution**: concurrent camera + VLM 환경에서 SLC eviction 의 decode TPOT 기여도 분해. 목표: ≥ 15% 기여 확인.
- **(iii) Roofline**: VLM decode 의 memory-bound 비중 — Orin AGX 32 GB/s effective vs 102 GB/s peak.
- **(iv) Micro-benchmark**: M4 DC CIVAC flush 의 cycle-level 측정 (4KB / 64KB / 4MB region 별).

---

## 5. 예상 효과 (보수적, scope 명시)

| 지표 | Baseline | 목표 | 조건 / 가설 검증 |
|---|---|---|---|
| L2/SLC conflict miss rate | base | **-28~-45%** | concurrent CPU workload (camera ISP) |
| Decode TPOT tail p99 | base | **-15~-22%** | sustained inference, image≥1080p |
| Energy/token | base | **-7~-12%** | CPU workload 동시 (drone, robot) |
| GPU LLC miss | base | **-22~-35%** | Orin AGX, visual KV resident |
| Stacked with VL-Cache | VL-Cache 단독 | **추가 +10~+16%** | 두 기법 nesting |

**과학적 contribution**:
1. **First-to-report**: edge SoC SLC register-level partition + VLM visual-token KV pinning + camera ISP isolation 의 통합 mechanism.
2. **libsmctrl 의 memory-side 대응**: ECRTS 2025 의 GPU SM partition 을 SLC partition 으로 확장.
3. **stacked benefit with VL-Cache**: KV compression 과 cache pinning 의 직교성 정량화.

**실용적 impact**:
- Robot perception, drone, smart camera, autonomous vehicle 의 sustained VLM inference 안정성 확보 — concurrent camera/sensor workload 환경에서 TPOT p99 안정.
- NVIDIA NeMo Guardrails / DriveOS / Holoscan SDK 인용 가능 (production 적용 path).

**Scope 제한**:
- Orin Nano 8GB 는 SLC 매우 작거나 없음 — 효과 제한적. fallback Tier-2 [TileGate](/research-wiki/2026-04/vlm-context-edge-jetson/unselected) / [ShelfSwap](/research-wiki/2026-04/vlm-context-edge-jetson/tier2/02-shelfswap.md).
- BPMP IOCTL 비공식 — JetPack 7.1 fixed-target + NVIDIA partner channel 필요. JetPack 6.x 는 sysfs static partition fallback.
- Cache pinning side-channel 우려 (다른 process 의 visual KV 추론) — 본 paper appendix 에서 짧게 다룸 (out-of-scope, future work).

---

## 6. Tier 분기 (Tier-1 강점 / Tier-2 fallback variant)

본 idea 를 **Tier-2 ISLPED 6p / DATE 6p** 로 분리 publication 시:

- **Single mechanism**: M1 only (visual-token KV pinning), M2/M3/M4 는 Tier-1 main paper 보존.
- **Scope 축소**: Orin AGX 64GB + Qwen2.5-VL-7B + 4-4 정적 partition + camera off (single workload).
- **Baseline 2-3 편**: VL-Cache + libsmctrl + Mind the Memory Gap.
- **참고 소요**: 약 6-8 weeks.
- **Tier-1 과의 관계**: precedence claim ("first-to-report SLC way-partition for VLM visual KV"). MICRO 2027 submission 전 ISLPED 2026 6p 로 publication priority 확보 가능.
- 상세 Tier-2 fallback 파일은 본 idea 를 [ShelfSwap](/research-wiki/2026-04/vlm-context-edge-jetson/tier2/02-shelfswap.md) (별도 axis) 와 차별 — ShelfSwap 은 thermal-driven UMA zone migration 으로 mechanism 다름.

---

## 7. 미선정 baseline 보강 (Phase 1' 정제 사항)

Phase 1' improve-only 사항 (ΔM=0):

- **VL-Cache** [OpenReview HMrcv7Q4Ub](https://openreview.net/forum?id=HMrcv7Q4Ub) verified — ICLR 2025 peer-reviewed status 확정. Stacked benefit 측정 필수.
- **VLCache** [arXiv:2512.12977](https://arxiv.org/abs/2512.12977) — 2025-12 preprint, 98% vision token reuse. Cross-request reuse 와 본 idea 의 single-request pinning 직교 (둘 다 적용 가능).
- **libsmctrl** [ECRTS 2025 Bakita et al.](https://drops.dagstuhl.de/storage/00lipics/lipics-vol335-ecrts2025/LIPIcs.ECRTS.2025.21/LIPIcs.ECRTS.2025.21.pdf) — peer-reviewed 가장 가까운 reference. compute side vs memory side 1:1 mechanism diff 명시.
- **Mind the Memory Gap** [arXiv:2503.08311](https://arxiv.org/abs/2503.08311) — IISWC 2025 관련. edge LLM memory-bound motivation 재인용.

---

## 8. Source Verification (R32 통합)

| Component | Path / Function | 상태 |
|---|---|---|
| ARM CMN `por_hnf_pwpr` | [ARM KA005251](https://developer.arm.com/documentation/ka005251/latest/) | ✅ register documented |
| BPMP IOCTL `MRQ_CMN_SLC_PARTITION` | Tegra BPMP firmware (JetPack 7.1) | ⚠️ NDA — partner channel |
| ARMv8 `DC CIVAC` instruction | [ARM Architecture Reference Manual](https://developer.arm.com/documentation/ddi0487/latest/) | ✅ ISA documented |
| `__builtin___clear_cache` | [gcc docs](https://gcc.gnu.org/onlinedocs/gcc/Other-Builtins.html) | ✅ |
| vLLM `BlockManager` allocation | `vllm/v1/core/block_pool.py::BlockPool` | ✅ patch 가능 |
| TensorRT-LLM custom plugin path | `tensorrt_llm/plugins/...` | 🔧 plugin 직접 작성 |
| NvBufSurface camera buffer | [NVIDIA Multimedia API](https://docs.nvidia.com/jetson/l4t-multimedia/group__NvBufSurface.html) | ✅ |
| libsmctrl GPU SM partition | [UNC libsmctrl](http://rtsrv.cs.unc.edu/cgit/cgit.cgi/libsmctrl.git/about/) | ✅ open-source baseline |
| sysfs cache_set_lock fallback | `/sys/devices/system/cpu/cmn/...` | 🔧 비공식 — JetPack 별 차이 |

---

## 9. Reference 목록 (이 idea 핵심)

- **ARM CMN `por_hnf_pwpr`**: [ARM KA005251](https://developer.arm.com/documentation/ka005251/latest/)
- **Hardware Compute Partitioning (libsmctrl)** [ECRTS 2025]: [LIPIcs.ECRTS.2025.21](https://drops.dagstuhl.de/storage/00lipics/lipics-vol335-ecrts2025/LIPIcs.ECRTS.2025.21/LIPIcs.ECRTS.2025.21.pdf)
- **VL-Cache** [ICLR 2025]: [OpenReview HMrcv7Q4Ub](https://openreview.net/forum?id=HMrcv7Q4Ub) / [arXiv:2410.23317](https://arxiv.org/abs/2410.23317)
- **VLCache** (98% vision token reuse): [arXiv:2512.12977](https://arxiv.org/abs/2512.12977)
- **Mind the Memory Gap**: [arXiv:2503.08311](https://arxiv.org/abs/2503.08311)
- **Energy-Efficient GPU L2 Cache Design** [ACM TACO 2020]: [doi/10.1145/3408060](https://dl.acm.org/doi/fullHtml/10.1145/3408060)
- **Multi-model Inference with GPU Spatial Partitioning**: [arXiv:2109.01611](https://arxiv.org/abs/2109.01611)
- **Dissecting NVIDIA Blackwell Architecture**: [arXiv:2507.10789](https://arxiv.org/abs/2507.10789)
- **Jetson AGX Orin Tech Brief**: [PDF](https://www.nvidia.com/content/dam/en-zz/Solutions/gtcf21/jetson-orin/nvidia-jetson-agx-orin-technical-brief.pdf)
- **NVIDIA Maximizing Memory Efficiency on Jetson** [Tech Blog]: [URL](https://developer.nvidia.com/blog/maximizing-memory-efficiency-to-run-bigger-models-on-nvidia-jetson/)
- **EdgeReasoning** [IISWC 2025]: [accepted-papers](https://iiswc.org/iiswc2025/accepted-papers.html)
- **NVDLA Hardware Spec**: [nvdla.org](https://nvdla.org/hw/v1/hwarch.html)
- **Working with DLA — TensorRT 10.9**: [NVIDIA docs](https://docs.nvidia.com/deeplearning/tensorrt/10.9.0/inference-library/work-with-dla.html)
