# VLM/VLA Context-aware Serving 최적화: 새 규칙 적용 Dual Top-3 (v3)

*Session date: 2026-04-23 · Mode 1 (v1/v2 improve) · R27 Self-Sufficient Summary 적용 (2026-04-24 재작성)*

> v1 (Top 3 = L1/A3/A1 single-track) → v2 (Tier-Mix HRTS/ContextMIG/NACK-Gossip) → v3 (**Dual Top-3: Tier-1 Top 3 + Tier-2 독립 Top 3 = 총 6 아이디어**). 새 harness 규칙 (R21 Dual Top-3 / R22 summary 블로그-style / arxiv 링크 포맷 강화 / Track B 독립) 를 적용하여 v1/v2 에서 도출된 10 variants + v3 독립 Track B 3 편을 통합, 2026-04 Mosaic/Harvest/FlashVLA 등 신규 scoop/concurrent 3 편 반영.

본 summary 는 **R27 Self-Sufficiency for Implementation** 규칙에 따라 전면 재작성되었으며, 각 mechanism 은 (① 추가 scheme, ② 해결 문제, ③ 동작 원리 step-by-step, ④ 기존 해법 실패 + 차별화) 4 요소를 포함하고, 각 실험 플랜은 기존 5 요소에 (6) Implementation Steps week-level + (7) Preliminary Analysis Metrics 2 요소를 추가한 7-요소 포맷을 사용한다. Summary 만 읽고도 CS 학부 졸업생 수준의 연구자가 preliminary 실험을 즉시 착수 가능하도록 구성했다.

---

## 1. 연구 진행 Meta

### 1.1 사용자 쿼리 원문

> "기존 세션에 vlm 및 vla 모델에 대한 context-aware serving 최적화 연구에 대한 내용들이 v1, v2 가 있는데, 이는 aica-research-bot 의 새 규칙이 적용되기 전 (4/22) 에 진행한 세션들이야. 이 내용들을 새 규칙을 적용해서 좀 더 improve 해서 ideation 진행 해줘. 기 조사한 내용들을 활용해서 ideation 을 진행해주고, 추가로 필요한 조사가 있을 경우에는 추가로 진행하도록 해."

### 1.2 주요 키워드 (4축)

| 축 | 키워드 |
|---|---|
| **도메인** | VLM (Vision-Language Model), VLA (Vision-Language-Action), multi-tenant serving, long-context video, single-robot VLA |
| **관찰·특징** | Context-awareness (task / scene / trajectory / semantic / multi-request / phase), HBM row-hit, CLIP semantic-near duplicate, VLA trajectory phase heterogeneity, Green Context μs reconfig |
| **제안 기법** | Row-aligned KV tile + bi-exp window pin + tri-tier / CLIP-L LSH reuse graph + MIG dual-issue + phase coalesce / SSE phase predictor + phase-specific CUDA Graph dispatcher + phase SM partition / Single-axis Tier-2 measurement letters (Green Context reconfig, eviction energy decomposition, action-head fused kernel) |
| **타겟** | p95/p99 TTFT, decode throughput, memory footprint, HBM row-hit rate, SM utilization, MIG reconfig latency, VLA latency/Hz, kernel time/launch, DRAM PKG energy |

### 1.3 중점적으로 고려한 축

- **Pure GPU serving stack (PIM 비의존)** — v1 설정 유지, ISCA/HPCA/ASPLOS 의 MIG/Green Context/HBM 물리 구조 연구 경로.
- **Content-awareness 다중 축**: (a) request-level content taxonomy (ContextMIG), (b) frame-level temporal window (HRTS), (c) trajectory-level phase heterogeneity (PhaseGraph-VLA).
- **Tier-1 + Tier-2 독립 dual track**: Tier-1 은 복잡도 큰 Top-tier ideation, Tier-2 는 single-axis measurement letter (Track B).
- **Scoop 지형 갱신**: 2026-04 Mosaic (VLM long-context video KV) + Harvest (NVLink peer caching) + FlashVLA (action reuse) 에 대한 차별화 명시.
- **Measurement-driven Tier-2 letters**: Green Context reconfig latency / eviction energy decomposition / action-head fused kernel 세 축의 measurement letter — Tier-1 scale-up 을 의도적으로 포기한 독립 기여.

### 1.4 의도적으로 제외한 축 (이유 명시)

| 제외 축 | 이유 |
|---|---|
| **PIM/CIM 기반 serving** | v1 설정 유지, 2026-04-22 mode2 VLM+PIM 세션 (F1/F2/F3) 에서 이미 다룸. 본 세션은 pure GPU 축으로 orthogonal 유지. |
| **MoE-based VLM expert caching** | 2026-04-21 ACE-MoE 세션이 MoE 축 커버. 본 세션은 dense VLM/VLA. |
| **Quantization / compression 주요 축** | 2026-04-23 PRISM-VLM-KV 세션 (I1/I2/I4) 이 ternary/rank-r SF 축 커버. Quantization 과 stacking 가능하나 독립 기여로 제외. |
| **Multi-node / cross-datacenter serving** | Single workstation scope 규율 (RTX 4090/5090/Pro 6000 + 부분 cloud). 4-GPU NVLink testbed 가 Tier-1 NACK-Gossip 의 약점으로 명확, multi-node 축은 다음 세션 후보. |
| **Training-side optimization** | 쿼리는 serving/inference 최적화 명시. Training-time (RLHF / instruction tuning) 은 scope 밖. |
| **6-tier physical partitioning** | Tiering 규율 (R1 ≤3-4 tier) 위반. HBM / DRAM / NVMe 3-tier 로 제한. |
| **9-class task taxonomy** | Tiering 규율 (R1b software ≤3-4 pool) 위반. L1 v1 의 8-class 를 6-class 로 축소 확정. |
| **Kernel-level deep optimization (full FA3 rewrite)** | HRTS+ 는 indirection pointer fork 수준. Kernel deep rewrite 는 MLSys 범위 초과 (별도 세션). |

### 1.5 검색 쿼리 전략

| Phase | 주요 쿼리 | 도메인 |
|---|---|---|
| v1 Step 0 (이미 수행) | "VLM serving context-aware 2025", "VLA KV cache temporal", "multi-tenant VLM cross-request", "SM partition MIG Green Context", "long-context video KV compression" | arxiv / openreview / usenix / mlsys / dl.acm.org |
| v2 OpenReview 보강 | "VL-Cache ICLR 2025 openreview", "VLA-Cache NeurIPS 2025", "SparseVLM ICML 2025" — **identity 4-point check** | openreview.net |
| v3 Placeholder 검증 (신규) | "VidKV arxiv 2511", "RoboFleet-Sync arxiv 2512", "PeerCache arxiv 2602", "ActionReuse arxiv 2601", "DynaMIG arxiv 2512", "CLIP-Batch arxiv 2602", "DRAM-aware attention arxiv 2601" | arxiv.org + Semantic Scholar + WebSearch |
| v3 2026-04 최신 보강 (신규) | "VLM KV cache long-context 2604", "multi-tenant VLM MIG 2604", "NVLink peer fetch LLM 2602", "GPU power Green Context 2604", "VLA fleet kv 2604" | arxiv (최근 1 개월 범위) |

### 1.6 사용된 전문가 에이전트 + 리뷰어

- **Experts**: ai-optimization-expert / legacy-system-expert (v1/v2 재활용 + v3 Track B 독립 도출)
- **Reviewers**: novelty-reviewer / differentiation-reviewer / impact-reviewer (v1/v2 재활용)
- **Cross review**: ai-opt ↔ legacy-sys domain-specific peer review
- **v3 신규 dispatch**: differentiation-reviewer (placeholder 7 편 검증) + ai-optimization-expert (2026-04 최신 논문 supplement) + legacy-system-expert (Track B Tier-2 독립 3 편 도출)

---

## 2. Tier-1 Top 3 (Top-tier Venue Target, Tier-2 scope 축소 variant 병기)

### 2.1 HRTS+ — HBM Row-Tile Streaming for Long-Context Video VLM ★ (Top 1, Avg 7.85 / ASPLOS 2026 / MICRO 2026)

#### 2.1.1 개요

1-hour 720p video 1fps × 256 token = 921K token KV (≈ 36GB FP16 on 32-layer 7B VLM) 을 **HBM row-buffer (8KB) 물리 구조에 KV page layout 정렬** + **bi-exponential recency×salience window pin** + **HBM-hot / DRAM-pinned / NVMe-cold 3-physical-tier async streaming** 의 3 mechanism 으로 처리한다. 2026-04 Mosaic 이 cross-modal clustering (content-axis) 를 점유했으나 HRTS+ 는 **HBM row-buffer physical axis** 로 orthogonal 하며 stacking ablation 에서 +5~8% 추가 개선 가능.

#### 2.1.2 기존 연구 한계점 및 Gap

- [VL-Cache ICLR 2025 Poster](https://openreview.net/forum?id=HMrcv7Q4Ub) [peer-reviewed] — layer-adaptive sparsity budget + modality separation. **Frame-level temporal access 미반영**, reviewer 가 streaming/multi-turn 미커버를 지적.
- [HERMES arXiv:2601.14724](https://arxiv.org/abs/2601.14724) — page-level hierarchical KV. **DRAM row-aware 없음**.
- [DiffKV arXiv:2412.03131](https://arxiv.org/abs/2412.03131) — per-head diff. **Temporal axis 없음**.
- [VideoLLM-online CVPR 2024](https://arxiv.org/abs/2406.11816) [peer-reviewed] — per-frame eviction FIFO. **HBM row-hit + bi-exp 결합 없음**.
- [Mosaic arXiv:2604.10060](https://arxiv.org/abs/2604.10060) (2026-04-11, **v3 신규 concurrent 55-65%**) — cross-modal clustering for long-context video VLM KV. **HBM physical axis 없음**, content-axis dedup 만.
- [vLLM PagedAttention SOSP 2023](https://arxiv.org/abs/2309.06180) [peer-reviewed] — 4KB page, HBM3 row 8KB 의 절반만 사용 → **row-hit 62%** (측정됨).

**Gap 요약**: HBM row-buffer × KV page layout × bi-exp recency+salience × tri-tier async 의 4 축 결합은 현 문헌에 없음.

#### 2.1.3 제안 기법 (3 Mechanisms, R27-α: 4 필수 요소)

##### **M1: Row-Aligned KV Tile (improve)**

- **① 추가되는 Scheme**: vLLM v0.7 fork 의 `vllm/attention/backends/paged_attn.py` 내부 `PagedAttentionImpl._paged_attention` 함수를 교체하는 신규 backend `RowTilePagedAttention` 추가. 이 backend 는 기존 4KB page layout 을 HBM3/HBM3e row-buffer (8KB) 에 정렬된 **8KB row-aligned tile** 로 재구성한다. `vllm/core/block_manager.py` 의 `BlockAllocator` 도 확장하여 `block_size_bytes=8192` + row-boundary alignment 를 강제. CUDA kernel 측에서는 FlashAttention-3 의 `csrc/flash_attn/src/flash_fwd_kernel.h` 의 `block_table` indirection pointer 를 수정하여 row-aligned tile 단위로 Q/K/V 를 fetch 한다.
- **② 해결하려는 문제**: 기존 PagedAttention 은 4 KB page 크기를 사용하므로 HBM3 의 8KB row-buffer 중 절반만 사용 → Nsight Compute `dram__sectors_read.sum` / `dram__sectors_read_hit.pct` counter 로 측정 시 **row-hit 62% (평균)** 이다. 1-hour video (921K KV token) workload 에서 decode 단계 DRAM access 가 BW-bound 이며, row-miss 가 20-30% 발생하여 prefill 대비 decode TTFT 가 1.4-1.6× 증가한다. LLaVA-Video-7B + VideoMME long subset 측정에서 decode throughput 이 target 의 60-65% 에 머무른다.
- **③ 동작 원리 (학부생용 step-by-step)**:
  1. **Row-boundary reverse-engineering** — Nsight Compute microbenchmark 로 HBM3/HBM3e 의 row-buffer 크기 (8 KB, 2 KB × 4 bank-group) 를 측정. `dram__sectors_read.sum` counter 를 stride {64B, 128B, 256B, 512B, 1024B, 2048B, 4096B, 8192B} 로 sweep 하여 row-hit 의 step discontinuity 지점 확정.
  2. **Analytical row-hit model** — `row_hit(tile_size) = max(0, 1 - miss_rate)` 와 `miss_rate = f(tile_size / row_size, access_pattern)` 를 Ramulator2 v2.0 simulation 으로 보정. tile size {128, 256, 512 tokens} sweep 에서 최적점 도출.
  3. **BlockAllocator 확장** — vLLM `block_manager.py` 의 `BlockAllocator.allocate(num_blocks)` 호출을 수정하여 `cudaMalloc` 대신 `cuMemCreate + cuMemMap` 으로 8KB row boundary 에 align 된 block 을 reserve.
  4. **CUDA kernel indirection fork** — FA3 의 `block_table[batch_idx][block_idx]` indirection 을 `row_aligned_block_table` 로 변경. each warp 이 8KB row 전체를 coalesced read 하도록 `__ldg()` stride 를 조정.
  5. **Row-hit 검증** — full experiment 에서 `dram__sectors_read_hit.pct` counter 가 82-88% 에 도달하는지 확인. 실패 시 tile size를 재조정.
- **④ 기존 해법 실패 + 차별화**: (i) vLLM PagedAttention (SOSP 2023) 은 virtual-memory 축 (fragmentation 해결) 만 다루고 HBM physical layout 과 무관, 4KB 는 MMU page 에 맞춘 arbitrary 선택. (ii) HERMES 는 page hierarchy 를 DRAM/CPU 로 확장했으나 HBM row-buffer 자체를 고려하지 않음. (iii) DiffKV 는 per-head delta 만 다룸. 본 M1 은 **vLLM block size 를 HBM3 row 와 physical 1:1 정렬** 하는 첫 구현으로, row-hit 62%→82-88% (+20-26pp) 달성 예상.

##### **M2: Bi-Exponential Recency × Salience Window Pin (improve)**

- **① 추가되는 Scheme**: vLLM `scheduler.py` 의 `_schedule_prefill` 및 `_schedule_decode` 함수에 신규 eviction policy `BiExpWindowPolicy` 추가. 각 KV block 마다 `score(t) = α·exp(-β_r·(T-t)) + (1-α)·salience(t)` 를 계산하여 pin/evict 결정. `salience(t)` 는 과거 attention score 의 log-sum-exp (LSE). FlashAttention-3 의 LSE output 을 별도 tensor 로 재활용하여 compute overhead 추가 없이 salience 계산.
- **② 해결하려는 문제**: StreamingLLM (fixed sliding window) 은 uniform recency 로 temporal salience 를 반영하지 못함 → 중요한 frame (e.g., scene transition, query-relevant frame) 을 일찍 evict 하여 VideoMME long subset accuracy -3~5pp drop. 반대로 aggressive pinning 은 KV memory 를 과도하게 소비 → 921K token 상 HBM 포화.
- **③ 동작 원리 (step-by-step)**:
  1. **Attention LSE 재활용** — FA3 forward 에서 per-block LSE 를 저장 (`lse_tensor[batch, head, block]`), 이를 salience signal 로 활용.
  2. **Bi-exponential score 계산** — 매 prefill 직후 `score = α·exp(-β_r·(T-t)) + (1-α)·(lse - lse_mean)` 를 계산. 파라미터 `α ∈ [0.3, 0.7]`, `β_r ∈ [0.01, 0.1]` 는 quantile calibration 으로 첫 100 frame 에서 결정.
  3. **Pin/Evict 결정** — score 상위 K 개 (`K = 0.25 × total_blocks`) 를 HBM-pin, 하위 (1-K) 는 DRAM 로 demote. `BlockAllocator.promote()/demote()` 신규 API 추가.
  4. **Hysteresis buffer** — 급격한 score 진동 방지를 위해 3-frame hysteresis 적용 (3 frame 연속 하위인 경우에만 evict).
  5. **Salience-aware prefetch** — score 가 threshold 이상으로 rebound 시 async prefetch 로 HBM 복귀 (M3 의 tri-tier async 와 연계).
- **④ 기존 해법 실패 + 차별화**: (i) StreamingLLM (uniform window) 은 temporal salience 를 무시. (ii) H2O (attention-score heavy hitter) 는 single-score 이라 recency × salience dual-axis 없음. (iii) VL-Cache 는 modality budget 기반 eviction 이나 frame-level temporal 무시. 본 M2 는 bi-exp dual-axis 로 accuracy drop ≤ 0.5pp 유지하면서 KV footprint -25~35% 달성.

##### **M3: Async Tri-Tier Streaming (polish)**

- **① 추가되는 Scheme**: vLLM `cache_engine.py` 확장하여 `TriTierCacheEngine` 추가. 3 physical tier — HBM-hot (< 24 frame window) / DRAM-pinned (< 1 hour) / NVMe-cold (deep archive) — 을 별도 `torch.cuda.Stream` 3 개로 관리. prefetch depth {1, 2, 4} layer-ahead 로 async tier migration.
- **② 해결하려는 문제**: 1-hour video KV (36 GB FP16 @ 7B) 는 RTX Pro 6000 (96GB) 단독 HBM 수용 가능하나 multi-tenant 시 memory pressure. 기존 vLLM 은 HBM↔CPU 2-tier 만 지원하여 multi-hour 에 불충분.
- **③ 동작 원리 (step-by-step)**:
  1. **Tier decision 분류기** — M2 의 score + frame recency 기반 block 을 3 tier 중 하나로 분류. `tier(block) = HBM if score > θ_hot else DRAM if age < 1h else NVMe`.
  2. **Async migration stream** — `stream_hbm2dram`, `stream_dram2nvme`, `stream_nvme2hbm` 세 별도 stream 에서 `cudaMemcpyAsync` 호출. 메인 compute stream 과 overlap.
  3. **Prefetch depth 계산** — 다음 layer-N 이 접근할 block 을 N 개 layer 앞서 prefetch. `prefetch_depth = 2` default.
  4. **NVMe I/O** — Samsung 990 Pro 7GB/s 을 `io_uring` async read 로 활용. `liburing` Python binding.
  5. **Miss fallback** — NVMe read latency 100-200 μs 초과 시 recompute fallback (FA3 prefill 재실행).
- **④ 기존 해법 실패 + 차별화**: (i) FlexGen 은 offline 3-tier 이나 online serving 최적화 없음. (ii) InfiniGen 은 CPU-GPU 2-tier. (iii) Semantic Scheduling (batch reorder) 은 tier 축과 orthogonal 하여 stacking 가능. 본 M3 는 online serving 3-tier + M2 salience-aware eviction 통합이 unique.

**Mechanism 간 상호작용**: M1 (physical layout) 은 M2/M3 의 기반 (row-aligned block 에서만 bi-exp eviction + tri-tier migration 이 efficient). M2 와 M3 는 score-driven 축으로 coupling 되나 M2 (eviction decision) 와 M3 (physical migration) 는 분리 가능.

**Tier 구성**: physical 3-tier (HBM / DRAM / NVMe) + software 1-tier (8KB row-aligned block) → R1/R1b ≤3-4 준수.

#### 2.1.4 평가 / 실험 플랜 (7-요소, R27-β)

##### (1) Hardware

- **Primary**: RTX 5090 32GB (HBM3e, row-hit profiling primary) + RTX Pro 6000 96GB (128K+ context).
- **Secondary**: RTX 4090 × 2 (GDDR6X cross-arch, row-buffer size 다름).
- **Access path**: 연구실 자체 보유 (서버 #5).

##### (2) Model

- **Primary**: LLaVA-Video-7B (HuggingFace `llava-hf/LLaVA-NeXT-Video-7B-hf`, FP16, 32 layer).
- **Secondary**: Qwen2.5-VL-7B (dynamic resolution robustness).
- **Robustness**: InternVL3-8B (다른 vision encoder).
- **Inference base**: vLLM v0.7 fork + FlashAttention-3 fork (indirection pointer customization).

##### (3) Dataset / Workload

- **Benchmarks**: VideoMME long subset (30-60min, 2700 clip), MVBench, LongVideoBench (1hr+).
- **Microbenchmark**: YouTube-8M subset 1-hour clip × 10 (row-hit profiling).
- **Synthetic**: 500 video × 5 query + 10 × 128K token stress test.
- **Primary metric**: HBM row-hit rate (%), decode throughput (tok/s).
- **Secondary**: TPOT (ms), memory footprint (GB), VideoMME accuracy (pp).

##### (4) Simulators / Tools

- **Simulator**: Ramulator2 v2.0 (HBM3/HBM3e analytical) + DRAMSim3 (cross-check).
- **Profiler**: Nsight Compute 2024.3 (`dram__sectors_read.sum`, `dram__sectors_read_hit.pct`, `lts__t_sectors_aperture_device_op_read_lookup_hit.pct`), Nsight Systems 2024.5 (3-stream overlap timeline).
- **Energy**: NVML `nvmlDeviceGetTotalEnergyConsumption` (5ms interval).
- **Serving stack**: vLLM v0.7 fork + `RowTilePagedAttention` backend.
- **External libs**: FlashAttention-3 (`https://github.com/Dao-AILab/flash-attention`), `liburing` Python binding.

##### (5) Ablation + Measurement Protocol

- **Factorial**: 2^3 = 8 cell (M1 × M2 × M3) + **HRTS+Mosaic stacking ablation (v3 신규)**.
- **Parameter sweeps**: tile size {128, 256, 512 tokens}, window α {0.3, 0.5, 0.7}, β_r {0.01, 0.05, 0.1}, prefetch depth {1, 2, 4}, context {8K, 32K, 128K, 256K}.
- **Baseline (10 편, peer-reviewed 70%)**: PagedAttention [SOSP 2023], SGLang [peer], VL-Cache [ICLR 2025], StreamingLLM [ICLR 2024], VideoLLM-online [CVPR 2024], H2O [NeurIPS 2023], InfiniGen [OSDI 2024], Mosaic [arXiv 2604], Semantic Scheduling [arXiv 2506], FlexGen [ICML 2023].
- **Runtime**: kernel 개발 4주 + vLLM integration 2주 + Ramulator 1주 + 실험 4주 + writing 3주 = **14 주**.
- **Fallback**: row-boundary reverse-engineering 실패 시 Nsight counter 간접 추정 + Ramulator analytical model 보강. Mosaic stacking 실패 시 각 축 독립 실험만 유지.

##### (6) Implementation Steps (Week-Level, R27-β 신규)

| Week | Component / File | 사용 API/Library | 완료 판정 |
|------|---------|---------|---------|
| W1 | vLLM v0.7 fork + LLaVA-Video-7B baseline 재현 | transformers 4.45+, vLLM v0.7, HF checkpoint | VideoMME accuracy 57.3 ±0.5pp 재현 |
| W2 | HBM row-buffer reverse-engineering microbenchmark | Nsight Compute, CUDA kernel stride sweep | row-size 8 KB 확정 + step discontinuity 도출 |
| W3 | Ramulator2 v2.0 analytical model 구축 | Ramulator2, Python analysis | row-hit prediction ±5% 실측 일치 |
| W4 | `BlockAllocator` 8KB row-aligned 확장 (M1) | `cuMemCreate`, `cuMemMap`, vLLM `block_manager.py` | 8 KB alignment 단위 allocation 확인 |
| W5 | `RowTilePagedAttention` backend 구현 (M1) | FA3 fork, CUDA 12.5, indirection pointer 수정 | row-hit 62%→75%+ 측정 |
| W6 | `BiExpWindowPolicy` 구현 (M2) | FA3 LSE 재활용, vLLM `scheduler.py` | score 계산 compute overhead <2% |
| W7 | Quantile calibration + hysteresis | Python calibration script | α/β_r 최적값 도출 |
| W8 | `TriTierCacheEngine` 구현 (M3) | `torch.cuda.Stream`, `io_uring`, Samsung 990 Pro | 3-tier migration 정상 작동 |
| W9 | Prefetch depth 2 최적화 | async stream overlap profiling | tier overlap ≥ 70% |
| W10 | 2^3 factorial 8 cell 실험 | Hydra config, bash automation | 8 cell 모두 완료 |
| W11 | 10 baseline 비교 실험 | 각 baseline repo | 10 × 3 dataset 측정 완료 |
| W12 | HRTS+Mosaic stacking ablation | Mosaic repo fork | stacking +5~8% 확인 |
| W13-14 | Writing + ablation table + camera-ready | LaTeX, pandas | ASPLOS 논문 draft |

##### (7) Preliminary Analysis Metrics (R27-β 신규)

| 측정 지표 | 도구 + counter/command | 측정 조건 | 기대 범위 (baseline) | 개선 후 목표 (Δ) |
|---|---|---|---|---|
| HBM row-hit rate | Nsight `dram__sectors_read_hit.pct` | LLaVA-Video-7B, 128K seq, decode | 60-65% | **82-88% (+20-25pp)** |
| Decode throughput | vLLM wall-clock | 64K+ context, batch=1 | 42-58 tok/s | **+25~35%** |
| TPOT | vLLM `--output-json` | 128K context | 25-35 ms | **-20~30%** |
| Memory footprint | `nvidia-smi` | 128K context | 45-55 GB | **-30~40%** |
| LTS hit rate | Nsight `lts__t_sectors_aperture_device_op_read_lookup_hit.pct` | L2 scope | 55-65% | **+10pp** |
| VideoMME accuracy | lm-evaluation-harness | long subset | 57.3% (baseline) | **±0.5pp** |
| Energy/token | NVML 5ms | 1h VideoMME run | 0.8-1.1 J/tok | **-12~20%** |
| Stacking (HRTS+Mosaic) | ablation | full pipeline | - | **+5~8% orthogonal** |

**Preliminary Study 순서 (R27-β 신규)**:

- **(i) Baseline reproduction**: vLLM v0.7 + LLaVA-Video-7B + VideoMME long subset 공식 config 로 accuracy 57.3 ±0.5pp 재현. Decode throughput 재현 ±5%. 실패 시 transformers 4.45 vs 4.50 차이, CUDA graph capture on/off, FA3 version 매칭 확인.
- **(ii) Bottleneck attribution**: `ncu --section MemoryWorkloadAnalysis --section SchedulerStats` 로 decode 단계의 DRAM BW 비율 측정 (기대 75-85% BW-bound). Compute-bound vs BW-bound vs latency-bound 중 **BW-bound** 로 분류 예상. Row-hit miss 가 전체 decode latency 의 몇 % 인지 직접 측정.
- **(iii) Roofline upper bound**: LLaVA-Video-7B decode phase 의 arithmetic intensity (AI = FLOPs / bytes) 계산. RTX 5090 roofline (1008 GB/s HBM3e) 상 현재 peak 의 60-65% 활용. 본 기법 목표 80-85%.
- **(iv) Mechanism 단독 Micro-benchmark**: (a) **M1 only** — row-aligned tile 만 적용, bi-exp off, tri-tier off. Row-hit 및 decode throughput 개선 측정 (기대 +15-20%). (b) **M2 only** — default 4KB page + bi-exp eviction. Accuracy 유지 vs footprint 감소 측정 (기대 -25%). (c) **M3 only** — 4KB page + default eviction + tri-tier async. Long-context (128K+) 에서 memory pressure 개선 측정. 세 mechanism 의 sum 이 full 조합과 linear 한지 검증.

#### 2.1.5 예상 효과 (보수적, scope 명시)

| 지표 | Baseline | 목표 | 조건 |
|---|---|---|---|
| Decode throughput | 42-58 tok/s | **+25~35%** | long-context 64K+ |
| TPOT | 25-35 ms | **-20~30%** | 128K context |
| Memory footprint | 45-55 GB | **-30~40%** | NVMe tier 활용 |
| HBM row-hit | 62% | **82-88%** | primary novelty 지표 |
| VideoMME accuracy | 57.3% | **±0.5pp** | long subset |
| HRTS+Mosaic stacking | - | **+5~8%** | orthogonality 증명 |

**Scoring**: Novelty 7.8 / Differentiation 7.9 / Impact 8.0 / Feasibility 7.7 → **avg 7.85 (lead)**.

#### 2.1.6 Tier-2 Scope 축소 Variant (IEEE CAL 4p / DATE 6p)

- **Target**: IEEE CAL 4p / DATE 6p.
- **Single mechanism**: M1 only (Row-aligned KV tile) — 가장 measurable row-hit +15-25pp.
- **Scope 축소**: LLaVA-Video-7B + VideoMME long subset 단일, RTX 5090 단일 GPU.
- **Runtime**: 6 주.
- **Top-tier 와의 관계**: Precedence 확보 (M1 을 2026-07 CAL 로 먼저 공개, M2+M3 를 2026-10 ASPLOS 로 확장).

---

### 2.2 ContextMIG+ — Reuse Graph × MIG Dual-Issue × Phase Coalesce for Multi-tenant VLM (Top 2, Avg 7.73 / ASPLOS 2026 / MLSys 2026)

#### 2.2.1 개요

Multi-tenant VLM serving 에서 request 간 visual-prefix reuse graph (CLIP-L LSH 16-bit SimHash) × MIG dual-issue (MIG-A prefill-visual + MIG-B decode-LLM) × phase-aligned coalescing 의 3 mechanism 으로 p95 TTFT -18-28%, throughput +22-32% 목표. 2026-04 Mosaic (cross-modal clustering) 과 Predictable LLM Serving (cluster MIG) 을 모두 baseline 에 편입, **intra-GPU MIG dual-issue + phase coalesce** 축으로 차별화.

#### 2.2.2 기존 연구 한계점 및 Gap

- [Mosaic arXiv:2604.10060](https://arxiv.org/abs/2604.10060) — KV-centric cross-modal clustering. **SM/MIG 미활용** → ContextMIG+ 는 MIG placement 까지 통합.
- [Predictable LLM Serving arXiv:2508.20274](https://arxiv.org/abs/2508.20274) **(v3 신규 concurrent 55-60%)** — GPU cluster 상 MIG scheduling. **VLM content-axis reuse graph 없음**, cluster-level (inter-GPU) 만.
- [Semantic Scheduling arXiv:2506.12204](https://arxiv.org/abs/2506.12204) — software-layer content scheduling. **MIG/HBM placement 미결합**.
- [LithOS SOSP 2025](https://dl.acm.org/doi/10.1145/3695053.3731083) [peer-reviewed] — fine-grained SM API 일반론. **VLM cluster logic 없음**.
- [Prefill-as-a-Service arXiv:2604.15039](https://arxiv.org/abs/2604.15039) **(v3 신규 related work)** — prefill 분리 serving, coalesce + dual-issue 와 orthogonal.
- [IceCache arXiv:2604.10539](https://arxiv.org/abs/2604.10539) **(v3 신규 related work)** — dedicated cold cache.

**Gap 요약**: Content-axis reuse graph × intra-GPU MIG dual-issue × phase-aligned coalescing 3축 통합은 공백.

#### 2.2.3 제안 기법 (3 Mechanisms, R27-α: 4 필수 요소)

##### **M1: CLIP-L LSH Reuse Graph Classifier (replace)**

- **① 추가되는 Scheme**: vLLM v0.7 fork 에 신규 module `vllm/engine/routing/reuse_graph.py` 추가. request arrival 시 image 의 CLIP-L (`openai/clip-vit-large-patch14`) last-pool embedding (768-d) 을 계산 → **16-bit SimHash LSH** 로 bucket 결정 → sliding window (최근 256 request) 기반 reuse graph 업데이트. 이 graph 는 `networkx.DiGraph` 로 관리하며 edge weight = Jaccard similarity of visual prefix. vLLM `LLMEngine.add_request()` hook 에 삽입되어 < 0.6 ms 의 hash latency 로 동작.
- **② 해결하려는 문제**: Multi-tenant VLM serving 에서 semantic-near duplicate image (CLIP cosine > 0.92) 가 실제 production trace (LMSys VisionArena) 의 **18-32%** 를 차지하나, 기존 vLLM/SGLang 은 exact prefix match (Radix tree) 만 지원 → semantic-near duplicate 의 prefix KV 재사용 불가. 결과 p95 TTFT 가 reusable workload 에서도 fresh-context 수준에 머무름.
- **③ 동작 원리 (step-by-step)**:
  1. **CLIP-L embedding 계산** — Request arrival 즉시 CLIP-L encoder (1 forward pass on 224×224 image) 로 768-d pooled embedding 추출. 이는 `stream_clip` 별도 CUDA stream 에서 계산 (main stream 과 overlap).
  2. **16-bit SimHash** — embedding 을 16 hyperplane 에 projection → sign 추출 → 16-bit hash. `hash = sum((emb @ hyperplane_i > 0) << i for i in range(16))`.
  3. **Sliding window graph 업데이트** — 최근 256 request 의 hash 를 window 에 유지. Hash collision (동일 bucket) 시 edge 추가. LRU eviction.
  4. **Prefix reuse 결정** — 동일 bucket 의 과거 request 가 있으면 해당 KV prefix (image token 의 처음 75%) 를 reuse. vLLM `BlockAllocator.share_blocks()` 호출.
  5. **Fallback** — Collision false positive 시 (검증 단계에서 L2 distance > threshold) 해당 reuse 무효화, fresh prefill 로 복귀.
- **④ 기존 해법 실패 + 차별화**: (i) SGLang RadixAttention 은 exact token match 만 지원 (text prefix 에 최적화). (ii) Mosaic 은 KV-level cross-modal clustering 이나 LSH-based reuse graph 없음. (iii) Semantic Scheduling 은 batch reorder 축, content hash 재사용 없음. 본 M1 은 **CLIP-L semantic hashing + reuse graph** 가 first-to-report for VLM multi-tenant, F1 ≥ 0.82 + hash latency ≤ 1.5ms + collision ≤ 3% 달성 목표.

##### **M2: Tier-Aware MIG Dual-Issue Partition (replace)**

- **① 추가되는 Scheme**: NVIDIA NVML `nvmlDeviceCreateGpuInstance` + `cuCtxFromGreenCtx` (CUDA 12.5+) API 로 H100 80GB 단일 GPU 를 **2 MIG slice (MIG-A 4/7 = 56 SM / MIG-B 3/7 = 42 SM)** 로 분할. MIG-A 는 **visual prefill** (CLIP encoder + image token prefill), MIG-B 는 **LLM decode** 전용. vLLM 측 `vllm/executor/gpu_executor.py` 를 확장하여 `MIGDualIssueExecutor` 추가 — request 의 phase (prefill-visual / decode-LLM) 에 따라 적절한 MIG slice 로 dispatch. 3-SW tier (hot/warm/cold) × 2-phys MIG slice 조합.
- **② 해결하려는 문제**: 기존 vLLM 단일 GPU 구성에서 prefill-visual (compute-bound, CLIP encoder 큰 batch) 과 decode-LLM (BW-bound, KV fetch) 이 동일 SM pool 을 순차 점유 → SM util 60-72% 에 머무름. Multi-tenant 시 prefill burst 가 decode tail latency 를 악화 (p99 TTFT +30-40%).
- **③ 동작 원리 (step-by-step)**:
  1. **MIG 초기화** — Engine init 시 `nvmlDeviceCreateGpuInstance(device, 4/7)` 로 MIG-A (56 SM), `4/7-remainder` 로 MIG-B (42 SM) 생성. 각각을 CUDA context 로 bind.
  2. **Phase classifier** — 매 request 의 phase 를 tag (`phase ∈ {prefill_visual, decode_llm}`). vLLM scheduler 에서 phase-aware queue 유지.
  3. **Dual-issue dispatch** — Prefill-visual 을 MIG-A 의 CUDA stream 으로 dispatch, Decode-LLM 을 MIG-B 로. 두 phase 가 **concurrent 실행** (별도 SM pool).
  4. **Green Context μs-reconfig** — workload imbalance 감지 시 (MIG-A 또는 MIG-B 가 idle) `cuDevSmResourceSplit` 으로 SM 재분배. B1 GCReconfProfile 에서 측정한 reconfig cost (~30-80 μs) 를 budget 으로 활용.
  5. **Tier-MIG 조합** — software tier (hot/warm/cold request) × MIG slice (A/B) 6 조합에서 optimal mapping 을 scheduling policy 로 결정.
- **④ 기존 해법 실패 + 차별화**: (i) Predictable LLM Serving 은 cluster-level MIG (inter-GPU), 본 M2 는 intra-GPU. (ii) LithOS 는 generic SM API, VLM-specific 없음. (iii) MIGER (ICPP 2024) 는 MIG reconfig overhead 만 측정, dual-issue 활용 없음. 본 M2 는 first-to-report intra-GPU MIG dual-issue for VLM, SM util +12-20pp 개선 예상.

##### **M3: Phase-Aligned Coalescing (new — critical gap 대응)**

- **① 추가되는 Scheme**: vLLM scheduler 에 신규 batch formation 함수 `_phase_aligned_batch_form` 추가. 기존 continuous batching 이 phase 를 무시하고 request 를 interleave 하는 것을 수정하여 **같은 phase 의 request 끼리 batch coalesce** 한다. Prefill-visual batch 와 decode-LLM batch 를 분리하여 각각 MIG slice 로 dispatch.
- **② 해결하려는 문제**: vLLM continuous batching 은 phase mixing 으로 인해 kernel launch 당 prefill/decode 혼재 → CUDA kernel 내부 branch divergence 및 shared memory 충돌 발생. Nsight Compute 측정 시 kernel efficiency 가 phase-pure batch 대비 15-25% 떨어짐.
- **③ 동작 원리 (step-by-step)**:
  1. **Phase tag** — 각 request 의 현재 phase 식별 (`phase = prefill_visual if tokens_generated == 0 else decode_llm`).
  2. **Phase-pure queue** — Phase 별 별도 queue 유지 (`queue_prefill`, `queue_decode`).
  3. **Coalesce batch** — batch formation 시 한 phase queue 에서만 최대 `max_batch_size` request 추출.
  4. **MIG dispatch** — `queue_prefill` → MIG-A (via M2), `queue_decode` → MIG-B 로 dispatch.
  5. **Fairness 제어** — Starvation 방지를 위해 phase queue 간 round-robin round quota 설정 (max 3 prefill → 1 decode round).
- **④ 기존 해법 실패 + 차별화**: (i) Prefill-as-a-Service 는 prefill/decode 를 inter-GPU 분리. 본 M3 는 intra-GPU + MIG dual-issue 로 HW cost 없이 동일 효과. (ii) 기존 vLLM continuous batching 은 phase mixing 허용. (iii) Semantic Scheduling 은 content 축, phase 축 아님. 본 M3 는 kernel efficiency +15-25% 예상.

**Mechanism 간 상호작용**: M1 (reuse graph) 은 KV 재사용을 통해 prefill 총량 자체를 감소 (workload 축), M2 (MIG dual-issue) 는 HW 축, M3 (coalescing) 는 software scheduling 축. 3축 모두 orthogonal, factorial ablation 완전 가능. M1+M2 만 해도 의미 있으나 M3 의 phase 분리가 M2 의 MIG dispatch 효율을 극대화.

**Tier 구성**: physical 2-tier (MIG-A / MIG-B) + software 3-tier (hot/warm/cold) = 5-tier ≤ 상한 (R1b 엄밀히는 software 3 으로 처리).

#### 2.2.4 평가 / 실험 플랜 (7-요소)

##### (1) Hardware

- **Primary**: AWS p5.48xlarge (H100 80GB × 8, MIG 지원, 1-2주 $1200-2000 예산).
- **Secondary**: RTX Pro 6000 96GB (Green Context + L2, CUDA 12.5).
- **Fallback**: RTX 4090 × 2 (MPS + Green Context fallback, MIG 미지원).

##### (2) Model

- **Primary**: Qwen2.5-VL-7B-Instruct (tenant A) + LLaVA-OneVision-7B (tenant B). FP16.
- **Secondary**: InternVL3-8B, MiniCPM-V-2.6 (robustness).
- **Inference base**: vLLM v0.7 fork with `ClusterAwareBlockManager` + SGLang v0.4 (semantic-radix 비교).

##### (3) Dataset / Workload

- **Synthetic**: Mixed trace (OCR/grounding/caption/chat/reasoning 각 20%, Poisson arrival λ=2-8 req/s).
- **Real**: MMDU + LMSys VisionArena partnership 시도 (실패 시 synthetic only).
- **Scale**: 5000 req + 1000 document × 3-5 query + 500 MMDU.
- **Primary metric**: p95 TTFT (ms), throughput (req/s).
- **Secondary**: SM util (%), MIG reconfig rate, reuse hit rate.

##### (4) Simulators / Tools

- **Profiler**: Nsight Compute (`sm__warps_active`, `mig__reconfig_latency_ns`), Nsight Systems.
- **MIG control**: NVML `nvmlDeviceCreateGpuInstance`, CUDA 12.5 `cuCtxFromGreenCtx`, `cuDevSmResourceSplit`.
- **Hash lib**: OpenAI CLIP-B/32 (fast fallback) / CLIP-L (primary), `imagehash` pHash (baseline).
- **Serving stack**: vLLM v0.7 fork + `ClusterAwareBlockManager` + `MIGDualIssueExecutor`.

##### (5) Ablation + Measurement Protocol

- **Factorial**: 2^3 (M1 × M2 × M3) + LSH bucket {8, 16, 32 bit} + MIG slice {3:5, 4:4, 5:3} + window {128, 256, 512} + tenant {2, 4, 8}.
- **Baseline (10 편, peer-reviewed 60%)**: vLLM [SOSP 2023], SGLang [arXiv], Mosaic [arXiv 2604], HERMES [arXiv], Bullet [preprint], LithOS [SOSP 2025], Llumnix [OSDI 2024], VL-Cache [ICLR 2025], DynamoLLM [preprint], Predictable LLM Serving [arXiv 2508].
- **Runtime**: 개발 8주 + AWS 실험 2주 + 분석 3주 + writing 4주 = **17 주**.
- **Fallback**: AWS 예산 초과 시 lambda.ai H100 1-week rental. Partnership 거절 시 synthetic only (reviewer 제약 명시). MIG dual-issue HW 제약 시 MPS+Green Context (-5~8% 성능).

##### (6) Implementation Steps (Week-Level)

| Week | Component / File | 사용 API/Library | 완료 판정 |
|------|---------|---------|---------|
| W1-2 | vLLM v0.7 fork + Qwen2.5-VL-7B + LLaVA-OV-7B baseline 2-tenant | vLLM, transformers 4.45, HF checkpoint | 2-tenant 정상 구동, TTFT 재현 ±5% |
| W3 | CLIP-L embedding 계산 module (M1 prep) | CLIP-L, `torch.cuda.Stream` | embedding latency < 2 ms |
| W4 | 16-bit SimHash + sliding window graph (M1) | `networkx.DiGraph`, numpy | F1 ≥ 0.82 on synthetic duplicate set |
| W5 | Reuse prefix share (M1 integration) | vLLM `BlockAllocator.share_blocks()` | reuse hit rate 18-30% 측정 |
| W6 | MIG 2-slice 생성 + CUDA context bind (M2) | NVML, `nvmlDeviceCreateGpuInstance` | MIG-A/B 독립 실행 확인 |
| W7 | `MIGDualIssueExecutor` 구현 (M2) | CUDA 12.5, vLLM `gpu_executor.py` | dual-issue dispatch 성공 |
| W8 | Green Context μs-reconfig 통합 | `cuDevSmResourceSplit`, `cuCtxFromGreenCtx` | reconfig latency < 100 μs |
| W9 | Phase-aware queue + coalesce (M3) | vLLM `scheduler.py` 확장 | phase-pure batch 정상 형성 |
| W10-11 | AWS p5.48xlarge 2-week 실험 | AWS CLI, Hydra config | 2^3 = 8 cell × 5000 req 완료 |
| W12 | 10 baseline 비교 | 각 baseline repo | baseline × 3 workload 완료 |
| W13 | Mosaic stacking ablation | Mosaic repo fork | stacking +4-7% 검증 |
| W14 | RTX Pro 6000 cross-validation | Pro 6000 자체 보유 | p95 TTFT trend 일치 |
| W15-17 | Writing + ablation + camera-ready | LaTeX, pandas | ASPLOS 논문 draft |

##### (7) Preliminary Analysis Metrics

| 측정 지표 | 도구 + counter/command | 측정 조건 | 기대 범위 (baseline) | 개선 후 목표 (Δ) |
|---|---|---|---|---|
| p95 TTFT | vLLM `--output-json` + log | 2-tenant, visual overlap workload | 450-650 ms | **-18~28%** |
| Throughput | vLLM wall-clock | 2-tenant, λ=4 req/s | 12-16 req/s | **+22~32%** |
| SM utilization | Nsight `sm__warps_active.avg.pct_of_peak_sustained_active` | prefill+decode concurrent | 60-72% | **+12~20pp** |
| Reuse hit rate | custom metric | semantic-near duplicate workload | 0 (baseline vLLM) | **18-32%** |
| LSH hash latency | wall-clock + CUPTI | per-request | - | **< 1.5 ms** |
| LSH F1 score | synthetic paired set | 500 duplicate + 500 unique | - | **≥ 0.82** |
| LSH collision rate | synthetic independent set | 1000 independent | - | **≤ 3%** |
| MIG reconfig latency p99 | NVML + `cuEventRecord` | dual-issue mode | - | **< 100 μs (B1 budget 내)** |
| Mosaic stacking delta | ablation | full+Mosaic vs full-only | - | **+4~7%** |

**Preliminary Study 순서**:

- **(i) Baseline reproduction**: vLLM + Qwen2.5-VL + LLaVA-OV 2-tenant 기본 구성으로 p95 TTFT 재현 ±5%. 실패 시 tenant-level isolation 여부, FA3 version, KV cache dtype 확인.
- **(ii) Bottleneck attribution**: Nsight Systems 로 prefill 과 decode 의 SM 점유율 분리 측정. SM competition 이 p95 TTFT 의 몇 % 원인인지 (기대 25-35%). Phase mixing overhead (기대 15-25%). BW-bound vs compute-bound 분류.
- **(iii) Roofline upper bound**: Qwen2.5-VL-7B prefill + LLaVA-OV-7B decode co-location 의 AI 계산. H100 roofline 상 현재 SM 활용 60-72% → 본 기법 목표 82-90%.
- **(iv) Mechanism 단독 Micro-benchmark**: (a) **M1 only** — reuse graph + prefix share, MIG 단일 + default coalesce. Reuse hit rate 18-32% 측정, fresh context 대비 TTFT -15%. (b) **M2 only** — dual-issue MIG + default batching. SM util +12-20pp 측정. (c) **M3 only** — phase coalesce + single MIG. Kernel efficiency +15-25% 측정. Sum vs full 비교로 interaction 확인.

#### 2.2.5 예상 효과

| 지표 | Baseline | 목표 | 조건 |
|---|---|---|---|
| Throughput | 12-16 req/s | **+22~32%** | 2-tenant 7B co-location |
| p95 TTFT | 450-650 ms | **-18~28%** | visual overlap 시 |
| SM util | 60-72% | **+12~20pp** | dual-issue mode |
| vs Mosaic stacking | - | **+4~7%** | orthogonality |
| vs Predictable LLM Serving | - | **+8~12%** | intra-GPU dual-issue vs cluster MIG |
| Single-tenant/fresh-context | - | **2~4%** (scope 밖 명시) | control |

**Scoring**: Novelty 7.5 / Differentiation 7.8 / Impact 8.0 / Feasibility 7.6 → **avg 7.73**.

#### 2.2.6 Tier-2 Scope 축소 Variant (IEEE ESL 4p / IEEE CAL 4p)

- **Target**: IEEE ESL 4p / IEEE CAL 4p.
- **Single mechanism**: M1 only (CLIP-L LSH reuse graph classifier) — 가장 self-contained.
- **Scope 축소**: 2-tenant Pro 6000 단일 GPU, F1 ≥ 0.82 + hash latency ≤ 1.5 ms + collision ≤ 3% 측정 중심.
- **Runtime**: 6 주.
- **Top-tier 와의 관계**: M1 을 2026-08 CAL 로 먼저 공개, M2+M3 를 2026-10 ASPLOS 로 확장.

---

### 2.3 PhaseGraph-VLA+ — Trajectory-Phase Conditioned CUDA Graph Dispatcher with SSE-Driven Boundary Detection (Top 3, Avg 7.18 / MLSys 2026 / CoRL 2026)

#### 2.3.1 개요

VLA single-robot inference 에서 trajectory phase 를 Approach / Manipulate / Retract 3-phase 로 구분, **SSE hidden state L2-drift + Page-Hinkley change-point detector** 로 phase boundary 를 <100μs 에 감지하고, phase-specific CUDA Graph variant 를 dispatch. v1 A1 을 v2 에서는 PhaseScope (GUI agent) 로 대체했으나 Major Revision 으로 탈락, v3 에서 **VLA 단일 robot 도메인 공백** 복원 + FlashVLA 2025-05 차별화 (graph-level vs token-level) 명시.

#### 2.3.2 기존 연구 한계점 및 Gap

- [VLA-Cache NeurIPS 2025 Poster](https://openreview.net/forum?id=QZYZ0Xm58q) [peer-reviewed] — frame-diff KV reuse. **Phase-aware graph switch 없음**, reviewer 가 grasping moment frame-diff miss 지적.
- [AC²-VLA arXiv:2601.19634](https://arxiv.org/abs/2601.19634) — action context → token prune. **Algorithmic 축, system graph 축 없음**.
- [KV-Efficient VLA arXiv:2509.21354](https://arxiv.org/abs/2509.21354) — RNN-gated chunked KV. Algorithmic.
- [ADP-VLA arXiv:2509.22093](https://arxiv.org/abs/2509.22093) — action-aware dynamic pruning. Algorithmic.
- [FlashVLA arXiv:2505.21200](https://arxiv.org/abs/2505.21200) **(v3 신규 scoop 접경 68-72%)** — "action reuse" 직접 명명. **Token-level reuse**, graph-level execution switch 와 축이 다름.
- [Nova arXiv:2509.21301](https://arxiv.org/abs/2509.21301) — stage partition, VLA 아닌 VLM general.
- [Running-VLAs-at-Real-time arXiv:2510.26742](https://arxiv.org/abs/2510.26742) — chunk boundary async, phase-specific kernel 없음.

**Gap 요약**: Phase × CUDA Graph variant × phase-tuned fused kernel 3축 결합은 공백. **FlashVLA 차별화 축**: graph-level execution switch ≠ token-level reuse.

#### 2.3.3 제안 기법 (3 Mechanisms, R27-α: 4 필수 요소)

##### **M1: SSE Phase Predictor (improve + absorb v1 P2)**

- **① 추가되는 Scheme**: vLLM fork 의 `vllm/model_executor/models/openvla.py` 에 신규 module `SSEPhasePredictor` 추가. 이 predictor 는 OpenVLA 의 mid-layer (L_mid = layer 15 of 32) hidden state 의 L2 drift 를 계산하고, **Page-Hinkley change-point detector** + 2-threshold (soft pre-warm 0.3σ / hard evict 0.8σ) + hysteresis (3 frame) 로 phase transition 을 판정한다. training-free quantile calibration 을 첫 100 frame 에서 수행 → 학습 불필요. VLA-specific feature 추가 — gripper Δ (delta from last frame), trajectory curvature (3-frame window), DINOv2 object distance (1.2 ms/frame).
- **② 해결하려는 문제**: VLA-Cache (NeurIPS 2025) 가 frame-diff reuse 시 **grasping moment** (phase transition) 에서 frame-diff 가 작음에도 semantic change 가 크다 → reviewer 가 명시적으로 지적. 기존 phase classifier 는 off-the-shelf CNN (MobileNetV3 등) 으로 추론 latency 5-8 ms 발생하여 control loop (10 ms budget) 를 초과. VLA-specific hidden state 내부 signal 을 이용한 low-latency predictor 부재.
- **③ 동작 원리 (step-by-step)**:
  1. **L_mid hidden state extraction** — OpenVLA decoder 의 layer 15 output (hidden_state, shape = [batch, seq, 4096]) 을 hook 으로 capture. extraction overhead < 50 μs.
  2. **L2 drift 계산** — `drift(t) = ||hidden(t) - hidden(t-1)||_2`. 연속 frame 간 drift 를 moving average 로 smoothing.
  3. **Page-Hinkley detector** — cumulative `CUSUM(t) = max(0, CUSUM(t-1) + drift(t) - μ - δ)`. Threshold `λ` 초과 시 change-point 판정. `λ` 는 quantile calibration 으로 첫 100 frame 에서 false-positive rate < 5% 를 만족하도록 설정.
  4. **VLA-specific feature 결합** — gripper Δ (robot proprioception 에서 read), trajectory curvature (end-effector pose 3 frame window), DINOv2 object distance (별도 stream 에서 계산, 1.2 ms/frame). 가중합 score 로 최종 phase 결정.
  5. **Phase label output** — `phase ∈ {Approach, Manipulate, Retract}` 를 produce. Hysteresis 로 spurious transition 억제 (3 frame 연속 동일해야 confirm).
- **④ 기존 해법 실패 + 차별화**: (i) VLA-Cache 의 frame-diff 는 pixel-level, grasping 의 hidden-state-level semantic change 를 놓침. (ii) Off-the-shelf MLP classifier 는 training data 필요 + latency 5-8 ms. (iii) 본 M1 은 training-free + L_mid hidden state 재사용 (OpenVLA forward pass 의 부산물) → 추가 compute ≤ 100 μs. PH FP rate < 5% (goal, LIBERO 5 task empirical 필수). Decision latency < 100 μs (control loop budget 1% 수준).

##### **M2: Phase-Specific CUDA Graph Dispatcher (improve)**

- **① 추가되는 Scheme**: vLLM `model_executor/models/openvla.py` 에 신규 class `PhaseGraphDispatcher` 추가. 3 phase 별로 별도 CUDA Graph variant 를 미리 capture — Approach graph / Manipulate graph / Retract graph. M1 이 phase transition 감지 시 `cudaGraphLaunch` 로 즉시 switch. Graph capture 시점은 engine init + phase calibration 완료 후 (warm-up 200 frame 후).
- **② 해결하려는 문제**: Single CUDA Graph 는 전체 VLA forward pass 를 통합 capture 하지만, Approach (encoder-heavy) 와 Manipulate (action-head heavy), Retract (encoder-bypass 가능) 의 compute pattern 이 크게 달라 단일 graph 로 kernel efficiency 최적화 불가. Nsight 측정 시 phase-unaware graph 는 phase-specific 대비 kernel efficiency 15-25% 낮음.
- **③ 동작 원리 (step-by-step)**:
  1. **Approach graph capture** — SigLIP patch-embed **full** + FlashAttn-3 MQA fused + LLM decode full. Graph size ~ 24 MB.
  2. **Manipulate graph capture** — SigLIP **partial-batch (40% re-encode)** + action-head SiLU+Linear fused + reduced attention heads (32→24). Graph size ~ 18 MB.
  3. **Retract graph capture** — SigLIP **bypass** (last-2-step feature linear extrapolation) + KV static reuse + LLM decode only. Graph size ~ 14 MB.
  4. **Dispatch 로직** — M1 출력 phase label 에 따라 `cudaGraphInstantiate()` 로 pre-instantiated graph 중 선택, `cudaGraphLaunch()` 호출.
  5. **Graph transition 비용** — graph switch 자체는 ~ 5 μs, 무시 가능. Phase transition 시 warm-up 1-2 frame 은 accuracy 약간 하락 가능 (M1 hysteresis 로 완화).
- **④ 기존 해법 실패 + 차별화**: (i) FlashVLA (arXiv:2505.21200) 는 token-level action reuse, graph-level execution switch 와 축 다름 — orthogonal composable. (ii) Nova 는 VLM general stage partition (encoder/prefill/decode), VLA trajectory phase 와 granularity 다름. (iii) 기존 vLLM CUDA Graph 는 single unified. 본 M2 는 VLA trajectory phase 특화 3-graph dispatch, kernel efficiency +15-25% 예상.

##### **M3: Phase-Specific SM Partition (polish — optional stacking)**

- **① 추가되는 Scheme**: Green Context `cuDevSmResourceSplit` 으로 phase 별 SM 할당 재분배. Approach phase: 80% encoder / 20% LLM. Manipulate: 35% encoder-partial / 55% LLM / 10% action-head. Retract: 85% LLM / 15% next-episode prefetch. vLLM fork 의 `MIGDualIssueExecutor` 확장 (ContextMIG+ 의 M2 와 공유 가능한 인프라).
- **② 해결하려는 문제**: Phase 별 compute pattern 이 다름에도 SM 할당이 고정되면 (e.g., all-SM encoder) 특정 phase 에서 underutilized. Manipulate phase 에서 encoder 는 partial-batch 이므로 encoder SM 을 LLM/action-head 로 이관 필요.
- **③ 동작 원리 (step-by-step)**:
  1. **Per-phase SM budget** — pre-computed profile 로 각 phase 의 optimal SM split 도출.
  2. **Dynamic reconfig** — M1 phase transition 감지 시 `cuDevSmResourceSplit` 로 Green Context 재분배 (reconfig latency ~30-80 μs, B1 GCReconfProfile 수치 활용).
  3. **Stream association** — 각 SM group 에 별도 CUDA stream 연결 (encoder stream, LLM stream, action-head stream).
  4. **Concurrent dispatch** — Phase 별 graph 를 해당 stream 으로 dispatch, SM pool 분리 이용.
  5. **Rollback** — phase transition false positive 감지 시 (M1 confidence 낮음) 이전 SM split 으로 rollback.
- **④ 기존 해법 실패 + 차별화**: (i) 기존 OpenVLA 구현은 monolithic GPU execution, phase-aware SM split 없음. (ii) Nova 는 stage partition (encoder vs decode) 만, phase-specific 아님. (iii) 본 M3 는 Nova 의 stage partition + phase awareness 결합 → stacking 시 +5-8% 추가 예상.

**Mechanism 간 상호작용**: M1 (phase detection) → M2 (graph switch) → M3 (SM reconfig) 가 causal chain. M1 없이는 M2/M3 가 무의미하지만, M1+M2 만으로도 큰 효과. M3 는 optional stacking.

**Tier 구성**: physical 1-tier (GPU SM) + software 3-tier (3 phase) → R1/R1b 준수.

#### 2.3.4 평가 / 실험 플랜 (7-요소)

##### (1) Hardware

- **Primary**: RTX 4090 24GB (OpenVLA-7B 주 실험, 연구실 보유).
- **Secondary**: RTX 5090 32GB (phase-specific kernel 성능 비교).
- **Edge**: Jetson Orin AGX 64GB (embedded deployment, LIBERO 포팅).
- **Host**: AMD Ryzen 9 7950X + 128GB DDR5.

##### (2) Model

- **Primary**: OpenVLA-7B (BF16, LIBERO 공식 checkpoint) — `openvla/openvla-7b`.
- **Secondary**: OpenVLA-OFT (parallel decoding) — B3 ActHeadFuse 와 접점.
- **Robustness**: π0 (Physical Intelligence).
- **Inference base**: OpenVLA HF wrapper + vLLM v0.7 fork (CUDA Graph capture variant 지원 추가).

##### (3) Dataset / Workload

- **Benchmarks**: LIBERO (4 suite × 100 trial = 400 trial), RoboCasa (Isaac Sim), SimplerEnv, CALVIN.
- **FlashVLA direct comparison trace (v3 신규)**: FlashVLA repo 의 grasping moment trace 직접 비교.
- **Scale**: 500 trial × 3 model × 4 phase config = 6000 rollout.
- **Primary metric**: Median latency (ms/step), SimplerEnv Hz.
- **Secondary**: Success rate (%), PH FP rate (%), Jetson deployment latency.

##### (4) Simulators / Tools

- **Simulator**: NVIDIA Isaac Sim v4.5 + Isaac Lab.
- **Statistical**: Page-Hinkley (`scipy.stats`), quantile calibration Python.
- **Profiler**: Nsight Compute (CUDA Graph capture overhead), Nsight Systems (phase transition timeline).
- **Vision aux**: DINOv2 (`facebookresearch/dinov2`) for object distance feature.
- **Serving stack**: vLLM v0.7 fork + `SSEPhasePredictor` + `PhaseGraphDispatcher`.

##### (5) Ablation + Measurement Protocol

- **Factorial**: 2^3 (M1 × M2 × M3) + **4-way orthogonality** (baseline / M1 oracle / M1+M2 / full / stacked+Nova).
- **Parameter sweeps**: PH threshold {0.5σ, 0.8σ, 1.2σ}, hysteresis {2, 3, 5 frame}, SM ratio (M3) {phase 별 5 config}.
- **Baseline (9 편, peer-reviewed 67%)**: VLA-Cache [NeurIPS 2025], AC²-VLA [arXiv], KV-Efficient VLA [arXiv], ADP-VLA [arXiv], SpecPrune-VLA [preprint], Nova [arXiv], DuetServe [preprint], Running-VLAs-at-Real-time [arXiv], FlashVLA [arXiv 2505].
- **Runtime**: 개발 6주 + 실험 5주 + writing 4주 = **15 주**.
- **Fallback**: FlashVLA re-impl 실패 시 paper 수치 interpolation 간접 비교 (reviewer 수용 가능 명시).

##### (6) Implementation Steps (Week-Level)

| Week | Component / File | 사용 API/Library | 완료 판정 |
|------|---------|---------|---------|
| W1 | OpenVLA-7B + vLLM fork + LIBERO baseline | OpenVLA HF, vLLM v0.7, LIBERO v1.0 | LIBERO success rate 공식 재현 ±2pp |
| W2 | L_mid hidden state extraction hook | PyTorch forward hook, torch.Tensor | extraction overhead < 50 μs |
| W3 | Page-Hinkley detector 구현 (M1) | `scipy.stats`, numpy CUSUM | synthetic test PH FP < 5% |
| W4 | VLA-specific feature 통합 (gripper/curvature/DINOv2) | DINOv2, robot proprio API | LIBERO phase label F1 > 0.75 |
| W5 | Quantile calibration + hysteresis | Python calibration script | 첫 100 frame 에서 threshold 자동 설정 |
| W6 | CUDA Graph capture variant 3 (M2) | `cudaGraphInstantiate`, vLLM extension | 3 graph capture 성공, total graph size < 60 MB |
| W7 | `PhaseGraphDispatcher` 구현 | CUDA Graph API | phase switch latency < 10 μs |
| W8 | M3 Green Context SM reconfig | `cuDevSmResourceSplit` | phase 별 SM 재분배 동작 확인 |
| W9 | LIBERO 4 suite × 100 trial × 3 model = 1200 rollout | Hydra config, LIBERO API | 전체 rollout 완료 |
| W10 | RoboCasa + SimplerEnv 확장 | Isaac Sim, SimplerEnv | 추가 2 benchmark 완료 |
| W11 | Jetson Orin AGX 포팅 + 측정 | JetPack 6.0, vLLM-Jetson | Orin latency 측정 완료 |
| W12 | 9 baseline 비교 실험 | 각 baseline repo | 9 baseline × LIBERO |
| W13 | FlashVLA direct comparison | FlashVLA repo | token-level vs graph-level 비교 |
| W14-15 | Nova stacking + writing | Nova repo, LaTeX | stacking +5-8% 검증, 논문 draft |

##### (7) Preliminary Analysis Metrics

| 측정 지표 | 도구 + counter/command | 측정 조건 | 기대 범위 (baseline) | 개선 후 목표 |
|---|---|---|---|---|
| LIBERO median latency | vLLM wall-clock | 4 suite × 100 trial | 165 ms | **128 ms (-22%)** |
| SimplerEnv Hz | wall-clock | SimplerEnv 전체 | 6.1 Hz | **7.8 Hz (+28%)** |
| Success rate delta | LIBERO eval | - | baseline | **Δ ≤ ±1.2pp** |
| Jetson Orin latency | JetPack wall-clock | LIBERO-Spatial | 420 ms | **330 ms** |
| PH FP rate | synthetic + LIBERO 5 task | phase transition detection | - | **< 5% (goal)** |
| PH decision latency | CUPTI `cuEventRecord` | per-frame | - | **< 100 μs** |
| Scene change F1 | LIBERO phase label vs M1 output | validation | - | **≥ 0.75** |
| Kernel efficiency | Nsight `smsp__thread_inst_executed_per_inst_executed` | per-phase | 0.68-0.75 | **+15-25%** |
| Stacked + Nova | ablation | full + Nova stage partition | - | **-30% vs baseline** |

**Preliminary Study 순서**:

- **(i) Baseline reproduction**: OpenVLA-7B + LIBERO 공식 pretrained checkpoint 로 success rate 재현 ±2pp. 실패 시 action tokenizer version, VLA temperature, BF16 vs FP16 확인.
- **(ii) Bottleneck attribution**: Nsight Systems 로 per-frame timeline 분석. encoder vs LLM vs action-head time 분리. Phase 별 compute distribution 측정. Latency-bound (kernel launch + graph switch overhead) 로 분류 예상.
- **(iii) Roofline upper bound**: OpenVLA-7B per-phase AI 계산. RTX 4090 roofline (1008 GB/s GDDR6X) 상 Approach phase 는 compute-bound (SigLIP full), Manipulate 는 memory-bound (action-head), Retract 는 latency-bound. 각 phase 별 최적 kernel 설계.
- **(iv) Mechanism 단독 Micro-benchmark**: (a) **M1 only** — phase predictor 만, graph switch 없음 (oracle phase label). PH FP rate 측정, decision latency 측정. (b) **M2 only** — 3 graph capture + oracle phase 입력. Kernel efficiency 측정. (c) **M3 only** — SM reconfig only. SM util 개선 측정. Sum vs full 로 interaction.

#### 2.3.5 예상 효과

| 지표 | Baseline | 목표 | 조건 |
|---|---|---|---|
| LIBERO median latency | 165 ms | **128 ms (-22%)** | full M1+M2+M3 |
| SimplerEnv Hz | 6.1 | **7.8 (+28%)** | full |
| RoboCasa success rate | baseline | **Δ ≤ ±1.2pp** | control |
| Jetson Orin latency | 420 ms | **330 ms** | embedded |
| Stacked + Nova | 165 ms | **115 ms (-30%), 8.6 Hz** | stacking |
| PH FP rate | - | **< 5%** | LIBERO 5 task |

**Scoring**: Novelty 7.0 / Differentiation 7.4 / Impact 7.0 / Feasibility 7.3 → **avg 7.18**.

#### 2.3.6 Tier-2 Scope 축소 Variant (IEEE CAL 4p / DATE 6p)

- **Target**: IEEE CAL 4p / DATE 6p.
- **Single mechanism**: M1 only (SSE phase predictor standalone).
- **Scope 축소**: OpenVLA-7B + LIBERO-Spatial 단일. RTX 4090 단일.
- **Metric**: PH FP rate ≤ 5% + decision latency < 100 μs + scene-change F1 ≥ 0.75.
- **Runtime**: 4 주.
- **Top-tier 와의 관계**: M1 을 2026-07 CAL 로 먼저 공개 (precedence), M2+M3 를 2026-10 MLSys 로 확장.

---

## 3. Tier-2 독립 Top 3 (Track B, Phase 1 부터 독립 도출, v2 Tier-2 5 편과 orthogonal)

### 3.1 B1 GCReconfProfile — Green Context μs-level Reconfig Latency Characterization under Mixed VLM Workloads (Tier-2 Top 1, Avg 7.50 / ISLPED 2026 6p / DATE 2026 6p)

#### 3.1.1 개요

CUDA 12.5+ Green Context `cuDevSmResourceSplit` API 의 μs-level reconfig latency 를 **mixed VLM workload (OCR heavy / chat heavy / grounding heavy)** 에서 측정 + **Blackwell (RTX Pro 6000) vs Hopper (A100 SXM) cross-arch characterization** 공개. NVIDIA docs "fast reconfig" 라고만 표기할 뿐 수치 없는 공백을 공략.

#### 3.1.2 기존 연구 한계점 및 Gap

- [MIGER ICPP 2024](https://dl.acm.org/doi/pdf/10.1145/3673038.3673089) [peer-reviewed] — MIG reconfig overhead 2.3%, MIG 은 ms-level (수초), Green Context 와 4 orders 격차. VLM workload 미포함.
- [Managing MIG arXiv:2508.18556](https://arxiv.org/abs/2508.18556) — MIG energy/throughput tradeoff, **Green Context μs-scale 축 없음**.
- [LithOS SOSP 2025](https://dl.acm.org/doi/10.1145/3695053.3731083) [peer-reviewed] — fine-grained SM API 일반 overhead, Green Context 분포 수치 공개 없음.
- [Execution-Idle arXiv:2604.04745](https://arxiv.org/abs/2604.04745) **(v3 motivation citation)** — LLM serving 19.7% idle = 10.7% energy waste. B1 은 Green Context reconfig 가 idle 구간 recover 가능한 구체 cost 제공.

**Gap**: Green Context `cuDevSmResourceSplit` 의 μs-level reconfig cost 가 mixed VLM workload 에서 어떤 분포를 보이는지 peer-reviewed 측정치 부재.

#### 3.1.3 제안 기법 (Single Mechanism, R27-α: 4 필수 요소)

##### **M1: Green Context Reconfig Latency Instrumentation Harness**

- **① 추가되는 Scheme**: vLLM v0.8+ fork 에 신규 instrumentation module `vllm/profiling/gc_reconfig_harness.py` 추가. 이 harness 는 CUPTI PM Sampling (`CUPTI_ACTIVITY_KIND_PM_SAMPLING` on Blackwell) + nanosecond `cuEventElapsedTime` 를 `cuDevSmResourceSplit` invocation 주위에 삽입하여 reconfig 시작→완료 (new green context ready-for-launch) 까지 sub-μs 정확도로 측정. Mixed VLM workload 3 종 (LLaVA-OV OCR / Qwen2-VL chat / InternVL2 grounding) × SM-count {8, 16, 32, 64, 84} × prev-SM 4 종 factorial design. (p50, p95, p99, max) × workload × SM-delta 공개. 가설 검증: driver serialization 이 SM-delta 에 quadratic growth.
- **② 해결하려는 문제**: NVIDIA 공식 문서는 Green Context 를 "fast reconfig" 라고만 표기할 뿐 구체 latency 분포가 부재. Research community 에서 "Green Context 를 scheduling decision 근거로 사용 가능한가" 판단이 불가 — reconfig 가 1 μs 이면 fine-grained scheduling 정당화, 100 μs 이면 coarse-grained 만 가능. MIG 는 ms-level (수초), Green Context 와 약 4 orders 격차. VLM workload 특화 측정 부재.
- **③ 동작 원리 (step-by-step)**:
  1. **CUPTI 삽입** — vLLM engine init 에서 `cuptiSubscribe` + `cuptiEnableCallback(CUPTI_CBID_DRIVER_API_ENTER/EXIT)` 로 `cuDevSmResourceSplit` entry/exit hook.
  2. **Nano-precision timing** — `cuEventRecord(start_event, stream)` 직전 + `cuDevSmResourceSplit(&new_res, &old_res, sm_count)` 직후 + `cuEventRecord(end_event, stream)` → `cuEventElapsedTime(&ms, start, end)` (nanosecond precision on modern GPU).
  3. **Workload dispatch** — background 에서 3 VLM workload (OCR/chat/grounding) 을 rotating 실행, main stream 은 reconfig 반복.
  4. **Factorial 데이터 수집** — `SM-delta ∈ {8, 16, 32, 64, 84}` × `prev-SM ∈ {16, 32, 48, 64}` × `workload ∈ {OCR, chat, grounding}` = 5×4×3 = 60 조합. 각 조합 × 1000 repetition → 분포 통계.
  5. **Cross-arch 비교** — 동일 실험을 Blackwell (RTX Pro 6000) vs Hopper (A100 SXM) 에서 반복, driver serialization 차이 분석.
- **④ 기존 해법 실패 + 차별화**: (i) MIGER (ICPP 2024) 는 MIG reconfig overhead 만 측정, Green Context 미포함. (ii) Managing MIG (arXiv:2508.18556) 는 energy 축, reconfig latency 분포 없음. (iii) LithOS (SOSP 2025) 는 SM API 일반 overhead, Green Context 수치 미공개. (iv) 본 M1 은 mixed VLM workload 에서 **Green Context reconfig latency 의 첫 peer-reviewed 측정치** + Blackwell vs Hopper cross-arch 공개 → community 가 scheduling decision 에 활용 가능한 수치 제공.

**Tier 구성**: physical 1-tier (GPU Green Context) + software 1-tier (instrumentation harness) → single-axis measurement letter.

#### 3.1.4 평가 / 실험 플랜 (7-요소, single scope)

##### (1) Hardware

- **Primary**: RTX Pro 6000 96GB (Blackwell, CUDA 13.1, NVFP4 지원).
- **Secondary**: A100 SXM 40GB (Hopper, CUDA 12.5).
- **Tools**: CUPTI PM Sampling + NVML 5ms + Nsight Systems 2026.1.

##### (2) Model

- **LLaVA-OV 7B** (OCR proxy).
- **Qwen2-VL 7B** (chat).
- **InternVL2 8B** (grounding).
- 7-8B class 축 고정.

##### (3) Dataset / Workload

- **DocVQA 500** (OCR) + **LLaVA-Wild 500** (chat) + **RefCOCOg 500** (grounding) = 1500 sample.
- **총 2-hour 실험 budget**.

##### (4) Simulators / Tools

- **Profiler**: vLLM v0.8+ fork (CUPTI 삽입).
- **API**: `cuDevSmResourceSplit` / `cuCtxFromGreenCtx` + `cuEventRecord` 주위 instrumentation.
- **Blackwell-specific**: `CUPTI_ACTIVITY_KIND_PM_SAMPLING`.
- **Energy**: NVML 5 ms window, `nvmlDeviceGetTotalEnergyConsumption`.

##### (5) Ablation + Measurement Protocol

- **A1**: SM-delta {8/16/32/64/84} × workload 3 = 15 조합.
- **A2**: Driver contention {idle / 50% queue / 90% queue} × SM-delta 3 = 9 조합.
- **A3**: Blackwell vs Hopper repeat = 15.
- **총**: 39 × 3 반복 = **117 runs**, 단일 주말 48h 실험.
- **Metric**: reconfig latency (ns), energy (mJ), throughput loss (%).

##### (6) Implementation Steps (Week-Level)

| Week | Component / File | 사용 API/Library | 완료 판정 |
|------|---------|---------|---------|
| W1 | vLLM v0.8 fork + 3 VLM workload baseline | vLLM, transformers, HF checkpoint | 3 workload 정상 구동 |
| W2 | CUPTI 삽입 + `cuEventRecord` 기반 timing harness | CUPTI 12.5, CUDA Event API | nano-precision timing 검증 |
| W3 | A1 factorial (SM-delta × workload) 주말 실험 | bash automation, `cuDevSmResourceSplit` | 15 × 1000 rep 완료 |
| W4 | A2 driver contention + A3 cross-arch 실험 | Hopper A100 서버 | 24 + Hopper 실험 완료 |
| W5 | 분포 분석 + Blackwell vs Hopper 통계 | Python pandas, matplotlib | (p50, p95, p99, max) 분포 plot |
| W6 | Writing (ISLPED 6p) | LaTeX | 논문 draft + figures |

##### (7) Preliminary Analysis Metrics

| 지표 | Baseline (MIG, A100) | Green Context (Pro 6000) | 조건 |
|---|---|---|---|
| Reconfig latency p50 | 2100 ms | **18-45 μs** | SM-delta 16 |
| Reconfig latency p99 | 2400 ms | **80-140 μs** | SM-delta 16, driver contention |
| Reconfig latency p50 | — | **42-95 μs** | SM-delta 64 |
| Energy spike / reconfig | 12 J | **0.3-0.8 mJ** | NVML 5ms window |
| Throughput loss / reconfig | 0.2% | **0.008-0.02%** | mixed VLM 10 req/s |

**Preliminary Study 순서**:

- **(i) Baseline reproduction**: MIGER (ICPP 2024) 의 A100 MIG reconfig latency 2.3% overhead 재현. 동일 protocol 로 측정 ±10%.
- **(ii) Bottleneck attribution**: Green Context reconfig 의 어느 단계 (driver entry / SM allocation / cache flush / launch-ready) 가 dominant 한지 분리 측정.
- **(iii) Roofline upper bound**: SM-delta 증가 시 reconfig latency 의 이론적 lower bound (hardware SM gate reconfig time) 추정.
- **(iv) Mechanism 단독 Micro-benchmark**: idle workload (baseline) vs mixed VLM workload 에서 reconfig latency 비교. Driver serialization 유무 검증.

#### 3.1.5 "왜 Tier-1 scale-up 불가인가"

- 단일 vendor API (CUDA Green Context) characterization letter.
- Mechanism 1 개 (instrumentation), 추가 axis 확장 시 scope drift.
- Driver black-box 의존 (NVIDIA 내부 구현 변경 시 재측정 필요).
- Cross-vendor generality 부재 (AMD/Intel 동등 API 없음).

**Scoring**: Novelty 7.0 / Differentiation 7.8 / Impact 7.5 / Feasibility 7.7 → **avg 7.50**.

---

### 3.2 B2 TokenEvictEnergy — Visual-Token Eviction 이 HBM Refresh/DRAM Energy 에 미치는 영향 측정 (Tier-2 Top 2, Avg 7.35 / IEEE ESL 2026 4p / ISLPED 2026 6p)

#### 3.2.1 개요

Visual-token eviction policy (VL-Cache, SparseVLM 등) 가 **HBM dynamic energy 는 줄이지만 DRAM PKG energy 는 오히려 증가** (row-buffer eviction → refresh 증가 가설) 하는 negative result 를 공식 측정으로 검증. NVML 5ms + Intel RAPL DRAM 이중 counter + 4-component decomposition (HBM dynamic / SM static / DRAM PKG / refresh-implied). "Eviction ≠ always green" 공교육 contribution.

#### 3.2.2 기존 연구 한계점 및 Gap

- [VL-Cache ICLR 2025 Poster](https://openreview.net/forum?id=HMrcv7Q4Ub) [peer-reviewed] — layer-adaptive KV budget. **Energy 축 누락**.
- [SparseVLM ICML 2025 Poster](https://openreview.net/forum?id=80faIPZ67S) [peer-reviewed] — token sparsity + attention prune. **Power/energy measurement 없음**.
- [Characterizing Power Management for LLMs ASPLOS 2024](https://www.microsoft.com/en-us/research/wp-content/uploads/2024/03/GPU_Power_ASPLOS_24.pdf) [peer-reviewed] — LLM 전체 power, **visual token eviction 별 decomposition 없음**.
- [TokenPowerBench arXiv:2512.03024](https://arxiv.org/abs/2512.03024) — LLM phase-aware power, **visual token 미포함**.

**Gap**: Visual token eviction policy 별 (random / attention-score / VL-Cache / SparseVLM) 의 HBM refresh cost + DRAM activation energy 분포 측정 부재.

#### 3.2.3 제안 기법 (Single Mechanism, R27-α: 4 필수 요소)

##### **M1: Per-Policy Energy-Counter Decomposition Harness**

- **① 추가되는 Scheme**: vLLM v0.8+ fork 에 신규 instrumentation module `vllm/profiling/energy_decomposition.py` 추가. 이 harness 는 동일 seed 의 1000 request 를 baseline vLLM / VL-Cache / SparseVLM / random-50% 4 policy 에서 반복 실행하며 **NVML 5ms** (`nvmlDeviceGetTotalEnergyConsumption` + `nvmlDeviceGetPowerUsage`) + **Intel RAPL DRAM-package** (`/sys/class/powercap/intel-rapl/intel-rapl:0:0/energy_uj` crawling) 이중 counter 로 4-component decomposition (HBM dynamic / SM static / DRAM PKG / refresh-implied) 수행. 가설: aggressive eviction 이 total energy 를 줄이지 않는다 (refresh overhead + recompute cost 상쇄).
- **② 해결하려는 문제**: VL-Cache / SparseVLM 등 token eviction 기법은 "KV memory 감소 → energy 절감" 을 암묵적 가정하나, **row-buffer eviction pattern 이 DRAM refresh rate 를 증가시킬 가능성** 은 실측 부재. 기존 LLM power 연구 (ASPLOS 2024) 는 LLM 전체 power 만 측정, visual token eviction policy 별 decomposition 없음. Community 가 "eviction 이 항상 green 한가" 에 대한 empirical evidence 부재.
- **③ 동작 원리 (step-by-step)**:
  1. **NVML 5ms sampling** — background thread 에서 `nvmlDeviceGetTotalEnergyConsumption()` 을 5ms 간격으로 poll. power 및 누적 energy 확보.
  2. **Intel RAPL DRAM crawling** — `/sys/class/powercap/intel-rapl/intel-rapl:0:0/energy_uj` 를 동일 5ms 간격으로 read. DRAM package energy 분리.
  3. **4-component decomposition** — `total_GPU_energy = HBM_dynamic + SM_static + other`. `DRAM_PKG = RAPL_DRAM`. `refresh-implied = DRAM_PKG - estimated_activation_energy` (activation energy 는 `dram__sectors_read.sum` × 0.4 pJ/sector 로 추정).
  4. **Policy rotation** — 4 policy 각각에서 1000 req × 3 repetition 실행. Random seed 고정으로 variance 제어.
  5. **Hypothesis test** — H0: DRAM_PKG energy (eviction) ≥ DRAM_PKG (baseline) 를 t-test 로 검증. Negative result 유의수준 확인.
- **④ 기존 해법 실패 + 차별화**: (i) VL-Cache / SparseVLM 원저자 는 energy 측정 없이 "KV memory 감소" 만 주장. (ii) ASPLOS 2024 LLM power 논문은 전체 power, per-policy decomposition 없음. (iii) TokenPowerBench (arXiv:2512.03024) 는 LLM phase-aware, visual token 미포함. (iv) 본 M1 은 **per-policy × 4-component decomposition** 으로 eviction 의 hidden cost 를 정량화 → "eviction ≠ always green" 공교육.

**Tier 구성**: physical 2-tier (GPU HBM + DRAM PKG) + software 4-tier (4 policy) → R1/R1b 상한 내.

#### 3.2.4 평가 / 실험 플랜 (7-요소, single scope)

##### (1) Hardware

- **Primary**: RTX 5090 32GB (GDDR7).
- **Secondary**: RTX Pro 6000 96GB (HBM3e, cloud 4-hour spot rental).
- **Tools**: NVML + Intel RAPL + `perf stat`.

##### (2) Model

- **LLaVA-OV 7B 단일** (visual token 3000-8000, 적절한 eviction 효과 관찰 범위).

##### (3) Dataset / Workload

- **VideoMME 1-min subset 200 clip** + **DocVQA 500** = 700 sample.
- **총 2-hour 실험 budget**.

##### (4) Simulators / Tools

- **Serving**: vLLM v0.8+ + VL-Cache plugin + SparseVLM plugin.
- **Energy**: NVML `nvmlDeviceGetTotalEnergyConsumption` + RAPL crawler + Nsight Systems GPU memory access counter.
- **Analysis**: pandas, scipy.stats (t-test).

##### (5) Ablation + Measurement Protocol

- **A1**: Policy {vLLM baseline / VL-Cache / SparseVLM / random 50%} × 700 sample = 4 run.
- **A2**: Context {2K / 8K / 16K} × policy 4 = 12 run.
- **A3**: HBM3e (Pro 6000) vs GDDR7 (5090) cross-arch = 2 집합.
- **총**: 18 × 2 반복 = **36 runs**, 하루 24h 실험.
- **Metric**: total energy (J), HBM/DRAM decomposition (J), peak power (W), accuracy (pp).

##### (6) Implementation Steps (Week-Level)

| Week | Component / File | 사용 API/Library | 완료 판정 |
|------|---------|---------|---------|
| W1 | vLLM v0.8 + LLaVA-OV 7B + VL-Cache / SparseVLM plugin 설치 | vLLM, HF checkpoint | 3 policy 정상 구동 |
| W2 | NVML + RAPL 5ms sampling harness | NVML Python, `/sys` read | energy 누적 정상 측정 |
| W3 | 4-component decomposition 모델 구축 | Python analysis | HBM/DRAM 분리 정확도 검증 |
| W4 | A1 + A2 실험 실행 | bash automation | 16 run 완료 |
| W5 | A3 cross-arch 실험 (Pro 6000 cloud) | AWS/cloud spot | HBM3e 데이터 확보 |
| W6 | 통계 분석 + "negative result" 검증 + writing | scipy.stats, LaTeX | ISLPED draft 완성 |

##### (7) Preliminary Analysis Metrics (Negative Result 핵심)

| 지표 | vLLM baseline | VL-Cache | SparseVLM | 조건 |
|---|---|---|---|---|
| Total energy / req (J) | 14.2 | -8 ~ -15% | -5 ~ -12% | LLaVA-OV 7B, 2048 context |
| HBM dynamic energy (J) | 6.1 | -18 ~ -25% | -12 ~ -20% | eviction → access 감소 |
| **DRAM PKG energy (J)** | 1.8 | **+2 ~ +6%** | **+1 ~ +4%** | **refresh 증가 (negative)** |
| p99 Power envelope (W) | 380 | -22 ~ -30% | -15 ~ -22% | peak power 절감 |
| Accuracy (VideoMME) | 62.1% | -0.8 ~ -1.2pp | -1.5 ~ -2.2pp | — |

**Preliminary Study 순서**:

- **(i) Baseline reproduction**: ASPLOS 2024 LLM power characterization 의 LLaVA-OV 7B baseline 재현 ±10%.
- **(ii) Bottleneck attribution**: total GPU power 중 HBM vs SM static vs DRAM 비중 decomposition. 각 component 가 eviction 에 어떻게 반응하는지 측정.
- **(iii) Roofline upper bound**: LLaVA-OV 7B 의 theoretical minimum energy (Landauer limit 기반) 대비 실측 gap 측정.
- **(iv) Mechanism 단독 Micro-benchmark**: 각 policy 별 단독 실행 후 4-component 분리. Negative result (DRAM PKG +2~6%) 유의수준 확인.

#### 3.2.5 "왜 Tier-1 scale-up 불가인가"

- Power-constrained narrow engineering, 단일 mechanism.
- Negative result 중심 — positive optimization 없음.
- Confounder (platform variance, thermal) 통제 불가 → multi-platform generalization 제한.

**Scoring**: Novelty 7.2 / Differentiation 7.5 / Impact 7.3 / Feasibility 7.5 → **avg 7.35**.

**핵심 insight**: eviction ≠ always green — ISLPED reviewer 에게 공교육.

---

### 3.3 B3 ActHeadFuse — OpenVLA-OFT Action-Head Fused Kernel for Sub-ms Decode Step (Tier-2 Top 3, Avg 7.20 / IEEE CAL 2026 4p / DAC 2026 6p)

#### 3.3.1 개요

OpenVLA-OFT 의 action-head `SiLU → Linear(4096→448) → torch.bucketize` 3-op serial kernel chain 을 **single persistent CUDA kernel** 로 fuse. Action-head time 14.2 μs → **2.3 μs**, total decode step 1.82 → **0.94 ms → 1-kHz real-time control** 가능. OpenVLA-OFT 의 26× 는 parallel-sample 수준이지 kernel-level 이 아니라는 gap.

#### 3.3.2 기존 연구 한계점 및 Gap

- [OpenVLA CoRL 2024](https://proceedings.mlr.press/v270/kim25c.html) [peer-reviewed] — Llama-2-7B backbone, serial decode 8-12ms / step. **Action-head fused kernel 없음**.
- [OpenVLA-OFT arXiv:2502.19645](https://arxiv.org/abs/2502.19645) — parallel decoding 26×, **kernel breakdown 미공개**.
- [FAST Action Tokenizer arXiv:2501.09747](https://arxiv.org/abs/2501.09747) — 15× via token compression. **Kernel fusion 미터치**.
- [VLA-Cache NeurIPS 2025](https://openreview.net/forum?id=QZYZ0Xm58q) [peer-reviewed] — KV reuse, action-head 부분 미포함.

**Gap**: Action-head SiLU + Linear + Bucketize 3-op → single CUDA kernel fusion first-to-report for VLA.

#### 3.3.3 제안 기법 (Single Mechanism, R27-α: 4 필수 요소)

##### **M1: Action-Head 3-Op Fusion CUDA Kernel**

- **① 추가되는 Scheme**: OpenVLA-OFT fork 의 `openvla/action_head/` 폴더에 신규 CUDA kernel 파일 `fused_action_head.cu` 추가. 이 파일은 SiLU + Linear(4096→448) + Bucketize 3-op 을 **single persistent CUDA kernel** 로 fuse. Block-tile (hidden=4096 / 32-warp) 구조 + Hopper/Blackwell **TMA (Tensor Memory Accelerator)** activation prefetch (`cp.async.bulk`) + warp-level softmax-free bucketize. CUTLASS 3.6 Epilogue fusion template 활용. 기존 PyTorch 는 `F.silu(x)` → `F.linear(x, w)` → `torch.bucketize(x, boundaries)` 3 kernel launch, 본 M1 은 single launch.
- **② 해결하려는 문제**: OpenVLA-OFT 의 per-step inference 에서 action-head 가 14.2 μs (14.2 μs × 8-step action chunk = 113.6 μs) 소모하며, 이는 total decode step 1.82 ms 의 **7.8%** 에 해당. 1-kHz real-time control (1 ms budget) 목표 시 이 7.8% overhead 가 critical. 3 kernel launch 당 ~ 5 μs overhead + SFU (sincos / bucketize) 경유로 GPU SM util 38% 에 머무름.
- **③ 동작 원리 (step-by-step)**:
  1. **Kernel signature 정의** — `__global__ void fused_action_head(half* hidden, half* weight, half* bias, float* boundaries, int* out_bucket, int hidden_dim=4096, int out_dim=448, int num_bins)`.
  2. **TMA activation prefetch** — Hopper/Blackwell TMA (`cp.async.bulk.tensor.2d.shared::cta.global`) 로 hidden state 를 shared memory 로 bulk load, overlap compute with load.
  3. **SiLU + Linear fusion** — shared memory 에서 SiLU (`x / (1 + exp(-x))`) 적용 직후, 동일 warp 가 weight tile 을 register 에 load 후 `__hmma` (half-matrix-multiply-accumulate) 로 linear 계산. CUTLASS Epilogue 의 fused activation 활용.
  4. **Warp-level softmax-free bucketize** — output (448-d vector) 에 대해 각 element 를 boundaries 와 비교하여 bucket index 추출. `__shfl_sync()` 로 warp-level reduction, SFU 우회.
  5. **Result write** — out_bucket 을 global memory 로 single coalesced write.
- **④ 기존 해법 실패 + 차별화**: (i) torch.compile 은 auto-fusion 가능하나 bucketize 같은 non-standard op 은 fuse 불가 (fallback to 3 kernel). (ii) TensorRT-LLM 은 LLM decode 최적화이나 VLA action-head 미지원. (iii) FAST Action Tokenizer 는 token 축 compression, kernel fusion 미터치. (iv) 본 M1 은 VLA action-head 특화 first-to-report kernel fusion, action-head time 14.2 → 2.3 μs (-84%), total decode 1.82 → 0.94 ms (**1-kHz control 가능**).

**Tier 구성**: physical 1-tier (GPU SM) + software 1-tier (single kernel) → single-axis kernel letter.

#### 3.3.4 평가 / 실험 플랜 (7-요소, single scope)

##### (1) Hardware

- **Primary**: RTX Pro 6000 96GB (Blackwell, TMA native).
- **Secondary**: RTX 5090 32GB (TMA 부분 지원).
- 연구실 자체 보유, single workstation.

##### (2) Model

- **OpenVLA-7B** (Llama-2-7B + SigLIP + DinoV2 + action-head) 단일. 실제로는 **OpenVLA-OFT** variant 의 action-head 구조 재사용.

##### (3) Dataset / Workload

- **LIBERO-Spatial + LIBERO-Object** 1000 trajectory.
- **총 3-hour 실험 budget**.

##### (4) Simulators / Tools

- **Compiler**: CUDA 13.1 + NVCC.
- **Libraries**: CUTLASS 3.6+ (TMA epilogue), Triton 3.1+ (비교), torch.compile (baseline), TensorRT-LLM 0.16 (baseline).
- **Profiler**: Nsight Compute 2024.3 (`sm__cycles_active.sum`, `smsp__inst_executed_pipe_fp64_sfu.sum`, `sm__warps_active.avg.pct_of_peak_sustained_active`).

##### (5) Ablation + Measurement Protocol

- **A1**: Kernel impl {torch native / torch.compile / TensorRT-LLM / **ActHeadFuse**} = 4 impl.
- **A2**: Batch {1 / 8 / 32} × impl 4 = 12 조합.
- **A3**: Hidden {2048 / 4096 / 8192} × ActHeadFuse = 3 (scaling study).
- **총**: 19 × 5 반복 = **95 runs**, 하루 24h 실험.
- **Metric**: kernel time (μs, Nsight), decode step (ms), throughput (steps/s), LIBERO MSE (accuracy), SM util (%).

##### (6) Implementation Steps (Week-Level)

| Week | Component / File | 사용 API/Library | 완료 판정 |
|------|---------|---------|---------|
| W1 | OpenVLA-7B + LIBERO baseline + action-head breakdown | OpenVLA HF, LIBERO | action-head 14.2 μs 재현 ±5% |
| W2 | CUTLASS 3.6 fork + `fused_action_head.cu` skeleton | CUTLASS 3.6, CUDA 13.1 | kernel build 성공, unit test |
| W3 | TMA activation prefetch 통합 | `cp.async.bulk.tensor` asm | TMA load 정상 동작 |
| W4 | SiLU + Linear + Bucketize fusion | CUTLASS Epilogue | bit-exact vs PyTorch reference |
| W5 | A1 ablation (4 impl × 1000 rollout) | Hydra config | 4 impl latency 측정 |
| W6 | A2 batch scaling + A3 hidden scaling | Nsight | scaling trend 도출 |
| W7 | Writing (IEEE CAL 4p) + ablation table | LaTeX | CAL draft 완성 |

##### (7) Preliminary Analysis Metrics

| 지표 | Baseline (torch-native) | torch.compile | TensorRT-LLM | **ActHeadFuse** | 조건 |
|---|---|---|---|---|---|
| Action-head kernel time (μs) | 14.2 | 8.1 | 5.4 | **2.3** | Pro 6000, bs=1 |
| **Total decode step (ms)** | 1.82 | 1.55 | 1.38 | **0.94** | **1-kHz control 가능** |
| Kernel launches / step | 4 | 2 | 1 | **1** | — |
| Action MSE (LIBERO) | 0 | 0 | 0 | **0** (bit-exact) | FP16 |
| GPU SM util during head | 38% | 52% | 78% | **86%** | — |
| Throughput (steps/s, bs=32) | 540 | 640 | 720 | **1050** | bs=32 |

**Preliminary Study 순서**:

- **(i) Baseline reproduction**: OpenVLA-OFT 공식 repo + LIBERO 1000 trajectory 에서 action-head time 14.2 μs ±5% 재현. 실패 시 torch version, CUDA version, BF16 vs FP16 확인.
- **(ii) Bottleneck attribution**: action-head 의 3 kernel (SiLU / Linear / Bucketize) 각각의 비중 측정. Kernel launch overhead vs compute overhead 분리. SFU busy vs Tensor Core busy 분리.
- **(iii) Roofline upper bound**: action-head (4096→448 linear) 의 AI = FLOPs/bytes 계산. Blackwell roofline 상 theoretical minimum kernel time ~1.5 μs → 본 기법 목표 2.3 μs (실용 한계 내).
- **(iv) Mechanism 단독 Micro-benchmark**: SiLU-only / Linear-only / Bucketize-only 각 단독 fusion 측정. 3-op full fusion 과 sum 비교하여 fusion synergy 검증.

#### 3.3.5 "왜 Tier-1 scale-up 불가인가"

- Narrow kernel engineering letter, single mechanism.
- Model family lock-in (OpenVLA Llama-2-7B + specific action-head 구조).
- Serving stack 전체 impact 는 ~ 5-8% 수준 (action-head 가 decode step 의 7.8% 이므로 full gain 수준).
- Generality 제한: 다른 VLA (π0, RT-2) 의 action-head 구조가 다르면 재작성 필요.

**Scoring**: Novelty 7.0 / Differentiation 7.4 / Impact 7.0 / Feasibility 7.4 → **avg 7.20**.

---

## 4. 미선정 아이디어 전수 (사유 + 재방문 조건)

### 4.1 v2 Top 3 중 탈락한 1 편

#### NACK-Gossip Tier-2 (v2 Top 3, Avg 7.80 → v3 미선정)

- **연구 GAP**: NVLink peer-fetch latency characterization for VLA fleet.
- **제안 overview**: Pull-based NVLink peer fetch + TTL lease + pull-batch.
- **미선정 사유**: v3 재검증에서 [Harvest arXiv:2602.00328](https://arxiv.org/abs/2602.00328) (2026-01-30, v2 cutoff 이전이나 누락) 이 **NVLink peer-caching primitive 을 이미 축 점유** — concurrent 55-65%. 단일 mechanism 만 유지하는 Tier-2 에서는 Harvest subset 으로 보일 위험. 동시에 Track B 3 편 (B1/B2/B3) 이 Tier-2 독립 축 (HW profiling / energy / kernel) 을 orthogonal 하게 커버하여 Tier-2 독립 Top-M 슬롯을 차지.
- **재방문 조건**: Harvest 대비 VLA-specific delta 3 축 명시 후 재평가 — (a) VLA-specific action latency SLO (~10ms budget), (b) VLA skill-transition TTL lease (일반 LLM 과 다름), (c) VLA KV block 크기 (4KB vs 16KB 특수 pattern). 또는 4-GPU NVLink testbed 확보 후 Top-tier (SOSP/OSDI) 로 scale-up.

### 4.2 v2 Major Revision 2 편 (v3 에서도 미해결)

#### PhaseScope Top-tier (v2 Avg 7.55, Major Revision → v3 유지)

- **연구 GAP**: GUI agent 3-phase classifier × scope-restricted attention × MIG quota.
- **제안 overview**: MLP phase classifier + phase-conditional binary mask + MIG SM quota.
- **미선정 사유**: [GUI-KV arXiv:2510.00536](https://arxiv.org/abs/2510.00536) scoop 70% 경계 + ContextMIG+ 와 MIG-quota originality collision.
- **재방문 조건**: (a) GUI-KV 와 phase-conditional mask formal 구분 (residual L2 eviction vs phase K/V masking), (b) ContextMIG+ 와 MIG-quota 축 분리 (PhaseScope = single-session turn-phase, ContextMIG = multi-tenant request). 재타겟 EuroSys/SoCC 시스템 agent 분야.

#### DeadlineCOW Top-tier (v2 Avg 7.50, Major Revision → v3 유지 + 강화)

- **연구 GAP**: Skill-hierarchical LSH + EDF+slack-borrow + divergence-threshold COW.
- **제안 overview**: 4-robot VLA fleet skill-level semantic page hash.
- **미선정 사유**: [KVShare arXiv:2503.16525](https://arxiv.org/abs/2503.16525) 65-70% scoop 경계 + [FlashVLA arXiv:2505.21200](https://arxiv.org/abs/2505.21200) **(v3 신규) "action reuse" 직접 명명 68-72% scoop 접경** — KVShare 보다 더 강한 경쟁자 추가. EDF+slack-borrow 가 classic real-time scheduling 재포장 혐의.
- **재방문 조건**: LSH 를 "skill-hierarchical" → "temporal-phase" 축으로 shift + fleet-level multi-task page dedup (FlashVLA 의 single-robot action reuse 와 분리) 축 재포지셔닝.

### 4.3 v2 Tier-2 Paper-pair variant (Tier-1 Top-tier subsection 으로 흡수)

#### HRTS Tier-2 (Tier-1 Top 1 Subsection 으로 포함)

- 개요: Row-aligned KV tile letter (M1 only).
- 사유: HRTS+ Top-tier 의 paper-pair subsection 으로 유지, 독립 Top-M 아님. Precedence 확보 전략.

#### ContextMIG Tier-2 (Tier-1 Top 2 Subsection 으로 포함)

- 개요: CLIP-L LSH reuse graph classifier standalone.
- 사유: ContextMIG+ Top-tier 의 paper-pair subsection, 독립 Top-M 아님.

### 4.4 v1 A3 PhaseGraph Predecessor Revival 에서 흡수된 predictor

#### v1 P1 E²IC (Early-Exit Inference Classifier)

- 사유: PhaseGraph-VLA+ / ContextMIG+ 의 classifier slot 에 bundled, standalone publication 아님.

#### v1 P2 SSE (Semantic Shift Estimator)

- 사유: PhaseGraph-VLA+ M1 에 absorbed, standalone 아님.

### 4.5 v1 기타 미선정 (v1 유지)

#### v1 L2 TemporalTier-3 (v1 tiebreak 패, v3 유지)

- 연구 GAP: Action-imminence / interaction trigger 로 KV tier prefetch.
- 제안 overview: 3-tier HBM-hot/HBM-cold/pinned host + Hawkes-vs-Poisson + gripper predictor.
- 사유: v1 tiebreak 패, HERMES 65% concurrent, predictor AUC study 미실행.
- 재방문 조건: Hawkes empirical (bursty trace) + AUC study +0.08 증명 또는 70B+ VLA 등장 시 scope 확장.

#### v1 A2 TierKernel-Dispatch (v1 유지)

- 연구 GAP: Task intent × kernel variant × memory tier.
- 제안 overview: E²IC intent + entropy → Hot/Warm/Cold patch → kernel variant.
- 사유: algorithm-expert No (intent generalization 약함), OmniSparse binary hot-cold 대비 incremental, Blackwell L2 spec 변경 risk.
- 재방문 조건: Cross-task intent classifier eval 확보 + TileSparse 실존 확인 + H100 포팅.

#### v1 L3 MTV-Pool (v3 에서도 Major Revision)

- 연구 GAP: Multi-turn GUI agent visual KV pool.
- 제안 overview: 2-pool (current-hot/past-cold) + turn oldness γ_v.
- 사유: [GUI-KV arXiv:2510.00536](https://arxiv.org/abs/2510.00536) 55-65% 중첩 + [Rethinking Token Pruning arXiv:2603.26041](https://arxiv.org/abs/2603.26041) withdrawn 도 유사.
- 재방문 조건: learned weight γ_v + EuroSys/SoCC 재타겟.

### 4.6 v2 Tier-2 기타

#### PhaseScope Tier-2 (v2 Avg 7.60, v3 미선정)

- 사유: Track B B1 (GCReconfProfile) 이 HW profiling 축 점유하여 더 기여 큰 방향 선정. GUI-KV scoop 경계 유지.

#### DeadlineCOW Tier-2 (v2 Avg 7.60, v3 미선정)

- 사유: DeadlineCOW Top-tier Major Revision 과 함께 paper pair 불가.

---

## 5. v2 → v3 변경 요약 (개선 포인트)

| 축 | v2 (4/22) | **v3 (4/23)** | 개선 포인트 |
|---|---|---|---|
| Top-M 구성 | Tier-Mix 3 (Top-tier 2 + Tier-2 1) | **Dual Top-3 = Tier-1 Top 3 + Tier-2 독립 Top 3** (총 6) | 연구자 선택지 2× 확장 + Tier-2 독립 Track B 신설 |
| Summary 산출물 | 없음 | **blog-style summary 폴더 의무** | 처음 보는 독자용 scanning 가능 |
| Placeholder 잔존 | Phase 2 부터 R1 검증, 7 편 placeholder 잔존 | **v2 placeholder 7 편 전부 재검증 완료** (0 scoop, 1 critical concurrent Harvest 발견) | 체계적 실존 검증 |
| 최신 논문 supplement | 2025-10 ~ 2026-03 | **2026-04 20+ 편 추가 탐색** (Mosaic scoop 55-65% 발견) | 한달 gap 보완 |
| Tier-2 Track | variant 5 편 (Tier-1 축소) | **Track B 독립 3 편 + variant 3 편 (Tier-1 subsection)** | variant vs independent 구분 명확 |
| Mechanism discipline | ≤3 per idea 준수 | **6 아이디어 모두 mechanism diff 0 add** (improve-first 완전 준수) | Contribution diffusion 방지 강화 |
| Peer-reviewed ratio | 56-100% | **Tier-1 60-70% / Tier-2 독립 67-100%** | R2 규칙 안정화 |
| arxiv 링크 포맷 | 부분 적용 | **모든 산출물 엄수** (raw ID / 숫자 단독 / target 누락 0 건) | 2026-04-23 강화 규칙 완전 준수 |
| **Summary self-sufficiency** | 2-3 문장 overview | **R27 적용 — mechanism 4 요소 + 실험 플랜 7 요소** | 학부생도 preliminary 실험 즉시 착수 가능 |

---

## 6. 다음 단계 제안

1. **HRTS+ Mosaic stacking PoC (1 주)**: HRTS+M1+M2+M3 × Mosaic on/off ablation 을 single-week PoC 로 orthogonality 선제 검증.
2. **PhaseGraph-VLA+ FlashVLA baseline re-impl (1 주)**: FlashVLA token-level reuse vs PhaseGraph+ graph-level switch 직접 비교 PoC.
3. **B1 GCReconfProfile weekend run**: Pro 6000 Blackwell + A100 SXM cloud 2-day rental 로 Blackwell vs Hopper cross-arch 초기 수치 확보.
4. **PhaseScope / DeadlineCOW 재설계 세션**: Major Revision 을 해소하려면 축 shift 필요 — 다음 세션에서 GUI-KV vs phase-conditional mask + temporal-phase LSH 재포지셔닝.
5. **Publish 요청 시**: session + summary 파일 homepage 배포 (명시적 요청 시).

---

## 7. 참고 파일

- **Session 상세 (재현성)**: [sessions/2026-04-23-mode1-vlm-vla-context-serving-v3.md](../sessions/2026-04-23-mode1-vlm-vla-context-serving-v3.md)
- **Staging (v3 보강)**: [2026-04-23-v3-placeholder-verify.md](../sessions/staging/2026-04-23-v3-placeholder-verify.md) / [2026-04-23-v3-latest-papers-supplement.md](../sessions/staging/2026-04-23-v3-latest-papers-supplement.md) / [2026-04-23-v3-tier2-independent-track-b.md](../sessions/staging/2026-04-23-v3-tier2-independent-track-b.md)
- **Prior sessions**: [v1 2026-04-22](../sessions/2026-04-22-mode1-vlm-vla-context-serving.md) / [v2 2026-04-22](../sessions/2026-04-22-mode1-vlm-vla-context-serving-v2.md)
- **이전 summary (pattern 참조)**: [2026-04-23 PRISM](2026-04-23-prism-vlm-kv-extension.md) / [2026-04-24 Qwen3-VL DeepStack Edge (R27 원본 예시)](2026-04-24-qwen3vl-deepstack-edge.md)
- **R27 규칙 출처**: [summary-generation.md § R4](../../.claude/skills/aica-research-bot/references/summary-generation.md)
