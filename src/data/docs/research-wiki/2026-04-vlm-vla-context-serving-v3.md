# VLM/VLA Context-aware Serving 최적화: 새 규칙 적용 Dual Top-3 (v3)

*Session date: 2026-04-23 · Mode 1 (v1/v2 improve)*

> v1 (Top 3 = L1/A3/A1 single-track) → v2 (Tier-Mix HRTS/ContextMIG/NACK-Gossip) → v3 (**Dual Top-3: Tier-1 Top 3 + Tier-2 독립 Top 3 = 총 6 아이디어**). 새 harness 규칙 (R21 Dual Top-3 / R22 summary 블로그-style / arxiv 링크 포맷 강화 / Track B 독립) 를 적용하여 v1/v2 에서 도출된 10 variants + v3 독립 Track B 3 편을 통합, 2026-04 Mosaic/Harvest/FlashVLA 등 신규 scoop/concurrent 3 편 반영.

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

#### 2.1.3 제안 기법 (3 Mechanisms, improve-first)

**M1 — Row-aligned KV tile (improve)**: analytical row-hit model + Nsight Compute `dram__sectors_read.sum` 측정 + tile size {128, 256, 512} sweep. HBM3 row 8KB ↔ vLLM page_size 8KB 정합 (v0.7 patch). 목표: row-hit 62% → 82-88%.

**M2 — Bi-exponential recency × salience window pin (improve)**: fixed sliding window 대신 adaptive pin. score(t) = α·exp(-β_r·(T-t)) + (1-α)·attention_salience(t), 여기서 β_r 은 recency decay, salience 는 past attention log-sum-exp. StreamingLLM (uniform sliding) 대비 adaptive.

**M3 — Async tri-tier streaming (polish)**: HBM-hot (< 24 frame window) / DRAM-pinned (<1 hour) / NVMe-cold (deep archive) 3-tier, prefetch depth {1, 2, 4} layer-ahead. Semantic Scheduling batch-level reorder 와 orthogonal.

#### 2.1.4 평가·실험 플랜 (5-요소)

| 요소 | 상세 |
|---|---|
| **Hardware** | RTX 5090 32GB (row-hit profiling, HBM3e) / RTX Pro 6000 96GB (128K+ context) / RTX 4090 × 2 (GDDR6X cross-arch) — 연구실 자체 보유. |
| **Model** | LLaVA-Video-7B (primary) / Qwen2.5-VL-7B (dynamic resolution) / InternVL3-8B (vision encoder cross). FP16 baseline + BF16 robustness. vLLM v0.7 fork + FlashAttention-3 fork (indirection pointer). |
| **Dataset/Workload** | VideoMME long subset (30-60min) + MVBench + LongVideoBench (1hr+). Row-hit microbenchmark: YouTube-8M subset 1-hour clip × 10. Scale: 500 video × 5 query + 10 × 128K token. |
| **Simulator/Tools** | Ramulator2 v2.0 (HBM3/HBM3e analytical) + DRAMSim3 cross-check. Nsight Compute (lts/dram counter) + Nsight Systems (3-stream overlap) + NVML. vLLM v0.7 `RowTilePagedAttention` 확장. |
| **Ablation/Protocol** | 2^3 factorial (M1×M2×M3) + **HRTS+Mosaic stacking ablation (v3 신규)** + tile/window/prefetch/context sweep. 10 baseline (peer 70%): PagedAttention/SGLang/VL-Cache/StreamingLLM/VideoLLM-online/H2O/InfiniGen/Mosaic/Semantic Scheduling/FlexGen. Runtime 14주 (kernel 4w + vLLM 2w + Ramulator 1w + 실험 4w + writing 3w). Fallback: row-boundary reverse-engineering 실패 시 Nsight 간접 추정 + Ramulator analytical. |

#### 2.1.5 예상 효과

- Decode throughput +25~35% (long-context 64K+).
- TPOT -20~30% (128K context).
- Memory footprint -30~40% (NVMe tier 활용).
- HBM row-hit 62% → **82-88%** (primary novelty 지표).
- VideoMME accuracy ≤ 0.5pp drop.
- HRTS+Mosaic stacking 추가 +5~8% (orthogonality 증명).

**Tier-2 paper-pair (HRTS Tier-2, IEEE CAL 4p / DATE 6p)**: Row-aligned KV tile letter (M1 only), LLaVA-Video-7B + VideoMME long subset 단일, HBM row-hit +15-25%p measurement-중심. Precedence 확보 후 Top-tier 제출 가능.

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

#### 2.2.3 제안 기법 (3 Mechanisms, replace-all from v1 Phase 1)

**M1 — CLIP-L LSH reuse graph classifier (replace)**: request arrival 시 CLIP-L last-pool embedding 계산 → 16-bit SimHash LSH → sliding window 256 req. 업데이트 <0.6ms.

**M2 — Tier-aware MIG dual-issue partition (replace)**: 3-SW tier (hot/warm/cold) × 2-phys MIG slice (MIG-A prefill-visual + MIG-B decode-LLM) dual-issue. Green Context μs-level reconfig (B1 GCReconfProfile 이 cost budget 정량화).

**M3 — Phase-aligned coalescing (new — critical gap 대응)**: prefill/decode phase 정렬 batch coalescing.

Replace-all 정당성: v1 TriadSM+RGSM 의 SM-allocation 중심 기여가 Mosaic scoop 대응 약함. Replace 로 "reuse graph → MIG → coalesce" coherent story 재구성.

#### 2.2.4 평가·실험 플랜 (5-요소)

| 요소 | 상세 |
|---|---|
| **Hardware** | AWS p5.48xlarge (H100 80GB × 8 MIG, 1-2주 $1200-2000) + RTX Pro 6000 96GB (Green Context + L2) + RTX 4090 × 2 (MPS+Green Context fallback). |
| **Model** | Qwen2.5-VL-7B-Instruct (tenant A) + LLaVA-OneVision-7B (tenant B) + InternVL3-8B (secondary) + MiniCPM-V-2.6 (robustness). FP16. vLLM v0.7 `ClusterAwareBlockManager` + SGLang v0.4 semantic-radix. |
| **Dataset/Workload** | Synthetic mixed trace (OCR/grounding/caption/chat/reasoning 각 20%, Poisson λ=2-8) + MMDU + 1000 document × 3-5 query + LMSys VisionArena partnership 시도. 5000 req + 1000 doc + 500 MMDU. |
| **Simulator/Tools** | vLLM `ClusterAwareBlockManager` + SGLang semantic-radix + CLIP-B/32 + `imagehash` pHash + NVML `nvmlDeviceCreateGpuInstance` + `cuCtxFromGreenCtx` (CUDA 12.5+) + Nsight Compute. |
| **Ablation/Protocol** | 2^3 factorial (M1×M2×M3) + LSH bucket {8/16/32 bit} + MIG slice {3:5/4:4/5:3} + window {128/256/512} + tenant {2/4/8}. 10 baseline (peer 60%): vLLM/SGLang/Mosaic/HERMES/Bullet/LithOS/Llumnix/VL-Cache/DynamoLLM/Predictable LLM Serving. Runtime 17주. Fallback: AWS 예산 초과 시 lambda.ai H100 1-week; partnership 거절 시 synthetic only (reviewer 제약); MIG dual-issue 제약 시 MPS+Green Context (-5~8%). |

#### 2.2.5 예상 효과

- Multi-tenant throughput +22~32% (2-tenant 7B co-location).
- p95 TTFT -18~28% (visual context overlap 시).
- SM util +12~20%p.
- vs Mosaic stacking +4~7% 추가 (orthogonality 증명).
- vs Predictable LLM Serving +8~12% 추가 (intra-GPU dual-issue vs cluster MIG).
- Single-tenant/fresh-context 에서는 2~4% (scope 밖 명시).

**Tier-2 paper-pair (ContextMIG Tier-2, IEEE ESL 4p / IEEE CAL 4p)**: CLIP-L LSH reuse graph classifier standalone (M1 only), 2-tenant Pro 6000 단일, F1 ≥ 0.82 + hash latency ≤ 1.5ms + collision ≤ 3%. Runtime 6주.

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

#### 2.3.3 제안 기법 (3 Mechanisms, improve-first)

**M1 — SSE phase predictor (improve + absorb v1 P2)**: L_mid hidden state L2 drift + EWMA + Page-Hinkley 2-threshold (soft pre-warm / hard evict) + hysteresis (3 frame). Training-free quantile calibration (첫 100 frame). VLA additional feature: gripper Δ, trajectory curvature, DINOv2 object distance (1.2ms/frame).

**M2 — Phase-specific CUDA Graph dispatcher (improve)**:
- Approach graph: SigLIP patch-embed full + FlashAttn-3 MQA fused.
- Manipulate graph: SigLIP partial-batch (40% re-encode) + action-head SiLU+Linear fused + reduced attention heads.
- Retract graph: SigLIP bypass (last-2-step feature linear extrapolation) + KV static reuse.

**M3 — Phase-specific SM partition (polish — optional stacking)**: Approach 80% encoder / Manipulate 35% encoder-partial + 55% LLM / Retract 85% LLM + 15% next-episode prefetch. Nova stacking 시 +5-8% 추가.

#### 2.3.4 평가·실험 플랜 (5-요소)

| 요소 | 상세 |
|---|---|
| **Hardware** | RTX 4090 24GB (primary, OpenVLA-7B 주 실험) / RTX 5090 32GB (phase-specific kernel 성능) / **Jetson Orin AGX 64GB (embedded deployment)**. AMD Ryzen 9 + 128GB DDR5. |
| **Model** | OpenVLA-7B (primary, BF16, LIBERO 지원) / OpenVLA-OFT (parallel decoding) / π0 (Physical Intelligence, robustness). OpenVLA HF wrapper + vLLM v0.7 fork (CUDA Graph capture variant). |
| **Dataset/Workload** | LIBERO (4 suite × 100 trial) + RoboCasa (Isaac Sim) + SimplerEnv + CALVIN + **FlashVLA direct comparison trace (v3 신규)**. 500 trial × 3 model × 4 phase config = 6000 rollout. |
| **Simulator/Tools** | NVIDIA Isaac Sim v4.5 + Isaac Lab + Page-Hinkley (scipy) + Nsight Compute/Systems (CUDA Graph capture overhead) + DINOv2 (object distance). |
| **Ablation/Protocol** | 2^3 factorial + 4-way orthogonality (baseline / M1 oracle / M1+M2 / full / stacked+Nova). PH threshold + hysteresis + SM ratio sweep. 9 baseline (peer 67%): VLA-Cache/AC²-VLA/KV-Efficient VLA/ADP-VLA/SpecPrune-VLA/Nova/DuetServe/Running-VLAs/FlashVLA. Runtime 15주. Fallback: FlashVLA re-impl 실패 시 paper 수치 interpolation 간접 비교. |

#### 2.3.5 예상 효과

- LIBERO median latency 165ms → **128ms (-22%)** (full M1+M2+M3).
- SimplerEnv Hz 6.1 → **7.8 (+28%)**.
- RoboCasa success rate Δ ≤ ±1.2pp.
- Jetson Orin 64GB latency 420 → **330ms**.
- Stacked + Nova: 165 → 115ms (-30%), Hz 8.6.
- Page-Hinkley FP rate < 5% (goal, LIBERO 5 task empirical 필수).

**Tier-2 paper-pair (PhaseGraph-VLA Tier-2, IEEE CAL 4p / DATE 6p)**: SSE phase predictor standalone (M1 only), OpenVLA-7B + LIBERO-Spatial 단일, PH FP rate ≤ 5% + decision latency < 100μs + scene-change F1 ≥ 0.75. Runtime 4주.

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

#### 3.1.3 제안 기법 (Single Mechanism)

**M1 — Green Context reconfig latency instrumentation harness**: vLLM v0.8+ fork 에 CUPTI PM Sampling + nanosecond `cuEventElapsedTime` 삽입. `cuDevSmResourceSplit` invocation → new green context ready-for-launch 까지 sub-μs 정확도. Mixed VLM workload 3 종 (LLaVA-OV OCR / Qwen2-VL chat / InternVL2 grounding) × SM-count {8, 16, 32, 64, 84} × prev-SM 4 종 factorial. (p50, p95, p99, max) × workload × SM-delta 공개. 가설 검증: driver serialization 이 SM-delta 에 quadratic growth.

#### 3.1.4 평가·실험 플랜 (5-요소, single scope)

| 요소 | 상세 |
|---|---|
| **Hardware** | RTX Pro 6000 (Blackwell, 96GB, CUDA 13.1) + A100 SXM 40GB (Hopper, CUDA 12.5, cross-arch). CUPTI PM Sampling + NVML 5ms + Nsight Systems 2026.1. |
| **Model** | LLaVA-OV 7B (OCR proxy) + Qwen2-VL 7B (chat) + InternVL2 8B (grounding). 7-8B class 축 고정. |
| **Dataset** | DocVQA 500 + LLaVA-Wild 500 + RefCOCOg 500 = 1500 sample, 2-hour. |
| **Tools** | vLLM v0.8+ fork (CUPTI 삽입) + `cuDevSmResourceSplit` / `cuCtxFromGreenCtx` 주위 `cuEventRecord` + Blackwell `CUPTI_ACTIVITY_KIND_PM_SAMPLING`. |
| **Ablation/Protocol** | (A1) SM-delta {8/16/32/64/84} × workload 3 = 15 조합. (A2) Driver contention {idle/50%/90% queue} × SM-delta 3 = 9 조합. (A3) Blackwell vs Hopper repeat = 15. Total 39 × 3 반복 = 117 runs, 단일 주말 48h 실험. Metric: reconfig latency (ns), energy (mJ), throughput loss (%). |

#### 3.1.5 예상 효과

| 지표 | Baseline (MIG, A100) | Green Context (Pro 6000) | 조건 |
|---|---|---|---|
| Reconfig latency p50 | 2100 ms | **18-45 μs** | SM-delta 16 |
| Reconfig latency p99 | 2400 ms | **80-140 μs** | SM-delta 16, driver contention |
| Reconfig latency p50 | — | **42-95 μs** | SM-delta 64 |
| Energy spike / reconfig | 12 J | **0.3-0.8 mJ** | NVML 5ms window |
| Throughput loss / reconfig | 0.2% | **0.008-0.02%** | mixed VLM 10 req/s |

**Tier-1 scale-up 불가 이유**: 단일 vendor API characterization letter, mechanism 1 개, driver black-box 의존, cross-vendor generality 부재.

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

#### 3.2.3 제안 기법 (Single Mechanism)

**M1 — per-policy energy-counter harness**: vLLM + VL-Cache / SparseVLM / PagedAttention baseline 동일 seed 1000 req. NVML 5ms + Intel RAPL DRAM-package crawling → (HBM dynamic / SM static / DRAM PKG / refresh-implied) 4-component decomposition. 가설: aggressive eviction 이 total energy 를 줄이지 않는다 (refresh overhead + recompute cost 상쇄).

#### 3.2.4 평가·실험 플랜 (5-요소, single scope)

| 요소 | 상세 |
|---|---|
| **Hardware** | RTX 5090 32GB (GDDR7) + RTX Pro 6000 96GB (HBM3e, cloud 4-hour spot rental). NVML + Intel RAPL + perf stat. |
| **Model** | LLaVA-OV 7B 단일 (visual token 3000-8000). |
| **Dataset** | VideoMME 1-min subset 200 clip + DocVQA 500 = 700 sample, 2-hour. |
| **Tools** | vLLM v0.8+ + VL-Cache + SparseVLM plugin + NVML `nvmlDeviceGetTotalEnergyConsumption` + `/sys/class/powercap/intel-rapl/intel-rapl:0:0/energy_uj` + Nsight Systems GPU memory access counter. |
| **Ablation/Protocol** | (A1) Policy {vLLM/VL-Cache/SparseVLM/random 50%} × 700 = 4. (A2) Context {2K/8K/16K} × policy 4 = 12. (A3) HBM3e vs GDDR7 cross-arch = 2. Total 18 × 2 반복 = 36 runs, 하루 24h. Metric: total energy (J), HBM/DRAM decomposition (J), peak power (W), accuracy (pp). |

#### 3.2.5 예상 효과 (Negative Result 핵심)

| 지표 | vLLM baseline | VL-Cache | SparseVLM | 조건 |
|---|---|---|---|---|
| Total energy / req (J) | 14.2 | -8 ~ -15% | -5 ~ -12% | LLaVA-OV 7B, 2048 context |
| HBM dynamic energy (J) | 6.1 | -18 ~ -25% | -12 ~ -20% | eviction → access 감소 |
| **DRAM PKG energy (J)** | 1.8 | **+2 ~ +6%** | **+1 ~ +4%** | **refresh 증가 (negative)** |
| p99 Power envelope (W) | 380 | -22 ~ -30% | -15 ~ -22% | peak power 절감 |
| Accuracy (VideoMME) | 62.1% | -0.8 ~ -1.2pp | -1.5 ~ -2.2pp | — |

**핵심 insight**: eviction ≠ always green — ISLPED reviewer 에게 공교육.

**Tier-1 scale-up 불가 이유**: Power-constrained narrow engineering, negative result 중심, confounder 통제 불가.

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

#### 3.3.3 제안 기법 (Single Mechanism)

**M1 — Action-head 3-op fusion CUDA kernel**: SiLU + Linear(4096→448) + Bucketize single persistent kernel + Block-tile (hidden=4096 / 32-warp) + TMA (Hopper/Blackwell) activation prefetch + warp-level softmax-free bucketize. Kernel launch 9-15 μs → 2-3 μs (single launch).

#### 3.3.4 평가·실험 플랜 (5-요소, single scope)

| 요소 | 상세 |
|---|---|
| **Hardware** | RTX Pro 6000 96GB (Blackwell, TMA) + RTX 5090 32GB (TMA 부분 지원). Single workstation. |
| **Model** | OpenVLA-7B (Llama-2-7B + SigLIP+DinoV2 + action-head) 단일. |
| **Dataset** | LIBERO-Spatial + LIBERO-Object 1000 trajectory, 3-hour. |
| **Tools** | CUDA 13.1 + NVCC + CUTLASS 3.6+ (TMA epilogue) + Triton 3.1+ + torch.compile + TensorRT-LLM 0.16 + Nsight Compute. |
| **Ablation/Protocol** | (A1) Kernel impl {torch/compile/TensorRT/**ActHeadFuse**} = 4. (A2) Batch {1/8/32} × impl 4 = 12. (A3) Hidden {2048/4096/8192} × ActHeadFuse = 3. Total 19 × 5 반복 = 95 runs, 하루 24h. Metric: kernel time (μs, Nsight), decode step (ms), throughput (steps/s), LIBERO MSE, SM util (%). |

#### 3.3.5 예상 효과

| 지표 | Baseline (torch-native) | torch.compile | TensorRT-LLM | **ActHeadFuse** | 조건 |
|---|---|---|---|---|---|
| Action-head kernel time (μs) | 14.2 | 8.1 | 5.4 | **2.3** | Pro 6000, bs=1 |
| **Total decode step (ms)** | 1.82 | 1.55 | 1.38 | **0.94** | **1-kHz control 가능** |
| Kernel launches / step | 4 | 2 | 1 | **1** | — |
| Action MSE (LIBERO) | 0 | 0 | 0 | **0** (bit-exact) | FP16 |
| GPU SM util during head | 38% | 52% | 78% | **86%** | — |
| Throughput (steps/s, bs=32) | 540 | 640 | 720 | **1050** | bs=32 |

**Tier-1 scale-up 불가 이유**: Narrow kernel engineering letter, model family lock-in (OpenVLA Llama-2-7B), serving stack 전체 impact ~5-8%.

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
- **이전 summary (pattern 참조)**: [2026-04-23 PRISM](2026-04-23-prism-vlm-kv-extension.md)
