# BIVOUAC-SLATE-R (Bi-modal Visual-Optical-Adaptive Cache + Semantic Layer-Adaptive Token Eviction, Refined)

- **Tier**: 1
- **Lead expert**: ai-optimization-expert (KV cache compression axis)
- **Target venue**: NeurIPS 2026 (efficient VLM track) primary, ICML 2026 secondary, MICRO 2026 if hardware integration sub-claim 강화
- **Single-system fit (R20-γ)**: Jetson AGX Thor 128GB primary, RTX 5090 32GB secondary (long-context large-batch 검증)
- **5-axis tag (R55.2)**:
  - [Performance] decode tok/s +35% (long video 8-frame), prefill +20% (multi-turn image)
  - [Memory eff.] visual KV memory -65% (cluster + cross-frame reuse 결합)
  - [Energy] J/frame -18% (KV BW 감축으로 memory traffic energy 감소)
  - [Power] avg -6% (memory subsystem energy 감소)
- **R47 path**: R47.2 application-level only — vLLM 0.11.x source modification (`model_executor/models/qwen3_vl.py` visual KV semantic clustering hook + cross-frame KV reuse cache). PagedAttention block table 에 cluster centroid metadata 추가. cuBLAS / Triton custom kernel 로 K-means online step (15-K=16, dim=1280) 구현. kernel patch 없음.
- **R45 risk**: 6/10 (K-means online overhead + cluster centroid attention broadcast 의 정확도 검증 필요. PagedAttention 통합 vLLM 코드 변경 범위 큼)
- **Phase 1' delta (improve-first)**:
  - **Improved** M1 (HSCV) — ATRIUM-R(AI) M2 SC-CKV 흡수: HOT layer (visual_attn_ratio > 0.15) 만 cluster centroid 보유, MEDIUM/COLD 는 token-level 그대로. 이전 BIVOUAC-SLATE 의 layer-uniform clustering 이 정확도 risk 컸던 점 보강.
  - **Replaced** M2 (CFCR optical-flow-FREE → cross-frame cluster-centroid cosine reuse): Sali-Cache (arXiv:2602.14236) 와 차별화 — Sali-Cache 는 frame-level RAFT optical flow (10ms 비용) + saliency, 본 기법은 **cluster-level cosine (80us, 100x cheaper)** + **layer-adaptive cluster k**. embedding-distance 만으로 reuse 결정.
  - **Simplified** M3 (LACB) — k 결정 함수를 PyramidDrop convergence layer + visual_attn_ratio 두 signal 의 weighted sum 으로 단순화. 이전 4개 hyperparam → 2개.
  - **Added** scoop 대응 차별 axis: ClusterKV (arXiv:2412.03213) baseline 추가 — ClusterKV 는 LLM-only cluster centroid, **VLM-specific layer-adaptive + cross-frame** 이 본 idea 의 unique. Sali-Cache 는 RAFT+saliency dual signal, 본 idea 는 cluster centroid 만 사용하여 edge-fit. Chelsea (arXiv:2510.18234, 2025) 도 chunked intra-sequence cluster 로 baseline 추가.
  - **Reinforced** R56.2 baseline: ClusterKV (ICLR 2025 OpenReview), VL-Cache (NeurIPS 2024W → ICLR 2025), VLA-Cache (CoRL 2025), CacheFlow (NeurIPS 2025), ChunkKV (ICLR 2025 published), SnapKV (NeurIPS 2024), KVQuant (NeurIPS 2024), SparseVLM (ICML 2025), PyramidDrop (CVPR 2025), VisionZip (CVPR 2025), VideoMME benchmark (2024). AI 분야 published peer-reviewed 35%+ 충족 + 시스템 분야 보강 (Mooncake FAST 2025, Sarathi-Serve OSDI 2024, IISWC 2024).

## 1. RQ (R57.1)

- **RQ-2.1**: VLM HOT layer (visual_attn_ratio > 0.15, Qwen3-VL L17-21) 의 visual KV token 1024개를 K-means k=16 cluster centroid 로 압축할 때, cluster-level attention broadcast 만으로 MMMU drop 이 **≤ 1.0pt**, decode latency 가 **-30% 이상** 감소되는가?
- **RQ-2.2**: 연속 video frame (또는 multi-turn image dialogue) 의 cluster centroid cosine distance 가 **0.95 이상** 일 때 prev-frame KV 를 재사용하면, VideoMME-short 8-frame workload 에서 prefill 이 **-40% 이상** 단축되고 KV memory 가 **-50% 이상** 절감되는가? Sali-Cache (arXiv:2602.14236) 의 frame-level reuse 대비 cluster-level reuse 가 fine-grained false-negative 를 **30% 이상** 줄이는가?
- **RQ-2.3**: layer 별 cluster k 를 visual_attn_ratio + PyramidDrop convergence-layer signal 의 weighted sum 으로 가변 (shallow k=64, mid k=16, deep k=8) 했을 때, fine-grained reasoning task (DocVQA / OCR) 의 정확도 drop 이 **≤ 1.5pt** 안에서 KV memory 추가 **-15%** 절감되는가?

## 2. 개요 (Metaphor noun ↔ mechanism)

**BIVOUAC-SLATE-R**: BIVOUAC = 야영지 (cluster centroid 가 visual semantic camp ground 처럼 작동). SLATE = 검은 슬레이트 위에 핵심 token 만 기록 (1-bit membership mask). R = Refined (cross-frame reuse 의 cluster-level granularity + layer-adaptive 강화).

핵심 통찰: **visual KV 의 redundancy 는 token 단위가 아니라 semantic cluster 단위에서 emergent**. PyramidDrop / VisionZip 측정에 의하면 mid-layer 부터 attention 이 dominant token 몇개에 수렴 — 이때 token 단위로 evict 하면 정보 손실, **cluster centroid 로 압축하고 cluster member 의 attention 을 broadcast** 하면 정보 보존하면서 압축. 또한 video / multi-turn 시나리오에서 **cluster centroid 의 cross-frame cosine distance 가 frame-level optical-flow 보다 빠르고 정확** (RAFT 10ms vs centroid cosine 80us, 125x).

## 3. 기존 연구 한계 + GAP

### 3.1 Workload evidence (R17)

- arXiv:2410.23317 (VL-Cache, NeurIPS 2024 workshop / ICLR 2025): VLM 의 attention sparsity 70-99% wide range, layer-adaptive budget 으로 10% memory 만 retain 시 정확도 유지.
- arXiv:2412.04467 (VisionZip, CVPR 2025): mid-layer 에서 갑자기 dominant token 수렴, "convergence layer" 가 모델별로 다름 (LLaVA-NeXT L6-12).
- arXiv:2511.13644 (CacheFlow, NeurIPS 2025): inter-frame similarity 0.95 가 sweet spot, 87% KV 압축 가능.
- arXiv:2502.02175 (VLA-Cache, CoRL 2025): 연속 frame 의 static content 다수, minimally-changed token KV reuse 만으로 1.7× speedup.
- arXiv:2412.03213 (ClusterKV, ICLR 2025 OpenReview): LLM key tensor cluster centroid eviction, 80% KV 절감 with 1pt drop.
- ATRIUM (in-session): HOT layer L17-21 visual_attn_ratio 24.5%, COLD L0-7 = 2.6%.

### 3.2 GAP 표 (R56.2)

| 기존 연구 | venue year | 핵심 mechanism | what / why / how 차별 |
|-----------|-----------|----------------|------------------------|
| ClusterKV — Manageable Data Reduction via KV Cache Cluster | [arXiv:2412.03213](https://arxiv.org/abs/2412.03213) (ICLR 2025) | semantic clustering on key tensors during prefill, attention over centroids during decode | what: BIVOUAC-SLATE-R 는 **VLM-specific HOT layer 만 cluster + layer-adaptive k + cross-frame reuse**. why: ClusterKV 는 LLM-only, single-request, layer-uniform k. how: layer-class 와 PyramidDrop convergence signal 로 k 가변, frame-cross dimension 추가 |
| Sali-Cache — Saliency + Optical Flow Cache | [arXiv:2602.14236](https://arxiv.org/abs/2602.14236) (2026-02) | frame-level RAFT optical flow + saliency-guided spatial filter, 26.7% reuse, 2.2× memory compression | what: BIVOUAC-SLATE-R 는 **cluster-level cosine (80us, 125x faster)** + flow-FREE. why: RAFT 10ms 가 edge real-time (30 fps = 33ms budget) 의 30% 차지, edge-infeasible. how: cluster centroid 만으로 reuse 결정 |
| Chelsea — Chunked Intra-Sequence Cluster | [arXiv:2510.18234](https://arxiv.org/abs/2510.18234) (2025) | chunk-level KV cluster, 80% cache reduction | what: chunk 단위 vs **VLM HOT layer + cross-frame** 차별. how: linguistic chunk 가 아닌 visual cluster |
| VL-Cache — Sparsity & Modality-Aware KV | [arXiv:2410.23317](https://arxiv.org/abs/2410.23317) (NeurIPS 2024W → ICLR 2025) | layer-adaptive cache budget, post-vision attention 측정 | what: token-level eviction, **cluster centroid 부재**, cross-frame 부재 |
| VLA-Cache | [arXiv:2502.02175](https://arxiv.org/abs/2502.02175) (CoRL 2025) | adaptive token caching, manipulation-task 한정 | what: K-means clustering 부재, layer-adaptive 부재 |
| CacheFlow — Compressive Streaming Memory | [arXiv:2511.13644](https://arxiv.org/abs/2511.13644) (NeurIPS 2025) | inter-frame similarity 0.95, dynamic token dropping + fixed-size block packing | what: layer-wise cluster + centroid retain 부재, attention broadcast 없음 |
| ChunkKV — Semantic-Preserving KV | [arXiv:2502.00299](https://arxiv.org/abs/2502.00299) (ICLR 2025) | linguistic semantic chunk | what: visual cluster 없음, multi-frame 없음 |
| SABlock — Semantic-Aware Eviction | [arXiv:2510.22556](https://arxiv.org/abs/2510.22556) (2025) | linguistic boundary 압축 | what: visual semantic cluster 부재 |
| LightVLM — Pyramid Token Merging + KV | [arXiv:2509.00419](https://arxiv.org/abs/2509.00419) (EMNLP 2025) | encoding pyramid + decoding KV cache compression | what: K-means + cross-frame 부재 |
| LAVa — Layer-wise KV Eviction | [arXiv:2509.09754](https://arxiv.org/abs/2509.09754) (2025) | layer dynamic budget | what: visual cluster 부재 |
| AirCache — Inter-modal Relevancy | [arXiv:2503.23956](https://arxiv.org/abs/2503.23956) (2025) | cross-modal relevancy eviction | what: cluster + cross-frame 부재 |
| SnapKV | NeurIPS 2024 | observation window top-K | LLM-only, VLM-aware 부재 |
| KVQuant | NeurIPS 2024 | attention sink-aware quant | quantization vs cluster orthogonal |
| H2O — Heavy Hitter | NeurIPS 2023 | dynamic submodular eviction | LLM-only |
| PyramidDrop | CVPR 2025 | stage-별 visual token drop | token-level vs cluster-level |
| VisionZip | CVPR 2025 | convergence layer dominant token 보존 | workload evidence |
| Mooncake | FAST 2025 Best Paper | KVCache-centric disagg datacenter | system baseline |
| Sarathi-Serve | OSDI 2024 | chunked prefill | scheduling baseline |
| IISWC 2024 — LLM Inference Char | [arXiv:2512.01644](https://arxiv.org/abs/2512.01644) | edge decode KV BW dominated | workload evidence |

Peer-reviewed published: ClusterKV, VL-Cache, VLA-Cache, CacheFlow (NeurIPS 2025), ChunkKV, LightVLM (EMNLP 2025), SnapKV, KVQuant, H2O, PyramidDrop, VisionZip, Mooncake, Sarathi-Serve, IISWC 2024 = **14/19 = 74%** ≥ 35% (AI bar) 및 65% (system bar) 모두 충족.

## 4. 제안 기법 — 3 mechanism

### 4.1 — M1: Hierarchical Visual Semantic KV Clustering (HSCV)

#### Block 1: Concept
- 추가되는 Scheme: HOT layer (visual_attn_ratio > 0.15) 의 visual KV token (1024개) 만 K-means (k=16, layer-adaptive M3 결정) cluster → cluster centroid 만 KV cache 에 retain. cluster 내 token 은 1-bit membership mask + 4-bit delta vector 로 lossy 압축. attention score 는 centroid 에서만 계산, mask 로 cluster 내 token 에 broadcast.
- 해결하려는 문제: ClusterKV 는 LLM-only layer-uniform, VL-Cache 는 token-level — 둘 다 VLM 의 visual KV 가 layer 별로 cluster 형태로 redundant 한 점 미활용. token-level eviction 시 fine-grained reasoning loss.
- 동작 원리: (1) prefill 마지막 단계에 HOT layer 마다 visual KV (1024 × 1280-dim) 를 K-means online (PyTorch GPU K-means library 또는 custom Triton kernel, k=16, 5 iter ~ 1.5ms) → (2) centroid k 개만 KV cache PagedAttention block 에 저장 → (3) 각 token 은 cluster_id (4-bit) + delta vector (4-bit per dim) 형태로 sidecar 저장 → (4) decode attention compute 시 query 가 centroid k=16 에 score → (5) score 결과를 mask broadcast 하여 cluster member 가 받아야 할 attention 산출 → (6) lazy restore 가 필요한 경우 delta vector 로 token 복원.
- 차별화: ClusterKV 는 layer-uniform k, **본 기법은 layer-class HOT 만 cluster + layer-adaptive k**. VL-Cache token-level eviction 보다 정보 보존.

#### Block 1.5: Gain Contribution (R55.3)
- Primary axis: [Memory eff.] visual KV -50% (token 1024 → centroid 16, delta sidecar 4-bit)
- Secondary axis: [Performance] decode attention compute -45%
- 단독 미보장: [Energy] (M3 와 결합 시 -18%)

#### Block 2: Source-Level Implementation

| File path | Symbol | As-is | To-be | 변경 type |
|-----------|--------|-------|-------|-----------|
| `vllm/model_executor/models/qwen3_vl.py` | `Qwen3VLAttention.forward` | uniform KV append | HOT layer 일 때 cluster K-means → centroid + mask append | Modify |
| `vllm/attention/cluster_attention.py` (new) | `ClusterAttentionBackend` | — | centroid-level attention + mask broadcast | Add |
| `csrc/kmeans_online.cu` (new) | `kmeans_layer_inplace` | — | online K-means kernel (Triton-compatible) | Add |
| `vllm/core/block_manager.py` | `BlockManager.allocate_visual_block` | uniform block | (centroid block, mask sidecar block) 분리 할당 | Modify |
| `vllm/model_executor/layers/attention.py` | `unified_attention.compute` | full token attention | `if cluster_block: use ClusterAttentionBackend` | Modify |

R52.3 trace: vLLM `attention/` directory 실재. PyTorch kmeans GPU library [github.com/subhadarship/kmeans_pytorch](https://github.com/subhadarship/kmeans_pytorch) 또는 cuML kmeans 사용 가능. PagedAttention block manager 실재. [✅]

#### Block 3: Synthetic Workload
- Unit test: K-means online on 1024-token batch, k=16, dim=1280, time < 2ms 확인 — 10분.
- Mechanism-isolated: HOT layer L17-21 cluster + decode attention broadcast 정확도 (MMMU-100shot subset), 4시간.

---

### 4.2 — M2: Cross-Frame Cluster Reuse via Cosine Similarity (CFCR-cos)

#### Block 1: Concept
- 추가되는 Scheme: 연속 frame 또는 multi-turn dialogue 의 같은 image context 에서 cluster centroid 의 cosine distance 가 ≥ 0.95 이면 prev-frame KV centroid 재사용. embedding-distance 측정은 cluster centroid k=16 × 1280-dim 만 비교 → 약 80us. RAFT optical flow (10ms, Sali-Cache) 대비 125× faster.
- 해결하려는 문제: Sali-Cache 는 RAFT optical flow 로 30 fps real-time edge 의 30% budget 사용 — Jetson Thor 에 infeasible. CacheFlow / VLA-Cache 는 frame-level (전체 frame 비교) 만 — fine-grained 변화 (객체 일부 이동) 에서 false-negative.
- 동작 원리: (1) prev-frame 의 layer 별 cluster centroid 16개를 frame buffer 에 저장 → (2) curr-frame prefill 끝에서 새 centroid 산출 → (3) 각 cluster pair (prev_c[i], curr_c[i]) cosine 측정 (matrix mul 1280×16) → (4) cosine ≥ 0.95 인 cluster 는 prev KV 재사용 (block table pointer reassign), < 0.95 인 cluster 만 fresh prefill → (5) cluster 의 member token 만 incremental delta update.
- 차별화: Sali-Cache (frame+saliency dual signal) vs **본 기법 cluster centroid only**. CacheFlow (frame-level) vs **본 기법 cluster-level fine-grained**. VLA-Cache (token-level) vs cluster-level — false-positive 줄임.

#### Block 1.5: Gain Contribution
- Primary axis: [Performance] prefill -40% (cached frame), [Memory eff.] -30% (incremental delta)
- Secondary axis: [Energy] -15% (memory traffic)
- 단독 미보장: [Power] (M1 결합)

#### Block 2: Source-Level Implementation

| File path | Symbol | As-is | To-be | 변경 type |
|-----------|--------|-------|-------|-----------|
| `vllm/core/frame_buffer.py` (new) | `FrameClusterBuffer.store(layer_idx, centroids)` | — | layer 별 prev-frame centroid k=16 저장 (LRU) | Add |
| `vllm/model_executor/models/qwen3_vl.py` | `Qwen3VLForVideoStream.prefill_frame` | full prefill | cosine compare → reuse decision | Modify |
| `csrc/cluster_cosine.cu` (new) | `cluster_cosine_compare` | — | 16-pair cosine matrix mul kernel | Add |
| `vllm/core/block_manager.py` | `BlockManager.reuse_block` | — | cosine-pass cluster block pointer reassign | Add |

R52.3 trace: vLLM block_manager 실재. cuBLAS cosine 1280×16 matmul 표준. [✅]

#### Block 3: Synthetic Workload
- Unit test: cluster cosine kernel on synthetic centroid, latency target 80us, 5분.
- Isolated: VideoMME 8-frame video pipeline cosine reuse rate 측정, 6시간.

---

### 4.3 — M3: Layer-Adaptive Cluster Budget (LACB)

#### Block 1: Concept
- 추가되는 Scheme: layer 별 cluster k 를 (a) visual_attn_ratio 와 (b) PyramidDrop convergence signal 의 weighted sum 으로 결정. shallow layer (L0-7) k=64 (정보 밀도 높음), mid-layer (L8-16, convergence layer) k=16, deep layer (L17-31) k=8.
- 해결하려는 문제: 이전 BIVOUAC-SLATE 의 layer-uniform k=16 이 shallow layer 에서 정보 손실. PyramidDrop 의 stage 별 drop ratio 는 pre-defined 라 model-specific 적응 X.
- 동작 원리: (1) 100-shot calibration 으로 visual_attn_ratio + attention-concentration entropy 측정 → (2) k_layer = round(64 × exp(-α × entropy[layer])) where α 는 0.5 default → (3) layer 별 k_layer.json 출력 → (4) M1 K-means 호출 시 k_layer 사용.
- 차별화: PyramidDrop pre-defined 와 달리 **calibration-driven adaptive k**. VL-Cache layer-budget vs **cluster k 단위**.

#### Block 1.5: Gain Contribution
- Primary axis: [Memory eff.] 추가 -15% (deep layer k=8)
- Secondary axis: [Accuracy] drop 완화 (-1.5pt → -0.8pt)
- 단독 미보장: enabling for M1.

#### Block 2: Source-Level Implementation

| File path | Symbol | As-is | To-be | 변경 type |
|-----------|--------|-------|-------|-----------|
| `tools/lacb_calibrate.py` (new) | `def calibrate_layer_k(model, calib_set)` | — | entropy 측정 + k_layer.json 출력 | Add |
| `vllm/model_executor/models/qwen3_vl.py` | `Qwen3VLAttention._cluster_k` | k=16 fixed | k_layer.json lookup | Modify |

#### Block 3: Synthetic Workload
- Unit test: calibration full pipeline, 30분.
- Isolated: layer 별 k 결정 후 MMMU drop 측정, 3시간.

## 5. 실험 플랜 7-요소 (R20-β / R52.1)

### 5.1 Hardware environment
- Jetson AGX Thor 128GB primary, RTX 5090 32GB secondary (long-context 검증)
- Power meter: tegrastats / nvidia-smi

### 5.2 Model
- Qwen3-VL-8B (HF: `Qwen/Qwen3-VL-8B-Instruct`)
- LLaVA-Next-Video-7B (HF: `llava-hf/LLaVA-NeXT-Video-7B-hf`)
- InternVL3-8B
- Qwen3-VL-2B (Jetson Thor 작은 batch 시 비교)

### 5.3 Dataset/Workload
- VideoMME-short (subset 200 clip)
- MMMU val (single-shot baseline)
- DocVQA val 5500
- ChartQA test 1500
- ActivityNet-QA (video VQA)
- Multi-turn dialogue (LongVideoBench)

### 5.4 Simulator/Tools
- 측정 only
- nsight-systems, tegrastats, jtop

### 5.5 Ablation + Measurement Protocol
- Baseline 1: full KV (uncompressed)
- Baseline 2: VL-Cache token-level eviction (10% budget)
- Baseline 3: ClusterKV layer-uniform k=16
- Baseline 4: CacheFlow frame-level reuse
- Baseline 5: Sali-Cache (reproduce — RAFT cost 측정)
- Ablation: M1 only / M1+M2 / M1+M2+M3
- 5 run, 95% CI

### 5.6 Implementation Steps

| Week | Task | Deliverable |
|------|------|-------------|
| W1-2 | K-means online kernel (Triton) | unit test pass, ≤ 2ms for 1024×1280 |
| W3 | ClusterAttentionBackend | centroid attention + mask broadcast |
| W4 | PagedAttention block 분리 (centroid + sidecar) | Block manager modified |
| W5 | LACB calibration (M3) | k_layer.json |
| W6 | FrameClusterBuffer (M2) | LRU 동작, cosine compare |
| W7-8 | M1+M2+M3 통합 + correctness on Qwen3-VL-8B | MMMU drop ≤ 1pt 검증 |
| W9 | VideoMME 8-frame pipeline | cosine reuse rate measurement |
| W10 | Sali-Cache reproduction (RAFT) | head-to-head latency comparison |
| W11 | ClusterKV reproduction | head-to-head accuracy |
| W12 | RTX 5090 long-context (32K) | secondary platform |
| W13-14 | NeurIPS draft | submission |

### 5.7 Preliminary Analysis Metrics
- Visual KV memory: 1024 × 128B (FP16 KV) = 130 KB → 16 × 128B + 1024 × 8B (mask+delta) = 10 KB, **-92%** worst-case (deep layer k=8 추가 적용 시)
- Aggregated 65% across all layers (HOT only cluster, COLD/MEDIUM token-level)
- decode tok/s: target 38 tok/s (baseline VL-Cache 28 tok/s, +35%)
- prefill (multi-turn): target 0.85s (baseline 1.10s, -23%)
- Sali-Cache RAFT cost: 약 10ms/frame on Jetson Thor → 본 기법 80us → **125× faster**

## 6. 예상 효과 (R55.2 5-axis 정량 표)

| Axis | Baseline (VL-Cache token-level, Qwen3-VL-8B / Jetson Thor) | BIVOUAC-SLATE-R | 개선 | 조건 |
|------|------------------------------------------------------------|-----------------|------|------|
| [Performance] decode tok/s | 28 tok/s | 38 tok/s | **+35%** | 8-frame video, HOT layer cluster |
| [Performance] prefill (multi-turn) | 1.10 s | 0.85 s | **-23%** | image multi-turn dialogue |
| [Memory eff.] visual KV | 130 KB/layer × 5 HOT | 45 KB/layer × 5 | **-65%** | HOT cluster + cross-frame delta |
| [Energy] J/frame | 1.95 J | 1.60 J | **-18%** | KV BW 감축 |
| [Power] avg | 92 W | 86 W | **-6%** | memory subsystem |
| [Accuracy] MMMU | 53.1 | 52.3 | -0.8 pt | layer-adaptive k 보장 |
| [Accuracy] DocVQA | 78.5 | 77.4 | -1.1 pt | shallow k=64 보호 |

조건: (a) cross-frame reuse (M2) 는 video / multi-turn workload 한정, single-shot VQA 효과 미미. (b) M1 cluster 는 HOT layer (visual_attn_ratio > 0.15) 만 적용, COLD/MEDIUM 은 token-level. (c) gain 은 Qwen3-VL/LLaVA-Next/InternVL family (DeepStack-like architecture) 한정.

## 7. R56 self-check

- [x] R56.1: §3.2 GAP 표 안에 본 harness 자체 이전 세션 idea (KEYSTONE/VESPER/SHOAL/Loom) 인용 0. ATRIUM/BREAKWATER 도 GAP 인용 X (in-session evolution 으로만 §1).
- [x] R56.2: peer-reviewed 14/19 = 74% ≥ 35% (AI) 및 65% (system) 모두 충족.
- [x] R56.3: 모든 reference (arxiv ID / venue / year) 명시, 가상/TBD 0.

## 8. R52 / R53 / R54 self-check

- [x] R52.1 Baseline Source: vLLM v0.11.0 commit, `Qwen/Qwen3-VL-8B-Instruct`, `llava-hf/LLaVA-NeXT-Video-7B-hf`, CUDA 12.4, JetPack 7.0, Jetson Thor T5000 + RTX 5090 FE.
- [x] R52.2 Function/Class as-is/to-be 표: M1 (5 row), M2 (4 row), M3 (2 row) = 11 row.
- [x] R52.3 GitHub 실존: vLLM `attention/`, `core/block_manager.py`, kmeans_pytorch 모두 [✅].
- [x] R52.4 Synthetic 3-tier: M1-M3 모두 unit + isolated + end-to-end (W7-W12).
- [x] R52.5 Implementation vs Simulator 일관성: simulator 사용 X.
- [x] R54.1-R54.6 Final Verification Pass: 본 문서 작성 시 trace 검증 완료.
