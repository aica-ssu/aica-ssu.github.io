# 최신 VLM 특성 기반 에너지 효율적 Edge VLM Inference: Parquet / Triptych / Cartographer / Sift / Verge

*Session date: 2026-04-23 · Mode 1 (기존 세션 참고 금지 완전 신규)*

> 초창기 LLaVA-1.5 (fixed 576 visual token) → 최신 Qwen2.5-VL / InternVL3 (dynamic 4-16,384 visual token + MRoPE 3D + pixel shuffle + video temporal packing) 으로 진화한 VLM 아키텍처가 Jetson Orin AGX / Thor / RTX 4060 같은 edge GPU 에서 에너지 효율적으로 서빙되려면 어떤 VLM-only 최적화가 필요한가? 3 명 전문가 ideation (ai-optimization / legacy-system / hw-pim) + Phase 1-2-1'-2'-1'' 5-phase 루프 + 2026-04-23 신규 규칙 4종 (R23 workload-driven / R24 kernel fusion triviality / R25 Platform-Usage Analysis / R26 Metaphor noun title) 전면 적용하여 **Tier-1 Top 2 (Parquet ★ / Triptych) + Tier-2 독립 Top 3 (Cartographer / Sift / Verge) = 총 5 선정**, I1 Tidal 1 개는 CodecSight (2026-04-07, 16일 차이) 68-72% scoop 으로 미선정.

---

## 1. 연구 진행 Meta

### 1.1 사용자 쿼리 원문

> "오픈소스로 공개된 혹은 많은 연구들이 표준적으로 사용하는 VLM 모델에 대해서, 초창기 모델 대비 최신 모델의 아키텍처를 고려해서 이를 Single-GPU 혹은 Edge GPU 에서 serving 시 성능/에너지 최적화에 대해 연구 ideation 을 진행. 기존 세션 참고 금지, Vision 모델 최적화나 LLM 모델 특징과 다른 VLM 만의 특징을 고려한 idea. Revisit 혹은 기존과 완전히 다른 방식 가능."

### 1.2 주요 키워드 (4축)

| 축 | 키워드 |
|---|---|
| **도메인** | VLM (Vision-Language Model), edge GPU (Jetson Orin Nano/NX/AGX, Jetson Thor, RTX 4060/4090), single-GPU serving |
| **관찰·특징** | 초기 vs 최신 VLM 아키텍처 diff (fixed 576 → dynamic 4-16K token, 1D RoPE → MRoPE 3D, AnyRes 1-12 tile, pixel shuffle 2×2, video 2-frame temporal packing), VLM-only (LLM 과 다른 vision encoder / projector / LLM 3-stage) |
| **제안 기법** | AnyRes tile-count unifying signal → CUDA Graph bucket + coupled DVFS + per-tile precision / Vision DLA + Projector Tensor core + LLM NVFP4 modality-stage mapping + UMA zero-copy + DLA preemption / MRoPE tri-axial LUT / Entropy-adaptive pixel shuffle / Thor vs Orin cross-arch VLM characterization |
| **타겟** | J/token, W, battery life, TTFT, energy / request, MMMU accuracy 99.5%+ |

### 1.3 중점적으로 고려한 축

- **초기 (LLaVA-1.5 2023) vs 최신 (Qwen2.5-VL / InternVL3 2024-2025) 아키텍처 diff 를 mechanism 에 직접 활용**: AnyRes tile variance 가 DVFS signal, MRoPE 3D 가 LUT 대상, pixel shuffle 이 entropy-adaptive dial.
- **Edge GPU 의 특수 HW**: Jetson DLA (NVDLA 2.0) 가 vision encoder INT8 에 최적이지만 VLM serving 에 미활용, UMA 가 zero-copy 가능하지만 vLLM 이 discrete GPU 전제 memcpy.
- **VLM-only 특징**: 3-stage (vision / projector / LLM), dynamic resolution, modality-specific precision.
- **성능 + 에너지 동시**: TTFT 속도, J/token 에너지, MMMU accuracy 3 축 Pareto.
- **기존 세션 재사용 금지**: v1/v2/v3 VLM/VLA context serving, PRISM-VLM-KV, ACE-MoE, VLM+PIM 전혀 reference 하지 않고 완전 신규 ideation.

### 1.4 의도적으로 제외한 축 (이유 명시)

| 제외 축 | 이유 |
|---|---|
| **Multi-tenant / cross-request sharing** | v1/v2/v3 VLM/VLA context serving 세션이 이미 다룸. 본 세션은 single-tenant edge. |
| **KV cache compression (quantization)** | PRISM-VLM-KV 세션 (I2 TernVLM-KV-LUT) 이 커버. 본 세션은 edge GPU compute / DVFS / DLA 축. |
| **MoE VLM** | ACE-MoE 세션 (VLM/VLA software 확장) 이 커버. 본 세션은 dense VLM. |
| **PIM / HBM-PIM** | 2026-04-22 VLM+PIM 세션이 커버. Edge (Jetson) 는 LPDDR 이므로 HBM-PIM 부적합. |
| **Multi-node serving** | Single-workstation / edge scope 규율. |
| **Training-time optimization** | Inference serving 쿼리 명시. |
| **Token pruning 단독** | VL-Cache / SparseVLM / FastV 가 이미 다룸. 본 세션은 이를 stacking 대상으로만 참조. |
| **Cloud A100/H100 deployment** | CodecSight 등 cloud 연구와 차별화 위해 edge-only (Jetson / 4060-class) 로 scope 제한. |

### 1.5 검색 쿼리 전략

| Phase | 주요 쿼리 | 도메인 |
|---|---|---|
| Step 0 ai-opt | "Qwen2.5-VL Jetson edge serving", "vLLM VLM AnyRes bucket", "VLM DVFS energy", "NVDEC motion vector LLM inference" | arxiv / OpenReview / MLSys / SOSP / HPCA |
| Step 0 legacy-sys | "Jetson Orin AGX LLM profiling energy", "edge GPU unified memory VLM", "LPDDR5 bandwidth LLM serving" | arxiv / ICPP / HPCA / proceedings |
| Step 0 hw-pim | "Jetson DLA NVDLA transformer", "NVFP4 VLM quantization edge", "LUT tensor core MRoPE" | arxiv / ISCA / MICRO / CVPR |
| Phase 2 Similarity | "video VLM NVDEC motion vector token skip", "VLM modality stage pipeline DLA", "edge VLM energy characterization Thor Orin" | arxiv 최근 6 개월 + OpenReview |

### 1.6 사용된 전문가 에이전트 + 리뷰어

- **Experts**: ai-optimization-expert / legacy-system-expert / hw-pim-accelerator-expert (3 명 병렬 dispatch)
- **Reviewers**: novelty-reviewer / differentiation-reviewer / impact-reviewer (3 인 병렬)
- **Similarity Critique**: differentiation-reviewer 4th dispatch (scoop ≥ 70% 발굴 전담)
- **Cross review**: ai-opt ↔ legacy-sys ↔ hw-pim 도메인 상호 리뷰 (Phase 2 staging 파일 내 기록)

---

## 2. Tier-1 Top 2 (Top-tier Venue Target)

> Tier-1 Top 3 에서 I1 Tidal 이 CodecSight scoop 로 drop — Top 2 로 축소. 각 idea 에 Tier-2 scope 축소 variant subsection 병기 (paper-pair 후보).

### 2.1 Parquet ★ (Tier-1 Top 1, Avg **7.54** / ASPLOS 2027 / MLSys 2027)

#### 2.1.1 개요 (Metaphor → Mechanism)

**"Parquet"** = 마루판 타일. AnyRes 의 가변 tile 1-12 개가 마루판 타일처럼 가지런히 정렬되면서도 각 판이 서로 다른 precision/빈도 성질을 가진 은유. 최신 VLM (LLaVA-NeXT / Qwen2.5-VL / InternVL3) 의 **AnyRes tile-count** 를 통합 signal 로 활용하여 (1) CUDA Graph bucket 을 tile-count-variant 로 확장, (2) GPU+DRAM DVFS 를 coupled 하게 제어, (3) tile 별 entropy 로 precision 동적 선택한다.

#### 2.1.2 기존 연구 한계점 및 Gap

- [vLLM v1 Multimodal CUDA Graph 문서](https://docs.vllm.ai/en/latest/design/cuda_graphs_multimodal/): batch-size variant **만** 지원. Tile-count variant 없음. 실측 4-bucket fill rate 55%.
- [DynamoLLM HPCA'25 arXiv:2408.00741](https://arxiv.org/abs/2408.00741) [peer-reviewed]: LLM DVFS/inference 에너지 관리. VLM 미포함. **Tile-count signal 없음**.
- [GreenLLM arXiv:2508.16449](https://arxiv.org/abs/2508.16449): SLO-aware frequency scaling. VLM AnyRes 고려 없음.
- [PolyThrottle MLSys'24 arXiv:2310.19991](https://arxiv.org/abs/2310.19991) [peer-reviewed]: DNN throttle 일반, VLM visual token variance 미반영.
- [MBQ CVPR'25 arXiv:2412.19509](https://arxiv.org/abs/2412.19509) [peer-reviewed]: modality-balanced quantization, per-layer 축만 있음. **Per-tile (sub-image) 축 없음**.
- [BiScale arXiv:2602.18755](https://arxiv.org/abs/2602.18755): BW-coupled DVFS, VLM 미터치.
- [throttLL'eM arXiv:2408.05235](https://arxiv.org/abs/2408.05235): LLM throttling, tile-count 미활용.
- **SparseDVFS [arXiv:2603.21908](https://arxiv.org/abs/2603.21908)**: sparse workload DVFS, VLM AnyRes 고려 없음.

**Gap 요약**: AnyRes tile-count 를 **CUDA Graph + DVFS + precision** 의 unifying control signal 로 활용하는 연구 공백. 3 축을 통합한 첫 edge VLM serving scheduler.

#### 2.1.3 제안 기법 (3 Mechanisms, improve-first 유지)

**M1 — Adaptive CUDA Graph Bucket by Tile-Count**
- vLLM v1 의 batch-size variant CUDA Graph 을 **tile-count variant** 로 확장.
- Bucket 크기 {1, 3, 6, 12} + dynamic overflow path (tile count 초과 시 JIT capture).
- 4-bucket fragmentation 55% → 75-85% 채움률 목표.
- Platform-Usage Analysis: vLLM v1 (이미 bucket 구조 존재, tile-count 로 확장 필요) / SGLang (RadixAttention 기반, tile-count bucket 미구현) / llama.cpp (batch API 제한적) — Step C: vLLM v1 에 upstream PR 경로.

**M2 — Coupled GPU+DRAM DVFS with Tile-Count Signal**
- Tile 많을 때 (1-tile 보다 6-tile) memory-bound → GPU freq 소폭 낮추고 DRAM freq 유지.
- Tile 적을 때 compute-bound → GPU freq max, DRAM freq 낮춤.
- Jetson Orin AGX `nvpmodel` + `jetson_clocks` 확장 (custom power profile).

**M3 — Per-Tile Entropy-Driven Precision Dispatching**
- Tile attention entropy (pre-trained SigLIP attention weight proxy) 상위 50% FP8, 하위 50% INT8.
- MBQ 의 per-layer 접근을 **per-tile (sub-image)** 로 확장. Edge Tensor core 의 FP8/INT8 이기종 활용.

#### 2.1.4 평가·실험 플랜 (5-요소)

| 요소 | 상세 |
|---|---|
| **Hardware** | RTX 4090 24GB (primary, desktop edge) / RTX 4060 8GB (consumer edge) / Jetson Orin AGX 64GB (embedded). Single workstation + single edge node. |
| **Model** | Qwen2.5-VL-7B (primary, AnyRes native) / LLaVA-NeXT-7B (AnyRes baseline) / InternVL3-8B (pixel shuffle baseline) / LLaVA-OneVision-7B (robustness). |
| **Dataset/Workload** | DocVQA (OCR, tile count 많음) + MMMU (mixed) + ChartQA + OK-VQA. 5000 request mixed AnyRes workload + per-request tile-count histogram 기록. |
| **Simulator/Tools** | vLLM v1 fork (tile-count bucket + DVFS hook) + Nsight Compute (`l1tex__t_sector_hit_rate`, `lts__t_sectors`, `dram__throughput`) + NVML (per-request energy) + Jetson INA3221 (edge power) + Intel RAPL (desktop). |
| **Ablation/Protocol** | 2^3 factorial (M1 × M2 × M3) + tile-count histogram sweep {1, 3, 6, 12} + DVFS policy sweep + precision tier sweep. 10 baseline (peer-reviewed 70%): vLLM v1 / SGLang / DynamoLLM HPCA'25 / GreenLLM / PolyThrottle MLSys'24 / BiScale / throttLL'eM / MBQ CVPR'25 / SparseDVFS / Nova. Runtime 14주. Fallback: vLLM v1 PR 변경 시 SGLang fork 로 대체. |

#### 2.1.5 예상 효과

| 지표 | Baseline | Parquet | 조건 |
|---|---|---|---|
| Energy / request | 1× | **-28~42%** | Mixed AnyRes workload |
| TTFT (ms) | 1× | +4~5% trade | - |
| CUDA Graph bucket fill rate | 55% | **75-85%** | Tile-count adaptive |
| DocVQA accuracy | 100% | ≥ 99.5% | Per-tile precision |
| Tile-uniform (text-only) | - | <3% gain | scope 밖 (명시) |

**Scoring Summary**: Nov 7.4 / Diff 7.5 / Imp 7.76 / Feas 7.5 → **avg 7.54 Accept strong**. 전문가 **3:0 unanimous**.

**Tier-2 scope 축소 variant (paper-pair 후보, IEEE CAL 4p)**: M1 only (Adaptive CUDA Graph bucket), Qwen2.5-VL-7B 단일 + Jetson Orin AGX 단일, bucket fill rate + latency 2 metric.

---

### 2.2 Triptych (Tier-1 Top 2, Avg **7.38** / ASPLOS 2027 / EuroSys 2027)

#### 2.2.1 개요 (Metaphor → Mechanism)

**"Triptych"** = 3 폭 패널 그림. Vision encoder (DLA INT8) / Projector (Tensor core FP16) / LLM (NVFP4 또는 INT4) 이 각각 독립된 패널로 구성되면서 하나의 통합된 serving 작품을 이루는 은유. Jetson Orin AGX 의 DLA 를 vision encoder 전용으로 활용하고, Jetson UMA 의 zero-copy 로 activation 을 memcpy 없이 hand-off, DLA fine-grained preemption 으로 pipeline bubble 제거한다.

#### 2.2.2 기존 연구 한계점 및 Gap

- [Nova arXiv:2509.21301](https://arxiv.org/abs/2509.21301): VLM 3-stage elastic partitioning, **desktop GPU only, DLA 미사용**. Phase 2 similarity **60% concurrent**.
- [Nanomind arXiv:2510.05109](https://arxiv.org/abs/2510.05109): module-level brick scheduling for tiny model, **DLA fine-grained preemption 없음**. Phase 2 similarity **60% concurrent**.
- [HeteroInfer arXiv:2501.14794](https://arxiv.org/abs/2501.14794): heterogeneous LLM serving, **server-class GPU, DLA 미포함**. 55% concurrent.
- [HydraInfer arXiv:2505.12658](https://arxiv.org/abs/2505.12658): hybrid inference, DLA 미사용.
- [llm.npu arXiv:2407.05858](https://arxiv.org/abs/2407.05858): mobile NPU + LLM, VLM 3-stage 미커버.
- [LiteVLM arXiv:2506.07416](https://arxiv.org/abs/2506.07416): edge VLM pipeline, **single compute unit, DLA 미사용**.
- [FastVLM CVPR'25 arXiv:2412.13303](https://arxiv.org/abs/2412.13303) [peer-reviewed]: vision encoder 최적화 (algorithmic), **DLA 미사용**.
- [NVIDIA DLA Jetson Orin blog](https://developer.nvidia.com/blog/maximizing-deep-learning-performance-on-nvidia-jetson-orin-with-dla/): DLA 를 convolution only 로 제시, transformer 미커버.

**Gap 요약**: Jetson DLA 는 NVIDIA 공식 문서조차 convolution only 로 가정 — transformer vision encoder + VLM serving 용 DLA scheduler 공백. UMA 는 하드웨어 지원 있으나 vLLM 이 discrete GPU 전제로 활용 안 함.

#### 2.2.3 제안 기법 (3 Mechanisms, Phase 1' M3 replace)

**M1 — Modality-Stage Heterogeneous Mapping**
- Vision encoder → **Jetson DLA INT8** (NVDLA 2.0).
- Projector → **Tensor core FP16**.
- LLM → **NVFP4 (Jetson Thor) 또는 INT4 (Orin)**.
- DLA INT8 는 vision encoder (ViT/SigLIP/InternViT) 에서 desktop GPU 대비 2-3× throughput/W 이점.

**M2 — UMA Zero-Copy Activation Hand-off**
- `ZeroCopyActivationRouter` (vLLM-Jetson fork) 구현.
- DLA output tensor → projector GPU kernel 직접 주입.
- Memcpy 제거 (Jetson UMA 물리 주소 동일 원리 활용).
- Platform-Usage Analysis: vLLM (discrete GPU 전제, UMA 미지원) / TensorRT-LLM (일부 Jetson support 있으나 activation routing 수동) / MLC-LLM (Jetson 지원, UMA 활용 부분적). Step C: vLLM-Jetson fork 로 upstream + MLC-LLM 보조.

**M3 (REPLACE from Green Context SM partition) — DLA Fine-Grained Preemptive Scheduling**
- Jetson DLA 의 sub-kernel preemption 활용 (NVDLA 2.0 기능).
- Multi-image batch 시 DLA / Tensor core / LLM 단계 간 pipeline bubble 제거.
- Nova 의 desktop SM partition 과 다른 **DLA preemption** 축.
- **Critical gap 방어**: Phase 2 Similarity 에서 Nova + Nanomind + HeteroInfer 68-72% concurrent 판정 → M3 replace 로 DLA axis 로 repositioning, concurrent → unique.

#### 2.2.4 평가·실험 플랜 (5-요소)

| 요소 | 상세 |
|---|---|
| **Hardware** | Jetson Orin AGX 64GB (primary, DLA 보유) / Jetson Thor DevKit (2026-06 gate, NVFP4) / RTX 4090 (desktop baseline, DLA 없음). |
| **Model** | Qwen2.5-VL-7B / InternVL3-8B / LLaVA-OneVision-7B (primary edge VLM). |
| **Dataset/Workload** | VideoMME short subset + MMMU + DocVQA + Multi-image LLaVA-Interleaved. 3000 req multi-image batch. |
| **Simulator/Tools** | vLLM-Jetson fork + `ZeroCopyActivationRouter` + NVDLA tools (`nvmedia-dla-sdk`) + Nsight Systems (3-stream overlap) + Jetson INA3221 (GPU/CPU/SoC power). |
| **Ablation/Protocol** | 2^3 factorial (M1 DLA mapping / M2 UMA zero-copy / M3 DLA preemption). Thor NVFP4 optional condition. 9 baseline (peer-reviewed 67%): Nova / Nanomind / HeteroInfer / HydraInfer / vLLM EPD / llm.npu / LiteVLM / FastVLM / TensorRT-LLM-Edge. Runtime 16주. Fallback: Thor DevKit 미확보 시 Orin-only (NVFP4 → INT4). |

#### 2.2.5 예상 효과

| 지표 | Baseline (vLLM all-GPU) | Triptych | 조건 |
|---|---|---|---|
| TTFT | 1× | **1.4-1.7× faster** | Multi-image batch 3+ |
| Energy / query | 1× | **-25~35%** | DLA 활용 |
| MMMU accuracy | 100% | ≥ 99.5% | Modality precision |
| Single-image VLM | - | 5-8% gain (scope 밖) | DLA 이점 축소 |
| Jetson Thor NVFP4 | +5-8% 추가 | - | Conditional on Thor DevKit |

**Scoring Summary**: Nov 6.5 (post-replacement) / Diff 8.0 / Imp 8.0 / Feas 7.0 → **avg 7.38 Accept**. 전문가 **3:0 unanimous (post-replacement)**.

**Tier-2 scope 축소 variant (IEEE ESL 4p, paper-pair 후보)**: M1 only (DLA INT8 vision encoder), Qwen2.5-VL-7B on Jetson Orin AGX, 에너지 + latency letter.

---

## 3. Tier-2 독립 Top 3 (Track B, Phase 1 부터 독립 도출)

### 3.1 Cartographer (Track B Top 1, Avg **7.06** / IEEE CAL 4p / DATE 6p)

#### 3.1.1 개요 (Metaphor → Mechanism)

**"Cartographer"** = 지도 제작자. MRoPE 의 3 axis (time × H × W) 좌표계를 precomputed LUT 으로 "지도화" 하여 runtime 계산을 치환. Edge GPU 의 SFU (Special Function Unit) 점유를 줄이고 memory BW 감소.

#### 3.1.2 GAP 및 단일 Mechanism

- Qwen2.5-VL / InternVL3 의 **MRoPE (Multi-dimensional RoPE, 3-axial: time × H × W)** 계산은 전용 kernel, SFU 점유 non-trivial (edge GPU 에서 더 크게 pronounced).
- [T-MAC EuroSys'25 arXiv:2407.00088](https://arxiv.org/abs/2407.00088) [peer-reviewed]: weight mpGEMM LUT, positional encoding 미커버.
- [LUT Tensor Core ISCA'25 arXiv:2408.06003](https://arxiv.org/abs/2408.06003) [peer-reviewed]: weight LUT, positional encoding 미적용.
- [RotateKV arXiv:2501.16383](https://arxiv.org/abs/2501.16383): rotation-based, LUT 치환 미도입.
- [Revisiting MRoPE arXiv:2510.23095](https://arxiv.org/abs/2510.23095): MRoPE 변형 연구, **LUT 기반 kernel 공백**.
- [SAIL arXiv:2509.25853](https://arxiv.org/abs/2509.25853): SRAM-LUT GEMV, VLM positional 미적용.

**Single Mechanism**: MRoPE (time × H × W) 3-axial positional 을 precomputed LUT 치환 + LPDDR row-aligned layout.

#### 3.1.3 평가·실험 플랜 (5-요소, single scope)

- **Hardware**: Jetson Orin AGX 64GB 단일.
- **Model**: Qwen2.5-VL-7B 단일 (MRoPE 3D).
- **Dataset**: VideoMME + DocVQA subset 500 req.
- **Tools**: CUTLASS LUT kernel impl + Nsight Compute (SFU busy %, `smsp__inst_executed_pipe_fp_hi`).
- **Ablation**: LUT size sweep {256, 512, 1024, 2048}. 5 baseline (peer-reviewed 60%): T-MAC / LUT Tensor Core / FlashAttention-3 / SAIL / Revisiting MRoPE. Runtime 4주.

#### 3.1.4 예상 효과

| 지표 | Baseline | Cartographer |
|---|---|---|
| MRoPE kernel latency | 1× | **-40~60%** |
| Energy / MRoPE call | 1× | **-15~25%** |
| MMMU accuracy | 100% | 100% (bit-exact LUT) |

**Scoring**: Nov 6.8 / Diff 6.6 / Imp 7.03 / Feas 7.8 → **avg 7.06 Accept CAL**.

**Tier-1 scale-up 불가 이유**: single-kernel characterization letter, MRoPE 에 한정.

---

### 3.2 Sift (Track B Top 2, Avg **6.85** / ISLPED 2026 6p)

#### 3.2.1 개요 (Metaphor → Mechanism)

**"Sift"** = 체로 거른다. Patch entropy 기준으로 visual token 을 체질하여 통과/압축 선택. InternVL3 / SmolVLM 의 pixel shuffle ratio 를 entropy-adaptive 하게 동적 선택한다 (기존 2×2 고정).

#### 3.2.2 GAP 및 단일 Mechanism

- [InternVL-X arXiv:2503.21307](https://arxiv.org/abs/2503.21307): RVTC (reduced visual token compression), **entropy-adaptive 미포함**.
- [PyramidDrop arXiv:2410.17247](https://arxiv.org/abs/2410.17247): layer-wise token drop, **projector-이전 pixel shuffle 축 없음**.
- [VisionZip arXiv:2412.04467](https://arxiv.org/abs/2412.04467): visual token compression, ratio 고정.
- [FastV arXiv:2403.06764](https://arxiv.org/abs/2403.06764): attention-based prune, projector-이전 미커버.
- [SparseVLM ICML'25](https://openreview.net/forum?id=80faIPZ67S) [peer-reviewed]: sparse attention, pixel shuffle 미커버.
- [SmolVLM arXiv:2504.05299](https://arxiv.org/abs/2504.05299): edge VLM, pixel shuffle 고정.

**Single Mechanism**: Patch attention entropy (pre-trained SigLIP attention weight proxy) 기반 pixel shuffle ratio {2, 4, 8} 동적 선택. Projector 이전에 적용되어 token count reduction 이 가장 효과적.

#### 3.2.3 평가·실험 플랜 (5-요소)

- **Hardware**: Jetson Orin Nano 8GB (tiny edge) + Orin AGX 64GB.
- **Model**: SmolVLM-2B / InternVL3-2B / MiniCPM-V-2.6.
- **Dataset**: DocVQA + COCO + MMMU patch entropy subset.
- **Tools**: PyTorch pixel shuffle modification + NVML + Jetson INA3221.
- **Ablation**: Entropy threshold sweep + pixel shuffle ratio {2, 4, 8} vs fixed. 6 baseline (peer 50%): InternVL-X / PyramidDrop / VisionZip / FastV / SparseVLM ICML'25 / SmolVLM. Runtime 5주.

#### 3.2.4 예상 효과

- Visual token count -40-60%, energy -20-30%, accuracy drop ≤ 1pp (DocVQA / MMMU).

**Scoring**: Nov 6.5 / Diff 6.5 / Imp 6.90 / Feas 7.5 → **avg 6.85 Accept ISLPED**.

**Tier-1 scale-up 불가 이유**: tiny VLM + projector-이전 pixel shuffle 축 narrow.

---

### 3.3 Verge (Track B Top 3, Avg **6.29** / IEEE ESL 4p letter Conditional)

#### 3.3.1 개요 (Metaphor → Mechanism)

**"Verge"** = 경계/가장자리. Jetson Thor (최신 Blackwell NVFP4) 와 Orin AGX (이전 세대 Ampere INT8) 의 경계에서 VLM-specific 에너지 차이를 측정.

#### 3.3.2 GAP 및 단일 Mechanism

- [ELANA arXiv:2512.09946](https://arxiv.org/abs/2512.09946): Thor + Orin Nano cross-arch, **일반 LLM serving only, VLM-specific stage breakdown 없음**.
- [Jetson Orin LLM profiling arXiv:2506.09554](https://arxiv.org/abs/2506.09554): Orin only, Thor cross-arch 없음.
- [TokenPowerBench arXiv:2512.03024](https://arxiv.org/abs/2512.03024): LLM token-level power, **VLM 미포함**.
- [Watt Counts arXiv:2604.09048](https://arxiv.org/abs/2604.09048): LLM energy, Jetson Thor 미커버.
- [Blackwell Microbench arXiv:2512.02189](https://arxiv.org/abs/2512.02189): Blackwell 일반 micro, VLM 미커버.

**Single Mechanism**: Qwen2.5-VL-7B / InternVL3-8B / LLaVA-OneVision-7B 3 모델을 Jetson Thor (Blackwell, NVFP4) vs Orin AGX (Ampere, INT8) 에서 VLM-specific stage (vision / projector / LLM prefill / decode) 별 J/token + W 측정.

#### 3.3.3 평가·실험 플랜 (5-요소, Conditional)

- **Hardware**: Jetson Thor DevKit (2026-06 gate, primary) + Jetson Orin AGX 64GB. Fallback: RTX 4090 + Orin AGX.
- **Model**: Qwen2.5-VL-7B / InternVL3-8B / LLaVA-OneVision-7B.
- **Dataset**: VideoMME / MMMU / DocVQA.
- **Tools**: Jetson INA3221 / NVML / PMU / nvprof.
- **Ablation**: Vision / projector / LLM prefill / decode 별 J/token 측정. 6 baseline (peer 40% Conditional): ELANA / Jetson Orin profiling / TokenPowerBench / E4 AAAI'25 / Blackwell Microbench / Watt Counts. Runtime 6주.

#### 3.3.4 예상 효과

- VLM-specific insight (FP4 의 LLM vs VLM 에너지 이점 차이, LPDDR5X vs LPDDR5 의 visual token burst write pattern).

**Scoring**: Nov 5.8 / Diff 6.2 / Imp 6.35 / Feas 6.8 → **avg 6.29 Conditional**.

**Conditional 조건**: Jetson Thor DevKit 2026-06 확보 필수.

**Tier-1 scale-up 불가 이유**: characterization-only letter.

---

## 4. 미선정 아이디어 전수 (사유 + 재방문 조건)

### 4.1 I1 Tidal (DROP by CodecSight Scoop)

- **연구 GAP (원래 의도)**: Video VLM (Qwen2.5-VL video / LLaVA-Video) 2-frame visual token cosine sim > 0.9 비율 42-65% 에서 NVDEC hardware motion vector + MRoPE-aware temporal KV dedup + LPDDR bank-aligned block layout.
- **Metaphor 의도**: "Tidal" = 조수 (temporal repetition, cyclical pattern 은유).
- **미선정 사유**: [CodecSight arXiv:2604.06036](https://arxiv.org/abs/2604.06036) (2026-04-07, Yulin Zou 등) 이 (a) NVDEC motion vector → token skip gate, (b) RoPE position-correction selective KVC refresh 를 이미 제안. **68-72% 직접 scoop**. 제출일 16일 차이로 precedence 확보 불가능. Edge-only repositioning 시도에도 core mechanism 이 1:1 중첩.
- **재방문 조건**:
  1. CodecSight 의 Jetson edge validation 부재 명확화 + VLM-specific edge benchmark 실측 novelty 확보.
  2. MRoPE 3D temporal axis 가 CodecSight 1D RoPE 와 formal 다른 별도 축으로 재설계 (예: VideoRoPE 변형 결합).
  3. LPDDR bank-aligned layout 을 I2 Parquet 의 sub-mechanism 으로 흡수 가능.

### 4.2 Phase 1 에서 통합된 원본 12 ideas (artificial split 방지)

| Original (Phase 1) | 통합 Idea | 합병 사유 |
|---|---|---|
| ai-opt Ember | **Triptych** (M2 UMA + Ember 의 per-stage DVFS) | Modality-stage pipeline 3-way 수렴 |
| ai-opt Mosaic | **Parquet** (M1 bucket + Mosaic 의 CUDA Graph) | AnyRes tile batching 3-way 수렴 |
| ai-opt Ripple | **Tidal** (M1 NVDEC + Ripple 의 MRoPE reuse) | Video VLM dedup 3-way 수렴 |
| ai-opt Lattice | **Sift** (Lattice 의 tiny edge entropy pixel shuffle) | Entropy-adaptive pixel shuffle 2-way 수렴 |
| legacy-sys TileTide | **Parquet** (M2 DVFS + TileTide 의 coupled GPU+DRAM) | AnyRes DVFS 3-way 수렴 |
| legacy-sys VistaGate | **Triptych** (M2 UMA zero-copy + VistaGate 의 LPDDR layout) | UMA pipeline 3-way 수렴 |
| legacy-sys EchoVault | **Tidal** (M2 KV dedup + EchoVault 의 LPDDR bank) | Video VLM 3-way 수렴 |
| legacy-sys PixelTram | **Sift** (PixelTram 의 ViT-tail pixel shuffle) | Entropy pixel shuffle 2-way 수렴 |
| hw-pim Triptych Pipeline | **Triptych** (M1 Modality mapping + DLA INT8) | 동일 컨셉, 이름 유지 |
| hw-pim Mosaic Tiler | **Parquet** (M3 per-tile precision + Mosaic Tiler 의 FP8/INT8) | AnyRes 3-way 수렴 |
| hw-pim Echo Chamber | **Tidal** (M3 + Echo Chamber 의 sparse projector/LLM) | Video VLM 3-way 수렴 |
| hw-pim Cartographer Cache | **Cartographer** (MRoPE LUT + LPDDR 독립) | 독립 Track B 유지 |

---

## 5. 이 세션의 독특한 점

| 축 | 이 세션의 특이점 |
|---|---|
| **기존 세션 완전 신규** | v1/v2/v3 VLM/VLA context serving, PRISM-VLM-KV, ACE-MoE, VLM+PIM 어느 것도 reference 하지 않음. 사용자 요구 충족. |
| **초기 vs 최신 VLM diff 직접 활용** | Parquet = AnyRes (LLaVA-1.5 미존) / Triptych = DLA × 3-stage (최신 pixel shuffle + projector) / Cartographer = MRoPE 3D (초기 1D RoPE 미존) / Sift = pixel shuffle (초기 미존). **4 idea 중 4 개 모두 최신 아키텍처 diff 활용**. |
| **VLM-only 특징 강제** | LLM-only or Vision-only 로 성립 불가함을 각 idea 에서 formal 증명. |
| **Edge GPU 이기종** | Jetson DLA (NVDLA 2.0), UMA zero-copy, NVFP4 (Thor), LPDDR5X 등 edge-specific HW 활용 — desktop GPU 적용 불가 축. |
| **R23 Workload-driven 엄수** | 6 idea 모두 Workload evidence 섹션에 숫자 근거 (Qwen2.5-VL 4-16K token, vLLM 55% bucket fill, video cosine 42-65% 등). |
| **R26 Metaphor Noun Title 엄수** | Parquet / Triptych / Cartographer / Sift / Verge / Tidal 모두 metaphor noun. |
| **CodecSight scoop 즉시 감지** | Phase 2 similarity critique 가 16일 전 공개 논문 발견 → Tidal 즉시 DROP. |

---

## 6. 다음 단계 제안

1. **Parquet Phase 3 entry**: vLLM v1 EncoderDisagg merge 상태 2026-04 snapshot 실측 (1주 PoC).
2. **Triptych DLA preemption API**: NVDLA SDK 접근 권한 확인 + Jetson Orin AGX DLA INT8 vision encoder 프로파일링 (2주).
3. **Cartographer MRoPE LUT PoC**: Nsight Compute SFU busy % 선행 측정 (1일).
4. **Sift entropy 분포 측정**: DocVQA + COCO entropy 분포 선행 (2일).
5. **Verge Thor DevKit gate**: 2026-06-01 확보 decision, 미확보 시 drop or Orin+RTX 4090 fallback.
6. **Tidal 재방문 세션**: CodecSight 후속 연구 모니터링 + MRoPE 3D vs 1D RoPE formal 차별 재설계 (Mode 1 재호출 시).
7. **Publish 요청 시**: summary 파일 homepage publish (명시 요청 시만).

---

## 7. 참고 파일

- **Session 상세 (재현성)**: [sessions/2026-04-23-mode1-energy-efficient-edge-vlm.md](../sessions/2026-04-23-mode1-energy-efficient-edge-vlm.md)
- **Staging**:
  - [aiopt expert](../sessions/staging/2026-04-23-edge-vlm-energy-aiopt-expert.md) (520 lines)
  - [legacy-sys expert](../sessions/staging/2026-04-23-edge-vlm-energy-legacy-sys-expert.md) (500 lines)
  - [hwpim expert](../sessions/staging/2026-04-23-edge-vlm-energy-hwpim-expert.md) (467 lines)
  - [Phase 1 integration](../sessions/staging/2026-04-23-edge-vlm-energy-phase1-integration.md)
  - [Phase 2 novelty](../sessions/staging/2026-04-23-edge-vlm-energy-phase2-novelty.md)
  - [Phase 2 differentiation](../sessions/staging/2026-04-23-edge-vlm-energy-phase2-diff.md)
  - [Phase 2 impact](../sessions/staging/2026-04-23-edge-vlm-energy-phase2-impact.md)
  - [Phase 2 similarity critique](../sessions/staging/2026-04-23-edge-vlm-energy-phase2-similarity.md)
  - [Phase 1'/2'/1''](../sessions/staging/2026-04-23-edge-vlm-energy-phase1prime-2prime-1primeprime.md)
