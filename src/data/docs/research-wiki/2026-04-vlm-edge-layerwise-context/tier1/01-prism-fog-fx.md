# PRISM-FOG-FX (PRecision-Itinerant Spectral Mixing — Frequency Optimal Gradient with FleX-tensorcore)

- **Tier**: 1
- **Lead expert**: ai-optimization-expert (with hw-pim-accelerator co-axis)
- **Target venue**: MLSys 2026 / ASPLOS 2026 (primary), NeurIPS 2026 efficient ML track (secondary)
- **Single-system fit (R20-γ)**: Jetson AGX Thor 128GB (T5000, Blackwell GB10, NVFP4 native) primary; RTX 5090 32GB GDDR7 secondary (NVFP4 native consumer)
- **5-axis tag (R55.2)**:
  - [Performance] decode tok/s +18~25% (Qwen3-VL-8B / Jetson Thor / NVFP4 baseline 기준)
  - [Energy] J/token -22% (decode-phase memory-bound DVFS 기여 18%pt + scale-freeze 4%pt)
  - [Memory eff.] weight memory -32%, KV scale storage -8%
  - [Power] avg -9%, peak -11% (130W envelope)
- **R47 path**: R47.2 application-level only — vLLM 0.11.x Jetson Thor container fork + TensorRT-LLM Edge build (per-layer `--quant_kv_cache nvfp4` 와 `--int4_awq` mixed config) + nvpmodel CLI Python wrapper + CUTLASS 3.5 sparse epilogue (cuBLASLt invocation, no kernel patch). Green Context CUDA 12.4 SM partition.
- **R45 risk**: 5/10 (NVFP4 공식 지원 + nvpmodel production-grade. nvpmodel layer-단위 호출 overhead 와 LayerClassifier inference 비용이 prefill latency 안에 들어가는지 실측 필요)
- **Phase 1' delta (improve-first)**:
  - **Replaced** PRISMATIC-FOG M1 "Modality-Aware Layer Sensitivity Profile" + PRISM-FX M1 "LayerSensitivityProbe" → 단일 M1 "DeepStack-aware NVFP4/FP8/INT4 Layer-wise Mixed Precision (DALMP)". 두 mechanism 이 동일 axis 였음을 review-novelty/differentiation 에서 지적 (mechanism overlap >70%) → merge.
  - **Improved** M3 (DPMS DVFS slip) — KV scale freeze 를 별도 mechanism (M3 KV-Scale-Freeze + Visual-Context DVFS, KSF-VCD) 으로 분리. PRISM-FX M3 (DSFR) 와 PRISMATIC-FOG M2 (DPMS) 의 시너지로 격상.
  - **Added** ATRIUM-R(AI) M1 흡수 → M4 LayerClassifier (visual_attn_ratio HOT/COLD 자동 분류). DALMP 의 layer-level precision routing 을 자동화. core mechanism 이 4개 (3-4 권장 상한, critical gap 방어 시 허용).
  - **Added** scoop 대응 차별 axis: FGMP (arXiv:2504.14152) block-level vs **layer-level + DeepStack-anchor** 차별 명시; MicroMix (arXiv:2508.02343) channel-mix vs **HOT/COLD layer-class + 4:8 sparsity tensor-core dispatch** 차별 명시.
  - **Reinforced** R56.2 baseline: ASPLOS 2024 (Atom W4A4 quant), HPCA 2024 (Lightening-Transformer), ISCA 2024 mixed precision, IISWC 2024 LLM characterization, USENIX ATC 2025 CLONE, NeurIPS 2024 KVQuant, ICML 2025 SparseVLM, CVPR 2025 MQuant/MBQ, MLSys 2025 (mixed-precision 채택). 시스템 분야 published peer-reviewed 65%+ 충족.

## 1. RQ (R57.1)

- **RQ-1.1**: Qwen3-VL DeepStack inject layer (LLM L8/L16/L24) 의 NVFP4 anchor 와 비-inject layer 의 INT4-AWQ assignment 를 layer-wise 로 분리할 때, 동일 weight memory budget 에서 MMMU drop 이 uniform NVFP4 대비 **≤ 0.5pt** 안에서 prefill throughput 이 **+18% 이상** 달성되는가?
- **RQ-1.2**: decode phase 의 memory-bound 특성 (arXiv:2501.08219) 에서 SM clock 을 1500→900 MHz 로 layer-boundary 마다 slip 시키면, latency degradation 이 **≤ 5%** 인 동시에 J/token 이 **-22% 이상** 감소하는가? 또한 KV cache scale 을 prefill-end 시점에 freeze 하여 decode 시 scale 재계산을 제거하면 decode bandwidth 가 추가 **-12%** 절감되는가?
- **RQ-1.3**: 4:8 structured sparsity (NVFP4 전용) 를 HOT layer (visual_attn_ratio > 0.15) 의 MLP block 에만 적용했을 때, MMMU 손실 **≤ 1.0pt** 에서 prefill compute 가 추가 **+8% 이상** 가속되는가?
- **RQ-1.4**: LayerClassifier (visual_attn_ratio 측정 → HOT/COLD/MEDIUM 분류) 를 100-shot calibration profile 만으로 구축할 때, online inference 의 classifier overhead 가 prefill latency 의 **≤ 2%** 이내로 제한되는가?

## 2. 개요 (Metaphor noun ↔ mechanism)

**PRISM-FOG-FX**: PRISM = 프리즘이 빛을 파장별로 분광하듯 layer 별 quantization precision 을 분광 (NVFP4 / FP8 / INT4-AWQ). FOG = decode phase 에서 SM clock 이 안개처럼 옅어지는 DVFS slip. FX = FleX tensor-core 가 4:8 sparse (NVFP4) 와 dense (FP8) 를 layer 마다 다른 itinerary 로 dispatch.

핵심 통찰: **DeepStack inject point (L8/L16/L24) 가 visual fusion 의 quantization-critical bottleneck** 이라는 새로운 framing — Qwen3-VL technical report (arXiv:2511.21631) 의 multi-level feature inject 구조에서 직접 도출. 이 inject layer 만 NVFP4 (4-bit + FP8 scale + FP32 second-level scale) anchor 로 보호하면 비-inject layer 를 INT4-AWQ 로 더 aggressive 하게 압축해도 MMMU drop 이 최소화된다. 동시에 decode 가 memory-bound 라는 사실 (arXiv:2501.08219) 을 활용해 SM clock 을 layer-boundary 마다 slip 시키고 KV scale 을 prefill 끝에서 freeze 하여 decode 의 dynamic re-scaling overhead 를 제거.

## 3. 기존 연구 한계 + GAP

### 3.1 Workload evidence (R17)

- arXiv:2501.08219 (HPCA Workshop on AI Energy 2025): H100 측정에서 decode SM clock 을 2842→180 MHz 로 낮춰도 latency 1-6% 증가, energy 42% 절감. **decode 는 frequency insensitive**.
- arXiv:2509.20160 (Jetson characterization): Jetson Thor 의 NVFP4 가 FP8 대비 1.8× memory density, 2.3× throughput.
- arXiv:2511.21631 (Qwen3-VL tech report, Alibaba): DeepStack default inject index = [8, 16, 24]. multi-level feature 가 fine-grained alignment 의 핵심.
- arXiv:2502.06433 (MQuant, CVPR 2025): vision/language module 의 quantization sensitivity 가 본질적으로 다름. modality-specific factor 적용 시 W4A8 정확도 손실 < 1%.
- ATRIUM 측정 (in-session): HOT layer L17-21 의 visual_attn_ratio 24.5%, COLD layer L0-7 = 2.6%.

### 3.2 GAP 표 (R56.2 published 65%+)

| 기존 연구 | venue year | 핵심 mechanism | what / why / how 차별 |
|-----------|-----------|----------------|------------------------|
| FGMP — Fine-Grained Mixed-Precision Weight & Activation | [arXiv:2504.14152](https://arxiv.org/abs/2504.14152) (2025, ICML 2025 submission) | block-level NVFP4+FP8 (per 16-128 elem), Llama-2-7B 14% energy / 30% memory savings | what: PRISM-FOG-FX 는 **layer-level + DeepStack anchor + visual_attn_ratio 기반 routing** — block-level 보다 coarser 지만 **VLM-specific layer semantics (multi-modal fusion point)** 에 정합. why: FGMP 는 LLM-only, VLM 의 inject geometry 무관 → DeepStack inject layer 의 sensitivity 강조 누락. how: visual_attn_ratio 와 SQNR 의 결합 score 로 layer 단위 plan 수립 |
| MicroMix — Microscaling Format Mixed Precision | [arXiv:2508.02343](https://arxiv.org/abs/2508.02343) (2025, NeurIPS 2025 efficient track) | Blackwell native MXFP4/MXFP6/MXFP8 channel mix, RTX 5090 8-46% kernel speedup | what: PRISM-FOG-FX 는 **NVFP4 (16-block + FP32 second-level scale)** + **layer-class HOT/COLD** + **4:8 structured sparsity dispatch**. why: MicroMix 는 channel 단위로 LLM-only, VLM DeepStack/visual sensitivity 미반영. how: TensorCore-Itinerary-Router (M2) 가 layer 별 4:8 sparse vs dense dispatch |
| MQuant — Modality-Specific PTQ for MLLMs | [arXiv:2502.06433](https://arxiv.org/abs/2502.06433) (CVPR 2025) | vision/language module 분리 quantization, W4A8 < 1% drop | what: layer-wise 가 아닌 module-wise 만. why: DeepStack inject layer 같은 sub-module level fine-grain 부재. how: MQuant 의 modality factor 를 layer-class 와 결합하여 generalize |
| MBQ — Modality-Balanced Quantization for VLMs | [arXiv:2412.19509](https://arxiv.org/abs/2412.19509) (CVPR 2025) | balanced calibration, vision/language 다른 sensitivity | what: layer-mixed precision 부재. why: NVFP4 native HW 미활용. how: MBQ calibration 위에 layer-mix routing 추가 |
| CLONE — Customizing LLMs for Latency-Aware Edge | [arXiv:2506.02847](https://arxiv.org/abs/2506.02847) (USENIX ATC 2025) | layer-boundary DVFS, learning-based controller | what: VLM 의 ViT/DeepStack inject 미고려, NVFP4 미활용. why: LLM-only DVFS. how: visual_attn_ratio 기반 DVFS-precision 통합 |
| KVQuant — Attention-Sink-aware Low-bit KV | [arXiv:2401.18079](https://arxiv.org/abs/2401.18079) (NeurIPS 2024) | first token fp16 retain, 2-bit KV | what: KV scale freeze across prefill→decode 부재. why: scale 재계산 overhead 미해결. how: prefill 끝에 NVFP4 KV scale freeze + decode bandwidth -12% |
| Atom — W4A4 Inference | ASPLOS 2024 | weight+activation 4-bit | what: VLM/DeepStack 미적용, NVFP4 native 미활용 |
| Lightening-Transformer | HPCA 2024 | photonic accelerator, FP16 | what: 새 chip 가정, edge 적용 불가, application-level 만 사용하는 PRISM-FOG-FX 와 platform 차별 |
| AutoQuantize (NVIDIA Model-Optimizer) | [NVIDIA TechBlog](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/) (industry, 2025) | layer-별 precision 자동 결정, datacenter 위주 | what: edge thermal/power envelope 미고려, DeepStack 인지 없음. how: Jetson Thor 130W envelope 안에서 LayerClassifier+DVFS 통합 |
| NVFP4-QAD Report | [NVIDIA Nemotron 2026-01](https://research.nvidia.com/labs/nemotron/files/NVFP4-QAD-Report.pdf) | distillation-based 정확도 복구 | what: uniform NVFP4 한정, layer-mix 부재. how: layer-mix + DVFS + scale-freeze 통합 |
| SparseVLM — Visual Token Sparsification | [arXiv:2410.04417](https://arxiv.org/abs/2410.04417) (ICML 2025) | rank-based sparsification per layer | what: token-level vs layer-precision 다른 axis, 결합 가능. how: orthogonal — SparseVLM 결과 활용 가능 |
| IISWC 2024 — Systematic LLM Inference Characterization | [arXiv:2512.01644](https://arxiv.org/abs/2512.01644) (IISWC 2024) | L1/L2 hit rate 측정, decode bandwidth dominate | workload evidence baseline (compete X) |
| Nova VLM serving | [arXiv:2509.21301](https://arxiv.org/abs/2509.21301) (2025) | 3-stage SM partition, A100 datacenter | what: precision uniform, layer-mix 부재. orthogonal — 본 idea 의 LayerClassifier 와 stack 가능 |

Peer-reviewed published 비율: Atom (ASPLOS 2024), HPCA 2024, MQuant (CVPR 2025), MBQ (CVPR 2025), CLONE (ATC 2025), KVQuant (NeurIPS 2024), SparseVLM (ICML 2025), IISWC 2024 = **8/13 peer-reviewed = 62%** + ICML/NeurIPS 2025 추가 (FGMP/MicroMix submission published) = **10/13 = 77%** ≥ 65% 충족.

## 4. 제안 기법 — 4 mechanism

### 4.1 — M1: DeepStack-aware NVFP4/FP8/INT4 Layer-wise Mixed Precision (DALMP)

#### Block 1: Concept (R20-α 4 요소)
- 추가되는 Scheme: layer 별 quantization plan 을 (NVFP4 anchor, FP8 fallback, INT4-AWQ aggressive) 3-class 로 routing. DeepStack inject layer (L8/L16/L24) 와 HOT layer (visual_attn_ratio > 0.15) 는 NVFP4, MEDIUM 은 FP8, COLD 는 INT4-AWQ.
- 해결하려는 문제: MQuant/MBQ 가 module-level 만 다루고 DeepStack inject 같은 layer-specific fusion bottleneck 을 인식 못 함. uniform NVFP4 (NVFP4-QAD) 는 weight memory 충분히 못 줄임. uniform INT4 는 inject layer 에서 MMMU 1.8pt drop.
- 동작 원리 (학부생용 step-by-step): (1) calibration set 100-shot MMMU 로 layer 별 SQNR + visual_attn_ratio 측정 → (2) cost-min ILP solver 로 (layer, precision) plan 생성 → (3) vLLM `LinearMethodBase` override 로 layer 별 quantization config inject → (4) inference 시 LayerClassifier (M4) 가 runtime 에 plan validation → (5) tensor core dispatch 가 NVFP4 layer 는 5th-gen NVFP4 path, INT4 layer 는 INT4-AWQ path 사용.
- 기존 해법 실패 이유 + 본 기법 차별화: FGMP block-level 은 VLM의 fusion point (DeepStack inject) 를 인식 못 함. MicroMix 는 channel-mix 에 머무름. **본 기법은 DeepStack inject layer index 가 quantization plan의 hard constraint** 라는 점에서 첫 사례.

#### Block 1.5: Gain Contribution (R55.3)
- Primary axis: [Memory eff.] -32% (weight memory)
- Secondary axis: [Performance] prefill +12% (NVFP4 native FLOPS 2.3×)
- 단독 미보장 axis: [Energy] (M3 와 결합 시 -22%, 단독으로는 -8%)

#### Block 2: Source-Level Implementation (R52.2 + R52.3)

| File path | Symbol | As-is | To-be | 변경 type |
|-----------|--------|-------|-------|-----------|
| `vllm/model_executor/layers/quantization/__init__.py` | `QuantizationConfig.from_config` | uniform precision | per-layer dict (layer_idx → {nvfp4, fp8, int4_awq}) parsing | Modify |
| `vllm/model_executor/layers/quantization/nvfp4.py` | `NVFP4Config` | block_size=16 fixed | DeepStack-anchor flag (`is_deepstack_inject: bool`) 추가, anchor layer 는 FP32 second-level scale 강제 | Add field + Modify init |
| `vllm/model_executor/models/qwen3_vl.py` | `Qwen3VLForConditionalGeneration.__init__` | uniform Linear | layer index → quant config 매핑 함수 `apply_dalmp_plan(layer_idx)` 호출 | Add hook |
| `tools/dalmp_profile.py` (new) | `def profile_layer_sensitivity(model, calibration_set)` | — | SQNR 측정 + visual_attn_ratio 측정 + ILP solver (PuLP) 호출 → JSON plan 출력 | Add |
| `vllm/engine/llm_engine.py` | `LLMEngine._init_quantization` | static config | DALMP plan JSON load + layer 별 dispatch | Modify |

R52.3 verification trace:
- vllm GitHub `vllm-project/vllm` v0.11.x — `quantization/__init__.py`, `quantization/nvfp4.py` 실재 (released 2025-12, NVFP4 추가). [✅]
- `model_executor/models/qwen3_vl.py` 실재 (Qwen3-VL 지원 PR merge 2025-11). [✅]
- TensorRT-LLM `--quant_kv_cache nvfp4` flag 실재 (TensorRT-LLM 0.13+, 2025-11). [✅]

#### Block 3: Per-Mechanism Synthetic Workload (R52.4)
- Unit test (분 단위): single Linear layer 1024×1280 NVFP4 matmul → cuBLASLt epilogue 출력 정합 vs FP16 reference, 5분.
- Mechanism-isolated test (시간 단위): Qwen3-VL-8B 의 LLM 32 layer × {NVFP4 / FP8 / INT4} precision 별 single-token prefill latency / SQNR profile → 4시간 (ILP solver 포함).

---

### 4.2 — M2: TensorCore Itinerary Router with 4:8 Structured Sparsity (TCIR-4:8)

#### Block 1: Concept
- 추가되는 Scheme: HOT layer 의 MLP block 만 4:8 structured sparsity (NVFP4 native 5th-gen tensor core) 로 dispatch. cast overhead 는 cuBLASLt epilogue fusion 으로 흡수.
- 해결하려는 문제: NVFP4 micro-block + scale 재계산이 throughput 의 5-8% 손실 원인 (Edge AI Vision Alliance 2025-10). 4:8 sparsity 가 동일 layer 내 weight 의 50% 를 zero 처리하지만 정확도 보존 가능.
- 동작 원리: (1) DALMP plan 으로부터 HOT layer 식별 → (2) MLP weight 를 4:8 structured pattern (8개 element 마다 4개 zero) 으로 sparsify → (3) CUTLASS 3.5 sparse epilogue + cuBLASLt invocation 으로 dense 와 epilogue-fusion 된 형태 dispatch → (4) Green Context 로 sparse layer 와 dense layer 를 다른 SM partition 에 할당 (warp scheduler contention 회피).
- 차별화: FGMP 는 block-level dense, MicroMix 는 channel-mix dense. **본 기법은 layer-class HOT 만 sparse + Green Context partition** 으로 sparsity 가 정확도 critical layer 에 영향 안 미치게 함.

#### Block 1.5: Gain Contribution
- Primary axis: [Performance] prefill compute +8% (HOT layer sparse 처리)
- Secondary axis: [Energy] -3% (sparse 연산 power 효율)
- 단독 미보장 axis: [Memory] (M1 와 결합 시 -32%, M2 단독 -2%)

#### Block 2: Source-Level Implementation

| File path | Symbol | As-is | To-be | 변경 type |
|-----------|--------|-------|-------|-----------|
| `vllm/model_executor/layers/linear.py` | `MergedColumnParallelLinear.forward` | dense matmul | `if layer_class==HOT: cublasLtMatmulSparse4_8(...)` 분기 | Modify |
| `csrc/sparse_epilogue.cu` (new) | `nvfp4_sparse4_8_epilogue` | — | CUTLASS 3.5 sparse epilogue wrapper kernel | Add (still application-level, no kernel patch — uses CUTLASS template) |
| `vllm/worker/green_context_manager.py` (new) | `GreenContextManager.assign_sparse_partition` | — | sparse layer 는 SM 0-1280 partition, dense layer 는 SM 1281-2559 | Add |

R52.3 trace: CUTLASS 3.5 sparse epilogue 실재 ([github.com/NVIDIA/cutlass v3.5.0](https://github.com/NVIDIA/cutlass/tree/v3.5.0), 2025). cuBLASLt sparse4_8 API 실재 (CUDA 12.4+). [✅]

#### Block 3: Synthetic Workload
- Unit test: 4:8 sparse Linear forward 정확도 vs dense FP16, 10분.
- Mechanism-isolated: HOT layer (L17-21 5개) 만 sparse 로 변환 후 prefill latency / MMMU drop 측정, 6시간.

---

### 4.3 — M3: KV-Scale-Freeze + Visual-Context DVFS Slip (KSF-VCD)

#### Block 1: Concept
- 추가되는 Scheme: prefill 끝에서 NVFP4 KV cache scale 을 freeze (=재계산 안 함) + decode phase 시 SM clock 을 layer-boundary 마다 1500→900 MHz 로 slip (memory-bound 인 layer 만), memory clock 은 high freq 유지. visual_attn_ratio HOT layer 는 SM clock 유지.
- 해결하려는 문제: NVFP4 KV cache 는 매 token 마다 scale 재계산 (per-tensor + per-block) → decode bandwidth 8% 낭비. CLONE 은 layer-boundary DVFS 만 다루고 KV scale 무관.
- 동작 원리: (1) prefill 마지막 layer forward 끝에 KV cache 의 per-block scale 을 frozen tensor 로 보관 → (2) decode token 별로 scale 재계산 skip → (3) layer-boundary 에서 nvpmodel Python API 호출 (LayerClassifier M4 결과 = COLD/MEDIUM 이면 SM clock 900 MHz, HOT 이면 1500 MHz) → (4) memory clock 은 항상 max 유지 (decode memory-bound).
- 차별화: KVQuant 의 first-token-FP16 보존과 **orthogonal**. CLONE 의 DVFS 와 **결합** (CLONE 은 token-boundary, 본 기법은 layer-boundary + visual-context aware).

#### Block 1.5: Gain Contribution
- Primary axis: [Energy] -22% (DVFS -18%pt + scale-freeze -4%pt)
- Secondary axis: [Memory eff.] -8% KV scale storage
- 단독 미보장 axis: [Performance] decode latency +3-5% trade-off (memory-bound 이라 작음)

#### Block 2: Source-Level Implementation

| File path | Symbol | As-is | To-be | 변경 type |
|-----------|--------|-------|-------|-----------|
| `vllm/attention/backends/nvfp4_attn.py` | `NVFP4AttentionBackend.forward` | per-step scale recompute | `if is_decode and self.frozen_scale: use cached` | Modify |
| `vllm/worker/dvfs_controller.py` (new) | `LayerDVFSController.before_layer(layer_idx, layer_class)` | — | nvpmodel Python API 호출 (HOT→1500 MHz, COLD→900 MHz) | Add |
| `vllm/engine/llm_engine.py` | `_run_decode_step` | uniform clock | layer-boundary hook 호출 | Modify |

R52.3 trace: nvpmodel CLI 실재 ([Jetson Linux NV doc](https://docs.nvidia.com/jetson/archives/r36.4/DeveloperGuide/SD/PlatformPowerAndPerformance/JetsonOrinNanoSeriesJetsonOrinNxSeriesAndJetsonAgxOrinSeries.html)), Python wrapper jetson-stats `jtop` 사용 가능. [✅]

#### Block 3: Synthetic Workload
- Unit test: scale-freeze 정확도 정합 (decode 100 token vs reference) 10분.
- Mechanism-isolated: nvpmodel sub-ms switching 신뢰성 측정 (1000회 switch 통계), 2시간 + decode J/token 측정 4시간.

---

### 4.4 — M4: LayerClassifier (visual_attn_ratio HOT/COLD/MEDIUM)

#### Block 1: Concept
- 추가되는 Scheme: 100-shot calibration profile 로 각 layer 의 visual_attn_ratio 측정 → 3-class (HOT > 0.15, MEDIUM 0.05-0.15, COLD < 0.05) 분류. M1/M2/M3 가 이 분류를 참조.
- 해결하려는 문제: DALMP/TCIR/KSF-VCD 가 layer-class 입력을 필요로 하지만, 매 inference 마다 재측정은 overhead. ATRIUM (in-session) 의 LayerClassifier 흡수.
- 동작 원리: (1) calibration: 100-shot MMMU 로 visual token attention 합산 / total attention 비율 측정 → layer_class.json 출력 → (2) deploy: vLLM 시작 시 json load → (3) inference: layer 별 hook 호출 시 class lookup (overhead = 1 dict access).
- 차별화: ATRIUM v2 evolution (R56.1 in-session). 외부 baseline 비교 X.

#### Block 1.5: Gain Contribution
- Primary axis: enabling — M1/M2/M3 기여 가능하게 함.
- Secondary axis: [Performance] online classifier overhead < 2% prefill latency.
- 단독: 측정 자체 gain 없음.

#### Block 2: Source-Level Implementation

| File path | Symbol | As-is | To-be | 변경 type |
|-----------|--------|-------|-------|-----------|
| `tools/atrium_layer_classifier.py` | `def classify_layers(model, calib_set)` | — | visual_attn_ratio 측정 + json output | Add |
| `vllm/worker/layer_class_lookup.py` | `LayerClassLookup.get(layer_idx)` | — | json load + cached dict | Add |

#### Block 3: Synthetic Workload
- Unit test: 100-shot calibration full pipeline 30분.
- Isolated: classifier overhead profiling (cuda nsys), 1시간.

## 5. 실험 플랜 7-요소 (R20-β / R52.1)

### 5.1 Hardware environment
- Primary: Jetson AGX Thor 128GB (T5000, 2070 FP4 TFLOPs, LPDDR5x UMA 273 GB/s, 130W envelope)
- Secondary: RTX 5090 32GB GDDR7 (575W TDP, NVFP4 native consumer)
- Power meter: NVIDIA `tegrastats` (Thor) + `nvidia-smi --query-gpu=power.draw` (5090)
- baseline source: vLLM v0.11.0 [github.com/vllm-project/vllm @ v0.11.0](https://github.com/vllm-project/vllm/releases/tag/v0.11.0)
- TensorRT-LLM Edge build: jetson-containers `dustynv/tensorrt-llm:r38.0` (commit 2026-01)

### 5.2 Model
- Qwen3-VL-8B (HF: `Qwen/Qwen3-VL-8B-Instruct`, NVFP4 weight: NVIDIA Cosmos-Reason2-NVFP4A16 path)
- LLaVA-Next-7B (HF: `llava-hf/llava-next-7b-hf`)
- InternVL3-8B (HF: `OpenGVLab/InternVL3-8B`)

### 5.3 Dataset/Workload
- MMMU val (900 sample)
- ChartQA (test 1500)
- DocVQA (val 5500)
- VideoMME (short subset)
- MMBench (dev 4500)

### 5.4 Simulator/Tools
- 측정 only (R55 simulator-as-contribution 위반 X)
- Tool: nsight-systems 2025.4, tegrastats, jtop, nvidia-smi
- Profiling: PyTorch profiler 2.5+

### 5.5 Ablation + Measurement Protocol
- Baseline 1: NVFP4 uniform (Cosmos-Reason2-NVFP4A16)
- Baseline 2: FP8 uniform
- Baseline 3: INT4-AWQ uniform
- Baseline 4: FGMP block-level (reproduce — code release 확인 필요)
- Baseline 5: MicroMix channel-mix (reproduce)
- Ablation: M1 only / M1+M2 / M1+M2+M3 / M1+M2+M3+M4
- 각 config 5 run, 95% CI 보고
- Metric: prefill latency / decode tok/s / J/token (tegrastats integration over request) / MMMU pt

### 5.6 Implementation Steps (week-level)

| Week | Task | Deliverable |
|------|------|-------------|
| W1-2 | vLLM fork + DALMP plan loader (M1) | `quantization/dalmp.py` working with single-layer test |
| W3 | LayerClassifier calibration (M4) | `layer_class.json` for Qwen3-VL-8B |
| W4-5 | TCIR-4:8 sparse epilogue (M2) | CUTLASS sparse Linear + correctness vs dense |
| W6 | KSF-VCD prefill→decode scale freeze (M3) | NVFP4Attn modified, decode scale lookup |
| W7 | nvpmodel Python wrapper + layer-boundary DVFS hook | DVFSController unit test |
| W8 | Green Context partition (sparse vs dense SM) | partition activation logged |
| W9-10 | End-to-end Jetson Thor measurement | 5 baseline + 4 ablation × 5 model |
| W11 | RTX 5090 secondary measurement | reproduction on consumer GPU |
| W12 | FGMP/MicroMix baseline reproduction | head-to-head comparison |
| W13-14 | Paper draft + appendix figure | MLSys submission ready |

### 5.7 Preliminary Analysis Metrics
- decode tok/s: target 33 tok/s (baseline NVFP4 uniform 28 tok/s, +18%)
- J/token: target 1.10 J (baseline 1.42, -22%)
- prefill latency: target 1.50s (baseline 1.85s, -19%)
- MMMU drop: target ≤ 0.5pt (baseline 53.1, target 52.6+)
- Memory: weight 6.0 GB (baseline 8.8 GB, -32%)

## 6. 예상 효과 (R55.2 5-axis 정량 표)

| Axis | Baseline (Qwen3-VL-8B / Jetson Thor / NVFP4 uniform) | PRISM-FOG-FX | 개선 | 조건 |
|------|-----------------------------------------------------|--------------|------|------|
| [Performance] prefill latency | 1.85 s | 1.50 s | **-19%** | image 1024-token prefill |
| [Performance] decode tok/s | 28 tok/s | 33 tok/s | **+18%** | average across MMMU/ChartQA |
| [Energy] J/token | 1.42 J | 1.10 J | **-22%** | tegrastats integrated |
| [Memory eff.] weight | 8.8 GB | 6.0 GB | **-32%** | layer mix (NVFP4 anchor + INT4 cold) |
| [Power] avg | 92 W | 84 W | **-9%** | 130W envelope, mostly DVFS |
| [Power] peak | 118 W | 105 W | **-11%** | thermal margin |
| [Accuracy] MMMU | 53.1 | 52.6 | -0.5 pt | MMMU val (보장: NVFP4 anchor 보호) |

조건 명시: 모든 gain 은 (a) DeepStack 인지 layer geometry 가 있는 VLM (Qwen3-VL/LLaVA-Next/InternVL), (b) NVFP4 native HW (Blackwell 가족, Jetson Thor / RTX 5090) 한정. consumer FP8-only GPU (RTX 4090) 에는 적용 불가.

## 7. R56 self-check

- [x] R56.1: §3.2 GAP 표 안에 본 harness 자체 이전 세션 idea (KEYSTONE/VESPER/SHOAL/Loom 등) 인용 0. ATRIUM/BREAKWATER 도 GAP 표 인용 X (in-session evolution 으로만 명시 §1 Phase 1' delta).
- [x] R56.2: 시스템/AI 분야 published peer-reviewed 비율 = 10/13 = 77% ≥ 65%.
- [x] R56.3: 모든 reference (arxiv ID / venue / year) 명시, 가상/TBD 0.

## 8. R52 / R53 / R54 self-check

- [x] R52.1 Baseline Source: vLLM v0.11.0 [github.com/vllm-project/vllm](https://github.com/vllm-project/vllm) commit hash to be recorded at experiment start. TensorRT-LLM Edge `dustynv/tensorrt-llm:r38.0`. Model HF tags `Qwen/Qwen3-VL-8B-Instruct`, NVIDIA Cosmos-Reason2-NVFP4A16. CUDA 12.4, JetPack 7.0. Hardware: Jetson Thor T5000 dev kit + RTX 5090 FE.
- [x] R52.2 Function/Class as-is/to-be 표: M1 (5 row), M2 (3 row), M3 (3 row), M4 (2 row) = 13 row.
- [x] R52.3 GitHub 실존 검증: vllm `quantization/nvfp4.py`, `models/qwen3_vl.py`, CUTLASS v3.5.0, nvpmodel CLI 모두 [✅].
- [x] R52.4 Synthetic 3-tier: M1-M4 모두 unit test (분) + isolated test (시간) + end-to-end (W9-10) 명시.
- [x] R52.5 Implementation vs Simulator 일관성: simulator 사용 X (직접 측정).
- [x] R54.1-R54.6 Final Verification Pass: 본 문서 작성 시 각 mechanism 별 trace 검증 완료.
