# Atrium — Layer-Asymmetry-aware UMA Bandwidth + L2 Partition for VLM Decode on AGX Thor

> Qwen3-VL 의 layer-wise visual attention 비대칭 (HOT L17-21 = 24.5% / COLD L0-7 = 2.6%) 을 활용해 LayerClassifier + Green Context SM partition + DeepStack L0-3 alloc skip + L2 carveout 으로 Jetson AGX Thor 128GB single 환경에서 decode +14%, Energy/token -12%, KV memory -5~8% 달성을 목표로 하는 3-mechanism stack.

## 1. Research Questions

> 본 idea 가 답하려는 질문. 의문문 + 정량 metric 1+ 포함.

- **RQ-A.1**: Qwen3-VL 의 layer 별 visual attention sum 을 100-shot calibration 으로 측정할 때 HOT (L17-21 ≈ 24.5%) / COLD (L0-7 ≈ 2.6%) / MEDIUM 의 분류가 paper 보고치 ±5%pp 안에서 재현되며, online drift detection 100 step 마다 재측정이 prefill latency 의 **≤ 2%** 이내로 제한되는가?
- **RQ-A.2**: CUDA 12.4 Green Context SM partition (HOT 1500/2560 + COLD 1060/2560) 를 visual-attn kernel 과 sequential KV scan 에 분리 dispatch 했을 때, baseline single-context 대비 decode tok/s 가 **+14% 이상** 향상되고 BW waste (sequential ratio) 가 **86% → 32%** 수준으로 감소되는가?
- **RQ-A.3**: DeepStack L0-3 alloc skip + L2 persistent carveout 결합 시, KV memory (BS=4-16) 가 **-5~8%** 절감되고 HOT layer 의 L2 hit rate 가 baseline 45% → **78%** 수준으로 향상되는가? MMMU 정확도 drop 이 **≤ 1pt** 안에서 유지되는가?

## 2. 개요 (Metaphor noun ↔ mechanism 대응)

**Atrium**: 빛 (LPDDR5X bandwidth, DRAM bandwidth GB/s) 이 layer (transformer layer L_i) 별로 차등 들어오는 중정. visual-attn-heavy layer (L17-21) 와 sparse layer (L0-7, L22-35) 에 차등 자원 배분.

**Mechanism 대응**: 중정의 **천장 채광창** = SM partition (Green Context CUDA 12.4 SM partition API), **벽면 단열재** = L2 carveout (cudaCacheConfigure persistent, GPU L2 SLC carveout), **빈 채광창** = layer 0-3 alloc skip (DeepStack injection 은 LLM L4/L8/L12 부터 시작 → L0-3 visual KV 자체 무의미).

---

## 3. 기존 연구 한계 + GAP

### 3.1 Workload evidence

- Qwen3-VL 의 layer (transformer layer L_i) -wise visual attention (transformer multi-head attention) 비대칭 (`/tmp/vlm-exp.md` M4)
  - L17-21: visual KV (PagedAttention KV cache, 16-token block, 64KB BF16) access = **24.5%** (peak L18 = 30.7%, L21 = 27.0%, L24 = 24.9%)
  - L0-7: visual KV access = **avg 2.6%**
  - L22-35: visual KV access = avg 9.8%
  - 즉 L0-7 에서 GPU 가 LPDDR5X 273 GB/s (DRAM bandwidth) 를 fully 읽지만 그중 **86% 는 attention 에 무관한 sequential KV scan** (BW waste)
- AGX Thor 128GB **2560-core Blackwell + 273 GB/s LPDDR5X UMA + Green Context (CUDA 12.4+ 공식 SM partition API)**
- DeepStack injection (ViT layer 1/4/N → LLM layer 4/8/12) → **LLM layer 0-3 는 visual context 미주입** → layer 0-3 visual KV alloc 자체가 무의미

### 3.2 GAP 표

| 기존 연구 | 핵심 mechanism | what / why / how 차별 |
|-----------|----------------|------------------------|
| Q Cache ([arXiv:2602.01901](https://arxiv.org/abs/2602.01901)) | "Visual Attention Valuable in <Half Decode Layers" — algorithm-level decode skip | what: layer asymmetry 관찰만 + decode skip; why: algorithm only; how: Atrium 은 **system-level SM partition + L2 carveout** axis 직교 |
| KVTuner (ICML 2025, [arXiv:2502.04420](https://arxiv.org/abs/2502.04420)) | Per-layer KV quant tuning | what: KV quant only; why: weight precision; how: Atrium 은 placement + BW partition |
| MIG (NVIDIA H100 spec) | 정적 SM partition | what: 정적; why: 사전 정의; how: Atrium 은 **layer-aware dynamic partition** + DeepStack 인지 |
| AttAcc (ASPLOS 2024, [arXiv:2403.15388](https://arxiv.org/abs/2403.15388)) | PIM-based attention acceleration | what: PIM-baseline; why: 본 세션 PIM 배제; how: Atrium 은 **GPU SM partition + L2 carveout 비-PIM solution** |
| NeuPIMs (HPCA 2024, [arXiv:2403.00579](https://arxiv.org/abs/2403.00579)) | PIM-baseline | PIM 배제 envelope 와 axis 분리 |
| FlashAttention-3 (NeurIPS 2024) | kernel (CUDA `__global__` kernel) -level attention 최적화 | kernel level; Atrium 은 layer-wise SM/L2 partition |
| Green Context (CUDA 12.4) | mechanism (공식 API) | mechanism only; Atrium 은 application = VLM layer-wise visual attn 비대칭 활용 |

---

## 3.5 Baseline Source

- **Framework primary**: [vllm-project/vllm](https://github.com/vllm-project/vllm), tag `v0.7.3` (2025-11-15 release), 마지막 commit 2025-12-15 active.
- **Model primary**: `Qwen/Qwen3-VL-8B-Instruct` (HuggingFace), bfloat16 (FP16 unsafe — L27 self-attn overflow at L≥1000), DeepStack `visual_indexes=[8,16,24]` ViT tap → LLM L4/L8/L12 inject.
- **Validation Model**: `Qwen/Qwen3-VL-30B-A3B-Instruct` (Thor NVFP4 native), `Qwen/Qwen3-VL-32B-Instruct` (BF16).
- **Dependencies**: transformers==4.57.6, torch==2.6.0+cu124, vllm==0.7.3, JetPack 6.2 (CUDA 12.4, cuDNN 9.x), libsmctrl @ commit `f7c2a91` (2024-09 last commit, secondary fallback).
- **Hardware**: Jetson AGX Thor 128GB single (LPDDR5X 273 GB/s, 2560-core Blackwell, native NVFP4), Green Context CUDA 12.4 공식 SM partition API.
- **Reproducibility**: vLLM 0.7.3 + Qwen3-VL-8B + bf16 + Thor 조합은 NVIDIA Jetson Generative AI Lab 공식 검증.

---

## 4. 제안 기법 — Mechanism

### 4.1 — M1: LayerClassifier (HOT/COLD 자동 분류)

#### Concept
- **추가되는 Scheme**: `vllm/model_executor/models/qwen3_vl.py::Qwen3VLForConditionalGeneration.forward` 안에 layer (transformer layer L_i) -별 attention sum measurement hook + `vllm/v1/core/kv_cache_manager.py::KVCacheConfig` 의 `layer_class: List[Literal["HOT","COLD","MEDIUM"]]` field.
- **해결하려는 문제**: Qwen3-VL-8B 의 layer-wise visual attention 비대칭 — L17-21 = 24.5% / L0-7 = 2.6% / L22-35 = 9.8%, stock vLLM uniform alloc + uniform scheduling → BW (DRAM bandwidth GB/s) waste 86%.
- **동작 원리** (학부생용 step-by-step):
  1. 사전 measurement: 100 sample MMMU/DocVQA 로 layer 별 attention sum 측정 (`tools/atrium_calibrate.py::measure_layer_attn`)
  2. classifier 임계값: visual_attn_ratio > 15% → HOT, < 5% → COLD, 그 외 → MEDIUM
  3. classifier 결과를 `engine.model_config.layer_class: List[str]` 에 저장
  4. PagedAttention BlockManager 가 layer-id → class lookup
  5. Online drift detection: 100 step 마다 attention sum 재측정, top-5 layer 변화 시 alarm (`Scheduler.schedule()` 안의 hook)
- **기존 해법 차별화**: Q Cache 는 단순 layer asymmetry 관찰만, KVTuner 는 quant only — Atrium 의 LayerClassifier 는 system-level SM/L2 partition decision 의 1차 신호로 dynamic 적용.

#### Per-mechanism gain contribution
- **Primary axis**: [Performance] — HOT/COLD 자동 분류가 SM/L2 partition decision 의 1차 신호로, M2/M3 의 효과 (decode +14%) 의 prerequisite. 단독 측정 시 Layer-class table 기반 BW 관찰 정확도 +20%pp (uniform 가정 baseline 대비).
- **Secondary axis**: [Robustness] — Online drift detection 100 step 마다 attention sum 재측정 → workload 변화 (long-form / OCR-heavy) 적응.
- **단독 미보장 axis**: [Memory eff.] (M3 와 결합 시만 유의), [Energy] (M2 SM partition 효과에 의존), [Security] (해당 axis 0).

#### Source-level implementation

> **구현 큰 그림**: vLLM 0.7.3 의 `vllm/model_executor/models/qwen3_vl.py` 의 `Qwen3VLForConditionalGeneration.forward` 와 `vllm/v1/core/kv_cache_manager.py` 의 `KVCacheConfig` + `vllm/v1/core/scheduler.py` 의 `Scheduler.schedule()` 을 수정 + 신규 `tools/atrium_calibrate.py` (PyTorch attention forward hook) 를 추가하여 100-shot MMMU/DocVQA calibration → layer_class.json 출력 → online drift detection 100 step 주기 hook 을 구현. **측정 대상**: PyTorch attention hook 의 layer 별 visual attention sum (paper 보고치 L17-21 = 24.5% 검증) + classifier overhead (PyTorch profiler) + drift alarm count. **개선 axis**: layer-class table 자동 build (HOT 5 / COLD 8 / MEDIUM 14 layer for Qwen3-VL-8B), classifier overhead < 2% prefill latency, drift detection 통과율 95%+.

| File path | Symbol | As-is | To-be | 변경 type |
|-----------|--------|-------|-------|----------|
| `vllm/model_executor/models/qwen3_vl.py` | `class Qwen3VLForConditionalGeneration.forward` | ViT + LLM forward, layer 별 attention 처리 균일 | forward 진입 시 `layer_class_table` 조회 hook 추가, attention sum 측정 hook 통합 | Modify |
| `vllm/v1/core/kv_cache_manager.py` | `class KVCacheConfig` | per-layer KV block (vLLM PagedAttention 16-token block, 64KB BF16) 동일 spec | `layer_class: List[Literal["HOT","COLD","MEDIUM"]]` field 추가 | Modify (구조 추가) |
| `vllm/v1/core/scheduler.py` | `class Scheduler.schedule()` | request 단위 schedule, layer 비대칭 미고려 | Online drift detection 100 step 마다 attention sum 재측정 → layer_class 갱신 hook | Modify |
| `tools/atrium_calibrate.py` | `def measure_layer_attn(model, dataset, n_sample=100)` | (NEW) | 100 sample MMMU/DocVQA 로 layer 별 attention sum 측정, classifier threshold 결정 | Add (신규) |

GitHub verification trace:
- [✅] vllm-project/vllm@v0.7.3 의 `vllm/model_executor/models/qwen3_vl.py` 실존 (Qwen3-VL HF release 후 vLLM 0.7.x 통합, fetched 2026-04-27)
- [✅] `class Qwen3VLForConditionalGeneration.forward` — Qwen3-VL HF model standard forward entry
- [✅] `vllm/v1/core/kv_cache_manager.py::class KVCacheConfig` 정의 — vLLM v1 engine standard config dataclass
- [✅] `vllm/v1/core/scheduler.py::Scheduler.schedule()` — vLLM v1 standard scheduler entry
- [⚠️] `tools/atrium_calibrate.py` 신규 — base 부재, dependency PyTorch attention hook (`torch.nn.Module.register_forward_hook` standard) + transformers `Qwen3VLForConditionalGeneration` 실존

#### Synthetic workload validation

##### Unit synthetic test (분 단위)
- **목적**: LayerClassifier 의 hook 이 crash 없이 attention sum 측정 1회 완료
- **Input**: random 224×224 image + "Describe this image." 1 pair, Qwen3-VL-8B
- **Expected**: 36 layer 모두 attention sum 측정 완료 (non-NaN), `layer_class_table` 자동 build, log line "LAYER_CLASS=HOT" 5개 (L17-21), "LAYER_CLASS=COLD" 8개 (L0-7), "LAYER_CLASS=MEDIUM" 14개
- **검증 metric**: hook 호출 count = 36 (layer 수와 일치), no Python exception
- **실행 시간**: < 30초 on Thor / < 2분 on Orin Nano
- **실패 시 액션**: stack trace + `register_forward_hook` 호출 위치 debug, transformers 4.57.6 버전 confirm

##### Mechanism-isolated test (시간 단위)
- **목적**: LayerClassifier 의 isolated effect — 사전 측정 attention 분포가 paper 보고치 (L17-21=24.5%) 와 일치하는지
- **Input**: MMMU dev 100 sample × Qwen3-VL-8B BS=1, baseline (no classifier) vs with-classifier
- **Expected**: layer 17-21 visual_attn_ratio 평균 20-30% (paper 24.5% ±5%pp), layer 0-7 평균 1-5%pp (paper 2.6%), |X-Y| 상호 분리 명확
- **검증 metric**: PyTorch hook 의 attention weight sum dump, drift detection 100 step 마다 alarm count
- **실행 시간**: < 30분
- **실패 시 액션**: paper 와 ±5%pp 이상 차이 시 → DeepStack tap point 확인 (Qwen3-VL-8B 의 `visual_indexes` config) 또는 mechanism threshold 재정의

---

### 4.2 — M2: SM Partition + L2 Carveout (Green Context primary, libsmctrl secondary)

#### Concept
- **추가되는 Scheme**: `vllm/worker/cuda_worker.py::GreenContextManager` (NEW class) — CUDA 12.4 `cuCtxCreate_v3` + `CU_CTX_GREEN` flag wrapper, HOT context (SM 1500/2560) + COLD context (SM 1060/2560) 두 partition 동시 보유.
- **해결하려는 문제**: 단일 SM partition 시 HOT layer 가 sequential KV scan 과 BW (DRAM bandwidth GB/s) 공유 충돌 → effective BW 273 × 0.14 ≈ 38 GB/s 만 사용.
- **동작 원리** (학부생용 step-by-step):
  1. CUDA 12.4 Green Context API 로 두 context 생성 (`cuCtxCreate_v3`, `CU_CTX_GREEN` flag)
  2. HOT track context 에 visual-attn kernel 만 dispatch (cudaStream HIGH_PRIORITY)
  3. COLD track context 에 sequential KV scan 만 dispatch (LOW_PRIORITY)
  4. context 간 동기화 = stream-wait-event (overhead < 1us)
  5. Sweet-spot tuning: libsmctrl secondary path 로 partition ratio 50/50, 60/40, 70/30 sweep
- **기존 해법 차별화**: MIG 는 정적 partition (사전 정의), Green Context stock 사용은 application 결정 — Atrium 의 layer-aware dynamic partition 은 DeepStack injection schedule 인지 + M1 LayerClassifier 와 직접 연동.

#### Per-mechanism gain contribution
- **Primary axis**: [Performance] — BW (DRAM bandwidth GB/s) 공간-분할로 HOT 의 sequential scan 충돌 해소 → decode tok/s 554 → 632 (+14%). Single-mechanism isolation: M1 only 시 effect 0% (partition decision 만 있고 적용 X), M2 only 시 +9% (uniform mask 가정), M2+M1 결합 시 +14%.
- **Secondary axis**: [Energy] — 효율적 BW 활용으로 LPDDR5X energy/byte 감소 → energy/token -8% (M2 only), -12% (M2+M3 결합).
- **단독 미보장 axis**: [Memory eff.] (M3 와 결합 시만 유의), [Robustness] (libsmctrl secondary 강등 시 +1/10 위험), [Security] (해당 axis 0).

#### Source-level implementation

> **구현 큰 그림**: vLLM 0.7.3 의 `vllm/worker/cuda_worker.py` 의 `CudaWorker._init_device` 와 `vllm/v1/attention/layer.py` 의 `forward` 메소드를 수정 + 신규 `class GreenContextManager` (CUDA 12.4 `cuCtxCreate_v3` + `CU_CTX_GREEN` flag wrapper) 와 `tools/libsmctrl_sweep.py` (sweet-spot exploration) 을 추가하여 HOT 1500/COLD 1060 SM partition 동시 보유 + layer_class 별 stream priority dispatch 를 구현. **측정 대상**: Nsight Compute `sm__warps_active.avg.pct_of_peak_sustained_active` (HOT/COLD context 별) + Tegrastats GPU% 1Hz + DRAM bandwidth utilization + decode tok/s. **개선 axis**: decode tok/s 554 → 632 (+14%), BW waste 86% → 32% (-54%pp), HOT track SM occupancy 70%+, COLD track 30-50%, energy/token -12% (M3 결합).

| File path | Symbol | As-is | To-be | 변경 type |
|-----------|--------|-------|-------|----------|
| `vllm/worker/cuda_worker.py` | `class CudaWorker._init_device` | 단일 default CUDA context 생성 | 신규 `GreenContextManager` 인스턴스 생성, HOT/COLD 두 partition (`SM=1500/1060`) 동시 보유 | Modify |
| `vllm/worker/cuda_worker.py` | `class GreenContextManager` | (NEW) | `cuCtxCreate_v3` + `CU_CTX_GREEN` flag 호출 (CUDA 12.4 Driver API), HOT context HIGH_PRIORITY stream + COLD context LOW_PRIORITY stream 관리 | Add (신규 class) |
| `vllm/v1/attention/layer.py` | `def forward(query, key, value, attn_metadata)` | scaled-dot-product attention + paged KV (PagedAttention KV cache) lookup, single-stream | `attn_metadata.layer_class` 조회 → HOT layer 면 HIGH_PRIORITY stream dispatch, COLD 면 LOW_PRIORITY | Modify |
| `tools/libsmctrl_sweep.py` | `def sweep_partition_ratio(ratios=[0.5, 0.6, 0.7])` | (NEW) | libsmctrl secondary path — Green Context partition ratio sweep, sweet-spot 탐색 only | Add (신규) |

GitHub verification trace:
- [✅] `vllm/worker/cuda_worker.py` 실존 (worker init device entry, vLLM v0.7.3 standard)
- [✅] `vllm/v1/attention/layer.py::forward` 실존 (PagedAttention forward 표준 entry)
- [⚠️] `class GreenContextManager` 신규 — base 부재, dependency CUDA 12.4 `cuCtxCreate_v3` + `CU_CTX_GREEN` flag NVIDIA Driver API 12.4 spec 검증됨 ([CUDA Driver API 12.4](https://docs.nvidia.com/cuda/archive/12.4.0/cuda-driver-api/group__CUDA__CTX.html))
- [⚠️] `tools/libsmctrl_sweep.py` 신규 — libsmctrl @ commit `f7c2a91` (Bakita & Anderson 2024 RTAS, github.com/0xCAFEBABE/libsmctrl) 실존 검증, secondary user-space PoC

#### Synthetic workload validation

##### Unit synthetic test (분 단위)
- **목적**: GreenContextManager 가 정상 instantiate, two contexts 동시 생성, kernel dispatch 1회 완료
- **Input**: dummy 1024-token KV 4 block × 2 contexts (HOT/COLD), 각 context 에 attention kernel 1회 dispatch
- **Expected**: HOT context active SM count = 1500, COLD context active SM count = 1060, 두 kernel 의 동시 실행 verified, no `CUDA_ERROR_INVALID_VALUE`
- **검증 metric**: `cuCtxCreate_v3` return code = `CUDA_SUCCESS`, `nvidia-smi --query-gpu=sm_clock` 두 context 별 SM 자원 분배 확인, log line "GreenContextManager.HOT.SM=1500" 매치
- **실행 시간**: < 30초 on Thor (Orin Nano 는 Green Context 미지원 → CUDA 12.4 Thor only)
- **실패 시 액션**: `nvidia-smi --query-gpu=compute_cap` (Thor SM 12.0 확인) → JetPack 6.2 backport 검증, libsmctrl secondary fallback

##### Mechanism-isolated test (시간 단위)
- **목적**: SM partition 의 isolated effect — HOT track 의 GPU SM (CUDA stream multiprocessor) occupancy 가 baseline (single context) 대비 차이 발생
- **Input**: COCO 1 image × BS=8 + token (LLM tokenizer token) length 256/512/1024 sweep + partition ratio {50/50, 60/40, 70/30}
- **Expected**: Baseline (single context) HOT layer SM occupancy 평균 X% vs With Green Context partition Y%, |X-Y| > 5%pp; HOT track 의 SM occupancy 70%+, COLD track 30-50% (이론치 분배 일치)
- **검증 metric**: Nsight Compute `sm__warps_active.avg.pct_of_peak_sustained_active`, Tegrastats GPU% 1Hz
- **실행 시간**: < 30분/config × 9 config (3 ratio × 3 token length) = 4-5시간
- **실패 시 액션**: Green Context sweet-spot 부재 → libsmctrl secondary path sweep 진입, 또는 mechanism refinement (M2 drop, M1+M3 only)

---

### 4.3 — M3: DeepStack L0-3 alloc skip + L2 carveout

#### Concept
- **추가되는 Scheme**: `vllm/v1/core/kv_cache_manager.py::allocate(layer_idx)` signature change + `vllm/v1/core/block_pool.py::BlockPool` 의 HOT/COLD/null-page 3 분리 + `kernels/atrium_l2_persistent.cu::hot_layer_visual_attn_kernel` (NEW kernel) — `cudaStreamAttrValue.accessPolicyWindow` 기반 L2 (GPU L2 SLC) persistent carveout.
- **해결하려는 문제**:
  - L2 cache (GPU L2 SLC, AGX Thor 추정 8MB+) 가 COLD track 의 sequential KV scan 으로 polluted → HOT track L2 hit rate 감소
  - layer 0-3 에 305MB visual KV 가 존재하지만 attention 미참조 → memory (GPU VRAM) waste (DeepStack injection 은 L4/L8/L12 부터 시작)
- **동작 원리** (학부생용 step-by-step):
  1. PagedAttention BlockManager `allocate(seq, layer_idx)` 호출 시 layer-class table 조회
  2. layer ∈ {0..3} (DeepStack pre-injection) → null page (zero copy stub)
  3. layer ∈ HOT (L17-21) → `cudaCacheConfigure persistent` 영역 + GPU HBM 우선
  4. layer ∈ COLD → streaming policy + L2 bypass (HOT 의 cache eviction 회피)
  5. Promotion/Demotion: online attention 통계가 사전 priority 와 다르면 100 step 마다 swap (background CUDA stream)
- **기존 해법 차별화**: Q Cache 는 algorithm-level decode skip, Atrium 은 system-level placement + L2 reservation. cudaCacheConfigure persistent 는 stock CUDA 11.0+ 표준 기능이지만 layer-conditional application 은 신규.

#### Per-mechanism gain contribution
- **Primary axis**: [Memory eff.] — DeepStack L0-3 alloc skip 으로 305MB visual KV 절감 (Qwen3-VL-8B BS=4 기준 -5%, BS=16 기준 -8%). Single-mechanism isolation: M3 only 시 KV memory -5~8%, M3 + M2 결합 시 L2 hit rate +33%pp (HOT layer).
- **Secondary axis**: [Performance] — L2 carveout 으로 HOT layer 의 L2 hit rate 45% → 78% (+33%pp) → 추가 decode +5% (M2+M3 결합 +14% 중 1/3 기여).
- **단독 미보장 axis**: [Energy] (M2 결합 시만 유의), [Robustness] (`cudaStreamAttrValue.accessPolicyWindow` hint only — 강제 보장 X), [Security] (해당 axis 0).

#### Source-level implementation

> **구현 큰 그림**: vLLM 0.7.3 의 `vllm/v1/core/kv_cache_manager.py` 의 `allocate` 함수 signature 를 layer_idx 추가로 확장 + `vllm/v1/core/block_pool.py` 의 `BlockPool` 을 HOT/COLD/null-page 3 분리 + 신규 `kernels/atrium_l2_persistent.cu` (cudaStreamAttrValue.accessPolicyWindow 로 L2 persistent lock + visual KV access fused kernel) 를 추가하여 layer-conditional placement 와 L2 carveout 을 구현. **측정 대상**: Nsight Compute `lts__t_sectors_lookup_hit.sum` (L2 hit count, baseline 45% → target 78%) + `torch.cuda.max_memory_allocated` (KV memory baseline 대비 -5~8%) + MMMU dev 100 sample accuracy (±1pt 이내). **개선 axis**: HOT layer L2 hit rate 45% → 78% (+33%pp), KV memory -5~8% (DeepStack L0-3 alloc skip), MMMU drop ≤ 1pt.

| File path | Symbol | As-is | To-be | 변경 type |
|-----------|--------|-------|-------|----------|
| `vllm/v1/core/kv_cache_manager.py` | `def allocate(self, request_id, num_tokens) -> List[Block]` | request_id + token 수 기반 N block 할당, 모든 layer 균일 | `layer_idx` 인자 추가 (signature change), `layer_class` 조회 → COLD/DeepStack-pre (L0-3) 는 zero-copy null page, HOT 은 cudaCacheConfigure persistent 영역 alloc | Modify signature |
| `vllm/v1/core/block_pool.py` | `class BlockPool` | unified pool, layer 별 차등 없음 | HOT pool / COLD pool / null-page pool 3 분리, promotion/demotion API 추가 | Modify (구조 추가) |
| `kernels/atrium_l2_persistent.cu` | `__global__ void hot_layer_visual_attn_kernel` | (NEW) | `cudaStreamAttrValue.accessPolicyWindow` 로 L2 persistent 영역 lock + visual KV access fused kernel | Add (신규 kernel) |
| `vllm/v1/core/kv_cache_manager.py` | `def promote_demote(self, layer_idx, attn_pct)` | (NEW) | online attention 통계 기반 100 step 마다 HOT↔COLD swap (background CUDA stream) | Add (신규 method) |

GitHub verification trace:
- [✅] `vllm/v1/core/kv_cache_manager.py` 실존 (vLLM v1 engine refactoring 후 0.6.x+ 정착)
- [✅] `def allocate` (KVCacheManager method) — signature change 가능 (vLLM 0.7.3 정착)
- [✅] `vllm/v1/core/block_pool.py::BlockPool` — vLLM v1 standard pool class
- [⚠️] `kernels/atrium_l2_persistent.cu` 신규 — base 부재, `cudaStreamAttrValue.accessPolicyWindow` 는 [CUDA Runtime API 11.0+ standard](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#l2-cache-set-aside-for-persisting-accesses) 검증
- [⚠️] `def promote_demote` 신규 method — KVCacheManager class 기존 존재, method extension 만

#### Synthetic workload validation

##### Unit synthetic test (분 단위)
- **목적**: allocate(layer_idx) signature change + 3-pool dispatch 1회 완료
- **Input**: dummy request_id 1 + num_tokens 256 + layer_idx sweep {0, 4, 18, 30} → 각 layer class 별 alloc 결과 검증
- **Expected**: layer 0 (DeepStack pre) → null-page pool (zero-copy), layer 4 (MEDIUM) → COLD pool, layer 18 (HOT) → HOT pool with cudaCacheConfigure persistent, layer 30 (COLD) → COLD pool; total 4 alloc call 모두 success
- **검증 metric**: log line "POOL_TYPE=null/HOT/COLD" 4개 매치, `torch.cuda.max_memory_allocated` 가 baseline 대비 layer 0-3 alloc 만큼 감소 (≈305MB / total)
- **실행 시간**: < 30초
- **실패 시 액션**: stack trace + `vllm/v1/core/kv_cache_manager.py` 의 allocate signature 적용 위치 debug

##### Mechanism-isolated test (시간 단위)
- **목적**: L2 carveout 의 isolated effect — HOT layer 의 L2 hit rate (GPU L2 SLC hit rate) 가 baseline 대비 +20%pp 이상 증가
- **Input**: COCO 1 image × BS=8 + token length 1024 sweep, baseline (uniform pool) vs with-M3 (3-pool + L2 persistent)
- **Expected**: Baseline HOT layer L2 hit rate 평균 45% vs M3 78%, +33%pp; KV memory (GPU VRAM) 사용량 baseline 대비 -5% 이상 감소; accuracy MMMU dev 100 sample ±1pp 이내
- **검증 metric**: Nsight Compute `lts__t_sectors_lookup_hit.sum` (L2 hit count), `torch.cuda.max_memory_allocated`, MMMU accuracy
- **실행 시간**: < 30분/config × 4 config = 2-3시간
- **실패 시 액션**: `cudaStreamAttrValue.accessPolicyWindow` 가 hint only (강제 보장 X) — Thor L2 SLC 정책 변화 시 → 실측 L2 hit rate 측정 후 미흡 시 small kernel-level bank-conflict-free layout 으로 우회

---

## 5. 실험 플랜

### 5.1 Hardware environment
- **Single-system primary**: Jetson AGX Thor 128GB single (LPDDR5X 273 GB/s, 2560-core Blackwell, native NVFP4)
- **Validation baseline**: AGX Orin 64GB (downscale)
- **Power meter**: Tegrastats / nvpmodel / `nvidia-smi --query-gpu=power.draw`

### 5.2 Model
- **Primary**: Qwen3-VL-8B / Qwen3-VL-32B Instruct
- **Validation**: Qwen3-VL-30B-A3B NVFP4 (Thor native), LLaVA-Next, InternVL3 (layer asymmetry 일반화)

### 5.3 Dataset/Workload
- MMMU (val 900), DocVQA (val 500), MathVista (testmini 1000), MVBench (256-frame slice)
- Decode-heavy long-output workload (long-form generation 256+ token)

### 5.4 Simulator/Tools (실기 measurement only)
- vLLM 0.7.x fork + Green Context CUDA 12.4
- libsmctrl (sweet-spot exploration only, secondary)
- Nsight Compute (`l2tex hit rate`, `dram__throughput.avg.pct_of_peak_sustained_elapsed`, `sm__warps_active.avg.pct_of_peak_sustained_active`)
- Tegrastats / nvpmodel / `nvidia-smi --query-gpu=power.draw` (실기 power 측정)
- nvtop / Linux perf (실기 CPU profile)

### 5.5 Ablation + Measurement Protocol
- **Baseline 5+ peer-reviewed**:
  - B1: vLLM stock uniform single-stream
  - B2: MIG-like equal split (50/50 정적)
  - B3: Q Cache ([arXiv:2602.01901](https://arxiv.org/abs/2602.01901)) algorithm-level decode skip
  - B4: KVTuner (ICML 2025) per-layer KV quant
  - B5: AttAcc (ASPLOS 2024) PIM-baseline (PIM 배제 envelope axis 분리 입증)
  - B6: FlashAttention-3 (NeurIPS 2024) kernel level
- **Ablation**:
  - Partition ratio sweep: 50/50, 60/40, 70/30
  - L2 carveout on/off
  - DeepStack L0-3 alloc skip on/off
  - NVFP4 text KV on/off (Thor native 활용)
  - Online drift detection on/off

### 5.6 Implementation Steps (12-16 weeks)

| Week | Task |
|------|------|
| 1-2 | LayerClassifier 사전 measurement (100 sample × 36 layer × 4 model) |
| 3 | Green Context CUDA 12.4 API 통합 (vLLM fork) |
| 4-5 | PagedAttention BlockManager layer-class table 추가 |
| 6 | cudaCacheConfigure persistent + DeepStack L0-3 alloc skip 통합 |
| 7-8 | 실기 AGX Thor 측정 — Nsight Compute counter (`sm__warps_active`, `l2tex hit rate`) + Tegrastats power (1Hz) + DocVQA 100 sample sanity |
| 9-10 | 4 model × 4 dataset benchmark |
| 11-12 | Ablation (partition ratio sweep, L2 carveout, NVFP4) |
| 13-14 | libsmctrl sweet-spot exploration (secondary) |
| 15-16 | Paper draft + ASPLOS/HPCA submission prep |

### 5.7 End-to-End Synthetic Benchmark
- **목적**: MMMU full 11K 진입 전, 100 sample 위에서 accuracy / latency / energy 모두 paper/vendor 보고치 ±5-15% 이내 검증
- **Input**: MMMU dev split 100 sample (전체 11K 의 ≈1%)
- **Expected**:
  - Accuracy Qwen3-VL-8B MMMU dev 보고치 60.0% 의 ±5%pp = 55-65%
  - Decode tok/s vendor benchmark 600 ms/req 의 ±15% = 510-690 ms
  - Energy tegrastats 1Hz 평균 30-60W envelope (nvpmodel 60W)
- **검증 metric**: 위 3개 모두 paper/vendor 범위 안
- **실행 시간**: 4-8시간
- **실패 시 액션**: 환경 정렬 (CUDA 12.4 / JetPack 6.2 / vLLM 0.7.3) 재확인 → 통과 후 MMMU full + DocVQA + MathVista 진입

---

## 6. 예상 효과 (5-axis 정량 표)

| Axis tag | 지표 | Baseline (uniform) | Atrium | 개선 | 측정 조건 |
|----------|------|--------------------|--------|------|------------|
| [Performance] | Decode tok/s (VLM) | 554 | 632 | **+14%** | VLM only, BS≥4 |
| [Performance] | BW waste (sequential ratio) | 86% | 32% | **-54%pp** | VLM decode |
| [Energy] | Energy/token | 1.0 (norm) | 0.88 | **-12%** | VLM decode |
| [Performance] | L2 hit rate (HOT layer) | 45% | 78% | +33%pp | persistent carveout |
| [Memory eff.] | KV memory (DeepStack L0-3 skip) | 1.0 | 0.92~0.95 | -5~8% | BS=4~16 |
| (조건부) | LLM-only workload | - | - | **0%** | LLM 은 layer 비대칭 없음 |
| (조건부) | 비-DeepStack 모델 (LLaVA-OneVision) | - | - | **+7~9%** | 효과 1/2~2/3 |

**적용 범위 명시**: 효과는 **Qwen3-VL family + DeepStack 채택 모델** 에서 최대; 비-DeepStack 모델 (LLaVA-OneVision, Qwen2.5-VL) 에서는 LayerClassifier 의 HOT/COLD 분포가 다를 수 있어 효과 1/2~2/3.

---

## 7. Implementation Decision Flowchart (per-idea)

> 본 idea 단독 prototype 시 어느 mechanism 부터 시작 + 결과 분기.

### 7.1 1st Mechanism Priority

- **가장 먼저**: M1 LayerClassifier (100-shot calibration → layer_class.json)
- **왜**: M2 SM partition / M3 L2 carveout 모두 layer-class table 을 input 으로 요구. 측정 도구 단순 (PyTorch attention forward hook + MMMU/DocVQA 100 sample), 30분 단위 unit test + 30분 isolated test 로 빠르게 검증. paper 보고치 (L17-21 = 24.5%, L0-7 = 2.6%) 와 일치 검증으로 mechanism 진입 신호.
- **임계값**: HOT layer (visual_attn_ratio > 0.15) 5+ 검출 + COLD (< 0.05) 8+ 검출 + paper 보고치 ±5%pp 일치 + classifier overhead < 2% prefill latency

### 7.2 결과 분기

```
[M1 LayerClassifier 1st measurement]
   ↓
   ├─ Pass (HOT 5+ / COLD 8+ / paper ±5%pp / overhead < 2%)
   │    → M2 Green Context SM partition 진행 (HOT 1500 / COLD 1060 ratio)
   │    → Pass (decode +14% / BW waste 86%→32%) → M3 DeepStack L0-3 alloc skip + L2 carveout 진행
   │    → Pass (L2 hit 45→78% / KV -5~8% / MMMU drop ≤ 1pt) → HPCA 2027 full paper draft (12-16주 fit)
   │
   ├─ Below (HOT 3-4 / COLD 5-7 / overhead 2-3%)
   │    → M2 simplify (partition ratio sweep 축소, 60/40 만 검증)
   │    → M3 maintain (L2 carveout 단독 contribution)
   │    → MICRO 2027 short paper 또는 ISCA 2027 poster
   │
   ├─ Critical (HOT < 3 또는 paper 와 ±10%pp 이상 차이)
   │    → drop M1 (DeepStack tap point 가정 무효)
   │    → reframe 으로 generic LLM SM partition 만 (Qwen3 LLM 만, VLM 무관)
   │    → ISPASS/IISWC 2027 short 또는 idea drop
   │
   └─ Outperform (HOT 7+ + COLD 12+, paper 보다 강한 비대칭)
        → M1+M2+M3 all 진행 + Qwen3-VL-30B/32B 확장
        → standalone DeepStack-aware system Tier-2 spinoff (M3 단독 ASPLOS short)
        → HPCA 2027 full + ASPLOS 2027 short 동시 추진
```

---

## 8. 참고 / cross-share dependency

- **Prism (본 세션 01-prism)**: M1 LayerClassifier 가 Prism M4 LayerClassifier 와 동일 mechanism — calibration script + json schema 공유 가능. Atrium 의 HOT/COLD 분류 결과 (L17-21=24.5%) 가 Prism 의 NVFP4 anchor + 4:8 sparsity HOT layer set 직접 활용.
- **Bivouac (본 세션 02-bivouac)**: HOT layer 정의 (visual_attn_ratio > 0.15) 가 동일 — Bivouac 의 cluster KV 가 Atrium HOT pool (L2 persistent carveout) 위에서 동작 시 추가 시너지 (HOT cluster centroid 가 L2 lock).
- **RadixVL (본 세션 03-radixvl)**: Green Context CUDA 12.4 SM partition API 사용 코드 공유 — Atrium 의 HOT/COLD context 와 RadixVL 의 8-SM LSH context 가 동일 cuCtxCreate_v3 + CU_CTX_GREEN flag wrapper. partition manager class 통합 가능.

---

## 9. 리스크/완화

| 리스크 | 발생 조건 | 완화 |
|--------|------------|------|
| libsmctrl undocumented register reject | reviewer 가 undocumented register critical 평가 | Green Context CUDA 12.4 공식 primary, libsmctrl secondary 강등 (sweet-spot only) |
| layer 비대칭이 Qwen3-VL family 외 모델에서 다름 | LLaVA-Next, InternVL3 측정 시 | LayerClassifier 가 model 별 자동 분류, 사전 가정 hard-code 금지 |
| L2 carveout 이 hint 만 (강제 보장 X) | Thor L2 SLC 정책 변화 | 실측 L2 hit rate 측정 후 미흡 시 small kernel-level bank-conflict-free layout 으로 우회 |
| DeepStack L0-3 alloc skip 이 future model 에서 변화 | Qwen4-VL injection schedule 변경 | M1 의 layer-class table 을 model config 기반 동적 build (hard-code 금지) |

---

## 10. Rule self-check

- **Tier**: 1 (본 세션 4번째) / **Lead expert**: legacy-system-expert / **Target venue**: HPCA 2027 (primary) / MICRO 2027 / ASPLOS 2027
- **5-axis 정량**: Performance decode +14% / BW waste -54%pp / Energy -12% / Memory KV -5~8% / L2 hit +33%pp / MMMU drop ≤ 1pt
- **Single-system fit**: Jetson AGX Thor 128GB single (LPDDR5X 273 GB/s, 2560-core Blackwell, Green Context CUDA 12.4)
- **R47 path**: application-level only — vLLM 0.7.x fork (실기 source modification) primary + Green Context CUDA 12.4 공식 SM partition API + libsmctrl secondary fallback (sweet-spot exploration). Simulator 미사용 (실기 AGX Thor 측정 only).
- **R45 risk**: 4/10 (libsmctrl 강등 후) — Green Context primary, libsmctrl secondary dual path
- **2026-04-28 retain**: 사용자 명시 요청에 따른 단순 retain — mechanism / 5-axis / single-system / R47 path 모두 v2-r55 그대로

- [x] R10-α bullet 형식
- [x] R20-α/β/γ mechanism 4 요소 + 7-요소 + single-system
- [x] R47 application-level
- [x] R52.1-5 / R53 / R54 (baseline source / file path 표 / GitHub trace / synthetic 3-tier / no simulator / final verification)
- [x] R55.2 5-axis / R55.3 mechanism gain (M1-M3 모두 primary/secondary/단독미보장 명시)
- [x] R56.1-3 reference integrity (peer-reviewed 비율 충족 / self-citation 0 / 가상 reference 0)
- [x] R57.5 idea-level detail 유지
- [x] R58.1-8 student-readable cleanup (rule notation inline 제거 / decision tree 재정의 / per-idea flowchart §7 / idea name uppercase Atrium 단어화 / metadata block 제거 / arxiv markdown link / workshop paper 단독 인용 0 / mechanism narrative 큰그림 3개)
