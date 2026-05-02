# T2.3 Hearth — Document VLM L2 Carveout + Tile Locality Boost

## 1. Research Questions

- **RQ6.1**: Mosaic scenario classifier 가 document VLM (DocVQA / ChartQA / OCRBench) 식별 시 `cudaAccessPolicyWindow` tile residence boost 가 tile attention latency -25%+ 가능한가?
- **RQ6.2**: AwaRes / VisionThink / AttWarp 의 algorithm-side 와 직교 axis (system-side L2 carveout) 로 paper-worthy contribution 도출 가능한가?

## 2. 개요

Document VLM (DocVQA / ChartQA / OCRBench) 의 tile attention 은 tile-locality 강함 — same tile 이 multiple attention head 에서 반복 read. AwaRes / VisionThink / AttWarp 는 algorithm-side spatial / RL / warp-attention, **system-side L2 cache 활용 미커버**. Hearth 는 (M1) Mosaic 의 document scenario dispatch hook, (M2) `cudaAccessPolicyWindow` tile residence boost, (M3) tile attention eviction priority 조정.

## 3. 기존 연구 한계 / GAP

- **AwaRes** ([arXiv:2603.16932](https://arxiv.org/abs/2603.16932)) — spatial on-demand 36% tokens, **system-side L2 carveout 미커버**.
- **VisionThink** ([ICLR 2026, arXiv:2507.13348](https://arxiv.org/abs/2507.13348)) — RL autonomous resolution, **L2 carveout 미커버**.
- **AttWarp** (ICLR 2026) — DocVQA attention warp, **system-side 미커버**.

**GAP**: Document VLM 의 system-side L2 carveout (`cudaAccessPolicyWindow`) + tile residence boost paper 0편 (HW arch novelty).

## 4. 제안 기법 (3 mechanism)

### 4.1 M1 — Mosaic Document Scenario Dispatch Hook

#### 동작 원리 (Mechanism)
- **추가 Scheme**: T1.1 Mosaic scenario classifier 가 document VLM 식별 시 Hearth carveout 활성화 hook 추가.
- **동작 원리**:
  1. Mosaic classifier output `request.metadata['scenario'] == 'document'` 검출
  2. Document scenario 시 Hearth `enable_l2_carveout()` 호출
  3. Inference 종료 시 `disable_l2_carveout()`
- **차별화**: 일반 VLM serving 은 scenario-agnostic L2, Hearth 는 document scenario-specific.

#### 기대 효과 (Gain)
- **Primary** [Performance]: Document scenario 한정 latency -25%+

#### 구현 변경점 (Code Changes)

| File path | Class · Function · Line region | As-is | To-be | 변경 type |
|-----------|-------------------------------|-------|-------|----------|
| (new) `vllm/v1/scenario/hearth_carveout.py` | `class HearthCarveoutManager` | (신규) | **신규 class**: `enable(stream)` / `disable(stream)`. invariant: `cudaAccessPolicyWindow` set/unset. | Add |
| `vllm/v1/core/scheduler.py` | `def _schedule_running(self, requests)` (line 380-460) | scenario-agnostic | **수정 line block**: line 410 직후 `for req in requests: if req.metadata['scenario'] == 'document': hearth.enable(req.stream)`. **invariant**: per-request L2 carveout. | Modify |

#### GitHub 실존 검증
- vLLM `Scheduler._schedule_running` 실재. [✅]
- CUDA `cudaAccessPolicyWindow` ([NVIDIA CUDA Runtime API](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html)) 실재. [✅]

#### 검증 시나리오 (Test Plan)
##### Unit test
- **목적**: Hook activation correctness
- **Input**: 6-scenario mock requests
- **Expected**: document scenario only L2 carveout enabled
- **실행 시간**: < 30초

### 4.2 M2 — `cudaAccessPolicyWindow` Tile Residence Boost

#### 동작 원리 (Mechanism)
- **추가 Scheme**: Document VLM tile attention 의 hot tile (high reuse) 에 `cudaAccessPolicyWindow` 로 L2 cache residence boost.
- **동작 원리**:
  1. Tile attention 진입 시 `cudaStreamSetAttribute` 로 access policy 설정
  2. `accessPolicyWindow.hitRatio = 1.0` (hot tile 100% L2 retain)
  3. `missProp = cudaAccessPropertyStreaming` (cold data 는 stream-through)
- **차별화**: AwaRes / VisionThink 는 algorithm-side spatial, Hearth 는 system-side L2.

#### 기대 효과 (Gain)
- **Primary** [Performance]: Tile attention L2 hit rate +30%, latency -25%

#### 구현 변경점 (Code Changes)

| File path | Class · Function · Line region | As-is | To-be | 변경 type |
|-----------|-------------------------------|-------|-------|----------|
| `vllm/v1/scenario/hearth_carveout.py` | `HearthCarveoutManager.enable` (M1 신규) | basic activation | **수정 line block**: `cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &window)` + `window.hitRatio = 1.0; window.missProp = cudaAccessPropertyStreaming`. **invariant**: hot tile L2 retention. | Modify |

#### GitHub 실존 검증
- CUDA `cudaStreamSetAttribute` ([NVIDIA CUDA Runtime API](https://docs.nvidia.com/cuda/cuda-runtime-api/)) 실재. [✅]

#### 검증 시나리오 (Test Plan)
##### Unit test
- **목적**: L2 carveout activation
- **Input**: synthetic tile attention 10 iter
- **Expected**: L2 hit rate (Nsight Compute counter `lts__t_sectors_hit_rate`) +30%
- **실행 시간**: < 30초

##### Mechanism-isolated test
- **목적**: DocVQA latency 측정
- **Input**: DocVQA 100 sample
- **Expected**: tile attention latency -25%
- **실행 시간**: ~ 3시간

### 4.3 M3 — Tile Attention Eviction Priority

#### 동작 원리 (Mechanism)
- **추가 Scheme**: L2 cache eviction 시 tile attention 의 hot tile 보호.
- **동작 원리**:
  1. `cudaAccessPropertyPersisting` 으로 hot tile 마킹
  2. `cudaCtxResetPersistingL2Cache` 로 inference 종료 시 reset
- **차별화**: 일반 L2 LRU 와 차별 — explicit persisting.

#### 기대 효과 (Gain)
- **Primary** [Performance]: 추가 latency -5%

#### 구현 변경점 (Code Changes)

| File path | Class · Function · Line region | As-is | To-be | 변경 type |
|-----------|-------------------------------|-------|-------|----------|
| `vllm/v1/scenario/hearth_carveout.py` | `HearthCarveoutManager.disable` | basic disable | **수정**: `cudaCtxResetPersistingL2Cache()` 호출 + persisting tile reset. | Modify |

#### GitHub 실존 검증
- CUDA `cudaCtxResetPersistingL2Cache` ([NVIDIA CUDA Driver API](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__CTX.html)) 실재. [✅]

## 5. 평가 / 실험 플랜

1. **Hardware**: RTX Pro 6000 96GB (Blackwell, L2 cache 큰 size).
2. **Model**: Qwen3-VL-30B-A3B-Instruct (document VLM 기능 보유).
3. **Benchmark**: **DocVQA** (DocVQA 2026 ICDAR), **ChartQA**, **OCRBench**, **AI2D**.
4. **Tools**: vLLM v0.10+, Nsight Compute (counter `lts__t_sectors_hit_rate`).
5. **Ablation**: 2³ (M1 dispatch × M2 carveout × M3 eviction priority). Baseline: vLLM no carveout, AwaRes (algorithm-side).
6. **Implementation Steps**: W1-2 dispatch hook / W3-4 cudaAccessPolicyWindow / W5 eviction priority / W6-7 DocVQA/ChartQA eval / W8 DATE letter.
7. **Preliminary Metrics**: L2 hit rate, tile attention latency, document VLM accuracy.

## 6. 예상 효과

| 지표 | Baseline (vLLM no carveout) | Hearth | 개선 |
|------|-----------------------------|--------|------|
| [Performance] Tile attention latency (DocVQA) | 1.0× | 0.70-0.75× | -25 ~ -30% |
| [Performance] L2 hit rate | baseline | +30pp | new |
| [Quality] DocVQA accuracy | baseline | 동일 | preserved |

## 7. Decision Tree

- **Pass (latency -25%+)**: DATE 2027 / IEEE ESL letter.
- **Below**: vLLM contribution PR.
- **Outperform**: MLSys workshop / Mosaic paper pair.

## 8. 기준 코드베이스

- vLLM v0.10.x
- Models: `Qwen/Qwen3-VL-30B-A3B-Instruct`
- Deps: `transformers==4.57.6`, `torch==2.6.0`, `cuda==12.4`
- Hardware: RTX Pro 6000 96GB (Blackwell L2 cache)

## 9. Self-Check

R31 (document VLM enterprise common case), R52.1-R52.5, R53 일반 단어, R55.2 (Performance + Memory), R19-α (Hearth 자연어 + full title vendor-neutral except CUDA generic), R28 모두 통과.
