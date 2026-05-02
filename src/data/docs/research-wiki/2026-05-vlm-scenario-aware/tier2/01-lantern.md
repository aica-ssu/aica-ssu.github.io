# T2.1 Lantern — NVDEC-Coupled Sliding-Window KV Eviction for Streaming Video

## 1. Research Questions

- **RQ4.1**: NVDEC frame metadata (frame stride / motion vector magnitude) + sliding window K=8 frame retention 으로 streaming video real-time 처리 시 latency < 100ms/frame + Q-Bench-Video accuracy 보존 가능한가?
- **RQ4.2**: Window 외 frame KV 즉시 evict + GPU memory recycle 로 1-hour streaming session 의 GPU memory bounded constant 유지 가능한가?

## 2. 개요

Streaming video real-time scenario (CCTV / robot agent / live transcription) 은 sliding window 기반 처리 필수 — 1-hour session 의 모든 frame KV 보존 시 GPU memory 폭발. Lantern 는 (M1) NVDEC frame metadata extract, (M2) sliding window K=8 retention, (M3) 윈도우 외 frame KV 즉시 evict.

## 3. 기존 연구 한계 / GAP

- **Nova** ([arXiv:2509.21301](https://arxiv.org/abs/2509.21301)) — agentic VLM real-time serving, **NVDEC HW co-design 미커버**.
- **One-Token-per-Frame** ([arXiv:2604.14149](https://arxiv.org/abs/2604.14149)) — frame compression, **streaming sliding window 미커버**.
- 이전 세션 Cadence (2026-04-30 vlm-rtx6000-cpu-cudagraph) — MRoPE T-axis, **NVDEC HW frame metadata + sliding window 미커버**.

**GAP**: NVDEC HW frame metadata 와 sliding window KV eviction 의 co-design paper 0편 (HW co-design unique axis).

## 4. 제안 기법 (3 mechanism)

### 4.1 M1 — NVDEC Frame Metadata Extract

#### 동작 원리 (Mechanism)
- **추가 Scheme**: vLLM video pipeline 에 NVDEC API hook 추가 → frame metadata (frame stride / motion vector magnitude / keyframe flag) 추출.
- **해결 문제**: 기존 video pipeline 은 NVDEC 만 frame decode 용도, metadata 활용 안함.
- **동작 원리**:
  1. NVDEC `cuvidParseVideoData` callback 에서 frame metadata extract
  2. Per-frame motion vector magnitude 계산
  3. Metadata 를 `request.metadata['frame_meta']` 에 attach
- **차별화**: Nova / One-Token-per-Frame 은 software-only, NVDEC HW co-design 미시도.

#### 기대 효과 (Gain)
- **Primary** [Performance]: Frame metadata extract latency < 1ms/frame
- **Secondary** [Energy]: NVDEC HW unit 활용 (GPU compute 절감)

#### 구현 변경점 (Code Changes)

| File path | Class · Function · Line region | As-is | To-be | 변경 type |
|-----------|-------------------------------|-------|-------|----------|
| (new) `vllm/multimodal/nvdec_hook.py` | `class NVDECMetadataExtractor` | (신규 module) | **신규 class**: PyNvVideoCodec / cuvid wrapper, `extract(frame) -> {'stride': int, 'motion_mag': float, 'keyframe': bool}`. invariant: NVDEC HW unit 활용. | Add |
| `vllm/multimodal/processing/processor.py` | `class BaseMultiModalProcessor.apply` (line 218-285) | OpenCV / decord frame decode | **수정 line block**: line 235 직후 video input 분기 추가 — `if input_type == 'video' and use_nvdec: frame_meta = nvdec_hook.extract(frame); request.metadata['frame_meta'] = frame_meta`. **invariant**: NVDEC fallback to OpenCV on unsupported codec. | Modify |

#### GitHub 실존 검증
- vLLM `vllm/multimodal/processing/` (type=dir, sub-files=5) 실재. [✅]
- vLLM `vllm/multimodal/processing/processor.py` (type=file) + `class BaseMultiModalProcessor` 실재. [✅]
- **R52.3 정정 trace**: initial `vllm/multimodal/processing.py` 는 subpackage refactor 로 directory. 정정 적용. [⚠️→✅]
- PyNvVideoCodec ([github.com/NVIDIA/PyNvVideoCodec](https://github.com/NVIDIA/PyNvVideoCodec)) 실재 (NVIDIA 공식). [✅]
- NVDEC `cuvidParseVideoData` API ([NVIDIA Video Codec SDK](https://developer.nvidia.com/nvidia-video-codec-sdk)) 실재. [✅]

#### 검증 시나리오 (Test Plan)
##### Unit test
- **목적**: NVDEC metadata extract correctness
- **Input**: synthetic 30-frame H.264 video
- **Expected**: 30 frame metadata extracted, motion_mag distribution non-zero
- **실행 시간**: < 30초

##### Mechanism-isolated test
- **목적**: NVDEC overhead measurement
- **Input**: 1-hour streaming video synthetic
- **Expected**: extract latency < 1ms/frame, NVDEC utilization 80%+
- **실행 시간**: ~ 1시간

### 4.2 M2 — Sliding Window K=8 Frame Retention

#### 동작 원리 (Mechanism)
- **추가 Scheme**: vLLM `KVCacheManager` 에 sliding window enforcement — frame index > current - K 인 frame KV 만 retain.
- **동작 원리**:
  1. Per-frame KV 에 frame_idx 첨부
  2. Decoding step 진입 시 current_frame_idx 갱신
  3. frame_idx <= current - K 인 KV block evict
- **차별화**: 기존 LRU eviction 대신 explicit window enforcement.

#### 기대 효과 (Gain)
- **Primary** [Memory]: 1-hour streaming session GPU memory bounded constant (window size × frame KV)

#### 구현 변경점 (Code Changes)

| File path | Class · Function · Line region | As-is | To-be | 변경 type |
|-----------|-------------------------------|-------|-------|----------|
| `vllm/v1/core/kv_cache_manager.py` | `def evict_lru(self, num_blocks)` (line 320-380) | LRU eviction | **수정 line block**: line 340 직후 `if request.scenario == 'streaming-video': evict_outside_window(current_frame_idx, K=8)` 추가. **invariant**: window enforcement priority over LRU. | Modify |
| (new) `vllm/v1/core/sliding_window_evictor.py` | `class SlidingWindowEvictor` | (신규) | **신규 class**: `evict_outside_window(current_idx, K)`. invariant: K configurable per-scenario. | Add |

#### GitHub 실존 검증
- vLLM `evict_lru` 실재. [✅]

#### 검증 시나리오 (Test Plan)
##### Unit test
- **목적**: Window eviction correctness
- **Input**: 100-frame streaming, K=8
- **Expected**: 항상 8 frame KV 만 GPU 에 잔존
- **실행 시간**: < 1분

##### Mechanism-isolated test
- **목적**: 1-hour streaming GPU memory measurement
- **Input**: 1-hour synthetic 30fps video
- **Expected**: GPU memory constant (window 8 frame × KV/frame)
- **실행 시간**: ~ 1시간

### 4.3 M3 — Window 외 Frame KV 즉시 Evict + GPU Memory Recycle

#### 동작 원리 (Mechanism)
- **추가 Scheme**: M2 의 evict 시 즉시 GPU memory free + new frame allocation 에 재사용.
- **동작 원리**:
  1. Window 외 frame KV evict 시 `cudaFree` 즉시 호출
  2. New frame allocation 시 freed memory pool 우선 재사용
  3. Memory fragmentation 방지 (fixed-size block pool)
- **차별화**: 기존 LRU 는 lazy free, Lantern 는 eager free.

#### 기대 효과 (Gain)
- **Primary** [Memory]: Memory fragmentation -50%, throughput +10%

#### 구현 변경점 (Code Changes)

| File path | Class · Function · Line region | As-is | To-be | 변경 type |
|-----------|-------------------------------|-------|-------|----------|
| `vllm/v1/core/sliding_window_evictor.py` | `evict_outside_window` (M2 신규) | evict only | **수정 line block**: evict 후 `cudaFree(block.ptr); freed_pool.append(block.ptr)` 추가. **invariant**: freed memory 즉시 재사용 가능. | Modify |

#### GitHub 실존 검증
- CUDA `cudaFree` API ([NVIDIA CUDA Runtime API](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html)) 실재. [✅]

#### 검증 시나리오 (Test Plan)
##### Unit test
- **목적**: Memory recycle correctness
- **Input**: 1000 frame streaming, K=8
- **Expected**: GPU memory peak < (window × frame KV × 1.1) (10% fragmentation budget)
- **실행 시간**: < 1시간

## 5. 평가 / 실험 플랜

1. **Hardware**: RTX Pro 6000 96GB.
2. **Model**: Qwen3-VL-30B-A3B-Instruct.
3. **Benchmark**:
   - **Q-Bench-Video** (CVPR 2025, video quality understanding)
   - Streaming synthetic (1-hour 30fps video)
4. **Tools**: vLLM v0.10+, PyNvVideoCodec, Nsight Systems (NVDEC utilization).
5. **Ablation**: 2³ (M1 NVDEC on/off × M2 sliding window on/off × M3 eager free on/off). Baseline: vLLM standard (no streaming).
6. **Implementation Steps**: W1-2 NVDEC hook / W3-4 sliding window / W5 eager free / W6-7 evaluation / W8 IEEE CAL letter writing.
7. **Preliminary Metrics**: per-frame latency, GPU memory peak, NVDEC utilization, accuracy delta.

## 6. 예상 효과

| 지표 | Baseline (no streaming) | Lantern | 개선 |
|------|------------------------|---------|------|
| [Performance] Per-frame latency | 200-300ms | < 100ms | -50%+ |
| [Memory] 1-hour session GPU memory peak | 30GB+ (unbounded) | 1.5GB (window 8 × KV) | -95% |
| [Energy] NVDEC utilization | 0% | 80%+ | new |

## 7. Decision Tree

- **Pass (latency <100ms, memory bounded)**: IEEE CAL letter + vLLM upstream PR.
- **Below**: vLLM contribution PR only.
- **Outperform**: paper pair with Mosaic dispatcher (streaming scenario class).

## 8. 기준 코드베이스 (Baseline Source)

- vLLM v0.10.x + PyNvVideoCodec ([github.com/NVIDIA/PyNvVideoCodec](https://github.com/NVIDIA/PyNvVideoCodec))
- Models: `Qwen/Qwen3-VL-30B-A3B-Instruct`
- Deps: `transformers==4.57.6`, `torch==2.6.0`, `cuda==12.4`, `pynvvideocodec==2.0+`
- Hardware: RTX Pro 6000 96GB

## 9. Self-Check

R31 (streaming video real-time common case for CCTV/robot/live agent), R52.1-R52.5, R53 일반 단어, R55.2 (Performance + Memory + Energy), R19-α (Lantern 자연어 + NVDEC 은 vendor-specific 이지만 industry-standard 통용 예외), R28 통과.
