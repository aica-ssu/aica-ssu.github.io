# T2.2 Compass — Ego-Motion-Aware Vision Token Compression for Egocentric Video

## 1. Research Questions

- **RQ5.1**: Optical flow zero-mean check (head turn / body pivot 등 ego-motion stable region detection) 으로 EgoSchema 1인칭 video 에서 token reduction 50%+ + accuracy 보존 가능한가?
- **RQ5.2**: Ego-motion vector 를 MRoPE T-axis 보강 신호로 추가 시 LongVU / AdaptToken 대비 ego scenario 에서 우수한 accuracy 달성 가능한가?

## 2. 개요

Egocentric video (EgoSchema 5063 video, 1인칭 시점) 는 head turn / body pivot 시 frame redundancy 매우 큼. LongVU / AdaptToken / PVC 등 long video compression 은 일반 third-person video 에 최적화 → ego-motion-aware compression 미커버. Compass 는 (M1) optical flow zero-mean check (ego-motion stable region detection), (M2) ego-stable region 만 token retention, (M3) ego-motion vector 를 MRoPE T-axis 보강 신호로 추가.

## 3. 기존 연구 한계 / GAP

- **LongVU** ([arXiv:2410.17434](https://arxiv.org/abs/2410.17434)) — spatiotemporal adaptive, **ego-motion 미고려**.
- **AdaptToken** (Microsoft Research) — entropy-based, **ego-motion 미고려**.
- **PVC** (CVPR 2025) — Progressive Visual Token Compression, **ego-motion 미고려**.
- **MM-Ego** (data-side 1인칭 dataset) — **compression 미커버**.

**GAP**: Egocentric video 의 ego-motion-aware compression paper 0편.

## 4. 제안 기법 (3 mechanism)

### 4.1 M1 — Optical Flow Zero-Mean Check

#### 동작 원리 (Mechanism)
- **추가 Scheme**: 각 frame pair 의 optical flow 계산 후 flow vector 의 mean 이 zero-mean 인 region (ego-stable, e.g., 가운데 fixed object) 식별.
- **해결 문제**: Ego-motion 시 frame 전체가 변하는 것처럼 보이지만, ego-stable region 은 의미 정보 보존.
- **동작 원리**:
  1. OpenCV `cv2.calcOpticalFlowFarneback` per-frame
  2. Flow vector 의 spatial mean 계산 → ego-motion direction
  3. Flow - mean = ego-stable residual → magnitude 작은 region = ego-stable
- **차별화**: LongVU/AdaptToken 은 entropy/attention 만, optical flow zero-mean 활용 미시도.

#### 기대 효과 (Gain)
- **Primary** [Quality]: EgoSchema accuracy 보존 (+0.5pt vs LongVU)
- **Secondary** [Performance]: Token reduction 50%+

#### 구현 변경점 (Code Changes)

| File path | Class · Function · Line region | As-is | To-be | 변경 type |
|-----------|-------------------------------|-------|-------|----------|
| (new) `vllm/multimodal/compass_egomotion.py` | `class EgoMotionDetector` | (신규) | **신규 class**: optical flow Farneback wrapper + zero-mean residual analysis. `detect_stable_region(frame_pair) -> mask`. invariant: mask 는 [0,1] tensor. | Add |
| `vllm/multimodal/processing/processor.py` | `class BaseMultiModalProcessor.apply` (line 218-285) | per-frame uniform | **수정 line block**: line 250 직후 ego scenario 분기 — `if scenario == 'egocentric': mask = ego_detector.detect_stable_region(frame_pair); apply_mask(vision_tokens, mask)`. | Modify |

#### GitHub 실존 검증
- OpenCV `cv2.calcOpticalFlowFarneback` 실재 (opencv-python 4.8+). [✅]

#### 검증 시나리오 (Test Plan)
##### Unit test
- **목적**: Ego-motion detection correctness
- **Input**: synthetic 1인칭 head-turn video 10 frame
- **Expected**: ego-stable region mask 60-80% pixel
- **실행 시간**: < 1분

##### Mechanism-isolated test
- **목적**: EgoSchema accuracy
- **Input**: EgoSchema 100 sample
- **Expected**: token reduction 50%+, accuracy ≤ -1.0pt vs full
- **실행 시간**: ~ 4시간

### 4.2 M2 — Ego-Stable Region Token Retention

#### 동작 원리 (Mechanism)
- M1 의 mask 적용 → ego-stable region token retain, motion region token compress.

#### 기대 효과 (Gain)
- **Primary** [Memory]: Vision token -50% on egocentric

#### 구현 변경점 (Code Changes)

| File path | Class · Function · Line region | As-is | To-be | 변경 type |
|-----------|-------------------------------|-------|-------|----------|
| `vllm/multimodal/compass_egomotion.py` | `EgoMotionDetector.apply_mask` (M1 신규) | mask only | **수정**: mask 적용 시 stable region full retention + motion region 50% compression. **invariant**: 50%+ reduction. | Modify |

### 4.3 M3 — Ego-Motion Vector → MRoPE T-axis 보강

#### 동작 원리 (Mechanism)
- **추가 Scheme**: 일반 MRoPE T-axis 는 frame index, Compass 는 ego-motion vector magnitude 추가 신호.
- **동작 원리**:
  1. Ego-motion vector magnitude 계산 (M1 결과)
  2. MRoPE T-axis frequency 에 ego-motion magnitude 를 frequency offset 으로 추가
  3. Same-position frame 도 ego-motion 다르면 다른 RoPE encoding
- **차별화**: 일반 MRoPE 는 spatial position only, Compass 는 ego-motion 보강.

#### 기대 효과 (Gain)
- **Primary** [Quality]: Ego-motion-aware position encoding → temporal localization 정확도 향상

#### 구현 변경점 (Code Changes)

| File path | Class · Function · Line region | As-is | To-be | 변경 type |
|-----------|-------------------------------|-------|-------|----------|
| `vllm/model_executor/layers/rotary_embedding.py` | `class MRotaryEmbedding.forward` (line 285-380) | T-axis frequency from frame index | **수정 line block**: line 320 직후 `if scenario == 'egocentric': ego_offset = compute_ego_offset(frame_meta); freq = freq + ego_offset` 추가. **invariant**: ego_offset 은 small perturbation (< 0.1 of base freq). | Modify |

#### GitHub 실존 검증
- vLLM `MRotaryEmbedding` 실재 ([github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/rotary_embedding.py](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/rotary_embedding.py)). [✅]

#### 검증 시나리오 (Test Plan)
##### Unit test
- **목적**: MRoPE ego-offset application
- **Input**: 8-frame ego-video, ego-motion mock
- **Expected**: ego-offset magnitude < 0.1 base freq
- **실행 시간**: < 30초

##### Mechanism-isolated test
- **목적**: Temporal localization 정확도
- **Input**: EgoSchema temporal QA subset 100
- **Expected**: localization accuracy +1pt vs no ego-offset
- **실행 시간**: ~ 3시간

## 5. 평가 / 실험 플랜

1. **Hardware**: RTX Pro 6000 96GB / RTX 5090 32GB.
2. **Model**: Qwen3-VL-30B-A3B-Instruct.
3. **Benchmark**: **EgoSchema** (5063 video), **EgoBench** (보조).
4. **Tools**: OpenCV, Nsight Compute.
5. **Ablation**: 2³ (M1 ego-detect × M2 retention × M3 MRoPE-offset).
6. **Implementation Steps**: W1-2 ego-motion detector / W3-4 retention + MRoPE / W5-6 EgoSchema eval / W7 DATE letter writing.
7. **Preliminary Metrics**: ego-motion detection accuracy, token reduction ratio, EgoSchema accuracy, temporal localization.

## 6. 예상 효과

| 지표 | Baseline (LongVU) | Compass | 개선 |
|------|-------------------|---------|------|
| [Memory] Vision token (egocentric) | 1.0× | 0.50× | -50% |
| [Quality] EgoSchema accuracy | -2pt vs full | ≤ -1pt | +1pt |
| [Quality] Temporal localization | baseline | +1pt | +1pt |

## 7. Decision Tree

- **Pass (token -50%, accuracy +1pt)**: DATE 2027 / ISLPED 2027.
- **Below**: IEEE ESL letter.
- **Outperform**: MLSys workshop.

## 8. 기준 코드베이스

- vLLM v0.10.x + OpenCV
- Models: `Qwen/Qwen3-VL-30B-A3B-Instruct`
- Deps: `transformers==4.57.6`, `torch==2.6.0`, `opencv-python==4.8+`
- Hardware: RTX Pro 6000

## 9. Self-Check

R31 (egocentric video for AR/VR common case), R52.1-R52.5, R53 일반 단어, R55.2 (Memory + Quality), R19-α (Compass 자연어 + full title vendor-neutral) 모두 통과.
