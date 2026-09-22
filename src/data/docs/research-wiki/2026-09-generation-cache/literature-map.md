# 선행연구 지형 — 2026-09-22 확인

## 먼저 읽을 비교 지도

각 행은 원 논문의 기여 범위를 요약한 것이다. 논문끼리 모델·GPU·step·품질 기준이 다르므로 보고 speedup 숫자를 한 순위표에 올리지 않는다. 원문이 어떤 입력·memory budget에서 검증되었는지는 상세 후보의 baseline 계약으로 다시 고정한다.

| 접근 | 대표 논문 | 이미 해결을 시도한 것 | 이번 학생이 물어볼 수 있는 경계 |
|---|---|---|---|
| step 변화량에 따른 reuse | [TeaCache](https://arxiv.org/abs/2411.19108), [EasyCache](https://arxiv.org/abs/2507.02860) | input/feature 변화에 따른 step 생략 | gate 비용을 포함해 같은 실제 latency에서 비교되는가 |
| cache 값을 예측 | [TaylorSeer](https://arxiv.org/abs/2503.06923) | history 차분으로 미래 feature forecast | history bytes와 차분의 수치 오차가 동시에 제약일 때 |
| history 저장 위치 축소 | [CG-Taylor](https://arxiv.org/abs/2508.02240), [FreqCa](https://arxiv.org/abs/2510.08669) | last-block/cumulative residual로 state 축소 | 이 단순한 축소 이후에도 더 복잡한 state allocator/codec가 필요한가 |
| shallow online probe | [DiCache](https://arxiv.org/abs/2508.17356) | 현재 sample에 맞춘 schedule/trajectory 정렬 | probe와 refresh가 history 신뢰도에 미치는 비용 |
| refresh 정책 최신 경쟁 | [ACID](https://arxiv.org/abs/2607.12358), [GP-Refiner](https://arxiv.org/abs/2609.05981) | adaptive threshold, prediction correction와 uncertainty refresh | 새로운 score 하나만으로 차별화하기 어려움 |
| 고정 compute budget | [BudCache](https://arxiv.org/abs/2606.13496) | offline cache schedule와 time discretization 정렬 | FLOPs/NFE budget과 physical byte/latency budget이 같은가 |
| downstream error | [EpaCache](https://arxiv.org/abs/2608.29264) | trajectory에서 오차 영향이 낮은 step에 reuse 배분 | propagation-aware라는 이름 자체가 신규 기여는 아님 |
| motion/token 단위 | [AdaCache](https://arxiv.org/abs/2411.02397), [ToCa](https://arxiv.org/abs/2410.05317), [DuCa](https://arxiv.org/abs/2412.18911) | motion 배분·token selection·random 대조 | dense/tile 실행에 실제 절약이 있는지와 특정 응용 실패 |
| guidance 중복 | [FasterCache](https://arxiv.org/abs/2410.19355) | conditional/unconditional 차이를 이용한 reuse | common/difference 표현 자체가 새롭다고 주장하지 않음 |
| 캐시와 양자화 결합 | [CacheQuant](https://arxiv.org/abs/2503.01323) | model quantization과 cache 오차를 함께 최적화 | model W/A quantization과 stored history codec의 차이를 증명해야 함 |
| 실제 memory 관리 | [LightCache](https://arxiv.org/abs/2510.05367), [Xema](https://arxiv.org/abs/2607.11136) | phase별 swapping/chunk/decode, tensor lifetime과 자원 계획 | free/reset 또는 기존 offload만으로 해결되지 않는 문제가 남는가 |
| 요청 간 reuse/서빙 | [Chorus](https://arxiv.org/abs/2604.04451), [FlexCache](https://arxiv.org/abs/2501.04012), [PixelFlow](https://arxiv.org/abs/2609.20723) | inter-request reuse, cache 저장·교체, token workload 관리 | 단일 GPU에서 실제 batching 수요와 fit이 먼저 성립하는가 |
| few-step AR 생성 | [X-Cache](https://arxiv.org/abs/2604.20289), [DisCa](https://arxiv.org/abs/2602.05449) | cross-chunk reuse 또는 distilled model에 맞춘 predictor | 모델 구조와 시간축을 바꾸는 것만으로 novelty를 주장할 수 없음 |
| 평가 | [VBench](https://arxiv.org/abs/2311.17982) | 여러 video quality 차원을 분리 | 평균 fidelity와 국소·tail 실패가 다를 때 응용 metric을 어떻게 고정할까 |

## 문헌을 읽으면서 작성할 카드

- 입력/출력: image 또는 video, offline 또는 autoregressive, conditioning 종류.
- 재사용 객체: block residual, module output, final denoiser output, K/V, inter-request latent 중 무엇인가.
- 상태: 저장 tensor shape/dtype/history 길이/수명, CPU·GPU 위치.
- 결정: 언제 reuse하며 언제 full compute하는가. 미래 정답을 보는 offline oracle과 online controller를 구분한다.
- 평가: no-cache 대비와 closest-cache 대비, denoiser-only와 end-to-end, cache bytes와 GPU peak, fidelity와 생성 품질.
- 반례: 이 논문이 직접 다룬 실패와 검증하지 않은 조건을 각각 한 줄로 쓴다.

## 검증한 참고문헌 목록

- 아래 verified는 **실존과 해당 source 내용 확인** 상태다. 기법 재현이나 우리 후보 신규성 승인이라는 뜻이 아니다.
- venue는 확인한 metadata를 유지한다. preprint를 임의로 학회 accepted로 승격하지 않는다.

| 논문 | 게시/게재 상태 | 실존 확인 | 역할 |
|---|---|---|---|
| [PixelFlow: Token-Level Workload Management for Efficient Distributed DiT Serving](https://arxiv.org/abs/2609.20723) | arXiv | 2026-09-22 / True | 경쟁·근거·평가 자료, 사용 범위는 후보별 문서 참조 |
| [Accelerating Diffusion Transformers with Gaussian Process Rectified Feature Cache](https://arxiv.org/abs/2609.05981) | accepted at ECCV 2026 (arXiv author metadata) | 2026-09-22 / True | 경쟁·근거·평가 자료, 사용 범위는 후보별 문서 참조 |
| [EpaCache: Error-Propagation-Aware Caching for Accelerating Diffusion-Based Visual Generation](https://arxiv.org/abs/2608.29264) | arXiv preprint; under review | 2026-09-22 / True | 경쟁·근거·평가 자료, 사용 범위는 후보별 문서 참조 |
| [BAG: Budget-Aware Gating for Diffusion Caching](https://arxiv.org/abs/2608.09231) | arXiv | 2026-09-22 / True | 경쟁·근거·평가 자료, 사용 범위는 후보별 문서 참조 |
| [OnlineCache: Learning Dynamic Caching Policies with Error Correction for Efficient Diffusion Inference](https://arxiv.org/abs/2607.29398) | arXiv | 2026-09-22 / True | 경쟁·근거·평가 자료, 사용 범위는 후보별 문서 참조 |
| [ACID: Adaptive Caching for vIDeo generation](https://arxiv.org/abs/2607.12358) | arXiv | 2026-09-22 / True | 경쟁·근거·평가 자료, 사용 범위는 후보별 문서 참조 |
| [Xema: Efficient Diffusion Serving through Fine-Grained Memory Management and Auto-Configuration](https://arxiv.org/abs/2607.11136) | arXiv | 2026-09-22 / True | 경쟁·근거·평가 자료, 사용 범위는 후보별 문서 참조 |
| [Budget-Constrained Step-Level Diffusion Caching](https://arxiv.org/abs/2606.13496) | ICML 2026 accepted | 2026-09-22 / True | 경쟁·근거·평가 자료, 사용 범위는 후보별 문서 참조 |
| [ReCache: Learning Budget-Aware Caching Schedules for Diffusion Models via REINFORCE](https://arxiv.org/abs/2606.06060) | arXiv | 2026-09-22 / True | 경쟁·근거·평가 자료, 사용 범위는 후보별 문서 참조 |
| [Motion-Aware Caching for Efficient Autoregressive Video Generation](https://arxiv.org/abs/2605.01725) | arXiv | 2026-09-22 / True | 경쟁·근거·평가 자료, 사용 범위는 후보별 문서 참조 |
| [X-Cache: Cross-Chunk Block Caching for Few-Step Autoregressive World Models Inference](https://arxiv.org/abs/2604.20289) | arXiv technical report | 2026-09-22 / True | 경쟁·근거·평가 자료, 사용 범위는 후보별 문서 참조 |
| [Beyond Few-Step Inference: Accelerating Video Diffusion Transformer Model Serving with Inter-Request Caching Reuse](https://arxiv.org/abs/2604.04451) | arXiv | 2026-09-22 / True | 경쟁·근거·평가 자료, 사용 범위는 후보별 문서 참조 |
| [SenCache: Accelerating Diffusion Model Inference via Sensitivity-Aware Caching](https://arxiv.org/abs/2602.24208) | CVPR 2026 | 2026-09-22 / True | 경쟁·근거·평가 자료, 사용 범위는 후보별 문서 참조 |
| [SeaCache: Spectral-Evolution-Aware Cache for Accelerating Diffusion Models](https://arxiv.org/abs/2602.18993) | CVPR 2026 | 2026-09-22 / True | 경쟁·근거·평가 자료, 사용 범위는 후보별 문서 참조 |
| [DisCa: Accelerating Video Diffusion Transformers with Distillation-Compatible Learnable Feature Caching](https://arxiv.org/abs/2602.05449) | CVPR 2026 | 2026-09-22 / True | 경쟁·근거·평가 자료, 사용 범위는 후보별 문서 참조 |
| [Forecast the Principal, Stabilize the Residual: Subspace-Aware Feature Caching for Efficient Diffusion Transformers](https://arxiv.org/abs/2601.07396) | CVPR 2026 | 2026-09-22 / True | 경쟁·근거·평가 자료, 사용 범위는 후보별 문서 참조 |
| [Tortoise and Hare Guidance: Accelerating Diffusion Model Inference with Multirate Integration](https://arxiv.org/abs/2511.04117) | NeurIPS 2025 | 2026-09-22 / True | 경쟁·근거·평가 자료, 사용 범위는 후보별 문서 참조 |
| [FreqCa: Accelerating Diffusion Models via Frequency-Aware Caching](https://arxiv.org/abs/2510.08669) | arXiv | 2026-09-22 / True | 경쟁·근거·평가 자료, 사용 범위는 후보별 문서 참조 |
| [LightCache: Memory-Efficient, Training-Free Acceleration for Video Generation](https://arxiv.org/abs/2510.05367) | arXiv | 2026-09-22 / True | 경쟁·근거·평가 자료, 사용 범위는 후보별 문서 참조 |
| [DiCache: Let Diffusion Model Determine Its Own Cache](https://arxiv.org/abs/2508.17356) | ICLR 2026 | 2026-09-22 / True | 경쟁·근거·평가 자료, 사용 범위는 후보별 문서 참조 |
| [Forecasting When to Forecast: Accelerating Diffusion Models with Confidence-Gated Taylor](https://arxiv.org/abs/2508.02240) | arXiv; publication status see source audit | 2026-09-22 / True | 경쟁·근거·평가 자료, 사용 범위는 후보별 문서 참조 |
| [Less is Enough: Training-Free Video Diffusion Acceleration via Runtime-Adaptive Caching](https://arxiv.org/abs/2507.02860) | arXiv | 2026-09-22 / True | 경쟁·근거·평가 자료, 사용 범위는 후보별 문서 참조 |
| [MagCache: Fast Video Generation with Magnitude-Aware Cache](https://arxiv.org/abs/2506.09045) | NeurIPS 2025 | 2026-09-22 / True | 경쟁·근거·평가 자료, 사용 범위는 후보별 문서 참조 |
| [MoDM: Efficient Serving for Image Generation via Mixture-of-Diffusion Models](https://arxiv.org/abs/2503.11972) | ASPLOS 2026 | 2026-09-22 / True | 경쟁·근거·평가 자료, 사용 범위는 후보별 문서 참조 |
| [From Reusing to Forecasting: Accelerating Diffusion Models with TaylorSeers](https://arxiv.org/abs/2503.06923) | ICCV 2025 | 2026-09-22 / True | 경쟁·근거·평가 자료, 사용 범위는 후보별 문서 참조 |
| [CacheQuant: Comprehensively Accelerated Diffusion Models](https://arxiv.org/abs/2503.01323) | CVPR 2025 | 2026-09-22 / True | 경쟁·근거·평가 자료, 사용 범위는 후보별 문서 참조 |
| [FlexCache: Flexible Approximate Cache System for Video Diffusion](https://arxiv.org/abs/2501.04012) | arXiv | 2026-09-22 / True | 경쟁·근거·평가 자료, 사용 범위는 후보별 문서 참조 |
| [Rethinking Token-wise Feature Caching: Accelerating Diffusion Transformers with Dual Feature Caching](https://arxiv.org/abs/2412.18911) | IEEE TIP 2026 | 2026-09-22 / True | 경쟁·근거·평가 자료, 사용 범위는 후보별 문서 참조 |
| [Timestep Embedding Tells: It's Time to Cache for Video Diffusion Model](https://arxiv.org/abs/2411.19108) | CVPR 2025 | 2026-09-22 / True | 경쟁·근거·평가 자료, 사용 범위는 후보별 문서 참조 |
| [Adaptive Caching for Faster Video Generation with Diffusion Transformers](https://arxiv.org/abs/2411.02397) | ICCV 2025 | 2026-09-22 / True | 경쟁·근거·평가 자료, 사용 범위는 후보별 문서 참조 |
| [FasterCache: Training-Free Video Diffusion Model Acceleration with High Quality](https://arxiv.org/abs/2410.19355) | ICLR 2025 | 2026-09-22 / True | 경쟁·근거·평가 자료, 사용 범위는 후보별 문서 참조 |
| [Accelerating Diffusion Transformers with Token-wise Feature Caching](https://arxiv.org/abs/2410.05317) | ICLR 2025 | 2026-09-22 / True | 경쟁·근거·평가 자료, 사용 범위는 후보별 문서 참조 |
| [Real-Time Video Generation with Pyramid Attention Broadcast](https://arxiv.org/abs/2408.12588) | ICLR 2025 | 2026-09-22 / True | 경쟁·근거·평가 자료, 사용 범위는 후보별 문서 참조 |
| [PipeFusion: Patch-level Pipeline Parallelism for Diffusion Transformers Inference](https://arxiv.org/abs/2405.14430) | arXiv | 2026-09-22 / True | 경쟁·근거·평가 자료, 사용 범위는 후보별 문서 참조 |
| [DistriFusion: Distributed Parallel Inference for High-Resolution Diffusion Models](https://arxiv.org/abs/2402.19481) | CVPR 2024 | 2026-09-22 / True | 경쟁·근거·평가 자료, 사용 범위는 후보별 문서 참조 |
| [DeepCache: Accelerating Diffusion Models for Free](https://arxiv.org/abs/2312.00858) | CVPR 2024 | 2026-09-22 / True | 경쟁·근거·평가 자료, 사용 범위는 후보별 문서 참조 |
| [VBench: Comprehensive Benchmark Suite for Video Generative Models](https://arxiv.org/abs/2311.17982) | CVPR 2024 | 2026-09-22 / True | 경쟁·근거·평가 자료, 사용 범위는 후보별 문서 참조 |

GATE literature-integrity · ran · P0/P2 exists logs + references-supplemental.md; 실존 검증과 실험 재현 구분 · 2026-09-22

## 최종 재검색에서 추가된 가까운 문헌

- [CachedSearch](https://arxiv.org/abs/2607.23159): cache가 test-time search의 candidate ranking을 보존하는지 평가한다. A2의 policy-ranking 질문과 비교해야 하며, ranking corruption이라는 broad 문제 자체를 새로 발견했다고 쓰지 않는다.
- [HiCache](https://arxiv.org/abs/2508.16984): ground-truth history로 초기화한 prediction simulation을 A4의 online history 개입과 구분하는 데 중요하다.
- 두 논문의 실존과 해당 PDF 절은 최종 재검색에서 확인했다. 이 목록을 포함한 candidate bibliography는 39편이고 기초 개념 문헌은 별도다.
