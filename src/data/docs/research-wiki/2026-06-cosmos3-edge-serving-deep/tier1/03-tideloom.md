# S1 — TIDELOOM (Tier-1 #3, 직전 avg 7.4)

**TIDELOOM: A Single-Device Unified Serving Runtime for Mixture-of-Transformers Omnimodal Models with Phase-Deterministic Tower Weight Residency**

- **Venue 목표**: MLSys / ASPLOS 2027 (신규 단일-device 런타임 시스템)
- **Metaphor↔Mechanism**: 조수(**TIDE**)처럼 결정적으로 오가는 AR/DM phase 에 맞춰, 베틀(**LOOM**)이 reasoner/generator 두 tower 의 weight 를 단일 device 메모리에 짜 넣었다(weave) 풀었다(unweave) 한다. 활성 tower 가 조수처럼 교대하고, 그 흐름을 한 베틀(단일 process·단일 forward)이 짜낸다.
- **계보**: A1(Loom, ai-opt: runtime FSM / 단일-process 통합) + L1(TIDELOCK, legacy: phase-결정론 weight residency / Orin UMA 물리제약 정확 대응 / denoise-slack staging) **merge** → S1. A3(Cascade, chunk-pipeline)는 M3 로 흡수.
- **소유 GAP**: **G1**(dual-regime 비대칭) + **G2**(2× 파라미터, 항상 1-tower 활성).
- **본 문서의 핵심 정정 (TRAP #3)**: cosmos3 generator 는 vLLM 의 AR scheduler/worker 스택을 **쓰지 않는다**. 두 phase(UND→GEN)는 **단일 diffusion forward 안에 in-process 로 순차 존재**한다. 따라서 TIDELOOM 의 FSM/hook 은 scheduler-level 이 아니라 **model-forward-level** 이다. 직전(2026-06-04) 버전 일부 서술이 "vLLM(AR)+vLLM-Omni(DM) 분리 엔진을 한 process 로 통합"이라 한 것은 *오해 소지*가 있었고(두 엔진을 합치는 것이 아니라, 이미 한 forward 안에 있는 두 phase 를 명시적 FSM 으로 구조화·관리하는 것), 본 문서에서 코드 수준으로 정정한다.

---

## 0. 약어 Glossary

| 약어 | 풀이 | 본 문서 맥락 |
|---|---|---|
| **MoT** | Mixture-of-Transformers | 한 모델 안에 modality/기능별로 분리된 dual-tower (reasoner=UND, generator=GEN). cosmos3 는 weight 가 `language_model.*`(UND) vs `gen_layers.*`(GEN) prefix 로 물리 분리. |
| **AR phase** | Autoregressive (understanding) phase | reasoner(UND) tower 가 text/조건 token 을 1회 forward 하여 K/V 를 만드는 단계. cosmos3 에선 `language_model(...)` 1회 호출. |
| **DM phase** | Diffusion (denoise) phase | generator(GEN) tower 가 noisy latent 를 N step 반복 denoise 하는 단계. `gen_layers` 가 cached UND K/V 를 cross-attn 으로 소비. |
| **FSM** | Finite State Machine | phase enum {VLM_ONLY, GEN_PREFILL(UND), GEN_DENOISE, POLICY_CHUNK} 간 전이를 관리하는 상태기계. |
| **UMA** | Unified Memory Architecture | Jetson SoC 의 CPU/GPU 공유 LPDDR. host↔device 가 물리적으로 같은 DRAM → "staging"이 실제론 같은 칩 내 복사. |
| **cMA** | `concurrentManagedAccess` (CUDA device attribute) | =1 이면 UVM demand-paging / `cudaMemPrefetchAsync` 가용. **Orin NX/AGX Orin = 0** → UVM prefetch 불가. **Thor = 1** → 정공법 가용. |
| **pinned memory** | page-locked host memory (`cudaHostAlloc`) | non-blocking H2D `cudaMemcpyAsync` 가 copy-engine 으로 async overlap 되려면 host buffer 가 pinned 여야 함. |
| **double buffering** | 2개 버퍼 교대 (compute on A while loading B) | denoise step n 계산 중 background stream 이 다음 phase tower weight 를 buffer B 로 적재 → step n+1 시작 시 swap. |
| **CUDA stream / event** | 비동기 작업 큐 / 동기화 마커 | compute stream(denoise) ∥ copy stream(H2D staging). `cudaEvent` 로 "staging 완료" 보장 후 swap. |
| **denoise slack** | DM phase 의 다단계 denoise 가 만드는 시간 여유 | policy 는 chunk 당 4 step × CFG2 = 8 forward (~2.1s). 이 window 동안 GEN tower 만 활성 → UND tower 잔류 불필요·다음 phase weight staging 기회. |
| **chunked policy serving** | robot action 을 chunk(예: 32-action) 단위로 생성·실행 | 15Hz 주기, chunk 당 한 번 AR re-prefill(새 관측) + DM denoise. chunk t 의 DM ∥ chunk t+1 의 AR overlap 여지. |
| **MPS** | Multi-Process Service (CUDA) | 여러 process/context 가 SM 을 공간 분할 공유. Orin JetPack 6.1+ verified. M3 의 AR∥DM 동시 dispatch fallback 경로. |
| **Green Context** | `cuGreenCtxCreate` SM-partition API | MIG식 isolation 아닌 SW 분배. Thor 헤드라인 기능, Orin 미검증 → Thor-scope 비교군. |
| **K_AR / K_DM** | UND tower 가 만든 K/V (조건) / GEN tower 자신의 self-K/V | cross-attn 은 `cat([k_und, k_gen])` 으로 두 K 를 결합 (코드 L648-649 계열). K_AR 은 denoise 전기간 read-only 불변. |

---

## 1. 개요 — RQ, metaphor, 소유 GAP

### 1.1 Research Question
> **RQ1**: 16B MoT omnimodal model 을, phase 별 활성 tower(active-8B)만 단일 edge GPU 메모리에 상주시키면서 **품질 0 손실(weight 무손실, swap 은 bit-exact)**로 서빙할 수 있는가?
>
> **RQ2**: phase 의 **결정성(determinism)**이 demand-paging 대비 *무엇을 사주는가* — 즉, "예측기 불필요 + miss penalty 0 + staging 을 compute slack 에 완전 hide" 가 실제 edge UMA 에서 성립하는가?

RQ2 가 본 idea 의 systems thesis 다. 기존 expert/neuron offload(PowerInfer-2, FluxMoE)는 **token-routing 또는 activation-predictor** 에 의존하므로 본질적으로 비결정 → miss 시 stall. TIDELOOM 의 주장은: MoT 의 AR→DM phase 는 **단일 forward 의 제어흐름으로 100% 결정**되어 있어, "다음에 무슨 tower 가 필요한가"를 **컴파일 타임에 안다** → predictor·miss penalty 자체가 사라진다는 것.

### 1.2 metaphor → mechanism 사상
- **TIDE (조수)** = phase 가 결정적으로 들고 남. AR phase 엔 UND tower 가 "밀물"(상주), DM 에선 GEN tower 가 "밀물", 비활성 tower 는 "썰물"(demote). → **M2 Phase-Deterministic Weight Residency**.
- **LOOM (베틀)** = 두 실(reasoner/generator)을 한 베틀(단일 forward·단일 process)에서 직조. phase 경계를 명시적 상태로 노출. → **M1 Tower-Phase FSM**.
- **조수의 교대 주기(chunk)** = policy 의 chunk 주기 동안 한 chunk 의 DM 과 다음 chunk 의 AR 을 겹쳐 짠다. → **M3 AR∥DM Chunk Overlap**.

### 1.3 소유 GAP (G1 + G2)
- **G1 (dual-regime 비대칭)**: 한 요청이 AR(memory-bound, KV-중심) → DM(compute-bound, KV-불변, step 반복)을 단일 edge GPU 에서 직렬 traverse 한다. 두 regime 은 자원 프로파일이 정반대지만 현재는 한 forward 안에서 직렬 실행될 뿐, 둘의 비대칭을 활용하는 메모리/스케줄 구조가 없다.
- **G2 (2× 파라미터, 1-tower 활성)**: 16B 총량 중 phase 별 활성은 **항상 8B 한 tower**다. 그런데 현 서빙(`Cosmos3VFMTransformer`)은 UND+GEN 두 tower weight 를 **상시 전량 device 상주**시킨다. 결과적으로 RTX Pro 6000 ×2 가 필요(직전 분석)하고 Orin NX 16GB 엔 BF16 16B(32GB) 가 애초에 안 올라간다.

---

## 2. Mechanism Summary Table (R70 7열)

| # | 메커니즘 | 해결 GAP | 핵심 아이디어 (1줄) | 코드 anchor (재확인) | 신규 객체 | 차별 baseline |
|---|---|---|---|---|---|---|
| **M1** | Tower-Phase FSM (model-forward-level) | G1 | 단일 diffusion forward 안의 UND→GEN phase 경계를 명시적 FSM state 로 노출, phase 별 active-module 집합을 LUT 로 관리 | `Cosmos3VFMTransformer.forward` L1459-1530 (UND-once cache 경계 ~L1480) → [vllm_omni/diffusion/models/cosmos3/transformer_cosmos3.py#L1459-L1530](https://github.com/vllm-project/vllm-omni/blob/main/vllm_omni/diffusion/models/cosmos3/transformer_cosmos3.py#L1459-L1530) (`@95d56cf`) | `TowerPhaseFSM` + phase enum + active-module LUT | vLLM-Omni multi-GPU disaggregation (단일 device 무의미) |
| **M2** | Phase-Deterministic Weight Residency Manager | G2 | 비활성 tower demote + 다음 phase tower 를 denoise slack 동안 double-buffered `cudaMemcpyAsync`(pinned) 로 staging; eviction 은 phase-결정적(predictor 불요) | tower prefix 분리 `language_model.*`/`gen_layers.*` (`_remap_ckpt_key` L530-550) → [vllm_omni/diffusion/models/cosmos3/pipeline_cosmos3.py#L530-L550](https://github.com/vllm-project/vllm-omni/blob/main/vllm_omni/diffusion/models/cosmos3/pipeline_cosmos3.py#L530-L550) (`@95d56cf`); `LayerwiseOffloadHook`(대비군) → [vllm_omni/diffusion/offloader/layerwise_backend.py](https://github.com/vllm-project/vllm-omni/blob/main/vllm_omni/diffusion/offloader/layerwise_backend.py) | `WeightResidencyManager.swap_tower()` | PowerInfer-2 activation-predictor / `LayerwiseOffloadHook` demand-driven per-block prefetch |
| **M3** | AR∥DM Chunk Overlap (policy mode) | G1/G5 | chunk t 의 DM denoise(GEN) ∥ chunk t+1 의 AR re-prefill(UND) 를 2-stream/MPS 로 병행; 간섭 예산화 | `diffuse()` N-step 루프 `for t in timesteps` L1492-1530 (slack 구간) → [vllm_omni/diffusion/models/cosmos3/pipeline_cosmos3.py#L1486-L1530](https://github.com/vllm-project/vllm-omni/blob/main/vllm_omni/diffusion/models/cosmos3/pipeline_cosmos3.py#L1486-L1530) (`@95d56cf`); `reset_cache()` L1281 → [transformer_cosmos3.py#L1281](https://github.com/vllm-project/vllm-omni/blob/main/vllm_omni/diffusion/models/cosmos3/transformer_cosmos3.py#L1281) | `PolicyPipelineScheduler` (2-stream orchestration) | Bullet/Nexus/DuetServe same-tower PD-multiplex; VLA-XPU V-AEFusion |

> R70 7열 = {번호, 메커니즘, 해결 GAP, 핵심, 코드 anchor, 신규 객체, 차별 baseline}.

---

## 3. GAP 심화 + Workload Evidence + Baseline Source (R52.1)

### 3.1 Workload evidence (numbers + clickable)
- **decode-지배성**: AR decode = 전체 inference time 의 **77-91%** ([arXiv:2501.08219](https://arxiv.org/abs/2501.08219)); AGX Orin 에서 **decode 가 prefill 대비 192-569× latency 지배** (EdgeReasoning IISWC'25, [arXiv:2511.01866](https://arxiv.org/abs/2511.01866)). AR action phase = e2e latency 의 최대 75%, memory-bound ([arXiv:2603.02271](https://arxiv.org/abs/2603.02271)).
- **policy denoise window (slack 의 출처)**: policy = chunk 당 **4 step × CFG2 = 8 forward @ 15Hz (~2.1s 주기)** (직전 techreport §4.2.5). GEN tower 가 이 long window 를 연속 점유 → UND tower weight idle = staging hiding 기회. (코드상 `diffuse()` L1492 `for t in timesteps` 가 step 루프, CFG 면 한 step 당 cond/uncond 2회 transformer 호출 L1497/L1511.)
- **PowerInfer-2 coarse FFN offload**: 7B 메모리 **40%↓**, 47B phone 11.68 tok/s ([arXiv:2406.06282](https://arxiv.org/abs/2406.06282)) — coarse offload 의 edge 실효는 입증됨(단 비결정 predictor).
- **co-location 간섭 예산 (M3 용)**: TBT **1.55×**, L2 pollution **2.15×**, bank conflict 3.75× ([arXiv:2501.16909](https://arxiv.org/abs/2501.16909)); Jetson 8-process throughput 95% 붕괴 ([arXiv:2508.08430](https://arxiv.org/abs/2508.08430)).
- **chunk 연속성**: RTC(real-time chunking) 식 chunk handoff ([arXiv:2506.07339](https://arxiv.org/abs/2506.07339)) 와 절대-시간 MRoPE 결합 → 새 chunk 위치 재계산 0.

### 3.2 메모리 수지표 (Cosmos3-Nano 16B, dual-tower 8B each)

| device | DRAM | BW (LPDDR5/x) | cMA | 16B BF16=32GB | 16B INT8=16GB | active-8B(INT8)=8GB | active-8B(INT4)=4GB | 판정 |
|---|---|---|---|---|---|---|---|---|
| **Orin NX 16GB** | 16GB | 102.4 GB/s | 0 | ❌ 불가 | ⚠ 빠듯 (KV/VAE/활성화 여유 無) | ✅ +KV+obs fit | ✅ 여유 | active-8B **必**, S1-mini 타깃 |
| **AGX Orin 64GB** (primary) | 64GB | 204.8 GB/s | 0 | ✅ 가능(but KV/VAE/활성화 여유 필요) | ✅ 편안 | ✅ | ✅ | **constrained primary**, double-buffer 실증 |
| **Thor 128GB** | 128GB | 273 GB/s | **1** | ✅ 정공법 | ✅ | ✅ | ✅ (FP4 native) | **forward-looking**, prefetch 대조 |
| **RTX Pro 6000 96GB** | 96GB | ~1.8 TB/s | 1 | ✅ | ✅ | ✅ | ✅ | ncu control + ×2→×1 단일화 대조 |

> 핵심 수지: AGX Orin 은 16B BF16(32GB) 가 64GB 안에 *수치상* 올라가지만, video VAE buffer + KV + 활성화가 동시 경합하면 peak 가 위험. active-8B residency 로 **상시 footprint 를 절반(~16GB→8-9GB+버퍼)으로** 낮추는 것이 G2 의 직접 해결. Orin NX 16GB 는 16B 전량이 **물리적으로 불가** → active-8B 가 enabling 의 *전제*다.
>
> ⚠ **mode-별 active-tower 수 정정**: active-8B(1-tower) 절감은 **generation(T2I/T2V)·VLM 단독 모드**에 성립. **policy 모드는 M3(AR∥DM overlap) 활성 시 reasoner+generator 양 tower 상주가 必**(chunk t+1 AR re-prefill 이 reasoner GPU 상주 요구 → demote 불가) → policy+M3 의 active footprint 는 **2-tower(예: NX 에서 2×INT4=8GB+KV+obs)**. NX policy fit 은 따라서 **저비트(INT4) 2-tower** 또는 **M3 비활성(sequential, 1-tower demote)** 둘 중 하나로만 성립 — S1-mini(M2 단독, M3 없음)는 1-tower demote 경로.

### 3.3 staging BW 수지 (M2 의 falsification 근거)
- active-8B(INT8)=8GB 를 한 chunk window(~2.1s) 안에 staging 해야 한다면: 8GB / 2.1s ≈ **3.8 GB/s**.
- 이는 LPDDR5 BW 의 비율로: Orin NX 102.4 GB/s 의 ~3.7%, AGX Orin 204.8 GB/s 의 **~1.9%**, Thor 273 GB/s 의 ~1.4%.
- 단순 transfer 시간: 8GB @ AGX 204.8GB/s = **~39ms**(이론), INT8 8GB @ NX 102.4GB/s = ~78ms. → **2.1s window 대비 압도적 여유**.
- ⚠ **진짜 gate 는 transfer 시간이 아니라 "경합 하 hiding%"**다: denoise compute 가 copy-engine 과 같은 DRAM 을 공유(UMA)하므로, denoise 의 memory traffic 이 staging copy 의 effective BW 를 깎는다. 따라서 falsification gate 는 "**경합 하 staging-hiding% > 70%**"로 정량화한다(직전 phase2' decision-tree 계승).

### 3.4 Baseline Source (R52.1) — 실행 가능한 비교군

| # | baseline | 종류 | 실행/구현 anchor | 무엇을 측정하나 |
|---|---|---|---|---|
| **(a)** | vllm-omni 기본 (전체 16B 상주) | 코드 baseline | `DiffusionModelRunner.load_model`, offload 미설정 (`enable_layerwise_offload=False`, `enable_cpu_offload=False`); `load_device = str(self.device)` (L124-125) → [vllm_omni/diffusion/worker/diffusion_model_runner.py#L124-L125](https://github.com/vllm-project/vllm-omni/blob/main/vllm_omni/diffusion/worker/diffusion_model_runner.py#L124-L125) (`@95d56cf`) | peak resident bytes 상한 (16B 전량) |
| **(b)** | vllm-omni layerwise CPU offload | **코드 baseline (정면 대비)** | `--enable-layerwise-offload` → `LayerWiseOffloadBackend.enable()` (layerwise_backend.py L287); per-block `pre_forward`(`prefetch_layer` L242)/`post_forward`(`offload_layer` L247) → [vllm_omni/diffusion/offloader/layerwise_backend.py#L242-L287](https://github.com/vllm-project/vllm-omni/blob/main/vllm_omni/diffusion/offloader/layerwise_backend.py#L242-L287) (`@95d56cf`) | demand-driven per-block prefetch 의 swap 오버헤드·hiding률 |
| **(c)** | vLLM-Omni 클러스터 분리(disaggregation) | 개념 baseline | `total = ulysses×ring×cfg×tp` 본질 multi-GPU ([arXiv:2602.02204](https://arxiv.org/abs/2602.02204)) → single-GPU 강제 축소 | 단일 device 로 강제했을 때의 부적합성 |
| **(d)** | FluxMoE demand-paging | 개념 baseline | expert-residency decoupling, token-routing 비결정 ([arXiv:2604.02715](https://arxiv.org/abs/2604.02715)) | 비결정 paging miss penalty vs 결정론 |
| **(e)** | MotuBrain (serving 없음) | **차별 anchor** | edge MoT 관련이나 serving runtime 부재 ([arXiv:2604.27792](https://arxiv.org/abs/2604.27792)) | "serving 미해결"임을 보이는 anchor (이 위 추가이득) |

#### baseline (b) 의 정확한 차이 (코드 수준, 본 idea 의 핵심 대비)
`LayerwiseOffloadHook` (실제 코드 재확인) 의 동작:
- 각 transformer **block** 에 hook 을 단다. `pre_forward` 가 자기 다음 block 을 `prefetch_layer(non_blocking=True)` 로 H2D 복사하고(L242), `post_forward` 가 자기 자신을 `offload_layer()` 로 GPU 에서 free(L247).
- 즉 **(그들) layer/block 단위, demand-driven, sliding-window** prefetch — "현재 실행 중인 block 이 다음 block 을 미리보기" 식.
- **(우리 M2) phase 단위, deterministic, tower 전체 잔류 결정**: TIDELOOM 은 block 이 아니라 **tower 전체**(UND 8B / GEN 8B)를 residency 단위로 보고, **phase 가 결정적이므로** "DM phase 가 시작되면 UND 는 통째로 demote, denoise-slack 동안 다음 phase(예: 다음 chunk 의 UND) tower 를 통째로 staging" 한다.
- 결정적 차이 3가지: ① 단위(block↔tower), ② 트리거(현재 block 실행↔phase 전이 LUT), ③ 잔류 결정(sliding window 항상 N-block↔phase 가 끝날 때까지 active tower 통째로 잔류 + slack 에 다음 tower 통째 staging). `LayerwiseOffloadHook` 는 high-level `tensor.copy_(non_blocking)` on `current_omni_platform.Stream()` 을 쓰며 `cudaMemcpyAsync`/pinned-double-buffer 의 명시적 stream/event 오케스트레이션은 우리가 추가.

> ✅ source verified: `vllm-omni@95d56cf` [vllm_omni/diffusion/offloader/layerwise_backend.py#L235-L249](https://github.com/vllm-project/vllm-omni/blob/main/vllm_omni/diffusion/offloader/layerwise_backend.py#L235-L249) — `pre_forward` 가 `self.prefetch_layer(non_blocking=True)` (다음 block H2D), `post_forward` 가 `self.offload_layer()` (자기 block free). pinned host = `torch.empty(..., pin_memory=pin_memory)` ([L144](https://github.com/vllm-project/vllm-omni/blob/main/vllm_omni/diffusion/offloader/layerwise_backend.py#L144)). copy stream = `current_omni_platform.Stream()` ([L284](https://github.com/vllm-project/vllm-omni/blob/main/vllm_omni/diffusion/offloader/layerwise_backend.py#L284)). **per-block sliding window 임을 코드로 확인.**

---

## 4. 제안 기법 (3 mechanism)

전제(model-forward-level, TRAP #3): 모든 hook 은 `Cosmos3VFMTransformer.forward` 내부 또는 `Cosmos3OmniDiffusersPipeline.diffuse` N-step 루프에 삽입된다. scheduler/worker 콜백이 아니다.

### M1. Tower-Phase FSM (model-forward-level unified runtime)

- **① scheme**: `Cosmos3VFMTransformer.forward` 의 UND→GEN 경계(~L1480)에 **phase-transition hook** 을 걸고, phase enum `{VLM_ONLY, GEN_PREFILL(UND), GEN_DENOISE, POLICY_CHUNK}` 와 각 phase 의 **active-module 집합 LUT** 를 노출하는 `TowerPhaseFSM` 을 둔다. plain-PyTorch reference(`Cosmos3OmniDiffusersPipeline`) 또는 vLLM-Omni 단일 확장 위에 구현 → 단일 process 자동 확보(CF-E: two-stack fork 금지).
- **② 해결 문제**: G1 — 현재 두 regime 의 phase 경계가 *암묵적*(if `cached_kv is None`)이라 residency/스케줄/DVFS 가 hook 할 지점이 없다. FSM 이 경계를 1급 객체로 만든다.
- **③ 동작 원리 (≥6 step)**:
  1. 요청 진입 시 generation-mode(VLM/T2I/T2V/policy)로 초기 phase 결정. policy 면 `POLICY_CHUNK`, VLM-only 면 `VLM_ONLY`(GEN 영구 비활성).
  2. `GEN_PREFILL(UND)`: `forward` 의 `if self.cached_kv is None:` 블록(L1460)이 `self.language_model(text_ids, freqs_und)` 1회 실행(L1474) → K_AR/V_AR 생성, padding trim(L1479). FSM 은 이 진입/완료를 state 전이로 기록.
  3. phase-transition trigger = `cached_kv` 채워짐(L1479 직후, ~L1480) → FSM `GEN_PREFILL → GEN_DENOISE`. 이 경계가 M2 의 demote/staging 트리거이자 M3 의 overlap 트리거.
  4. K_AR/V_AR block 을 **read-only pin** (zero-copy 핸드오프, 동일 device·동일 cached_kv list — 코드상 이미 `self.cached_kv` 로 step 간 보존되므로 추가 복사 0).
  5. `GEN_DENOISE`: `gen_layers` 가 매 step `cached_kv[layer_idx]` 를 cross-attn 으로 소비(L1499-1520). FSM 은 step 카운터를 들고 "slack 잔여"를 M2 에 보고.
  6. denoise 종료 → `DECODE_OUT`(VAE) 또는(policy) skip. policy 면 다음 chunk 로 `POLICY_CHUNK` 재진입 → step 2 로 loop. VLM-only 면 GEN demote 유지.
- **④ 차별화**: vLLM-Omni([arXiv:2602.02204](https://arxiv.org/abs/2602.02204))는 stage 를 **다른 GPU** 로 disaggregate(본질 multi-GPU) → 단일 device 에선 stage connector 무의미. FluxMoE([arXiv:2604.02715](https://arxiv.org/abs/2604.02715))는 MoE expert paging(token-routing 비결정). TIDELOOM 은 MoT **tower 단위 phase-결정론 FSM** + 단일 forward 내 in-process phase 관리. Layered Prefill([arXiv:2510.08055](https://arxiv.org/abs/2510.08055))은 단일-모델 LLM layer-group, MoT dual-tower 아님.

**pseudo-code (M1: hook + FSM 스켈레톤)**:
```python
class Phase(Enum):
    VLM_ONLY, GEN_PREFILL, GEN_DENOISE, POLICY_CHUNK = range(4)

# phase -> 활성 tower module 집합 (compile-time LUT)
ACTIVE_MODULES = {
    Phase.VLM_ONLY:     {"language_model"},                 # GEN 영구 demote
    Phase.GEN_PREFILL:  {"language_model"},                 # UND 1회 forward
    Phase.GEN_DENOISE:  {"gen_layers", "norm_moe_gen", "proj_out"},
    Phase.POLICY_CHUNK: {"gen_layers", "language_model"},   # chunk loop (M3 overlap)
}

class TowerPhaseFSM:
    def __init__(self, residency_mgr, scheduler):
        self.state = None; self.rm = residency_mgr; self.sched = scheduler
    def on_und_done(self, transformer):           # hook @ forward L~1480
        # K_AR/V_AR 는 transformer.cached_kv 에 이미 상주 (zero-copy pin)
        self._transition(Phase.GEN_DENOISE)
    def on_denoise_step(self, step_idx, n_steps):  # hook @ diffuse L1492 loop
        slack = n_steps - step_idx
        self.rm.tick(self.state, slack)            # M2: slack 동안 다음 tower staging
        self.sched.maybe_overlap(self.state, slack)# M3: chunk t+1 AR prefill dispatch
    def _transition(self, nxt):
        self.rm.apply_residency(ACTIVE_MODULES[nxt], evict=ACTIVE_MODULES.get(self.state, set()))
        self.state = nxt
```

> ✅ source verified: `vllm-omni@95d56cf` [vllm_omni/diffusion/models/cosmos3/transformer_cosmos3.py#L1459-L1530](https://github.com/vllm-project/vllm-omni/blob/main/vllm_omni/diffusion/models/cosmos3/transformer_cosmos3.py#L1459-L1530) — `if self.cached_kv is None:`(L1460) → `self.language_model(text_ids, freqs_und)`(L1474) → trim(L1479) → `for layer, (k_und, v_und) in zip(self.gen_layers, self.cached_kv)`(L1499-1507). **UND 1회 → GEN N step 직렬, 경계 ~L1480 확인.** `reset_cache()` ([L1281-L1283](https://github.com/vllm-project/vllm-omni/blob/main/vllm_omni/diffusion/models/cosmos3/transformer_cosmos3.py#L1281-L1283)) 가 per-request UND cache 무효화.

### M2. Phase-Deterministic Weight Residency Manager (active-8B lock)

- **① scheme**: tower 단위 weight group(`Cosmos3Und*` ↔ `Cosmos3Gen*`, ckpt prefix `language_model.*` ↔ `gen_layers.*`)을 residency 단위로 두는 `WeightResidencyManager`. residency policy 는 **mode × device 메모리 등급 LUT** 로 결정. 비활성 tower demote + 다음 phase tower 를 **denoise slack 동안 double-buffered `cudaMemcpyAsync`(pinned host↔device)** 로 staging. eviction 은 **phase-결정적**(predictor 불요 — FluxMoE 와의 본질 차).
- **② 해결 문제**: G2 — 16B 상시 상주를 active-8B 로. Orin NX 16GB enabling(현 OOM), AGX/Thor 는 VAE/KV 경합 완화.
- **③ 동작 원리 (≥6 step)**:
  1. **compile-time LUT 구성**: generation-mode → tower-activation 결정론 schedule. 예: policy → `[UND, GEN, GEN, ..., GEN]`(UND 1회 후 GEN N step), VLM → `[UND]*`(GEN 영구 demote).
  2. **device 등급별 residency policy** (아래 표): Orin NX = active-8B INT8/INT4 必, AGX = INT8 16GB 편안, Thor = BF16/FP4 native.
  3. **demote**: `GEN_PREFILL → GEN_DENOISE` 전이 시 UND tower 의 param storage 를 placeholder 로 치환(코드 패턴은 `LayerwiseOffloadHook._set_tensor_storage` 의 tower-단위 확장). 단, K_AR/V_AR(cached_kv)는 잔류 — weight 만 demote.
  4. **staging (slack hiding)**: denoise step n 실행 중 `stream_h2d`(copy stream)로 **다음 phase tower** weight 를 buffer B 로 `cudaMemcpyAsync`. 필요 BW = 8GB / 2.1s ≈ **3.8 GB/s** vs LPDDR5 204.8 GB/s 의 **~1.9%** (§3.3 수지) → 이론상 hiding 여유 충분, 실측 gate 는 경합-하 hiding%.
  5. **double-buffer swap + event 동기**: `cudaEvent` 로 staging 완료를 마지막 denoise step *전*에 보장. 미완 시 fallback(동기 copy 또는 전량 상주) + 측정.
  6. **device 경로 분기**: **Orin (cMA=0)** → UVM prefetch 불가 → double-buffered `cudaMemcpyAsync` + pinned host + `cudaStreamAttachMemAsync(cudaMemAttachSingle)`; **Thor (cMA=1)** → `cudaMemPrefetchAsync` 정공법 대조군 (UVM-prefetch 페널티 정량).

**residency policy LUT (mode × device 메모리 등급)**:

| mode | Orin NX 16GB (INT8/INT4) | AGX Orin 64GB (BF16/INT8) | Thor 128GB (BF16/FP4) |
|---|---|---|---|
| VLM_ONLY | UND-INT4 상주(4GB), GEN 영구 demote | UND-BF16(16GB) 상주 | UND-BF16 상주 |
| GEN_DENOISE (T2I/T2V) | active-GEN INT4(4GB)+staging, UND demote | active-GEN INT8(8GB) | both BF16 fit, prefetch 대조 |
| POLICY_CHUNK | M3 off: active-tower INT4(4GB)+KV+obs; **M3 on: 양 tower(UND+GEN) 상주 必**(AR re-prefill 위해 reasoner GPU 상주) → 8GB(2×INT4)+KV+obs | M3 off: active INT8+double-buffer; **M3 on: UND+GEN 양 상주**(demote 불가) | both fit, M3 full |

- **④ 차별화**: PowerInfer-2([arXiv:2406.06282](https://arxiv.org/abs/2406.06282)) = neuron-level activation-**predictor**(비결정, miss penalty). D²MoE([arXiv:2504.15299](https://arxiv.org/abs/2504.15299))/Importance-Driven Expert Scheduling([arXiv:2508.18983](https://arxiv.org/abs/2508.18983)) = 단일-모델 MoE expert routing(비결정). `LayerwiseOffloadHook`(코드 baseline) = block 단위 demand-driven sliding window. TIDELOOM = **tower-level phase-100%-결정론**(예측기 불필요) + MoT 의 modality/기능-disjoint weight 가 swap 에 유리하다는 **구조적 property** + Orin UVM-prefetch 미지원 정확 대응(double-buffer). M2 weight-streaming 을 *단독 novelty 로 주장하지 않음* — 결정론·tower-단위·UMA-제약 대응의 *조합*이 차별(직전 score 주석 계승).
- **⚠ M2↔M3 충돌 (정정)**: **policy mode 에서는 reasoner(UND) tower demote 불가**. M3(AR∥DM overlap)는 chunk t 의 DM 중 chunk t+1 의 AR re-prefill(`language_model`)을 실행하므로 **reasoner tower 가 GPU 에 상주**해야 한다 → "DM phase 진입 시 UND 통째 demote"(③3) 와 정면 충돌. 따라서 **M2 의 active-8B(1-tower) 절감은 generation(T2I/T2V)·VLM 단독 모드 중심**이고, **policy 모드는 (M3 활성 시) 양 tower 상주가 기본** = M2 demote 이득 미적용(또는 M3 비활성·sequential fallback 시에만 demote 가능). 즉 M2 단독(S1-mini, M3 없음)은 policy 에서 demote 성립하나, M2+M3 결합 policy 는 양 tower 상주. (§3.2 수지표·아래 LUT 의 POLICY_CHUNK 행은 이 분기를 반영.)

**pseudo-code (M2: `swap_tower()` double-buffer 로직)**:
```python
class WeightResidencyManager:
    def __init__(self, device, towers):  # towers: {"und": [params...], "gen": [...]}
        self.dev = device
        self.copy_stream = torch.cuda.Stream()           # H2D background
        self.pinned = {t: _flatten_pinned(towers[t]) for t in towers}  # pin_memory=True
        self.gpu_buf = {t: None for t in towers}         # double buffer slots
        self.evt = {t: None for t in towers}

    def stage(self, tower, slack_steps):                 # called each denoise step
        if self.gpu_buf[tower] is not None: return       # already staged
        with torch.cuda.stream(self.copy_stream):        # Orin: cudaMemcpyAsync(pinned)
            buf = torch.empty_like(self.pinned[tower], device=self.dev)
            buf.copy_(self.pinned[tower], non_blocking=True)   # copy-engine, hidden under denoise
            self.gpu_buf[tower] = buf
            e = torch.cuda.Event(); e.record(self.copy_stream); self.evt[tower] = e

    def swap_tower(self, evict, load):                    # at phase boundary
        if self.evt[load] is not None:
            torch.cuda.current_stream().wait_event(self.evt[load])  # staging done?
        else:                                            # staging missed -> sync fallback (measured)
            self.gpu_buf[load] = self.pinned[load].to(self.dev, non_blocking=False)
        _rebind_params(load, self.gpu_buf[load])         # tower 전체 통째 rebind
        _evict_to_placeholder(evict)                     # demote (storage -> empty)
        # Thor (cMA=1) 경로: 위 copy 대신 cudaMemPrefetchAsync(uvm_ptr, dev, copy_stream)
```

> ✅ source verified: `vllm-omni@95d56cf` [transformer_cosmos3.py#L1058](https://github.com/vllm-project/vllm-omni/blob/main/vllm_omni/diffusion/models/cosmos3/transformer_cosmos3.py#L1058)(`self.language_model = Cosmos3LanguageModel(...)`), [#L1096-L1111](https://github.com/vllm-project/vllm-omni/blob/main/vllm_omni/diffusion/models/cosmos3/transformer_cosmos3.py#L1096-L1111)(`self.gen_layers = nn.ModuleList([Cosmos3GenDecoderLayer(...)...])`) — **두 tower 가 별도 nn.Module 트리**임을 확인 → tower-단위 demote/rebind 가능. [pipeline_cosmos3.py#L530-L550](https://github.com/vllm-project/vllm-omni/blob/main/vllm_omni/diffusion/models/cosmos3/pipeline_cosmos3.py#L530-L550) `_remap_ckpt_key` — `transformer.language_model.{k}`(UND) vs `transformer.gen_layers.{layer_idx}`(GEN) prefix 분리 확인 → state_dict prefix 로 tower group 화 가능. grep 결과 `cudaMemcpyAsync`/`memPrefetch`/`AccessPolicyWindow`/`StreamAttach` = **diffusion 스택 전체 0건** → 명시적 stream/event staging 은 greenfield(우리 기여).

### M3. AR∥DM Chunk Overlap (policy mode)

- **① scheme**: `PolicyPipelineScheduler` — chunk t 의 DM denoise(GEN, 4 step × CFG)와 chunk t+1 의 AR re-prefill(새 관측 ViT+UND)을 **CUDA stream priority 또는 MPS context** 로 병행 dispatch. compute-bound AR ∥ memory-bound DM = resource-complementary. Green Context 는 Thor-scope 비교군.
- **② 해결 문제**: G1/G5 — chunk 주기 2.1s 내 AR(compute)→DM 직렬 idle. SM issue-slot 25-40% headroom([arXiv:2508.08430](https://arxiv.org/abs/2508.08430))을 cross-regime 으로 메움.
- **③ 동작 원리 (≥6 step)**:
  1. chunk t 가 `GEN_DENOISE` 진입(FSM hook) → `PolicyPipelineScheduler.maybe_overlap` 호출.
  2. chunk t+1 의 새 관측(ViT encode + UND prefill)을 **별도 stream/MPS context** 로 dispatch (chunk t 의 denoise 와 동시).
  3. chunk t denoise 는 high-priority stream, chunk t+1 AR prefill 은 low-priority stream (`cudaStreamCreateWithPriority`).
  4. chunk t DM 완료 시 chunk t+1 의 K_AR(cached_kv)이 준비 완료 → 즉시 DM 시작(stall 0). 이때 M2 가 다음 GEN tower staging 을 이미 hide.
  5. 절대-시간 MRoPE(G7) + RTC handoff([arXiv:2506.07339](https://arxiv.org/abs/2506.07339))로 새 chunk 위치 재계산 0.
  6. split ratio(AR 30%/DM 70% 등)는 부하 비율로 튜닝. **간섭 예산**: co-location mem-BW 1.55× TBT, L2 pollution 2.15× ([arXiv:2501.16909](https://arxiv.org/abs/2501.16909)) 측정 후 동적 조정. 간섭이 overlap 이득을 상쇄하면 sequential fallback.
- **④ 차별화**: Bullet([arXiv:2504.19516](https://arxiv.org/abs/2504.19516)) / Nexus([arXiv:2507.06608](https://arxiv.org/abs/2507.06608)) / DuetServe([arXiv:2511.04791](https://arxiv.org/abs/2511.04791)) PD-multiplexing = **동일 tower** LLM prefill∥decode SM-split. (직전 phase2' 정정: 2507.06608=Nexus, Bullet 진짜=2504.19516 — 라벨 분리 표기.) M3 = **dual-tower MoT 의 AR-prefill(compute, reasoner) ∥ DM-denoise(memory, generator) cross-regime** + VLA 실시간(15Hz) + 절대-시간 MRoPE chunk handoff. VLA-XPU V-AEFusion([arXiv:2604.24447](https://arxiv.org/abs/2604.24447))이 가장 근접(2-phase async)하나 단일 edge GPU intra-device partition pipeline 미커버 → **정면 baseline 의무**.

**pseudo-code (M3: 2-stream orchestration)**:
```python
class PolicyPipelineScheduler:
    def __init__(self, transformer, residency_mgr):
        self.tf = transformer; self.rm = residency_mgr
        self.s_dm = torch.cuda.Stream(priority=-1)   # high: 현재 chunk denoise
        self.s_ar = torch.cuda.Stream(priority=0)    # low : 다음 chunk AR prefill
        self.next_obs = None

    def maybe_overlap(self, state, slack):
        if state is not Phase.POLICY_CHUNK or self.next_obs is None: return
        if slack < 1: return                          # 마지막 step 전엔 시작 안함
        with torch.cuda.stream(self.s_ar):            # MPS context fallback 도 동일 패턴
            kv_next = self.tf.language_model(self.next_obs.text_ids, self.next_obs.freqs)
            self.staged_kv = kv_next                   # chunk t+1 K_AR 준비
            e = torch.cuda.Event(); e.record(self.s_ar); self.ar_done = e

    def begin_next_chunk(self):                        # chunk t DM 완료 시
        torch.cuda.current_stream().wait_event(self.ar_done)   # K_AR ready
        self.tf.cached_kv = self.staged_kv             # zero-copy handoff
        self.tf.reset_cache_for_new_chunk()            # GEN 재사용, UND 갱신만
```

> ✅ source verified: `vllm-omni@95d56cf` [vllm_omni/diffusion/models/cosmos3/pipeline_cosmos3.py#L1486-L1530](https://github.com/vllm-project/vllm-omni/blob/main/vllm_omni/diffusion/models/cosmos3/pipeline_cosmos3.py#L1486-L1530) — `do_cfg` 경로의 `for t in self.progress_bar(timesteps)` 루프가 매 step `self.transformer(...)` 를 cond(L1497)/uncond(L1511) 2회 호출, `cond_cache`/`uncond_cache` 로 UND KV step 재사용(L1496/L1507/L1510/L1521). 이 N-step 루프가 **M3 가 chunk t+1 AR 를 숨길 slack 구간**. `reset_cache()`([L1281-L1283](https://github.com/vllm-project/vllm-omni/blob/main/vllm_omni/diffusion/models/cosmos3/transformer_cosmos3.py#L1281-L1283))가 chunk 경계 hook 지점.

### 4.X 의존성 그래프 (M1 → M2 → M3)
```
        +----------------------+
        | M1 TowerPhaseFSM     |  phase enum + active-module LUT (forward-level hook ~L1480)
        |  (model-forward)     |
        +----+------------+----+
             | phase 경계 |  step counter / slack
             v            v
  +----------------+   +------------------------+
  | M2 Residency   |   | M3 PolicyPipeline      |
  | Manager        |   | Scheduler              |
  | demote+staging |   | AR∥DM 2-stream/MPS     |
  +-------+--------+   +-----------+------------+
          |  swap_tower(evict,load)|  begin_next_chunk()
          +-----------+------------+
                      v
            zero-copy K_AR handoff (cached_kv)
```
- **M1 은 M2/M3 의 전제**: phase 경계·slack 정보를 둘 다 M1 hook 에서 받는다. M1 단독으로도 "단일-process unified runtime" 기여 성립.
- **M2 는 M1 위 독립 가치**(active-8B fit enabling) → **S1-mini = M2 단독**.
- **M3 는 M1+M2 위 stretch**(policy-mode only, MPS 간섭 risk). budget borderline 시 stretch 로 분리(직전 sanity-check 권고 계승: M1+M2 = 14주 core, M3 = stretch).

---

## 5. 평가 · 실험 플랜 (R20-β 7요소)

### 5.1 HW
- **AGX Orin 64GB (primary)**: cMA=0, 204.8 GB/s, 4MB L2. double-buffer staging 1차 실증 + falsification gate. ncu 불가 → Nsight Systems + tegrastats.
- **Orin NX 16GB (stretch)**: cMA=0, 102.4 GB/s. active-8B 必. **INT8/INT4 조합 = Q2(ANCHOR, static K_AR quant) / Q1(DRIFT, tower-asymmetric PTQ) 산출물 명시 차용** — TIDELOOM 자체는 quant-orthogonal이나 NX fit 엔 저비트가 component. (Q2=KV INT4, Q1=GEN tower vs UND tower 비대칭 bit.)
- **Thor (conditional)**: cMA=1, 273 GB/s, FP4 native, Green Context 정식. `cudaMemPrefetchAsync` 정공법 대조 + Green Context vs MPS 대조.
- **RTX Pro 6000 96GB (control)**: ncu 지원 (sm_120) → 커널 카운터 검증 + ×2→×1 단일화 대조 + 품질 대량 eval.

### 5.2 모델
- **Cosmos3-Nano 16B** (BF16-only 공식 → active-8B 는 INT8/INT4 component 가정 명시) + **Policy-DROID** (policy mode).
- fallback: BAGEL-7B-MoT 14B(7B-active, vLLM 미지원 → Diffusers/자체 path) — 16B Cosmos3 edge 미실증 risk 대비.

### 5.3 Workload (3 phase mix)
- **policy**: Policy-DROID 15Hz 32-action 연속 **100 chunk** (POLICY_CHUNK, M3 핵심).
- **T2I-512p** (GEN_DENOISE, staging hiding 검증).
- **VLM-mode**: MMMU subset (VLM_ONLY, GEN 영구 demote 검증).
- T2V-256p 짧은clip(2차). T2V-720p on-device 금지.

### 5.4 Tools
- vllm-omni **fork** (또는 Diffusers `Cosmos3OmniPipeline` 위 FSM) — TRAP #3 반영, forward-level hook.
- **Nsight Systems** (range marker, phase 경계 timeline — ncu Orin 불가).
- **tegrastats** (EMC%/전력, J/chunk 적분).
- RTX 에서만 **Nsight Compute(ncu)** 커널 카운터 cross-reference.

### 5.5 Ablation (3축)
- residency: {all-resident(a), layerwise-offload(b), **tower-FSM(ours)**}
- staging: {sync, double-buffered}
- device: {AGX Orin, Orin NX, Thor} (RTX control)
- **metrics**: chunk latency p50/p99, J/chunk, **경합-하 staging-hiding%**, 첫-chunk **cold latency**, peak resident bytes, phase-transition stall(ms, Nsight range).

### 5.6 주차별 표 (~14주; 행마다 파일경로/API/완료판정)

| 주차 | 작업 | 파일경로 / API | 완료 판정 |
|---|---|---|---|
| **W1-2** | baseline (a)(b) 구동 + 측정 harness | `diffusion_model_runner.py:load_model` (offload off/`--enable-layerwise-offload`); Nsight Systems range marker; tegrastats J/chunk 적분 | (a) 16B peak resident, (b) layerwise swap 오버헤드 수치 확보; AGX/NX/Thor 3종 구동 |
| **W3-5** | **M1 TowerPhaseFSM** | `transformer_cosmos3.py:Cosmos3VFMTransformer.forward` L~1480 phase hook; `diffuse()` L1492 step hook; phase enum + active-module LUT | phase 전이가 Nsight timeline 에 range 로 노출; VLM_ONLY 에서 GEN 미실행 확인 |
| **W6-9** | **M2 WeightResidencyManager** + **falsification gate** | `swap_tower()` double-buffer (pinned `cudaMemcpyAsync` + `cudaEvent` + `cudaStreamAttachMemAsync`); tower prefix `language_model.*`/`gen_layers.*` group; Thor `cudaMemPrefetchAsync` 분기 | **GATE: AGX 경합-하 staging-hiding% > 70%?** + active-8B peak ≤ 9GB+버퍼; 미달 시 S1-mini 축소 분기 |
| **W10-12** | **M3 AR∥DM overlap** (policy) | `PolicyPipelineScheduler` 2-stream(`cudaStreamCreateWithPriority`)/MPS; 간섭 측정·split ratio 동적 조정; 절대-시간 MRoPE handoff | chunk t+1 K_AR 준비가 chunk t DM 완료 전 끝남(stall 0); 간섭 1.55× 이내 |
| **W13-14** | full eval | Thor prefetch 대조 + RTX ×2→×1 단일화 + baseline (a-e) + 3축 ablation | 모든 metric (chunk latency p50/p99, J/chunk, hiding%, cold latency, peak) 표 완성 |

### 5.7 Preliminary (4단계)
1. **reproduction**: cookbooks 정확 실행 명령 — `vllm serve nvidia/Cosmos3-Nano`(reasoner) + generator `run_id_with_vllm.ipynb`(policy: `num_inference_steps=30, guidance_scale=1.0, flow_shift=10.0, fps=10`); `--enable-layerwise-offload` baseline 재현.
2. **attribution**: phase 별 시간/메모리 분해 — Nsight range 로 UND-prefill / DM-denoise-window / VAE 분리, tegrastats 로 phase 별 J.
3. **roofline**: staging BW 수지 — 3.8 GB/s 필요 vs LPDDR5 실측 effective BW(경합 하), staging-hiding% 의 BW-이론 상한 확인.
4. **micro-benchmark**: tower swap 단독 시간 측정 — UND 8GB demote+GEN 8GB load 의 순수 latency(경합 없는 환경, sync vs double-buffer).

---

## 6. 예상 효과 · Risk · Tier 분류 · Tier-2 variant · Scoring · Decision-tree

### 6.1 예상 효과
- **보수치**: peak weight 메모리 **~50%↓** (16GB→8-9GB active+버퍼); 경합-하 staging-hiding ≥70% 달성 시 chunk latency 오버헤드 **<5%**; phase-transition zero-copy 로 e2e **−10~18%**.
- **chunk latency**: M3 overlap 성공 시 **−15~25%** (간섭 상쇄 후).
- **best-case**: RTX Pro 6000 **×2 → ×1 단일화** (16B INT8=16GB / BF16=32GB 가 96GB 1장에 KV+활성화 포함 fit, quant 없이 성립); Thor FP4 에서 full-residency 대비 iso-latency 달성.

### 6.2 Risk + 완화
- **R1 staging-hiding 실패** (경합-하 hiding < 70%): → **S1-mini 축소** (M2 단독, policy-only, NX). M1+M3 단독 이득 유지.
- **R2 MPS 간섭** (mem-BW/L2 1.55-2.15× 가 overlap 이득 상쇄): → **sequential fallback**; AR-prefill L2-light 스케줄, split ratio 동적 조정. 간섭 측정은 RTX/Thor(ncu), Orin 은 wall-clock proxy.
- **R3 cold-start** (첫-chunk tower load 가 SLO 초과): → warm-up 전량 상주 → 첫 chunk 후 demote 시작. cold > 2× 면 demote 분기.
- **R4 16B edge 미실증**: → BAGEL-7B-MoT fallback + Thor 128GB 정공법.

### 6.3 R52.4 Tier-A/B/C
- **Tier-A (반드시 달성)**: M1 FSM + M2 active-8B residency 가 **품질 0 손실(weight bit-exact swap)**로 AGX Orin 에서 peak ~50%↓ 실증. (= 핵심 contribution)
- **Tier-B (목표)**: 경합-하 staging-hiding ≥70% → chunk latency 오버헤드 <5%; Orin NX 16GB policy fit enabling.
- **Tier-C (stretch)**: M3 AR∥DM overlap chunk latency −15~25%; Thor FP4 iso-latency; RTX ×2→×1.

### 6.4 Tier-2 variant — S1-mini
- **"Policy-mode-only active-8B residency on Orin NX 16GB"** (1 mechanism = **M2 단독**).
- scope: VLM/생성 제외, policy 단일. GEN tower 연속 점유 long window(2.1s) 동안 reasoner demote, active-8B(INT4=8GB) + KV + 3-view obs 만 16GB fit. M1(FSM)·M3(overlap) 제외.
- 1 mechanism: phase-deterministic tower demote + denoise-slack double-buffer staging.
- venue: DATE/ISLPED/IISWC short (measurement + enabling letter).

| 지표 | Baseline | S1-mini | 개선 |
|---|---|---|---|
| 배포 HW | RTX Pro 6000 ×2 | Orin NX 16GB ×1 | on-robot fit enabling |
| NX staging 이론 | — | 8GB(INT4)@102.4GB/s ≈ 78ms << 2.1s | 여유 충분 (KV/obs BW 경합이 진짜 gate) |
| peak weight | 16B 전량 | active-8B(INT4)=4GB+버퍼 | ~50%↓ |

- **승격 조건**: NX 경합-하 hiding% > 70 확인 + AGX 로 M1+M3 통합 → full S1(Tier-1).

### 6.5 Scoring (직전 7.4 계승)
- **avg 7.4** 유지. ★**impact 8** (RTX ×2→×1 단일화 + Orin NX fit enabling, TODAY 워크로드). ▼**novelty 7** (weight-streaming 일반 메커니즘 포화 — 결정론·tower-단위·KV-handoff·단일-device runtime 통합의 *조합*만 차별; M2 단독 novelty 주장 금지).
- step0-refresh(2026-06-05) 재확인: 신규 정면 scoop **0건**, worst-overlap ~50%(Layered Prefill/D²MoE 류, 단 단일-모델 expert ≠ tower-결정론). 판정 유지.

### 6.6 Decision-tree 분기 (직전 임계값 계승)
```
W6-9 falsification gate: 경합-하 staging-hiding% ?
  ├─ ≥ 70%  → M2 full 유지, W10-12 M3 진입
  │            └─ M3 간섭 < 1.55× & chunk latency −15%↑ → Tier-C 달성
  │            └─ Thor FP4 에서 full-residency 대비 iso-latency 달성 → ★ASPLOS 본선
  ├─ < 70%  → S1-mini 축소 (M2 단독, policy-only NX) + M1/M3 단독 이득 보고
  └─ cold-start > 2× (첫 chunk SLO 초과) → demote 정책 demote / warm-resident fallback
```

---

## 부록 A. 확인한 source anchor (R72.3 trace 요약)

| anchor | repo@sha 경로#라인 (clickable) | 확인 내용 |
|---|---|---|
| AR→DM 경계 (FSM hook) | `vllm-omni@95d56cf` [transformer_cosmos3.py#L1459-L1530](https://github.com/vllm-project/vllm-omni/blob/main/vllm_omni/diffusion/models/cosmos3/transformer_cosmos3.py#L1459-L1530) | `if self.cached_kv is None:`(L1460)→`language_model`(L1474)→trim(L1479)→`gen_layers` zip cached_kv(L1499-1507). UND 1회→GEN N step, 경계 ~L1480 |
| tower weight 분리 단위 | `@95d56cf` [transformer_cosmos3.py#L1058](https://github.com/vllm-project/vllm-omni/blob/main/vllm_omni/diffusion/models/cosmos3/transformer_cosmos3.py#L1058), [#L1096-L1111](https://github.com/vllm-project/vllm-omni/blob/main/vllm_omni/diffusion/models/cosmos3/transformer_cosmos3.py#L1096-L1111) | `self.language_model`(UND) / `self.gen_layers`(GEN) 별도 nn.Module 트리 |
| decoder layer 클래스 | `@95d56cf` [transformer_cosmos3.py#L723-L845](https://github.com/vllm-project/vllm-omni/blob/main/vllm_omni/diffusion/models/cosmos3/transformer_cosmos3.py#L723-L845) | `Cosmos3UndDecoderLayer`(self-attn+MLP) vs `Cosmos3GenDecoderLayer`(cross-attn to UND K/V) — tower asymmetry |
| ckpt prefix routing | `@95d56cf` [pipeline_cosmos3.py#L530-L550](https://github.com/vllm-project/vllm-omni/blob/main/vllm_omni/diffusion/models/cosmos3/pipeline_cosmos3.py#L530-L550) `_remap_ckpt_key` | `language_model.{k}` vs `gen_layers.{layer_idx}` prefix → tower group 화 |
| denoise N-step slack | `@95d56cf` [pipeline_cosmos3.py#L1486-L1530](https://github.com/vllm-project/vllm-omni/blob/main/vllm_omni/diffusion/models/cosmos3/pipeline_cosmos3.py#L1486-L1530) `diffuse()` | `for t in timesteps` CFG cond/uncond 2회 호출, cond/uncond_cache step 재사용 |
| per-request cache 무효화 | `@95d56cf` [transformer_cosmos3.py#L1281-L1283](https://github.com/vllm-project/vllm-omni/blob/main/vllm_omni/diffusion/models/cosmos3/transformer_cosmos3.py#L1281-L1283) `reset_cache()` | `cached_kv=None; cached_freqs_gen=None` — chunk 경계 hook |
| baseline (b) 메커니즘 | `@95d56cf` [layerwise_backend.py#L22-L249](https://github.com/vllm-project/vllm-omni/blob/main/vllm_omni/diffusion/offloader/layerwise_backend.py#L22-L249) | `LayerwiseOffloadHook`: per-block sliding window, `pre_forward`→`prefetch_layer`(다음 block H2D, pinned), `post_forward`→`offload_layer`(자기 free) |
| greenfield 확인 | grep diffusion/ 전체 ([vllm_omni/diffusion/](https://github.com/vllm-project/vllm-omni/blob/main/vllm_omni/diffusion/)) | `cudaMemcpyAsync`/`memPrefetch`/`AccessPolicyWindow`/`StreamAttach` = **0건** → 명시 stream/event staging 은 우리 기여 |
| offload enable site | `@95d56cf` [diffusion_model_runner.py#L124-L167](https://github.com/vllm-project/vllm-omni/blob/main/vllm_omni/diffusion/worker/diffusion_model_runner.py#L124-L167) | `load_device = "cpu" if enable_*_offload else str(device)`; `get_offload_backend(...).enable(pipeline)` |

## 부록 B. 인용 (clickable)
- [arXiv:2501.08219](https://arxiv.org/abs/2501.08219) decode 77-91% / GPU-freq 둔감
- [arXiv:2603.02271](https://arxiv.org/abs/2603.02271) AR action phase ≤75% memory-bound
- [arXiv:2511.01866](https://arxiv.org/abs/2511.01866) EdgeReasoning IISWC'25 (decode 192-569× prefill)
- [arXiv:2406.06282](https://arxiv.org/abs/2406.06282) PowerInfer-2 (activation-predictor, 메모리 40%↓)
- [arXiv:2602.02204](https://arxiv.org/abs/2602.02204) vLLM-Omni (multi-GPU disaggregation)
- [arXiv:2604.02715](https://arxiv.org/abs/2604.02715) FluxMoE (expert demand-paging)
- [arXiv:2604.27792](https://arxiv.org/abs/2604.27792) MotuBrain (serving 없음, 차별 anchor)
- [arXiv:2510.08055](https://arxiv.org/abs/2510.08055) Layered Prefill (단일-모델 LLM)
- [arXiv:2504.15299](https://arxiv.org/abs/2504.15299) D²MoE (on-device MoE)
- [arXiv:2508.18983](https://arxiv.org/abs/2508.18983) Importance-Driven Expert Scheduling
- [arXiv:2501.16909](https://arxiv.org/abs/2501.16909) co-location 간섭 (TBT 1.55×, L2 2.15×)
- [arXiv:2508.08430](https://arxiv.org/abs/2508.08430) Jetson SM issue-slot 25-40% / 8-process 95% 붕괴
- [arXiv:2504.19516](https://arxiv.org/abs/2504.19516) Bullet (PD-multiplex)
- [arXiv:2507.06608](https://arxiv.org/abs/2507.06608) Nexus (intra-GPU PD disaggregation)
- [arXiv:2511.04791](https://arxiv.org/abs/2511.04791) DuetServe (PD-multiplex)
- [arXiv:2604.24447](https://arxiv.org/abs/2604.24447) VLA-across-XPUs V-AEFusion (M3 정면 baseline)
- [arXiv:2506.07339](https://arxiv.org/abs/2506.07339) RTC real-time chunking handoff
