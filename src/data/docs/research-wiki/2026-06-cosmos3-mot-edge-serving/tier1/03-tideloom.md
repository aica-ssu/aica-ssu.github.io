# S1 TIDELOOM (Tier-1 #3, avg 7.4)

**A Single-Device Unified Serving Runtime with Phase-Deterministic Tower-Weight Residency for Dual-Tower Mixture-of-Transformers on Edge Unified Memory** · MLSys / ASPLOS 2027

---

## 1. 개요

- **metaphor↔mechanism**: TIDELOOM = TIDE(조수 — 한 tower 가 phase 마다 들고 남, A↔DM 교대) + LOOM(베틀 — Reasoner/Generator 두 실을 한 베틀에서 직조 = 단일 runtime 통합). 조수처럼 활성 tower 가 교대하고, 그 흐름을 한 베틀(단일 process)이 짜낸다.
- **한 줄**: vLLM(AR)+vLLM-Omni(DM) 분리 스택을 단일 edge GPU·단일 process 통합 런타임으로 합치고, 16B 중 phase 별 활성 8B tower 만 상주시키는 phase-결정론 weight residency.
- **Tier**: 1 (신규 단일-device 런타임 시스템). A1(Loom, ai-opt) + L1(TIDELOCK, legacy) merge, A3(Cascade)는 M3 흡수.

## 2. 기존 연구 한계 · GAP (workload evidence 수치 포함)

- **G1(dual-regime 비대칭) + G2(2× 파라미터, 1-tower 활성)**: 한 요청이 AR(memory-bound, KV-중심, token-by-token) → DM(compute-bound, KV-불변, step 반복)을 단일 edge GPU 에서 직렬 traverse 하는데, 현 스택은 vLLM(AR)·vLLM-Omni(DM)가 분리 엔진 → phase 전환을 한 process·한 메모리 풋프린트로 못 다룸.
- **workload evidence**:
  - decode = 전체 inference time 의 **77-91%** ([arXiv:2501.08219](https://arxiv.org/abs/2501.08219)); AR action phase = e2e latency 의 **최대 75%, memory-bound** ([arXiv:2603.02271](https://arxiv.org/abs/2603.02271)).
  - AGX Orin 64GB LPDDR5 **204.8GB/s**, 8B decode **7.8 tok/s** ([arXiv:2511.01866](https://arxiv.org/abs/2511.01866)). 8B(INT8=16GB) double-buffer staging ≈ Orin 204.8GB/s 에서 **~78ms**, NX 102.4GB/s 에서 ~156ms → denoise window hiding gate.
  - policy denoise window = **4 step × CFG2 = 8 forward/chunk @ 15Hz (~2.1s 주기)** (§4.2.5).
  - PowerInfer-2 coarse FFN offload 7B **메모리 40%↓** ([arXiv:2406.06282](https://arxiv.org/abs/2406.06282)).
  - 현 정책이 RTX Pro 6000 **×2** 필요(§4.2.5) = single-device 미해결. Cosmos3-Nano 16B BF16=32GB / INT8=16GB / INT4=8GB.

## 3. 제안 기법 (mechanism ≤3)

### M1. Tower-phase FSM (단일-process unified runtime)

- **① 추가 scheme**: plain-PyTorch(`Cosmos3OmniPipeline`) 또는 vLLM-Omni 단일 확장 위 `TowerPhaseFSM` (states: `AR_PREFILL→AR_DECODE→DM_DENOISE[k]→(loop)→DECODE_OUT`/VAE) + phase별 active-tower descriptor.
- **② 해결 문제**: G1 — 분리 엔진 KV 핸드오프·중복 메모리, 한 요청 두 regime 직렬 traverse.
- **③ 동작 원리**: (1) `AR_PREFILL`: Reasoner tower forward [S_AR], paged KV 에 K_AR/V_AR 적재. (2) phase 전환 trigger = AR EOS/conditioning 완료 → K_AR/V_AR block read-only pin (zero-copy 핸드오프, 동일 device·동일 KV pool). (3) `DM_DENOISE`: Generator tower [K_AR;K_DM] full-attn, step k loop, active-tower 만 CUDA stream dispatch, 비활성 tower op skip. (4) `DECODE_OUT`: VAE decode (policy 는 skip → S4 연계).
- **④ 기존 실패 이유 + 차별화**: vLLM-Omni([arXiv:2602.02204](https://arxiv.org/abs/2602.02204))는 stage 를 다른 GPU 로 disaggregate(본질 multi-GPU) → 단일 device 에선 stage connector 무의미. FluxMoE([arXiv:2604.02715](https://arxiv.org/abs/2604.02715))는 MoE expert paging(token-routing 비결정). TIDELOOM 은 MoT tower 단위 phase-결정론 + 단일 요청 AR→DM 직렬 FSM + KV zero-copy.

### M2. Phase-deterministic tower weight residency (active-8B lock)

- **① 추가 scheme**: `WeightResidencyManager` — generation-mode→tower-activation LUT(compile-time)로 비활성 tower demote, 다음 phase weight 를 double-buffered `cudaMemcpyAsync`(pinned host↔device) staging + `cudaStreamAttachMemAsync`. Orin prefetch 미지원 대응.
- **② 해결 문제**: G2 — 16B→상시 8B footprint. Orin NX 16GB 에 16B+KV+VAE fit enabling(현 OOM).
- **③ 동작 원리**: (1) generation-mode→tower-activation LUT(compile-time). (2) AR→DM 전환 예고(AR decode-length 예측) 시 Generator weight prefetch 시작, denoise 8-forward window 동안 copy-engine background stream staging, `cudaEvent` 로 마지막 step 전 완료 보장. (3) VLM-mode 는 Generator 영구 demote. (4) staging 미완 시 fallback(동기 copy / 전량 상주) + 측정. (5) Thor(cMA=1)는 `cudaMemPrefetchAsync` 대조군.
- **④ 기존 실패 이유 + 차별화**: PowerInfer-2([arXiv:2406.06282](https://arxiv.org/abs/2406.06282)) = neuron-level activation-predictor(비결정, miss penalty). TIDELOOM = tower-level phase-100%-결정론(예측기 불필요). MoT 의 modality/기능-disjoint weight 가 swap 에 유리하다는 구조적 property + Orin UVM-prefetch 미지원 정확 대응(double-buffer)이 novelty.

### M3. Chunk-pipelined AR∥DM cross-regime overlap (A3 흡수, MPS fallback)

- **① 추가 scheme**: `PolicyPipelineScheduler` — chunk t 의 DM denoise(8 forward, compute/memory-bound)와 chunk t+1 의 AR re-prefill(new obs, compute-bound)을 MPS partition(Orin JetPack 6.1+ verified) 또는 2-CUDA-stream priority 로 동시 dispatch. Green Context 는 Thor-scope 비교군.
- **② 해결 문제**: G1/G5 — chunk 주기 2.1s 내 AR(compute)→DM 직렬 idle. compute-bound AR ∥ memory-bound DM resource-complementary.
- **③ 동작 원리**: (1) chunk t DM denoise 시작과 동시 chunk t+1 AR prefill 별도 MPS context/stream dispatch. (2) chunk t DM 완료 시 chunk t+1 K_AR 준비 완료 → 즉시 DM 시작. (3) 절대-시간 MRoPE(G7)로 새 chunk 위치 재계산 0. (4) partition split ratio(AR 30%/DM 70% 등)는 부하 비율 튜닝, 간섭(mem-BW 1.55×, L2 pollution 2.15× [arXiv:2501.16909](https://arxiv.org/abs/2501.16909)) 측정 후 동적 조정.
- **④ 기존 실패 이유 + 차별화**: Bullet([arXiv:2504.19516](https://arxiv.org/abs/2504.19516))/Nexus([arXiv:2507.06608](https://arxiv.org/abs/2507.06608))/DuetServe([arXiv:2511.04791](https://arxiv.org/abs/2511.04791)) PD-multiplexing = 동일 tower LLM prefill∥decode SM-split. M3 = dual-tower MoT 의 AR-prefill(compute, reasoner) ∥ DM-denoise(memory, generator) cross-regime + VLA 실시간(15Hz) + 절대-시간 MRoPE chunk handoff. VLA-across-XPUs V-AEFusion([arXiv:2604.24447](https://arxiv.org/abs/2604.24447))이 가장 근접(2-phase async)하나 단일 edge GPU intra-device partition pipeline 미커버 → 정면 baseline 의무.

## 4. 평가 · 실험 플랜 (R20-β 7요소)

- **HW**: Thor 128GB(forward-looking primary, 16B 정공법 + prefetch 대조) / AGX Orin 64GB(constrained, INT8 16GB + double-buffer staging) / Orin NX 16GB(capacity-wall stretch, active-8B fit) / RTX Pro 6000(×2→×1 단일화 대조 + ncu control).
- **모델**: Cosmos3-Nano 16B (BF16-only → active-8B 는 INT8 component); fallback BAGEL-7B-MoT 14B(7B-active, vLLM 미지원 → Diffusers path).
- **워크로드**: policy-DROID 32-action @15Hz(1차) + T2I-512p(1차) + T2V-256p 짧은clip(2차) + VLM-mode(MMMU sample). T2V-720p on-device 금지.
- **도구**: Diffusers `Cosmos3OmniPipeline`, Nsight Systems range marker(CF-A: ncu 불요), tegrastats J/chunk.
- **ablation + baseline**: M1/M2/M3 분리 이득. baseline — (b1) vLLM+vLLM-Omni 분리 2-process(cluster→1-GPU 강제 [arXiv:2602.02204](https://arxiv.org/abs/2602.02204)), (b2) plain-PyTorch full-16B-resident, (b3) PowerInfer-2-style activation-offload([arXiv:2406.06282](https://arxiv.org/abs/2406.06282)), (b4) VLA-XPU V-AEFusion 단일-GPU 축소([arXiv:2604.24447](https://arxiv.org/abs/2604.24447)), (b5) MotuBrain-menu composed 위 추가이득([arXiv:2604.27792](https://arxiv.org/abs/2604.27792)).
- **주차별 구현**:

  | 주차 | 작업 |
  |---|---|
  | W1-3 | Diffusers 위 TowerPhaseFSM 골격 + AGX Orin staging-hiding **falsification gate**(8B double-buffer 가 8-forward window 안에 경합 하 hiding%>70?) |
  | W4-7 | WeightResidencyManager(double-buffer `cudaMemcpyAsync` + `cudaStreamAttachMemAsync`) + VLM-mode demote + AR decode-length 예측 early-prefetch |
  | W8-11 | M3 MPS partition AR∥DM overlap + 간섭 측정·split ratio 동적 조정 + 절대-시간 MRoPE handoff |
  | W12-14 | Thor prefetch 대조군 + RTX ×2→×1 단일화 + baseline + ablation |

- **preliminary metrics**: 경합 하 tower-swap hidden%(W1-3 gate), peak resident bytes.

## 5. 예상 효과 (보수치, scope 명시)

- peak resident **−35~45%** (active-8B + double-buffer, **streaming hiding 성공 시 best-case**).
- phase-transition zero-copy 로 e2e **−10~18%** (보수).
- chunk latency **−15~25%** (M3 overlap, **간섭 상쇄 후**).
- **RTX Pro 6000 ×2 → ×1 단일화 실증** (16B INT8=16GB / BF16=32GB 가 96GB 1장에 KV+activation 포함 fit — quant 없이 성립).
- scope: policy + T2I 1차, T2V-256p 2차, T2V-720p on-device 제외.

## Tier-2 scope-축소 variant (R15-β.3) = S1-mini

- **Policy-mode-only active-8B residency on Orin NX 16GB** — 1 mechanism(M2 단독).
- **mechanism 축소 benefit**: VLM/생성 제외, policy mode 단일 — Generator tower 연속 점유 long window(2.1s) 동안 reasoner demote, active-8B(INT4=8GB)+KV+3-view obs 만 16GB fit. M1(FSM)·M3(overlap) 제외.
- **venue**: DATE/ISLPED/IISWC short (measurement+enabling letter).
- **정량 표**:

  | 지표 | Baseline | variant | 개선 |
  |---|---|---|---|
  | 배포 HW | RTX Pro 6000 ×2 | Orin NX 16GB ×1 | on-robot fit enabling |
  | NX staging hiding | — | 156ms << 2.1s window | 여유 충분 (KV/obs BW 경합 측정) |

- **승격 가능성**: NX staging-hiding%>70 확인 + AGX Orin 으로 M1+M3 통합 시 full S1(Tier-1)로 승격.
