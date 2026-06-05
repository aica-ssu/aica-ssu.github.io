# Cosmos3 Nano-급 MoT(Mixture-of-Transformers) dual-tower 모델의 단일 edge GPU 효율 서빙 — DEEP RE-RUN (R72)

> NVIDIA Cosmos3 는 language/image/video/audio/**action** 을 하나의 MoT dual-tower 아키텍처(Reasoner=understanding/AR tower + Generator=diffusion tower, **attention 연산만 공유**)에서 처리하는 omnimodal world model 이다. 현 정책 배포가 RTX Pro 6000 **×2** 를 요구하는(tech report §4.2.5) 16B(활성 8B) dual-tower 를 단일 Jetson Orin/Thor 급 edge GPU 에서 효율적으로 서빙하기 위한 6 idea 묶음. 핵심 특이성은 **(1) AR↔DM dual-regime 직렬 공존**, **(2) 16B 중 phase 별 8B 한 tower 만 활성**, **(3) denoise 전 기간 read-only 인 정적 K_AR/V_AR**, **(4) video≫audio≫action 모달리티 토큰 밀도 비대칭**.

> **세션 일자**: 2026-06-05 · **Mode**: 1 (문장 수준 ideation 의 **source-code 깊이 재수행**) · **계보**: 2026-06-04 세션(`mode2-cosmos3-mot-edge-serving`)의 **DEPTH RE-RUN** — 사용자 피드백("구현 디테일이 너무 얇다")에 따라 신설된 **Rule R72** 의 **첫 적용 사례**. Top-6 idea 와 점수는 직전 세션을 계승하되, 6개 tier 문서를 **local repo 클론 source-code anchoring** 으로 전면 재작성. · **검증**: Phase-1-deep 작성 → Phase-2 adversarial verification 리뷰어 co-sign (47/48 anchor 정확, 5 fix-now 정정 반영).

---

## 1. 연구 진행 Meta

### 1.1 쿼리 원문

- 사용자 쿼리(재수행): **"edge 에서 cosmos3 nano 수준의 omni model 서빙 최적화 — R72 깊이 재수행."**
- 원(原) 쿼리(2026-06-04 계승): "엣지에서 nano 수준의 모델을 deploy 할 때 Mixture of transformer 구조를 효율적으로 서빙할 수 있는 ideation — 기존 VLM/LLM 과의 모델 아키텍처 차이, 다양한 modality 지원 차이 고려 최적화 포인트 (Cosmos3 tech report 면밀 분석 기반)."

### 1.2 키워드

- Mixture-of-Transformers (MoT) dual-tower
- omnimodal world model (Cosmos3)
- single edge GPU serving (Jetson Orin/Thor, unified LPDDR5 / UMA)
- AR↔diffusion dual-regime
- static conditioning KV (K_AR/V_AR)
- tower-aware weight residency / asymmetric PTQ
- classifier-free guidance (CFG) on single device
- **source-code anchoring (R72): vllm-omni / cosmos-framework local clone 기반 line-level 검증**

### 1.3 중점 축 (포함)

- **MoT dual-tower 구조 특이성**: layer 마다 Reasoner/Generator 두 파라미터 세트(`language_model.*` vs `gen_layers.*`)가 존재하나 **attention 연산만 공유**. 이 구조적 property 가 weight swap·tower 차등 정밀도·cross-tower KV 공유의 근거 (gaps G1/G2/G4).
- **edge 단일 GPU**: 현 정책이 RTX Pro 6000 ×2 인 16B 를 단일 Jetson UMA 로 내림. multi-GPU disaggregation(vLLM-Omni)이 불가능한 영역 — capacity·BW·energy 제약이 1차 변수 (G2/G8).
- **정적 K_AR**: conditioning 이 sampling trajectory 동안 고정 → reasoner 1회 forward 후 K_AR/V_AR 을 모든 denoising step 이 read-only 로 소비. decode-time growing-KV 와 요구가 정반대라 1-shot quant amortize·L2 pin·layout dedup 의 자유도가 큼.
- **R72 깊이**: 모든 mechanism 을 실제 file#L anchor 로 grounding (vllm-omni@`95d56cf` / cosmos-framework@`003d66d`). 각 문서 363-404 줄, R72.4 depth-gate 통과.

### 1.4 의도적 제외 축 (이유 명시)

- **학습/훈련 최적화 제외**: 본 ideation 은 inference serving 한정. pre-training MFU·distillation 학습은 쿼리 밖.
- **multi-GPU 클러스터 제외**: vLLM-Omni([arXiv:2602.02204](https://arxiv.org/abs/2602.02204))가 cluster stage-disaggregation 을 이미 점유. 본 세션은 그 메뉴를 **단일 edge GPU UMA·DVFS·offload·L2** 로 축소·특화한 빈 영역만 노림.
- **HW 신설계/PIM 제외**: 상용 Jetson + 실기기 측정 우선 — ncu 미지원(Orin sm_87)·`cudaMemPrefetchAsync` 미지원(cMA=0) 등 실제 제약을 정직히 수용.

### 1.5 직전 세션과의 관계 (Top-6 계승 + source-code 깊이 재작성)

- **Top-6 동일 유지**: Tier-1 {Q2 ANCHOR 7.6, Q1 DRIFT 7.4, S1 TIDELOOM 7.4} / Tier-2 {S2 DUOCLOCK 7.4, L3 KEELKV 6.6, S4 SIDEPOOL 6.2}.
- **Q2×L3 pair 격상**: 직전엔 "INT4 → 3MB L2 에 여유 pin (보너스)" 톤이었으나, source-anchored 재산정 결과 **policy 현실 K_AR(3-view 관측 지배 ~1,050 tok) = BF16 4.3MB/layer > 3MB L2 set-aside** 임이 확정 → **Q2 INT4(1.08MB) 가 L3 per-layer 전량 pin 의 사실상 전제**로 격상 (단순 보너스 아님).
- **재작성 산출물**: 6 tier 문서를 line-level anchor 로 전면 재작성, R72.4 depth-gate(363-404줄) 통과, adversarial verification(47/48 anchor 정확) 후 5 fix-now 정정 반영.

### 1.6 사용 인프라 (local clones 4 repo + sha)

| repo | HEAD sha | scope |
|---|---|---|
| **cosmos-framework** | `003d66d` | MoT 모델 정의(`unified_mot.py`), reference 추론 루프, tokenizer/VAE |
| **vllm-omni** | `95d56cf` | 서빙 스택. `diffusion/models/cosmos3/`(generator), quant config, offloader |
| **cosmos** | `7f5797f` | 배포 cookbooks + `inference_benchmarks.md` (벤치/배포 config) |
| **vllm** | `063ce98` | upstream vLLM (reasoner LM 서빙·paged KV·attention backend) |

> 모든 anchor 는 `> ✅ source verified: {repo}@{sha} {path}#L{a}-{b}` 형식. 검증 리뷰어가 4 repo HEAD sha 가 문서 표기와 일치함을 재확인. · **R72 신설로 본 세션이 첫 적용 사례** (R72.1 본문 길이 무제한 · R72.2 per-document 전담 agent · R72.3 source-code 직접 검증 · R72.4 depth-gate · R72.5 staging 완화).

---

## 2. Tier-1 Top 3

| Idea | Score | Domain | Venue |
|---|---|---|---|
| ① [Q2 ANCHOR](tier1/01-anchor.md) | **7.6** | static-KV 1-shot quant + flow-step×layer 누적 denoising error bound | MLSys / ICML 2027 |
| ② [Q1 DRIFT](tier1/02-drift.md) | 7.4 | tower 비대칭 PTQ (co-init quant-sensitivity divergence ρ_ℓ) | NeurIPS / ICML / MLSys 2027 |
| ③ [S1 TIDELOOM](tier1/03-tideloom.md) | 7.4 | 단일-device MoT 통합 serving runtime + phase-결정 tower residency | MLSys / ASPLOS 2027 |

### ① Q2 ANCHOR (avg 7.6)

- **(a) mechanism 별 정성 benefit (실제 anchor 포함)**:
  - **M1** Hadamard-rotated 1-shot K_AR/V_AR quant — `transformer_cosmos3.py#L1474-1479`(`cached_kv` 할당 직후 `QuantizedKVCache` wrapper). cached K 가 **post-RoPE 저장**(`Cosmos3CausalAttention.forward` L548, RoPE L545 직후 return)이므로 Hadamard 를 RoPE 뒤에 적용해 MRoPE 위치정보 보존. dequant hook 은 **local·SP 두 소비처**(`_forward_local#L648` + `_forward_sp#L675`) 모두 필요.
  - **M2** flow-step-N × layer 이중 누적 denoising error bound — `softmax_scale=1/√d`(L628) + step loop(`pipeline_cosmos3.py#L1492`) + K_AR 이 **36 gen layer 전부에서 read**(L1499-1507) 를 입력으로, `‖x̂₀−x₀‖ ≤ (Σ_{n=1}^N Δ_n·L_v^{(n)})·(Σ_{ℓ=1}^{L}(Π_{j>ℓ}L_blk^{(j)})·(L_softmax^{(ℓ)}‖ΔK_AR^{(ℓ)}‖+‖ΔV_AR^{(ℓ)}‖))`. **N 작을수록(policy N=4) bound tight = 저비트에 관대**라는 검증가능 예측.
  - **M3** bound-validation + bit 자동선택 — `component_config.py#L63-73 resolve()`(longest-prefix) per-layer bit 라우팅 + `reset_cache()`(L1281-1283) hook.
- **(b) closest competitor + 차별 axis**: QuantKeys([arXiv:2605.26266](https://arxiv.org/abs/2605.26266))/33-Method study([arXiv:2603.27469](https://arxiv.org/abs/2603.27469)) 는 video-diffusion KV quant 실증이나 **autoregressive self-forcing growing-KV + bound 부재**. 인접 OT-Quant([arXiv:2511.11418](https://arxiv.org/abs/2511.11418))는 flow-matching **weight** quant 로 step 전파를 정성 언급할 뿐 closed-form 아님. 차별축 = **static prefix K_AR 라서만 가능한 N×L 의존 closed-form bound + 1-shot Hadamard amortize**.
- **(c) 예상 gain (보수치, 4-column)**:

  | 지표 | Baseline | 본 idea | 개선 |
  |---|---|---|---|
  | conditioning-KV footprint | BF16 K_AR ~4.3MB/layer (~1,050 tok, CFG 2벌) | INT4 1-shot ~1.08MB/layer | 2-4× 압축 (품질손실 <1%) |
  | policy mode bit (N=4) | INT8 균일 | INT3, bound-favorable 시 INT2 | action-MSE <2% (bound 허용 시만) |
  | step KV-read traffic | bf16 ~155MB/step (per CFG pass) | INT4 ~39MB/step | bit 비례 BW 절감 |
  | bound-tightness | (경쟁작 전부 부재) | bit별 step×layer 누적 tightness plot | 핵심 novelty artifact |

- **(d) Tier 강등 risk**: bound 가 실측 출력 error 대비 5-10× loose(L_v 과대 + 36-layer 누적 보수성) → bound 를 상대 ordering 가이드로 강등, "1-shot Hadamard static-KV quant (bound 없이)"로 reposition (novelty↓, 여전히 publishable). >10× 또는 N-의존성 falsify(policy N=4 vs T2V N=50 bound-tightness 차이 미발생, W7-8 1차 gate) → 핵심 가설 붕괴.
- **(e) Outperform 가능성**: bound 가 load-bearing novelty — N-의존성 실증 시 KV-quant 경쟁작 전부 부재한 수학 객체로 outperform. policy INT2 성공(action-MSE <2% + bound 허용) → ICML 본선.
- **(f) Implementation envelope**: BF16-only Cosmos3-Nano fork(llm-compressor/QuaRot Hadamard) + L_v jacobian power-iteration calib + SP/TP **두 경로** dequant. `QuantizedKVCache`(+50 LoC) + dequant hook(~14 LoC) + `anchor/bound.py`(+120 LoC) + config 라우팅(~15 LoC). **~13주**, RTX Pro 6000 calibration(ncu BW) + AGX Orin footprint/latency proxy.

### ② Q1 DRIFT (avg 7.4)

- **(a) mechanism 별 정성 benefit (실제 anchor 포함)**:
  - **M1** ρ_ℓ divergence 측정 (구 Q4 SIEVE 흡수) — tower 의 물리 분리가 `transformer_cosmos3.py#L1058-1111`(`language_model` vs `gen_layers` 가 **동일 hyperparameter 로 co-instantiate**)로 확정. framework 측은 `unified_mot.py#L430-495`의 `_moe_gen` 접미사. co-init controlled experiment 로 `ρ_ℓ = S_ℓ^R/S_ℓ^G`(Hessian-trace × quant-perturbation) + weight-drift δ_ℓ 상관 측정. **reasoner-detach 하 generator HVP** 프로토콜로 cross-attn gradient 누설 차단.
  - **M2** UMA 예산 제약 asymmetric bit ILP — `min Σ S_ℓ^τ(b) s.t. Σ mem(b)≤B`. 결과를 prefix dict 로 `ComponentQuantizationConfig` 에 직배선. **결정적 precedent**: `qwen3_omni_moe_thinker.py#L1150-1151` 이 이미 `ComponentQuantizationConfig({"language_model": ...})` tower-prefix 라우팅을 **실사용** → DRIFT M2 는 작동 코드의 확장(gen_layers 라우팅이 greenfield).
  - **M3** iso-memory 배포 검증 — asymmetric vs uniform 동일 footprint 에서 3-mode(VLM/gen/policy) 횡단 tower 분리 품질.
- **(b) closest competitor + 차별 axis**: World-Model-Quant([arXiv:2602.02110](https://arxiv.org/abs/2602.02110))가 encoder/predictor 모듈 비대칭을 선점하나 **서로 다른 구조·init = confounded**. 차별축 = **동일 init·동일 구조 co-init 두 tower 가 두 objective(AR-CE vs flow-MSE)로 발산하는 controlled attribution + serving-stack 통합 bit 배분**.
- **(c) 예상 gain (보수치, ρ_ℓ 발산 클 때만)**:

  | 지표 | Baseline | 본 idea | 개선 |
  |---|---|---|---|
  | 품질 @동일 메모리 | 균일-W4 whole-model | per-tower bit-alloc (ILP) | 0.5-1.5%p 회복 |
  | footprint @동일 품질 | 균일 정밀도 | tower 비대칭 | 25-40% 절감 |
  | ρ_ℓ 분포 + corr(δ_ℓ,ρ_ℓ) | (측정 없음) | tower별 sensitivity plot | novel artifact |

- **(d) Tier 강등 risk**: ρ_ℓ < 1.5× across ≥80% layers (판정 bit 고정, 예 b=4) → measurement-only letter(IEEE CAL)로 강등. 균일-W4 대비 회복 <0.5%p (이득 측정불가) → 2차 gate demote.
- **(e) Outperform 가능성**: 발산 material 하면 tower-agnostic baseline 전부를 conditional-gain 으로 outperform; BAGEL 일반화 시 "법칙" 격상. 발산 작으면 negative-but-publishable.
- **(f) Implementation envelope**: BF16-only + SVDQuant/AWQ 이식 fork + Hutchinson HVP double-backward calib, 2 PTQ method × {8,4} per tower prune. **~13주**, RTX calib + AGX Orin 배포 metric(ncu 불요).

### ③ S1 TIDELOOM (avg 7.4)

- **(a) mechanism 별 정성 benefit (실제 anchor 포함)**:
  - **M1** Tower-Phase FSM (model-forward-level) — `Cosmos3VFMTransformer.forward#L1459-1530` 의 UND→GEN 경계(~L1480, `if self.cached_kv is None:` L1460 → `language_model` L1474 → trim L1479 → `gen_layers` zip L1499-1507)를 phase enum + active-module LUT 로 1급 객체화. **TRAP #3 정정**: 두 phase 는 **단일 diffusion forward 안에 in-process 순차** 존재 → hook 은 scheduler-level 이 아니라 model-forward-level (직전 "분리 엔진 통합" 서술 정정).
  - **M2** Phase-Deterministic Weight Residency — 비활성 tower demote + 다음 phase tower 를 denoise slack 동안 double-buffered `cudaMemcpyAsync`(pinned). **greenfield 입증**: diffusion 스택 전체에 `cudaMemcpyAsync`/`memPrefetch`/`AccessPolicyWindow`/`StreamAttach` **0건**; 기존 `LayerwiseOffloadHook` 은 per-block sliding window(demand-driven)이라 우리의 tower-단위 phase-결정론과 다름.
  - **M3** AR∥DM Chunk Overlap (policy) — chunk t 의 DM ∥ chunk t+1 의 AR re-prefill 을 2-stream/MPS 로. CFG cond/uncond 2-call(L1497/L1511)이 만드는 slack window 활용.
- **(b) closest competitor + 차별 axis**: vLLM-Omni([arXiv:2602.02204](https://arxiv.org/abs/2602.02204))는 stage 를 다른 GPU 로 disaggregate(본질 multi-GPU), FluxMoE([arXiv:2604.02715](https://arxiv.org/abs/2604.02715))는 MoE expert paging(token-routing 비결정). 차별축 = **MoT tower 단위 phase-100%-결정론(predictor·miss penalty 부재) + 단일-device in-process FSM + Orin UVM-prefetch(cMA=0) 미지원 double-buffer 대응** (조합이 차별, M2 단독 novelty 금지).
- **(c) 예상 gain (보수치)**:

  | 지표 | Baseline | 본 idea | 개선 |
  |---|---|---|---|
  | peak resident weight | full-16B-resident | active-8B + double-buffer | −35~50% (경합 하 hiding ≥70% 성공 시) |
  | e2e latency | (분리 엔진 핸드오프 가정) | zero-copy phase 전환 | −10~18% |
  | chunk latency | AR→DM 직렬 | M3 2-stream overlap (간섭 상쇄 후) | −15~25% |
  | 배포 GPU 수 | RTX Pro 6000 ×2 | ×1 (96GB 단일) | 단일화 실증 (quant 없이) |

- **(d) Tier 강등 risk**: W6-9 falsification gate — 경합 하 staging-hiding% <70% → S1-mini 축소(M2 단독, policy-only NX). cold-start >2× → warm-resident fallback. **M2↔M3 충돌(정정)**: policy 모드 + M3 활성 시 reasoner demote 불가(AR re-prefill 위해 양 tower 상주 必) → policy active footprint 는 2-tower.
- **(e) Outperform 가능성**: VLA-XPU V-AEFusion([arXiv:2604.24447](https://arxiv.org/abs/2604.24447)) 정면 baseline 으로 intra-device partition pipeline 으로 outperform; Thor FP4 iso-latency → ASPLOS 본선.
- **(f) Implementation envelope**: 단일-process vllm-omni fork(또는 Diffusers reference) 위 forward-level hook. M1+M2 = 14주 core, M3 = stretch. **최장 ~14주**, AGX Orin constrained primary + Orin NX stretch + Thor + RTX ×2→×1 대조.

---

## 3. Tier-2 독립 Top 3

| Idea | Score | Domain | Venue |
|---|---|---|---|
| ④ [S2 DUOCLOCK](tier2/01-duoclock.md) | **7.4** | intra-request AR↔DM EMC/GPU 분리 DVFS + J/chunk ledger | DATE / ISLPED / IISWC 2027 |
| ⑤ [L3 KEELKV](tier2/02-keelkv.md) | 6.6 | static cross-tower K_AR 의 per-layer L2 set-aside pinning | DAC / DATE 2027 |
| ⑥ [S4 SIDEPOOL](tier2/03-sidepool.md) | 6.2 | frozen tokenizer accelerator-complex placement LUT + async overlap | DATE / ISPASS / DAC 2027 |

### ④ S2 DUOCLOCK (avg 7.4)

- **(a) mechanism 별 정성 benefit (실제 anchor 포함)**:
  - **M1** Phase-Anchored Frequency Plan — `forward` UND→GEN 경계(L1459-1500) / `diffuse` step loop(`pipeline_cosmos3.py#L1492`)에서 userspace governor daemon 에 **결정적 통지(predictor 없음)** → AR(memory-bound)에 GPU↓·EMC max, DM(compute-bound)에 GPU max·EMC↓. **EMC 제어 = BPMP debugfs 3-step(`mrq_rate_locked/state/rate`, experimental)** → on-robot 프로덕션 1차 경로는 **nvpmodel 프리셋 fallback**(debugfs 강등).
  - **M2** J/chunk Measurement Ledger — INA3221 33-50ms > step 이라 J/step 불가 → `DiffusionPipelineProfilerMixin.stage_durations`(`diffusion_pipeline_profiler.py#L80-113`) 정렬 + chunk(policy 8 forward/2.1s)·phase(gen) 회계 + **freq settle-time kill 경계** 정식화.
- **(b) closest competitor + 차별 axis**: SparseDVFS([arXiv:2603.21908](https://arxiv.org/abs/2603.21908))가 CPU/GPU/EMC triplet + edge + intra-inference granularity 선점. 생존축 = **intra-request AR↔DM modality-regime 전환(diffusion denoise 포함) + phase-결정성(predictor 불필요) + MoT omnimodal 최초**.
- **(c) 예상 gain (보수치)**:

  | 지표 | Baseline | 본 idea | 개선 |
  |---|---|---|---|
  | J/chunk (policy) | 고정 nvpmodel MAXN | phase별 EMC/GPU 독립 | −15~30% (상호작용·transition 차감 보수치) |
  | chunk p99 latency | 고정 freq | chunk-단위 전환 | +<5% |
  | freq settle 경계 | (미정량) | settle vs phase 길이 kill-gate | idea 생존선 정식화 |

- **(d) Tier 강등 risk**: settle_time > phase_length → drop. J/chunk <15% 또는 SparseDVFS-style 대비 순증분 <5%p → 측정-only letter. EMC 가 어느 경로로도 phase-제어 불가 → GPU-only DVFS 축소(AR phase GPU↓ 42%급 절감만 생존).
- **(e) Outperform 가능성**: multi-tenant phase-mix governor 로 확장 시 Tier-1(MLSys/HPCA) 승격. on-robot battery 시나리오에서 GreenLLM/DualScale outperform.
- **(f) Implementation envelope**: 표준 Jetson sysfs/debugfs + INA3221 devkit (최저 risk, arch-impl 8). S1 의 phase boundary hook 재사용. **~10주**, AGX Orin devkit 1차.

### ⑤ L3 KEELKV (avg 6.6)

- **(a) mechanism 별 정성 benefit (실제 anchor 포함)**:
  - **M1** Persistent-Window Pinning — `cached_kv`(L1474-1479)를 연속 buffer 로 재배치 후 `cudaAccessPolicyWindow{Persisting}` 으로 L2 set-aside 에 pin; video-K_DM 은 `Streaming` demote. layer 순차 실행(L1499-1507)으로 rolling-window. 소비 site `cat([k_und,k])`(L648-649).
  - **M2** Verification & Fallback — RTX Pro 6000 ncu(`l2_tex_hit_rate`)로 hit-rate 인과 증명 → Orin(ncu 미지원) latency/EMC proxy; set-aside 미지원 시 SMEM 수동 캐시 fallback.
- **(b) closest competitor + 차별 axis**: Async KV Prefetch→L2([arXiv:2504.06319](https://arxiv.org/abs/2504.06319))가 KV 를 L2 로 prefetch(상주 pin 아님)·decode-time growing-KV·discrete GPU. 차별축 = **denoise 전기간 read-only static cross-tower K_AR 의 per-layer L2 set-aside pin + video-K_DM streaming demote** (Orin sm_87 4MB L2, set-aside 실측). vLLM/SGLang/llama.cpp/TRT 전부 `accessPolicyWindow` 미사용 = greenfield.
- **(c) 예상 gain (보수치, K_AR 비중 조건부)**:

  | 지표 | Baseline | 본 idea | 개선 |
  |---|---|---|---|
  | per-step DRAM read | K_AR DRAM 재독출 | L2-pin (K_AR 비중만큼) | 10-20% 절감 |
  | per-step latency | no L2 pin | pin on | 5-12% 절감 |
  | `l2_tex_hit_rate`(K_AR) | reasoner-caching only | pin (RTX 측정) | hit↑ |

- **(d) Tier 강등 risk**: RTX ncu pin on/off 유의차 없음(K_AR 비중<10%) 또는 Orin Δlatency <3% → negative-characterization measurement letter(IEEE CAL). set-aside 실측 <3MB → INT4 CFG 2벌(2.16MB) 마진 소진 → cond-only pin(1.08MB). set-aside 미지원 → SMEM fallback.
- **(e) Outperform 가능성**: T2V video-K_DM 클수록 pollution 회피 이득 큼. rolling-window 일반화 + 멀티 워크로드 + Orin ≥10% latency↓ → Tier-1(DAC full → MLSys/ASPLOS) 승격.
- **(f) Implementation envelope**: custom CUDA ext(pybind `cudaStreamSetAttribute`/`cudaDeviceSetLimit`) + 연속 재배치 hook. **~9주** 최견고. **Q2 INT4 K_AR 이 per-layer 전량 pin 의 전제** (BF16 4.3MB > 3MB set-aside).

### ⑥ S4 SIDEPOOL (avg 6.2)

- **(a) mechanism 별 정성 benefit (실제 anchor 포함)**:
  - **M1 (standalone 핵심)** Tokenizer Placement Characterization (LUT) — frozen tokenizer{video VAE, audio VAE, ViT}×가속기{iGPU/DLA/PVA/CPU}×해상도×dtype 의 latency/energy/PSNR/**fallback%** grid. VAE op 은 소스 확인(`wan2pt2_vae_4x16x16.py`: `CausalConv3d` L74 / `RMS_norm` L102 / `AttentionBlock`(SDPA) L256). **DLA 비호환**: CausalConv3d/RMS_norm/AttentionBlock → **encode 측 2D subgraph 만 DLA 후보**. AGX Orin = **NVDLA v2.0**(직전 "3.1" 정정).
  - **M2 (보조)** LUT-Driven Async Dispatch — `_decode_latents`(L987-1004, 동기 L1003; diffuse 종료 후 L1867)를 별도 stream/cuDLA 로 비동기 dispatch → 다음 요청 denoise 와 overlap. stream/event greenfield(0건).
- **(b) closest competitor + 차별 axis**: PipeDiT([arXiv:2511.12056](https://arxiv.org/abs/2511.12056))/DeDiVAE([arXiv:2512.07350](https://arxiv.org/abs/2512.07350))가 "VAE∥denoise overlap" 선점하나 **multi-GPU group 분리**. 차별축 = **단일 edge SoC 이종 가속기 복합체(iGPU/DLA-2Dsubgraph/PVA/CPU) placement LUT** ("어느 가속기" 문제 자체가 multi-GPU 엔 없음). overlap(M2)은 LUT 의 한 활용처.
- **(c) 예상 gain (보수치, generation-mode 한정)**:

  | 지표 | Baseline | 본 idea | 개선 |
  |---|---|---|---|
  | e2e chunk latency | all-on-iGPU serial | placement + async (VAE critical path 비중만큼) | −5~15% (DLA 무관, GPU-stream/CPU 로도) |
  | J/chunk | naive serial | DLA/CPU 2D-subgraph 오프로드(부분) | −5~15% (fallback% 의존) |
  | denoise GPU occupancy | VAE 경합 | VAE 분리 | 회복 |

- **(d) Tier 강등 risk**: VAE share <8%(256p 기준 고정) → drop. DLA-supported subgraph <20% → DLA drop, placement={GPU,CPU,PVA} 한정 특성화 letter. M2 overlap% ≤ PipeDiT-식 baseline → M1 placement LUT 단독 letter. policy 는 video-latent decode skip → generation-mode 한정 scope.
- **(e) Outperform 가능성**: 단일-SoC 이종 가속기 placement = direct prior art 부재. placement LUT 가 단일 자원 대비 e2e 25%+ + DLA fallback<30% + 멀티모델 일반화 → Tier-1(systems runtime) 승격.
- **(f) Implementation envelope**: ncu 불요 Orin 측정 완결 + `trtexec --useDLACore` fallback% gate + cuDLA hybrid + INT8 DLA calibration set 구축. **~10주**, arch-impl 7(특성화-only floor 견고).

---

## 4. 미선정 아이디어 요약 (직전 9건 계승)

> 9건 모두 2026-06-04 세션에서 탈락, 2026-06-05 step0-refresh 재검토 결과 **판정 변동 없음**. 4-요소 상세(GAP/시도/사유/재방문)와 Phase 별 탈락 지점은 → [unselected.md](unselected.md), 원본 상세는 → [2026-06-04 세션 unselected](/research-wiki/2026-06/cosmos3-mot-edge-serving/unselected).

- **A2 Keystone** (G4 runtime layout + CFG dedup): read-only static-KV layout + exact CFG cond/uncond shared-context batching. **탈락** — M2 "exact DM-self matmul 공유"가 layer-recursion 누락으로 layer-1-only(36L 중 1L) 붕괴 → FasterCache approx 영역과 차별축 소멸 (Phase-2' Task1). M1 layout 은 S1 에 흡수.
- **Q5 RELAY** (modality bit/step 비대칭): video→action attention perturbation bound + modality ILP. **탈락** — UD-VLA "joint denoising > decoupled" 반증 + p_v 결과 의존 → Tier-1 보류. 재방문: p_v 분포 pilot 에서 attention-mass 집중 시 elective 승격.
- **S3 LEDGERMARK** (edge MoT 특성화 producer): per-phase J/chunk + co-residency 간섭 + transition stall 통합 ledger. **탈락** — 측정 방법론을 각 selected idea 의 preliminary 로 흡수(novelty 상한 낮음). 재방문: Thor + ncu-on-Thor 확인 시 IISWC/ISPASS letter.
- **A3 Cascade** (chunk-pipeline scheduler): Green Context SM-split intra-GPU pipeline. **탈락** — standalone novelty ~55%(Bullet/Nexus/DuetServe PD-multiplexing) → S1-M3 으로 흡수.
- **A4+L4** (Compass transition + Ledgerline energy): phase-transition stall + energy/interference 측정. **탈락** — 4 reviewer 만장일치 "S3 로 통합" 후 S3 자체가 미선정. 재방문: S3 와 동일.
- **Q4 SIEVE** (measurement-only divergence study): ρ_ℓ/κ/δ 측정 letter. **탈락** — Q1-M1 과 측정핵심 100% 중복(자가중복) → Q1 motivation 으로 흡수. 재방문: Q1 분리 필요 시 IEEE CAL 단독 spinoff.
- **A5 Herald** (cross-tower DM step-skip gate): AR semantic 변화 신호로 DM step skip. **DROP** — DISK([arXiv:2602.00440](https://arxiv.org/abs/2602.00440))가 cross-modal skip 발상 ~55-60% 선점. 재방문: AR(non-diffusion) tower 신호가 step-skip 정확도 >10%p 개선 증명 시.
- **A6 Switchback** (VLM↔생성 tower multiplex): Green Context 동시 dispatch co-scheduling. **DROP-PARK** — GenServe([arXiv:2604.04335](https://arxiv.org/abs/2604.04335)) concurrent diffusion co-serve 선점 + Green Context Orin 미가용. 재방문: MPS 재구현 + 멀티테넌트 workload + GenServe baseline 비교 시.
- **Q3 PRISM** (CFG attention partition-decomposition 공유): log-sum-exp partition-merge exact 공유 custom kernel. **DROP** — exact 공유가 layer-1-only + FasterCache 영역 + kernel overhead. exact-sharing 자산은 A2-②로 흡수.

---

## 5. Essential Reading List (5편)

본 세션 idea 구현 전 반드시 읽어야 할 최강 경쟁자 + baseline (직전 세션 동일):

| Paper | 왜 읽어야 하는가 | baseline idea |
|---|---|---|
| [MotuBrain (arXiv:2604.27792)](https://arxiv.org/abs/2604.27792) | **🔴 전체 세션 최강 경쟁자 (~70% overlap)**. three-stream MoT world-action 이 거의 동일 클래스 + 거의 동일 추론 메뉴(action-only suffix=video skip, DiT cache, FP8, chunked closed-loop)로 50× speedup/11Hz 달성. 단 **edge(Jetson) UMA·offload·DVFS·L2 미다룸** — 전 세션의 유일한 숨구멍. | 전 idea 의 차별화 anchor |
| [vLLM-Omni (arXiv:2602.02204)](https://arxiv.org/abs/2602.02204) | **S1 primary baseline**. any-to-any 를 stage graph 로 fully-disaggregate (per-stage batching/GPU 할당, denoising caching, RingAttention CP). 단 **multi-GPU 클러스터 전제** → 단일 Jetson tower 관리로 축소가 명확한 edge gap. | S1 (baseline) |
| [QuantKeys (arXiv:2605.26266)](https://arxiv.org/abs/2605.26266) | **Q2 closest**. video diffusion KV quant + Jensen-bias 보정. 단 autoregressive self-forcing growing-KV + 순수 실증, **N-의존 closed-form bound 부재** → Q2 의 static-prefix N×L bound 차별축. | Q2 (baseline) |
| [SparseDVFS (arXiv:2603.21908)](https://arxiv.org/abs/2603.21908) | **S2 closest**. CPU/GPU/EMC freq triplet + discrete-edge + intra-inference granularity (EMC축 선점). 단 **operator-sparsity 기반**이지 AR↔DM modality-regime 전환 아님 → S2 의 dual-regime + phase-결정성 순증분이 생존축. | S2 (baseline) |
| [VLA-across-XPUs (arXiv:2604.24447)](https://arxiv.org/abs/2604.24447) | **특성화 경쟁 (~55% overlap)**. "compute-bound VLM backbone → memory-bound Action Expert" 2-phase + edge NPU 6× speedup + leaderboard. 단 dual-**tower** param-separation·CFG·VAE·world-model 모드 미커버 → S1-M3 의 정면 대조 의무. | S1 (M3 baseline) |

---

## 6. Implementation-Priority Decision Tree (R14.4)

### 6.0 학생 구현 우선순위 결정 트리 (SVG)

> 학생이 어떤 순서로 무엇을 먼저 구현하고, 각 정량 gate 에서 어디로 분기하는지를 보이는 implementation-priority 결정 트리. ASCII 원본(§6.1) 과 branch 표(§6.2) 의 시각화.

<svg viewBox="0 0 1180 980" style="width:100%;max-width:1180px;height:auto" xmlns="http://www.w3.org/2000/svg" font-family="system-ui, -apple-system, Segoe UI, Roboto, sans-serif">
  <defs>
    <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M0,0 L10,5 L0,10 z" fill="#1E2761"/>
    </marker>
    <marker id="arrowred" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M0,0 L10,5 L0,10 z" fill="#F96167"/>
    </marker>
  </defs>
  <rect x="0" y="0" width="1180" height="980" fill="#ffffff"/>
  <rect x="300" y="20" width="580" height="78" rx="8" fill="#CADCFC" stroke="#1E2761" stroke-width="2"/>
  <text x="590" y="46" text-anchor="middle" font-size="15" font-weight="700" fill="#1E2761">W0-2 공통 인프라 (모든 idea 의존)</text>
  <text x="590" y="68" text-anchor="middle" font-size="12.5" fill="#1E2761">vllm-omni cosmos3 구동 (Cosmos3VFMTransformer.forward) + 측정 harness</text>
  <text x="590" y="86" text-anchor="middle" font-size="12.5" fill="#1E2761">tegrastats + Nsight Systems + wall-clock · J/chunk 재정의 (ncu Orin 미지원)</text>
  <line x1="590" y1="98" x2="590" y2="128" stroke="#1E2761" stroke-width="2" marker-end="url(#arrow)"/>
  <text x="200" y="150" text-anchor="middle" font-size="13" font-weight="700" fill="#2C5F2D">먼저 (RTX calib)</text>
  <text x="940" y="150" text-anchor="middle" font-size="13" font-weight="700" fill="#1E2761">병렬 트랙 (edge 비의존)</text>
  <polygon points="590,130 760,200 590,270 420,200" fill="#F96167" stroke="#1E2761" stroke-width="2"/>
  <text x="590" y="185" text-anchor="middle" font-size="14" font-weight="700" fill="#ffffff">Q2 ANCHOR · 13주</text>
  <text x="590" y="205" text-anchor="middle" font-size="12" fill="#ffffff">INT4 bound-tightness</text>
  <text x="590" y="223" text-anchor="middle" font-size="12" fill="#ffffff">@N=4 (step×layer 누적)</text>
  <line x1="420" y1="200" x2="250" y2="200" stroke="#F96167" stroke-width="2" marker-end="url(#arrowred)"/>
  <text x="335" y="192" text-anchor="middle" font-size="11.5" fill="#F96167">5-10× / &gt;10×·N-falsify</text>
  <rect x="40" y="172" width="200" height="56" rx="6" fill="#F9E795" stroke="#1E2761" stroke-width="1.5"/>
  <text x="140" y="195" text-anchor="middle" font-size="11.5" fill="#1E2761">5-10× → ordering 강등</text>
  <text x="140" y="214" text-anchor="middle" font-size="11.5" fill="#1E2761">&gt;10× → CRITICAL-FAIL</text>
  <text x="600" y="288" text-anchor="start" font-size="11.5" font-weight="700" fill="#2C5F2D">≤3× PASS → INT4 K_AR 1.08MB 산출</text>
  <line x1="590" y1="270" x2="590" y2="312" stroke="#1E2761" stroke-width="2.5" marker-end="url(#arrow)"/>
  <text x="775" y="298" text-anchor="middle" font-size="11.5" font-weight="700" fill="#1E2761">══필수══ (격상 dependency edge)</text>
  <polygon points="590,312 760,382 590,452 420,382" fill="#F96167" stroke="#1E2761" stroke-width="2"/>
  <text x="590" y="365" text-anchor="middle" font-size="14" font-weight="700" fill="#ffffff">L3 KEELKV · 9주</text>
  <text x="590" y="385" text-anchor="middle" font-size="11.5" fill="#ffffff">RTX ncu l2_tex_hit_rate</text>
  <text x="590" y="403" text-anchor="middle" font-size="11.5" fill="#ffffff">pin on/off (Q2-INT4 전제)</text>
  <line x1="420" y1="382" x2="250" y2="382" stroke="#F96167" stroke-width="2" marker-end="url(#arrowred)"/>
  <text x="335" y="374" text-anchor="middle" font-size="11.5" fill="#F96167">&lt;3% / set-aside 미지원</text>
  <rect x="40" y="354" width="200" height="56" rx="6" fill="#F9E795" stroke="#1E2761" stroke-width="1.5"/>
  <text x="140" y="377" text-anchor="middle" font-size="11.5" fill="#1E2761">Orin Δlat&lt;3% → CAL letter</text>
  <text x="140" y="396" text-anchor="middle" font-size="11.5" fill="#1E2761">미지원 → SMEM fallback</text>
  <rect x="430" y="468" width="320" height="40" rx="6" fill="#2C5F2D"/>
  <text x="590" y="493" text-anchor="middle" font-size="12.5" font-weight="700" fill="#ffffff">Δhit≥+15%p &amp; Orin Δlat≥5% → DAC/DATE full</text>
  <line x1="940" y1="98" x2="940" y2="540" stroke="#1E2761" stroke-width="1.5" stroke-dasharray="5,4"/>
  <polygon points="940,160 1085,215 940,270 795,215" fill="#F96167" stroke="#1E2761" stroke-width="2"/>
  <text x="940" y="200" text-anchor="middle" font-size="13" font-weight="700" fill="#ffffff">S1 TIDELOOM · 14주</text>
  <text x="940" y="219" text-anchor="middle" font-size="11.5" fill="#ffffff">W6-9 경합-하</text>
  <text x="940" y="237" text-anchor="middle" font-size="11.5" fill="#ffffff">staging-hiding% gate</text>
  <rect x="800" y="284" width="280" height="38" rx="6" fill="#F9E795" stroke="#1E2761" stroke-width="1.5"/>
  <text x="940" y="300" text-anchor="middle" font-size="11" fill="#1E2761">≥70% PASS / 50-70% M3 stretch</text>
  <text x="940" y="316" text-anchor="middle" font-size="11" fill="#1E2761">&lt;70% → CRITICAL S1-mini (M2 단독)</text>
  <line x1="940" y1="270" x2="940" y2="284" stroke="#1E2761" stroke-width="2" marker-end="url(#arrow)"/>
  <polygon points="940,338 1085,393 940,448 795,393" fill="#F96167" stroke="#1E2761" stroke-width="2"/>
  <text x="940" y="385" text-anchor="middle" font-size="13" font-weight="700" fill="#ffffff">Q1 DRIFT · 13주</text>
  <text x="940" y="404" text-anchor="middle" font-size="11.5" fill="#ffffff">ρ_ℓ divergence (b=4 고정)</text>
  <line x1="940" y1="322" x2="940" y2="338" stroke="#1E2761" stroke-width="2" marker-end="url(#arrow)"/>
  <rect x="800" y="462" width="280" height="38" rx="6" fill="#F9E795" stroke="#1E2761" stroke-width="1.5"/>
  <text x="940" y="478" text-anchor="middle" font-size="11" fill="#1E2761">≥1.5×@&gt;20%L PASS / &lt;1.5×@80%L → CAL</text>
  <text x="940" y="494" text-anchor="middle" font-size="11" fill="#1E2761">회복&lt;0.5%p → CRITICAL (측정불가)</text>
  <line x1="940" y1="448" x2="940" y2="462" stroke="#1E2761" stroke-width="2" marker-end="url(#arrow)"/>
  <line x1="590" y1="540" x2="590" y2="582" stroke="#1E2761" stroke-width="2" marker-end="url(#arrow)"/>
  <text x="590" y="560" text-anchor="middle" font-size="12.5" font-weight="700" fill="#1E2761">공통 측정 harness 소비 (측정-중심 Tier-2)</text>
  <polygon points="350,584 520,654 350,724 180,654" fill="#F96167" stroke="#1E2761" stroke-width="2"/>
  <text x="350" y="637" text-anchor="middle" font-size="13" font-weight="700" fill="#ffffff">S2 DUOCLOCK · 10주</text>
  <text x="350" y="656" text-anchor="middle" font-size="11.5" fill="#ffffff">SparseDVFS 순증분</text>
  <text x="350" y="674" text-anchor="middle" font-size="11.5" fill="#ffffff">J/chunk + settle vs phase</text>
  <rect x="210" y="740" width="280" height="56" rx="6" fill="#F9E795" stroke="#1E2761" stroke-width="1.5"/>
  <text x="350" y="763" text-anchor="middle" font-size="11" fill="#1E2761">≥5%p &amp; J/chunk≥15% PASS · &lt;5%p letter</text>
  <text x="350" y="782" text-anchor="middle" font-size="11" fill="#1E2761">settle&gt;phase → CRITICAL (drop)</text>
  <line x1="350" y1="724" x2="350" y2="740" stroke="#1E2761" stroke-width="2" marker-end="url(#arrow)"/>
  <polygon points="830,584 1000,654 830,724 660,654" fill="#F96167" stroke="#1E2761" stroke-width="2"/>
  <text x="830" y="637" text-anchor="middle" font-size="13" font-weight="700" fill="#ffffff">S4 SIDEPOOL · 10주</text>
  <text x="830" y="656" text-anchor="middle" font-size="11.5" fill="#ffffff">VAE share% (256p 고정)</text>
  <text x="830" y="674" text-anchor="middle" font-size="11.5" fill="#ffffff">→ trtexec DLA fallback%</text>
  <rect x="690" y="740" width="280" height="56" rx="6" fill="#F9E795" stroke="#1E2761" stroke-width="1.5"/>
  <text x="830" y="763" text-anchor="middle" font-size="11" fill="#1E2761">share≥8% &amp; DLA≥20% PASS · DLA&lt;20% M1만</text>
  <text x="830" y="782" text-anchor="middle" font-size="11" fill="#1E2761">share&lt;8% → CRITICAL (drop, 비병목)</text>
  <line x1="830" y1="724" x2="830" y2="740" stroke="#1E2761" stroke-width="2" marker-end="url(#arrow)"/>
  <line x1="590" y1="582" x2="350" y2="584" stroke="#1E2761" stroke-width="2" marker-end="url(#arrow)"/>
  <line x1="590" y1="582" x2="830" y2="584" stroke="#1E2761" stroke-width="2" marker-end="url(#arrow)"/>
  <rect x="40" y="828" width="1100" height="120" rx="8" fill="#ffffff" stroke="#1E2761" stroke-width="1.5"/>
  <text x="60" y="852" text-anchor="start" font-size="12.5" font-weight="700" fill="#1E2761">범례 / dependency edges</text>
  <rect x="60" y="864" width="20" height="14" fill="#F96167" stroke="#1E2761"/><text x="86" y="876" font-size="11.5" fill="#1E2761">decision gate (정량 임계)</text>
  <rect x="300" y="864" width="20" height="14" fill="#CADCFC" stroke="#1E2761"/><text x="326" y="876" font-size="11.5" fill="#1E2761">process (인프라)</text>
  <rect x="490" y="864" width="20" height="14" fill="#F9E795" stroke="#1E2761"/><text x="516" y="876" font-size="11.5" fill="#1E2761">분기 액션 callout</text>
  <rect x="680" y="864" width="20" height="14" fill="#2C5F2D"/><text x="706" y="876" font-size="11.5" fill="#1E2761">Tier-1 outcome (DAC/DATE full)</text>
  <text x="60" y="900" font-size="11.5" fill="#1E2761">Q2→L3 (필수 격상): policy K_AR BF16 4.3MB &gt; 3MB L2 set-aside → INT4(1.08MB, CFG 2벌=2.16MB≤3MB) 가 per-layer 전량 pin 의 전제.</text>
  <text x="60" y="920" font-size="11.5" fill="#1E2761">Q2 INT4 → S1 (시너지): 저비트 K_AR → double-buffer staging 여유↑. · S1 TowerPhaseFSM hook → S2 phase boundary notify 재사용.</text>
  <text x="60" y="940" font-size="11.5" fill="#1E2761">공통 측정 harness(W0-2) → S2(J/chunk ledger)·S4(VAE placement) 공유. · 임계값: Phase-1-deep 작성 → Phase-2 리뷰어 co-sign (47/48 anchor).</text>
</svg>

### 6.1 양식 A — ASCII flowchart

```
[주차 0-2: 공통 인프라 (모든 idea 의존)]
  │
  ├─→ vllm-omni cosmos3 구동: Cosmos3OmniDiffusersPipeline / Cosmos3VFMTransformer.forward
  │     ├─ Nano 16B BF16 RTX Pro 6000 fit? → Yes: 진행
  │     └─ No / edge OOM → BAGEL-7B-MoT (14B/7B-active, vLLM 미지원→Diffusers) fallback
  │
  ├─→ 측정 harness: tegrastats(EMC%/전력 20-30Hz) + Nsight Systems(range marker) + wall-clock
  │     │   (ncu Orin sm_87 미지원 → 커널카운터는 RTX Pro 6000/Thor 에서만)
  │     ├─ J/step 직접? → No (33-50ms<step) → J/chunk(policy 8 forward/2.1s)·J/phase 재정의
  │     └─ harness 검증 OK → idea 분기
  │
  ▼
[Q2 ANCHOR — 먼저 (13주, RTX Pro 6000 calibration, edge 의존성 없음)]
  │   INT4 bound-tightness ratio @N=4 (step×layer 누적 bound)
  │     ├─ ≤3× → PASS (Tier-1, bit 자동선택 + INT4 K_AR 산출물 → L3 공급)
  │     ├─ 5-10× → BELOW (bound 를 상대 ordering 으로 강등, quant-only reposition)
  │     ├─ >10× / N-의존성 falsify(W7-8) → CRITICAL-FAIL (핵심 가설 붕괴)
  │     └─ policy INT2 (action-MSE<2% + bound 허용) → OUTPERFORM (ICML 본선)
  │   │
  │   └─[edge: Q2 INT4 K_AR 1.08MB] ══필수══┐ (격상된 dependency edge)
  ▼                                          │
[L3 KEELKV — Q2 INT4 전제 소비 (9주, paper-pair)]
  │   RTX ncu l2_tex_hit_rate pin on/off + per-step DRAM
  │     ├─ Δhit≥+15%p & DRAM↓ → PASS (Orin Δlatency≥5% → DAC/DATE full)
  │     ├─ Orin Δlatency <3% → BELOW (한계-특성화 IEEE CAL letter)
  │     ├─ set-aside 미지원 / K_AR 비중<10% → CRITICAL-FAIL (SMEM fallback / negative-char)
  │     └─ T2V K_DM pollution 회피 큼 + rolling 일반화 → OUTPERFORM (Tier-1 승격)
  │
  ▼ (Q2 와 병렬 가능)
[S1 TIDELOOM — runtime, 최장 14주]
  │   W6-9 경합-하 staging-hiding% falsification gate
  │     ├─ ≥70% → PASS (Tier-1, M1+M2 core, M3 진입)
  │     ├─ 50-70% → BELOW (M3 stretch, M1+M2 core)
  │     ├─ <70% (M2 fail) → CRITICAL-FAIL (S1-mini 축소, M2 단독 policy-only NX)
  │     └─ Thor FP4 iso-latency / RTX ×2→×1 단일화 → OUTPERFORM (ASPLOS 본선)
  │
  ▼ (독립)
[Q1 DRIFT — 독립 calibration (13주, RTX calib)]
  │   ρ_ℓ divergence across layers (판정 bit 고정, 예 b=4)
  │     ├─ ≥1.5× across >20% layers → PASS (Tier-1)
  │     ├─ <1.5× across ≥80% layers → BELOW (measurement letter Q4 SIEVE 강등)
  │     ├─ 균일-W4 대비 회복 <0.5%p → CRITICAL-FAIL (이득 측정불가)
  │     └─ BAGEL 일반화 "법칙" → OUTPERFORM
  │
  ▼ (공통 측정 harness 소비)
[S2 DUOCLOCK (10주) / S4 SIDEPOOL (10주) — 측정 중심]
  │   S2: SparseDVFS-style 순증분 J/chunk
  │     ├─ ≥5%p & J/chunk≥15% → PASS (Tier-2)  ├─ <5%p → BELOW (측정-only letter)
  │     ├─ settle>phase → CRITICAL-FAIL (drop)  └─ multi-tenant governor → OUTPERFORM
  │   S4: VAE share% (256p 고정) → trtexec DLA fallback%
  │     ├─ share≥8% & DLA subgraph≥20% → PASS (Tier-2)
  │     ├─ DLA<20% → BELOW (placement LUT={GPU,CPU,PVA}, M1 단독)
  │     ├─ share<8% → CRITICAL-FAIL (drop, VAE 비병목)  └─ placement 다양성 → OUTPERFORM
```

### 6.2 양식 C — 6 idea × 단계 × 4 branch 액션 표

| Idea | 주차/단계 | Pass | Below | Critical-fail | Outperform |
|---|---|---|---|---|---|
| **Q2 ANCHOR** | W5-8 bound 유도+tightness | N×L 닫힌식 + INT4 ≤3× | 5-10× ordering 강등 | >10× / N-falsify | N=4 tight 입증 |
| | W9-12 bit sweep+RoboLab | <1% 품질손실 | INT3 만 | action-MSE >2% | INT2 bound-favorable |
| **Q1 DRIFT** | W5-7 ρ_ℓ 측정(gate) | ρ_ℓ≥1.5× material | <1.5% across 80% → CAL | 회복<0.5%p 측정불가 | (δ_ℓ,ρ_ℓ) 인과 + BAGEL |
| | W8-13 bit-alloc + iso-mem | 품질 0.5-1.5%p 회복 | footprint만 절감 | — | tower-asymmetric 우월 |
| **S1 TIDELOOM** | W6-9 hiding gate | ≥70% M1+M2 full | 50-70% M3 stretch | <70% S1-mini | RTX ×2→×1 |
| | W10-14 M3 overlap+baseline | chunk −15~25% | 간섭 상쇄 후만 | 간섭이 이득 상쇄 | Thor FP4 iso-latency |
| **S2 DUOCLOCK** | W4-7 캘리브+settle | phase별 곡선 + settle<phase | J/phase 만 | settle>phase drop | — |
| | W8-10 governor vs SparseDVFS | J/chunk≥15% + 순증분≥5%p | <5%p 측정-only | 순증분 0 | multi-tenant 승격 |
| **L3 KEELKV** | W5-6 RTX ncu A/B | Δhit≥+15%p & read↓ | — | 유의차 없음 negative | — |
| | W7-9 Orin proxy + Q2 pair | Δlatency≥5% DAC | <3% CAL letter | set-aside 미지원 SMEM | rolling+멀티WL 승격 |
| **S4 SIDEPOOL** | W1-4 VAE share + DLA fallback | share≥8% & DLA≥20% | DLA<20% M1 단독 | share<8% drop | — |
| | W7-10 LUT + async overlap | overlap > PipeDiT-식 | overlap≤baseline M1만 | — | accelerator-complex 최초 |

### 6.3 inter-idea dependency edge

- **Q2 → L3 (필수화 격상)**: policy 현실 K_AR(BF16 4.3MB/layer)는 3MB L2 set-aside 초과 → Q2 의 INT4(1.08MB) 가 L3 의 **per-layer 전량 pin 의 사실상 전제** (CFG ON 2벌 = 2×1.08 = 2.16MB ≤ 3MB). 직전 "보너스 producer"에서 "전제"로 격상. 단 BF16 앞쪽 token segment 부분 pin·INT8 1벌 fit 으로 L3 **standalone** 도 성립 (전량 pin=INT4 전제(강) / 부분 pin=optional 의 조건 분기).
- **공통 측정 harness → S2/S4**: 주차 0-2 의 tegrastats + Nsight Systems + `stage_durations` harness 를 S2(J/chunk ledger)·S4(VAE placement latency/J)가 공유.
- **Q2 INT4 → S1 (시너지)**: Q2 의 저비트 K_AR 이 S1 의 active-tower residency 압축과 시너지 (KV footprint↓ → double-buffer staging 여유↑).
- **S1 TowerPhaseFSM hook → S2**: S1 의 phase boundary hook(forward ~L1480)을 S2 phase boundary notify callback 으로 재사용.

### 6.4 2-agent 검증 trace

- 분기 임계값은 **Phase-1-deep 전문가(systems / algorithm)가 작성**, **Phase-2 검증 리뷰어(ai-impl + arch-sys 통합)가 co-sign**.
- co-sign 증거: Phase-2 adversarial verification 이 47/48 anchor 정확 확인 + **5 fix-now 정정**을 co-sign — 그 중 (i) **KEELKV K_AR 재산정**(ViT patch 14×14→16×16, 채택 1,500→**~1,050 tok**, per-layer BF16 6.0→**4.3MB**·INT4 1.5→**1.08MB**, 출처를 tech report §4.2.5 로 정정)과 (ii) **ANCHOR bound 의 36-layer 누적 항 추가**(구 single-layer 식은 optimistic-invalid → step×layer 이중 누적으로 정정)가 대표적 co-sign 정정 사례다.

---

## 7. Deep-Dive 신규 발견 (오늘 세션 코드-수준, 10건)

R72 source-anchoring 으로 직전 세션엔 없던 코드-수준 사실 10건을 확정. 각 항목은 근거 문서 §로 링크.

1. **cached K 는 post-RoPE 저장** — `Cosmos3CausalAttention.forward`(L548, RoPE L545 직후 return) → Hadamard-on-K 는 RoPE 후 적용해야 MRoPE 위치정보 보존. → [ANCHOR §4.M1](tier1/01-anchor.md)
2. **dequant/pin hook 2곳 필요** — `_forward_local#L648` + `_forward_sp#L675`(SP/TP 경로 별도 분기). 누락 시 SP 무손상 깨짐. → [ANCHOR §4.M1](tier1/01-anchor.md) · [KEELKV §3.1](tier2/02-keelkv.md)
3. **tower-prefix 라우팅 실사용 precedent** — `qwen3_omni_moe_thinker.py#L1150-1151` 이 이미 `ComponentQuantizationConfig({"language_model": ...})` tower-prefix 라우팅 실사용 → DRIFT M2 는 작동 코드의 확장(gen_layers 라우팅이 greenfield). → [DRIFT §4.M2](tier1/02-drift.md)
4. **vllm-omni quant 실제 메뉴** — `factory.py#L142-153 _OVERRIDES` = `{int8, mxfp8, mxfp4, mxfp4_dualscale, inc(=auto-round), gguf}` + vLLM registry(GPTQ/AWQ/fp8 등) → reasoner=AWQ/GPTQ/int8, generator=mxfp4 를 prefix 별 배선 가능. → [DRIFT §4.M2](tier1/02-drift.md)
5. **diffusion 스택 stream/event/pinned staging 0건** — `cudaMemcpyAsync`/`memPrefetch`/`AccessPolicyWindow`/`StreamAttach` grep 0건 → TIDELOOM M2 greenfield 입증. 기존 `LayerwiseOffloadHook` 은 per-block sliding window(demand-driven). → [TIDELOOM §3.4·M2](tier1/03-tideloom.md)
6. **policy 도 CFG ON (guidance 3.0)** — CFG cond/uncond 2-call(`pipeline_cosmos3.py` L1497/L1511) + DROID policy guidance scale 3.0 (`action_policy_robolab_server.md`·tech report L2007) → 직전 일부 문서의 "policy CFG off(guidance=1.0)" 오기 정정. K_AR cond/uncond **2벌**. → [ANCHOR §0·§3.2](tier1/01-anchor.md) · [KEELKV §1.3](tier2/02-keelkv.md) · [DUOCLOCK §5.2](tier2/01-duoclock.md)
7. **K_AR 토큰 수 정정 + 코드/tech-report 시나리오 병기** — ViT 16×16 patch + 2×2 merge → policy 관측 ~1,050 tok, per-layer BF16 4.3 / INT8 2.15 / INT4 1.08 MB. **단 코드상 serving K_AR 는 text-id 토큰만**(`L1474` `embed_tokens(text_ids)`, 관측은 GEN tower 로 별도 주입) — tech-report 관측-지배 시나리오(A, worst-case)와 코드 text-only 시나리오(B) 병기. → [KEELKV §1.3](tier2/02-keelkv.md)
8. **ANCHOR bound 에 36-layer 누적 항 추가** — K_AR 이 36 gen layer 전부에서 read(L1499-1507) → step×layer 이중 누적. 구 single-layer 식은 layer-간 전파 누락으로 optimistic-invalid 였음. → [ANCHOR §4.M2](tier1/01-anchor.md)
9. **Orin EMC 제어 = BPMP debugfs 3-step (experimental)** — `clk/emc/{mrq_rate_locked,state,rate}` 직접 write 는 devkit 특성화 경로 → 프로덕션 1차 경로는 nvpmodel 프리셋 fallback. → [DUOCLOCK §M1·§6.2](tier2/01-duoclock.md)
10. **DLA 비호환 op + 버전 정정** — `CausalConv3d`/`RMS_norm`/`AttentionBlock`(SDPA) DLA 미지원 → encode 측 2D subgraph 만 후보. AGX Orin = **NVDLA v2.0**(직전 "3.1" 정정). → [SIDEPOOL §3.2·§1.4](tier2/03-sidepool.md)
