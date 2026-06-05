# Cosmos3 Nano-급 MoT(Mixture-of-Transformers) dual-tower 모델의 단일 edge GPU 효율 서빙

> **⚠️ Superseded (2026-06-05)**: 본 번들의 tier1/tier2 문서는 구현 디테일 보강을 위해 [2026-06-05 deep 번들](/research-wiki/2026-06/cosmos3-edge-serving-deep) 로 대체되었습니다 (R72 적용, source-code anchored). 본 파일은 이력 보존용. 일부 수치 정정 있음 (K_AR 산정 / policy CFG ON / ANCHOR bound 누적 항).

> NVIDIA Cosmos3 (2026-06-01) 는 language/image/video/audio/**action** 을 한 MoT dual-tower 아키텍처(Reasoner AR tower + Generator diffusion tower, **attention 연산만 공유**)에서 처리하는 omnimodal world model 이다. 현 정책 배포가 RTX Pro 6000 **×2** 를 요구하는(tech report §4.2.5) 16B(활성 8B) dual-tower 를 단일 Jetson Orin/Thor 급 edge GPU 에서 효율적으로 서빙하기 위한 6 idea 묶음. 핵심 특이성은 **(1) AR↔DM dual-regime 직렬 공존**, **(2) 16B 중 phase 별 8B 한 tower 만 활성**, **(3) denoise 전 기간 read-only 인 정적 K_AR/V_AR**, **(4) video≫audio≫action 모달리티 토큰 밀도 비대칭**.

> **세션 일자**: 2026-06-04 · **Mode**: 2 (local PDF: `cosmos3 omni model technical-report.pdf`, NVIDIA) + 1 (문장 수준 ideation) 하이브리드 · **참여 expert**: ai-optimization + legacy-system + algorithm · **참여 reviewer**: novelty + differentiation + impact + ai-implementation + arch-system-implementation (5인) · **세션 로그**: staging `2026-06-04-*`

---

## 1. 연구 진행 Meta

### 1.1 쿼리 원문

- 사용자 쿼리: "엣지에서 nano 수준의 모델을 deploy 할 때 Mixture of transformer 구조를 효율적으로 서빙할 수 있는 ideation — 기존 VLM/LLM 과의 모델 아키텍처 차이, 다양한 modality 지원 차이 고려 최적화 포인트 (Cosmos3 tech report 면밀 분석 기반)."

### 1.2 키워드 (4-8)

- Mixture-of-Transformers (MoT) dual-tower
- omnimodal world model (Cosmos3)
- single edge GPU serving (Jetson Orin/Thor, unified LPDDR5)
- AR↔diffusion dual-regime
- static conditioning KV (K_AR/V_AR)
- tower-aware weight residency
- classifier-free guidance (CFG) on single device

### 1.3 중점 축 (포함)

- **MoT dual-tower 구조 특이성**: layer 마다 Reasoner/Generator 두 파라미터 세트가 존재하나 **attention 연산만 공유** (LayerNorm+QKV/O+FFN 전부 분리). 이 구조적 property 가 weight swap·tower 차등 정밀도·cross-tower KV 공유의 근거 (gaps G1/G2).
- **edge 단일 GPU**: 현 정책이 RTX Pro 6000 ×2 인 16B 를 단일 Jetson UMA(통합 LPDDR5)로 내림. multi-GPU disaggregation(vLLM-Omni)이 불가능한 영역 — capacity·BW·energy 제약이 1차 변수 (G2/G8).
- **modality 비대칭**: video VAE 토큰(수만) ≫ audio(25 tok/s) ≫ action(~32) 의 밀도 차 + 절대 시간축 MRoPE 정렬. 모달리티별 sparsity/precision/step 차등화 여지 (G6/G7).
- **정적 K_AR**: conditioning(text+image) 이 sampling trajectory 동안 고정 → reasoner 1회 forward 후 K_AR/V_AR 을 모든 denoising step 이 read-only 로 소비 (G4). decode-time growing KV 와 요구사항이 정반대라 1-shot quant amortize·layout dedup·L2 pin 의 자유도가 큼.

### 1.4 의도적 제외 축 (이유 명시)

- **학습/훈련 최적화 제외**: 본 ideation 은 inference serving 한정. pre-training MFU·data curation·distillation 학습은 쿼리("deploy ... 서빙") 밖 → 모든 idea 가 inference-time mechanism.
- **multi-GPU 클러스터 제외**: vLLM-Omni([arXiv:2602.02204](https://arxiv.org/abs/2602.02204))가 cluster stage-disaggregation 을 이미 점유 (worst-overlap ~55-60%). 본 세션은 그 메뉴를 **단일 edge GPU UMA·DVFS·offload** 로 축소·특화하는 빈 영역만 노림.
- **HW 신설계/PIM 제외**: action-bottleneck 논문([arXiv:2603.02271](https://arxiv.org/abs/2603.02271))이 HBM/PIM 신설계를 제안하나, 본 세션은 **상용 Jetson + 실기기 측정 우선** 원칙 — commodity SW-only 로 ncu 미지원(Orin)·`cudaMemPrefetchAsync` 미지원 등 실제 제약을 정직히 수용.

### 1.5 검색 전략 (3-agent)

- **unified-serving agent**: AR+Diffusion 통합 아키텍처(MoT/Transfusion/BAGEL) + DiT 가속(caching/distillation/quant/token reduction) + 멀티모달 서빙 시스템(vLLM-Omni/GenServe). 결론: 정면 scoop 없음, vLLM-Omni 가 유일한 ≥50% 위험.
- **VLA-edge agent**: nano-급 MoT world-action 모델의 efficient edge serving 선행/경쟁 (17편). 최강 경쟁자 = MotuBrain([arXiv:2604.27792](https://arxiv.org/abs/2604.27792)), 특성화 경쟁 = VLA-across-XPUs([arXiv:2604.24447](https://arxiv.org/abs/2604.24447)).
- **workload-char agent**: IISWC/ISPASS 측정 + 벤치마크 기술보고 (R17 IISWC≥3·ISPASS≥3 충족). phase 이질성·모달리티 에너지 비대칭·Jetson 활용 갭·co-location 간섭 정량 evidence 확보.

### 1.6 검증 trace

- **분기 임계값은 Phase-1' 전문가(systems / algorithm)가 작성**, **Phase-2' 통합 패널이 co-sign** (TASK 5 decision-tree branch co-validation 에서 6 idea 임계값 전부 방향 동의 + 5건 강화).

---

## 2. Tier-1 Top 3

| Idea | Score | Domain | Venue |
|---|---|---|---|
| ① [Q2 ANCHOR](tier1/01-anchor.md) | 7.6 | static-KV 1-shot quant + flow-step denoising error bound | MLSys / ICML 2027 |
| ② [Q1 DRIFT](tier1/02-drift.md) | 7.4 | tower 비대칭 PTQ (co-init sensitivity divergence) | NeurIPS / ICML / MLSys 2027 |
| ③ [S1 TIDELOOM](tier1/03-tideloom.md) | 7.4 | 단일-device MoT 통합 serving runtime + phase-결정 tower residency | MLSys / ASPLOS 2027 |

### ① Q2 ANCHOR (avg 7.6)

- **(a) mechanism 별 정성 benefit**:
  - ① Hadamard-rotated 1-shot K_AR/V_AR quant: K_AR 이 정적·1회·read-only 라 비싼 per-channel asymmetric INT4/INT3 + Hadamard rotation + MSE-optimal clipping 을 1회만 치르고 모든 denoising step 에서 amortize.
  - ② Flow-matching denoising error bound: denoising 출력 오차 `‖x̂_0−x_0‖ ≤ (Σ_{n=1}^{N} Δ_n·L_v^{(n)})·(L_softmax·‖ΔK_AR‖+‖ΔV_AR‖)` 를 flow-step 수 N 에 명시 의존하는 닫힌 형식으로 유도 → N 작을수록(policy N=4) bound tight = policy 가 저비트에 관대하다는 검증가능 예측.
  - ③ Deploy & bound-validation: bit{8,6,4,3,2}별 bound-tightness plot 산출 후 tolerance 로 layer별 bit 자동 선택.
- **(b) closest competitor + 차별 axis**: QuantKeys([arXiv:2605.26266](https://arxiv.org/abs/2605.26266))/33-Method KV-quant study([arXiv:2603.27469](https://arxiv.org/abs/2603.27469)) 둘 다 video-diffusion KV quant 를 sweep 하나 **autoregressive self-forcing growing-KV 대상 + 순수 실증**. ANCHOR 의 차별축 = **static prefix K_AR 라서만 가능한 N-의존 closed-form bound + 1-shot Hadamard amortize** (WebFetch 로 두 경쟁작에 bound 부재 확인 → clear scoop 회피).
- **(c) 예상 gain (보수치)**:

  | 지표 | Baseline | 본 idea | 개선 |
  |---|---|---|---|
  | conditioning-KV footprint | FP16/BF16 K_AR | INT4/INT3 1-shot | 2-4× 압축 (품질손실 <1%) |
  | policy mode bit (N=4) | INT8 균일 | INT3, bound-favorable시 INT2 | action-MSE <2% (bound 허용시만) |
  | step KV-read latency | BF16 반복 read | 저비트 read | bit 비례 BW 절감 |
  | bound-tightness | (경쟁작 없음) | bit별 tightness plot | 핵심 artifact 산출 |

- **Tier 강등 risk**: bound 가 실측 출력 error 대비 >5× loose(L_v 과대) 시 empirical-tightening fallback, >10× 시 "1-shot Hadamard static-KV quant (bound 없이)" 로 demote.
- **Outperform 가능성**: bound 가 load-bearing novelty — N-의존성이 실증(policy N=4 vs T2V N=50 의 bound-tightness 차이)되면 KV-quant 경쟁작 전부 부재한 수학 객체로 outperform 가능.
- **Implementation envelope**: BF16-only Cosmos3-Nano fork(llm-compressor/QuaRot) + L_v jacobian calib ~13주, RTX Pro 6000 calibration(ncu BW 측정) + AGX Orin footprint/latency.

### ② Q1 DRIFT (avg 7.4)

- **(a) mechanism 별 정성 benefit**:
  - ① M1 = ρ_ℓ divergence 측정 study: 두 tower 가 동일 Qwen3-VL-8B co-init 인데 서로 다른 objective(AR-CE vs flow-MSE)로 학습 → quant 민감도 발산을 `ρ_ℓ=S_ℓ^R/S_ℓ^G`(HAWQ-style Hessian trace×weight-drift)로 측정, 교란변수(init·구조) 통제 첫 사례.
  - ② M2 = ILP/greedy bit allocation: UMA 예산 하 `min Σ S_ℓ^τ(b) s.t. Σ bits ≤ B`, reasoner=AWQ/GPTQ·generator=SVDQuant 경로 분리.
  - ③ M3 = deploy validation: MMMU/GenEval/RoboLab-120 subset.
- **(b) closest competitor + 차별 axis**: Mix-QViT([arXiv:2501.06357](https://arxiv.org/abs/2501.06357))/KL-Lens([arXiv:2604.13440](https://arxiv.org/abs/2604.13440)) 은 layer-importance mixed-precision 으로 성숙. 차별축 = **co-init 된 두 tower 가 두 objective 로 발산하는 ρ_ℓ divergence 측정각** — 기존은 전부 tower-agnostic / 단일 objective.
- **(c) 예상 gain (보수치, ρ_ℓ 발산 클 때만)**:

  | 지표 | Baseline | 본 idea | 개선 |
  |---|---|---|---|
  | 품질 @동일예산 | 균일-W4 whole-model | per-tower bit-alloc | 0.5-1.5%p 회복 |
  | footprint | 균일 정밀도 | tower 비대칭 | 10-18% 절감 |
  | ρ_ℓ 분포 | (측정 없음) | tower별 sensitivity plot | novel artifact |

- **Tier 강등 risk**: ρ_ℓ divergence <1.5× across ≥80% layers → measurement letter(Q4 SIEVE, IEEE CAL)로 demote; 균일-W4 대비 품질 회복 <0.5%p 면 이득 측정불가로 demote (2차 gate).
- **Outperform 가능성**: 발산이 material 하면 tower-agnostic baseline 전부를 conditional-gain 으로 outperform; 발산 작으면 negative-but-publishable("균일이 충분").
- **Implementation envelope**: BF16-only + SVDQuant 이식 fork, 2 PTQ method × {8,4} per tower 로 prune ~14주, RTX calib + AGX Orin 배포 metric(ncu 불요).

### ③ S1 TIDELOOM (avg 7.4)

- **(a) mechanism 별 정성 benefit**:
  - M1 Tower-phase FSM: plain-PyTorch(`Cosmos3OmniPipeline`) 또는 vLLM-Omni 단일 확장 위 `AR_PREFILL→AR_DECODE→DM_DENOISE[k]→DECODE_OUT` 상태기계 + K_AR/V_AR zero-copy read-only pin → 분리 엔진 KV 핸드오프·중복 메모리 제거 (G1).
  - M2 Phase-deterministic tower residency: 비활성 tower 를 demote, 다음 phase weight 를 denoise 8-forward window 동안 double-buffered `cudaMemcpyAsync` 로 staging (Orin prefetch 미지원 대응) → 16B 를 상시 8B footprint 로 (G2).
  - M3 Chunk-pipelined AR∥DM overlap: chunk t 의 DM denoise 와 chunk t+1 의 AR re-prefill 을 MPS partition/2-stream priority 로 동시 dispatch (compute-bound AR ∥ memory-bound DM 상보).
- **(b) closest competitor + 차별 axis**: vLLM-Omni([arXiv:2602.02204](https://arxiv.org/abs/2602.02204))는 stage 를 다른 GPU 로 disaggregate(본질 multi-GPU), FluxMoE([arXiv:2604.02715](https://arxiv.org/abs/2604.02715))는 MoE expert 단위 paging(token-routing 비결정). 차별축 = **MoT tower 단위 phase-100%-결정론 + 단일-device AR→DM 직렬 FSM + KV zero-copy + Orin UVM-prefetch 미지원 double-buffer 대응** (조합이 차별, 단일 메커니즘 단독 주장 금지).
- **(c) 예상 gain (보수치)**:

  | 지표 | Baseline | 본 idea | 개선 |
  |---|---|---|---|
  | peak resident bytes | full-16B-resident | active-8B + double-buffer | −35~45% (streaming hiding 성공시) |
  | e2e latency | 분리 2-process KV 핸드오프 | zero-copy phase 전환 | −10~18% |
  | chunk latency | AR→DM 직렬 | M3 MPS overlap (간섭 상쇄후) | −15~25% |
  | 배포 GPU 수 | RTX Pro 6000 ×2 | ×1 (96GB 단일) | 단일화 실증 (quant 없이) |

- **Tier 강등 risk**: W1-3 staging-hiding gate fail (8B double-buffer 가 8-forward window 안에 **경합 하 hiding%>70** 못 채움) → M2 재설계(+4주 위험), 불가시 M1+M3 만 유지(S1-mini 축소).
- **Outperform 가능성**: VLA-XPU V-AEFusion 단일-GPU 축소([arXiv:2604.24447](https://arxiv.org/abs/2604.24447))를 정면 baseline 으로, intra-device partition pipeline 으로 outperform 가능.
- **Implementation envelope**: 단일-process(CF-E 로 22주→~14주), Thor 128GB primary + AGX Orin constrained + Orin NX stretch + RTX ×2→×1 대조. 최장 14주.

---

## 3. Tier-2 독립 Top 3

| Idea | Score | Domain | Venue |
|---|---|---|---|
| ④ [S2 DUOCLOCK](tier2/01-duoclock.md) | 7.4 | intra-request AR↔DM EMC/GPU 분리 DVFS + J/chunk LUT | DATE / ISLPED / IISWC 2027 |
| ⑤ [L3 KEELKV](tier2/02-keelkv.md) | 6.6 | static cross-tower K_AR 의 per-layer L2 set-aside pinning | DAC / DATE 2027 |
| ⑥ [S4 SIDEPOOL](tier2/03-sidepool.md) | 6.2 | VAE∥denoise GPU stream-overlap + accelerator-complex placement LUT | DATE / ISPASS 2027 |

### ④ S2 DUOCLOCK (avg 7.4)

- **(a) mechanism 별 정성 benefit**:
  - M1 decoupled EMC/GPU DVFS governor: 한 요청 내부 AR phase(memory-bound) 에 EMC max + GPU↓, DM phase(compute-bound) 에 GPU max + EMC↓ 를 `/sys/class/devfreq/{gpu,emc}` userspace governor 로 독립 set → 단일 nvpmodel 이 못 잡는 상반 phase 동시 최적.
  - M2 J/chunk + J/inference-phase 특성화 LUT: tegrastats 33-50ms < step 한계로 J/step 직접 불가 → J/chunk(policy 15Hz 2.1s 주기)·J/phase 로 재정의, INA3221 rail 적분.
- **(b) closest competitor + 차별 axis**: SparseDVFS([arXiv:2603.21908](https://arxiv.org/abs/2603.21908))가 CPU/GPU/EMC freq triplet + edge + intra-inference granularity 까지 도달(EMC축 선점). DUOCLOCK 생존축 = **intra-request AR↔DM dual-regime(diffusion denoise 포함) modality-phase 전환** + J/chunk 재정의 — SparseDVFS 는 operator-sparsity 기반이지 AR↔DM modality-regime 전환 아님.
- **(c) 예상 gain (보수치)**:

  | 지표 | Baseline | 본 idea | 개선 |
  |---|---|---|---|
  | J/chunk | 고정 nvpmodel MAXN | phase별 EMC/GPU 독립 | −25~38% (phase별 독립 가정 명시, 단순합 아님) |
  | SLO 위반 (policy 2.1s) | 고정 freq | chunk-단위 전환 | 추가 <3%p |
  | latency overhead | — | freq settle (chunk 흡수) | <6% |

- **Tier 강등 risk**: freq settle latency 가 J/chunk 절감의 >30% 잠식 → EMC-only 축소; SparseDVFS triplet 대비 modality-regime-aware 순증분 J/chunk <5%p → S3 producer-data 로 흡수.
- **Outperform 가능성**: GreenLLM([arXiv:2508.16449](https://arxiv.org/abs/2508.16449))/DualScale([arXiv:2602.18755](https://arxiv.org/abs/2602.18755)) phase-DVFS 를 EMC 분리 + diffusion denoise 포함으로 edge 에서 outperform 가능.
- **Implementation envelope**: 표준 Jetson sysfs(최저 risk), AGX Orin 1차, ~10주. S1 의 TowerPhaseFSM hook 재사용.

### ⑤ L3 KEELKV (avg 6.6)

- **(a) mechanism 별 정성 benefit**:
  - M1 per-layer K_AR/V_AR persisting L2 window + video-K_DM streaming demote: 진입 시 `cudaStreamSetAttribute accessPolicyWindow(hitProp=Persisting, base=K_AR)`, video-K_DM 은 Streaming prop → 수만 token video-K_DM 의 K_AR 축출(L2 pollution 2.15×) 방지.
  - M2 SMEM staging fallback: set-aside 미지원 HW 에서 GQA-8 K_AR tile 을 SMEM(192KB/SM) step 1회 stage.
- **(b) closest competitor + 차별 axis**: Async KV Prefetch→L2([arXiv:2504.06319](https://arxiv.org/abs/2504.06319))가 KV 를 L2 로 prefetch(attention 2.15×)하나 decode-time growing KV 대상·prefetch(상주 pin 아님)·discrete GPU. 차별축 = **denoise 전기간 read-only static cross-tower K_AR 을 per-layer L2 set-aside pin + video-K_DM streaming demote** (Orin sm_87 3MB 확정, per-layer policy 1.2MB BF16 / 0.3MB INT4 fit).
- **(c) 예상 gain (보수치)**:

  | 지표 | Baseline | 본 idea | 개선 |
  |---|---|---|---|
  | per-step DRAM read | K_AR DRAM 재독출 | L2-pin (K_AR 비중만큼) | 10-20% 절감 |
  | per-step latency | no L2 pin | pin on | 5-12% 절감 |
  | `l2_tex_hit_rate`(K_AR) | reasoner-caching only | pin (RTX 측정) | hit↑ |

- **Tier 강등 risk**: RTX ncu `l2_tex_hit_rate` 가 pin on/off 유의차 없거나(K_AR 비중<10%) per-step DRAM −10% 미달 → negative-characterization demote; T2V K_AR>3MB(긴 conditioning) → policy-mode-only scope 축소.
- **Outperform 가능성**: vLLM/SGLang/llama.cpp/TRT 전부 L2 set-aside 미사용 → static cross-tower KV pin 은 미점유 영역. T2V video-K_DM 클수록 pollution 회피 이득 큼.
- **Implementation envelope**: ~9주 최견고, L2-hit 인과검증 = RTX Pro 6000(ncu), Orin 은 latency/BW delta 간접. **Q2 INT4 → L3 pin paper-pair**.

### ⑥ S4 SIDEPOOL (avg 6.2, placement-LUT 중심 rescope)

- **(a) mechanism 별 정성 benefit**:
  - M1 GPU stream-priority overlap of VAE ∥ next-chunk denoise: VAE decode(현 chunk)를 낮은 priority CUDA stream + `cudaEvent` 동기로 denoise(다음 chunk, 높은 priority)와 overlap → VAE 가 denoise SM/BW 경합(Edge/Nano 급 비-amortized 지배 비용) 보전. DLA 무관.
  - M2 accelerator-complex placement LUT: VAE 종류·해상도·chunk frame 수별 best placement(iGPU/DLA-2Dsubgraph/PVA/CPU). DLA 는 검증된 2D-subgraph 만 선택 오프로드(3D conv/GroupNorm 은 GPU).
- **(b) closest competitor + 차별 axis**: PipeDiT([arXiv:2511.12056](https://arxiv.org/abs/2511.12056))/DeDiVAE([arXiv:2512.07350](https://arxiv.org/abs/2512.07350))가 "VAE∥denoise pipeline overlap" 핵심 발상을 이미 보유(단 multi-GPU group 분리). 차별축 = **단일 edge GPU stream-priority overlap + iGPU/DLA-2Dsubgraph/PVA/CPU placement LUT** (placement-LUT M2 가 standalone 핵심, M1 단독 주장 금지).
- **(c) 예상 gain (보수치)**:

  | 지표 | Baseline | 본 idea | 개선 |
  |---|---|---|---|
  | e2e chunk latency | all-on-iGPU serial | M1 stream-overlap (VAE critical path 비중만큼) | −15~30% (DLA 무관) |
  | J/chunk | naive serial | DLA 2D-subgraph 오프로드(부분) | −10~20% |
  | denoise GPU occupancy | VAE 경합 | VAE 분리 | 회복 |

- **Tier 강등 risk**: `trtexec` DLA-supported subgraph 비율 step0 실측 <20% → DLA ablation drop, M1 GPU-overlap 만; M1 overlap% 가 PipeDiT-style 대비 우위 없으면 → placement LUT(M2)만 standalone.
- **Outperform 가능성**: 모달리티-aware accelerator placement(G6+G8) 는 미존재 → placement study 로 direct prior art 부재가 강점.
- **Implementation envelope**: ncu 불요 Orin 측정 완결, ~10주. generation-mode 한정(policy 는 video-latent decode skip → VAE 이득 無).

---

## 4. 미선정 아이디어 전체 (9건)

> 상세 + Phase 별 탈락 지점은 [unselected.md](unselected.md).

- **A2 Keystone** (G4 runtime layout + CFG dedup): GAP — vLLM paged KV 가 read-only·step-invariant cross-tower KV layout/dedup 자유도 미활용. 시도 — read-only static-KV layout(무손실) + exact CFG cond/uncond shared-context batching(B=2). 미선정 사유 — M2 "exact DM-self matmul 공유" 가 **layer-recursion 누락으로 layer-1-only(36L 중 1L) 붕괴** → FasterCache([arXiv:2410.19355](https://arxiv.org/abs/2410.19355)) approx-cache 영역과 차별축 소멸 (Phase-2' Task1). 재방문 — per-layer cond/uncond divergence 실측이 작으면 bounded-approx 로 재구성하되 FasterCache 와 정면 비교. (M1 layout 은 S1 에 흡수.)
- **Q5 RELAY** (modality bit/step 비대칭): GAP — policy 에서 video token 이 decode 도 안 되는데 attention 에 잔존, denoise fidelity 의 action 영향 미정량. 시도 — video→action attention perturbation bound `‖Δa‖≤p_v·(L_softmax‖ΔK_v‖+‖ΔV_v‖)`, modality별 연속 precision/step 배분. 미선정 사유 — UD-VLA([arXiv:2511.01718](https://arxiv.org/abs/2511.01718)) "joint denoising > decoupled" 반증 출현 + p_v 결과 의존 → Tier-1 보류. 재방문 — 1주차 p_v 분포 pilot 에서 attention-mass 집중 확인 시 elective 승격.
- **S3 LEDGERMARK** (edge MoT 특성화 producer-letter): GAP — diffusion+AR 을 한 요청이 같은 edge GPU 에서 직렬 traverse 하는 omnimodal MoT 통합 특성화 공백. 시도 — per-phase J/chunk + co-residency 간섭 + phase-transition stall 통합 ledger (2-tier 측정). 미선정 사유 — 측정 방법론을 각 selected idea 의 preliminary study 로 흡수(novelty 상한 낮음). 재방문 — Thor 입수 + ncu-on-Thor 확인 시 IISWC/ISPASS letter.
- **A3 Cascade** (chunk-pipeline scheduler): GAP — chunk 주기 2.1s 내 AR(compute)→DM(memory) 직렬 idle. 시도 — Green Context SM-split intra-GPU pipeline. 미선정 사유 — standalone novelty concurrent ~55%(Bullet/Nexus/DuetServe PD-multiplexing) → S1-M3 으로 흡수. 재방문 — (S1 의 M3 로 생존, standalone 부활 조건 없음.)
- **A4+L4** (S3 통합 전): GAP — phase-transition stall + energy/interference 측정. 시도 — Compass(transition) + Ledgerline(energy) 별도. 미선정 사유 — 4 reviewer 만장일치 "S3 로 통합" 후 S3 자체가 미선정. 재방문 — S3 와 동일.
- **Q4 SIEVE** (measurement-only divergence study): GAP — co-init 두 tower 의 sensitivity 발산 측정. 시도 — ρ_ℓ/κ_ℓ/δ_ℓ 측정 letter. 미선정 사유 — Q1-M1 과 측정핵심 100% 중복(자가중복) → Q1 motivation 으로 흡수. 재방문 — Q1 분리 필요 시 IEEE CAL 단독 spinoff.
- **A5 Herald** (cross-tower DM step-skip gate): GAP — AR semantic 변화 신호로 DM step skip 예측. 시도 — K_AR 상수화 인지 cross-tower skip gate. 미선정 사유 — **DROP**: DISK([arXiv:2602.00440](https://arxiv.org/abs/2602.00440))가 cross-modal skip 핵심 발상 concurrent 55-60% 선점. 재방문 — AR(non-diffusion) tower semantic 변화가 DM step-skip 정확도를 >10%p 개선 ablation 증명 시.
- **A6 Switchback** (VLM↔생성 tower-complementary multiplex): GAP — G2 의 tower-disjoint 단일-device multiplex 미활용. 시도 — Green Context 동시 dispatch tower co-scheduling. 미선정 사유 — **DROP-PARK**: GenServe([arXiv:2604.04335](https://arxiv.org/abs/2604.04335)) concurrent diffusion co-serve 선점 + Green Context Orin 미가용. 재방문 — MPS-multiplex 재구현 + edge-server 멀티테넌트 workload 명시 + GenServe baseline 정면 비교 시.
- **Q3 PRISM** (CFG attention partition-decomposition 공유): GAP — Eq.8 단일 softmax 에서 DM-DM self 성분 cond/uncond 공유. 시도 — log-sum-exp partition-merge exact 공유 custom kernel. 미선정 사유 — **DROP**: exact 공유가 layer-1-only + FasterCache([arXiv:2410.19355](https://arxiv.org/abs/2410.19355)) 영역 + custom kernel overhead. 재방문 — (exact-sharing 자산은 A2-②로 흡수, standalone 부활 조건 없음.)

---

## 5. Essential Reading List (5편)

본 세션 idea 구현 전 반드시 읽어야 할 최강 경쟁자 + baseline:

| Paper | 왜 읽어야 하는가 | baseline idea |
|---|---|---|
| [MotuBrain (arXiv:2604.27792)](https://arxiv.org/abs/2604.27792) | **🔴 전체 세션 최강 경쟁자 (~70% overlap)**. three-stream MoT world-action 이 거의 동일 클래스 + 거의 동일 추론 메뉴(action-only suffix=video skip, DiT cache, FP8, chunked closed-loop)로 50× speedup/11Hz 달성. 단 **edge(Jetson) UMA·offload·DVFS 미다룸** — 이 지점이 본 세션 전체의 유일한 숨구멍. | Q5/S1/전 idea 의 차별화 anchor |
| [vLLM-Omni (arXiv:2602.02204)](https://arxiv.org/abs/2602.02204) | **S1 primary baseline**. any-to-any 멀티모달을 stage graph 로 fully-disaggregate(per-stage batching/GPU 할당, denoising caching, RingAttention CP). 단 **multi-GPU 클러스터 전제** → 단일 Jetson tower disaggregation 으로 축소가 명확한 edge gap. | S1, A2 (baseline) |
| [QuantKeys (arXiv:2605.26266)](https://arxiv.org/abs/2605.26266) | **Q2 closest**. "Quantized Keys Steal Attention" — video diffusion KV quant + Jensen-bias 보정. 단 autoregressive self-forcing growing-KV + 순수 실증, **N-의존 closed-form bound 부재** → Q2 의 static-prefix N-bound 차별축. | Q2 (baseline) |
| [SparseDVFS (arXiv:2603.21908)](https://arxiv.org/abs/2603.21908) | **S2 closest**. CPU/GPU/EMC freq triplet + discrete-edge + intra-inference granularity (EMC축 선점). 단 **operator-sparsity 기반**이지 AR↔DM modality-regime 전환 아님 → S2 의 dual-regime 순증분이 생존축. | S2 (baseline) |
| [VLA-across-XPUs (arXiv:2604.24447)](https://arxiv.org/abs/2604.24447) | **특성화 경쟁 (~55% overlap)**. "compute-bound VLM backbone → memory-bound Action Expert" 2-phase + edge NPU 6× speedup + leaderboard. 단 dual-**tower** param-separation·CFG·VAE·world-model 모드 미커버 → S1-M3/S3 의 정면 대조 의무. | S1(M3 baseline), S3 |

---

## 6. Implementation-Priority Decision Tree (R14.4)

### 6.0 학생 구현 우선순위 결정 트리 (SVG, compact)

> ⚠️ 이 번들은 superseded. 최신 정량 임계값·격상된 dependency 는 [deep 번들 §6](/research-wiki/2026-06/cosmos3-edge-serving-deep) 참조. 아래는 본 번들 시점의 compact 결정 트리.

<svg viewBox="0 0 960 640" style="width:100%;max-width:960px;height:auto" xmlns="http://www.w3.org/2000/svg" font-family="system-ui, -apple-system, Segoe UI, Roboto, sans-serif">
  <defs>
    <marker id="ar4" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M0,0 L10,5 L0,10 z" fill="#1E2761"/>
    </marker>
  </defs>
  <rect x="0" y="0" width="960" height="640" fill="#ffffff"/>
  <rect x="270" y="16" width="420" height="60" rx="8" fill="#CADCFC" stroke="#1E2761" stroke-width="2"/>
  <text x="480" y="42" text-anchor="middle" font-size="14" font-weight="700" fill="#1E2761">W0-2 공통 인프라 (구 S3 측정 방법론)</text>
  <text x="480" y="63" text-anchor="middle" font-size="12" fill="#1E2761">Cosmos3 구동 + tegrastats/Nsight harness · J/chunk 재정의</text>
  <line x1="480" y1="76" x2="480" y2="104" stroke="#1E2761" stroke-width="2" marker-end="url(#ar4)"/>
  <polygon points="240,106 400,166 240,226 80,166" fill="#F96167" stroke="#1E2761" stroke-width="2"/>
  <text x="240" y="152" text-anchor="middle" font-size="13" font-weight="700" fill="#ffffff">Q2 ANCHOR (먼저)</text>
  <text x="240" y="172" text-anchor="middle" font-size="11.5" fill="#ffffff">bound-tightness @INT4</text>
  <text x="240" y="190" text-anchor="middle" font-size="11.5" fill="#ffffff">≤3× PASS / &gt;10× FAIL</text>
  <line x1="240" y1="226" x2="240" y2="262" stroke="#1E2761" stroke-width="2.5" marker-end="url(#ar4)"/>
  <text x="370" y="248" text-anchor="middle" font-size="11" font-weight="700" fill="#1E2761">INT4 K_AR (paper-pair)</text>
  <polygon points="240,262 400,322 240,382 80,322" fill="#F96167" stroke="#1E2761" stroke-width="2"/>
  <text x="240" y="308" text-anchor="middle" font-size="13" font-weight="700" fill="#ffffff">L3 KEELKV</text>
  <text x="240" y="328" text-anchor="middle" font-size="11.5" fill="#ffffff">RTX ncu l2_tex_hit</text>
  <text x="240" y="346" text-anchor="middle" font-size="11.5" fill="#ffffff">DRAM −10~20% PASS</text>
  <polygon points="700,106 860,166 700,226 540,166" fill="#F96167" stroke="#1E2761" stroke-width="2"/>
  <text x="700" y="152" text-anchor="middle" font-size="13" font-weight="700" fill="#ffffff">S1 TIDELOOM (14주)</text>
  <text x="700" y="172" text-anchor="middle" font-size="11.5" fill="#ffffff">staging-hiding gate</text>
  <text x="700" y="190" text-anchor="middle" font-size="11.5" fill="#ffffff">&gt;70% PASS / &lt;70% mini</text>
  <line x1="480" y1="104" x2="240" y2="106" stroke="#1E2761" stroke-width="2" marker-end="url(#ar4)"/>
  <line x1="480" y1="104" x2="700" y2="106" stroke="#1E2761" stroke-width="2" marker-end="url(#ar4)"/>
  <polygon points="700,262 860,322 700,382 540,322" fill="#F96167" stroke="#1E2761" stroke-width="2"/>
  <text x="700" y="308" text-anchor="middle" font-size="13" font-weight="700" fill="#ffffff">Q1 DRIFT</text>
  <text x="700" y="328" text-anchor="middle" font-size="11.5" fill="#ffffff">ρ_ℓ divergence</text>
  <text x="700" y="346" text-anchor="middle" font-size="11.5" fill="#ffffff">≥1.5× PASS / &lt;1.5× CAL</text>
  <line x1="700" y1="226" x2="700" y2="262" stroke="#1E2761" stroke-width="2" marker-end="url(#ar4)"/>
  <line x1="480" y1="392" x2="480" y2="424" stroke="#1E2761" stroke-width="2" marker-end="url(#ar4)"/>
  <text x="480" y="410" text-anchor="middle" font-size="12" font-weight="700" fill="#1E2761">공통 harness 소비 (측정-중심 Tier-2)</text>
  <polygon points="300,426 460,486 300,546 140,486" fill="#F96167" stroke="#1E2761" stroke-width="2"/>
  <text x="300" y="472" text-anchor="middle" font-size="13" font-weight="700" fill="#ffffff">S2 DUOCLOCK</text>
  <text x="300" y="492" text-anchor="middle" font-size="11.5" fill="#ffffff">SparseDVFS 순증분</text>
  <text x="300" y="510" text-anchor="middle" font-size="11.5" fill="#ffffff">≥5%p PASS / settle&gt;step</text>
  <polygon points="660,426 820,486 660,546 500,486" fill="#F96167" stroke="#1E2761" stroke-width="2"/>
  <text x="660" y="472" text-anchor="middle" font-size="13" font-weight="700" fill="#ffffff">S4 SIDEPOOL</text>
  <text x="660" y="492" text-anchor="middle" font-size="11.5" fill="#ffffff">trtexec DLA subgraph</text>
  <text x="660" y="510" text-anchor="middle" font-size="11.5" fill="#ffffff">≥20% PASS / &lt;20% M1만</text>
  <line x1="480" y1="424" x2="300" y2="426" stroke="#1E2761" stroke-width="2" marker-end="url(#ar4)"/>
  <line x1="480" y1="424" x2="660" y2="426" stroke="#1E2761" stroke-width="2" marker-end="url(#ar4)"/>
  <rect x="40" y="566" width="880" height="60" rx="8" fill="#F9E795" stroke="#1E2761" stroke-width="1.5"/>
  <text x="60" y="588" font-size="11.5" fill="#1E2761">Q2→L3 (paper-pair): Q2 INT4 K_AR 산출 → L3 per-layer L2 set-aside pin 여유. · Q2 INT4 → S1 시너지 (double-buffer 여유↑).</text>
  <text x="60" y="610" font-size="11.5" fill="#1E2761">decision gate = CORAL diamond · process = ICE box · callout = GOLD · 임계값 Phase-1' 작성 + Phase-2' 패널 co-sign.</text>
</svg>

### 6.1 양식 A — ASCII flowchart

```
[주차 0-2: 공통 인프라 (모든 idea 의존, 구 S3 LEDGERMARK 측정 방법론)]
  │
  ├─→ Cosmos3 reference path (Diffusers Cosmos3OmniPipeline) 구동
  │     ├─ Nano 16B BF16 RTX Pro 6000 fit? → Yes: 진행
  │     └─ No / edge OOM → BAGEL-7B-MoT (14B/7B-active, vLLM 미지원→Diffusers) fallback
  │
  ├─→ 측정 harness: tegrastats(EMC%/전력 20-30Hz) + Nsight Systems(range marker) + wall-clock
  │     │   (ncu Orin sm_87 미지원 → 커널카운터는 RTX Pro 6000/Thor 에서만)
  │     ├─ J/step 직접? → No (33-50ms<step) → J/chunk(policy 2.1s)·J/phase 재정의
  │     └─ harness 검증 OK → idea 분기
  │
  ▼
[Q2 ANCHOR — 먼저 (RTX Pro 6000 calibration, edge 의존성 없음)]
  │   bound-tightness ratio @INT4
  │     ├─ ≤3× → PASS (Tier-1, bit 자동선택 + INT4 산출물 → L3 공급)
  │     ├─ 3-5× → BELOW (empirical-tightening fallback)
  │     ├─ >10× → CRITICAL-FAIL (bound 없는 static-KV quant 로 demote)
  │     └─ N=4 vs N=50 bound-tightness 예측대로 → OUTPERFORM (N-의존성 실증)
  │   │
  │   └─[edge: Q2 INT4 K_AR] ──┐ (dependency edge)
  ▼                            │
[L3 KEELKV — Q2 산출물 소비 (paper-pair)]
  │   RTX ncu l2_tex_hit_rate pin on/off + per-step DRAM
  │     ├─ DRAM −10~20% → PASS (Tier-2, INT4 K_AR 이 3MB L2 에 4× 여유 pin)
  │     ├─ −10% 미달 → BELOW (BF16 에서도 성립 = pair 보너스 상실)
  │     ├─ K_AR 비중<10% 유의차 없음 → CRITICAL-FAIL (negative-characterization)
  │     └─ T2V K_DM pollution 회피 큼 → OUTPERFORM
  │
  ▼ (Q2 와 병렬 가능)
[S1 TIDELOOM — runtime, 최장 14주]
  │   W1-3 staging-hiding falsification gate (경합 하 hiding%)
  │     ├─ >70% → PASS (Tier-1, M1+M2+M3 full)
  │     ├─ 50-70% → BELOW (M3 stretch goal, M1+M2 core)
  │     ├─ <70% (M2 fail) → CRITICAL-FAIL (S1-mini 로 축소, M2 단독)
  │     └─ RTX ×2→×1 단일화 실증 → OUTPERFORM
  │
  ▼ (독립)
[Q1 DRIFT — 독립 calibration (RTX calib)]
  │   ρ_ℓ divergence across layers
  │     ├─ ≥1.5× across >20% layers → PASS (Tier-1)
  │     ├─ <1.5× across ≥80% layers → BELOW (measurement letter Q4 SIEVE 로 강등)
  │     ├─ 균일-W4 대비 회복 <0.5%p → CRITICAL-FAIL (이득 측정불가)
  │     └─ tower-asymmetric gain material → OUTPERFORM
  │
  ▼ (공통 측정 harness 소비)
[S2 DUOCLOCK / S4 SIDEPOOL — 측정 중심 (S3 방법론 공유)]
  │   S2: SparseDVFS 대비 순증분 J/chunk
  │     ├─ ≥5%p → PASS (Tier-2)  ├─ <5%p → BELOW (S3 흡수)
  │     ├─ settle >30% 잠식 → CRITICAL-FAIL (EMC-only)  └─ on-robot battery → OUTPERFORM
  │   S4: trtexec DLA-supported subgraph 비율
  │     ├─ ≥20% → PASS (Tier-2)  ├─ <20% → BELOW (DLA drop, M1만)
  │     ├─ M1 PipeDiT 대비 우위 없음 → CRITICAL-FAIL (M2 placement LUT 만)  └─ placement 다양성 → OUTPERFORM
```

### 6.2 양식 C — 6 idea × 3 단계 × 4 branch 액션 표

| Idea | 단계 | Pass | Below | Critical-fail | Outperform |
|---|---|---|---|---|---|
| **Q2 ANCHOR** | preliminary (bound 유도) | N-bound 닫힌식 성립 | constant 대체 | bound 발산 → quant-only | — |
| | MVP (bit sweep) | INT4 bound-tightness ≤3× → PASS | 3-5× empirical-tighten | >10× bit-selection 무용 | N=4 tight 입증 |
| | full (RoboLab/VBench) | <1% 품질손실 | INT3 만 | action-MSE >2% | INT2 bound-favorable |
| **Q1 DRIFT** | preliminary (ρ_ℓ 측정) | ρ_ℓ≥1.5× material | <1.5% across 80% → CAL letter | — | (δ_ℓ,ρ_ℓ) 인과 확인 |
| | MVP (bit-alloc) | 품질 0.5-1.5%p 회복 | footprint만 절감 | 회복 <0.5%p 측정불가 | tower-asymmetric 우월 |
| | full (cross-model) | BAGEL 일반화 | Cosmos 한정 | — | "법칙" 격상 |
| **S1 TIDELOOM** | preliminary (W1-3 gate) | hiding%>70 | 50-70% M3 stretch | <70% M2 재설계 | — |
| | MVP (M1+M2 runtime) | peak −35~45% | −20~35% | OOM 잔존 | ×2→×1 단일화 |
| | full (M3 overlap+baseline) | chunk −15~25% | M3 간섭 상쇄 후만 | 간섭이 이득 상쇄 | VLA-XPU outperform |
| **S2 DUOCLOCK** | preliminary (J/chunk LUT) | phase별 곡선 산출 | J/phase 만 | settle>step | — |
| | MVP (governor) | J/chunk −25~38% | EMC-only −15% | settle >30% 잠식 | — |
| | full (vs SparseDVFS) | 순증분 ≥5%p | <5%p → S3 흡수 | 순증분 0 | on-robot battery |
| **L3 KEELKV** | preliminary (fit 수학) | per-layer 3MB fit | INT4 만 fit | T2V>3MB scope 축소 | — |
| | MVP (L2 pin) | DRAM −10~20% | −5~10% | 유의차 없음 negative | T2V pollution 회피 |
| | full (Q2 pair) | INT4 4× 여유 pin | BF16-only(보너스 상실) | — | BW+SRAM 동시 절감 |
| **S4 SIDEPOOL** | preliminary (trtexec) | DLA subgraph ≥20% | <20% DLA drop | — | — |
| | MVP (M1 overlap) | chunk −15~30% | overlap% 낮음 | PipeDiT 대비 무우위 → M2만 | — |
| | full (placement LUT) | LUT 다양성 | iGPU-only | — | accelerator-complex 최초 |

### 6.3 inter-idea dependency edge

- **Q2 → L3** (paper-pair): Q2 의 INT4 K_AR 산출물이 L3 의 per-layer L2 set-aside pin 을 3MB 에 4× 여유롭게 만드는 producer-consumer (BW·SRAM 동시 절감, G4 ownership table 에서 계층 비충돌 확정 — Q2=numeric, L3=placement).
- **공통 측정 harness → S2/S4**: 주차 0-2 의 tegrastats + Nsight Systems 측정 harness(구 S3 방법론)를 S2(J/chunk LUT)·S4(VAE placement latency/J)가 공유.
- **Q2 INT4 → S1**: Q2 의 저비트 K_AR 이 S1 의 active-tower residency 압축과 시너지(KV footprint↓ → double-buffer staging 여유↑).
- **S1 TowerPhaseFSM hook → S2/S3**: S1 의 phase boundary hook 이 S2 phase boundary callback + S3 timestamp 동기로 재사용.

### 6.4 2-agent 검증 trace

- 분기 임계값은 **Phase-1' 전문가(systems / algorithm)가 작성**, **Phase-2' 통합 패널이 co-sign**.
- co-sign reservation: Phase-2' Task5 가 6 idea 임계값 전부 방향 동의 + 5건 강화 — (Q2) N-의존성 falsify gate 를 1차 gate 로 추가, (Q1) 이득-측정가능성 2차 gate(회복<0.5%p), (S1) gate 를 "hiding 여부"가 아니라 **경합 하 hiding%>70** 으로 정량화, (S2) SparseDVFS 순증분 gate, (S4) M1→M2 standalone 전환.
