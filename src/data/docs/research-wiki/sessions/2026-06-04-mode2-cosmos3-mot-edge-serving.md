# Session — Cosmos 3 MoT (16B dual-tower) 단일 edge GPU Omnimodal 서빙 Ideation

- **Date**: 2026-06-04
- **Mode**: 2+1 하이브리드 (Mode 2 = local PDF 정독 + Mode 1 = 문장 수준 ideation)
- **사용자 쿼리 (원문 요지)**: "Cosmos3 의 기존 모델 대비 특징을 분석하고, **edge 에서 Nano 급 MoT(Mixture-of-Transformers) 구조를 효율적으로 서빙**하는 ideation 을 해달라. 기존 VLM/LLM 대비 아키텍처 차이 + 다양한 modality 지원 차이에 기반한 최적화 포인트를 도출하라."
- **Source PDF**: *Cosmos 3: Omnimodal World Models for Physical AI* (NVIDIA, 2026-06-01) — `/home/yhgong/claude/research/VLM/cosmos3 omni model technical-report.pdf`
- **참여 expert (3)**: ai-optimization-expert (systems/serving), legacy-system-expert (memory/HW-hierarchy), algorithm-expert (quantization/numerics/theory)
- **리뷰어 (5)**: novelty / differentiation+impact / ai-implementation / arch-system-implementation (Phase 2) + integrated re-reviewer (Phase 2')
- **세션 결과 (1줄)**: 16 idea pool → Phase 1'/2' 정제 후 **Top-6 확정** (Tier-1 Q2 ANCHOR 7.6 / Q1 DRIFT 7.4 / S1 TIDELOOM 7.4 + Tier-2 S2 DUOCLOCK 7.4 / L3 KEELKV 6.6 / S4 SIDEPOOL 6.2), paper pair Q2×L3, net drop 3 (A5/A6/Q3).

> Summary 번들: [summary/2026-06-04-mode2-cosmos3-mot-edge-serving/README.md](../summary/2026-06-04-mode2-cosmos3-mot-edge-serving/README.md)

---

## Section 0 — Executive Summary (R10 의무)

### Top-M 6 (R10-α.2)

#### ① Q2 — ANCHOR (Tier-1, MLSys/ICML 2027, avg 7.6 ★ 최강)
- **전제**: Cosmos3 의 generator tower 는 `O_DM = Attn_full(Q_DM, [K_AR;K_DM], [V_AR;V_DM])` (Eq.8) 로 reasoner 의 K_AR/V_AR 을 **N denoising step × CFG 2 pass 전 기간 read-only static prefix** 로 cross-attend 한다. Reasoner-tower caching (tech report §5.3.2) 으로 K_AR/V_AR 은 이미 1회 계산 후 캐시됨.
- **GAP bullets**:
  - decode-time KV quant(KIVI/KVQuant)은 online·매 토큰 cheap + growing KV 가정 — Cosmos3 K_AR/V_AR 은 정반대(정적·1회·read-only).
  - 정적이라 (i) 비싼 1-shot 양자화(Hadamard rotation + MSE-optimal clipping)를 amortize 가능, (ii) denoising 출력 error 를 KV quant error 의 함수로 **flow-step 수 N 에 명시 의존하는 닫힌 bound** 로 유도 가능 — online 엔 step 개념 자체가 없어 불가능.
  - 이 static-KV 특화 bound 가 문헌에 부재.
- **Overview Mechanism**:
  - M1: Hadamard-rotated 1-shot per-channel K_AR/V_AR quantization (INT4/INT3, bit∈{8,6,4,3,2} sweep).
  - M2: Flow-matching denoising error bound `‖x̂_0−x_0‖ ≤ (Σ_n Δ_n·L_v^{(n)})·(L_softmax·‖ΔK_AR‖ + ‖ΔV_AR‖)` — N 작을수록(policy N=4) bound tight = **policy mode 가 KV 저비트에 더 관대**라는 검증가능 예측.
  - M3: Deploy + bound-validation (L_v power-iteration 측정 → bit-tightness plot → bound-tolerance 로 bit 자동 선택).
- **예상 효과 (정량)**:
  - conditioning-KV footprint **2-4× 압축** (INT4/INT3), 품질손실 <1%.
  - policy N=4 → INT3, bound-favorable 시 INT2 (action-MSE <2%).
  - step KV-read BW 비례 절감.
- **Scoring (5축)**: nov 8 / diff 8 / impact 8 / ai-impl 7 / arch-impl 7 → **7.6**. ★ nov: flow-step-N 의존 closed-form bound = 경쟁 KV-quant(QuantKeys [arXiv:2605.26266], 33-method study [arXiv:2603.27469]) 전부 부재(WebFetch 확인). ▼ ai-impl: BF16-only fork + L_v jacobian calib ~13wk.
- **Tier-2 variant**: `policy-mode 4-step static-KV INT4 + measured action-MSE budget` — DATE/IEEE-CAL letter, bound 의 N-의존 예측을 RoboLab 에서 실증.

#### ② Q1 — DRIFT (Tier-1, NeurIPS/ICML/MLSys, avg 7.4)
- **전제**: 두 tower 는 동일 Qwen3-VL-8B 에서 co-init (tech report line 933) 되나 서로 다른 objective(AR next-token CE vs rectified-flow velocity MSE)로 학습되며 **양자화 민감도(loss curvature)가 비대칭 발산**한다.
- **GAP bullets**:
  - 이 발산을 *교란변수 통제(init·구조 고정) 하* 측정한 연구 부재.
  - 단일 UMA 메모리 예산 하 per-tower bit allocation 부재.
  - LLM(AR) outlier 민감 vs DiT(generator) W4A4 내성 — 두 정설이 같은 모델 두 tower 에 공존하는 첫 사례.
- **Overview Mechanism**:
  - M1: ρ_ℓ divergence 측정 study (Q4 흡수) — `S_ℓ^τ = Tr(H_ℓ^τ)·‖ΔW_ℓ^τ‖_F²` (Hutchinson HVP), **divergence ratio ρ_ℓ = S_ℓ^R/S_ℓ^G** (null: co-init 이라 ρ_ℓ≈1) + outlier kurtosis κ + weight-drift δ_ℓ 와 (δ_ℓ,ρ_ℓ) 인과 상관.
  - M2: UMA budget 하 ILP/greedy bit allocation (reasoner=AWQ/GPTQ, generator=SVDQuant low-rank).
  - M3: deploy validation (MMMU/GenEval/RoboLab-120 subset).
- **예상 효과 (정량)**:
  - 동일 예산서 균일-W4 대비 **품질 0.5-1.5%p 회복 또는 footprint 10-18% 절감** (ρ_ℓ 발산 material 할 때만).
  - 발산 작으면 "균일이 충분"이라는 negative-but-publishable 결과 (conditional-gain 정직).
- **Scoring (5축)**: nov 7 / diff 8 / impact 8 / ai-impl 7 / arch-impl 7 → **7.4**. ★ diff: 모든 baseline tower-agnostic, co-init ρ_ℓ divergence 측정각 유일. ▼ nov: sensitivity mixed-precision 기법 자체는 성숙(Mix-QViT [arXiv:2501.06357], KL-Lens [arXiv:2604.13440]).
- **Tier-2 variant**: Q4 SIEVE = measurement-only co-init divergence letter (IEEE CAL/DAC-short), 해법 제외 순수 측정 + cross-model(BAGEL) 일반성 "법칙" 격상.

#### ③ S1 — TIDELOOM (Tier-1, MLSys/ASPLOS, avg 7.4)
- **전제**: 한 요청이 AR(memory-bound, KV-중심) → DM(compute-bound, KV-불변, step 반복)을 단일 edge GPU 에서 직렬 traverse 하나, 현 스택은 vLLM(AR)·vLLM-Omni(DM)가 **분리 엔진**. 16B 중 phase 별 활성은 항상 8B 한 tower. 현 policy 배포가 RTX Pro 6000 ×2 필요 (tech report §4.2.5).
- **GAP bullets**:
  - 분리 엔진은 phase 전환을 한 process·한 메모리 풋프린트로 못 다룸 (G1).
  - 16B 전량 상주 = capacity·BW 낭비, 활성 8B 만 fast-tier 상주 정책 미개척 (G2).
  - MotuBrain/VLA-XPU 는 알고리즘 가속만, weight residency·런타임 통합 부재.
- **Overview Mechanism** (A1+L1+A3 merge, 9→3):
  - M1: Tower-phase FSM (`TowerPhaseFSM`, plain-PyTorch `Cosmos3OmniPipeline` 단일-process 위) — AR_PREFILL→AR_DECODE→DM_DENOISE[k] loop→DECODE_OUT, K_AR zero-copy 핸드오프.
  - M2: Phase-deterministic tower weight residency (active-8B lock, double-buffered `cudaMemcpyAsync` + `cudaStreamAttachMemAsync`, Orin prefetch 미지원 대응).
  - M3: Chunk-pipelined AR∥DM cross-regime overlap (A3 흡수, MPS partition / 2-stream, Green Context=Thor-scope).
- **예상 효과 (정량)**:
  - peak resident **−35~45%** (active-8B + double-buffer, streaming hiding 성공 시).
  - phase-transition zero-copy 로 e2e **−10~18%**.
  - chunk latency **−15~25%** (M3 overlap), **RTX Pro 6000 ×2 → ×1 단일화** 실증.
- **Scoring (5축)**: nov 7 / diff 8 / impact 8 / ai-impl 7 / arch-impl 7 → **7.4**. ★ impact: RTX×2→×1 단일화 + Orin NX fit enabling, TODAY 워크로드. ▼ nov: weight-streaming 일반 메커니즘 포화(FluxMoE [arXiv:2604.02715]/DuoServe-MoE [arXiv:2509.07379]), 조합만 차별.
- **Tier-2 variant**: S1-mini = policy-mode-only active-8B residency on Orin NX 16GB (M2 단독, DATE/ISLPED/IISWC short).

#### ④ S2 — DUOCLOCK (Tier-2, DATE/ISLPED/IISWC energy, avg 7.4)
- **전제**: 한 inference **내부**에서 AR(memory-bound) ↔ DM(compute-bound)가 교대 — 단일 nvpmodel 모드(고정 GPU+EMC freq)는 두 상반 phase 에 동시 최적일 수 없음 (G1).
- **GAP bullets**:
  - 기존 phase-DVFS(DualScale/GreenLLM/SparseDVFS)는 across-request / 동일 AR 모델 prefill-decode — 한 요청 내 AR↔DM tower 전환 freq 비대칭 미개척.
  - 데이터센터 DVFS 에 EMC(memory) clock 분리 개념 자체가 없음.
  - tegrastats 33-50ms < step → J/step 직접 측정 불가 (CF-B).
- **Overview Mechanism**:
  - M1: Phase-transition-triggered decoupled EMC/GPU DVFS governor (AR: EMC max + GPU↓ / DM: GPU max + EMC↓, chunk-granularity 전환).
  - M2: J/chunk + J/inference-phase 특성화 LUT (INA3221 rail 적분, policy forward 263ms 직접 / gen step <50ms N-step 적분).
- **예상 효과 (정량)**:
  - J/chunk **−25~38%** (AR phase GPU-freq↓ + DM phase EMC-freq↓, phase별 독립 가정 명시).
  - SLO 위반 추가 <3%p, latency overhead <6% (freq settle chunk-단위 흡수).
- **Scoring (5축)**: nov 6 / diff 7 / impact 8 / ai-impl 8 / arch-impl 8 → **7.4**. ★ ai-impl: 표준 Jetson sysfs, 최저 risk + arch-impl J/chunk feasible. ▼ nov: SparseDVFS [arXiv:2603.21908] EMC-triplet 축 선점(6 고정).
- **Tier-2 variant**: 이미 Tier-2 (Tier-1→2 강등) — intra-request + UMA EMC/GPU + J/chunk reposition.

#### ⑤ L3 — KEELKV (Tier-2, DAC/DATE/CAL, avg 6.6)
- **전제**: K_AR/V_AR 은 denoise 전 기간 read-only 인데 매 step streaming 되는 video-K_DM(수만 토큰)이 L2 를 점유해 K_AR 축출 → DRAM 재독출 (L2 pollution 2.15×). AGX Orin L2=4MB, set-aside 3MB 확정 지원.
- **GAP bullets**:
  - vLLM/SGLang/llama.cpp/TRT 전부 L2 set-aside 미사용 (Platform-Usage scan).
  - per-layer K_AR(policy ~300 tok, GQA8) = BF16 1.2MB / INT4 0.3MB → **3MB L2 set-aside 에 per-layer fit 확정**.
  - static cross-tower KV 의 L2-pin 은 어느 엔진도 미적용.
- **Overview Mechanism**:
  - M1: per-layer K_AR/V_AR persisting L2 window (`cudaAccessPolicyWindow`) + video-K_DM streaming demote.
  - M2: SMEM staging fallback (192KB/SM, L2 미지원 HW 대비).
- **예상 효과 (정량)**:
  - per-step DRAM read **−10~20%** (K_AR 비중만큼), per-step latency **−5~12%**.
  - T2V K_DM streaming 클수록 pollution 회피 이득 큼.
- **Scoring (5축)**: nov 6 / diff 6 / impact 6 / ai-impl 8 / arch-impl 7 → **6.6**. ★ ai-impl: 3MB L2 set-aside per-layer fit 검증, ~9wk 최견고. ▼ imp: per-step DRAM −10~20%, T2V 한정 단일 metric.
- **Tier-2 variant**: 이미 Tier-2; **Q2(INT4)→L3(L2-pin) paper-pair** (INT4 K_AR 이 3MB 에 4× 여유로 pin).

#### ⑥ S4 — SIDEPOOL (Tier-2, DAC/DATE/ISPASS placement, avg 6.2)
- **전제**: VAE encode/decode 가 denoise 와 같은 iGPU 에서 직렬 실행 → SM·BW 경합. Edge/Nano 급에서 VAE 가 비-amortized 지배 비용 (tech report §3.4/5.2.6). frozen VAE 의 지배 연산(3D causal conv/GroupNorm)이 DLA 미지원.
- **GAP bullets**:
  - vLLM-Omni VAE-Patch-Parallel 은 multi-GPU 전제, LiteVLA-Edge 는 AR backbone only.
  - "frozen VAE→DLA" 단독은 GPU fallback 빈발로 ROI ceiling 낮음 (ai-impl flag).
  - 단일 edge GPU VAE∥denoise stream-overlap 미개척.
- **Overview Mechanism** (L5 rescope, R-S5 option a+b):
  - M1: GPU stream-priority overlap of VAE ∥ next-chunk denoise (DLA 무관, `cudaStreamCreateWithPriority`).
  - M2: Accelerator-complex placement LUT (iGPU/DLA-2Dsubgraph/PVA/CPU, DLA 는 검증된 2D-subgraph 만 선택 오프로드).
- **예상 효과 (정량)**:
  - e2e chunk latency **−15~30%** (M1 GPU stream-overlap, VAE critical path 비중만큼).
  - J/chunk **−10~20%** (DLA 2D-subgraph 오프로드 성공 시), denoise GPU occupancy 회복.
- **Scoring (5축)**: nov 6 / diff 6 / impact 6 / ai-impl 6 / arch-impl 7 → **6.2**. ★ arch-impl: ncu 불요 Orin 측정 완결 + stream-priority 견고. ▼ diff: PipeDiT [arXiv:2511.12056]/DeDiVAE [arXiv:2512.07350] VAE∥denoise overlap 선점(7→6).
- **Tier-2 variant**: 이미 Tier-2; generation-mode scope 만(policy 는 video-latent decode skip → VAE 이득 無).

### 미선정 9건 (R10-α.3)

1. **A2 Keystone** — GAP: read-only static K_AR/V_AR 의 runtime layout/dedup 자유도 (G4). / 시도: read-only layout + CFG cond/uncond exact shared-context batching + modality tiering. / 미선정 사유: **Phase 2' Task1 audit 에서 M2 "exact CFG sharing" = EXACT-LAYER-1-ONLY 로 붕괴** (layer-recursion 누락 → ℓ≥2 발산), FasterCache [arXiv:2410.19355] 영역과 차별축 소멸. / 재방문: M1 layout-only 무손실 기여를 S1-M1 의 K_AR read-only pin feature 로 흡수.
2. **Q5 RELAY** — GAP: policy 에서 video token 이 decode 안 되나 attention 잔존, action 정확도 기여를 정량화 (p_v info-flow bound). / 시도: video→action attention-mass p_v perturbation bound + modality 비대칭 bit/step ILP. / 미선정 사유: **UD-VLA [arXiv:2511.01718] "joint denoising > decoupled" 반증** + VLA joint-denoise 분야 혼잡, p_v 결과 의존(7.0). / 재방문: p_v pilot 측정에서 p_v 가 작게 나오면(video 가 action 에 무관) Q2 bound 의 modality 축 확장 spinoff 로 재방문.
3. **A3 Cascade** — GAP: chunk t DM denoise ∥ chunk t+1 AR re-prefill 단일 edge GPU pipeline (G1/G5/G7). / 시도: Green Context SM-partition cross-regime overlap + video-token prune. / 미선정 사유: standalone novelty concurrent ~55% (Bullet/Nexus PD-multiplexing), cross-regime overlap 만 가치. / 재방문: **→ S1-M3 으로 흡수** (MPS fallback), standalone idea 소멸.
4. **A4 Compass** — GAP: 단일 edge GPU MoT dual-regime phase-transition 측정 공백. / 시도: Nsight 기반 phase-transition profiler letter. / 미선정 사유: L4 와 측정 산출물 정면 중복 + ncu Orin 미지원으로 issue-slot/TC 측정 불가. / 재방문: **→ S3 LEDGERMARK 으로 통합** (transition 관점), 단일 producer-letter.
5. **L4 LEDGERLINE** — GAP: edge MoT J/denoise-step + two-tower co-location 간섭 최초 ledger. / 시도: per-phase 에너지 분해 + co-residency profiling. / 미선정 사유: A4 와 중복 (energy/interference 관점). / 재방문: **→ S3 LEDGERMARK 으로 통합** (S3 가 채택된 producer-letter, S1/S2 motivation).
6. **Q4 SIEVE** — GAP: co-init tower quantization-sensitivity divergence 측정 공백. / 시도: ρ_ℓ/κ/δ 측정-only study. / 미선정 사유: Q1 DRIFT 의 M1 과 측정핵심 100% 자가중복. / 재방문: **→ Q1-M1 흡수** (Q1 full 제출 시 motivation, 분리 필요 시 IEEE CAL spinoff).
7. **A5 Herald** — GAP: G3/G4 + cross-tower cache-invalidation 공백 (AR semantic 변화로 DM step skip 예측). / 시도: K_AR-정적성 인지 cross-tower DM step-skip gate + idle-SM speculative prefetch. / 미선정 사유: **DROP — killing paper DISK [arXiv:2602.00440]** (두 coupled diffusion transformer cross-modal skip, training-free, concurrent ~55-60%). / 재방문: AR(non-diffusion)→DM 비대칭이 step-skip 정확도를 >10%p 개선함을 ablation 으로 증명 시 단일 기여 재제출.
8. **A6 Switchback** — GAP: G2 multiplexing 미활용 (VLM=reasoner-only ↔ 생성=generator-heavy tower-disjoint). / 시도: tower-complementary single-device multiplex. / 미선정 사유: **DROP-PARK — GenServe [arXiv:2604.04335] scoop** + Green Context Orin 미가용 + 두 tower 동시 상주 NX 16GB 불가 + edge VLM+생성 동시 수요 speculative. / 재방문: MPS-based multiplex 재구현 + edge-server/멀티-테넌트 scope + GenServe-style baseline 정면 비교 시.
9. **Q3 PRISM** — GAP: CFG cond/uncond 가 K_AR 만 다름 — attention-level decomposition 으로 single-GPU CFG compute 절감 (G3). / 시도: CFG attention decomposition + guidance-aware uncond caching. / 미선정 사유: **DROP — FasterCache [arXiv:2410.19355] uncond-redundancy 영역 선점** + partition-decomposed custom kernel overhead 위험 (~13wk). exact-sharing 자산은 A2-M2 로 흡수했으나 Task1 에서 layer-1-only 붕괴. / 재방문: 없음 (근사 캐시 영역 red ocean).

> **Net drop 3** (A5 Herald, A6 Switchback, Q3 PRISM) — 나머지 6건(A2/Q5/A3/A4/L4/Q4)은 전부 자산 흡수 또는 통합.

---

## Section 1 — Step 0 요약

### 1.1 Tech Report 분석 핵심 (Mode 2 Primary Source)
- **Cosmos 3 정체**: 옴니모달(omnimodal) world model 패밀리 — language/image/video/audio/**action** 을 단일 MoT 아키텍처에서 공동 처리/생성. VLM + T2I/T2V/I2V/V2V + audio-visual + world simulator + world-action(policy/forward/inverse dynamics)를 단일 모델 통합.
- **MoT Dual-Tower (핵심)**: 각 decoder layer 가 두 세트 독립 파라미터 보유 — **Reasoner tower**(AR subsequence, language+vision token, next-token) + **Generator tower**(DM subsequence, VAE latent+audio+action, flow-matching denoising). **공유 지점은 attention 연산 하나뿐** (dual-stream joint attention). 두 tower 모두 Qwen3-VL 8B/32B 로 co-init.
  - `O_AR = Attn_causal(Q_AR, K_AR, V_AR)` (AR 은 DM 을 절대 안 봄), `O_DM = Attn_full(Q_DM, [K_AR;K_DM], [V_AR;V_DM])` (DM 은 AR context+DM 전체 bidirectional).
- **Variants (Tab.2)**: Cosmos3-Edge 4B(2B/tower, 미공개), **Cosmos3-Nano 16B(8B/tower, Qwen3-VL-8B init)**, Cosmos3-Super 64B(32B/tower). GQA KV-heads 8 (전 variant).
- **8 GAPs (G1-G8)**:

| GAP | 핵심 |
|---|---|
| G1 | dual-regime 비대칭 — AR memory-bound·token-by-token / DM compute-bound·KV불변·step반복. vLLM(AR)+vLLM-Omni(DM) 분리 스택이 한 요청 안에 직렬 공존 |
| G2 | 2× 파라미터·1-tower 활성 — 16B 지만 phase 별 활성 8B. 16B 전량 상주는 낭비 |
| G3 | CFG 2× on single GPU — CFG-parallel 은 2-GPU 전제, 단일 edge GPU 는 step 마다 2회 forward |
| G4 | K_AR/V_AR 정적 컨텍스트 — denoising 전 기간 read-only, 정밀도/배치/형식 최적화 자유도 |
| G5 | Policy 실시간 — 4 step × CFG2 = 8 forward/chunk @ 15Hz, video-latent decode skip 하나 attention 잔존 |
| G6 | 모달리티별 토큰 밀도 비대칭 — video VAE(수만) ≫ audio(25 tok/s) ≫ action(~32), audio/action h=w=0 |
| G7 | 절대 시간축(MRoPE+FPS modulation) — chunk 단위 incremental 시 위치 재계산 불필요 |
| G8 | VAE 인코더/디코더 병목 — Edge/Nano 급에서 비-amortized 지배 비용 |

- **서빙 증거**: Nano T2V-720p 1-GPU H100 ~286-297s, T2I-720p H100 4.21/3.44s (Fig.16). Policy-DROID: 4 diffusion step + CFG3 + video-decode skip, **RTX Pro 6000 ×2 배포** (§4.2.5). Reasoner-tower caching = K_AR/V_AR 1회 계산 후 전 step 재사용·품질손실 0 (§5.3.2).

### 1.2 외부 탐색 (3-agent, Mode 1)
- **수집 편수**: 3 search 파일 합 ~50편 — unified AR+diffusion(MoT/Transfusion/BAGEL/Show-o2/MammothModa2), DiT 가속(TeaCache/SVDQuant/AGD/DMD2/ToMA), 서빙(vLLM-Omni/GenServe), VLA edge(MotuBrain/UVA/RTC/FASTER/BitVLA/LiteVLA-Edge/Spec-VLA), workload characterization(IISWC/ISPASS 18편).
- **Scoop 위험 3건**:
  - **MotuBrain [arXiv:2604.27792]** ~70% — three-stream MoT world-action, 추론 메뉴(action-only/DiT-cache/FP8/chunk) 50×/11Hz. 단 edge/Jetson/UMA·tower-aware residency 미커버 = 숨구멍.
  - **vLLM-Omni [arXiv:2602.02204]** ~55-60% — AR+diffusion stage disaggregation, **multi-GPU cluster 전제** → single-Jetson tower 공존/시분할 미해결.
  - **VLA-across-XPUs [arXiv:2604.24447]** ~55% — "compute-bound backbone→memory-bound action-expert" 2-phase 특성화+가속, edge NPU 6×. dual-tower param-separation·CFG·world-model 미커버.
- **R17 Step 0-α 충족**: IISWC ≥3 (A7 LLM-on-CPU IISWC'24 + A8 LLMServingSim IISWC'24 + A10 EdgeReasoning IISWC'25) + ISPASS ≥3 (A11 CPU-GPU Coupled ISPASS'25 + A12 Embodied ISPASS'25 + A13 Generative-AI-Beyond-LLMs ISPASS'24) ✅.

---

## Section 2 — Phase 1 로그 (16 idea pool)

| ID | 이름 | Expert | Tier | 1줄 mechanism |
|---|---|---|---|---|
| A1 | Loom | ai-opt | 1 | 단일 edge GPU tower-phase 상태기계 + active-8B weight streaming + two-tower interleaved CUDA graph |
| A2 | Keystone | ai-opt | 1 | read-only 정적 K_AR/V_AR layout·dedup + CFG cond/uncond context 공유 + modality-asymmetric tiering |
| A3 | Cascade | ai-opt | 1 | chunk t DM denoise ∥ chunk t+1 AR re-prefill, Green Context SM-partition pipeline + video-token prune |
| A4 | Compass | ai-opt | 2 | 단일 edge GPU MoT dual-regime phase-transition·co-residency 측정 letter |
| A5 | Herald | ai-opt | 2 | K_AR-정적성 인지 cross-tower DM step-skip gate + idle-SM speculative prefetch |
| A6 | Switchback | ai-opt | 2 | VLM(reasoner) ↔ 생성(generator) tower-complementary 단일-device multiplex |
| L1 | TIDELOCK | legacy | 1 | phase-deterministic 비활성 tower UMA demote + denoise-slack staging (active-8B lock) |
| L2 | TWOCLOCKS | legacy | 1 | AR(mem-bound)↔DM(compute-bound) phase 별 EMC/GPU freq 분리 DVFS governor |
| L3 | KEELKV | legacy | 2 | static K_AR/V_AR 을 denoise loop 동안 L2 set-aside 에 pin (video-KV streaming demote) |
| L4 | LEDGERLINE | legacy | 2 | edge MoT J/denoise-step + two-tower co-location 간섭 최초 통합 ledger |
| L5 | TIDEPOOL | legacy | 2 | frozen VAE 를 DLA(cuDLA)로 오프로드 + next-chunk denoise overlap (placement LUT) |
| Q1 | DRIFT | algorithm | 1 | co-init 두 tower quant-sensitivity 발산 ρ_ℓ + per-tower 비대칭 bit allocation |
| Q2 | ANCHOR | algorithm | 1 | static K_AR/V_AR 1-shot quant + flow-matching denoising error bound |
| Q3 | PRISM | algorithm | 2 | CFG cond/uncond attention decomposition + guidance-aware uncond caching |
| Q4 | SIEVE | algorithm | 2 | co-init tower quant-sensitivity divergence 측정-only study |
| Q5 | RELAY | algorithm | 1-borderline | video→action info-flow perturbation bound p_v + modality 비대칭 bit/step allocation |

---

## Section 3 — Phase 2 로그 (5-리뷰어)

### 5축 score 표 (16 idea × novelty/diff/impact/ai-impl/arch-impl)

| ID | 이름 | nov | diff | impact | ai-impl | arch-impl |
|---|---|---|---|---|---|---|
| A1 | Loom | 7 | 8 | 8 | 5 | 7 |
| A2 | Keystone | 7 | 7 | 7 | 7 | 7 |
| A3 | Cascade | 6 | 7 | 7 | 5 | 6 |
| A4 | Compass | 5 | 6 | 6 | 7 | 4 |
| A5 | Herald | 5 | 7 | 6 | 6 | 6 |
| A6 | Switchback | 5 | 6 | 6 | 5 | 6 |
| L1 | TIDELOCK | 7 | 7 | 8 | 7 | 8 |
| L2 | TWOCLOCKS | 6 | 7 | 8 | 8 | 8 |
| L3 | KEELKV | 6 | 6 | 6 | 8 | 5 |
| L4 | LEDGERLINE | 5 | 5 | 6 | 7 | 4 |
| L5 | TIDEPOOL | 6 | 7 | 7 | 5 | 7 |
| Q1 | DRIFT | 7 | 8 | 8 | 7 | 7 |
| Q2 | ANCHOR | 8 | 8 | 8 | 7 | 8 |
| Q3 | PRISM | 5 | 6 | 7 | 6 | 6 |
| Q4 | SIEVE | 5 | 5 | 6 | 7 | 8 |
| Q5 | RELAY | 6 | 7 | 8 | 7 | 7 |

### 주요 판정
- **DROP 2**: A5 Herald (killing: DISK [arXiv:2602.00440]), Q4 SIEVE (Q1 자가중복 → 흡수).
- **통합 2쌍**: A1⟷L1 (weight residency 중복), A4⟷L4 (measurement 산출물 중복).
- **REPOSITION 4**: A3 Cascade (Bullet/Nexus PD-multiplex), A6 Switchback (GenServe), L2 TWOCLOCKS (DualScale/GreenLLM/SparseDVFS), Q3 PRISM (FasterCache).
- **G4 3중 중복 분리**: A2(layout)/L3(L2-SRAM)/Q2(quant+bound) 계층 분리.

### 치명 발견 (HW/측정 제약)
- **ncu(Nsight Compute) Orin(ga10b sm_87) 미지원** — `l2_tex_hit_rate`/`dram__bytes_read`/issue-slot/TC util 커널 카운터 측정 불가 → A4/L4/L3/A2/Q2 측정 근간 깨짐. RTX Pro 6000/Thor 로 이전 필요.
- **VAE 지배 연산(3D causal conv/GroupNorm) DLA 비호환** — "frozen VAE→DLA" 전제 fragile (L5 ROI ceiling).
- **Cosmos3-Nano BF16-only** (model card "FP4/FP8/FP16 not officially supported") — Q1/Q2/Q5/A2 양자화는 비공식 정밀도 신규도입(fork 부담).
- **Green Context Orin L4T 가용성 미확인** — NVIDIA staff "isolation/QoS 보장 없음", Thor 헤드라인 기능 → A3/A6/A5-M2 SM-partition 은 MPS fallback gate 필수.
- 부수: INA3221 ~33-50ms sampling < denoise step (J/step 직접 측정 불가); cudaMemPrefetchAsync Orin 미지원(cMA=0); **L2 set-aside ≈3MB 는 Orin 지원 확정** (L3 risk 과대평가였음 — 긍정 정정); BAGEL "×2/vLLM 지원" 부분 hallucination (14B 단일 MoT-experts, vLLM upstream 미지원); 16B Cosmos3-Nano edge 미실증(공개=Cosmos Reason 2B VLM-only).

---

## Section 4 — Phase 1' 로그 (merge/reposition)

### Merge / Reposition 결과 (R12 improve-first, ≤3 mechanism/idea)

| Phase 1 | Phase 1' | 종류 | 비고 |
|---|---|---|---|
| A1(Loom) + L1(TIDELOCK) + A3(Cascade) | **S1 TIDELOOM** (Tier-1) | merge | A1=runtime FSM/CUDA-graph, L1=phase-결정 residency/UMA 제약, A3=cross-regime overlap 흡수(M3) |
| L2(TWOCLOCKS) | **S2 DUOCLOCK** (Tier-1→2) | reposition | intra-request + UMA EMC/GPU 분리 + J/chunk reposition, Tier 강등 |
| A4(Compass) + L4(LEDGERLINE) | **S3 LEDGERMARK** (Tier-2) | merge | A4=transition 관점, L4=energy/interference 관점, 2-tier 측정 |
| L5(TIDEPOOL) | **S4 SIDEPOOL** (Tier-2) | rescope | GPU stream-overlap 핵심 + DLA 2D-subgraph 강등 (R-S5 a+b) |
| Q2 ANCHOR / Q1 DRIFT / Q5 RELAY / A2 Keystone / L3 KEELKV | (refined 유지) | — | algorithm track 정제 |
| A5(Herald) | **DROP** | — | DISK scoop |
| A6(Switchback) | **DROP/PARK** | — | GenServe scoop + Green Context Orin |
| Q3(PRISM) | **DROP** | — | FasterCache + kernel overhead (exact-share 자산은 A2 흡수 시도) |
| Q4(SIEVE) | **→ Q1-M1 흡수** | — | 자가중복 |

### Mechanism diff 요약
- **S1 = 9→3** (A1 3 + L1 3 + A3 3 → M1/M2/M3, delta −6): two-tower CUDA graph remove, NVMe 2nd tier remove, video-token prune→Q5 위임, RTC handoff→M3 흡수.
- **S2 = 2→2** (metric replace: J/step → J/chunk + J/phase, Tier 강등).
- **S3 = 4→3** (A4 1 + L4 3 → M1/M2/M3, delta −1 + 통합): "ncu issue-slot/TC" remove, 2-tier 측정 replace.
- **S4 = 2→2** (M1 replace: DLA 핵심 → GPU stream-priority 핵심, scope add).

### G4 ownership 표 (no double-claiming)

| Idea | G4 소유 (계층) | 핵심 metric | Venue | 결합 시너지 |
|---|---|---|---|---|
| **Q2 ANCHOR** | **numeric** (quant + flow-matching bound) | bound vs 실측 error tightness, KV footprint 압축비 | T1 (MLSys/ICML) | INT4 K_AR → L3 L2-pin 4× 여유 + A2 dedup pool BW 절감 |
| **A2 Keystone** | **runtime layout** + CFG dedup (quant off 무손실) | per-step K_AR traffic, CFG-pair 저장 −50%, 품질 Δ=0 | T1 (MLSys/NeurIPS) | cond/uncond K_AR 1부 dedup → Q2 가 그 1부만 양자화 |
| **L3 KEELKV** | **on-chip L2 placement** (`cudaAccessPolicyWindow` pin) | l2_tex_hit_rate, per-step DRAM read −10~20% | T2 (DAC/DATE) | Q2(INT4)→L3(pin) paper-pair |

> 계층 분리 원칙: Q2=수치(bit/bound), A2=runtime(layout/dedup), L3=HW(L2-SRAM). 세 축이 같은 G4 자원(static K_AR)을 노리되 추상화 계층이 달라 비충돌·상호 producer-consumer.

---

## Section 5 — Phase 2' 로그 (integrated re-review)

### Task1 — A2 exact-CFG audit verdict: **EXACT-LAYER-1-ONLY**
A2-M2 의 "DM-self partial-softmax 통계가 cond/uncond byte-identical → n_DM×n_DM matmul 1회 공유 zero-quality-change" 주장은 **layer-recursion 을 누락**했다. 알고리즘 요약 (5-10줄):
```
cond pass = K_AR(real prompt p), uncond pass = K_AR(null ∅)
초기:  h_DM^(1)[cond] = h_DM^(1)[uncond] = (동일 noisy latent x_t)        ← 입력 동일
Layer 1: Q/K/V_DM^(1) = Proj(h_DM^(1))  → cond/uncond 동일 → DM-stats byte-identical ✅ EXACT
         O_DM^(1) = merge(AR-cross-partial(K_AR), DM-self-partial)  → AR항만 분기 다름
         ⇒ O_DM^(1)[cond] ≠ O_DM^(1)[uncond]   (K_AR(p) ≠ K_AR(∅))
Layer 2: h_DM^(2) = O_DM^(1) + residual  → h_DM^(2)[cond] ≠ h_DM^(2)[uncond]   ← 발산 시작
         Q/K/V_DM^(2) = Proj(h_DM^(2)) → cond/uncond 서로 다름 → DM-stats NOT identical ❌
Layer ℓ≥2: 발산 누적 → DM-self matmul 분기별 상이, 공유 불가
```
- 함의: byte-identical exact 공유는 **layer 1뿐 (≈1/36 = 2.8%)** → 사실상 무이득. "exact n_DM×n_DM matmul 공유" headline 철회. ℓ≥2 는 bounded-approximation → FasterCache approx-cache 영역과 충돌 = 차별축 소멸 (severity HIGH).
- **단**: A2-M1(read-only static-KV layout, 무손실)·M3(modality tiering)·CFG-pair K_AR 저장 −50%(K_AR block 자체 dedup, layer-recursion 무관)는 건재. → A2 nov 7→6, diff 7→6 강등, Top-M 제외 (S1-M1 흡수).

### 신규 scoop 알람 (Phase 2')
- **PipeDiT [arXiv:2511.12056] / DeDiVAE [arXiv:2512.07350]** → S4 SIDEPOOL: "VAE decode ∥ denoise pipeline overlap" 핵심 발상 선점 (multi-GPU group 분리이나 발상 동일). M1 단독 기여 위험 → placement LUT(M2) 1차 기여로 재정위 (concurrent 30→45-55).
- **SparseDVFS [arXiv:2603.21908]** → S2 DUOCLOCK: CPU/GPU/EMC freq triplet + edge + intra-inference granularity 도달 → EMC분리(b)축 선점. Tier-2 강등 정당, (a)intra-request modality-regime + (c)J/chunk 두 축만 유일.
- **UD-VLA [arXiv:2511.01718]** → Q5 RELAY: "joint denoising > decoupled" 실증 → modality-asymmetric 절감 전제와 긴장, p_v 큰 경우 negative 가능성 = Tier 강등 risk.

### 최종 score 표 (9 surviving × 5축, Phase 1' fix 반영)

| ID | 이름 | nov | diff | impact | ai-impl | arch-impl | avg |
|---|---|---|---|---|---|---|---|
| Q2 | ANCHOR | 8 | 8 | 8 | 7 | 7 | **7.6** |
| Q1 | DRIFT | 7 | 8 | 8 | 7 | 7 | **7.4** |
| S1 | TIDELOOM | 7 | 8 | 8 | 7 | 7 | **7.4** |
| S2 | DUOCLOCK | 6 | 7 | 8 | 8 | 8 | **7.4** |
| Q5 | RELAY | 6 | 7 | 8 | 7 | 7 | **7.0** |
| L3 | KEELKV | 6 | 6 | 6 | 8 | 7 | **6.6** |
| A2 | Keystone | 6 | 6 | 7 | 7 | 7 | **6.6** |
| S4 | SIDEPOOL | 6 | 6 | 6 | 6 | 7 | **6.2** |
| S3 | LEDGERMARK | 5 | 6 | 6 | 7 | 6 | **6.0** |

### Reference check (R13.1, WebFetch) — 4/5 verified + Bullet 정정
- ✅ DualScale [arXiv:2602.18755], FluxMoE [arXiv:2604.02715], DISK [arXiv:2602.00440], QuantKeys [arXiv:2605.26266] — 전부 실재·내용 일치.
- 🚩 **Bullet 정정**: Phase 1' 가 "Bullet/Nexus/DuetServe [arXiv:2507.06608]"로 묶어 인용했으나 **2507.06608 = Nexus**. Bullet 실제 = [arXiv:2504.19516](https://arxiv.org/abs/2504.19516) "Boosting LLM Serving through Spatial-Temporal GPU Resource Sharing", DuetServe = [arXiv:2511.04791]. 논거 무손상, 라벨-번호 매핑 오류 (severity MEDIUM, 카메라레디 전 정정).

---

## Section 6 — Phase 1'' (Top-M 확정)

- **Tier-1 Top 3**: Q2 ANCHOR (7.6) · Q1 DRIFT (7.4) · S1 TIDELOOM (7.4).
- **Tier-2 독립 Top 3**: S2 DUOCLOCK (7.4, energy) · L3 KEELKV (6.6, on-chip placement) · S4 SIDEPOOL (6.2, VAE placement).
- **Paper Pair (≤1)**: **Q2 ANCHOR × L3 KEELKV** — Q2 가 K_AR 을 INT4 로 줄이면 L3 의 per-layer L2 set-aside pin 이 3MB 에 4× 여유로 들어가 BW·SRAM 동시 절감. 수학(bound)→배치(L2)의 producer-consumer, G4 ownership table 에서 계층 비충돌 확정.
- **다양성 확인**: Tier-1 셋이 **수학 객체(Q2 bound) / 측정-법칙(Q1 ρ_ℓ) / 시스템 런타임(S1)** 으로 기여 종류 직교. **quant 쏠림 회피** — 3개 중 Q2만 quant 기여(Q1 은 sensitivity-law vehicle, S1 은 quant-orthogonal). Tier-2 셋은 energy(S2)/on-chip BW(L3)/VAE-accelerator(S4)로 자원축 분리.
- **2-agent 분기 검증 trace** (Phase 2' Task5): 6 Top-M 의 kill/demote threshold 전부 방향 동의. 강화 포인트 = (Q2) N-의존성 falsify gate 추가, (Q1) 이득-측정가능성 2차 gate, (S1) 경합-하-hiding%>70 정량화, (S2) SparseDVFS 순증분 J/chunk gate, (S4) M1→M2 standalone 전환. A2 는 Task1 붕괴로 branch 생략 (S1-M1 흡수).

> Summary 번들: [summary/2026-06-04-mode2-cosmos3-mot-edge-serving/README.md](../summary/2026-06-04-mode2-cosmos3-mot-edge-serving/README.md)
