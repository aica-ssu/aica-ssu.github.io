# 미선정 아이디어 (9건)

각 아이디어: GAP / 시도 overview / 미선정 사유 / 재방문 조건 (R10-α.3 4-요소) + Phase 별 탈락 지점.

---

## A2 Keystone — Static Cross-Attention Context Caching for Diffusion Towers

- **GAP**: vLLM paged KV 는 decode-time KV(매 step 성장·R/W) 가정 → read-only·step-invariant cross-tower KV 의 layout/dedup 자유도 미활용. CFG-parallel 은 2-GPU 전제(G3 단일-device 무대응).
- **시도 overview**: read-only static-KV layout(quant off 무손실) + CFG cond/uncond exact shared-context batching(B=2, DM-self partial-softmax 통계 cond/uncond byte-identical 공유) + modality-asymmetric KV tiering.
- **미선정 사유**: M2 "exact n_DM×n_DM matmul 1회 공유, zero quality change" headline 이 **layer-recursion 누락으로 layer-1-only(36L 중 1L=2.8%) 붕괴** — layer 1 의 O_DM 가 K_AR 분기를 흡수해 ℓ≥2 의 h_DM 가 cond/uncond 발산 → DM-self matmul 분기별로 달라 공유 불가. exact 주장 철회 시 FasterCache([arXiv:2410.19355](https://arxiv.org/abs/2410.19355)) approx-cache 영역과 차별축 소멸.
- **재방문 조건**: per-layer cond/uncond DM-self divergence ‖Δh_DM^(ℓ)‖ 실측이 작으면(초기 layer ε 작음) bounded-approx sharing 으로 재구성하되 FasterCache 와 정면 비교 필요. (M1 read-only static-KV layout 은 S1-M1 의 K_AR read-only pin 으로 흡수.)
- **Phase 탈락**: **Phase-2' Task1** (CRITICAL MATH AUDIT 에서 M2 EXACT-LAYER-1-ONLY verdict → nov 7→6, diff 7→6 강등 → Top-M 제외).

## Q5 RELAY — Modality-Asymmetric Precision/Step Allocation via Video→Action Information-Flow Bound

- **GAP**: policy 에서 video token 은 decode 도 안 되는데 attention 에 잔존(§4.2.5) — action head 가 Eq.8 의 K_DM 중 video 부분으로만 읽음. video token 의 denoise fidelity(bits/steps)가 action 정확도에 얼마나 필요한지 미정량.
- **시도 overview**: video→action attention perturbation bound `‖Δa‖≤p_v·(L_softmax‖ΔK_v‖+‖ΔV_v‖)` (p_v=action query 의 video attention mass) + modality ILP + p_v-guided video token prune(보조).
- **미선정 사유**: UD-VLA([arXiv:2511.01718](https://arxiv.org/abs/2511.01718))가 "joint denoising > decoupled" 실증 → RELAY 의 modality-asymmetric 절감 전제와 긴장 (p_v 큰 경우 negative). VLA joint-denoise 분야 혼잡(DiT4DiT/DUST). Tier-1 진입 보류, Q2 bound 의 자연 확장(modality축)이라 Q2 채택 시 spinoff 흡수 가능.
- **재방문 조건**: 1주차 p_v 분포 pilot 에서 attention-mass 집중(p_v 작음) 확인 시 elective 승격 (bound + RoboLab gain 둘 다 있으면 Tier-1, 아니면 strong Tier-2 DATE/ICRA-workshop).
- **Phase 탈락**: **Phase-2' Task2/Task3** (UD-VLA joint-우월 반증 + p_v 결과 의존으로 Tier-1 보류, avg 7.0 이나 Top-6 제외 — Tier-1 에 넣으면 algorithm 4 vs systems 1 불균형, S1 유지가 다양성 우월).

## S3 LEDGERMARK — 첫 edge MoT dual-regime 특성화 (phase전환/co-residency/J-chunk)

- **GAP**: G1+G2 측정 부재 — EdgeReasoning(AR-only)·Generative-AI-Beyond-LLMs(diffusion-only)·MLPerf v5.1 가 각각 한쪽만 커버. diffusion+AR 을 한 요청이 같은 edge GPU 에서 직렬 traverse 하는 omnimodal MoT 통합 특성화 공백.
- **시도 overview**: per-phase J/chunk + J/inference-phase 에너지 분해 + two-tower co-residency 간섭 profiling(2-tier: Orin timeline / RTX ncu) + phase-transition stall + modality working-set 비대칭 통합 ledger.
- **미선정 사유**: 측정 방법론을 각 selected idea 의 preliminary study 로 흡수(measurement letter 라 novelty 상한 낮음, nov 5/10). VLA-XPU([arXiv:2604.24447](https://arxiv.org/abs/2604.24447))가 Jetson profiling 일부 점유.
- **재방문 조건**: Thor 입수 + ncu-on-Thor 가용성 확인 시 (MoT dual-tower 특유 co-loc K_DM→K_AR L2 pollution + J/chunk + phase-transition stall + EMC sensitivity 한정) IISWC/ISPASS letter.
- **Phase 탈락**: **Phase-1' systems** (A4+L4 → S3 통합 producer-letter, Tier-2) → **Phase-2'** Top-6 제외 (측정 방법론은 공통 인프라 주차 0-2 로 흡수 — S2/S4 가 J/chunk·간섭 LUT 재사용).

## A3 Cascade — chunk-pipeline scheduler

- **GAP**: chunk 주기 2.1s 내 AR(compute, reasoner)→DM(memory, generator) 직렬 idle. compute-bound AR ∥ memory-bound DM resource-complementary.
- **시도 overview**: Green Context SM-split 으로 chunk t DM denoise ∥ chunk t+1 AR re-prefill intra-GPU pipeline + video-token prune + RTC handoff.
- **미선정 사유**: standalone novelty concurrent ~55% (Bullet[arXiv:2504.19516](https://arxiv.org/abs/2504.19516)/Nexus[arXiv:2507.06608](https://arxiv.org/abs/2507.06608)/DuetServe[arXiv:2511.04791](https://arxiv.org/abs/2511.04791) PD-multiplexing) — "intra-GPU SM-partition" 단독 기여는 scooped. Green Context Orin 미가용.
- **재방문 조건**: (S1-M3 으로 cross-regime overlap 흡수 — standalone 부활 조건 없음. A3-M2 video prune 는 Q5 위임 drop, A3-M3 RTC handoff 는 S1-M3 흡수.)
- **Phase 탈락**: **Phase-1' systems (R-S3 결정)** (A3 standalone drop → S1-M3 으로 흡수, MPS fallback 재작성).

## A4 + L4 — Compass(transition) + Ledgerline(energy/interference)

- **GAP**: phase-transition stall 측정(A4) + phase별 energy/co-location interference 측정(L4) 각각 부재.
- **시도 overview**: A4 = Nsight Compute issue-slot/TC 로 미개척 워크로드 공간 측정 기준점; L4 = phase별 J/step + co-residency 간섭 원장.
- **미선정 사유**: 4 reviewer 만장일치 "둘 중 하나로 통합 — 합치면 더 강함" → S3 LEDGERMARK 로 merge. 그 후 S3 자체가 Top-6 미선정 (측정 letter 상한). A4 의 "ncu issue-slot/TC" 는 ncu Orin 미지원(CF-A)으로 근간 깨짐.
- **재방문 조건**: S3 와 동일 (Thor + ncu-on-Thor 확인 시 통합 letter).
- **Phase 탈락**: **Phase-1' systems** (A4+L4 → S3 merge) → **Phase-2'** (S3 와 함께 Top-6 제외).

## Q4 SIEVE — measurement-only co-init divergence study

- **GAP**: co-init 두 tower 의 quant-sensitivity 발산이 init/구조 교란 없이 측정된 적 없음.
- **시도 overview**: ρ_ℓ/κ_ℓ/δ_ℓ 측정 + (δ_ℓ,ρ_ℓ) 상관 + cross-model(BAGEL) 일반성으로 "법칙" 격상한 순수 측정 letter.
- **미선정 사유**: novelty 5/10 + Q1 DRIFT 의 M1 과 측정핵심 100% 중복(자가중복) → Q1 의 motivation + 제1 contribution 으로 완전 흡수.
- **재방문 조건**: Q1 full paper 제출 시 motivation 섹션으로, 분리 필요 시에만 IEEE CAL 단독 spinoff (해법 bit-alloc 제외 순수 측정, ~6주).
- **Phase 탈락**: **Phase-1' algorithm (Q4 ABSORPTION RECORD)** (Q1-M1 흡수 → Q1 Tier-2 spinoff 로 명시).

## A5 Herald — cross-tower DM step-skip gate

- **GAP**: G3/G4 + cross-tower cache-invalidation 공백 — AR semantic 변화 신호로 DM step skip 예측.
- **시도 overview**: K_AR 정적성 인지 cross-tower DM step-skip gate(Cache-DiT 위 누적) + idle-SM speculative prefetch.
- **미선정 사유**: **DROP**. killing paper = DISK([arXiv:2602.00440](https://arxiv.org/abs/2602.00440)) — 두 coupled diffusion transformer 를 dual-branch controller 의 cross-modal skip decision 으로 조정(training-free, 2× trajectory/1.6× video), "cross-tower/cross-stream skip 신호" 핵심 발상이 concurrent ~55-60% 선점. DiT step-skip 문헌 포화(TeaCache/DPCache). M2(idle-SM)는 Green Context Orin 미가용.
- **재방문 조건**: DISK 가 두 diffusion DiT 간만 다루므로, AR(autoregressive, non-diffusion) tower 의 semantic 변화가 DM step-skip 정확도를 유의하게(>10%p) 개선함을 ablation 으로 증명 시 "AR-conditioned DM skip" 단일 기여로 재제출.
- **Phase 탈락**: **Phase-1' systems (DROP-1)** (DISK scoop → drop, salvage: K_AR 상수화 cache-invalidation 단순화 통찰은 S1-M1 흡수, idle-SM 은 S1-M3 흡수).

## A6 Switchback — VLM↔생성 tower-complementary multiplex

- **GAP**: G2 의 multiplexing 미활용 — VLM(reasoner-only) ↔ 생성(generator-heavy) tower-disjoint 단일-device multiplex.
- **시도 overview**: Green Context 동시 dispatch tower-complementary co-scheduling + mode-aware residency.
- **미선정 사유**: **DROP-PARK**. novelty concurrent ~50-55% (GenServe[arXiv:2604.04335](https://arxiv.org/abs/2604.04335) heterogeneous diffusion co-serve / TetriServe / SCORPIO). "complementary workload SM multiplex" 일반 발상 포화. Green Context Orin 미가용 + 두 tower 동시 상주는 AGX Orin 64GB만(NX 16GB swap thrash). workload(edge 에서 VLM+생성 동시 수요)가 speculative — on-robot 은 보통 policy 단일 모드.
- **재방문 조건**: (1) MPS-based multiplex(Orin 검증 경로) 재구현 + (2) workload 를 edge-server/멀티-테넌트(on-robot 단일모드와 구분) 명시 + (3) GenServe-style preemptive co-scheduler baseline 정면 비교 시 Tier-2 재제출.
- **Phase 탈락**: **Phase-1' systems (DROP/PARK-2)** (GenServe scoop + Green Context Orin 미가용 → drop/park, salvage: mode-aware residency 는 S1-M2 WeightResidencyManager multiplex 확장으로 흡수).

## Q3 PRISM — CFG attention partition-decomposition 공유

- **GAP**: Eq.8 가 [K_AR;K_DM] 단일 softmax 라 cond/uncond DM-DM self-attention 성분(Õ_DM, Z_DM)을 정확히 공유할 수 있는가.
- **시도 overview**: log-sum-exp partition decomposition `O_DM=(Z_AR·Õ_AR+Z_DM·Õ_DM)/(Z_AR+Z_DM)` 으로 비싼 n_DM×n_DM self-attn matmul 1회 공유하는 custom kernel.
- **미선정 사유**: **DROP**. exact 공유가 layer-1-only (Phase-2' Task1 의 A2-M2 붕괴와 동일 — DM-self partial-softmax 통계는 layer 1 만 byte-identical, ℓ≥2 발산) + FasterCache([arXiv:2410.19355](https://arxiv.org/abs/2410.19355))가 uncond-redundancy/캐시 공간 선점 + partition-decomposed custom kernel 이 fused flash-attn 대비 overhead 로 이득 상쇄 위험(~13wk 커널 엔지니어링).
- **재방문 조건**: (exact-sharing 자산은 A2-②로 흡수 — standalone 부활 조건 없음. "DM-self partial-softmax exact 공유" 가 A2 의 CFG dedup 을 수학적으로 정당화하나 A2 자체가 Phase-2' 미선정.)
- **Phase 탈락**: **Phase-1' algorithm (Q3 DISPOSITION RECORD)** (standalone DROP, exact-sharing 자산 → A2-② 흡수) → A2 도 **Phase-2'** 미선정.

---

## 요약: Phase 별 탈락 분포

| Idea | 탈락 Phase | 핵심 사유 |
|---|---|---|
| A2 Keystone | Phase-2' Task1 | M2 exact-CFG layer-1-only 붕괴 |
| Q5 RELAY | Phase-2' Task2/3 | UD-VLA joint-우월 반증 + diversity 균형 |
| S3 LEDGERMARK | Phase-1' merge → Phase-2' | 측정 letter 상한, preliminary study 흡수 |
| A3 Cascade | Phase-1' R-S3 | PD-multiplexing scoop → S1-M3 흡수 |
| A4+L4 | Phase-1' merge | S3 통합 후 미선정 |
| Q4 SIEVE | Phase-1' 흡수 | Q1-M1 자가중복 |
| A5 Herald | Phase-1' DROP-1 | DISK scoop |
| A6 Switchback | Phase-1' DROP/PARK-2 | GenServe scoop + Green Context Orin |
| Q3 PRISM | Phase-1' DROP | layer-1-only + FasterCache + kernel overhead |
