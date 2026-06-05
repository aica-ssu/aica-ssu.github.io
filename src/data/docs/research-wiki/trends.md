# Research Trends

`aica-research-bot` Mode 2 / Mode 3 실행에서 도출된 학회별 또는 분야별 트렌드를 시간 역순으로 기록.

---

### Trend M1 (R72): Document-Depth Discipline 도입 — ideation 산출물의 source-code-anchored 깊이 표준화 — 2026-06-05 cosmos3-edge-serving-deep 세션
- **Period**: 2026-06-05 (harness 규칙 진화)
- **배경**: 직전 2026-06-04 cosmos3 세션 tier 문서가 ~80줄 압축으로 R52.2(코드 anchor)/R70(완료 판정)/pseudo-code/preliminary 누락 → 구현 가능성 검증 불가.
- **내용**: tier 문서를 **350-400줄** 깊이로 표준화 (길이상한 금지·per-document 전담 agent·문서 agent 의 source-code 직접 검증·depth-gate). 적용 결과 **anchor 검증율 98%** (47/48), depth-gate 가 1차 4 FAIL(github 링크 0건) 포착 → 규칙이 실제 결함을 잡아내는 효과 입증.
- **핵심 관찰 (코드-수준 발견이 ideation 정정을 유발)**: 문서 agent 가 repo 를 직접 검증하자 ideation 단계의 가정이 코드로 반증·정정됨 — (1) **serving K_AR text-only** (관측은 GEN tower 주입 → 두 시나리오 병기), (2) **policy CFG ON** (구 문서 일부 OFF 기술 → guidance 3.0 + CFG-parallelism = cond/uncond 2벌). 즉 "ideation→문서화→실측" 직렬 대신 "문서화 단계에서 소스 검증을 당겨" 가정 오류를 조기 차단하는 흐름이 표준화됨.

### Trend T10: Omnimodal MoT(Mixture-of-Transformers) dual/multi-tower 아키텍처의 부상 (2024 H2 - 2026 H1) — 2026-06 cosmos3 세션
- **Period**: 2024 H2 - 2026 H1
- **Backbone**:
  - MoT 원논문 [arXiv:2411.04996](https://arxiv.org/abs/2411.04996) — modality별 FFN/proj/LN 분리 + global attention 공유 (sparse multimodal)
  - [BAGEL (arXiv:2505.14683)](https://arxiv.org/abs/2505.14683) — 7B active/14B total MoT-experts decoder-only
  - **Cosmos 3** (NVIDIA 2026-06-01) — Reasoner AR + Generator diffusion dual-tower, Qwen3-VL co-init, language/image/video/audio/action 5-modality
  - [MotuBrain (arXiv:2604.27792)](https://arxiv.org/abs/2604.27792) — three-stream MoT world-action (video/action/text)
  - π0/GR00T action-expert, [Transfusion (arXiv:2408.11039)](https://arxiv.org/abs/2408.11039), [Show-o2 (arXiv:2506.15564)](https://arxiv.org/abs/2506.15564)
- **Key axis 진화**:
  - ~2024: 단일 transformer 통합(Transfusion/Chameleon, AR+diffusion 한 파라미터)
  - 2024 H2: MoT 패턴 정립 — 파라미터 분리 + attention 만 공유 (간섭 최소화, VLM weight 보존)
  - 2025-2026: 8B×2급 dual-tower 가 omnimodal 사실상 표준 (BAGEL/Cosmos3 동형) + action 을 core modality 로
- **본 세션 white space**: MoT 가 신규 카테고리가 아니라 *기존 MoT 계열의 dual-tower 특화 인스턴스* → 서빙 연구 일반화 가능성 높음. tower 가 modality/기능-disjoint → phase 별 1-tower 만 활성(AR memory-bound vs DM compute-bound).

### Trend T11: AR + Diffusion 이중 체제(dual-regime) 서빙의 분리 스택 문제 (2026) — 2026-06 cosmos3 세션
- **Period**: 2026 H1
- **Backbone**: [vLLM-Omni (arXiv:2602.02204)](https://arxiv.org/abs/2602.02204) (any-to-any stage disaggregation, 2026-02), [GenServe (arXiv:2604.04335)](https://arxiv.org/abs/2604.04335) (이종 diffusion co-serve, 2026-04), [VLA-across-XPUs (arXiv:2604.24447)](https://arxiv.org/abs/2604.24447) (2-phase 특성화+가속, 2026-04)
- **Key axis 진화**:
  - AR(reasoner)은 paged-KV/continuous-batch(vLLM), DM(generator)은 denoising few-step(vLLM-Omni) — 서로 다른 엔진
  - 2026: 서빙 시스템이 "AR+diffusion 혼합 disaggregation" 정조준 (diffusion step 경계 preemptibility 가 공통 무기)
  - **모두 multi-GPU 클러스터 전제** — edge(단일 가속기) 시나리오 미개척
- **본 세션 white space**: cross-GPU disaggregation 을 **단일 edge GPU 내 시분할/weight-streaming/메모리공유**로 내리는 연구 부재 (S1 TIDELOOM). MoT 의 attention-공유(KV 공유) 구조를 stage 경계로 끊지 않고 살리는 cross-tower 최적화 부재.

### Trend T12: Edge 디바이스 세대 교체 — Orin(sm_87) → Thor(Blackwell) (2025-2026) — 2026-06 cosmos3 세션
- **Period**: 2025 H2 - 2026 H1
- **Key axis 진화**:
  - Orin(Ampere sm_87): `concurrentManagedAccess=0`(UVM prefetch 미지원), L2 4MB(set-aside ~3MB), **ncu(Nsight Compute) ga10b 미지원**(커널 카운터 측정 불가), Green Context soft(isolation 비보장), 204.8GB/s LPDDR5
  - Thor(Blackwell, JetPack 7.x): cMA=1(prefetch 지원), native FP4 TE, Green Context 정식, 273GB/s, **TensorRT Edge-LLM(2026-01, NVFP4/EAGLE-3/chunked prefill)**
- **본 세션 함의**: Thor = 16B dual-tower 의 유일한 "정공법" edge 타깃(forward-looking primary), Orin = constrained(INT8/INT4 + active-tower 관리, 측정은 RTX/Thor 이전). 측정 idea(S3)는 ncu Orin 미지원 → 2-tier 측정(Orin=timeline/전력, Thor/RTX=커널 카운터) 필수. Cosmos3-Nano BF16-only(저비트 비공식).

### Trend T13: CFG·step-cache 계열 근사 캐시의 포화 (red ocean, 2024-2026) — 2026-06 cosmos3 세션
- **Period**: 2024 H2 - 2026 H1
- **Backbone**: [FasterCache (arXiv:2410.19355)](https://arxiv.org/abs/2410.19355) (cond/uncond residual redundancy), [TeaCache (arXiv:2411.19108)](https://arxiv.org/abs/2411.19108) (CVPR'25 timestep cache), [AGD (arXiv:2503.07274)](https://arxiv.org/abs/2503.07274) (adapter CFG distill), [DISK (arXiv:2602.00440)](https://arxiv.org/abs/2602.00440) (cross-modal step-skip), Adaptive Guidance/LinearAG, X-Cache/WorldCache
- **Key axis 진화**: training-free step caching + uncond-branch 캐시/skip 이 2024-2026 폭발 → DiT 가속의 red ocean
- **본 세션 함의**: **Q3 PRISM(CFG attention decomposition) DROP** (FasterCache 영역 선점), **A5 Herald(cross-tower step-skip) DROP** (DISK scoop). exact-CFG-sharing 시도(A2-M2)도 Phase 2' Task1 에서 layer-recursion 누락으로 layer-1-only 붕괴 → 근사 캐시 영역으로 환원되어 차별축 소멸. 정적 K_AR *block dedup*(저장 −50%)만 layer-recursion 무관하게 생존.

### Trend T14: 특성화 공백 — MoT dual-regime J/chunk 통합 측정 (2026) — 2026-06 cosmos3 세션
- **Period**: 2026 H1
- **Backbone**: [EdgeReasoning (arXiv:2511.01866)](https://arxiv.org/abs/2511.01866) (IISWC'25, AR-only edge), [Generative-AI-Beyond-LLMs (arXiv:2312.14385)](https://arxiv.org/abs/2312.14385) (ISPASS'24, diffusion-only), MLPerf v5.1 (SDXL+LLM SLO), [VLA-XPU (arXiv:2604.24447)](https://arxiv.org/abs/2604.24447) (server 2-phase)
- **Key axis 진화**: phase 이질성(prefill compute / decode memory)은 정설+프로덕션 인프라(DistServe/Splitwise → Dynamo/vLLM 표준). 모달리티가 에너지 1차 변수(video=image 100×, [arXiv:2601.22076](https://arxiv.org/abs/2601.22076)).
- **본 세션 white space**: **diffusion+AR 을 한 요청이 같은 edge GPU 에서 직렬 traverse 하는 omnimodal MoT 통합 특성화는 부재** (S3 LEDGERMARK). tegrastats 33-50ms < step → J/step 직접 측정 불가 → **J/chunk + J/inference-phase 재정의**가 측정 방법론 기여. MoT-특유(co-loc K_DM→K_AR L2 pollution, phase-transition stall, EMC sensitivity) 한정 "최초 통합 ledger".

---

### Production VLM Scenario Diversity 와 Single-Path Optimization 의 Gap — 2026-05-02 시점

- **Date analyzed**: 2026-05-02
- **Session**: [Landing](/research-wiki/2026-05/vlm-scenario-aware)
- **요약**: vLLM v0.10+ / SGLang v0.4.x 가 production prefix hit 60-85% (agent loop / multi-tenant SaaS / repo QA / long-doc) measured 환경에서도 scenario-agnostic single-path optimization. ECVL-ROUTER ([arXiv:2510.27256](https://arxiv.org/abs/2510.27256), ICLR'26) 가 model-tier routing 으로 등장했으나 single-model serving-stack config dispatch (KV budget / prefix policy / compression rate / hot scenario pinning) 는 미커버. 6 scenario class (image-single / image-multi-turn / video-single / video-multi-turn / document / agent) 의 production diversity 가 lightweight classifier (60M) + scenario-conditional config table 로 cover 가능 axis 식별.
- **시사점**: VLM serving optimization 의 다음 frontier 가 scenario-aware dispatcher + scenario-specific mechanism 결합. Mosaic / Lattice / Bramble paper pair (OSDI + MLSys + OSDI) 가 다음 1-2년 emergence 예상.

### Vision Token Frame-Level Radix Tree — Production Multi-Turn Gap — 2026-05-02 시점

- **Date analyzed**: 2026-05-02
- **요약**: SGLang RadixAttention ([ICLR 2024, arXiv:2312.07104](https://arxiv.org/abs/2312.07104)) 의 LLM token prefix radix tree 가 production multi-turn 의 60-85% hit rate 의 backbone, 그러나 vision token frame-level radix tree 는 미커버. mlx-vlm Issue #832 (Qwen3.5 매 turn 마다 vision tower re-run + full re-prefill) 가 production gap 직접 evidence. VLCache ([arXiv:2512.12977](https://arxiv.org/abs/2512.12977), 2025-12) 가 single-session image-patch reuse 시작 (concurrent), cross-session frame radix tree + privacy boundary 는 다음 emergence axis.
- **시사점**: vLLM PagedAttention + SGLang RadixAttention 의 vision token extension 이 MLSys/OSDI 2027 의 핵심 paper 영역.

### VLM Edge Layer-Wise + Context-Semantic 통합 트렌드 — 2026-04-28 시점 (R57 신규 적용)

- **Date analyzed**: 2026-04-28
- **Session**: [세션](/research-wiki/2026-04/vlm-edge-layerwise-context) / [Summary](summary/2026-04-28-mode1-vlm-edge-layerwise-context/)
- **트렌드 한 줄**: VLM edge inference 의 production deployment 가 단순 KV compression 단일 axis 에서 **layer-wise mixed precision (NVFP4/FP8/INT4 Blackwell native) + Context/Semantic-aware KV management (cross-frame cluster reuse / phase-aware caching) + Power-envelope adaptive DVFS** 3-축 통합으로 부상. RTX 5090 32GB GDDR7 single-GPU 의 large MoE local serving 도 신규 axis.
- **근거 paper (peer-reviewed backbone, R56.2 시스템 65%+)**:
  - **VL-Cache** ([ICLR 2025 arXiv:2410.23317](https://arxiv.org/abs/2410.23317)) — VLM KV cache eviction 의 published baseline
  - **Mooncake** (USENIX FAST 2025 Best Paper) — KVCache disaggregation의 production direction
  - **AttAcc** ([ASPLOS 2024 arXiv:2403.15388](https://arxiv.org/abs/2403.15388)) — KV PIM accelerator
  - **CLONE** ([USENIX ATC 2025 arXiv:2506.02847](https://arxiv.org/abs/2506.02847)) — LLM inference energy DVFS
  - **Sarathi-Serve** ([USENIX OSDI 2024](https://www.usenix.org/conference/osdi24/presentation/agrawal)) — chunked prefill + continuous batching
- **주요 supporting evidence (arxiv preprint)**:
  - **FGMP** ([arXiv:2504.14152](https://arxiv.org/abs/2504.14152)) — fine-grained mixed precision
  - **MicroMix** ([arXiv:2508.02343](https://arxiv.org/abs/2508.02343)) — NVFP4+FP8 layer-wise
  - **ClusterKV** ([arXiv:2412.03213](https://arxiv.org/abs/2412.03213)) — KV cluster centroid
  - **VLCache** ([arXiv:2512.12977](https://arxiv.org/abs/2512.12977)) — visual encoder cache
  - **Spec-LLaVA** ([arXiv:2509.11961](https://arxiv.org/abs/2509.11961)) — VLM speculative

### Single-GPU Large MoE Local Serving 트렌드 — 2026-04-28 시점

- **트렌드 한 줄**: RTX 5090 32GB GDDR7 (Blackwell consumer) 가 Qwen3-VL-30B-A3B (3B activated) MoE 의 local serving 을 가능하게 하는 turning point — cluster GPU 가정 없이 single-GPU 에서 large VLM serving.
- **근거**: RTX 5090 32GB + Qwen3-VL-30B-A3B (3B activated, expert routing 후 ~12 GB working set) + DynaExq ([arXiv:2511.15015](https://arxiv.org/abs/2511.15015)) MoE expert routing
- **세션 idea**: Obelisk (Tier-2 T3)

---

### Edge VLM 의 5-Axis Production Readiness 트렌드 — 2026-04 시점 (R55 적용 + v1+v2 종합)

- **Date analyzed**: 2026-04-27 (v2-r55 통합)
- **Session**: 세션 / [Summary](summary/2026-04-27-mode1-vlm-llm-asym-dual-jetson-v2-r55/)
- **트렌드 한 줄**: Edge VLM serving 의 production deployment 가 단순 [Performance]/[Energy]/[Memory eff.] 단일 axis 만 아닌 **[Power]** (sustained thermal envelope) + **[Robustness]** (adversarial / silent corruption / OOD) + **[Security]** (multi-tenant / visual injection / PII) 7-axis 통합으로 부상. 자동차 / robotics / smart camera 산업의 직접 정합.
- **근거 paper (peer-reviewed backbone)**:
  - **NVIDIA TensorRT Edge-LLM** (2026, Jetson Thor + DRIVE AGX) + **NVIDIA Cosmos Reason2** (2026) — 자동차 / robotics 산업의 5-axis production paths
  - **GPUHammer** ([USENIX Sec 2025 arXiv:2507.08166](https://arxiv.org/abs/2507.08166)) — HW fault attack 의 edge VLM 영향
  - **Visual Prompt Injection** ([CMU 2024 arXiv:2403.09766](https://arxiv.org/abs/2403.09766)) — image-based instruction hijack
  - **Robust ViT** ([CVPR 2024 arXiv:2402.07004](https://arxiv.org/abs/2402.07004)) — adversarial training
  - **Energy-OOD** ([NeurIPS 2020 arXiv:2010.03759](https://arxiv.org/abs/2010.03759)) — OOD baseline
- **본 세션 활용**: VEILSEAL-KV ([Security]), STORMGLASS ([Power]), ROBUSTOKEN ([Robustness]) 3 신규 axis idea — v1 의 [Performance]/[Energy]/[Memory eff.] 3 axis 와 결합하여 7-axis 모두 cover. JETTYSIM 류 simulator infrastructure 자체 contribution idea 는 R55.1 자동 배제 — production-ready idea 만 retain.

### Simulator-Building 자체 Contribution 의 본 harness 배제 — 2026-04-27 R55 등록 (사용자 명시)

- **Date analyzed**: 2026-04-27
- **트렌드 한 줄**: 본 harness 의 ideation scope 를 **명확한 gain/benefit (5-axis)** 으로 한정 — simulator infrastructure / extension / tool / benchmark / dataset 신규 release 자체를 main contribution 으로 하는 idea 자동 배제 (R55.1). ISPASS / IISWC tool track 류 paper 는 별도 venue 진행.
- **근거**: Rule 27 (R55) 신규 등록. 학생 / 사용자 입장에서 measurable gain (성능 / robustness / 에너지 / 전력 / 보안 / 메모리 / 비용) 만 본 harness 의 selected idea 로 진입.
- **본 세션 활용**: v1 의 JETTYSIM (simulator infra paper) 자동 배제 → v2 의 VEILSEAL-KV / STORMGLASS / ROBUSTOKEN 3 신규 axis idea 진입.

### LLM↔VLM Asymmetry 의 system-level 활용 부상 — 2026-04 시점 (VLM edge inference 트렌드)

- **Date analyzed**: 2026-04-27
- **Session**: 세션 / [Summary](summary/2026-04-27-mode1-vlm-llm-asym-dual-jetson/)
- **트렌드 한 줄**: VLM 의 LLM 대비 비대칭성 (prefill TTFT 22.4× / visual KV 305MB vs LLM 18MB 17× / layer-wise visual attention L17-21=24.5% vs L0-7=2.6% 5-10× 비대칭) 이 단순 token pruning (SparseVLM/FastV) 을 넘어 **system-level 자원 분할** (SM partition / L2 carveout / DeepStack injection schedule mapping) 의 axis 로 부상.
- **근거 paper (peer-reviewed backbone)**:
  - **DeepStack** ([NeurIPS 2024, arXiv:2406.04334](https://arxiv.org/abs/2406.04334)) — ViT intermediate output → LLM 여러 layer inject, layer-wise visual context distribution 변형
  - **Qwen3-VL Tech Report** ([arXiv:2511.21631](https://arxiv.org/abs/2511.21631)) — DeepStack integration native + 256K context
  - **SparseVLM** ([ICML 2025, arXiv:2410.04417](https://arxiv.org/abs/2410.04417)) — attention pattern → visual token importance 1-shot 추론
  - **DistServe** ([OSDI 2024, arXiv:2401.09670](https://arxiv.org/abs/2401.09670)) — prefill compute-bound vs decode memory-bound 분리
  - **DuetServe** ([arXiv:2511.04791](https://arxiv.org/abs/2511.04791)) — SM-level partitioning prefill/decode contention 해결
  - **EPDServe** (ICML 2025) — encode-prefill-decode 3-way disaggregation, dual-Jetson disagg cluster scoop trigger
  - 내부 측정 (`/home/yhgong/paper/vlm/VLM_exploration_PIM_260407.pdf`) — Qwen3-VL-4B layer-wise visual attn 24.5% vs 2.6% 직접 측정 + TTFT 22.4×
- **본 세션 활용**: ATRIUM (layer-aware SM partition), BREAKWATER-T (ViT-internal split + DeepStack tap), CASCADE-PREFILL (chunked prefill AI dispatch GPU↔CPU), BIMODAL-MASK-T2 (modality outlier topology) 모두 본 trend 의 직접 mechanism.

### Dual-Jetson edge serving 의 USB-C/Ethernet topology 첫 commercialization — 2026-04 시점

- **Date analyzed**: 2026-04-27
- **Session**: 세션
- **트렌드 한 줄**: NVIDIA TensorRT Edge-LLM (2026, Jetson Thor + DRIVE AGX) + Holoscan SDK + Cosmos Nemotron 의 출시로 dual-Jetson 연결 inference 가 자동차 / robotics 산업의 default 가 됨. NVLink 가 Jetson 에서 미가용 → USB-C 3.2 gen2x2 (20 Gbps) 또는 2.5/10 GbE 가 sole interconnect. EPDServe/DiP-SD (2026-04) 류 distributed pipelined speculative decoding 이 edge 도 cluster 도 아닌 **dual-edge** 환경으로 진출.
- **근거 paper**:
  - **DiP-SD** ([arXiv:2604.20919](https://arxiv.org/abs/2604.20919), 2026-04) — distributed pipelined speculative decoding for **edge devices**
  - **DSD** ([arXiv:2511.21669](https://arxiv.org/abs/2511.21669)) — distributed speculative decoding
  - **NVIDIA TensorRT Edge-LLM blog** (2026)
  - **Distributed VLMs** (Columbia 2025) — cloud-edge collaboration
  - **NVIDIA Holoscan SDK** — sensor-proximity dual-device reference
- **본 세션 활용**: BREAKWATER-T (sensor-proximity dual-{Orin Nano + AGX Orin} ViT-internal split via USB-C 20Gbps) + JETTYSIM (LLMServingSim dual-Jetson UMA + USB-C topology extension first-to-report).

### Phoenix CVE 시대 host-MC RAS 의 부상 — 2026-04 시점 (rowhammer 후속 트렌드)

- **Date analyzed**: 2026-04-26 (PM)
- **Session**: 세션 / [Summary](summary/2026-04-26-mode1-rowhammer-ecc-ras/)
- **트렌드 한 줄**: SK Hynix DDR5 의 in-DRAM mitigation 이 Phoenix CVE-2025-6202 (S&P'26, 2025-09 disclosure) 에서 109초 만에 우회되면서, **host-side memory controller** 가 다시 RAS 의 핵심 위치로 복귀.
- **근거 paper**: McSee ([USENIX Sec'25](https://www.usenix.org/conference/usenixsecurity25/presentation/jattke)) 가 측정한 Intel/AMD MC 의 RFM 명령 미발행 + Phoenix CVE post-disclosure context + ARFM ([arXiv:2501.14328](https://arxiv.org/abs/2501.14328)) workload-aware MC throttle + RogueRFM ([arXiv:2501.06646](https://arxiv.org/abs/2501.06646)) covert channel + TPRAC ([arXiv:2505.10111](https://arxiv.org/abs/2505.10111)) timing channel = **host-MC scheduler 가 4 attack vector 모두 응답해야 한다** 는 합의.
- **본 세션 활용**: RFM-COP idea 의 직접 motivation. 4-pillar 통합 narrative 핵심.

### Datatype-asymmetric ECC 의 commodity 화 (REACH 점유) — 2026-04 시점

- **Date analyzed**: 2026-04-26 (PM)
- **Session**: 세션
- **트렌드 한 줄**: REACH ([arXiv:2512.18152](https://arxiv.org/abs/2512.18152), 2025-12) + Domain-Specific ECC AI ([arXiv:2507.02654](https://arxiv.org/abs/2507.02654), 2025-09) 가 **FP16/BF16 exponent 우선 보호 + tunable importance + controller-managed two-level RS** 의 핵심 contribution 을 점유 — datatype-asymmetric ECC 가 commodity 화. 후속 work 은 (a) MoE expert routing weight 비대칭, (b) INT4/INT8 magnitude bit, (c) speculative decoding draft model 등 niche axis 만 가능.
- **본 세션 활용**: Cluster A (S1/H4/L2) 전체 drop. KV cache (KEYSTONE) 와 cross-policy hot-row (RAMPART) 로 reposition.

### CXL controller integrated RAS engine 의 commercialization — 2026-04 시점

- **Date analyzed**: 2026-04-26 (PM)
- **Session**: 세션
- **트렌드 한 줄**: Microchip SMC2100 (2025 commercial CXL controller) 이 Patrol Scrub Engine + DPA poison list + ECS mailbox + MER ring buffer 통합 RAS engine 을 제공하기 시작 → host CPU mailbox polling 부담 경감. Linux 6.16 EDAC mainline 도 generic RAS Control Feature Driver 제공. 학계의 unique contribution 은 **workload-phase-adaptive** + **app-driven hint** 두 차원.
- **근거 paper**: CXL RAS Whitepaper (CXL Consortium 2023) + Linux 6.16 EDAC ([LWN.net 982190](https://lwn.net/Articles/982190/)) + Melody (ASPLOS'25) + M5 (ASPLOS'25) + Microchip SMC2100 (2025).
- **본 세션 활용**: HARBOR idea (H2+L4 merge) 의 motivation — workload-phase-adaptive scrub rate + PyTorch/vLLM annotation app-hint stack.

---

### KV cache reliability + Modern Memory Standard (LPDDR5x/HBM3/CXL 3.x) — 2026-04 시점 미니 트렌드 (v2)

- **Date analyzed**: 2026-04-26
- **Session**: [링크](/research-wiki/2026-04/kv-ecc-ras-v2)
- **Experts**: system-robustness-expert (primary), legacy-system-expert, ai-optimization-expert
- **Papers analyzed**: 24 paper + 9 workload sources + 4 modern memory standard documents

#### 신호 1 — **CXL 3.x CCI mailbox + ECS 가 KV cache reliability research 의 신규 surface 로 부상 (2026-Q1 이후)**

CXL 3.1 RAS Whitepaper (Aug 2024) + CXL 3.2 Patrol Scrub Control + Linux 6.16 EDAC scrub_subsystem upstream (2025-08) 가 동시 도착하면서 **CXL-attached KV cache (LMCache [arXiv:2510.09665](https://arxiv.org/abs/2510.09665) / TraCT [arXiv:2512.18194](https://arxiv.org/abs/2512.18194) / Beluga [arXiv:2511.20172](https://arxiv.org/abs/2511.20172))** 의 reliability axis 가 갑자기 mature 함. KVCache-in-Wild USENIX ATC'25 ([arXiv:2506.02634](https://arxiv.org/abs/2506.02634)) 가 prefix lifetime hour-scale 측정 — long-lived prefix 와 scrub interval alignment 의 first work 가 가능. 본 세션 P1 PrefixGuard / V1 PrefixGuard-Lite / P3 Quarantine / V3 Quarantine-Mini 모두 이 trend 직접 활용. **3 LMCache/TraCT/Beluga concurrent paper 가 latency/throughput axis 만 — reliability axis 는 향후 1-2 년 first-mover 기회**.

#### 신호 2 — **HBM3 PAT counter 의 KV cache hot block identification 활용 가능성**

HBM3 JESD238B 의 **PAT (Pseudo-channel Activation Timing)** counter 가 hot-row 를 hardware-level 로 식별. PRAC family (MOAT [arXiv:2407.09995](https://arxiv.org/abs/2407.09995) / QPRAC [arXiv:2501.18861](https://arxiv.org/abs/2501.18861) / CnC-PRAC [arXiv:2506.11970](https://arxiv.org/abs/2506.11970)) 가 모두 DRAM-level mitigation 만 — KV-aware 부재. Meta Reliability ([arXiv:2410.21680](https://arxiv.org/abs/2410.21680)) Llama-3 405B 16384 H100 cluster 의 54일 419 failure 중 HBM3 72건 (3hr 당 1건) 측정으로 hot row 가 fault dominant origin 확인. 본 세션 P4 PATroller / P8 ECS-Trace / V4 PATroller-Solo 가 이 trend 직접 활용. **clear (<25%) 분류 — first work 기회 강함**.

#### 신호 3 — **Bit-flip attack 의 KV cache 표적이 LM weight 표적 대비 단가 1/100 (SilentStriker 2024 / Two-Decade-Old Prophecy 2025)**

SilentStriker ([arXiv:2509.06939](https://arxiv.org/abs/2509.06939)) 가 INT8-quant Llama-3.1-8B 에서 50 bits flip 으로 GSM8K 65.7% → 7.6% naturalness 유지. Two-Decade-Old Prophecy ([arXiv:2510.00490](https://arxiv.org/abs/2510.00490)) 가 .gguf quantized LLM 에서 **단 1 bit** flip 으로 73.5% → 0% (31.7s @464.3 flips/s). GenBFA ([arXiv:2411.13757](https://arxiv.org/abs/2411.13757)) 도 INT4 quant 에서 3 critical bits → MMLU 0%. **공통 finding**: vulnerable bit 이 attention mechanism + output layer 집중 — KV cache attention key 도 동일 vulnerability 영역. 단 KV cache 는 weight 보다 **temporal volatility 가 높아 detection latency budget 이 sub-second 필요**. Defense paper (LM-Fix [arXiv:2511.02866](https://arxiv.org/abs/2511.02866) / RoR [arXiv:2603.16382](https://arxiv.org/abs/2603.16382) / BitFlipScope [arXiv:2512.22174](https://arxiv.org/abs/2512.22174)) 가 모두 weight 영역 — **KV cache 영역 defense 는 본 세션 P3 Quarantine + V3 Quarantine-Mini + P5 Watermark 의 first stack-able layer**.

#### Top-tier venue alignment 2026-2027

- **OSDI 2027 / ASPLOS 2027**: CXL prefix cache reliability + multi-tenant LLM serving (P1 PrefixGuard).
- **USENIX Security 2027**: multi-agent KV poison isolation + agent-level BFA defense (P3 Quarantine).
- **HPCA 2027 / DSN 2027**: HBM3 PAT counter × KV migration (P4 PATroller).
- **ITC 2027 / DSN short**: HBM3 ECS mailbox × LLM lifetime (P8 ECS-Trace).
- **DAC 2027 / DATE 2027 6p**: CXL DPA poison recompute latency profile, EDAC scrub interval calibration (V3, V1).

#### 본 세션과의 연결

본 세션 6 final idea (P1 / P3 / P4 / P8 / V3 / V1) 모두 위 3 신호 중 1+ 직접 활용. **R50.2 modern memory standard 의 신규 RAS feature 핵심 mechanism integration** 이 v1 (2026-04-25) 의 "feature 인용만 motivation" 사고를 회피하는 핵심 차별점.

---

### VLM + PIM / GPU-PIM Heterogeneous Serving — 2026-04 시점 미니 트렌드
- **Date analyzed**: 2026-04-22
- **Session**: [링크](/research-wiki/2026-04/vlm-pim-extension)
- **Experts**: hw-pim-accelerator-expert (primary), ai-optimization-expert, legacy-system-expert

**신호 1 — VLM-specific HW accelerator 가 2025-Q4 ~ 2026-Q1 폭발적 증가**: Focus ([arXiv:2512.14661](https://arxiv.org/abs/2512.14661), HPCA 2026 Best Paper Candidate, Duke), V-Rex ([arXiv:2512.12284](https://arxiv.org/abs/2512.12284), HPCA 2026, KAIST), ORCHES (MICRO 2025, vision reasoning 3.10×), Pimba ([arXiv:2507.10178](https://arxiv.org/abs/2507.10178), MICRO 2025 post-transformer) 등 4편이 6개월 내 연속 등장. 공통 테마: **VLM 특이성 (visual token 비대칭, multi-image long-context, video streaming) 을 HW 에 직접 반영**. AttAcc (ASPLOS 2024) / NeuPIMs / IANUS / TransPIM 의 LLM 가정은 이미 outdated.

**신호 2 — KV hierarchy 는 레드오션화**: PAM ([arXiv:2602.11521](https://arxiv.org/abs/2602.11521), 2026.02, HBM-PIM + DRAM-PIM + SSD-PIM 3-tier + PAMattention), PIMphony/LoL-PIM ([arXiv:2412.20166](https://arxiv.org/abs/2412.20166), HPCA 2026), LongSight (MICRO 2025, NPU HBM dense + CXL sparse), Scalable PNM ([arXiv:2511.00321](https://arxiv.org/abs/2511.00321), 2025.11 CXL multi-PNM 1M-Token), LeoAM ([arXiv:2506.20187](https://arxiv.org/abs/2506.20187), adaptive GPU-CPU-Disk), Strata ([arXiv:2508.18572](https://arxiv.org/abs/2508.18572), hierarchical context caching) — 3-tier + hot/cold migration 이 이미 6편 이상. **단독 3-tier 제안은 통과 불가**.

**신호 3 — VLM serving scheduling 도 레드오션화**: ModServe ([arXiv:2502.00937](https://arxiv.org/abs/2502.00937), 2025.02), RPS-Serve ([arXiv:2603.26498](https://arxiv.org/abs/2603.26498), 2026.03, MMMU tail 해결), ElasticMM ([arXiv:2507.10069](https://arxiv.org/abs/2507.10069), 2025.07), Dual-Pool Token-Budget ([arXiv:2604.08075](https://arxiv.org/abs/2604.08075), 2026.04), PolyServe ([arXiv:2507.17769](https://arxiv.org/abs/2507.17769)), SLOs-Serve ([arXiv:2504.08784](https://arxiv.org/abs/2504.08784)) — 1년 사이 6편 이상. Modality-aware admission / disaggregation / pool partition 모두 선점.

**신호 4 — Image-hash KV reuse 가 SGLang 에 공식 구현**: VLCache ([arXiv:2512.12977](https://arxiv.org/abs/2512.12977), 2025.12) 가 Qwen3-VL 대상 2% 계산 + 98% 재사용 + 1.2-16× TTFT 달성. vLLM prefix caching 도 image hash 공식 지원 (PR #11187). **Cross-request image KV sharing** 은 더 이상 novel axis 아님. MPIC ([arXiv:2502.01960](https://arxiv.org/abs/2502.01960)), KVShare ([arXiv:2503.16525](https://arxiv.org/abs/2503.16525)), CacheBlend ([arXiv:2405.16444](https://arxiv.org/abs/2405.16444)), DroidSpeak ([arXiv:2411.02820](https://arxiv.org/abs/2411.02820)) 추가.

**신호 5 — VLM KV quantization 은 per-sample / per-layer 분리 시작**: VL-Cache ([arXiv:2410.23317](https://arxiv.org/abs/2410.23317), ICLR 2025 layer-adaptive budget), AKVQ-VL ([arXiv:2501.15021](https://arxiv.org/abs/2501.15021) attention-aware saliency), MBQ ([arXiv:2412.19509](https://arxiv.org/abs/2412.19509), CVPR 2025 modality-balanced), SparseVLM ([arXiv:2410.04417](https://arxiv.org/abs/2410.04417), ICML 2025), SparseVILA ([arXiv:2510.17777](https://arxiv.org/abs/2510.17777), ICCV 2025). 모두 **layer 또는 token granularity**. **Per-sample fragility gating 은 아직 empty axis**.

**신호 5-보강 (2026-04-22 post-literature survey) — VLM quantization × attention distribution dynamics 는 측정 공백**: 본 연구의 W8A8 +66.85pp visual attention collapse (5.73% → 72.58%) 관찰을 계기로 MBQ/Q-VLM/VLMQ/MQuant/AKVQ-VL/VL-Cache/LLM.int8/SmoothQuant/AWQ/QuaRot/SpinQuant/QUIK 전체 정독 결과: **누구도 layer-wise visual attention ratio 를 quantization 전/후로 직접 측정하지 않음**. MBQ 는 gradient sensitivity 로 vision 10× 둔감 보고, VLMQ 는 Hessian 편향, MQuant 는 per-tensor scale mismatch, LLM.int8 는 outlier → softmax mass 방향 영향 — 모두 **간접 signal**. Visual Attention Sink ([arXiv:2503.03321](https://arxiv.org/abs/2503.03321)) 는 FP16 내재적 sink 발견이지만 quantization 영향 미분석. 실배포 W8A8 모델카드 (RedHatAI 등) 도 task accuracy 만 보고. **+66pp extreme collapse 정량화는 공백 + Qwen-VL family specific 측정도 공백**. F2 는 이 공백을 측정 axis + runtime detection axis 로 채움.

**VLM quantization 의 세 가지 mechanism 통합 (F2 post-survey 제안)**: (1) **Vision-side activation outlier 증폭** (MBQ + LLM.int8 합성) — per-tensor W8A8 이 vision-channel outlier 를 기준으로 scale 설정 → softmax logit 분포 왜곡. (2) **Attention sink 증폭** (Visual Attention Sink + Attention Sinks and Compression Valleys [arXiv:2510.06477](https://arxiv.org/abs/2510.06477)) — massive activation quantization 으로 sink-KV dot-product magnitude spiky 증가. (3) **Rotation-invariance 미적용** (MQuant + SpinQuant [arXiv:2405.16406](https://arxiv.org/abs/2405.16406) + QuaRot) — rotation-free recipe 에서 vision-channel outlier 보존. 이 세 mechanism 이 VLM 배포 setting 에서 catastrophic regime 로 수렴하는 경로를 본 연구의 W8A8 collapse 로 최초 정량화. "Seeing but Not Believing" ([arXiv:2510.17771](https://arxiv.org/abs/2510.17771)) 의 attention-accuracy decoupling 으로 MBQ "W8A8 lossless (task acc)" 주장과 +66pp collapse 양립 설명.

**신호 6 — Post-transformer PIM 이 독립 영역으로 분화**: Pimba ([arXiv:2507.10178](https://arxiv.org/abs/2507.10178), Mamba/SSM/linear attn), P3-LLM ([arXiv:2511.06838](https://arxiv.org/abs/2511.06838), NPU-PIM mixed precision with dynamic smoothing), HPIM ([arXiv:2509.12993](https://arxiv.org/abs/2509.12993), SRAM-PIM + HBM-PIM 2-tier), AQPIM (HPCA 2026 in-memory activation quant). Qwen3.5 hybrid linear 같은 post-transformer VLM 은 Pimba 계열이 strong baseline.

**신호 7 — FP16 overflow 가 VLM 특화 numerical safety 이슈로 부상**: 본 연구의 Qwen3-VL L27 self-attn Q·Kᵀ > 65504 overflow 발견 (bf16 migration 강제). 이는 LM head 직전 sharpened representation 의 FP16 5-bit exponent 한계 문제 — **VLM 의 긴 sequence (L≥1000) 에서 정규화**. FlashAttention-3 ([arXiv:2407.08608](https://arxiv.org/abs/2407.08608)) FP32 softmax rescale 은 표준 해법이지만, VLM PIM macro 에서 HW-level 감지 + fallback 은 미탐색.

**HW venue 전략적 교훈**:
1. VLM HW 는 SW+HW co-design 필수 — single-layer contribution 은 distinct competitor 5편 이상에 밀림.
2. **DeepStack topology + quantization robustness 는 2026 시점 unique axes**. Architecture-level novelty 가 단독으로는 약해도 결합 시 defensible.
3. 3-tier hierarchy / VLM serving scheduling 은 DROP 또는 사용 시 반드시 VLM-specific 추가 axis 결합.
4. **Cross-request image KV sharing 은 scooped** — coherence protocol / coarse-grained consistency 축으로 전환 필요.
5. Post-transformer (Mamba, hybrid linear) 확장은 Pimba/P3-LLM baseline 강력 — Qwen3-VL 같은 standard transformer scope 로 좁히는 것이 안전.

**Top-tier venue alignment 2026-2027**:
- **HPCA**: VLM accelerator + quantization + coherence (Focus/V-Rex/AQPIM)
- **MICRO**: GPU-PIM heterogeneous scheduling + post-transformer PIM (ORCHES/Pimba)
- **ASPLOS / MLSys**: end-to-end VLM serving (EPD disaggregation, chunked prefill extensions)
- **ISCA**: large-scale memory hierarchy + cross-tier consistency (PAM 계열 follow-up)
- **NeurIPS / ICLR**: VLM algorithm (per-sample fragility, modality-balanced, DeepStack-aware)

---

### BNN/TNN + CIM + Non-Image 도메인 — 2026-04 시점 미니 트렌드
- **Date analyzed**: 2026-04-22
- **Session**: 링크
- **Experts**: algorithm-expert, hw-pim-accelerator-expert
- **Papers analyzed**: 25+ (ISLPED/ISSCC/Interspeech/arxiv 2022-2026)
- **Trigger**: 공영호 교수의 PRISM (ISLPED'26 투고) 확장 쿼리. scratch training + small model + CIM target 제약.

#### Hot Topics (2024-2026 반복 등장)
| Topic | 대표 논문 | 한 줄 설명 |
|-------|---------|----------|
| **Low-rank scaling factor decomposition** | [LoRDS](https://arxiv.org/abs/2601.22716), PRISM (ISLPED'26 투고) | PRISM 과 LLM PTQ 쪽 (LoRDS) 이 2026 년 거의 동시에 "SF 를 low-rank 로 분해" 원리를 독립 발견 — domain-specific 확장이 open |
| **Streaming/temporal CIM** | [PSCNN](https://arxiv.org/abs/2205.01569), [CIMR-V](https://arxiv.org/abs/2503.22072), [Nonlinear Analog CIM LSTM KWS](https://arxiv.org/abs/2512.06362) | SRAM-CIM 의 streaming audio 가 production 에 근접 — 885 TOPS/W 달성. Crossbar 구조에서 temporal reuse / warm-LUT 는 미개척 |
| **Event-driven SNN on CIM (2025-11 집중)** | [CADC](https://arxiv.org/abs/2511.22166), [SOT-MRAM Event-Driven Spiking CIM](https://arxiv.org/abs/2511.03203), [ASTER](https://arxiv.org/abs/2511.06770), [SpikeFit](https://arxiv.org/abs/2604.14487) | 2025-11 에 SNN-CIM 논문 집중 발표 (243.6 TOPS/W) — **이 트랙은 scoop 매우 위험, 본 세션에서 미선정** |
| **Test-Time Adaptation for quantization** | [TTAQ](https://arxiv.org/abs/2412.09899), [Tent](https://arxiv.org/abs/2006.10726), [RTF-Q](https://arxiv.org/abs/2408.05752) | FP/INT8 양자화 모델이 distribution shift 하에서 성능 저하 — on-device 보정. BNN 쪽은 empty |
| **Biosignal BNN (매우 희박)** | [Arrhythmia-BNN](https://arxiv.org/abs/2304.01568), [BNN-HAR RISC-V](https://arxiv.org/abs/2205.12781), [MINIMALIST](https://arxiv.org/abs/2505.08599) | 2022-2023 의 소수 ancestor. **2024-2026 BNN+CIM+biosignal+subject-shift 4-way intersection 은 empty** (blue ocean) |
| **Wearable KWS BNN** | [BiFSMNv2](https://arxiv.org/abs/2211.06987), [SparkNet](https://arxiv.org/abs/2406.06634) | CPU/MCU 타겟은 성숙, CIM 결합은 "PSCNN 외 없음" — KWS+BNN+CIM+streaming 4-way 도 sparse |

#### Emerging Trends
| Trend | 관련 논문/신호 | 해석 |
|-------|---------------|------|
| Rank decomposition 이 양자화 도메인 전반으로 확산 | LoRDS(2026-01) + PRISM(2026-04 투고) | LoRA 의 training efficiency 이득이 inference quantization 로 확장 중. BNN/domain-특화 확장은 **지금이 golden window** |
| SNN-CIM 2025-11 에 집중 발표 | CADC+SOT-MRAM+ASTER+SpikeFit+ETBW-DVS 5편이 2개월 내 집중 | 2025-11 은 SNN-CIM 의 "hype moment". 후발 주자는 매우 빠른 출시 혹은 composition 전략 필요 |
| Wearable always-on 이 BNN 의 killer use case | ISSCC 2025/2026 의 ultra-low-power biomedical SoC 다수 | <100 µW 24h 대역이 연구→산업 전환점. Medical + HAR 융합이 clinical narrative 강점 |
| RRAM/SRAM collab CIM 실물 chip 가 ISSCC 에 | ISSCC 2025/2026 | precision mix 가 multi-modal workload 용 기술 성숙 중. per-modality partition 은 미답 |
| Streaming / online learning CIM | CIMR-V 등 | batched CIM 에서 streaming 으로 전환 중. Cold-start / warm-LUT 이슈가 새 연구 공간 |

#### Open Problems (이번 세션 도출)
| Open Problem | 관련 idea |
|-------------|----------|
| **Time-axis rank decomposition**: rank decomposition 을 spatial 외 time dimension 으로 확장하는 조건 | F1 TempoPRISM-CoDesign |
| **Subject-shift 를 scalar shift 로 근사 가능한 조건** (rank-1 Taylor 근사의 tightness) | F2 PhysioPRISM |
| **RRAM endurance 보존한 per-subject adaptation**: shadow SRAM 혹은 hybrid 메모리 설계 | F2 VitalXbar |
| **Heterogeneous-precision 다중 region CIM 의 joint training 안정화**: stagewise 외 다른 방법 | F2 VitalXbar (scope 축소) |
| **BNN + CIM + biosignal + subject-shift 4-way intersection** 최초 탐사 | F2 |
| **SNN-CIM 후발 주자의 합리적 repositioning** (CADC 와 composition / N_active-binned SF workshop) | SNN track (미선정) |

#### 유의사항 (Caveats)
- PRISM 의 rank decomposition 과 LoRDS 의 동시 발견 → **rank decomposition for quantization 은 2026 년 hot area** 로 전환 중. 후속 연구가 계속 쏟아질 것.
- SNN-CIM 영역은 이미 포화 — 단순 "RRAM 위에서 같은 아이디어 재실행" 은 scoop 위험. Composition / theoretical lemma 필요.
- Biosignal BNN 은 blue ocean 이나 subject-shift 평가의 variance 가 커 reviewer 의 statistical significance 요구가 높음 — paired t-test / cluster-balanced LOSO 필수.

---

### MoE Expert Fingerprinting (Security + Systems Dual Axis) — 2026-04 시점 미니 트렌드
- **Date analyzed**: 2026-04-21
- **Session**: 링크
- **Experts**: system-robustness-expert, ai-optimization-expert
- **Papers analyzed**: 30+ (Phase 1 expert-memory 기반 + Phase 2 리뷰어 arxiv 재검색)
- **Trigger**: User 공영호 교수가 "fingerprinting accuracy 85-90%, expert 많은 Qwen3에서 더 잘 됨" premise 공유 후 보안+시스템 연구 주제 도출 요청.

#### Hot Topics (2024-2026에 반복 등장)
| Topic | 대표 논문 | 한 줄 설명 |
|-------|---------|----------|
| **MoE expert prediction** | MoE-Beyond([2508.17137](https://arxiv.org/abs/2508.17137), 97.5%), Pre-Attention([2511.10676](https://arxiv.org/abs/2511.10676), 93-94%), OD-MoE([2512.03927](https://arxiv.org/abs/2512.03927), 99.94%), PROBE([2602.00509](https://arxiv.org/abs/2602.00509)) | 2025년 하반기부터 predictor accuracy가 90%+ 고지를 본격 돌파 — premise 85-90%가 오히려 conservative |
| **MoE side-channel attack (NEW)** | MoEcho([2508.15036](https://arxiv.org/abs/2508.15036), CCS'25), Expert Selections([2602.04105](https://arxiv.org/abs/2602.04105)), Stealing User Prompts([2410.22884](https://arxiv.org/abs/2410.22884)) | 2025-2026에 MoE 고유의 routing leakage가 security 영역에서 확립. 향후 2-3년 주요 threat class |
| **MoE serving infrastructure** | vLLM/SGLang EP deployment, Kimi K2.6, GLM-5.1 (754B MoE), DeepSeek-V3 EPLB | 2026년 현재 production MoE 서빙은 fine-grained (128+ experts) 중심 |
| **MoE prefetch family** | HOBBIT, ProMoE, PreScope, BuddyMoE, DuoServe-MoE, SP-MoE, MoE-SpeQ, MoE-SpAc | 포화 상태 — novelty는 formal bound / schedulability 방향으로 이동 |
| **MoE affinity dispatcher** | Semantic Parallelism, Gimbal, DualMap, XShare, METRO | 2025-2026 신흥 영역 — demand-side routing이 supply-side(EPLB) 보완 축으로 부상 |
| **MoE privacy defense** | CryptoMoE, SecMoE (crypto 100×), NoEsis (training DP), CacheSolidarity (KV cache) | Routing-layer에 대한 practical-overhead 방어는 blue ocean |

#### Emerging Trends
| Trend | 관련 논문/신호 | 해석 |
|-------|---------------|------|
| Predictor accuracy crosses threshold | OD-MoE 99.94% | 60-70% 정확도로 설계된 기존 prefetcher들의 "miss 경로 과부하" 설계 전제가 무너지는 중 |
| Side-channel-as-universal-threat | MoEcho + Whisper Leak + NetEcho | LLM에 대한 side-channel은 hardware(MoEcho), network(Whisper), cache timing(InputSnatch) 다축으로 성숙 |
| WCRT for GPU-based inference | RTAS 2025-2026 | GPU serving에서 formal real-time 분석이 점차 수용됨. 단 probabilistic-RT (p99.9) framing 필수 |
| Cross-request expert affinity | Semantic Parallelism, Gimbal | batch 단일 → replica pool 레벨로 dispatcher가 "올라가는" 추세 |
| MoE architecture growth trend | Mixtral-8 → Qwen3-Next-128 → DeepSeek-V3-256 | Expert granularity가 privacy liability (attack) + defense enabler (decoy budget) 양면성 |

#### Open Problems (이번 세션 도출)
| Open Problem | 관련 idea |
|-------------|----------|
| **Fingerprint-driven dispatcher의 signature-entropy scaling law** 형식화 (LSH cohort purity vs N_experts) | I4 FARD-C |
| **MoE expert scheduling의 WCRT theorem** (bounded-miss EDF 변형) | I3 ZMSP |
| **Routing-layer formal MI bound** as function of (k, ε, N_experts) | I2 PhantomRoute |
| **Privacy-vs-expert-count 측정** (MoEcho가 측정 안 한 scaling trend) | I2 motivation (I1 흡수) |
| **VLA phase-conditional fingerprint의 schedulability** | I5 (deferred) |
| Cross-idea predictor 공유 시 artificial-split rejection 회피 전략 | 전체 세션 output strategy |

#### 유의사항 (Caveats)
- MoEcho의 존재를 Phase 1 expert-memory가 놓침 → Phase 2 arxiv 재검색의 중요성 재확인. Web-based related work search는 **모든 세션에서 필수**.
- 85-90% premise는 OD-MoE(99.94%) / MoE-Beyond(97.5%) 대비 conservative. 상한 accuracy로 설계 재평가 시 idea 가치 상승 가능.
- MoE architecture는 few-expert(Mixtral-8) → many-expert(128+)로 명백한 수렴 중. 모든 아이디어는 many-expert regime에 최적화.

---

### VLM/VLA Software-Side Serving Optimization — 2026-04 시점 미니 트렌드 분석
- **Date analyzed**: 2026-04-21
- **Session**: 링크
- **Experts**: ai-optimization-expert
- **Papers analyzed**: 2 (in-depth) + 7 (referenced)
- **Trigger**: ACE-MoE의 future direction (VLM/VLA 확장)을 user가 명시적으로 요청

#### Hot Topics (반복 등장)
| Topic | 대표 논문 | 한 줄 설명 |
|-------|---------|----------|
| MoE expert offloading + caching | ACE-MoE [iccad-2026], HybriMoE [DAC'25], Edge-MoE [ICCAD'23] | GPU memory 한계 극복 위한 expert를 CPU/PIM/Storage로 offload + 활성화 시 transfer |
| Visual KV cache compression / eviction | VL-Cache [arXiv'24], PyramidKV [NeurIPS'24], SnapKV [NeurIPS'24], H2O [NeurIPS'23] | VLM의 visual KV가 sequence 86%+ 차지하나 attention 11% 만 받음 — software-side compression 활발 |
| Importance-driven scheduling | ACE caching, Cache-Conditional MoE [arXiv'24], cumulative score 류 | "hit rate가 아닌 accuracy preservation" 패러다임 전환 |

#### Emerging Trends
| Trend | 대표 논문 | 왜 주목 |
|-------|---------|--------|
| VLM-MoE inference (사용자 워크로드) | MoE-LLaVA [arXiv'24], Uni-MoE [arXiv'24], DeepSeek-VL2 | 모델 크기 증가 + 추론 cost 압박으로 VLM에서도 MoE 채택 가속 |
| VLA serving (real-time robotics) | OpenVLA [RSS'24], Pi-0 [arXiv'24], RT-2-X | LLM-style serving optimization을 hard real-time constraint에 적용해야 |
| Hierarchical multimodal injection | Qwen3-VL DeepStack, BLIP-3, Mantis | ViT 중간 출력을 LM 여러 layer에 inject하는 trend → layer-uniform 최적화의 가정 깨짐 |
| Cross-request sharing (multi-turn / multi-user) | vLLM prefix caching, SGLang RadixAttention, PromptCache | KV-level 공유에서 expert/visual 단위로 확장될 흐름 |
| Joint multi-axis resource budget | (emerging, no canonical paper yet) | 단일 axis 최적화의 한계, multi-resource (token×expert×bandwidth) 최적화 필요성 인지 시작 |

#### Open Problems (저자들이 언급한 미해결)
| Problem | 언급 출처 | 현재 접근법의 한계 |
|---------|---------|-----------------|
| VLM의 prefill TTFT가 LLM 대비 6-22× 길어짐 | VLM exploration (Choi 2026) | software-side로는 visual token 줄이기 외 직접 attack 부재 |
| ACE-MoE의 multimodal 확장 | ACE-MoE 결론 | future work로 명시, 아직 답 없음 |
| Layer-wise visual attention asymmetry 활용한 software-side 최적화 | VL-Cache, ACE-MoE 둘 다 layer dimension 다름 | 두 dim(token×layer, expert×layer)을 모두 다룬 통합 framework 부재 |
| VLA의 hard real-time inference + 정확도 보존 | OpenVLA 후속 | latency budget formulation의 control-theoretic 통합 부재 |
| Cross-request visual sharing의 expert-level extension | vLLM prefix caching 한계 | KV에 한정, expert activation 정보는 미공유 |

#### Cross-domain Opportunities
| Domain A | Domain B | 연결 시나리오 |
|----------|----------|-------------|
| MoE expert offloading | VLM serving | ACE 원칙을 modality-aware로 확장 (I1) |
| KV cache compression | MoE caching | Token+expert joint budget (I3) |
| LLM/VLM serving | Robotics control | Hard real-time SLO 도입 (I5) |
| Production workload analysis | Cache management | Image-level cross-request sharing (I6) |
| Model architecture analysis | Optimization | DeepStack/HIL boundary handling (I1 흡수) |

---

_이 트렌드 분석은 ACE-MoE 본 논문과 VLM exploration 1편의 깊은 분석에서 파생됨. Mode 3 (전 분야 3년 스캔)을 통해 보다 종합적인 트렌드 분석을 수행하면 본 entry가 보강될 것._

---

### 2026-04-21 보강 — 2024-2026 arxiv 관련 연구 검색 결과 기반 업데이트

arxiv API WebFetch 검색(9개 쿼리, 100+ 논문 검토)으로 아래 추가 트렌드 발견:

#### 가속 트렌드 (2025-2026 sub-themes)
- **VLA-specific inference optimization 폭증** (2025-09 이후): ActionFlow(2.55× FPS), HyperVLA(120× speedup via hypernetwork), CogVLA(NeurIPS'25), A1(72% latency reduction via budget-aware truncation), SemanticVLA(2.7×) 등. **VLA serving optimization이 2025년 하반기부터 집중 연구 분야**로 부상.
- **MoE edge inference의 budget/precision co-design 증가**: DyMoE, Dynamic Expert Quantization, Context-Aware MoE on CXL-NDP, SliceMoE 등이 모두 **importance + precision + scheduling의 joint approach** 채택. 연구 설계 자체의 **multi-axis 방향으로의 shift**.
- **Speculative + Prefetch + Quantize 3-way co-design 유행**: MoE-SpeQ, SpecMoE, MoE-SpAc 등.
- **Modality-aware vs modality-agnostic 논쟁 가시화**: AlignMamba-2 (modality-specific + shared experts) vs ERNIE 5.0 (modality-agnostic routing). I1과 같은 direction이 단일 옳은 답이 아님을 인지해야.
- **Null expert / big-little expert 방식 등장**: Improving MoE Compute Efficiency (null experts), MoBiLE (big+little), BuddyMoE (redundancy fallback) — **expert 자체의 heterogeneity 탐구**.

#### Emerging Trends (updated)
| Trend | 대표 논문 | 왜 주목 |
|-------|---------|--------|
| Budget-constrained optimization formulation | Dynamic Expert Quantization (2025-11), A1 (2026-04), DyMoE (2026-03) | "budget-aware"가 paper title 차원에서 급증 — optimization의 formalism이 정착 중 |
| VLA + MoE (architectural fusion) | HEX (2026-04), HY-Embodied-0.5 (2026-04), Qwen3.5-Omni (2026-04) | VLA/embodied foundation에 MoE 채택이 2026년부터 본격화 |
| Speculative-driven MoE inference | MoE-SpeQ (2025-11), SpecMoE (2026-04), MoE-SpAc (2026-02) | speculation으로 prefetch 예측 정확도 ↑ |
| Visual token pruning의 메타 학습 | VisPCO (2026-04, Pareto-frontier), PixelPrune (2026-04, predictive coding) | task-specific에서 generic framework로 이동 |
| KV cache의 multi-agent sharing | TokenDance (2026-04), MemServe | multi-agent serving 패턴 정착 |

#### Open Problems (updated)
| Problem | 최신 언급 |
|---------|---------|
| Modality-aware vs modality-agnostic routing의 empirical 비교 부재 | AlignMamba-2 vs ERNIE 5.0 — 직접 비교 실험 없음 |
| VLA의 action chunk boundary가 ACE / caching의 ideal granularity인가? | Adaptive Action Chunking (2026-04) 제기한 문제, ACE-VLA의 direct question |
| Joint token-expert budget optimizer의 closed-form 존재 여부 | 여전히 미해결, I3 papers에서 정면 접근 시 contribution |
| Visual encoder + LM decoder + expert skip의 end-to-end orchestration | SemanticVLA/CogVLA/ACE-VLA 세 방향이 병렬 전개 중 |

_본 보강은 이 세션의 Related Work 강화를 위해 WebFetch(arxiv API)로 수집. 향후 Mode 2/3 실행 시 매번 관련 분야 최신 6-12개월 논문을 재검색할 것을 권장 (`references/related-work-search.md` 워크플로우 참조)._
