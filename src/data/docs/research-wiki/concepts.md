# Concept Glossary

`aica-research-bot` 실행에서 반복 등장하거나 주요하게 논의된 핵심 개념을 알파벳 순으로 정리한 용어집.

---

## C35. MoT Dual-Tower Serving Asymmetry (phase 별 1-tower 활성) — 2026-06 cosmos3-edge-serving-deep 세션
- **정의**: Cosmos3 MoT 는 각 decoder layer 가 Reasoner(AR) + Generator(diffusion) 두 세트 독립 파라미터를 보유하나 generation phase 마다 **한 tower 만 활성**한다 (VLM mode = generator 완전 미사용, denoising loop = reasoner FFN/proj 미사용·K_AR/V_AR 캐시만 소비). 한 요청이 AR phase(memory-bound, KV-중심, token-by-token) → DM phase(compute-bound, KV-불변, step 반복)를 단일 edge GPU 에서 직렬 traverse — 두 phase 의 자원 요구가 정반대 (G1/G2).
- **기존 baseline**: vLLM(AR paged-KV/continuous-batch) + [vLLM-Omni (arXiv:2602.02204)](https://arxiv.org/abs/2602.02204) (DM denoising, multi-GPU cluster) **분리 스택**; [VLA-across-XPUs (arXiv:2604.24447)](https://arxiv.org/abs/2604.24447) "compute-bound backbone→memory-bound action-expert" 2-phase; DistServe/Splitwise phase disaggregation (cross-GPU).
- **차별화 axis**: 기존 phase 이질성은 across-request 또는 동일 AR 모델 prefill-decode; MoT 는 *한 요청 내 AR↔DM tower 전환* + 16B 중 활성 8B + attention-만-공유라는 구조적 사실. cross-GPU disaggregation 은 edge 단일 가속기에서 불가 → 단일-device 변형이 신규.
- **관련 idea**: S1 TIDELOOM (단일-device 통합 runtime), S2 DUOCLOCK (phase DVFS), S3 LEDGERMARK (phase-transition 측정).

## C36. Static Conditioning KV (read-only K_AR/V_AR, one-shot quant + flow-step error bound) — 2026-06 cosmos3-edge-serving-deep 세션
- **정의**: Cosmos3 generator 는 `O_DM = Attn_full(Q_DM, [K_AR;K_DM], [V_AR;V_DM])` 로 reasoner 의 K_AR/V_AR 을 **N denoising step × CFG 2 pass 전 기간 read-only static prefix** 로 cross-attend 한다 (reasoner-tower caching, 1회 계산·전 step 재사용·품질손실 0). decode-time KV(online·매 토큰 cheap·growing)와 정반대 — (i) 비싼 1-shot 양자화(Hadamard rotation + MSE-optimal clipping)를 amortize 가능, (ii) denoising 출력 error 를 KV quant error 의 함수로 **flow-step 수 N 에 명시 의존하는 닫힌 bound** 로 유도 가능 (`‖x̂_0−x_0‖ ≤ (Σ_n Δ_n·L_v^{(n)})·(L_softmax‖ΔK_AR‖+‖ΔV_AR‖)`, N 작을수록 tight = policy 가 저비트에 관대).
- **기존 baseline**: [KIVI (arXiv:2402.02750)](https://arxiv.org/abs/2402.02750)/[KVQuant (arXiv:2401.18079)](https://arxiv.org/abs/2401.18079) (online growing decode-KV), [QuantKeys (arXiv:2605.26266)](https://arxiv.org/abs/2605.26266) (video diffusion KV quant + Jensen-bias, autoregressive growing-KV), [33-method study (arXiv:2603.27469)](https://arxiv.org/abs/2603.27469) (순수 실증, bound 없음), SVDQuant (weight/activation only).
- **차별화 axis**: 정적·1회·read-only prefix 라서만 가능한 N-의존 closed-form denoising error bound — online decode-KV 엔 step 개념 자체가 없음. 경쟁 KV-quant 전부 bound 부재(WebFetch 확인).
- **관련 idea**: Q2 ANCHOR (numeric, bound 소유), A2 Keystone (runtime layout/dedup 위임), L3 KEELKV (L2 placement 위임). G4 ownership table 계층 분리.
- **[2026-06-05 보강]** (source-code 직접 검증, R72): (1) 코드상 serving K_AR 는 **text-only** (관측 시에는 GEN tower 에 주입 — 두 시나리오 병기). (2) policy 는 **CFG ON** (guidance 3.0 + CFG-parallelism → cached K_AR cond/uncond 2벌; guidance_scale=1.0 은 forward/inverse-dynamics·padding pass 용). (3) denoising error bound 는 단일-layer 가 아니라 **36 gen layer 누적** 항을 포함해야 valid (구 식 single-layer = optimistic-invalid). (4) cosmos3 `cached_kv` 는 **post-RoPE** 저장(L548) → Hadamard 를 RoPE 뒤에 적용해야 위치정보 보존. K_AR per-layer footprint = 16×16 patch / policy ~1,050 tok / GQA 8-head / 36 layer 기준 BF16 ~4.3MB · INT4 ~1.08MB.

## C37. Phase-Deterministic Weight Residency (활성 8B/16B, FluxMoE demand-paging 대비) — 2026-06 cosmos3-edge-serving-deep 세션
- **정의**: MoT 의 tower 활성은 **phase 로 100% 결정론적** — generation mode 가 어느 tower 를 쓰는지 컴파일-타임에 확정. 비활성 tower weight 를 UMA 에서 demote, 다음 phase weight 를 현재 phase denoise compute 와 overlap 하여 double-buffered `cudaMemcpyAsync` staging (Orin `concurrentManagedAccess=0` prefetch 미지원 대응). denoise long window(8 forward/chunk @15Hz)가 staging hiding 기회.
- **기존 baseline**: [PowerInfer-2 (arXiv:2406.06282)](https://arxiv.org/abs/2406.06282) (neuron-level activation-predictor, 비결정 miss penalty), [FluxMoE (arXiv:2604.02715)](https://arxiv.org/abs/2604.02715) (expert paging, token-routing 비결정), [DuoServe-MoE (arXiv:2509.07379)](https://arxiv.org/abs/2509.07379) (dual-phase prefetch).
- **차별화 axis**: 예측기 기반 demand-paging 과 달리 phase-결정성은 예측기 불필요·더 coarse·정확 — MoT 의 modality/기능-disjoint weight 가 swap 에 유리한 구조적 property. Orin UVM-prefetch 미지원 제약 정확 대응(double-buffer)이 systems novelty.
- **관련 idea**: S1 TIDELOOM-M2 (residency full method), S1-mini (policy-only NX 16GB fit enabling).

## C38. Modality Token-Density Asymmetry (video ≫ audio ≫ action, h=w=0 MRoPE) — 2026-06 cosmos3-edge-serving-deep 세션
- **정의**: Cosmos3 의 모달리티 token 밀도 = video VAE(공간그리드×프레임, 수만) ≫ audio(48kHz hop 1920 → **25 token/s**) ≫ action(~32 future joint-position @15Hz) (G6). audio/action 토큰은 3D MRoPE 에서 **h=w=0**(temporal 축만), video 는 (t,h,w) 모두 사용 → attention 구조·layout 단순성 차등. video gen 이 image 대비 100× 에너지([arXiv:2601.22076](https://arxiv.org/abs/2601.22076)) — video token 이 edge 비용 지배하나 policy 에선 출력은 action 뿐(video-latent decode skip 하나 attention 잔존).
- **기존 baseline**: [Modality Inflation (arXiv:2512.22695)](https://arxiv.org/abs/2512.22695) (MLLM 에너지 17-94%), [UVA (arXiv:2503.00200)](https://arxiv.org/abs/2503.00200) (video gen bypass, 이분법), MotuBrain action-only suffix, [VLA-Pruner (arXiv:2511.16449)](https://arxiv.org/abs/2511.16449) (token salience prune).
- **차별화 axis**: 기존은 video skip 이분법(전부 skip or keep) 또는 token prune; modality 별 *연속 precision/step* 을 정보흐름 bound 로 배분(video→action attention-mass p_v)은 미개척. [UD-VLA (arXiv:2511.01718)](https://arxiv.org/abs/2511.01718) "joint > decoupled" 반증과 긴장.
- **관련 idea**: Q5 RELAY (p_v bound 기반 modality bit/step, p_v pilot 재방문), L3/S3 modality working-set 측정, A2-M3 modality tiering.

## C39. Intra-Request Dual-Regime DVFS (AR↔DM 위상 전환, J/chunk metric) — 2026-06 cosmos3-edge-serving-deep 세션
- **정의**: 한 inference **내부**에서 AR(memory-bound, EMC-freq 민감) ↔ DM(compute-bound, GPU-freq 민감)이 교대 — Jetson 의 EMC(memory) clock 과 GPU clock 을 phase 에 맞춰 **물리적으로 분리 제어**(AR: EMC max + GPU↓ / DM: GPU max + EMC↓, chunk-granularity 전환으로 freq settle 흡수). tegrastats 33-50ms < step → J/step 직접 측정 불가 → **J/chunk(policy 15Hz 2.1s) + J/inference-phase 재정의**가 측정 방법론 기여.
- **기존 baseline**: [DualScale (arXiv:2602.18755)](https://arxiv.org/abs/2602.18755) (disaggregated LLM phase-DVFS, across-request), [GreenLLM (arXiv:2508.16449)](https://arxiv.org/abs/2508.16449) (prefill/decode GPU-freq, EMC 무, LLM-only), [SparseDVFS (arXiv:2603.21908)](https://arxiv.org/abs/2603.21908) (CPU/GPU/EMC triplet, edge, operator-sparsity), [DynamoLLM (arXiv:2408.00741)](https://arxiv.org/abs/2408.00741) (cluster GPU-freq).
- **차별화 axis**: 데이터센터 DVFS 에 EMC 분리 개념 자체가 없음; SparseDVFS 가 EMC-triplet+edge 도달했으나 operator-sparsity 기반이지 AR↔DM modality-regime 전환 아님 → (a)intra-request dual-regime + (c)J/chunk 재정의 두 축만 유일 (Tier-2 강등 정당).
- **관련 idea**: S2 DUOCLOCK (governor + J/chunk LUT), S3 LEDGERMARK (J/phase 측정 producer).

---

## Scenario-Aware VLM Serving Dispatcher — 2026-05-02 vlm-scenario-aware 세션
**정의**: Production VLM (vLLM/SGLang) 의 scenario diversity (image/video × single/multi-turn × document/agent 6 class) 를 lightweight classifier (DistilBERT-mini 60M) 로 사전 분류 후 scenario-conditional config table dispatch (KV budget / prefix policy / compression rate). ECVL-ROUTER ([arXiv:2510.27256](https://arxiv.org/abs/2510.27256), ICLR'26) 의 model-tier routing 과 직교 axis (single-model serving-stack config). Production hit rate 60-85% measured common case 활용.
**관련 자료**: ECVL-ROUTER ICLR'26, vLLM Prefix Caching, SGLang RadixAttention [ICLR 2024], LMCache [arXiv:2510.09665](https://arxiv.org/abs/2510.09665)
**관련 아이디어**: Mosaic (Tier-1, 2026-05-02)

## Frame-Level Radix Tree Vision KV Cache — 2026-05-02 vlm-scenario-aware 세션
**정의**: Per-frame perceptual hash (pHash, OpenCV `cv2.img_hash.pHash` 16-bit) 를 frame ID 로 사용 + frame ID 시퀀스를 SGLang RadixAttention pattern 의 frame-level 확장 radix tree 에 insert. Multi-turn video QA 의 turn 2+ vision tower re-run 제거 + cross-session frame prefix sharing (privacy classifier CLIP-based 95%+). VLCache ([arXiv:2512.12977](https://arxiv.org/abs/2512.12977), 2025-12) single-session 과 직교 axis (cross-session). PrefixKV ([NeurIPS 2025](https://github.com/THU-MIG/PrefixKV)) layer-wise binary search 결합.
**관련 자료**: VLCache 2025-12, PrefixKV NeurIPS'25, SGLang RadixAttention ICLR'24, mlx-vlm Issue #832 (production gap)
**관련 아이디어**: Lattice (Tier-1, 2026-05-02)

## Cross-Image Vision Token Pool with Privacy Boundary — 2026-05-02 vlm-scenario-aware 세션
**정의**: Multi-image agent loop (MileBench 6+ image / request) 의 cross-image redundancy (same logo / UI element) 를 image-level pHash + tenant-shared LRU pool + CLIP-based public/private classifier (95%+ accuracy) + reference counting + cuckoo filter + zeroize on evict. PolyKV ([arXiv:2604.24971](https://arxiv.org/abs/2604.24971)) / KVShare ([arXiv:2503.16525](https://arxiv.org/abs/2503.16525)) 의 LLM token cross-request 와 직교 axis (vision token cross-image).
**관련 자료**: PolyKV [arXiv:2604.24971], KVShare [arXiv:2503.16525], OxyGen [arXiv:2603.14371](https://arxiv.org/abs/2603.14371), cuckoo filter
**관련 아이디어**: Bramble (Tier-1, 2026-05-02)

## DeepStack-aware NVFP4 Anchor — 2026-04-28 vlm-edge-layerwise-context 세션
**정의**: Blackwell native NVFP4 (E2M1, 16-element block scaling) 의 transformer layer 별 sensitivity 분포를 Qwen3-VL DeepStack architecture (visual_indexes=[8,16,24] inject point) 와 정합시켜 layer 별 NVFP4/FP8/INT4 mixed precision 결정. ATRIUM-R(AI) M1 LayerClassifier 의 evolution 으로 본 세션 Prism 의 unique axis. FGMP / MicroMix 의 LLM-only mixed precision 과 차별 (DeepStack 인지).
**관련 자료**: [NVFP4 NVIDIA Tech Blog](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/), [FGMP arXiv:2504.14152](https://arxiv.org/abs/2504.14152), [MicroMix arXiv:2508.02343](https://arxiv.org/abs/2508.02343), [Qwen3-VL arXiv:2511.21631](https://arxiv.org/abs/2511.21631)
**관련 아이디어**: Prism (Tier-1, 2026-04-28), Obelisk (Tier-2 per-stage cap)

## Phase-aware Visual LSH Hash — 2026-04-28 vlm-edge-layerwise-context 세션
**정의**: VLM serving 의 encode / prefill / decode 3 phase 마다 다른 LSH (Locality-Sensitive Hashing) granularity policy 적용 — encode 단계 fine-grain (visual token 단위), prefill 단계 coarse-grain (image-level), decode 단계 cache-only (lookup). VLCache (encoder cache + content hash) 와 차별 — single-policy → phase-policy 분리. RadixVL 의 unique axis. Green Context CUDA 12.4 SM partition 과 통합되어 LSH lookup overhead 분산.
**관련 자료**: [VLCache arXiv:2512.12977](https://arxiv.org/abs/2512.12977), [SimCache (CVPR 2025W)](https://www.lmsys.org/blog/2024-01-17-sglang/), [SGLang RadixAttention arXiv:2312.07104](https://arxiv.org/abs/2312.07104)
**관련 아이디어**: RadixVL (Tier-1 W12 동적 분기, 2026-04-28)

## Cross-Frame Visual KV Cluster Reuse (CFCR) — 2026-04-28 vlm-edge-layerwise-context 세션
**정의**: 연속 프레임 간 visual token 의 semantic embedding similarity (cosine ≥ 0.85) 을 활용해 K-means cluster centroid level 에서 KV reuse. Layer-Adaptive Cluster Budget (LACB, HOT layer 더 많은 cluster) 와 결합. ClusterKV (LLM only, single-frame) + Sali-Cache (saliency single-axis) 와 차별 — VLM-specific + cross-frame + layer-adaptive. Mosaic ([arXiv:2604.10060](https://arxiv.org/abs/2604.10060)) 와 55-65% concurrent (Phase 2' 검출).
**관련 자료**: [ClusterKV arXiv:2412.03213](https://arxiv.org/abs/2412.03213), [Sali-Cache arXiv:2602.14236](https://arxiv.org/abs/2602.14236), [VL-Cache ICLR 2025 arXiv:2410.23317](https://arxiv.org/abs/2410.23317)
**관련 아이디어**: Bivouac (Tier-1 🥈, 2026-04-28)

## Stratified KV Layout (L2 / GDDR7 / LPDDR5x UMA) — 2026-04-28 vlm-edge-layerwise-context 세션
**정의**: VLM KV cache 를 layer score (visual_attn_ratio) 기반으로 4-tier 가 아닌 3-tier physical layout 에 매핑 — HOT layer → GPU L2 carveout (cudaCacheConfigure persistent) / MEDIUM → GDDR7 또는 LPDDR5x UMA (RTX 5090 / Jetson Thor 다름) / COLD → LPDDR5x bottom rank. Page-color affinity 로 UMA bank-level partitioning 추가 (ATRIUM-R(Sys) 흡수). PagedAttention single-tier 와 차별. Bluefield-4 ICMS 4-tier 와 다른 axis (single-system + page-color).
**관련 자료**: [NVIDIA Bluefield-4 ICMS](https://developer.nvidia.com/blog/introducing-nvidia-bluefield-4-powered-inference-context-memory-storage-platform-for-the-next-frontier-of-ai/), [Sarathi-Serve OSDI 2024](https://www.usenix.org/conference/osdi24/presentation/agrawal)
**관련 아이디어**: Strata (Tier-2 T1, 2026-04-28)

---

## R55 5-Axis Gain Target — 본 harness ideation scope (2026-04-27 v2-r55 등록)
**정의**: 본 harness 의 selected idea 는 다음 5(+2 sub) axis 중 1+ 정량 gain target 의무. **[Performance]** (latency / throughput / TTFT / decode tps) / **[Robustness]** (adversarial / fault tolerance / silent corruption) / **[Energy]** (energy/token / energy/inference) / **[Power]** (thermal envelope / nvpmodel / DVFS) / **[Security]** (encryption / GMAC unforgeable / privacy ε-DP) + sub-axis **[Memory eff.]** (KV / weight / max batch) / **[Cost eff.]** (cost per token / 학기 budget fit). **R55.1** simulator infrastructure / extension / tool / benchmark / dataset 신규 release 자체 contribution idea 는 자동 배제. **R55.2** 예상 효과 표 첫 column 에 [axis] tag 의무.
**관련 자료**: skill.md Rule 27, references/no-simulator-building-concrete-gain.md
**관련 아이디어**: 모든 2026-04-27 v2-r55 selected idea (ATRIUM/BREAKWATER-T/VEILSEAL-KV/CASCADE-PREFILL/STORMGLASS/ROBUSTOKEN) 5-axis 균형 cover

## VEILSEAL-KV (Adversarial-Secure Multi-Tenant Edge VLM KV Cache) — 2026-04-27 v2-r55 Tier-1
**정의**: Edge multi-tenant VLM 의 SGLang RadixAttention prefix block 에 per-tenant ARMv8 PMULL GMAC seal + visual prompt injection detector + PII redaction + ε-DP composition 통합 idea. 본 harness 의 [Security] axis 첫 selected idea. WARDEN-KV (linear CRC, forgeable) + EPSILON-VEIL (ε-DP) + PIIVEIL-Q (PII) sub-mechanism 흡수.
**관련 자료**: USENIX Security / S&P / OSDI 2027 target. ARMv8 PMULL spec, [Visual Prompt Injection arXiv:2403.09766](https://arxiv.org/abs/2403.09766), [Mironov 2017 ε-DP arXiv:1702.07476](https://arxiv.org/abs/1702.07476)
**관련 아이디어**: VEILSEAL-KV (Tier-1, v2-r55), KEYSTONE (2026-04-26 rowhammer 세션 KV GMAC for rowhammer)

## ROBUSTOKEN (5-Cluster Robustness Merge) — 2026-04-27 v2-r55 Tier-2
**정의**: VLM robustness 의 통합 paradigm — adversarial (FGSM/PGD on visual token) + silent corruption (FP16 NaN/Inf) + OOD detection 3 axis 통합 inference-time + training-free + edge-fit. 5 sub-mechanism cluster merge: SCRIVENER (warp-ballot NaN detect) + NAN-SAFENET (INT8 fallback) + PROBE-DECODE (cross-attention spike detect) + REEFCAST (5-class confidence reject + smaller-model fallback) + OOD-CALIB (energy-based OOD).
**관련 자료**: [Energy-OOD NeurIPS 2020 arXiv:2010.03759](https://arxiv.org/abs/2010.03759), [Robust ViT CVPR 2024 arXiv:2402.07004](https://arxiv.org/abs/2402.07004), [GPUHammer USENIX Sec 2025 arXiv:2507.08166](https://arxiv.org/abs/2507.08166)
**관련 아이디어**: ROBUSTOKEN (Tier-2, v2-r55, NeurIPS / ICLR target)

## STORMGLASS (Skin-Temp Leading-Indicator Power Tuning) — 2026-04-27 v2-r55 Tier-2
**정의**: Edge VLM 의 thermal envelope (7-130W) 안 sustained workload 안정화 idea. Tegrastats `temp` field 1Hz polling + skin-temp leading-indicator hysteresis (±2°C) 로 nvpmodel mode 동적 switch + UMA GPU/CPU power split + thermal-aware admission control 통합. 자동차/robotics 산업 직접 정합.
**관련 자료**: NVIDIA Jetson Power Tuning Guide, Tegrastats spec, nvpmodel CLI
**관련 아이디어**: STORMGLASS (Tier-2, v2-r55, ISLPED 6p / DAC target)

## ARMv8 PMULL (AArch64 Crypto Extension) — 2026-04-27 v2-r55 (VEILSEAL-KV M1)
**정의**: ARMv8.2-A Crypto Extension 의 polynomial multiplication HW instruction (`aes64gmac`). GCM/GMAC 의 HW accel — 1.5-3 cycle per byte. Jetson AGX Thor (Neoverse-V3AE = ARMv9-A + crypto) / AGX Orin (Cortex-A78AE = ARMv8.2-A + crypto) 모두 지원. gcc `-march=armv8.2-a+crypto` flag 표준.
**관련 자료**: ARMv8.2-A spec, [NIST SP 800-38D AES-GCM](https://csrc.nist.gov/pubs/sp/800/38/d/final)
**관련 아이디어**: VEILSEAL-KV M1 (multi-tenant block GMAC, 1.5-3 cycle/byte)

## ε-DP / Rényi-DP Composition — 2026-04-27 v2-r55 (VEILSEAL-KV M3)
**정의**: ε-Differential Privacy 의 Rényi-DP composition (Mironov 2017) — multi-request 누적 budget tracking. Inference-time KV outlier mask 에 noise injection 시 reconstruction attack 방어. ε=4.0 / δ=10⁻⁶ default.
**관련 자료**: [Mironov 2017 arXiv:1702.07476](https://arxiv.org/abs/1702.07476), [Abadi DP-SGD arXiv:1607.00133](https://arxiv.org/abs/1607.00133), Opacus PyTorch framework
**관련 아이디어**: VEILSEAL-KV M3 (PII redaction + ε-DP composition)

## DeepStack Injection Schedule — VLM Architecture (2026-04-27 세션 ATRIUM/BREAKWATER-T 활용)
**정의**: Qwen3-VL ([arXiv:2511.21631](https://arxiv.org/abs/2511.21631)) 의 핵심 architecture upgrade — ViT intermediate output 을 LLM 의 여러 layer 에 inject. Default schedule: ViT layer-1 → LLM layer-4 inject, ViT layer-4 → LLM layer-8, ViT layer-N → LLM layer-12. 일반 VLM (LLaVA-1.5) 의 last-layer-only injection 대비 minimal additional cost 로 visual token 4× 증가 효과. **DeepStack 자체는 [NeurIPS 2024 arXiv:2406.04334](https://arxiv.org/abs/2406.04334) (Meng et al., Fudan/Microsoft)** 에서 처음 제안됨.
**관련 논문**: [DeepStack NeurIPS 2024](https://arxiv.org/abs/2406.04334), [Qwen3-VL Tech Report](https://arxiv.org/abs/2511.21631)
**관련 아이디어**: ATRIUM (DeepStack L0-3 alloc skip 으로 GPU HBM 절약), BREAKWATER-T (ViT split point = DeepStack tap point 일치), BIMODAL-MASK-T2 (DeepStack OR-merge per-layer mask)

## Layer-wise Visual Attention Asymmetry — VLM Inference Property (2026-04-27 세션 motivation)
**정의**: Qwen3-VL-4B inference 에서 measured layer-wise visual token attention 비율 — L17-21 평균 24.5% / L0-7 평균 2.6% (5-10× 비대칭). LLM 의 forward pass 동안 visual token 이 각 layer 별로 받는 attention weight 가 매우 비균등 분포. L0-7 region 의 BW waste 86% sequential read / 11% attention 측정 (내부 측정 자료 `VLM_exploration_PIM_260407.pdf`). 이 비대칭이 layer-aware system optimization (SM partition / L2 carveout / KV residence policy) 의 motivation.
**관련 논문**: [Qwen3-VL Tech Report](https://arxiv.org/abs/2511.21631), [SparseVLM ICML 2025](https://arxiv.org/abs/2410.04417)
**관련 아이디어**: ATRIUM (layer-aware SM partition + L2 carveout), BIMODAL-MASK-T2 (modality outlier topology Moran's I), CASCADE-PREFILL (chunk-AI dispatch GPU↔CPU)

## VLM Prefill TTFT Explosion — VLM vs LLM Workload Asymmetry (2026-04-27 세션 motivation)
**정의**: VLM 의 visual token (1 image → 100~4096 token) 추가로 인한 prefill phase 길이 폭증. 측정: Qwen3-VL-4B 672×672 server BS=8 = 353 ms vs LLM 56 ms (6.13×), FHD 1920×1080 = 1285 ms vs 57 ms (22.4×). Decode tok/s 는 LLM=573 vs VLM=554 거의 동일 — visual KV 가 있어도 decode 성능 차이 미미, 즉 **prefill 이 VLM 의 unique bottleneck**. 이 axis 가 dual-Jetson disaggregation (BREAKWATER-T) + chunked prefill GPU↔CPU dispatch (CASCADE-PREFILL) 의 motivation.
**관련 논문**: 내부 측정 (`VLM_exploration_PIM_260407.pdf`), [DistServe OSDI 2024](https://arxiv.org/abs/2401.09670), [Nexus arXiv:2507.06608](https://arxiv.org/abs/2507.06608)
**관련 아이디어**: BREAKWATER-T (prefill -28% via dual-Jetson sensor-proximity split), CASCADE-PREFILL (prefill -12~17% via chunk-AI GPU↔CPU dispatch)

## Visual KV Capacity Saturation — VLM Memory Asymmetry (2026-04-27 세션 motivation)
**정의**: VLM visual KV size 가 LLM 대비 17× 큰 capacity 점유. 측정: Qwen3-VL-4B FHD 1920×1080 = 305 MB/req vs LLM 18 MB/req. Max batch = 118 vs 2002 (16.97× 감소). 단일 FHD image 가 KV cache 의 거의 전부 점유 → edge serving 에서 capacity bottleneck. 본 세션 ATRIUM 의 DeepStack L0-3 alloc skip + BIMODAL-MASK-T2 의 modality-conditioned dtype dispatch 의 motivation.
**관련 논문**: 내부 측정 (`VLM_exploration_PIM_260407.pdf`), [VL-Cache arXiv:2410.23317](https://arxiv.org/abs/2410.23317)
**관련 아이디어**: ATRIUM (alloc skip), BIMODAL-MASK-T2 (per-block dtype dispatch -23~29% memory)

## USB-C 3.2 gen2x2 Edge Interconnect — Dual-Jetson topology (2026-04-27 세션 BREAKWATER-T)
**정의**: Jetson 은 PCIe / NVLink 미가용 (datacenter GPU 와 차별화). Dual-Jetson serving 의 sole 고대역 interconnect 는 **USB-C 3.2 gen2x2 (20 Gbps, full-duplex)** 또는 Ethernet (1/2.5/10 GbE). 단방향 30us roundtrip + ±5ms cable burst variance 가정. NVIDIA Holoscan SDK 가 이 topology 의 reference cable + spec.
**관련 자료**: NVIDIA Holoscan SDK, [TensorRT Edge-LLM blog](https://developer.nvidia.com/blog/accelerating-llm-and-vlm-inference-for-automotive-and-robotics-with-nvidia-tensorrt-edge-llm/), [DiP-SD arXiv:2604.20919](https://arxiv.org/abs/2604.20919)
**관련 아이디어**: BREAKWATER-T (4-bit channel-wise tap stream USB-C 부담 ¼), JETTYSIM (USB-C link model 30us+variance), LIGHTLINK-SD reframed (USB-C link-aware speculative decoding)

## Modality-conditioned Outlier Topology (Moran's I) — VLM KV Quantization (2026-04-27 BIMODAL-MASK-T2)
**정의**: VLM KV cache 의 outlier (high-magnitude value) 분포가 modality 마다 다른 spatial topology — vision KV 는 spatial cluster (Moran's I = 0.34, 인접 토큰 끼리 outlier 집중), text KV 는 channel-uniform (Moran's I = 0.02). Spatial autocorrelation 통계량 Moran's I 가 modality-aware quantization 의 정량 지표.
**관련 논문**: KIVI ([ICML 2024](https://arxiv.org/abs/2402.02750)), KVTuner (ICML 2025), MBQ (CVPR 2025), MadaKV (ACL 2025)
**관련 아이디어**: BIMODAL-MASK-T2 (Moran's I 기반 cluster=BF16 + sparse=NVFP4 + weight band=INT8 dispatch)

## RFM (Refresh Management) — DDR5/LPDDR5 표준 명령 (2026-04-26 PM 세션 RFM-COP 활용)
**정의**: DDR5 JESD79-5C / LPDDR5 JESD209-5C 표준의 host MC → DRAM 명령. PRAC counter 가 ACT-N 도달 시 MC 가 발행 → DRAM 이 victim row 를 preventive refresh. **McSee (USENIX Sec'25)** 측정: 29 DDR5 UDIMM 중 16 RFM 지원, 4 require — 그러나 Intel/AMD CPU 어느 것도 rowhammer workload 에서 RFM 명령을 보내지 않음. **RFM-COP** 는 이 host-MC 부재 문제를 직접 해결.
**관련 논문**: [McSee USENIX Sec'25](https://www.usenix.org/conference/usenixsecurity25/presentation/jattke), [ARFM arXiv:2501.14328](https://arxiv.org/abs/2501.14328), [QPRAC arXiv:2501.18861](https://arxiv.org/abs/2501.18861), JEDEC JESD79-5C
**관련 아이디어**: RFM-COP (4-pillar host-MC scheduler), RAMPART (Hot bucket RFM trigger 활용)

## DRFM (Directed Refresh Management) — DDR5 표준 (2026-04-26 PM)
**정의**: RFM 명령에 specific row id 를 첨부하여 victim row 만 정밀 refresh 하는 DDR5 표준. Sampling 기반 statistical mitigation 가능성 — Salman Qazi (Google) DRAMSec'25.
**관련 논문**: [DRFM DRAMSec'25 paper](https://dramsec.ethz.ch/dramsec25-papers/drfm-dramsec25.pdf)
**관련 아이디어**: RFM-COP Mechanism 4 (DRFM packet builder ~2k gates)

## TPRAC (Timing channel attack on PRAC) — 2026-04-26 PM
**정의**: ISCA'25 발표 ([arXiv:2505.10111](https://arxiv.org/abs/2505.10111))의 attack 명. PRAC counter update 가 tRC 를 증가시켜 attacker 가 victim 의 access pattern 추론 가능. Defense: constant-noise RFM injection (LFSR Fuzzing).
**관련 논문**: [TPRAC arXiv:2505.10111](https://arxiv.org/abs/2505.10111)
**관련 아이디어**: RFM-COP Mechanism 3 (RogueRFM Fuzzing 으로 TPRAC mutual info < 0.1 bit/access 차단)

## Phoenix CVE-2025-6202 (2026-04-26 PM 세션 motivation)
**정의**: 2025-09 disclosure 된 CVE — ETH Zurich 가 SK Hynix DDR5 의 in-DRAM mitigation (PRAC + auto-RFM) 을 109초 만에 우회. S&P'26 publication 예정. host-MC fallback 의 결정적 가치를 입증.
**관련 논문**: S&P 2026 Phoenix paper (TBD arxiv), 2025-09 NVD entry CVE-2025-6202
**관련 아이디어**: RFM-COP 의 핵심 narrative

## PAT (Probabilistic Aggressor Tracker) — HBM3 base die counter (2026-04-26 PM RAMPART)
**정의**: HBM3 base die 의 in-memory counter — bank/row 별 activation 빈도를 tracking. 본 세션 RAMPART 가 2-tier (Hot 8K SRAM-PAT + Cold 64K CAM-PAT) 로 분리하여 PRAC counter promotion + ECC tier promotion 동시 운영.
**관련 논문**: [HBM3 RAS Gurumurthi & Lee, AMD](https://www.semanticscholar.org/paper/HBM3-RAS:-Enhancing-Resilience-at-Scale-Gurumurthi-Lee/68d2787d4edd16dd52b0bf789b31692529fbd59c), [Hydra HPCA'22](https://safari.ethz.ch/architecture_seminar/spring2023/lib/exe/fetch.php?media=hydra.pdf), [Mithril DSN'22], [PVAC arXiv:2604.20576](https://arxiv.org/abs/2604.20576)
**관련 아이디어**: RAMPART (PAT + ECC tier coupling), RFM-COP (PAT 활용 indirectly)

## GMAC (Galois/Counter Mode Authentication Code) — KEYSTONE (2026-04-26 PM)
**정의**: AES-128-GCM 의 authentication 부분 — universal hash function (over GF(2^128)) 으로 unforgeable MAC 생성. Linear checksum (CRC-32, Adler) 의 forgeable 약점을 해결. Block 64KB 당 64-bit tag, FAR < 2⁻⁶⁴.
**관련 논문**: [NIST SP 800-38D AES-GCM](https://csrc.nist.gov/publications/detail/sp/800-38d/final), Bonsai Merkle Tree (Intel TDX), Apple MIE (2025-09), AMD SEV-SNP, [KV-Cache Bit-Flip arXiv:2604.17249](https://arxiv.org/abs/2604.17249)
**관련 아이디어**: KEYSTONE (PagedAttention 64KB block GMAC adversarial-secure)

## DPA Poison List (CXL.mem RAS register) — 2026-04-26 PM HARBOR
**정의**: CXL 3.x 표준의 RAS register set — Device Physical Address (DPA) 단위 poison entry 를 host 에서 add/remove 가능. Linux 6.16 EDAC mainline 이 generic interface 제공. HARBOR 의 64-entry CAM 으로 hardware lookup.
**관련 논문**: [CXL RAS Whitepaper](https://computeexpresslink.org/wp-content/uploads/2023/12/CXL-RAS-Whitepaper-Post-WG-Revision_FINAL.pdf), [Linux 6.16 EDAC docs](https://lwn.net/Articles/982190/)
**관련 아이디어**: HARBOR Mechanism 1 (DPA poison CAM + RAS coordinator)

## Memory Event Record (MER) — CXL RAS (2026-04-26 PM HARBOR)
**정의**: CXL 3.x RAS register — DRAM error event 를 hardware-stream 으로 host 에 push. Mailbox polling 회피. HARBOR 의 256-entry × 256-bit ring buffer (8KB SRAM) 로 구현.
**관련 논문**: CXL 3.0/3.1 spec
**관련 아이디어**: HARBOR Mechanism 1

## ECS (Error Check & Scrub) — HBM3 / CXL self-diagnostic mode
**정의**: HBM3 / CXL Type-3 device 의 self-diagnostic mode — self-refresh idle 또는 host refresh-all 명령 시 internal scrub 수행. **HBM3** 은 self-refresh-only (active workload 미적용) — sliding-window scrub 필요성. **CXL** 은 ECS mailbox 로 host 가 trigger 가능.
**관련 논문**: [HBM3 ISSCC 2022 IEEE 9830391](https://ieeexplore.ieee.org/document/9830391/), CXL RAS Whitepaper
**관련 아이디어**: HARBOR (CXL ECS mailbox 통합), L1 MOSAIC (HBM3 ECS scheduler), LIGHTHOUSE (ECS scheme 비교)

## CXL Patrol Scrub Control (P1 PrefixGuard / V1 origin)
**정의**: CXL 3.2 spec 의 background scrub mechanism. **host 가 hour-단위 interval 을 device 에 설정**할 수 있는 read/correct/writeback 정책. Linux 6.16 EDAC scrub_subsystem 으로 mainstream upstream 됨 (2025-08, [docs.kernel.org/edac/scrub.html](https://docs.kernel.org/edac/scrub.html)). sysfs `/sys/bus/edac/devices/<dev>/scrubX/cycle_duration` interface. CXL Type-3 device 가 internally read-correct-writeback 수행 — host bandwidth 영향 0, ECS mailbox query 만 host 측 latency overhead (ms 단위).
**관련 논문**: [CXL 3.1 RAS Whitepaper](https://computeexpresslink.org/wp-content/uploads/2024/08/An-Overview-of-RAS-for-Compute-Express-Link-3.1-Whitepaper.pdf) / [Linux 6.16 EDAC docs](https://docs.kernel.org/edac/scrub.html) / [Phoronix 2025 CXL upstream](https://www.phoronix.com/news/Linux-6.16-CXL)
**관련 아이디어**: P1 PrefixGuard (3-tier scrub interval) / V1 PrefixGuard-Lite (p75 lifetime calibration) (2026-04-26 세션)
**Open Questions**: (a) Linux 6.16 EDAC sysfs minimum interval 5min 의 vendor-specific 실측, (b) prefix lifetime histogram 의 p75 vs p90 alignment 의 corruption rate 차이, (c) CXL 3.2 Patrol Scrub 의 device-internal bandwidth 가 host bandwidth 와 contention 발생 여부.

## HBM3 PAT Counter (P4 PATroller / V4 origin)
**정의**: HBM3 JESD238B 의 **PAT (Pseudo-channel Activation Timing)** counter. pseudo-channel 단위 row activation count tracking. PRAC (Per-Row Activation Counter, DDR5 JESD79-5C April 2024) 의 HBM 버전. **IEEE 1500 Test Access Port (TAP)** 으로 host 가 mailbox query 가능 (vendor-specific register layout). PAT counter top-k row = hot row 식별 → KV cache reliability-aware migration trigger 로 재활용 가능. Meta Reliability ([arXiv:2410.21680](https://arxiv.org/abs/2410.21680)) 의 Llama-3 405B 16384 H100 cluster 54일 419 failure 중 HBM3 72건 (3hr 당 1건) — hot row 가 fault dominant origin.
**관련 논문**: [HBM3 JESD238B](https://www.jedec.org/standards-documents/docs/jesd238b01) / [MOAT arXiv:2407.09995](https://arxiv.org/abs/2407.09995) [HPCA 2025] / [QPRAC arXiv:2501.18861](https://arxiv.org/abs/2501.18861) [HPCA 2025] / [CnC-PRAC arXiv:2506.11970](https://arxiv.org/abs/2506.11970) [DSN 2025] / [Meta Reliability arXiv:2410.21680](https://arxiv.org/abs/2410.21680)
**관련 아이디어**: P4 PATroller (1s polling, top-32) / V4 PATroller-Solo (polling overhead profile, 미선정) (2026-04-26 세션)
**Open Questions**: (a) IEEE 1500 TAP register layout 의 NVIDIA H100 / SK hynix HBM3 / Samsung HBM3e vendor 별 차이, (b) PAT polling interval 1s default 의 8K activations/sec threshold 가 PRAC family threshold (10K-50K) 와 align 하는가, (c) NeuroSim V1.4 cell wear model 이 Meta C5 의 fault rate 와 quantitatively cross-check 가능한가.

## DPA Poison Tracking (P3 Quarantine / V3 origin)
**정의**: CXL 3.x spec 의 corrupted data 격리 mechanism. HPA (Host Physical Address) → DPA (Device Physical Address) translation 후 **device 가 corrupted DPA range 를 internally mark + Memory Event Record 로 host 에 report**. CXL Type-3 device 의 mailbox interface 로 query. AER (Advanced Error Reporting) 의 "memory error address 미log" 한계를 보완. multi-tenant CXL pool 에서 cross-tenant Rowhammer / SDC 의 격리 핵심.
**관련 논문**: [CXL 3.1 RAS Whitepaper §8.2.9.9.11.2](https://computeexpresslink.org/wp-content/uploads/2024/08/An-Overview-of-RAS-for-Compute-Express-Link-3.1-Whitepaper.pdf) / [CacheSolidarity arXiv:2603.10726](https://arxiv.org/abs/2603.10726) / [Targeted BFA on Agents arXiv:2603.10042](https://arxiv.org/abs/2603.10042)
**관련 아이디어**: P3 Quarantine (agent_id × DPA_range 2-level sparse hash) / V3 Quarantine-Mini (single-agent recompute latency profile) (2026-04-26 세션)
**Open Questions**: (a) CXL Type-3 device mailbox interface 의 vendor-specific layout 이 sim emulated 만으로 충분한가 (Phase 2 industrial deployment 추가 검증 필요), (b) agent_id × DPA_range 2-level sparse hash 의 worst-case 128MB → average 4-16MB 가정의 fill rate 1/8 이 multi-tenant cloud 실측과 일치, (c) vLLM RFC #19329 / vLLM-ascend RFC #5067 의 mainstream upstream timeline (Q2 2026 추정).

## Outlier-Aware ECC for KV Cache (KV cache reliability axis)
**정의**: KV cache 의 양자화 후 outlier 분포 (KIVI [arXiv:2402.02750](https://arxiv.org/abs/2402.02750) / KVQuant [arXiv:2401.18079](https://arxiv.org/abs/2401.18079) / VecInfer [arXiv:2510.06175](https://arxiv.org/abs/2510.06175) 등) 를 ECC strength 결정 input 으로 활용. 1% outlier channel 이 99% bit-flip sensitivity — strong ECC (DEC-3) 적용, 나머지 SEC-DED. Outlier 가 spatially clustered (특정 channel 에 누적) → bit-protection target 의 spatial locality 가짐. Pre-RoPE Key 에서 channel 단위 strong, post-RoPE 에서 RoPE rotation 으로 spatially scrambling.
**관련 논문**: [KIVI ICML 2024 arXiv:2402.02750](https://arxiv.org/abs/2402.02750) / [KVQuant NeurIPS 2024 arXiv:2401.18079](https://arxiv.org/abs/2401.18079) / [VecInfer arXiv:2510.06175](https://arxiv.org/abs/2510.06175) / [KITTY arXiv:2511.18643](https://arxiv.org/abs/2511.18643) / [Outlier Tokens Tracing ACL 2025 Findings](https://aclanthology.org/2025.acl-long.631.pdf)
**관련 아이디어**: v1 OAEP-KV (outlier-channel × ECC strength, 2026-04-25) / v2 P5 Watermark (KV-cache-specific lightweight CRC, 2026-04-26 세션)
**Open Questions**: (a) Hadamard transform 후 outlier magnitude variance 균일화 시 strong code 균일 적용 vs adaptive 의 trade-off, (b) Channel sensitivity ranking 이 calibration 100 prompt 만으로 95%+ stable 한가 (KITTY 측정), (c) outlier 가 token-axis × channel-axis 양쪽 분산 — ECC tagging 에 2D index 가 single index 보다 정확도 차이.

## KV Cache Block Granularity Page Retirement (BlockShard origin, v1)
**정의**: vLLM PagedAttention 의 16-token block (≈ 512B-2KB INT4 / 4KB FP16) 단위로 page retirement 수행하는 RAS 정책. Linux mcelog 의 4KB / 2MB hugepage 단위 retirement 가 8-64× stranding 발생. block table 의 logical→physical mapping update 만으로 부분 retire 가능. **vLLM 가 ECC granularity unit 을 OS page (4KB) 가 아닌 KV block (16-token) 으로 재정의**. v1 BlockShard 의 핵심 axis. v2 P3 Quarantine 의 agent_id × DPA_range 격리 정책의 base.
**관련 논문**: [vLLM PagedAttention SOSP 2023 arXiv:2309.06180](https://arxiv.org/abs/2309.06180) / [Linux mcelog hard-offline](http://www.mcelog.org/badpageofflining.html) / [vLLM RFC #19329 graceful KV connector error](https://github.com/vllm-project/vllm/issues/19329)
**관련 아이디어**: v1 BlockShard (Linux soft-offline ABI, 2026-04-25) / v2 P3 Quarantine (agent_id × DPA, 2026-04-26)
**Open Questions**: (a) Linux soft-offline ABI 가 sub-page granularity 로 확장 가능한가 (kernel patch 필요 vs application-level emulation), (b) vLLM RFC #19329 token-range recompute path 의 vLLM-ascend RFC #5067 mainstream upstream timeline.

## KVSink Sink Token (P2 SinkShield origin, 미선정)
**정의**: Attention sink token 의 활성 outlier 기반 dynamic 예측. StreamingLLM ([arXiv:2309.17453](https://arxiv.org/abs/2309.17453)) 의 fixed first-N (PFN) 대비 향상. KVSink ([arXiv:2508.04257](https://arxiv.org/abs/2508.04257), COLM 2025) 가 plug-and-play 형식. SinkQ ([OpenReview bJ33TvbJW0](https://openreview.net/forum?id=bJ33TvbJW0)) 가 2-bit KV quant + dynamic sink tracking. 본 세션 P2 SinkShield 가 hardware refresh axis (HBM3 RFM frequency boost) 로 attention sink 보존을 시도했으나 KVSink/SinkQ concurrent (motivation 동일) + v1 EntropyECC overlap 으로 미선정.
**관련 논문**: [StreamingLLM ICLR 2024 arXiv:2309.17453](https://arxiv.org/abs/2309.17453) / [KVSink COLM 2025 arXiv:2508.04257](https://arxiv.org/abs/2508.04257) / [SinkQ OpenReview bJ33TvbJW0](https://openreview.net/forum?id=bJ33TvbJW0) / [Visual Attention Sink arXiv:2503.03321](https://arxiv.org/abs/2503.03321)
**관련 아이디어**: P2 SinkShield (미선정, 2026-04-26 세션) — 재방문 조건: KVSink stack-able layer reposition.
**Open Questions**: (a) HBM3 RFM granularity (row/bank-level) 와 sink token block (16-token) 의 mapping aggregation 정확도, (b) attention sink dynamic 예측 의 quantization layer (KVSink) 와 hardware refresh layer (P2) stack 가능성.

---

## Adaptive Adversarial Robustness of Discrete Routing (DISCRETE-VEIL origin)
**정의**: MoE router 의 top-k selection 은 **discrete argmax** 연산이므로, continuous hidden state (dense LLM 의 residual stream) 에 대한 embedding-space PGD 공격 (e.g., Obfuscated Activations 가 recall 100→0 로 파괴) 과 **질적으로 다른 attack surface** 를 가질 수 있다는 가설. DRO-Attack 은 joint loss `L = CE(jailbreak) + λ·MSE(topk_surrogate, benign_pattern)` 를 Gumbel-softmax hard=True surrogate 로 discrete argmax 를 미분 가능하게 풀어 PGD 적용. **핵심 측정**: "clean WildJailbreak recall → DRO-attack 후 recall" drop 이 dense hidden-state linear probe 대비 얼마나 작은가 (목표: > 30%p 격차 시 discrete-robust 가설 성립).
**관련 논문**: [Obfuscated Activations arXiv:2412.09565](https://arxiv.org/abs/2412.09565) (dense probe 무력화) / [V-MoE Adversarial Robustness OpenReview Fd05J4Bu5Sp](https://openreview.net/pdf?id=Fd05J4Bu5Sp) (image PGD, 가설과 상충 가능) / [GateBreaker arXiv:2512.21008](https://arxiv.org/abs/2512.21008) (weight-level ablation) / [Expert Selections Reveal arXiv:2602.04105](https://arxiv.org/abs/2602.04105) (privacy attack)
**관련 아이디어**: DISCRETE-VEIL' (2026-04-24 세션) / DISCRETE-VEIL-Lite (Tier-2 spinoff)
**Open Questions**: (a) Gumbel-softmax temperature τ 가 작아질수록 PGD 수렴 얼마나 어려워지는가 (combinatorial hardness 수식화), (b) text embedding PGD 는 image PGD (V-MoE) 와 attack 공간 진짜 구분되는가 (Venn diagram 필수), (c) 2 모델 (Qwen3 + Mixtral) 간 robustness 차이의 원인이 expert 수 (128 vs 8) 에 있는가 또는 top-k (8 vs 2) 에 있는가.

## MoE Expert-Activation Fingerprint (source experiment origin)
**정의**: MoE LLM 의 forward 1회 (`max_tokens=1`) 만으로 각 MoE layer 에서 4 종 tensor (`router_logits` / `softmax_scores` / `topk_weights` / `activation_count`) 를 ctx-mean pool → prompt 당 `(L, E)` 행렬로 얻는 training-free 관측치. **모델 수정 없음**, vLLM custom hook 만 필요. Domain 분류 (MMLU 4-cat 96.2%) + safety 분류 (WildJailbreak 2-way 94.0%) 를 동일 fingerprint 에서 k-NN 으로 처리. 이방산 학생 (SSU AICA Lab) 2026-04-18~24 실험 4 모델 × 608 configs × ~5,900 runs 기준.
**관련 논문 (인접)**: [Task-Cond. Routing arXiv:2603.11114](https://arxiv.org/abs/2603.11114) (domain 92.5%, OLMoE single) / [MoE Lens arXiv:2603.05806](https://arxiv.org/abs/2603.05806) (top-1 cosine 0.95) / [RouteMark arXiv:2508.01784](https://arxiv.org/abs/2508.01784) (JS divergence IP) / [Expert Selections Reveal arXiv:2602.04105](https://arxiv.org/abs/2602.04105) (privacy attack)
**관련 아이디어**: DISCRETE-VEIL' / LOOM' / BEACON-GUARD-Lite (2026-04-24 세션)
**Open Questions**: (a) Qwen1.5-MoE 의 routing 변별력 제한 (Non-MoE+Shared 13%) 은 model-specific 인가 model-class-specific 인가, (b) 판별 layer 위치 (Qwen3 L22/48 vs Qwen1.5 L02/24) 가 fingerprint 정보량의 분포 위상을 정의하는가, (c) cross-model fingerprint alignment (Procrustes/CCA) 가 본 실험 데이터에서 가능한가.

## Token × Layer 2D Early-Exit Pareto (LOOM' M2 origin)
**정의**: MoE fingerprint convergence 를 **token prefix length k*** (k=1, 4, 16, 64, full) 와 **layer depth L_k** (L=2, 5, 10, 22, full) 두 축에서 joint Pareto frontier 로 최적화하는 framework. FJD ([arXiv:2509.14558](https://arxiv.org/abs/2509.14558)) 는 token axis k=1 만, HSF / HiddenDetect 는 layer axis L=final 만 사용 → 2D joint 는 미개척. Adaptive layer budget (margin m_high/m_low) runtime 으로 per-prompt 별 최적 (k, L_k) 적응.
**관련 논문**: [FJD arXiv:2509.14558](https://arxiv.org/abs/2509.14558) [EMNLP 2025 Findings] (token axis k=1) / [HSF arXiv:2409.03788](https://arxiv.org/abs/2409.03788) (layer axis) / [HiddenDetect arXiv:2502.14744](https://arxiv.org/abs/2502.14744) (VLM layer) / [Do Internal Layers Reveal arXiv:2510.06594](https://arxiv.org/abs/2510.06594)
**관련 아이디어**: LOOM' M2 (2026-04-24 세션, EMBER+THRESHOLD 흡수)
**Open Questions**: (a) Token axis convergence 가 layer axis 보다 모델 크기에 더 민감한가 (Qwen3 vs Qwen1.5), (b) 2D joint 최적 (k*, L_k*) 이 task (domain / safety) 에 따라 다른가, (c) batch-packed early-exit (BPEE) 의 mid-layer slot swap 이 KV cache 무결성에 영향 없는가.

## Information-Theoretic Shared-Substrate Pareto (LOOM' systems-theory origin)
**정의**: MoE fingerprint F 를 detection / residency / prefetch 등 N 개 consumer 에 공유 공급하는 시스템에서, F 의 Fisher information 이 N task 에 분배 될 때의 Pareto upper bound 를 `I_max(task_i) ≤ trace(F^{-1}) · eigenvalue(task_correlation)` 형태로 bound 하는 systems-theory framework. 단순 "shared read 공짜" engineering 주장 (vLLM Semantic Router Iris 수준) 을 넘어서 **task correlation matrix 의 eigenvalue 분포** 에 의존하는 formal guarantee 를 제공.
**관련 논문**: vLLM Semantic Router Iris v0.1 (2026-01 production) / [MoE-Infinity arXiv:2401.14361](https://arxiv.org/abs/2401.14361) [ATC 2024] / [DuoServe arXiv:2509.07379](https://arxiv.org/abs/2509.07379) / [Gimbal arXiv:2602.21626](https://arxiv.org/abs/2602.21626) [MLSys 2026]
**관련 아이디어**: LOOM' overall reframe (2026-04-24 세션)
**Open Questions**: (a) 3 task (detection + residency + prefetch) 의 correlation 이 실제로 높은가 낮은가 (기존 5,900 runs 로 측정 가능), (b) Pareto proof 가 workstation 2-replica scope 에서 실증 가능한가 multi-replica 에서만 의미 있는가, (c) Fisher info 분배 bound 가 실제 latency/throughput gain 과 correlate?

## DeepStack Multi-Layer Visual Residual Injection (Qwen3-VL integration)
**정의**: Vision encoder 의 3 level feature (low/mid/high) 를 LLM decoder 의 layer 8/16/24 (HF 공식 `deepstack_visual_indexes=[8,16,24]`) 에 residual 로 주입하는 구조. Context length 늘리지 않고 multi-level visual information 전달. [DeepStack NeurIPS 2024 arXiv:2406.04334](https://arxiv.org/abs/2406.04334) 원본에서 제안, Qwen3-VL ([arXiv:2511.21631](https://arxiv.org/abs/2511.21631)) 에 공식 integration. **Key insight**: layer 0-7 은 vision-independent, layer 8/16/24 가 injection point 이므로 sub-graph partitioning + DLA offload 가능.
**관련 논문**: [DeepStack NeurIPS 2024](https://arxiv.org/abs/2406.04334) / [Cross-Layer Injection arXiv:2601.10710](https://arxiv.org/abs/2601.10710) / [Qwen3-VL Technical Report](https://arxiv.org/abs/2511.21631) / [HF transformers Qwen3-VL modeling](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_vl/modeling_qwen3_vl.py)
**관련 아이디어**: I1 Mangrove (4-stage sub-graph + LPDDR bank + DLA offload) / I3 Vault' (DeepStack × MoE L2 contention) (2026-04-24 세션)
**Open Questions**: (a) deepstack_visual_indexes sweep 의 optimal 이 모델 크기별 다른가? (b) Injection point 가 많을수록 DLA offload 이점 증가하는가 vs memory sync overhead? (c) MoE variant 에서 vision residual 이 expert L2 residency 를 어떻게 교란하는가.

## Interleaved MRoPE (Qwen3-VL origin)
**정의**: Qwen3-VL 의 positional encoding, Qwen2-VL/2.5-VL 의 chunked MRoPE (t/h/w 를 embedding dimension chunk 로 분할) 대신 **t/h/w 를 low/high frequency band 에 round-robin interleave**. Qwen3-VL 공식 구현 `[24,20,20]`. 기존 chunk 방식의 frequency spectrum imbalance 해소 + long video 성능 향상. **Kernel-level implication**: chunked 3-table LUT 가 아닌 single unified permuted LUT 필요.
**관련 논문**: [Qwen3-VL Technical Report arXiv:2511.21631](https://arxiv.org/abs/2511.21631) / [Revisiting MRoPE ICLR'26 arXiv:2510.23095](https://arxiv.org/abs/2510.23095) / [vLLM PR #22593 partial rotary](https://blog.vllm.ai/2025/09/11/qwen3-next.html)
**관련 아이디어**: I2 Loom (unified LUT + FA3 fused rotation + texture unit) (2026-04-24 세션)
**Open Questions**: (a) Permuted LUT size 의 optimal 이 edge GPU L1/L2 에 어떻게 fit? (b) FA3 tile-internal fused rotation 이 Blackwell FP4 path 와 호환? (c) Interleaved pattern 이 Kimi Linear 등 linear attention 과 결합 가능성.

## DeepStack × MoE L2 Contention (Vault' origin, VLM-MoE edge coupling)
**정의**: Qwen3-VL-30B-A3B 같은 MoE variant 에서 DeepStack vision residual 이 layer 8/16/24 에 주입될 때, 해당 layer 의 expert L2 residency 를 교란하는 문제. Vision token 의 access pattern 이 expert weight cache line 을 evict 하여 top-2 gating 의 L2 hit rate 저하. **Activation-aware L2 pinning** + **GDN dual working-set zone** 으로 대응. 본 개념은 VLM-MoE + edge (Jetson Thor) 교집합 공백.
**관련 논문**: [Qwen3-VL MoE 30B-A3B](https://docs.vllm.ai/projects/ascend/en/latest/tutorials/models/Qwen3-VL-30B-A3B-Instruct.html) / [Jetson Thor vLLM MoE gap forum 2026-02](https://forums.developer.nvidia.com/t/jetson-agx-thor-vllm-26-02-moe-performance-significantly-below-reference-missing-fused-moe-config/364663) / [VEQ arXiv:2602.01037](https://arxiv.org/abs/2602.01037) / [DyMoE arXiv:2603.19172](https://arxiv.org/abs/2603.19172)
**관련 아이디어**: I3 Vault' (Major Revision post-NVFP4 scoop) (2026-04-24 세션)
**Open Questions**: (a) Analytical model 의 input 이 무엇인가 (visual token rate × expert activation frequency × L2 way count)? (b) Thor LPDDR5X 128GB bank 단위 expert placement 의 최적 분할은? (c) GDN recurrent state 의 L2 residency 가 expert pinning 과 어떻게 trade-off?

## Gated Delta Net (GDN) Constant-Memory Fast Path (Gale origin)
**정의**: Qwen3-Next/3.5 에 도입된 3:1 GDN:Attention hybrid 에서 GDN layer 는 linear attention 의 constant-memory recurrent state 만 유지 → 256K long context 에서도 KV paging 불필요. Attention 8 layer 만 KV cache 필요 → 3-tier storage (GPU/DRAM/NVMe) 에 분산 가능. DeepStack layer-aware eviction priority 추가 시 edge 256K context 실현 가능.
**관련 논문**: [Gated Delta Networks ICLR 2025 arXiv:2412.06464](https://arxiv.org/abs/2412.06464) / [Qwen3.5-Omni Tech Report](https://arxiv.org/abs/2604.15804) / [vLLM Qwen3-Next blog](https://blog.vllm.ai/2025/09/11/qwen3-next.html) / [Kimi Linear + KDA]
**관련 아이디어**: T1 Gale (2026-04-24 세션)
**Open Questions**: (a) Qwen3-VL 이 GDN 공식 포함하는지 (Qwen3-Next/3.5 는 확실, VL variant 공식 config 미확인). (b) GDN recurrent state 의 L2 residency trade-off (Forge 와 결합 시). (c) DeepStack layer priority eviction 의 formal policy (layer 8/16/24 은 NVMe 강퇴 금지?).

## Qwen3.5-Omni Thinker-Talker Heterogeneous (Forge origin)
**정의**: Qwen3.5-Omni 의 Thinker (MoE + GDN LLM, text 생성) 와 Talker (multi-codebook RVQ + MTP + Code2Wav causal ConvNet, speech 생성) 을 분리한 구조. ARIA 로 text-speech unit dynamic align. Edge 에서 Thinker → Tensor Core, Talker Code2Wav → DLA (ConvNet 기반 → DLA 완전 적합), GDN recurrent state → L2 resident 매핑으로 first-packet latency 감소.
**관련 논문**: [Qwen3.5-Omni Tech Report arXiv:2604.15804](https://arxiv.org/abs/2604.15804) / [Qwen3-Omni Tech Report arXiv:2509.17765](https://arxiv.org/abs/2509.17765) / [Qwen2.5-Omni arXiv:2503.20215](https://arxiv.org/abs/2503.20215)
**관련 아이디어**: T2 Forge (2026-04-24 세션)
**Open Questions**: (a) Code2Wav ConvNet kernel 의 NVDLA 2.0 지원 범위? (b) ARIA dynamic alignment 가 single-Jetson 에서 어떤 sync overhead? (c) Thinker MoE + GDN + Talker DLA + L2 configuration 의 Pareto 선.

## AnyRes Tile-Count Unifying Signal (Parquet origin)
**정의**: LLaVA-NeXT / Qwen2.5-VL / InternVL3 의 AnyRes (multi-resolution tiling) 에서 이미지마다 생성되는 **visual token tile 수 (1-12 가변)** 를 하나의 scheduler-level signal 로 사용하여 (1) CUDA Graph bucket 선택, (2) GPU+DRAM coupled DVFS, (3) per-tile precision (FP8/INT8) 을 결정하는 unifying control axis. Tile 많을수록 memory-bound, 적을수록 compute-bound 라는 실측 observation 에 기반. 2026-04-23 Parquet Tier-1 Top 1 의 primary mechanism.
**관련 논문**: [DynamoLLM HPCA'25](https://arxiv.org/abs/2408.00741) (coupled DVFS), [PolyThrottle MLSys'24](https://arxiv.org/abs/2310.19991) (DNN throttle), [MBQ CVPR'25](https://arxiv.org/abs/2412.19509) (per-layer quantization — Parquet 는 per-tile 확장), [BiScale](https://arxiv.org/abs/2602.18755), [throttLL'eM](https://arxiv.org/abs/2408.05235), [SparseDVFS](https://arxiv.org/abs/2603.21908), [vLLM v1 Multimodal CUDA Graph docs](https://docs.vllm.ai/en/latest/design/cuda_graphs_multimodal/)
**관련 아이디어**: Parquet Tier-1 Top 1 (2026-04-23 세션)
**Open Questions**: (a) vLLM v1 EncoderDisagg merge 이후 tile-count bucket 의 정확한 구현 경로는? (b) Per-tile entropy 를 SigLIP attention weight proxy 로 구하는 overhead 는 tile count 와 어떤 관계인가? (c) Coupled GPU+DRAM DVFS 의 transition latency 가 tile-count 변화 주기와 어떻게 interact 하는가?

## Jetson DLA-Aware VLM Serving (Triptych origin)
**정의**: Jetson Orin AGX 의 DLA (Deep Learning Accelerator, NVDLA 2.0) 를 VLM 의 vision encoder (ViT / SigLIP / InternViT) INT8 inference 에 할당하여 desktop GPU 대비 2-3× throughput/W 이점을 확보하고, projector 는 GPU Tensor core FP16, LLM 은 NVFP4 (Thor) 또는 INT4 (Orin) 로 modality-stage 별 이기종 dataflow 구성. UMA (Unified Memory Architecture) 로 activation hand-off 시 memcpy 제거. NVDLA 2.0 의 sub-kernel preemption 으로 multi-image batch pipeline bubble 제거. 2026-04-23 Triptych Tier-1 Top 2 의 핵심.
**관련 논문**: [Nova arXiv:2509.21301](https://arxiv.org/abs/2509.21301) (desktop GPU only, DLA 없음), [Nanomind arXiv:2510.05109](https://arxiv.org/abs/2510.05109) (tiny model module scheduling, DLA 없음), [HeteroInfer arXiv:2501.14794](https://arxiv.org/abs/2501.14794) (server-class), [llm.npu arXiv:2407.05858](https://arxiv.org/abs/2407.05858) (mobile NPU), [LiteVLM arXiv:2506.07416](https://arxiv.org/abs/2506.07416) (edge VLM single compute unit), [FastVLM CVPR'25](https://arxiv.org/abs/2412.13303), [NVIDIA DLA blog](https://developer.nvidia.com/blog/maximizing-deep-learning-performance-on-nvidia-jetson-orin-with-dla/) (convolution only 언급)
**관련 아이디어**: Triptych Tier-1 Top 2 (2026-04-23 세션)
**Open Questions**: (a) NVDLA 2.0 의 transformer attention 지원 범위와 성능 ceiling 은? (b) UMA zero-copy 가 PCIe 가 없어도 실제 latency gain 이 있는가 vs CUDA cudaMallocManaged 기본 경로의 차이? (c) DLA preemption granularity 가 multi-image VLM batch scheduling 에 적절한가?

## MRoPE Tri-Axial LUT (Cartographer origin)
**정의**: Qwen2-VL 이후 VLM 에 도입된 MRoPE (Multi-dimensional RoPE) 의 3-axial positional encoding (time × H × W) 을 runtime 계산 대신 precomputed LUT 치환하여 edge GPU 의 SFU (Special Function Unit) 점유를 줄이고 memory BW 감소. LPDDR row-aligned layout 과 결합하여 row-hit 증가. 1D RoPE (LLM) 와 수학적 구조가 달라 기존 T-MAC / LUT Tensor Core 등 weight-LUT 연구와 orthogonal. 2026-04-23 Cartographer Tier-2 독립 Top 1.
**관련 논문**: [T-MAC EuroSys'25](https://arxiv.org/abs/2407.00088) (weight mpGEMM LUT), [LUT Tensor Core ISCA'25](https://arxiv.org/abs/2408.06003) (weight LUT), [RotateKV](https://arxiv.org/abs/2501.16383) (rotation-based, LUT 미치환), [SAIL](https://arxiv.org/abs/2509.25853) (SRAM-LUT GEMV), [Revisiting MRoPE](https://arxiv.org/abs/2510.23095) (MRoPE variant algorithmic), [VideoRoPE](https://arxiv.org/abs/2502.05173), [VRoPE](https://arxiv.org/abs/2502.11664), [Qwen2.5-VL Tech Report](https://arxiv.org/abs/2502.13923) (MRoPE 3D 원출처)
**관련 아이디어**: Cartographer Tier-2 독립 Top 1 (2026-04-23 세션)
**Open Questions**: (a) MRoPE LUT size {256, 512, 1024, 2048} 의 Pareto 선은 edge GPU 별로 어떻게 다른가? (b) LUT approximation error 가 bit-exact 을 유지하려면 table fidelity 어느 수준? (c) Jetson Orin LPDDR row (보통 1-2KB) 과 MRoPE LUT 의 정합 layout 실측.

## Entropy-Adaptive Pixel Shuffle (Sift origin)
**정의**: InternVL3 / SmolVLM 의 projector-이전 pixel shuffle (2×2 고정, token 4× 압축) 을 patch attention entropy (pre-trained SigLIP attention weight proxy) 기반으로 ratio {2, 4, 8} 동적 선택. High-entropy patch 는 낮은 압축, low-entropy 는 높은 압축. Projector 이전 단계에 적용되어 downstream LLM 의 token count 을 대폭 감소시키는 VLM 고유 기법. 2026-04-23 Sift Tier-2 독립 Top 2.
**관련 논문**: [InternVL-X arXiv:2503.21307](https://arxiv.org/abs/2503.21307) (RVTC, entropy-adaptive 미포함), [PyramidDrop](https://arxiv.org/abs/2410.17247) (layer-wise drop, projector-이전 없음), [VisionZip](https://arxiv.org/abs/2412.04467) (ratio 고정), [FastV](https://arxiv.org/abs/2403.06764) (attention prune), [SparseVLM ICML'25](https://openreview.net/forum?id=80faIPZ67S), [SmolVLM](https://arxiv.org/abs/2504.05299) (pixel shuffle 고정)
**관련 아이디어**: Sift Tier-2 독립 Top 2 (2026-04-23 세션)
**Open Questions**: (a) Pre-trained SigLIP attention weight 를 entropy proxy 로 사용하는 overhead 는 배치당 얼마? (b) Ratio 8 까지 증가 시 MMMU accuracy drop curve 가 domain 별로 어떻게 다른가? (c) Tiny VLM (256M-2B) 과 medium VLM (7-8B) 의 Pareto 선 차이.

## HBM Row-Buffer Aligned KV Tile (HRTS origin)
**정의**: HBM3/HBM3e row buffer (8 KB) 의 물리 구조를 LLM/VLM KV cache page layout 에 노출하여, page 단위 (vLLM PagedAttention 기본 4 KB) 를 row-size 와 정합시키는 tile 설계. Row-hit rate 62% (default) → 82-88% (aligned) 의 측정-중심 novelty. Analytical row-hit model + Nsight Compute `dram__sectors_read.sum` 으로 reverse-engineering 후 tile size {128, 256, 512} sweep. 2026-04-23 v3 세션에서 HRTS+ Top 1 의 primary mechanism.
**관련 논문**: [vLLM PagedAttention SOSP 2023](https://arxiv.org/abs/2309.06180), [HERMES arXiv:2601.14724](https://arxiv.org/abs/2601.14724), [Ramulator2](https://github.com/CMU-SAFARI/ramulator2), Mosaic [arXiv:2604.10060](https://arxiv.org/abs/2604.10060) (content-axis vs physical-axis 차별화)
**관련 아이디어**: HRTS+ Tier-1 Top 1 / HRTS Tier-2 paper-pair (2026-04-23 v3 세션)
**Open Questions**: (a) Blackwell HBM3e row boundary 가 Hopper HBM3 과 같은지 vendor 미공개 — reverse-engineering cross-check 필요. (b) Row-aligned tile 이 Mosaic content-axis clustering 과 stacking 시 orthogonality empirical 증명 — +5-8% 추가 가정의 실증. (c) NVMe cold tier 의 wall-clock reproducibility — analytical model vs 실측 consistency.

## CLIP-LSH Reuse Graph (ContextMIG origin)
**정의**: Multi-tenant VLM serving 에서 request 간 visual-prefix reuse 를 CLIP-L last-pool embedding 의 16-bit SimHash 로 fuzzy 매칭, sliding window (256 req) 내 edge density 가 ≈ 0.31 임을 활용한 reuse 그래프. Exact token-level match 를 요구하는 vLLM PagedAttention refcount 와 달리 cosine ≥0.92 의 semantic-near duplicate 를 포착. 업데이트 latency < 0.6 ms/req. Mosaic 의 cross-modal clustering 과 차별화: Mosaic = KV dedup only, CLIP-LSH Reuse Graph = MIG placement + phase coalesce 와 결합.
**관련 논문**: Mosaic [arXiv:2604.10060](https://arxiv.org/abs/2604.10060), KVShare [arXiv:2503.16525](https://arxiv.org/abs/2503.16525), MPIC [arXiv:2502.01960](https://arxiv.org/abs/2502.01960), Semantic Scheduling [arXiv:2506.12204](https://arxiv.org/abs/2506.12204), Predictable LLM Serving [arXiv:2508.20274](https://arxiv.org/abs/2508.20274)
**관련 아이디어**: ContextMIG+ Tier-1 Top 2 / ContextMIG Tier-2 paper-pair (2026-04-23 v3 세션)
**Open Questions**: (a) 16-bit SimHash bucket collision 이 실제 workload 에서 ≤ 3% 달성 가능한지. (b) CLIP-L 대신 SigLIP 이나 더 small embedding 으로 latency 감축 가능성. (c) Hash table maintenance overhead 가 tenant 수 (2/4/8) 와 sub-linear 관계인가.

## SSE + Page-Hinkley Phase Predictor (PhaseGraph-VLA origin)
**정의**: L_mid hidden state 의 L2 drift + EWMA baseline 을 Page-Hinkley change-point test (soft pre-warm / hard evict 2-threshold) 로 모니터링, VLA trajectory phase (Approach / Manipulate / Retract) boundary 를 < 100 μs 에 감지하는 predictor. Training-free quantile calibration (첫 100 frame). VLA 에서는 gripper Δ, trajectory curvature, DINOv2 object distance (1.2ms/frame) 를 additional feature 로 결합. FlashVLA 의 token-level reuse 와 축 분리 — SSE 는 graph-level switch 를 위한 boundary detection.
**관련 논문**: Page-Hinkley (1954), VLA-Cache [NeurIPS 2025](https://openreview.net/forum?id=QZYZ0Xm58q), FlashVLA [arXiv:2505.21200](https://arxiv.org/abs/2505.21200), AC²-VLA [arXiv:2601.19634](https://arxiv.org/abs/2601.19634), Running-VLAs [arXiv:2510.26742](https://arxiv.org/abs/2510.26742), DINOv2 [arXiv:2304.07193](https://arxiv.org/abs/2304.07193)
**관련 아이디어**: PhaseGraph-VLA+ Tier-1 Top 3 / PhaseGraph-VLA Tier-2 paper-pair (2026-04-23 v3 세션)
**Open Questions**: (a) PH FP rate < 5% 가 LIBERO 5 task 에서 실제 empirical 하게 달성되는지. (b) VLA 외 domain (VLM chat multi-turn) 으로의 SSE 일반화 가능성. (c) Hysteresis {2, 3, 4 frames} 의 optimal 이 VLA control frequency 에 의존하는가.

## Green Context μs-level Reconfig (CUDA 12.5+)
**정의**: NVIDIA CUDA 12.5 에서 도입된 `cuDevSmResourceSplit` API 를 통해 SM partition 을 **μs-level** (MIG 의 ms-level 과 4 orders 격차) 에 재구성하는 기능. `cuCtxFromGreenCtx` 로 green context 획득 후 새 SM partition 즉시 가용. VLM phase-transition-heavy workload (prefill↔decode↔vision-encode) 에서 per-request reshape 가 가능해진다. NVIDIA docs 는 "fast reconfig" 라고만 표기하고 수치 공개 없음 — B1 GCReconfProfile 이 Blackwell vs Hopper cross-arch characterization 공개.
**관련 논문**: [NVIDIA CUDA 12.5 Green Contexts docs](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/green-contexts.html), [MIGER ICPP 2024](https://dl.acm.org/doi/pdf/10.1145/3673038.3673089), [Managing MIG arXiv:2508.18556](https://arxiv.org/abs/2508.18556), [LithOS SOSP 2025](https://dl.acm.org/doi/10.1145/3695053.3731083), [Execution-Idle arXiv:2604.04745](https://arxiv.org/abs/2604.04745)
**관련 아이디어**: ContextMIG+ M2 (Tier-1 Top 2) / B1 GCReconfProfile (Tier-2 독립 Top 1) (2026-04-23 v3 세션)
**Open Questions**: (a) Driver serialization 이 SM-delta 에 quadratic growth 하는지 empirical 가설 검증. (b) Blackwell 과 Hopper 의 reconfig cost 가 driver firmware 에 의존하는가 아니면 PHY 에 의존하는가. (c) Pending kernel queue drain 이 reconfig latency 의 p99 tail 에 미치는 비중.

## Eviction Energy Decomposition (B2 TokenEvictEnergy origin)
**정의**: Visual token eviction policy (VL-Cache, SparseVLM, random) 의 energy impact 를 (HBM dynamic / SM static / DRAM PKG / refresh-implied) 4-component 으로 분해하는 methodology. NVML 5ms window + Intel RAPL `intel-rapl:0:0/energy_uj` dual-counter crawling. **Hidden insight**: aggressive eviction 이 HBM row-buffer locality 상실 → DRAM refresh 증가 → DRAM PKG energy 오히려 +2~+6%. "Eviction ≠ always green" 공교육 negative result.
**관련 논문**: VL-Cache [ICLR 2025](https://openreview.net/forum?id=HMrcv7Q4Ub), SparseVLM [ICML 2025](https://openreview.net/forum?id=80faIPZ67S), [Characterizing Power Management for LLMs ASPLOS 2024](https://www.microsoft.com/en-us/research/wp-content/uploads/2024/03/GPU_Power_ASPLOS_24.pdf), [TokenPowerBench arXiv:2512.03024](https://arxiv.org/abs/2512.03024)
**관련 아이디어**: B2 TokenEvictEnergy (Tier-2 독립 Top 2) (2026-04-23 v3 세션)
**Open Questions**: (a) HBM3e refresh counter 가 Blackwell 에서 CUPTI/Nsight 으로 노출되는지. (b) Refresh-implied energy 의 lower bound 를 analytical model 로 tight 하게 bound 가능한가. (c) Eviction policy 가 DRAM row-buffer locality 에 미치는 영향의 analytical model (row-buffer hit rate × refresh overhead 교차).

## VLA Action-Head Fused Kernel (B3 ActHeadFuse origin)
**정의**: OpenVLA-OFT 의 action-head `SiLU → Linear(hidden 4096 → 448 action logits) → torch.bucketize` 3-op serial chain 을 **single persistent CUDA kernel** 로 fuse. Kernel launch overhead 각 3-5 μs × 3 op = 9-15 μs → single launch 2-3 μs. Decode step 1.82 → **0.94 ms (1-kHz real-time control 가능)**. Block-tile (hidden=4096 / 32-warp) + TMA (Hopper/Blackwell) + warp-level softmax-free bucketize. OpenVLA-OFT 의 26× 가속은 parallel-sample level 이지 kernel-level 이 아니라는 gap 보완.
**관련 논문**: OpenVLA [CoRL 2024](https://proceedings.mlr.press/v270/kim25c.html), OpenVLA-OFT [arXiv:2502.19645](https://arxiv.org/abs/2502.19645), FAST [arXiv:2501.09747](https://arxiv.org/abs/2501.09747), VLA-Cache [NeurIPS 2025](https://openreview.net/forum?id=QZYZ0Xm58q), [CUTLASS 3.6+ TMA epilogue](https://github.com/NVIDIA/cutlass)
**관련 아이디어**: B3 ActHeadFuse (Tier-2 독립 Top 3) (2026-04-23 v3 세션)
**Open Questions**: (a) OpenVLA 외 VLA family (π0, RT-2) 의 action-head 구조가 fused kernel 로 일반화 가능한가. (b) TMA async memory 가 Blackwell RTX 5090 (consumer) 에서 full support 되는가. (c) FP16 vs BF16 bit-exact 일치 검증의 action MSE 의 임계치.

## Decompose-then-Merge Scaling Factor (PRISM origin)
**정의**: Training 시에 scaling factor matrix S ∈ R^(P_N × P_M) 을 rank-r 로 분해 S = S1·S2 (S1 ∈ R^(P_N × r), S2 ∈ R^(r × P_M)) 하여 r×(P_N+P_M) 파라미터로 expressive power 확장, Inference 시 S = S1·S2 를 pre-compute 후 per-tile scalar 하나로 merge 하여 overhead 0. 본질은 "training fidelity 와 inference cost 를 decouple" 하는 rank-decoupling 원리. CIM-BNN PRISM(ISLPED'26) 에서 제안, BN-Free 의 dynamic range 제약 해소 효과 (σ 2.44→4.42, 1.66→8.01). 본 세션에서 VLM ternary QAT (I1') / VLM KV cache (I2') / PIM dequant LUT (I4') 로 확장.
**관련 논문**: PRISM ISLPED'26 submission, MDBF [arXiv:2512.24545](https://arxiv.org/abs/2512.24545) (weight-space factorization), LoRDS [arXiv:2601.22716](https://arxiv.org/abs/2601.22716) (continuous scale S=BA LLM-only), ARB-LLM [arXiv:2410.03129](https://arxiv.org/abs/2410.03129) (rank-1 row×col), GEAR [arXiv:2403.05527](https://arxiv.org/abs/2403.05527) (error residual low-rank)
**관련 아이디어**: I1' TernVLM-RankSF, I2' TernVLM-KV-LUT, I4' PRISMKV-PIM-DequantLUT (2026-04-23 세션)
**Open Questions**: (a) Scale-space Hessian vs weight-space Hessian (MDBF 차별화 축) 의 empirical sensitivity 차이 측정 부재. (b) rank-r SF 가 VLM modality-split 에서 vision wide dynamic range 복원하는 mechanism 의 수학적 증명 부재 (Lipschitz upper bound 만). (c) Inference merge 후 flat SF 의 per-tile distortion ε 이 누적되는 attention 깊이에서 bound 가 O(L·ε) 이 typical 인지 more-tight 가능한지.

## OPTIC (Outlier-preserving Threshold-based Inlier Compression, PRISM origin)
**정의**: SF 분포의 inlier (평균 근처 τ·σ 이내) 를 single representative value μ_in 으로 대체하고, outlier (distribution tail) 는 원본 preservation 하는 quantile-based compression. SF 개수를 target compression ratio ρ 까지 줄이되 accuracy 는 유지. PRISM 에서 26-59.5% SF 감소 + +1.07-2.04% accuracy 개선. 본 세션에서 **saliency-weighted OPTIC** (attention entropy × gradient magnitude) 로 I2' 에 확장, contact-specific saliency 로 I3' 에 확장, mobile-VLM k-means clustering 으로 T1' OPTIC-SF-Lite 에 직접 이식.
**관련 논문**: PRISM ISLPED'26, BiLLM [arXiv:2402.04291](https://arxiv.org/abs/2402.04291) (salient weight preservation), PrefixQuant [arXiv:2410.05265](https://arxiv.org/abs/2410.05265) (static outlier isolation), HIGGS [arXiv:2411.17525](https://arxiv.org/abs/2411.17525) (RHT near-normal distribution), AKVQ-VL [arXiv:2501.15021](https://arxiv.org/abs/2501.15021) (pivot token saliency), Oaken [arXiv:2503.18599](https://arxiv.org/abs/2503.18599) (offline outlier threshold).
**관련 아이디어**: I2' TernVLM-KV-LUT, I3' PRISM-VLA-Temporal, T1' OPTIC-SF-Lite (2026-04-23 세션)
**Open Questions**: (a) Saliency metric 선택 (attention entropy / gradient magnitude / L2 norm) 이 OPTIC accuracy 에 미치는 impact Pareto 부재. (b) Cumulative distortion bound O(N·(1-τ)) 이 temporal 축에서 타 quantization 기법 (rotation, low-rank error) 와 어떻게 interact 하는지 미탐구. (c) Outlier 1% quota 가 VLM vision token 에서 optimal 인지 tile 크기 의존성 study 부재.

## LUT-Bypass Attention (decoupled from LUT-GEMM/weight LUT)
**정의**: Attention 의 Q·K^T 연산에서 K 가 binary/ternary quantization 되어 dot product 출력 분포가 near-normal (좁고 zero 집중) 일 때, Q sub-tile × K sub-tile dot product 를 precomputed LUT 으로 치환하는 kernel. PRISM 의 "freed buffer → LUT" 원리를 GPU attention 으로 포팅. sub-tile 크기 4 로 group 하면 K ternary LUT size 3^4 = 81 entries, Q INT8 quantize 시 LUT size 81 × 256 = 20,736 entries × 16-bit ≈ 40 KB per head (H100 shared memory 한도 내). FA3-dual path kernel 설계 (hot WGMMA / cold LUT) 로 tensor-core throughput 유지하며 extreme regime speedup 확보.
**관련 논문**: PRISM ISLPED'26 (origin), T-MAC [arXiv:2407.00088](https://arxiv.org/abs/2407.00088) (weight group LUT, CPU), LUT Tensor Core [arXiv:2408.06003](https://arxiv.org/abs/2408.06003) (weight mpGEMM GPU), SAIL [arXiv:2509.25853](https://arxiv.org/pdf/2509.25853) (SRAM-LUT GEMV), Vec-LUT [arXiv:2512.06443](https://arxiv.org/abs/2512.06443), FlashAttention-3 (WGMMA tensor-core baseline), AQPIM [arXiv:2604.18137](https://arxiv.org/abs/2604.18137) (DRAM-row-resident LUT for inner-product, PIM).
**관련 아이디어**: I2' TernVLM-KV-LUT (FA3-dual path), I4' PRISMKV-PIM-DequantLUT (DRAM bank), I2-Tier2 (LUT hit rate measurement letter) (2026-04-23 세션)
**Open Questions**: (a) Q sub-tile INT8 quantization 의 attention sharpness 손실 threshold — Q 를 INT4 까지 내릴 수 있는지 SoftMax Lipschitz 분석. (b) Hot/cold path 전환 threshold (2K tokens) 가 workload 에 따라 dynamic 해야 하는지 (prefill-heavy vs decode-heavy). (c) LUT hit rate 가 VLM vision-token (spatial locality) vs text-token 에서 분포 차이 — saliency-aware hit rate 설계 여지.

## FA3-Dual Path Attention Kernel
**정의**: FlashAttention-3 (H100 WGMMA tensor-core) 의 hot path 와 PRISM LUT-bypass 의 cold path 를 조합한 attention kernel. Decode 의 recent tokens (e.g., 2K 이하) 은 기존 FA3 path 로 throughput 유지, cold prefix (2K 이상) 은 LUT-bypass path 로 extreme quantization 이득 확보. Prefill-heavy workload (VLM large image token count, VLA action chunk) 에서 cold path activate. Production serving stack (vLLM/SGLang) 과 compatible 설계의 engineering 돌파구.
**관련 논문**: FlashAttention-3, vLLM PagedAttention (page_size 정합 8KB), T-MAC GPU port, PRISM ISLPED'26
**관련 아이디어**: I2' TernVLM-KV-LUT (2026-04-23 세션)
**Open Questions**: (a) Hot/cold transition 이 attention score 연속성에 미치는 영향 (boundary token 에서 discontinuity). (b) Multi-query / grouped-query attention 에서 dual path 의 kernel complexity. (c) Speculative decoding 과의 interaction (draft model 의 cold path activate 기준).

## Hessian-Aware Adaptive Rank (scale-space vs weight-space)
**정의**: Rank-r decomposition 에서 r 값을 **per-layer Hessian eigenvalue decay** 로 결정하는 전략. Cumulative eigenvalue ratio 90% 에 해당하는 rank 를 effective rank 로 채택. MDBF/LoRDS 등이 **weight-space Hessian** (∂²L/∂W²) 을 기준으로 하는 반면, PRISM 의 rank-r SF decomposition 은 **scale-space Hessian** (∂²L/∂S²) 을 기준으로 — 축이 다름. Scale-space 가 weight-space 보다 eigenvalue decay 가 빨라 작은 r 로도 expressive power 확보 가능 가설 (미검증).
**관련 논문**: MDBF [arXiv:2512.24545](https://arxiv.org/abs/2512.24545), LoRDS [arXiv:2601.22716](https://arxiv.org/abs/2601.22716), PyHessian estimator ([github.com/amirgholami/PyHessian](https://github.com/amirgholami/PyHessian))
**관련 아이디어**: I1' TernVLM-RankSF (2026-04-23 세션)
**Open Questions**: (a) Scale-space vs weight-space Hessian 의 eigenvalue decay 속도 차이 empirical 측정 부재. (b) Modality-split rank (vision r=8, LLM r=4, head r=1) 이 per-layer adaptive 와 interact 할 때 optimal 조합. (c) Training 중 rank 를 adaptive 하게 update 할 수 있는가 (현재는 fixed after Hessian pre-pass).

## Tile-Aligned / Xbar-Aligned Quantization (PRISM origin)
**정의**: SF granularity 를 physical hardware partition 경계 (CIM crossbar, GPU tile, DRAM bank, HBM row) 에 맞추는 전략. Channel-wise SF 가 HW partition 을 가로지르면 per-column buffer read N 개 필요하지만, partition-aligned SF 로는 1 read 로 충분. PRISM 에서 xbar-wise SF 로 CIM buffer access N× 감소 증명. 본 세션에서 GPU tile (BitBLAS 128), HBM row (HBM3 8KB), PagedAttention page (vLLM 기본 16KB) 4-tuple 정합으로 확장 (T2' PRISM-Tile).
**관련 논문**: PRISM ISLPED'26, PagedAttention (vLLM), BitBLAS [arXiv:2407.11722](https://arxiv.org/abs/2407.11722), XNOR-RRAM
**관련 아이디어**: I1' TernVLM-RankSF, I2' TernVLM-KV-LUT, T2' PRISM-Tile (2026-04-23 세션)
**Open Questions**: (a) HBM3 vs HBM3e row 크기 차이가 alignment 전략에 미치는 영향. (b) Multi-GPU tensor parallel 시 tile alignment 의 cross-device 일관성. (c) Dynamic page_size 가 throughput 에 미치는 실측 impact (현재 static 가정).

## Action Imminence Score (β_a)
**정의**: VLA 의 action horizon 상 "다음 수백 ms 내 어떤 KV 페이지가 hot 이 될지" 를 예측하는 scalar [0,1]. 3-feature: (1) gripper state change rate (last 3 action 의 gripper dim 미분), (2) trajectory curvature (직전 8 action position 2차 미분 norm), (3) end-effector ↔ detected-object distance (DINOv2 mask 기반, 1.2ms/frame). Logistic regression. RTX 4090 0.3ms. β_a > 0.7 인 frame 의 KV 는 HBM top-tier 에 pin.
**관련 논문**: VLA-Cache [arXiv:2502.02175](https://arxiv.org/abs/2502.02175), KV-Efficient VLA [arXiv:2509.21354](https://arxiv.org/abs/2509.21354), AC²-VLA [arXiv:2601.19634](https://arxiv.org/abs/2601.19634)
**관련 아이디어**: L2 v2 TemporalTier-3 (VLA 경로)
**Open Questions**: Gripper state + trajectory curvature + object-distance 3-signal 이 VLA-Cache frame-diff 대비 action-boundary AUC 가 +0.08 이상 우월한지 예측기 study 필수. RLBench/LIBERO/OpenX-Embodiment 에서만 available — real-world robot trace 부재.

## Content-Axis Taxonomy
**정의**: 서빙 요청을 **content semantic** 에 따라 분류하는 taxonomy. 본 세션에서 L1 v2 는 6-class effective taxonomy (prefill-long / decode-memory / decode-compute / vision-encode / audio-decode / retrieval-heavy) 제안. 초안 8-class 에서 α-pairwise MAD < 15% 인 VQA-simple/VQA-reasoning 병합. DistilBERT-tiny (2M param, 0.4ms) 또는 P1 E²IC distilled (L_early=2 activation 10-dim probe, 260 param, 50μs) 로 분류. 기존 serving scheduler (Nova, DuetServe, RPS-Serve) 는 stage/size-based 로 content-blind.
**관련 논문**: RPS-Serve [arXiv:2603.26498](https://arxiv.org/abs/2603.26498) (size-based), Nova [arXiv:2509.21301](https://arxiv.org/abs/2509.21301) (stage), DuetServe [arXiv:2511.04791](https://arxiv.org/abs/2511.04791) (P/D)
**관련 아이디어**: L1 v2 ContextSM-Tri
**Open Questions**: 6-class 가 실제로 (α_SM, α_BW, α_KV) 의 pairwise MAD > 15% 를 충족하는지 Azure LLM trace 로 검증 필요. Unseen workload 에 대한 default class fallback 정책.

## Copy-on-Write (COW) Reference-Counted KV
**정의**: vLLM PagedAttention block 단위 refcount 메타 확장. Shared block 에 write 발생 시 해당 block 만 fork (HBM local 할당 + copy), read 는 공유. Cross-request vision KV sharing 에서 partial overwrite (top-k attention 만 recompute) 를 O(k/N) 비용으로 지원. Mosaic/KVShare/MPIC 등 기존 cross-request cache 연구는 full-share/full-copy 이분으로 partial write 미지원.
**관련 논문**: Mosaic [arXiv:2604.10060](https://arxiv.org/abs/2604.10060), KVShare [arXiv:2503.16525](https://arxiv.org/abs/2503.16525), MPIC [arXiv:2502.01960](https://arxiv.org/abs/2502.01960)
**관련 아이디어**: A3 v2 SemCOW-Deadline
**Open Questions**: Cluster false-positive → output divergence 방지용 COW validation step (first-token logit divergence check) overhead 는 0.3 ms 예상. Write-on-shared fork rate 가 예상 14% 초과 시 memory saving 무효화.

## Deadline-Aware Green Context SM Yielding
**정의**: CUDA 12.5+ Green Context API 로 SM partition 을 per-request 로 동적 재분배하되, 결정 입력으로 **tenant SLO deadline miss risk** (time_to_deadline / estimated_remaining_compute) 를 사용. Risk > τ tenant 에 SM 확장, risk < τ 는 축소. Adrenaline [arXiv:2503.20552](https://arxiv.org/abs/2503.20552) 의 idle yielding (throughput) 과 orthogonal 한 SLO 축.
**관련 논문**: Adrenaline [arXiv:2503.20552](https://arxiv.org/abs/2503.20552), Hummingbird [arXiv:2601.04071](https://arxiv.org/abs/2601.04071) (microsec preemption)
**관련 아이디어**: A3 v2 SemCOW-Deadline, L1 v2 ContextSM-Tri (Green Context partition)
**Open Questions**: Green Context reconfig latency 가 NVIDIA 문서상 ms 급인지 실측 확인 필요. Tenant 수 < 4 에서는 yielding 효과 미미.

## Green Context (CUDA 12.5+ SM Partition API)
**정의**: NVIDIA CUDA 12.5 이상에서 노출되는 SM 그룹 (4 SM 단위) 동적 재구성 API. `cuGreenCtxCreate` + `cuCtxFromGreenCtx`. Ada 는 128 SM → 32 partition, Blackwell 은 더 세분화 가능. MIG (H100/H200) 의 수 초 reconfig 대비 **ms 단위 reconfig** 가능. Persistent kernel 로 context switch 오버헤드 (~50μs) 를 pay-once amortize. Hopper 에서는 MIG slice (초 단위) 와 nested 사용 가능.
**관련 논문**: Bullet [arXiv:2504.19516](https://arxiv.org/abs/2504.19516), DuetServe [arXiv:2511.04791](https://arxiv.org/abs/2511.04791), LithOS (SOSP'25), Nova [arXiv:2509.21301](https://arxiv.org/abs/2509.21301)
**관련 아이디어**: L1 v2 ContextSM-Tri, A3 v2 SemCOW-Deadline, A1 v2 PhaseGraph-VLA (optional stacking)
**Open Questions**: RTX Pro 6000 Blackwell 의 MIG 지원 여부 미확인. AMD MI300 portability 없음 (NVIDIA-only).

## HBM Bandwidth Carving (cudaAccessPolicyWindow)
**정의**: HW partition 불가능한 HBM BW 를 간접 제어하는 기법. (a) `cudaStreamSetAttribute(CU_STREAM_ATTRIBUTE_ACCESS_POLICY_WINDOW)` 로 특정 KV tensor 를 L2 persistent 로 마킹 (Ada 4090 L2=72MB, Blackwell RTX 5090 L2=128MB, Hopper H100 L2=128MB). (b) `cudaStreamCreateWithPriority` 로 memory controller queue 에서 차례 vote. (c) compute-heavy stream / BW-heavy stream / prefetch stream 3-stream orchestration. 목표: BW-heavy turn 의 L2 hit rate 를 측정 가능하게 상승 (NCU `lts__t_sectors_aperture_device_op_read_lookup_hit` 으로 검증).
**관련 논문**: POD-Attention (ASPLOS'25), DynamoLLM (HPCA'25), Understanding GPU Resource Interference [arXiv:2501.16909](https://arxiv.org/abs/2501.16909)
**관련 아이디어**: L1 v2 ContextSM-Tri (α_BW knob), A2 v2 TierKernel-Dispatch (L2 persistent hint)
**Open Questions**: L2 persistent hint 효과가 Hot tile 의 실제 BW contention 해소 까지 연결되는지는 workload 별 검증 필요. Blackwell 이후 L2 spec 변경 시 portability risk.

## Hawkes Process Arrival Model
**정의**: Self-exciting point process. 과거 event 가 미래 event 의 intensity 를 증가 (kernel function 예: exponential, power-law). VLM 사용자 interaction (문서 view → query 연쇄) 이나 VLA action sequence 의 bursty 특성을 모델링. β_v (VLM 용 interaction imminence score) 에서 scene cut + audio energy + Hawkes arrival rate 를 결합.
**관련 논문**: LiveVLM [arXiv:2505.15269](https://arxiv.org/abs/2505.15269) (interaction 직전 10초 eviction mistake 73%)
**관련 아이디어**: L2 v2 TemporalTier-3 (β_v VLM streaming 경로)
**Open Questions**: Hawkes vs Poisson (exponential) ablation 에서 bursty workload 만 의미 있는지. Kernel function 선택 (exponential/power-law) 에 따른 robustness.

## Page-Hinkley Change-Point Detector
**정의**: Classical sequential change-point detection statistic. $m_T = \sum_{t=1}^{T}(s_t - \bar{s} - \delta)$, $M_T = \min_{t \le T} m_t$, change detected when $m_T - M_T > \lambda$. Gradual drift 를 single-threshold 이 놓치는 경우에 backup 으로 사용. Training-free quantile calibration 과 결합. 본 세션의 SSE (Semantic Shift Estimator) 에서 소프트/하드 2-threshold 의 보완으로 채택.
**관련 논문**: Expected Attention [arXiv:2510.00636](https://arxiv.org/abs/2510.00636)
**관련 아이디어**: A1 v2 PhaseGraph-VLA (SSE phase boundary detection), P2 SSE predictor
**Open Questions**: LIBERO 5 task 에서 FP rate empirical 측정 필요 (A1 v2 Phase 3 entry action).

## Phase-Conditioned CUDA Graph Dispatcher
**정의**: VLA trajectory phase (Approach/Manipulate/Retract) 별로 서로 다른 **kernel fusion boundary + SM allocation** 을 미리 pre-captured CUDA Graph 로 storage 하고 runtime 에 dispatch. vLLM 의 CUDA Graph capture 는 batch-size variant 만 지원 — content/phase-aware variant 는 공백. 본 세션의 A1 v2 에서 phase × batch 2D table 제안.
- **Approach graph**: SigLIP patch-embed full + FlashAttn-3 MQA fused (ViT-specific).
- **Manipulate graph**: SigLIP partial-batch (40% re-encode) + action-head SiLU+Linear fused + reduced attention heads.
- **Retract graph**: SigLIP bypass (last 2-step linear extrapolation) + KV static reuse.
**관련 논문**: Running-VLAs-Real-time [arXiv:2510.26742](https://arxiv.org/abs/2510.26742), Nova [arXiv:2509.21301](https://arxiv.org/abs/2509.21301), DuetServe [arXiv:2511.04791](https://arxiv.org/abs/2511.04791)
**관련 아이디어**: A1 v2 PhaseGraph-VLA
**Open Questions**: CUDA Graph capture overhead (~수 MB graph memory × 3 variant) amortization 이 phase transition 주기 (> 100 step) 내에 충족되는지.

## Reference Count COW (vs Full-Share)
**정의**: Cross-request vision KV sharing 의 두 패러다임:
- **Full-share**: Mosaic/KVShare 방식. Cluster 전체 member 가 KV 공유 후 divergence 발생 시 cluster 전체 fork. Worst-case memory ×N.
- **Ref-count COW**: vLLM PagedAttention block 단위 refcount, write 발생 block 만 fork. Partial write (top-k recompute) 에서 훨씬 효율적.
**관련 아이디어**: A3 v2 SemCOW-Deadline.

## Semantic Shift Estimator (SSE)
**정의**: Per-frame (video VLM) 또는 per-chunk (VLA action chunk) scene/state shift detector. Input: L_mid LLM hidden state cosine sim + EWMA baseline + Jensen-Shannon divergence + action chunk delta. Decision: 2-threshold (soft pre-warm / hard evict) + hysteresis 3 frames + Page-Hinkley backup. Training-free quantile calibration (처음 100 frame). Scene change F1 0.72-0.80, decision latency <100μs. 본 세션의 A1 v2 에서 phase predictor 로 채택 (P2 흡수).
**관련 논문**: Event-VStream [arXiv:2601.15655](https://arxiv.org/abs/2601.15655), CodecSight [arXiv:2604.06036](https://arxiv.org/abs/2604.06036), VLA-Cache [arXiv:2502.02175](https://arxiv.org/abs/2502.02175) (frame-diff 비교)
**관련 아이디어**: A1 v2 PhaseGraph-VLA, L2 v2 TemporalTier-3 (β_v 일반형)
**Open Questions**: Task-conditioned EWMA (P1→P2 cascade) 의 cold-start — 첫 request 에서 P1 uncertain 인 경우 P2 baseline.

## Top-k Partial Recompute
**정의**: Cross-request cluster-hit 시 전체 KV 재계산 대신 **attention top-k block 만 recompute** 하는 기법. Full recompute 대비 O(k/N). COW refcount 와 결합하여 top-k block 만 fork. k 선택 sensitivity 가 accuracy/latency trade-off 의 key parameter.
**관련 논문**: OmniSparse [arXiv:2511.12201](https://arxiv.org/abs/2511.12201), SparseVILA [arXiv:2510.17777](https://arxiv.org/abs/2510.17777), FlashVLM [arXiv:2512.20561](https://arxiv.org/abs/2512.20561)
**관련 아이디어**: A3 v2 SemCOW-Deadline

## Tri-Knob Resource Control (α_SM, α_BW, α_KV)
**정의**: Serving 요청에 대한 3-axis resource allocation vector:
- **α_SM**: SM 할당 비율 (Green Context 4-SM 단위 partition).
- **α_BW**: HBM bandwidth carving (cudaAccessPolicyWindow L2 persistent + stream priority).
- **α_KV**: KV budget (prefill chunk 크기 + decode reserved block count).
세 축 모두 content taxonomy 별로 서로 다른 최적 비율을 가진다. Single-knob baseline (SM-only 또는 BW-only) 대비 joint Pareto 개선 가능. 본 세션의 L1 v2 제안.
**관련 논문**: Nova [arXiv:2509.21301](https://arxiv.org/abs/2509.21301) (SM-only stage), DuetServe [arXiv:2511.04791](https://arxiv.org/abs/2511.04791) (SM-only P/D), POD-Attention (ASPLOS'25)
**관련 아이디어**: L1 v2 ContextSM-Tri
**Open Questions**: Tri-knob factorial ablation 에서 tri > any single × 1.4 목표. 각 knob 의 Marginal gain isolation 실험 필요.

## Turn Oldness (Inter-Turn Visual Coldness)
**정의**: Multi-turn GUI agent (Claude Computer Use, VisualWebArena) 에서 screenshot-history visual token 의 "이 token 이 몇 turn 전 생성되었는가" 축. Long-tail distribution — 대부분 과거 token 은 재참조 안 되지만 2% 는 critical. γ_v(t) = w1·turn_oldness(t) + w2·log-sum-exp(past_attn_scores) + w3·tile_entropy(t) 로 정의, turn_oldness 가 dominant term. MMInference (offline static permutation) vs 본 idea (per-turn dynamic) 차별 포인트.
**관련 논문**: MMInference [arXiv:2504.16083](https://arxiv.org/abs/2504.16083), OmniSparse [arXiv:2511.12201](https://arxiv.org/abs/2511.12201) (intra-turn only), DiffKV [arXiv:2412.03131](https://arxiv.org/abs/2412.03131)
**관련 아이디어**: L3 v2 MTV-Pool (미선정 Major Revision)
**Open Questions**: Claude Computer Use 내부 trace 접근. "GUIAgent-KV 2025-11 (placeholder)" 유사도 65% 실존 확인.

---

## AttAcc (Attention Accelerator PIM)
**정의**: ASPLOS 2024 Park et al. 의 PIM 기반 attention accelerator. HBM3 bank-level 에 16 FP16 MUL+ADD/bank GEMV unit 을 배치하여 decode phase 의 memory-bound attention GEMV (Q·Kᵀ, softmax·V) 를 9× internal BW (242 TB/s) 로 가속. 1,024 banks / 256 BGs / 8 DRAM dies + 1 Buffer. Throughput 최대 5.93×, energy 66.8% 감소 (GPT-3 175B / LLAMA 65B). AttAcc_bank + head-level/batch-level pipelining + FC co-processing 구조. 본 연구의 baseline simulator.
**관련 논문**: AttAcc (ASPLOS 2024), NeuPIMs [arXiv:2403.00579](https://arxiv.org/abs/2403.00579), IANUS, TransPIM (HPCA'22), HBM-PIM ISSCC'21
**관련 아이디어**: F1-VLM (DeepStack-Native 6-Tier), F2-VLM (KL-Collapse Detector Macro)
**Open Questions**: Original simulator 는 decode-only (prefill GPU route) — prefill PIM 지원을 위해 4-file extension (config/system/model/ramulator_wrapper) 필요. VLM 의 chunked prefill 을 m=1 GEMV 로 decompose 하는 것이 feasible.

## Bank Imbalance (PIM Mixed-Batch)
**정의**: Mixed LLM + VLM batch 에서 bank group 간 load 편차가 최대 13.62× (25% VLM mix 시) 발생하는 현상. 소수의 큰 VLM request 가 hot spot 생성. AttAcc HBM3 (1,024 banks, 256 BGs) 기준 modality-aware spreading 으로 22% 완화, same-image sharing 으로 BS=128 에서 38% 메모리 절약.
**관련 논문**: AttAcc (ASPLOS 2024), 본 연구의 VLM_exploration_PIM_260407
**관련 아이디어**: 본 연구의 4-C, F1-VLM (일부 흡수)

## Chunked Prefill
**정의**: LLM serving 에서 긴 input prefill 을 작은 chunk C 로 나눠 처리하는 기법. 각 chunk 는 이전 chunk 의 누적 KV 전체에 attend → memory-bound 로 전환. vLLM/Sarathi-Serve production default. 본 연구의 측정: C=4 AI=4.0, C=16 AI=15.6, C=32 AI=37.2 (모두 A100 ridge 87.6 ops/byte 미만 memory-bound); C≥256 부터 compute-bound 전환. E1 실측 C=4 SDPA 870ms (full prefill 33ms 의 26×).
**관련 논문**: Sarathi-Serve ([arXiv:2403.02310](https://arxiv.org/abs/2403.02310)), Mnemosyne ([arXiv:2409.17264](https://arxiv.org/abs/2409.17264)), POD-Attention ([arXiv:2410.18038](https://arxiv.org/abs/2410.18038))
**관련 아이디어**: F1-VLM (C-adaptive dispatcher)
**Open Questions**: C=16 sweet spot 이 HW topology (A6000 vs A100 vs LPDDR-PIM) 와 workload (text vs VLM DeepStack) 에 universal 한가?

## DeepStack (VLM Multi-Layer Visual Injection)
**정의**: Qwen3-VL 등 VLM 에서 ViT 의 intermediate layer 출력을 LLM decoder 의 여러 layer (예: L4, L8, L12) 에 inject 하는 architecture (NeurIPS 2024). Single-injection (Qwen2.5-VL 등) 대비 vision information 이 LLM 전체에 분산. 본 연구의 가설: L17-21 peak (visual attn 24.5%) 는 L12 inject 이후 visual 정보가 self-attn 으로 확산되어 누적된 결과.
**관련 논문**: DeepStack ([arXiv:2406.04334](https://arxiv.org/abs/2406.04334)), Cross-Layer Injection ([arXiv:2601.10710](https://arxiv.org/abs/2601.10710))
**관련 아이디어**: F1-VLM (6-tier boundary), F3-VLM (재설계 시 injection-aware coherence)
**Open Questions**: Injection topology [4,8,12] 가 다른 VLM family (LLaVA early-fusion, InternVL) 에서 유사 peak pattern 을 만드는가?

## FP16 L27 Overflow (VLM Numerical Safety)
**정의**: Qwen3-VL L≥1000 long-sequence 에서 LLM 마지막 layer (L27) self-attn 의 Q·Kᵀ abs_max 가 FP16 max (65,504) 를 초과하여 softmax(exp(∞)) = NaN 발생. 원인: FP16 5-bit exponent 한계; L27 은 LM head 직전 sharpened representation 으로 score 절대값 최대. BF16 (8-bit exponent) 에서는 동일 입력 L27 abs_max=860 finite. ViT 의 L31 abs_max=42,368 (FP16 65%) 는 위험하지만 finite. **VLM 특화 numerical safety 이슈**.
**관련 논문**: FlashAttention-3 ([arXiv:2407.08608](https://arxiv.org/abs/2407.08608) FP32 softmax rescale)
**관련 아이디어**: F2-VLM (HW KL-collapse detector + BF16 fallback path)
**Open Questions**: 다른 VLM (LLaVA-NeXT, InternVL) 의 마지막 layer 에서도 L27-like overflow 가 발생? ViT tail layer (L31) 의 FP16 65% 에 근접도 HW fallback 트리거 필요한지?

## Layer Asymmetry (VLM Visual Attention)
**정의**: VLM inference 에서 visual KV 가 sequence 의 86.2% 를 차지하면서도 attention 기여도는 11.4% 만 (Qwen3-VL-4B) — BW 7.6× over-consumption. Layer 별로는 L0-7 평균 2.6%, L17-21 평균 24.5% (peak L18=30.7%), worst-case waste L7=116× (visual_bw_fraction / visual_attn_fraction). **Uniform placement 는 비효율**. 5-model 교차검증 (Qwen3-VL/Qwen2.5-VL/InternVL3/Qwen3.5/Llama-3.2-V) 결과 Qwen3-VL/Qwen2.5-VL/InternVL3 dense band 공통.
**관련 논문**: VL-Cache ([arXiv:2410.23317](https://arxiv.org/abs/2410.23317)), AKVQ-VL ([arXiv:2501.15021](https://arxiv.org/abs/2501.15021)), SparseVLM ([arXiv:2410.04417](https://arxiv.org/abs/2410.04417))
**관련 아이디어**: 본 연구의 4-B (Layer-Adaptive Placement), F1-VLM (6-tier)
**Open Questions**: L17-21 peak 의 DeepStack inject causality 를 ablation 으로 증명 가능? Hybrid linear (Qwen3.5 48/64) 와 cross-attn (Mllama) 에서 어떻게 일반화?

## MOESI (Cache Coherence Protocol)
**정의**: Modified / Owned / Exclusive / Shared / Invalid 5-state cache coherence protocol (Sweazey & Smith, ISCA'86). Modified = 유일한 dirty copy, Owned = dirty shared (write-back 책임), Exclusive = 유일한 clean, Shared = 다중 clean, Invalid = 무효. LazyPIM ([arXiv:1706.03162](https://arxiv.org/abs/1706.03162)) 이 PIM-CPU coherence 에 응용. TraCT ([arXiv:2512.18194](https://arxiv.org/abs/2512.18194)) 는 multi-TB CXL scale 에서 cache-line coherence impractical 주장.
**관련 논문**: LazyPIM, TraCT, CacheBlend ([arXiv:2405.16444](https://arxiv.org/abs/2405.16444))
**관련 아이디어**: F3-VLM (image-hash granularity MOESI, 미선정 Major Revision)
**Open Questions**: KV 는 append-only 성질이 강해 Modified/Owned state 빈도 낮을 가능성 — 5-state 전부 필요한가 MSI 로 충분한가? Image-identity granularity (pHash + SimHash verification) 로 cache-line 실용성 회피 가능?

## Per-Sample Fragility (VLM Policy)
**정의**: VLM attention pattern 이 aggregate (125 sample 평균) 레벨에서는 corr 0.996 으로 안정적이지만, individual sample 레벨에서는 sample-to-full corr minimum 0.357 까지 하락하는 현상. 즉 정적 layer-budget 정책을 개별 request 에 적용 시 35% 수준 상관 request 에서 catastrophic KV misallocation 위험. VL-Cache (ICLR'25) 가 layer-adaptive budget 을 주장하면서 report 하지 않은 리스크. **Per-sample gating 축은 VLM KV 문헌에 empty**.
**관련 논문**: VL-Cache, AKVQ-VL, MBQ ([arXiv:2412.19509](https://arxiv.org/abs/2412.19509)), CAM ([arXiv:2406.02069](https://arxiv.org/abs/2406.02069))
**관련 아이디어**: F2-VLM (PatternGuard per-sample fragility gating via L0-L1 probe + W8A8 KL disagreement)
**Open Questions**: Aggregate stable 인데 individual noisy 한 원인 (central limit effect? DeepStack inject distribution?) 의 causal 해명; fragility classifier 10차원 feature 로 >85% 정확도 달성 가능?

## W8A8 Pattern Collapse (VLM Quantization)
**정의**: RedHatAI W8A8 quantization 이 Qwen3-VL 의 layer-wise visual attention pattern 을 완전히 붕괴시키는 현상 — FP16 baseline 5.73% (entropy 0.451) → W8A8 72.58% (entropy 0.90) = **Δ +66.85pp extreme collapse**. 반대로 **weight-only INT4/INT8 은 Δ<0.1pp 로 pattern 보존** (INT4 per-group 128 에서 -0.06pp, INT8 per-channel 에서 -0.03pp). FP8-Dynamic activation 은 반대 방향 -4.17pp (opposite collapse, visual → text 로 몰림). 즉 **activation quantization** 이 visual attention distribution 을 파괴하는 핵심이며, activation 보존 시 PIM-friendly quantization 가능.

**선행 보고 상태 (2026-04 기준 literature survey)**: **+66pp extreme collapse 는 선행 논문 어디에도 직접 보고된 바 없음**. MBQ/Q-VLM/VLMQ/MQuant/AKVQ-VL/VL-Cache/LLM.int8/SmoothQuant/QuaRot/SpinQuant 등 주요 VLM quantization 연구 **모두 task accuracy (MMMU/VQA score) 만 보고, layer-wise visual attention ratio 를 quantization 전/후로 직접 측정하지 않음**. Qwen-VL family + W8A8 + attention ratio 정량화는 완전 공백. 관련 선행: MBQ 는 "vision 10× gradient 둔감" (modality sensitivity), LLM.int8 는 "outlier zero 화 시 softmax mass 감소" (**반대 방향**), Visual Attention Sink ([arXiv:2503.03321](https://arxiv.org/abs/2503.03321)) 는 **FP16 내재적 sink** (양자화 영향 미분석), MQuant 는 per-tensor scale mismatch 만 논의.

**Mechanism Triangulation (세 선행 연구의 합성으로 가설)**: (1) **Vision-side outlier 증폭** — Per-tensor W8A8 이 outlier 기준 scale 설정 → non-outlier 가 INT8 격자 하위 bit 로 압축 → softmax(QKᵀ) logit 분산 왜곡 → vision token 쪽 mass 집중 (LLM.int8 의 반대 방향 현상). (2) **Attention sink 증폭** — FP16 에 이미 존재하는 visual sink 가 massive activation quantization 으로 더 spiky 해지면서 sink-KV dot-product magnitude 비정상 증가. (3) **Rotation-invariance 미적용** — RedHatAI W8A8 은 per-tensor SmoothQuant 계열 (SpinQuant/QuaRot 미적용), vision-channel outlier 보존.

**관련 논문**: SmoothQuant ([arXiv:2211.10438](https://arxiv.org/abs/2211.10438)), AWQ ([arXiv:2306.00978](https://arxiv.org/abs/2306.00978)), P3-LLM ([arXiv:2511.06838](https://arxiv.org/abs/2511.06838)), MBQ ([arXiv:2412.19509](https://arxiv.org/abs/2412.19509)), LLM.int8() ([arXiv:2208.07339](https://arxiv.org/abs/2208.07339)), Q-VLM ([arXiv:2410.08119](https://arxiv.org/abs/2410.08119)), VLMQ ([arXiv:2508.03351](https://arxiv.org/abs/2508.03351)), MQuant ([arXiv:2502.00425](https://arxiv.org/abs/2502.00425)), Visual Attention Sink ([arXiv:2503.03321](https://arxiv.org/abs/2503.03321)), Seeing but Not Believing ([arXiv:2510.17771](https://arxiv.org/abs/2510.17771)), Attention Sinks and Compression Valleys ([arXiv:2510.06477](https://arxiv.org/abs/2510.06477)), SpinQuant ([arXiv:2405.16406](https://arxiv.org/abs/2405.16406)), QuaRot, Fallback Quantization ([arXiv:2503.08040](https://arxiv.org/abs/2503.08040)), Empirical Qwen3 Quant ([arXiv:2505.02214](https://arxiv.org/abs/2505.02214))

**관련 아이디어**: F2-VLM (KL-Collapse Detector Macro + Per-Sample Fragility Gating; measurement contribution = +66pp 정량화 최초)

**Open Questions**: (1) W8A8 collapse 를 **signal 로 재해석** — W8A8 vs W-only KL divergence 를 per-sample fragility proxy 로 활용 가능성. (2) **Bit-width 연속 곡선** (W6A6/W4A4/W4A8) 에서 A 축 감소 시 Δ 단조 증가인지, W8A16 은 preserve 인지. (3) **Rotation 기반 (QuaRot/SpinQuant)** 이 collapse 완화하면 mechanism (c) rotation-invariance 검증 → F2 는 "rotation 적용 불가한 legacy 배포" scope 축소. (4) LLaVA-1.6/InternVL2.5/MiniCPM-V 등 Qwen-VL 외 family 에서도 W8A8 attention collapse 발생? (5) Collapse 가 특정 layer (middle) 에서 시작되는지 all-layer 동시 현상인지 heatmap. (6) Task accuracy 와 attention distribution 의 decoupling ("Seeing but Not Believing" 현상) — collapse 에도 MMMU top-1 유지 시 reasoning grounding 관점 기여 확장 가능.

---

## ACE (Accuracy-Critical Expert)
_(아래 ACE 정의는 기존 MoE 세션용. 본 섹션 위에 BNN/CIM 관련 용어 블록이 있다.)_

## BNN (Binary Neural Network)
**정의**: weight 와 activation 을 모두 1-bit (`{-1, +1}`) 로 양자화한 신경망. XNOR + popcount 로 1-bit MAC 을 구현 가능 — CIM crossbar 와 자연스럽게 매칭된다. 정확도 회복을 위해 **scaling factor(SF)** 를 곱하는 것이 표준 (XNOR-Net 이래).
**관련 논문**: XNOR-Net, Bi-Real-Net, ReActNet, A&B BNN ([arXiv:2406.03718](https://arxiv.org/abs/2406.03718))
**관련 아이디어**: PRISM (ISLPED'26 투고본), F1 TempoPRISM-CoDesign, F2 PhysioPRISM-VitalXbar
**관련 개념**: [[TNN]], [[Xbar-wise SF]], [[OPTIC]], [[Scratch Training]]
**Open Questions**: Non-image 도메인 (KWS, biosignal) 에서 BNN 1-bit 이 task-specific 정확도를 얼마나 보존? INT4 와의 hybrid 는 언제 필요?

## TNN (Ternary Neural Network)
**정의**: weight 를 `{-1, 0, +1}` 3값으로 양자화한 신경망. BNN 대비 0 sparsity 활용 가능 → CIM 에서 WL gating 으로 추가 에너지 절감. Activation 은 BNN 수준 (1-bit) 혹은 2-bit.
**관련 논문**: TWN, TTQ, CIM-Explorer ([arXiv:2505.14303](https://arxiv.org/abs/2505.14303))
**관련 아이디어**: F2 PhysioPRISM-VitalXbar (scope 축소 후 PPG region drop, 향후 복구 시 ternary PPG 고려)

## Xbar-wise SF (Crossbar-wise Scaling Factor)
**정의**: PRISM 이 도입한 개념. BNN 의 scaling factor 를 channel 당이 아닌 **물리 crossbar 당 1개**로 할당. N × M crossbar 마다 SF 1개 → buffer read 1회. 기존 channel-wise (N buffer reads per crossbar) 대비 N 배 감소.
**관련 논문**: [[islped-2026-prism]], Bai et al. array-wise quantization ([arXiv:2023-CIMQ], [arXiv:2023-Partial-Sum])
**관련 아이디어**: PRISM, F1, F2 — 모두 Xbar-wise SF 를 base primitive 로 사용
**관련 개념**: [[Decompose-then-Merge]], [[OPTIC]], [[LUT-Bypass]]
**Open Questions**: Xbar 물리 크기 (16×16, 32×32, 64×64, 128×128, 256×256) 에 따라 representational fidelity vs memory overhead trade-off — 어느 sweet spot 이 domain 별로 다른가?

## Decompose-then-Merge
**정의**: PRISM 의 핵심 training-inference decoupling 기법. **훈련 시** SF 행렬을 `S_xbar ∈ R^(P_N × P_M)` 를 직접 학습하지 않고 두 작은 matrix `S1 ∈ R^(P_N × r)`, `S2 ∈ R^(r × P_M)` 로 rank-r 분해해 학습 (representational capacity 확장). **추론 시** `S_PRISM = S1 × S2` merge 하여 physical crossbar partitioning dimension 만 남김 — zero inference overhead. LoRA 유사하나 목적 반대 (LoRA: 훈련 효율, PRISM: 표현력 확장 + 추론 overhead 0).
**관련 논문**: [[islped-2026-prism]], LoRA ([arXiv:2106.09685](https://arxiv.org/abs/2106.09685)), DoRA ([arXiv:2402.09353](https://arxiv.org/abs/2402.09353)), **LoRDS ([arXiv:2601.22716](https://arxiv.org/abs/2601.22716))** (LLM PTQ 로 독립 발견된 유사 원리)
**관련 아이디어**: F1 TempoPRISM-CoDesign 이 **time-axis rank decomposition** 으로 확장, F2 PhysioPRISM-VitalXbar 는 per-subject μ 적응에 scalar-shift 근사 사용
**관련 개념**: [[Xbar-wise SF]], [[Feature Expansion Effect]]
**Open Questions**: rank r 의 최적값이 task / model / crossbar 크기에 따라 어떻게 결정? Time-axis / modality-axis 확장은 어디까지 가능?

## OPTIC (Outlier-preserving Threshold-based Inlier Compression)
**정의**: PRISM 의 SF 압축 기법. SF 값 분포에서 평균 근처 inlier 는 공통 대표값 `μ_in` 으로 교체, tail 의 outlier 는 보존. Crossbar 가 작을 때 (P > Cout) baseline 대비 SF 수 초과 문제 해결. Threshold `τ` 를 반복 증가시켜 target compression ratio ρ 달성.
**관련 논문**: [[islped-2026-prism]], Oiso outlier-isolated data format ([arXiv:2025-Oiso])
**관련 아이디어**: F2 PhysioPRISM-VitalXbar 의 **subject-adaptive μ_in(u)** 은 OPTIC 의 per-subject 확장
**관련 개념**: [[Xbar-wise SF]], [[Subject-Adaptive Calibration]]
**Open Questions**: row-wise vs layer-wise statistics 의 선택 기준? Multimodal workload 에서 OPTIC threshold 가 modality 별 다르게 설정되어야 하는가?

## LUT-Bypass (Lookup-Table-based Multiplication Bypass)
**정의**: PRISM 이 SF 감소로 확보한 buffer 에 pre-computed `SF × integer_crossbar_output` 을 저장. 추론 시 crossbar 출력이 LUT 엔트리에 해당하면 FP32 곱셈 대신 1회 buffer read 로 대체. crossbar 출력이 near-normal 분포로 0 주변에 집중하므로 작은 LUT (N entries for N×N Xbar) 만으로 hit rate >82%.
**관련 논문**: [[islped-2026-prism]]
**관련 아이디어**: F1 의 **non-volatile warm-LUT** (streaming KWS 의 cold-start 제거), F2 의 **delta-LUT on shadow SRAM** (per-subject 패치, RRAM endurance 보존)
**관련 개념**: [[Xbar-wise SF]], [[Warm-LUT]], [[Shadow SRAM]]
**Open Questions**: Streaming workload 에서 LUT hit rate 의 time-locality 는? Non-volatile vs volatile LUT 의 area/power/cold-start trade-off 는?

## Frame-Phase Clustering
**정의**: F1 TempoPRISM-CoDesign 이 도입. Streaming KWS 의 각 프레임을 음향 특성 (silence / onset / voiced / offset) 으로 4-8개 cluster 에 할당. 첫 convolution layer activation norm 의 k-means 로 학습 (training 동적, inference 고정). 이 cluster index 를 (a) rank-2 decomposition 의 time-axis 로, (b) HW clock-gate signal 로 공유.
**관련 논문**: SparkNet ([arXiv:2406.06634](https://arxiv.org/abs/2406.06634)) 의 input-dependent gate 는 K=2 special case
**관련 아이디어**: F1 TempoPRISM-CoDesign
**관련 개념**: [[Decompose-then-Merge]], [[Clock-Gate Signal Reuse]], [[Warm-LUT]]
**Open Questions**: 클러스터 수 K 의 최적 범위? 언어 간 전이 (MSWC multilingual) 시 cluster semantics 가 보존되는가?

## Subject-Adaptive Calibration
**정의**: F2 PhysioPRISM-VitalXbar 의 핵심. Wearable BNN 에서 사용자별 신호 분포 shift 를 30초 unsupervised 수집 데이터로 보정. 가정: shift 가 per-layer activation mean 의 scalar shift 로 근사 가능 (rank-1 approximation). `μ_in(u, layer) = μ_baseline(layer) + α·(μ̂_u - μ_baseline)` 의 scalar α 를 meta-learning.
**관련 논문**: TTAQ ([arXiv:2412.09899](https://arxiv.org/abs/2412.09899)), Tent ([arXiv:2006.10726](https://arxiv.org/abs/2006.10726)), AdaBN ([arXiv:1603.04779](https://arxiv.org/abs/1603.04779)), RTF-Q ([arXiv:2408.05752](https://arxiv.org/abs/2408.05752))
**관련 아이디어**: F2
**관련 개념**: [[OPTIC]], [[Shadow SRAM]], [[Scalar-Shift Approximation]]
**Open Questions**: Scalar-shift 가정이 얼마나 정확? 어느 task/subject 조건에서 rank-1 부족?

## Shadow SRAM
**정의**: F2 VitalXbar 가 도입. RRAM CIM 에서 per-subject delta-LUT (μ_in adaptation 결과) 를 본체 RRAM 이 아닌 보조 SRAM bank 에 기록. RRAM write 는 endurance 10⁶-10⁹ cycle 로 제한되므로 weekly recalibration 시 장기 수명 문제 유발 → shadow SRAM 에 쓰면 RRAM write count 0.
**관련 논문**: ISSCC 2024 Jia et al. 16nm RRAM-CIM (endurance 분석), F2 세션 신규 도입
**관련 아이디어**: F2
**관련 개념**: [[LUT-Bypass]], [[Subject-Adaptive Calibration]], [[RRAM Endurance]]

## Heterogeneous-Precision Multi-Region CIM
**정의**: F2 VitalXbar 가 제안. 한 chip 에 여러 region 을 두고 region 마다 precision 다르게 (binary HAR / INT4 ECG / ternary PPG 등). 각 region 이 독립 crossbar + 공유 time-multiplexed ADC + shared shadow SRAM bank. Activity-gated duty cycling 으로 24/7 <50 µW 가능.
**관련 논문**: ISSCC 2025 RRAM/SRAM collab CIM (adjacent, ~25%)
**관련 아이디어**: F2 (scope 2-region: binary HAR + INT4 ECG)
**관련 개념**: [[Stagewise Training]], [[Shadow SRAM]], [[Activity-Gated Duty Cycling]]
**Open Questions**: Precision 수와 region 수의 trade-off? Joint training 불안정성을 stagewise 로 완전히 해결할 수 있는가?

## Stagewise Training (heterogeneous precision 용)
**정의**: Binary STE 와 INT4 LSQ 를 공동 훈련 시 gradient scale 10²-10³ 차이로 수렴 실패 발생. Stagewise 로는 region 1 (binary) 먼저 독립 훈련 → freeze → region 2 (INT4) 독립 훈련 → fusion head post-hoc. F2 의 기본 훈련 schedule.
**관련 아이디어**: F2 VitalXbar

## Scratch Training
**정의**: PRISM 계열 연구의 제약 조건. Pretrained 큰 모델을 fine-tune 하는 경로 없이 **작은 모델을 밑바닥부터** 훈련. CIM 타겟 BNN 은 1-bit weight 특성상 knowledge distillation 외에는 pretrained 활용이 제한적. 공영호 lab 의 PRISM 및 본 세션 F1/F2 모두 scratch training 기반.
**관련 아이디어**: PRISM, F1, F2
**관련 개념**: [[BNN]], [[TNN]], [[Small Model Constraint]]

## DeepStack (Hierarchical Visual Injection)
**정의**: MoE 모델에서 누적 expert score(`Σ_step s_i`)와 variance-aware global ranking으로 식별된 "정확도에 결정적인 expert" 집합. ACE-MoE에서는 cache hit rate 최대화 대신 ACE preservation을 caching 목표로 함.
**관련 논문**: [[iccad-2026-ace-moe]]
**관련 아이디어**: I1 (modality-aware extension), I3 (joint budget), I5 (real-time VLA), I2 (token transfer to ACT)
**관련 개념**: [[Variance-Aware Global Selection]], [[ACT (Accuracy-Critical Token)]], [[Time-Budgeted Skipping]]
**Open Questions**: 
- Modality dimension으로 자연스럽게 확장 가능한가? (I1에서 시도)
- Hard real-time budget에서 ACE 갱신 빈도 trade-off는?

## ACT (Accuracy-Critical Token)
**정의**: ACE 원칙을 visual KV token에 transfer한 개념 (이번 세션 도입). per-token cumulative attention contribution score를 누적하고 variance-aware global ranking으로 보존할 token을 선정.
**관련 논문**: [[iccad-2026-ace-moe]] (source method), [[internal-2026-vlm-pim-exploration]] (motivation O3/O4)
**관련 아이디어**: I2 (standalone, 미선정), I3 (joint budget의 일부)
**관련 개념**: [[ACE (Accuracy-Critical Expert)]], [[Visual KV Asymmetry]]
**Open Questions**: 
- ACT가 unseen image에 generalize되는가? (calibration 일반화)
- DeepStack inject layer에서 ACT score가 jump하는 패턴은?

## Decoy Expert (ε-weighted)
**정의**: MoE inference에서 실제 top-K expert 외에 **ε=0.05-scale 의 작은 가중치로** 추가 활성화되는 fake expert. Real output shift ≤ ε·‖h‖로 task accuracy는 거의 유지하지만 GPU-visible compute/memory pattern은 (K+k) top-(K+k)으로 보여 side-channel observer의 trace가 "다른 input"처럼 위장됨.
**관련 논문**: [[arxiv-2508-15036-moecho]] (threat), [[arxiv-2602-04105-expert-selections]] (defense hint)
**관련 아이디어**: I2 PhantomRoute
**관련 개념**: [[Routing Obfuscation]], [[Observable vs Functional Expert Set]]
**Open Questions**: 
- Adaptive attacker가 decoy 분포를 retraining set에 포함시키면 방어력 얼마나 degrade?
- N_experts < 16일 때 decoy 1개당 mass 비중이 너무 커 accuracy trade-off 감당 불가 — lower bound where?

## DeepStack (Hierarchical Visual Injection)
**정의**: VLM 아키텍처 패턴 중 하나로, ViT의 여러 intermediate layer 출력을 LLM의 특정 layer (예: L4, L8, L12)에 각각 inject하는 구조. Qwen3-VL이 채택. visual information이 LLM 전체 layer에 distribute됨.
**관련 논문**: [[internal-2026-vlm-pim-exploration]] (M5 분석), Qwen3-VL technical report
**관련 아이디어**: I4 (DeepStack-aware routing, I1에 흡수), I1 (HIL boundary handling)
**관련 개념**: [[Hierarchical Injection Layer (HIL)]], [[Visual KV Asymmetry]]
**Open Questions**: 
- DeepStack 외 다른 hierarchical injection 모델 (BLIP-3, Mantis)도 동일 패턴인가?
- Inject layer 인근의 importance jump 정량화

## Expert Fingerprinting
**정의**: MoE 모델에서 per-request activated expert 집합을 외부 관측 가능한 signal (hardware side-channel OR 값싼 predictor)로 identify/predict하는 것. 2026-04 premise: ~85-90% 정확도, expert 수가 많을수록 정확도 상승 (Qwen3-Next 128+ > Mixtral 8). 최신 논문: MoE-Beyond 97.5%, Pre-Attention Prediction 93-94%, OD-MoE 99.94%.
**관련 논문**: [[arxiv-2508-17137-moe-beyond]], [[arxiv-2511-10676-pre-attention]], [[arxiv-2512-03927-od-moe]], [[arxiv-2603-11114-task-conditioned]]
**관련 아이디어**: I1 ExpertEcho (attack signal), I2 PhantomRoute (defense target), I3 ZMSP (prefetch signal), I4 FARD-C (dispatch signal), I5 FF-ACE-VLA (eviction signal)
**관련 개념**: [[Signature Entropy]], [[WCRT Schedulability]], [[MoEcho Threat]]
**Open Questions**: 
- Noisy hardware channel (cache occupancy, power) vs cheap predictor (early-layer MLP) 간 정확도 envelope?
- OOD prompt class drift 하에서 85-90% 유지 가능한가?
- "Accuracy grows with expert count" 이론적 증명?

## LSH Cohort Purity / Signature Entropy
**정의**: FARD-C(I4)에서 도입한 개념. MoE request의 predicted expert footprint를 SimHash(1024-bit signature)로 인코딩 후 LSH bucket별 "같은 request class"만 몰리는 정도(purity)를 측정. 이론: log₂(K/k) × layers bits의 signature entropy가 있어야 cohort purity가 usefulness threshold 넘음. **Mixtral-8 (3 bits/layer): 실패. Qwen3-Next-128 (7 bits/layer): 성공.** → expert count와 dispatch quality가 joint하게 scaling.
**관련 논문**: [[arxiv-2503-04398-semantic-parallelism]] (within-replica version), [[arxiv-2602-21626-gimbal]]
**관련 아이디어**: I4 FARD-C
**관련 개념**: [[Expert Fingerprinting]]
**Open Questions**:
- Cohort purity → throughput gain 번역 함수의 closed form?
- Cross-replica dispatcher에서 purity vs load-balance trade-off Pareto?

## MoEcho Threat (Multi-Channel MoE Side Channel)
**정의**: MoEcho 논문([arXiv:2508.15036](https://arxiv.org/abs/2508.15036), CCS 2025)이 정립한 MoE LLM 서빙에 대한 4종 아키텍처 side channel: (1) Cache Occupancy, (2) Pageout+Reload, (3) Performance Counter, (4) TLB Evict+Reload. DeepSeek-V2 / Qwen1.5-MoE / TinyMixtral에서 Prompt Inference 99.8%, Visual Inference, Response Reconstruction 92.8% 달성.
**관련 논문**: [[arxiv-2508-15036-moecho]]
**관련 아이디어**: I1 ExpertEcho (scooped), I2 PhantomRoute (primary defense target)
**관련 개념**: [[Routing Obfuscation]], [[Decoy Expert]]
**Open Questions**:
- 128-expert vs 8-expert 모델 간 leakage scaling (MoEcho 측정 안 함)?
- Cross-process/cross-batch attacker(MoEcho는 co-resident)까지 확장 가능?

## Routing Obfuscation
**정의**: MoE 서빙에서 side-channel observer가 얻는 routing trace를 **input topic과 (조건부) 독립적**으로 만드는 defense 패러다임. Crypto(2PC/MPC) 기반은 100× slowdown으로 비현실적; 본 세션의 PhantomRoute는 ε-decoy로 1.3-1.5× overhead에 MI 5× 감소 목표. Constant-time crypto code의 MoE 판으로 해석 가능.
**관련 논문**: [[arxiv-2511-01197-cryptomoe]], [[arxiv-2601-06790-secmoe]], [[arxiv-2504-18147-noesis]] (training-only)
**관련 아이디어**: I2 PhantomRoute
**관련 개념**: [[Decoy Expert]], [[Observable vs Functional Expert Set]], [[MoEcho Threat]]
**Open Questions**:
- Formal MI upper bound I(trace;topic) ≤ f(k, ε, N_experts) 증명?
- Adaptive attacker가 decoy distribution을 학습 corpus에 포함시키면 방어력 변화?

## Observable vs Functional Expert Set
**정의**: PhantomRoute(I2)의 핵심 설계 원칙. MoE의 expert 활성화를 **(A) 실제 output에 기여하는 functional set** (real top-K)과 **(B) GPU 자원 trace상 보이는 observable set** (real top-K ∪ decoy-k)로 분리. ε·‖h‖ bound로 (A)의 variance를 제한하면서 (B)를 obfuscate.
**관련 논문**: [[arxiv-2602-04105-expert-selections]] (defense section sketch)
**관련 아이디어**: I2 PhantomRoute
**관련 개념**: [[Decoy Expert]], [[Routing Obfuscation]]

## WCRT Schedulability (for MoE Expert Scheduling)
**정의**: ZMSP(I3)에서 도입한 real-time 이론 기반 framework. MoE serving을 bounded-miss EDF 변형으로 모델링, per-layer wait ≤ max(T_compute, T_PCIe) + 2·T_INT4_fallback + aging_buffer 형태로 WCRT(Worst-Case Response Time) bound 증명. GPU의 non-deterministic 특성 때문에 **probabilistic-RT (p99.9)** framing 필수 — hard-RT 주장 금지.
**관련 논문**: Liu-Layland 1973 foundation, bounded-miss EDF 변형 (real-time systems literature)
**관련 아이디어**: I3 ZMSP, I5 FF-ACE-VLA (deferred)
**관련 개념**: [[Expert Fingerprinting]]
**Open Questions**:
- GPU scheduling non-determinism의 가정 모델(Gaussian? Pareto?)이 실측과 일치?
- Miss rate가 premise upper bound (15%) 초과 시 graceful degradation 증명?
- FF-ACE-VLA의 phase transition에서 WCRT 유지 증명 가능?

## Hierarchical Injection Layer (HIL)
**정의**: 본 세션에서 도입한 일반화 용어. 외부 modality encoder(ViT, audio encoder)의 intermediate output이 main backbone(LLM)의 특정 layer에 inject되는 layer를 가리킴. DeepStack이 instance.
**관련 논문**: 본 wiki에서 정의
**관련 아이디어**: I1 (boundary handling)
**관련 개념**: [[DeepStack]]
**Open Questions**:
- HIL 인근 ±2 layer를 boundary로 처리하는 optimal radius는?

## Sequence-Level Skipping
**정의**: ACE-MoE prefill phase에서 사용. token-level skipping(score < threshold token만 skip)이 prefill에서는 무력하므로, **predicted expert score를 모든 token에 대해 sum**한 aggregate score로 expert를 prefill 전체 sequence에 대해 skip하는 기법. Token 단위가 아닌 sequence(=request) 단위 결정.
**관련 논문**: [[iccad-2026-ace-moe]] Sec 3.4
**관련 아이디어**: I2 (visual token에 같은 원칙 적용)
**관련 개념**: [[Time-Budgeted Skipping]]

## Time-Budgeted Skipping
**정의**: GPU computation 시간을 hard budget으로 두고 그 안에 transfer 가능한 expert만 prefetch하고 나머지는 skip하는 기법. ACE-MoE에서 prefill/decoding 각각 다른 정책 적용.
**관련 논문**: [[iccad-2026-ace-moe]] Sec 3.4
**관련 아이디어**: I3 (joint multi-axis budget으로 일반화), I5 (VLA hard real-time budget으로 transfer)
**관련 개념**: [[Sequence-Level Skipping]], [[ACE]]
**Open Questions**:
- Multi-axis budget formulation의 optimizer 형태 (closed-form vs DP vs RL)?
- Hard real-time guarantee의 statistical bound 제공 방법?

## Variance-Aware Global Selection
**정의**: ACE-MoE의 cache 슬롯 분배 메커니즘. 누적 expert score를 layer-별 activation variance로 normalize한 후 모든 layer expert를 single global ranking에 pool. variance가 낮은 (= 더 sensitive한) layer에 자동으로 더 많은 슬롯 할당.
**관련 논문**: [[iccad-2026-ace-moe]] Sec 3.3, [Li et al. 2023, Chen et al. 2025] (variance sensitivity 근거)
**관련 아이디어**: I1 (modality dim 확장하여 2D variance-aware), I2 (token×layer로 transfer)
**관련 개념**: [[ACE]]
**Open Questions**:
- Modality 차원으로 확장 시 variance 정의는?

## Visual KV Asymmetry
**정의**: VLM에서 visual token이 sequence의 86.2%를 차지하지만 attention contribution은 11.4%만 받는 현상. Layer-wise로도 비대칭 (L17-21 dense 24.5%, L0-7 sparse 2.6%, peak L18=30.7%, worst case BW waste 116×).
**관련 논문**: [[internal-2026-vlm-pim-exploration]] M4 분석
**관련 아이디어**: I2 (ACT), I3 (token retention r_t), I1 (modality-aware), 모든 VLM idea
**관련 개념**: [[ACT]], [[DeepStack]]
**Open Questions**:
- 모델별(LLaVA, InternVL, Qwen-VL) layer asymmetry pattern이 얼마나 다른가?

## VLA (Vision-Language-Action Model)
**정의**: Vision encoder + language model + action decoder가 결합된 robotics control 모델. RT-2, OpenVLA, Octo, Pi-0 등. Closed-loop control frequency 30-100Hz의 hard real-time constraint를 가짐.
**관련 논문**: OpenVLA [RSS'24], RT-2 [CoRL'23], Pi-0 [arXiv'24]
**관련 아이디어**: I5 (ACE-VLA real-time)
**관련 개념**: [[Time-Budgeted Skipping]], [[ACE]]
**Open Questions**:
- Action expert specialization (manipulation skill별 expert)이 가능한가?
- Visual context freshness scoring과 ACE caching의 결합?
