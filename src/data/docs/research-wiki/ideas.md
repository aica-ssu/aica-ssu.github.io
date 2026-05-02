# Research Ideas

`aica-research-bot` 에서 도출된 모든 아이디어를 시간 역순으로 기록.

---

## Selected Ideas (2026-05-02 — VLM Scenario-Aware Optimization)

### T1.1 Mosaic — Workload-Adaptive Serving Configuration Dispatcher
- **Date**: 2026-05-02 | **Session**: [Landing](/research-wiki/2026-05/vlm-scenario-aware) / [Mosaic](/research-wiki/2026-05/vlm-scenario-aware/tier1/01-mosaic)
- **Tier**: Tier-1 🥇 | **Target**: OSDI 2027 / ASPLOS 2027 | **Score**: novelty 6 / diff 7 / impact 9.5 → mean **7.5**
- **5-axis Gain**: [Performance] mixed-scenario TTFT -30~-35% / throughput +30~+40% / [Memory] GPU VRAM peak -25% / [Cost eff.] $/req -35% / prefix hit 60-85%→65-90%
- **Mechanism (3)**: ① DistilBERT-mini 60M lightweight classifier 6-class < 2ms ② scenario-conditional config dispatch (KV budget + prefix policy + compression) ③ hit-rate adaptive hot scenario GPU pinning (LMCache static 직교)
- **Closest competitor**: ECVL-ROUTER [arXiv:2510.27256](https://arxiv.org/abs/2510.27256) ICLR'26 (model-tier 직교 axis), vLLM Prefix Caching, SGLang RadixAttention

### T1.2 Lattice — Cross-Turn Frame-Indexed Radix Vision KV Cache for Multi-Turn Video QA
- **Date**: 2026-05-02 | **Session**: [Lattice](/research-wiki/2026-05/vlm-scenario-aware/tier1/02-lattice)
- **Tier**: Tier-1 | **Target**: MLSys 2027 / NeurIPS 2026 | **Score**: novelty 8 / diff 6 / impact 8 → mean **7.5**
- **5-axis Gain**: [Performance] Multi-turn turn 2+ TTFT -60~-80% / [Memory] Vision KV 5-turn session 7.5GB→1.5GB (-75%) / cross-session frame prefix hit 0%→50-70% (new) / [Quality] ≤-1.0pt
- **Mechanism (3)**: ① per-frame pHash + frame radix tree ② PrefixKV [NeurIPS 2025] 기반 layer-wise prefix retention ③ cross-session frame prefix sharing + CLIP privacy classifier 95%+
- **Closest competitor**: VLCache [arXiv:2512.12977](https://arxiv.org/abs/2512.12977) 2025-12 (single-session, concurrent 50-70%), PrefixKV [NeurIPS 2025], SGLang RadixAttention

### T1.3 Bramble — Cross-Image Vision Token Pool for Multi-Image Agent Loop
- **Tier**: Tier-1 | **Target**: MLSys 2027 / OSDI 2027 | **Score**: novelty 8 / diff 5 / impact 8.5 → mean **7.2**
- **5-axis Gain**: [Memory] GPU KV 24GB→14GB (-40%) / max concurrent tenants 2× / [Cost eff.] -30~-45% / [Security] privacy leakage 0
- **Mechanism (3)**: ① image pHash + tenant-shared LRU pool ② CLIP privacy classifier (public/private) + boundary verification ③ reference counting + zeroize on evict + cuckoo filter
- **Closest competitor**: PolyKV [arXiv:2604.24971](https://arxiv.org/abs/2604.24971), KVShare [arXiv:2503.16525](https://arxiv.org/abs/2503.16525), OxyGen [arXiv:2603.14371](https://arxiv.org/abs/2603.14371) (모두 LLM token only)

### T2.1 Lantern — NVDEC-Coupled Sliding-Window KV for Streaming Video
- **Tier**: Tier-2 | **Target**: IEEE CAL letter + vLLM upstream PR | **Score**: novelty 5 / diff 8 / impact 6 → mean **6.0**
- **Mechanism**: NVDEC frame metadata + sliding window K=8 + eager free + memory recycle. HW co-design 0-30% scoop unique.

### T2.2 Compass — Ego-Motion-Aware Compression for Egocentric Video
- **Tier**: Tier-2 | **Target**: DATE 2027 / ISLPED 2027 | **Score**: novelty 7 / diff 7 / impact 6 → mean **6.0**
- **Mechanism**: optical flow zero-mean ego-motion stable region detection + ego-stable region retention + ego-motion vector → MRoPE T-axis 보강. EgoSchema 50%+ token reduction.

### T2.3 Hearth — Document VLM L2 Carveout + Tile Locality Boost
- **Tier**: Tier-2 | **Target**: DATE 2027 / IEEE ESL letter | **Score**: novelty 6 / diff 6 / impact 7 → mean **6.0**
- **Mechanism**: Mosaic dispatch hook + `cudaAccessPolicyWindow` tile residence boost + `cudaCtxResetPersistingL2Cache`. DocVQA tile latency -25%+.

### Dropped / Absorbed (2026-05-02)

- **DROP 14**: A3 (PrefixKV scoop) / A4 (Nova scoop) / A6 (AwaRes scoop) / B1 (VisionThink scoop) / B2 (LongVU scoop) / B3 (METok scoop) / B4 (TSG 91.4% scoop) / B6 (AttWarp scoop) / C4 (single axis weak)
- **흡수 6**: A5→Lantern / B7→Bramble / C1→Lattice / C3→Bramble / C5→Lattice / C6→Mosaic

---

## Selected Ideas (2026-04-28 — VLM Edge Layer-Wise + Context-Semantic Optimization, R57 신규 적용 첫 세션 — ATRIUM Tier-1 4번째 retain 포함)

### ATRIUM (2026-04-28 retain) — Layer-Asymmetry-aware UMA Bandwidth + L2 Partition for Dual-Track VLM Decode on AGX Thor
- **Date**: 2026-04-28 (사용자 명시 요청 retain, v2-r55 origin) | **Session**: [세션](/research-wiki/2026-04/vlm-edge-layerwise-context) / Summary
- **Tier**: Tier-1 4️⃣ (본 세션 4번째) | **Target**: HPCA 2027 / MICRO 2027 / ASPLOS 2027
- **Score**: novelty 7.7 / diff 7.5 / impact 8.0 (v2-r55 score retain)
- **5-axis Gain (R55.2)**: **[Performance]** decode +14% / **[Energy]** -12% / **[Memory eff.]** DeepStack L0-3 alloc skip
- **Single-system fit (R20-γ)**: single-{AGX Thor 128GB, 273 GB/s LPDDR5X, 2560-core Blackwell}
- **Mechanism (3)**: ① LayerClassifier (HOT/COLD/MEDIUM visual_attn_ratio 자동 분류, 100-shot calibration), ② Green Context CUDA 12.4 SM partition (HOT layer SM 1500/총 2560 할당, COLD 1060), ③ L2 SLC carveout (cudaCacheConfigure persistent) + DeepStack L0-3 alloc skip (visual KV alloc 자체 skip)
- **Cross-share dependency**: 본 세션 Prism M4 LayerClassifier 와 동일 visual_attn_ratio measurement infrastructure 공유 (`tools/atrium_calibrate.py::measure_layer_attn`). Sequential development path: ATRIUM W1-12 prototype → Prism 추가 W13-24 NVFP4 mixed precision 적용 가능
- **Note**: 사용자 명시 요청에 따른 retain. mechanism / 5-axis / single-system / R47 path 모두 v2-r55 그대로



### Prism — Layer-wise NVFP4/FP8/INT4 Mixed Precision + 4:8 Sparsity + Decode DVFS Slip + LayerClassifier (Cluster A merge)
- **Date**: 2026-04-28 | **Session**: [세션](/research-wiki/2026-04/vlm-edge-layerwise-context) / Summary
- **Tier**: Tier-1 🥇 | **Target**: MLSys 2027 / ASPLOS 2027 / NeurIPS 2026
- **Score**: novelty 7.5 / diff 8.0 / impact 9.0
- **5-axis Gain (R55.2)**: **[Performance]** decode +18-25% / **[Energy]** -22% energy/token / **[Memory eff.]** -32% (KV scale freeze) / **[Power]** -9% peak (DVFS slip)
- **Single-system fit (R20-γ)**: Jetson Thor 128GB (NVFP4 native primary) + RTX 5090 secondary
- **Mechanism (4)**: ① DeepStack-aware NVFP4/FP8/INT4 layer-wise mixed precision (Blackwell native E2M1), ② 4:8 structured sparsity tensor core dispatch (HOT layer L17-21 only), ③ KV cache scale freeze + decode-phase visual-context DVFS slip, ④ LayerClassifier (HOT/COLD visual_attn_ratio 자동 분류, ATRIUM-R(AI) M1 흡수)
- **Phase 2 → Phase 1' delta**: FGMP ([arXiv:2504.14152](https://arxiv.org/abs/2504.14152)) + MicroMix ([arXiv:2508.02343](https://arxiv.org/abs/2508.02343)) baseline 명시 + DeepStack-aware NVFP4 anchor 차별 axis 강화 + R56.2 published 77%

### Bivouac — Hierarchical Visual Semantic KV Clustering + Cross-Frame Cluster Reuse + Layer-Adaptive Cluster Budget
- **Date**: 2026-04-28 | **Session**: [세션](/research-wiki/2026-04/vlm-edge-layerwise-context) / Summary
- **Tier**: Tier-1 🥈 | **Target**: NeurIPS 2026 / ICML 2026 / MICRO 2027
- **Score**: novelty 7.5 / diff 8.5 / impact 8.0
- **5-axis Gain (R55.2)**: **[Performance]** decode +35% / **[Memory eff.]** KV memory -65% / **[Energy]** secondary
- **Single-system fit (R20-γ)**: Jetson Thor / RTX 5090
- **Mechanism (3)**: ① Hierarchical Visual Semantic KV Clustering (HSCV) — visual token semantic embedding K-means cluster centroid, ② Cross-Frame Cluster Reuse (CFCR) — consecutive frame similarity 기반 cluster centroid 공유, ③ Layer-Adaptive Cluster Budget (LACB) — HOT layer 더 많은 cluster, COLD 적은 cluster
- **Critical scoop 잔존**: Mosaic ([arXiv:2604.10060](https://arxiv.org/abs/2604.10060), 2026-04-11) 55-65% concurrent — Phase 1'' baseline 추가 의무
- **Phase 1' delta**: ClusterKV ([arXiv:2412.03213](https://arxiv.org/abs/2412.03213)) + Sali-Cache ([arXiv:2602.14236](https://arxiv.org/abs/2602.14236)) baseline + cross-frame cluster-level granularity + layer-adaptive k 차별

### RadixVL — Phase-aware Visual LSH Hash + RadixAttention Second-level Visual Prefix Branch + Green Context SM Partition with LSH Lookup
- **Date**: 2026-04-28 | **Session**: [세션](/research-wiki/2026-04/vlm-edge-layerwise-context) / Summary
- **Tier**: Tier-1 🥉 (W12 Tier-1/Tier-2 동적 분기) | **Target**: OSDI 2026 / SOSP 2027
- **Score**: novelty 5.5 / diff 7.5 / impact 8.5
- **5-axis Gain (R55.2)**: **[Performance]** TTFT -22-30% / **[Memory eff.]** -30-45% (visual prefix sharing) / **[Energy]** -15-20%
- **Single-system fit (R20-γ)**: Jetson Thor / RTX 5090
- **Mechanism (3)**: ① Phase-aware Visual LSH Hash (encode 단계 fine-grain, prefill 단계 coarse, decode 단계 cache-only), ② RadixAttention Second-level Visual Prefix Branch (text 와 다른 tree), ③ Green Context SM Partition with LSH Lookup
- **Phase 1' delta**: VLCache ([arXiv:2512.12977](https://arxiv.org/abs/2512.12977)) + SimCache (CVPR 2025W) scoop 대응 — Phase-aware policy + RadixAttention second-level branch + GC-LSH 가 unique axis. 시스템 분야 published 65%+

### Strata — Stratified KV Layout (L2/GDDR7/LPDDR5x UMA) + Layer-Score Tier 매핑 + Page-Color Affinity Bank Partitioning (ATRIUM-R(Sys) page-color 흡수)
- **Date**: 2026-04-28 | **Session**: [세션](/research-wiki/2026-04/vlm-edge-layerwise-context) / Summary
- **Tier**: Tier-2 T1 | **Target**: DAC 2027 / DATE 2027
- **Score**: novelty 6.5 / diff 8.0 / impact 7.5
- **5-axis Gain (R55.2)**: **[Performance]** +15-22% / **[Memory eff.]** +40-55% / **[Energy]** -10-15%
- **Single-system fit (R20-γ)**: RTX 5090 + Jetson Thor
- **Mechanism (3)**: ① Stratified KV layout (L2 carveout / GDDR7 / LPDDR5x UMA), ② Layer score 기반 tier 매핑 (HOT → L2, COLD → LPDDR5x), ③ Page-color affinity (UMA bank-level partitioning)
- **Phase 1' delta**: ATRIUM-R(Sys) page-color affinity 흡수, ASPLOS 2024 (memory tiering) / FAST 2025 (Mooncake) baseline 보강 + R56.2 published 86%

### Harbinger — Visual-confidence-aware Speculative Draft + Visual-cluster-heat Early-Exit + Power-envelope Adaptive Frequency Locking
- **Date**: 2026-04-28 | **Session**: [세션](/research-wiki/2026-04/vlm-edge-layerwise-context) / Summary
- **Tier**: Tier-2 T2 | **Target**: ISLPED 2027 / DATE 2027
- **Score**: novelty 7.0 / diff 7.5 / impact 6.5
- **5-axis Gain (R55.2)**: **[Performance]** decode +32% / **[Energy]** -28% / **[Power]** Peak power -18%
- **Single-system fit (R20-γ)**: Jetson Orin NX 16GB (10-25W power envelope)
- **Mechanism (3)**: ① Visual-confidence-aware Speculative Draft Length (visual context confidence > 0.8 → draft 8-token, < 0.5 → draft 2-token), ② Visual-cluster-heat Early-Exit (cluster heat 기반), ③ Power-envelope Adaptive Frequency Locking (Jetson Orin NX 25W envelope)
- **Phase 1' delta**: Spec-LLaVA ([arXiv:2509.11961](https://arxiv.org/abs/2509.11961)) + Fast Speculative Edge-Cloud ([arXiv:2505.21594](https://arxiv.org/abs/2505.21594)) baseline — visual-cluster-heat + power-envelope locking 이 unique axis

### Obelisk — Qwen3-VL-30B-A3B MoE Local Serving on RTX 5090 32GB GDDR7 + Expert Routing Layer-Aware + Per-stage Power Cap
- **Date**: 2026-04-28 | **Session**: [세션](/research-wiki/2026-04/vlm-edge-layerwise-context) / Summary
- **Tier**: Tier-2 T3 | **Target**: MLSys 2027 / DAC 2027
- **Score**: novelty 6.0 / diff 6.5 / impact 7.5
- **5-axis Gain (R55.2)**: **[Performance]** Throughput / **[Memory eff.]** GDDR7 32GB local fit / **[Power]** PCIe 600W envelope
- **Single-system fit (R20-γ)**: RTX 5090 single
- **Mechanism (3)**: ① Qwen3-VL-30B-A3B MoE local serving on 32GB GDDR7, ② Expert routing layer-aware, ③ PCIe 600W envelope 관리 + per-stage power cap
- **Critical scoop 잔존**: DynaExq ([arXiv:2511.15015](https://arxiv.org/abs/2511.15015), 2025-11-19) 35-45% adjacent
- **Phase 1' delta**: DynaExq baseline + R56.2 47% → 73% 보강 의무 (W7 추가)

## Dropped Ideas (4, 2026-04-28)

- **CARILLON** (3-phase scheduling) — Nova ([arXiv:2509.21301](https://arxiv.org/abs/2509.21301), 2025) 80%+ scoop
- **BREAKWATER-T-R** (dual-Jetson MIG reframe) — Jetson Thor T5000 MIG single-partition only (2026-04 시점) infeasible
- **TIDEGATE** (DVFS slip) — R56.2 30% 미달 + Prism M3 흡수
- **HARBOR-DLA** (DLA-aware ViT) — DLA 2.0 LayerNorm 미지원, R20-γ platform mismatch (Orin Ampere only)

---

## Selected Ideas (2026-04-27 v2-r55 종합 — VLM↔LLM Asymmetry, R55 No-Simulator + 5-Axis Gain 적용)

### VEILSEAL-KV — Adversarial-Secure Multi-Tenant Edge VLM KV Cache (GMAC + Visual Prefix Defense + PII Redaction) ⭐ NEW v2 Tier-1 lead
- **Date**: 2026-04-27 (v2-r55) | **Session**: 세션 / Summary
- **Tier**: Tier-1 lead | **Target**: USENIX Security 2026 / S&P 2027 / OSDI 2027 / ASPLOS 2027
- **Score**: Novelty 8.5 / Diff 8.0 / Impact 8.5 → mean **8.3**
- **5-axis Gain (R55.2)**: **[Security]** cross-tenant integrity 100% (forgery 2⁻¹²⁸) + visual prompt injection TPR ≥92% + PII recall ≥95% + Reconstruction attack -79pp at ε=4 / **[Memory eff.]** -28% (BIMODAL-MASK-T2 흡수) / **[Performance]** overhead < 6% (ARMv8 PMULL HW accel)
- **Hypothesis**: Edge multi-tenant VLM (JPS / robotics) 의 critical security gap — SGLang RadixAttention prefix block 의 cross-tenant leak + visual prompt injection attack + PII leakage 통합 해결
- **Mechanism (3)**: ① Multi-tenant block GMAC (per-tenant ARMv8 PMULL seal + HKDF derivation), ② Visual prefix injection detector (ONNX YOLO-Nas-S + PaddleOCR + 5K signature DB), ③ PII redaction (Presidio + custom YOLO finetune) + ε-DP Rényi composition
- **R47 path**: SGLang 0.4.x fork + AutoAWQ + Presidio + Opacus + ARMv8 PMULL HW accel. application-level only. **Simulator 미사용**.
- **R55 self-check**: ✅ R55.1 simulator-building 부재, R55.2 5-axis tag, R55.3 mechanism Block 1.5
- **흡수**: WARDEN-KV (legacy v2 staging multi-tenant GMAC), EPSILON-VEIL (algo v2 ε-DP), PIIVEIL-Q (algo v2 PII redaction), BIMODAL-MASK-T2 (v1 modality outlier mask)

### STORMGLASS — Skin-Temp Leading-Indicator nvpmodel Hysteresis + UMA Power Split + Thermal-Aware Admission ⭐ NEW v2 Tier-2
- **Date**: 2026-04-27 (v2-r55) | **Session**: 세션 / Summary
- **Tier**: Tier-2 독립 | **Target**: ISLPED 2027 6p / DAC 2027 6p / IEEE ESL
- **Score**: Novelty 7.0 / Diff 7.5 / Impact 7.5 → mean **7.3**
- **5-axis Gain (R55.2)**: **[Power]** 30분 sustained throughput +18%, thermal envelope 안 안정화 / **[Energy]** -14%
- **Hypothesis**: Edge VLM 의 thermal envelope (7-130W) 안 sustained workload 안정화 — Tegrastats `temp` field 1Hz polling + skin-temp leading-indicator hysteresis + UMA GPU/CPU power split + admission control 통합
- **Mechanism (3)**: ① Skin-temp leading-indicator nvpmodel hysteresis (±2°C), ② UMA GPU/CPU power split (LPDDR5X 273 GB/s 동적 allocation), ③ Thermal-aware admission control (skin temp >75°C 시 backpressure)
- **R47 path**: vLLM 0.7.x fork + nvpmodel CLI + Tegrastats parser. application-level only. **Simulator 미사용**.
- **R55 self-check**: ✅ 모두 통과
- **흡수**: THERMOGUARD-V (ai-opt v2 staging) FP16 NaN fallback sub-mechanism

### ROBUSTOKEN — VLM Adversarial-Robust Token Saliency + Numerical Tolerance + OOD Rejection (5-Cluster Merge) ⭐ NEW v2 Tier-2
- **Date**: 2026-04-27 (v2-r55) | **Session**: 세션 / Summary
- **Tier**: Tier-2 독립 | **Target**: NeurIPS 2026 main / ICLR 2027 / EMNLP 2026 findings / TMLR
- **Score**: Novelty 8.0 / Diff 7.5 / Impact 7.5 → mean **7.7**
- **5-axis Gain (R55.2)**: **[Robustness]** PGD-8/255 attack +14-22pp / silent corruption detection 99.7% / OOD F1 ≥0.88 / hallucination 18-22% → ≤4% / **[Performance]** overhead -3~-8% / **[Cost eff.]** smaller-model fallback latency -40%
- **Hypothesis**: VLM robustness 의 통합 paradigm — adversarial / silent corruption / OOD 3 axis 통합 inference-time + training-free + edge-fit
- **Mechanism (3, 5-cluster merge)**: ① Adversarial-Robust Token Saliency (PGD/FGSM defense + cross-attention spike detection, PROBE-DECODE 흡수), ② Silent Corruption Tolerant Path (warp-ballot NaN/Inf detect + INT8/BF16 fallback + LPDDR5X ODECC syndrome cross-check, SCRIVENER + NAN-SAFENET 흡수), ③ Energy-Based OOD Rejection (5-class confidence + smaller-model fallback, REEFCAST + OOD-CALIB 흡수)
- **R47 path**: vLLM 0.7.x fork + ART v1.18 + TextAttack + lm-evaluation-harness + Linux EDAC subsystem. application-level only. **Simulator 미사용**.
- **R55 self-check**: ✅ 모두 통과
- **흡수**: SCRIVENER (legacy v2), NAN-SAFENET (algo v2), PROBE-DECODE (ai-opt v2), REEFCAST (legacy v2), OOD-CALIB (algo v2) — 5 cluster merge

### v1 retain (R55 5-axis tag 적용 후 retain 3)

- **ATRIUM** (Tier-1, retained from v1) — [Performance] decode +14% / [Energy] -12% / [Memory eff.] DeepStack L0-3 alloc skip. AGX Thor SM partition + L2 carveout + DeepStack alloc skip. 상세: Summary
- **BREAKWATER-T** (Tier-1, retained from v1) — [Performance] prefill -28% / throughput +45% / [Energy] -18%. dual-Jetson ViT layer-N/2 split + 4-bit channel-wise tap + USB-C 3.2 gen2x2. Summary
- **CASCADE-PREFILL** (Tier-2, repositioned from v1 Tier-1) — [Performance] prefill -12-17% / [Energy] -8-13% / [Cost eff.] $249 Orin Nano fit. NEON BF16 fused chunked prefill GPU↔CPU. Summary

### v1 → v2 미선정 (3)

- **HARMONY-LANE-MERGED** (v1 Tier-2 → v2 미선정): semantic routing axis 가 ATRIUM layer-aware dispatch 와 부분 overlap, narrow scope. 재방문 조건: Multi-VLM model serving 시나리오 (5+ model)
- **BIMODAL-MASK-T2** (v1 Tier-2 → v2 미선정): KV memory eff. axis 가 ATRIUM (alloc skip) + VEILSEAL-KV (-28%) 와 overlap, axis 중복. modality-conditioned dtype 만 VEILSEAL-KV sub-mechanism 흡수
- **JETTYSIM** (v1 Tier-2 → v2 자동 배제): **R55.1 위반** (simulator infrastructure 자체 contribution). 재방문 조건: ISPASS / IISWC tool track 별도 paper

---

## Selected Ideas (2026-04-27 v1 — VLM↔LLM Asymmetry on Single/Dual-Jetson Edge)

### ATRIUM — AGX Thor Layer-Asymmetric SM Partition + L2 Carveout + DeepStack L0-3 Alloc Skip (Tier-1 Top 1)
- **Date**: 2026-04-27 | **Session**: 세션 / Summary
- **Tier**: Tier-1 lead | **Target**: HPCA / MICRO / ASPLOS 2027
- **Score**: Novelty 8.0 / Diff 8.0 / Impact 7.0 → mean **7.7**
- **Hypothesis**: Qwen3-VL-4B 의 layer-wise visual attention 이 L17-21=24.5% / L0-7=2.6% 로 5-10× 비대칭, AGX Thor 273 GB/s LPDDR5X UMA 를 시간-분할이 아닌 **공간-분할** (Green Context SM 1500/1060 split + L2 4MB carveout + DeepStack L0-3 alloc skip) 시 BW 공유 충돌 회피.
- **Mechanism (3)**: ① LayerClassifier (visual_attn_pct 24.5% 기준 HOT/COLD 분류, hook-based, hot=L17-21 cold=L0-7,L22-35), ② SM Partition + L2 Carveout (Green Context CUDA 12.4 공식 API, 1500/1060 split, libsmctrl secondary), ③ DeepStack L0-3 alloc skip (visual KV 의 sparse layer 만 GPU HBM, hot layer 만 L2 carveout 활용).
- **R47 path**: vLLM fork + LLMServingSim+Ramulator2 plugin (R47.3) + 실기 AGX Thor primary.
- **R45 risk**: 4/10 (Green Context user-space ✅, libsmctrl secondary fallback).
- **GAP**: Q Cache (arXiv 2602.01901) algorithm decode skip only / KVTuner (ICML 2025) KV quant only / MIG (NVIDIA H100) 정적 partition — layer-aware dynamic SM partition + L2 carveout + DeepStack L0-3 alloc skip 통합 axis 직교.
- **예상 효과** (보수치, VLM-only): decode +14% (BS≥4) / BW waste 86%→32% / energy/token -12% / LLM-only 0% / single AGX Thor 128GB.

### BREAKWATER-T — Dual-Jetson ViT-Internal Layer Split + DeepStack Tap Compression (Tier-1 Top 2)
- **Date**: 2026-04-27 | **Session**: 세션 / Summary
- **Tier**: Tier-1 | **Target**: MICRO / ISCA / MLSys 2027
- **Score**: Novelty 7.5 / Diff 8.0 / Impact 7.4 → mean **7.6**
- **Hypothesis**: ViT 를 layer N/2 지점에서 split — device A (Orin Nano 8GB, sensor-proximity, MIPI-CSI 직결) = ViT 전반부 (low-level edge/depth) + device B (AGX Orin 64GB) = ViT 후반부 + LLM 전체. **DeepStack tap point = split point 일치**로 자연 layer 동기화 + 4-bit channel-wise tap stream 으로 USB-C 부담 ¼ 축소.
- **Mechanism (3)**: ① ViT layer-N/2 split (Orin Nano = layers 1..N/2, AGX Orin = layers N/2+1..N + LLM), ② 4-bit channel-wise tap quantization (DeepStack 의 ViT-L1 / L4 / L_N tap point 마다, RD curve 기반 channel-wise scale), ③ Pipeline depth=1 batching (next frame ViT 전반부 || current frame ViT 후반부+LLM) + Cosmos Reason2 single-tap fallback.
- **R47 path**: vLLM + LLMServingSim ViT extension (R47.4) + DRAMSim3 LPDDR5X timing + 실기 PyTorch hook (R45.2 user-space).
- **R45 risk**: 5/10 (USB-C 3.2 gen2x2 20Gbps cable variance 의존, NVIDIA Holoscan SDK reference). 모두 user-space.
- **GAP**: EPDServe (ICML 2025) encode 전체 single device / PipeDream / EdgeShard LLM only / Distributed VLMs (Columbia 2025) cloud-edge collaboration ViT 통째 edge — BREAKWATER-T 는 ViT-internal layer split + DeepStack tap quantized streaming + sensor-proximity dual-Jetson axis.
- **예상 효과**: prefill -28% / throughput +45% (streaming VLM ≥30 FPS) / energy -18% (sensor-proximity).

### CASCADE-PREFILL — Arithmetic-Intensity-Driven GPU↔CPU NEON Chunked Prefill (Tier-1 Top 3)
- **Date**: 2026-04-27 | **Session**: 세션 / Summary
- **Tier**: Tier-1 | **Target**: MLSys / IISWC / ASPLOS 2027
- **Score**: Novelty 7.5 / Diff 7.5 / Impact 8.2 → mean **7.7**
- **Hypothesis**: VLM chunked prefill 의 small-chunk tail (C ≤ 32 → AI < 60 강한 memory-bound region, GPU 8 분의 1 만 활용) 을 **ARM A78AE NEON BF16 fused VLM kernel** (200 GFLOPS, hidden=2560 head_dim=128 fused GEMM+RMSNorm+SiLU) 로 offload + UMA pinned KV barrier coherence (1-2us vs cudaMemcpy 70us).
- **Mechanism (3)**: ① RoofShim (chunked prefill chunk size C 별 AI 계산 + roofline 비교 → GPU/CPU dispatcher), ② NeonKernelPak (ARM NEON BF16 fused VLM kernel, hidden=2560 head_dim=128, llama.cpp ggml CPU backend extension), ③ UMAKVCoherence (cudaHostAllocMapped pinned KV + memory barrier 1-2us, modality-aware chunk size 흡수: vision chunk C=32, text chunk C=128).
- **R47 path**: llama.cpp fork + ggml CPU NEON backend / UMA pinned KV / user-space only (R47.2).
- **R45 risk**: 4/10 (Orin Nano 8GB single $249 reproducibility).
- **GAP**: PowerInfer (SOSP 2024) activation-level hot/cold split / FlexGen (ICML 2023) batch throughput / Hybrid CPU/GPU Inference (ACM 2025) generic LLM AVX-512 / llama.cpp ggml stock single-backend per graph — CASCADE-PREFILL 은 chunk-level AI dispatch + VLM-specific Qwen3-VL fused kernel + UMA barrier coherence axis.
- **예상 효과**: prefill -12~17% / throughput +17~25% / energy -8~13% / GPU idle 18%→9% / Orin Nano $249 reproducibility.

### HARMONY-LANE-MERGED — Semantic-Class Multi-Axis Routing (Cluster Merge of HARMONY-LANE + HALYARD + SEMACLASS-PRUNE) (Tier-2 Top 1)
- **Date**: 2026-04-27 | **Session**: 세션 / Summary
- **Tier**: Tier-2 독립 | **Target**: DAC 2027 6p / EMNLP 2026 findings / IISWC
- **Score**: Novelty 6.5 / Diff 7.0 / Impact 8.4 → mean **7.3**
- **Hypothesis**: 3 expert (ai-opt HARMONY-LANE + legacy HALYARD + algorithm SEMACLASS-PRUNE) 가 동일 axis (semantic class router) 도출 → 1 idea 통합 + multi-axis (resolution / class-conditional pruning ratio / Bayesian retention / chunk size) 모두 흡수.
- **Mechanism (3)**: ① SemanticOracle (5-class: Spatial/OCR/VQA/Counting/Video classifier, MiniLM-L6 tiny 4-bit quant, Orin Nano 0.3 ms), ② ResolutionLane + class-conditional pruning ratio (Spatial=672/45% pruning, OCR=1344/15%, VQA=896/30%, Counting=1344/20%, Video=448/55%), ③ Bayesian retention + ConfBackoff entropy retry.
- **R47 path**: Orin NX 16GB single / vLLM fork + ECVL-ROUTER scheme reference (R47.2).
- **R45 risk**: 3/10 (모두 user-space).
- **GAP**: ECVL-ROUTER ([arXiv:2510.27256](https://arxiv.org/abs/2510.27256)) scenario-aware VLM routing / vLLM Semantic Router ([arXiv:2603.04444](https://arxiv.org/abs/2603.04444)) signal-driven / SparseVLM ([arXiv:2410.04417](https://arxiv.org/abs/2410.04417)) attention sparsity — HARMONY-LANE-MERGED 는 5-class semantic + class-conditional resolution + Bayesian retention + chunk size axis 통합.
- **예상 효과**: TTFT P99 -22~34% (mixed workload) / ViT compute -40~60% / accuracy ≤0.5pp drop.

### JETTYSIM — LLMServingSim Dual-Jetson Topology Extension (Tier-2 Top 2, Infrastructure)
- **Date**: 2026-04-27 | **Session**: 세션 / Summary
- **Tier**: Tier-2 독립 (infrastructure) | **Target**: ISPASS 2026 / IISWC tool track
- **Score**: Novelty 8.5 / Diff 6.5 / Impact 6.0 → mean **7.0**
- **Hypothesis**: 본 세션 Tier-1 3 idea (ATRIUM/BREAKWATER-T/CASCADE-PREFILL) 모두 dual-Jetson + UMA + USB-C 토폴로지 가정, 기존 LLMServingSim 은 NVLink/PCIe cluster 가정. dual-Jetson edge topology 의 simulator 부재 → 자체 contribution + Tier-1 3 의 R47.4 backbone.
- **Mechanism (3)**: ① UMA model + USB-C link model (10/20 Gbps, 30us roundtrip + variance ±5ms cable burst), ② DeepStack tap support + ViT-internal split simulation, ③ Spec-decode trace + Power model (15W/25W/30W/60W/130W per-Jetson).
- **R47 path**: simulator-only (R47.3 LLMServingSim extension).
- **R45 risk**: 2/10.
- **GAP**: LLMServingSim ([arXiv:2408.05086](https://arxiv.org/abs/2408.05086)) NVLink/PCIe cluster only / Splitwise (ISCA 2024) cluster / DistServe (OSDI 2024) cluster — JETTYSIM 은 dual-Jetson UMA + USB-C topology first-to-report.
- **예상 효과**: infrastructure (정량 효과 X). Tier-1 3 + 후속 연구의 simulation backbone.

### BIMODAL-MASK-T2 — KV Modality Outlier Topology + Per-Block Dtype Dispatch (Tier-2 Top 3)
- **Date**: 2026-04-27 | **Session**: 세션 / Summary
- **Tier**: Tier-2 독립 | **Target**: EMNLP 2026 findings / NeurIPS 2026 ENLSP / ACL short
- **Score**: Novelty 5.0 / Diff 7.5 / Impact 7.6 → mean **6.7**
- **Hypothesis**: VLM KV cache 의 outlier 가 modality 마다 다른 topology — vision Moran's I = 0.34 (spatial cluster), text Moran's I = 0.02 (channel-uniform). modality-conditioned outlier mask 로 per-block dtype dispatch (NVFP4 / INT8 / BF16) + STACK-BAND weight band 흡수.
- **Mechanism (3)**: ① Modality-conditioned outlier mask (vision spatial Moran's I=0.34 cluster mask vs text channel uniform mask), ② Per-block dtype dispatch (cluster=BF16, sparse=NVFP4, weight band=INT8), ③ DeepStack OR-merge (ViT-L1/L4/L_N injection 별 mask OR 결합 + LLM layer-wise application).
- **R47 path**: Orin NX 16GB / Thor 128GB / dual-Jetson / vLLM custom block manager (R47.2).
- **R45 risk**: 3/10.
- **GAP**: KIVI (ICML 2024) 2-bit KV / KVTuner (ICML 2025) KV quant / MBQ (CVPR 2025) modality-balanced quant / MadaKV (ACL 2025) — BIMODAL-MASK-T2 는 modality outlier topology (Moran's I) + per-block dtype + DeepStack OR-merge axis.
- **예상 효과**: KV memory -23~29% / accuracy ≤0.4pp drop.

---

## Selected Ideas (2026-04-26 PM — Rowhammer + Memory ECC + RAS)

### RFM-COP — Phoenix-Aware Memory Controller RFM Scheduler with RogueRFM Fuzzing and TPRAC Defense (Tier-1 Top 1)
- **Date**: 2026-04-26 (PM) | **Session**: 세션 / Summary
- **Tier**: Tier-1 lead | **Target**: USENIX Security 2026 / S&P'26 / MICRO'26
- **Score**: Novelty 7.5 / Diff 8.5 / Impact 8.5 → mean **8.2**
- **Hypothesis**: McSee (USENIX Sec'25) 가 측정한 "Intel/AMD MC 어느 것도 rowhammer 시 RFM 명령을 보내지 않음" 을 host-MC scheduler 에서 직접 해결. Phoenix CVE-2025-6202 (S&P'26, 2025-09) 가 SK Hynix DDR5 in-DRAM mitigation 109초 우회 → host-MC fallback 의 결정적 가치.
- **Mechanism (4, attack-vector exception)**: ① RFM Issue Scheduler (PRAC ACT-N polling + ARFM race-close), ② Workload-aware RFM throttle (LLM exponent / graph hub 우선), ③ RogueRFM Fuzzing (LFSR-32 ±5 cycle), ④ DRFM directed RFM 통합.
- **Verilog 합성**: ~10K gate, 0.005-0.006 mm² @ TSMC 7nm, 5-8 mW. DDR5 MC IP 의 0.1-0.2%.
- **Simulator path**: Ramulator 2.0 PRAC+RFM plugin + ChampSim+DRAMSim3 rowhammer fork + FaultSim Monte Carlo + Hammulator+gem5 Inter-VM.
- **GAP**: ARFM ([arXiv:2501.14328](https://arxiv.org/abs/2501.14328)) 단일 mechanism. RFM-COP 4-pillar 통합 unique. Phoenix urgency response.
- **예상 효과**: Phoenix bypass < 1% (vs vanilla 99%, in-DRAM 4-8%, ARFM 0.5%), slowdown 1.0-2.5%, RogueRFM covert ≤8 bps (vs 1-10 kbps), TPRAC mutual info < 0.1 bit/access.

### KEYSTONE — Cryptographic GMAC for KV Cache Integrity in LLM Serving (Tier-1 Top 2)
- **Date**: 2026-04-26 (PM) | **Session**: 세션 / Summary
- **Tier**: Tier-1 | **Target**: USENIX Security 2026 / S&P'26 / ASPLOS'26
- **Score**: Novelty 7.0 / Diff 8.0 / Impact 8.0 → mean **7.7**
- **Hypothesis**: GPUHammer ([arXiv:2507.08166](https://arxiv.org/abs/2507.08166)) + KV-Cache Bit-Flip ([arXiv:2604.17249](https://arxiv.org/abs/2604.17249)) 의 attack 면에서 vLLM PagedAttention 64KB block 단위 AES-128 GMAC 으로 multi-bit + targeted attack 의 adversarial-secure unforgeable detection (collision 2⁻⁶⁴).
- **Mechanism (3)**: ① MC AES-128 GMAC core (~12K gates) + 64-entry block-id TLB, ② vLLM PagedAttention allocator hook → MC mailbox 등록, ③ Read-time MAC verify → mismatch 시 poison flag + IRQ → block-level re-prefill.
- **Verilog 합성**: ~17.5K gate + 1K FF, 0.018 mm² @ 7nm, 9 mW.
- **Simulator path**: Ramulator2 KEYSTONE plugin + ChampSim+DRAMSim3 + FaultSim FAR/FRR + Hammulator targeted bit-flip + lm-eval-harness.
- **GAP**: N2 KV-Cache Bit-Flip linear CRC-32 forgeable, GMAC unforgeable. Bonsai Merkle Tree 4KB granularity vs PagedAttention 64KB. SEV-SNP / Apple MIE / NVIDIA H100 CC cache-line 단위.
- **예상 효과**: Multi-bit + targeted attack 100% detection, FAR < 2⁻⁶⁴, throughput overhead 1.5-3%, MMLU accuracy preservation < 0.5% drop (vs -30~40% under attack).

### HARBOR — App-Hint-Driven CXL Controller Integrated RAS Engine with Workload-Phase-Adaptive Scrubbing (H2+L4 merge, Tier-1 Top 3)
- **Date**: 2026-04-26 (PM) | **Session**: 세션 / Summary
- **Tier**: Tier-1 | **Target**: ASPLOS'26 / HPCA'26 / DSN'26
- **Score**: Novelty 7.0 / Diff 7.5 / Impact 8.0 → mean **7.5**
- **Hypothesis**: CXL 3.x RAS register set (Patrol Scrub Control + ECS mailbox + DPA poison list + MER + AER + IDE) 이 표준화되었지만 host CPU mailbox polling 의존 → workload-aware 부재. Microchip SMC2100 commercial 도 phase-adaptive + app-hint 부재.
- **Mechanism (3, H2+L4 merge from 8)**: ① Integrated RAS Engine (HW backend, ~0.014 mm²), ② App→Kernel Hint Bridge (PyTorch `torch.utils.cxl_hint.mark_region()` + Linux `cxl_hint_bridge.ko` ~300 LoC + vLLM hook), ③ CXL.io vendor-defined message + page migration trigger (M5 cooperative).
- **Verilog 합성**: ~15K gate + 8KB SRAM, 0.014 mm² @ 7nm, 18 mW. CXL controller IP 3-5 mm² 의 0.3-0.5%.
- **Simulator path**: gem5 v23.0+ CXL Type-3 + Ramulator2 backend + DRAMSim3 + Hammulator Inter-VM + Linux 6.16 EDAC mainline reference + M5 cooperative.
- **GAP**: TPP (OSDI'23) / Pond (ASPLOS'23) / HMSDK (ASPLOS'24) placement-hint family — RAS-hint 으로의 확장 부재. Microchip SMC2100 commercial — workload-phase-adaptive + app hint 부재.
- **예상 효과**: Hot region scrub rate 8-16×, MER polling overhead 95% reduction, M5 page migration latency -40%, end-to-end LLM serving overhead < 1.5%.

### LIGHTHOUSE — Workload-Stress-Aware Lifetime Reliability Methodology (Tier-2 Top 1)
- **Date**: 2026-04-26 (PM) | **Session**: 세션 / Summary
- **Tier**: Tier-2 독립 | **Target**: DSN'26 / IISWC'26
- **Score**: Novelty 7.0 / Diff 7.0 / Impact 8.0 → mean **7.3**
- **Hypothesis**: FaultSim (TACO'16) 의 fault model 은 G2 field statistics 와 일치 정확하지만 simulator core 가 workload thermal stress + activation pattern 미반영. SCREME ([arXiv:2509.06101](https://arxiv.org/abs/2509.06101)) 도 spare-chip pool only.
- **Mechanism (3)**: ① Workload Stress Profiler (ChampSim+DRAMSim3 thermal trace 시계열), ② Stress-Modulated Fault Injector (G2 baseline + Arrhenius + McSee rowhammer overlay), ③ Multi-ECC Comparison Engine + 99th-percentile UER reporter.
- **R50-α mitigation**: FaultSim core grace, fault model only import, simulator core 는 본 idea 의 새 wrapper.
- **예상 효과**: 7-year UER prediction accuracy ±50% → ±15% (1.5-3× more accurate), 1M trial × 5 scheme 1200 sec parallelizable.

### MOSAIC-TRACE — Open-Source Standardized Rowhammer/ECC Workload-Trace Suite (Tier-2 Top 2)
- **Date**: 2026-04-26 (PM) | **Session**: 세션 / Summary
- **Tier**: Tier-2 독립 | **Target**: IISWC'26 / ISPASS'26 / OSDI'26 artifact
- **Score**: Novelty 6.0 / Diff 6.0 / Impact 8.5 → mean **6.8**
- **Hypothesis**: 본 세션 6 idea 의 reproducibility + community benchmark 부재. Mess ([arXiv:2405.10170](https://arxiv.org/abs/2405.10170)) / Antmicro / Google rowhammer-test 는 fault injection only, ECC scheme cross-product 부재.
- **Mechanism (3)**: ① Unified Trace Format + Workload Suite (5 simulator adapter + 18 workload × 10 SimPoint = 180 trace), ② PRAC Plugin Suite (MOAT/MoPAC/AutoRFM/MINT/QPRAC/ARFM 6종 unified API), ③ McSee Calibration Harness (90 calibration point).
- **Deploy**: Docker / Nix flake / Zenodo DOI artifact.

### RAMPART — Hot-Row Infrastructure with Dual PRAC + ECC Tier Promotion (S2+H1 merge, Tier-2 Top 3)
- **Date**: 2026-04-26 (PM) | **Session**: 세션 / Summary
- **Tier**: Tier-2 독립 | **Target**: HPCA'26 short / DAC'26 / DATE'26 / DSN'26
- **Score**: Novelty 7.0 / Diff 7.5 / Impact 7.0 → mean **7.2**
- **Hypothesis**: PRAC counter (rowhammer) 와 ECC tier (multi-bit) 가 별도 infrastructure → area + power 중복. Hydra (HPCA'22) / Mithril (DSN'22) 2-tier 가 있지만 ECC coupling 부재.
- **Mechanism (3, S2+H1 merge from 8)**: ① Hot-Row Tracking Infrastructure (Hot 8K SRAM-PAT + Cold 64K CAM-PAT + MC sideband layer-id hint), ② PRAC Counter Promotion (Hot 71, Cold 256, Cold→Hot 승격), ③ ECC Tier Promotion (Hot SECDED→DECTED + 인접 row 자동 DECTED, Cold SECDED 유지).
- **Verilog 합성**: ~25K gate + 540Kbit SRAM/CAM, 0.025-0.03 mm² @ 7nm, ~31 mW. HBM3 base die 80mm² 의 0.04%.
- **GAP**: PVAC ([arXiv:2604.20576](https://arxiv.org/abs/2604.20576)) victim-row counting (root cause) vs RAMPART aggressor-counter + sideband + ECC coupling.

---

## 미선정 — 2026-04-26 PM Rowhammer ECC RAS

| ID | Title | Status | 사유 |
|----|-------|--------|-----|
| S1 SANCTUM | Datatype-asymmetric ECC LLM | DROP | REACH ([arXiv:2512.18152](https://arxiv.org/abs/2512.18152)) + Domain-Specific ECC AI ([arXiv:2507.02654](https://arxiv.org/abs/2507.02654)) 80-85% 이중 scoop |
| H4 SURGE-ECC | Base die datatype DECTED | DROP | REACH base-die variant 80% overlap |
| L2 JANUS | OS page dual-mode ECC | DROP | REACH + Domain-Specific 70-75% scoop |
| S5 CHRONOSCRUB | Refresh-cycle stealth scrub | ABSORBED | L1 MOSAIC superset, S5 specific implementation |
| L1 MOSAIC | Phase-aware ECS scheduler | Reserve | Tier-2 Conditional, Top-6 우선순위 후순 |
| H5 TWIN-CLOCK PRAC | Cross-layer twin counter | Reserve | Chronus ([arXiv:2502.12650](https://arxiv.org/abs/2502.12650)) 50-60% concurrent |
| S4 STRIPE-ECC | Pseudo-channel RS(10,8) | Reject | REACH outer-RS 50-55% overlap |

---

## Selected Ideas (2026-04-26 AM — KV cache ECC + Memory RAS v2)

### PrefixGuard ★ — Reliability-Aware Prefix Cache Eviction & Scrub Scheduling for CXL-Attached KV Storage in Multi-Tenant LLM Serving (Tier-1 Top 1)
- **Date**: 2026-04-26 | **Session**: [링크](/research-wiki/2026-04/kv-ecc-ras-v2) / Summary
- **Tier**: Top-tier lead | **Target**: OSDI 2027 (13p, primary) / ASPLOS 2027 / DSN 2027
- **Experts**: system-robustness-expert (메인). Phase 1'' scores: Nov **9.0** / Diff **9.5** / Imp **8.0** / Feas **9.0** → avg **8.9**
- **Hypothesis**: CXL-attached prefix block 의 hour-scale lifetime (KVCache-in-Wild USENIX ATC'25) 에 맞춰 Patrol Scrub Control hour-단위 interval 차등 (long-lived 1hr / short-lived disabled) → silent corruption 90% 감소 + scrub overhead <30%.
- **Mechanism (3, improve-only, ΔM=0)**:
  - M1 **8-bit Lifetime Tracker Tag**: PagedAttention 16-token block hash table 에 lifetime tag 추가 (0=ephemeral / 1-127=lifetime in 10-min units / 128-255=permanent prefix).
  - M2 **3-tier scrub interval × prefix lifetime alignment**: `LMCacheConnector` 측 별도 thread 가 Linux 6.16 EDAC scrub_subsystem sysfs polling — long-lived 1hr scrub, medium-lived 5min, short-lived disabled.
  - M3 **LLMServingSim 8-32 GPU + 256GB CXL pool cluster**: ECS history × prefix hit ratio trade-off + Meta C5 cross-check.
- **GAP**: D1 LMCache ([arXiv:2510.09665](https://arxiv.org/abs/2510.09665)) / D2 TraCT ([arXiv:2512.18194](https://arxiv.org/abs/2512.18194)) / Beluga ([arXiv:2511.20172](https://arxiv.org/abs/2511.20172)) 가 모두 latency/throughput axis — reliability/scrub axis 부재. CXL 3.2 Patrol Scrub Control + Linux 6.16 EDAC scrub_subsystem 활용 first work.
- **예상 효과**: silent corruption rate 90% 감소, scrub overhead <30%, prefix hit ratio drop <2%.

### Quarantine — Per-Agent DPA-Level KV Cache Poison Isolation for Agentic Multi-Turn LLM Serving on CXL Pools (Tier-1 #2)
- **Date**: 2026-04-26 | **Session**: [링크](/research-wiki/2026-04/kv-ecc-ras-v2) / Summary
- **Tier**: Top-tier | **Target**: USENIX Security 2027 (13p, primary) / OSDI 2027 / DSN 2027
- **Experts**: system-robustness-expert (메인). Phase 1'' scores: Nov **9.0** / Diff **8.5** / Imp **8.5** / Feas **8.0** → avg **8.5**
- **Hypothesis**: KVFlow multi-agent prefix cache 환경에서 한 agent 의 KV CE 가 다른 agent 로 silent propagate. Targeted BFA on Agents ([arXiv:2603.10042](https://arxiv.org/abs/2603.10042)) 직접 threat. CXL 3.x DPA tracking + poison + ECS mailbox + Memory Event Record 4 feature 결합 → cross-agent corruption 100% 차단 + throughput drop <5%.
- **Mechanism (3, improve-only, ΔM=0)**:
  - M1 **agent_id × DPA_range 2-level sparse hash table**: top agent_id → DPA-block-list, leaf DPA → KV block hash. worst 128MB → average 4-16MB.
  - M2 **CXL ECS mailbox query (5s poll + on-demand poison event)**: trusted (within-tenant) / untrusted (cross-tenant) zoning + vLLM RFC #19329 affected token-range recompute.
  - M3 **LLMServingSim 4-16 multi-agent cluster**: corruption injection × isolation effectiveness 측정.
- **GAP**: CacheSolidarity ([arXiv:2603.10726](https://arxiv.org/abs/2603.10726)) timing side-channel 만, KVFlow ([arXiv:2507.07400](https://arxiv.org/abs/2507.07400)) reliability axis 부재, TraCT 도 poison/DPA tracking 미언급. CXL DPA × multi-agent KV poison isolation first work.
- **예상 효과**: cross-agent corruption rate 100% 차단, multi-agent throughput drop <5%, recompute < TPOT 5× (200ms).

### PATroller — HBM3 Pseudo-channel Activation Timing Counter as a Hot-Block Identifier for Reliability-Aware KV Migration (Tier-1 #3)
- **Date**: 2026-04-26 | **Session**: [링크](/research-wiki/2026-04/kv-ecc-ras-v2) / Summary
- **Tier**: Top-tier | **Target**: HPCA 2027 (12p, primary) / DSN 2027 / MICRO 2027
- **Experts**: system-robustness-expert (메인). Phase 1'' scores: Nov **8.0** / Diff **8.0** / Imp **8.0** / Feas **8.5** → avg **8.1**
- **Hypothesis**: HBM3 PAT counter top-k row (1s polling, top-32) 를 software-level KV migration trigger 로 활용 (8K activations/sec threshold) → Rowhammer-induced silent corruption 95% 감소 + migration overhead <3%.
- **Mechanism (3, improve-only, ΔM=0)**:
  - M1 **PAT-counter polling thread + reverse-mapping table**: 1s default polling, top-32 row, KV block hash → HBM row single-direction map.
  - M2 **Threshold-based migration via BlockManager.swap**: PAT > 8K activations/sec threshold, BlockManager.swap_in/swap_out 50 line.
  - M3 **NeuroSim V1.4 HBM3 cell wear validation + LLMServingSim cluster**: BTI/HCI fault rate calibration + Meta C5 cross-check.
- **GAP**: PRAC family (MOAT [arXiv:2407.09995](https://arxiv.org/abs/2407.09995) / QPRAC [arXiv:2501.18861](https://arxiv.org/abs/2501.18861) / CnC-PRAC [arXiv:2506.11970](https://arxiv.org/abs/2506.11970)) 모두 KV-aware 부재. v1 LayerTier 도 PAT counter 미사용. HBM3 PAT × KV migration first work (similarity clear <25%).
- **예상 효과**: silent corruption 95% 감소, migration overhead <3%, throughput drop <2%.

### ECS-Trace — HBM3 Error-Check-Scrub Mailbox History as Reliability Trace for KV Cache Block Lifetime Management (Tier-2 독립 #1)
- **Date**: 2026-04-26 | **Session**: [링크](/research-wiki/2026-04/kv-ecc-ras-v2) / Summary
- **Tier**: Tier-2 독립 | **Target**: ITC 2027 6p (primary) / IEEE TCAD short / DSN short
- **Experts**: system-robustness-expert. Phase 1'' scores: Nov **8.0** / Diff **8.0** / Imp **7.5** / Feas **8.0** → avg **7.9**
- **Hypothesis**: HBM3 ECS mailbox (IEEE 1500 TAP, 10s query interval, self-refresh cycle 32ms 와 align) history 를 KV block lifetime reliability trace 로 활용 → 누적 CE 가 collapse 직전인 block prefetch eviction → long-context (>128k) silent corruption 90% 차단.
- **Mechanism (3, improve-only)**:
  - M1 **ECS-history query thread (10s)**: IEEE 1500 TAP register sim emulated query.
  - M2 **LRU + ECS history threshold stack**: evict priority = LRU rank * 0.6 + cumulative CE rate * 0.4.
  - M3 **LLMServingSim + NeuroSim V1.4 (Meta C5 cross-check + Aliyun replay)**: 8 category 별 ECS pattern.
- **GAP**: HCache (EuroSys'25) state restoration, NACL/CaR eviction — 모두 ECS history axis 부재. HBM3 ECS mailbox × LLM KV lifetime first work.
- **예상 효과**: silent corruption 90% 차단, hit ratio drop <3%, ECS query overhead <0.1%.

### Quarantine-Mini — Single-Agent CXL DPA Poison Detection-to-Recompute Latency Profile for vLLM Token-Range Recovery (Tier-2 독립 #2, paper pair V3 ↔ P3)
- **Date**: 2026-04-26 | **Session**: [링크](/research-wiki/2026-04/kv-ecc-ras-v2) / Summary
- **Tier**: Tier-2 독립 (Paper pair P3 ↔ V3 1 쌍) | **Target**: DAC 2027 6p / IEEE TCAD short / IEEE CAL 4p
- **Phase 1'' scores**: Nov **7.0** / Diff **7.0** / Imp **7.0** / Feas **9.0** → avg **7.5**
- **Hypothesis**: CXL ECS mailbox poison event detect 후 vLLM RFC #19329 affected token-range recompute latency 가 token range × layer linear → 8K token + 32 layer 에서 detect-to-recompute < 200ms (TPOT 40ms × 5 budget 내).
- **Mechanism (1, scope-shrink design)**: M1 single-agent ECS poison detect-to-recompute latency profile only.
- **GAP**: vLLM RFC #19329 / vLLM-ascend RFC #5067 (mainstream Q2 2026 upstream) 의 production-ready latency profile 부재.
- **예상 효과**: detect-to-recompute < 200ms, throughput drop <3%, ECS poll overhead <0.05%.

### PrefixGuard-Lite — Empirical Calibration of Linux 6.16 EDAC Scrub Interval for CXL-Attached LLM Prefix Cache (Tier-2 독립 #3)
- **Date**: 2026-04-26 | **Session**: [링크](/research-wiki/2026-04/kv-ecc-ras-v2) / Summary
- **Tier**: Tier-2 독립 (P1 의 M2 분리, sister paper) | **Target**: DATE 2027 6p / IEEE CAL 4p
- **Phase 1'' scores**: Nov **7.0** / Diff **7.0** / Imp **6.0** / Feas **9.0** → avg **7.25**
- **Hypothesis**: Linux 6.16 EDAC scrub_subsystem hour-단위 interval knob 을 vLLM prefix lifetime histogram 에 fit → optimal scrub interval = p75 lifetime → silent corruption 70% 감소 + scrub overhead <5%.
- **Mechanism (1, scope-shrink design)**: M1 prefix lifetime histogram (1-min bucket, 8 Aliyun category) → p75 lifetime → EDAC sysfs write calibration.
- **GAP**: Linux 6.16 EDAC scrub_subsystem 은 kernel-side primitive 만 — application-level prefix lifetime 과 align 한 calibration first work.
- **예상 효과**: corruption 70% 감소, scrub overhead <5%, throughput drop <2%.

---

## 미선정 로그 (2026-04-26 KV cache ECC + Memory RAS v2)

- **P2 SinkShield (Tier-1 후보)** → **REPLACE 권고**. 사유: KVSink ([arXiv:2508.04257](https://arxiv.org/abs/2508.04257)) + SinkQ concurrent 55-65%, v1 EntropyECC 와 차별성 약화, HBM3 RFM granularity row/bank-level mismatch. 재방문 조건: KVSink 와 stack-able layer (RFM hardware × quantization software) reposition + sink_flag → row aggregation 정확 문서화.
- **P5 Watermark (Tier-2 후보)** → **Tier-2 보조 (Top 3 외)**. 사유: LM-Fix ([arXiv:2511.02866](https://arxiv.org/abs/2511.02866)) / RoR ([arXiv:2603.16382](https://arxiv.org/abs/2603.16382)) / BitFlipScope ([arXiv:2512.22174](https://arxiv.org/abs/2512.22174)) concurrent 60-70%, R21 paper pair 한계. 재방문 조건: KV cache vs weight axis stack 가치 정량 입증, attention kernel fuse 0.2-0.4ms overhead 실측.
- **P6 VideoVeil (Tier-2 후보)** → **REPLACE 권고**. 사유: Sali-Cache ([arXiv:2602.14236](https://arxiv.org/abs/2602.14236)) frame-importance source 동일 (compression vs reliability axis 차별화 부족), v1 VLM-MAP overlap. 재방문 조건: video-specific RAS 의 다른 axis (예: vision token outlier × HBM3 RFM, video stream temporal outlier × ECS) reposition.
- **P7 EdgeARM (Tier-2 후보)** → **Tier-2 보조 (Top 3 외)**. 사유: Kelle ([arXiv:2510.16040](https://arxiv.org/abs/2510.16040)) MICRO 2025 adjacent 35-45%, R21 paper pair 1 쌍 한계 (V3+V1 우선), Qualcomm/Samsung/MediaTek vendor-specific TAP register layout 검증 필요. 재방문 조건: 3 vendor register map 검증 + commodity SoC deployment generality 백서 + Kelle 와 deployment differential.
- **V4 PATroller-Solo (Tier-2 variant)** → **R21 paper pair 1 쌍 한계로 미선정 (Option C 선택)**. 사유: V3 ↔ P3 + V4 ↔ P4 동시 포함 시 2 쌍 위반. system-robustness-expert 권고: V3 가 vLLM RFC #5067 mainstream supplement 로 production impact 강함, narrative fit 우선. 재방문 조건: 추가 Tier-2 venue slot 확보 시 (IEEE CAL 추가 등) V3 와 swap 또는 독립 진입.

---

## Selected Ideas (2026-04-24 MoE Fingerprint Security+Serving)

### DISCRETE-VEIL' ★ — MoE Discrete Routing Adaptive-Adversarial Robustness (Tier-1 S&P lead)
- **Date**: 2026-04-24 | **Session**: [링크](/research-wiki/2026-04/moe-fingerprint-security-serving) / [Summary](/research-wiki/2026-04/moe-fingerprint-security-serving)
- **Tier**: Top-tier lead | **Target**: IEEE S&P 2027 (13p) / (Tier-2 DISCRETE-VEIL-Lite IEEE CAL 4p or DSN practical 6p)
- **Experts**: system-robustness-expert (메인). 3 reviewers Phase 2' scores: Nov **7.0** / Diff **8.0** / Imp **9.0** / Feas **8.0** → avg **8.00**
- **Metaphor**: "Veil" — 얇은 가림막이지만 combinatorial 구조가 공격을 차단하는지 검증
- **Mechanism (2, improve-only, ΔM=0)**:
  - M1 **DRO-Attack**: embedding-space PGD with Gumbel-softmax surrogate for discrete top-k argmax (joint CE + routing-pattern loss). Qwen3 + Mixtral 2 models × PAIR + GCG 2 attacks × white-box + gray-box 2 threat models.
  - M2 **Entropy-Sharpen-KS Tripwire**: DRO-attack 특화 entropy sharpening KS-test (ASE 2025 NIER "routing entropy OOD" 와 구분 — adversarial-specific forensics).
- **GAP (공격 공간 Venn diagram)**: [Obfuscated Activations arXiv:2412.09565](https://arxiv.org/abs/2412.09565) (dense hidden probe recall 100→0) / [V-MoE Adversarial OpenReview Fd05J4Bu5Sp](https://openreview.net/pdf?id=Fd05J4Bu5Sp) (image PGD) / [GateBreaker arXiv:2512.21008](https://arxiv.org/abs/2512.21008) (weight ablation ASR 7.4→64.9%) / [Expert Selections Reveal arXiv:2602.04105](https://arxiv.org/abs/2602.04105) (privacy leakage 공격) — 본 연구 **embedding-PGD on text MoE routing classifier** 는 4 공간 모두 non-overlap.
- **예상 효과**: Qwen3 WildJailbreak recall 94% → DRO-attack 후 ≥55% 유지 (하한), hidden-state linear probe 는 0-20% 붕괴 → **> 30%p 격차 면 discrete-robust 가설 성립**.
- **Phase 3 entry**: W1 baseline reproduction + Venn diagram, W2 PAIR/GCG 400 prompts, W3 DRO-Attack 구현, W4 fingerprint 재추출.

### LOOM' — MoE Fingerprint Shared-Substrate with Token×Layer 2D Early-Exit + Compressed Index (Tier-1 #2, merged)
- **Date**: 2026-04-24 | **Session**: [링크](/research-wiki/2026-04/moe-fingerprint-security-serving)
- **Tier**: Top-tier | **Target**: MLSys 2027 또는 ASPLOS 2027 (18p) / (Tier-2 LOOM-Core-Lite 별도 분리 안 함 — LOOM' 내 subset)
- **Experts**: ai-optimization-expert (메인). Phase 2' scores: Nov **6.0** / Diff **7.5** / Imp **8.5** / Feas **7.5** → avg **7.38**
- **Metaphor**: "Loom" — 실(fingerprint) 하나를 베틀에서 여러 가닥(consumer)으로 엮음
- **Mechanism (4, critical merge, ΔM=+1 for EMBER+THRESHOLD+TALLY artificial split 방어)**:
  - M1 **EPRT** (End-of-Prefill Pooled Router Tap): vLLM `_execute_model_forward` 말미 forward hook, prefill 종료 시 `(L, E)` pooled tensor 를 `SchedulerOutput` fingerprint_tensor 필드로 publish.
  - M2 **Token×Layer 2D Early-Exit Pareto**: (token prefix k*, layer depth L_k) 2D grid 에서 F1 / FLOPs / latency Pareto frontier. Adaptive layer budget runtime.
  - M3 **Multi-Consumer Fan-Out + Safety-Aware Admission**: 3 consumer (detection + residency + prefetch) + admission (reject / sandbox / full_decode).
  - M4 **LEAP + QIVF Compressed Index**: layer-expert variance pruning 256-dim + FAISS IVF-PQ 32B/vec → 253K pool ~8MB / ≤3ms query.
  - **추가 Systems-theory**: Fisher information Pareto proof (fingerprint F 의 N task 분배 상한 bound, task correlation eigenvalue 의존).
- **GAP**: [MoE-Infinity arXiv:2401.14361](https://arxiv.org/abs/2401.14361) [ATC 2024] (batch trace, detection 없음) / [PreScope arXiv:2509.23638](https://arxiv.org/abs/2509.23638) (prefetch only) / [DuoServe arXiv:2509.07379](https://arxiv.org/abs/2509.07379) (safety admission 없음) / [Gimbal arXiv:2602.21626](https://arxiv.org/abs/2602.21626) (expert footprint 미사용) / vLLM Semantic Router Iris v0.1 (2026-01 merge, LoRA classifier heads only, serving consumer 없음) / [FJD arXiv:2509.14558](https://arxiv.org/abs/2509.14558) [EMNLP 2025 Findings] (dense 1D token axis).
- **예상 효과**: detection F1 93-94%, expert cache miss rate -20-25%, decode p50 latency -15%, memory cost +10MB vs WildGuard +28GB.
- **사용자 lab 이전 세션 차별화**: 2026-04-21 FARD-C (외부 side-channel 85-90% 가정) 과 전제·가정 독립.

### BEACON-GUARD-Lite — Training-free Multi-task Unified Guard (Tier-1 #3 / Tier-2 primary, paper pair with LOOM')
- **Date**: 2026-04-24 | **Session**: [링크](/research-wiki/2026-04/moe-fingerprint-security-serving)
- **Tier**: Top-tier Tier-2 (paper pair with LOOM' Tier-1) | **Target**: USENIX ATC 2027 (12p) 또는 DATE 2027 (6p)
- **Experts**: system-robustness-expert (원안) → ai-optimization pivot (systems/deployment). Phase 2' scores (Tier-2 rubric): Nov **5.5** / Diff **6.5** / Imp **8.0** / Feas **9.0** → avg **7.25**
- **Metaphor**: "Beacon" — 하나의 등대가 여러 방향 (domain + safety) 을 비춤
- **Mechanism (2, improve+pivot, ΔM=0)**:
  - M1 **URFB** (Unified Routing-Feature Bank): `(L, 4E)` 통합 bank + FAISS IVF IndexIVFFlat(nlist=64) + 2-head k-NN. 사용자 5,900 runs 100% 재활용.
  - M2 **Multi-Signal Fingerprint Fusion Interpretability**: topk_index (discrete) + softmax_scores (continuous) + activation_count (integer) 3 signal per-class MMD + JS divergence. Signal-specific discriminability + weighted ensemble.
  - ~~M2.2 OOD-gate~~: DROPPED (ASE 2025 NIER scoop + DISCRETE-VEIL' defense-in-depth 로 이식)
- **GAP**: [WildGuard arXiv:2406.18495](https://arxiv.org/abs/2406.18495) [NeurIPS 2024] (safety only, +200ms +28GB) / LlamaGuard-3 (safety only, +100ms) / [OmniGuard arXiv:2505.23856](https://arxiv.org/abs/2505.23856) (safety only, 120x faster) / [FJD arXiv:2509.14558](https://arxiv.org/abs/2509.14558) [EMNLP 2025 Findings] (safety only) / [Task-Cond. Routing arXiv:2603.11114](https://arxiv.org/abs/2603.11114) (domain only, OLMoE preprint) / [MultiTaskGuard arXiv:2504.19333](https://arxiv.org/abs/2504.19333) (multi-task but LoRA training required) / vLLM Semantic Router Iris v0.1 (LoRA classifier heads, training required) — 본 연구는 **완전 no-train k-NN + 3 signal geometric analysis**.
- **예상 효과**: WildGuard 대비 latency 50-100× ↓ (200ms → 2-4ms), memory 28GB → <50MB, 2-task accuracy 96.2%/94% 유지.
- **Deployment path**: vLLM plugin, NeMo Guardrails adapter.

### DISCRETE-VEIL-Lite — Qwen3 + PAIR Only Precedence (Tier-2 독립 #1)
- Scope: Qwen3 + PAIR 200 prompts, Mech M1 only (DRO-Attack), 2 baselines (FJD + OmniGuard)
- **Target**: IEEE CAL 4p 또는 DSN practical 6p
- **Date**: 2026-04-24 | **Session**: [링크](/research-wiki/2026-04/moe-fingerprint-security-serving)
- 8주 완결. Tier-1 S&P 투고 전 "first-to-report MoE embedding-PGD" precedence claim.

### TALLY-Spinoff — LEAP MoE-specific Interpretability 축 (Tier-2 독립 #2)
- Scope: LEAP + SAFEx-style expert-j-at-layer-i (safety-critical / domain-critical / general) label attachment
- **Target**: DATE 2027 4p WIP
- **Date**: 2026-04-24 | **Session**: [링크](/research-wiki/2026-04/moe-fingerprint-security-serving)
- 4주 완결 가능. LOOM' Section 7 과 중복 회피하여 단독 venue.

### BEACON-GUARD-Lite DATE fallback — Multi-signal Fusion 단독 (Tier-2 독립 #3)
- ATC reject 시 DATE 6p 재제출. 2주 압축.

---

## 미선정 로그 (2026-04-24 MoE Fingerprint Security+Serving)

- **DISCRETE-VEIL 원안 (3 models × 3 attacks)** → **Refined → DISCRETE-VEIL'** (scope 2 models × 2 attacks, entropy tripwire → KS-test 재정의). 재방문 조건: GateBreaker 대응 defense 추가 시 Mech 3 확장.
- **BEACON-GUARD 원안 (USENIX Security Tier-1)** → **Tier-2 ATC/DATE 강등**. 사유: Task-Cond Routing + MultiTaskGuard + vLLM Semantic Router Iris + ASE 2025 NIER concurrent 압박. OOD Mech 2.2 는 DISCRETE-VEIL' 로 이식. 재방문 조건: Multi-signal interpretability proof + vLLM upstream PR 완료.
- **THRESHOLD (NDSS 단독)** → **DROP, LOOM' M2 token-axis 에 흡수**. 사유: EMBER 와 early-exit axis 중복 (artificial split). 재방문 조건: Crescendo 공격이 production 주류 되면 LOOM' streaming Section 확장.
- **EMBER (ATC/OSDI 단독)** → **MERGED, LOOM' M2 layer-axis 로 흡수**. 사유: LOOM 과 vLLM fork/FusedMoE hook 공유. 재방문 조건: LOOM' rejection 시 standalone ATC resubmit.
- **TALLY (NeurIPS Systems 단독)** → **DOWNGRADED, LOOM' M4 흡수 + DATE 4p WIP spinoff**. 사유: FAISS IVF-PQ + axis pruning incremental novelty 부족.

---

## Selected Ideas (2026-04-24 Qwen3-VL DeepStack Edge)

### Loom ★ — Interleaved MRoPE Unified LUT + FA3 Fused Rotation + Texture Unit (Tier-1 Top 1, lead)
- **Date**: 2026-04-24 | **Session**: [링크](/research-wiki/2026-04/qwen3vl-deepstack-edge) / [Summary](/research-wiki/2026-04/qwen3vl-deepstack-edge)
- **Tier**: Top-tier lead | **Target**: MLSys 2026 / ISCA 2027
- **Experts**: ai-opt + legacy-sys + hw-pim 3:0 unanimous (lead)
- **Review scores** (Phase 2'): Nov **7.4** / Diff **8.0** / Imp **7.7** / Feas **8.5** → **avg 7.90**
- **Metaphor**: "Loom" = 베틀, t/h/w 실이 low/high frequency band 에 interleave 되어 positional encoding 을 직조.
- **Mechanism (3, improve-first)**: (M1) Unified permuted LUT (chunked 3-table → interleaved single), (M2) FA3 tile-internal fused rotation (SFU 우회), (M3) Texture unit LUT fetch + register pack.
- **Qwen3-VL 아키텍처 diff 활용**: Interleaved MRoPE `[24,20,20]` round-robin frequency → chunked MRoPE (Qwen2-VL/2.5-VL) 에서 불가.
- **GAP**: [Revisiting MRoPE ICLR'26 arXiv:2510.23095](https://arxiv.org/abs/2510.23095) algorithm only / [vLLM PR #22593](https://blog.vllm.ai/2025/09/11/qwen3-next.html) generic partial rotary / Cartographer (2026-04-23) chunked 전제 / [T-MAC EuroSys'25](https://arxiv.org/abs/2407.00088) + [LUT Tensor Core ISCA'25](https://arxiv.org/abs/2408.06003) weight only.
- **예상 효과**: MRoPE kernel -30~45%, prefill TTFT -12~15%, MMMU 0 drop (bit-exact).
- **Phase 3 entry**: Nsight SFU busy % 실측 + CUTLASS FA3 fused rotation prototype.

### Mangrove — DeepStack Layer-Aware 4-Stage Pipeline + LPDDR Bank + DLA Offload (Tier-1 Top 2)
- **Date**: 2026-04-24 | **Session**: [링크](/research-wiki/2026-04/qwen3vl-deepstack-edge)
- **Tier**: Top-tier | **Target**: ASPLOS 2027 / MLSys 2026
- **Review scores**: Nov **6.6→7.0** (CLI cite) / Diff **7.5** / Imp **8.06** / Feas **7.8** → **avg 7.60**
- **Metaphor**: "Mangrove" = 맹그로브 나무, 여러 깊이 뿌리 (layer 8/16/24) 가 하나의 나무 지지.
- **Mechanism (3)**: (M1) Layer-aware 4-stage sub-graph + CUDA dual-stream (layer 0-7 vision-independent), (M2) LPDDR5X bank-aligned vision residual write, (M3) Layer 8/16/24 injection point DLA offload.
- **Qwen3-VL diff**: DeepStack `visual_indexes=[8,16,24]` multi-layer residual → Qwen2.5-VL single-layer concat 에서 불가.
- **GAP**: [DeepStack NeurIPS'24](https://arxiv.org/abs/2406.04334) + [Cross-Layer Injection arXiv:2601.10710](https://arxiv.org/abs/2601.10710) algorithm only, system-level first / [Nova arXiv:2509.21301](https://arxiv.org/abs/2509.21301) modality-stage axis orthogonal.
- **예상 효과**: TTFT 1.3-1.6× (multi-image batch), energy -18~28%.

### Vault' — DeepStack × MoE L2 Contention with LPDDR Bank Placement + Activation-Aware L2 Pin (Tier-1 Top 3, post-Major Revision)
- **Date**: 2026-04-24 | **Session**: [링크](/research-wiki/2026-04/qwen3vl-deepstack-edge)
- **Tier**: Top-tier (Major Revision) | **Target**: ASPLOS 2027 / HPCA 2027
- **Review scores**: Nov **5.4→6.8** (post-replacement) / Diff **8.5** / Imp **8.28 flagship** / Feas **5.6** (Thor gate) → **avg 7.30**
- **Metaphor**: "Vault" = 금고, 128 expert 를 Thor LPDDR5X 에 보관 + top-2 gating 으로 꺼내 사용.
- **Major Revision**: VEQ + ARCQuant + DyMoE + CC-MoE 4-way scoop 68-72% → NVFP4 per-expert axis drop, **DeepStack × MoE L2 contention** 으로 repositioning.
- **Mechanism (3, M1 REPLACE)**: (M1 replace) DeepStack × MoE L2 contention analytical model, (M2 유지) LPDDR5X bank-aligned expert placement, (M3 유지) Activation-aware L2 pinning + GDN dual working-set.
- **Qwen3-VL/3.5 diff 활용**: 30B-A3B MoE (Qwen3 신규) + NVFP4 (Blackwell/Thor) + GDN (Qwen3-Next/3.5) + DeepStack (Qwen3-VL).
- **GAP**: [VEQ arXiv:2602.01037](https://arxiv.org/abs/2602.01037) / [ARCQuant arXiv:2601.07475](https://arxiv.org/abs/2601.07475) / [DyMoE arXiv:2603.19172](https://arxiv.org/abs/2603.19172) / [CC-MoE arXiv:2509.25689](https://arxiv.org/abs/2509.25689) — DeepStack×MoE L2 contention 축 공백. 산업 motivation: [Jetson Thor vLLM MoE gap forum](https://forums.developer.nvidia.com/t/jetson-agx-thor-vllm-26-02-moe-performance-significantly-below-reference-missing-fused-moe-config/364663).
- **예상 효과**: MoE decode 1.2-1.4×, energy -20~30%, Thor DevKit 확보 전제.

### Gale — GDN:Attn 3:1 Hybrid Constant-Memory + 256K KV 3-Tier DeepStack-Aware Eviction (Tier-2 독립 Top 1)
- **Date**: 2026-04-24 | **Session**: [링크](/research-wiki/2026-04/qwen3vl-deepstack-edge)
- **Tier**: Tier-2 독립 | **Target**: IEEE ESL 4p / ISLPED 6p
- **Review scores**: Nov **6.2** / Diff **6.5** / Imp **7.02** / Feas **7.0** → **avg 6.70**
- **Metaphor**: "Gale" = 강풍, 256K long context 지속 흐름 + dual wind direction (GDN + Attention).
- **Single mechanism + 1 ablation**: GDN 24 layer constant-memory fast path (KV paging 제외) + 256K KV 3-tier (GPU/DRAM/NVMe) DeepStack layer-aware eviction.
- **Qwen3.5 diff 활용**: Hybrid MoE + GDN 3:1 ratio (Qwen3-Next/3.5 신규).
- **GAP**: [Gated Delta Net ICLR'25 arXiv:2412.06464](https://arxiv.org/abs/2412.06464) algorithm / [TTKV arXiv:2604.19769](https://arxiv.org/abs/2604.19769) 2-tier / Kimi Linear + KDA 3:1 ratio overlap 있으나 **edge Thor LPDDR5X specific** 차별화.
- **예상 효과**: KV memory -75%, decode TPOP -37~50% (long context).
- **Tier-1 scale-up 불가 이유**: GDN 은 Qwen3-Next/3.5 한정, Qwen3-VL 공식 config 미확인 → narrow scope.

### Forge — Thinker/Talker Heterogeneous with Tensor Core + DLA + GDN L2 (Tier-2 독립 Top 2)
- **Date**: 2026-04-24 | **Session**: [링크](/research-wiki/2026-04/qwen3vl-deepstack-edge)
- **Tier**: Tier-2 독립 | **Target**: IEEE CAL 4p / DAC 6p
- **Review scores**: Nov **5.9** / Diff **7.0** / Imp **6.68** / Feas **6.8** → **avg 6.50**
- **Metaphor**: "Forge" = 대장간, Tensor Core + DLA + L2 이기종 재료 단조.
- **Scoop risk**: [Qwen3.5-Omni Tech Report arXiv:2604.15804](https://arxiv.org/abs/2604.15804) (본 세션 1일 전 공개) — 빠른 2026-05 CAL submit 권고.
- **Single mechanism**: Qwen3.5-Omni Thinker → Tensor Core, Talker Code2Wav causal ConvNet → Jetson DLA, GDN recurrent state → L2 resident.
- **GAP**: vLLM-Omni multi-GPU → single-Jetson Code2Wav DLA 실기 novelty.
- **예상 효과**: First-packet latency <200ms (Qwen3.5-Omni-Flash 235ms 대비 -15~30%).
- **Tier-1 scale-up 불가 이유**: single-Jetson Code2Wav kernel characterization letter.

## 미선정 (2026-04-24)

### Echo — Video Timestamp Hash for DeepStack Feature Dedup (DROP)
- **Date**: 2026-04-24 | **Session**: [링크](/research-wiki/2026-04/qwen3vl-deepstack-edge)
- **연구 GAP (의도)**: Qwen3-VL text-based video timestamp token 을 hash key 로 DeepStack 3-level visual feature dedup.
- **미선정 사유**:
  1. [VLCache arXiv:2512.12977](https://arxiv.org/abs/2512.12977) (2025-12-15) — 2% vision + 98% reuse + pixel hash, **72-78% direct scoop**.
  2. [CodecSight arXiv:2604.06036](https://arxiv.org/abs/2604.06036) (2026-04-07, 17일 전) — NVDEC motion vector + RoPE KVC refresh.
  3. [STC arXiv:2512.00891](https://arxiv.org/abs/2512.00891) — temporal cache 축.
  4. 이전 2026-04-23 edge-vlm-energy 세션 **Tidal (DROP)** 과 60-68% 재진입.
- **재방문 조건**: (1) Cross-request timestamp sharing 등 완전 새 축, (2) Tokenizer-deterministic hash formal property 분석, (3) DeepStack 3-level 중 특정 level 만 timestamp-conditional dedup.

---

## 미선정 로그 (최근)

### Tidal — Video VLM Temporal Token Dedup (2026-04-23 DROP by CodecSight scoop)
- **Date**: 2026-04-23 | **Session**: [링크](/research-wiki/2026-04/energy-efficient-edge-vlm)
- **연구 GAP (원래 의도)**: Video VLM (Qwen2.5-VL video / LLaVA-Video) 2-frame visual token cosine sim > 0.9 비율 42-65% 에서 NVDEC motion vector + MRoPE temporal KV dedup + LPDDR bank-aligned block layout.
- **Metaphor**: "Tidal" = 조수 (cyclical pattern).
- **Phase 1 기여자 (통합 원본)**: ai-opt Ripple + legacy-sys EchoVault + hw-pim Echo Chamber 3-way 통합.
- **미선정 사유**: [CodecSight arXiv:2604.06036](https://arxiv.org/abs/2604.06036) (2026-04-07, Yulin Zou 등) 이 NVDEC motion vector → token skip gate + RoPE position-correction KVC refresh 를 이미 제안. **68-72% direct scoop**. 제출일 16일 차이, precedence 확보 불가. Edge-only scope repositioning 시도에도 core mechanism 1:1 중첩.
- **재방문 조건**: (1) CodecSight 의 Jetson edge validation 부재 확인 + VLM-specific edge benchmark 실측 novelty, (2) MRoPE 3D 가 CodecSight 1D RoPE 와 formal 다른 별도 축으로 재설계 (VideoRoPE 결합 등), (3) LPDDR bank-aligned layout 을 Parquet M3 sub-mechanism 으로 흡수.

---

## Selected Ideas

### Parquet ★ — AnyRes Tile-Aware Adaptive Batching with Coupled GPU+DRAM DVFS and Per-Tile Precision for Edge VLM (2026-04-23 Tier-1 Top 1)
- **Date**: 2026-04-23 | **Mode**: 1 | **Session**: [링크](/research-wiki/2026-04/energy-efficient-edge-vlm) / [Summary](/research-wiki/2026-04/energy-efficient-edge-vlm)
- **Tier**: Top-tier | **Target venue**: ASPLOS 2027 / MLSys 2027
- **Experts**: ai-optimization + legacy-system + hw-pim (3:0 unanimous post-integration)
- **Review Scores** (Phase 2'): Nov **7.4** / Diff **7.5** / Imp **7.76** / Feas **7.5** → 평균 **7.54** (Accept strong).
- **Metaphor 근거**: "Parquet" = 마루판 타일. AnyRes 가변 tile 1-12 이 마루판처럼 정렬 + 서로 다른 precision/빈도 은유.
- **Core mechanisms (3, improve-first)**: (1) **Adaptive CUDA Graph bucket by tile-count** (vLLM v1 batch-size variant → tile-count variant 확장, bucket {1,3,6,12} + overflow path, fragmentation 55% → 75-85%), (2) **Coupled GPU+DRAM DVFS with tile-count signal** (tile 많음 → memory-bound → GPU freq 낮춤 + DRAM freq 유지, Jetson Orin `nvpmodel` 확장), (3) **Per-tile entropy-driven precision dispatching** (attention entropy 상위 FP8 / 하위 INT8, MBQ per-layer 를 per-tile 확장).
- **VLM-only 정당화**: LLM 은 tile 개념 없음; Vision-only 는 prefill/decode 이원화 없음.
- **초기 vs 최신 VLM diff**: LLaVA-1.5 fixed 576 → LLaVA-NeXT/Qwen2.5-VL/InternVL3 dynamic 1-12 tile (signal 존재) 직접 활용.
- **유사 연구 대응**: DynamoLLM [HPCA'25, [arXiv:2408.00741](https://arxiv.org/abs/2408.00741)] / GreenLLM [[arXiv:2508.16449](https://arxiv.org/abs/2508.16449)] / PolyThrottle [MLSys'24, [arXiv:2310.19991](https://arxiv.org/abs/2310.19991)] / MBQ [CVPR'25] / BiScale [[arXiv:2602.18755](https://arxiv.org/abs/2602.18755)] / throttLL'eM [[arXiv:2408.05235](https://arxiv.org/abs/2408.05235)] / SparseDVFS [[arXiv:2603.21908](https://arxiv.org/abs/2603.21908)] / Nova [[arXiv:2509.21301](https://arxiv.org/abs/2509.21301)] — 모두 tile-count unifying signal 축 없음. Peer-reviewed 70%.
- **Platform-Usage Analysis (R25)**: vLLM v1 (batch-size bucket 존재, tile-count 확장 필요 — 이미 사용 분류 i) / SGLang (RadixAttention, tile-count bucket 미구현 — 분류 iii) / llama.cpp (batch API 제한 — 분류 iii). Step C: vLLM v1 upstream PR 경로.
- **예상 개선**: Energy / request -28~42%, TTFT +4~5% trade, bucket fill rate 55% → 75-85%, DocVQA accuracy drop ≤ 0.5pp. Scope: mixed AnyRes workload only; tile-uniform (text-only) 에서는 <3% gain.
- **Tier-2 paper-pair (IEEE CAL 4p)**: M1 only (Adaptive bucket), Qwen2.5-VL-7B + Jetson Orin AGX 단일, bucket fill rate + latency 2 metric.
- **Phase 3 entry actions**: (a) vLLM v1 EncoderDisagg merge 상태 2026-04 snapshot 실측, (b) Qwen2.5-VL-7B on RTX 4060/4090 + Jetson Orin AGX tile-count histogram + bucket fill rate, (c) MBQ per-layer → per-tile ablation 선행 1주 PoC.

### Triptych — Edge-Specific Three-Stage Modality Pipeline with DLA-Aware Heterogeneous Compute and UMA Zero-Copy (2026-04-23 Tier-1 Top 2)
- **Date**: 2026-04-23 | **Mode**: 1 | **Session**: [링크](/research-wiki/2026-04/energy-efficient-edge-vlm)
- **Tier**: Top-tier | **Target venue**: ASPLOS 2027 / EuroSys 2027
- **Experts**: 3:0 unanimous post-replacement
- **Review Scores** (Phase 2' post Major Revision): Nov **6.5** / Diff **8.0** / Imp **8.0** / Feas **7.0** → 평균 **7.38** (Accept).
- **Metaphor 근거**: "Triptych" = 3 폭 패널 그림. Vision encoder (DLA) / Projector (Tensor core) / LLM (NVFP4) 각 패널 + 통합 serving 은유.
- **Core mechanisms (3, Phase 1' M3 replace)**: (1) **Modality-stage heterogeneous mapping** (vision → Jetson DLA INT8 / projector → Tensor core FP16 / LLM → NVFP4 Thor 또는 INT4 Orin), (2) **UMA zero-copy activation hand-off** (`ZeroCopyActivationRouter` vLLM-Jetson fork, memcpy 제거), (3) **DLA fine-grained preemptive scheduling** (Nova/HeteroInfer overlap 해소용 replace, NVDLA 2.0 sub-kernel preemption).
- **VLM-only 정당화**: LLM-only single-stage 로 DLA × Tensor core × NVFP4 이기종 부재; Vision-only 는 projector + LLM 없음.
- **초기 vs 최신 VLM diff**: LLaVA-1.5 CLIP-ViT-L + 단순 MLP projector → Qwen2.5-VL/InternVL3 dynamic encoder + pixel shuffle projector → 3-stage 이원화 이점 증대.
- **유사 연구 대응**: Nova [[arXiv:2509.21301](https://arxiv.org/abs/2509.21301)] 60% concurrent desktop GPU (DLA 없음) / Nanomind [[arXiv:2510.05109](https://arxiv.org/abs/2510.05109)] 60% concurrent tiny model / HeteroInfer [[arXiv:2501.14794](https://arxiv.org/abs/2501.14794)] 55% server-class / HydraInfer [[arXiv:2505.12658](https://arxiv.org/abs/2505.12658)] / llm.npu [[arXiv:2407.05858](https://arxiv.org/abs/2407.05858)] / LiteVLM [[arXiv:2506.07416](https://arxiv.org/abs/2506.07416)] / FastVLM [CVPR'25, [arXiv:2412.13303](https://arxiv.org/abs/2412.13303)] — 모두 DLA 미활용. NVIDIA DLA blog 조차 convolution only. Peer-reviewed 67%. **DLA axis 로 repositioning 후 unique**.
- **Platform-Usage Analysis (R25)**: vLLM (discrete GPU 전제, UMA 미지원 — 분류 iii) / TensorRT-LLM (Jetson 부분 지원, activation routing 수동 — 분류 i subset) / MLC-LLM (Jetson 지원, UMA 부분 — 분류 i subset). Step C: vLLM-Jetson fork upstream + MLC-LLM 보조.
- **예상 개선**: TTFT 1.4-1.7× faster (multi-image batch 3+), energy -25~35% (DLA 활용), MMMU accuracy ≥ 99.5%, Thor NVFP4 추가 +5-8%.
- **Tier-2 paper-pair (IEEE ESL 4p)**: M1 only (DLA INT8 vision encoder), Qwen2.5-VL-7B on Jetson Orin AGX, 에너지 + latency letter.
- **Phase 3 entry actions**: (a) Jetson Orin AGX DLA INT8 vision encoder (SigLIP/InternViT) profiling, (b) vLLM-Jetson fork `ZeroCopyActivationRouter` 4주 구현, (c) Jetson Thor DevKit 2026-06 gate or Orin fallback, (d) Nova + Nanomind + HeteroInfer 를 ablation baseline 에 포함 confirm.

### Cartographer — MRoPE Tri-Axial LUT + LPDDR Row-Aligned Layout (2026-04-23 Tier-2 독립 Top 1)
- **Date**: 2026-04-23 | **Mode**: 1 | **Session**: [링크](/research-wiki/2026-04/energy-efficient-edge-vlm)
- **Tier**: Tier-2 독립 (Track B) | **Target venue**: IEEE CAL 4p / DATE 6p
- **Experts**: legacy-system + hw-pim (primary)
- **Review Scores** (Phase 2'): Nov **6.8** / Diff **6.6** / Imp **7.03** / Feas **7.8** → 평균 **7.06** (Accept CAL).
- **Metaphor 근거**: "Cartographer" = 지도 제작자. MRoPE 3 axis (time × H × W) 좌표계를 LUT 로 지도화.
- **Core mechanism (1, Tier-2 rubric)**: MRoPE (time × H × W) 3-axial positional → precomputed LUT 치환 + LPDDR row-aligned layout.
- **VLM-only 정당화**: 1D RoPE (LLM) 과 다른 2D/3D RoPE, VLM (Qwen2-VL 이후) 만 존재.
- **유사 연구 대응**: T-MAC [EuroSys'25, [arXiv:2407.00088](https://arxiv.org/abs/2407.00088)] weight only / LUT Tensor Core [ISCA'25, [arXiv:2408.06003](https://arxiv.org/abs/2408.06003)] weight only / RotateKV [[arXiv:2501.16383](https://arxiv.org/abs/2501.16383)] rotation-based / SAIL [[arXiv:2509.25853](https://arxiv.org/abs/2509.25853)] SRAM-LUT GEMV / Revisiting MRoPE [[arXiv:2510.23095](https://arxiv.org/abs/2510.23095)] algorithmic. MRoPE LUT 축 공백. Peer-reviewed 60%.
- **예상 개선**: MRoPE kernel latency -40~60%, energy -15~25% (single kernel scope), MMMU 0 drop (bit-exact).
- **Tier-1 scale-up 불가 이유**: single-kernel characterization letter, MRoPE 한정.

### Sift — Entropy-Adaptive Pixel Shuffle for Tiny Edge VLM (2026-04-23 Tier-2 독립 Top 2)
- **Date**: 2026-04-23 | **Mode**: 1 | **Session**: [링크](/research-wiki/2026-04/energy-efficient-edge-vlm)
- **Tier**: Tier-2 독립 (Track B) | **Target venue**: ISLPED 2026 6p / IEEE ESL 4p
- **Review Scores**: Nov **6.5** / Diff **6.5** / Imp **6.90** / Feas **7.5** → 평균 **6.85** (Accept ISLPED).
- **Metaphor 근거**: "Sift" = 체로 거른다. Patch entropy 기준으로 visual token 체질.
- **Core mechanism (1)**: Patch attention entropy (pre-trained SigLIP attention weight proxy) 기반 pixel shuffle ratio {2, 4, 8} 동적 선택. Projector 이전 적용.
- **VLM-only 정당화**: Pixel shuffle 이 projector 이전 vision token reduction (VLM 고유).
- **초기 vs 최신 diff**: LLaVA-1.5 pixel shuffle 미존 → InternVL3 / SmolVLM fixed 2×2 → 본 idea adaptive.
- **유사 연구 대응**: InternVL-X [[arXiv:2503.21307](https://arxiv.org/abs/2503.21307)] RVTC / PyramidDrop [[arXiv:2410.17247](https://arxiv.org/abs/2410.17247)] layer-wise / VisionZip [[arXiv:2412.04467](https://arxiv.org/abs/2412.04467)] / FastV [[arXiv:2403.06764](https://arxiv.org/abs/2403.06764)] / SparseVLM [ICML'25](https://openreview.net/forum?id=80faIPZ67S) / SmolVLM [[arXiv:2504.05299](https://arxiv.org/abs/2504.05299)] — 모두 ratio 고정 또는 projector-이후. Peer-reviewed 50%.
- **예상 개선**: Visual token count -40-60%, energy -20-30%, accuracy drop ≤ 1pp.
- **Tier-1 scale-up 불가 이유**: tiny VLM + projector-이전 pixel shuffle narrow scope.

### Verge — Jetson Thor vs Orin AGX Cross-Arch VLM Energy Characterization (2026-04-23 Tier-2 독립 Top 3, Conditional)
- **Date**: 2026-04-23 | **Mode**: 1 | **Session**: [링크](/research-wiki/2026-04/energy-efficient-edge-vlm)
- **Tier**: Tier-2 독립 (Track B, Conditional) | **Target venue**: IEEE ESL 4p letter
- **Review Scores**: Nov **5.8** / Diff **6.2** / Imp **6.35** / Feas **6.8** → 평균 **6.29** (Conditional).
- **Metaphor 근거**: "Verge" = 경계/가장자리. Thor (최신 Blackwell NVFP4) 와 Orin AGX (이전 Ampere INT8) 의 경계 측정.
- **Core mechanism (1)**: Qwen2.5-VL-7B / InternVL3-8B / LLaVA-OneVision-7B 3 모델을 Jetson Thor vs Orin AGX 에서 VLM-specific stage (vision / projector / LLM prefill / decode) 별 J/token + W 측정.
- **유사 연구 대응**: ELANA [[arXiv:2512.09946](https://arxiv.org/abs/2512.09946)] Thor+Orin Nano 일반 LLM (VLM-specific stage breakdown 없음) / Jetson Orin LLM profiling [[arXiv:2506.09554](https://arxiv.org/abs/2506.09554)] Orin only / TokenPowerBench [[arXiv:2512.03024](https://arxiv.org/abs/2512.03024)] LLM token power / Watt Counts [[arXiv:2604.09048](https://arxiv.org/abs/2604.09048)] / Blackwell Microbench [[arXiv:2512.02189](https://arxiv.org/abs/2512.02189)]. VLM-specific cross-arch 공백. Peer-reviewed 40% (Conditional).
- **Conditional 조건**: Jetson Thor DevKit 2026-06 확보 필수. 미확보 시 Orin AGX + RTX 4090 fallback.
- **Tier-1 scale-up 불가 이유**: characterization-only letter.

### HRTS+ ★ — HBM Row-Tile Streaming for Long-Context Video VLM (v3 Tier-1 Top 1)
- **Date**: 2026-04-23 (v3) | **Mode**: 1 | **Session**: 링크 / Summary
- **Tier**: Top-tier | **Target venue**: ASPLOS 2026 / MICRO 2026
- **Experts**: ai-optimization-expert + legacy-system-expert (2:0 unanimous)
- **Review Scores** (v3 Phase 2'): Nov **8.0** / Diff **7.8** (Mosaic concurrent 반영 -0.3) / Imp **8.0** / Feas **7.3**. 평균 **7.85** (Accept).
- **Core mechanisms (3, improve-first)**: (1) Row-aligned KV tile (HBM3 row 8KB ↔ vLLM page 8KB 정합, tile {128/256/512} sweep), (2) Bi-exponential recency × salience window pin, (3) Async tri-tier streaming (HBM-hot / DRAM-pinned / NVMe-cold).
- **Tiering**: physical 3-tier (HBM / DRAM / NVMe) + software window 2-bucket. R1/R1b 규율 준수.
- **유사 연구 대응 (v3 보강)**: VL-Cache [ICLR'25], HERMES [arXiv:2601.14724](https://arxiv.org/abs/2601.14724), DiffKV [arXiv:2412.03131](https://arxiv.org/abs/2412.03131), VideoLLM-online [CVPR'24], PagedAttention [SOSP'23], **Mosaic ([arXiv:2604.10060](https://arxiv.org/abs/2604.10060)) v3 신규 concurrent 55-65%** — Mosaic = content-axis cross-modal clustering, HRTS+ = HBM row-buffer physical axis, orthogonal + stacking 가능. Baseline 10편, peer-reviewed 70%.
- **OpenReview feedback 활용**: VL-Cache [HMrcv7Q4Ub](https://openreview.net/forum?id=HMrcv7Q4Ub) reviewer 의 streaming/multi-turn 미커버 지적을 M2 bi-exp window pin 으로 대응.
- **예상 개선**: Decode throughput +25~35% (long-context 64K+) / TPOT -20~30% (128K) / HBM row-hit 62% → 82-88% / Memory footprint -30~40% / HRTS+Mosaic stacking +5~8% 추가.
- **핵심 가설**: HBM3 row buffer 8KB 의 물리 구조를 KV page layout 에 노출하면 row-hit 이 20%p 이상 개선되며, 이는 Mosaic 의 content-axis dedup 와 orthogonal 한 추가 이득 제공.
- **차별점**: VL-Cache sparsity-only / HERMES page-level non-DRAM / Mosaic content-axis only. HRTS+ 는 **HBM physical axis + bi-exp adaptive window + tri-tier async** 의 3축 novel 교차.
- **Phase 3 entry actions**: (a) Nsight Compute row-boundary probing, (b) VideoMME/MVBench 30-min subset 확보, (c) Mosaic stacking ablation protocol 설계.
- **Tier-2 paper-pair (HRTS Tier-2)**: Row-aligned tile letter (M1 only), LLaVA-Video-7B + VideoMME long subset 단일, HBM row-hit +15-25%p. IEEE CAL 4p / DATE 6p. Precedence 확보.

### ContextMIG+ — Reuse Graph × MIG Dual-Issue × Phase Coalesce for Multi-tenant VLM (v3 Tier-1 Top 2)
- **Date**: 2026-04-23 (v3) | **Mode**: 1 | **Session**: 링크
- **Tier**: Top-tier | **Target venue**: ASPLOS 2026 / MLSys 2026
- **Experts**: ai-optimization-expert + legacy-system-expert + algorithm-expert (3:0 unanimous)
- **Review Scores** (v3 Phase 2'): Nov **8.3** / Diff **8.0** (Predictable LLM Serving baseline 추가) / Imp **7.8** / Feas **6.8**. 평균 **7.73** (Accept).
- **Core mechanisms (3, replace-all from v1 Phase 1)**: (1) CLIP-L LSH reuse graph classifier (16-bit SimHash, sliding window 256 req, <0.6ms), (2) Tier-aware MIG dual-issue partition (3-SW tier × 2-phys MIG slice, MIG-A prefill-visual + MIG-B decode-LLM, Green Context μs reconfig), (3) Phase-aligned coalescing (new, critical gap 대응).
- **Tiering**: 3-SW tier × 2-phys MIG, R1/R1b 규율 준수. Mechanism 3.
- **유사 연구 대응 (v3 보강)**: Mosaic [arXiv:2604.10060](https://arxiv.org/abs/2604.10060) KV-centric clustering / **Predictable LLM Serving ([arXiv:2508.20274](https://arxiv.org/abs/2508.20274)) v3 신규 concurrent 55-60%** cluster-level MIG / Semantic Scheduling [arXiv:2506.12204](https://arxiv.org/abs/2506.12204) software only / LithOS [SOSP'25] fine-grained SM / vLLM [SOSP'23] / SGLang [NeurIPS'24] / HERMES [ISCA'24] / LithOS / Llumnix [OSDI'24] / VL-Cache [ICLR'25] / DynamoLLM [HPCA'25]. v3 신규 related work: Prefill-as-a-Service [arXiv:2604.15039](https://arxiv.org/abs/2604.15039), IceCache [arXiv:2604.10539](https://arxiv.org/abs/2604.10539). Baseline 10편, peer-reviewed 60%.
- **예상 개선**: Multi-tenant throughput +22~32% / p95 TTFT -18~28% (visual context overlap) / SM util +12~20%p / vs Mosaic stacking +4~7% / vs Predictable LLM Serving +8~12%.
- **핵심 가설**: Content-axis reuse graph × intra-GPU MIG dual-issue × phase coalesce 3축 결합이 multi-tenant VLM 에서 p99 TTFT 개선.
- **차별점**: Predictable LLM Serving = cluster-level MIG (inter-GPU), ContextMIG+ = **intra-GPU MIG dual-issue + content reuse graph + phase coalesce**.
- **Tier-2 paper-pair (ContextMIG Tier-2)**: CLIP-L LSH classifier standalone (M1 only), Pro 6000 2-tenant, F1 ≥ 0.82 + hash latency ≤ 1.5ms. IEEE ESL 4p / CAL 4p.

### PhaseGraph-VLA+ — Trajectory-Phase Conditioned CUDA Graph Dispatcher with SSE (v3 Tier-1 Top 3, v1 A1 revival)
- **Date**: 2026-04-23 (v3) | **Mode**: 1 | **Session**: 링크
- **Tier**: Top-tier | **Target venue**: MLSys 2026 / CoRL 2026
- **Experts**: ai-optimization-expert + legacy-system-expert + algorithm-expert (Conditional PH FP rate)
- **Review Scores** (v3 Phase 2'): Nov **6.9** (+0.1 FlashVLA 차별화) / Diff **7.5** / Imp **6.8** / Feas **7.5**. 평균 **7.18** (Conditional Accept).
- **Core mechanisms (3, improve-first + P2 absorb)**: (1) SSE phase predictor (L_mid hidden state L2 drift + Page-Hinkley 2-threshold + hysteresis, VLA add-feature: gripper Δ + trajectory curvature + DINOv2 object distance), (2) Phase-specific CUDA Graph dispatcher (Approach/Manipulate/Retract 3-graph, phase × batch 2D), (3) Phase-specific SM partition (optional stacking with Nova).
- **유사 연구 대응 (v3 보강)**: VLA-Cache [NeurIPS'25], AC²-VLA [arXiv:2601.19634](https://arxiv.org/abs/2601.19634), KV-Efficient VLA [arXiv:2509.21354](https://arxiv.org/abs/2509.21354), ADP-VLA [arXiv:2509.22093](https://arxiv.org/abs/2509.22093), **FlashVLA ([arXiv:2505.21200](https://arxiv.org/abs/2505.21200)) v3 신규 scoop 접경 68-72%** (token-level reuse, PhaseGraph+ 는 graph-level switch — 축 다름), Nova [arXiv:2509.21301](https://arxiv.org/abs/2509.21301), DuetServe [arXiv:2511.04791](https://arxiv.org/abs/2511.04791), SpecPrune-VLA [arXiv:2509.05614](https://arxiv.org/abs/2509.05614), Running-VLAs [arXiv:2510.26742](https://arxiv.org/abs/2510.26742). Baseline 9편, peer-reviewed 67%.
- **예상 개선**: LIBERO median latency 165 → 128ms (-22%) / SimplerEnv 6.1 → 7.8 Hz (+28%) / Jetson Orin 420 → 330ms / Stacked + Nova 115ms (-30%).
- **차별점**: FlashVLA = token-level reuse, PhaseGraph-VLA+ = **graph-level execution switch** (CUDA Graph variant 교체).
- **Tier-2 paper-pair (PhaseGraph-VLA Tier-2)**: SSE predictor standalone (M1 only), OpenVLA-7B + LIBERO-Spatial 단일, PH FP rate ≤ 5% + decision <100μs. IEEE CAL 4p / DATE 6p.

### B1 GCReconfProfile — Green Context μs-level Reconfig Characterization (v3 Tier-2 독립 Top 1)
- **Date**: 2026-04-23 (v3) | **Mode**: 1 | **Session**: 링크
- **Tier**: Tier-2 독립 (Track B) | **Target venue**: ISLPED 2026 6p / DATE 2026 6p
- **Experts**: legacy-system-expert (primary, HW profiling 축)
- **Review Scores** (v3 Phase 2'): Nov **7.2** / Diff **7.6** / Imp **7.5** / Feas **7.7**. 평균 **7.50** (Accept).
- **Core mechanism (1, Tier-2 rubric)**: Green Context reconfig latency instrumentation harness — vLLM v0.8+ fork 에 CUPTI PM Sampling + nanosecond `cuEventElapsedTime` 삽입 → `cuDevSmResourceSplit` invocation 의 μs-level latency 를 mixed VLM workload (LLaVA-OV OCR / Qwen2-VL chat / InternVL2 grounding) × SM-count {8/16/32/64/84} × prev-SM 4 factorial.
- **유사 연구 대응**: MIGER [ICPP'24] MIG 2.3% reconfig (ms-level), LithOS [SOSP'25] fine-grained SM API, Power Management ASPLOS 2024, Execution-Idle [arXiv:2604.04745](https://arxiv.org/abs/2604.04745) (v3 motivation citation). Baseline 3편, peer-reviewed 100%.
- **예상 결과**: Reconfig latency p50 **18-45 μs** (vs MIG 2100ms, 4 orders 격차) / p99 **80-140 μs** / energy spike **0.3-0.8 mJ**. **Blackwell vs Hopper cross-arch characterization primary contribution**.
- **Tier-1 scale-up 불가 이유**: 단일 vendor API characterization letter, mechanism 1 개, cross-vendor generality 부재.

### B2 TokenEvictEnergy — Visual-Token Eviction 의 HBM Refresh/DRAM Energy Negative Result (v3 Tier-2 독립 Top 2)
- **Date**: 2026-04-23 (v3) | **Mode**: 1 | **Session**: 링크
- **Tier**: Tier-2 독립 (Track B) | **Target venue**: IEEE ESL 2026 4p / ISLPED 2026 6p
- **Experts**: legacy-system-expert (primary, energy 축)
- **Review Scores** (v3 Phase 2'): Nov **7.3** / Diff **7.4** / Imp **7.3** / Feas **7.5**. 평균 **7.35** (Conditional Accept, error bar).
- **Core mechanism (1)**: Per-policy energy-counter harness — NVML 5ms + Intel RAPL DRAM-package 이중 counter → (HBM dynamic / SM static / DRAM PKG / refresh-implied) 4-component decomposition.
- **유사 연구 대응**: VL-Cache [ICLR'25] / SparseVLM [ICML'25] 모두 energy 축 누락, Power Management [ASPLOS'24], TokenPowerBench [arXiv:2512.03024](https://arxiv.org/abs/2512.03024). Baseline 3편, peer-reviewed 100%.
- **Hidden insight**: aggressive eviction 이 HBM row-buffer locality 상실 → DRAM refresh 증가 → **DRAM PKG energy 오히려 +2~+6%** (negative result). "eviction ≠ always green" 공교육.
- **예상 결과**: Total energy -8~15% (VL-Cache) / HBM dynamic -18~25% / **DRAM PKG +2~+6% (negative)** / p99 power -22~30%.
- **Tier-1 scale-up 불가 이유**: Power-constrained narrow engineering, negative result 중심, confounder 통제 불가.

### B3 ActHeadFuse — OpenVLA-OFT Action-Head Fused Kernel for Sub-ms Decode Step (v3 Tier-2 독립 Top 3)
- **Date**: 2026-04-23 (v3) | **Mode**: 1 | **Session**: 링크
- **Tier**: Tier-2 독립 (Track B) | **Target venue**: IEEE CAL 2026 4p / DAC 2026 6p
- **Experts**: legacy-system-expert (primary, kernel 축)
- **Review Scores** (v3 Phase 2'): Nov **7.0** / Diff **7.3** / Imp **7.2** / Feas **7.5**. 평균 **7.20** (Accept).
- **Core mechanism (1)**: Action-head 3-op fusion CUDA kernel — SiLU + Linear(4096→448) + Bucketize single persistent kernel + Block-tile (hidden=4096 / 32-warp) + TMA (Hopper/Blackwell) activation prefetch + warp-level softmax-free bucketize.
- **유사 연구 대응**: OpenVLA [CoRL'24], OpenVLA-OFT [arXiv:2502.19645](https://arxiv.org/abs/2502.19645) (parallel-sample 26×, kernel-level 아님), FAST [arXiv:2501.09747](https://arxiv.org/abs/2501.09747), VLA-Cache [NeurIPS'25]. Baseline 3편, peer-reviewed 67%.
- **예상 결과**: Action-head kernel time 14.2 μs → **2.3 μs** (Pro 6000 bs=1) / Total decode step 1.82 → **0.94 ms (1-kHz real-time control 가능)** / Throughput 540 → **1050 steps/s** (bs=32) / LIBERO MSE **0 (bit-exact)**.
- **Tier-1 scale-up 불가 이유**: Narrow kernel engineering letter, model family lock-in (OpenVLA Llama-2-7B), serving stack 전체 impact ~5-8%.

### I2' TernVLM-KV-LUT ★ — Xbar-aligned Ternary KV + Rank-4 SF + FA3-Dual LUT Attention (VLM/VLA)
- **Date**: 2026-04-23 | **Mode**: 1 | **Session**: 링크 / Summary
- **Tier**: Top-tier (**primary**) | **Target venue**: MLSys 2026 (2026-10) / NeurIPS 2026
- **Experts**: algorithm-expert + ai-optimization-expert + hw-pim-accelerator-expert (3 공통 최고 novel)
- **Review Scores** (Phase 2' 재평가): Nov **8.0**/10, Impact **8.6**/10, Algo peer **7.5**, AI-opt peer **6.5→7.5** (FA3-dual path 해소 후), HW peer **7.0**. 평균 **7.95** (Strong Accept, oral 후보).
- **Core mechanisms (3, improved)**: (1) K pre-RoPE ternary + V INT4 tile-aligned (bit-exact prefix-cache 보존) + logit scale β 학습, (2) rank-4 modality-split SF (vision r=4, text r=2) + saliency-weighted OPTIC + Lipschitz bound ||S - S_merged||_F ≤ σ_{r+1}·√(G_c·G_t), (3) **FA3-dual path attention kernel** — hot path WGMMA (recent 2K tokens), cold path Q INT8 × K ternary sub-tile=4 LUT 81×256 entry (prefill-heavy).
- **Tiering**: modality 3-tier (vision/text/projector, architecture-natural 규율 준수). Physical storage 3-tier (HBM-tier / CPU / NVMe, unchanged). Mechanism 3.
- **유사 연구 대응**: KIVI [ICML'24] / KVQuant [NeurIPS'24] / GEAR [NeurIPS'24] / SKVQ [COLM'24] / RotateKV [IJCAI'25] / VL-Cache [ICLR'25] / Oaken [ISCA'25] / ResQ [ICML'25] / AKVQ-VL ([arXiv:2501.15021](https://arxiv.org/abs/2501.15021)) / KVTQ ([OpenReview eZAlb8fX5y](https://openreview.net/pdf?id=eZAlb8fX5y)) / T-MAC EuroSys'25 / LUT Tensor Core ISCA'25 — **peer-reviewed 75%**. 모든 기존 연구 scalar/rotation/error-residual SF 만, rank-r SF matrix × ternary × LUT-attention × FA3 compat × VLM modality-split 의 5축 결합 공백.
- **OpenReview feedback 활용**: VL-Cache [HMrcv7Q4Ub](https://openreview.net/forum?id=HMrcv7Q4Ub) reviewer 의 "modality-aware cache budget extension" 을 modality-split rank 에 반영. SpinQuant [ogO6DGE6FZ](https://openreview.net/forum?id=ogO6DGE6FZ) reviewer "learned rotation × rank-r" 를 saliency-weighted OPTIC 근거.
- **예상 개선**: LLaVA-OneVision-7B W8A8 + KV(K=1.58b/V=INT4) 에서 MMMU drop ≤ 0.5% vs FP16. Decode latency 1.5-2.0× speedup vs VL-Cache. KV memory VL-Cache 10% × ternary 8× × V INT4 4× average = 추가 4-6× 절감 (긴 video/multi-image).
- **핵심 가설**: vision-token KV 의 near-normal 분포 + K pre-RoPE ternary 의 좁은 dynamic range → FA3-dual path (hot WGMMA + cold LUT 81×256 entry) 로 FA3 throughput 유지하며 extreme quantization 가능.
- **차별점**: KIVI/KVQuant/GEAR scalar SF / RotateKV rotation / VL-Cache sparsity-only / Oaken HW-only / KVTQ ASIC-only. I2' 는 **rank-r SF matrix + GPU-commodity + VLM + FA3-dual path** 의 5축 novel 교차.
- **Phase 3 entry actions**: (a) T-MAC LUT kernel GPU fork, (b) vLLM page_size=8KB patch, (c) Hessian-aware rank test (I1 공유), (d) LongVideoBench 32K context profile.
- **Tier-2 variant (I2-Tier2, paper-pair)**: sub-tile=4 LUT attention only (rank-r SF / OPTIC 제거), LLaVA-OV-7B + LongVideoBench only, LUT hit rate 70-85% measurement letter. MLSys workshop 또는 ISLPED 2026 short 6p. **Tier-2 독립 Top 3 에 선정** — I2' primary submission 과 precedence 확보.

### I4' PRISMKV-PIM-DequantLUT — DRAM-PIM Bank SF-LUT Dequant Replacing MAC
- **Date**: 2026-04-23 | **Mode**: 1 | **Session**: 링크
- **Tier**: Top-tier (HW track) | **Target venue**: HPCA 2027 (2026-08) / ASPLOS 2027
- **Experts**: hw-pim-accelerator-expert (primary)
- **Review Scores** (Phase 2'): Nov **7.5** (+0.3 AQPIM 차별화), Impact **7.4**, HW peer **7.5**. 평균 **7.35** (Accept, HW secondary).
- **Core mechanisms (3, simplified)**: (1) PIM bank internal SRAM (SF_idx, quant_value) → dequant scalar LUT (Q-independent static, 64-128 entries), (2) single SF-LUT + 1-bit modality tag bit in index (dual-channel 축소), (3) chunk retention (optional ablation axis).
- **Tiering**: physical channel single (≤ 3-4 규율). HW tier 3 (DRAM bank + SRAM LUT + host driver).
- **유사 연구 대응**: Oaken [ISCA'25, [arXiv:2503.18599](https://arxiv.org/abs/2503.18599)] / P3-LLM ([arXiv:2511.06838](https://arxiv.org/abs/2511.06838)) / AttAcc [ASPLOS'24] / NeuPIMs [ASPLOS'24] / AQPIM [HPCA'26, [arXiv:2604.18137](https://arxiv.org/abs/2604.18137)] / SAIL ([arXiv:2509.25853](https://arxiv.org/pdf/2509.25853)) / LUT Tensor Core [ISCA'25, [arXiv:2408.06003](https://arxiv.org/abs/2408.06003)] — peer-reviewed 100%. **AQPIM 차별화 핵심**: AQPIM LUT = inner-product pre (Q-dependent runtime), I4' LUT = dequant scalar pre (static, Q-independent). Replace 대상이 다름.
- **예상 개선**: P3-LLM 대비 KV dequant energy 80-90% 감소, end-to-end TOPS/W 2-3× 개선 (P3-LLM 4.9× × dequant-LUT stacking).
- **핵심 가설**: PIM bank 내 MAC-based dequant 가 TOPS/W 병목 → static dequant LUT 로 치환하면 AQPIM 의 inner-product LUT 와 orthogonal 한 추가 이득.
- **차별점**: Oaken/P3-LLM MAC dequant 유지, AQPIM inner-product pre (Q-dependent), SAIL SRAM-LUT weight mpGEMM. I4' 는 **DRAM-PIM bank + KV cache + dequant scalar pre (static)** novel 교차.
- **Tier-2 variant (I4-Tier2)**: SRAM-PIM SF-LUT (DRAM 대신 SRAM), single device, BitVLA-2B subset, ICCAD/ISLPED 2027 8p.

### I1' TernVLM-RankSF — Hessian-aware Modality-Split Rank-r SF for Ternary VLM
- **Date**: 2026-04-23 | **Mode**: 1 | **Session**: 링크
- **Tier**: Top-tier | **Target venue**: ICLR 2027 (2026-09) / NeurIPS 2026
- **Experts**: algorithm-expert + ai-optimization-expert + hw-pim-accelerator-expert (3 공통)
- **Review Scores** (Phase 2'): Nov **6.8** (+0.6 Hessian+Lipschitz), Impact **7.4**, Algo peer **7.0**. 평균 **7.05** (Conditional → Accept after 차별화 보강).
- **Core mechanisms (3, improved)**: (1) Hessian eigenvalue decay-driven adaptive rank (per-layer cumulative 90% × modality base r: vision r=8 / LLM r=4 / head r=1), (2) Xbar-aligned tile decomposition (channel-group × token-group, tile=128), (3) inference merge + Lipschitz-bounded error ||S_rank-r - S_merged||_F ≤ σ_{r+1}·√(G_c·G_t). TP sharding: hidden-axis (rank-r linearity, 재학습 불필요).
- **Tiering**: modality 3-tier (vision/LLM/head, architecture-natural). tile 단일.
- **유사 연구 대응**: BitNet b1.58 ([arXiv:2402.17764](https://arxiv.org/abs/2402.17764)) / BitVLA ([arXiv:2506.07530](https://arxiv.org/abs/2506.07530)) / Bi-VLM ([arXiv:2509.18763](https://arxiv.org/abs/2509.18763)) / MBQ [CVPR'25] / TernaryLLM ([arXiv:2406.07177](https://arxiv.org/abs/2406.07177)) / MDBF ([arXiv:2512.24545](https://arxiv.org/abs/2512.24545)) / LoRDS ([arXiv:2601.22716](https://arxiv.org/abs/2601.22716)) / QSVD ([arXiv:2510.16292](https://arxiv.org/abs/2510.16292)) / MASQuant ([arXiv:2603.04800](https://arxiv.org/html/2603.04800)) / ARB-LLM ([arXiv:2410.03129](https://arxiv.org/abs/2410.03129)) — peer-reviewed 60%. **MDBF vs I1' formal 차별화**: MDBF weight-space factorization (W ≈ sign(W)⊙uv^T), I1' scale-space factorization (W ≈ ternary(W)⊙S1·S2) — Hessian sensitivity 축 다름.
- **예상 개선**: LLaVA-OneVision-7B W1.58A8 MMMU drop ≤ 2.5% vs FP16 (BitVLA 대비 gap 30% 축소). Training convergence 30% 감소. Inference TOPS/W +5~10%. Memory overhead < 1%.
- **핵심 가설**: BN-Free σ expansion 효과가 VLM QAT 에서도 재현 가능, modality-split 이 vision wide dynamic range 복원 핵심.
- **차별점**: ARB-LLM rank-1 (row×col), MDBF weight factorization, LoRDS LLM-only scale S=BA. I1' 는 **rank-r SF matrix × Hessian-adaptive × VLM modality-split × tile-aligned × inference merge** 5축 novel.
- **Tier-2 variant (I1-Tier2)**: rank-r SF only (modality-split 제거), LLaVA-OV-7B + H100 + MMMU only, DATE/ISLPED short 6p.

### T2' PRISM-Tile (Tier-2 독립) — HBM Row × Page × Tile × Flat-SF 4-tuple Alignment Measurement
- **Date**: 2026-04-23 | **Mode**: 1 | **Session**: 링크
- **Tier**: Tier-2 (독립) | **Target venue**: DATE 2027 / ISLPED 2026 late / IEEE ESL
- **Experts**: ai-optimization-expert (primary)
- **Review Scores** (Tier-2 rubric): Nov 4.7 / Impact 7.8 / AI-opt peer 5.5 / HW peer 5.5. 평균 **5.8 / 7.8 (Tier-2 rubric Accept)**.
- **Core mechanism (1, Tier-2)**: HBM3 row (8KB) × vLLM PagedAttention page × BitBLAS tile × flat SF 의 4-tuple 8KB align strategy.
- **예상 개선**: HBM row-hit 62% → 82-88%. Decode latency 5-15% 개선.
- **차별점**: first-to-report measurement letter. PRISM xbar-alignment 원리의 GPU-HBM stack 포팅.

### T1' OPTIC-SF-Lite (Tier-2 독립) — Mobile VLM SF Clustering + Jetson Edge Measurement
- **Date**: 2026-04-23 | **Mode**: 1 | **Session**: 링크
- **Tier**: Tier-2 (독립) | **Target venue**: DATE / ICCAD short / ISLPED
- **Experts**: algorithm-expert (primary)
- **Review Scores** (Tier-2 rubric): Nov 4.2 / Impact 7.7 / peers 5.0-6.0. 평균 **5.3 / 7.7 (Tier-2 rubric Accept)**.
- **Core mechanism (1, Tier-2)**: AWQ SF K-means clustering (K=8/16/32) + top-1% outlier quantile 보존.
- **예상 개선**: MobileVLM-1.4B MMMU -1.5% 이내, SF storage 50-90% 감소.
- **차별점**: first-to-report mobile-VLM SF distribution characterization.

### I2-Tier2 (paper-pair with I2') — Ternary KV LUT Attention Hit Rate Measurement
- **Date**: 2026-04-23 | **Mode**: 1 | **Session**: 링크
- **Tier**: Tier-2 (companion) | **Target venue**: MLSys workshop / ISLPED short 6p
- **Core mechanism (1)**: Sub-tile=4 LUT attention (rank-r SF / OPTIC 제거), Q INT8 × K ternary 81×256 entry LUT.
- **예상 개선**: LUT hit rate 70-85% on LongVideoBench, decode latency 1.2-1.5× speedup vs FA3.
- **Paper pair 전략**: I2-Tier2 6월 submission (ISLPED short) 으로 precedence 확보, I2' 10월 MLSys primary.

---

### (v2) HRTS: HBM Row-Tile Streaming for Long-Context Video VLM
- **Date**: 2026-04-22 (v2) | **Mode**: 1 | **Session**: 링크
- **Tier**: Top-tier | **Target venue**: ASPLOS 2026 / MICRO 2026
- **Experts**: legacy-system-expert (primary)
- **Review Scores** (Phase 2' 최종): Nov **8.2**/10, Diff **8.1**/10, Imp **8.0**/10, Feas **7.3**/10, 평균 **7.90** (본 세션 1위, Accept)
- **Core mechanisms (3)**: (1) Row-aligned KV tile (analytical row-hit model + Nsight row-probing), (2) bi-exponential recency×salience window pin, (3) HBM/DRAM/NVMe async tri-tier streaming.
- **Tier**: 3-physical (HBM/DRAM/NVMe) + 3-software (row-pinned/row-evictable/pinned-host). 규칙 ≤3-4 준수.
- **유사 연구 대응**: HERMES [[arXiv:2601.14724](https://arxiv.org/abs/2601.14724)] (page-level only), DiffKV [[arXiv:2412.03131](https://arxiv.org/abs/2412.03131)] (head-diff only), VL-Cache [ICLR 2025 Poster, [arXiv:2410.23317](https://arxiv.org/abs/2410.23317)] (layer-adaptive budget only), StreamingLLM [ICLR 2024, [arXiv:2309.17453](https://arxiv.org/abs/2309.17453)], VideoLLM-online [CVPR 2024, [arXiv:2406.11816](https://arxiv.org/abs/2406.11816)], H2O [NeurIPS 2023], InfiniGen [OSDI 2024], Semantic Scheduling [[arXiv:2506.12204](https://arxiv.org/abs/2506.12204)], FlexGen [ICML 2023] — **peer-reviewed 6/9 = 67%**. Scoop risk Low-Medium.
- **OpenReview feedback 활용**: VL-Cache (ICLR 2025) reviewer 지적 "streaming/multi-turn 미커버" 를 window pin 으로 정면 대응.
- **예상 개선**: Decode throughput +25~35% (64K context), TPOT -20~30% (128K context), memory footprint -30~40% (NVMe tier). VideoMME accuracy ≤0.5pp drop.
- **핵심 가설**: HBM row buffer 물리 구조 노출 + bi-exp window pin + tri-tier 로 long-context video VLM 에서 HBM row-hit 62%→85% + NVMe backing.
- **차별점**: Mosaic/KVShare/HERMES/DiffKV 모두 token/page-level 축. HRTS 는 **HBM row-buffer physical alignment** 신규 축.
- **Phase 3 entry actions**: (a) Nsight Compute row-boundary probing, (b) 30-min 영상 subset, (c) NVMe 3-tier baseline.
- **Tier-2 variant** (IEEE CAL / DATE): Row-aligned KV tile only (M1), 1-GPU 64K context, HBM row-hit +15~25%p, attention kernel -8~12%. Precedence 확보용 self-contained sub-contribution. Conditional Accept.

### (v2) ContextMIG: CLIP-L Reuse Graph × MIG Dual-Issue × Phase Coalesce (Multi-tenant VLM merged from TriadSM+RGSM)
- **Date**: 2026-04-22 (v2) | **Mode**: 1 | **Session**: 링크
- **Tier**: Top-tier | **Target venue**: ASPLOS 2026 / MLSys 2026
- **Experts**: ai-optimization-expert + legacy-system-expert (merged, Axis A Phase 2 merger)
- **Review Scores** (Phase 2' 최종): Nov **8.3**/10, Diff **8.2**/10, Imp **7.7**/10, Feas **6.8**/10, 평균 **7.75** (본 세션 2위, Accept)
- **Mechanism replace-all justification**: Phase 1 TriadSM (SM allocation heuristic) + RGSM (reuse graph post-hoc) 모두 Mosaic [arXiv:2604.10060] scoop 70% 위험. Phase 1' 에서 3 mechanism 모두 replace 로 재설계 — critical gap 방어 근거 명시.
- **Core mechanisms (3, replace-all from Phase 1)**: (1) CLIP-L LSH reuse graph classifier (16-bit SimHash, sliding window 256 req, <0.6ms), (2) tier-aware MIG dual-issue partition (3-SW tier hot/warm/cold × 2-phys MIG slice MIG-A prefill-visual + MIG-B decode-LLM), (3) phase-aligned coalescing.
- **Tier**: 3-software (hot/warm/cold) + 2-physical MIG slice. 규칙 ≤3-4 준수.
- **유사 연구 대응**: Mosaic [arXiv:2604.10060] (cross-modal clustering KV-centric only, **정면 대응 replace-all 로 차별화**), Semantic Scheduling [arXiv:2506.12204] (software-only), Nova [arXiv:2509.21301] (phase-axis only), DuetServe [arXiv:2511.04791], Bullet [arXiv:2504.19516], HERMES [ISCA 2024], LithOS [EuroSys 2025], Llumnix [OSDI 2024], DynamoLLM [HPCA 2025] — **peer-reviewed 5/9 = 56%**. Scoop risk Low.
- **OpenReview feedback 활용**: VL-Cache (ICLR 2025) reviewer 지적 "deployment-scale baseline 부재 (vLLM integration 없음)" 를 vLLM+MIG 통합 경로로 정면 대응.
- **예상 개선**: Multi-tenant throughput +22~32% (2-tenant 7B co-location), p95 TTFT -18~28%, SM utilization +12~20%p.
- **적용 범위**: Multi-tenant VLM only (≥2 tenants, visual context overlap 존재 시).
- **차별점 vs v1 L1 ContextSM-Tri**: v1 의 content-axis tri-knob (α_SM, α_BW, α_KV) 은 Semantic Scheduling [arXiv:2506.12204] 과 concurrent 45-55% 로 novelty 축소됐음. v2 는 **3 mechanism replace-all** (reuse graph × MIG × coalesce) 로 Mosaic 정면 대응하며 tier/mechanism budget 모두 준수.
- **전문가 합의**: ai-opt Y (strong) + legacy-sys Y + algorithm Y (3:0 unanimous).
- **Phase 3 entry actions**: (a) CLIP-L LSH hashing infra, (b) vLLM fork + MIG dual-issue scheduler hook, (c) 2-tenant (Qwen2.5-VL-7B + LLaVA-OneVision-7B) trace.
- **Tier-2 variant** (IEEE ESL / IEEE CAL): CLIP-L LSH reuse graph classifier only (M1), 2-tenant trace replay, F1 ≥ 0.82, hash collision ≤ 3%, hash latency ≤ 1.5ms/req. Accept but paper pair 사용 안 함 (Top 3 slot NACK-Gossip Tier-2 에 양보).

### (v2) NACK-Gossip Tier-2: Pull-based NVLink Peer-Fetch Latency Profiling for VLA Inference
- **Date**: 2026-04-22 (v2) | **Mode**: 1 | **Session**: 링크
- **Tier**: **Tier-2** | **Target venue**: IEEE ESL (4p) / ISLPED 2026 (6p)
- **Experts**: legacy-system-expert (primary, tier-2 rubric 허용)
- **Review Scores** (Phase 2' 최종): Nov **7.4**/10, Diff **7.7**/10, Imp **7.2**/10, Feas **8.8**/10, 평균 **7.80** (본 세션 Tier-2 1위, Conditional Accept)
- **Conditional Accept 조건**: ISLPED 제출 시 **power measurement (SM dynamic power) 1개 추가 필수**. 현재 latency + BW utilization 까지만.
- **Single mechanism (tier-2 rubric, 1 권장)**: Pull-based NVLink peer fetch with TTL lease. 2-GPU NVLink node (A100 또는 H100 × 2), VLA KV block (4 KB) peer fetch. TTL lease {100ms, 500ms, 2s} sweep, pull-batch size {4, 16, 64} sweep.
- **Scope**: 1 hardware (2-GPU NVLink node), 1 workload (OpenVLA-7B batch 8), VLA-specific KV access pattern.
- **Baselines (3편, peer-reviewed 100%)**: vLLM PagedAttention [SOSP 2023, [arXiv:2309.06180](https://arxiv.org/abs/2309.06180)], Llumnix [OSDI 2024], NCCL P2P baseline [SC 2019].
- **예상 효과 (best-case sub-table)**: Peer-fetch latency p50 -15~25%, NVLink BW utilization +10~18%p, SM dynamic power -5~10% (ISLPED 요건).
- **Top-tier 와의 관계**: **Precedence 확보**. NACK-Gossip Top-tier (SOSP/OSDI) 의 M2 를 독립 letter 로 분리. ESL paper 는 SOSP Section 3.2 building block 으로 인용.
- **전문가 합의**: ai-opt N (AI optimization 관여 적음, tier-2 rubric 상 허용) + legacy-sys Y (strong, core).
- **Phase 3 entry actions**: (a) 2-GPU NVLink node 확보, (b) VLA KV block peer-fetch microbenchmark, (c) **power measurement 추가** (Conditional → Accept 조건).

---

### L1 v2 ContextSM-Tri: Content-Axis SM/BW/KV Tri-Partition with Reconfig-Latency-Bounded Green Context
- **Date**: 2026-04-22 | **Mode**: 1 | **Session**: 링크
- **Experts**: legacy-system-expert (primary) + algorithm-expert (P1 E²IC distilled classifier 흡수)
- **Review Scores** (Phase 2' 최종, post-verification): Nov **6.8**/10 (↓ from 7.2 after [arXiv:2506.12204](https://arxiv.org/abs/2506.12204) concurrent discovery), Diff **7.3**/10, Imp **6.8**/10, Feas **7.5**/10, 평균 **7.00** (본 세션 1위 유지)
- **Phase 2 → Phase 2' Score Delta**: +0.87 → +0.67 (verification 반영)
- **Phase 2' 판정**: **Accept** (조건: Semantic Scheduling 대비 novelty 차별 명시)
- **유사 연구 대응**: Nova [arXiv:2509.21301](https://arxiv.org/abs/2509.21301) (50% concurrent, stage partition) + DuetServe [arXiv:2511.04791](https://arxiv.org/abs/2511.04791) (65% concurrent, P/D partition) 대비 **content axis (6-class taxonomy) × tri-knob (α_SM, α_BW, α_KV)** orthogonal. MIG (H100 수 초 reconfig) vs Green Context (ms reconfig) 축에서 "per-request short-lived tenant" niche. Bullet [arXiv:2504.19516](https://arxiv.org/abs/2504.19516) (60%), LithOS (SOSP'25, 55%), Hummingbird [arXiv:2601.04071](https://arxiv.org/abs/2601.04071) (adjacent). **Post-verification 추가 실존 논문**: [arXiv:2506.12204](https://arxiv.org/abs/2506.12204) Semantic Scheduling for LLM Inference (2025-06, 45-55% concurrent — content-axis semantic scheduling 이미 제안, 다만 tri-knob 과 MIG nested 는 없음), [arXiv:2603.07917](https://arxiv.org/abs/2603.07917) SageSched, [arXiv:2510.03243](https://arxiv.org/abs/2510.03243) PARS, [arXiv:2510.03334](https://arxiv.org/abs/2510.03334) Semantic-Aware Scheduling for GPU Clusters, [arXiv:2504.08784](https://arxiv.org/abs/2504.08784) SLOs-Serve, [arXiv:2601.10729](https://arxiv.org/abs/2601.10729) OrbitFlow, [arXiv:2508.18572](https://arxiv.org/abs/2508.18572) Strata (hierarchical context caching) 모두 baseline 에 포함. "NestedGPU" / "Taxonomy-Sched" placeholder 실존 부재 확인.
- **핵심 가설**: VLM/VLA multi-tenant 서빙에서 "요청 content taxonomy" 가 SM/HBM BW/KV budget 의 최적 비율을 다르게 결정한다. 6-class effective taxonomy (prefill-long / decode-memory / decode-compute / vision-encode / audio-decode / retrieval-heavy) + DistilBERT-tiny (또는 P1 E²IC distilled 10-dim activation L_early=2 probe) 로 <0.5ms 분류, 각 class 에 tri-knob profile (α_SM, α_BW, α_KV) 적용. Green Context (CUDA 12.5+) 4-SM 단위 partition + MIG H100 nested + cudaAccessPolicyWindow L2 persistent BW carving + cudaStreamCreateWithPriority + vLLM BlockSpaceManager KV budget extension.
- **접근법**:
  1. 6-class taxonomy classifier (DistilBERT-tiny 2M param 0.4ms 또는 P1 E²IC 260-param 50μs).
  2. α_SM → Green Context + MIG nested (datacenter H100) / Green Context only (RTX 5090 no-MIG).
  3. α_BW → cudaAccessPolicyWindow L2 residency + stream priority.
  4. α_KV → per-taxonomy prefill chunk + decode reserved block count.
  5. Nova orthogonality single ablation (Nova-only / L1-only / both stacked).
  6. A1 phase predictor cascade extension — multi-robot VLA fleet serving.
- **예상 개선**: L1-only vs Nova 평균 mixed 6-class workload p99 TTFT **-10~13%, TPOT -9~12%**. Nova+L1 stacked **-17~20% / -15~18%** (-23/-20% 초안에서 보수 하향). L1 vs MIG H100 per-request reconfig **-14~18% TTFT**. A1 cascade VLA fleet **-18~22% p99 latency**. Uniform-taxonomy batch (retrieval-only) **<3% (scope 밖)**.
- **적용 범위**: Datacenter H100/H200 (MIG + Green Context nested) + consumer RTX 5090 / Pro 6000 Blackwell (Green Context only). Per-request short-lived tenant mix 가 주 scope. **미적용**: uniform batch workload (예: 동일 task 만), Ampere/older GPU (Green Context 미지원). **불확실**: RTX Pro 6000 Blackwell 의 MIG 지원 여부 (확인 필요).
- **실험**: RTX 4090 (primary no-MIG), RTX Pro 6000 96GB, AWS p5 H100 (MIG 비교 대여 1-2주) / Models Qwen2.5-VL-7B + LLaVA-NeXT-13B + InternVL3-8B / Workload LMMs-Eval synthetic mixed trace (OCR/grounding/caption/chat/reasoning Poisson λ=2-8) + MMDU multi-turn / Baselines vLLM, SGLang, Nova (reimpl), DuetServe (reimpl), Bullet, LithOS, Adrenaline, Helix, Llumnix, JITServe, Scorpio, Hummingbird, POD-Attention, DynamoLLM, Aegaeon, **MIG H100 per-request reconfig** / Metrics p50/p90/p99 TTFT/TPOT, SM occupancy (NCU), HBM BW (NVML), L2 hit rate (NCU) / Ablation (a) SM only, (b) BW only, (c) KV only, (d) tri-knob, (e) classifier ground-truth oracle vs DistilBERT vs P1 E²IC vs random / Expected runtime ~2개월.
- **차별점**: (1) **content-axis taxonomy → tri-knob mapping** 은 Nova/DuetServe/Bullet/LithOS 모두 미적용 (stage or size based). (2) **MIG + Green Context nested** 구조 (slow outer × fast inner) 는 published 보고 부재. (3) **10 baseline (최다)** 으로 positioning comprehensive. (4) A1 phase predictor cascade 로 VLA fleet serving 확장. Target venue: **MICRO 2026 / HPCA 2027**.
- **Phase 3 entry actions**: (a) Azure LLM trace 에서 6-class exhaustiveness 검증, (b) Green Context ms-level reconfig NVIDIA benchmark 확인 또는 실측, (c) Pro 6000 MIG 지원 확인.
- **남은 risk**: DuetServe v2 가 content-aware 추가 시 novelty 침식. Green Context reconfig 가 수십 ms 이면 claim 수정. MIG 는 H100/H200 only.
- **상세**: 세션 Executive Summary Top 1 + Section 4.1 L1 refinement

---

### A3 v2 SemCOW-Deadline: Copy-on-Write Reference-Counted Vision KV with Deadline-Aware Green Context SM Yielding for Multi-Tenant VLM Serving
- **Date**: 2026-04-22 | **Mode**: 1 | **Session**: 링크
- **Experts**: ai-optimization-expert (primary) + legacy-system-expert (SM yielding stream priority)
- **Review Scores** (Phase 2' 최종, post-verification): Nov **6.5**/10, Diff **8.0**/10, Imp **8.0**/10, Feas **6.0**/10, 평균 **7.15** (본 세션 2위, verification 에서 +0.02)
- **Phase 2 → Phase 2' Score Delta**: **+0.46 (최대)** → +0.48 — cluster detection 포기 repositioning 성공 + placeholder 부재 확인으로 sliver 재확인
- **Phase 2' 판정**: **Conditional Accept (Strong)** — 3-인 전문가 strong consensus
- **유사 연구 대응**: Phase 2 에서 **Mosaic [arXiv:2604.10060](https://arxiv.org/abs/2604.10060) 75% + KVShare [arXiv:2503.16525](https://arxiv.org/abs/2503.16525) 72% + MPIC [arXiv:2502.01960](https://arxiv.org/abs/2502.01960) 70% 3-way scoop** — cluster detection novelty 포기하고 **sliver = page-granular reference-counted COW + top-k partial recompute + deadline-aware Green Context SM yielding** 로 narrow. Adrenaline [arXiv:2503.20552](https://arxiv.org/abs/2503.20552) 는 throughput 기반 SM yielding (idle yielding) — 본 idea 는 **SLO deadline miss risk 기반** yielding (orthogonal). ContextCache [arXiv:2506.22791](https://arxiv.org/abs/2506.22791), Nexus [arXiv:2507.06608](https://arxiv.org/abs/2507.06608), Llumnix (OSDI'24), DroidSpeak [arXiv:2411.02820](https://arxiv.org/abs/2411.02820) adjacent. **Post-verification 추가**: [arXiv:2603.14371](https://arxiv.org/abs/2603.14371) OxyGen (2026-03 VLA unified KV management, single-request 내 multi-task, ~35% adjacent — cross-task sharing axis 다름), [arXiv:2504.08784](https://arxiv.org/abs/2504.08784) SLOs-Serve, [arXiv:2501.12162](https://arxiv.org/abs/2501.12162) AdaServe, [arXiv:2601.10729](https://arxiv.org/abs/2601.10729) OrbitFlow, [arXiv:2511.02230](https://arxiv.org/abs/2511.02230) Continuum (multi-turn KV TTL) 모두 baseline 추가. **Foundation**: [arXiv:2309.06180](https://arxiv.org/abs/2309.06180) PagedAttention 의 기존 refcount + COW 위에 partial recompute + deadline-aware SM yielding 으로 extension. "SLOServe-GC" / "Refcount-KV" placeholder 실존 부재 확인 → sliver novelty 보존.
- **핵심 가설**: VLM multi-tenant 서빙에서 cluster-shared vision KV 는 **partial overwrite (top-k recompute) 가 빈번**하므로 block-level refcount COW 가 full-share/full-copy 이분보다 효율적이다. Tenant SLO deadline 을 입력으로 Green Context SM 을 **deadline miss risk** 기반 동적 재분배하면 Adrenaline 의 idle yielding 대비 p99 latency 를 추가 개선할 수 있다.
- **접근법**:
  1. Cluster detector 는 Mosaic/KVShare 에서 차용 (novelty claim X).
  2. vLLM PagedAttention block 단위 **refcount COW** — shared block write 시 해당 block 만 fork.
  3. **Top-k partial recompute** — cluster hit 이지만 attention top-k 다르면 top-k block 만 fork+recompute (full recompute 대비 O(k/N)).
  4. **Deadline-aware Green Context SM yielding** — 매 decode step 마다 tenant 별 miss risk = (time_to_deadline / estimated_remaining_compute) 계산, risk > τ tenant 에 SM 확장, risk < τ 는 축소.
  5. **Reference-count GC** — COW fork rate > threshold 시 eager eviction.
  6. A3 v2 ablation: detector / COW only / Deadline only / Full / stacked with Adrenaline / cluster hit rate sweep 30/50/70%.
- **예상 개선**: Cluster-hit 50% p99 TTFT **-26% (520→385 ms)**, SLO attainment (deadline=1s) **82→94%**, Vision KV memory (8 tenants) **-31% (18.5→12.8 GB)**, Throughput (mixed) **+29% (24→31 req/s)**, COW fork rate 14%. **Cluster hit <15%** 에서는 **-3~5% 역효과 가능** (COW overhead), threshold GC 필수. **Tenant 수 <4** 에서는 deadline yielding 효과 미미.
- **적용 범위**: VLM multi-tenant serving (enterprise document understanding, robot fleet remote-serving, multi-turn image chat). 동일 이미지 multi-query workload. **미적용**: single-user workstation (multi-tenancy 부재). **불확실**: Write-on-shared fork rate 가 14% 예상 초과 시 memory saving 무효화.
- **실험**: RTX Pro 6000 96GB (multi-tenant 시뮬레이션 primary) / Models Qwen2.5-VL-7B + LLaVA-OneVision-7B / Workload LMSys VisionArena multimodal subset (접근 시도) + synthetic cluster hit 30/50/70% + robot fleet synthetic + cluster miss control / Baselines vLLM+Mosaic detector, KVShare, MPIC, Adrenaline, ContextCache, Nexus, Llumnix, DroidSpeak, LMCache / Ablation (a) COW only FIFO, (b) Deadline only, (c) Full, (d) +KVShare token edit, (e) cluster hit rate sweep, (f) top-k sensitivity / Expected runtime ~3개월.
- **차별점**: (1) **page-granular refcount COW + top-k partial recompute** 는 Mosaic/KVShare/MPIC 모두 write-time 분기 없음. (2) **deadline miss risk 기반 SM yielding** 은 Adrenaline 의 idle yielding 과 orthogonal (throughput vs SLO). (3) Cluster detection 포기 로 scoop 축 정면 회피 + sliver 정면 집중. Target venue: **OSDI 2027 / SOSP 2026 / NSDI 2027**.
- **Phase 3 entry actions**: (a) LMSys VisionArena / WildVision trace 확보, (b) Mosaic+KVShare detector 차용 path 확정, (c) CUDA 12.5 Green Context SM yielding 실측, (d) top-k sensitivity study 선행.
- **남은 risk**: Mosaic/KVShare v2 가 COW 추가 시 sliver 축소. Single-workstation multi-tenant "simulation" 의 realism 을 reviewer 가 의심 가능 → real cloud trace replay 로 보강.
- **상세**: 세션 Executive Summary Top 2 + Section 4.1 A3 repositioning

---

### A1 v2 PhaseGraph-VLA: Trajectory-Phase Conditioned CUDA Graph Dispatcher with SSE-Driven Boundary Detection
- **Date**: 2026-04-22 | **Mode**: 1 | **Session**: 링크
- **Experts**: ai-optimization-expert (primary) + algorithm-expert (P2 SSE Page-Hinkley 흡수)
- **Review Scores** (Phase 2' 최종, post-verification): Nov **6.8**/10, Diff **7.5**/10, Imp **6.5**/10, Feas **7.5**/10, 평균 **7.08** (본 세션 3위, L2 와 tiebreak 승: impact×feasibility 48.75>43.40). Verification 후 변동 없음.
- **Phase 2 → Phase 2' Score Delta**: +0.70 (SSE 흡수 + 5개 baseline + 4-way ablation + SimplerEnv/RoboCasa/Jetson 확장)
- **Phase 2' 판정**: **Conditional Accept** (Page-Hinkley FP rate empirical 필요)
- **유사 연구 대응**: AC²-VLA [arXiv:2601.19634](https://arxiv.org/abs/2601.19634) (algorithmic token prune + component skip, orthogonal axis 유지), ADP-VLA [arXiv:2509.22093](https://arxiv.org/abs/2509.22093) (65% concurrent, action-aware pruning), Nova [arXiv:2509.21301](https://arxiv.org/abs/2509.21301) (60% concurrent, SM partition VLM 아님 VLA), VLA-Cache [arXiv:2502.02175](https://arxiv.org/abs/2502.02175) (55%), KV-Efficient VLA [arXiv:2509.21354](https://arxiv.org/abs/2509.21354), SpecPrune-VLA [arXiv:2509.05614](https://arxiv.org/abs/2509.05614), TTF-VLA [arXiv:2508.19257](https://arxiv.org/abs/2508.19257), Running-VLAs-Real-time [arXiv:2510.26742](https://arxiv.org/abs/2510.26742), VLA-Pruner [arXiv:2511.16449](https://arxiv.org/abs/2511.16449). **Post-verification**: "Helix-VLA" placeholder 부재 확인 (Figure AI Helix 는 blog only, 논문 없음). [arXiv:2505.03912](https://arxiv.org/abs/2505.03912) OpenHelix (dual-system VLA survey+opensource) adjacent baseline 추가.
- **핵심 가설**: VLA trajectory 의 **phase heterogeneity** (Approach / Manipulate / Retract) 는 execution graph 수준의 kernel fusion boundary 차이를 만든다. 이를 **SSE (Hidden State Shift Estimator, Page-Hinkley 2-threshold)** 로 online 검출하고 **phase-specific CUDA Graph** (phase × batch 2D, 3 pre-captured variant) 를 dispatch 하면, VLA-Cache/AC²-VLA 의 algorithmic axis 와 **orthogonal stacking** 으로 LIBERO median latency 를 -22~30% 단축한다.
- **접근법**:
  1. **SSE phase predictor** — L_mid hidden state L2 drift + EWMA + Page-Hinkley 2-threshold τ_enter/τ_exit + hysteresis 3 frame. VLA 에서는 gripper state Δ + trajectory curvature + DINOv2 object distance (1.2ms/frame) feature 추가. 결정 <100μs.
  2. **Phase-specific CUDA Graph dispatcher (phase × batch 2D)**:
     - Approach graph: SigLIP patch-embed full + FlashAttn-3 MQA fused.
     - Manipulate graph: SigLIP partial-batch 40% re-encode + action-head SiLU+Linear fused + reduced attention heads.
     - Retract graph: SigLIP bypass (2-step linear extrapolation) + KV static reuse.
  3. **Phase-specific SM partition (optional stacking)** — Nova 와 stacking 축 분리. Approach 80% encoder, Manipulate 35/55/10 split, Retract 0/85/15.
  4. 4-way ablation: baseline / phase-graph-only (oracle phase) / SSE-only / full (graph+SSE) / stacked-with-Nova.
- **예상 개선**: LIBERO median latency 165→**128 ms (-22%)**, full+Nova stacked **115 ms (-30%)**, SimplerEnv Hz 6.1→**7.8**, RoboCasa success rate Δ **±1.2pp**, Jetson Orin 64GB latency **420→330 ms**, **sub-120ms median end-to-end (real-time marketing 금지 보수 표현)**. Short 1-shot pick 에서는 ±3% (scope 밖).
- **적용 범위**: Autoregressive dense VLA (OpenVLA/OpenVLA-OFT/CogACT/RT-2 계열). LIBERO, SimplerEnv, RoboCasa. **부분적용**: π0 계열 (diffusion action head 는 추가 설계). **미적용**: pure VLM, end-to-end learned policy with no phase structure. **불확실**: open-ended manipulation 에서 3-class phase predictor 정확도 저하.
- **실험**: RTX 4090 24GB (primary), RTX Pro 6000 96GB (cross-GPU), Jetson Orin AGX 64GB (optional embedded) / Models OpenVLA-7B (primary), OpenVLA-OFT-7B / Datasets LIBERO 4 split + SimplerEnv Google Robot + WidowX + RoboCasa (optional) / Baselines OpenVLA vanilla, VLA-Cache, AC²-VLA, KV-Efficient VLA, PD-VLA, **DuetServe, TTF-VLA, Scorpio, SpecPrune-VLA, Running-VLAs-Real-time** / Ablation 4-way + Nova stacked / Expected runtime ~2개월.
- **차별점**: (1) **phase × batch 2D CUDA Graph dispatcher** 는 vLLM 1D batch-only variant 와 분리. (2) **SSE Page-Hinkley** 는 training-free quantile calibration. (3) **Phase-specific kernel fusion boundaries** (ViT-MQA / SiLU-Linear / KV static reuse) 는 VLA 문헌 unique. (4) **Orthogonal with Nova + VLA-Cache + AC²-VLA** — 4-way ablation 으로 stacked gain 증명. Target venue: **MLSys 2026 / CoRL 2026 / NeurIPS 2026 D&B**.
- **Phase 3 entry actions**: (a) Page-Hinkley FP rate LIBERO 5 task empirical, (b) SimplerEnv + RoboCasa env 구축, (c) Jetson Orin 접근 (optional), (d) CUDA Graph capture overhead amortization.
- **남은 risk**: Nova stacking gain < 3% 이면 merge 요구. Open-ended trajectory 에서 3-class phase structure 부정. Helix-VLA placeholder 실존 시 재평가.
- **상세**: 세션 Executive Summary Top 3 + Section 4.1 A1 refinement

---

### F2-VLM Quantization-Robust Layered Defense for VLM-PIM (HW KL-Collapse Detector + SW Per-Sample Fragility Gating)
- **Date**: 2026-04-22 | **Mode**: 2 | **Session**: [링크](/research-wiki/2026-04/vlm-pim-extension)
- **Experts**: hw-pim-accelerator-expert + ai-optimization-expert (fused layered defense)
- **Review Scores** (Post-Literature Survey 최종): Nov **8.5**/10, Diff 8.5/10, Imp **8.3**/10, 평균 **8.43** (본 세션 최고)
- **Phase 2 → Phase 2' → Post-Survey Score Delta**: +0.5 (layered 구조) → +0.2 (W8A8 collapse 선행 보고 부재 검증으로 measurement contribution 추가)
- **Phase 2' 판정**: **Accept**
- **Post-Literature Survey Verdict (핵심 novelty 근거)**: MBQ/Q-VLM/VLMQ/MQuant/AKVQ-VL/VL-Cache/LLM.int8/SmoothQuant/QuaRot/SpinQuant 등 주요 VLM quantization 논문 **모두 task accuracy 만 보고, layer-wise visual attention ratio 를 quantization 전/후로 직접 측정한 사례 부재**. Qwen-VL + W8A8 + attention ratio 정량화는 완전 공백. 본 연구의 +66.85pp (5.73% → 72.58%) extreme collapse 는 **선행 보고 없는 novel observation** — F2 에 **measurement contribution** 추가됨.
- **유사 연구 대응**: P3-LLM ([arXiv:2511.06838](https://arxiv.org/abs/2511.06838), ~55-65% concurrent) 은 static per-layer precision vs runtime KL trigger + BF16 fallback 축으로 차별화. Fallback Quantization ([arXiv:2503.08040](https://arxiv.org/abs/2503.08040), ~25% on refined claim) 은 training GEMM 대상이고 F2 는 inference PIM macro. VL-Cache ([arXiv:2410.23317](https://arxiv.org/abs/2410.23317)) / AKVQ-VL ([arXiv:2501.15021](https://arxiv.org/abs/2501.15021)) 은 layer-level policy 로 orthogonal. MBQ ([arXiv:2412.19509](https://arxiv.org/abs/2412.19509)) 는 calibration sensitivity 축으로 runtime detection 과 구분. LLM.int8 ([arXiv:2208.07339](https://arxiv.org/abs/2208.07339)) 은 outlier → softmax mass 감소 방향 보고 (+66pp 의 반대 방향), Visual Attention Sink ([arXiv:2503.03321](https://arxiv.org/abs/2503.03321)) 는 FP16 내재적 sink 현상 (양자화 영향 미분석). Seeing but Not Believing ([arXiv:2510.17771](https://arxiv.org/abs/2510.17771)) 은 attention vs answer correctness decoupling 근거 — F2 의 "MMMU acc 유지하나 attention 붕괴" 해석 지지.
- **핵심 가설**: VLM W8A8 pattern collapse (+66.85pp visual attn) 와 FP16 L27 self-attn overflow (>65504 token 3.2%) 는 (a) **HW macro-level KL-collapse detector + BF16 fallback path** (layer granularity) 와 (b) **SW per-sample fragility gating via L0-L1 probe + W8A8/weight-only KL disagreement** (sample granularity) 의 **layered defense** 로 robust 하게 해결 가능하다. Aggregate corr 0.996 vs individual 0.357 gap 은 sample 간 fragility variance 가 aggregate 평균에 masking 됨을 증명.
- **접근법**: (1) AttAcc simulator 내부에 per-layer KL comparator logic 추가 (W8A8 output vs weight-only-INT8 shadow output), KL > threshold 시 해당 layer 를 BF16 dedicated path 로 escalation. (2) L27 self-attn 의 Q·Kᵀ abs_max 가 FP16 max 근접하면 즉시 BF16 rescue. (3) SW side: L0-L1 probe 로 per-sample fragility score 계산 (LR classifier, input feature ≤10차원, <3% FLOPs). (4) W8A8 shadow inference 를 처음 4 layer 만 실행하여 weight-only-INT8 와의 KL divergence signal 을 추출 — KL > τ 인 sample 은 fragile 분류. (5) Policy dispatcher: stable 85% → 4-B Layer-Adaptive PIM placement 적용; fragile 10-15% → L17-21 peak layer 만 GPU HBM 강제 + 나머지 PIM; highly fragile 2% → full GPU attn. (6) HW-SW 인터페이스 프로토콜 명시 (race condition 방지, rollback cost 정량).
- **예상 개선**: W8A8 MMMU accuracy drop -12.4% (collapse) → **-1.5~-2.5%**; FP16 L27 overflow rate 3.2% → **<0.3%** (10× reduction); per-sample worst-case corr **0.357 → 0.70-0.80** (2× robustness); macro area overhead <3%; latency overhead <5%; aggregate corr 0.996 유지.
- **적용 범위**: VLM PIM 서빙 (Qwen3-VL-4B/LLaVA-NeXT-7B/InternVL3-8B W8A8/W4A8 regime), production quantized deployment (edge/server). **미적용**: FP16/BF16 full-precision regime (detector idle), pure text-LLM (VLM 특화 FP16 L27 overflow 없음). **불확실**: 10-15% fallback budget 이 long-tail OCR/dense-document MMMU 에서 초과할 가능성.
- **실험**: 2×A6000 (48GB each) + AttAcc sim macro-level KL comparator extension / Models Qwen3-VL-4B (W8A8, W4A8, FP16 각각), LLaVA-NeXT-7B, InternVL3-8B / Datasets MMMU, MMBench, TextVQA, DocVQA, MMStar / Fragility score = per-sample Hessian trace proxy (gradient variance on calibration) / Ablation: (a) HW KL detector only, (b) SW PatternGuard only, (c) layered both, (d) P3-LLM static / Baselines: SmoothQuant [arXiv:2211.10438](https://arxiv.org/abs/2211.10438), AWQ [arXiv:2306.00978](https://arxiv.org/abs/2306.00978), VL-Cache, AKVQ-VL, P3-LLM, KVQuant [arXiv:2401.18079](https://arxiv.org/abs/2401.18079) / Expected runtime ~3개월.
- **차별점**: (1) **HW macro detector (layer granularity) + SW fragility gating (sample granularity) 의 cross-granularity layered defense 는 VLM PIM 문헌에 없음**. P3-LLM/MBQ/AKVQ-VL 모두 single-granularity. (2) **W8A8 +66pp visual attention collapse 정량화 최초** — 기존 VLM quant 문헌은 task accuracy 만 보고 (measurement contribution). (3) **Mechanism triangulation** — LLM.int8 softmax outlier 왜곡 + Visual Attention Sink 증폭 + MBQ modality gradient gap 3 메커니즘이 VLM W8A8 에서 catastrophic regime 로 수렴함을 통합 설명하는 첫 연구. (4) VLM-specific numerical-safety (FP16 L27 overflow) 의 sub-microarchitecture-level 증명은 최초. Target venue: **HPCA 2026 / MICRO 2026** (HW macro + quant robustness 축).
- **본 연구의 E5 추가 권장 실험 (novelty 공고화)**: (a) Bit-width sweep W6A6/W4A4/W4A8/W4A6 — collapse 가 activation precision 함수인지 연속 곡선 확인 (예상: A 축 감소 시 Δ 증가, W8A16 은 preserve). (b) Recipe sweep SmoothQuant vs QuaRot vs SpinQuant vs MBQ — rotation 기반이 collapse 완화하면 mechanism (c) rotation-invariance 검증. (c) Per-modality outlier analysis (vision-token activation max/99.9-percentile). (d) Sink dimension tracking (ϕ(x) metric FP16 vs W8A8). (e) Task-attention decoupling (MMMU acc 유지 but attention 붕괴 sample 을 Seeing but Not Believing 과 연결). (f) Cross-family 재현 (LLaVA-1.6, InternVL2.5, MiniCPM-V). (g) Layer-localization heatmap (collapse 발생 layer 정확한 범위).
- **남은 risk**: KL threshold tuning model-specific (5-model calibration protocol) / Layered HW-SW 인터페이스 race condition (프로토콜 1절 추가 조건) / Fallback budget 초과 시 SLO 위반 (long-tail profiling 필요) / Recipe sweep 에서 QuaRot/SpinQuant 가 W8A8 collapse 를 SW-only 로 해결하면 HW detector 필요성 약화 위험 — 이 경우 F2 는 "rotation 적용 불가한 legacy 배포 환경" scope 로 축소.
- **상세**: [세션 5절 F2 + 10.5절 post-literature survey](/research-wiki/2026-04/vlm-pim-extension)

---

### F1-VLM DeepStack-Native Prefill-Decode Pipeline with 6-Tier KV Tiering (GPU+PIM)
- **Date**: 2026-04-22 | **Mode**: 2 | **Session**: [링크](/research-wiki/2026-04/vlm-pim-extension)
- **Experts**: hw-pim-accelerator-expert + ai-optimization-expert (A3 architecture-aware adapter 흡수)
- **Review Scores** (Phase 2' 최종): Nov 7/10, Diff 8/10, Imp 7.8/10, 평균 **7.60**
- **Phase 2 → Phase 2' Score Delta**: +0.8 (DeepStack topology + AI inflection 교집합으로 tier 정당화, C-adaptive dispatcher 추가)
- **Phase 2' 판정**: **Conditional Accept** (DeepStack 외 VLM family ablation + AI inflection 이동성 sensitivity 조건)
- **유사 연구 대응**: STARC ([arXiv:2505.05772](https://arxiv.org/abs/2505.05772), ~40% adjacent, AttAcc sim 공통) 는 token clustering bank remapping vs layer-topology + AI inflection dispatch 로 경쟁 axis 분리 + 베이스라인 필수. VLCache ([arXiv:2512.12977](https://arxiv.org/abs/2512.12977), ~45-55% concurrent) 는 attention sparsity tier vs DeepStack inject topology tier 로 narrative 좁힘. Jenga ([arXiv:2503.18292](https://arxiv.org/abs/2503.18292), ~50-60% concurrent) 는 generic layer placement vs VLM-DeepStack-specific vision injection. Mnemosyne ([arXiv:2409.17264](https://arxiv.org/abs/2409.17264), STILL CLEAR) latency equalization 목적 vs AI inflection 목적 구분.
- **핵심 가설**: Qwen3-VL 의 DeepStack injection layer **[L4, L8, L12]** 는 PIM offloadable KV 의 **arithmetic intensity inflection point (AI<60 memory-bound 구간)** 와 구조적으로 일치하며, 이 교집합을 exploit 한 **chunked-prefill dispatcher (C∈{8,16,32,64}) + 6-tier KV placement** 가 AttAcc baseline 대비 E2E 1.45-1.60× speedup 을 달성한다. A3 architecture-aware adapter 는 F1 의 evaluation generalization 모듈로 흡수되어 LLaVA-NeXT/InternVL3/Qwen3-VL 3 family cross-validation 을 제공.
- **접근법**: (1) AttAcc simulator 4-file extension — config/system/model/ramulator_wrapper — 를 본 연구의 P1 작업과 병행하여 prefill PIM path 활성화. (2) DeepStack inject layer [4,8,12] + L17-21 peak + L27 overflow 기준 6-tier 정의 (Tier 0: L0-L3 pre-inject, Tier 1: L4-L7 post-inject-1, Tier 2: L8-L11 post-inject-2, Tier 3: L12-16 post-inject-3, Tier 4: L17-L21 peak=GPU HBM full bf16, Tier 5: L22-L27 tail=BF16 escalation with overflow safe). (3) C-adaptive dispatcher: C∈{8,16,32,64} sweep, video L=8948 regression 시 C=8 로 하향. (4) Cross-request ViT feature cache: 같은 image 의 L4/L8/L12 inject 는 ViT 출력 재사용 (4-C 연장). (5) A3 adapter — Mllama cross-attn 은 PIM immutable region, Qwen3.5 hybrid linear 는 KV skip. (6) Per-tier compression: Tier 0/5 INT4 weight-only, Tier 1-3 aggressive eviction 60-80%, Tier 4 GPU HBM bf16 강제.
- **예상 개선**: E2E latency (Qwen3-VL-4B, prefill L=4096) AttAcc 대비 **0.62-0.69× (1.45-1.60×)** ; TTFT chunked C=16 **0.55-0.65× (1.53-1.80×)** ; decode throughput 20-30% 향상; video L=8948 PIM **0.81× → 0.95-1.05× (regression 회복)**; FHD TTFT 1,285ms → 650-800ms.
- **적용 범위**: DeepStack VLM (Qwen3-VL family 주), 단일/다중 이미지 VQA, chunked prefill production serving. A3 adapter 로 LLaVA-NeXT/InternVL3 early-fusion 확장 가능. **미적용**: Mllama cross-attn (A3 immutable region 으로만 부분 지원), Qwen3.5 hybrid linear (KV skip 전용 path), pure text LLM. **불확실**: Non-DeepStack VLM 에서 AI inflection 위치 이동 (sensitivity study 필수).
- **실험**: 2×A6000 (48GB each) + AttAcc cycle-accurate simulator extension / Models Qwen3-VL-4B (primary), Qwen2-VL-7B, LLaVA-NeXT-7B, InternVL3-8B, MiniCPM-V-2.6 (5-model generalization) / Workloads LLaVA-Bench-Wilder, VideoMME, MMMU, TextVQA / Chunk size sweep C∈{8,16,32,64}, seq len {512, 2048, 4096, 8948} / Ablation: (a) DeepStack-aware tier vs naive 6-tier, (b) C-adaptive vs fixed C=16, (c) A3 adapter on/off, (d) tier count sweep {2,4,6,8,12} / Baselines AttAcc, NeuPIMs [arXiv:2403.00579](https://arxiv.org/abs/2403.00579), STARC, VLCache, Jenga, vLLM+PagedAttention [arXiv:2309.06180](https://arxiv.org/abs/2309.06180) / Expected runtime ~3-4개월.
- **차별점**: **DeepStack inject topology × arithmetic intensity inflection 교집합 tier** 는 VLM PIM 문헌에 없음. STARC/VLCache/Jenga 는 각각 token clustering / attention sparsity / generic layer placement 로 축 다름. C-adaptive dispatcher 로 video regression 회복은 unique. 5-model A3 adapter 로 VLM family generalization. Target venue: **ASPLOS 2026 / MLSys 2026** (시스템 + 서빙 축).
- **남은 risk**: DeepStack 외 family 일반화 (A3 adapter 로 mitigate, ablation 필수) / AI inflection 이동성 (HBM-PIM vs LPDDR-PIM sensitivity) / C=16 sweet spot production workload universal 여부 (dispatcher oscillation risk).
- **상세**: [세션 5절 F1 + 7절](/research-wiki/2026-04/vlm-pim-extension)

---

### F1 TempoPRISM-CoDesign: Streaming KWS 용 Time-Axis Xbar-wise SF + VAD-Gated Warm-LUT 공동 설계
- **Date**: 2026-04-22 | **Mode**: 1 | **Session**: 링크
- **Experts**: algorithm-expert + hw-pim-accelerator-expert (fused co-design)
- **Review Scores** (Phase 2' 최종): Nov 7.5/10, Diff 8/10, Imp 8/10, 평균 **8.0**
- **Phase 2 → Phase 2' Score Delta**: +1.0 (fuse 효과로 SparkNet/LoRDS scoop 방어)
- **Phase 2' 판정**: Accept
- **유사 연구 대응**: SparkNet ([arXiv:2406.06634](https://arxiv.org/abs/2406.06634), ~45% 겹침) 은 rank-1 K=2 특수화로 reframe, LoRDS ([arXiv:2601.22716](https://arxiv.org/abs/2601.22716), ~45%) 는 LLM PTQ vs BNN-CIM-scratch 축 차이로 방어 (baseline 에 LoRDS-transferred-to-BNN 포함)
- **핵심 가설**: Streaming KWS 의 4-8 frame-phase cluster (silence/onset/voiced/offset) 를 rank-2 decomposition 의 time-axis 로 편입 + 동일 gate 를 HW clock-gate signal 로 재활용하면, SF buffer 접근 98% 제거 + warm-LUT hit rate 95%+ + 2.5-3배 에너지/frame 감소가 algorithm 단독 (SparkNet) / HW 단독 (PSCNN) 접근 모두에서 불가능한 수준으로 성립.
- **접근법**: (1) 4-8 cluster k-means 로 frame-phase 학습 (silence/onset/voiced/offset). (2) `S_xbar(c,p) = Σ_r S_time(r,c) × S_space(r,p)` rank-2 time-space decomposition. (3) VAD 신호 = cluster index = HW clock-gate signal (gate MLP 의 SRAM/FP overhead 완전 제거). (4) Non-volatile RRAM meta region 에 warm-LUT 저장 → cold-start 제거. (5) K=8 frame 주기 full-refresh 로 rolling-residue analog σ 누적 방어. (6) Gate entropy 기반 periodic refresh trigger.
- **예상 개선**: SC v2 Top-1 93.0-94.0% (BiFSMNv2 93.5% 경쟁), SF buffer reads ≤80 per frame (98% 감소), warm-LUT hit rate 93-96% (PRISM 82% 기준), streaming-amortized 에너지 2-2.5배 PSCNN, p99 streaming latency 12-14ms.
- **적용 범위**: Streaming KWS (SC v2, MSWC, 35-class, BiFSMN/BC-ResNet 규모 1-5M params). **미적용**: Transformer-KWS, rich-vocabulary ASR/TTS. **불확실**: DEMAND 0-5dB SNR 하 VAD false-negative cold-start 복귀.
- **실험**: 서버 #5 (RTX Pro 6000) algo training + NeuroSIM + XNOR-RRAM + FinCACTI / BiFSMN-small (2.5M) + BC-ResNet-8 (5M) scratch / Baselines: BiFSMNv2, PRISM-channel-wise, Static Xbar-wise, LoRDS-transferred-to-BNN, SparkNet+PRISM, PSCNN, CIMR-V / **~9일**.
- **차별점**: SparkNet (rank=1 gate-only, CPU 타겟) 은 본 idea 의 strict special case — 실험에서 ablation 으로 명시 포함. LoRDS (LLM PTQ post-hoc) 는 scratch training + BNN + CIM 축에서 구조적으로 다름. BiFSMNv2 (CPU channel-wise) 는 CIM substrate 자체가 다름 — NeuroSIM 공통 mapping 으로 fair 비교. Target venue: ISLPED 2027 / DATE 2027.
- **상세**: 세션 6.1절

---

### F2 PhysioPRISM-VitalXbar: Subject-Adaptive OPTIC + Heterogeneous-Precision 2-Region CIM for Wearable Biosignal
- **Date**: 2026-04-22 | **Mode**: 1 | **Session**: 링크
- **Experts**: algorithm-expert + hw-pim-accelerator-expert (fused co-design)
- **Review Scores** (Phase 2' 최종): Nov 8.5/10, Diff 8.5/10, Imp 8.5/10, 평균 **8.5** (본 세션 최고점)
- **Phase 2 → Phase 2' Score Delta**: +0.95 (2-region scope 축소로 training 안정성 확보, shadow SRAM 으로 RRAM endurance 해결)
- **Phase 2' 판정**: Accept
- **유사 연구 대응**: TTAQ ([arXiv:2412.09899](https://arxiv.org/abs/2412.09899), ~35% 겹침) 는 FP INT8 CPU/GPU vs BNN+CIM+hetero-precision 축 차이로 분리, baseline 필수 포함. Tent ([arXiv:2006.10726](https://arxiv.org/abs/2006.10726)) / AdaBN ([arXiv:1603.04779](https://arxiv.org/abs/1603.04779)) 은 BN 있는 FP 모델 전제 — BN-free A&B BNN 에서 직접 적용 불가, μ_in 기반 우리 방법이 등가물. Arrhythmia-BNN ([arXiv:2304.01568](https://arxiv.org/abs/2304.01568)) 의 "binary-only 95% 달성" 반박은 VT/VF rare class confusion matrix 로 INT4 필요성 증명.
- **핵심 가설**: Wearable BNN-CIM 배포에서 **subject shift 는 activation mean μ 의 scalar shift 로 근사 가능**하며, 30초 unsupervised 신호 → per-layer μ̂_u 계산 → delta-LUT 패치만으로 subject-shift 정확도 gap 의 50-70% 회복. Shadow SRAM 에 delta 저장하면 RRAM weight 0 write → 24h 연속 동작 하에서 device lifetime 5년 이상 보장.
- **접근법**: (1) Binary HAR region (ResNet1D-18, ~500K) + INT4 ECG region (ResNet1D-ECG, ~1M) 2-region heterogeneous-precision CIM (scope 축소). (2) Stagewise training: HAR 먼저 binary STE → freeze → ECG INT4 LSQ (gradient scale 10² 차이 회피). (3) Per-layer scalar α 로 μ_in(u) = μ_baseline + α·(μ̂_u - μ_baseline) meta-learning. (4) Delta-LUT 를 4KB shadow SRAM bank 에 기록 (RRAM 불변). (5) CORAL covariance alignment (primary) + GRL (optional, 불안정 시 disable). (6) Activity-gated duty cycling: IMU motion 감지 시 HAR region 활성, ECG 는 상시 저듀티. (7) Dual ADC 혹은 1초 buffer pipeline 으로 250Hz ECG sampling 충돌 회피.
- **예상 개선**: UCI-HAR LOSO **76-80%** (Fixed-SF PRISM ~70% 대비 +6~10pp), MIT-BIH VT/VF sensitivity 95%+ (INT4), continuous power **<50 µW** (24h, 10% motion), RRAM write count 0, area +4-6%.
- **적용 범위**: Wearable/implantable BNN-CIM (IMU + ECG), subject adaptation 필요 시나리오. **미적용**: FP32 환경, 12-lead ICU-grade continuous diagnostic (INT4 부족). **불확실**: 고강도 motion 지속 시 HAR region duty 100% → 50µW 목표 초과.
- **실험**: 서버 #5 (RTX Pro 6000) / NeuroSIM + FinCACTI + 28nm Synopsys synthesis / ResNet1D-18 (HAR) + ResNet1D-ECG stagewise / UCI-HAR LOSO (30 subjects) + MIT-BIH AAMI 5-class + PAMAP2 ablation / **~10일** / Baselines: Fixed-SF PRISM LOSO, TTAQ, Tent-on-BNN, AdaBN-on-BNN, Arrhythmia-BNN, RTF-Q / Metrics: LOSO acc, VT/VF sensitivity, 24h power, RRAM endurance.
- **차별점**: **BNN + CIM + biosignal + subject-shift 4-way intersection 이 2024-2026 arxiv 에서 empty** (최대 강점). Delta-LUT on shadow SRAM 은 RRAM endurance 보존하는 HW-algo co-design novelty. Heterogeneous-precision 2-region (binary HAR + INT4 ECG 한 칩) 은 ISSCC 2025 RRAM/SRAM collab 의 per-modality partition 확장. Scalar-shift closed-form subject adaptation theorem 형식화 가능. Target venue: ISSCC 2027 / JSSC extended / TBioCAS / NeurIPS 2026 small-model track (algorithm 비중) 듀얼.
- **상세**: 세션 6.2절

---

### FARD-C: Fingerprint-Aware Request Dispatcher & Coalescing for Many-Expert MoE Serving
- **Date**: 2026-04-21 | **Mode**: 1 | **Session**: 링크
- **Experts**: ai-optimization-expert (main)
- **Review Scores** (refined): Nov 7/10, Diff 6/10, Imp 7/10, 평균 **6.67** (Top 1, I2와 공동)
- **핵심 가설**: 85-90% fingerprint 정확도를 활용해 동일 expert 사용 패턴을 가진 request를 같은 batch/replica로 묶으면, per-token unique-expert-loads가 2-3배 감소하고 decode throughput이 1.4-1.8배 증가한다. 60-70% 정확도에서는 오분류로 재사용 이득이 절반 이하로 떨어져 실현 불가.
- **접근법**: (1) DistilBERT 규모 prompt classifier (CPU 2-5ms) + 1024-bit SimHash 서명. (2) cross-replica dispatcher — replica별 `active_batch_fingerprint` EMA와의 Hamming 거리 최소화. (3) intra-replica batch reordering — 2-4개 fingerprint cohort를 micro-batch 단위로 dispatch. (4) signature-entropy 스케일링 정리 — LSH cohort purity가 유의미 임계를 넘는 공식 조건 `N_e ≥ 32` (Mixtral-8 실패, Qwen3-Next-128 성공). (5) 공정성 aging 메커니즘 (Jain index ≥ 0.90을 primary metric으로). (6) EPLB 보완 — demand-side replica routing이 supply-side expert 재배치를 보완.
- **예상 개선**: decode throughput 1.4-1.8배 (batch 16+, HBM 50% 제약), unique-expert-loads/layer 0.33-0.50배, Jain fairness ≥0.90, 정확도 변화 없음. miss rate 15%에서는 throughput 1.4-1.8배 유지; 40% 초과 시 자동으로 vanilla vLLM fallback (regression floor 없음).
- **적용 범위**: multi-replica MoE 서빙, Qwen3-Next-A3B / Qwen3-30B-A3B / DeepSeek-V2-Lite, batch 16+. **미적용**: single-request interactive, Mixtral-8 (signature-entropy 부족, 명시적 negative control). **불확실**: 워크로드 class heavy-tail로 인한 load imbalance.
- **실험**: 서버 #5 RTX Pro 6000 96GB (MIG 분할) 또는 #4 (4090×2 FSDP) / vLLM 0.9 + FingerprintDispatcher scheduler / LMSYS-Chat-1M + BurstGPT + 혼합 synthetic trace / 4-6일.
- **차별점**: Semantic Parallelism ([2503.04398](https://arxiv.org/abs/2503.04398), 2025-03, same-replica affinity)과 Gimbal ([2602.21626](https://arxiv.org/abs/2602.21626), 2026-02, load+affinity sticky) 대비 **cross-replica dispatch + 형식적 signature-entropy 정리 + EPLB 보완**. XShare(post-hoc)와 pre-batch 축 차이. METRO는 token-level, FARD-C는 request-level.
- **상세**: 세션 4.2절

---

### PhantomRoute: Routing Obfuscation Defense via Calibrated Dummy-Expert Activations
- **Date**: 2026-04-21 | **Mode**: 1 | **Session**: 링크
- **Experts**: system-robustness-expert (main)
- **Review Scores** (refined): Nov 7.5/10, Diff 6/10, Imp 6.5/10, 평균 **6.67** (Top 1, I4와 공동)
- **핵심 가설**: MoE의 observable expert set과 functional expert set을 decouple하면 (ε-weighted k=2 decoy 추가, 실제 output shift ≤ ε·‖h‖), side-channel topic classifier의 MI(trace; topic)가 5배 이상 감소한다. 이때 정확도 하락 ≤1.5pp, 지연 overhead ≤1.5배. expert 128개 모델에서는 decoy 하나당 mass 0.8%로 저렴하지만, expert 8개 Mixtral은 decoy당 12.5%로 동일 privacy 목표 달성 시 정확도 10pp 이상 손실 (명시적 negative control).
- **접근법**: (1) 오프라인으로 expert 공동활성화 clustering (T개 topic group), 각 group마다 반대 group의 decoy 후보 pool 선정. (2) 온라인으로 첫 layer router 통계로 group 추정 → 반대 group pool에서 k=2 decoy 샘플링. (3) vLLM fused-MoE kernel fork에서 ε=0.05 weighted gather. (4) per-request HMAC(server_secret, req_id) seed로 adaptive attacker 재학습 방어. (5) 형식적 MI upper bound 증명 `I(observable; topic) ≤ log T − H(decoy | topic)`. (6) **primary threat = MoEcho ([2508.15036](https://arxiv.org/abs/2508.15036), CCS'25)** 공격 4종에 대한 평가. (7) I1 ExpertEcho의 scaling-leakage 법칙(expert 수 증가 → privacy 감소)을 motivation 1-2쪽으로 흡수.
- **예상 개선**: MoEcho Prompt Inference 99.8% → ≤30% (Qwen3-Next), adaptive attacker ≤45%, MMLU −1.0±0.5pp, 지연 1.3-1.5배, MI ≤ 1/5배 baseline. Mixtral-8 negative control: 동일 MI 목표 시 MMLU −10pp 이상.
- **적용 범위**: Qwen3-Next-MoE / DeepSeek-V2-Lite / DeepSeek-V3-Lite (expert ≥64). **미적용**: expert-choice routing (별도 분석), Mixtral-class expert ≤16 (명시적 negative result, MoE architecture 설계 지침으로 활용). **불확실**: 방어 trace를 수집한 adaptive attacker.
- **실험**: 서버 #5 RTX Pro 6000 / vLLM + custom fused-MoE fork + MoEcho 공격 harness 재현 / 8-10일 (adaptive 재학습 2일 포함).
- **차별점**: 최초의 **inference-time, practical overhead** MoE routing-obfuscation defense (CryptoMoE/SecMoE는 100배 느림, NoEsis는 training 전용). 최초의 형식적 MI bound. expert granularity에 대한 (privacy/utility/compute) Pareto 최초 측정. MoEcho의 mitigations 섹션 및 저비용 baseline(logit noise, random permutation) 대비 우위.
- **상세**: 세션 4.4절

---

### ZMSP: Two-Tier Zero-Miss Speculative Expert Prefetch with WCRT Schedulability
- **Date**: 2026-04-21 | **Mode**: 1 | **Session**: 링크
- **Experts**: ai-optimization-expert (main)
- **Review Scores** (refined): Nov 6.5/10, Diff 5.5/10, Imp 7.5/10, 평균 **6.50** (Top 3)
- **핵심 가설**: 85-90% fingerprint 정확도에서 남은 10-15% miss를 **bounded-concurrency JIT path**로 처리하면 per-layer wait ≤ max(T_compute, T_PCIe) + 2·T_INT4 + aging_buffer가 성립해 p99 TPOT에 대한 형식적 WCRT 정리를 증명할 수 있다. 60-70% 정확도에서는 JIT 경로가 과부하로 p99이 오히려 악화 → 본 설계는 premise에 의존.
- **접근법**: (1) 처음 2개 MoE block의 router output → 3-layer MLP (12M, 80μs CUDA graph) — I4와 구조적으로 다른 predictor (artificial-split 방지). (2) Tier-1 prefetch: horizon H layer 앞서 top-(K+α). (3) Tier-2 JIT: INT4 HBM fallback 우선, 없으면 priority PCIe (**concurrent JIT cap = 2**). (4) **WCRT schedulability 정리** (핵심 기여). (5) safety valve: miss rate >25% 관찰 시 vanilla HOBBIT fallback. (6) LRU-on-miss로 INT4 pool 재구성. (7) probabilistic-RT framing (p99.9 bound, hard-RT 주장 회피).
- **예상 개선**: TPOT p50 0.75-0.85배 (HOBBIT 기준), **p99 0.65-0.75배** (primary), throughput vanilla 대비 2.0-2.5배 / HybriMoE 대비 1.2-1.4배. HBM 45-55%. 정확도 −0.3 ~ −0.8%. WCRT bound 하에서 p99.9 period-miss rate <0.1%. miss 15% 예상 수준은 대응 가능, 30% 스트레스도 safety valve로 bounded 유지.
- **적용 범위**: Qwen3-Next-A3B / Qwen3-30B-A3B / DeepSeek-V2-Lite. **미적용**: Mixtral-8 (fingerprint SNR 부족, signature-entropy 이론과 일관), prefill-heavy 워크로드 (ACE-MoE가 지배). **불확실**: OOD prompt class drift.
- **실험**: 서버 #5 RTX Pro 6000 (또는 #4 4090×2) / vLLM 0.9 + custom prefetcher / LMSYS + ShareGPT + MMLU + GSM8K / 5-7일.
- **차별점**: **BuddyMoE ([2511.10054](https://arxiv.org/abs/2511.10054), substitution)와의 차이는 INT4 fallback이 identity를 유지하면서 WCRT bound를 추가한 점**. PreScope ([2509.23638](https://arxiv.org/abs/2509.23638), async prefetch)는 bound 없음. HOBBIT은 miss를 정밀도 강등으로 처리, ZMSP는 bound가 있는 스케줄링 이벤트로 처리. 80μs MLP를 써서 SP-MoE/MoE-SpeQ의 draft 모델(50-300ms) 대비 predictor overhead가 현저히 작음. ACE-MoE와 composable (backward score + forward fingerprint).
- **상세**: 세션 4.3절

---

### ACE-VLA: Real-Time Action Decoding under Latency Budget for Vision-Language-Action Models
- **Date**: 2026-04-21 | **Mode**: 2 | **Session**: 링크
- **Experts**: ai-optimization-expert (main), algorithm-expert (sub)
- **Review Scores** (refined): Nov 9/10, Diff 9/10, Imp 9/10, 평균 **9.0** (Top 1)
- **Review Scores** (post-2026-04-21 related work augmentation): Nov 8/10, Diff 9/10, Imp 9/10, 평균 **8.5** — ActionFlow(2025-12, 2.55× FPS)/A1(2026-04, 72% latency)/HEX(2026-04, MoE VLA 선례) 등장으로 novelty 소폭 하락. Expert-level + hard real-time + visual freshness의 조합은 여전히 unique
- **Additional augmentation (AdaMoE-VLA, [arXiv:2510.14300](https://arxiv.org/abs/2510.14300), 2025-10)**: 가장 직접적 VLA-MoE 선례 발견 — dense VLA FFN을 sparse MoE로 대체, Scale Adapter로 expert selection/weighting decouple. **Training-time 아키텍처** 혁신이라 ACE-VLA(inference-time)와 **strictly complementary**. Base model 선택을 OpenVLA retrofit 대신 **AdaMoE-VLA 재현(Pi-0 base) → ACE-VLA 적용**으로 재설계. 총 5-7주 구현 roadmap 확보. 실제 공개된 VLA+MoE 연구는 극히 드물어 (AdaMoE-VLA / HEX / HY-Embodied 정도) blue-ocean 유지.
- **핵심 가설**: VLA 모델은 robotics control loop의 hard real-time(30-100Hz) constraint 가짐. ACE-MoE의 time-budgeted skipping을 action chunk별 hard latency budget(예: 33ms)으로 재정의하고 visual context와 action prediction expert에 modality-aware ACE 적용 시, action latency 30-50% 단축하면서 task success rate 95%+ 유지 가능.
- **접근법**: (1) Hard budget = 1/control_freq 명시. (2) Action chunk별 ACE update. (3) Visual freshness scoring (optical flow 기반). (4) Safety-aware skip whitelist (collision avoidance 등). (5) Streaming visual prefill (incremental ViT). (6) VLA-MoE retrofit recipe (OpenVLA FFN을 expert로 분해). (7) Schedulability analysis (WCET model로 probabilistic real-time guarantee).
- **예상 개선**: per-action latency 50ms→25-35ms (RTX 4090, 7B class), task success rate -1~-3%, throughput 1.4-2.0×, memory 50-65%
- **적용 범위**: VLA 모델 (OpenVLA, RT-2-X, Octo, Pi-0), MoE 변형 가용 시. manipulation/navigation. **미적용**: pure simulation only, language-only robots. **불확실**: real robot eval은 협력 lab 필요
- **실험**: #1 또는 #3 server / OpenVLA-7B + custom inference / SimplerEnv (Bridge/RoboCasa), LIBERO
- **차별점**: VLA + expert offloading + ACE은 거의 미연구 영역. Quar-VLA(양자화만)와 명확히 차별. Robotics + LLM serving 두 community에 영향 가능.
- **상세**: 세션 Phase 2/4의 I5 섹션

---

### Joint Token-Expert Budget Allocator: Cross-Modal Resource Scheduling for VLM Inference
- **Date**: 2026-04-21 | **Mode**: 2 | **Session**: 링크
- **Experts**: ai-optimization-expert (main), algorithm-expert (sub)
- **Review Scores** (refined): Nov 9/10, Diff 8/10, Imp 9/10, 평균 **8.7** (Top 2)
- **Review Scores** (post-2026-04-21 related work augmentation): Nov 7.5/10, Diff 8/10, Imp 8.5/10, 평균 **8.0** — DyMoE(2026-03, depth-adaptive + importance + mixed-precision + prefetch)와 Dynamic Expert Quantization(2025-11, budget-constrained optimization formulation)이 가까운 선례. I3의 차별 포인트는 **token axis + visual KV asymmetry 활용**으로 축소 재정의 필요
- **핵심 가설**: VLM-MoE에서 visual token skip과 expert skip은 공유된 PCIe/HBM bandwidth budget을 두고 경쟁한다. 두 axis를 joint하게 layer-wise budget allocation으로 풀면 단일 axis 최적화 대비 latency 추가 20-30% 단축 + iso-accuracy Pareto 15-25% gain.
- **접근법**: (1) Layer별 (token attention compute + expert transfer) 두 cost 모델링, 총 layer budget T_layer. (2) Per-layer optimizer (closed-form 또는 DP)로 token retention r_t와 expert prefetch r_e 결정. (3) Modality-conditional (dense layer L17-21 token 우선, sparse layer L0-7 expert 우선). (4) Online adaptation (첫 N requests로 cost 재추정, EWMA). (5) 차등 quantization (sparse layer KV INT8, dense FP16). (6) vLLM의 first-class scheduler API design.
- **예상 개선**: TTFT (FHD VLM-MoE) 800ms → 500-650ms (BS=4-8), decode throughput 600 → 700-800 tok/s, accuracy -0.5~-1.0%, Pareto gain 15-25%
- **적용 범위**: VLM-MoE + dense VLM 모두에서 ACE-MoE-style 동시 적용 시나리오. **미적용**: 단일 modality (LLM only - 불필요), VLA closed-loop (별도 budget, I5와 통합 가능). **불확실**: optimizer가 architecture별 generalize 여부
- **실험**: #5 / vLLM + custom budget scheduler / Qwen3-VL-4B/8B, LLaVA-Mixtral / MMMU, ChartQA, MS-COCO
- **차별점**: Hobbit (mixed precision expert axis만), FlexGen (coarse-grained), HybriMoE (compute placement만) — cross-axis budget formulation이 unique. SARATHI chunked prefill과 orthogonal하게 결합 가능. vLLM PR 수준의 system contribution 가능.
- **상세**: 세션 Phase 2/4의 I3 섹션

---

### Modality-Aware ACE for VLM-MoE (with Hierarchical Injection Layer Boundary)
- **Date**: 2026-04-21 | **Mode**: 2 | **Session**: 링크
- **Experts**: ai-optimization-expert (main), algorithm-expert (sub)
- **Review Scores** (refined): Nov 8/10, Diff 8/10, Imp 8/10, 평균 **8.0** (Top 3)
- **Review Scores** (post-2026-04-21 related work augmentation): Nov 7/10, Diff 7.5/10, Imp 8/10, 평균 **7.5** — AlignMamba-2(2026-03, modality-specific + modality-shared experts)가 I1의 철학을 이미 제시, ERNIE 5.0(2026-02, modality-agnostic routing)은 counter-design. I1은 ACE caching 측면의 새로움 + HIL boundary handling으로 positioning 필요
- **핵심 가설**: ACE-MoE의 cumulative expert score를 modality dimension(visual/text)으로 분할하여 두 stream으로 유지하고, modality-conditional variance normalization을 적용하면, VLM-MoE 추론에서 동일 accuracy 달성 cache ratio를 50%→25-35%로 절반 감소 가능. DeepStack 같은 hierarchical injection layer는 boundary group으로 별도 처리.
- **접근법**: (1) Token마다 modality tag 부여, DeepStack injection 이후 token은 visual mass에 weight fading 포함. (2) Layer-wise expert score를 visual/text 두 stream으로 누적. (3) Modality-conditional variance normalization. (4) Global selection: weighted sum α (entropy-based dynamic, token modality entropy 기반). (5) ACE 슬롯 일부(30%) visual-critical reserve. (6) Modality-aware time-budgeted prefetch. (7) Hierarchical Injection Layer (HIL) boundary handling — inject layer ±2는 별도 ACE pool, inject layer 자체는 ACE-MoE layer 0과 동일 quantized GPU resident.
- **예상 개선**: cache ratio 50%→25-35% (iso-accuracy), prefill latency 0.85-0.95× ACE-MoE single, memory footprint 60-75%, accuracy -0.5~-1.0% at 50% cache
- **적용 범위**: VLM-MoE 모델 (Qwen3-VL-MoE 출시 시 / MoE-LLaVA / Uni-MoE / DeepSeek-VL2 / LLaVA-Mixtral), batch ≥ 4. **미적용**: 단일 modality LLM, dense VLM (I2 적용). **불확실**: VLM-MoE 공개 모델 부족 → MoE-LLaVA 같은 기존 모델로 우선 검증 후 future-ready로 positioning
- **실험**: #5 / vLLM + ACE-MoE patch + modality tag / MoE-LLaVA-7B, Uni-MoE-8B, LLaVA-Mixtral / MMMU, ScienceQA, VQAv2, ChartQA
- **차별점**: Edge-MoE(task-level sparsity), Cache-Conditional MoE(modality 미구분), MoE-LLaVA(serving 최적화 미제시) — 본 idea는 token-modality level expert importance + 2D variance-aware. 직계 ACE-MoE 후속이며 modality-aware MoE serving은 미답.
- **상세**: 세션 Phase 2/4의 I1+I4 섹션

---

## 미선정 로그

### I3' PRISM-VLA-Temporal — 미선정 (Major Revision → Deferred, DyQ-VLA/SD-VLA 65%+ scoop)
- **Date**: 2026-04-23 | **Mode**: 1 | **Session**: 링크 / Summary
- **Score avg**: Phase 2 6.0 → Phase 2' (refined) 6.45. Novelty 5.3→6.0, Impact 7.4, Algo peer 7.0, AI-opt 5.0, HW 5.5.
- **Phase 2' 판정**: Major Revision → **Deferred** (Phase 1'' A/B empirical test 결과 조건부 재진입).
- **연구 GAP**: VLA action sequence 의 temporal KV SF delta decomposition — VLA-Cache ([arXiv:2502.02175](https://arxiv.org/abs/2502.02175)) 는 caching only + FP16 KV 유지, quantization 축 공란.
- **제안 overview**: Continuous rank-r delta SF (S_t = S_0 + Σ α_k(t)·S1_k·S2_k, α_k ∈ [0,1]) + action-head-aware OPTIC (contact/navigation 2-tier, gripper + force threshold 0.5N) + gated freeze (δ threshold, rank-2→rank-1 fallback). BitVLA-2B + OpenVLA-7B 타겟, LIBERO-Long ≥50% success, control freq +25-30%, KV memory 7-10% of FP16.
- **유사 연구 대응**: [DyQ-VLA arXiv:2603.07904](https://arxiv.org/abs/2603.07904) (2026-03) **65%+ 일치** — temporal-dynamic-aware VLA quantization + kinematic proxy bit-switch. [SD-VLA arXiv:2602.03983](https://arxiv.org/abs/2602.03983) (2026-02) **55% 일치** — static/dynamic binary disentanglement + recache gate + LIBERO-Memory benchmark. [QVLA arXiv:2602.03782](https://arxiv.org/abs/2602.03782) (ICLR'26) **60%** — action-space sensitivity channel bit allocation. [VLA-Cache [NeurIPS'25]](https://arxiv.org/abs/2502.02175) (45%, caching only), [KV-Efficient VLA arXiv:2509.21354](https://arxiv.org/abs/2509.21354) (40%), [EaqVLA arXiv:2505.21567](https://arxiv.org/abs/2505.21567) (module mixed-precision).
- **미선정 사유**: (1) DyQ-VLA/SD-VLA 가 "temporal dynamic + VLA quantization" 축을 binary 방식으로 선점. (2) I3' 의 **continuous rank-r delta 이득 분리 증명**이 Phase 1'' A/B empirical test 전에는 확정 불가. (3) 전문가 평균 6.45 는 Tier-1 선정 기준 이하.
- **재방문 조건**: (a) LIBERO-Long 에서 continuous rank-r vs binary-{SD-VLA style} A/B test → success rate gap >0.5% 확인 시 Tier-1 재진입 (CoRL 2026 workshop / ICRA 2027). (b) 또는 Tier-2 downgrade 경로: I3-Tier2 continuous rank-r delta measurement letter, BitVLA-2B + LIBERO-Spatial only, ICCAD/DATE 8p. (c) DyQ-VLA/SD-VLA follow-up 6개월 모니터링 후 novelty gap 재확인.
- **상세**: 세션 Section 4.4.1 (미선정 처리) / Summary § 4.1

### T3 LightTri-LUT — 미선정 (I4 Self-scoop, 6개월 gap 필요)
- **Date**: 2026-04-23 | **Mode**: 1 | **Session**: 링크 / Summary
- **Score avg (Tier-2 rubric)**: Nov 4.3 / Impact 6.7 / peers 5.5. 평균 5.5 / **6.7 (Tier-2 rubric Conditional Accept)**.
- **Phase 2' 판정**: Tier-2 Conditional Accept. **미선정 사유**: I4' PRISMKV-PIM-DequantLUT 와 핵심 mechanism "SF-LUT precomputation" 중복 (device RRAM vs DRAM 차이만). 연속 publication 시 reviewer 가 "same author, incremental" 로 판정 가능성. Tier-2 rubric 6.7 은 borderline.
- **연구 GAP**: Single RRAM tile (128×256, 28nm PTM) 에서 BitVLA-2B attention+action-head 의 PRISM LUT 효율 measurement letter 부재.
- **제안 overview**: RRAM tile 단위 SF-LUT + BitVLA-2B subset profiling. TOPS/W + LUT hit rate measurement.
- **재방문 조건**: (a) I4' (HPCA 2027) submission 후 6개월 이후. (b) RRAM-specific novel contribution 추가 (endurance 측정, retention drift, read disturb). (c) RRAM-exclusive workload (edge VLA inference on resistive memory) 로 포지셔닝.
- **상세**: Summary § 4.2

### L2 v2 TemporalTier-3: Action-Imminence-Driven Hierarchical KV (VLA + Streaming VLM) — 미선정 (Tiebreak 패배)
- **Date**: 2026-04-22 | **Mode**: 1 | **Session**: 링크
- **Score avg** (post-verification): Phase 2 6.5 → Phase 2' **7.08** (+0.58). Nov 7.5 / Diff 7.6 / Imp 6.2 / Feas 7.0. 전문가 합의 2.5:0.5. Verification 후 변동 없음.
- **Phase 2' 판정**: Conditional Accept. **미선정 사유**: A1 v2 PhaseGraph-VLA 와 평균 **동점 7.08**, **impact × feasibility product tiebreak 에서 패배 (L2: 43.40 < A1: 48.75)**.
- **연구 GAP**: VLA action stream / VLM video streaming 에서 "다음 수백 ms 내 어떤 KV 페이지가 hot 인지" 를 task-specific signal 로 예측. 기존 VLA-Cache / KV-Efficient VLA / HERMES ([arXiv:2601.14724](https://arxiv.org/abs/2601.14724)) 은 single-tier 또는 generic hierarchical.
- **제안 overview**: 초안 4-tier (HBM top/bottom/UVM/pinned/disk) → Phase 1' 에서 **3-tier (HBM-hot/HBM-cold/pinned host), UVM 제거**. VLA **action imminence β_a** (gripper Δ + trajectory curvature + object distance) / VLM **interaction trigger β_v** (scene cut + audio + Hawkes arrival). FA-3 indirection-pointer fork (wrapper vs kernel-rewrite 2-variant). Grace-Hopper NVLink-C2C analytical variant. Hawkes vs Poisson ablation. Gripper predictor AUC study (VLA-Cache frame-diff 대비 +0.08 목표).
- **유사 연구 대응**: HERMES [arXiv:2601.14724](https://arxiv.org/abs/2601.14724) (65% concurrent, 2026-01 hierarchical KV for streaming video). VLA-Cache (45-55%), KV-Efficient VLA (40-50%), MadaKV [arXiv:2506.15724](https://arxiv.org/abs/2506.15724) (55%), PRESERVE [arXiv:2501.08192](https://arxiv.org/abs/2501.08192) (55%), KVSwap [arXiv:2511.11907](https://arxiv.org/abs/2511.11907) (52%), TTF-VLA [arXiv:2508.19257](https://arxiv.org/abs/2508.19257), Lethe [arXiv:2511.06029](https://arxiv.org/abs/2511.06029), Event-VStream [arXiv:2601.15655](https://arxiv.org/abs/2601.15655). **Post-verification 추가**: [arXiv:2603.14371](https://arxiv.org/abs/2603.14371) OxyGen (2026-03, VLA unified KV management under multi-task parallelism, 30% adjacent — cross-task sharing axis), [arXiv:2502.04077](https://arxiv.org/abs/2502.04077) AttentionPredictor (Temporal Patterns Matter for KV Cache Compression, temporal 축 adjacent). "TempoKV" / "EventPrefetch-Robot" placeholder 실존 부재 확인 → HERMES 65% concurrent 유일한 major threat.
- **개선 가능성 / 재방문 조건**: (a) Hawkes vs Poisson ablation 이 bursty workload (tool-use dialogue) 에서 유의미하면 reposition. (b) Gripper predictor empirical AUC +0.08 이상 증명. (c) 70B+ VLA 등장 (scope 확장) 시. (d) Grace Hopper NVLink-C2C 실기 접근 시. 다음 세션에서 single-PoC (gripper signal AUC study 1주) 후 재평가 가능.

### A2 v2 TierKernel-Dispatch: Three-Tier Patch Routing with Warp-Specialized Kernel — 미선정 (Top 3 컷오프)
- **Date**: 2026-04-22 | **Mode**: 1 | **Session**: 링크
- **Score avg** (post-verification): Phase 2 5.67 → Phase 2' **6.88** (+1.18). Nov 7.0 / Diff 7.5 / Imp 6.5 / Feas 6.5. 전문가 합의 **2:1** (algorithm-expert **No** — intent classifier cross-task generalization 증명 약함). Verification 후 변동 없음.
- **Phase 2' 판정**: Conditional Accept. **미선정 사유**: (1) 평균 6.88 은 Top 3 컷오프 하회 (L1 7.00 > A3 7.15 > A1 7.08, post-verification). (2) algorithm-expert 반대표. (3) OmniSparse [arXiv:2511.12201](https://arxiv.org/abs/2511.12201) 의 binary hot-cold 대비 3-tier incremental contribution 이 challenge. (4) Blackwell L2 spec 변경 risk.
- **연구 GAP**: Task intent × kernel variant × memory tier 3축 통합 공백.
- **제안 overview**: E²IC (P1 흡수) 10-dim activation probe 또는 DistilBERT-tiny → intent + attention entropy 이중 gating → Hot/Warm/Cold 3-tier patch → per-tile kernel variant (CUTLASS warp-specialized Hot producer-consumer 4+8 warp / vanilla FA Warm / eviction Cold) + L2 cudaAccessPolicyWindow (Blackwell 288MB) + Hopper H100 128MB L2 fallback variant. DocVQA TTFT **-23%**, OCRBench 유지, MMStar 1.24×.
- **유사 연구 대응**: OmniSparse (62-68% concurrent), MMInference [arXiv:2504.16083](https://arxiv.org/abs/2504.16083) (62% concurrent), SparseVLM [arXiv:2410.04417](https://arxiv.org/abs/2410.04417) (55%), VLCache [arXiv:2512.12977](https://arxiv.org/abs/2512.12977) (adjacent), FlashInfer MLSys 2025 (50%), PAT [arXiv:2511.22333](https://arxiv.org/abs/2511.22333), PureKV [arXiv:2510.25600](https://arxiv.org/abs/2510.25600), VL-Cache (45%), SparseVILA [arXiv:2510.17777](https://arxiv.org/abs/2510.17777) (40%). **Post-verification 추가**: [arXiv:2507.09071](https://arxiv.org/abs/2507.09071) BlindSight (input-template-aware sparsity, adjacent), [arXiv:2510.14719](https://arxiv.org/abs/2510.14719) Tawa (warp specialization, adjacent), [arXiv:2510.00536](https://arxiv.org/abs/2510.00536) GUI-KV (secondary adjacent — GUI-agent 특화지만 tile-intent hint 공유). "TileSparse" / "L2-Persist-Attn" placeholder 실존 부재 확인.
- **개선 가능성 / 재방문 조건**: (a) Intent classifier cross-task generalization eval (최소 3 VLM benchmark) 결과 확보 시. (b) BlindSight / Tawa 와의 명확한 차별화 formal. (c) B200 이후 L2 spec 확정 후 재검증.

### L3 v2 MTV-Pool: Multi-Turn-Aware Visual KV Pool — 미선정 (Major Revision)
- **Date**: 2026-04-22 | **Mode**: 1 | **Session**: 링크
- **Score avg** (post-verification): Phase 2 6.00 → Phase 2' **6.68** → **6.50 (verified, 강화)**. Nov 5.9 (↓ from 6.3 after GUI-KV scoop 실존 확인) / Diff 7.2 / Imp 6.0 / Feas 7.2. 전문가 합의 **1.5:1.5 분열** (legacy-sys **No** — pool 단일 도입 MICRO scope 엔 얕음, ai-opt Conditional — γ_v weight overfitting 우려).
- **Phase 2' 판정**: **Major Revision (근거 강화)**. **미선정 사유 (검증 반영)**: placeholder "GUIAgent-KV 65%" 의 실제 논문 = [arXiv:2510.00536](https://arxiv.org/abs/2510.00536) **GUI-KV: Efficient GUI Agents via KV Cache with Spatio-Temporal Awareness** (2025-10-01, Kung-Hsiang Huang, Haoyi Qiu, Yutong Dai) 실존 확인. GUI-KV 메커니즘 = (a) residual stream L2 norm spatial saliency + (b) previous frames' keys → current frame key subspace projection 으로 redundant history 제거. AgentNetBench 5-screenshot: FLOPs -38.9%, accuracy +4.1%. 메커니즘 1:1 일치는 아니나 target scenario (multi-turn GUI agent screenshot history KV 압축) + positioning 완전 중첩 → contribution novelty claim 잠식. 전문가 합의 분열 유지.
- **연구 GAP**: Screenshot-history 기반 GUI agent (Claude Computer Use, VisualWebArena, OSWorld) 의 turn 별 inter-turn visual coldness — MMInference static permutation / OmniSparse intra-turn 미커버.
- **제안 overview**: 초안 3-pool (visual-hot / text-hot / visual-cold) → Phase 1' 에서 **2-pool (current-turn-hot / past-turn-cold)** 축소. γ_v = w1·turn_oldness + w2·log-sum-exp(past attn) + w3·tile_entropy. Multi-turn GUI agent (VisualWebArena, OSWorld) workload. Reorg stream overlap > 80% 정량화. Text token always-hot. MMInference (static) vs MTV-Pool (dynamic per-turn) formal 차별.
- **유사 연구 대응**: OmniSparse (65-75% scoop), VL-Cache (60-70% scoop), DiffKV [arXiv:2412.03131](https://arxiv.org/abs/2412.03131) (55-65% scoop), MMInference (68% concurrent), MadaKV (58%), FlowMM [arXiv:2511.05534](https://arxiv.org/abs/2511.05534) (55%), SparseVLM (45% adjacent), SparseVILA (42%). **Post-verification 추가**: **[arXiv:2510.00536](https://arxiv.org/abs/2510.00536) GUI-KV (55-65% scoop, 실존 확인 최우선 경쟁)**, [arXiv:2603.26041](https://arxiv.org/abs/2603.26041) Rethinking Token Pruning for Historical Screenshots in GUI Visual Agents (2026-03-27 제출, 2026-03-31 withdrawn — temporal recency-budget 은 turn_oldness 와 정면 유사하나 withdrawn 상태), [arXiv:2509.17396](https://arxiv.org/abs/2509.17396) EpiCache (episodic KV), [arXiv:2510.09038](https://arxiv.org/abs/2510.09038) Auto-Scaling Continuous Memory for GUI Agent, [arXiv:2511.02230](https://arxiv.org/abs/2511.02230) Continuum, [arXiv:2603.20284](https://arxiv.org/abs/2603.20284) STAC. "ScreenHistory-Cache" placeholder 실존 부재 확인.
- **개선 가능성 / 재방문 조건 (검증 후 재정의)**: (1) GUI-KV 실존 확인됨 — scoop 위험 확정. **GUI-KV 의 key-subspace projection 과 orthogonal 한 축** (예: pool reorg 의 SM overlap 정량화, INT4 dequant + log-sum-exp kernel 이득) 을 primary contribution 으로 재설계. (2) Rethinking Token Pruning (2603.26041) 재제출 시 재평가. (3) Claude Computer Use 내부 trace 접근. (4) learned weight γ_v online learn prototype. (5) Venue 재타겟 (MICRO → EuroSys / SoCC system-agent track).

### F3-VLM VLM-MOESI Image-Hash Coherence (GPU-HBM ↔ PIM-DRAM) — 미선정 (Major Revision 권고)
- **Score avg**: Phase 1' 7.67 → Phase 2' **6.17** (VLCache scoop 로 Novelty 8→6)
- **미선정 사유**: VLCache ([arXiv:2512.12977](https://arxiv.org/abs/2512.12977), 2025.12) 가 image-hash 기반 encoder/KV cache 재사용을 이미 SGLang 에 구현 + 1.2-16× TTFT speedup 실증. F3 의 핵심 claim (shared-image TTFT 1.8-2.5×) 을 ~70-75% 선점. Phase 1' 에서는 F3 가 단독 Novelty 8 로 유지됐으나 Phase 2' 의 6-month fresh similarity search 에서 발견된 VLCache 는 2주 전 공개 논문으로 prior session 에서도 반영되지 않은 신정보. Image-hash KV reuse axis 는 더 이상 safe harbor 아님.
- **개선 가능성 / repositioning 옵션**: (a) Narrative 를 "KV 재사용" 대신 **"GPU+PIM 이종 메모리 MOESI coherence 프로토콜"** 로 전면 피벗. 수치 claim 을 TTFT speedup 이 아닌 **coherence traffic 감소율 / invalidation cost / write-back bandwidth saving** 으로 재정의. VLCache 가 다루지 않는 **Modified/Owned state transition** (PIM bank pinned KV 의 invalidation 비용, cross-tenant image update 시 coherence message) 을 주 기여로 제시. TraCT ([arXiv:2512.18194](https://arxiv.org/abs/2512.18194)) 의 "cache-line coherence impractical at multi-TB CXL" 반박에 image-identity granularity 응답. (b) 또는 F1 의 cross-request visual KV sharing sub-module 로 흡수 (독립 논문 포기). 예상 추가 2-4주 ideation 필요.
- **재방문 트리거**: (1) VLCache 논문 정독 후 MOESI 축 narrative draft 완성, (2) LLaVA-Bench-Wilder real batch trace duplication rate ≥20% 확인, (3) AttAcc simulator 에 coherence protocol extension 가능성 검토.
- **Session**: [링크](/research-wiki/2026-04/vlm-pim-extension) — F3 (Phase 2' Major Revision)

### H2-VLM Hierarchical 3-Tier KV (GPU HBM / PIM bank / CXL-SSD) — DROP (SCOOP)
- **Score avg**: Phase 2 Novelty 4/10, Differentiation 4/10 (단독 통과 불가)
- **미선정 사유**: **PAM ([arXiv:2602.11521](https://arxiv.org/abs/2602.11521), 2026.02)** 이 HBM-PIM + DRAM-PIM + SSD-PIM 3-tier + context locality migration + PAMattention 알고리즘 모두 선점 (~75-80% 메커니즘 일치). FlexGen ([arXiv:2303.06865](https://arxiv.org/abs/2303.06865)) 이 이미 3-tier, InfiniGen ([arXiv:2406.19707](https://arxiv.org/abs/2406.19707)) 이 length-adaptive KV, LoL-PIM ([arXiv:2412.20166](https://arxiv.org/abs/2412.20166)) 이 long-context PIM hierarchy — 전체적으로 2024-2025 유행 주제. VLM-specific 특성만으로는 얇음.
- **개선 가능성 / repositioning**: "video long-context (L>8K) 구간에서 VLM-specific prefetch scheduler" 로 narrow 하거나, DeepStack-aware tier policy (visual token hot tier 우선, injection 시점 기반 promotion) 을 결합해 **F1-VLM 의 sub-module 로 흡수**. 독립 논문은 PAM 이후 불가.
- **Session**: [링크](/research-wiki/2026-04/vlm-pim-extension) — H2 (Phase 2 DROP)

### L1-VLM SLO Admission Control & Dual-Pool Batching — DROP (SCOOP)
- **Score avg**: Phase 2 Novelty 7/10, Differentiation 6/10, Impact 7/10 → Similarity SCOOP 로 통과 불가
- **미선정 사유**: 3편이 80%+ 선점 — **ModServe ([arXiv:2502.00937](https://arxiv.org/abs/2502.00937), 2025.02)** modality-aware disaggregation + image-text vs text-only routing, **RPS-Serve ([arXiv:2603.26498](https://arxiv.org/abs/2603.26498), 2026.03)** "Rocks/Pebbles/Sand" video/image/text 3-tier scheduling + MMMU TTFT tail 해결, **Dual-Pool Token-Budget Routing ([arXiv:2604.08075](https://arxiv.org/abs/2604.08075), 2026.04)** 용어 · 구조 직접 충돌. ElasticMM ([arXiv:2507.10069](https://arxiv.org/abs/2507.10069)), PolyServe ([arXiv:2507.17769](https://arxiv.org/abs/2507.17769)), SLOs-Serve ([arXiv:2504.08784](https://arxiv.org/abs/2504.08784)) 추가. 2025-2026 상반기 VLM serving scheduling 은 레드오션.
- **개선 가능성 / repositioning**: "PIM bank contention-aware admission" 으로 narrow 하여 **F1-VLM dispatcher 의 component** 로 흡수. 단독 system 논문은 생존 불가.
- **Session**: [링크](/research-wiki/2026-04/vlm-pim-extension) — L1 (Phase 2 DROP)

### Visual Token ACT — Sequence-Level Pruning via Cumulative Attention Score
- **Score avg**: 7.0 (refined, original 6.0)
- **미선정 사유**: KV compression 분야(VL-Cache, PyramidKV, SnapKV, H2O, Quest, Look-M 등)가 매우 crowded. ACE 원칙의 KV transfer는 직관적이지만 단독 paper로는 differentiation 약함.
- **개선 가능성**: I3(Joint Budget) 또는 I1(Modality-Aware ACE)의 sub-component로 흡수 시 가치 있음. 또는 sequence-level pruning을 prefill phase의 attention compute 자체에서 skip하는 방향으로 강화하면 standalone 부활 가능.
- **Session**: 링크 — Phase 2 I2

### Hierarchical Injection-Aware Routing (DeepStack as Instance) — 흡수
- **Score avg**: 7.0 (standalone)
- **미선정 사유**: DeepStack-specific 한정 시 적용 범위 좁음. Hierarchical Injection Layer (HIL) framework로 일반화한 후 I1에 흡수.
- **개선 가능성**: I1의 boundary handling sub-component로 통합 완료 (HIL handling).
- **Session**: 링크 — Phase 2 I4

### Image-Level Cross-Request Cache (KV + Expert Profile)
- **Score avg**: 7.0
- **미선정 사유**: Novelty incremental (vLLM prefix caching의 visual extension). System contribution은 가치 있으나 standalone academic paper로는 약함.
- **개선 가능성**: vLLM/SGLang에 PR 형태로 직접 contribute. Production workload reuse rate 통계가 강력하면 standalone 가능 (e-commerce/document QA 시나리오 강화).
- **Session**: 링크 — Phase 2 I6

### FF-ACE-VLA: Fingerprint-Forward ACE Eviction for VLA Real-Time Serving — 미선정 (deferred)
- **Score avg**: 5.83 (post-refinement 6.2)
- **미선정 사유**: (1) VLA-MoE 시장 성숙도 의존 — AdaMoE-VLA(2025-10), HiMoE-VLA(2025-12) 모두 최근; 실제 robot deployment은 2027+ bet. (2) 직계 ACE-MoE 연장선 → reviewer들이 "incremental over team's own prior work"로 볼 위험. (3) Cross-idea overlap 문제 — I3 ZMSP + I4 FARD-C + I5 FF-ACE-VLA 3편이 같은 fingerprint predictor를 공유해 3-paper split 시 artificial-split rejection 위험. I4+I3 2편 먼저 검증 후 진행이 합리적.
- **개선 가능성**: Phase-conditional schedulability analyzer만 분리 → RTAS/CoRL 워크숍 페이퍼로 publishable. AdaMoE-VLA/HiMoE-VLA production 신호 관찰 시 재방문. 실제 robot evaluation partner 확보 시 priority 상승.
- **재방문 트리거**: (1) VLA-MoE production deployment 신호, (2) ICCAD/MICRO ACE-MoE 결과 발표 후, (3) robot lab partner 확보.
- **Session**: 링크 — 미선정 I5

### ExpertEcho: Cross-Tenant Topic Recovery via Hardware Side-Channel — DROP
- **Score avg**: 4.0
- **미선정 사유**: **MoEcho ([arXiv:2508.15036](https://arxiv.org/abs/2508.15036), CCS 2025)에 의해 scooped**. 99.8% Prompt Inference + 92.8% response reconstruction, GPU cache occupancy + TLB + perf counter 공격 모두 DeepSeek-V2 / Qwen1.5-MoE / TinyMixtral에서 이미 달성. 공영호 lab에 hardware side-channel 실적 부재로 S&P/USENIX Security 수용 위험도 높음 (2-3개월 infra 선행).
- **흡수**: 유일한 novel angle인 **scaling-leakage law (privacy ↓ as N_experts ↑)**를 **I2 PhantomRoute의 motivation section (1-2 pages)으로 흡수**. 별도 paper 불필요. Security 외부 랩과 coauthor 시에만 standalone 부활 고려.
- **Session**: 링크 — DROP I1

### SpikeXbar + SpikeRoute-Xbar: Event-Driven BNN-SNN on Crossbar (SNN track) — 미선정 (deferred)
- **Score avg**: I-A3 5.3 + I-H2 6.7 (fused ~6.5)
- **미선정 사유**: (1) **Concurrent work 3-4편 집중 등장 (2025-11)**: SOT-MRAM Event-Driven Spiking CIM ([arXiv:2511.03203](https://arxiv.org/abs/2511.03203)) 이 이미 243.6 TOPS/W 달성 — HW 축에서 scoop 위험 ~55%. CADC ([arXiv:2511.22166](https://arxiv.org/abs/2511.22166)) 의 80-88% psum sparsity 가 output sparsity 축 점유. SpikeFit ([arXiv:2604.14487](https://arxiv.org/abs/2604.14487), 2026-04) 이 SNN codebook 양자화 선점. ASTER ([arXiv:2511.06770](https://arxiv.org/abs/2511.06770)) 467배 가속 SNN transformer. (2) **공영호 lab 의 SNN surrogate gradient + BPTT 훈련 실적 부재** — researcher-fit 약함. (3) 9일 budget 빡빡 (BPTT × T=8 은 ANN 대비 4-8배 훈련 시간). (4) Algorithm expert 상호 리뷰: "zero-spike LUT collapse 단독은 에너지 10-15% 수준 개선 → I-H2 와 병합 필수". HW expert 상호 리뷰: "실제 energy save 는 peripheral(ADC/S/H) 뿐, 주장 5-10× 는 과대."
- **개선 가능성 / repositioning 옵션**:
  - **A. N_active-binned SF 단일 기여로 workshop 축소**: event-driven 이 아닌 dense SNN 에서도 active neuron count bin 별 SF — concurrent work 없는 유일한 각도. √N_active scaling law 로 formalize 후 AICAS/ICONS workshop.
  - **B. CADC 저자와 composition**: CADC output sparsity + 우리 input sparsity dual-sparsity synergy 분석 — 저자 조율 필요.
  - **C. Lab SNN infrastructure 구축 후 2027 H1 재방문**: surrogate gradient + BPTT 파이프라인 정비 + 실제 robot lab partner 확보 시 재시도.
- **재방문 트리거**: (1) N_active-binned SF 의 standalone workshop 성사, (2) CADC 혹은 SOT-MRAM 저자와의 joint work 가능성, (3) lab 의 SNN 훈련 infra 정비 완료, (4) 2027 ISSCC 제출 deadline (2026-09) 가 뜨거운 시점이면 급행 처리 고려.
- **Session**: 링크 — 미선정 SNN track (I-A3 + I-H2)
