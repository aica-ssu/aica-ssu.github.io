# 2026-05-02 — VLM Scenario-Aware Optimization

> **Mode 1 (sentence-input)** · 시나리오별 사전 분류 후 최적화 + multi-turn vision KV reuse + 기존 benchmark 활용 실험 플랜

본 세션은 사용자 input — "VLM 추론은 시나리오에 따라 크게 다르므로 사전 구분 후 최적화, 특히 video 기반 multi-turn QA visual info + accuracy 보존, 기존 benchmark 활용 실험 플랜" — 을 R31 검증 통과 후 21 candidate × 5-reviewer dispatch → Top-M 6 으로 finalize.

**핵심 trigger**: production VLM (Qwen3.5/mlx-vlm Issue #832) 에서 multi-turn 마다 vision tower re-run + full re-prefill — vLLM Prefix Caching / SGLang RadixAttention 이 LLM token prefix 만 cover, **vision token frame-level KV reuse 는 production gap**. 측정 prefix hit rate 60-85% (agent loop / multi-tenant SaaS / repo QA / long-doc) 가 multi-turn 이 valid common case 임을 입증.

---

## 1. Research Questions

### Master RQ

- **MQ1**: VLM serving 의 scenario diversity (image/video × single/multi-turn × document/agent) 에서 단일 generic optimizer 보다 scenario-aware dispatcher 가 production prefix hit rate 60-85% 환경에서 추가 30%+ TTFT 감소 가능한가?
- **MQ2**: Multi-turn video QA 의 frame-level radix tree vision KV cache 가 VLCache ([arXiv:2512.12977](https://arxiv.org/abs/2512.12977)) 의 single-session image-patch reuse 와 직교 axis 로 cross-session frame prefix hit 50%+ 달성 가능한가?
- **MQ3**: Multi-image agent loop 의 cross-image vision token pool 이 PolyKV / KVShare 의 LLM token cross-request 와 직교 axis 로 multi-tenant batch>4 환경에서 KV memory -40%+ 가능한가?

### Per-Idea RQ

- **T1.1 Mosaic**: Lightweight classifier (DistilBERT-mini 60M) 가 6 scenario class (image-single / image-multi-turn / video-single / video-multi-turn / document / agent) 를 95%+ 정확도로 분류 + scenario-specific KV budget / prefix policy / token compression rate dispatch 시, vLLM 단일 path 대비 TTFT -30%, throughput +35% 달성 가능한가?
- **T1.2 Lattice**: Per-frame perceptual hash (pHash) 기반 frame radix tree + layer-wise prefix retention (PrefixKV 기반) 으로 video multi-turn QA 에서 vision tower re-run 제거 + cross-session frame prefix hit 50%+ + accuracy 보존 ≤ -1.0pt on MMBench-Video 가능한가?
- **T1.3 Bramble**: Image perceptual hash + tenant-shared LRU pool + privacy boundary 검증 시 multi-image agent loop (MileBench) 에서 KV memory -40% + cross-tenant share 정확도 ≤ -1.0pt 가능한가?
- **T2.1 Lantern**: NVDEC frame metadata + sliding window K=8 + 윈도우 외 frame KV 즉시 evict 시 streaming video real-time (latency < 100ms/frame) + Q-Bench-Video 정확도 보존 가능한가?
- **T2.2 Compass**: Ego-motion estimation (optical flow zero-mean check) + ego-stable region 만 retention 시 EgoSchema accuracy 보존 + token reduction 50%+ 가능한가?
- **T2.3 Hearth**: Scenario classifier 의 document VLM dispatch + `cudaAccessPolicyWindow` tile residence boost 시 DocVQA/ChartQA tile attention latency -25%+ 가능한가?

---

## 2. Essential Reading List

| # | Paper | Why read | Idea baseline |
|---|-------|---------|--------------|
| 1 | **VLCache** ([arXiv:2512.12977](https://arxiv.org/abs/2512.12977), 2025-12) — vision token 2-5% compute, 1.2-16× TTFT speedup | T1.2 Lattice 의 closest competitor (concurrent 50-70%, single-session 만 cover, multi-turn frame radix 미커버) | T1.2 |
| 2 | **PrefixKV** ([NeurIPS 2025, github.com/THU-MIG/PrefixKV](https://github.com/THU-MIG/PrefixKV)) — adaptive layer-wise KV retention, binary search | T1.2 Lattice 의 layer-wise retention 의 기반 | T1.2 |
| 3 | **SGLang RadixAttention** ([Zheng et al., ICLR 2024, arXiv:2312.07104](https://arxiv.org/abs/2312.07104)) — production LLM token prefix radix tree, 60-85% hit rate | T1.2 Lattice 의 frame-level 확장 + T1.3 Bramble 의 cross-image 확장 baseline | T1.2, T1.3 |
| 4 | **AdaptToken** (Microsoft Research, 2025, [microsoft.com/research](https://www.microsoft.com/en-us/research/publication/adapttoken-entropy-based-adaptive-token-selection-for-mllm-long-video-understanding/)) — entropy-based, 4 long video benchmark 일관 +6.7 over Qwen2.5-VL 7B | T2.2 Compass 의 long video baseline + Mosaic dispatcher target | T2.2 |
| 5 | **PolyKV / KVShare** ([arXiv:2604.24971](https://arxiv.org/abs/2604.24971) / [arXiv:2503.16525](https://arxiv.org/abs/2503.16525)) — LLM token cross-request KV pool | T1.3 Bramble 의 closest competitor (LLM token only, vision token 미커버) | T1.3 |

---

## 3. 연구 개요 + GAP outline

### 3.1 연구 개요

- **타겟**: VLM serving 의 scenario diversity 활용 — image/video × single/multi-turn × document/agent 6 scenario class
- **Production landscape (2026-04 측정)**:
  - vLLM Prefix Caching / SGLang RadixAttention prefix hit 60-85% on agent loop / multi-tenant SaaS / repo QA / long-doc ([particula.tech 2026](https://particula.tech/blog/sglang-vs-vllm-inference-engine-comparison))
  - SGLang 29% throughput edge on prefix-heavy workload
  - Production VLM (Qwen3.5 mlx-vlm Issue #832): multi-turn 마다 vision tower re-run + full re-prefill (gap)
- **Workload evidence**:
  - Long video (MLVU 1+ hr): 16K-32K vision token, ViT 1996-4040ms
  - Multi-turn video (MMBench-Video): 26 capabilities, vision KV reuse 미커버
  - Multi-image agent (MileBench): cross-image context, prefix hit 70%+
  - Document VLM (DocVQA/ChartQA): 32K+ vision token tile attention

### 3.2 GAP outline

- **G1**: vLLM Prefix Caching / SGLang RadixAttention 모두 LLM token prefix 만 cover. **Vision token frame-level radix tree (multi-turn video) 미커버**.
- **G2**: PolyKV ([arXiv:2604.24971](https://arxiv.org/abs/2604.24971)) / KVShare ([arXiv:2503.16525](https://arxiv.org/abs/2503.16525)) 가 LLM token cross-request 만, **vision token cross-image / cross-tenant pool (agent loop) 미커버**.
- **G3**: ECVL-ROUTER ([arXiv:2510.27256](https://arxiv.org/abs/2510.27256), ICLR'26 submission) 가 model-tier routing (small vs large VLM) 만 cover, **serving-stack config dispatch (KV budget / prefix policy / compression rate) 미커버**.
- **G4**: AdaptToken / LongVU / PVC 가 long video uniform compression, **EgoSchema ego-motion-aware compression 미커버**.
- **G5**: AwaRes / VisionThink / AttWarp 가 algorithm-side document VLM, **system-side L2 carveout + tile residence boost (document scenario-specific) 미커버**.
- **G6**: Cadence / EVS / WFS-SB 가 frame compression, **NVDEC HW frame metadata + sliding window real-time streaming 미커버**.

### 3.3 GAP → idea mapping

| GAP | 매칭 idea | Tier |
|-----|---------|------|
| G1 (vision frame radix) | T1.2 Lattice | Tier-1 |
| G2 (cross-image vision pool) | T1.3 Bramble | Tier-1 |
| G3 (serving-stack config dispatch) | T1.1 Mosaic | Tier-1 |
| G4 (ego-motion compression) | T2.2 Compass | Tier-2 |
| G5 (document L2 carveout) | T2.3 Hearth | Tier-2 |
| G6 (NVDEC streaming window) | T2.1 Lantern | Tier-2 |

---

## 4. Implementation-Priority Decision Tree

### 양식 A — Inline SVG Flowchart (R40 / R44)

<svg viewBox="0 0 900 740" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:900px;height:auto;font-family:system-ui,sans-serif">
<defs>
<marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="#1E2761"/></marker>
</defs>
<rect x="270" y="20" width="360" height="60" rx="6" fill="#CADCFC" stroke="#1E2761" stroke-width="2"/>
<text x="450" y="46" text-anchor="middle" fill="#1E2761" font-size="14" font-weight="bold">Wk 1-3 · Baseline Reproduction</text>
<text x="450" y="66" text-anchor="middle" fill="#1E2761" font-size="11">vLLM Prefix + SGLang RadixAttention 60-85% hit 재현</text>
<line x1="450" y1="80" x2="450" y2="115" stroke="#1E2761" stroke-width="2" marker-end="url(#arr)"/>
<polygon points="450,120 620,165 450,210 280,165" fill="#F9E795" stroke="#1E2761" stroke-width="2"/>
<text x="450" y="160" text-anchor="middle" fill="#1E2761" font-size="13" font-weight="bold">Wk 4-7 · T1.1 Mosaic + T1.2 Lattice (parallel)</text>
<text x="450" y="180" text-anchor="middle" fill="#1E2761" font-size="11">classifier 95%+ ? · TTFT -30%+ ? · frame prefix 50%+ ?</text>
<line x1="450" y1="210" x2="450" y2="245" stroke="#1E2761" stroke-width="2" marker-end="url(#arr)"/>
<rect x="40" y="250" width="240" height="70" rx="6" fill="#2C5F2D" stroke="#1E2761" stroke-width="2"/>
<text x="160" y="278" text-anchor="middle" fill="white" font-size="12" font-weight="bold">PASS → Tier-1 진행</text>
<text x="160" y="298" text-anchor="middle" fill="white" font-size="10">Mosaic OSDI 2027 / Lattice MLSys</text>
<text x="160" y="312" text-anchor="middle" fill="white" font-size="10">Wk 8-10 T1.3 Bramble 진입</text>
<rect x="320" y="250" width="260" height="70" rx="6" fill="#F96167" stroke="#1E2761" stroke-width="2"/>
<text x="450" y="278" text-anchor="middle" fill="white" font-size="12" font-weight="bold">BELOW (-15~-30%) → Tier-2 reposition</text>
<text x="450" y="298" text-anchor="middle" fill="white" font-size="10">단일 scenario dispatcher / IEEE CAL</text>
<text x="450" y="312" text-anchor="middle" fill="white" font-size="10">또는 single-session frame radix only</text>
<rect x="620" y="250" width="240" height="70" rx="6" fill="#1E2761" stroke="#1E2761" stroke-width="2"/>
<text x="740" y="278" text-anchor="middle" fill="white" font-size="12" font-weight="bold">CRITICAL FAIL → DROP</text>
<text x="740" y="298" text-anchor="middle" fill="white" font-size="10">classifier &lt;90% / TTFT &lt;-15%</text>
<text x="740" y="312" text-anchor="middle" fill="white" font-size="10">VLCache axis scoop 시 hard reframe</text>
<line x1="160" y1="320" x2="160" y2="365" stroke="#1E2761" stroke-width="2" marker-end="url(#arr)"/>
<rect x="40" y="370" width="240" height="70" rx="6" fill="#CADCFC" stroke="#1E2761" stroke-width="2"/>
<text x="160" y="398" text-anchor="middle" fill="#1E2761" font-size="12" font-weight="bold">Wk 8-10 · T1.3 Bramble</text>
<text x="160" y="418" text-anchor="middle" fill="#1E2761" font-size="10">image pHash + privacy + cuckoo filter</text>
<text x="160" y="432" text-anchor="middle" fill="#1E2761" font-size="10">MileBench KV mem -40%+ 평가</text>
<line x1="160" y1="440" x2="160" y2="485" stroke="#1E2761" stroke-width="2" marker-end="url(#arr)"/>
<polygon points="160,490 280,525 160,560 40,525" fill="#F9E795" stroke="#1E2761" stroke-width="2"/>
<text x="160" y="525" text-anchor="middle" fill="#1E2761" font-size="11" font-weight="bold">KV -40%+ ? privacy 0?</text>
<text x="160" y="540" text-anchor="middle" fill="#1E2761" font-size="10">accuracy ≤-1.0pt ?</text>
<line x1="280" y1="525" x2="380" y2="525" stroke="#1E2761" stroke-width="2" marker-end="url(#arr)"/>
<rect x="380" y="490" width="240" height="70" rx="6" fill="#2C5F2D" stroke="#1E2761" stroke-width="2"/>
<text x="500" y="518" text-anchor="middle" fill="white" font-size="12" font-weight="bold">Wk 11-13 · Tier-2 (Lantern/Compass/Hearth)</text>
<text x="500" y="538" text-anchor="middle" fill="white" font-size="10">streaming / ego-motion / DocVQA L2</text>
<text x="500" y="552" text-anchor="middle" fill="white" font-size="10">IEEE CAL / DATE 2027 letter</text>
<line x1="500" y1="560" x2="500" y2="605" stroke="#1E2761" stroke-width="2" marker-end="url(#arr)"/>
<rect x="280" y="610" width="440" height="80" rx="6" fill="#1E2761" stroke="#1E2761" stroke-width="2"/>
<text x="500" y="638" text-anchor="middle" fill="white" font-size="13" font-weight="bold">Wk 14-16 · Paper Writing + Submission</text>
<text x="500" y="658" text-anchor="middle" fill="white" font-size="10">T1.1 Mosaic OSDI / T1.2 Lattice MLSys / T1.3 Bramble MLSys+OSDI</text>
<text x="500" y="675" text-anchor="middle" fill="white" font-size="10">T2.1 IEEE CAL · T2.2/T2.3 DATE 2027 · paper pair (Mosaic + Lattice)</text>
<line x1="500" y1="690" x2="500" y2="715" stroke="#1E2761" stroke-width="2"/>
<text x="500" y="730" text-anchor="middle" fill="#1E2761" font-size="11" font-style="italic">benchmark 분담: MMBench-Video / MLVU / MileBench / EgoSchema / Q-Bench-Video / DocVQA</text>
</svg>

### 양식 B — ASCII Flowchart (보조)

```
[Week 1-3: Baseline Reproduction]
   ├─ vLLM v0.10+ Prefix Caching + SGLang RadixAttention baseline
   ├─ 6 scenario benchmark 측정 (MLVU / MMBench-Video / MileBench / DocVQA / EgoSchema / Q-Bench-Video)
   └─ Production prefix hit rate 측정 (60-85% 재현)

[Week 4-7: T1.1 Mosaic Dispatcher Preliminary]
   ├─ Scenario classifier (DistilBERT-mini 60M) 학습 (6 class, weak supervision from query+image meta)
   ├─ Scenario-specific KV budget / prefix policy table 설계
   ├─ Pass (TTFT -30%+, throughput +35%+) → Tier-1 OSDI/ASPLOS 진행
   ├─ Below (-15~-30%) → Tier-2 reposition (단일 scenario dispatcher)
   └─ Critical fail (<-15%) → DROP

[Week 4-7: T1.2 Lattice Frame Radix Vision KV] (병렬 with Mosaic)
   ├─ Per-frame pHash + frame ID radix tree
   ├─ Layer-wise prefix retention (PrefixKV 기반 binary search)
   ├─ MMBench-Video / Video-MME multi-turn 평가
   ├─ Pass (frame prefix hit 50%+, accuracy ≤-1.0pt) → MLSys/OSDI 진행
   └─ Critical fail (VLCache 와 axis 중복) → cross-tenant axis 으로 hard reframe

[Week 8-10: T1.3 Bramble Cross-Image Pool] (의존: T1.2 의 hash 재사용)
   ├─ Image-level pHash + tenant-shared LRU + privacy boundary
   ├─ MileBench multi-image agent 평가
   ├─ Pass (KV mem -40%+, accuracy ≤-1.0pt) → MLSys 진행
   └─ Critical fail (PolyKV/KVShare scoop) → drop

[Week 11-13: Tier-2 idea (Lantern / Compass / Hearth)]
   ├─ T2.1 Lantern: NVDEC + Q-Bench-Video real-time
   ├─ T2.2 Compass: EgoSchema ego-motion
   ├─ T2.3 Hearth: DocVQA/ChartQA L2 carveout
   └─ 각 idea Pass / Below / Drop branch

[Week 14-16: Paper writing + ablation]
   ├─ T1.1 Mosaic → OSDI 2027 또는 ASPLOS 2027
   ├─ T1.2 Lattice → MLSys 2027 또는 NeurIPS 2026
   ├─ T1.3 Bramble → MLSys 2027 (T1.1 paper pair 가능)
   ├─ T2.1 Lantern → IEEE CAL letter
   ├─ T2.2 Compass → DATE 2027 또는 ISLPED 2027
   └─ T2.3 Hearth → DATE 2027 또는 IEEE ESL letter
```

### Inter-Idea Dependency

- **T1.1 Mosaic ← T1.2 Lattice / T1.3 Bramble / T2.1-T2.3**: Mosaic dispatcher 가 나머지 5 idea 의 scenario classifier 역할 → paper pair 가능 (OSDI Mosaic + MLSys Lattice/Bramble system companion)
- **T1.2 Lattice ↔ T1.3 Bramble**: pHash + radix tree infrastructure 공유 → 동일 base 위 build
- **T2.2 Compass ↔ T2.3 Hearth**: 독립 (다른 scenario)

---

## 5. Tier-1 Top 3

| ID | Idea | Domain | Mean Score | Target Venue |
|----|------|--------|-----------|------------|
| **T1.1** | [Mosaic](tier1/01-mosaic.md) — Workload-Adaptive Serving Configuration Dispatcher | serving + system | **7.50** | OSDI 2027 / ASPLOS 2027 |
| **T1.2** | [Lattice](tier1/02-lattice.md) — Cross-Turn Frame-Indexed Radix Vision KV Cache for Multi-Turn Video QA | serving + algorithm | **7.50** | MLSys 2027 / NeurIPS 2026 |
| **T1.3** | [Bramble](tier1/03-bramble.md) — Cross-Image Vision Token Pool for Multi-Image Agent Loop | serving + system | **7.20** | MLSys 2027 / OSDI 2027 |

### T1.1 Mosaic Contribution

- **(a) Mechanism 정성적 benefit**:
  - **M1 — Lightweight Scenario Classifier**: DistilBERT-mini 60M weak-supervised on (query keyword + image/video meta + history depth), 6-way softmax, latency < 2ms/request
  - **M2 — Scenario-Conditional Config Dispatch**: scenario class → KV budget table (long video 5%, multi-turn 30%, agent loop pool) + prefix policy (radix vs LRU) + token compression rate
  - **M3 — Online Hit-Rate Adaptive Calibration**: per-scenario hit rate counter → 가장 hot scenario 의 KV/prefix GPU memory pinning (LMCache pin API 와 직교 — scenario-classifier-driven)
- **(b) Closest competitor**: vs **ECVL-ROUTER** ([arXiv:2510.27256](https://arxiv.org/abs/2510.27256), ICLR'26 submission) — model-tier routing (small vs large VLM); Mosaic 은 single-model serving-stack config 직교 axis. vs **vLLM Prefix Caching** (single path, scenario-agnostic).
- **(c) 예상 gain 표**:

| 지표 | Baseline (vLLM single path) | Mosaic | 개선 |
|------|----------------------------|--------|------|
| [Performance] TTFT (mixed scenario batch) | 1.0× | **0.65-0.70×** | **-30 ~ -35%** |
| [Performance] Throughput (multi-tenant) | 1.0× | **1.30-1.40×** | **+30 ~ +40%** |
| [Memory] GPU VRAM (peak, batch=8) | 78GB | 58GB | -25% |
| [Cost eff.] $/req | 1.0× | **0.65×** | -35% |

- **(d) Tier 강등 risk**: Phase 1' 에서 ECVL-ROUTER 와 axis 차별화 부족 발견 시 Tier-2 reposition (single-scenario optimizer).
- **(e) Outperform 가능성**: Production vLLM/SGLang upstream PR + 6 scenario benchmark 일관 향상 → OSDI distinguished.

### T1.2 Lattice Contribution

- **(a) Mechanism**:
  - **M1 — Per-Frame pHash + Frame Radix Tree**: 각 video frame 의 16-bit perceptual hash → frame ID 시퀀스 prefix 가 일치하면 vision KV 재사용
  - **M2 — Layer-Wise Prefix Retention**: PrefixKV ([NeurIPS 2025](https://github.com/THU-MIG/PrefixKV)) binary search 기반 layer-wise KV retention ratio + Lattice radix tree node 별 layer-wise budget
  - **M3 — Cross-Session Frame Prefix Sharing**: 동일 video 가 다른 session 에서 다른 question 으로 재진입 시 frame radix tree 재사용 (privacy boundary 검증 — public video 만)
- **(b) Closest competitor**: vs **VLCache** ([arXiv:2512.12977](https://arxiv.org/abs/2512.12977), 2025-12) — single-session image-patch reuse, multi-session frame radix 미커버. vs **PrefixKV** ([NeurIPS 2025](https://github.com/THU-MIG/PrefixKV)) — vision instruction-following layer-wise, frame radix 미커버. vs **SGLang RadixAttention** ([ICLR 2024, arXiv:2312.07104](https://arxiv.org/abs/2312.07104)) — LLM token prefix radix only, vision token 미커버.
- **(c) 예상 gain 표**:

| 지표 | Baseline (vLLM Prefix + LLM only) | Lattice | 개선 |
|------|----------------------------------|---------|------|
| [Performance] Multi-turn TTFT (turn 2+) | 1.0× | **0.20-0.40×** (vision tower re-run 제거) | **-60 ~ -80%** |
| [Memory] Vision KV (per-session, multi-turn 5 turn) | 5×1.5GB = 7.5GB | **1.5GB + radix overhead** | **-75%** |
| [Quality] MMBench-Video accuracy drop | n/a | **≤ -1.0pt** | acceptable |
| [Performance] Frame prefix hit rate (cross-session) | 0% (no vision prefix) | **50-70%** | new capability |

- **(d) Tier 강등 risk**: VLCache concurrent 50-70% — Phase 1' 에서 cross-session axis 차별화 명확화 필요. Risk 시 Tier-2 reposition (single-session 만).
- **(e) Outperform 가능성**: Multi-turn video QA + cross-session frame radix 가 production VLM (Qwen3.5/mlx-vlm Issue #832) 의 직접적 gap 해결 → MLSys outstanding paper.

### T1.3 Bramble Contribution

- **(a) Mechanism**:
  - **M1 — Image-Level pHash + Tenant-Shared LRU Pool**: 각 image 의 pHash 를 global pool key 로 사용 + LRU eviction
  - **M2 — Privacy Boundary Verification**: pool 진입 전 image content public/private classifier (CLIP-based, 95%+ accuracy) → public 만 cross-tenant share, private 는 tenant-isolated
  - **M3 — Reference Counting + Privacy-Preserving Access**: pool entry 별 reference count, eviction 시 마지막 reference 이후 zeroize
- **(b) Closest competitor**: vs **PolyKV** ([arXiv:2604.24971](https://arxiv.org/abs/2604.24971)) — LLM token cross-request KV pool, vision token 미커버. vs **KVShare** ([arXiv:2503.16525](https://arxiv.org/abs/2503.16525)) — LLM token only. vs **OxyGen** ([arXiv:2603.14371](https://arxiv.org/abs/2603.14371)) — agent KV management, cross-tenant vision 미커버.
- **(c) 예상 gain 표**:

| 지표 | Baseline (per-tenant isolated) | Bramble | 개선 |
|------|-------------------------------|---------|------|
| [Memory] GPU KV (multi-tenant batch=8, 6 image avg) | 24GB | **14GB** | **-40%** |
| [Memory] Max concurrent tenants | 4 | **8** | 2× |
| [Cost eff.] $/req | 1.0× | **0.55-0.70×** | **-30 ~ -45%** |
| [Quality] MileBench accuracy drop | n/a | ≤-1.0pt | acceptable |

- **(d) Tier 강등 risk**: PolyKV/KVShare 가 vision-aware 확장 시 scoop risk → Phase 1' 에서 vision-token-specific axis (image pHash + privacy boundary) 강조.
- **(e) Outperform 가능성**: Multi-tenant agent loop 에서 cost -45% 측정 시 production deployment value → MLSys + OSDI dual.

---

## 6. Tier-2 독립 Top 3

| ID | Idea | Domain | Mean Score | Target Venue |
|----|------|--------|-----------|------------|
| **T2.1** | [Lantern](tier2/01-lantern.md) — NVDEC-Coupled Sliding-Window KV Eviction for Streaming Video | system + serving | **6.00** | IEEE CAL letter + vLLM upstream PR |
| **T2.2** | [Compass](tier2/02-compass.md) — Ego-Motion-Aware Vision Token Compression for Egocentric Video | algorithm | **6.00** | DATE 2027 / ISLPED 2027 |
| **T2.3** | [Hearth](tier2/03-hearth.md) — Document VLM L2 Carveout + Tile Locality Boost | system | **6.00** | DATE 2027 / IEEE ESL letter |

### T2.1 Lantern Contribution

- **Mechanism**: (M1) NVDEC frame metadata (frame stride / motion vector) extract → (M2) sliding window K=8 frame retention → (M3) window 외 frame KV 즉시 evict (GPU memory recycle)
- **Closest competitor**: 0-30% scoop (HW-decoder co-design 희소). 이전 세션 Cadence cross-share 가능.
- **예상 gain**: Streaming video latency < 100ms/frame, GPU memory window-bounded
- **Benchmark 활용**: Q-Bench-Video (CVPR 2025) + streaming synthetic

### T2.2 Compass Contribution

- **Mechanism**: (M1) optical flow zero-mean check (ego-motion stable region detection) → (M2) ego-stable region 만 token retention → (M3) ego-motion vector 를 MRoPE T-axis 보강 신호로 추가
- **Closest competitor**: 30-50% scoop (MM-Ego 는 data-side, compression 미커버)
- **예상 gain**: EgoSchema accuracy 보존 + token reduction 50%+
- **Benchmark 활용**: EgoSchema (5063 video, ego-motion dominant)

### T2.3 Hearth Contribution

- **Mechanism**: (M1) Mosaic scenario classifier → document VLM 식별 → (M2) `cudaAccessPolicyWindow` tile residence boost → (M3) tile attention eviction priority 조정
- **Closest competitor**: 30-50% scoop (AttWarp adjacent algorithm-side, system-side L2 carveout 미커버)
- **예상 gain**: DocVQA/ChartQA tile attention latency -25%+
- **Benchmark 활용**: DocVQA (DocVQA 2026 ICDAR) + ChartQA + OCRBench

---

## 7. 미선정 / DROP / 흡수 아이디어

→ [unselected.md](unselected.md)

### Quick Summary

- **DROP 14개** — 모두 70%+ scoop or single axis weak:
  - **A3 KV Budget**: PrefixKV/AdaptToken adaptive policy 의 scenario conditioning 만 → incremental
  - **A4 Multi-Tenant Batch**: Nova ([arXiv:2509.21301](https://arxiv.org/abs/2509.21301)) cross-stage parallel 70%+ scoop
  - **A5 Streaming**: Nova + One-Token-per-Frame ([arXiv:2604.14149](https://arxiv.org/abs/2604.14149)) 50-70% scoop, T2.1 Lantern 으로 흡수
  - **A6 Doc Multi-Turn ROI**: AwaRes 70%+ scoop
  - **B1 Question-Conditional**: VisionThink 70%+ scoop
  - **B2 Long Video Scene Boundary**: LongVU/PVC 70%+ scoop
  - **B3 Action Boundary**: METok ([arXiv:2506.02850](https://arxiv.org/abs/2506.02850)) 50-70% scoop
  - **B4 Event-Graph KV**: TSG ([arXiv:2601.06097](https://arxiv.org/abs/2601.06097)) 91.4% token reduction 70%+ scoop
  - **B6 Doc ROI Prune**: AttWarp 70%+ scoop
  - **B7 Multi-Image Dedup**: Lossless Ultimate Compression ([arXiv:2512.09010](https://arxiv.org/abs/2512.09010)) 50-70%, T1.3 Bramble 으로 흡수
  - **C1 Vision KV Warm Cache**: VLCache + LMCache 70%+ scoop, T1.2 Lattice 로 흡수
  - **C3 Cross-Tenant Pool**: OxyGen + KVShare 70%+ scoop, T1.3 Bramble 로 흡수
  - **C4 Batch Optimizer**: T1.1 Mosaic 의 sub-component
  - **C6 Hot Pinning**: LMCache pin API 50-70%, T1.1 Mosaic 으로 흡수

- **흡수 1개**: C5 Scene Demote → T1.2 Lattice 의 layer-wise sub-mechanism

---

## 8. 참고 (Cumulative file links)

- [`__research_wiki/index.md`](../../index.md) — 전 세션 timeline
- [`__research_wiki/ideas.md`](../../ideas.md) — Tier-1 + Tier-2 idea log
- [`__research_wiki/papers.md`](../../papers.md) — 본 세션 분석 논문
- [`__research_wiki/trends.md`](../../trends.md) — Scenario-aware VLM serving trend
- [`__research_wiki/concepts.md`](../../concepts.md) — Frame Radix / Cross-Tenant Vision Pool / Scenario Dispatcher 신규 개념

---

## 9. 약어 / 핵심 용어 풀이 (CTRL+F / Cmd+F 로 본 섹션 검색)

### 9.1 도메인 약어
- **VLM** — Vision-Language Model
- **TTFT** — Time-To-First-Token
- **pHash** — perceptual hash (16-bit average hash 등)
- **Radix tree** — prefix-tree 자료구조 (SGLang RadixAttention 의 base)
- **LMCache / vLLM Prefix Caching** — production prefix caching framework
- **NVDEC** — NVIDIA hardware video decoder
- **MMBench-Video / MLVU / Video-MME / LongVideoBench / MVBench / NExT-QA / EgoSchema / Q-Bench-Video / DocVQA / ChartQA / MileBench** — VLM scenario benchmarks
- **MRoPE** — Multimodal Rotary Position Embedding

### 9.2 Polysemous Term Disambiguation (R51-α)
- **prefix** — radix tree prefix (SGLang) ≠ document prefix
- **frame** — video frame (per-frame pHash) ≠ stack frame
- **pool** — KV cache pool (cross-tenant) ≠ thread pool
- **token** — vision token (Lattice) ≠ LLM text token
- **session** — user session (privacy boundary) ≠ TCP session
- **window** — sliding window K=8 frame ≠ attention window

### 9.3 Idea Metaphor Noun → Mechanism 대응
- **Mosaic**: 시나리오별 다른 조각이 모여 전체 그림 (workload-adaptive dispatcher)
- **Lattice**: radix tree 격자 구조 (frame prefix sharing)
- **Bramble**: 가시 덤불 — 여러 image 가 얽힌 multi-image agent
- **Lantern**: 등불 — sliding window streaming
- **Compass**: ego-motion direction
- **Hearth**: 화로 — locality / tile heat

---

## R31 검증 결과 (cross-reference)

본 세션 사용자 input "video multi-turn QA visual info 최적화 + accuracy 보존" 은 production prefix hit 60-85% measured common case 로 R31 검증 통과. Wrong-insight 아님.

상세: 본 세션 staging 파일 (`sessions/2026-05-02-mode1-vlm-scenario-aware-staging.md`, local-only, R29.1 publish 제외) 의 § Section 0 — Phase 1 raw idea pool + R31 verification trace + workload evidence + Step 0/0-α 외부 검색 결과 모두 기록.
