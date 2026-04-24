# Qwen3-VL (DeepStack) + Qwen3.5 Edge 최적화: Loom / Mangrove / Vault / Gale / Forge

*Session date: 2026-04-24 · Mode 1 (sentence input + 3 PDF)*

> Qwen3-VL 의 3 core upgrade — **DeepStack multi-layer injection** (`visual_indexes=[8,16,24]`), **Interleaved MRoPE** (t/h/w round-robin frequency), **text-based video timestamp** — 과 Qwen3.5-Omni 의 **Hybrid MoE + Gated Delta Net + Thinker-Talker + ARIA** 최신 아키텍처를 Jetson Orin AGX / Jetson Thor 128GB LPDDR5X / RTX 4060-5090 edge GPU 에 배포 시, 이전 Qwen2.5-VL / LLaVA-1.5 로는 가능하지 않은 새 최적화를 도출했다. **R23 Step 0-α** (IISWC/ISPASS + benchmark technical report 우선 탐색) 신규 규칙 첫 적용 세션. 3 experts × 4 idea = 12 → 3-way artificial split 통합 → **Tier-1 Top 3 (Loom ★ lead / Mangrove / Vault') + Tier-2 독립 Top 2 (Gale / Forge) = 5 selected**, T3 Echo 는 VLCache scoop 72-78% + 이전 세션 Tidal 재진입 공간으로 DROP.

---

## 1. 연구 진행 Meta

### 1.1 사용자 쿼리 원문

> "/home/paper/260424 sota llm vlm/ 폴더에, 기존에 탐색했던 Qwen2.5-VL 이후에 나온 Qwen3에 대한 tech report 와 이에 적용된 deepstack 문서가 있어. 또한, LLM 최신 모델인 Qwen3.5 모델이 있는데, 이러한 최신 모델 아키텍처의 엣지 적용 시, 이전의 VLM/LLM 모델 대비 아키텍처 차이를 고려해서, 성능/에너지 최적화 방안에 대한 새로운 ideation 을 진행해줘."

### 1.2 입력 3 PDF

1. **Qwen3-VL Technical Report** ([arXiv:2511.21631](https://arxiv.org/abs/2511.21631), 2025-11) — 3 core upgrade: Interleaved MRoPE / DeepStack integration / explicit video timestamp. Dense (2B/4B/8B/32B) + MoE (30B-A3B/235B-A22B). 256K native context.
2. **DeepStack** ([arXiv:2406.04334](https://arxiv.org/abs/2406.04334), NeurIPS 2024) — visual tokens 을 N group 으로 stack → N decoder layer 에 residual 주입. Early decoder layer 가 visual processing 에 효과적. `deepstack_visual_indexes=[8,16,24]` HF 공식 default.
3. **Qwen3.5-Omni Technical Report** ([arXiv:2604.15804](https://arxiv.org/abs/2604.15804), 2026-04-22) — Thinker-Talker, Hybrid MoE + **Gated Delta Net (GDN)**, ARIA, multi-codebook RVQ + MTP, Code2Wav ConvNet.

### 1.3 주요 키워드 (4축)

| 축 | 키워드 |
|---|---|
| **도메인** | Qwen3-VL, Qwen3.5-Omni, Edge GPU (Jetson Orin AGX / Thor 128GB LPDDR5X / RTX 4060-5090), Dense + MoE 30B-A3B |
| **관찰·특징** | DeepStack multi-layer residual injection (layer 8/16/24), Interleaved MRoPE frequency interleave, Hybrid MoE + GDN, Thinker-Talker + ARIA, 256K native context, NVFP4 Blackwell |
| **제안 기법** | Layer-aware sub-graph + LPDDR bank-aligned + DLA offload / Unified permuted LUT + FA3 fused rotation + texture unit / DeepStack×MoE L2 contention + bank placement + activation-aware L2 pin / GDN constant-memory + 256K 3-tier / Thinker Tensor Core + Talker DLA |
| **타겟** | TTFT, TPOP, J/token, W, first-packet latency, MMMU accuracy, 256K context 처리 |

### 1.4 중점적으로 고려한 축

- **이전 세대로 불가능한 VLM-only 아키텍처 diff 직접 mechanism 화**: DeepStack multi-layer injection 은 Qwen2.5-VL single-layer concat 에서는 적용 불가. Interleaved MRoPE 는 chunked MRoPE 에서는 단일 LUT 불가.
- **R23 Step 0-α 신규 규칙 첫 적용**: IISWC + ISPASS + MLPerf/Jetson benchmark report 우선 탐색 (15편 수집).
- **Jetson Thor 128GB LPDDR5X + NVFP4 생태계**: 30B-A3B MoE full 탑재 가능 환경 활용.
- **[NVIDIA forum 2026-02 Jetson Thor vLLM MoE gap](https://forums.developer.nvidia.com/t/jetson-agx-thor-vllm-26-02-moe-performance-significantly-below-reference-missing-fused-moe-config/364663)**: 산업계 실제 gap 해결 motivation.

### 1.5 의도적으로 제외한 축

| 제외 축 | 이유 |
|---|---|
| 이전 2026-04-23 edge-vlm-energy 세션 아이디어 (Parquet/Triptych/Cartographer/Sift/Verge/Tidal) | 이 세션은 Qwen2.5-VL 기반, 본 세션은 Qwen3-VL/3.5 신규 아키텍처 diff 특화로 완전 독립. |
| Multi-node / cloud serving | Edge GPU scope 규율 |
| LLM-only quantization | 2026-04-23 PRISM-VLM-KV 세션 등에서 다룸. 본 세션은 DeepStack + MRoPE + MoE + Thinker-Talker specific. |
| Training-time optimization | Inference serving 쿼리 명시. |
| HBM-PIM | Jetson LPDDR 기반, HBM-PIM 부적합. |

### 1.6 Step 0-α Workload Characterization Sources (신규)

R23 2026-04-24 신규 규칙 — IISWC/ISPASS + benchmark technical report 우선 탐색 의무.

- **IISWC**: [LLMServingSim IISWC 2024 arXiv:2408.05499](https://arxiv.org/abs/2408.05499), [EdgeReasoning IISWC 2025 arXiv:2511.01866](https://arxiv.org/abs/2511.01866), [Systematic LLM Characterization arXiv:2512.01644](https://arxiv.org/pdf/2512.01644), LLM-on-CPU IISWC 2025, IISWC 2024 MLLM Workshop
- **ISPASS/PAISE**: [Arya & Simmhan PAISE 2025 arXiv:2506.09554](https://arxiv.org/abs/2506.09554), Mind-the-Memory-Gap ISPASS 2025
- **Benchmark reports**: [MLPerf Inference v5.0](https://mlcommons.org/2025/04/mlperf-inference-v5-0-results/) / [v5.1](https://mlcommons.org/2025/09/mlperf-inference-v5-1-results/) / [Jetson Thor 7x Gen AI Blog](https://developer.nvidia.com/blog/unlock-faster-smarter-edge-models-with-7x-gen-ai-performance-on-nvidia-jetson-agx-thor/) / [vLLM Qwen3-Next Blog](https://blog.vllm.ai/2025/09/11/qwen3-next.html) / [vLLM EPD Blog](https://blog.vllm.ai/2025/12/15/vllm-epd.html) / [Qwen3-VL-30B-A3B vLLM-Ascend](https://docs.vllm.ai/projects/ascend/en/latest/tutorials/models/Qwen3-VL-30B-A3B-Instruct.html) / [Jetson AI Lab GenAI Benchmarking](https://www.jetson-ai-lab.com/tutorials/genai-benchmarking/) / [Thor MoE forum](https://forums.developer.nvidia.com/t/jetson-agx-thor-vllm-26-02-moe-performance-significantly-below-reference-missing-fused-moe-config/364663) / [NVFP4 blog](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/)

수집 총 **15편** (R23 Step 0-α 완전 준수).

### 1.7 사용된 전문가 + 리뷰어

- Experts: ai-optimization-expert / legacy-system-expert / hw-pim-accelerator-expert
- Reviewers: novelty / differentiation / impact / similarity-critique (4 병렬)

---

## 2. Tier-1 Top 3

### 2.1 ★ Top 1 (lead) — **Loom** (avg 7.90, MLSys 2026 / ISCA 2027)

#### 2.1.1 개요

**"Loom"** = 베틀. t/h/w 실이 low/high frequency band 에 interleave 되어 positional encoding 을 직조 — Qwen3-VL 의 Interleaved MRoPE `[24,20,20]` round-robin frequency allocation 의 정확한 은유.

Qwen3-VL 의 Interleaved MRoPE 는 t/h/w 를 chunked 분할 (Qwen2-VL/2.5-VL) 이 아닌 embedding dimension 에 round-robin interleave 하여 frequency balance 를 확보한다. 이는 기존 3-table chunked LUT 구조 (이전 세션 Cartographer) 로 표현 불가능하며 **single unified permuted LUT** 가 필요하다. Kernel-level 에서 FA3 tile 내부에 fused rotation 을 적용하면 SFU (Special Function Unit) 를 우회할 수 있고, edge GPU 의 texture unit 으로 LUT 을 fetch 하여 SIMT divergence 를 제거한다.

#### 2.1.2 기존 연구 한계점 및 Gap

- [Revisiting MRoPE ICLR 2026 arXiv:2510.23095](https://arxiv.org/abs/2510.23095) — algorithm-level design only, kernel-level 구현 공백.
- [vLLM PR #22593 partial rotary](https://blog.vllm.ai/2025/09/11/qwen3-next.html) — generic implementation, Interleaved-specific fusion 없음.
- Cartographer (2026-04-23 edge-vlm-energy 세션) — chunked MRoPE 전제, Interleaved 에 적용 불가.
- [T-MAC EuroSys'25 arXiv:2407.00088](https://arxiv.org/abs/2407.00088) / [LUT Tensor Core ISCA'25 arXiv:2408.06003](https://arxiv.org/abs/2408.06003) — weight mpGEMM LUT, positional encoding 미커버.
- [FlashAttention-3](https://arxiv.org/abs/2407.08608) + FA4 — rotation 을 tile-internal 에 fuse 한 예 없음.

#### 2.1.3 제안 기법 (3 Mechanism, improve-first)

- **M1 Unified permuted LUT** — Interleaved MRoPE 의 t/h/w round-robin 을 단일 LUT 으로 통합. Chunked 3-table register pressure 제거.
- **M2 FA3 tile-internal fused rotation** — MRoPE rotation 을 FlashAttention-3 tile loop 내부에 fuse 하여 SFU 우회 + memory BW 절감.
- **M3 Texture unit LUT fetch + register pack** — edge GPU 의 texture unit 으로 LUT fetch, register pack 으로 SIMT divergence 제거.

#### 2.1.4 평가·실험 플랜 (5-요소)

| 요소 | 상세 |
|---|---|
| Hardware | Jetson Orin AGX 64GB (primary) + Thor 128GB (FP4) + RTX 5090 |
| Model | Qwen3-VL-8B (Interleaved MRoPE native) + Qwen3-VL-30B-A3B + Qwen3-VL-2B |
| Dataset | VideoMME long + MMMU + DocVQA + 256K long video stress |
| Tools | CUTLASS 3.6+ FA3 fused rotation + Nsight Compute (SFU busy %) + vLLM v0.8+ Interleaved MRoPE integration |
| Ablation/Protocol | 2^3: M1 unified LUT × M2 FA3 fused × M3 texture unit + permuted layout sweep. 8 baseline (peer 63%): Revisiting MRoPE, FA3, T-MAC EuroSys'25, LUT Tensor Core ISCA'25, RotateKV, SAIL, FA4, Kimi Linear. Runtime 8주. |

#### 2.1.5 예상 효과

- MRoPE kernel latency -30~45%
- Prefill TTFT -12~15%
- MMMU accuracy 0 drop (bit-exact LUT)
- **Scoring**: Nov 7.4 / Diff 8.0 / Imp 7.7 / Feas 8.5 → **avg 7.90 (3:0 unanimous lead)**

---

### 2.2 Top 2 — **Mangrove** (avg 7.60, ASPLOS 2027 / MLSys 2026)

#### 2.2.1 개요

**"Mangrove"** = 맹그로브 나무. 여러 깊이 뿌리 (layer 8/16/24) 가 하나의 나무 (Qwen3-VL) 를 지지하는 구조 — DeepStack multi-layer residual injection 은유.

DeepStack 의 `deepstack_visual_indexes=[8,16,24]` 에서 layer 0-7 은 vision-independent 하다. 이 특성을 활용해 **4-stage sub-graph partitioning** 으로 CUDA dual-stream concurrent dispatch 가 가능하다. 또한 vision token residual write 를 **LPDDR5X bank-aligned layout** 으로 정렬하면 row-buffer hit rate 가 상승하며, **layer 8/16/24 injection point 를 Jetson DLA 에 offload** 하여 energy 절감한다.

#### 2.2.2 기존 연구 한계점 및 Gap

- [DeepStack NeurIPS 2024 arXiv:2406.04334](https://arxiv.org/abs/2406.04334) — algorithm original, system-level paper 공백 (first system-level 가능).
- [Cross-Layer Injection arXiv:2601.10710](https://arxiv.org/abs/2601.10710) — algorithm-level 후속, system axis 와 orthogonal.
- [Nova arXiv:2509.21301](https://arxiv.org/abs/2509.21301) — modality-stage partition (encoder/prefill/decode), vision-layer axis (residual injection point) 와 orthogonal.
- [vLLM EPD](https://blog.vllm.ai/2025/12/15/vllm-epd.html) — generic disaggregation, DeepStack layer-specific 없음.
- Triptych (2026-04-23 edge-vlm-energy 세션) — modality-stage (vision/projector/LLM), vision-layer axis 와 50% overlap 이나 write-traffic (Mangrove) vs read-traffic (Triptych) 으로 분리.

#### 2.2.3 제안 기법 (3 Mechanism)

- **M1 Layer-aware 4-stage sub-graph partitioning + CUDA dual-stream** — layer 0-7 vision-independent, layer 8/16/24 residual injection, layer 24+ final.
- **M2 LPDDR5X bank-aligned vision residual write** — row-buffer hit rate 증가.
- **M3 Layer 8/16/24 injection point DLA offload** — dual-band pipeline (DLA + GPU).

#### 2.2.4 평가·실험 플랜

| 요소 | 상세 |
|---|---|
| Hardware | Jetson Orin AGX 64GB (DLA primary) + Thor (DLA-Next) + RTX 4090 baseline |
| Model | Qwen3-VL-8B / 30B-A3B (DeepStack native) + LLaVA-NeXT baseline (no DeepStack) |
| Dataset | MMMU / DocVQA / ChartQA / VideoMME / interleaved multi-image |
| Tools | vLLM-Jetson fork 4-stage sub-graph scheduler + NVDLA SDK + Nsight Systems + Jetson INA3221 |
| Ablation | 2^3: M1 × M2 × M3 + deepstack_visual_indexes sweep {[4,12,20], [8,16,24], [12,18,24]}. 9 baseline (peer 67%): DeepStack NeurIPS'24, CLI 2601.10710, Nova, DuetServe, vLLM EPD, CacheFlow, FastVLM, LiteVLM, TensorRT-LLM Edge. Runtime 12주. |

#### 2.2.5 예상 효과

- TTFT 1.3-1.6× (image workload multi-image batch)
- Energy -18~28%
- **Scoring**: Nov 6.6→7.0 (CLI cite) / Diff 7.5 / Imp 8.06 / Feas 7.8 → **avg 7.60**

---

### 2.3 Top 3 — **Vault'** (avg 7.30, Accept post-Major Revision, ASPLOS 2027 / HPCA 2027)

#### 2.3.1 개요 (Major Revision 사유 명시)

**"Vault"** = 금고. 30B-A3B 의 128 expert 를 Thor LPDDR5X 에 보관 + top-2 gating 으로 필요한 2개만 꺼내 사용 — expert storage + activation-aware pinning 은유.

**Phase 2 novelty 5.4**: VEQ [arXiv:2602.01037] + ARCQuant [arXiv:2601.07475] + DyMoE [arXiv:2603.19172] + CC-MoE [arXiv:2509.25689] 4-way scoop 68-72%. NVFP4 per-expert 축 포기하고 **DeepStack vision residual × MoE expert L2 contention** 이라는 VLM-MoE × edge 교집합 공백으로 재설계.

#### 2.3.2 기존 연구 한계점 및 Gap

- [VEQ arXiv:2602.01037](https://arxiv.org/abs/2602.01037) — Qwen3-VL MoE modality-adaptive quant (직접 scoop, axis repositioning 필요).
- [ARCQuant arXiv:2601.07475](https://arxiv.org/abs/2601.07475) — NVFP4 전용.
- [DyMoE arXiv:2603.19172](https://arxiv.org/abs/2603.19172) — edge MoE mixed-precision.
- [CC-MoE arXiv:2509.25689](https://arxiv.org/abs/2509.25689) — communication-compute MoE.
- **DeepStack × MoE L2 contention** — layer 8/16/24 vision residual 주입이 expert L2 residency 를 교란하는 문제는 **공백** (VLM-MoE + edge 교집합 novelty).
- [Jetson Thor forum 2026-02](https://forums.developer.nvidia.com/t/jetson-agx-thor-vllm-26-02-moe-performance-significantly-below-reference-missing-fused-moe-config/364663) — 산업 현안 motivation.

#### 2.3.3 제안 기법 (3 Mechanism, Phase 1' REPLACE M1)

- **M1 (REPLACE)** DeepStack × MoE L2 contention analytical model — vision residual 주입 layer 8/16/24 에서 expert L2 contention 측정, 교란 회피 정책.
- **M2 유지** LPDDR5X bank-aligned expert placement — physical layout, VEQ 축과 orthogonal.
- **M3 유지** Activation-aware L2 pinning + GDN dual working-set zone.

#### 2.3.4 평가·실험 플랜

| 요소 | 상세 |
|---|---|
| Hardware | **Jetson Thor 128GB LPDDR5X (primary, 2026-06 gate)** + Orin AGX fallback (INT4) + RTX 4090 (no NVFP4) |
| Model | Qwen3-VL-30B-A3B (128 expert) + Qwen3-Next variants |
| Dataset | MMMU / DocVQA / ChartQA / multi-image 2-tenant |
| Tools | vLLM Thor fork fused MoE config + CUPTI L2 residency + NVML + INA3221 |
| Ablation | 2^3: M1 L2 contention × M2 bank-aligned × M3 activation-aware pinning + GDN. 8 baseline (peer 50%): VEQ, DyMoE, ARCQuant, CC-MoE, Four Over Six 2512.02010, DynaExq 2511.15015, HOBBIT, OD-MoE. Runtime 14주 (Thor gate). |

#### 2.3.5 예상 효과

- MoE decode 1.2-1.4×
- Energy -20~30%
- **Scoring**: Nov 6.8 (post-replacement) / Diff 8.5 / Imp 8.28 (flagship) / Feas 5.6 → **avg 7.30**
- Impact 8.28 flagship, feasibility 는 Thor DevKit 확보 gate.

---

## 3. Tier-2 독립 Top 2 (T3 DROP — Top 3 → Top 2)

### 3.1 **Gale** (avg 6.70, IEEE ESL 4p / ISLPED 6p)

#### 3.1.1 개요

**"Gale"** = 강풍, 256K long context 지속 흐름 + GDN:Attention 3:1 hybrid dual wind direction 은유.

Qwen3-Next/3.5 의 3:1 GDN:Attention ratio 에서 GDN 24 layer 는 constant-memory recurrent state → KV paging 제외. Attention 8 layer 만 256K KV 3-tier (GPU/DRAM/NVMe) + DeepStack layer priority eviction.

#### 3.1.2 GAP 및 Mechanism

- [Gated Delta Net ICLR'25 arXiv:2412.06464](https://arxiv.org/abs/2412.06464) — algorithm, system-level edge 적용 없음.
- [TTKV arXiv:2604.19769](https://arxiv.org/abs/2604.19769) — 2-tier, 3-tier 공백.
- [Kimi Linear + KDA](https://arxiv.org/abs/2604.15804) — 3:1 ratio 축 overlap 있으나 edge Thor LPDDR5X specific 차별화.
- **Single mechanism + 1 ablation**: GDN constant-memory fast path + 256K 3-tier DeepStack-aware eviction.

#### 3.1.3 예상 효과

- KV memory -75%
- Decode TPOP -37~50% (long context)
- **Scoring**: Nov 6.2 / Diff 6.5 / Imp 7.02 / Feas (estimated) 7.0 → **avg 6.70**

### 3.2 **Forge** (avg 6.50, IEEE CAL 4p / DAC 6p)

#### 3.2.1 개요

**"Forge"** = 대장간. Tensor Core (Thinker) + DLA (Talker Code2Wav) + L2 (GDN state) 이기종 재료 단조 은유.

Qwen3.5-Omni Thinker (MoE+GDN LLM) → Tensor Core, Talker Code2Wav causal ConvNet → Jetson DLA, GDN recurrent state → L2 resident.

#### 3.2.2 GAP 및 Mechanism

- [Qwen3.5-Omni Tech Report arXiv:2604.15804](https://arxiv.org/abs/2604.15804) (본 세션 1일 전 공개) — Light variant module on-demand load 공식화 → **빠른 2026-05 CAL submit 권고**.
- vLLM-Omni multi-GPU → single-Jetson Code2Wav DLA 실기 novelty.
- **Single mechanism**: Thinker/Talker heterogeneous mapping + GDN L2 residency.

#### 3.2.3 예상 효과

- First-packet latency <200ms (현재 Qwen3.5-Omni-Flash 235ms 대비 -15~30%)
- Energy -15~20%
- **Scoring**: Nov 5.9 / Diff 7.0 / Imp 6.68 / Feas 6.8 → **avg 6.50**

---

## 4. 미선정 아이디어 (DROP + 재방문 조건)

### 4.1 T3 Echo — Video Timestamp Hash for DeepStack Feature Dedup (DROP)

- **연구 GAP (의도)**: Qwen3-VL text-based video timestamp token 을 hash key 로 DeepStack 3-level visual feature dedup (VLCache 확장).
- **Metaphor**: "Echo" = 메아리, 반복 frame visual feature 재활용.
- **미선정 사유**:
  1. [VLCache arXiv:2512.12977](https://arxiv.org/abs/2512.12977) (2025-12-15) — 2% vision + 98% reuse + pixel hash + encoder cache. **72-78% direct scoop**.
  2. [CodecSight arXiv:2604.06036](https://arxiv.org/abs/2604.06036) — NVDEC motion vector + RoPE-correction KVC refresh (본 세션 16일 전).
  3. [STC arXiv:2512.00891](https://arxiv.org/abs/2512.00891) — temporal cache 축.
  4. 이전 2026-04-23 edge-vlm-energy 세션 **Tidal (DROP)** 과 60-68% 재진입 공간.
- **재방문 조건**:
  1. **Cross-request timestamp sharing** 등 완전 새 축 재설계 — 단일 request 내 dedup 은 VLCache 에 포섭됨.
  2. Tokenizer-deterministic hash 의 formal property 분석 (VLCache 의 continuous pixel hash 와 수학적 차별화).
  3. DeepStack 3-level feature 중 특정 level 만 timestamp-conditional dedup 하는 axis 분리.

### 4.2 원본 12 ideas 통합 흡수

| Original | 통합 Idea | 합병 사유 |
|---|---|---|
| ai-opt RESIDUA | **Mangrove** (M3 DLA offload) | DeepStack layer scheduling 3-way |
| ai-opt PRISMATIC | **Loom** (M1 unified LUT + M2 FA3 fused) | Interleaved MRoPE LUT 3-way |
| ai-opt DELTANET TIDE | **Gale** (M1 GDN constant-mem) | GDN:Attn hybrid KV 2-way |
| ai-opt EXPERT MURMUR | **Vault'** (M3 activation-aware L2 pin) | MoE on Thor 3-way |
| ai-opt TEMPO BEACON | **Echo** (DROP) | Video timestamp dedup |
| legacy BankWeave | **Mangrove** (M2 LPDDR bank) | DeepStack 3-way |
| legacy Prism Lanes | **Loom** (M3 texture unit + register pack) | Interleaved MRoPE 3-way |
| legacy HybridPond | **Vault'** (M2 bank placement + GDN dual working-set) | MoE 3-way |
| legacy TierShelf | **Gale** (M2 3-tier DeepStack eviction) | 256K KV 2-way |
| hw-pim Estuary Scheduler | **Mangrove** (M1 4-stage sub-graph) | DeepStack 3-way |
| hw-pim Harmonic Table | **Loom** (M3 permuted LUT + texture) | Interleaved MRoPE 3-way |
| hw-pim NVFP4 Expert Vault | **Vault'** (M1 replaced — NVFP4 scoop) | MoE 3-way → Major Revision |
| hw-pim Dual-Engine Echo | **Forge** | Thinker/Talker 독립 |

---

## 5. 이 세션의 독특한 점

| 축 | 본 세션의 특이점 |
|---|---|
| **완전 신규 VLM/LLM** | Qwen3-VL (2025-11) + DeepStack (NeurIPS 2024) + Qwen3.5-Omni (2026-04-22) — 최신 3 paper 의 아키텍처 diff 를 mechanism 에 직접 활용. |
| **R23 Step 0-α 신규 규칙 첫 적용** | IISWC + ISPASS + MLPerf/Jetson benchmark report 우선 탐색 (15편 수집). |
| **이전 세대로 불가능한 mechanism** | Mangrove M1 4-stage sub-graph = DeepStack multi-layer 구조에서만 가능 / Loom M1 unified LUT = Interleaved MRoPE frequency interleave 에서만 가능 / Vault' M1 L2 contention = DeepStack × MoE 교집합. |
| **산업 현안 motivation** | Jetson Thor + vLLM MoE performance gap forum (2026-02) 을 Vault' 의 직접 동기. |
| **이전 세션 완전 독립** | 2026-04-23 edge-vlm-energy (Parquet/Triptych/Cartographer/Sift/Verge/Tidal) 와 metaphor + mechanism 모두 독립 (T3 Echo Tidal 재진입으로 DROP). |
| **Major Revision 정면 수용** | I3 Vault NVFP4 scoop 을 receipt 하고 DeepStack×MoE L2 contention 으로 repositioning — scope bloat 아닌 critical gap 방어. |
| **T3 DROP 의 정직한 선택** | VLCache 72-78% scoop + Tidal 재진입 → Tier-2 Top 3 → Top 2 축소 수용. |

---

## 6. 다음 단계 제안

1. **Loom CUTLASS prototype (1-week PoC)** — lead candidate, SFU busy % 실측 + FA3 fused rotation 검증.
2. **Mangrove DLA transformer offload feasibility** — Jetson Orin AGX DLA-Next spec 확인.
3. **Vault' Thor DevKit 확보 gate (2026-06)** — 미확보 시 Orin AGX INT4 fallback.
4. **Forge 2026-05 CAL submit** — Qwen3.5-Omni Tech Report 1일 전 공개로 빠른 precedence 확보 필요.
5. **T3 Echo 재방문 세션** — cross-request timestamp sharing 등 완전 새 축 재설계 시.
6. **Homepage publish 요청 시** summary publish (명시 요청 시만).

---

## 7. 참고 파일

- **Session 상세**: [sessions/2026-04-24-mode1-qwen3vl-deepstack-edge.md](../sessions/2026-04-24-mode1-qwen3vl-deepstack-edge.md)
- **Staging**:
  - [ai-opt expert](../sessions/staging/2026-04-24-qwen3vl-deepstack-edge-aiopt-expert.md) (626줄)
  - [legacy-sys expert](../sessions/staging/2026-04-24-qwen3vl-deepstack-edge-legacy-sys-expert.md) (528줄)
  - [hw-pim expert](../sessions/staging/2026-04-24-qwen3vl-deepstack-edge-hwpim-expert.md) (543줄)
  - [Phase 1 integration](../sessions/staging/2026-04-24-qwen3vl-deepstack-edge-phase1-integration.md)
  - [Phase 2 novelty](../sessions/staging/2026-04-24-qwen3vl-deepstack-edge-phase2-novelty.md)
  - [Phase 2 differentiation](../sessions/staging/2026-04-24-qwen3vl-deepstack-edge-phase2-diff.md)
  - [Phase 2 impact](../sessions/staging/2026-04-24-qwen3vl-deepstack-edge-phase2-impact.md)
  - [Phase 2 similarity](../sessions/staging/2026-04-24-qwen3vl-deepstack-edge-phase2-similarity.md)
  - [Phase 1'/2'/1''](../sessions/staging/2026-04-24-qwen3vl-deepstack-edge-phase1prime-2prime-1primeprime.md)
