# VLM Serving × Speculative Decoding × Chunked Prefill — CPU-GPU Cooperation + Energy + Accuracy

> **세션 일자**: 2026-05-26 · **Mode**: 1 (deep-dive single-topic) · **참여 expert**: ai-optimization + algorithm + legacy-system · **참여 reviewer**: novelty + differentiation + impact + ai-implementation + arch-system-implementation (5인)

VLM serving 에서 (a) draft 의 visual-aware accuracy, (b) chunked prefill 의 modality-boundary 정합성, (c) CPU-GPU 협업 (vision encode CPU offload / draft placement / KV tier), (d) energy 최적화 (3-stage DVFS), (e) accuracy 보존 (KL bound + lossless quantization) 의 5축 결합 최적화 idea 도출. Phase 1 expert × 3 → 20 raw idea → Phase 2' 17 정식 평가 idea → Top-M 6 (Tier-1 3 + Tier-2 3) + 1 paper pair.

---

## 1. Research Questions (R28.1.1)

### Master RQ

**MLSys/ASPLOS/MICRO/NeurIPS 2027 진입 가능한 single-machine (RTX Pro 6000 96GB OR RTX 5090 32GB OR datacenter 8×H100) VLM serving 의 5축 결합 최적화 (SpecDec + Chunked Prefill + CPU-GPU + Energy + Accuracy) idea 6 개를 어떻게 도출할 것인가?**

- **RQ-Master-1 (Accuracy bound)**: VLM draft 와 target 의 modality-marginal KL divergence ε 의 closed-form bound (Pinsker `TV ≤ √(KL/2)` + BH `TV ≤ √(1 - e^{-KL})`) 가 acceptance rate α 의 하한을 ε ≤ 0.01 nats 환경에서 ≥ 0.65 로 보장하는가? (B9' direct check)
- **RQ-Master-2 (Memory enabling)**: Visual KV cache 의 1.58-bit ternary 양자화 + EAGLE-shared first-1-2 layer 가 modality-asymmetric Hessian (visual ≤ 25% of text) 환경에서 Memory -50% + accuracy drop < 0.5pp 동시 만족하는가? (B11' direct check)
- **RQ-Master-3 (System throughput)**: Blackwell SM120 TBC `__cluster_dims__(8,1,1)` distributed shmem + pinned triple-buffer + 3-stream priority 가 prefill/decode/verify cross-phase overlap 으로 throughput +20-28% + HBM3e BW eff +22-30% 달성하는가? (C4L' direct check)

### 6 idea 별 RQ (1줄 each)

- **Tier-1 #1 B9'**: VLM draft loss 에 modality-conditional KL term 추가 시 ε ≤ 0.01 nats 보장 + serving 변경 0 + universal orthogonal infra 가능한가?
- **Tier-1 #2 B11'**: 1.58-bit visual KV ternary + 4-bit text KV asymmetric + Marlin sub-2-bit kernel + B14' embedded calibration 으로 RTX 5090 32GB enabling 가능한가?
- **Tier-1 #3 C4L'**: TBC cluster-8 + pinned triple-buffer + 3-stream priority 의 cross-phase overlap 이 ASPLOS/MICRO/ISCA target throughput delivers 가능한가?
- **Tier-2 #1 A6'**: Cross-attn saliency tree EWMA + PWL(`H_v`) dynamic `num_spec` 가 wasted draft compute -35% + acceptance +6pp 가능한가? (300 LOC)
- **Tier-2 #2 C5L'**: `cudaAccessPolicyWindow` L2 set-aside 25-40% + acceptance-aware SM partition 이 vLLM 의 high-QPS slowdown (1.4-1.8×) 회복 가능한가?
- **Tier-2 #3 A2'**: Datacenter GPU saturation 환경 (vLLM b6553be1 + H100 8-batch) 에서 CPU AMX/AVX-512 1B INT8 draft + visual encode hiding 이 HiViS GPU drafter 와 orthogonal (≥10% incremental gain) 인가?

---

## 2. Essential Reading (R28.1.2, R15-α — 3-5편)

학생이 본 세션 idea 구현 시작 전 반드시 읽어야 할 paper (5편):

| Paper | Venue | 왜 읽어야 하는가 (1-2줄) | Idea 분담 |
|---|---|---|---|
| [SpecVLM: Fast Speculative Decoding in VLMs (arXiv:2509.11815)](https://arxiv.org/abs/2509.11815) | arXiv 2025-09 (AMD 추정) | EagleVLM baseline 대비 1.5-2.3× → 2.5-2.9× (5 epoch online distillation). VLM prefill 의 visual token dominance 와 acceptance rate 의 직접 evidence. B9' / B11' 의 차별화 axis (modality KL, Hessian asymmetry) 의 baseline | B9', B11', A6' |
| [CoVSpec: Device-Edge Co-Inference for VLMs via SpecDec (arXiv:2605.02218)](https://arxiv.org/abs/2605.02218) | arXiv 2026-05 | 본 세션 mechanism 과 70%+ 충돌 가능 paper. Chunked prefill 명시 통합 부재 + energy 미명시 + device-edge 가정 → 본 세션은 single-machine CPU-GPU + chunked prefill + energy 의 3-axis 차별화 | A2', A6', C5L' (scoop defense) |
| [Sarathi-Serve: Taming Throughput-Latency Tradeoff (arXiv:2403.02310)](https://arxiv.org/abs/2403.02310) | OSDI 2024 | Chunked prefill stall-free scheduling 의 production baseline. Mistral-7B 2.6× / Yi-34B 3.7× / Falcon-180B 5.6×. C4L' / A1' / C1L' / C5L' 의 chunked prefill invariant (token budget 정합) 공유 | C4L', A1', C5L' |
| [Cost-Efficient Multimodal LLM Inference via Cross-Tier GPU Heterogeneity (HeteroServe, arXiv:2603.12707)](https://arxiv.org/abs/2603.12707) | arXiv 2026-03 | Modality boundary 에서 partition 시 KV (GB-scale) → embedding (MB-scale) 전송, PCIe 충분, 37% Tokens/$ 향상. A2' / B10' / C2L' (CPU vision offload) 의 direct motivation + 정량 근거 | A2', B10' (merge candidate) |
| [EAGLE: Speculative Sampling Rethinking Feature Uncertainty (arXiv:2401.15077)](https://arxiv.org/abs/2401.15077) | ICML 2024 | Feature-level autoregression 의 2.7-3.5× speedup + 3.2-4.5 tok/forward. B9' KL bound 의 mathematical foundation + B11' EAGLE-shared first 1-2 layer scope 의 architectural constraint | B9', B11', A6' |

추가 참고 (papers.md 전체 기록): [SpecVLM-Video (arXiv:2508.16201)](https://arxiv.org/abs/2508.16201), [Spec-LLaVA (arXiv:2509.11961)](https://arxiv.org/abs/2509.11961), [ViSpec (arXiv:2509.15235)](https://arxiv.org/abs/2509.15235), [MASSV (arXiv:2505.10526)](https://arxiv.org/abs/2505.10526), [Nova (arXiv:2509.21301)](https://arxiv.org/abs/2509.21301), [Dovetail (arXiv:2412.18934)](https://arxiv.org/abs/2412.18934), [VLMOpt (arXiv:2604.26334, MLSys 2026 Industry)](https://arxiv.org/abs/2604.26334), [VoltanaLLM (arXiv:2509.04827)](https://arxiv.org/abs/2509.04827), [GreenLLM (arXiv:2508.16449)](https://arxiv.org/abs/2508.16449), [EnergyLens-Multimodal (arXiv:2605.10556)](https://arxiv.org/abs/2605.10556), [POD-Attention (arXiv:2410.18038)](https://arxiv.org/abs/2410.18038), [Mooncake (arXiv:2407.00079)](https://arxiv.org/abs/2407.00079), [Splitwise (arXiv:2311.18677)](https://arxiv.org/abs/2311.18677), [DistServe (arXiv:2401.09670)](https://arxiv.org/abs/2401.09670), [FlexGen (arXiv:2303.06865)](https://arxiv.org/abs/2303.06865), [Llama-3.2-Vision (arXiv:2504.00557)](https://arxiv.org/abs/2504.00557), [SparseVILA (arXiv:2510.17777)](https://arxiv.org/abs/2510.17777).

---

## 3. 연구 개요 + GAP outline (R28.1.3)

### 3.1 연구 개요

- **VLM serving 의 3-phase taxonomy 확립**: LVLM Survey ([arXiv:2604.05546](https://arxiv.org/abs/2604.05546), ACL 2026 Findings) 에 따라 (1) Encoding compute-bound, (2) Prefilling quadratic complexity, (3) Decoding memory wall 의 3-regime 으로 정의됨
- **2025-2026 VLM SpecDec 의 폭발적 등장**: SpecVLM / ViSpec / MASSV / Spec-LLaVA / FastVLM 5편 (2025 Q3-Q4) — visual token KV cache 가 draft latency 의 1차 bottleneck 명시
- **Chunked Prefill 의 production-grade standardization**: vLLM 2024 retrospective 에서 chunked prefill + speculative decoding 의 **결합 최적화** 가 2025 priority 로 공식 선언 ([vLLM 2024 retrospective blog](https://blog.vllm.ai/2025/01/10/vllm-2024-wrapped-2025-vision.html))
- **CPU-GPU 협업의 production-scale evidence**: VLMOpt (NVIDIA, MLSys 2026 Industry) Cosmos-Reason1 의 10× VRAM 감소 + 6.7× TTFT 개선; HeteroServe 의 modality boundary partition 시 PCIe 만으로 충분
- **CPU LLM Inference 의 경쟁력 입증**: IISWC 2024 CPU LLM Inference (Georgia Tech) — AMX/AVX-512 활용 시 작은 batch CPU draft 가 GPU-경쟁 가능 — A2' Tortoise & Hare 의 direct motivation
- **VLM 의 visual token 비중**: SparseVILA / SpecVLM / Llama-3.2-Vision 모두 visual token 이 prefill latency 의 dominant 부분임을 정량 — visual-aware chunking + draft acceptance 의 axis 가 미개척
- **Energy 의 30-48% 감소가 새 baseline**: VoltanaLLM / GreenLLM / DualScale 의 phase-aware DVFS — 단 vision encode phase 미통합 (A3' Power Grid 의 motivation)
- **CPU-GPU coupled architecture 의 ISPASS 2025 evidence**: GH200 closely-coupled prefill 이 PCIe A100/H100 대비 1.9-2.7× ([arXiv:2504.11750](https://arxiv.org/abs/2504.11750)) — 단 GH200 도 4× batch 까지 CPU-bound — CPU draft 의 motivation
- **Blackwell TBC distributed shared memory**: NVIDIA Blackwell Tuning Guide 의 portable cluster size 8 confirmation — C4L' / C1L' 의 직접 활용
- **Sub-2-bit quantization 의 modality-asymmetric novelty**: KIVI/KVQuant/AsymKV 가 modality 미구분 + 2-bit floor — B11' 의 4-axis 차별화 (sub-2-bit / modality-asym / EAGLE-shared / Hessian)

### 3.2 GAP outline (Step 0 verified)

- **GAP-1 (VLM SpecDec accuracy bound 부재)**: SpecVLM / ViSpec / MASSV 모두 acceptance rate empirical 측정만 — modality-marginal KL 의 formal bound (Pinsker / BH dual) 부재 — `cite: [arXiv:2509.11815](https://arxiv.org/abs/2509.11815), [arXiv:2509.15235](https://arxiv.org/abs/2509.15235), [arXiv:2505.10526](https://arxiv.org/abs/2505.10526)`
- **GAP-2 (Visual KV sub-2-bit quantization)**: KIVI/KVQuant/AsymKV/QuantSpec 모두 modality-asymmetric Hessian + sub-2-bit 의 4-axis combination 부재 — `cite: [arXiv:2510.22641](https://arxiv.org/abs/2510.22641), QuantSpec related`
- **GAP-3 (Chunked Prefill × SpecDec cross-phase overlap)**: Sarathi-Serve / POD-Attention / Mooncake 모두 single-phase SM optimization 만 — TBC distributed shmem × pinned triple-buffer × 3-stream priority 의 cross-phase combination 부재 — `cite: [arXiv:2403.02310](https://arxiv.org/abs/2403.02310), [arXiv:2410.18038](https://arxiv.org/abs/2410.18038), [arXiv:2407.00079](https://arxiv.org/abs/2407.00079)`
- **GAP-4 (Visual-aware dynamic draft tree depth)**: Spec-LLaVA / ViSpec 의 dynamic tree 가 image content 와 무관 — cross-attn saliency 기반 dynamic `num_spec` 미개척 — `cite: [arXiv:2509.11961](https://arxiv.org/abs/2509.11961), [arXiv:2509.15235](https://arxiv.org/abs/2509.15235)`
- **GAP-5 (High-QPS slowdown recovery)**: vLLM spec decode 의 high-QPS 1.4-1.8× slowdown 회복 mechanism 부재 — L2 persistent × SM partition × POD-Attention CTA fusion combination 부재 — `cite: [vLLM blog 2024-10-17](https://vllm.ai/blog/2024-10-17-spec-decode), [arXiv:2410.18038](https://arxiv.org/abs/2410.18038)`
- **GAP-6 (Datacenter CPU draft placement)**: Dovetail (consumer GPU) + HiViS (GPU drafter) 의 third axis (datacenter CPU placement) 부재 — `cite: [arXiv:2412.18934](https://arxiv.org/abs/2412.18934), HiViS`
- **GAP-7 (Vision encode DVFS phase 통합)**: VoltanaLLM / GreenLLM / DualScale 의 phase-aware DVFS 가 vision encode 미통합 — 3-stage (encode/prefill/decode) 부재 — `cite: [arXiv:2509.04827](https://arxiv.org/abs/2509.04827), [arXiv:2508.16449](https://arxiv.org/abs/2508.16449), [arXiv:2602.18755](https://arxiv.org/abs/2602.18755)`
- **GAP-8 (Multimodal energy model 에 acceptance rate parameter)**: EnergyLens 12-param closed-form 이 spec-decode acceptance rate 미모델링 — `cite: [arXiv:2605.10556](https://arxiv.org/abs/2605.10556)`
- **GAP-9 (Long-context VLM KV tier hierarchy)**: Mooncake KVCache-centric 의 single-machine + multi-tier (HBM/DDR5/NVMe Gen5) 변종 부재 — `cite: [arXiv:2407.00079](https://arxiv.org/abs/2407.00079)`
- **GAP-10 (Region-atomic chunk × spec window)**: Sarathi-Serve stall-free 가 visual region split → spec acceptance window desync — `cite: [arXiv:2403.02310](https://arxiv.org/abs/2403.02310)`

### 3.3 GAP → Idea Mapping

| GAP | Tier-1 | Tier-2 | Phase 3 elective |
|---|---|---|---|
| GAP-1 (Accuracy bound) | **#1 B9' KL-Bounded Distillation** | — | — |
| GAP-2 (Sub-2-bit visual KV) | **#2 B11' Sub-2-bit Visual KV** (+ B14' embedded) | — | — |
| GAP-3 (Cross-phase overlap) | **#3 C4L' KV-Stream-TBC** | — | C1L' VTAP-Sched (HW layered partner) |
| GAP-4 (Visual saliency dynamic γ) | — | **#4 A6' VAST-Sched** | B12'/B8' VAST-γ merged |
| GAP-5 (High-QPS slowdown recovery) | — | **#5 C5L' SpecVerify-L2** | — |
| GAP-6 (Datacenter CPU draft) | — | **#6 A2' Tortoise & Hare** | — |
| GAP-7 (Vision encode DVFS) | — | — | A3' Power Grid Conductor |
| GAP-8 (Energy model acceptance) | — | — | A3' / B12' coupling |
| GAP-9 (KV tier hierarchy) | — | — | **C6L' UnifiedKV-NVMe (rank 1)** |
| GAP-10 (Region-atomic chunk) | — | — | A1' Mosaic Maestro (Phase 3 partner) |

---

## 4. Implementation-Priority Decision Tree (R14.4 양식 A)

```
[Step 0: 학기 시작 — 1 학생 + hardware 확인]
  │
  ├─→ RTX Pro 6000 96GB 보유?
  │     ├─ Yes → Phase 1'' pilot 가능 (모든 Top-M 6)
  │     ├─ Only RTX 5090 32GB → B11' (1.58-bit visual KV enabling), C4L' (TBC SM120) 만 + Tier-2 A6'/C5L'
  │     └─ Only datacenter 8×H100 → A2' Tortoise & Hare 우선 (HiViS 비교)
  │
  ├─→ AMX 보유 (SPR/GNR Intel) 또는 AMD Zen 5 AVX-512 VNNI BF16?
  │     ├─ AMX (Intel SPR/GNR) → A2' Path A (CPU draft) + C2L' Path A (vision offload)
  │     ├─ AMD Zen 5 → A2' Path B (AVX-512 VNNI BF16 fallback) + C2L' Path B
  │     └─ Neither → A2' DROP, C2L' DROP (혹은 Tier-2 simulator gem5 만)
  │
  ▼
[Step 1: Phase 1'' Pilot — Week 1-2]
  │
  ├─→ NVML calibration (A3', C1L', C4L')
  │     ├─ `nvmlDeviceGetSupportedGraphicsClocks` 출력 ≥10 clocks → Pass
  │     ├─ ≤5 clocks → DVFS sub-mechanism scope 축소
  │     └─ 0 clocks → A3' DROP, C4L' DVFS sub-mechanism 제거
  │
  ├─→ TBC kernel PoC (C4L', C1L')
  │     ├─ `__cluster_dims__(8,1,1)` numerical correctness → Pass
  │     ├─ Cluster size 4 only → C4L' partial scope
  │     └─ TBC unsupported on hardware → C4L' DROP
  │
  ├─→ HiViS head-to-head (A2' gate)
  │     ├─ ≥10% incremental gain → A2' Pass
  │     ├─ 5-10% → Tier-2 reposition (DAC/DATE 6p)
  │     └─ <5% → A2' DROP
  │
  ├─→ MRoPE PPL gate (B13' if pursued)
  │     ├─ PPL < 1.0% → Conditional Pass
  │     ├─ 1.0% < PPL < 1.5% → "cross-chunk only" reposition
  │     └─ PPL ≥ 1.5% → B13' DROP
  │
  ├─→ cudaAccessPolicyWindow set-aside 측정 (C1L', C5L')
  │     ├─ Verify kernel L2 hit ratio +20pp → Pass
  │     ├─ +5-20pp → C5L' Tier-2 narrow scope 유지
  │     └─ <5pp → C5L' DROP
  │
  ▼
[Step 2: Tier-1 main paper 작성 — Week 3-8]
  │
  ├─→ Week 3-4: B9' KL-Bounded Distillation training pipeline
  │     ├─ ε ≤ 0.01 nats 달성 → Phase 3 main paper
  │     ├─ 0.01 < ε ≤ 0.02 → λ tuning 재실행
  │     └─ ε > 0.02 → Cosine similarity proxy fallback
  │
  ├─→ Week 5-6: B11' Sub-2-bit Visual KV (B14' embedded)
  │     ├─ Marlin sub-2-bit kernel + Hessian ratio ≤ 25% → Phase 3 main paper
  │     ├─ Hessian ratio 25-40% → 2-bit visual fallback (B11'-Lite)
  │     └─ Hessian ratio > 40% → scope narrow (specific VLM model)
  │
  └─→ Week 7-8: C4L' KV-Stream-TBC 통합 + Pinned triple-buffer + 3-stream priority
        ├─ Throughput +20-28% 달성 → Phase 3 main paper
        ├─ +10-20% → mechanism 축소 (TBC + 3-stream only)
        └─ <+10% → Tier-2 reposition
  │
  ▼
[Step 3: Tier-2 paper 작성 — Week 9-10]
  │
  ├─→ A6' VAST-Sched (300 LOC, B12' propose signature 공유) → DAC/DATE 6p
  ├─→ C5L' SpecVerify-L2 (L2 persistent + SM partition) → DAC/DATE/ICCAD
  └─→ A2' Tortoise & Hare (datacenter CPU draft, HiViS head-to-head 통과 시) → DAC/MLSys-Industry
  │
  ▼
[Step 4: Cross-share integration + benchmark sweep — Week 11-14]
  │
  ├─→ B9' draft + B11' quantized KV + C4L' system 결합 evaluation
  ├─→ MMMU/MMBench/Video-MME/LLaVA-Bench/OCRBench 5-axis benchmark sweep
  └─→ R55.2 5-axis Gain Target ≥ 20% verify
  │
  ▼
[Step 5: Paper writing + Phase 3 elective 진입 — Week 15-16]
  │
  └─→ Phase 3 elective priority (Tier-1 추가): C6L' > C1L' > VAST-γ merged > A3' > C2L'
```

### 6 idea × 3 Phase × 4 Branch 액션 표

| Idea | Phase 1'' Pilot (Week 1-2) | Phase 1''' Main (Week 3-10) | Phase 3 Benchmark (Week 11-14) |
|---|---|---|---|
| **B9'** | NVML/TBC 불필요, training infra 셋업 | KL term + Pinsker + BH dual bound 학습 (5 epoch) | LLaVA-Bench + MMMU acceptance rate +12pp ↑ |
| **B11'** | Hessian ratio 측정 (128 sample × 5 task) | Marlin sub-2-bit kernel + 1.58-bit ternary | RTX 5090 32GB Llama-3.2-Vision-11B accuracy < 0.5pp |
| **C4L'** | TBC kernel PoC + 3-stream priority test | TBC + Pinned + 3-stream priority 통합 | Throughput +20-28% + HBM3e BW +22-30% |
| **A6'** | EWMA + PWL signature 공유 test (B12' 와) | 300 LOC + dynamic `num_spec` | Wasted draft compute -35% + acceptance +6pp |
| **C5L'** | `cudaAccessPolicyWindow` set-aside fraction 실측 | Acceptance-aware SM partition + L2 persistent | High QPS +24-41% slowdown recovery |
| **A2'** | HiViS head-to-head (datacenter) | CPU draft worker + visual encode pre-warm | Throughput +10-15% (datacenter only) |

Branch action codes: **Pass** (target 달성 → 진행), **Partial** (50-80% 달성 → scope 축소 진행), **Reposition** (Tier-2 reposition), **DROP** (target 미달성, idea 폐기).

---

## 5. Tier-1 Top 3 (R15-β contribution bullet)

| Idea | Score | Domain | Venue target |
|---|---|---|---|
| **#1 B9' KL-Bounded VLM Draft Distillation v2** | 8.55 | Algorithm / Training | NeurIPS / ICML 2027 |
| **#2 B11' Sub-2-bit Visual KV Asymmetric Quantization** | 8.40 | Algorithm / Quantization | MLSys / ASPLOS / NeurIPS 2027 |
| **#3 C4L' KV-Stream-TBC v2** | 8.40 | System / GPU Kernel | ASPLOS / MICRO / ISCA 2027 |

### Tier-1 #1: B9' KL-Bounded VLM Draft Distillation v2 (R15-β 4-6 bullets)

- **(a) Mechanism 정성적 benefit**: Pinsker `TV ≤ √(KL/2)` + Bretagnolle-Huber `TV ≤ √(1 - e^{-KL})` 의 dual bound 가 modality-marginal KL ε ≤ 0.01 nats 환경에서 acceptance rate α 의 closed-form 하한을 보장 (영역별 tight bound 선택). VLM serving 의 vLLM b6553be1 변경 0 + training-only orthogonal infra 로 모든 다른 system idea (B11'/C4L'/A6'/C5L'/A2') 와 자유 결합 가능.
- **(b) Closest competitor**: SpecVLM ([arXiv:2509.11815](https://arxiv.org/abs/2509.11815)) — 차별화 axis: SpecVLM 은 online-logit distillation 의 empirical training-time scaling 만 측정, modality-marginal KL 의 formal bound 부재. 본 idea 는 dual bound 의 grid-search λ tuning + λ sensitivity analysis 추가.
- **(c) 예상 gain 정량 표**:

  | Metric | Baseline (SpecVLM) | B9' Target | Delta |
  |---|---|---|---|
  | Acceptance rate α | 0.53 (text-only baseline) | ≥0.65 | +12pp |
  | Draft training time | 1× | 2.3× (5 epoch) | training-only overhead |
  | Accuracy drop (MMMU) | 0pp (lossless) | 0pp (lossless) | maintained |
  | Serving runtime change | n/a | 0 LoC | universal orthogonal |

- **(d) Tier 강등 risk**: ε > 0.02 nats 시 bound 가 vacuous → cosine similarity proxy fallback (Tier-2 reposition). Pinsker/BH dual bound 의 grid-search λ 가 5 epoch 내 수렴 못하면 epoch 늘리거나 partial scope (text-only first, visual 나중).
- **(e) Outperform 가능성**: NeurIPS/ICML 2027 main track 적합 — universal orthogonal training infra 가 7 Strong Accept idea 중 mean 8.55 최고; Phase 3 paper-pair 의 "free combine partner" 로 모든 다른 idea 의 draft checkpoint 공급.
- **(f) Implementation envelope**: Training-only, single workstation (RTX Pro 6000 96GB) + ZeRO-2 가능, ~600 LOC training script, vLLM serving 변경 0, Phase 1'' pilot risk 없음 (Phase 3 즉시 진입 가능).

### Tier-1 #2: B11' Sub-2-bit Visual KV Asymmetric Quantization (+ B14' embedded)

- **(a) Mechanism 정성적 benefit**: Per-modality Hessian profiling (B14' embedded, ~400 LoC training-free calibration) 로 visual KV 의 Hessian magnitude 가 text KV 의 25% ↓ 검증 후, visual KV 1.58-bit ternary (5-ternary/byte storage layout) + text KV 4-bit asymmetric quantization + EAGLE draft 의 first 1-2 layer KV 공유 (verify 와 동일 양자화 → lossless 보장) + Marlin sub-2-bit CUDA kernel.
- **(b) Closest competitor**: KIVI / KVQuant / AsymKV (modality 미구분, 2-bit floor) + QuantSpec (self-spec + 4-bit hierarchical만). 차별화 4-axis: sub-2-bit / modality-asymmetric / EAGLE-shared first-N-layer / per-modality Hessian.
- **(c) 예상 gain 정량 표**:

  | Metric | Baseline (KIVI 2-bit) | B11' Target | Delta |
  |---|---|---|---|
  | Memory footprint | 100% (16-bit KV baseline 대비 50%) | 50% (16-bit baseline 대비 25%) | -50% |
  | RTX 5090 32GB Llama-3.2-Vision-11B fit | OOM | Fit | enabling |
  | Accuracy drop (MMMU/MMBench) | <0.3pp | <0.5pp | within budget |
  | Throughput (Marlin kernel) | 1× (KIVI) | 1.1-1.2× | side benefit |

- **(d) Tier 강등 risk**: Hessian ratio > 40% 시 visual KV 의 1.58-bit 무효 → 2-bit visual fallback (B11'-Lite, paper pair companion); Marlin sub-2-bit kernel 의 5-ternary/byte SIMD-friendly dequant 구현 risk 시 INT2 path fallback (vLLM existing).
- **(e) Outperform 가능성**: MLSys/ASPLOS/NeurIPS 2027 main track — Memory -50% 가 19 idea 중 magnitude top + RTX 5090 32GB enabling 이 contribution 핵심. B14' calibration framework 가 paper sub-contribution 으로 자연 포함 (별도 paper 불필요).
- **(f) Implementation envelope**: Single workstation (RTX 5090 32GB OR RTX Pro 6000 96GB), Marlin sub-2-bit kernel 신규 (1 CUDA kernel, ~300 lines), B14' Hessian calibration (~400 LoC training-free), vLLM b6553be1 patch ~200 LoC, Phase 1'' Hessian ratio 측정 pilot 의무.

### Tier-1 #3: C4L' KV-Stream-TBC v2

- **(a) Mechanism 정성적 benefit**: Blackwell SM120 의 TBC (Thread Block Cluster) `__cluster_dims__(8,1,1)` distributed shared memory 가 cross-SM data exchange enable; CUDA `cudaStreamCreateWithPriority` 3-stream (prefill / verify / draft) priority scheduling 으로 phase 간 SM occupancy 충돌 해소; pinned host memory triple-buffer 로 H2D/D2H copy 의 async overlap.
- **(b) Closest competitor**: Sarathi-Serve ([arXiv:2403.02310](https://arxiv.org/abs/2403.02310), OSDI 2024) / POD-Attention ([arXiv:2410.18038](https://arxiv.org/abs/2410.18038), ASPLOS 2025) / Mooncake ([arXiv:2407.00079](https://arxiv.org/abs/2407.00079), FAST 2025) / ModServe (SoCC 2025) / VLMOpt (MLSys 2026 Industry) — 모두 single-phase SM optimization 만, cross-phase TBC × stream priority × pinned triple-buffer combination 부재.
- **(c) 예상 gain 정량 표**:

  | Metric | Baseline (Sarathi-Serve OSDI 2024) | C4L' Target | Delta |
  |---|---|---|---|
  | Throughput (chunked prefill + spec decode) | 1× | 1.20-1.28× | +20-28% |
  | HBM3e BW efficiency | 65-70% | 87-95% | +22-30%pp |
  | TBC cluster occupancy | 0 (not used) | 8-SM cluster | new lever |
  | 3-stream priority overhead | n/a | <2% scheduling | acceptable |

- **(d) Tier 강등 risk**: TBC `__cluster_dims__(8,1,1)` numerical correctness 미통과 시 cluster size 4 partial scope (gain -50%); 3-stream priority 가 verify kernel 의 SM occupancy 충돌 시 2-stream fallback.
- **(e) Outperform 가능성**: ASPLOS/MICRO/ISCA 2027 main track — 6 peer-reviewed anchor (reference diversity 10/10, 19 idea 중 최고); 100% Tier-1 application-level (모든 4 mechanism 이 application-level); TBC kernel prototype 통과 시 mechanism novelty 8→9 상승.
- **(f) Implementation envelope**: Single workstation (RTX Pro 6000 96GB Blackwell SM120 필수), vLLM b6553be1 + new kernel `csrc/kv_reduce_tbc8.cu` (~400 LoC) + Pinned triple-buffer 통합 (~200 LoC) + 3-stream priority (~100 LoC), Phase 1'' TBC PoC 의무.

---

## 6. Tier-2 독립 Top 3 (R15-β contribution bullet)

| Idea | Score | Domain | Venue target |
|---|---|---|---|
| **#4 A6' VAST-Sched** (independent) | 7.25 | Application Scheduling | DAC / DATE 2027 (6p short) |
| **#5 C5L' SpecVerify-L2 v2** | 7.95 | GPU L2 / SM partition | DAC / DATE / ICCAD 2027 |
| **#6 A2' Tortoise & Hare v2** | 7.30 | CPU-GPU heterogeneous | DAC / MLSys-Industry 2027 (6p short) |

### Tier-2 #4: A6' VAST-Sched (Visual Heartbeat → Dynamic num_spec)

- **(a) Mechanism 정성적 benefit**: Cross-attention saliency tree H_v 의 EWMA 가 visual region token-level information 변화율 capture; PWL(`H_v`) function 으로 dynamic `num_spec` runtime 재읽기 (high saliency → spec depth ↓ 으로 acceptance 유지).
- **(b) Closest competitor**: Spec-LLaVA / ViSpec 의 dynamic tree (image content 와 무관). CoVSpec parallel branching (sequential adaptive 와 orthogonal).
- **(c) 예상 gain 정량 표**:

  | Metric | Baseline (static γ=4) | A6' Target | Delta |
  |---|---|---|---|
  | Wasted draft compute | 100% | 65% | -35% |
  | Acceptance rate | 0.52 | 0.58 | +6pp |
  | Energy | 100% | 89% | -11% |
  | Implementation LOC | n/a | 300 LOC | minimal |

- **(d) Tier 강등 risk**: Cross-attn saliency export overhead > 1.5ms/step 시 fallback (static γ); B12' 와 propose signature 공유 충돌 시 sequential implementation.
- **(e) Outperform 가능성**: DAC/DATE 6p short paper 적합 (Tier-2 magnitude). B12' VAST-γ merged 시 Tier-1 elective 진입 가능.
- **(f) Implementation envelope**: 300 LOC, 3 file modify (`vllm/v1/spec_decode/eagle.py`, scheduler, ewma_tracker.py 신규), ~1주 implementation budget, single workstation fit.

### Tier-2 #5: C5L' SpecVerify-L2 v2 (Acceptance-Aware SM Partition + L2 Persistent Window)

- **(a) Mechanism 정성적 benefit**: CUDA 11.0+ `cudaAccessPolicyWindow` public API 로 L2 set-aside fraction 25-40% 지정 (verify kernel L2 hit ratio 향상); acceptance-rate online SM partition resize (Green Context 대안); shmem carveout 33-50%; POD-Attention CTA fusion 의 cross-phase 통합.
- **(b) Closest competitor**: vLLM spec decode (high-QPS 1.4-1.8× slowdown, [vLLM blog](https://vllm.ai/blog/2024-10-17-spec-decode)). POD-Attention ASPLOS 2025 (single-phase SM only).
- **(c) 예상 gain 정량 표**:

  | Metric | Baseline (vLLM v0.9.1 high QPS) | C5L' Target | Delta |
  |---|---|---|---|
  | Low QPS throughput | 1× (1.4-1.8× spec gain) | maintained | 0 |
  | Mid QPS throughput | 1× (gain collapse 시작) | 1.20-1.33× | +20-33% |
  | High QPS throughput | 0.55-0.71× (slowdown) | 1.24-1.41× | +24-41% recovery |
  | L2 hit ratio (verify) | 45% | 65-85% | +20-40pp |

- **(d) Tier 강등 risk**: `cudaAccessPolicyWindow` set-aside fraction 효과 미달 (<5pp) 시 DROP; high QPS magnitude scope narrow → Tier-2 venue 정합.
- **(e) Outperform 가능성**: DAC/DATE/ICCAD 6-8p; ASPLOS Tier-1 가능하나 magnitude 측면에서 Tier-2 venue 가 정합. Single-axis 차별화 + production-grade fallback.
- **(f) Implementation envelope**: Single workstation (RTX Pro 6000 96GB 188 SM + 128MB L2), vLLM b6553be1 patch (~500 LoC), Phase 1'' set-aside fraction 실측 의무.

### Tier-2 #6: A2' Tortoise & Hare v2 (Datacenter CPU Draft Spinoff)

- **(a) Mechanism 정성적 benefit**: Datacenter GPU saturation 환경 (vLLM b6553be1 + H100 8-batch) 에서 CPU AMX/AVX-512 1B INT8 draft 가 GPU-경쟁 가능 (IISWC 2024 CPU LLM Inference 검증). Visual encode hiding 으로 PCIe transfer 와 GPU verify overlap. BaseProposer 추상화 PR-level 분리.
- **(b) Closest competitor**: Dovetail ([arXiv:2412.18934](https://arxiv.org/abs/2412.18934), EMNLP 2025) — consumer GPU 3-7GB VRAM, text-only. HiViS — GPU drafter only. 3-way 의 third axis (datacenter CPU placement) 가 본 idea unique.
- **(c) 예상 gain 정량 표**:

  | Metric | Baseline (vLLM H100 8-batch + GPU draft) | A2' Target | Delta |
  |---|---|---|---|
  | Throughput (datacenter, batch ≥8) | 1× | 1.10-1.15× | +10-15% |
  | PCIe transfer overhead | 50-100ms/step | hidden | absorbed |
  | CPU AMX utilization | 0% | 70-85% | new lever |
  | Implementation scope | n/a | datacenter only | narrow |

- **(d) Tier 강등 risk**: HiViS head-to-head pilot 의 incremental gain < 10% 시 Tier-3 reposition 또는 DROP; AMX (Intel SPR/GNR) 없으면 AMD Zen 5 AVX-512 VNNI BF16 fallback (15-25% slower).
- **(e) Outperform 가능성**: DAC/MLSys-Industry 6p short paper 적합. Datacenter scope 명시로 HiViS scoop 회피.
- **(f) Implementation envelope**: Datacenter (8×H100 + Intel SPR or AMD Zen 5), vLLM `MultiprocExecutor._init_workers_*` (multiproc_executor.py:L46) patch (~400 LoC), HiViS reference 구현 필요 (head-to-head benchmark).

---

## 7. 미선정 12 + Embedded 1 = 13 entry (R10-α.3 3요소 + R67 sub-axis Scoring)

상세 사유 + 재방문 조건 + R67 ★/▼: [unselected.md](unselected.md)

요약 (Phase 2' mean score 순):

| # | Idea | Score | Grade | 처리 분류 | Phase 3 Status | ★ 최고 sub-axis | ▼ 최저 sub-axis |
|---|---|---|---|---|---|---|---|
| 1 | **A1' Mosaic Maestro** | 7.85 | Accept | layered partner | Phase 3 partner | arch-sys R47 (9) | diff Positioning (7) |
| 2 | **A3' Power Grid (=C3L')** | 8.05 | Strong Accept | DVFS axis solo | elective rank 4 | arch-sys HW-fit/R47 (9/9) | diff Positioning (7) |
| 3 | **A4' Pipeline Conveyor** | 7.65 | Accept | CPU vision overlap | Tier-2 reposition | diff Positioning (8) | impact Magnitude (8) |
| 4 | **A5' Heartbeat Sync** | 7.65 | Accept | KV materialization | Phase 3 component | arch-sys 4axis (8) | impact Magnitude (7) |
| 5 | **B8' VAST-Spec** | 7.95 | Accept-strong | VAST-γ merge | elective rank 3 | diff Cov/Contrib (9/9) | novelty Mech (7) |
| 6 | **B10' Modality-Boundary** | 8.05 | Accept | C2L' merge | merge candidate | diff Cov (9) + impact Adopt (9) | novelty Mech/arch Sim (8/8) |
| 7 | **B12' Entropy-Gated γ*** | 8.40 | Strong Accept | VAST-γ merge | elective rank 3 | arch-sys 4axis 모두 (9) | impact Mag/Appl (8) |
| 8 | **B13' Cross-Chunk MRoPE** | 7.30 | Conditional | gate-dependent | DROP risk | novelty Mech/Hyp (8) | impact Comb (6) |
| 9 | **B14' Hessian Calib** | n/a | embedded | → B11' | Phase 3 sub-step | n/a (embedded) | n/a (embedded) |
| 10 | **C1L' VTAP-Sched** | 8.05 | Strong Accept | HW chunk axis | elective rank 2 | nov Mech (9) + arch 3axis (9) | impact Appl/Adopt (7) |
| 11 | **C2L' CDVV-Lite** | 7.75 | Accept | CPU vision main | elective rank 5 | nov Comb (9) + diff Cov (9) | arch-sys R47 (7) |
| 12 | **C6L' UnifiedKV-NVMe** | 8.15 | Strong Accept | KV tier solo | **elective rank 1** | impact Mag/Adopt (9) + diff Cov (9) | novelty Hyp (8) |

---

## 8. 참고 자료 (Step 0 paper 30+편 list)

본 세션이 Step 0 / Step 0-α 에서 검증한 30+편 paper 의 압축 list. 전체 detail 은 `papers.md` 누적 wiki 참조.

### Axis 1 — VLM + Speculative Decoding
- [SpecVLM (arXiv:2509.11815)](https://arxiv.org/abs/2509.11815) — 2.5-2.9× speedup, EagleVLM baseline
- [SpecVLM-Video (arXiv:2508.16201)](https://arxiv.org/abs/2508.16201) — verifier-guided 2-stage pruning
- [Spec-LLaVA (arXiv:2509.11961)](https://arxiv.org/abs/2509.11961) — ICML 2025 TTODLer workshop, 3.28× lossless
- [FastVLM (arXiv:2510.22641)](https://arxiv.org/abs/2510.22641) — IJCNLP-AACL 2025 Main, self-spec
- [ViSpec (arXiv:2509.15235)](https://arxiv.org/abs/2509.15235) — NeurIPS 2025
- [MASSV (arXiv:2505.10526)](https://arxiv.org/abs/2505.10526) — Cerebras, 1.46× e2e
- [Spec-VLA (arXiv:2507.22424)](https://arxiv.org/abs/2507.22424) — EMNLP 2025 main, relaxed acceptance

### Axis 2 — VLM Chunked Prefill / PD-Disagg
- [Nova (arXiv:2509.21301)](https://arxiv.org/abs/2509.21301) — elastic spatial partitioning
- [HydraInfer (arXiv:2505.12658)](https://arxiv.org/abs/2505.12658) — 3-stage hybrid disagg, 4× throughput
- [ModServe (arXiv:2502.00937)](https://arxiv.org/abs/2502.00937) — SoCC 2025, 128-GPU production
- [ElasticMM (arXiv:2507.10069)](https://arxiv.org/abs/2507.10069) — NeurIPS 2025 Oral, TTFT 4.2×
- [HeteroServe (arXiv:2603.12707)](https://arxiv.org/abs/2603.12707) — cross-tier GPU heterogeneity 37% Tokens/$
- [KV-Efficient VLA (arXiv:2509.21354)](https://arxiv.org/abs/2509.21354) — RNN-gated chunked KV
- [PPE (arXiv:2510.22936)](https://arxiv.org/abs/2510.22936) — ICLR 2026, MRoPE positional preservation

### Axis 3 — CPU-GPU Heterogeneous
- [Dovetail (arXiv:2412.18934)](https://arxiv.org/abs/2412.18934) — EMNLP 2025 main, GPU draft + CPU target 1.79-10.1×
- [Ghidorah (arXiv:2505.23219)](https://arxiv.org/abs/2505.23219) — Jetson NX HCMP, 7.6×
- [Mirror Speculative Decoding (arXiv:2510.13161)](https://arxiv.org/abs/2510.13161) — Apple GPU-NPU 2.8-5.8×
- [VLMOpt (arXiv:2604.26334)](https://arxiv.org/abs/2604.26334) — MLSys 2026 Industry, 10× VRAM 감소
- [AHASD (arXiv:2604.25326)](https://arxiv.org/abs/2604.25326) — DAC 2026, NPU-PIM 1.5× tput, 1.24× energy
- [FlexSpec (arXiv:2601.00644)](https://arxiv.org/abs/2601.00644) — edge-cloud collaborative
- [CoVSpec (arXiv:2605.02218)](https://arxiv.org/abs/2605.02218) — VLM device-edge 2.21× tput
- [SpecOffload (arXiv:2505.10259)](https://arxiv.org/abs/2505.10259) — GPU utilization 4.49×

### Axis 4 — Energy / Power
- [VoltanaLLM (arXiv:2509.04827)](https://arxiv.org/abs/2509.04827) — feedback DVFS, 36.3% energy
- [GreenLLM (arXiv:2508.16449)](https://arxiv.org/abs/2508.16449) — SLO-aware DVFS, 34% energy
- [DualScale (arXiv:2602.18755)](https://arxiv.org/abs/2602.18755) — MPC prefill / slack decode, 39%/48%
- [Energy-Performance Tradeoff (arXiv:2501.08219)](https://arxiv.org/abs/2501.08219) — decode 77-91% time + frequency-insensitive
- [EnergyLens (arXiv:2605.10556)](https://arxiv.org/abs/2605.10556) — multimodal closed-form 12-param
- [Camel (arXiv:2508.09173)](https://arxiv.org/abs/2508.09173) — Jetson MAB EDP 12.4-29.9%
- [SpecDec Energy Benchmark (arXiv:2602.09113)](https://arxiv.org/abs/2602.09113) — EACL Findings 2026

### Step 0-α — Workload Characterization (peer-reviewed published)
- [LLMServingSim (arXiv:2408.05499)](https://arxiv.org/abs/2408.05499) — IISWC 2024, 14.7% error, 91.5× speed
- CPU LLM Inference IISWC 2024 (Georgia Tech, Seonjin Na) — AMX/AVX-512 characterization
- HEX-SIM IISWC 2024 — multi-modal chiplet NPU
- Lotus IISWC 2024 — ML preprocessing pipeline
- [CPU-GPU Coupled (arXiv:2504.11750)](https://arxiv.org/abs/2504.11750) — ISPASS 2025, GH200 1.9-2.7× prefill, 4× CPU-bound
- ADOR / NonGEMM ISPASS 2025
- [LVLM Survey (arXiv:2604.05546)](https://arxiv.org/abs/2604.05546) — ACL 2026 Findings, 3-phase taxonomy
- [Llama-3.2-Vision (arXiv:2504.00557)](https://arxiv.org/abs/2504.00557) — CVPR 2025 Workshop, 50% visual reduction
- [SparseVILA (arXiv:2510.17777)](https://arxiv.org/abs/2510.17777) — 4× prefill, 2.5× decode, 2.6× e2e
- [DeepSeek-VL2 (arXiv:2412.10302)](https://arxiv.org/abs/2412.10302) — MLA latent KV compression
- [Compact VLM Recipes (arXiv:2603.16987)](https://arxiv.org/abs/2603.16987) — CPU-side latency 53%/93%
- [FlexGen (arXiv:2303.06865)](https://arxiv.org/abs/2303.06865) — ICML 2023, OPT-175B 16GB GPU
- [SARATHI (arXiv:2308.16369)](https://arxiv.org/abs/2308.16369) — chunked prefill 10× decode tput
- [Sarathi-Serve (arXiv:2403.02310)](https://arxiv.org/abs/2403.02310) — OSDI 2024, 2.6-5.6×
- [DistServe (arXiv:2401.09670)](https://arxiv.org/abs/2401.09670) — OSDI 2024, 7.4× requests
- [Splitwise (arXiv:2311.18677)](https://arxiv.org/abs/2311.18677) — ISCA 2024, 2.35× tput
- [POD-Attention (arXiv:2410.18038)](https://arxiv.org/abs/2410.18038) — ASPLOS 2025, 59% max attn speedup
- [Mooncake (arXiv:2407.00079)](https://arxiv.org/abs/2407.00079) — FAST 2025, 525% throughput
- [EAGLE (arXiv:2401.15077)](https://arxiv.org/abs/2401.15077) — ICML 2024, 2.7-3.5×
- [MMSpec (arXiv:2603.14989)](https://arxiv.org/abs/2603.14989) — VLM spec decode benchmark

### Benchmark Reports
- [MLPerf v5.0 (NVIDIA blog)](https://developer.nvidia.com/blog/nvidia-blackwell-delivers-massive-performance-leaps-in-mlperf-inference-v5-0/) — Blackwell 3.4× / 30×
- [MLPerf Llama 2 70B (MLCommons)](https://mlcommons.org/2024/03/mlperf-llama2-70b/) — 24 submitters, 33k tok/s
- [vLLM Spec Decode Blog 2024-10-17](https://vllm.ai/blog/2024-10-17-spec-decode) — 2.8× / 1.5× / 1.4-1.8× slowdown
- [NVIDIA Spec Decode Tutorial / EAGLE-3](https://developer.nvidia.com/blog/an-introduction-to-speculative-decoding-for-reducing-latency-in-ai-inference/) — 58% latency reduction
- [vLLM 2024 Retrospective + 2025 Vision](https://blog.vllm.ai/2025/01/10/vllm-2024-wrapped-2025-vision.html) — chunked prefill + spec decode 결합 priority

---

## 9. 약어 / 핵심 용어 풀이 (R35-α 통합 glossary, 맨 뒤)

> **CTRL+F (Mac: Cmd+F) 로 탐색**

### 0.1 도메인 약어 / 핵심 용어 정의

- **VLM**: Vision-Language Model (e.g., Llama-3.2-Vision, Qwen2-VL, LLaVA-1.5/1.6, DeepSeek-VL2, InternVL3)
- **SpecDec**: Speculative Decoding (Leviathan et al. 2023). Draft + Target 두 모델 통해 multi-token 병렬 생성
- **EAGLE / EAGLE-3**: hidden-state-based draft head + dynamic tree drafting ([arXiv:2401.15077](https://arxiv.org/abs/2401.15077), [arXiv:2503.01840](https://arxiv.org/abs/2503.01840))
- **Chunked Prefill**: Prefill 을 chunk 단위로 분할하여 decode 와 piggyback (SARATHI, Sarathi-Serve, [arXiv:2403.02310](https://arxiv.org/abs/2403.02310))
- **PD-Disagg**: Prefill-Decode Disaggregation (DistServe, Splitwise, Mooncake)
- **MMMU / MMBench / Video-MME / LLaVA-Bench / OCRBench**: VLM benchmark suite (본 세션의 shared evaluation suite)
- **KV cache**: Key-Value cache (auto-regressive decoding 의 transformer state)
- **MRoPE**: Multimodal Rotary Position Embedding (multimodal token 의 위치 encoding)
- **MLA**: Multi-head Latent Attention (DeepSeek-VL2 의 KV cache latent vector 압축)
- **TBC**: Thread Block Cluster (NVIDIA Blackwell SM120 의 multi-SM cluster, distributed shared memory)
- **DSMEM**: Distributed Shared Memory (TBC cluster 의 cross-SM data exchange)
- **HBM3 / HBM3e**: High Bandwidth Memory 3 / 3e (NVIDIA Blackwell GPU memory)
- **ECS**: Error Correction Scrubbing (HBM3e RFM/ECS application API)
- **AMX**: Intel Advanced Matrix Extensions (SPR/GNR CPU, BF16/INT8 matmul)
- **AVX-512 VNNI BF16**: AMD Zen 5 vector neural network instructions (AMX fallback path)
- **PCIe Gen5 / NVMe Gen5**: PCIe 5.0 interconnect / NVMe SSD over PCIe Gen5
- **CXL 3.x**: Compute Express Link 3.x (Tier-2 simulator only, 2026-05 commercial 부재)
- **DVFS**: Dynamic Voltage Frequency Scaling
- **NVML**: NVIDIA Management Library (`nvmlDeviceGetSupportedGraphicsClocks`)
- **L2 set-aside**: CUDA 11.0+ `cudaAccessPolicyWindow` 의 L2 cache persistent fraction
- **Marlin kernel**: vLLM 의 sub-INT4 matmul CUDA kernel (sub-2-bit 확장 본 세션)
- **Pinsker inequality**: `TV(p, q) ≤ √(KL(p||q) / 2)` (B9' bound axis 1)
- **Bretagnolle-Huber (BH)**: `TV(p, q) ≤ √(1 - e^{-KL(p||q)})` (B9' bound axis 2)
- **TV**: Total Variation distance
- **KL**: Kullback-Leibler divergence
- **EWMA**: Exponentially Weighted Moving Average (A6' VAST-Sched signal)
- **PWL**: Piecewise Linear function (A6' VAST-Sched output)
- **TPOT / TTFT**: Time Per Output Token / Time To First Token (VLM serving SLO metrics)
- **GH200**: NVIDIA Grace Hopper Superchip (closely-coupled CPU-GPU, ISPASS 2025)
- **POD-Attention**: Prefill-Decode-Overlap Attention CTA-level fusion (ASPLOS 2025)
- **MoE**: Mixture-of-Experts (DeepSeek-VL2 의 MoE VLM)
- **Llama.cpp / SVE2 / NEON**: ARM SoC CPU inference backend

### 0.2 Polysemous Term Disambiguation (R51-α)

- **"draft"**:
  - SpecDec draft model — Target 의 작은 보조 모델 (Llama-3.2-1B INT8 등)
  - EAGLE-3 draft head — hidden-state 기반 별도 head, target 의 layer 1-2 KV 공유
  - imitation-learned self-draft — FastVLM 의 single-model self-spec
- **"verify / verification"**:
  - SpecDec verification — draft token 의 target distribution 일치 검증
  - L2 cache verification — `cudaAccessPolicyWindow` 의 L2 hit ratio 측정
- **"chunk"**:
  - Chunked Prefill chunk — vLLM 의 token-budget-based prefill 분할
  - KV cache chunk — KV cache 의 layer-wise 또는 region-wise 분할
  - Region-atomic chunk (A1') — visual region 을 split 하지 않는 chunk
- **"prefill"**:
  - LLM prefill — autoregressive 생성 시작 전 context 의 KV 생성
  - Vision prefill — VLM 의 ViT encoder output 의 KV cache 생성
  - M3 spec prelude prefill (A4') — spec decode 의 first chunk prefill
- **"saliency"**:
  - Cross-attn saliency (A6' / B8' / B12') — VLM 의 cross-attention attention probability
  - Image saliency (computer vision) — pixel-level salience map (본 세션 미사용)
- **"context"**:
  - LLM context window — token length (4K / 8K / 32K)
  - CUDA Green Context — SM partition unit (Hopper SM90+ / Blackwell SM120, deprecated in C5L' for `cudaAccessPolicyWindow`)
- **"kernel"**:
  - CUDA kernel — GPU 함수 (Marlin, TBC, POD-Attention)
  - OS kernel — Linux kernel (R45.1 위반 위험, 본 세션은 application-level만)
- **"partition / split"**:
  - SM partition — GreenContext / `cudaAccessPolicyWindow` 의 SM 분할
  - KV cache partition — Hot/Warm/Cold tier (C6L')
  - Modality partition — vision encoder CPU + LLM decoder GPU (B10' / C2L' / A2')
- **"stream"**:
  - CUDA stream — `cudaStreamCreateWithPriority` (C4L' 3-stream priority)
  - Triple-buffer stream — pinned memory 의 producer-consumer pipeline
- **"draft tree"**:
  - Static draft tree — fixed γ depth (vLLM default)
  - Dynamic draft tree — EAGLE-2/3 / Spec-LLaVA / A6' 의 runtime construction
- **"acceptance"**:
  - Token acceptance — strict equality verification
  - Relaxed acceptance — Spec-VLA 의 semantic equivalence
  - Acceptance rate α — accepted / proposed token ratio

### 0.3 Idea Metaphor Noun → Mechanism 대응

- **"Mosaic Maestro" (A1')** → Region-atomic chunked prefill scheduling 의 비유 noun. 실제 mechanism: `min(token_budget, round_up_to_region(...))` + spec acceptance window 동기화.
- **"Tortoise & Hare" (A2')** → CPU (느린 tortoise) draft + GPU (빠른 hare) verify 의 비유. 실제 mechanism: CPU AMX/AVX-512 1B INT8 draft worker via MultiprocExecutor.
- **"Power Grid Conductor" (A3')** → 3-phase DVFS 의 전력망 metaphor. 실제 mechanism: 3-sub-phase NVML clock control + HBM clock + acceptance×power feedback.
- **"Pipeline Conveyor" (A4')** → Vision pipeline 의 컨베이어 belt metaphor. 실제 mechanism: M3 spec prelude primary + cross/self-attention 분기.
- **"Heartbeat Synchronizer" (A5')** → Async prefetch heartbeat 의 비유. 실제 mechanism: 3-alternative signal source (Triton / LVSpec-logit / sub-module) + double-buffer.
- **"VAST-Sched / VAST-Spec / VAST-γ" (A6' / B8' / B12')** → Visual-Aware Speculative scheduling. 실제 mechanism: cross-attn saliency tree EWMA + PWL(H_v) → dynamic `num_spec`.
- **"VTAP-Sched" (C1L')** → Visual Token Atomic Page-aligned scheduling. 실제 mechanism: HBM3e 256B page-table align + L2 persistent + TBC.
- **"CDVV-Lite" (C2L')** → CPU Decoupled Vision Verify Lite. 실제 mechanism: AMX/VNNI dual-path + acceptance-aware dGPU partition + per-stage DVFS.
- **"KV-Stream-TBC" (C4L')** → KV stream + Thread Block Cluster. 실제 mechanism: 3-stream priority + Blackwell TBC cluster-8 + pinned triple-buffer.
- **"SpecVerify-L2" (C5L')** → Spec Verification L2 persistence. 실제 mechanism: `cudaAccessPolicyWindow` set-aside + acceptance-aware SM resize + shmem carveout + POD-Attention CTA fusion.
- **"UnifiedKV-Hier-NVMe" (C6L')** → Unified KV Hierarchical NVMe. 실제 mechanism: HBM3e Hot / DDR5 Warm / NVMe Gen5 Cold tier + spec attention IoU prefetch.

---

*Last updated: 2026-05-26 — Step 7 wiki bundle (Top-M 6 final + Phase 3 elective 5 + DROP risk 3). R28.1 9 섹션 순서 준수 + R15-β 36+ contribution bullet + R67 sub-axis Scoring (18 idea) + R14.4 Decision Tree 양식 A.*
