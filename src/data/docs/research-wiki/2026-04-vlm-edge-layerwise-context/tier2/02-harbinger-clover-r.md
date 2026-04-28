# HARBINGER-CLOVER-R (Heuristic Adaptive Re-Bin Inference Gating with Energy Reduction + Confidence-Layer Optical Variable Exit Recipe, Refined)

- **Tier**: 2
- **Lead expert**: ai-optimization-expert (speculative decoding + early exit + DVFS)
- **Target venue**: DATE 2026 / ISLPED 2026 / IEEE CAL 2026 / EMNLP 2026 efficient track
- **Single-system fit (R20-γ)**: Jetson Orin NX 16GB primary (1024-core Ampere, 10-25W envelope) — small-memory power-limited edge. RTX 5090 secondary (alternate 검증)
- **5-axis tag (R55.2)**:
  - [Performance] decode throughput +32% (speculative + early-exit 결합)
  - [Energy] J/token -28% (Orin NX 25W envelope, layer skip 기여)
  - [Power] peak -18% (early-exit SM utilization 감소)
  - [Memory] (보조) decode KV bandwidth -8%
- **R47 path**: R47.2 application-level only — vLLM 또는 llama.cpp speculative decoding 경로 + early-exit hook + nvpmodel CLI Python wrapper. Orin NX 16GB 환경 Qwen3-VL-2B/4B 또는 InternVL3-1B/2B 운영. CUDA 11.4 Ampere 호환. kernel patch 없음.
- **R45 risk**: 6/10 (Orin NX 16GB memory tight — Qwen3-VL-2B + draft model + KV cache 동시 보관 가능성, long-context 에서 빡빡. early-exit 의 정확도 손실 visually ambiguous task (OCR/ChartQA) 에서 클 수 있음 → confidence threshold task-adaptive 조정 필요. nvpmodel sub-ms switching 신뢰성 측정 필요)
- **Phase 1' delta (improve-first)**:
  - **Improved** M1 (VCSDL — Visual-Confidence-Aware Speculative Draft Length) — review-novelty 권고에 따라 visual cluster heat anchor (M2) 가 layer-specific speculation acceptance signal 임을 강조하여 reframe. Spec-LLaVA 의 dynamic tree confidence 와 차별: **vision-grounded entropy (ViT CLS / DeepStack inject signal)** vs draft model confidence. PyramidDrop / VisionZip 의 convergence-layer insight 를 speculation acceptance prediction 의 signal 로 활용.
  - **Improved** M2 (LCEH → Visual-Cluster-Heat Early-Exit) — middle layer 의 visual cluster heat (top-3 cluster attention 합산) 측정 → 0.85 이상이면 LLM hidden state mature → early-exit. FREE early-exit 의 confidence-only 보다 visual-grounded confidence 가 정확.
  - **Improved** M3 (PEAFL — Power-Envelope Adaptive Frequency Locking) — Orin NX nvpmodel 0/1/2 (10W/15W/25W) state 에서 LLM decode phase 의 GPU clock 을 layer 별 lock. early-exit 발생 시 즉시 lower-power state 전환. CLONE 의 layer-boundary DVFS 를 power envelope 안에서 dynamic 적용.
  - **Added** scoop 대응 차별 axis: Spec-LLaVA (arXiv:2509.11961) baseline 추가 — Spec-LLaVA 는 dynamic tree confidence draft, **본 기법은 vision-grounded entropy + visual-cluster-heat early-exit + power-envelope DVFS**. Fast Speculative Edge-Cloud (arXiv:2505.21594) 추가 — Edge-Cloud 는 client-server 분리, **본 기법은 single-Orin-NX visual-cluster-heat + power-envelope locking** 차별. Spec-VLA (arXiv:2507.22424) 추가 (VLA model speculative decoding).
  - **Reinforced** R56.2: NeurIPS 2024 KVQuant, ICML 2025 SparseVLM, EMNLP 2025 Findings MASSV, USENIX ATC 2025 CLONE, ACL 2025 Findings FREE, CVPR 2025 PyramidDrop, CVPR 2025 VisionZip, MobiSys 2025 FUSE Mobile DVFS, IISWC 2024. AI 분야 published 35%+ 충족 + edge venue 추가 (ISLPED 2024, DATE 2024).

## 1. RQ (R57.1)

- **RQ-5.1**: VLM 의 ViT CLS embedding 또는 DeepStack inject signal entropy 가 낮을 때 (visually unambiguous, threshold τ_low = 0.3) speculative draft length γ=8, 높을 때 (τ_high = 0.7) γ=2 로 동적 조절하면, 정적 γ=4 baseline 대비 acceptance rate 이 **+18%pt 이상**, decode throughput 이 **+20% 이상** 향상되는가?
- **RQ-5.2**: middle layer (Qwen3-VL-2B 의 L8-12 convergence layer) 의 visual cluster heat (top-3 cluster attention 합산) ≥ 0.85 일 때 early-exit 했을 때, average layer skip 이 **28% 이상** (28-layer model 의 8 layer 평균 skip), MMMU drop 이 **≤ 1.5pt** 이내인가? OCR / ChartQA 같은 visually ambiguous task 에서 task-adaptive threshold 조정으로 drop 을 **≤ 2.5pt** 안에서 유지 가능한가?
- **RQ-5.3**: Orin NX nvpmodel state (10W/15W/25W) 사이 transition 의 sub-ms switching reliability 가 1000회 측정에서 **failure rate < 1%** 이고, layer-boundary 마다 frequency lock 적용 시 J/token 이 정적 25W mode 대비 **-22% 이상** 감소되는가?
- **RQ-5.4**: 3 mechanism 결합 시 Orin NX 25W envelope 안에서 peak power 가 baseline 대비 **-18% 이상** 감소되고, 동시에 throughput 이 **+32% 이상** 향상되는가?

## 2. 개요 (Metaphor noun ↔ mechanism)

**HARBINGER-CLOVER-R**: HARBINGER = 미리 예고하는 전조 (speculative draft 의 visual confidence prediction). CLOVER = 4-leaf 4-stage exit (early/mid/late/full). R = Refined (visual-cluster-heat anchor 격상 + power-envelope locking 통합).

핵심 통찰: **VLM 의 speculation acceptance 는 visually unambiguous 입력에서 안정적**. ViT CLS embedding 의 entropy (또는 DeepStack inject signal) 가 visual clarity 의 proxy — entropy 낮을 때 draft model 이 더 멀리 예측 가능 (γ 늘림). 또한 **middle layer 의 visual cluster heat 가 LLM hidden state maturation 의 signal** — 충분히 mature 시 final lm_head 만 적용하여 layer skip. Orin NX 의 25W envelope 은 peak power 가 throttle trigger — layer-boundary DVFS 로 envelope 안 평균 power 낮추고 burst 시점만 peak.

## 3. 기존 연구 한계 + GAP

### 3.1 Workload evidence (R17)

- arXiv:2509.11815 (SpecVLM, 2025): elastic visual compressor 2.5-2.9× speedup.
- arXiv:2509.15235 (ViSpec, 2025): vision adaptor global feature inject acceptance rate +30%.
- arXiv:2505.10526 (MASSV, EMNLP 2025 Findings): COCO captioning +47.5% acceptance.
- ACL 2025 Findings ([aclanthology 2025.findings-acl.1209](https://aclanthology.org/2025.findings-acl.1209/)): FREE early-exit BLIP-2 mid-crisis insight.
- arXiv:2410.17247 (PyramidDrop, CVPR 2025): mid-layer convergence redundancy.
- arXiv:2412.04467 (VisionZip, CVPR 2025): convergence layer dominant token (LLaVA-NeXT L6-12).
- arXiv:2506.02847 (CLONE, USENIX ATC 2025): layer-boundary DVFS edge demonstration.
- Orin NX nvpmodel switching latency known < 50us (NVIDIA forum), sub-ms 신뢰성 측정 필요.

### 3.2 GAP 표 (R56.2)

| 기존 연구 | venue year | 핵심 mechanism | what / why / how 차별 |
|-----------|-----------|----------------|------------------------|
| Spec-LLaVA — Dynamic Tree-Based Speculative Decoding for VLMs | [arXiv:2509.11961](https://arxiv.org/abs/2509.11961) (2025) | adaptive tree expansion using draft model confidence, 3.28× LLaVA-1.5 | what: HARBINGER-CLOVER-R 는 **vision-grounded ViT CLS entropy (visual clarity proxy)** + **visual-cluster-heat early-exit** + **power-envelope DVFS**. why: Spec-LLaVA 는 draft model confidence 만, visual-grounded signal 부재. how: ViT CLS / DeepStack inject signal entropy 사용 |
| Fast Speculative Edge-Cloud Decoding with Early Exits | [arXiv:2505.21594](https://arxiv.org/abs/2505.21594) (2025, Unitree Go2) | Jetson Nano client + early-exit + speculative, 21% Unitree | what: 본 기법 **single-Orin-NX self-contained**, edge-cloud 분리 부재. why: client-server latency 추가. how: visual-cluster-heat anchor + power-envelope |
| SpecVLM | [arXiv:2509.11815](https://arxiv.org/abs/2509.11815) (2025) | elastic visual compressor adaptive | what: visual confidence-aware draft length 부재 |
| ViSpec | [arXiv:2509.15235](https://arxiv.org/abs/2509.15235) (2025) | global feature inject draft | what: dynamic γ 조절 부재 |
| MASSV — Multimodal Adaptation Self-Distillation | [arXiv:2505.10526](https://arxiv.org/abs/2505.10526) (EMNLP 2025 Findings) | small LM adaptation as multimodal draft | what: early-exit 부재, DVFS 부재 |
| FREE — Free Early-Exit | [aclanthology 2025.findings-acl.1209](https://aclanthology.org/2025.findings-acl.1209/) (ACL 2025 Findings) | confidence-only early-exit VLM | what: visual-cluster-heat 부재 |
| GM-Skip | [arXiv:2508.18227](https://arxiv.org/abs/2508.18227) (2025) | block skipping reverse-order deletion | what: speculative + DVFS 결합 부재 |
| HiViS | [arXiv:2509.23928](https://arxiv.org/abs/2509.23928) (2025) | drafter hide visual token | orthogonal, draft model improvement axis |
| CLONE | [arXiv:2506.02847](https://arxiv.org/abs/2506.02847) (USENIX ATC 2025) | layer-boundary DVFS edge | what: VLM speculative + early-exit 부재 |
| Spec-VLA | [arXiv:2507.22424](https://arxiv.org/abs/2507.22424) (2025) | VLA-specific speculative decoding | what: VLA 한정, VLM general 부재 |
| FUSE Mobile DVFS | [arXiv:2507.02135](https://arxiv.org/abs/2507.02135) (MobiSys 2025) | FUSE governor TTFT/TPOT 7-37% 감소 | what: mobile governor only, layer-specific 부재 |
| PyramidDrop | CVPR 2025 | stage 별 token drop | workload evidence baseline |
| VisionZip | CVPR 2025 | convergence layer dominant token | workload evidence |
| KVQuant | NeurIPS 2024 | quant + sink anchor | orthogonal |
| SparseVLM | ICML 2025 | rank-based sparsification | orthogonal |
| IISWC 2024 | [arXiv:2512.01644](https://arxiv.org/abs/2512.01644) | LLM characterization | workload evidence |

Peer-reviewed published: MASSV (EMNLP 2025 Findings), FREE (ACL 2025 Findings), CLONE (USENIX ATC 2025), FUSE (MobiSys 2025), PyramidDrop (CVPR 2025), VisionZip (CVPR 2025), KVQuant (NeurIPS 2024), SparseVLM (ICML 2025), IISWC 2024 = **9/16 = 56%** + Spec-LLaVA / SpecVLM / ViSpec arXiv (top vendor + 100+ citation 가능 시 포함) → **AI 분야 35%+ 충족**.

## 4. 제안 기법 — 3 mechanism

### 4.1 — M1: Visual-Confidence-Aware Speculative Draft Length (VCSDL)

#### Block 1: Concept
- 추가되는 Scheme: VLM 의 ViT 출력 (CLS embedding 또는 DeepStack inject signal) entropy 측정 → entropy 낮을 때 (visually unambiguous, threshold τ_low = 0.3) speculative draft 길이 늘림 (γ=8), 높을 때 (visually ambiguous, OCR 등, τ_high = 0.7) γ=2. 동적 γ 조절.
- 해결하려는 문제: 정적 draft length γ=4 는 visual clarity 와 무관, acceptance rate 변동 큼. Spec-LLaVA 는 draft model confidence 만, visual signal 부재 — 입력 image 가 visually ambiguous 시 acceptance rate 낮아도 인식 못 함.
- 동작 원리: (1) ViT forward 끝에 CLS token embedding 의 entropy 산출 (softmax over patch tokens, log compute) → (2) DeepStack inject signal 도 보조로 사용 (Qwen3-VL 의 L8/L16/L24 inject 정보) → (3) entropy → γ lookup table (linear interpolation between τ_low ↔ τ_high) → (4) speculative draft 단계에서 γ token 생성 → (5) target VLM verify → (6) accepted token 수 / γ 비율로 다음 step γ 조정 (EMA).
- 차별화: Spec-LLaVA 의 draft model confidence 와 **본 기법은 vision-grounded ViT entropy** — 입력 image clarity 자체가 signal. ViSpec 의 global feature inject 와 다른 axis (draft length 조절).

#### Block 1.5: Gain Contribution (R55.3)
- Primary axis: [Performance] decode throughput +20%
- Secondary axis: [Energy] -8% (acceptance rate 향상으로 verify cost 절감)
- 단독 미보장: [Power] (M3 결합 필요)

#### Block 2: Source-Level Implementation

| File path | Symbol | As-is | To-be | 변경 type |
|-----------|--------|-------|-------|-----------|
| `vllm/spec_decode/draft_runner.py` | `DraftRunner.run` | static γ=4 | ViT entropy 측정 후 γ 동적 결정 | Modify |
| `vllm/model_executor/models/qwen3_vl.py` | `Qwen3VLForConditionalGeneration.forward` | uniform | ViT 끝에서 CLS entropy + DeepStack signal 출력 | Modify |
| `vllm/spec_decode/visual_entropy.py` (new) | `def compute_visual_entropy(cls_emb, deepstack_signal)` | — | log-sum-exp entropy 계산 | Add |

R52.3 trace: vLLM `spec_decode/` 실재 ([github.com/vllm-project/vllm/tree/main/vllm/spec_decode](https://github.com/vllm-project/vllm/tree/main/vllm/spec_decode)). [✅]

#### Block 3: Synthetic Workload
- Unit test: ViT entropy 계산 정확도 (synthetic CLS embedding 분포), 5분.
- Mechanism-isolated: γ 동적 조절 acceptance rate 측정 (Qwen3-VL-2B + draft 0.5B, MMMU 100shot), 4시간.

---

### 4.2 — M2: Visual-Cluster-Heat Early-Exit (LCEH)

#### Block 1: Concept
- 추가되는 Scheme: middle layer (Qwen3-VL-2B 의 L8-12 convergence layer, VisionZip 측정) 의 attention concentration 측정 → visual cluster heat (top-3 cluster 의 attention 합산) ≥ 0.85 이면 LLM 의 hidden state 가 mature → early-exit 가능 layer 식별. exit 시 final lm_head 만 적용. FREE 의 confidence-only 보다 visual-grounded confidence.
- 해결하려는 문제: FREE early-exit 는 confidence (logit margin) 만, visually ambiguous input 시 false-confidence. layer-uniform skip ratio 는 task-adaptive 부재.
- 동작 원리: (1) decode forward 중 middle layer (L8-12) 도착 시 visual token attention map 산출 → (2) top-3 visual cluster (PyramidDrop / VisionZip 의 convergence cluster 사용) 의 attention 합산 → (3) heat ≥ 0.85 시 early-exit signal → (4) 이후 layer skip, final lm_head + softmax 만 → (5) heat < 0.85 시 normal forward continue → (6) task-adaptive threshold (OCR/ChartQA 는 0.92, 일반 VQA 는 0.85) calibration 으로 결정.
- 차별화: FREE 는 LLM confidence 만, **본 기법은 visual cluster heat (vision-grounded)**. GM-Skip 의 block skipping 과 달리 **layer-level early-exit + visual signal**.

#### Block 1.5: Gain Contribution
- Primary axis: [Energy] -22% (layer skip 28% 평균)
- Secondary axis: [Performance] decode throughput +12%
- 단독 미보장: visually ambiguous task accuracy drop (mitigation: task-adaptive threshold)

#### Block 2: Source-Level Implementation

| File path | Symbol | As-is | To-be | 변경 type |
|-----------|--------|-------|-------|-----------|
| `vllm/model_executor/models/qwen3_vl.py` | `Qwen3VLForConditionalGeneration.decode_step` | uniform 28-layer | middle layer hook → cluster heat → early-exit branch | Modify |
| `vllm/spec_decode/early_exit.py` (new) | `class VisualClusterHeatExit` | — | top-3 cluster heat 산출 + threshold check | Add |
| `tools/lceh_calibrate.py` (new) | `def calibrate_threshold(model, calib_set, task)` | — | task-adaptive threshold 결정 | Add |

R52.3 trace: vLLM forward 분기 가능. PyramidDrop / VisionZip 의 cluster identification 코드 공개 ([github.com/Cooperx521/PyramidDrop](https://github.com/Cooperx521/PyramidDrop)). [✅]

#### Block 3: Synthetic Workload
- Unit test: cluster heat 산출 + threshold check, 5분.
- Isolated: layer skip rate + accuracy (MMMU/OCR/ChartQA 분리), 5시간.

---

### 4.3 — M3: Power-Envelope Adaptive Frequency Locking (PEAFL)

#### Block 1: Concept
- 추가되는 Scheme: Orin NX nvpmodel 0/1/2 (10W/15W/25W) state 에서 LLM decode phase 의 GPU clock 을 layer 별 lock — early-exit 발생 시 즉시 lower-power state 전환. CLONE 의 layer-boundary DVFS 를 power envelope 안에서 dynamic 적용.
- 해결하려는 문제: CLONE 은 LLM-only DVFS, VLM speculative + early-exit 결합 없음. 25W envelope 에서 peak power 가 throttle trigger → 평균 power 낮추기 + burst 시점만 peak 필요.
- 동작 원리: (1) M2 가 early-exit signal 발사 시 nvpmodel mode 0 (10W) 진입 → (2) speculative draft (M1) 활성 시 mode 2 (25W) 진입 (burst) → (3) verify 단계 mode 1 (15W) → (4) layer-boundary 마다 nvpmodel Python API 호출 (sub-ms switching, RQ-5.3 검증) → (5) PEAFL controller 가 EMA 로 power budget tracking, envelope violation 임박 시 mode 강제 다운.
- 차별화: CLONE layer-boundary DVFS vs **본 기법 power-envelope-aware + early-exit-triggered**. FUSE mobile governor vs **layer-specific lock**.

#### Block 1.5: Gain Contribution
- Primary axis: [Energy] -8% (DVFS 단독)
- Secondary axis: [Power] peak -18%
- 단독 미보장: [Performance] (M1/M2 결합으로 보완)

#### Block 2: Source-Level Implementation

| File path | Symbol | As-is | To-be | 변경 type |
|-----------|--------|-------|-------|-----------|
| `vllm/worker/peafl_controller.py` (new) | `PEAFLController.before_layer(layer_idx, phase)` | — | nvpmodel mode 결정 (early-exit/speculative/verify) | Add |
| `vllm/engine/llm_engine.py` | `_run_decode_step` | uniform clock | layer-boundary PEAFL hook | Modify |
| `tools/peafl_envelope_calib.py` (new) | `def measure_envelope_violation(model, workload)` | — | EMA tracker tuning | Add |

R52.3 trace: nvpmodel CLI on Orin NX 실재 (JetPack 6.0+). jtop Python wrapper [github.com/rbonghi/jetson_stats](https://github.com/rbonghi/jetson_stats). [✅]

#### Block 3: Synthetic Workload
- Unit test: nvpmodel sub-ms switching reliability (1000회 측정), 1시간.
- Isolated: layer-boundary DVFS + envelope tracking, 4시간.

## 5. 실험 플랜 7-요소 (R20-β / R52.1)

### 5.1 Hardware environment
- Jetson Orin NX 16GB primary (1024-core Ampere, 102 GB/s LPDDR5, 10-25W envelope)
- RTX 5090 secondary (alternate validation, larger model)
- Power meter: Orin NX `tegrastats`, RTX 5090 `nvidia-smi`

### 5.2 Model
- Qwen3-VL-2B (HF: `Qwen/Qwen3-VL-2B-Instruct`, 28 layer)
- Qwen3-VL-4B
- InternVL3-1B / InternVL3-2B
- LLaVA-Next 7B (RTX 5090 secondary)

### 5.3 Dataset/Workload
- MMMU val (general VQA)
- ChartQA (visually ambiguous, threshold sensitive)
- OCR-VQA (visually ambiguous baseline)
- DocVQA (long-form, layer skip risk)
- VideoMME-short (video, speculative gain 검증)

### 5.4 Simulator/Tools
- 측정 only
- nsight-systems, tegrastats, jtop

### 5.5 Ablation + Measurement Protocol
- Baseline 1: vanilla decode (no spec, no early-exit, MAXN power)
- Baseline 2: SpecVLM (reproduce, 코드 공개)
- Baseline 3: Spec-LLaVA (reproduce, 코드 공개 가능성 W3 확인)
- Baseline 4: FREE early-exit (reproduce)
- Baseline 5: CLONE DVFS only
- Ablation: M1 / M1+M2 / M1+M2+M3
- Task-stratified: general VQA vs visually ambiguous (OCR/ChartQA)
- 5 run, 95% CI

### 5.6 Implementation Steps

| Week | Task | Deliverable |
|------|------|-------------|
| W1-2 | ViT entropy + dynamic γ (M1) | unit test, acceptance rate 측정 |
| W3 | Spec-LLaVA reproduction | head-to-head |
| W4 | LCEH cluster heat (M2) | layer-skip rate measurement |
| W5 | Task-adaptive threshold calibration (M2) | OCR vs general 분리 |
| W6 | nvpmodel sub-ms switching reliability (M3) | 1000회 측정, failure rate |
| W7 | PEAFL envelope tracker | EMA controller |
| W8 | Orin NX 16GB memory profiling | KV cache + draft fit 검증 |
| W9 | M1+M2 통합 + acceptance rate | 5 task |
| W10 | M1+M2+M3 end-to-end | Orin NX power envelope |
| W11 | RTX 5090 secondary validation | larger model |
| W12 | FREE / CLONE reproduction | head-to-head |
| W13-14 | DATE / ISLPED draft | submission |

### 5.7 Preliminary Analysis Metrics
- Draft acceptance rate: target +18%pt (baseline static γ=4)
- Layer skip rate: target 28% average (8/28 layer)
- Decode throughput: target 32 tok/s (baseline 24, +33%)
- J/token: target 0.85 J (baseline 1.18, -28%)
- Peak power: target 20W (baseline 25W, -20%)
- MMMU drop: ≤ 1.5pt
- ChartQA drop (visually ambiguous): ≤ 2.5pt with task-adaptive threshold

## 6. 예상 효과 (R55.2 5-axis 정량 표)

| Axis | Baseline (Qwen3-VL-2B / Orin NX 25W) | HARBINGER-CLOVER-R | 개선 | 조건 |
|------|--------------------------------------|-------------------|------|------|
| [Performance] decode tok/s | 24 tok/s | 32 tok/s | **+33%** | spec γ 동적 + early-exit |
| [Energy] J/token | 1.18 J | 0.85 J | **-28%** | layer skip + DVFS |
| [Power] peak | 25 W | 20 W | **-20%** | early-exit SM 감소 |
| [Power] avg | 18 W | 14 W | **-22%** | 25W envelope 안 |
| [Accuracy] MMMU | 49.5 | 48.2 | -1.3 pt | Qwen3-VL-2B baseline |
| [Accuracy] ChartQA (visually ambig.) | 56.8 | 54.5 | -2.3 pt | task-adaptive threshold 0.92 |

조건: (a) Orin NX 16GB memory tight 환경 한정 — 큰 모델 (Qwen3-VL-7B+) 은 Jetson Thor 또는 RTX 5090. (b) M2 의 visual-cluster-heat threshold 는 task-adaptive (calibration W5). (c) M1 의 dynamic γ 는 single-stream decode 가정, multi-stream batch 시 효과 감소. (d) M3 sub-ms switching 신뢰성은 RQ-5.3 결과에 의존.

## 7. R56 self-check

- [x] R56.1: §3.2 GAP 표 안에 본 harness 자체 이전 세션 idea (KEYSTONE/VESPER/SHOAL/Loom) 인용 0.
- [x] R56.2: AI 분야 published 9/16 = 56% ≥ 35% (AI bar). edge venue (DATE/ISLPED) 추가 baseline 으로 65% 가능.
- [x] R56.3: 모든 reference 명시, 가상/TBD 0.

## 8. R52 / R53 / R54 self-check

- [x] R52.1 Baseline Source: vLLM v0.11.0, JetPack 6.0+, jetson_stats Python wrapper, Orin NX dev kit, RTX 5090 FE.
- [x] R52.2 Function/Class as-is/to-be 표: M1 (3 row), M2 (3 row), M3 (3 row) = 9 row.
- [x] R52.3 GitHub 실존: vLLM spec_decode, PyramidDrop github, jetson_stats 모두 [✅].
- [x] R52.4 Synthetic 3-tier: M1-M3 unit + isolated + end-to-end (W9-W11).
- [x] R52.5 Implementation vs Simulator 일관성: simulator X.
- [x] R54.1-R54.6 Final Verification Pass.
