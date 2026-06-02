# FOCUS-COVERAGE: A Multi-Focus Recall Benchmark and Query-Distribution-Marginal Retention with Coverage-Energy Pareto for Compression Methods

**Tier-2 · Grade: Conditional Accept (mean 7.40 / Total mean 7.18) · BENCH-D producer · community enabler (Breadth ★8)**

> **Metaphor noun:** *FOCUS-COVERAGE* — "focus" 가 multi-turn query focus, "coverage" 가 expected focus-set coverage (distribution-marginal).
> **Merge 출처:** FOCUS-RECALL(ai-opt) + algo-B FutureFocus + algo-G FocusBench + F2(legacy) + algo-A SubCover 흡수 (coverage 는 distribution-marginal 로만). F2 benchmark harness + GT protocol + G 의 FCR/Coverage-Energy Pareto metric 정식 + B 의 marginal-over-focus importance + TV-distance mismatch bound.

📖 약어 / 핵심 용어:
- **FCR (Focus-Coverage Recall)**: 같은 image 의 N개 focus 시나리오에서 보존된 KV 가 각 focus 의 GT region 을 얼마나 cover 하는가 (recall@k per focus 의 집계). 본 idea 신규 metric.
- **Coverage-Energy Pareto**: FCR vs J/turn (tegrastats energy) Pareto — 압축 정책의 system-relevance 정량.
- **Distribution-marginal retention**: single-query conditional `attn(t,q)` 대신 예상 focus 분포 marginal `I(t)=Σ_f P(f)·attn(t,f)` 로 보존 (expected coverage maximize).
- **TV-δ bound**: distribution-mismatch (예상 P_focus vs 실제) 가 TV-distance δ 작으면 retention 손실이 Lipschitz bound.
- **GT (Ground Truth)**: focus 별 region/segment — human-verified subset (≥50) + Grounding-DINO/SAM region 검증 + MileBench cross-val.

---

## 1. Research Questions

- **Master RQ**: single-query pruning 의 multi-focus retention 손실을 정량화할 metric 이 부재한 상황에서, focus-trajectory 벤치마크 + FCR/Coverage-Energy Pareto metric + 분포-marginal retention 을 설계하면 multi-focus 손실을 정량화하고 동일 budget 에서 recall 을 개선할 수 있는가?
- **RQ-1**: 같은 image 에 N개 focus 시나리오에서 single-query retention 의 FCR 이 N 증가에 따라 **단조 감소**하는가 (정량화 가능)?
- **RQ-2**: 예상 focus 분포 marginal retention `I(t)=Σ_f P(f)·attn(t,f)` 으로 바꾸면 동일 KV budget 에서 recall@k 를 **+6~12%p** 개선하고, distribution-mismatch (TV-δ) 가 작으면 손실이 bound 되는가?
- **RQ-3**: GT focus trajectory (LLM 생성) 가 실제 human multi-turn focus drift 분포와 상관 ≥ threshold 이며, Coverage-Energy Pareto 가 Jetson tegrastats energy 로 system-relevant 한가?

## 2. Two-Sentence Pitch

기존 query-aware pruning(SparseVLM/CROP)은 single-query conditional 이라 같은 image 의 여러 focus 를 묻는 multi-focus 시나리오에서 retention 손실을 정량화할 metric 자체가 부재하다. 본 idea 는 (a) focus-trajectory multi-turn 벤치마크 + Focus-Coverage Recall(FCR) + Coverage-Energy Pareto metric 을 정의하고, (b) single-query 대신 예상 focus 분포에 대한 marginal `I(t)=Σ_f P(f)·attn(t,f)` retention (보존이 미래 focus-set coverage 를 maximize, distribution-mismatch TV-δ Lipschitz bound) 을 설계한다.

## 3. 가설 + Falsification

**가설:** 같은 image 에 N개 focus 시나리오에서 single-query retention 의 FCR 은 N 증가에 단조 감소(정량화 가능)하며, 예상 focus 분포 marginal retention 으로 바꾸면 동일 KV budget 에서 recall@k 를 +6~12%p 개선하고, distribution-mismatch (TV-δ)가 작으면 손실이 bound 된다.

**Falsification (5):**
1. single-query retention 의 multi-focus recall 이 N 무관 평탄 (focus-shift 손실 없음) → R31 전제 재검증 기각 (단 CSP/RVIS 가 손실 입증, 가능성 낮음).
2. distribution-marginal retention 이 uniform random 대비 recall <5%p → 분포 모델링 무의미.
3. **GT focus trajectory (LLM 생성) 가 실제 human multi-turn focus drift 분포와 상관 낮음** → benchmark 외적 타당성 붕괴 (diff F04: synthetic-only over-claim).
4. focus 분포 skew (한 region 집중) 시 distribution-marginal 이 single-query 와 수렴 → "diverse-focus only" scope 축소.
5. focus prior 가 cross-dataset 일반화 실패 → training-free saliency+autocorrelation fallback.

## 4. Workload Evidence (Step 0-α 정량 인용)

- **Multi-focus 손실 (verified, CSP):** MileBench multi-turn embodied dialogue 에서 over-pruning **+41%** 회복 → single-query prune 이 multi-focus 에서 손실. CROP/SparseVLM 은 single-query region — focus-level recall 미분해.
- **Single-turn 평가 편향 (verified):** corpus A 가 사용한 Video-MME/MLVU/MVBench 는 모두 **single-turn MCQ** → multi-turn 평가 공백. "Are We Using the Right Benchmark" [arXiv:2510.07143](https://arxiv.org/abs/2510.07143) 도 compression bench noise 비판 (multi-turn eval 미제안).
- **기존 multi-turn 벤치 (CF, novelty F02):** MultiVerse(647 dialog, 4-turn), MMDU, ConvBench, MileBench 존재 → "부재" framing 무효. 단 compression-under-multi-focus FCR/Coverage-Energy Pareto metric 미정의.
- **Vision 비중:** sample 당 6,272 visual vs 109 text token → focus 별 region grounding 필요.
- **Edge bottleneck (PAISE'25):** decode memory-bound → Coverage-Energy Pareto 가 tegrastats energy 로 system-relevant.

## 5. 기준 코드베이스 (R52.1)

- **vLLM** `main` commit `7c37096`. scorer hook = **LLM cross-attention layer** (query 존재 시점), `_execute_mm_encoder`(L2869, query-independent encoder) **아님** (CF, F03/H4). retention apply = `KVCacheManager.allocate_slots`(L238).
- **SGLang** `e958f45` / **llama.cpp** `dbe9c0c` (cross-framework 벤치 호환).
- **HF model:** `Qwen/Qwen2.5-VL-7B-Instruct`, `llava-hf/llava-onevision-qwen2-7b-ov-hf` (region-grounded QA).
- **deps:** lm-eval-harness skill (FCR/Pareto task), CLIP/SAM skill (region grounding), Grounding-DINO (GT 검증).
- **Jetson Orin NX 16GB 정정 spec (CF-4):** 벤치 생성·측정은 **RTX 5090/4090 (Ada sm_89 — Ampere 아님, kernel dev 부적합)** 또는 DGX Spark; retention 정책 deploy 는 Jetson Orin NX (1024 core, LPDDR5 102.4GB/s, sm_87, `concurrentManagedAccess=0`). **Coverage-Energy Pareto 는 Jetson tegrastats energy 측정 포함** (arch-sys: FOCUS-RECALL 보다 system-relevant).

## 6. 동작 원리 (R53 inline)

### M1 — Focus-Trajectory Benchmark + FCR / Coverage-Energy Pareto Metric

**① 동작원리 4요소**
- **(무엇을)** image/video + GT region/segment per turn (drift param 제어) 벤치마크 + FCR + Coverage-Energy Pareto (FCR vs J/turn) metric.
- **(왜 작동)** RVIS 가 보인 focus shift → single-query importance 는 미래 focus underweight. multi-turn 벤치는 존재하나 compression-multi-focus FCR 미정의 → 신규 측정 도구.
- **(구현변경점, CF)** GT = human-verified subset (≥50) + Grounding-DINO/SAM region 검증 + MileBench cross-val (arch-sys F03 GT protocol). lm-eval-harness new task.
- **(검증 시나리오)** GT focus trajectory vs real human drift 상관 측정 (synthetic-only 타당성 gate).

**② 기대효과**: multi-focus 손실 첫 정량화 + Coverage-Energy Pareto 신metric.
**③ 구현변경점 (R52.2 7-col 표)**

| 컴포넌트 | 역할 | 프레임워크 | 파일 (GitHub, R68 main+line-anchored) | 수정 유형 | 난이도 | 검증 |
|---|---|---|---|---|---|---|
| Focus benchmark harness | FCR/Pareto 측정 loop | lm-eval-harness | [`~/skills/AI-Research-SKILLs/11-evaluation`](https://github.com) (local skill) | new task | 중 | skill ✓ |
| Region-text relevance | focus→region grounding | CLIP/SAM | [`~/skills/AI-Research-SKILLs/18-multimodal/clip`](https://github.com) (local skill) | reuse | 하 | skill ✓ |

**④ 검증 trace (R52.3)**: benchmark/CLIP/SAM skill-grounded. GT = human subset + Grounding-DINO/SAM + MileBench cross-val. **CF: 기존 multi-turn 벤치 인정, compression-multi-focus 차별화.**

### M2 — Query-Distribution-Marginal Retention (scorer = LLM cross-attention, TV-δ bound)

**① 동작원리 4요소**
- **(무엇을)** single-query importance scorer 를 query-set marginal `I(t)=Σ_f P(f)·attn(t,f)` 로 교체 (P_focus = saliency prior + Bayesian update).
- **(왜 작동)** 분포-marginal 은 expected coverage maximize → MMTok 의 single-prompt within-instance 와 대비되는 multi-turn distribution 이 생존 differential.
- **(구현변경점, CF)** scorer hook = **LLM cross-attention layer** (query 존재 시점), `_execute_mm_encoder`(query-independent encoder) **아님** (ai-impl F03/arch-sys CF). SparseVLM/FastV scorer 를 marginal 로 교체.
- **(검증 시나리오)** distribution-marginal vs single-query vs uniform random recall@k (focus 분포 skew sweep), submodularity (diminishing returns) toy 검증.

**② 기대효과**: multi-focus FCR +6~12%p (vs single-query, diverse-focus N≥3).
**③ 구현변경점 (R52.2 7-col 표)**

| 컴포넌트 | 역할 | 프레임워크 | 파일 (GitHub, R68 main+line-anchored) | 수정 유형 | 난이도 | 검증 |
|---|---|---|---|---|---|---|
| Distribution scorer | query-set marginal (CF: LLM cross-attn) | vLLM V1 | [`vllm/v1/worker/gpu_model_runner.py#L2869-L2900`](https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/gpu_model_runner.py#L2869-L2900) `_execute_mm_encoder` 아님 → LLM attention layer | hook | 중 | clone ✓ |
| Retention apply | budget 내 token select | vLLM V1 | [`vllm/v1/core/kv_cache_manager.py#L238-L300`](https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/kv_cache_manager.py#L238-L300) `allocate_slots` | extend | 중 | clone ✓ |

**④ 검증 trace (R52.3)**: scorer hook=LLM cross-attention (CF, `_execute_mm_encoder`(L2869)=encoder query-independent 명시 배제). `allocate_slots`(L238) clone ✓.

## 7. End-to-End Evaluation

- **벤치 토대:** LLaVA-OV / Qwen2.5-VL region-grounded QA + lm-eval-harness. MultiVerse / MMDU / ConvBench / MileBench cross-val.
- **합성:** focus-trajectory drift param 제어 (N=2~6 focus, skew sweep) — FCR vs N 단조성, distribution-marginal vs single-query.
- **System:** Coverage-Energy Pareto (Jetson tegrastats J/turn) + R1 hot-tier plug-in 1실험 (16GB footprint+recall trade-off).
- **Baseline:** single-query retention (SparseVLM), uniform random, MMTok(single-prompt coverage), CSP(MileBench), StreamMem(generic-query), CROP(single-region).

## 8. 실험 7요소 (12-16주)

1. **Hardware**: 벤치 생성·측정 RTX 5090/4090 (Ada sm_89) 또는 DGX Spark; retention deploy Jetson Orin NX (1024 core). Coverage-Energy Pareto = Jetson tegrastats.
2. **Model**: LLaVA-OneVision-7B / Qwen2.5-VL-7B (region-grounded QA).
3. **Framework**: lm-eval-harness new task + vLLM cross-attention scorer hook. training-free.
4. **Energy 측정**: TRACE-C(R2) 재사용 — Coverage-Energy Pareto J/turn.
5. **Gate**: GT focus trajectory vs human drift 상관 측정 (synthetic-only 타당성) + submodularity toy 검증. 상관 낮으면 human-curated 50 video 축소.
6. **Steps**: (a) focus-trajectory benchmark + GT (human ≥50 + Grounding-DINO/SAM + MileBench cross-val) → (b) FCR/Pareto metric → (c) focus autocorr data 측정 (R7 전제) → (d) distribution scorer (LLM cross-attn) → (e) single-query 대비 recall.
7. **Metrics**: FCR (per-focus recall@k), Coverage-Energy Pareto (FCR vs J/turn), GT-human 상관, distribution-mismatch TV-δ.

## 9. 예상효과 5-axis 표 (Energy 강조)

| Axis | 예상 개선 | 조건/scope |
|---|---|---|
| Performance ★ | multi-focus FCR +6~12%p (vs single-query) | diverse-focus, N≥3 |
| **Energy/Power** | recall 개선 → re-prefill 빈도 −15~25% (간접) + Coverage-Energy Pareto 신metric | focus-shift 빈번 |
| Cost eff. ★ | benchmark = device-agnostic community 자산 (long-tail citation) | 후속 연구 enabler |
| Memory eff. | 동일 budget 에서 더 높은 coverage | budget-neutral |
| Latency | 불필요 re-prefill 회피 평균 TTFT −10~20% | multi-turn only |

## 10. 관련연구 + 차별화

- **MultiVerse** [arXiv:2510.16641](https://arxiv.org/abs/2510.16641) (647 dialog, 4-turn) / **MMDU / ConvBench / MileBench** — **CF: 기존 multi-turn VLM 벤치 (novelty F02, "부재" framing 무효).** → **차별화: compression-under-multi-focus FCR/Coverage-Energy Pareto 미정의 (일반 conversational capability). 본 idea = compression turn-간 degradation 정량.**
- **MMTok** [arXiv:2508.18264](https://arxiv.org/abs/2508.18264) (maximum coverage, single-prompt) + **Adaptive Greedy Frame** [arXiv:2603.20180](https://arxiv.org/abs/2603.20180) (explicit (1−1/e), single-query) — **CF: A SubCover 의 coverage+(1−1/e) ~70% scoop (novelty F07 critical, A standalone DROP).** → **차별화: coverage 를 future-focus distribution marginal 로만 흡수 (single-prompt 아님), (1−1/e) standalone 주장 폐기.**
- **CSP** [arXiv:2412.04652](https://arxiv.org/abs/2412.04652) — MileBench multi-turn +41% over-pruning, focus-level recall 미분해. → **차별화: FCR 분해.**
- **StreamMem** [arXiv:2508.15717](https://arxiv.org/abs/2508.15717) — "questions in advance impractical", generic-query 단일 vector. → **차별화: 명시적 focus 분포 P_focus.**
- **CROP** [arXiv:2505.21233](https://arxiv.org/abs/2505.21233) (EMNLP'25) — single-query region localization. → **차별화: focus-set union.**
- **Are We Using the Right Benchmark** [arXiv:2510.07143](https://arxiv.org/abs/2510.07143) — compression bench noise critique, multi-turn eval 미제안.

## 11. Implementation Consistency

- **R47 path**: Application-level. vLLM cross-attention scorer hook + lm-eval-harness task. training-free.
- **CF 일관성**: scorer hook = LLM cross-attention (encoder 아님, F03). "multi-turn 벤치 부재" framing 무효 → MultiVerse/MMDU/ConvBench 인정 + compression-multi-focus 차별화 (F02). A SubCover (1−1/e) standalone DROP (MMTok/Adaptive Greedy scoop, F07). A 흡수=distribution-marginal coverage 만.
- **BENCH-D producer**: R1/R3/R5/R6 measurement 토대 + R7 focus autocorr data source.

## 12. Reproducibility Checklist (5 필드)

1. **Clone Spec**: vLLM `7c37096`. `_execute_mm_encoder`(L2869, encoder query-independent 배제), `allocate_slots`(L238) verified, hallucinated 0건.
2. **Environment**: CUDA 12.x, lm-eval-harness, CLIP/SAM/Grounding-DINO, Python 3.10+. 측정 RTX/DGX Spark, deploy Jetson.
3. **Build**: lm-eval-harness new task (FCR/Pareto) + vLLM scorer hook 패치 + GT pipeline.
4. **Patch List**: M1(benchmark harness + CLIP/SAM region grounding + GT) / M2(distribution scorer LLM cross-attn + allocate_slots).
5. **Smoke Test**: FCR 측정 동작, GT-human 상관 ≥ threshold, distribution-marginal vs single-query recall +Yp, Coverage-Energy Pareto J/turn.

## 13. Scoring 및 이유 (R67, phase2prime Section D)

| Reviewer | sub1 | sub2 | sub3 | sub4 | rev-mean |
|---|---|---|---|---|---|
| Novelty (Mech/Comb/Hypo/D2) | 6 ▼ | 6 ▼ | 7 | 7 | 6.5 |
| Differentiation (RW/Clarity/Pos/Scope) | 8 | 8 | 8 | 6 | 7.5 |
| Impact (Mag/Breadth/Adopt/D1) | 6 | **8** ★ | 7 | 6.5 | 6.875 |
| AI-impl (Src/Kernel/Integ/D6) | 6.5 | 7.5 | 7 | **8** | 7.25 |
| Arch-sys (R47/fit/HW/D6) | 8 | 8 | 8 | 7 | 7.75 |

- **★ 전체 최고: Impact-Breadth = 8 + AI-D6 8** — device/model-agnostic, long-tail citation (유일 FastV scale 후보), benchmark 독립 artifact = 최저 마찰.
- **▼ 전체 최저: Novelty-Mech/Comb = 6** — benchmark 점진성.
- **이유**: community enabler. A SubCover scoop 해소 후 **novelty 5.0→6.5 단일 최대 회복** (A drop + distribution-marginal 만 흡수). Coverage-Energy Pareto 고유 metric. R1/R5 measurement 토대 + R3/R7 focus-autocorr data source. **Total mean 7.18 / Grade Conditional Accept.**

## 14. R14.4 Decision Tree

```
preliminary gate: GT focus trajectory vs real human drift 상관
├─ 상관 ≥ threshold (synthetic 타당)
│   ├─ distribution-marginal recall +6%p 이상 → 벤치+retention 묶음 [NeurIPS D&B / ACL benchmark track]
│   └─ marginal ~ uniform (F2) → 벤치 metric 만 contribution (retention drop)
├─ 상관 낮음 (F3) → human-curated 50 video 로 축소 (synthetic over-claim 회피)
├─ focus 분포 skew → single-query 수렴 (F4) → "diverse-focus only" scope 축소
└─ focus prior cross-dataset 실패 (F5) → training-free saliency+autocorr fallback
```

## 15. Inter-idea Dependency

- **R4 = BENCH-D producer**: focus-trajectory benchmark + FCR metric + GT + focus spatial autocorrelation data. **R1/R3/R5/R6 measurement 토대, R7 predictor 학습 data (R7 gate 의 전제).**
- **R4 ← R2 (TRACE-C)**: Coverage-Energy Pareto energy 측정.
- **R4 → R1 (1실험)**: R1 hot-tier 선정 기준 plug-in (16GB footprint+recall trade-off).
- **R4 → R3 (warm-start prior)**: R3 의 T 짧을 때 focus distribution prior 제공.
- **독립 paper unit**: R4 = benchmark NeurIPS D&B / ACL.

## 16. Stakeholder (7-row)

| Stakeholder | 관심사 | R4 제공 가치 |
|---|---|---|
| Compression 연구 커뮤니티 | multi-focus 평가 도구 부재 | FCR/Coverage-Energy Pareto metric |
| Benchmark / D&B 저자 | device/model-agnostic 자산 | long-tail citation enabler |
| R1/R3/R5/R6 저자 | measurement 토대 | BENCH-D producer (재사용) |
| R7 저자 | focus autocorr 입증 data | spatial autocorrelation source |
| Edge 시스템 연구자 | system-relevance | Jetson tegrastats Coverage-Energy Pareto |
| 데이터 annotation 팀 | GT 신뢰성 | human ≥50 + Grounding-DINO/SAM + MileBench cross-val |
| 후속 retention 연구자 | distribution-marginal 방어 | TV-δ Lipschitz bound |

## 17. Boundary (5-axis)

| Axis | In-scope | Out-of-scope |
|---|---|---|
| Task | compression-under-multi-focus FCR | 일반 conversational capability (MMDU) |
| Coverage | future-focus distribution marginal | single-prompt within-instance coverage (MMTok) |
| Scorer | LLM cross-attention (query 존재) | encoder query-independent (`_execute_mm_encoder`) |
| GT | human subset + region grounding | synthetic-only over-claim (gate) |
| Device | benchmark agnostic + Jetson energy axis | HW-specific 측정만 |

## 18. Self-Check

- [x] RQ 3개 의문문 + 정량 (FCR 단조 감소 / +6~12%p·TV-δ / GT-human 상관)
- [x] CF-1 (Coverage-Energy 는 tegrastats user-space, prefetch API 무관)
- [x] CF-2 (scorer=LLM cross-attn, `_execute_mm_encoder` encoder query-independent 배제)
- [x] CF-3 (re-prefill 회피, NVMe/LPDDR5 는 R1/R5)
- [x] CF-4 (Jetson 1024 core, RTX 4090=Ada sm_89 kernel dev 부적합 명시)
- [x] R68 GitHub link `blob/main/{path}#L{A}-L{B}` (fixed path)
- [x] R8 arxiv clickable markdown (MMTok/Adaptive Greedy/CSP/StreamMem/CROP/MultiVerse)
- [x] R67 ★(Impact-Breadth 8·AI-D6 8) / ▼(Novelty-Mech/Comb 6)
- [x] vendor-neutral title (device명 title 없음)
- [x] 18 의무 섹션 전부
