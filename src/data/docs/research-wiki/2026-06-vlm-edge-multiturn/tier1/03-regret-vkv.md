# REGRET-VKV: Regret-Bounded Cross-Turn Importance Re-estimation over Shared-Prefix Vision-KV for Multi-Turn VLM Serving

**Tier-1 · Grade: Conditional Accept (mean 7.25 / Total mean 7.03) · Paper Pair layer (+R1) · weight-only 최경량**

> **Metaphor noun:** *REGRET-VKV* — "regret" 가 online learning (MWU) 의 regret bound, "VKV" 가 vision-KV.
> **Merge 출처:** algo-C OnlineImp + SHARED-VKV(ai-opt). algo-C 의 MWU regret √((T/2)ln n) backbone + SHARED-VKV 의 shared-prefix(RadixAttention) 구현 vehicle.

📖 약어 / 핵심 용어:
- **MWU (Multiplicative Weights Update)**: prediction-with-expert-advice online learning. token=expert, 관측 attention=gain, regret ≤ √((T/2)ln n) (Arora-Hazan-Kale'12, ToC'12 verified).
- **Importance staleness**: 모든 baseline 은 prefill 시점 **한 번** vision token importance 추정 (static). multi-turn 에서는 turn 마다 실제 쓰인 token 이 관측되므로 prefill-salience 가 stale 됨 (RVIS 가 입증).
- **Shared-prefix (RadixAttention)**: 같은 image, 다른 query 의 vision KV 를 1회 prefill 후 재사용 — SGLang `RadixCache` 의 **현존 기능** (신규 아님, CF). 본 idea 의 신규 기여는 budget 초과 시 어느 vision-KV 를 evict 하나 + cross-turn 갱신.
- **regret-gap**: MWU 누적 보존 정확도가 oracle-best-fixed-subset 대비 얼마나 가까운가 (theorem ≠ 실용, F09 — 경험적 threshold 필요).

---

## 1. Research Questions

- **Master RQ**: multi-turn (T≥5) VLM 에서 prefill importance staleness 로 budget-제약 보존 손실이 누적되는데, turn-level MWU re-estimation (관측 attention 으로 보존 set 갱신) 을 shared-prefix vision-KV 위에서 수행하면 static-importance 대비 누적 accuracy 를 회복하면서 regret 을 sublinear 로 줄일 수 있는가?
- **RQ-1**: cross-turn MWU re-estimation 이 static-importance 대비 multi-turn 누적 accuracy 를 **+6~12%p** 회복하는가 (budget-제약, T≥5)?
- **RQ-2**: MWU 누적 보존 정확도가 oracle-best-fixed-subset 의 일정 X% 이내로 수렴하고, regret 이 T 증가에 따라 **sublinear** (√((T/2)ln n)) 로 줄어드는가?
- **RQ-3**: shared-prefix(RadixAttention hit) 로 turn 당 TTFT 를 **−40~60%** 유지하면서 importance 갱신에 의한 cache thrashing (promote/evict 반복 bandwidth) 이 발생하지 않는가?

## 2. Two-Sentence Pitch

모든 baseline 은 prefill 시점 한 번 vision token importance 를 추정(static)하나, multi-turn 에서는 turn 마다 어떤 token 이 실제 쓰였는지 관측되므로 이는 online learning(prediction with expert advice) 문제다. 본 idea 는 turn-level multiplicative-weights(MWU) 로 보존 importance 를 갱신(regret ≤√((T/2)ln n), Arora-Hazan-Kale'12)하여 prefill-salience staleness 를 해소하되, vision KV 는 SGLang RadixAttention 의 shared-prefix(같은 image, 다른 query)로 1회 prefill·재사용한다.

## 3. 가설 + Falsification

**가설:** multi-turn (T≥5) VLM 에서 prefill importance 가 stale 되어 budget-제약 보존 손실이 누적되며, cross-turn MWU re-estimation(관측 attention 으로 보존 set 갱신)을 추가하면 static-importance 대비 누적 accuracy 를 +6~12%p 회복하고, regret 이 T 증가에 따라 sublinear 로 줄며, shared-prefix 로 turn 당 prefill 절감(RadixAttention hit)을 유지한다.

**Falsification (4):**
1. prefill importance 가 turn 간 stable (staleness 없음) → re-estimation 무의미 (단 RVIS 가 shift 입증, 반증 가능성 낮음).
2. **MWU 누적 보존 정확도가 oracle-best-fixed-subset 의 X% 이내로 수렴 못하거나 static 대비 +Yp 미만** → 경험적 regret-gap 기각 (novelty F09: theorem ≠ 실용 이득).
3. importance 갱신으로 보존 set 자주 변경 → cache thrashing (promote/evict 반복) → bandwidth 폭증, 기각.
4. **실사용 multi-turn 길이 분포가 짧으면 (대부분 1-3 turn) MWU prior 미수렴 → static 과 동등** (diff F10 scope risk) → warm-start prior(R4 distribution) 결합으로 완화.

## 4. Workload Evidence (Step 0-α 정량 인용)

- **Importance shift (★ verified, RVIS [arXiv:2604.12358](https://arxiv.org/abs/2604.12358)):** Relevant Visual Information Shift — decoding step 마다 중요한 visual token 이 달라짐; complex reasoning 에서 static pruning 이 generalize 실패. → 본 idea 가 intra-response → inter-turn 으로 일반화.
- **Multi-turn 취약성 (verified, SparseVILA):** "permanently remove ... lossy in multi-turn", query-aware token 을 each conversation round 마다 retrieve 해야 함 → turn 별 fresh importance signal 필요.
- **CSP MileBench:** multi-turn embodied dialogue 에서 over-pruning **+41%** 회복 (KV budget 13.6%↓) → static prune 손실이 multi-turn 에서 누적됨을 직접 측정.
- **Vision 비중:** sample 당 6,272 visual vs 109 text token → shared-prefix leverage 큼 (vision KV 재사용 가치 ≫ text).

## 5. 기준 코드베이스 (R52.1)

- **SGLang** `main` commit `e958f45` (1순위 vehicle). `RadixCache.match_prefix`(L361, +28 drift, `params: *Params` dataclass), `insert`(L421), `evict`(L561). **CF: shared vision prefix 는 RadixAttention 현존 기능** (신규 아님, ai-impl F11). `multimodal_cache.py`(embedding 캐시) ≠ `radix_cache.py`(KV). SGLang multimodal radix 지원 여부 **GitHub 선확인 의무** (미지원 시 vLLM mm_hash 경로).
- **vLLM** `main` commit `7c37096` (fallback). `_gen_mm_extra_hash_keys`(L395) mm_hash 경로.
- **llama.cpp** `main` commit `dbe9c0c` (Jetson 핵심 — SGLang Jetson build 검증 약함).
- **HF model:** `Qwen/Qwen2.5-VL-7B-Instruct`.
- **deps:** model-pruning skill (MWU online 확장), Arora-Hazan-Kale'12 MWU.
- **Jetson Orin NX 16GB 정정 spec (CF-4):** CUDA core **1024**, LPDDR5 **102.4 GB/s**, nvpmodel **10/15/25/40W (40W=JetPack6.2)**, Ampere **sm_87**, `concurrentManagedAccess=0`. weight vector (token 당 scalar) = 메모리 무시가능 (세션 최경량 mechanism). SGLang Jetson build 미검증 → Jetson 핵심은 llama.cpp/vLLM, SGLang RTX 대조.

## 6. 동작 원리 (R53 inline)

### M1 — Cross-Turn MWU Importance Re-estimation (regret-bounded)

**① 동작원리 4요소**
- **(무엇을)** 매 turn 관측 attention 으로 vision token 보존 importance 를 MWU 갱신 (budget-제약 보존 set). token=expert, attention=gain, regret≤√((T/2)ln n).
- **(왜 작동)** RVIS 가 보인 importance shift 의 inter-turn 일반화 → 현재 turn attention 이 fresh signal. MWU 가 사후 최적 고정 set 에 수렴 (regret sublinear).
- **(구현변경점, CF)** scorer = **LLM cross-attention layer** (query 존재 시점), `_execute_mm_encoder`(query-independent encoder) 아님. weight vector = 메모리 무시가능.
- **(검증 시나리오)** MWU vs oracle-best-fixed-subset regret-gap 측정 (T sweep), static-importance 대비 누적 accuracy.

**② 기대효과**: multi-turn 누적 accuracy +6~12%p (vs static, T≥5).
**③ 구현변경점 (R52.2 7-col 표)**

| 컴포넌트 | 역할 | 프레임워크 | 파일 (GitHub, R68 main+line-anchored) | 수정 유형 | 난이도 | 검증 |
|---|---|---|---|---|---|---|
| MWU weight update | turn별 element-wise exp | model-pruning | [`~/skills/AI-Research-SKILLs/19-emerging-techniques/model-pruning`](https://github.com) (local skill) | online 확장 | 중 | skill ✓ |
| Importance scorer | LLM cross-attention (query 존재 시점) | SGLang/vLLM | LLM attention layer (encoder 아님, CF) | hook | 중 | — |
| Importance-guided evict | MWU 보존 set 갱신 | SGLang | [`python/sglang/srt/mem_cache/radix_cache.py#L561-L600`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/mem_cache/radix_cache.py#L561-L600) `evict` (`EvictParams`) | guided evict | 상 | clone ✓ |

**④ 검증 trace (R52.3)**: `evict`(L561, params-dataclass) clone ✓. **CF: scorer=LLM cross-attention (encoder 아님). `multimodal_cache.py`(embedding) ≠ `radix_cache.py`(KV).**

### M2 — Shared-Prefix Vision-KV Reuse (RadixAttention 활용 — 신규 아님)

**① 동작원리 4요소**
- **(무엇을)** 같은 image, 다른 query 의 vision KV 를 RadixAttention shared prefix 로 1회 prefill·재사용. miss 시 R1 cold tier retrieve.
- **(왜 작동)** vision 98-99% 비중 → prefix-share leverage 큼. turn 당 vision 재계산 회피.
- **(구현변경점, CF)** **"shared prefix" 는 RadixAttention 활용 명시 (신규 아님, ai-impl F11)** — 차별화는 budget 초과 시 어느 vision KV evict 하나(M1) + cross-turn 갱신. SGLang multimodal radix 미지원 시 vLLM mm_hash 경로.
- **(검증 시나리오)** RadixAttention hit-rate, turn 당 TTFT 절감, cache thrashing (promote/evict 빈도) 측정.

**② 기대효과**: turn 당 TTFT −40~60% (prefix hit), prefill energy −40~50%.
**③ 구현변경점 (R52.2 7-col 표)**

| 컴포넌트 | 역할 | 프레임워크 | 파일 (GitHub, R68 main+line-anchored) | 수정 유형 | 난이도 | 검증 |
|---|---|---|---|---|---|---|
| Shared vision prefix | RadixAttention 활용 (신규 아님) | SGLang | [`python/sglang/srt/mem_cache/radix_cache.py#L361-L420`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/mem_cache/radix_cache.py#L361-L420) `match_prefix` | reuse | 하 | clone ✓ |
| mm_hash fallback | vLLM 경로 (multimodal radix 미지원 시) | vLLM V1 | [`vllm/v1/core/kv_cache_utils.py#L395-L420`](https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/kv_cache_utils.py#L395-L420) `_gen_mm_extra_hash_keys` | reuse | 중 | clone ✓ |

**④ 검증 trace (R52.3)**: `match_prefix`(L361, +28 drift) clone ✓. **CF: shared-prefix=RadixAttention 현존, 신규 주장 철회 (정직성↑).**

## 7. End-to-End Evaluation

- **Multi-turn / multi-focus:** R4 FOCUS-COVERAGE benchmark + MMDU / ConvBench / MileBench(CSP 41% 회복 비교) / MultiVerse(647 dialog 4-turn).
- **합성:** multi-turn 길이 T sweep (1~10) — regret sublinear 검증, T 분포 motivation.
- **Baseline:** static-importance (FastV/SparseVLM prefill-once), Fast-dDrive(출력 fork, importance 무갱신), VLCache(same-input), RadixAttention(static prefix), oracle-best-fixed-subset (regret-gap 측정).

## 8. 실험 7요소 (12-16주)

1. **Hardware**: Jetson Orin NX 16GB (1024 core). secondary RTX (SGLang 대조).
2. **Model**: Qwen2.5-VL-7B (LLM cross-attention scorer hook).
3. **Framework**: SGLang RadixCache (RTX 우선) + Jetson llama.cpp/vLLM (mm_hash 경로). SGLang multimodal radix 선확인.
4. **Energy 측정**: TRACE-C(R2) 재사용 — prefill energy 절감 (per-turn J).
5. **Gate**: 실사용 multi-turn T 분포 motivation 측정 + MWU vs oracle regret-gap. T 짧으면 warm-start prior(R4) 결합.
6. **Steps**: (a) SGLang multimodal radix 선확인 → (b) shared-prefix reuse → (c) LLM cross-attn scorer + MWU update → (d) importance-guided evict → (e) static 대비 누적 accuracy + regret-gap.
7. **Metrics**: 누적 accuracy (T별), regret-gap (vs oracle), RadixAttention hit-rate, turn 당 TTFT, cache thrashing 빈도.

## 9. 예상효과 5-axis 표 (Energy 강조)

| Axis | 예상 개선 | 조건/scope |
|---|---|---|
| Performance | multi-turn 누적 accuracy +6~12%p (vs static, T≥5) | budget-제약, focus-shift |
| **Energy/Power** | prefix-hit prefill 절감 → prefill energy −40~50% | turn 당 재계산 회피 |
| Memory eff. | weight-only (token 당 scalar) 무시가능 + shared prefix 중복 제거 | multi-turn 多 |
| Latency | turn 당 TTFT −40~60% (prefix hit) | prefix 일치 시 |
| Cost eff. | training-free re-estimation | — |

## 10. 관련연구 + 차별화

- **RVIS / Why Pruning Fails** [arXiv:2604.12358](https://arxiv.org/abs/2604.12358) — Relevant Visual Information Shift (decoding-step 내). → **차별화: intra-response → inter-turn shift 일반화 + regret bound (RVIS 는 진단만).**
- **Fast-dDrive** [arXiv:2605.23163](https://arxiv.org/abs/2605.23163) — shared-prefix KV fork. → **차별화: 출력 구조 fork → 입력 vision prefix share + importance update.**
- **VLCache** [arXiv:2512.12977](https://arxiv.org/abs/2512.12977) — same-input 98% reuse. → **차별화: same-input → different-focus inter-turn re-estimation. baseline 격상.**
- **SGLang RadixAttention** (`radix_cache.py match_prefix`, clone ✓) — **CF: 현존 기능 (신규 아님). 신규 기여=importance-guided evict + cross-turn 갱신.**
- **MMDU** [arXiv:2510.16641](https://arxiv.org/abs/2510.16641) / **Quest**(ICML'24) — scoop 인접: MMDU(multi-turn bench)·Quest(query-aware page selection) 와 본 idea(regret-bounded inter-turn re-estimation) 차별화. concurrent 미발견 (scoop 4/10).
- Arora-Hazan-Kale'12 MWU regret √((T/2)ln n) (ToC'12 verified).

## 11. Implementation Consistency

- **R47 path**: Application-level. SGLang RadixCache importance-guided evict + MWU. GPU 우선(RTX), Jetson 검증.
- **CF 일관성**: "shared vision prefix" 신규 기여 철회 (RadixAttention 현존, F11) → 기여를 importance-guided evict + cross-turn MWU 로 한정. scorer=LLM cross-attention (encoder 아님). `radix_cache.py`(KV) vs `multimodal_cache.py`(embedding) 분리. SGLang multimodal radix 선확인.
- **Paper Pair {R1+R3}**: retrieve-on-miss 가 R1 cold tier 에 의존 (단독 미완 — R1 INFRA-A/B 필수).

## 12. Reproducibility Checklist (5 필드)

1. **Clone Spec**: SGLang `e958f45` / vLLM `7c37096`. `match_prefix`(L361, +28 drift), `evict`(L561, EvictParams), `_gen_mm_extra_hash_keys`(L395) verified, hallucinated 0건.
2. **Environment**: JetPack 6.2 (Jetson), CUDA 12.x, SGLang (RTX), Python 3.10+, model-pruning skill.
3. **Build**: SGLang multimodal radix 지원 확인 → importance-guided evict 패치. 미지원 시 vLLM mm_hash.
4. **Patch List**: M1(MWU update + LLM cross-attn scorer + radix_cache evict) / M2(match_prefix reuse + mm_hash fallback).
5. **Smoke Test**: shared-prefix hit 확인, MWU weight 갱신 동작, static 대비 누적 accuracy +Yp, regret-gap < threshold.

## 13. Scoring 및 이유 (R67, phase2prime Section D)

| Reviewer | sub1 | sub2 | sub3 | sub4 | rev-mean |
|---|---|---|---|---|---|
| Novelty (Mech/Comb/Hypo/D2) | 7 | 7 | 8 ★ | 7 | 7.25 |
| Differentiation (RW/Clarity/Pos/Scope) | 8 | 8 | 8 | 6 ▼ | 7.5 |
| Impact (Mag/Breadth/Adopt/D1) | 6 | 6 | 6 | 7 | 6.25 |
| AI-impl (Src/Kernel/Integ/D6) | 6 | 6.5 | 6.5 | 7 | 6.5 |
| Arch-sys (R47/fit/HW/D6) | 8 | 7.5 | 7.5 | 7.5 | 7.625 |

- **★ 전체 최고: Diff-Clarity/Pos = 8 + Novelty-Hypo 8 + Arch-fit 8** — 모든 경쟁자 static, regret bound unique (concurrent 미발견).
- **▼ 전체 최저: Diff-Scope = 6** — T≥5 실사용 빈도 미검증 (warm-start prior R4 결합으로 완화).
- **이유**: weight-only 세션 최경량 mechanism (Arch 8/10) + AHK'12 regret bound + RVIS inter-turn 일반화. shared-prefix novelty 철회는 정직성↑. retrieve-on-miss 가 R1 cold tier 의존 (결합 자연). **Total mean 7.03 / Grade Conditional Accept.**

## 14. R14.4 Decision Tree

```
preliminary gate: 실사용 multi-turn T 분포 + MWU vs oracle regret-gap
├─ T 분포 충분히 길음 (T≥5 빈도 높음)
│   ├─ regret-gap < threshold + 누적 accuracy +6%p 이상 → R1+R3 통합 [MLSys Paper Pair]
│   └─ regret-gap 큼 / static 대비 미미 (F2) → MWU 제거, 정적 budget 재할당만
├─ T 분포 짧음 (대부분 1-3 turn, F4) → warm-start prior(R4 distribution) 결합 필수
├─ importance 갱신 → cache thrashing (F3) → evict hysteresis (변경 임계) 추가
└─ prefill importance turn-stable (F1) → re-estimation 무의미 → static fallback (가능성 낮음)
```

## 15. Inter-idea Dependency

- **R3 ← R1 (INFRA-A/B)**: retrieve-on-miss 가 R1 cold tier 의존 (evict 대상 KV block). **Paper Pair {R1+R3}**: R1=system backbone, R3=hot-tier 선정/evict 정책 layer.
- **R3 ← R4 (BENCH-D)**: 누적 accuracy/recall 평가 토대 + warm-start prior (T 짧을 때).
- **R3 ← R2 (TRACE-C)**: prefill energy 측정 재사용.

## 16. Stakeholder (7-row)

| Stakeholder | 관심사 | R3 제공 가치 |
|---|---|---|
| MLSys / online learning 연구자 | regret bound 이론 | AHK'12 √((T/2)ln n) inter-turn |
| 대화형 VLM 서비스 | multi-turn 누적 fidelity | static 대비 +6~12%p |
| Framework maintainer (SGLang) | importance-guided evict | radix_cache.py evict 정책 PR |
| Edge 운영자 | turn 당 TTFT/prefill 절감 | shared-prefix −40~60% |
| R1 paper pair 저자 | hot-tier 선정 정책 | cold tier 위 evict layer |
| 후속 연구자 | training-free re-estimation | weight-only 무시가능 overhead |
| 측정 커뮤니티 | regret-gap 경험 검증 | theorem→실용 threshold |

## 17. Boundary (5-axis)

| Axis | In-scope | Out-of-scope |
|---|---|---|
| Modality | vision-KV importance (inter-turn) | text KV re-estimation |
| Shift | inter-turn focus shift | intra-response shift (RVIS 영역) |
| Mechanism | MWU regret-bounded re-estimation | shared-prefix 신규 주장 (RadixAttention 현존) |
| Dependency | R1 cold tier 위 evict 정책 | standalone retrieve infra (R1 필수) |
| Turn length | T≥5 multi-turn | 1-3 turn (warm-start prior 결합 필요) |

## 18. Self-Check

- [x] RQ 3개 의문문 + 정량 (+6~12%p / sublinear regret / TTFT −40~60%)
- [x] CF-1 (retrieve-on-miss 는 R1 framework KV buffer 의존, prefetch API 무관)
- [x] CF-2 (scorer=LLM cross-attn, radix_cache KV ≠ multimodal_cache embedding)
- [x] CF-3 (cold retrieve 는 R1 LPDDR5 logical 의존)
- [x] CF-4 (1024 core, 40W=JetPack6.2, weight-only 무시가능)
- [x] R68 GitHub link `blob/main/{path}#L{A}-L{B}` (fixed path)
- [x] R8 arxiv clickable markdown
- [x] R67 ★(Diff-Clarity/Pos·Novelty-Hypo·Arch-fit 8) / ▼(Diff-Scope 6)
- [x] vendor-neutral title (device명 title 없음)
- [x] 18 의무 섹션 전부
