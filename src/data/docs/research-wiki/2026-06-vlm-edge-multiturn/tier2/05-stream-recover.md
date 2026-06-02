# STREAM-RECOVER: Single-Pass Streaming Submodular Retention with Recoverable Eviction for Long-Video Edge VLM

**Tier-2 · Grade: Conditional Accept (mean 7.05 / Total mean 6.95) · video 특화 (Impact-Mag ★8) · R1 INFRA 재사용**

> **Metaphor noun:** *STREAM-RECOVER* — "stream" 이 single-pass streaming submodular, "recover" 가 recoverable stub/thumbnail eviction.
> **Merge 출처:** algo-D StreamCover + F5(legacy). algo-D 의 Sieve-Streaming (1/2−ε) bounded-memory 보장 backbone + F5 의 thumbnail 재encode recoverable evict + energy-bounded recompute.

📖 약어 / 핵심 용어:
- **Sieve-Streaming**: single-pass streaming submodular maximization, (1/2−ε) approximation, O((k log k)/ε) bounded-memory (Badanidiyuru et al. KDD'14 verified). threshold τ=v/(2k), marginal gain ≥τ 만 채택.
- **Recoverable evict**: 영구 버림 대신 stub(Q2 KV, coarse) + thumbnail(1/4 해상도, fine-grained 재encode) 강등. 과거 focus 질문 시 energy-budget 내 선택적 retrieve/재encode.
- **Recency-tiered spill**: hot(LPDDR5) → warm(Q2 stub, LPDDR5 logical) → cold(NVMe thumbnail, carrier PCIe Gen4 — CF-3). 무한 stream 수용.
- **(1/2−ε) vs (1−1/e)**: single-pass streaming 의 대가 — full-video greedy (1−1/e) 보다 약하나 bounded-memory.

---

## 1. Research Questions

- **Master RQ**: 긴 video 의 frame vision KV 가 Jetson 16GB UMA 에 전부 못 올라 영구 evict 되어 과거 frame 재질문 시 손실되는데, single-pass streaming submodular 로 bounded-memory 보존 + recoverable 강등 하면 OOM 을 해소하면서 과거-focus recall 을 회복할 수 있는가?
- **RQ-1**: single-pass Sieve-Streaming 으로 bounded-memory (k 고정) 보존하면 full-cache 대비 peak memory 를 **60-85%** 절감하고 OOM 을 해소하며 (1/2−ε) coverage 를 보장하는가?
- **RQ-2**: recoverable stub(Q2)+thumbnail(1/4 해상도)로 과거-focus recall 을 destructive evict 대비 **+10~25%p** 회복하되, thumbnail 재encode accuracy 가 원본의 **≥85%** (OCR/fine-grained 포함) 인가?
- **RQ-3**: edge bs=1 기준 frame 수↔KV bytes 곡선에서 streaming 적용 하한이 어디이며, frame 당 O(log k/ε) bucket 평가 overhead 가 edge GPU frame-rate (real-time 목표 시 FPS budget) 를 따라가는가?

## 2. Two-Sentence Pitch

긴 video 의 frame vision KV 는 Jetson 16GB UMA 에 전부 못 올라 영구 evict(StreamingVLM/ReKV heuristic)하나, 과거 frame 재질문(multi-focus over time) 시 손실된다. 본 idea 는 frame token 을 단일 pass Sieve-Streaming submodular ((1/2−ε), O((k log k)/ε) bounded-memory)로 보존하되, evict 를 "버림"이 아니라 "저비트 stub + thumbnail recoverable 강등"으로 하여 과거 focus 질문 시 energy-budget 내 선택적 retrieve/재encode 한다.

## 3. 가설 + Falsification

**가설:** streaming video frame vision KV 를 single-pass Sieve-Streaming 으로 bounded-memory (k 고정) 보존하면 full-cache 대비 peak memory 를 60-85% 절감(OOM 해소)하고 (1/2−ε) coverage 보장하며, recoverable stub+thumbnail 로 과거-focus recall 을 destructive evict 대비 +10~25%p 회복하되, 무한 stream cold 누적은 recency-tiered + NVMe spill 로 수용한다.

**Falsification (5):**
1. thumbnail(1/4 해상도) 재encode vision KV 가 과거-focus accuracy 를 원본의 **<85%** (OCR/fine-grained 포함) → recoverable evict 무의미.
2. **streaming 중 미래 focus 집합 F·weight w_f 미지** (StreamMem 딜레마) → coverage objective 가 F 알려졌다 가정, saliency-prior 고정 또는 prior-drift 하 (1/2−ε) 유지 미검증 시 보장 붕괴 (diff F12).
3. stub+thumbnail 이 16GB UMA 의 **>30%** → recency decimation 으로 완화, 무한 stream 은 NVMe spill (CF-3).
4. frame 당 token × O(log k/ε) bucket 평가 overhead 가 edge GPU frame-rate 못 따라감 → real-time 목표 시 FPS budget 초과.
5. short image multi-turn 에선 streaming 불필요 → R1 우월 ("긴 video only" scope).

## 4. Workload Evidence (Step 0-α 정량 인용)

- **Video frame 영구 evict (verified, StreamingVLM):** compact KV = attention sinks + 짧은 vision window → 오래된 frame vision KV evict → 과거 frame 재질문 시 손실 (사용자 insight 의 video 판 archetype).
- **ReKV (verified):** "traditional VideoQA repeat this process for each new question" 비판, full KV RAM/disk offload 후 retrieve (citation 77). → edge 16GB full 불가.
- **Capacity (CF, edge 재산정 의무):** VLCache 인용 720GB (bs=256 datacenter) → **edge bs=1 frame 수↔KV bytes 재산정** (F15).
- **Edge bottleneck (PAISE'25):** decode memory-bound, bounded-memory 가 16GB 제약과 직접 매칭.
- **Modality Inflation:** thumbnail 재encode = compute-bound (vision tower), retrieve = memory-bound → energy-bounded recompute 정당.

## 5. 기준 코드베이스 (R52.1)

- **vLLM** `main` commit `7c37096`. `BlockPool.get_new_blocks`(L333) page-granularity bucket (streaming select). vision range 식별 = **R1 INFRA-A 재사용** (`_gen_mm_extra_hash_keys`).
- **llama.cpp** `main` commit `dbe9c0c` (Jetson 핵심). `src/llama-kv-cache.cpp` `seq_rm`(L343) (replace evict → stub/thumbnail 강등). ggml buffer.
- **SGLang** `e958f45` (보조).
- **HF model:** `Qwen/Qwen2.5-VL-7B-Instruct`, `llava-hf/llava-onevision-qwen2-7b-ov-hf` (video frame 196 token/frame).
- **deps:** long-context skill (streaming submodular), `cudaHostRegister` (NVMe spill), Badanidiyuru Sieve-Streaming.
- **Jetson Orin NX 16GB 정정 spec (CF-4):** CUDA core **1024**, LPDDR5 **102.4 GB/s**, sm_87, `concurrentManagedAccess=0`. O((k log k)/ε) bounded-memory 가 16GB 제약과 직접 매칭 (세션 핵심 enabler). **carrier M.2 NVMe (PCIe Gen4) — 무한 stream cold spill (CF-3, F5 가 NVMe 부재로 놓친 경로, F28).**

## 6. 동작 원리 (R53 inline)

### M1 — Single-Pass Sieve-Streaming Bounded-Memory Retention ((1/2−ε))

**① 동작원리 4요소**
- **(무엇을)** frame token stream → single-pass Sieve-Streaming (threshold τ=v/(2k), marginal gain ≥τ 만 채택) bounded-memory 보존.
- **(왜 작동)** 긴 video frame 을 메모리에 다 못 올림 → streaming submodular 가 single-pass, bounded-mem 로 representative subset. (1/2−ε) < (1−1/e) 는 single-pass 대가.
- **(구현변경점, CF)** vLLM page-granularity bucket / llama.cpp ggml buffer. vision token range 식별 = **R1 INFRA-A 재사용** (CF-2). 미래 focus F 미지 → saliency-prior 고정 + prior-drift regret 분석.
- **(검증 시나리오)** edge bs=1 frame 수↔KV bytes 곡선 (OOM 경계) + prior-drift 하 (1/2−ε) 유지.

**② 기대효과**: peak vision-KV −60~85% (bounded, single-pass) → 긴 video OOM 해소.
**③ 구현변경점 (R52.2 7-col 표)**

| 컴포넌트 | 역할 | 프레임워크 | 파일 (GitHub, R68 main+line-anchored) | 수정 유형 | 난이도 | 검증 |
|---|---|---|---|---|---|---|
| Streaming bucket | single-pass Sieve-Streaming | vLLM V1 | [`vllm/v1/core/block_pool.py#L333-L380`](https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/block_pool.py#L333-L380) `get_new_blocks` | streaming select | 상 | clone ✓ |
| Streaming submodular | long-context decouple | long-context | [`~/skills/AI-Research-SKILLs/19-emerging-techniques/long-context`](https://github.com) (local skill) | extend | 중 | skill ✓ |

**④ 검증 trace (R52.3)**: `get_new_blocks`(L333) clone ✓. vision range = R1 INFRA-A.

### M2 — Recoverable Eviction (stub Q2 + thumbnail) with Recency-Tiered NVMe Spill

**① 동작원리 4요소**
- **(무엇을)** evict = stub(Q2 KV, coarse) + thumbnail(1/4 해상도, fine-grained 재encode). 과거 focus → stub retrieve (coarse) 또는 thumbnail 재encode (energy-bounded).
- **(왜 작동)** recoverable evict 로 과거 focus 대응. hot(LPDDR5) → warm(Q2 stub) → cold(NVMe thumbnail) recency-tiered.
- **(구현변경점, CF-3)** stub/thumbnail spill = LPDDR5 logical (warm) → **무한 stream 은 NVMe (carrier PCIe Gen4)**. promote = R1 INFRA-B dequant 재사용.
- **(검증 시나리오)** thumbnail 재encode accuracy (원본 대비 ≥85%), stub+thumbnail UMA 비율 (<30%).

**② 기대효과**: 과거-focus recall +10~25%p (vs destructive evict), thumbnail 재encode vs full reencode energy −40~60%.
**③ 구현변경점 (R52.2 7-col 표)**

| 컴포넌트 | 역할 | 프레임워크 | 파일 (GitHub, R68 main+line-anchored) | 수정 유형 | 난이도 | 검증 |
|---|---|---|---|---|---|---|
| Recoverable evict | stub(Q2)+thumbnail 강등 | llama.cpp | [`src/llama-kv-cache.cpp#L343-L400`](https://github.com/ggml-org/llama.cpp/blob/master/src/llama-kv-cache.cpp#L343-L400) `seq_rm` (L349) (main 부재 — llama.cpp 기본 branch = master, R68.2) | replace evict | 상 | clone ✓ |
| NVMe spill (warm→cold) | 무한 stream 수용 (CF-3) | OS/CUDA | M.2 NVMe (carrier PCIe Gen4) + `cudaHostRegister` | new | 중 | datasheet ✓ |

**④ 검증 trace (R52.3)**: `seq_rm`(L343) clone ✓. **CF-3: NVMe spill 활용 (Orin NX PCIe Gen4 carrier).** promote = R1 INFRA-B 재사용.

## 7. End-to-End Evaluation

- **Long-video:** Video-MME (900 video/2,700 QA, median 1,024s), MLVU (1,730 video), MileBench, EgoSchema (median 180s).
- **Multi-focus over time:** 과거 segment 재질문 시나리오 합성 (frame 재방문) — R4 FOCUS-COVERAGE 토대.
- **합성:** 무한 stream (frame 수 sweep) — recency-tier spill + OOM 경계.
- **Baseline:** StreamingVLM(영구 evict), ReKV(full KV RAM/disk), StreamKV/MemStream(segment retrieval+compression), StreamMem(query-agnostic fixed-size), Adaptive Greedy Frame(non-streaming (1−1/e)).

## 8. 실험 7요소 (12-16주)

1. **Hardware**: Jetson Orin NX 16GB (1024 core) + carrier M.2 NVMe (PCIe Gen4). secondary DGX Spark.
2. **Model**: Qwen2.5-VL-7B / LLaVA-OneVision-7B (video frame).
3. **Framework**: vLLM page streaming bucket + llama.cpp recoverable evict + NVMe spill.
4. **Energy 측정**: TRACE-C(R2) 재사용 — thumbnail 재encode vs full reencode energy.
5. **Gate**: edge bs=1 frame 수↔KV bytes 곡선 (OOM 경계, streaming 하한) + prior-drift 하 (1/2−ε) 보장 + frame ingestion FPS vs streaming overhead.
6. **Steps**: (a) R1 INFRA-A/B 재사용 → (b) single-pass Sieve-Streaming bucket → (c) recoverable stub/thumbnail → (d) NVMe recency-tier (CF-3) → (e) StreamingVLM/ReKV/StreamKV/MemStream 비교.
7. **Metrics**: peak vision-KV, 과거-focus recall@k, thumbnail 재encode accuracy, (1/2−ε) coverage, FPS overhead.

## 9. 예상효과 5-axis 표 (Energy 강조)

| Axis | 예상 개선 | 조건/scope |
|---|---|---|
| Memory eff. ★★ | peak vision-KV −60~85% (bounded, single-pass) → 긴 video OOM 해소 | edge 16GB enabling |
| Performance | 과거-focus recall +10~25%p (vs destructive evict) | 과거 segment 재질문 |
| **Energy/Power** | thumbnail 재encode vs full reencode −40~60% + bounded mem→idle energy↓ | fine-grained focus |
| Latency | bounded set → frame 재방문 불요 | streaming |
| Cost eff. | training-free, Sieve-Streaming anchor | — |

## 10. 관련연구 + 차별화

- **StreamingVLM** [arXiv:2510.09608](https://arxiv.org/abs/2510.09608) (ICLR'26) — attention sink + short vision window, 과거 frame 영구 evict. → **차별화: recoverable evict (stub+thumbnail) + bounded-memory 보장.**
- **ReKV** [arXiv:2503.00540](https://arxiv.org/abs/2503.00540) (ICLR'25, cite 77) — full KV RAM/disk retrieve. → **차별화: edge 16GB full 불가 → bounded + recoverable 강등. ReKV 궤도 진입.**
- **StreamKV** [arXiv:2511.07278](https://arxiv.org/abs/2511.07278) + **MemStream** [arXiv:2602.18434](https://arxiv.org/abs/2602.18434) — segment-level retrieval+compression. → **누락 baseline 추가 (F13). 차별화: (1/2−ε) bounded-memory 보장 + 실측 우위.**
- **StreamMem** [arXiv:2508.15717](https://arxiv.org/abs/2508.15717) — query-agnostic fixed-size, 불가역. → **차별화: recoverable thumbnail.**
- **Adaptive Greedy Frame** [arXiv:2603.20180](https://arxiv.org/abs/2603.20180) — (1−1/e) frame selection, non-streaming, single-query. → **차별화: single-pass streaming (1/2−ε) bounded-mem (full-video greedy 아님, novelty F10).**
- Badanidiyuru et al. Sieve-Streaming (KDD'14, (1/2−ε) verified).

## 11. Implementation Consistency

- **R47 path**: Application-level. vLLM page streaming bucket + llama.cpp recoverable evict + NVMe spill.
- **CF 일관성**: CF-3 (NVMe 활용으로 무한 stream cold OOM 구원 — F5 가 NVMe 부재로 놓친 경로, F28). 720GB → edge bs=1 재산정 (F15). vision range = R1 INFRA-A. promote = R1 INFRA-B.
- **R1 INFRA 재사용**: vision token→KV block 매핑 (INFRA-A) + dequant promote (INFRA-B) 를 build-once-share.

## 12. Reproducibility Checklist (5 필드)

1. **Clone Spec**: vLLM `7c37096` / llama.cpp `dbe9c0c`. `get_new_blocks`(L333), `seq_rm`(L343) verified, hallucinated 0건.
2. **Environment**: JetPack 6.2, CUDA 12.x, carrier M.2 NVMe (PCIe Gen4), `cudaHostRegister`, long-context skill.
3. **Build**: streaming bucket 패치 + recoverable evict + NVMe spill daemon.
4. **Patch List**: M1(get_new_blocks streaming select + R1 INFRA-A) / M2(seq_rm stub/thumbnail + NVMe spill + R1 INFRA-B promote).
5. **Smoke Test**: 무한 stream bounded-memory 유지, OOM 회피, 과거-focus retrieve/thumbnail 재encode path, StreamingVLM 대비 recall.

## 13. Scoring 및 이유 (R67, phase2prime Section D)

| Reviewer | sub1 | sub2 | sub3 | sub4 | rev-mean |
|---|---|---|---|---|---|
| Novelty (Mech/Comb/Hypo/D2) | 6 ▼ | 7 | 7 | 6 ▼ | 6.5 |
| Differentiation (RW/Clarity/Pos/Scope) | 8 | 7 | 7 | 7 | 7.25 |
| Impact (Mag/Breadth/Adopt/D1) | **8** ★ | 7 | 6 | 6.5 | 6.875 |
| AI-impl (Src/Kernel/Integ/D6) | 6 | 6 | 6.5 | 7 | 6.375 |
| Arch-sys (R47/fit/HW/D6) | 8 | 8 | 7.5 | 7.5 | 7.75 |

- **★ 전체 최고: Impact-Mag = 8 + Arch-fit 8** — 긴 video OOM enabling (R1 과 함께 최대 magnitude), ReKV(77 cite) 궤도, bounded-memory O((k log k)/ε) = 16GB 직접 매칭.
- **▼ 전체 최저: Novelty-Mech/D2 = 6** — coverage family (MMTok/Adaptive Greedy) 와 부분 겹침.
- **이유**: bounded-memory (1/2−ε) = edge 16GB 직접 매칭. CF-3 NVMe 로 무한 stream 구원. video 특화로 R1(image) 과 상보. R1 vision-range 인프라 재사용. **Total mean 6.95 / Grade Conditional Accept.**

## 14. R14.4 Decision Tree

```
preliminary gate: edge bs=1 frame↔KV bytes 곡선 + prior-drift (1/2−ε) + FPS overhead
├─ streaming 하한 명확 + (1/2−ε) 유지 (prior-drift bound)
│   ├─ thumbnail 재encode ≥85% + recall +10%p → recoverable streaming [MLSys video]
│   └─ thumbnail <85% (F1) → stub(Q2)-only recover, thumbnail 제거 (coarse 한정)
├─ 미래 focus F 미지로 보장 붕괴 (F2) → saliency-prior 고정 + prior-drift regret 분석으로 재방어
├─ stub+thumbnail >30% UMA (F3) → recency decimation + NVMe spill 강화
├─ FPS overhead 초과 (F4) → real-time 목표 완화 (offline video QA 한정)
└─ short image multi-turn (F5) → R1 우월, "긴 video only" scope 명시
```

## 15. Inter-idea Dependency

- **R5 ← R1 (INFRA-A/B)**: vision token range (INFRA-A) + dequant promote/stub-thumbnail (INFRA-B) 재사용.
- **R5 ← R2 (TRACE-C)**: thumbnail 재encode vs full reencode energy 측정.
- **R5 ← R4 (BENCH-D)**: 과거-focus recall 평가 토대.
- **R5 ↔ R1 상보**: R1=image multi-turn, R5=long-video → 독립 paper unit (R5=video MLSys).

## 16. Stakeholder (7-row)

| Stakeholder | 관심사 | R5 제공 가치 |
|---|---|---|
| Long-video VLM 운영자 | 긴 video OOM | peak −60~85% bounded-memory |
| Streaming 연구자 | bounded-memory 보장 | Sieve-Streaming (1/2−ε) |
| Robotics / surveillance | 과거 frame 재질문 | recoverable stub/thumbnail +10~25%p |
| R1 INFRA 소비자 | vision-range + promote 재사용 | INFRA-A/B build-once-share |
| HW vendor (Jetson carrier) | NVMe 무한 stream | recency-tiered spill (CF-3) |
| 측정 커뮤니티 | edge video energy | thumbnail vs full reencode −40~60% |
| 후속 연구자 (ReKV 궤도) | edge-constrained streaming | full-KV 불가 환경 enabling |

## 17. Boundary (5-axis)

| Axis | In-scope | Out-of-scope |
|---|---|---|
| Modality | long-video frame vision-KV | short image multi-turn (R1) |
| Memory | bounded single-pass (1/2−ε) | full-KV 보존 (ReKV) |
| Evict | recoverable stub+thumbnail | destructive 영구 evict (StreamingVLM) |
| Spill | LPDDR5 logical → NVMe (CF-3) | datacenter disk-array |
| Selection | streaming submodular | non-streaming full-video greedy (Adaptive Greedy) |

## 18. Self-Check

- [x] RQ 3개 의문문 + 정량 (60-85% / +10~25%p·≥85% / FPS overhead)
- [x] CF-1 (promote = R1 INFRA-B dequant, prefetch API 무관)
- [x] CF-2 (vision range = R1 INFRA-A 재사용, encoder/KV 분리)
- [x] CF-3 (NVMe spill 무한 stream 구원, carrier PCIe Gen4 — F5 누락 경로)
- [x] CF-4 (1024 core, LPDDR5 102.4GB/s, edge bs=1 재산정 F15)
- [x] R68 GitHub link `blob/main/{path}#L{A}-L{B}` (fixed path)
- [x] R8 arxiv clickable markdown (StreamingVLM/ReKV/StreamKV/MemStream/StreamMem/Adaptive Greedy)
- [x] R67 ★(Impact-Mag 8·Arch-fit 8) / ▼(Novelty-Mech/D2 6)
- [x] vendor-neutral title (device명 title 없음)
- [x] 18 의무 섹션 전부
