# VLM Edge Serving on Jetson Orin NX 16GB — Multi-turn / Multi-focus Vision-Token Cache-Reuse Vulnerability

> Vision token pruning 은 single-turn 에선 거의 무손실이지만, 같은 image/video 에 **query focus 가 turn 마다 이동하는 multi-turn/multi-focus** 시나리오에선 turn-1 기준 prune 된 vision KV 가 turn-2 의 다른 focus 정보를 이미 버려서 무너진다. 이 현상은 SparseVILA·CSP·RVIS 가 **이미 확인**했으나, 누구도 **Jetson 16GB UMA memory-bound + energy/DVFS + multi-focus recall 정량** 의 교집합을 다루지 않았다 — 이 narrow gap 을 reconstruction/regret/competitive/coverage 보장과 함께 점유하는 6 idea.

> **세션 일자**: 2026-06-02 · **Mode**: 2 (local PDF corpus: Focus HPCA'26 / BLASST MLSys'26 / Fast-dDrive NVIDIA) · **참여 expert**: legacy-system (lead) + ai-optimization + algorithm (sub: hw-pim-accelerator) · **참여 reviewer**: novelty + differentiation + impact + ai-implementation + arch-system-implementation (5인) · **세션 로그**: ideation harness 내부 세션 기록에 보관 (Phase 1→2→1'→2'→1'' 상세, homepage 미게시)

---

## 1. Research Questions

**Master RQ**: Jetson Orin NX 16GB UMA 라는 memory-bound·energy-bound edge 제약 하에서, multi-turn/multi-focus VLM serving 의 vision-token cache-reuse 취약성을 (a) destructive prune 없이 recoverable 하게 방어하고, (b) prune-vs-retrieve 의 energy trade-off 를 정량화하며, (c) multi-focus recall 을 처음으로 측정 가능하게 하는 single-system 구현 가능한 idea 6 개를 어떻게 도출할 것인가?

- **RQ-Master-1 (Memory enabling)**: SparseVILA 의 "retain most of visual cache" 가 물리적으로 불가능한 16GB UMA 에서, vision KV 를 reconstruction-bounded recoverable tier 로 관리하여 full-cache 대비 footprint −60~76% + multi-focus accuracy ≥92% 회복이 가능한가? (R1, R5)
- **RQ-Master-2 (Energy)**: commodity Jetson SW-only 에서 prune→재encode(compute) vs retrieve(memory) vs recompute 의 energy-optimal modality 가 nvpmodel power-mode 에 따라 역전(cross-over)하는가, 그리고 ski-rental 정책이 고정 정책 대비 session energy −15~30% + 2-competitive 를 보장하는가? (R2, R6)
- **RQ-Master-3 (Multi-focus recall 정량)**: multi-turn focus trajectory 위에서 single-query retention 의 focus-coverage recall 이 focus 수 N 에 단조 감소함을 측정하고, distribution-marginal / regret-bounded online re-estimation 이 +6~12%p 개선하는가? (R3, R4)

| idea | RQ | 한 줄 |
|---|---|---|
| **R1 RECOVER-TIER** | M1 | hot(low-rank+quant, `‖V−V̂‖_F≤ε`) + cold(recoverable, LPDDR5 logical) dual-tier, focus-shift 시 block dequant promote |
| **R2 DVFS-PILOT** | M2 | 3-way prune/retrieve/recompute energy modality 를 nvpmodel × ski-rental(2-competitive)로 online 선택 |
| **R3 REGRET-VKV** | M3 | shared-prefix vision-KV 위 turn-level MWU importance re-estimation (regret ≤ √((T/2)ln n)) |
| **R4 FOCUS-COVERAGE** | M3 | multi-focus FCR/Coverage-Energy Pareto benchmark + distribution-marginal retention (TV-δ bound) |
| **R5 STREAM-RECOVER** | M1 | 긴 video frame 의 single-pass Sieve-Streaming ((1/2−ε) bounded-mem) + recoverable stub/thumbnail evict |
| **R6 VKV-SKIP** | M2 | non-destructive vision-aware softmax-threshold V-load skip on Ampere SM87 (cache 비파괴 = multi-focus 안전) |

---

## 2. Essential Reading (3-5편, R28.1.2)

학생이 본 세션 idea 구현 전 반드시 읽어야 할 baseline + scoop 후보:

| Paper | Venue | 왜 읽어야 하는가 | Idea 분담 |
|---|---|---|---|
| [SparseVILA: Decoupling Visual Sparsity for Efficient VLM Inference (arXiv:2510.17777)](https://arxiv.org/abs/2510.17777) | ICCV 2025 | **🔴 closest competitor**. "permanently remove visual tokens ... quite lossy in multi-turn" 문제 정의 + "retaining most of the visual cache so that query-aware tokens can be retrieved" 해법. 단 GPU/AWQ, edge/Jetson/memory/energy 무 + no formal guarantee → R1/R3 의 차별화 anchor (16GB 에선 full 보존 불가 + reconstruction/regret bound) | R1, R3 |
| [Focus: A Streaming Concentration Architecture for Efficient VLMs (arXiv:2512.14661)](https://arxiv.org/abs/2512.14661) | HPCA 2026 (corpus A) | **primary 차별화 대상**. SEC 가 query-conditional destructive prune — Fig.2(a) 가 prompt 별 attention 이동을 스스로 입증하나 single-turn 한정. "query-conditional pruning = single-turn-safe but multi-turn-vulnerable" 의 대표 architecture. 단 dedicated systolic-array HW (Jetson GPU 이식 불가) | R1, R6 (commodity SW 대조) |
| [BLASST: Dynamic BLocked Attention Sparsity via Softmax Thresholding (arXiv:2512.12087)](https://arxiv.org/abs/2512.12087) | MLSys 2026 (corpus B) | **positive reference / design principle**. online-softmax running-max 재사용으로 **KV 비파괴 + query 별 V-load(HBM) skip** (decode 1.48×@73.2%). "non-destructive query-adaptive" 의 LLM-side 선례 → R6 가 vision-aware + Ampere kernel 로 확장 (단 text-LLM/Blackwell kernel) | R6 (직접 base) |
| [MMTok: Maximum Coverage for VLM Vision Token Selection (arXiv:2508.18264)](https://arxiv.org/abs/2508.18264) | (2025-08) | **scoop 회피 필독**. coverage + (1−1/e) for VLM token (single-prompt within-instance) → A SubCover 의 coverage+(1−1/e) ~70% 선점. R4 가 coverage 를 **multi-turn future-focus distribution marginal** 로만 흡수 (single-prompt 아님), (1−1/e) standalone 주장 폐기 | R4 (차별화), A SubCover DROP 근거 |
| [Modality Inflation: Energy Characterization for MLLM Inference (arXiv:2512.22695)](https://arxiv.org/abs/2512.22695) | (2025-12) | R2/R6 의 workload evidence. energy overhead 17-94%, vision-encode=compute-bound / decode=memory-bandwidth-bound 이분법 + substantial GPU underutilization (→ DVFS 여지 + R7 compute-idle 근거). prune(compute)/retrieve(memory) energy 비대칭의 직접 근거 | R2, R6, R7 |

추가 필독 (전체는 §8 + papers.md): [StreamMem (arXiv:2508.15717)](https://arxiv.org/abs/2508.15717) (query 사전미지 = multi-focus 본질), [Cross-Self KV/CSP (arXiv:2412.04652)](https://arxiv.org/abs/2412.04652) (MileBench multi-turn +41%), [RVIS/Why Pruning Fails (arXiv:2604.12358)](https://arxiv.org/abs/2604.12358) (decoding-step relevant visual info shift), [ReKV (arXiv:2503.00540)](https://arxiv.org/abs/2503.00540) (video KV offload+retrieve), [KVSwap (arXiv:2511.11907)](https://arxiv.org/abs/2511.11907) (UMA capacity wall, text-only), [PAISE'25 (arXiv:2506.09554)](https://arxiv.org/abs/2506.09554) (Jetson decode memory-bound, freq↓ energy +72%).

---

## 3. 연구 개요 + GAP outline (R28.1.3)

### 3.1 연구 개요

- **vision token 이 VLM input 의 98-99%** (LLaVA-OneVision @ VideoMME sample 당 6,272 visual vs 109 text token, corpus A 본문 verified) → vision KV 가 edge 16GB 의 binding capacity constraint.
- **single-turn pruning 은 거의 무손실** — FastV(50% prune), VisionZip(10% token→95%), PyramidDrop(FLOPs −55%), FastVID(98.0%@90.3% prune) 가 일관 입증.
- **그러나 multi-turn/multi-focus 에선 무너진다** — SparseVILA("quite lossy in multi-turn"), CSP(MileBench over-pruning +41% 회복), RVIS(decoding-step relevant visual info drastic shift), VisionZip("previous methods underperform in multi-turn") 가 현상을 **외부에서 직접 확인** (R31: wrong-insight 아님).
- **Jetson Orin decode 는 memory-bandwidth-bound** (memory freq↓ 시 latency +370%/energy +72%, PAISE'25 verified). vision-encode=compute-bound / decode=memory-bound 이분법 (Modality Inflation) → prune(재encode, compute) vs retrieve(memory) 의 energy 비대칭.
- **UMA 특유성**: Orin NX 는 CPU+iGPU 가 동일 LPDDR5 (102.4 GB/s) 공유 — hot/cold 가 **같은 물리 메모리** (zero-copy SoC DRAM). "promote" = physical migration 이 아니라 **dequant compute**. `concurrentManagedAccess=0` 이라 `cudaMemPrefetchAsync` 미지원 (CF-1).
- **세 corpus 의 설계 원리**: BLASST(non-destructive query-adaptive skip) + Fast-dDrive(shared-prefix KV fork) + SparseVILA/StreamMem(decouple = query-agnostic cache 보존 + per-turn query-aware retrieval) 의 교집합에 본 세션 design space 가 위치.
- **R31 reframe**: 순수 "문제 발견" contribution 은 scooped → "edge-constrained 정량 + 방어 메커니즘 + 형식 보장" 으로 재정렬.

### 3.2 GAP outline (reference cite)

- **GAP-1 (Edge UMA memory-bound)**: SparseVILA([arXiv:2510.17777](https://arxiv.org/abs/2510.17777))의 "retain most of visual cache" 가 16GB UMA 에선 불가 (VLCache 1,000-frame KV ~720GB [⚠️ bs=256 datacenter, edge bs=1 재산정 의무]). KVSwap([arXiv:2511.11907](https://arxiv.org/abs/2511.11907))은 UMA capacity wall 다루나 **text-only**. → vision-aware recoverable tier 공백.
- **GAP-2 (Energy/DVFS-aware)**: Modality Inflation([arXiv:2512.22695](https://arxiv.org/abs/2512.22695))이 stage-DVFS 효과 확인하나 **vision-KV recover 정책 미결합 + A100(edge 아님)**. V-Rex([arXiv:2512.12284](https://arxiv.org/abs/2512.12284), HPCA'26)는 dedicated HW. Sustainability-aware([arXiv:2512.04088](https://arxiv.org/abs/2512.04088))는 text-LLM routing. → commodity Jetson SW-only 3-way recover energy model 공백.
- **GAP-3 (Multi-focus recall 정량)**: query-aware pruning(SparseVLM, CROP [arXiv:2505.21233](https://arxiv.org/abs/2505.21233))은 single-query conditional. CSP([arXiv:2412.04652](https://arxiv.org/abs/2412.04652))는 multi-turn 손실 측정하나 focus-level recall 미분해. MMDU/ConvBench/MileBench/MultiVerse([arXiv:2510.16641](https://arxiv.org/abs/2510.16641))는 일반 conversational capability — compression-under-multi-focus FCR metric 미정의. → multi-focus recall metric 자체 공백.
- **GAP-4 (정적 importance staleness)**: 모든 baseline 이 prefill 시점 한 번 importance 추정. RVIS([arXiv:2604.12358](https://arxiv.org/abs/2604.12358))는 intra-response shift 만 — inter-turn online re-estimation + regret bound 공백.
- **GAP-5 (destructive vs non-destructive)**: 모든 vision pruning 이 destructive (cache 파괴, multi-focus lossy). BLASST([arXiv:2512.12087](https://arxiv.org/abs/2512.12087))의 non-destructive skip 은 vision-modality redundancy 미활용 + Blackwell/Hopper kernel. Quest(ICML'24)/MInference(NeurIPS'24) 인접. → vision-aware non-destructive skip on Ampere 공백 (단 scoop risk 잔존).

### 3.3 GAP → Idea Mapping

| GAP | Tier-1 | Tier-2 | 미선정 |
|---|---|---|---|
| GAP-1 Edge UMA memory-bound | **R1 RECOVER-TIER** (dual-tier + reconstruction) | **R5 STREAM-RECOVER** (streaming + recoverable evict) | — |
| GAP-2 Energy/DVFS-aware | **R2 DVFS-PILOT** (3-way ski-rental) | **R6 VKV-SKIP** (V-load skip energy) | R7 (latency) |
| GAP-3 Multi-focus recall 정량 | **R3 REGRET-VKV** (recall via re-estimation) | **R4 FOCUS-COVERAGE** (FCR benchmark + Pareto) | A SubCover→R4 |
| GAP-4 정적 importance staleness | **R3 REGRET-VKV** (MWU inter-turn) | **R4 FOCUS-COVERAGE** (distribution-marginal) | — |
| GAP-5 destructive vs non-destructive | — | **R6 VKV-SKIP** (non-destructive skip) | — |

---

## 4. Implementation-Priority Decision Tree (R14.4 양식 A)

```
[Step 0: 학기 시작 — 1 학생, Jetson Orin NX 16GB primary]
  │
  ├─→ Jetson Orin NX JetPack/CUDA + concurrentManagedAccess 실측 (=0 예상)?
  │     ├─ Yes → cudaStreamAttachMemAsync/Priority path (CF-1). 진행
  │     └─ Jetson Thor (Blackwell, =1) 보유 → prefetch-API 대조군 추가
  │
  ├─→ Qwen2.5-VL-7B Q4_K (4.4GB) + dual-tier KV 가 16GB fit?
  │     ├─ Yes → Step 1
  │     └─ No (OOM) → LLaVA-OV-0.5B / Qwen2.5-VL-3B fallback
  │
  ▼
[Step 1: Producer-first — 측정 토대 먼저 (모든 idea 의존)]
  │
  ├─→ [BENCH-D] R4 focus-trajectory benchmark + FCR/Coverage-Energy Pareto + focus autocorr 측정
  │     │   (GT: human ≥50 + Grounding-DINO/SAM + MileBench cross-val)
  │     ├─ GT vs human drift 상관 OK → BENCH-D 확정 (R1/R3/R5/R6 평가 토대 + R7 전제)
  │     └─ 상관 낮음 → human-curated 50 video 로 축소
  │
  ├─→ [TRACE-C] R2 rail-level energy probe (INA3221 sysfs) + cross-over pilot (3 modality × 3 mode)
  │     ├─ energy 역전(cross-over) 존재 → R2 = Tier-1 MLSys (ski-rental selector)
  │     └─ 역전 부재 → R2 = measurement contribution (IISWC/DATE), drop 아님
  │
  ▼
[Step 2: R1 flagship INFRA 구축 → consumer 가 재사용]
  │
  ├─→ [INFRA-A/B] R1 vision token→KV block 매핑 (신규) + block requantize + dequant promote
  │     │   uniform-Q4(no-tier) vs dual-tier ablation + SVD compute pilot
  │     ├─ tiering·low-rank marginal gain ≥5% (vs quant-only) → R1 Accept (flagship)
  │     └─ marginal <5% → quant-only 로 축소 (정직한 fallback)
  │
  ├─→ [Paper Pair] R3 = R1 cold tier 위 importance-guided evict + cross-turn MWU
  │     │   T 분포 motivation + MWU vs oracle regret-gap 측정
  │     ├─ T≥5 빈번 + regret-gap 수렴 → R1+R3 통합 MLSys 1편
  │     └─ T 짧음 → warm-start prior(R4) 결합
  │
  ▼
[Step 3: Tier-2 consumer (R1 INFRA + R4 BENCH + R2 TRACE 재사용)]
  │
  ├─→ R5 video: INFRA-A/B 재사용 + Sieve-Streaming (1/2−ε) + recoverable stub/thumbnail + NVMe recency-tier
  │     └─ edge bs=1 frame↔KV 곡선 + prior-drift (1/2−ε) 보장 + FPS overhead gate
  │
  ├─→ R6 (조건부): vision vs text redundancy 분포 pilot (KL/Wasserstein, accuracy-first)
  │     ├─ 분포 분리 + Quest marginal gain 입증 → SM87 block-skip kernel (FlashInfer Ampere)
  │     └─ 미입증 / kernel 실패 → accuracy-only contribution 또는 DROP
  │
  └─→ R7 (component, time-permitting): R1 INFRA-B 재사용 + compute-idle dequant + R4 autocorr predictor
        ├─ compute-idle 존재 + autocorr > uniform → R1 latency-hiding component 부활
        └─ gate 실패 → R1 reactive promote 로 충분 (R7 보류)
```

### 6 idea × phase × branch 표 (cross-share producer 명시)

| idea | producer/consumer | 의존 인프라 | gate (pilot) | pass → | fail → |
|---|---|---|---|---|---|
| **R1** RECOVER-TIER | **producer: INFRA-A/B** | (자체 구축) | uniform-Q4 vs dual-tier + SVD pilot | Tier-1 Accept (flagship), MLSys/ASPLOS | quant-only 축소 |
| **R2** DVFS-PILOT | **producer: TRACE-C** | (자체 구축) | cross-over (3 mod × 3 mode) | Tier-1 MLSys | measurement (IISWC/DATE) |
| **R3** REGRET-VKV | consumer | R1 cold tier (Paper Pair) | T 분포 + MWU regret-gap | R1+R3 MLSys pair | warm-start prior 결합 |
| **R4** FOCUS-COVERAGE | **producer: BENCH-D** | (자체 구축) | GT vs human drift 상관 | NeurIPS-D&B/ACL benchmark | human-curated 50 video |
| **R5** STREAM-RECOVER | consumer | R1 INFRA-A/B + R2 TRACE-C | bs=1 frame↔KV + FPS overhead | video MLSys | scope 축소 (긴 video only) |
| **R6** VKV-SKIP | consumer | R4 BENCH-D + R2 TRACE-C | KL/Wasserstein 분리 + Quest gain | DAC/CAL (T1 조건부) | engineering 격하 / DROP |

**Producer-first 순서 (12-16주)**: R4 BENCH-D (W2-4) / R2 TRACE-C (W3-5) → R1 INFRA-A/B (W5-8) → R3 Paper Pair (W8-10) → R5 video (W9-12) → R6 조건부 (W11-14) → R7 component (W14-16). 핵심: **R1=INFRA producer, R4=BENCH producer, R2=TRACE producer; 나머지는 consumer 라 producer 가 먼저.**

---

## 5. Tier-1 Top 3 (R15-β contribution bullet)

### R1 RECOVER-TIER — Dual-Tier Recoverable Vision-KV with Reconstruction-Bounded Cold Spill (Accept, mean 7.75, scoop 4/10)

- **Mechanism / benefit**: vision KV 를 hot(low-rank rank-r + b-bit quant, reconstruction `‖V−V̂‖_F≤ε` 보장, 항상 attend) + cold(Q2/Q4 recoverable, **같은 LPDDR5 logical partition — 추가 메모리 없음**) 로 이원화. focus-shift turn 이 cold region 요구 시 framework-level block dequant 로 promote → destructive prune 의 multi-focus 손실을 reconstruction bound 내에서 복원.
- **Closest competitor 차별화**: SparseVILA([arXiv:2510.17777](https://arxiv.org/abs/2510.17777)) = full visual cache 보존 (GPU/AWQ, edge/energy/memory 무, no formal guarantee). R1 = **16GB UMA 에서 full 보존 불가** → recoverable-quant dual-tier 로 capacity 충족 + **reconstruction error bound** (경쟁자 전무, Eckart-Young + quant). KVSwap = text-only UMA swap → vision-token-aware + LPDDR5 logical spill (NVMe round-trip µs 대비 ns, CF-3).
- **예상 gain (정량)**:

  | Axis | 개선 | 조건 |
  |---|---|---|
  | Memory ★ | GPU-resident KV −60~76% (α∈[0.25,0.4], **단일 76% 금지**) | Jetson 16GB, quant 본질 |
  | Energy | hot-only turn per-turn −2~4× (vs cold retrieve) | hot hit-rate 높을 때 |
  | Accuracy | multi-focus full-cache 의 ≥92% 회복; destructive prune 대비 +8~15%p | N=4-6 focus/video |
  | Latency | cold promote TTFT full re-prefill 대비 −30~50% | promote < recompute 시 |

- **Tier 강등 risk**: uniform-Q4(no-tier) 대비 dual-tier·low-rank 의 marginal footprint·accuracy 이득 <5% 이면 tiering 이 quant 와 분리된 기여 없음 → quant-only 로 축소 (preliminary-study gate).
- **Outperform/envelope**: SparseVILA full-cache(OOM on 16GB) ↔ prefill-prune-discard(SparseVLM, multi-focus lossy) ↔ uniform-Q4(no-tier ablation) 의 envelope 안 — capacity-fit + recoverable + bound 의 유일 점.

### R2 DVFS-PILOT — Energy/DVFS 3-way Prune-vs-Retrieve-vs-Recompute Policy with Ski-Rental Bound (Accept, mean 7.75, scoop 3/10 최저)

- **Mechanism / benefit**: turn 별 vision 정보 recover 의 3-way modality — (a) prune 후 vision-encoder **재encode**(compute-bound), (b) cache **retrieve**(memory-bound), (c) 부분 **recompute**(attention) — 를 nvpmodel(10/15/25W) × focus-shift × KV-residency 입력 rail-level energy cost model 로 online 선택. 미래 focus 재방문 미지 → ski-rental (retain 누적 = recompute 1회 시 전환).
- **Closest competitor 차별화**: V-Rex([arXiv:2512.12284](https://arxiv.org/abs/2512.12284), HPCA'26) = edge KV retrieval 3.1-18.5× energy but **dedicated HW**. R2 = commodity Jetson **SW-only**. Modality Inflation = stage-DVFS 확인하나 vision-KV recover 정책 미결합 + A100. Sustainability-aware([arXiv:2512.04088](https://arxiv.org/abs/2512.04088)) = text-LLM routing → vision-KV recover modality energy + **ski-rental 2-competitive 보장** (경쟁자 전무).
- **예상 gain (정량)**:

  | Axis | 개선 | 조건 |
  |---|---|---|
  | Energy ★★ | session energy −15~30% (재방문 skew) + 2-competitive worst-case | cross-over regime 존재 시 |
  | Latency | EDP −12~20%, e2e −10~20% (freq↓ 회피) | thermal-bound 아닌 구간 |
  | Cost | SW-only, HW 추가 없이 V-Rex 류 효과 일부 | commodity Jetson |

- **Tier 강등 risk**: 3-way modality energy 순위가 모든 nvpmodel mode 에서 동일 (cross-over 없음) → DVFS-aware 가치 소멸 → **measurement-only 로 reposition** ("edge VLM 3-way recover energy 첫 정량 characterization", IISWC/DATE; drop 아님).
- **Outperform/envelope**: always-prune(SparseVLM 재encode 잦음) / always-retrieve(SparseVILA full cache) 고정 정책의 envelope — ski-rental 이 worst-case 2× 안에서 cross-over 점유.

### R3 REGRET-VKV — Regret-Bounded Cross-Turn Importance Re-estimation on Shared-Prefix Vision-KV (Conditional Accept, mean 7.25, scoop 4/10)

- **Mechanism / benefit**: vision KV 를 SGLang RadixAttention shared-prefix(같은 image, 다른 query)로 1회 prefill·재사용 + 매 turn 관측 attention 으로 보존 importance 를 **multiplicative-weights(MWU)** 갱신 (token=expert, attention=gain). prefill-salience staleness 를 inter-turn online learning 으로 해소. retrieve-on-miss 는 R1 cold tier 의존 (Paper Pair).
- **Closest competitor 차별화**: SparseVILA/CSP/StreamMem 전부 **static one-shot 추정** (turn 간 학습 없음). R3 = (a) turn-level online learning 정식화, (b) **regret ≤ √((T/2)ln n)** (Arora-Hazan-Kale'12 — 누구도 보장 없음), (c) 관측 attention 누적 갱신. RVIS([arXiv:2604.12358](https://arxiv.org/abs/2604.12358)) = intra-response shift → **inter-turn 일반화** (concurrent 미발견). shared-prefix 는 RadixAttention 현존 (신규 아님, 정직하게 철회) → 기여는 importance-guided evict + cross-turn MWU 로 한정.
- **예상 gain (정량)**:

  | Axis | 개선 | 조건 |
  |---|---|---|
  | Accuracy | multi-turn 누적 accuracy +6~12%p (vs static) | T≥5, budget-제약 |
  | Energy | prefix-hit prefill energy −40~50% | turn 당 재계산 회피 |
  | Memory | weight-only (token 당 scalar) = 무시가능 + prefix 중복 제거 | 세션 최경량 mechanism |
  | Latency | turn 당 TTFT −40~60% (prefix hit) | prefix 일치 시 |

- **Tier 강등 risk**: 실사용 multi-turn 길이 분포가 짧으면 (대부분 1-3 turn) MWU prior 미수렴 → static 과 동등 (Diff-Scope 6) → warm-start prior(R4 distribution) 결합으로 완화.
- **Outperform/envelope**: Fast-dDrive(출력 fork, importance 무갱신) / VLCache(same-input) / RadixAttention(static prefix) 대비 — different-focus + fresh importance + regret bound 의 사후-최적 고정 set 수렴.

---

## 6. Tier-2 독립 Top 3 (R15-β contribution bullet)

### R4 FOCUS-COVERAGE — Multi-Focus Recall Benchmark + Distribution-Marginal Retention + Coverage-Energy Pareto (Conditional Accept, mean 7.40, scoop 5/10)

- **Mechanism / benefit**: (a) focus-trajectory multi-turn 벤치 (image/video + GT region/segment per turn, drift param 제어) + **Focus-Coverage Recall (FCR) + Coverage-Energy Pareto (FCR vs J/turn)** metric, (b) single-query 대신 **예상 focus 분포 marginal** `I(t)=Σ_f P(f)·attn(t,f)` retention (P_focus = saliency prior + Bayesian update). scorer hook = LLM cross-attention (query 존재 시점, `_execute_mm_encoder` query-independent encoder 아님, CF-2).
- **Closest competitor 차별화**: MMDU/ConvBench/MileBench/MultiVerse([arXiv:2510.16641](https://arxiv.org/abs/2510.16641))는 일반 conversational capability — **compression-under-multi-focus FCR/Coverage-Energy Pareto 미정의** ("벤치 부재" framing 철회, 정면 인용). MMTok([arXiv:2508.18264](https://arxiv.org/abs/2508.18264))/Adaptive Greedy([arXiv:2603.20180](https://arxiv.org/abs/2603.20180)) = single-prompt coverage+(1−1/e) → R4 는 **multi-turn future-focus distribution marginal** 로만 흡수. CSP = focus-level 미분해 → FCR 분해. StreamMem = generic-query 단일 vector → 명시적 P_focus.
- **예상 gain (정량)**:

  | Axis | 개선 | 조건 |
  |---|---|---|
  | Accuracy/recall ★ | multi-focus FCR +6~12%p (vs single-query) | diverse-focus, N≥3 |
  | Energy | re-prefill 빈도 −15~25% (간접) + Coverage-Energy Pareto 신metric | focus-shift 빈번 |
  | Cost ★ | benchmark = device-agnostic community 자산 (long-tail citation) | 후속 연구 enabler |
  | Latency | 불필요 re-prefill 회피 평균 TTFT −10~20% | multi-turn only |

- **Tier 강등 risk**: GT focus trajectory (GPT-4/LLM 생성) 가 실제 human multi-turn focus drift 와 상관 낮으면 (synthetic-only) 외적 타당성 붕괴 → human-curated 50 video 축소 (preliminary-study gate).
- **Outperform/envelope**: single-query retention(SparseVLM)의 multi-focus 손실을 처음 정량화 + distribution-marginal 방어 + Coverage-Energy Pareto (MultiVerse/Right-Benchmark 미수행) — 측정도구 + 방어 두 기둥. R1/R3/R5/R6 measurement 토대 + R7 focus-autocorr data source (최광 breadth, 유일 FastV-scale citation 후보).

### R5 STREAM-RECOVER — Single-Pass Streaming Submodular (1/2−ε) Retention with Recoverable Eviction for Edge Video (Conditional Accept, mean 7.05, scoop 5-6/10)

- **Mechanism / benefit**: 긴 video frame token 을 **single-pass Sieve-Streaming** (threshold τ=v/(2k), marginal gain ≥τ 만 채택, (1/2−ε), O((k log k)/ε) bounded-memory, Badanidiyuru KDD'14) 로 보존 + evict = "버림" 이 아니라 stub(Q2 KV) + thumbnail(1/4 해상도) **recoverable 강등**. 과거 focus → stub retrieve(coarse) 또는 thumbnail 재encode(fine-grained, energy-bounded). 무한 stream = NVMe(carrier PCIe Gen4, CF-3) recency-tier. R1 vision-range 인프라 재사용.
- **Closest competitor 차별화**: StreamingVLM([arXiv:2510.09608](https://arxiv.org/abs/2510.09608), ICLR'26) = attention sink + short vision window, 과거 frame **영구 evict**. ReKV([arXiv:2503.00540](https://arxiv.org/abs/2503.00540), ICLR'25, 77 cite) = full KV RAM/disk retrieve ("repeat for each new question" 비판) → edge 16GB full 불가 → **bounded + recoverable 강등**. StreamKV([arXiv:2511.07278](https://arxiv.org/abs/2511.07278))/MemStream([arXiv:2602.18434](https://arxiv.org/abs/2602.18434)) = heuristic segment retrieval → **(1/2−ε) bounded-memory 보장**. Adaptive Greedy([arXiv:2603.20180](https://arxiv.org/abs/2603.20180)) = non-streaming full-video greedy → single-pass.
- **예상 gain (정량)**:

  | Axis | 개선 | 조건 |
  |---|---|---|
  | Memory ★★ | peak vision-KV −60~85% (bounded, single-pass) → 긴 video OOM 해소 | edge 16GB enabling |
  | Accuracy | 과거-focus recall +10~25%p (vs destructive evict) | 과거 segment 재질문 |
  | Energy | thumbnail 재encode vs full reencode −40~60% + bounded mem→idle energy↓ | fine-grained focus |

- **Tier 강등 risk**: streaming 중 미래 focus 집합 F·weight w_f 미지 (StreamMem 딜레마) → saliency-prior 고정 + prior-drift 하 (1/2−ε) 보장 유지 미검증 시 보장 붕괴 → prior-drift regret 분석 (gate). short image multi-turn 엔 streaming 불필요 → R1 우월 (긴 video only scope).
- **Outperform/envelope**: heuristic streaming evict (StreamingVLM/ReKV/StreamKV/MemStream) 의 envelope — (1/2−ε) bounded-memory 보장 + recoverable 의 유일 점. R1(image) 과 video 로 상보.

### R6 VKV-SKIP — Non-Destructive Vision-Aware Cache-Resident Block Skipping on Jetson Ampere (Conditional Accept borderline, mean 6.20, scoop 5/10)

- **Mechanism / benefit**: decode attention kernel 에서 vision KV block 에 **vision-aware softmax threshold (λ_vision < λ_text)** 로 V-load/PV-matmul skip. **cache 보존** (query 변경 시 skip pattern 재평가) → 다음 focus turn 에서 동일 cache 재skip (multi-focus 안전). vision 98-99% 비중 → 대부분 block post-softmax≈0 → memory-bound decode 의 V-load(LPDDR5) skip = bandwidth(=energy) 절감.
- **Closest competitor 차별화**: BLASST([arXiv:2512.12087](https://arxiv.org/abs/2512.12087), corpus) = text-LLM, vision 미구분, Blackwell/Hopper kernel → **vision-aware λ + Ampere SM87 kernel + multi-focus 무손실 평가**. Quest(ICML'24, query-aware page selection without eviction)/MInference(NeurIPS'24) 인접 → **vision-modality redundancy 분포 차이 명시 활용 (KL/Wasserstein) + non-destructive softmax-threshold (selection 아닌 skip)**. VisionZip = destructive merge → non-destructive skip. Focus = dedicated HW → commodity Jetson.
- **예상 gain (정량)**:

  | Axis | 개선 | 조건 |
  |---|---|---|
  | Energy | decode V-load skip → LPDDR5 read −20~30% (energy 비례) | memory-bound decode |
  | Latency | Jetson decode TPOT −15~25% (Ampere 선측정, B200 transfer 금지) | long-vision context |
  | Memory | cache 보존 (절감 아님) — multi-focus 안전이 본질 | — |

- **Tier 강등 risk**: vision block post-softmax redundancy 분포가 text 와 통계적 차이 없으면 (KL/Wasserstein) vision-aware λ 무의미 → "BLASST 재현 + vision tag" 격하. Quest 대비 marginal gain 미입증 → engineering 격하/DROP. **AI-Kernel 4.5 (전체 최저)** — from-scratch SM87 Ampere kernel (`fattn-tile.cu` block-skip 부재) 이 최대 risk.
- **Outperform/envelope**: BLASST-uniform(vision 미구분) / VisionZip(destructive) / Quest(modality-agnostic selection) 의 envelope — vision-aware non-destructive skip + multi-focus 무손실. accuracy-first(PyTorch/Triton ref) 입증 후 speedup 분리.

---

## 7. 미선정 (R10-α.3 + R67 ★/▼)

상세는 [unselected.md](unselected.md). 1줄 요약 (R67 ★최고 / ▼최저 sub-axis):

- **R7 COMPUTE-PREFETCH** (Major Revision, mean 5.90, R1 component): speculative compute-idle dequant latency-hiding. ★ AI-impl-D6 7 (CF-1 dead→working path) / ▼ **Novelty-Mech 5 (전체 최저, prefetch=교과서)**. 미선정 = standalone novelty 부족 + R1 종속 + focus autocorr evidence 0건. 재방문 = R4 autocorr>uniform 입증 + Orin compute-idle pilot.
- **A SubCover** (drop→R4 흡수): coverage+(1−1/e) multi-focus 보존. ★ 수학적 보장 / ▼ **Novelty 6 (MMTok/Adaptive Greedy scoop ~70%)**. distribution-marginal 로만 흡수.
- **G FocusBench** (drop→R4 흡수): standalone FCR benchmark. ★ Impact-Breadth / ▼ **D2=4 (standalone triviality)**. retention+Pareto 묶음으로 R4 흡수.
- **VISION-BLASST** (drop→R6 흡수): F6 부분집합 (modality-aware λ 만).
- **PREFETCH standalone** (drop→R7): Orin prefetch API 미지원 + R1 종속.

---

## 8. 참고 (Step 0 paper)

본 세션 Step 0 에서 R13.1 검증된 핵심 paper (전체는 [papers.md](../../papers.md)):

- **Vision token pruning (single-turn-safe)**: [FastV (arXiv:2403.06764)](https://arxiv.org/abs/2403.06764) ECCV'24, [SparseVLM (arXiv:2410.04417)](https://arxiv.org/abs/2410.04417) ICML'25, [VisionZip (arXiv:2412.04467)](https://arxiv.org/abs/2412.04467) CVPR'25, [PyramidDrop (arXiv:2410.17247)](https://arxiv.org/abs/2410.17247) CVPR'25, [When Token Pruning is Worse than Random (arXiv:2512.07580)](https://arxiv.org/abs/2512.07580) CVPR'26.
- **Multi-turn / multi-focus 취약성 (insight 검증)**: [SparseVILA (arXiv:2510.17777)](https://arxiv.org/abs/2510.17777) ICCV'25, [Cross-Self KV/CSP (arXiv:2412.04652)](https://arxiv.org/abs/2412.04652), [RVIS/Why Pruning Fails (arXiv:2604.12358)](https://arxiv.org/abs/2604.12358), [StreamMem (arXiv:2508.15717)](https://arxiv.org/abs/2508.15717), [VLCache (arXiv:2512.12977)](https://arxiv.org/abs/2512.12977), [CROP (arXiv:2505.21233)](https://arxiv.org/abs/2505.21233) EMNLP'25.
- **Video long-context KV**: [StreamingVLM (arXiv:2510.09608)](https://arxiv.org/abs/2510.09608) ICLR'26, [LiveVLM (arXiv:2505.15269)](https://arxiv.org/abs/2505.15269) DAC'26, [ReKV (arXiv:2503.00540)](https://arxiv.org/abs/2503.00540) ICLR'25, [StreamKV (arXiv:2511.07278)](https://arxiv.org/abs/2511.07278), [V-Rex (arXiv:2512.12284)](https://arxiv.org/abs/2512.12284) HPCA'26, [MemStream (arXiv:2602.18434)](https://arxiv.org/abs/2602.18434), [FastVID (arXiv:2503.11187)](https://arxiv.org/abs/2503.11187) NeurIPS'25.
- **Edge Jetson + KV + energy**: [KVSwap (arXiv:2511.11907)](https://arxiv.org/abs/2511.11907), [Sustainability-Aware LLM (arXiv:2512.04088)](https://arxiv.org/abs/2512.04088), [CHIME (arXiv:2601.19908)](https://arxiv.org/abs/2601.19908), [PAISE'25 (arXiv:2506.09554)](https://arxiv.org/abs/2506.09554), [Modality Inflation (arXiv:2512.22695)](https://arxiv.org/abs/2512.22695), [EdgeReasoning (arXiv:2511.01866)](https://arxiv.org/abs/2511.01866).
- **Corpus (Mode 2 PDF)**: [Focus (arXiv:2512.14661)](https://arxiv.org/abs/2512.14661) HPCA'26, [BLASST (arXiv:2512.12087)](https://arxiv.org/abs/2512.12087) MLSys'26, [Fast-dDrive (arXiv:2605.23163)](https://arxiv.org/abs/2605.23163).
- **이론 anchor + scoop**: Nemhauser-Wolsey-Fisher 1978 ((1−1/e)), Badanidiyuru KDD'14 (Sieve-Streaming (1/2−ε)), Arora-Hazan-Kale'12 (MWU regret), ski-rental 2-competitive, [MMTok (arXiv:2508.18264)](https://arxiv.org/abs/2508.18264), [Adaptive Greedy Frame (arXiv:2603.20180)](https://arxiv.org/abs/2603.20180), [MultiVerse (arXiv:2510.16641)](https://arxiv.org/abs/2510.16641), Quest (ICML'24), MInference (NeurIPS'24), InfiniGen (OSDI'24).

> ⚠️ **Publish gate**: 미래(2026) arxiv ID (RVIS 2604.12358, Fast-dDrive 2605.23163, Adaptive Greedy 2603.20180, MemStream 2602.18434, CHIME 2601.19908) 는 Phase 3 publish 전 re-WebFetch 의무. VLCache 720GB KV 수치 = bs=256 datacenter search-reported [⚠️ unverified] → edge bs=1 재산정.

---

## 9. 약어 / 용어 Glossary (CTRL+F / Cmd+F)

- **Jetson Orin NX 16GB**: edge SoC. 1024-core Ampere (SM87, 32 Tensor core), 128-bit LPDDR5 102.4 GB/s, 16GB UMA (CPU+iGPU 물리 공유), nvpmodel 10/15/25W+MAXN (40W=JetPack 6.2 Super Mode). `concurrentManagedAccess=0`.
- **UMA (Unified Memory Architecture)**: CPU+GPU 가 같은 물리 DRAM 공유. Orin NX 는 zero-copy SoC DRAM → "promote"=physical migration 아닌 dequant compute. (discrete GPU 의 PCIe spill 과 근본 다름.)
- **multi-turn / multi-focus**: 같은 image/video 에 여러 turn 에 걸쳐 다른 region/object/시점(focus)을 질문. query focus 가 캐시된 vision KV 위에서 이동.
- **vision token pruning/compression**: VLM 의 visual token(98-99% 비중)을 줄여 prefill/KV 절감. query-conditional(FastV/SparseVLM/Focus-SEC, single-turn-safe·multi-turn-vulnerable) vs query-agnostic(VisionZip-merge/StreamMem, 재사용 가능·정보 손실 불가역).
- **RVIS (Relevant Visual Information Shift)**: decoding step 마다 중요한 visual token 이 이동 ([arXiv:2604.12358](https://arxiv.org/abs/2604.12358)). 본 세션은 이를 **inter-turn** (turn 간 focus 이동)으로 일반화.
- **dual-tier recoverable KV**: hot(compress, 항상 attend) + cold(recoverable, 같은 LPDDR5 logical partition, dequant 전까지 제외). R1 핵심.
- **multi-focus recall / FCR (Focus-Coverage Recall)**: 보존된 token 이 focus trajectory 의 각 focus 를 cover 하는 가중 비율. `FCR(π,T)=(1/n)Σ_i 𝟙[answer-token(f_i)∈reachable(S_π)]`. R4 신규 metric.
- **Coverage-Energy Pareto**: (FCR, energy/turn) frontier. R4 신규 metric (Jetson tegrastats energy 포함).
- **ski-rental prune-vs-retrieve**: 미래 focus 재방문 미지 하 online decision — 보존 누적 비용 = 재계산 1회 비용 도달 시 evict. deterministic 2-competitive (randomized e/(e−1)≈1.58). R2 핵심.
- **non-destructive block skip**: cache 보존, query 별 V-load/PV-matmul 만 동적 skip (BLASST online-softmax threshold). prune(destructive)과 달리 multi-focus 안전. R6.
- **shared-prefix vision-KV**: 같은 image 의 vision KV 를 1회 prefill 후 turn 간 재사용 (SGLang RadixAttention 현존). R3 vehicle.
- **MWU (Multiplicative Weights Update)**: token=expert, attention=gain 으로 turn-level online importance 갱신. regret ≤ √((T/2)ln n) (Arora-Hazan-Kale'12). R3.
- **Sieve-Streaming**: single-pass streaming submodular maximization, (1/2−ε) 근사, O((k log k)/ε) bounded-memory (Badanidiyuru KDD'14). R5.
- **reconstruction error bound**: `‖V−V̂‖_F ≤ ε` (Eckart-Young low-rank + quant). hot tier 의 lossy compress 가 복원 가능함을 보장. R1.
- **decode memory-bandwidth-bound**: Jetson decode 의 dominant bottleneck = LPDDR5 bandwidth (memory freq↓ 시 latency +370%/energy +72%, PAISE'25). V-load skip / quant 이 energy 직결.
- **CF-1/2/3/4 (critical fix)**: CF-1 = Orin `cudaMemPrefetchAsync` 미지원 → framework-level KV buffer + `cudaStreamAttachMemAsync`. CF-2 = `EncoderCacheManager`(embedding) ≠ LLM KV(`BlockPool`); vision token→KV block 매핑 신규 인프라. CF-3 = Orin NX PCIe Gen4 NVMe 가능, capacity 본질은 quantization. CF-4 = 1024 core(1792 아님), 40W=JetPack6.2, 4090=Ada sm_89.
- **nvpmodel / tegrastats / INA3221**: Jetson power-mode CLI / telemetry / on-board rail-level power sensor (VDD_GPU_SOC, VDD_CPU_CV). R2 energy 측정.
- **producer / consumer (cross-share)**: R1=INFRA producer (vision→KV 매핑 + KV buffer), R4=BENCH producer (FCR/autocorr), R2=TRACE producer (energy probe). 나머지 consumer → producer-first 구현.
- **Paper Pair**: {R1 + R3} — R1(dual-tier capacity, system) backbone + R3(regret-bounded re-estimation, algorithm) evict 정책 layer = MLSys/ASPLOS 1편.
