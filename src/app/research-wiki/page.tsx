import Link from "next/link";
import { Metadata } from "next";

export const metadata: Metadata = {
  title: "Research Wiki - AICA Lab",
  robots: { index: false, follow: false },
};

type SessionCard = {
  date: string;
  title: string;
  href: string;
  keywords: string[];
  summary: string;
  detailed?: boolean;
  tierTable?: { tier: string; idea: string; venue: string; score: string; href: string }[];
  extraLinks?: { label: string; href: string }[];
};

const recentSessions: SessionCard[] = [
  {
    date: "2026-06-04",
    title: "🥇 Edge Multi-Agent Hybrid-SSM/MoE Serving — NVIDIA Nemotron 3 Nano/Super 분석 기반 (Mode 2)",
    href: "/research-wiki/2026-06/nemotron3-edge-agentic-serving",
    keywords: [
      "NVIDIA Nemotron 3 Nano 30B-A3B (Mamba-2 23층 / MoE 23층 / Attention 6층, KV-head 2)",
      "Nemotron 3 Super 120B-A12B (LatentMoE d/ℓ=4 + MTP native spec decoding + NVFP4 25T)",
      "agent-count scaling — expert working-set union cliff N* closed-form 예측",
      "SSM state 48MB/agent 상수 — state-vs-KV 비대칭의 1급 swap 객체화",
      "per-token expert read 1.84GB (batch-1 FP8) → AGX Orin ~9 tok/s 상한",
      "SSM state quant verbosity +40% (Super Eq3) — 오차 예산 = throughput 예산",
      "thermal 정상상태 sustainable agent-count 곡선 (cold 대비 −40~60% 가설)",
      "LPDDR row-buffer locality of expert read (trace-driven Ramulator2/DRAMSim3)",
      "agent workload: decode-dominated 91-98.6% / prefix reuse 84.6-99.5% / turn 12-62",
      "AGX Orin 64GB FP8 full-resident / Orin NX 16GB는 W4(22GB)도 불가",
      "llama.cpp hybrid 로딩 깨짐(#20570) → vLLM 단일 path (clone 검증)",
      "MLSys/HPCA/ASPLOS/NeurIPS 2025-26 agentic serving 34편 + characterization 13편",
      "SPENDTHRIFT × BLOAT paper pair (이론 ↔ 인과 측정)",
    ],
    summary:
      "사용자 input — Nemotron 3 Nano/Super tech report 분석 + 2026 agentic AI 최적화 논문 탐색 + \"Super 대비 Nano 특이점 × agent 수 증가 × edge 리소스 특성\" ideation. **핵심 비대칭**: Nano 는 52층 중 attention 6층(KV-head 2)만 KV cache 를 갖는 hybrid Mamba-MoE → per-agent context 메모리가 사실상 상수(SSM state 48MB) 가 되고 병목이 **expert read (1.84GB/token, LPDDR BW 직격)** 로 이동 — Super 는 LatentMoE(d/ℓ=4 압축)로 응답했으나 edge 후보인 Nano 에는 압축 부재. Super 보고서가 자인한 **SSM state quant verbosity +40%** (recurrent bias accumulation) 와 long-horizon rollout batching 충돌이 hidden insight. Step 0 에서 시스템 학회 16편 + edge multi-agent 18편 + characterization 13편 검증 수집 (decode-dominated 91-98.6% / prefix reuse 84.6-99.5% / DRAM util 20-60% memory-bound / expert migration 90%+). 3 expert × 15 candidate → 5-reviewer (vLLM clone 검증 포함) + cross-review → 12 refined (NIMBLE×RELAY→**MOORING** 병합, TEMPER→GOVERNOR / FORGE→KILN rename, PILOT drop [Mamba Drafters [arXiv:2506.01206](https://arxiv.org/abs/2506.01206) 45% scoop], QUARRY trace-driven 강등 구제 +1.4) → Phase 2' 신규 scoop 0 → **Top-M 6**. **Tier-1 Top 3** (TIDEMARK 7.6 / MOORING 7.8 / SPENDTHRIFT 7.4) + **Tier-2 독립 Top 3** (KILN 6.8 / QUARRY 7.1 / BLOAT 6.6) + {SPENDTHRIFT×BLOAT} paper pair. 치명 발견 — llama.cpp hybrid GGML_ASSERT(#20570) / W4 실측 22GB > Orin NX 16GB / ncu Orin iGPU 미지원 / RTX(1.79TB/s)는 edge BW-bound proxy 불가 / cross-review \"expert 46-60GB\" 주장 재검산 30GB 정정. 구현 우선순위: KILN(인프라) → BLOAT(인과 gate) → MOORING MVP → TIDEMARK → SPENDTHRIFT → QUARRY.",
    detailed: true,
    tierTable: [
      { tier: "🥇 Tier-1", idea: "TIDEMARK (Critical Agent-Count Prediction via Expert Working-Set Union Growth — 2-state Markov N* closed-form + diversity-aware admission)",
        venue: "MLSys 2027", score: "7.6 (scoop <30% 최저)",
        href: "/research-wiki/2026-06/nemotron3-edge-agentic-serving/tier1/01-tidemark" },
      { tier: "🥈 Tier-1", idea: "MOORING (LPDDR-Warm Tiered SSM-State Pool with Tool-Window Swap — state-vs-KV 비대칭 1급 객체화, NIMBLE×RELAY 병합)",
        venue: "MLSys 2027 / EuroSys 2027", score: "7.8 (nov 8, white space)",
        href: "/research-wiki/2026-06/nemotron3-edge-agentic-serving/tier1/02-mooring" },
      { tier: "🥉 Tier-1", idea: "SPENDTHRIFT (Spectral Verbosity Budget for State-Quantized Hybrid Decoding — bias-drift→verbosity closed-form + spectral allocator)",
        venue: "ICML / NeurIPS / MLSys 2027", score: "7.4 (이론 최강)",
        href: "/research-wiki/2026-06/nemotron3-edge-agentic-serving/tier1/03-spendthrift" },
      { tier: "T1", idea: "KILN (Thermal-Bounded Sustainable Agent-Count Characterization — 정상상태 천장 측정, Track-A baseline 인프라 hub)",
        venue: "DATE / ISLPED / IISWC 2027", score: "6.8 (arch 9 최안전)",
        href: "/research-wiki/2026-06/nemotron3-edge-agentic-serving/tier2/01-kiln" },
      { tier: "T2", idea: "QUARRY (Row-Buffer Locality of Expert Read on LPDDR — fused_moe trace → Ramulator2/DRAMSim3)",
        venue: "DAC / DATE / IEEE CAL 2027", score: "7.1 (강등 구제 +1.4)",
        href: "/research-wiki/2026-06/nemotron3-edge-agentic-serving/tier2/02-quarry" },
      { tier: "T3", idea: "BLOAT (Verbosity Cost of SSM-State Quantization — det vs stochastic rounding 인과 식별, SPENDTHRIFT pair)",
        venue: "IISWC / ISPASS / IEEE CAL 2027", score: "6.6 (pair)",
        href: "/research-wiki/2026-06/nemotron3-edge-agentic-serving/tier2/03-bloat" },
    ],
    extraLinks: [
      { label: "📊 Landing (RQ + Essential Reading 5편 + Nano vs Super 비교표 + Implementation-Priority Decision Tree)",
        href: "/research-wiki/2026-06/nemotron3-edge-agentic-serving" },
      { label: "📜 미선정 / Drop 로그 (HEARTH 7.3 FineMoE 인접 / TOLLGATE stack 후보 / PILOT Mamba-Drafters scoop 등 8건)",
        href: "/research-wiki/2026-06/nemotron3-edge-agentic-serving/unselected" },
    ],
  },
  {
    date: "2026-06-02",
    title: "🥈 VLM Edge Multi-Turn / Multi-Focus Serving — Vision-KV Pruning Cache 재사용 취약성 (Mode 2, R69/R70/R71 적용)",
    href: "/research-wiki/2026-06/vlm-edge-multiturn",
    keywords: [
      "Jetson Orin NX 16GB edge (primary)",
      "DGX Spark 128GB unified / RTX 5090·4090 (secondary)",
      "multi-turn + multi-focus (query focus shift over cached vision KV)",
      "vision token pruning single-turn-safe but multi-turn-vulnerable",
      "dual-tier recoverable vision-KV (hot quant + cold LPDDR5 logical spill)",
      "energy/DVFS prune-vs-retrieve-vs-recompute (ski-rental 2-competitive)",
      "cross-turn MWU regret-bounded importance re-estimation",
      "multi-focus recall (FCR) benchmark + distribution-marginal retention",
      "streaming Sieve-Streaming (1/2−ε) bounded-memory",
      "non-destructive softmax-threshold vision block skip",
      "corpus: Focus HPCA'26 / BLASST MLSys'26 / Fast-dDrive NVIDIA",
      "R31 verdict: 정당한 underexplored gap (wrong-insight 아님)",
      "tegrastats/INA3221 energy 측정, cudaMemPrefetchAsync 미지원(SM87) 정정",
      "SparseVILA / VLCache / CSP / RVIS / MMTok / Quest 차별화",
    ],
    summary:
      "사용자 input — Jetson Orin NX (16GB edge) primary [secondary DGX Spark 128GB unified / RTX 5090·4090] 에서 VLM serving 최적화 + insight 검증 (vision token pruning 이 single-turn 엔 안전하나 multi-turn/multi-focus 에선 turn-1 기준 prune 된 vision KV 가 turn-2 의 다른 focus 정보를 이미 버려 취약). **R31 판정 = 정당한 underexplored gap (wrong-insight 아님)** — 메커니즘이 SparseVILA([arXiv:2510.17777](https://arxiv.org/abs/2510.17777) ICCV'25)·CSP·RVIS·VisionZip 로 직접 확증되나, 순수 문제 발견은 scoop → **생존 gap = Jetson 16GB UMA memory-bound + energy/DVFS + multi-focus recall 정량 의 교집합**. Mode 2 (local PDF corpus: Focus [HPCA'26](https://arxiv.org/abs/2512.14661) / BLASST [MLSys'26](https://arxiv.org/abs/2512.12087) / Fast-dDrive [NVIDIA](https://arxiv.org/abs/2605.23163)) + 외부 31편 + R69 superpowers brainstorming + R70 AI-Research-SKILLs grounding. 3 expert × 5-7 = **19 candidate → 7 refined → Top-M 6**. **Tier-1 Top 3** (RECOVER-TIER 7.75 / DVFS-PILOT 7.75 / REGRET-VKV 7.25) + **Tier-2 독립 Top 3** (FOCUS-COVERAGE 7.40 / STREAM-RECOVER 7.05 / VKV-SKIP 6.20). Phase 2 reviewer 가 critical 구현 결함 정정 — **cudaMemPrefetchAsync 미지원 (Orin Ampere SM87 concurrentManagedAccess=0)** → zero-copy mapped + framework KV requant 재설계, vLLM KV cache modality-agnostic + EncoderCacheManager=embedding 캐시 혼동 해소, NVMe 재정의 (M.2 가능), HW spec 정정 (CUDA 1024 core / 40W=JetPack6.2 / RTX4090=Ada sm_89). GitHub link main/master 100% line-anchored verified.",
    detailed: true,
    tierTable: [
      { tier: "🥇 Tier-1", idea: "RECOVER-TIER (Dual-Tier Recoverable Vision-KV with Reconstruction-Bounded Cold Spill)",
        venue: "MLSys 2027 / ASPLOS 2027", score: "7.75 (Accept)",
        href: "/research-wiki/2026-06/vlm-edge-multiturn/tier1/01-recover-tier" },
      { tier: "🥈 Tier-1", idea: "DVFS-PILOT (Energy/DVFS 3-way Prune-vs-Retrieve-vs-Recompute, Ski-Rental Bound)",
        venue: "MLSys 2027 / IISWC / DATE", score: "7.75 (Accept)",
        href: "/research-wiki/2026-06/vlm-edge-multiturn/tier1/02-dvfs-pilot" },
      { tier: "🥉 Tier-1", idea: "REGRET-VKV (Cross-Turn MWU Regret-Bounded Importance Re-estimation, R1 paper-pair)",
        venue: "MLSys 2027 (R1+R3 pair)", score: "7.25 (Cond.)",
        href: "/research-wiki/2026-06/vlm-edge-multiturn/tier1/03-regret-vkv" },
      { tier: "T1", idea: "FOCUS-COVERAGE (Multi-Focus Recall Benchmark + Distribution-Marginal Retention + Coverage-Energy Pareto)",
        venue: "NeurIPS D&B 2027 / ACL", score: "7.40 (Cond.)",
        href: "/research-wiki/2026-06/vlm-edge-multiturn/tier2/04-focus-coverage" },
      { tier: "T2", idea: "STREAM-RECOVER (Single-Pass Sieve-Streaming (1/2−ε) Bounded-Memory + Recoverable Eviction)",
        venue: "MLSys 2027 (video)", score: "7.05 (Cond.)",
        href: "/research-wiki/2026-06/vlm-edge-multiturn/tier2/05-stream-recover" },
      { tier: "T3", idea: "VKV-SKIP (Non-Destructive Vision-Aware Cache-Resident Block Skipping on Ampere)",
        venue: "DAC 2027 / IEEE CAL", score: "6.20 (Cond. borderline)",
        href: "/research-wiki/2026-06/vlm-edge-multiturn/tier2/06-vkv-skip" },
    ],
    extraLinks: [
      { label: "📊 Landing (RQ + Essential Reading + Decision Tree + producer-first 12-16주 순서 + Tier 표)",
        href: "/research-wiki/2026-06/vlm-edge-multiturn" },
      { label: "📜 미선정 / Drop 로그 (R7 COMPUTE-PREFETCH + A SubCover[MMTok scoop] / G FocusBench / VISION-BLASST 흡수)",
        href: "/research-wiki/2026-06/vlm-edge-multiturn/unselected" },
    ],
  },
  {
    date: "2026-05-27",
    title: "🥈 AI-Workload-Aware Memory Error Mitigation & Page Migration on HBM/GDDR/LPDDR (R67/R68 적용)",
    href: "/research-wiki/2026-05/ai-memory-error-mitigation",
    keywords: [
      "ECC telemetry → frame-health map (KEEPER)",
      "Protect–Migrate–Recompute trichotomy (TRIAD)",
      "Inference-induced read-disturbance (ROWPRESS-AI)",
      "Block-syndrome KV checksum (TALLY)",
      "Error-robust KV quantization (TEMPER)",
      "Reliability-cost KV tiering term (BALLAST)",
      "Ramulator2 PRAC/ECS + GoldenTransformer mimic",
      "HBM3/GDDR6/LPDDR5x on-package only (no CXL testbed)",
      "vLLM/InfiniGen/LMCache application-level",
      "Single GPU 검증 (R20-γ)",
      "Heavy-hitter KV criticality term",
      "DNN-Defender 차별화 (weight→KV)",
      "GPUHammer / Story-of-Two-GPUs motivation",
      "Capacity 손실 0 (offline 대비)",
    ],
    summary:
      "최신 LLM/VLM serving 의 메모리 access·error-sensitivity skew (weights vs KV-heavy-hitter vs transient activation) 를 1급 신호로 삼아 **on-package 메모리(HBM/GDDR/LPDDR)** 의 error mitigation + page migration 을 재설계. CXL testbed 없이 **Ramulator2 + vLLM/GoldenTransformer mimic** 으로 single GPU 검증. 3 expert × 6-7 idea + 5 reviewer dispatch — **Tier-1 Top 3 (KEEPER 8.16 ASPLOS/MICRO / TRIAD 7.96 MICRO/ISCA / ROWPRESS-AI 7.86 ISCA) + Tier-2 독립 Top 3 (TALLY DAC/DATE / TEMPER DAC/DATE / BALLAST EuroSys-short/DAC)**. GPUHammer ([arXiv:2507.08166](https://arxiv.org/abs/2507.08166), USENIX Security 2025) GDDR6 single bit-flip → DNN accuracy −80% + Story of Two GPUs ([arXiv:2503.11901](https://arxiv.org/abs/2503.11901)) H100 HBM3 per-GB MTBE −24% baseline motivation. DNN-Defender (weight row swap) 와 직교한 **dynamic KV criticality migration** 이 차별화 핵심. R67 5-reviewer × 4 sub-axis ★/▼ + R68 GitHub link main branch + line-anchored 의무 모두 충족.",
    detailed: true,
    tierTable: [
      { tier: "🥇 Tier-1", idea: "KEEPER (ECC-Telemetry-Driven Critical-KV Cache Migration)",
        venue: "ASPLOS 2027 / MICRO 2027", score: "8.16",
        href: "/research-wiki/2026-05/ai-memory-error-mitigation/tier1/01-keeper" },
      { tier: "🥈 Tier-1", idea: "TRIAD (Protect–Migrate–Recompute Trichotomy Controller)",
        venue: "MICRO 2027 / ISCA 2027", score: "7.96",
        href: "/research-wiki/2026-05/ai-memory-error-mitigation/tier1/02-triad" },
      { tier: "🥉 Tier-1", idea: "ROWPRESS-AI (Inference-Induced Read-Disturbance Characterization + Mitigation)",
        venue: "ISCA 2027 (main)", score: "7.86",
        href: "/research-wiki/2026-05/ai-memory-error-mitigation/tier1/03-rowpress-ai" },
      { tier: "T1", idea: "TALLY (Block-Syndrome KV Checksum Detector)",
        venue: "DAC 2027 / DATE 2027 (6p)", score: "Tier-2",
        href: "/research-wiki/2026-05/ai-memory-error-mitigation/tier2/01-tally" },
      { tier: "T2", idea: "TEMPER (Error-Robust KV Quantization Calibration)",
        venue: "DAC 2027 / DATE 2027 (6p)", score: "Tier-2",
        href: "/research-wiki/2026-05/ai-memory-error-mitigation/tier2/02-temper" },
      { tier: "T3", idea: "BALLAST (Reliability-Cost KV Tiering Placement Term)",
        venue: "EuroSys 2027 short / DAC 2027", score: "Tier-2",
        href: "/research-wiki/2026-05/ai-memory-error-mitigation/tier2/03-ballast" },
    ],
    extraLinks: [
      { label: "📊 Landing (Research Questions + Essential Reading + Decision Tree + Tier 표)",
        href: "/research-wiki/2026-05/ai-memory-error-mitigation" },
      { label: "📜 미선정 / Drop 로그 (DNN-Defender weight axis / TRIAGE 흡수 등)",
        href: "/research-wiki/2026-05/ai-memory-error-mitigation/unselected" },
    ],
  },
  {
    date: "2026-05-26",
    title: "🥈 VLM Speculative Decoding + Chunked Prefill 결합 최적화 (CPU-GPU + Energy + Accuracy, R67/R68 첫 적용)",
    href: "/research-wiki/2026-05/vlm-specdec-chunked-prefill",
    keywords: [
      "KL Pinsker + Bretagnolle-Huber bound (KL-Bounded Distillation)",
      "Modality-asymmetric Hessian quantization (Sub-2-bit Visual KV)",
      "3-stream priority + pinned triple-buffer + TBC (KV-Stream-TBC)",
      "Cross-attn saliency tree (VAST-Sched)",
      "cudaAccessPolicyWindow L2 set-aside (SpecVerify-L2)",
      "CPU-resident draft + AMX/VNNI dual-path (Tortoise & Hare)",
      "vLLM v1 + EAGLE training",
      "Memory −50% (RTX 5090 32GB enabling)",
      "CoVSpec/HiViS/LVSpec/QuantSpec orthogonal differentiation",
      "Sarathi-Serve / POD-Attention / Mooncake baseline",
      "R52.2 7-column + R68 GitHub link main + line-anchored",
      "R67 5-reviewer × 4 sub-axis ★/▼ reason",
    ],
    summary:
      "VLM serving 의 5축 결합 최적화 (Speculative Decoding + Chunked Prefill + CPU-GPU 협업 + Energy + Accuracy 보존) ideation. 3 expert × 6-7 idea = 19 candidate → Phase 2 (5 reviewer 병렬, R30.6 clone-based codebase verification 의무) → Phase 1' refinement (vLLM v0→v1 path 정정, CoVSpec/HiViS/LVSpec/QuantSpec scoop 차별화 axis 보강) → Phase 2' integrated 재리뷰 (Grade mapping) → **Tier-1 Top 3** (KL-Bounded Distillation 8.55 NeurIPS/ICML / Sub-2-bit Visual KV 8.40 MLSys/ASPLOS / KV-Stream-TBC 8.40 ASPLOS/MICRO/ISCA) + **Tier-2 독립 Top 3** (VAST-Sched 7.25 / SpecVerify-L2 7.95 / Tortoise & Hare 7.30, 모두 DAC/DATE/ICCAD/MLSys-Industry). Hallucinated path 6건 (EAGLE `train/loss.py`→`train/main.py:L231-L237` / vLLM `attention/ops/paged_attn.py`→`csrc/attention/paged_attention_v1.cu` / `multimodal/processing.py`→`processing/processor.py:L972` Subpackage Refactor / `quantization/gptq.py`→`auto_gptq.py:L95` 등) 모두 curl 실존 검증 후 정정. R67/R68 신규 rule 첫 적용 — 모든 R52.2 표 GitHub link main branch 100% + line-anchored.",
    detailed: true,
    tierTable: [
      { tier: "🥇 Tier-1", idea: "KL-Bounded VLM Draft Distillation (Pinsker + Bretagnolle–Huber)",
        venue: "NeurIPS 2027 / ICML 2027", score: "8.55",
        href: "/research-wiki/2026-05/vlm-specdec-chunked-prefill/tier1/01-kl-bounded-distillation" },
      { tier: "🥈 Tier-1", idea: "Sub-2-bit Visual KV Asymmetric Quantization (Hessian per-modality)",
        venue: "MLSys 2027 / ASPLOS 2027 / NeurIPS 2027", score: "8.40",
        href: "/research-wiki/2026-05/vlm-specdec-chunked-prefill/tier1/02-sub-2bit-visual-kv" },
      { tier: "🥉 Tier-1", idea: "KV-Stream-TBC (3-Stream Priority + Pinned Triple-Buffer + TBC Distributed Shmem)",
        venue: "ASPLOS 2027 / MICRO 2027 / ISCA 2027", score: "8.40",
        href: "/research-wiki/2026-05/vlm-specdec-chunked-prefill/tier1/03-kv-stream-tbc" },
      { tier: "T1", idea: "VAST-Sched (Visual Saliency Adaptive Speculative Tree Scheduler)",
        venue: "DAC 2027 / DATE 2027 (6p)", score: "7.25",
        href: "/research-wiki/2026-05/vlm-specdec-chunked-prefill/tier2/04-vast-sched" },
      { tier: "T2", idea: "SpecVerify-L2 (Acceptance-Aware SM Partition + L2 Persistent Window)",
        venue: "DAC 2027 / DATE 2027 / ICCAD 2027", score: "7.95",
        href: "/research-wiki/2026-05/vlm-specdec-chunked-prefill/tier2/05-specverify-l2" },
      { tier: "T3", idea: "Tortoise & Hare (Datacenter CPU-Resident Draft + AMX/VNNI Dual-Path)",
        venue: "DAC 2027 / MLSys-Industry 2027 (6p)", score: "7.30",
        href: "/research-wiki/2026-05/vlm-specdec-chunked-prefill/tier2/06-tortoise-hare" },
    ],
    extraLinks: [
      { label: "📊 Landing (RQ + Essential Reading + Implementation-Priority Decision Tree + Tier 표)",
        href: "/research-wiki/2026-05/vlm-specdec-chunked-prefill" },
      { label: "📜 미선정 / Drop 13 로그 (R67 sub-axis ★/▼ scoring 포함)",
        href: "/research-wiki/2026-05/vlm-specdec-chunked-prefill/unselected" },
    ],
  },
  {
    date: "2026-05-02",
    title: "VLM Scenario-Aware Optimization (Mosaic / Lattice / Bramble — R45 신규 적용)",
    href: "/research-wiki/2026-05/vlm-scenario-aware",
    keywords: ["Scenario-Aware Dispatcher (Mosaic)", "Frame-Indexed Radix Vision KV (Lattice)", "Cross-Image Vision Token Pool with Privacy (Bramble)", "NVDEC Sliding-Window KV (Lantern)", "Ego-Motion Compression (Compass)", "DocVQA L2 Carveout (Hearth)", "vLLM Prefix 60-85% hit", "mlx-vlm Issue #832 multi-turn gap", "VLCache cross-session 직교", "PolyKV/KVShare vision token 직교", "MMBench-Video / MLVU / MileBench / EgoSchema 11 benchmark", "R52.2 6-column / R53 일반 단어 / R19-α 자연어 acronym"],
    summary:
      "사용자 input — VLM 시나리오별 사전 분류 + multi-turn QA visual info + accuracy 보존 + 기존 benchmark 활용. R31 검증 통과 (production prefix hit 60-85% measured common case via vLLM/SGLang RadixAttention, mlx-vlm Issue #832 multi-turn gap evidence). 3 expert × 7 idea = 21 candidate × 5 reviewer dispatch — Differentiation reviewer web search 로 critical scoop 12편 발견 (ECVL-ROUTER ICLR'26 / Nova [arXiv:2509.21301](https://arxiv.org/abs/2509.21301) / VLCache [arXiv:2512.12977](https://arxiv.org/abs/2512.12977) 2025-12 / PolyKV [arXiv:2604.24971](https://arxiv.org/abs/2604.24971) / KVShare [arXiv:2503.16525](https://arxiv.org/abs/2503.16525) / VisionThink ICLR'26 / LongVU / TSG 91.4% / AwaRes / AttWarp / OxyGen / METok). **Tier-1 Top 3** (Mosaic 7.5 / Lattice 7.5 / Bramble 7.2) + **Tier-2 독립 Top 3** (Lantern / Compass / Hearth, 모두 6.0). **DROP 14** (70%+ scoop) + **흡수 6** (A5→Lantern / B7→Bramble / C1→Lattice / C3→Bramble / C5→Lattice / C6→Mosaic). 6-axis cover (Performance / Memory / Cost / Energy / Quality / Security) + 11 benchmark 분담. R52.2 6-column 강화 + R53 일반 단어 section title (동작 원리 / 기대 효과 / 구현 변경점 / 검증 시나리오) + R19-α 자연어 acronym + full title vendor-neutral.",
    detailed: true,
    tierTable: [
      { tier: "🥇 Tier-1", idea: "Mosaic (Workload-Adaptive Serving Configuration Dispatcher)", venue: "OSDI 2027 / ASPLOS 2027", score: "6/7/9.5 → 7.5",
        href: "/research-wiki/2026-05/vlm-scenario-aware/tier1/01-mosaic" },
      { tier: "🥈 Tier-1", idea: "Lattice (Cross-Turn Frame-Indexed Radix Vision KV Cache for Multi-Turn Video QA)", venue: "MLSys 2027 / NeurIPS 2026", score: "8/6/8 → 7.5",
        href: "/research-wiki/2026-05/vlm-scenario-aware/tier1/02-lattice" },
      { tier: "🥉 Tier-1", idea: "Bramble (Cross-Image Vision Token Pool for Multi-Image Agent Loop)", venue: "MLSys 2027 / OSDI 2027", score: "8/5/8.5 → 7.2",
        href: "/research-wiki/2026-05/vlm-scenario-aware/tier1/03-bramble" },
      { tier: "T1", idea: "Lantern (NVDEC-Coupled Sliding-Window KV for Streaming Video)", venue: "IEEE CAL letter + vLLM upstream PR", score: "5/8/6 → 6.0",
        href: "/research-wiki/2026-05/vlm-scenario-aware/tier2/01-lantern" },
      { tier: "T2", idea: "Compass (Ego-Motion-Aware Compression for Egocentric Video)", venue: "DATE 2027 / ISLPED 2027", score: "7/7/6 → 6.0",
        href: "/research-wiki/2026-05/vlm-scenario-aware/tier2/02-compass" },
      { tier: "T3", idea: "Hearth (Document VLM L2 Carveout + Tile Locality Boost)", venue: "DATE 2027 / IEEE ESL letter", score: "6/6/7 → 6.0",
        href: "/research-wiki/2026-05/vlm-scenario-aware/tier2/03-hearth" },
    ],
    extraLinks: [
      { label: "📊 Landing (Inline SVG Decision Tree + Tier-1/Tier-2 link 표 + 11 benchmark 분담)",
        href: "/research-wiki/2026-05/vlm-scenario-aware" },
      { label: "📜 미선정 / Drop 14 + 흡수 6 로그 (VLCache / PolyKV / KVShare / VisionThink / LongVU / TSG / AwaRes / AttWarp / OxyGen scoop)",
        href: "/research-wiki/2026-05/vlm-scenario-aware/unselected" },
    ],
  },
  {
    date: "2026-04-28",
    title: "🥈 VLM Edge Layer-Wise + Context-Semantic Optimization on Single-GPU/Jetson (R57 신규 적용 첫 세션, ATRIUM Tier-1 4번째 retain)",
    href: "/research-wiki/2026-04/vlm-edge-layerwise-context",
    keywords: ["R57 Summary Entry-Friendly", "Research Questions 최상단", "Essential Reading 5편", "Glossary 맨 뒤 + CTRL+F", "RTX 5090", "Jetson Thor", "Jetson Orin NX", "NVFP4 mixed precision", "DeepStack-aware", "visual KV cluster", "cross-frame reuse", "Phase-aware LSH", "ATRIUM Tier-1 retain", "Bivouac Mosaic concurrent 잔존", "Obelisk DynaExq adjacent"],
    summary:
      "최신 VLM (Qwen3-VL/LLaVA-Next/InternVL3) 의 RTX 5090 / Jetson Thor (LPDDR5x 273 GB/s) / Jetson Orin NX (10-25W) edge inference 에서 **layer-wise 최적화 + Context/Semantic 특성** 활용 ideation. 3 expert × 13 candidate → Phase 2 (3 reviewer) → Phase 1' refinement (cluster merge: PRISMATIC-FOG ⊕ PRISM-FX → Prism, ATRIUM-R(AI) M2 ⊕ BIVOUAC-SLATE → Bivouac) → Phase 2' (최근 6개월 scoop 재검증) → **Tier-1 Top 4** (Prism / Bivouac / RadixVL W12 분기 / **ATRIUM** 사용자 명시 retain) **+ Tier-2 독립 Top 3** (Strata / Harbinger / Obelisk). **Drop 4**: CARILLON (Nova arXiv:2509.21301 80%+ scoop), BREAKWATER-T-R (Jetson Thor T5000 MIG single-partition only 2026-04 infeasible), TIDEGATE (R56.2 30% 미달), HARBOR-DLA (DLA 2.0 LayerNorm 제외). **Critical scoop 잔존**: Bivouac ↔ Mosaic ([arXiv:2604.10060](https://arxiv.org/abs/2604.10060), 2026-04-11) 55-65% concurrent / Obelisk ↔ DynaExq ([arXiv:2511.15015](https://arxiv.org/abs/2511.15015), 2025-11-19) 35-45% adjacent. **R57 신규 적용 첫 세션** — Summary README 가 학부생/AI agent 진입장벽 완화 구조 (Research Questions → Essential Reading → 연구 개요+GAP outline → Decision Tree → Tier-1/2 + contribution bullet → Glossary 맨 뒤 + CTRL+F).",
    detailed: true,
    tierTable: [
      { tier: "🥇 Tier-1", idea: "Prism (R58 cleanup, NVFP4 mixed precision)", venue: "MLSys 2027 / ASPLOS 2027 / NeurIPS 2026", score: "7.5/8.0/9.0",
        href: "/research-wiki/2026-04/vlm-edge-layerwise-context/tier1/01-prism" },
      { tier: "🥈 Tier-1", idea: "Bivouac", venue: "NeurIPS 2026 / ICML 2026 / MICRO 2027", score: "7.5/8.5/8.0",
        href: "/research-wiki/2026-04/vlm-edge-layerwise-context/tier1/02-bivouac" },
      { tier: "🥉 Tier-1", idea: "RadixVL (W12 동적 분기)", venue: "OSDI 2026 / SOSP 2027", score: "5.5/7.5/8.5",
        href: "/research-wiki/2026-04/vlm-edge-layerwise-context/tier1/03-radixvl" },
      { tier: "4️⃣ Tier-1", idea: "Atrium (retain, v2-r55 origin)", venue: "HPCA 2027 / MICRO 2027 / ASPLOS 2027", score: "7.7/7.5/8.0",
        href: "/research-wiki/2026-04/vlm-edge-layerwise-context/tier1/04-atrium" },
      { tier: "T1", idea: "Strata (Stratified KV layout)", venue: "DAC 2027 / DATE 2027", score: "6.5/8.0/7.5",
        href: "/research-wiki/2026-04/vlm-edge-layerwise-context/tier2/01-strata" },
      { tier: "T2", idea: "Harbinger", venue: "ISLPED 2027 / DATE 2027", score: "7.0/7.5/6.5",
        href: "/research-wiki/2026-04/vlm-edge-layerwise-context/tier2/02-harbinger" },
      { tier: "T3", idea: "Obelisk (RTX 5090 large MoE)", venue: "MLSys 2027 / DAC 2027", score: "6.0/6.5/7.5",
        href: "/research-wiki/2026-04/vlm-edge-layerwise-context/tier2/03-obelisk" },
    ],
    extraLinks: [
      { label: "📊 Landing (R57 신규 구조 README — RQ + Essential Reading + GAP outline + Decision Tree)",
        href: "/research-wiki/2026-04/vlm-edge-layerwise-context" },
      { label: "📜 미선정 / Drop 4 로그 (CARILLON Nova scoop / BREAKWATER-T-R MIG infeasible / TIDEGATE / HARBOR-DLA)",
        href: "/research-wiki/2026-04/vlm-edge-layerwise-context/unselected" },
    ],
  },
  {
    date: "2026-04-25",
    title: "🥈 KV Cache Memory ECC + RAS for Quantized AI Serving",
    href: "/research-wiki/2026-04/kv-ecc-ras",
    keywords: ["KV cache", "ECC", "Memory RAS", "Page retirement", "Quantization-aware", "Outlier-aware", "DRAM error", "Simulator-only", "R45/R46 strict"],
    summary:
      "LLM/VLM prefill stage KV cache memory error correction (ECC) + RAS (page retirement / page migration) — quantization-aware. 시간 제약 학생 (12-16주 simulator-only) 시나리오 위주. 3 expert × 7 idea = 21 candidate → 3 reviewer 평가 (Kelle MICRO 2025 [arXiv:2510.16040](https://arxiv.org/abs/2510.16040) critical missing baseline 식별 + 6 self-scoop pair merge) → Tier-1 Top 3 + Tier-2 Top 3. **R45 strict** (kernel patch / register write 금지, application-level + simulator only): gem5+DRAMSim3 / ChampSim / NeuPIMs / AttAcc / NeuroSim V1.4 / LLMServingSim. **R46 strict**: 모든 reference WebFetch verified (PerfVec 류 hallucination 방지).",
    detailed: true,
    tierTable: [
      { tier: "🥇 Tier-1", idea: "OAEP-KV (lead)", venue: "DSN/HPCA/MICRO 2027 (12p)", score: "8+ / risk 3",
        href: "/research-wiki/2026-04/kv-ecc-ras/tier1/01-oaep-kv" },
      { tier: "🥈 Tier-1", idea: "BlockShard", venue: "ASPLOS/OSDI 2027 (12-15p)", score: "8 / risk 3",
        href: "/research-wiki/2026-04/kv-ecc-ras/tier1/02-blockshard" },
      { tier: "🥉 Tier-1", idea: "LayerTier", venue: "MICRO/DSN 2027 (12p)", score: "7.5 / risk 4",
        href: "/research-wiki/2026-04/kv-ecc-ras/tier1/03-layertier" },
      { tier: "T1", idea: "VLM-MAP", venue: "DATE 6p / IEEE TCAD", score: "6.5 / risk 3",
        href: "/research-wiki/2026-04/kv-ecc-ras/tier2/01-vlm-map" },
      { tier: "T2", idea: "FrostFloor", venue: "DATE 6p", score: "6 / risk 3",
        href: "/research-wiki/2026-04/kv-ecc-ras/tier2/02-frostfloor" },
      { tier: "T3", idea: "EntropyECC", venue: "ITC/MTS/IEEE TCAD", score: "6 / risk 4",
        href: "/research-wiki/2026-04/kv-ecc-ras/tier2/03-entropy-ecc" },
    ],
    extraLinks: [
      { label: "📊 Landing (Student Decision Tree SVG)",
        href: "/research-wiki/2026-04/kv-ecc-ras" },
      { label: "📜 미선정 15편 로그 (6 self-scoop merge 포함)",
        href: "/research-wiki/2026-04/kv-ecc-ras/unselected" },
    ],
  },
  {
    date: "2026-04-25",
    title: "VLM Context-aware Serving on Jetson Edge (R45 적용)",
    href: "/research-wiki/2026-04/vlm-context-edge-jetson",
    keywords: ["VLM", "Edge", "Jetson Thor", "Jetson Orin NX/Nano", "Context-aware", "Energy", "R45 implementation feasibility", "register-write 금지", "simulator-path"],
    summary:
      "Jetson Thor 128GB / Orin NX 16GB / Orin Nano 8GB 의 4 edge constraint (UMA / DLA / LPDDR5X / thermal) 신규 ideation. 3 expert × 7 idea = 21 candidate → 3 reviewer (novelty / diff / impact+scoop) 평가 → 6 selected + 16 미선정. **R45 신규 적용 (구현·검증 난이도 + risk discipline)**: undocumented BPMP IOCTL / kernel patch / register write 등 vendor 비공식 mechanism 은 default 제외 — novelty 강할 시에만 gem5 / DRAMSim3 / ChampSim simulator path 로 reframe + risk 7-10/10 명시. ★ 1차 publish 의 CacheVeil (Tier-1 lead) 는 ARM CMN partition register undocumented BPMP IOCTL 의존 → R45.1 위반 → Tier-2 simulator-path spinoff (CacheVeil-Sim) 으로 demote. Glacier Migrate 도 DLA SRAM physical addr exposure + 12-20주 simulator overhead 로 unselected 이동.",
    detailed: true,
    tierTable: [
      { tier: "🥇 Tier-1", idea: "VESPER (lead)", venue: "OSDI/SOSP 2027 (12-15p)", score: "8.10 / risk 3",
        href: "/research-wiki/2026-04/vlm-context-edge-jetson/tier1/03-vesper" },
      { tier: "🥈 Tier-1", idea: "SHOAL", venue: "MLSys/ASPLOS 2027 (12-18p)", score: "8.07 / risk 4",
        href: "/research-wiki/2026-04/vlm-context-edge-jetson/tier1/02-shoal" },
      { tier: "🥉 Tier-1", idea: "DualLane", venue: "ISCA/MICRO 2027 (12-15p)", score: "8.00 / risk 4",
        href: "/research-wiki/2026-04/vlm-context-edge-jetson/tier1/04-duallane" },
      { tier: "T1", idea: "TUFA", venue: "IEEE CAL 4p", score: "6.47 / risk 3",
        href: "/research-wiki/2026-04/vlm-context-edge-jetson/tier2/01-tufa" },
      { tier: "T2", idea: "ShelfSwap", venue: "ISLPED/DATE 6p", score: "6.07 / risk 4",
        href: "/research-wiki/2026-04/vlm-context-edge-jetson/tier2/02-shelfswap" },
      { tier: "T3", idea: "CacheVeil-Sim (R45 spinoff)", venue: "DAC/ISLPED 6p (gem5+ChampSim)", score: "6.50 / risk 7",
        href: "/research-wiki/2026-04/vlm-context-edge-jetson/tier2/03-cacheveil-sim" },
    ],
    extraLinks: [
      { label: "📊 Landing (R45 결정 트리 SVG)",
        href: "/research-wiki/2026-04/vlm-context-edge-jetson" },
      { label: "🚨 CacheVeil 원안 (R45 demoted)",
        href: "/research-wiki/2026-04/vlm-context-edge-jetson/tier1/01-cacheveil" },
      { label: "📜 미선정 16편 로그",
        href: "/research-wiki/2026-04/vlm-context-edge-jetson/unselected" },
    ],
  },
  {
    date: "2026-04-24",
    title: "MoE Expert-Activation Fingerprint 기반 보안 + 서빙 최적화",
    href: "/research-wiki/2026-04/moe-fingerprint-security-serving",
    keywords: ["MoE", "Security", "Adversarial", "Serving", "Fingerprint", "S&P 2027", "MLSys 2027"],
    summary:
      "이방산 학생의 4 MoE 모델 × ~5,900 runs 실험 (WildJailbreak 2-way 94.0%) 기반. 3 paper-worthy 축 확정: (1) MoE discrete routing adaptive-adversarial, (2) token×layer 2D early-exit Pareto, (3) information-theoretic shared-substrate. Tier-1 Top 3 (DISCRETE-VEIL S&P 8.00 / LOOM MLSys 7.38 / BEACON-GUARD ATC 7.25) + Tier-2 Top 3.",
  },
  {
    date: "2026-04-24",
    title: "Qwen3-VL (DeepStack) + Qwen3.5 Edge 배포 최적화",
    href: "/research-wiki/2026-04/qwen3vl-deepstack-edge",
    keywords: ["Qwen3-VL", "DeepStack", "Qwen3.5-Omni", "Jetson Thor", "LPDDR5X", "MoE 30B-A3B", "Interleaved MRoPE", "Edge GPU"],
    summary:
      "Qwen3-VL (arXiv:2511.21631) + Qwen3.5-Omni (arXiv:2604.15804) 3 PDF 입력. R23 Step 0-α 신규 규칙 (IISWC/ISPASS/benchmark report 우선) 첫 적용. 5 Top idea: ★ Loom (Interleaved MRoPE LUT + FA3 fused) / Mangrove (DeepStack 4-stage + LPDDR bank + DLA) / Vault' (DeepStack×MoE L2 contention) + Tier-2 Gale (GDN:Attn 3:1) / Forge (Thinker-Talker Tensor Core).",
  },
  {
    date: "2026-04-23",
    title: "에너지 효율적 Edge VLM Inference",
    href: "/research-wiki/2026-04/energy-efficient-edge-vlm",
    keywords: ["VLM", "Edge GPU", "Energy", "Qwen2.5-VL", "InternVL3", "AnyRes Tile", "MRoPE 3D", "R23-R26 첫 적용"],
    summary:
      "LLaVA-1.5 (fixed 576 token) vs 최신 Qwen2.5-VL/InternVL3 아키텍처 diff 활용. 5 Top idea: Parquet (AnyRes tile-count signal × DVFS × per-tile precision) / Triptych (DLA + UMA zero-copy + DLA preemption) + Tier-2 Cartographer / Sift / Verge. Tidal DROP (CodecSight 2604.06036 68-72% scoop).",
  },
  {
    date: "2026-04-22",
    title: "VLM+PIM 연구 보완·확장",
    href: "/research-wiki/2026-04/vlm-pim-extension",
    keywords: ["VLM", "PIM", "AttAcc", "Mode 2", "Quant-Robust"],
    summary:
      "AttAcc(ASPLOS'24) baseline + Qwen3-VL-4B 기반. 8 ideas → 3 fused → Top 2 (F2 Quant-Robust Layered Defense / F1 DeepStack-Native 6-Tier Pipeline). W8A8 +66pp visual attention collapse 선행 보고 부재 검증.",
  },
];

export default function ResearchWikiIndexPage() {
  return (
    <div className="max-w-5xl mx-auto px-6 py-12">
      <h1 className="text-3xl font-bold mb-2" style={{ color: "var(--text-primary)" }}>
        AICA Research Wiki
      </h1>
      <p className="text-sm mb-2" style={{ color: "var(--text-muted)" }}>
        SSU AICA Lab — 5 전문가 + 3 리뷰어 협업 ideation harness 의 누적 결과 · 좌측 사이드바에서 모든 세션 클릭 탐색
      </p>
      <p className="text-xs italic mb-10" style={{ color: "var(--text-muted)" }}>
        Hidden link · 네비게이션 비노출 · 본 페이지 + 사이드바는 외부 색인되지 않음.
      </p>

      <h2 className="text-xl font-semibold mb-6" style={{ color: "var(--text-primary)" }}>
        ⭐ 최근 세션
      </h2>

      <div className="space-y-8">
        {recentSessions.map((s, i) => (
          <article
            key={s.href}
            className="rounded-lg p-6"
            style={{
              border: "1px solid var(--border)",
              backgroundColor: i === 0 ? "var(--bg-secondary, #f4f6fb)" : "transparent",
            }}
          >
            <div className="flex items-baseline gap-3 mb-2 flex-wrap">
              <span className="text-xs" style={{ color: "var(--text-muted)" }}>
                {s.date}
              </span>
              <Link
                href={s.href}
                className="text-lg font-semibold hover:underline"
                style={{ color: "var(--accent)" }}
              >
                {s.title}
              </Link>
            </div>

            <div className="flex flex-wrap gap-1.5 mb-3">
              {s.keywords.map((k) => (
                <span
                  key={k}
                  className="text-[10px] font-medium px-2 py-0.5 rounded"
                  style={{
                    backgroundColor: "var(--bg-secondary)",
                    color: "var(--text-muted)",
                    border: "1px solid var(--border)",
                  }}
                >
                  {k}
                </span>
              ))}
            </div>

            <p className="text-sm mb-3" style={{ color: "var(--text-secondary)" }}>
              {s.summary}
            </p>

            {s.detailed && s.tierTable && (
              <div className="mt-4 mb-2 overflow-x-auto">
                <table className="text-xs w-full" style={{ color: "var(--text-secondary)" }}>
                  <thead>
                    <tr style={{ borderBottom: "1px solid var(--border)" }}>
                      <th className="text-left py-1 pr-3 font-semibold">Tier</th>
                      <th className="text-left py-1 pr-3 font-semibold">Idea</th>
                      <th className="text-left py-1 pr-3 font-semibold">Venue</th>
                      <th className="text-right py-1 font-semibold">Score</th>
                    </tr>
                  </thead>
                  <tbody>
                    {s.tierTable.map((row) => (
                      <tr key={row.idea} style={{ borderBottom: "1px solid var(--border)" }}>
                        <td className="py-1 pr-3">{row.tier}</td>
                        <td className="py-1 pr-3">
                          <Link href={row.href} className="hover:underline" style={{ color: "var(--accent)" }}>
                            {row.idea}
                          </Link>
                        </td>
                        <td className="py-1 pr-3">{row.venue}</td>
                        <td className="text-right py-1 font-mono">{row.score}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}

            {s.extraLinks && (
              <div className="flex gap-3 text-xs mt-3 flex-wrap">
                {s.extraLinks.map((l) => (
                  <Link
                    key={l.href}
                    href={l.href}
                    className="hover:underline"
                    style={{ color: "var(--text-muted)" }}
                  >
                    {l.label}
                  </Link>
                ))}
              </div>
            )}
          </article>
        ))}
      </div>

      <h2 className="text-xl font-semibold mt-12 mb-4" style={{ color: "var(--text-primary)" }}>
        🔍 빠른 탐색
      </h2>
      <div className="grid grid-cols-2 md:grid-cols-3 gap-3 text-sm">
        {[
          { l: "📅 최근 세션 timeline", h: "/research-wiki/index" },
          { l: "💡 모든 아이디어", h: "/research-wiki/ideas" },
          { l: "📚 분석된 논문", h: "/research-wiki/papers" },
          { l: "📖 개념 용어집", h: "/research-wiki/concepts" },
          { l: "📈 학회/분야 트렌드", h: "/research-wiki/trends" },
        ].map((q) => (
          <Link
            key={q.h}
            href={q.h}
            className="rounded p-3 hover:underline"
            style={{ border: "1px solid var(--border)", color: "var(--text-secondary)" }}
          >
            {q.l}
          </Link>
        ))}
      </div>
    </div>
  );
}
