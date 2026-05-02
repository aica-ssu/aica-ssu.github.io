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
    date: "2026-05-02",
    title: "🥇 VLM Scenario-Aware Optimization (Mosaic / Lattice / Bramble — R45 신규 적용)",
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
    title: "🥇 KV Cache Memory ECC + RAS for Quantized AI Serving",
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
