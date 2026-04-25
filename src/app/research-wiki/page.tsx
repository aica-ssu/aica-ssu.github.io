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
    date: "2026-04-24",
    title: "🥇 MoE Expert-Activation Fingerprint 기반 보안 + 서빙 최적화",
    href: "/research-wiki/2026-04/moe-fingerprint-security-serving",
    keywords: ["MoE", "Security", "Adversarial", "Serving", "Fingerprint", "S&P 2027", "MLSys 2027", "R28-R37 첫 적용"],
    summary:
      "이방산 학생의 4 MoE 모델 × ~5,900 runs 실험 (MMLU 4-cat 96.2% / WildJailbreak 2-way 94.0%) 기반. 사용자 insight (\"no-finetune + multi-task = paper-worthy\") 의 Tier-1 novelty 엄밀 검증 결과 3 진짜 paper-worthy 축 확정 — (1) MoE discrete routing 의 adaptive-adversarial robustness, (2) token×layer 2D early-exit Pareto, (3) information-theoretic shared-substrate. R28-R37 (hierarchical bundle / flow chart / source verified / R35 약어 glossary / R36 link validation / R37 cover-page-less) 첫 적용 세션.",
    detailed: true,
    tierTable: [
      { tier: "🥇 Tier-1", idea: "DISCRETE-VEIL", venue: "IEEE S&P 2027 (13p)", score: "8.00",
        href: "/research-wiki/2026-04/moe-fingerprint-security-serving/tier1/01-discrete-veil" },
      { tier: "🥈 Tier-1", idea: "LOOM (merged)", venue: "MLSys/ASPLOS 2027 (18p)", score: "7.38",
        href: "/research-wiki/2026-04/moe-fingerprint-security-serving/tier1/02-loom" },
      { tier: "🥉 Tier-1", idea: "BEACON-GUARD", venue: "USENIX ATC 2027 (12p)", score: "7.25",
        href: "/research-wiki/2026-04/moe-fingerprint-security-serving/tier1/03-beacon-guard" },
      { tier: "T1", idea: "DISCRETE-VEIL-Lite", venue: "IEEE CAL 4p", score: "7.10",
        href: "/research-wiki/2026-04/moe-fingerprint-security-serving/tier2/01-discrete-veil-lite" },
      { tier: "T2", idea: "TALLY-Spinoff", venue: "DATE 4p WIP", score: "6.80",
        href: "/research-wiki/2026-04/moe-fingerprint-security-serving/tier2/02-tally-spinoff" },
      { tier: "T3", idea: "BEACON-GUARD-DATE", venue: "DATE 6p", score: "6.90",
        href: "/research-wiki/2026-04/moe-fingerprint-security-serving/tier2/03-beacon-guard-date" },
    ],
    extraLinks: [
      { label: "📊 Landing (Flow Chart + 약어집)",
        href: "/research-wiki/2026-04/moe-fingerprint-security-serving" },
      { label: "📜 미선정 로그",
        href: "/research-wiki/2026-04/moe-fingerprint-security-serving/unselected" },
    ],
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
    date: "2026-04-23",
    title: "VLM/VLA Context-aware Caching & Serving (v3)",
    href: "/research-wiki/2026-04/vlm-vla-context-serving-v3",
    keywords: ["VLM", "VLA", "Caching", "Serving", "Dual Top-3", "Track B", "R21-R22 첫 적용", "OpenReview"],
    summary:
      "R21 Dual Top-3 + R22 summary 블로그-style 첫 적용. 6 Top: HRTS+ (HBM row-tile streaming) / ContextMIG+ (CLIP-L LSH × MIG) / PhaseGraph-VLA+ + Tier-2 독립 GCReconfProfile / TokenEvictEnergy / ActHeadFuse. v2 placeholder 7편 재검증 + 2026-04 20+편 신규 탐색.",
  },
  {
    date: "2026-04-22",
    title: "VLM/VLA Context-aware Caching & Serving (v2)",
    href: "/research-wiki/2026-04/vlm-vla-context-serving-v2",
    keywords: ["VLM", "VLA", "Tier-Mix", "Dual-Track", "OpenReview verified"],
    summary:
      "업데이트된 harness 규칙 (tiering ≤3-4, dual-track Top-tier+Tier-2, improve-first, OpenReview identity check) 전체 적용. 6 ideas → Top 3 Tier-Mix (HRTS Top-tier 7.90 / ContextMIG 7.75 / NACK-Gossip Tier-2 7.80 Conditional). OpenReview 3편 verified.",
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
