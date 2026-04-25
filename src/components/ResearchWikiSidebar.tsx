import Link from "next/link";

type SidebarSection = {
  title: string;
  items: { label: string; href: string; emphasis?: boolean }[];
};

const SIDEBAR: SidebarSection[] = [
  {
    title: "🏠 Top",
    items: [
      { label: "Home", href: "/research-wiki" },
      { label: "최근 세션 timeline", href: "/research-wiki/index" },
      { label: "모든 아이디어", href: "/research-wiki/ideas" },
      { label: "분석된 논문", href: "/research-wiki/papers" },
      { label: "개념 용어집", href: "/research-wiki/concepts" },
      { label: "학회 트렌드", href: "/research-wiki/trends" },
    ],
  },
  {
    title: "⭐ 최근 세션 (2026-04-24, 가장 최신)",
    items: [
      {
        label: "Landing — MoE Fingerprint Security+Serving",
        href: "/research-wiki/2026-04/moe-fingerprint-security-serving",
        emphasis: true,
      },
      {
        label: "🥇 DISCRETE-VEIL (S&P 2027) lead",
        href: "/research-wiki/2026-04/moe-fingerprint-security-serving/tier1/01-discrete-veil",
      },
      {
        label: "🥈 LOOM (MLSys/ASPLOS 2027)",
        href: "/research-wiki/2026-04/moe-fingerprint-security-serving/tier1/02-loom",
      },
      {
        label: "🥉 BEACON-GUARD (ATC/DATE)",
        href: "/research-wiki/2026-04/moe-fingerprint-security-serving/tier1/03-beacon-guard",
      },
      {
        label: "T1 DISCRETE-VEIL-Lite (IEEE CAL)",
        href: "/research-wiki/2026-04/moe-fingerprint-security-serving/tier2/01-discrete-veil-lite",
      },
      {
        label: "T2 TALLY-Spinoff (DATE)",
        href: "/research-wiki/2026-04/moe-fingerprint-security-serving/tier2/02-tally-spinoff",
      },
      {
        label: "T3 BEACON-GUARD-DATE (DATE 6p)",
        href: "/research-wiki/2026-04/moe-fingerprint-security-serving/tier2/03-beacon-guard-date",
      },
      {
        label: "미선정 / 변경 / 흡수 로그",
        href: "/research-wiki/2026-04/moe-fingerprint-security-serving/unselected",
      },
    ],
  },
  {
    title: "📚 이전 Summary",
    items: [
      { label: "2026-04-24 Qwen3-VL DeepStack Edge", href: "/research-wiki/2026-04/qwen3vl-deepstack-edge" },
      { label: "2026-04-23 Energy-Efficient Edge VLM", href: "/research-wiki/2026-04/energy-efficient-edge-vlm" },
      { label: "2026-04-23 VLM/VLA Context Serving v3", href: "/research-wiki/2026-04/vlm-vla-context-serving-v3" },
      { label: "2026-04-22 VLM/VLA Context Serving v2", href: "/research-wiki/2026-04/vlm-vla-context-serving-v2" },
      { label: "2026-04-22 VLM/VLA Context Serving v1", href: "/research-wiki/2026-04/vlm-vla-context-serving" },
      { label: "2026-04-22 VLM+PIM Extension", href: "/research-wiki/2026-04/vlm-pim-extension" },
      { label: "2026-04-21 ACE-MoE VLM/VLA Extension", href: "/research-wiki/2026-04/ace-moe-vlm-vla" },
    ],
  },
];

export default function ResearchWikiSidebar() {
  return (
    <aside
      className="research-wiki-sidebar shrink-0 border-r"
      style={{
        width: "320px",
        maxHeight: "calc(100vh - 64px)",
        overflowY: "auto",
        position: "sticky",
        top: "64px",
        borderColor: "var(--border)",
        backgroundColor: "var(--bg-secondary, #f4f6fb)",
        padding: "24px 16px",
      }}
    >
      <h2
        className="text-base font-bold mb-4"
        style={{ color: "var(--text-primary, #1E2761)" }}
      >
        AICA Research Wiki
      </h2>
      {SIDEBAR.map((section) => (
        <section key={section.title} className="mb-6">
          <h3
            className="text-xs font-semibold uppercase mb-2 tracking-wide"
            style={{ color: "var(--text-muted, #8893B8)" }}
          >
            {section.title}
          </h3>
          <ul className="space-y-1">
            {section.items.map((item) => (
              <li key={item.href}>
                <Link
                  href={item.href}
                  className="block text-sm py-1 px-2 rounded hover:underline"
                  style={{
                    color: item.emphasis
                      ? "var(--accent, #F96167)"
                      : "var(--text-secondary, #333)",
                    fontWeight: item.emphasis ? 600 : 400,
                  }}
                >
                  {item.label}
                </Link>
              </li>
            ))}
          </ul>
        </section>
      ))}
      <p
        className="text-[10px] mt-8 italic"
        style={{ color: "var(--text-muted, #8893B8)" }}
      >
        Hidden link — 외부 노출되지 않음 · 사이드바는 모든 research-wiki 페이지에 동일 노출.
      </p>
    </aside>
  );
}
