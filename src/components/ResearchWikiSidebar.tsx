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
    title: "⭐ 최근 세션 (2026-04-25, 가장 최신)",
    items: [
      {
        label: "Landing — VLM Context-aware Jetson Edge",
        href: "/research-wiki/2026-04/vlm-context-edge-jetson",
        emphasis: true,
      },
      {
        label: "🥇 CacheVeil (MICRO/HPCA 2027) lead",
        href: "/research-wiki/2026-04/vlm-context-edge-jetson/tier1/01-cacheveil",
      },
      {
        label: "🥈 SHOAL (MLSys/ASPLOS 2027)",
        href: "/research-wiki/2026-04/vlm-context-edge-jetson/tier1/02-shoal",
      },
      {
        label: "🥉 VESPER (OSDI/SOSP 2027)",
        href: "/research-wiki/2026-04/vlm-context-edge-jetson/tier1/03-vesper",
      },
      {
        label: "T1 TUFA (IEEE CAL 4p)",
        href: "/research-wiki/2026-04/vlm-context-edge-jetson/tier2/01-tufa",
      },
      {
        label: "T2 ShelfSwap (ISLPED/DATE 6p)",
        href: "/research-wiki/2026-04/vlm-context-edge-jetson/tier2/02-shelfswap",
      },
      {
        label: "T3 Glacier Migrate (ICCAD 8p)",
        href: "/research-wiki/2026-04/vlm-context-edge-jetson/tier2/03-glacier-migrate",
      },
      {
        label: "미선정 15편 로그",
        href: "/research-wiki/2026-04/vlm-context-edge-jetson/unselected",
      },
    ],
  },
  {
    title: "📚 이전 Summary",
    items: [
      { label: "2026-04-24 MoE Fingerprint Security+Serving", href: "/research-wiki/2026-04/moe-fingerprint-security-serving" },
      { label: "2026-04-24 Qwen3-VL DeepStack Edge", href: "/research-wiki/2026-04/qwen3vl-deepstack-edge" },
      { label: "2026-04-23 Energy-Efficient Edge VLM", href: "/research-wiki/2026-04/energy-efficient-edge-vlm" },
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
