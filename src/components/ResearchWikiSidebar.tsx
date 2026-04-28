"use client";

import Link from "next/link";
import { useState, useEffect } from "react";
import { usePathname } from "next/navigation";

type SidebarChild = {
  label: string;
  href: string;
  emphasis?: boolean;
};

type SidebarItem = {
  label: string;
  href: string;
  emphasis?: boolean;
  children?: SidebarChild[];
};

type SidebarSection = {
  title: string;
  items: SidebarItem[];
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
    title: "📚 모든 Summary (역시간순)",
    items: [
      {
        label: "⭐ 2026-04-28 VLM Edge Layer-Wise + Context",
        href: "/research-wiki/2026-04/vlm-edge-layerwise-context",
        emphasis: true,
        children: [
          { label: "Landing — VLM Edge Layer-Wise + Context-Semantic", href: "/research-wiki/2026-04/vlm-edge-layerwise-context" },
          { label: "🥇 PRISM-FOG-FX (MLSys 2027)", href: "/research-wiki/2026-04/vlm-edge-layerwise-context/tier1/01-prism-fog-fx" },
          { label: "🥈 BIVOUAC-SLATE-R (NeurIPS 2026)", href: "/research-wiki/2026-04/vlm-edge-layerwise-context/tier1/02-bivouac-slate-r" },
          { label: "🥉 PRISM-VL-R (OSDI 2026, W12 분기)", href: "/research-wiki/2026-04/vlm-edge-layerwise-context/tier1/03-prism-vl-r" },
          { label: "T1 STRATA-K-R (DAC 2027)", href: "/research-wiki/2026-04/vlm-edge-layerwise-context/tier2/01-strata-k-r" },
          { label: "T2 HARBINGER-CLOVER-R (ISLPED 2027)", href: "/research-wiki/2026-04/vlm-edge-layerwise-context/tier2/02-harbinger-clover-r" },
          { label: "T3 OBELISK-5090-R (MLSys 2027)", href: "/research-wiki/2026-04/vlm-edge-layerwise-context/tier2/03-obelisk-5090-r" },
          { label: "미선정 로그", href: "/research-wiki/2026-04/vlm-edge-layerwise-context/unselected" },
        ],
      },
      {
        label: "2026-04-26 KV cache ECC + RAS v2",
        href: "/research-wiki/2026-04/kv-ecc-ras-v2",
        children: [
          { label: "Landing — KV Cache ECC + RAS v2", href: "/research-wiki/2026-04/kv-ecc-ras-v2" },
          { label: "🥇 PrefixGuard (OSDI 2027)", href: "/research-wiki/2026-04/kv-ecc-ras-v2/tier1/01-prefixguard" },
          { label: "🥈 Quarantine (USENIX Security 2027)", href: "/research-wiki/2026-04/kv-ecc-ras-v2/tier1/02-quarantine" },
          { label: "🥉 PATroller (HPCA 2027)", href: "/research-wiki/2026-04/kv-ecc-ras-v2/tier1/03-patroller" },
          { label: "T1 ECS-Trace (ITC 2027 / DSN short)", href: "/research-wiki/2026-04/kv-ecc-ras-v2/tier2/01-ecs-trace" },
          { label: "T2 Quarantine-Mini (DAC 2027 6p)", href: "/research-wiki/2026-04/kv-ecc-ras-v2/tier2/02-quarantine-mini" },
          { label: "T3 PrefixGuard-Lite (DATE 2027 6p)", href: "/research-wiki/2026-04/kv-ecc-ras-v2/tier2/03-prefixguard-lite" },
          { label: "미선정 로그", href: "/research-wiki/2026-04/kv-ecc-ras-v2/unselected" },
        ],
      },
      {
        label: "2026-04-25 VLM Context-aware Jetson Edge",
        href: "/research-wiki/2026-04/vlm-context-edge-jetson",
        children: [
          { label: "Landing — VLM Context-aware Jetson Edge", href: "/research-wiki/2026-04/vlm-context-edge-jetson" },
          { label: "🥇 CacheVeil", href: "/research-wiki/2026-04/vlm-context-edge-jetson/tier1/01-cacheveil" },
          { label: "🥈 Shoal", href: "/research-wiki/2026-04/vlm-context-edge-jetson/tier1/02-shoal" },
          { label: "🥉 Vesper", href: "/research-wiki/2026-04/vlm-context-edge-jetson/tier1/03-vesper" },
          { label: "4️⃣ DualLane", href: "/research-wiki/2026-04/vlm-context-edge-jetson/tier1/04-duallane" },
          { label: "T1 Tufa", href: "/research-wiki/2026-04/vlm-context-edge-jetson/tier2/01-tufa" },
          { label: "T2 ShelfSwap", href: "/research-wiki/2026-04/vlm-context-edge-jetson/tier2/02-shelfswap" },
          { label: "T3 CacheVeil-Sim", href: "/research-wiki/2026-04/vlm-context-edge-jetson/tier2/03-cacheveil-sim" },
          { label: "미선정 로그", href: "/research-wiki/2026-04/vlm-context-edge-jetson/unselected" },
        ],
      },
      {
        label: "2026-04-24 MoE Fingerprint Security+Serving",
        href: "/research-wiki/2026-04/moe-fingerprint-security-serving",
        children: [
          { label: "Landing — MoE Fingerprint Security+Serving", href: "/research-wiki/2026-04/moe-fingerprint-security-serving" },
          { label: "🥇 DISCRETE-VEIL", href: "/research-wiki/2026-04/moe-fingerprint-security-serving/tier1/01-discrete-veil" },
          { label: "🥈 LOOM", href: "/research-wiki/2026-04/moe-fingerprint-security-serving/tier1/02-loom" },
          { label: "🥉 BEACON-GUARD", href: "/research-wiki/2026-04/moe-fingerprint-security-serving/tier1/03-beacon-guard" },
          { label: "T1 DISCRETE-VEIL-Lite", href: "/research-wiki/2026-04/moe-fingerprint-security-serving/tier2/01-discrete-veil-lite" },
          { label: "T2 TALLY-Spinoff", href: "/research-wiki/2026-04/moe-fingerprint-security-serving/tier2/02-tally-spinoff" },
          { label: "T3 BEACON-GUARD-DATE", href: "/research-wiki/2026-04/moe-fingerprint-security-serving/tier2/03-beacon-guard-date" },
          { label: "미선정 로그", href: "/research-wiki/2026-04/moe-fingerprint-security-serving/unselected" },
        ],
      },
      { label: "2026-04-24 Qwen3-VL DeepStack Edge", href: "/research-wiki/2026-04/qwen3vl-deepstack-edge" },
      { label: "2026-04-23 Energy-Efficient Edge VLM", href: "/research-wiki/2026-04/energy-efficient-edge-vlm" },
      { label: "2026-04-22 VLM+PIM Extension", href: "/research-wiki/2026-04/vlm-pim-extension" },
      { label: "2026-04-21 ACE-MoE VLM/VLA Extension", href: "/research-wiki/2026-04/ace-moe-vlm-vla" },
    ],
  },
];

function isActiveSession(itemHref: string, pathname: string | null): boolean {
  if (!pathname) return false;
  // 활성 조건: pathname 이 itemHref 와 정확히 일치하거나, itemHref + "/" 로 시작
  return pathname === itemHref || pathname.startsWith(itemHref + "/");
}

export default function ResearchWikiSidebar() {
  const [open, setOpen] = useState(false);
  const pathname = usePathname();

  // 라우트 변경 시 자동 닫기 (모바일)
  useEffect(() => {
    setOpen(false);
  }, [pathname]);

  // 사이드바 open 시 body 스크롤 잠금 (모바일)
  useEffect(() => {
    if (open) {
      document.body.style.overflow = "hidden";
      return () => {
        document.body.style.overflow = "";
      };
    }
  }, [open]);

  const sidebarContent = (
    <>
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
            {section.items.map((item) => {
              const active = isActiveSession(item.href, pathname);
              const showChildren = active && item.children && item.children.length > 0;
              return (
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
                  {showChildren && (
                    <ul
                      className="mt-1 ml-3 pl-3 space-y-0.5"
                      style={{
                        borderLeft: "2px solid var(--border, #E5E7EB)",
                      }}
                    >
                      {item.children!.map((child) => {
                        const childActive = pathname === child.href;
                        return (
                          <li key={child.href}>
                            <Link
                              href={child.href}
                              className="block text-xs py-0.5 px-2 rounded hover:underline"
                              style={{
                                color: childActive
                                  ? "var(--accent, #F96167)"
                                  : "var(--text-secondary, #333)",
                                fontWeight: childActive ? 600 : 400,
                                backgroundColor: childActive
                                  ? "rgba(249, 97, 103, 0.08)"
                                  : "transparent",
                              }}
                            >
                              {child.label}
                            </Link>
                          </li>
                        );
                      })}
                    </ul>
                  )}
                </li>
              );
            })}
          </ul>
        </section>
      ))}
      <p
        className="text-[10px] mt-8 italic"
        style={{ color: "var(--text-muted, #8893B8)" }}
      >
        Hidden link — 외부 노출되지 않음 · 사이드바는 모든 research-wiki 페이지에 동일 노출. 활성 세션의 top idea 는 자동 펼침.
      </p>
    </>
  );

  return (
    <>
      {/* Mobile-only hamburger toggle button */}
      <button
        type="button"
        className="md:hidden fixed top-20 left-3 z-40 rounded-md px-3 py-2 text-sm font-medium shadow-md"
        style={{
          backgroundColor: "var(--bg-secondary, #f4f6fb)",
          color: "var(--text-primary, #1E2761)",
          border: "1px solid var(--border)",
        }}
        onClick={() => setOpen((v) => !v)}
        aria-expanded={open}
        aria-controls="research-wiki-sidebar"
      >
        {open ? "✕ Close" : "☰ Wiki Menu"}
      </button>

      {/* Mobile-only backdrop */}
      {open && (
        <div
          className="md:hidden fixed inset-0 z-40 bg-black/40"
          onClick={() => setOpen(false)}
          aria-hidden="true"
        />
      )}

      {/* Desktop sidebar (always visible at md+) */}
      <aside
        className="research-wiki-sidebar shrink-0 border-r hidden md:block"
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
        {sidebarContent}
      </aside>

      {/* Mobile drawer (off-canvas, fixed overlay) */}
      <aside
        id="research-wiki-sidebar"
        className={`md:hidden fixed inset-y-0 left-0 z-50 overflow-y-auto transition-transform ${
          open ? "translate-x-0" : "-translate-x-full"
        }`}
        style={{
          width: "min(85vw, 320px)",
          backgroundColor: "var(--bg-secondary, #f4f6fb)",
          borderRight: "1px solid var(--border)",
          padding: "72px 16px 24px 16px",
        }}
      >
        {sidebarContent}
      </aside>
    </>
  );
}
