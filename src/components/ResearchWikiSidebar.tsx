"use client";

import Link from "next/link";
import { useState, useEffect } from "react";
import { usePathname } from "next/navigation";

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
    title: "⭐ 최근 세션 (2026-04-25 #2, KV cache ECC + RAS)",
    items: [
      {
        label: "Landing — KV Cache ECC + RAS",
        href: "/research-wiki/2026-04/kv-ecc-ras",
        emphasis: true,
      },
      {
        label: "🥇 OAEP-KV (DSN/HPCA/MICRO 2027) lead",
        href: "/research-wiki/2026-04/kv-ecc-ras/tier1/01-oaep-kv",
      },
      {
        label: "🥈 BlockShard (ASPLOS/OSDI 2027)",
        href: "/research-wiki/2026-04/kv-ecc-ras/tier1/02-blockshard",
      },
      {
        label: "🥉 LayerTier (MICRO/DSN 2027)",
        href: "/research-wiki/2026-04/kv-ecc-ras/tier1/03-layertier",
      },
      {
        label: "T1 VLM-MAP (DATE/IEEE TCAD)",
        href: "/research-wiki/2026-04/kv-ecc-ras/tier2/01-vlm-map",
      },
      {
        label: "T2 FrostFloor (DATE 6p)",
        href: "/research-wiki/2026-04/kv-ecc-ras/tier2/02-frostfloor",
      },
      {
        label: "T3 EntropyECC (ITC/MTS/IEEE TCAD)",
        href: "/research-wiki/2026-04/kv-ecc-ras/tier2/03-entropy-ecc",
      },
      {
        label: "미선정 15편 로그",
        href: "/research-wiki/2026-04/kv-ecc-ras/unselected",
      },
    ],
  },
  {
    title: "📚 이전 Summary",
    items: [
      { label: "2026-04-25 VLM Context-aware Jetson Edge (R45 적용)", href: "/research-wiki/2026-04/vlm-context-edge-jetson" },
      { label: "2026-04-24 MoE Fingerprint Security+Serving", href: "/research-wiki/2026-04/moe-fingerprint-security-serving" },
      { label: "2026-04-24 Qwen3-VL DeepStack Edge", href: "/research-wiki/2026-04/qwen3vl-deepstack-edge" },
      { label: "2026-04-23 Energy-Efficient Edge VLM", href: "/research-wiki/2026-04/energy-efficient-edge-vlm" },
      { label: "2026-04-22 VLM+PIM Extension", href: "/research-wiki/2026-04/vlm-pim-extension" },
      { label: "2026-04-21 ACE-MoE VLM/VLA Extension", href: "/research-wiki/2026-04/ace-moe-vlm-vla" },
    ],
  },
];

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
