import Link from "next/link";
import { Metadata } from "next";

export const metadata: Metadata = {
  title: "Research Wiki - AICA Lab",
  robots: { index: false, follow: false },
};

type WikiEntry = {
  slug: string;
  title: string;
  description: string;
  date: string;
  tags: string[];
};

const entries: Record<string, WikiEntry[]> = {
  "2026-04": [
    {
      slug: "2026-04/ace-moe-vlm-vla",
      title: "ACE-MoE 확장: VLM/VLA Software 연구 주제 Top 3",
      description:
        "ACE-MoE(ICCAD'26)을 VLM/VLA로 확장하는 3개 연구 주제 — ACE-VLA(real-time), Joint Token-Expert Budget, Modality-Aware ACE. arxiv 60+편 재검색 기반 positioning 포함.",
      date: "2026-04-21",
      tags: ["Mode 2", "MoE", "VLM", "VLA"],
    },
  ],
};

export default function ResearchWikiIndexPage() {
  const months = Object.keys(entries).sort().reverse();

  return (
    <div className="max-w-4xl mx-auto px-4 py-16">
      <h1
        className="text-2xl font-bold mb-1"
        style={{ color: "var(--text-primary)" }}
      >
        Research Wiki
      </h1>
      <p className="text-sm mb-2" style={{ color: "var(--text-muted)" }}>
        연구실 내부 공유용 ideation 노트. 네비게이션 비노출(hidden link).
      </p>
      <p className="text-xs mb-10" style={{ color: "var(--text-muted)" }}>
        aica-research-bot harness로 도출된 연구 주제 및 관련연구 정리.
      </p>

      <div className="space-y-12">
        {months.map((month) => (
          <section key={month}>
            <h2
              className="text-sm font-semibold mb-3 tracking-wide uppercase"
              style={{ color: "var(--text-muted)" }}
            >
              {month}
            </h2>
            <div
              className="space-y-0 divide-y"
              style={{ borderColor: "var(--border)" }}
            >
              {entries[month].map((entry) => (
                <Link
                  key={entry.slug}
                  href={`/research-wiki/${entry.slug}`}
                  className="block group"
                >
                  <div
                    className="py-5 flex items-start gap-3"
                    style={{ borderColor: "var(--border)" }}
                  >
                    <div className="flex flex-col gap-1 shrink-0 min-w-[82px]">
                      {entry.tags.map((tag) => (
                        <span
                          key={tag}
                          className="text-[10px] font-medium px-2 py-0.5 rounded inline-block text-center"
                          style={{
                            backgroundColor: "var(--bg-secondary)",
                            color: "var(--text-muted)",
                            border: "1px solid var(--border)",
                          }}
                        >
                          {tag}
                        </span>
                      ))}
                    </div>
                    <div>
                      <h3
                        className="text-base font-semibold group-hover:underline"
                        style={{ color: "var(--accent)" }}
                      >
                        {entry.title}
                      </h3>
                      <p
                        className="text-sm mt-1"
                        style={{ color: "var(--text-secondary)" }}
                      >
                        {entry.description}
                      </p>
                      <p
                        className="text-xs mt-2"
                        style={{ color: "var(--text-muted)" }}
                      >
                        {entry.date}
                      </p>
                    </div>
                  </div>
                </Link>
              ))}
            </div>
          </section>
        ))}
      </div>
    </div>
  );
}
