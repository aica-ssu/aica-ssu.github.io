import Link from "next/link";
import { Metadata } from "next";

export const metadata: Metadata = {
  title: "Guides - AICA Lab",
};

const guides = [
  {
    title: "AICA Skills for Claude Code",
    description: "Outline Assistant + Writing Assistant 설치 및 사용법, Decision Log 운영 가이드",
    href: "/guides/aica-skills",
    tag: "AI Skills",
  },
  {
    title: "논문 Outline 작성 가이드라인 (v2)",
    description: "요약본 + 클릭하면 펼쳐지는 상세 설명이 통합된 버전",
    href: "/guides/paper-outline-v2",
    tag: "통합",
  },
  {
    title: "논문 Outline 작성 가이드라인 (요약본)",
    description: "핵심 원칙과 Section별 요약을 한눈에 볼 수 있는 요약 버전",
    href: "/guides/paper-outline-summary",
    tag: "요약",
  },
  {
    title: "논문 Outline 작성 가이드라인",
    description: "Section별 Outline 작성법, 자기 검토 체크리스트, Reviewer 관점 검토 등 상세 가이드",
    href: "/guides/paper-outline",
    tag: "상세",
  },
];

export default function GuidesPage() {
  return (
    <div className="max-w-4xl mx-auto px-4 py-16">
      <h1 className="text-2xl font-bold mb-1" style={{ color: "var(--text-primary)" }}>
        Guides
      </h1>
      <p className="text-sm mb-10" style={{ color: "var(--text-muted)" }}>
        연구실 학생을 위한 가이드 문서
      </p>

      <div className="space-y-0 divide-y" style={{ borderColor: "var(--border)" }}>
        {guides.map((guide) => (
          <Link key={guide.href} href={guide.href} className="block group">
            <div className="py-5 flex items-start gap-3" style={{ borderColor: "var(--border)" }}>
              <span
                className="text-[11px] font-medium px-2 py-0.5 mt-0.5 shrink-0 rounded"
                style={{ backgroundColor: "var(--bg-secondary)", color: "var(--text-muted)", border: "1px solid var(--border)" }}
              >
                {guide.tag}
              </span>
              <div>
                <h2
                  className="text-base font-semibold group-hover:underline"
                  style={{ color: "var(--accent)" }}
                >
                  {guide.title}
                </h2>
                <p className="text-sm mt-1" style={{ color: "var(--text-secondary)" }}>
                  {guide.description}
                </p>
              </div>
            </div>
          </Link>
        ))}
      </div>
    </div>
  );
}
