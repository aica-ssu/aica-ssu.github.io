import fs from "fs";
import path from "path";
import { Metadata } from "next";
import { markdownToHtml } from "@/lib/markdown";

export const metadata: Metadata = {
  title: "논문 Outline 작성 가이드라인 (요약) - AICA Lab",
};

export default async function PaperOutlineSummaryPage() {
  const filePath = path.join(process.cwd(), "src/data/docs/paper-outline-summary.md");
  const content = fs.readFileSync(filePath, "utf-8");
  const html = await markdownToHtml(content);

  return (
    <div className="max-w-4xl mx-auto px-4 py-16">
      <div className="mb-8 rounded-lg border border-[var(--accent)] bg-[color-mix(in_srgb,var(--accent)_8%,transparent)] px-5 py-4">
        <p className="m-0 text-sm leading-relaxed" style={{ color: "var(--text-primary)" }}>
          <strong>Updated version available</strong> — 이 가이드의 통합 개정판(v2)이 있습니다.
          요약 + 상세를 한 페이지에서 확인할 수 있습니다.{" "}
          <a href="/guides/paper-outline-v2" className="font-semibold underline">
            v2 가이드라인으로 이동 →
          </a>
        </p>
      </div>
      <article
        className="md-content"
        dangerouslySetInnerHTML={{ __html: html }}
      />
    </div>
  );
}
