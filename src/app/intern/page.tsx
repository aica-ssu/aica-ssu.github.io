import fs from "fs";
import path from "path";
import { Metadata } from "next";
import { markdownToHtml } from "@/lib/markdown";

export const metadata: Metadata = {
  title: "인턴 지원자 학습 가이드 - AICA Lab",
};

export default async function InternGuidePage() {
  const filePath = path.join(process.cwd(), "src/data/docs/intern-guide.md");
  const content = fs.readFileSync(filePath, "utf-8");
  const html = await markdownToHtml(content);

  return (
    <div className="max-w-4xl mx-auto px-4 py-16">
      <article
        className="md-content"
        dangerouslySetInnerHTML={{ __html: html }}
      />
    </div>
  );
}
