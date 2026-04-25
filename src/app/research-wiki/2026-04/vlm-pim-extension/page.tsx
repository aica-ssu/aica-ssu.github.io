import fs from "fs";
import path from "path";
import Link from "next/link";
import { Metadata } from "next";
import { markdownToHtml } from "@/lib/markdown";
import MermaidScript from "@/components/MermaidScript";

export const metadata: Metadata = {
  title: "VLM+PIM Heterogeneous KV Management Extension Ideas - AICA Lab",
  robots: { index: false, follow: false },
};

export default async function VlmPimExtensionPage() {
  const filePath = path.join(
    process.cwd(),
    "src/data/docs/research-wiki/2026-04-vlm-pim-extension.md"
  );
  const content = fs.readFileSync(filePath, "utf-8");
  const html = await markdownToHtml(content);

  return (
    <div className="max-w-4xl mx-auto px-4 py-16">
      <Link
        href="/research-wiki"
        className="text-xs mb-6 inline-block hover:underline"
        style={{ color: "var(--text-muted)" }}
      >
        ← Research Wiki
      </Link>
      <article
        className="md-content"
        dangerouslySetInnerHTML={{ __html: html }}
      />
      <MermaidScript />
    </div>
  );
}
