import fs from "fs";
import path from "path";
import Link from "next/link";
import { Metadata } from "next";
import { markdownToHtml } from "@/lib/markdown";
import MermaidScript from "@/components/MermaidScript";

export const metadata: Metadata = {
  title: "최신 VLM 특성 기반 에너지 효율적 Edge VLM Inference (Parquet / Triptych / Cartographer / Sift / Verge) - AICA Lab",
  robots: { index: false, follow: false },
};

export default async function EnergyEfficientEdgeVlmPage() {
  const filePath = path.join(
    process.cwd(),
    "src/data/docs/research-wiki/2026-04-energy-efficient-edge-vlm.md"
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
