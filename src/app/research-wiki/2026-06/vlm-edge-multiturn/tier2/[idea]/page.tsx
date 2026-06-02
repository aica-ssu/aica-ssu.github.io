import fs from "fs";
import path from "path";
import Link from "next/link";
import { Metadata } from "next";
import { notFound } from "next/navigation";
import { markdownToHtml } from "@/lib/markdown";
import MermaidScript from "@/components/MermaidScript";

export const metadata: Metadata = {
  title: "Tier-2 Idea — VLM Edge Multi-Turn / Multi-Focus Serving - AICA Lab",
  robots: { index: false, follow: false },
};

const ROUTE_BASE_DIR = "/research-wiki/2026-06/vlm-edge-multiturn/tier2";

export async function generateStaticParams() {
  return [
    { idea: "04-focus-coverage" },
    { idea: "05-stream-recover" },
    { idea: "06-vkv-skip" },
  ];
}

export default async function Tier2IdeaPage({ params }: { params: Promise<{ idea: string }> }) {
  const { idea } = await params;
  const filePath = path.join(
    process.cwd(),
    `src/data/docs/research-wiki/2026-06-vlm-edge-multiturn/tier2/${idea}.md`
  );
  if (!fs.existsSync(filePath)) notFound();
  const content = fs.readFileSync(filePath, "utf-8");
  const html = await markdownToHtml(content, ROUTE_BASE_DIR);

  return (
    <div className="max-w-4xl mx-auto px-6 py-12">
      <Link
        href="/research-wiki/2026-06/vlm-edge-multiturn"
        className="text-xs mb-6 inline-block hover:underline"
        style={{ color: "var(--text-muted)" }}
      >
        ← VLM Edge Multi-Turn / Multi-Focus Serving (Landing)
      </Link>
      <article className="md-content" dangerouslySetInnerHTML={{ __html: html }} />
      <MermaidScript />
    </div>
  );
}
