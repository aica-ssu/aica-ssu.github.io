import fs from "fs";
import path from "path";
import Link from "next/link";
import { Metadata } from "next";
import { notFound } from "next/navigation";
import { markdownToHtml } from "@/lib/markdown";
import MermaidScript from "@/components/MermaidScript";

export const metadata: Metadata = {
  title: "Tier-1 Idea — VLM SpecDec + Chunked Prefill Optimization - AICA Lab",
  robots: { index: false, follow: false },
};

const ROUTE_BASE_DIR = "/research-wiki/2026-05/vlm-specdec-chunked-prefill/tier1";

export async function generateStaticParams() {
  return [
    { idea: "01-kl-bounded-distillation" },
    { idea: "02-sub-2bit-visual-kv" },
    { idea: "03-kv-stream-tbc" },
  ];
}

export default async function Tier1IdeaPage({ params }: { params: Promise<{ idea: string }> }) {
  const { idea } = await params;
  const filePath = path.join(
    process.cwd(),
    `src/data/docs/research-wiki/2026-05-vlm-specdec-chunked-prefill/tier1/${idea}.md`
  );
  if (!fs.existsSync(filePath)) notFound();
  const content = fs.readFileSync(filePath, "utf-8");
  const html = await markdownToHtml(content, ROUTE_BASE_DIR);

  return (
    <div className="max-w-4xl mx-auto px-6 py-12">
      <Link
        href="/research-wiki/2026-05/vlm-specdec-chunked-prefill"
        className="text-xs mb-6 inline-block hover:underline"
        style={{ color: "var(--text-muted)" }}
      >
        ← VLM SpecDec + Chunked Prefill Optimization (Landing)
      </Link>
      <article className="md-content" dangerouslySetInnerHTML={{ __html: html }} />
      <MermaidScript />
    </div>
  );
}
