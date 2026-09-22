import fs from "fs";
import path from "path";
import Link from "next/link";
import type { Metadata } from "next";
import { notFound } from "next/navigation";
import { markdownToHtml } from "@/lib/markdown";
import MermaidScript from "@/components/MermaidScript";

const DOCS = ["student-foundations", "decision-report", "scoring-calibration", "literature-map"];
export const metadata: Metadata = { title: "Generation Caching Notes — AICA Lab", robots: { index: false, follow: false } };

export function generateStaticParams() { return DOCS.map((doc) => ({ doc })); }

export default async function GenerationCacheDocPage({ params }: { params: Promise<{ doc: string }> }) {
  const { doc } = await params;
  if (!DOCS.includes(doc)) notFound();
  const content = fs.readFileSync(path.join(process.cwd(), `src/data/docs/research-wiki/2026-09-generation-cache/${doc}.md`), "utf-8");
  const html = await markdownToHtml(content, `/research-wiki/2026-09/generation-cache/${doc}`);
  return <div className="max-w-4xl mx-auto px-6 py-12"><Link href="/research-wiki/2026-09/generation-cache" className="text-xs mb-6 inline-block hover:underline" style={{ color: "var(--text-muted)" }}>← Generation Caching</Link><article className="md-content" dangerouslySetInnerHTML={{ __html: html }} /><MermaidScript /></div>;
}
