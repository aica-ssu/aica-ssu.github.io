import fs from "fs";
import path from "path";
import Link from "next/link";
import type { Metadata } from "next";
import { notFound } from "next/navigation";
import { markdownToHtml } from "@/lib/markdown";
import MermaidScript from "@/components/MermaidScript";

const IDEAS = ["01-s2", "02-a2"];
export const metadata: Metadata = { title: "Tier-1 Generation Caching Idea — AICA Lab", robots: { index: false, follow: false } };
export function generateStaticParams() { return IDEAS.map((idea) => ({ idea })); }

export default async function GenerationCacheTier1Page({ params }: { params: Promise<{ idea: string }> }) {
  const { idea } = await params;
  if (!IDEAS.includes(idea)) notFound();
  const content = fs.readFileSync(path.join(process.cwd(), `src/data/docs/research-wiki/2026-09-generation-cache/tier1/${idea}.md`), "utf-8");
  const html = await markdownToHtml(content, `/research-wiki/2026-09/generation-cache/tier1`);
  return <div className="max-w-4xl mx-auto px-6 py-12"><Link href="/research-wiki/2026-09/generation-cache" className="text-xs mb-6 inline-block hover:underline" style={{ color: "var(--text-muted)" }}>← Generation Caching</Link><article className="md-content" dangerouslySetInnerHTML={{ __html: html }} /><MermaidScript /></div>;
}
