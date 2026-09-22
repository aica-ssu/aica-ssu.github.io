import fs from "fs";
import path from "path";
import Link from "next/link";
import type { Metadata } from "next";
import { markdownToHtml } from "@/lib/markdown";
import MermaidScript from "@/components/MermaidScript";

export const metadata: Metadata = {
  title: "Generation Caching Research — AICA Lab",
  robots: { index: false, follow: false },
};

const ROOT = "/research-wiki/2026-09/generation-cache";

export default async function GenerationCacheLandingPage() {
  const content = fs.readFileSync(
    path.join(process.cwd(), "src/data/docs/research-wiki/2026-09-generation-cache/README.md"),
    "utf-8"
  );
  const html = await markdownToHtml(content, ROOT);
  return (
    <div className="max-w-4xl mx-auto px-6 py-12">
      <Link href="/research-wiki" className="text-xs mb-6 inline-block hover:underline" style={{ color: "var(--text-muted)" }}>
        ← Research Wiki
      </Link>
      <article className="md-content" dangerouslySetInnerHTML={{ __html: html }} />
      <MermaidScript />
    </div>
  );
}
