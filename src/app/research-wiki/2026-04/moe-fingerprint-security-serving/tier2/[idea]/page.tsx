import fs from "fs";
import path from "path";
import Link from "next/link";
import { Metadata } from "next";
import { notFound } from "next/navigation";
import { markdownToHtml } from "@/lib/markdown";
import MermaidScript from "@/components/MermaidScript";

export const metadata: Metadata = {
  title: "Tier-2 Idea — MoE Fingerprint Security + Serving - AICA Lab",
  robots: { index: false, follow: false },
};

const ROUTE_BASE_DIR =
  "/research-wiki/2026-04/moe-fingerprint-security-serving/tier2";

export async function generateStaticParams() {
  return [
    { idea: "01-discrete-veil-lite" },
    { idea: "02-tally-spinoff" },
    { idea: "03-beacon-guard-date" },
  ];
}

export default async function Tier2IdeaPage({
  params,
}: {
  params: Promise<{ idea: string }>;
}) {
  const { idea } = await params;
  const filePath = path.join(
    process.cwd(),
    `src/data/docs/research-wiki/2026-04-moe-fingerprint-security-serving/tier2/${idea}.md`
  );
  if (!fs.existsSync(filePath)) {
    notFound();
  }
  const content = fs.readFileSync(filePath, "utf-8");
  const html = await markdownToHtml(content, ROUTE_BASE_DIR);

  return (
    <div className="max-w-4xl mx-auto px-4 py-16">
      <Link
        href="/research-wiki/2026-04/moe-fingerprint-security-serving"
        className="text-xs mb-6 inline-block hover:underline"
        style={{ color: "var(--text-muted)" }}
      >
        ← MoE Fingerprint Security + Serving (Session Overview)
      </Link>
      <article
        className="md-content"
        dangerouslySetInnerHTML={{ __html: html }}
      />
      <MermaidScript />
    </div>
  );
}
