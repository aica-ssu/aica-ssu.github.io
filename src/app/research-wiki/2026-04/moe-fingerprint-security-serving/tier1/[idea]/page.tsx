import fs from "fs";
import path from "path";
import Link from "next/link";
import { Metadata } from "next";
import { notFound } from "next/navigation";
import { markdownToHtml } from "@/lib/markdown";

export const metadata: Metadata = {
  title: "Tier-1 Idea — MoE Fingerprint Security + Serving - AICA Lab",
  robots: { index: false, follow: false },
};

export async function generateStaticParams() {
  return [
    { idea: "01-discrete-veil" },
    { idea: "02-loom" },
    { idea: "03-beacon-guard" },
  ];
}

export default async function Tier1IdeaPage({
  params,
}: {
  params: Promise<{ idea: string }>;
}) {
  const { idea } = await params;
  const filePath = path.join(
    process.cwd(),
    `src/data/docs/research-wiki/2026-04-moe-fingerprint-security-serving/tier1/${idea}.md`
  );
  if (!fs.existsSync(filePath)) {
    notFound();
  }
  const content = fs.readFileSync(filePath, "utf-8");
  const html = await markdownToHtml(content);

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
    </div>
  );
}
