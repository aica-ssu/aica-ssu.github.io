import fs from "fs";
import path from "path";
import Link from "next/link";
import { Metadata } from "next";
import { markdownToHtml } from "@/lib/markdown";
import MermaidScript from "@/components/MermaidScript";

export const metadata: Metadata = {
  title: "Unselected Ideas — MoE Fingerprint Security + Serving - AICA Lab",
  robots: { index: false, follow: false },
};

const ROUTE_BASE_DIR =
  "/research-wiki/2026-04/moe-fingerprint-security-serving";

export default async function UnselectedPage() {
  const filePath = path.join(
    process.cwd(),
    "src/data/docs/research-wiki/2026-04-moe-fingerprint-security-serving/unselected.md"
  );
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
