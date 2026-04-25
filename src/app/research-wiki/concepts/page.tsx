import fs from "fs";
import path from "path";
import Link from "next/link";
import { Metadata } from "next";
import { markdownToHtml } from "@/lib/markdown";
import MermaidScript from "@/components/MermaidScript";

export const metadata: Metadata = {
  title: "concepts — Research Wiki - AICA Lab",
  robots: { index: false, follow: false },
};

const ROUTE_BASE_DIR = "/research-wiki";

export default async function Page() {
  const filePath = path.join(
    process.cwd(),
    "src/data/docs/research-wiki/concepts.md"
  );
  const content = fs.readFileSync(filePath, "utf-8");
  const html = await markdownToHtml(content, ROUTE_BASE_DIR);

  return (
    <div className="max-w-5xl mx-auto px-6 py-12">
      <Link
        href="/research-wiki"
        className="text-xs mb-6 inline-block hover:underline"
        style={{ color: "var(--text-muted)" }}
      >
        ← Research Wiki
      </Link>
      <article className="md-content" dangerouslySetInnerHTML={{ __html: html }} />
      <MermaidScript />
    </div>
  );
}
