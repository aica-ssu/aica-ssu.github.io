import fs from "fs";
import path from "path";
import { Metadata } from "next";
import { markdownToHtml } from "@/lib/markdown";

export const metadata: Metadata = {
  title: "AICA Skills for Claude Code - AICA Lab",
};

export default async function AicaSkillsPage() {
  const filePath = path.join(process.cwd(), "src/data/docs/aica-skills.md");
  const content = fs.readFileSync(filePath, "utf-8");
  const html = await markdownToHtml(content);

  return (
    <div className="max-w-4xl mx-auto px-4 py-16">
      <article
        className="md-content"
        dangerouslySetInnerHTML={{ __html: html }}
      />
    </div>
  );
}
