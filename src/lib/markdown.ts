import { unified } from "unified";
import remarkParse from "remark-parse";
import remarkGfm from "remark-gfm";
import remarkRehype from "remark-rehype";
import rehypeRaw from "rehype-raw";
import rehypeSlug from "rehype-slug";
import rehypeStringify from "rehype-stringify";
import { visit } from "unist-util-visit";
import type { Root as MdastRoot } from "mdast";
import type { Root as HastRoot } from "hast";

/**
 * Convert relative .md links and mermaid code blocks to homepage-friendly forms.
 *
 * - linkBaseUrl: absolute URL prefix corresponding to the current markdown
 *   file's directory in the published homepage. Example: for the file
 *   `src/data/docs/research-wiki/2026-04-foo/tier1/01-bar.md` rendered at
 *   `/research-wiki/2026-04/foo/tier1/01-bar`, pass linkBaseUrl =
 *   `/research-wiki/2026-04/foo/tier1`.
 *
 * Behavior:
 * - `[txt](file.md)` → `[txt](linkBaseUrl/file)` (extension stripped, route)
 * - `[txt](sub/file.md)` → `[txt](linkBaseUrl/sub/file)`
 * - `[txt](../README.md)` → `[txt](parentOfBase)` (one level up, README → "")
 * - `[txt](../../foo/bar.md)` → resolved relative to linkBaseUrl
 * - Absolute URLs (http://, https://, /...) and anchors (#) untouched
 * - ```mermaid ... ``` code blocks → `<pre class="mermaid">...</pre>` raw HTML
 *   so client-side mermaid script can render them.
 */
export async function markdownToHtml(md: string, linkBaseUrl?: string) {
  const result = await unified()
    .use(remarkParse)
    .use(remarkGfm)
    .use(rewriteMdLinks, { baseUrl: linkBaseUrl })
    .use(mermaidToHtml)
    .use(remarkRehype, { allowDangerousHtml: true })
    .use(rehypeRaw)
    .use(rehypeSlug)
    .use(rehypeStringify)
    .process(md);
  return String(result);
}

function rewriteMdLinks({ baseUrl }: { baseUrl?: string } = {}) {
  return (tree: MdastRoot) => {
    visit(tree, "link", (node) => {
      const url = node.url;
      if (!url) return;
      if (/^[a-z]+:\/\//i.test(url) || url.startsWith("#") || url.startsWith("mailto:")) {
        return;
      }
      const [pathPart, hashPart] = url.split("#");
      if (!/\.md(?:$|[?#])/.test(pathPart)) {
        return;
      }

      // (a) Absolute path ending in .md — strip extension + map README → dir.
      if (pathPart.startsWith("/")) {
        const segs = pathPart.split("/").filter(Boolean);
        let last = segs.pop() ?? "";
        last = last.replace(/\.md$/, "");
        if (last.toLowerCase() === "readme") last = "";
        let newUrl = "/" + segs.concat(last ? [last] : []).join("/");
        if (newUrl === "//") newUrl = "/";
        if (hashPart) newUrl += "#" + hashPart;
        node.url = newUrl;
        return;
      }

      // (b) Relative .md link — needs baseUrl to resolve.
      if (!baseUrl) return;
      const base = baseUrl.replace(/\/$/, "");
      const segments = pathPart.split("/");
      const baseSegments = base.split("/").filter(Boolean);
      const resolved: string[] = [...baseSegments];
      for (const seg of segments) {
        if (seg === "" || seg === ".") continue;
        if (seg === "..") {
          resolved.pop();
        } else {
          resolved.push(seg);
        }
      }
      let last = resolved.pop() ?? "";
      last = last.replace(/\.md$/, "");
      if (last.toLowerCase() === "readme") {
        last = "";
      }
      let newUrl = "/" + resolved.concat(last ? [last] : []).filter(Boolean).join("/");
      if (newUrl === "//") newUrl = "/";
      if (hashPart) newUrl += "#" + hashPart;
      node.url = newUrl;
    });
  };
}

function mermaidToHtml() {
  return (tree: MdastRoot) => {
    visit(tree, "code", (node) => {
      if (node.lang !== "mermaid") return;
      // Convert to a raw HTML node consumed by remark-rehype + rehype-raw
      const html = `<pre class="mermaid">${escapeHtml(node.value)}</pre>`;
      (node as unknown as { type: string; value: string }).type = "html";
      (node as unknown as { type: string; value: string }).value = html;
    });
  };
}

function escapeHtml(s: string): string {
  return s
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

// Re-export for type-checker (no runtime use)
export type { HastRoot };
