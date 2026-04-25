#!/usr/bin/env node
/**
 * Research-wiki 모든 markdown link 검증.
 *
 * 검증 대상:
 *   - src/data/docs/research-wiki/**\/*.md 안의 모든 [text](url) 링크
 *
 * 검증 규칙:
 *   - 외부 URL (http/https/mailto) → skip
 *   - anchor only (#foo)         → skip
 *   - .md 상대 링크              → rewriteMdLinks 와 동일 로직으로 절대 경로로 변환,
 *                                   그 경로에 해당하는 src/app/research-wiki/.../page.tsx
 *                                   라우트가 존재하는지 확인
 *   - .md 절대 링크 (/research-wiki/...) → 해당 라우트 page.tsx 존재 확인
 *
 * Sidebar (`ResearchWikiSidebar.tsx`) 에 hard-coded href 도 함께 검증 (옵션).
 *
 * 실패 시 process.exit(1).
 */
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = path.resolve(__dirname, "..");
const DATA_ROOT = path.join(REPO_ROOT, "src/data/docs/research-wiki");
const APP_ROOT = path.join(REPO_ROOT, "src/app/research-wiki");

function* walk(dir) {
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    if (entry.name.startsWith("_")) continue; // skip _archive/, _draft/, etc.
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) yield* walk(full);
    else if (entry.isFile() && entry.name.endsWith(".md")) yield full;
  }
}

/**
 * 데이터 파일 경로 (src/data/docs/research-wiki/...) → published route base 경로.
 *
 * 매핑 규칙:
 *   src/data/docs/research-wiki/index.md
 *      → /research-wiki/index
 *   src/data/docs/research-wiki/papers.md
 *      → /research-wiki/papers
 *   src/data/docs/research-wiki/2026-04-foo.md
 *      → /research-wiki/2026-04/foo                       (single-file)
 *   src/data/docs/research-wiki/2026-04-foo/README.md
 *      → /research-wiki/2026-04/foo                       (hierarchical landing)
 *   src/data/docs/research-wiki/2026-04-foo/tier1/01-bar.md
 *      → /research-wiki/2026-04/foo/tier1/01-bar
 *   src/data/docs/research-wiki/2026-04-foo/unselected.md
 *      → /research-wiki/2026-04/foo/unselected
 *
 * linkBaseUrl 은 해당 markdown 의 directory 에 대응하는 route 의 디렉토리.
 */
function dataPathToRouteBase(absDataPath) {
  const rel = path.relative(DATA_ROOT, absDataPath); // e.g. "2026-04-foo/tier1/01-bar.md"
  const segments = rel.split(path.sep);
  const fileName = segments.pop();
  const basename = fileName.replace(/\.md$/, "");

  // top-level files: index.md, papers.md, ideas.md, concepts.md, trends.md
  if (segments.length === 0) {
    if (basename.toLowerCase() === "readme") {
      return { routePath: "/research-wiki", routeDir: "/research-wiki" };
    }
    // session single-file: 2026-04-foo.md → /research-wiki/2026-04/foo
    const m = /^(\d{4}-\d{2})-(.+)$/.exec(basename);
    if (m) {
      const route = `/research-wiki/${m[1]}/${m[2]}`;
      return { routePath: route, routeDir: `/research-wiki/${m[1]}` };
    }
    return {
      routePath: `/research-wiki/${basename}`,
      routeDir: "/research-wiki",
    };
  }

  // 1st segment: "2026-04-foo" → "2026-04/foo"
  const top = segments[0];
  const m = /^(\d{4}-\d{2})-(.+)$/.exec(top);
  if (!m) return { routePath: null, routeDir: null };

  const subSegs = segments.slice(1); // tier1, tier2, ...
  const routeDirParts = ["/research-wiki", m[1], m[2], ...subSegs].join("/");
  const routePath =
    basename.toLowerCase() === "readme"
      ? routeDirParts
      : `${routeDirParts}/${basename}`;
  return { routePath, routeDir: routeDirParts };
}

/**
 * Same logic as rewriteMdLinks in src/lib/markdown.ts.
 * Resolve relative .md link against linkBaseUrl.
 */
function resolveMdLink(url, linkBaseUrl) {
  if (!linkBaseUrl) return null;
  const base = linkBaseUrl.replace(/\/$/, "");
  if (/^[a-z]+:\/\//i.test(url) || url.startsWith("/") || url.startsWith("#") || url.startsWith("mailto:")) {
    return null; // skip
  }
  const [pathPart, hashPart] = url.split("#");
  if (!/\.md(?:$|[?#])/.test(pathPart)) return null;

  const segments = pathPart.split("/");
  const baseSegments = base.split("/").filter(Boolean);
  const resolved = [...baseSegments];
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
  if (last.toLowerCase() === "readme") last = "";
  let newUrl = "/" + resolved.concat(last ? [last] : []).filter(Boolean).join("/");
  if (newUrl === "//") newUrl = "/";
  return hashPart ? `${newUrl}#${hashPart}` : newUrl;
}

/**
 * Check if a /research-wiki/... URL has a corresponding Next.js route.
 *
 * Strategy: convert URL → relative path inside src/app, then look for page.tsx
 * (allowing dynamic segments [foo]).
 */
function routeExists(url) {
  // Strip hash fragment for filesystem check
  const noHash = url.split("#")[0];
  if (!noHash.startsWith("/research-wiki")) return false;

  const rel = noHash.slice("/research-wiki".length).replace(/^\/|\/$/g, "");
  const segments = rel.split("/").filter(Boolean);

  // Walk segments matching either literal dir or [param] dir
  let cur = APP_ROOT;
  for (const seg of segments) {
    const literal = path.join(cur, seg);
    if (fs.existsSync(literal) && fs.statSync(literal).isDirectory()) {
      cur = literal;
      continue;
    }
    // Try dynamic [param] dir
    let matched = null;
    if (fs.existsSync(cur)) {
      for (const e of fs.readdirSync(cur)) {
        if (/^\[.+\]$/.test(e)) {
          const dyn = path.join(cur, e);
          if (fs.statSync(dyn).isDirectory()) {
            matched = dyn;
            break;
          }
        }
      }
    }
    if (matched) {
      cur = matched;
      continue;
    }
    return false;
  }
  return fs.existsSync(path.join(cur, "page.tsx"));
}

const LINK_RE = /\[([^\]]*)\]\(([^)]+)\)/g;

let totalLinks = 0;
let mdLinks = 0;
let absoluteLinks = 0;
const errors = [];

for (const file of walk(DATA_ROOT)) {
  const { routeDir } = dataPathToRouteBase(file);
  if (!routeDir) continue;
  const content = fs.readFileSync(file, "utf-8");
  const lines = content.split("\n");
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    LINK_RE.lastIndex = 0;
    let m;
    while ((m = LINK_RE.exec(line)) !== null) {
      const text = m[1];
      const url = m[2];
      totalLinks++;

      // Skip external + anchors
      if (/^[a-z]+:\/\//i.test(url) || url.startsWith("#") || url.startsWith("mailto:")) {
        continue;
      }

      // Absolute /research-wiki/... → must have route
      if (url.startsWith("/research-wiki")) {
        absoluteLinks++;
        // Strip .md from absolute path (markdown.ts auto-rewrites at render time).
        let routeUrl = url.split("#")[0];
        if (routeUrl.endsWith(".md")) {
          routeUrl = routeUrl.replace(/\.md$/, "");
          if (routeUrl.toLowerCase().endsWith("/readme")) {
            routeUrl = routeUrl.slice(0, -"/readme".length);
          }
        }
        if (!routeExists(routeUrl)) {
          errors.push({
            file: path.relative(REPO_ROOT, file),
            line: i + 1,
            text,
            url,
            kind: "absolute-route-missing",
          });
        }
        continue;
      }

      // Other absolute URLs (e.g., /img/foo.png) — skip
      if (url.startsWith("/")) continue;

      // Relative .md link
      if (/\.md(?:$|[#?])/.test(url)) {
        mdLinks++;
        const resolved = resolveMdLink(url, routeDir);
        if (!resolved) continue;
        if (!routeExists(resolved)) {
          errors.push({
            file: path.relative(REPO_ROOT, file),
            line: i + 1,
            text,
            url,
            resolved,
            kind: "relative-md-route-missing",
          });
        }
      }
    }
  }
}

// Additional check (R44): no blank lines inside <svg> blocks (CommonMark splits HTML blocks at blank lines).
for (const file of walk(DATA_ROOT)) {
  const content = fs.readFileSync(file, "utf-8");
  const lines = content.split("\n");
  let inSvg = false;
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    if (/^<svg /.test(line)) inSvg = true;
    else if (/^<\/svg>/.test(line)) inSvg = false;
    else if (inSvg && /^\s*$/.test(line)) {
      errors.push({
        file: path.relative(REPO_ROOT, file),
        line: i + 1,
        text: "(blank)",
        url: "",
        kind: "blank-line-inside-svg-r44",
      });
    }
  }
}

console.log(`Scanned: ${totalLinks} total links, ${mdLinks} relative .md links, ${absoluteLinks} absolute /research-wiki links`);
if (errors.length === 0) {
  console.log("OK: all research-wiki links resolve to existing routes.");
  process.exit(0);
}

console.error(`\nFAIL: ${errors.length} broken link(s):\n`);
for (const e of errors) {
  console.error(`  ${e.file}:${e.line}  [${e.text}](${e.url})`);
  if (e.resolved) console.error(`      → resolved to: ${e.resolved} (route NOT found)`);
  else console.error(`      → kind: ${e.kind}`);
}
process.exit(1);
