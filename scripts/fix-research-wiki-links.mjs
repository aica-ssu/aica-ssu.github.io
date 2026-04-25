#!/usr/bin/env node
/**
 * Research-wiki broken-link 자동 보정.
 *
 * 보정 규칙 (validation 스크립트가 발견한 broken 링크 처리):
 *
 *   (a) `[text](sessions/foo.md)` 또는 `[text](../sessions/foo.md)`
 *       → 동일 세션 published landing route 가 있으면 absolute URL 로 교체.
 *         예) sessions/2026-04-23-mode1-energy-efficient-edge-vlm.md
 *             → /research-wiki/2026-04/energy-efficient-edge-vlm
 *       → 없으면 link wrapper 제거, plain text 만 유지.
 *
 *   (b) `[text](../sessions/staging/...md)` (staging) → text 만
 *
 *   (c) `[text](.claude/skills/...)` 또는 `[text](../../.claude/...)` → text 만
 *
 *   (d) `[text](other-old-session.md)` 같은 unpublished 직접 ref → text 만
 *
 *   (e) 같은 모듈 안 published 세션 cross-ref (예: `../2026-04-foo.md`)
 *       → absolute /research-wiki/2026-04/foo URL 로 교체
 *
 *   (f) `[text](summary/2026-04-foo.md)` (timeline 의 summary 컬럼)
 *       → 해당 published landing 으로 교체
 *
 * In-place 수정. dry-run 모드 (--dry) 지원.
 */
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = path.resolve(__dirname, "..");
const DATA_ROOT = path.join(REPO_ROOT, "src/data/docs/research-wiki");
const APP_ROOT = path.join(REPO_ROOT, "src/app/research-wiki");
const DRY = process.argv.includes("--dry");

function* walk(dir) {
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    if (entry.name.startsWith("_")) continue; // skip _archive/, etc.
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) yield* walk(full);
    else if (entry.isFile() && entry.name.endsWith(".md")) yield full;
  }
}

/**
 * 세션 slug → published route 매핑 (data dir 스캔으로 자동 빌드).
 *
 *   slug = "energy-efficient-edge-vlm" → "/research-wiki/2026-04/energy-efficient-edge-vlm"
 *   슬러그가 같은 published route 가 없으면 매핑 없음.
 *
 * Data dir scan:
 *   - 2026-04-foo.md (single-file) → published route /research-wiki/2026-04/foo
 *   - 2026-04-foo/README.md (hierarchical) → /research-wiki/2026-04/foo
 */
function buildSlugMap() {
  const map = new Map();
  for (const entry of fs.readdirSync(DATA_ROOT, { withFileTypes: true })) {
    if (entry.isFile() && entry.name.endsWith(".md")) {
      const m = /^(\d{4}-\d{2})-(.+)\.md$/.exec(entry.name);
      if (m) map.set(m[2], `/research-wiki/${m[1]}/${m[2]}`);
    } else if (entry.isDirectory()) {
      const m = /^(\d{4}-\d{2})-(.+)$/.exec(entry.name);
      if (!m) continue;
      const readme = path.join(DATA_ROOT, entry.name, "README.md");
      if (fs.existsSync(readme))
        map.set(m[2], `/research-wiki/${m[1]}/${m[2]}`);
    }
  }
  return map;
}

const slugMap = buildSlugMap();
console.log("Published session routes:");
for (const [slug, route] of slugMap) console.log(`  ${slug.padEnd(40)} → ${route}`);

/**
 * 링크 url 을 분석해 보정된 (newUrl, newText) 반환.
 *
 *   - 보정 가능 → { url: newUrl } (link 유지)
 *   - link 자체 제거 (text 만) → { stripLink: true }
 *   - 보정 불필요 → null
 */
function rewrite(url) {
  // Skip externals, anchors, mailto, absolute /research-wiki
  if (/^[a-z]+:\/\//i.test(url) || url.startsWith("#") || url.startsWith("mailto:")) return null;
  if (url.startsWith("/research-wiki/") || url === "/research-wiki") return null;
  if (url.startsWith("/")) return null; // 다른 루트 경로

  // .md 가 아닌 상대 링크는 skip
  if (!/\.md(?:$|[?#])/.test(url)) return null;

  // (c) skill / external project 참조 → text only
  if (/\.claude\//i.test(url)) return { stripLink: true };

  // (b) sessions/staging/ → text only
  if (/(?:^|\/)sessions\/staging\//i.test(url)) return { stripLink: true };

  // (a) sessions/foo.md (with or without ../) → published landing 시도
  const sessionMatch = url.match(/(?:^|\/)sessions\/([^?#]+?)\.md(?:[#?]|$)/i);
  if (sessionMatch) {
    const fileBase = sessionMatch[1]; // e.g. "2026-04-23-mode1-energy-efficient-edge-vlm"
    // 정규화: date prefix + mode 부분 제거 → slug
    // "2026-04-23-mode1-foo-bar"  → "foo-bar"
    // "2026-04-22-mode2-foo"      → "foo"
    const slugMatch = fileBase.match(/^\d{4}-\d{2}-\d{2}-(?:mode\d-)?(.+)$/);
    if (slugMatch) {
      const slug = slugMatch[1];
      if (slugMap.has(slug)) {
        const hash = (url.match(/#(.+)$/) || [, ""])[1];
        return { url: slugMap.get(slug) + (hash ? `#${hash}` : "") };
      }
    }
    // 매핑 못 찾음 → text only
    return { stripLink: true };
  }

  // (f) summary/foo.md (timeline) → published landing
  const summaryMatch = url.match(/(?:^|\/)summary\/([^?#]+?)\.md(?:[#?]|$)/i);
  if (summaryMatch) {
    const fileBase = summaryMatch[1]; // "2026-04-24-moe-fingerprint-security-serving"
    // 같은 정규화 + slug 직접 (date 만 제거)
    const slugMatch = fileBase.match(/^\d{4}-\d{2}-\d{2}-(.+)$/) || fileBase.match(/^\d{4}-\d{2}-(.+)$/);
    if (slugMatch && slugMap.has(slugMatch[1])) {
      return { url: slugMap.get(slugMatch[1]) };
    }
    return { stripLink: true };
  }

  // (e) cross-session ref: "../2026-04-foo.md" or "2026-04-foo.md"
  const crossMatch = url.match(/(?:^|\/)(\d{4}-\d{2})-([^?#/]+?)\.md(?:[#?]|$)/i);
  if (crossMatch) {
    const ym = crossMatch[1];
    let rest = crossMatch[2];
    // strip leading "DD-modeN-" if present (full session log naming)
    rest = rest.replace(/^\d{2}-(?:mode\d-)?/, "");
    if (slugMap.has(rest)) {
      const hash = (url.match(/#(.+)$/) || [, ""])[1];
      return { url: slugMap.get(rest) + (hash ? `#${hash}` : "") };
    }
    // (d) unpublished old session → text only
    return { stripLink: true };
  }

  return null;
}

const LINK_RE = /\[([^\]]*)\]\(([^)]+)\)/g;

let totalFixed = 0;
let totalStripped = 0;
let totalUnchanged = 0;

for (const file of walk(DATA_ROOT)) {
  const content = fs.readFileSync(file, "utf-8");
  let modified = false;
  const newContent = content.replace(LINK_RE, (match, text, url) => {
    const r = rewrite(url);
    if (!r) {
      totalUnchanged++;
      return match;
    }
    if (r.stripLink) {
      totalStripped++;
      modified = true;
      return text;
    }
    if (r.url) {
      totalFixed++;
      modified = true;
      return `[${text}](${r.url})`;
    }
    return match;
  });
  if (modified) {
    if (DRY) {
      console.log(`(dry) would update: ${path.relative(REPO_ROOT, file)}`);
    } else {
      fs.writeFileSync(file, newContent, "utf-8");
      console.log(`updated: ${path.relative(REPO_ROOT, file)}`);
    }
  }
}

console.log(`\nSummary: ${totalFixed} fixed, ${totalStripped} stripped (link removed, text kept), ${totalUnchanged} unchanged`);
if (DRY) console.log("(dry-run, no files written)");
