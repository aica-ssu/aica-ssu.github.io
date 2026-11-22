// 멤버별 CV 파일(PDF, Markdown)을 public/cv/ 에 만든다.
//   npm run gen-cv   (= npm run build 후 이 스크립트 실행)
// - Markdown: src/lib/cv.ts 의 cvToMarkdown() 으로 데이터에서 바로 생성
// - PDF: 빌드된 out/ 을 로컬 서버로 띄우고 헤드리스 크롬으로 /members/<id> 를 인쇄
//   (--no-pdf-header-footer 로 날짜·주소 머리말/꼬리말 없음, 인쇄 레이아웃은 globals.css 의 print 규칙)
// publications.ts / patents.ts / members.ts 를 고치면 이 스크립트를 다시 돌리고 public/cv 도 함께 커밋한다.
// PDF 는 생성 시각이 들어가 매번 달라지므로, 내용(Markdown)이 바뀐 사람만 다시 만든다.
// 레이아웃(CSS·컴포넌트)을 바꿨을 때는 --force 로 전원 PDF 를 다시 만든다.
import { spawn, spawnSync } from "node:child_process";
import { createServer } from "node:http";
import type { AddressInfo } from "node:net";
import { existsSync, mkdirSync, readdirSync, readFileSync, rmSync, statSync, writeFileSync } from "node:fs";
import { extname, join, resolve } from "node:path";
import { members } from "@/data/members";
import { buildCv, cvFileBase, cvToMarkdown } from "@/lib/cv";

const ROOT = resolve(__dirname, "..");
const OUT = join(ROOT, "out");
const DEST = join(ROOT, "public", "cv");
const FORCE = process.argv.includes("--force");

const MIME: Record<string, string> = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript",
  ".css": "text/css",
  ".json": "application/json",
  ".txt": "text/plain; charset=utf-8",
  ".svg": "image/svg+xml",
  ".png": "image/png",
  ".jpg": "image/jpeg",
  ".jpeg": "image/jpeg",
  ".webp": "image/webp",
  ".ico": "image/x-icon",
  ".woff2": "font/woff2",
  ".woff": "font/woff",
};

function findChrome() {
  const candidates = [process.env.CHROME_PATH, "google-chrome", "google-chrome-stable", "chromium", "chromium-browser"].filter(Boolean) as string[];
  return candidates.find((c) => spawnSync(c, ["--version"], { stdio: "ignore" }).status === 0);
}

// out/ 정적 파일 서버 (/members/x → out/members/x.html)
function serveOut() {
  const server = createServer((req, res) => {
    const path = decodeURIComponent(new URL(req.url ?? "/", "http://x").pathname);
    const base = join(OUT, path);
    const file = [base, `${base}.html`, join(base, "index.html")].find((f) => f.startsWith(OUT) && existsSync(f) && statSync(f).isFile());
    if (!file) {
      res.writeHead(404).end();
      return;
    }
    res.writeHead(200, { "Content-Type": MIME[extname(file)] ?? "application/octet-stream" }).end(readFileSync(file));
  });
  return new Promise<{ port: number; close: () => void }>((ok) =>
    server.listen(0, "127.0.0.1", () => ok({ port: (server.address() as AddressInfo).port, close: () => server.close() })),
  );
}

// 크롬은 비동기로 띄운다 (spawnSync 는 같은 프로세스의 로컬 서버를 멈춰 세운다)
function run(cmd: string, args: string[], timeoutMs: number) {
  return new Promise<void>((done) => {
    const child = spawn(cmd, args, { stdio: "ignore" });
    const timer = setTimeout(() => child.kill("SIGKILL"), timeoutMs);
    child.on("close", () => {
      clearTimeout(timer);
      done();
    });
  });
}

async function main() {
  if (!existsSync(join(OUT, "members"))) {
    console.error("out/ 이 없습니다. npm run gen-cv (빌드 포함) 로 실행하세요.");
    process.exit(1);
  }
  const chrome = findChrome();
  if (!chrome) {
    console.error("Chrome/Chromium 을 찾지 못했습니다. CHROME_PATH 환경변수로 경로를 지정하세요.");
    process.exit(1);
  }

  mkdirSync(DEST, { recursive: true });
  const targets = members.filter((m) => m.category !== "alumni").map((m) => ({ cv: buildCv(m), base: cvFileBase(buildCv(m)) }));

  // 더 이상 없는 멤버의 파일은 지운다
  const keep = new Set(targets.flatMap((t) => [`${t.base}.pdf`, `${t.base}.md`]));
  for (const f of readdirSync(DEST)) if (/_CV\.(pdf|md)$/.test(f) && !keep.has(f)) rmSync(join(DEST, f));

  const server = await serveOut();
  let failed = 0;
  let updated = 0;
  try {
    for (const { cv, base } of targets) {
      const mdPath = join(DEST, `${base}.md`);
      const pdf = join(DEST, `${base}.pdf`);
      const md = cvToMarkdown(cv);
      const unchanged = existsSync(mdPath) && readFileSync(mdPath, "utf8") === md && existsSync(pdf);
      if (unchanged && !FORCE) {
        console.log(`= ${base} (변경 없음)`);
        continue;
      }
      writeFileSync(mdPath, md);
      rmSync(pdf, { force: true });
      updated++;
      await run(
        chrome,
        [
          "--headless=new",
          "--disable-gpu",
          "--no-sandbox",
          "--no-pdf-header-footer",
          "--virtual-time-budget=10000",
          `--print-to-pdf=${pdf}`,
          `http://127.0.0.1:${server.port}/members/${cv.slug}`,
        ],
        60_000,
      );
      const ok = existsSync(pdf) && statSync(pdf).size > 0;
      if (!ok) failed++;
      console.log(`${ok ? "✓" : "✗"} ${base}.pdf / ${base}.md 갱신`);
    }
  } finally {
    server.close();
  }
  if (failed) {
    console.error(`PDF ${failed}개 생성 실패`);
    process.exit(1);
  }
  console.log(`public/cv/: ${targets.length}명 중 ${updated}명 갱신`);
}

main();
