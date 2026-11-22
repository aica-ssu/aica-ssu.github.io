// CV 페이지의 PDF / Markdown 다운로드 버튼.
// 파일은 scripts/gen-cv.ts (npm run gen-cv) 가 public/cv/ 에 미리 만들어 둔다.
import { existsSync } from "node:fs";
import { join } from "node:path";

const DownloadIcon = () => (
  <svg className="w-3.5 h-3.5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
    <path d="M12 3v12m0 0-4-4m4 4 4-4M5 21h14" />
  </svg>
);

export default function CvActions({ fileBase }: { fileBase: string }) {
  // 아직 생성되지 않은 파일은 버튼을 숨긴다 (빌드 시점에 확인)
  const files = [
    { ext: "pdf", label: "PDF" },
    { ext: "md", label: "Markdown" },
  ].filter(({ ext }) => existsSync(join(process.cwd(), "public", "cv", `${fileBase}.${ext}`)));
  if (files.length === 0) return null;

  return (
    <div className="no-print flex items-center gap-1.5">
      {files.map(({ ext, label }) => (
        <a
          key={ext}
          href={`/cv/${fileBase}.${ext}`}
          download
          title={`Download ${fileBase}.${ext}`}
          className="inline-flex items-center gap-1.5 text-xs font-medium px-2.5 py-1.5 border transition-colors hover:bg-[var(--bg-secondary)]"
          style={{ color: "var(--accent)", borderColor: "var(--border)" }}
        >
          <DownloadIcon />
          {label}
        </a>
      ))}
    </div>
  );
}
