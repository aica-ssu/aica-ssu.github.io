export interface PhotoAlbum {
  title: string;
  date: string;
  folder: string;
  cover?: number; // 대표사진 index (0부터 시작, 미지정 시 첫 번째 사진)
}

// 메타데이터만 관리. photos 파일 목록은 빌드 시 자동 생성됨 (npm run gen-photos)
export const albumMeta: PhotoAlbum[] = [
  {
    title: "컴퓨터시스템소사이어티 동계학술대회, 용평",
    date: "2026.01",
    folder: "2026-01-컴퓨터시스템소사이어티-동계학술대회,-용평",
  },
  {
    title: "KSC 2025 여수",
    date: "2025.12",
    folder: "2025-12-KSC-2025-여수",
  },
  {
    title: "권예빈 석사 졸업",
    date: "2025.08",
    folder: "2025-08-권예빈-석사-졸업",
  },
  {
    title: "ISCA 2025, Tokyo",
    date: "2025.07",
    folder: "2025-07-ISCA-2025,-Tokyo",
  },
  {
    title: "컴퓨터시스템소사이어티 동계학술대회, 용평",
    date: "2025.02",
    folder: "2025-02-컴퓨터시스템소사이어티-동계학술대회,-용평",
  },
  {
    title: "전자공학회, 제주",
    date: "2024.06",
    folder: "2024-06-전자공학회,-제주",
  },
  {
    title: "박우혁 석사 졸업",
    date: "2023.01",
    folder: "2023-01-박우혁-석사-졸업",
  },
];
