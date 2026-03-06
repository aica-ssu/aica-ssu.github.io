export interface PhotoAlbum {
  title: string;
  date: string;
  folder: string;
  photos: string[];
  cover?: number; // 대표사진 index (0부터 시작, 미지정 시 첫 번째 사진)
}

export const albums: PhotoAlbum[] = [
  {
    title: "컴퓨터시스템소사이어티 동계학술대회, 용평",
    date: "2026.01",
    folder: "2026-01-컴퓨터시스템소사이어티-동계학술대회,-용평",
    photos: Array.from({ length: 10 }, (_, i) => `KakaoTalk_20260122_103716450_${String(i + 1).padStart(2, "0")}.jpg`),
  },
  {
    title: "KSC 2025 여수",
    date: "2025.12",
    folder: "2025-12-KSC-2025-여수",
    photos: [
      ...Array.from({ length: 23 }, (_, i) => `IMG_${7791 + i}.JPG`),
      "KakaoTalk_20251216_193536831_01.jpg",
    ],
  },
  {
    title: "권예빈 석사 졸업",
    date: "2025.08",
    folder: "2025-08-권예빈-석사-졸업",
    photos: [
      "KakaoTalk_20250822_132427930.jpg",
      "KakaoTalk_20250822_132427930_01.jpg",
      "KakaoTalk_20250822_132427930_02.jpg",
      "KakaoTalk_20250822_132427930_03.jpg",
    ],
  },
  {
    title: "ISCA 2025, Tokyo",
    date: "2025.07",
    folder: "2025-07-ISCA-2025,-Tokyo",
    photos: [2, 3, 4, 6, 7, 8, 9, 15, 16, 17, 19, 24, 25].map(
      (n) => `KakaoTalk_20250709_144215226_${String(n).padStart(2, "0")}.jpg`
    ),
  },
  {
    title: "컴퓨터시스템소사이어티 동계학술대회, 용평",
    date: "2025.02",
    folder: "2025-02-컴퓨터시스템소사이어티-동계학술대회,-용평",
    photos: Array.from({ length: 9 }, (_, i) => `${String(i + 1).padStart(2, "0")}.jpg`),
  },
  {
    title: "전자공학회, 제주",
    date: "2024.06",
    folder: "2024-06-전자공학회,-제주",
    photos: Array.from({ length: 11 }, (_, i) => `${String(i + 1).padStart(2, "0")}.jpg`),
  },
  {
    title: "박우혁 석사 졸업",
    date: "2023.01",
    folder: "2023-01-박우혁-석사-졸업",
    photos: ["KakaoTalk_Photo_2023-02-15-22-52-45.jpeg"],
  },
];
