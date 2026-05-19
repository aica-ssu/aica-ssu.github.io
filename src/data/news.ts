export interface NewsItem {
  date: string;
  content: string;
  contentKo: string;
}

export const news: NewsItem[] = [
  { date: "2026.05", content: "Paper accepted to ISLPED 2026 (LayUp): Layer-wise Parallelization for Energy-Efficient Edge LLM Training", contentKo: "ISLPED 2026에 논문이 채택되었습니다. (LayUp: 엣지 LLM 학습을 위한 레이어-단위 병렬화)" },
  { date: "2026.03", content: "Our project proposal has been chosen for funding by the 2026 NRF Core Research Program", contentKo: "핵심연구(도약형) 연구과제에 선정되었습니다." },
  { date: "2026.01", content: "One paper accepted to IEEE Computer Architecture Letters", contentKo: "IEEE CAL에 논문이 채택되었습니다." },
  { date: "2025.12", content: "Awards at KSC2025: Seok-Hwan Kim (Outstanding Paper Award), Sang-Jun Moon (Undergraduate Junior Paper Competition Award)", contentKo: "KSC2025에서 김석환 학생이 우수논문상, 문상준 학생이 학부생주니어논문경진대회 수상을 받았습니다." },
  { date: "2025.11", content: "Two papers accepted to KSC2025", contentKo: "KSC2025에 2편의 논문이 채택되었습니다." },
  { date: "2025.09", content: "Ye-Bin Kwon (M.S.) joins DeepX. SQUAD paper accepted to IEEE Access.", contentKo: "권예빈 석사가 DeepX에 입사하였습니다. SQUAD 논문이 IEEE Access에 게재 승인되었습니다." },
  { date: "2025.07", content: "Paper accepted to ISLPED 2025 (SHIFT ECC)", contentKo: "ISLPED 2025에 논문이 채택되었습니다. (SHIFT ECC)" },
  { date: "2025.04", content: "New government-funded PIM AI semiconductor research project launched", contentKo: "PIM AI 반도체 관련 정부 과제가 새롭게 시작되었습니다." },
  { date: "2024.08", content: "ISLPED 2024 Best Paper Award! Sparrow ECC wins the best paper award.", contentKo: "ISLPED 2024 Best Paper Award 수상! Sparrow ECC 논문이 최우수 논문상을 받았습니다." },
  { date: "2024.03", content: "One paper accepted to HPCA 2024 (Bandwidth-Effective DRAM Cache for GPUs)", contentKo: "HPCA 2024에 논문이 채택되었습니다." },
  { date: "2023.08", content: "HedgeRank paper published in Micromachines", contentKo: "HedgeRank 논문이 Micromachines에 게재되었습니다." },
  { date: "2023.04", content: "Twin ECC paper accepted to DATE 2023", contentKo: "DATE 2023에 Twin ECC 논문이 채택되었습니다." },
];
