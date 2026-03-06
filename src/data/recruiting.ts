export interface ResearchTrack {
  title: string;
  titleKo: string;
  desc: string;
  papers: { title: string; venue: string }[];
  news: { title: string; source: string; date: string; url?: string }[];
}

export const researchTracks: ResearchTrack[] = [
  {
    title: "AI & Next-Generation Computing System Optimization",
    titleKo: "AI & 차세대 컴퓨팅 시스템 최적화",
    desc: "On-Device AI 및 LLM의 효율적 실행을 위한 자원 제약 디바이스 최적화 기술을 연구합니다.",
    papers: [
      { title: "ReAx: Resource-efficient Asynchronous Execution for Accelerating LLM Fine-tuning at the Edge", venue: "IEEE ESL, 2026" },
      { title: "SQUAD: A Scalable Quantization Accelerator Toward Energy-Efficient On-Device QAT", venue: "IEEE Access, 2025" },
      { title: "Scale-CIM: Precision-Scalable Computing-in-Memory for Energy-Efficient Quantized Neural Networks", venue: "JSA, 2023" },
    ],
    news: [
      { title: "전자제품 '터치' 익숙해지듯 … AI가 모든 기기 기본값될 것", source: "매일경제", date: "Jan. 2026", url: "https://www.mk.co.kr/news/business/11939100" },
      { title: "Why In-Memory Computation Is So Important For Edge AI", source: "Semi Engineering", date: "Oct. 2025", url: "https://semiengineering.com/why-in-memory-computation-is-so-important-for-edge-ai/" },
      { title: "반도체 저전력 기술 열풍… AI로 인한 전력 문제 해결이 관건", source: "조선비즈", date: "Sep. 2025", url: "https://www.chosun.com/economy/tech_it/2025/09/29/VMGWRUFTHZHNDCTW6BSUQ65VAA/" },
      { title: "'AI 확장·첨단 패키징·지속 가능성'... 2025 반도체 업계 3대 키워드", source: "디지털투데이", date: "Jan. 2025", url: "https://www.digitaltoday.co.kr/news/articleView.html?idxno=547849" },
      { title: "AI 시대 '고성능 메모리' 강자가 시장 지배", source: "동아일보", date: "Jan. 2024", url: "https://www.donga.com/news/Opinion/article/all/20240114/123045745/1" },
    ],
  },
  {
    title: "High-Reliability Memory Systems & HBM Optimization",
    titleKo: "고신뢰 메모리 시스템 및 HBM 최적화",
    desc: "HBM/DDR5 메모리의 RowHammer 등 신뢰성 문제를 해결하고 지능형 메모리 기술을 개발합니다.",
    papers: [
      { title: "SHIFT ECC: A Value Converting HBM ECC Approach for Refresh Energy Efficient Integer Quantized DNN Inference", venue: "ISLPED, 2025" },
      { title: "Bandwidth-Effective DRAM Cache for GPUs with Storage-Class Memory", venue: "HPCA, 2024" },
      { title: "Sparrow ECC: A Lightweight ECC Approach for HBM Refresh Reduction", venue: "ISLPED, 2024 (Best Paper)" },
      { title: "ZEC ECC: A Zero-byte Eliminating Compression Based ECC Scheme", venue: "IEEE Access, 2024" },
      { title: "Twin ECC: A Data Duplication Based ECC for Strong DRAM Error Resilience", venue: "DATE, 2023" },
      { title: "Stealth ECC: A Data-Width Aware Adaptive ECC Scheme", venue: "DATE, 2022" },
    ],
    news: [
      { title: "AI가 요구하는 새로운 메모리 구조: HBM을 넘어 HBF 시대로", source: "네이버", date: "Jan. 2026", url: "https://naver.me/5Otp5wbC" },
      { title: "엔비디아 GPU, Rowhammer 취약점 발견돼…AI 연산 및 클라우드 환경에 심각한 위협", source: "데일리시큐", date: "Jul. 2025", url: "https://www.dailysecu.com/news/articleView.html?idxno=167976" },
      { title: "Understanding Memory's RowHammer Challenge", source: "Electronic Design", date: "Nov. 2024", url: "https://www.electronicdesign.com/technologies/embedded/digital-ics/memory/article/55241243/rambus-understanding-memorys-rowhammer-challenge" },
      { title: "JEDEC Extends DDR5 Memory Specification to 8800 MT/s, Adds Anti-Rowhammer Features", source: "AnandTech", date: "Apr. 2024", url: "https://www.anandtech.com/show/21363/jedec-extends-ddr5-specification-to-8800-mts-adds-anti-rowhammer-features" },
      { title: "Memory's Future Hinges On Reliability", source: "Semi Engineering", date: "2024", url: "https://semiengineering.com/memorys-future-hinges-on-reliability/" },
    ],
  },
  {
    title: "Low-Power Design & Thermal Management",
    titleKo: "저전력 설계 및 열 관리",
    desc: "Chiplet 기반 2.5D/3D 적층 시스템 및 Edge AI 반도체의 열 제어 및 아키텍처 최적화를 연구합니다.",
    papers: [
      { title: "Quantifying the Impact of Monolithic 3D (M3D) Integration on L1 Caches", venue: "IEEE TETC, 2021" },
      { title: "M3D-based SRAM/MRAM Hybrid Memory for an Energy-efficient Unified L2 TLB-Cache", venue: "IEEE Access, 2021" },
      { title: "Monolithic 3D Stacked Multiply-Accumulate Units", venue: "VLSI Journal, 2021" },
      { title: "Exploring the Relation between M3D L1 GPU Cache Capacity and Warp Scheduling", venue: "ISLPED, 2019" },
      { title: "Thermal Modeling and Validation of a Real-World Mobile AP", venue: "IEEE D&T, 2018" },
    ],
    news: [
      { title: "칩렛·3D SoC 가 향후 패키징 산업의 핵심 기술", source: "더일렉", date: "Feb. 2023", url: "https://www.thelec.kr/news/articleView.html?idxno=19857" },
      { title: "칩렛으로 진화한 4세대 제온 SP, 인텔 미래에 중요하다", source: "테크레시피", date: "Jan. 2023", url: "https://techrecipe.co.kr/posts/49292" },
      { title: "반도체 업계, 미세 공정 한계…대안으로 '칩렛' 각광", source: "EPNC", date: "Oct. 2022", url: "https://www.epnc.co.kr/news/articleView.html?idxno=228562" },
      { title: "What is 3D V-cache?", source: "Dot Esports", date: "2023", url: "https://dotesports.com/hardware/news/what-is-3d-v-cache" },
      { title: "반도체 미세화 한계 뛰어넘을 신공법 'M3D'", source: "더일렉", date: "2019", url: "http://www.thelec.kr/news/articleView.html?idxno=2720" },
    ],
  },
];
