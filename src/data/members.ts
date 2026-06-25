export interface Member {
  name: string;
  nameKo: string;
  role: string;
  category: 'pi' | 'graduate' | 'undergraduate' | 'alumni';
  image?: string;
  email?: string;
  linkedin?: string;        // LinkedIn 프로필 URL
  googleScholar?: string;   // Google Scholar 프로필 URL
  research?: string;       // 개별 페이지용 (상세)
  researchShort?: string;  // Members 목록용 (요약)
  background?: string;
  year?: string;
  current?: string;
  // PI-specific fields
  education?: string[];
  career?: string[];
  awards?: string[];
  teaching?: { university: string; period: string; courses: string[] }[];
  services?: string[];
}

export const members: Member[] = [
  {
    name: "Young-Ho Gong",
    nameKo: "공영호",
    role: "Associate Professor",
    category: "pi",
    image: "/images/members/Young-Ho Gong.jpg",
    email: "yhgong@ssu.ac.kr",
    linkedin: "https://www.linkedin.com/in/young-ho-gong-453b86118/",
    googleScholar: "https://scholar.google.com/citations?user=j_7MT9kAAAAJ&hl=en",
    //research: "Edge AI System Design/Optimization\nRobust Memory Systems\nNext-generation Memory Technologies (FeRAM, MRAM, ReRAM, PRAM)\nApplication-specific Performance Optimization\nThermal-aware System Optimization",
    background: "School of Software, Soongsil University",
    education: [
      "Ph.D., Computer Science & Engineering, Korea University (2018)\n- Thesis: Thermal-/Energy-aware Architectural Design Techniques (Advisor: Prof. Sung Woo Chung)",
      "B.S., Computer Science & Engineering, Korea University (2012)",
    ],
    career: [
      "Associate Professor, Soongsil University (Mar. 2026–Present)",
      "Assistant Professor, Soongsil University (Mar. 2023–Feb. 2026)",
      "VPP Consulting Professor, SK Hynix (Apr. 2025–Sep. 2025)",
      "Assistant Professor, Kwangwoon University (Mar. 2020–Feb. 2023)",
      "Staff Engineer, Samsung Electronics Memory Division (Sep. 2018–Feb. 2020)",
      "Research Assistant, Korea University (Mar. 2012–Aug. 2018)",
    ],
    awards: [
      "Research Excellence Award, Korea University Department (2017)",
    ],
    teaching: [
      {
        university: "Soongsil University",
        period: "2023–Present",
        courses: [
          "Computer Architecture",
          "Operating Systems",
          "System Programming",
          "Next-generation Memory Systems",
          "Advanced Computer Architecture",
        ],
      },
      {
        university: "Kwangwoon University",
        period: "2020–2022",
        courses: [
          "GPU Computing",
          "C++ Programming",
          "Digital Logic Design",
          "VLSI Design",
          "3D Semiconductors",
          "Thermal Management",
          "Next-generation Memory",
        ],
      },
    ],
    services: [
      "Reviewer: IEEE/ACM DAC, ISLPED, ICCD",
      "Journal Reviewer: IEEE TC, TACO, TVLSI, IEEE Access",
    ],
  },
  {
    name: "Bang-San Lee",
    nameKo: "이방산",
    role: "Integrated M.S./Ph.D. Student (2025~)",
    category: "graduate",
    image: "/images/members/Bang-San Lee.png",
    email: "atks990210@gmail.com",
    researchShort: "Edge LLM Training, MoE Architecture Optimization",
    research: "Edge LLM Training, MoE Architecture Optimization",
    background: "B.S. Computer and Information Engineering, Kwangwoon University (2025)",
  },
  {
    name: "Jun-Hyeok Ha",
    nameKo: "하준혁",
    role: "M.S. Student (2025~)",
    category: "graduate",
    image: "/images/members/Jun-Hyeok Ha.png",
    email: "hamoci@naver.com",
    researchShort: "Memory Reliability & HW Optimization",
    research: "Memory system reliability, \nHardware optimization, \nHigh-level synthesis",
    background: "B.S. Computer and Information Engineering, Kwangwoon University (2025)",
  },
  {
    name: "Seok-Hwan Kim",
    nameKo: "김석환",
    role: "M.S. Student (2025~)",
    category: "graduate",
    image: "/images/members/Seok-Hwan Kim.jpg",
    email: "kchk0628@naver.com",
    researchShort: "HW-SW Co-optimization for AI",
    research: "HW-SW Co-optimization for AI, \nKV cache optimization",
    background: "B.S. Software, Soongsil University (2025)",
  },
  {
    name: "Sang-Jun Moon",
    nameKo: "문상준",
    role: "M.S. Student (2026~)",
    category: "graduate",
    image: "/images/members/Sang-Joon Moon.jpg",
    email: "moonsangjun.bizz@gmail.com",
    researchShort: "Model Quantization",
    research: "Layer-wise model quantization",
    background: "B.S. Software, Soongsil University (Expected 2026)",
  },
  {
    name: "Yoon Hong Min",
    nameKo: "민윤홍",
    role: "B.S. Student (2025~)",
    category: "undergraduate",
    image: "/images/members/Yoon Hong Min.png",
    email: "picomin1027@gmail.com",
    researchShort: "Edge AI",
    research: "Edge AI",
    background: "B.S. School of AI Software, Soongsil University (Expected 2029)",
  },
/*  {
    name: "Jun-Hyeok Lee",
    nameKo: "이준혁",
    role: "Undergraduate Intern",
    category: "undergraduate",
    email: "wnsx0000@gmail.com",
    researchShort: "Quantization & Edge LLM",
    research: "Model quantization and edge LLM serving",
    background: "B.S. Software, Soongsil University (Expected 2026)",
  },*/
  {
    name: "Ye-Bin Kwon",
    nameKo: "권예빈",
    role: "M.S. 2025",
    category: "alumni",
    year: "2025",
    current: "DeepX",
    linkedin: "https://www.linkedin.com/in/yebin-gwon/",
  },
  {
    name: "Woo Hyuck Park",
    nameKo: "박우혁",
    role: "M.S. 2023",
    category: "alumni",
    year: "2023",
    current: "SK Hynix",
    linkedin: "https://www.linkedin.com/in/%EC%9A%B0%ED%98%81-%EB%B0%95-321820239/",
  },
  {
    name: "Jun-Hyeok Lee",
    nameKo: "이준혁",
    role: "B.S. 2027 (Expected)",
    category: "alumni",
    year: "2026",
  },
  {
    name: "Ryoonki Hong",
    nameKo: "홍륜기",
    role: "B.S. 2025",
    category: "alumni",
    year: "2025",
    current: "Univ. Southern California (M.S in CS)",
    linkedin: "https://www.linkedin.com/in/ryoonkihong/",
  },
  {
    name: "Dong-Woo Kim",
    nameKo: "김동우",
    role: "B.S. 2023",
    category: "alumni",
    year: "2023",
    current: "Samsung Electronics",
  },
  {
    name: "Han-Gil Lee",
    nameKo: "이한길",
    role: "B.S. 2023",
    category: "alumni",
    year: "2023",
    current: "Samsung Foundry",
  },
];
