export interface ResearchArea {
  title: string;
  titleKo: string;
  description: string;
  descriptionKo: string;
  image?: string;
  topics: string[];
}

export const researchAreas: ResearchArea[] = [
  {
    title: "Edge AI System Design & Optimization",
    titleKo: "Edge AI 시스템 설계 및 최적화",
    description: "Developing lightweight AI models and optimization techniques for resource-constrained edge devices, including pruning, quantization, knowledge distillation, and PIM/CIM-based acceleration.",
    descriptionKo: "Pruning, Quantization, Knowledge Distillation 등의 기법을 통해 제한된 자원의 디바이스에서 효율적인 AI 추론 및 학습이 가능하도록 합니다.",
    image: "/images/research/edge-ai.png",
    topics: ["Model Quantization", "Edge Devices", "Neural Processing Unit", "Edge LLM Optimization"],
  },
  {
    title: "Robust Memory Systems",
    titleKo: "강건한 메모리 시스템",
    description: "Designing innovative error correction codes (ECC) for HBM and DRAM systems, optimizing refresh mechanisms for energy-efficient DNN inference.",
    descriptionKo: "HBM, DDR5 등 최신 메모리 시스템의 신뢰성 향상을 위한 혁신적인 ECC 기법을 개발합니다.",
    image: "/images/research/memory.jpg",
    topics: ["Error Correction Code (ECC)", "Reliability", "HBM Refresh Optimization", "Memory for AI"],
  },
  {
    title: "Next-Generation Memory Architecture",
    titleKo: "차세대 메모리 아키텍처",
    description: "Exploring monolithic 3D integration, hybrid SRAM/MRAM memories, Processing-in-Memory (PIM), Computing-in-Memory (CIM), and emerging non-volatile memory technologies for advanced computing systems.",
    descriptionKo: "Monolithic 3D 집적, SRAM/MRAM 하이브리드 메모리, PIM/CIM 기반 연산 가속, 차세대 비휘발성 메모리 기반 시스템을 연구합니다.",
    image: "/images/research/3d-memory.jpg",
    topics: ["Monolithic 3D Integration", "Processing-in-Memory (PIM)", "Computing-in-Memory (CIM)", "SRAM/MRAM Hybrid", "FeRAM, STT-MRAM, ReRAM"],
  },
  {
    title: "Thermal-Aware System Optimization",
    titleKo: "발열을 고려한 시스템 최적화",
    description: "Developing thermal modeling methodologies and thermal-aware microarchitecture designs for mobile processors and 3D stacked systems.",
    descriptionKo: "모바일 프로세서 및 3D 적층 시스템의 열 모델링 및 발열을 고려한 설계 기법을 연구합니다.",
    image: "/images/research/thermal.jpg",
    topics: ["Thermal Modeling", "Temperature-Aware Refresh", "Chiplet Thermal Management"],
  },
];
