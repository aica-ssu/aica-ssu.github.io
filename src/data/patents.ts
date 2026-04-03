export interface Patent {
  title: string;
  inventors: string;
  number: string;
  date: string;
  status: "Registered" | "Applied";
}

export const internationalPatents: Patent[] = [
  { title: "Apparatus for Compensating for Training Operation Variation of Computation-in-Memory Based Artificial Neural Network", inventors: "Young-Ho Gong, Woo Hyuck Park, Ye Bin Kwon", number: "US18/472145", date: "Sep. 2023", status: "Applied" },
  { title: "Apparatus of Interconnect with On-the-Fly Quantization and Operating Method Thereof", inventors: "Young-Ho Gong, Woo Hyuck Park, Ye Bin Kwon, Dong Gyu Sim", number: "US18/472137", date: "Apr. 2024", status: "Applied" },
  { title: "Storage Device and Operating Method Thereof", inventors: "Hyunjin Park, Young-Ho Gong, Hoyoung Kim, Sunghan Ahn", number: "US17/067698 / EU 20217748.1", date: "Jul. 2021", status: "Applied" },
  { title: "Hybrid Memory System and Refresh Method Thereof Based on a Read-to-Write Ratio of a Page", inventors: "Young-Ho Gong, Jaehoon Chung, Hyun Ho Cho, Sung Woo Chung", number: "US10198211", date: "Feb. 2019", status: "Registered" },
];

export const domesticPatents: Patent[] = [
  { title: "객체 감지 모델 최적화 장치 및 방법", inventors: "Young-Ho Gong, Seok-Hwan Kim, Dong Gyu Sim", number: "10-2026-0059664", date: "Dec. 2025", status: "Applied" },
  { title: "비전 트랜스포머 모델의 최적화 장치 및 방법", inventors: "Young-Ho Gong, Sang-Jun Moon", number: "10-2026-0052641", date: "Dec. 2025", status: "Applied" },
  { title: "이기종 프로세싱이 적용된 통합 메모리 시스템 기반의 인공지능 모델 훈련 방법 및 이를 수행하는 장치", inventors: "Young-Ho Gong, Bang-San Lee, Dong Gyu Sim", number: "10-2025-0082144", date: "Jun. 2025", status: "Applied" },
  { title: "다중 워드라인 및 수직 비트라인 기반 최댓값 탐색 가능한 3차원 프로세싱인메모리", inventors: "Young-Ho Gong", number: "10-2889341", date: "Nov. 2025", status: "Registered" },
  { title: "인메모리 연산 기반 인공신경망의 훈련 연산 변이 보상 장치", inventors: "Young-Ho Gong, Woo Hyuck Park, Ye Bin Kwon", number: "10-2794894", date: "Mar. 2025", status: "Registered" },
  { title: "엣지 디바이스 간 워크로드 분산 방법", inventors: "Young-Ho Gong, Woo Hyuck Park", number: "10-2787219", date: "Mar. 2025", status: "Registered" },
  { title: "비동기 병렬 실행을 통한 딥러닝 모델 미세 조정 가속 장치 및 그 방법", inventors: "Hyukju Na, Daeseon Choi, Young-Ho Gong, Young Geun Kim", number: "10-2025-0034190", date: "Mar. 2025", status: "Applied" },
  { title: "다중 워드라인 및 수직 비트라인 기반 최댓값 탐색 가능한 3차원 프로세싱인메모리", inventors: "Young-Ho Gong", number: "10-2023-0146400", date: "Oct. 2023", status: "Applied" },
  { title: "메모리 레이어 및 프로세싱 로직회로가 결합된 3차원 프로세싱 인 메모리", inventors: "Young-Ho Gong", number: "10-2023-0146399", date: "Oct. 2023", status: "Applied" },
  { title: "레이어-단위 양자화 신경망을 위한 인-메모리 가속기 및 이의 동작 방법", inventors: "Young-Ho Gong, Young Seo Lee, Sung Woo Chung", number: "10-2507461", date: "Mar. 2023", status: "Registered" },
  { title: "인메모리 연산 기반 인공신경망의 훈련 연산 변이 보상 장치", inventors: "Young-Ho Gong, Woo Hyuck Park, Ye Bin Kwon", number: "10-2022-0186115", date: "Dec. 2022", status: "Applied" },
  { title: "양자화 인터커넥트 장치 및 이의 동작 방법", inventors: "Young-Ho Gong, Woo Hyuck Park, Ye Bin Kwon", number: "10-2022-0129916", date: "Oct. 2022", status: "Applied" },
  { title: "엣지 디바이스 간 워크로드 분산 방법", inventors: "Young-Ho Gong, Woo Hyuck Park", number: "10-2022-0092497", date: "Jul. 2022", status: "Applied" },
  { title: "양자화 신경망을 위한 정밀도 변환 가능 메모리 내부 연산 방법 및 장치", inventors: "Sung Woo Chung, Young Seo Lee, Young-Ho Gong", number: "10-2022-0018230", date: "Feb. 2022", status: "Applied" },
  { title: "통합 L2 캐시-변환 색인 버퍼 메모리의 제어 방법 및 장치", inventors: "Young-Ho Gong", number: "10-2021-0041985", date: "Mar. 2021", status: "Applied" },
  { title: "저장 장치 및 이의 동작 방법", inventors: "Hyunjin Park, Young-Ho Gong, Hoyoung Kim, Sunghan Ahn", number: "10-2021-0087350", date: "Jan. 2020", status: "Applied" },
  { title: "모놀리식 3D 집적 기술 기반 캐시 메모리", inventors: "Young-Ho Gong, Joonho Kong, Sung Woo Chung", number: "10-1913930", date: "Oct. 2018", status: "Registered" },
  { title: "모놀리식 3D 기반 뉴로모픽 칩", inventors: "Young-Ho Gong, Sang Jun Nam, Sung Woo Chung", number: "10-2018-0015670", date: "Feb. 2018", status: "Applied" },
  { title: "모놀리식 3D 집적 구조 기반 캐시메모리", inventors: "Young-Ho Gong, Sung Woo Chung", number: "10-1780586", date: "Sep. 2017", status: "Registered" },
  { title: "모놀리식 3D 집적 구조 기반 메인메모리 장치 및 이와 연결되는 프로세서 장치", inventors: "Young-Ho Gong, Sung Woo Chung", number: "10-1771350", date: "Aug. 2017", status: "Registered" },
  { title: "모놀리식 3차원 집적 기술 기반 캐시 메모리 및 이의 제어 방법", inventors: "Young-Ho Gong, Sung Woo Chung", number: "10-2017-0096900", date: "Jul. 2017", status: "Applied" },
  { title: "메모리의 리프레시 방법 및 장치", inventors: "Young-Ho Gong, Sung Woo Chung", number: "10-2014-0115368", date: "Sep. 2014", status: "Applied" },
  { title: "메모리의 리프레시 방법", inventors: "Young-Ho Gong, Sung Woo Chung", number: "10-2013-0140390", date: "Nov. 2013", status: "Applied" },
];
