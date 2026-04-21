# AICA Lab 인턴 지원자 학습 가이드

**대상: 연구 인턴십을 지원하거나, 연구실에서 다루는 주제를 미리 공부하고 싶은 학부생**

> 연구실에서 주로 다루는 주제는 **AI System / Architecture Optimization**입니다.
> 아래 자료들은 인턴 시작 전(또는 지원 전) 단계적으로 학습하면 좋은 내용들을 정리한 것입니다.

---

## 학습 전에 명심할 것 (Mindset)

이 가이드는 **시험 범위**가 아닙니다. 본인의 속도로, 본인의 방식으로 공부해 나가면 되는 자료입니다. 다만 몇 가지는 꼭 염두에 두고 시작하길 바랍니다.

### 1. 마감은 없습니다 — 본인의 템포에 맞게

학습 완료 **due date는 따로 정해두지 않습니다.** 빨리 끝내는 것보다, 자신이 이해한 깊이가 더 중요합니다. 일정에 쫓기기보다 각자 맞는 호흡으로 진행하세요.

### 2. 학습 내용은 "정리"하면서, 궁금한 것은 "질문"하세요

- 공부한 내용은 본인이 알아볼 수 있는 형태로 **반드시 정리**하길 권장합니다 (메모, 노션, 블로그, 손글씨, 형식은 자유).
- 정리 과정에서 궁금한 점, 이해가 안 되는 부분, 자료 간에 상충되어 보이는 내용 등이 있으면 **주저하지 말고 교수에게 직접 질문**하세요. 메일, 미팅, 언제든 좋습니다.
- 같이 discussion 하는 과정 자체가 연구실 생활의 본질입니다. 질문은 "잘 몰라서"가 아니라 "더 잘 이해하려고" 하는 것입니다.

### 3. 가이드의 진짜 목적

위 두 가지가 중요한 이유는 — 이 학습 과정을 통해 교수가 다음을 파악하기 위함입니다.

- **어떤 스타일로 공부**하는지 (혼자 파고드는지, 다양한 자료를 비교하는지, 코드로 확인하는지 등)
- **어떤 주제에 흥미**를 느끼고, **어떤 부분을 어려워**하는지
- **교수와 소통하는 방식**은 어떤지 (질문의 형태, 응답에 대한 반응, discussion 흐름 등)

이 정보를 토대로 **각자에게 맞는 연구 방향과 지도 방식을 설계**합니다. 단순히 지식을 주입하는 과정이 아니라, 서로를 알아가고 맞춰가는 과정이라고 생각하면 좋겠습니다.

---

## 1. 인공지능 기초 학습

### 1.1 모두를 위한 딥러닝 시즌2 (PyTorch)

- [강의 페이지](https://deeplearningzerotoall.github.io/season2/lec_pytorch.html)

시즌1은 TensorFlow 기반이었지만, 실제 연구에 쓰는 건 거의 PyTorch 입니다. PyTorch 버전(시즌2)을 수강하는 것을 권장합니다.

단, 이 강의에는 Transformer 파트가 빠져 있습니다. Transformer는 아래 자료로 보완하면 좋습니다.

### 1.2 Transformer 집중 학습

**[딥러닝 기계 번역] Transformer: Attention Is All You Need (꼼꼼한 딥러닝 논문 리뷰와 코드 실습)**

- [YouTube 강의](https://www.youtube.com/watch?v=AA621UofTUA)

논문 기반으로 설명 + 실제 코드까지 제공합니다. 이외에도 YouTube에 Transformer를 시각적으로 잘 풀어 설명한 영상이 많으니, 검색해서 여러 자료를 비교해 보는 것을 추천합니다.

### 1.3 서울대 이준석 교수님 ML/DL 강의 (2024-Fall)

- [YouTube 재생목록](https://www.youtube.com/playlist?list=PL0E_1UqNACXA5u65LBjzFCAVSZ4xuBWqj)

**Lecture 19부터 Transformer 내용**이 나옵니다. Transformer 파트부터 이어서 수강하는 것을 권장합니다.

---

## 2. Efficient AI / ML

### 2.1 MIT Han Lab — EfficientML

- [efficientml.ai](https://efficientml.ai)
- [2023 Fall 전체 강의](https://hanlab.mit.edu/courses/2023-fall-65940)

**AI System / Architecture Optimization 관련 최고 수준의 강의**입니다. 연구실 주제와 가장 직결되는 내용이니 반드시 수강하길 권장합니다.

---

## 3. System / Architecture 연구 시작 전 참고 자료

### 3.1 GPU / CPU 기본 동작 원리

- [GPU는 어떻게 작동할까?](https://www.youtube.com/watch?v=ZdITviTD3VM)
- [CPU는 어떻게 작동할까?](https://www.youtube.com/watch?v=Fg00LN30Ezg) — 비교용 참고
- [AI가 GPU를 좋아하는 이유](https://www.youtube.com/watch?v=AKob3yZT0I4)

### 3.2 GPU 병렬 프로그래밍 (CUDA)

- [GPU CUDA 병렬 프로그래밍 재생목록](https://www.youtube.com/watch?v=zdqZjVxIHT4&list=PLKZ28p5qq0DGLcO6QZdMSG_jsprRtG15C)

5년 전 강의지만, **GPU 병렬 프로그래밍 모델을 코드 레벨에서** 설명하는 자료로는 위 영상들보다 훨씬 자세합니다.

---

## 4. Paper Search

- [cspapers.org](https://cspapers.org) — Systems/Architecture 관련 논문 검색

### 주제별 논문 리스트

- **Edge AI**: [Edge-AI-Paper-List](https://github.com/xumengwei/Edge-AI-Paper-List/)
- **System-level optimization**:
  - [Systems-for-Foundation-Models](https://github.com/inpluslab-wuhui/Systems-for-Foundation-Models)
  - [ml-systems-papers](https://github.com/byungsoo-oh/ml-systems-papers)

### DRAM / NVM 관련

- [CMU Parallel Data Laboratory](https://www.pdl.cmu.edu/)
- [ETH SAFARI Research Group Publications](https://ihpcs.ethz.ch/research/publications-safari-research-group.html)

---

## 학습 순서 권장

1. **1.1 PyTorch 딥러닝 기초** — 기본기가 없다면 여기서 시작
2. **1.2 / 1.3 Transformer** — 최근 모든 AI 연구의 기반
3. **3.1 GPU 기본 동작** — System 연구의 전제
4. **2.1 MIT EfficientML** — 연구실 주제와 직결. 시간 충분히 투자할 것
5. **3.2 CUDA 병렬 프로그래밍** — 실제 최적화 실습이 필요할 때
6. **4. Paper Search** — 연구 주제가 좁혀지면 관련 논문 탐색 시작

---

> **문의 사항이나 추가 질문은 언제든 메일로 연락해 주세요.**
