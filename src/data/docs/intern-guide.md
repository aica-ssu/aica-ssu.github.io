# AICA Lab 인턴 지원자 학습 가이드

**대상: 연구 인턴십을 지원하거나, 연구실에서 다루는 주제를 미리 공부하고 싶은 학부생**

> 연구실에서 주로 다루는 주제는 **AI System / Architecture Optimization**입니다.
> 아래 자료들은 인턴 시작 전(또는 지원 전) 단계적으로 학습하면 좋은 내용들을 정리한 것입니다.

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
