# 논문 Outline 작성 가이드라인 (요약본)

**작성: 공영호 | 대상: 석/박사과정 학생**

> 상세 버전: `논문_outline_가이드라인.md` 참조

---

## 0. 모든 Section을 관통하는 원칙: 두괄식 서술

**중요한 것부터 먼저 쓴다.** 이것이 논문 writing의 가장 기본적인 원칙이다.

한국어 화법에서는 "말은 끝까지 들어봐야 안다"고 하지만, 논문은 정반대다. **핵심이 처음에 나와야 한다.** 발표자료에서 Executive Summary가 맨 앞에 오는 것과 같은 이유다. Reviewer는 피곤하고, 독자는 바쁘다. 핵심을 먼저 보여주지 않으면, 끝까지 읽어주지 않는다.

이 원칙은 논문의 모든 단위에 적용된다:

| 단위 | 적용 |
|------|------|
| **논문 전체** | Abstract → Introduction에서 핵심 기여와 결과를 먼저 제시 |
| **각 Section** | 해당 section의 핵심 메시지를 첫 문단에서 제시 |
| **각 문단** | 첫 문장(topic sentence)이 그 문단의 요지 |
| **결과 서술** | 가장 인상적인 결과를 먼저, 세부 분석은 그 다음 |
| **Contribution** | 가장 중요한 기여를 첫 번째 bullet으로 |

> **자가 점검법**: 각 section과 문단의 **첫 문장만** 이어서 읽어본다. 그것만으로 논문의 전체 논리가 통하면 잘 쓴 것이다.

---

## 1. 시작 전 반드시 할 것

- **Contribution을 먼저 정리한다.** 각 기여점에 (1) 무엇을, (2) 왜, (3) 기존 대비 차이를 명시한다.
- **핵심 용어를 하나로 통일한다.** 동의어 혼용 금지. Google Scholar에서 해당 분야의 표준 표현을 확인한다.
- **타겟 venue의 page limit, format, 최근 논문의 실험 구성을 확인한다.**
- **한 줄 = 하나의 아이디어 = 최종 논문의 한 문단.** 이 원칙을 지키며 outline을 작성한다.

---

## 2. Section별 핵심 원칙

### Abstract — X-Y-Z 구조

| X (What+Why) | Y (Why hard) | Z (Contribution) | 검증 (핵심 결과) |
|---|---|---|---|
| 무엇을, 왜 중요한가 | 왜 어려운가 | 우리의 기여 | 구체적 수치 |

- Stand-alone이어야 한다. 약어 정의 없이 사용하지 않는다.
- "significant improvement"가 아니라 **구체적 숫자**를 쓴다.

### Introduction — 논리 흐름 7단계

```
Common tech → Target 중요성 → 약점 → 기존 노력 → Gap → Gap이 Critical한 이유 → 우리 Novelty
```

- **Contribution을 bullet point로 명시적 나열한다.**
- **첫 페이지에 Figure 1을 배치한다.**
- Technical detail은 여기서 장황하게 쓰지 않는다 (Background으로).
- "Gap이 Critical한 이유"를 빠뜨리지 않는다 (가장 흔한 누락).

### Background/Motivation

- 제안 기법 이해에 **필요한 것만** 설명한다 (교과서 복사 금지).
- 본인에게 당연한 것도 reviewer에게는 아닐 수 있다 — 빠뜨리지 않는다.
- Notation과 용어를 여기서 정의한다.

### Related Work

- 단순 나열이 아니라 **compare and contrast**한다.
- 각 문단 첫 문장 = main idea. 카테고리별로 분류한다.
- **"Similar to their scheme, our~" 절대 금지.** "Different from~"으로 차별성을 부각한다.
- 기존 연구를 **비난하지 않는다** — 그 저자가 reviewer일 수 있다.

### Proposed Method

- **핵심 아이디어를 먼저, 상세 설계는 그 다음.** Cooking recipe처럼 쓰지 않는다.
- Overhead는 여기서 논의하지 않는다 (Evaluation으로).
- 설계 결정마다 **왜 이렇게 했는지 (design rationale)**를 포함한다.
- Overview figure를 반드시 포함한다.

### Evaluation

- **Setup**: HW/SW 환경, baseline, metric을 **정확히** 명시한다.
- **결과**: 가장 인상적인 것부터. Figure가 있으면 텍스트에서는 대표 사례만 언급한다.
- **Ablation**: 모든 핵심 설계 결정에 대해 필수 (비협상 사항).
- **약한 결과**: 변명이 아닌 분석으로 다룬다. Workload grouping으로 맥락화하는 것도 효과적이다.
- **Overhead**: 간략한 표 또는 짧은 discussion으로 처리한다.

### Conclusion

- **Introduction과 다르게 쓴다.** Intro = novelty 강조, Conclusion = 결과의 의의 강조.
- 가장 인상적인 수치 1~2개만 언급한다 (전체 나열 금지).
- **Limitation + Future Work 반드시 포함.**

---

## 3. Reviewer를 고려한 자기 검토

### 모든 것을 아는 Reviewer

- 모든 관련 연구를 인용했는가? (간접 관련 연구 포함)
- 기존 연구와의 차별점이 명확한가?

### 아무것도 모르는 Reviewer

- Background가 self-contained인가? 다른 논문을 읽지 않아도 이해 가능한가?
- "너무 당연해서 쓰기 지루한" 내용이 Background에 있는가? → 있다면 올바른 방향이다.

### 피곤하고 짜증난 Reviewer

- 각 section의 핵심이 첫 문장에서 드러나는가?
- 건너뛰며 읽어도 전체 논리를 따라갈 수 있는가?
- Figure/table이 텍스트 없이도 대략 이해 가능한가?
- **Reject 사유가 될 만한 빈틈이 없는가?**

---

## 4. 흔한 실수 Top 10 (한 줄 요약)

| # | 실수 | 핵심 |
|---|------|------|
| 1 | 핵심 결정이 하위 bullet에 묻힘 | 중요한 것은 눈에 띄는 위치에 |
| 2 | "한편" 남용 | 논리적 연결 없이 문단을 잇지 않는다 |
| 3 | 한 문장에 4+ 개념 압축 | 문장을 나눈다 |
| 4 | 모호한 주장 | "사전에 결정됨" → 언제, 어떤 기준으로? |
| 5 | 반복 = 강조 착각 | 같은 문장 복붙은 게으름이다 |
| 6 | Related Work에서 비난 | 저자가 reviewer일 수 있다 |
| 7 | 숫자 나열식 결과 | Figure가 있으면 대표 사례만 텍스트로 |
| 8 | Ablation 누락 | 설계 결정 당 ablation 필수 |
| 9 | 약한 결과를 변명으로 처리 | 분석적 reasoning 또는 맥락화 |
| 10 | Conclusion = Introduction 복사 | 관점을 달리하고 Limitation/Future Work 포함 |

---

## 5. Outline 작성 순서

```
Contribution 정리 → Method → Evaluation → Related Work → Background → Introduction → Conclusion → Abstract
```

Abstract는 가장 마지막에 확정하되, **일찍 draft를 써두면 story의 허점을 빨리 발견할 수 있다.**

---

> 상세한 bullet level 가이드, 예시, 체크리스트는 `논문_outline_가이드라인.md` 참조.
