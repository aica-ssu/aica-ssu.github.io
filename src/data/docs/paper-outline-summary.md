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

### 문장 원칙: 주어·목적어·동사를 명확히

두괄식과 함께 지켜야 할 원칙이다. **모든 문장에서 "누가 — 무엇을 — 어떻게 한다"가 한눈에 보여야 한다.**

한국어는 주어·목적어를 생략하기 쉽고, 동사가 문장 끝에 와서 의미가 뒤늦게 전달된다. 영어로 쓰라는 것이 아니라, **주어를 명확히, 목적어를 명확히, 동사를 명확히** 쓰라는 것이다.

- 각 문장의 의도가 뚜렷하면 → 문장이 모인 **문단**에서 무엇에 집중하는지가 달라진다.
- 주어·목적어가 흐릿한 문장이 쌓이면 → 문단 전체가 무엇을 말하려는지 알 수 없게 된다.

| 나쁜 예 | 좋은 예 |
|---------|---------|
| "분석을 통해 성능이 향상된다" | "제안 기법은 profiling을 통해 throughput을 향상시킨다" |
| "이를 기반으로 최적화한다" | "profiling 결과를 기반으로 expert 배치를 최적화한다" |
| "~에 대해 고려한다" | "~의 overhead를 측정하고, trade-off를 분석한다" |

> **자가 점검법**: 각 문장을 읽고 "누가, 무엇을, 어떻게"를 즉시 답할 수 있는가?

### Claim의 강도를 evidence에 맞출 것

**과장된 표현은 rejection의 대표적 사유이다.** "원천적으로 제거", "완전히 해소" 같은 표현은 실제 결과와 조금이라도 괴리가 있으면 reviewer의 신뢰를 잃는다. 표현을 약간 낮추는 것이 논문의 신뢰도를 높인다.

| 위험한 표현 | 대안 |
|------------|------|
| "원천적으로 제거" | "대부분 제거", "최소화" |
| "크게/현저히 향상" | 구체적 수치 (e.g., "Up to 2.1x") |
| "세계 최초" | "최초"를 쓰려면 범위를 한정: "A에 대해 B를 적용한 최초의 연구" |

### 논문 전체에서 설명 순서를 일관되게 유지할 것

**Introduction에서 A → B → C 순서로 소개했다면, Method·Evaluation·Conclusion에서도 동일한 순서를 따른다.** Figure와 본문의 설명 순서도 일치시킨다. 순서 불일치는 reader를 혼란스럽게 만든다.

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

- **Contribution을 bullet point로 명시적 나열한다.** 각 bullet은 그 자체로 완결적이어야 한다 (부연 없이 "무엇을, 왜, 어떻게"가 전달).
- **Contribution은 3개 내외.** 5개 이상이면 "unfocused" 인상을 준다.
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
- **Subsection 제목은 핵심 메시지를 반영한다.** "3.2 Expert Skip" (X) → "3.2 Limitation of Token-level Selection" (O).
- **Implementation detail은 novelty가 아니면 줄인다.** 단, systems 분야에서는 구현 자체가 contribution일 수 있으므로 판단 필요.
- Overhead는 여기서 논의하지 않는다 (Evaluation으로).
- 설계 결정마다 **왜 이렇게 했는지 (design rationale)**를 포함한다.
- Overview figure를 반드시 포함한다.

### Evaluation

- **Setup**: HW/SW 환경, baseline, metric을 **정확히** 명시한다.
- **결과**: 가장 인상적인 것부터. Figure가 있으면 텍스트에서는 대표 사례만 언급한다. **그래프에는 baseline 기준선을 표시하고, 데이터가 아닌 story를 보여줄 것.** Figure와 본문의 설명 순서를 일치시킨다.
- **Ablation**: 자명하지 않은 설계 결정 — 포함/제외에 따라 결과가 좋아질 수도 나빠질 수도 있는 요소 — 에 대해 수행.
- **약한 결과**: 변명이 아닌 분석으로 다룬다. Workload grouping으로 맥락화하는 것도 효과적이다.
- **Overhead**: 간략한 표 또는 짧은 discussion으로 처리한다.

### Conclusion

- **Introduction과 다르게 쓴다.** Intro = novelty 강조, Conclusion = 결과의 의의 강조.
- 가장 인상적인 수치 1~2개만 언급한다 (전체 나열 금지).
- **Future Work 반드시 포함.** Limitation을 직접 나열하지 말고, 기존 기법과 orthogonal하게 적용 가능함을 강조하거나, 확장 가능성(opportunity)으로 자연스럽게 전환한다.

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

## 4. 흔한 실수 (한 줄 요약)

| # | 실수 | 핵심 |
|---|------|------|
| 1 | 핵심 결정이 하위 bullet에 묻힘 | 중요한 것은 눈에 띄는 위치에 |
| 2 | "한편" 남용 | 논리적 연결 없이 문단을 잇지 않는다 |
| 3 | 한 문장에 4+ 개념 압축 + 주어·목적어 실종 | 문장을 나누고, 각 문장에서 주어·목적어·동사를 명확히 |
| 4 | 모호한 주장 | "사전에 결정됨" → 언제, 어떤 기준으로? |
| 5 | 반복 = 강조 착각 | 같은 문장 복붙은 게으름이다 |
| 6 | Related Work에서 비난 | 저자가 reviewer일 수 있다 |
| 7 | 숫자 나열식 결과 | Figure가 있으면 대표 사례만 텍스트로 |
| 8 | Ablation 누락 | 자명하지 않은 설계 결정에 대해 필수 |
| 9 | 약한 결과를 변명으로 처리 | 분석적 reasoning 또는 맥락화 |
| 10 | Conclusion = Introduction 복사 | 관점을 달리하고, limitation은 확장 가능성으로 전환 |
| 11 | 과장 표현 ("원천적 제거" 등) | Claim 강도 ≤ evidence. Reviewer의 먹잇감이 됨 |
| 12 | Section 간 설명 순서 불일치 | Intro·Method·Eval에서 동일 순서 유지 |

---

## 5. Outline 작성 순서

```
Contribution 정리 → Method → Evaluation → Related Work → Background → Introduction → Conclusion → Abstract
```

Abstract는 가장 마지막에 확정하되, **일찍 draft를 써두면 story의 허점을 빨리 발견할 수 있다.**

---

## 6. 한글로 Outline 작성 시 주의사항

- **한국어식 지시어를 기술 용어로 대체한다.** "바깥" → "top-k+α 범위의 expert" 등 구체적으로 풀어쓰기, "양쪽" → "prefill과 decoding 모두", "이를" → 구체적 대상 명시.
- **한글 bullet이 3줄 이상이면 영어로 쓸 때 한 문장이 너무 길어진다.** Outline 단계에서 미리 나눈다.
- **한국어 동사 후치 구조에 주의한다.** 긴 수식절이 동사 앞에 쌓이면 영어 전환 시 읽기 어려운 문장이 됨.

---

> 상세한 bullet level 가이드, 예시, 체크리스트는 `논문_outline_가이드라인.md` 참조.
