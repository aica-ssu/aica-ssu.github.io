# 논문 Outline 작성 가이드라인

**작성: 공영호 | 대상: 석/박사과정 학생**

> Outline은 건물의 설계도다. 설계도 없이 건물을 짓는 사람은 없다.
> 논문도 마찬가지다. Outline 단계에서 구조를 확실히 잡아야 나중에 수정이 쉽다.

<details>
<summary><strong>📖 왜 Outline부터 써야 하는가?</strong></summary>

- **구조 변경이 쉽다**: 완성된 글의 section 순서를 바꾸는 것은 고통스럽다. Outline에서는 bullet을 드래그하면 끝이다.
- **논리적 허점이 보인다**: Outline 단계에서 "이 주장의 근거가 어디에 있지?"를 확인할 수 있다.
- **공동 작업이 효율적이다**: 지도교수와 outline을 먼저 합의하면, 나중에 전체 rewrite하는 비극을 피할 수 있다.
- **분량 조절이 가능하다**: Bullet 수를 세면 예상 page 수를 추정할 수 있다 (대략 bullet 3~4개 = 1 column).

</details>

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

<details>
<summary><strong>📖 상세: 두괄식 서술의 나쁜 예 / 좋은 예 + 자가 점검 테스트</strong></summary>

| 단위 | 나쁜 예 | 좋은 예 |
|------|---------|---------|
| **논문 전체** | 배경 설명이 2페이지 이후에야 contribution 등장 | 첫 페이지 안에 contribution과 Figure 1 |
| **각 Section** | Method section이 배경 설명으로 시작 | "제안 기법의 핵심은 ~이다"로 시작 |
| **각 문단** | 세부 내용을 나열한 뒤 마지막에 요약 | 결론을 먼저 제시하고 근거를 이어서 설명 |
| **결과 서술** | 작은 결과부터 나열하여 클라이맥스 구조 | 가장 큰 성과를 첫 subsection에 배치 |
| **Contribution** | 부수적 기여가 먼저, 핵심이 마지막 | 핵심 알고리즘 제안이 C1 |
| **Related Work** | 개별 논문 소개로 시작 | "~를 위한 기법들이 제안되어 왔다"로 시작 |

**자가 점검 테스트:**

```
[Test] Introduction 첫 문장들만 이어 읽기:

"대규모 언어 모델의 추론 효율성이 중요해지고 있다."
→ "MoE는 이를 위한 대표적 기법이지만, expert 배치 문제가 있다."
→ "기존 연구는 static placement에 의존하여 workload 변화에 대응하지 못한다."
→ "본 논문에서는 adaptive expert placement를 제안한다."
→ 논리가 통한다 ✓
```

</details>

---

## 1. 시작 전 반드시 할 것

- **Contribution을 먼저 정리한다.** 각 기여점에 (1) 무엇을, (2) 왜, (3) 기존 대비 차이를 명시한다.
- **핵심 용어를 하나로 통일한다.** 동의어 혼용 금지. Google Scholar에서 해당 분야의 표준 표현을 확인한다.
- **타겟 venue의 page limit, format, 최근 논문의 실험 구성을 확인한다.**
- **한 줄 = 하나의 아이디어 = 최종 논문의 한 문단.** 이 원칙을 지키며 outline을 작성한다.

<details>
<summary><strong>📖 상세: Contribution 정리 예시, Venue 분석 체크리스트, 표현 검증법</strong></summary>

### Contribution 정리 예시

Contribution을 bullet point로 작성한다. 이 bullet point는 나중에 Introduction의 contribution list가 된다.

```
- (C1) CPU-GPU 이기종 시스템에서 MoE expert를 동적으로 배치하는
       adaptive placement 알고리즘 제안
- (C2) Expert 활용 패턴을 사전 분석하여 placement를 최적화하는
       profiling 기법 개발
- (C3) 다양한 MoE 모델에서 기존 대비 최대 2.1배 throughput 향상 검증
```

**주의: 반복 = 강조가 아니다!** 동일한 문장을 복사-붙여넣기 하는 것은 강조가 아니라 게으름이다. Introduction에서는 novelty 관점에서, Conclusion에서는 결과와 의의 관점에서 다른 방식으로 서술해야 한다.

### 타겟 Venue 분석 체크리스트

| 항목 | 확인 내용 |
|------|----------|
| Page limit | 본문 몇 페이지? Reference 포함/미포함? |
| Format | Single/double column? Template 종류? |
| 기대하는 결과 | 해당 venue의 최근 논문들이 어떤 실험을 포함하는지 확인 |
| Related work 위치 | Introduction 다음? Evaluation 다음? (venue마다 관행이 다름) |
| Supplementary | 허용 여부, 분량 제한 |

### 핵심 용어 정의 및 통일

- 논문 전체에서 핵심 용어는 하나로만 사용한다. 예: "refresh interval"과 "refresh period"를 혼용하지 않는다.
- **새로운 용어를 만드는 것은 가급적 피한다.** Google Scholar에서 해당 분야 주요 논문들이 쓰는 표현을 확인한다.
- Acronym은 반드시 첫 등장 시 full name과 함께 정의하고, 정의한 후에만 사용한다.

### 표현 검증 습관

어떤 표현을 쓰기 전에 Google Scholar에서 검색한다.

- 검색 시 따옴표로 묶어서 exact match 검색: `"As we mentioned"`
- 검색도구에서 `"Transactions on"` 또는 `"IEEE International Conference on"`을 소스 필터로 넣으면, 학술 논문에서 실제로 쓰이는 표현인지 확인 가능
- 검색 결과가 거의 없는 표현은 사용하지 않는다

### Outline 작성 후 확장하는 법

Outline이 확정되면, 각 bullet 위에 LaTeX 주석으로 요약을 남기고 그 아래에 paragraph를 확장한다.

```latex
% TL;DR: 기존 MoE 라우팅은 load balancing만 고려하여 전문가 활용도가 낮다
기존 Mixture-of-Experts (MoE) 모델의 라우팅 알고리즘은 주로 load balancing에
초점을 맞추어 설계되어 왔다. 그러나 이러한 접근은 각 expert의 실제 전문성을
고려하지 않아, 결과적으로 expert utilization이 낮아지는 문제를 야기한다.
```

</details>

---

## 2. Section별 핵심 원칙

### Abstract — X-Y-Z 구조

| X (What+Why) | Y (Why hard) | Z (Contribution) | 검증 (핵심 결과) |
|---|---|---|---|
| 무엇을, 왜 중요한가 | 왜 어려운가 | 우리의 기여 | 구체적 수치 |

- Stand-alone이어야 한다. 약어 정의 없이 사용하지 않는다.
- "significant improvement"가 아니라 **구체적 숫자**를 쓴다.

<details>
<summary><strong>📖 상세: Abstract 작성 가이드 + 좋은 예 / 나쁜 예</strong></summary>

#### 목적과 역할
- 논문 전체의 TL;DR. **모든 독자가 가장 먼저 읽는 부분**이다.
- Stand-alone이어야 한다: Abstract만 읽어도 논문의 핵심을 파악할 수 있어야 한다.
- 흥미를 유발하고, 연구의 필요성이 드러나야 한다.

#### 흔한 실수
- 장황하게 배경을 설명하는 것 (Abstract에서 배경은 1~2문장이면 충분)
- 구체적 수치 없이 "significant improvement"만 언급하는 것
- 논문 본문에서 그대로 복사한 문장을 사용하는 것
- 약어를 정의 없이 사용하는 것

#### Bullet Level 가이드

```
Abstract (하나의 문단, bullet 4~6개)
├── X: 문제 정의 + 중요성 (1~2 bullet)
├── Y: 기존 접근의 한계 (1 bullet)
├── Z: 우리의 핵심 기여 (1~2 bullet)
└── 검증: 핵심 결과 요약 (1 bullet)
```

#### 나쁜 예
```
최근 대규모 언어 모델이 다양한 분야에서 활용되고 있다. Mixture-of-Experts는
이러한 모델의 효율성을 높이는 기법이다. 본 논문에서는 새로운 기법을 제안한다.
실험 결과 성능이 향상되었다.
```
문제점: X-Y-Z 중 Y(왜 어려운가)가 없고, 구체적 수치가 없으며, "새로운 기법"이 무엇인지 알 수 없다.

#### 좋은 예
```
Mixture-of-Experts (MoE) 모델의 추론 효율성은 expert의 물리적 배치에 크게
좌우된다 (X). 그러나 기존 static placement 방식은 입력 분포의 변화에 대응하지
못하여, 특정 expert에 부하가 집중되는 문제를 야기한다 (Y). 본 논문에서는
runtime에 expert 활용 패턴을 분석하고, CPU-GPU 간 expert를 동적으로
재배치하는 adaptive placement 알고리즘을 제안한다 (Z). LLaMA-MoE,
Mixtral, DeepSeek-MoE에서 평가한 결과, GPU-only baseline 대비 최대 2.1배의
throughput 향상을 달성하였다 (검증).
```

</details>

### Introduction — 논리 흐름 7단계

```
Common tech → Target 중요성 → 약점 → 기존 노력 → Gap → Gap이 Critical한 이유 → 우리 Novelty
```

- **Contribution을 bullet point로 명시적 나열한다.** 각 bullet은 그 자체로 완결적이어야 한다 (부연 없이 "무엇을, 왜, 어떻게"가 전달).
- **Contribution은 3개 내외.** 5개 이상이면 "unfocused" 인상을 준다.
- **첫 페이지에 Figure 1을 배치한다.**
- Technical detail은 여기서 장황하게 쓰지 않는다 (Background으로).
- "Gap이 Critical한 이유"를 빠뜨리지 않는다 (가장 흔한 누락).

<details>
<summary><strong>📖 상세: Introduction 논리 흐름 상세 + Bullet Level 가이드</strong></summary>

#### 필수 포함 요소 (논리 흐름)

```
1. Common technology 소개
   → "현재 ~가 널리 사용되고 있다"

2. Target의 중요성 강조
   → "그중 ~는 전체 성능/에너지의 큰 부분을 차지한다"

3. Target의 약점/문제점 제시
   → "그러나 ~에는 다음과 같은 한계가 있다"

4. 기존 해결 노력 소개
   → "이를 해결하기 위해 ~가 제안되어 왔다 [ref]"

5. 기존 연구의 gap 지적
   → "하지만 기존 연구들은 ~를 고려하지 못했다"

6. 이 gap이 critical함을 강조
   → "이는 ~때문에 반드시 해결되어야 하는 문제이다"

7. 우리의 novelty 제시
   → "본 논문에서는 ~를 제안하여 이 문제를 해결한다"

8. Contribution bullet points
   → 구체적인 기여 사항 나열

9. (선택) 핵심 결과 미리보기
   → "제안 기법은 ~에서 ~% 향상을 달성하였다"
```

#### 흔한 실수
- **Technical detail을 장황하게 설명하는 것**: Introduction이 너무 길어지면 Background/Motivation section을 별도로 만든다.
- **확인되지 않은 사항을 기술하는 것**: 논문 전체에서 주의할 사항이지만, Introduction에서 특히 위험하다.
- **Contribution과 결과를 혼동하는 것**: "2.1배 향상"은 결과이지 contribution이 아니다. Contribution은 "adaptive placement 알고리즘을 제안"하는 것이다.
- 기존 연구를 비난하는 어조: 기존 연구의 "한계"를 지적하되, 그 연구 자체를 폄하하지 않는다.

#### Bullet Level 가이드

```
Introduction
├── [1문단] Common technology 배경
├── [1문단] Target의 중요성
├── [1문단] 문제점 + Figure 1 참조
├── [1문단] 기존 연구 노력과 그 gap
├── [1문단] Gap의 심각성 + 우리 접근의 핵심 아이디어
├── [Bullet list] Contributions (3~4개)
└── [1문단] 논문 구성 (Section 2에서는... 선택사항)
```

**주의:** 핵심 contribution이나 결정이 하위 bullet에 묻혀서는 안 된다. 가장 중요한 내용은 최상위 레벨에 위치시킨다.

</details>

### Background/Motivation

- 제안 기법 이해에 **필요한 것만** 설명한다 (교과서 복사 금지).
- 본인에게 당연한 것도 reviewer에게는 아닐 수 있다 — 빠뜨리지 않는다.
- Notation과 용어를 여기서 정의한다.

<details>
<summary><strong>📖 상세: Background 구성 + Bullet Level 가이드</strong></summary>

#### 목적과 역할
- 독자가 제안 기법을 이해하기 위해 필요한 사전 지식을 제공한다.
- **논문은 self-contained이어야 한다**: 다른 논문을 읽지 않아도 이해할 수 있어야 한다.
- Motivation section이 포함되는 경우, "왜 이 문제가 중요한가"를 데이터로 보여준다.

#### 필수 포함 요소
- 제안 기법이 의존하는 기반 기술 설명
- 문제 정의 (Problem definition): 입력, 출력, 가정, 표기법
- (Motivation 포함 시) 기존 접근의 한계를 보여주는 분석/데이터

#### 흔한 실수
- 교과서를 통째로 옮겨 적는 것: 제안 기법 이해에 **필요한 부분만** 설명한다.
- 용어와 notation을 여기서 정의하지 않고 Method section에서 갑자기 사용하는 것

#### Bullet Level 가이드

```
Background / Motivation
├── [Subsection] 기반 기술 A 설명
│   ├── 핵심 개념
│   └── 본 논문과의 관련성
├── [Subsection] 기반 기술 B 설명
│   ├── 핵심 개념
│   └── 본 논문과의 관련성
├── [Subsection] Problem Definition
│   ├── 입력/출력 정의
│   └── 핵심 notation 소개
└── [Subsection] Motivation (선택)
    ├── 기존 접근의 한계를 보여주는 실험/분석
    └── 이로부터 도출되는 핵심 관찰 (Key Observation)
```

</details>

### Related Work

- 단순 나열이 아니라 **compare and contrast**한다.
- 각 문단 첫 문장 = main idea. 카테고리별로 분류한다.
- **"Similar to their scheme, our~" 절대 금지.** "Different from~"으로 차별성을 부각한다.
- 기존 연구를 **비난하지 않는다** — 그 저자가 reviewer일 수 있다.

<details>
<summary><strong>📖 상세: Related Work 작성 전략 + 위치 결정 + Bullet Level 가이드</strong></summary>

#### 절대로 피해야 할 표현
```
(X) "Similar to their scheme, our proposed scheme~"
    → 이전 연구와 비슷하다면 novelty가 없어 보인다.

(O) "Different from their scheme, our proposed scheme~"
    → 차별성을 강조하여 novelty를 부각한다.
```

#### Related Work의 위치
- 본 논문의 실험에서 이전 연구의 기법과 직접 비교하는 경우: Introduction 바로 다음에 배치
- Conventional scheme과만 비교하는 경우: Evaluation 다음에 배치

#### 작성 전략
1. 관련 논문을 모두 리스트업한다
2. 각 논문에 대해 1~2줄 요약을 쓴다
3. 카테고리로 분류한다 (2~4개)
4. 각 카테고리에 대해 paragraph를 작성한다

#### Bullet Level 가이드

```
Related Work
├── [1문단] Category A: ~를 위한 기법들
│   ├── 연구 1의 핵심 아이디어 + 한계
│   ├── 연구 2의 핵심 아이디어 + 한계
│   └── "Different from these, our work~"
├── [1문단] Category B: ~를 활용한 기법들
│   ├── 연구 3의 핵심 아이디어 + 한계
│   └── "Our approach differs in that~"
├── [1문단] 특별히 가까운 연구 (주요 경쟁자)
│   └── 구체적 차이점 강조
└── (선택) [표] Comparison table
```

</details>

### Proposed Method

- **핵심 아이디어를 먼저, 상세 설계는 그 다음.** Cooking recipe처럼 쓰지 않는다.
- **Subsection 제목은 핵심 메시지를 반영한다.** "3.2 Expert Skip" (X) → "3.2 Limitation of Token-level Selection" (O).
- **Implementation detail은 novelty가 아니면 줄인다.** 단, systems 분야에서는 구현 자체가 contribution일 수 있으므로 판단 필요.
- Overhead는 여기서 논의하지 않는다 (Evaluation으로).
- 설계 결정마다 **왜 이렇게 했는지 (design rationale)**를 포함한다.
- Overview figure를 반드시 포함한다.

<details>
<summary><strong>📖 상세: Cooking recipe vs 아이디어 중심 서술 + Bullet Level 가이드</strong></summary>

#### "Cooking recipe처럼 쓰지 말 것"

Method section은 코드의 README가 아니다. **알고리즘을 서술하는 것이지, 코드를 설명하는 것이 아니다.**

```
(X) "먼저 입력을 받아 전처리한다. 그다음 모델에 넣는다. 그 출력을 후처리한다."
    → Cooking recipe

(O) "제안하는 알고리즘의 핵심은 expert 활용 패턴의 temporal locality를
     exploitation하는 것이다. 이를 위해..."
    → 아이디어 중심 서술
```

#### 추가 주의사항
- **이 section에서 overhead를 기술하지 않는다.** Overhead는 Evaluation에서 다룬다.
- Pipeline 순서대로 설명할 필요가 없다. **핵심 component부터 먼저 설명**한다.
- 한 문장에 4개 이상의 개념을 압축하지 않는다.

#### Bullet Level 가이드

```
Proposed Method
├── [Subsection] Overview
│   ├── 전체 framework 설명 (overview figure 참조)
│   └── 각 component 간략 소개 + 어디서 설명할지 안내
├── [Subsection] 핵심 Component A (가장 중요한 것 먼저)
│   ├── 핵심 아이디어 / 직관
│   ├── 구체적 설계
│   └── 왜 이렇게 설계했는지 (design rationale)
├── [Subsection] 핵심 Component B
│   ├── 핵심 아이디어 / 직관
│   └── Design rationale
└── [Subsection] 통합 / 전체 알고리즘
    └── (선택) Pseudocode
```

</details>

### Evaluation

- **Setup**: HW/SW 환경, baseline, metric을 **정확히** 명시한다.
- **결과**: 가장 인상적인 것부터. Figure가 있으면 텍스트에서는 대표 사례만 언급한다. **그래프에는 baseline 기준선을 표시하고, 데이터가 아닌 story를 보여줄 것.** Figure와 본문의 설명 순서를 일치시킨다.
- **Ablation**: 자명하지 않은 설계 결정 — 포함/제외에 따라 결과가 좋아질 수도 나빠질 수도 있는 요소 — 에 대해 수행.
- **약한 결과**: 변명이 아닌 분석으로 다룬다. Workload grouping으로 맥락화하는 것도 효과적이다.
- **Overhead**: 간략한 표 또는 짧은 discussion으로 처리한다.

<details>
<summary><strong>📖 상세: Evaluation 작성 가이드 + 약한 결과 처리 전략 + Bullet Level 가이드</strong></summary>

#### 필수 포함 요소

| 항목 | 내용 |
|------|------|
| Experimental setup | HW/SW 환경, dataset, baseline, metric |
| 주요 결과 | Baseline 대비 성능 비교 (가장 인상적인 것 먼저) |
| Ablation study | 각 핵심 설계 결정의 효과 검증 (**비협상 사항**) |
| Sensitivity analysis | 주요 parameter에 대한 민감도 분석 |
| Overhead analysis | 제안 기법의 overhead (간략하게) |

#### 추가 주의사항
- **강한 주장에는 구체적 근거를 제시한다.** "GPU-only보다 빠르다" → 정확한 조건 (어떤 모델, 어떤 batch size, 어떤 hardware에서)을 명시한다.

#### 약한 결과 처리 전략

**방법 1: reasoning 제공**
```
(X) "Workload G에서는 오히려 5% 성능이 떨어지는데, 이는 어쩔 수 없는 한계이다."
    → 변명 톤

(O) "Workload G에서는 이득이 제한적인데, 이는 해당 workload의 expert 활용 패턴이
     균일하여 재배치의 여지가 적기 때문이다."
    → 분석 톤
```

**방법 2: workload grouping으로 맥락화**
```
(O) "Memory-intensive workload 그룹에서 평균 35%,
     compute-intensive 그룹에서 평균 12%의 향상을 보인다."
```

**방법 3: Setup에서 적용 범위 명시**
```
(O) [Setup] "제안 기법은 expert 간 활용 빈도 차이가 클수록 효과적이다."
    [Results] "예상대로, 활용 패턴 편차가 큰 workload에서 가장 큰 향상을 보인다."
```

#### Bullet Level 가이드

```
Evaluation
├── [Subsection] Experimental Setup
│   ├── Hardware/Software 환경
│   ├── Dataset / Workload
│   ├── Baseline 기법들
│   └── Evaluation metrics
├── [Subsection] 주요 결과 (가장 인상적인 것 먼저)
│   ├── 결과 1: 전체 성능 비교 (Figure/Table 참조)
│   └── 핵심 takeaway
├── [Subsection] Ablation Study
│   ├── Component A의 효과
│   └── Component B의 효과
├── [Subsection] Sensitivity Analysis
└── [Subsection] Overhead Discussion (간략하게)
```

</details>

### Conclusion

- **Introduction과 다르게 쓴다.** Intro = novelty 강조, Conclusion = 결과의 의의 강조.
- 가장 인상적인 수치 1~2개만 언급한다 (전체 나열 금지).
- **Future Work 반드시 포함.** Limitation을 직접 나열하지 말고, 기존 기법과 orthogonal하게 적용 가능함을 강조하거나, 확장 가능성(opportunity)으로 자연스럽게 전환한다.

<details>
<summary><strong>📖 상세: Conclusion 작성 예시 + Limitation 처리 전략 + Bullet Level 가이드</strong></summary>

#### Conclusion ≠ Introduction 복사

```
(X) Introduction의 contribution 문장을 그대로 가져오는 것
(O) Introduction에서는 "무엇을 제안했는가 (novelty)"를 강조하고,
    Conclusion에서는 "그 결과가 가지는 의의 (impact)"를 강조한다
```

예시:
- **Introduction**: "본 논문에서는 expert 활용 패턴의 temporal locality를 활용한 adaptive placement 알고리즘을 제안한다."
- **Conclusion**: "실험 결과, adaptive placement를 통해 기존 GPU-only 대비 최대 2.1배의 throughput 향상을 달성하였으며, 이는 MoE 모델의 실용적 배포에 있어 이기종 시스템 활용의 가능성을 보여준다."

#### Limitation 처리 전략

Conclusion에서 limitation을 직접적으로 나열하는 것은 위험하다. 대신 **확장 가능성이나 future work로 자연스럽게 전환**한다.

```
(X) "본 연구는 single-GPU 환경만을 대상으로 하였으며, multi-GPU 환경에서는
     검증하지 못하였다."
    → 약점을 직접 노출

(O) "제안 기법은 기존의 multi-GPU 병렬화 기법과 orthogonal하게 적용 가능하며,
     향후 multi-GPU 환경으로의 확장을 통해 추가적인 성능 향상이 기대된다."
    → 한계를 확장 가능성으로 전환

(O) "본 연구는 MoE 모델에 초점을 맞추었으나, 제안하는 adaptive placement의
     핵심 원리는 다른 sparse model에도 적용 가능하다."
    → 한계를 범용성 논의로 전환
```

**핵심 원칙:** 약점을 나열하는 것이 아니라, **남은 기회(opportunity)**로 프레이밍하는 것이 핵심이다.

#### Bullet Level 가이드

```
Conclusion
├── [1~2문단] Contribution 요약 (결과/의의 관점)
│   └── 핵심 결과 수치 (대표 1~2개만)
└── [1문단] Future Work + 확장 가능성
    ├── 기존 기법과의 orthogonality / 호환성 언급
    └── 구체적인 확장 방향 제시 (limitation을 opportunity로 전환)
```

</details>

---

## 3. Reviewer를 고려한 자기 검토

**논문은 worst-case reviewer를 만족시키도록 써야 한다.** Best-case reviewer(충분한 시간, 전문가, 호의적)라도 완벽한 설명, 충분한 background, 논리적 완결성은 필요로 한다. 그러나 모든 reviewer가 이상적이지는 않으므로, 아래 세 가지 worst-case를 기준으로 검토한다.

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

<details>
<summary><strong>📖 상세: Reviewer 유형별 상세 검토 질문 + Section별 예상 질문표</strong></summary>

#### 모든 것을 아는 Reviewer — 상세 검토

> "이 reviewer는 관련 연구를 모조리 읽었다."

- [ ] 모든 직접 관련 연구를 인용했는가?
- [ ] 간접적으로라도 관련되어 보이는 연구를 인용했는가?
- [ ] 인용한 논문의 정확한 버전을 인용했는가? (arXiv vs. conference version)
- [ ] 기존 연구와의 차별점이 명확히 서술되어 있는가?

#### 아무것도 모르는 Reviewer — 상세 검토

> "이 reviewer는 해당 분야의 배경 지식이 부족하지만, 본인은 안다고 생각한다."

- [ ] 최근 5년 이내의 새로운 기술에 의존하는가? → 설명이 있는가?
- [ ] 기술을 색다른 방식으로 조합하여 사용하는가? → 설명이 있는가?
- [ ] 논문이 self-contained인가?

#### 피곤하고 짜증난 Reviewer — 상세 검토

> "이 reviewer는 15편의 논문을 이틀 안에 읽어야 하며, reject 사유를 찾고 있다."

- [ ] 논문의 구조가 명확하고 예측 가능한가?
- [ ] 문법 오류와 typo가 없는가?
- [ ] Contribution이 Introduction에서 bullet point로 명확히 제시되는가?

#### Section별 Reviewer 예상 질문

**Abstract / Introduction:**

| Reviewer 질문 | 확인할 것 |
|-------------|----------|
| "이게 왜 중요한데?" | Motivation이 충분한가? |
| "이전 연구와 뭐가 다른데?" | Novelty가 명확한가? |
| "이건 trivial한 거 아닌가?" | 왜 어려운지 (Y)가 설명되어 있는가? |

**Method:**

| Reviewer 질문 | 확인할 것 |
|-------------|----------|
| "왜 이렇게 설계했는데?" | Design rationale이 있는가? |
| "다른 방법은 고려 안 했는가?" | Alternative approach를 논의했는가? |
| "이건 이전 연구 X와 같은 거 아닌가?" | 차별점이 명확한가? |

**Evaluation:**

| Reviewer 질문 | 확인할 것 |
|-------------|----------|
| "Baseline이 공정한가?" | 동일 framework, 동일 조건 비교인가? |
| "Ablation은?" | 자명하지 않은 설계 결정에 ablation이 있는가? |
| "Overhead는?" | Overhead 분석이 포함되어 있는가? |
| "Scalability는?" | 다양한 규모에서 테스트했는가? |

#### Strong Claim 체크리스트

- [ ] "X는 Y보다 빠르다" → **어떤 조건에서?**
- [ ] "최초로 ~를 제안한다" → **정말 최초인가?**
- [ ] "기존 연구는 ~를 고려하지 않았다" → **정말 없는가?**
- [ ] "significant improvement" → **구체적 수치는?**

</details>

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

<details>
<summary><strong>📖 상세: 각 실수별 나쁜 예 / 좋은 예</strong></summary>

### 실수 1: 핵심 결정이 하위 bullet에 묻힘
```
(X) 3.2.1절의 세 번째 문단에서 갑자기 핵심 알고리즘이 등장
(O) 3절 첫 문단에서 핵심 아이디어를 제시하고, 이후 상세 설명으로 전개
```

### 실수 3: 한 문장에 너무 많은 개념 압축 + 주어·목적어 실종

긴 문장일수록 주어와 목적어가 사라지기 쉽다. **문장을 나누되, 나눈 각 문장에서 "누가 무엇을 어떻게 한다"가 명확해야 한다.**
```
(X) "제안하는 기법은 expert의 활용 빈도를 실시간으로 모니터링하여
     temporal locality를 분석하고, 이를 기반으로 CPU-GPU 간 expert
     migration 비용을 고려한 ILP를 풀어 최적의 배치를 결정하며,
     동시에 pipeline stall을 최소화하기 위한 prefetching을 수행한다."

(O) "제안하는 기법의 핵심은 expert 활용 빈도의 temporal locality를
     활용하는 것이다. 이를 위해 먼저 실시간 모니터링으로 활용 패턴을
     분석한다. 분석된 패턴을 기반으로, migration 비용을 고려한 ILP를
     통해 최적의 CPU-GPU 배치를 결정한다."
```

### 실수 4: 모호한 주장
```
(X) "Expert의 배치는 사전에 결정된다."
    → 언제? 어떤 기준으로? 누가?

(O) "Expert의 배치는 inference 시작 전 profiling 단계에서, 각 expert의
     평균 활용 빈도를 기준으로 결정된다."
```

### 실수 6: Related Work에서 비난
```
(X) "이전 연구 [5]는 이 중요한 문제를 완전히 간과하였다."
(O) "이전 연구 [5]는 A 문제를 효과적으로 해결하였으나, B 측면은
     고려하지 않았다. 본 논문에서는 이 B 측면까지 함께 해결한다."
```

### 실수 7: 숫자 나열식 결과 서술
```
(X) "Workload A에서 15%, B에서 23%, C에서 31%, D에서 18%,
     E에서 27%, F에서 42%의 향상을 보인다."

(O) "Figure 5에서 보이는 바와 같이, 제안 기법은 평균 26%의 성능 향상을
     달성하며, 특히 memory-intensive workload (F)에서 최대 42%까지
     향상된다."
```

### 실수 11: 과장 표현
```
(X) "GPU stall을 원천적으로 제거한다."
    → 실제 결과가 1.0x가 아니면 거짓

(O) "GPU stall을 대부분 제거하여 near-optimal throughput을 달성한다."
```

</details>

---

## 5. Outline 자기 검토 체크리스트

<details>
<summary><strong>📖 전체 체크리스트 펼치기</strong></summary>

### Bullet Level 검증
- [ ] 핵심 결정/주장이 하위 bullet에 묻혀 있지 않은가?
- [ ] 각 bullet은 정확히 하나의 아이디어만 담고 있는가?
- [ ] Bullet의 계층 구조가 논리적인가? (상위 bullet이 하위 bullet을 포괄하는가?)
- [ ] 한 section 내에서 bullet 수가 적절한가? (너무 많으면 subsection으로 분리)
- [ ] 같은 section 내에서 동일한 내용을 중복 서술하고 있지 않은가? (특히 contribution의 상위/하위 bullet 간 중복에 주의)

### Logic Flow 검증
- [ ] 각 section 간의 전환이 자연스러운가?
- [ ] Section A에서 제기한 문제가 Section B에서 해결되는가?
- [ ] "한편", "또한" 등의 약한 연결어에 과도하게 의존하고 있지 않은가?
- [ ] 각 paragraph의 첫 문장만 이어 읽었을 때 전체 논리가 통하는가?
- [ ] Introduction의 논리 흐름이 빠짐없이 있는가?

### 문장 복잡도 검증
- [ ] 한 bullet에 4개 이상의 개념이 압축되어 있지 않은가?
- [ ] 모호한 주장이 없는가?
- [ ] 모든 strong claim에 구체적 근거가 있는가?

### 주장-근거 검증
- [ ] 모든 주장에 근거(실험, 인용, 분석)가 outline에 포함되어 있는가?
- [ ] 근거 없는 주관적 주장 (보통 형용사가 red flag)이 없는가?
- [ ] 인용으로 뒷받침해야 할 claim이 빠져 있지 않은가?

### 그림/표 배치 검증
- [ ] Figure 1이 첫 페이지에 배치될 수 있는가?
- [ ] 각 주요 결과에 대응하는 figure/table이 계획되어 있는가?
- [ ] Overview figure가 있는가?
- [ ] 그림 없이 텍스트로만 설명하는 부분이 너무 길지 않은가?

### 글쓰기 기본 사항
- [ ] 동사 시제: 과거형은 Introduction/Related Work에서만 사용
- [ ] 주어 일관성: 같은 paragraph 내에서 주어를 자주 바꾸지 않는가? (특히 "We"와 시스템명을 혼용하면 행위 주체가 불명확해질 수 있다)
- [ ] 핵심 용어가 전체에서 통일되어 있는가?
- [ ] Filler words를 제거했는가 (can, in order to, shall 등)
- [ ] 첫 draft 작성 후 약 1/3을 삭제할 수 있는가? (그만큼 불필요한 내용이 있다는 의미)
- [ ] LaTeX 인용: 저자가 주어일 때 `\citet{}`, 아닐 때 `~\citep{}`
- [ ] Acronym을 정의 전에 사용하지 않았는가
- [ ] 인용 시 arXiv 버전이 아닌 conference/journal 버전을 인용했는가

</details>

---

## 6. 그림 및 표 계획

<details>
<summary><strong>📖 그림 배치 원칙 + 표 vs 그림 선택 기준 + 그래프 Story + 체크리스트</strong></summary>

### 언제 그림이 필요한가

다음 중 하나에 해당하면 그림이 필요하다:

- **시스템 구조/pipeline**: Overview figure는 거의 필수다
- **비교 결과**: 여러 기법의 성능 비교는 그래프가 텍스트보다 효과적이다
- **동작 과정**: 시간 순서가 있는 알고리즘은 다이어그램이 이해를 돕는다
- **데이터 분포/패턴**: Motivation에서 문제를 보여줄 때 효과적이다

### 그림 배치 원칙

| 그림 | 위치 | 목적 |
|------|------|------|
| **Figure 1** (Teaser) | 첫 페이지 | 핵심 아이디어의 직관적 전달 |
| Overview figure | Method section 시작 | 전체 framework의 big picture |
| Motivation figure | Background/Motivation | 문제의 심각성을 데이터로 보여줌 |
| 결과 figure | Evaluation | Baseline 대비 성능 비교 |
| Ablation figure | Evaluation | 각 component의 기여도 |

### 표 vs 그림 선택 기준

| 상황 | 추천 |
|------|------|
| 정확한 수치 비교가 중요할 때 | **표** |
| 전반적인 추세/경향을 보여줄 때 | **그림** |
| 기법 간 기능 비교 (있다/없다) | **표** (checkmark table) |
| 분포나 패턴을 보여줄 때 | **그림** (scatter, heatmap) |
| 실험 setup 요약 | **표** |

### 그래프는 데이터가 아니라 Story를 보여줘야 한다

그래프는 단순히 숫자를 시각화하는 것이 아니다. **Reader가 그래프를 보고 즉시 "이 논문의 기법이 왜 좋은가"를 파악할 수 있어야 한다.**

- **Baseline 기준선을 시각적으로 표시한다.** 예: 1.0x 위치에 점선을 긋고 "GPU-only baseline" 라벨을 붙인다. 기준선이 없으면 reader가 수치의 의미를 즉시 파악하지 못한다.
- **가장 중요한 takeaway를 그래프 안에 annotation으로 표시한다.** 예: "최대 2.1x" 화살표, 특정 구간에 대한 설명 라벨 등.
- **그래프에서 실제 결과보다 과장된 인상을 주지 않는다.** Reviewer는 그래프와 숫자를 대조한다.

### Figure와 본문의 순서를 일치시킬 것

그림에서 A → B 순서로 보여주면, 본문에서도 A → B 순서로 설명한다. 그림은 Prefill → Decoding 순인데 본문이 Decoding → Prefill 순으로 설명하면 reader가 혼란스럽다.

이 원칙은 multi-panel figure (a), (b), (c)에도 적용된다. 본문에서 (b)를 먼저 설명하고 (a)를 나중에 설명하는 것은 피한다.

### 그림 분석 체크리스트 (Outline 단계)

Outline 단계에서 각 계획된 그림에 대해 다음을 판단한다:

- **추가해야 할 그림**: 텍스트만으로 설명이 어려운 부분이 있는가?
- **수정해야 할 그림**: 기존 그림이 메시지를 명확히 전달하는가?
- **이동해야 할 그림**: 이 그림이 현재 위치에서 가장 효과적인가?
- **삭제해야 할 그림**: 이 그림이 없어도 이해에 문제가 없는가? (공간이 부족할 때)

</details>

---

## 7. 분야별 특수 고려사항 (Computer Architecture / AI Systems)

<details>
<summary><strong>📖 HW 환경 명시, Baseline Fairness, Latency Context 등</strong></summary>

### 하드웨어 환경 명시 필수
- GPU 모델, 수량, 메모리 용량
- CPU 모델, 코어 수, 메모리 용량
- 인터커넥트 (NVLink, PCIe 버전 등)
- CUDA/cuDNN/PyTorch 버전

### Baseline Fairness
**같은 framework, 같은 조건에서 비교한다.**
```
(X) "우리의 PyTorch 구현이 논문에서 보고된 TensorFlow 결과보다 빠르다"
(O) "동일한 PyTorch 환경에서 baseline을 재구현하여 비교하였다"
```

- Baseline을 직접 재구현했다면, 원 논문의 결과와 재구현 결과의 차이를 보고한다
- 가능하면 공식 코드를 사용하되, 동일한 하드웨어에서 실행한다

### Latency/Throughput에 Context 필요

| 빠짐없이 명시할 것 | 예시 |
|------------------|------|
| Model 정보 | LLaMA-MoE 7B, 8 experts |
| Batch size | 32 |
| Sequence length | 2048 tokens |
| Precision | FP16 / BF16 / INT8 |
| 측정 방식 | Warm-up 후 100회 반복의 평균 |

### Ablation Study
**자명하지 않은 설계 결정** — 포함/제외에 따라 결과가 좋아질 수도 나빠질 수도 있는 요소 — 에 대해 ablation을 수행한다. 누구나 동의할 표준적 설계까지 ablation할 필요는 없다.

### Overhead는 별도 Discussion으로
- Method section에서 언급하지 않는다 (novelty에 집중)
- Evaluation의 마지막에서 간략히 다룬다
- 가능하면 표 하나로 요약한다
- Overhead가 불가피하다면, 그로 인한 이득이 충분히 크다는 것을 보여준다

</details>

---

## 8. Outline 작성 순서

```
Contribution 정리 → Method → Evaluation → Related Work → Background → Introduction → Conclusion → Abstract
```

Abstract는 가장 마지막에 확정하되, **일찍 draft를 써두면 story의 허점을 빨리 발견할 수 있다.**

---

## 부록: 글쓰기 기본 규칙 체크리스트

<details>
<summary><strong>📖 제출 전 최종 점검 리스트</strong></summary>

- [ ] 맞춤법/문법 검사를 여러 번 실행했는가 (Word grammar checker 포함)
- [ ] 3인칭 단수 동사에 -s를 빠뜨리지 않았는가
- [ ] 조동사 뒤에 동사원형을 사용했는가
- [ ] 과거형을 Intro/Related Work 이외에서 사용하지 않았는가
- [ ] 수동태를 과도하게 사용하지 않았는가
- [ ] Filler words를 제거했는가 (can, in order to, shall, ...)
- [ ] 동일 paragraph 내에서 단어 반복이 없는가
- [ ] LaTeX 인용: 저자가 주어일 때 `\citet{}`, 아닐 때 `~\citep{}`
- [ ] LaTeX 따옴표: ` ``correct'' ` 형식 사용
- [ ] 수식 뒤에 마침표/쉼표를 빠뜨리지 않았는가
- [ ] Broken reference/citation (**??**)이 없는가
- [ ] Acronym을 정의 전에 사용하지 않았는가
- [ ] 인용 시 arXiv 버전이 아닌 conference/journal 버전을 인용했는가
- [ ] Page limit을 초과하지 않았는가

</details>

---

## 부록: 한글로 Outline 작성 시 주의사항

- **한국어식 지시어를 기술 용어로 대체한다.** "바깥" → "non-top-k", "양쪽" → "prefill과 decoding 모두", "이를" → 구체적 대상 명시.
- **한글 bullet이 3줄 이상이면 영어로 쓸 때 한 문장이 너무 길어진다.** Outline 단계에서 미리 나눈다.
- **한국어 동사 후치 구조에 주의한다.** 긴 수식절이 동사 앞에 쌓이면 영어 전환 시 읽기 어려운 문장이 됨.

<details>
<summary><strong>📖 상세: 한국어 문장 구조의 영어 전환 예시</strong></summary>

| 한국어 습관 | 문제 | 수정 |
|------------|------|------|
| "Top-k **바깥**의 expert" | "바깥"이 영어로 어색 | "top-k+α 범위의 expert" 또는 구체적으로 풀어쓴다 |
| "**양쪽** phase에서" | 무엇을 가리키는지 불명확 | "Prefill과 decoding phase 모두에서" |
| "**이를** 기반으로 최적화" | "이"가 무엇인지 모호 | "Profiling 결과를 기반으로 배치를 최적화" |

```
한국어 outline:
"Expert의 활용 빈도를 실시간으로 모니터링하여 temporal locality를
분석하고, 이를 기반으로 CPU-GPU 간 migration 비용을 고려한
ILP를 풀어 최적의 배치를 결정한다."
→ 동사가 끝에 와서 한국어로는 자연스럽지만, 영어로 옮기면 한 문장에 4개 이상 개념

수정:
"제안 기법은 expert 활용 빈도의 temporal locality를 분석한다.
분석 결과를 기반으로, migration 비용을 고려한 ILP를 통해
최적의 CPU-GPU 배치를 결정한다."
→ 한 문장 = 한 행위
```

</details>

---

> **마지막으로**: 좋은 논문은 좋은 연구에서 나오지만, 좋은 연구도 잘 쓰지 않으면 impact를 잃는다.
> Outline에 충분한 시간을 투자하라. 그것이 결국 전체 작성 시간을 줄여준다.
