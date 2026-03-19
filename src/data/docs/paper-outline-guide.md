# 논문 Outline 작성 가이드라인

**작성: 공영호 | 대상: 석/박사과정 학생**

> Outline은 건물의 설계도다. 설계도 없이 건물을 짓는 사람은 없다.
> 논문도 마찬가지다. Outline 단계에서 구조를 확실히 잡아야 나중에 수정이 쉽다.

---

## 목차

0. [모든 Section을 관통하는 원칙: 두괄식 서술](#0-모든-section을-관통하는-원칙-두괄식-서술)
1. [서론: Outline이 중요한 이유](#1-서론-outline이-중요한-이유)
2. [논문 작성 전 준비사항](#2-논문-작성-전-준비사항)
3. [Section별 Outline 작성법](#3-section별-outline-작성법)
4. [Outline 자기 검토 체크리스트](#4-outline-자기-검토-체크리스트)
5. [그림 및 표 계획](#5-그림-및-표-계획)
6. [Reviewer를 고려한 Outline 검토](#6-reviewer를-고려한-outline-검토)
7. [분야별 특수 고려사항 (Computer Architecture / AI Systems)](#7-분야별-특수-고려사항-computer-architecture--ai-systems)
8. [흔한 실수 Top 12](#8-흔한-실수-top-12)

---

## 0. 모든 Section을 관통하는 원칙: 두괄식 서술

**중요한 것부터 먼저 쓴다.** 이것이 논문 writing의 가장 기본적인 원칙이다.

한국어 화법에서는 "말은 끝까지 들어봐야 안다"고 하지만, 논문은 정반대다. **핵심이 처음에 나와야 한다.** 발표자료에서 Executive Summary가 맨 앞에 오는 것과 같은 이유다. Reviewer는 피곤하고, 독자는 바쁘다. 핵심을 먼저 보여주지 않으면, 끝까지 읽어주지 않는다.

### 이 원칙이 적용되는 모든 단위

| 단위 | 적용 | 나쁜 예 | 좋은 예 |
|------|------|---------|---------|
| **논문 전체** | Abstract → Introduction에서 핵심 기여와 결과를 먼저 제시 | 배경 설명이 2페이지 이후에야 contribution 등장 | 첫 페이지 안에 contribution과 Figure 1 |
| **각 Section** | 해당 section의 핵심 메시지를 첫 문단에서 제시 | Method section이 배경 설명으로 시작 | "제안 기법의 핵심은 ~이다"로 시작 |
| **각 문단** | 첫 문장(topic sentence)이 그 문단의 요지 | 세부 내용을 나열한 뒤 마지막에 요약 | 결론을 먼저 제시하고 근거를 이어서 설명 |
| **결과 서술** | 가장 인상적인 결과를 먼저, 세부 분석은 그 다음 | 작은 결과부터 나열하여 클라이맥스 구조 | 가장 큰 성과를 첫 subsection에 배치 |
| **Contribution** | 가장 중요한 기여를 첫 번째 bullet으로 | 부수적 기여가 먼저, 핵심이 마지막 | 핵심 알고리즘 제안이 C1 |
| **Related Work** | 각 문단의 첫 문장이 해당 카테고리의 main idea | 개별 논문 소개로 시작 | "~를 위한 기법들이 제안되어 왔다"로 시작 |

### 자가 점검법

**각 section과 문단의 첫 문장만 이어서 읽어본다.** 그것만으로 논문의 전체 논리가 통하면 잘 쓴 것이다. 통하지 않으면, 핵심이 문단 중간이나 끝에 묻혀 있다는 뜻이다.

```
[Test] Introduction 첫 문장들만 이어 읽기:

"대규모 언어 모델의 추론 효율성이 중요해지고 있다."
→ "MoE는 이를 위한 대표적 기법이지만, expert 배치 문제가 있다."
→ "기존 연구는 static placement에 의존하여 workload 변화에 대응하지 못한다."
→ "본 논문에서는 adaptive expert placement를 제안한다."
→ 논리가 통한다 ✓
```

이 원칙은 이후 모든 section별 작성법에서 반복적으로 적용된다.

### 두괄식과 함께 지켜야 할 문장 원칙: 주어·목적어·동사를 명확히

두괄식 서술만큼 중요한 것이 **문장 단위의 명확성**이다. 한국어 논문 초고에서 가장 흔한 문제는 주어, 목적어, 동사가 불명확한 문장이다.

**원칙: 모든 문장에서 "누가(주어) — 무엇을(목적어) — 어떻게 한다(동사)"가 한눈에 보여야 한다.**

영어로 논문을 쓰는 것이지만, outline 단계에서 한국어로 메모할 때도 이 습관을 들여야 한다. 한국어에서는 주어와 목적어를 생략하기 쉽고, 동사가 문장 끝에 오기 때문에 의미가 뒤늦게 전달된다. 논문에서는 이것이 치명적이다.

```
(X) "활용 패턴을 분석하여 효율적으로 처리한다."
    → 누가 분석하는가? 무엇을 처리하는가? 어떻게 효율적인가?

(O) "제안 기법은 expert의 활용 패턴을 실시간으로 분석하고,
     분석 결과를 기반으로 expert를 GPU에 동적 배치한다."
    → 주어(제안 기법), 목적어(활용 패턴, expert), 동사(분석한다, 배치한다) 모두 명확
```

이 원칙이 중요한 이유는 단순히 문장 하나의 가독성 때문이 아니다. **각 문장의 의도가 뚜렷해지면, 그 문장들이 모인 문단에서 무엇에 집중하는지가 자연스럽게 드러난다.** 반대로, 주어·목적어가 흐릿한 문장이 쌓이면 문단 전체가 무엇을 말하려는지 알 수 없게 된다.

| 확인 항목 | 나쁜 예 | 좋은 예 |
|-----------|---------|---------|
| **주어 누락** | "분석을 통해 성능이 향상된다" | "제안 기법은 profiling을 통해 throughput을 향상시킨다" |
| **목적어 불분명** | "이를 기반으로 최적화한다" | "profiling 결과를 기반으로 expert 배치를 최적화한다" |
| **동사 모호** | "~에 대해 고려한다" | "~의 overhead를 측정하고, trade-off를 분석한다" |

> **자가 점검법**: 각 문장을 읽고 "**누가, 무엇을, 어떻게**"를 즉시 답할 수 있는가? 답할 수 없다면 문장을 다시 쓴다.

### Claim의 강도를 evidence에 맞출 것

**논문에서 가장 위험한 것은 과장된 표현(overclaiming)이다.** Reviewer는 claim의 강도와 실제 evidence 사이의 괴리를 즉시 감지하며, 이는 탑티어 학회에서 rejection의 대표적 사유 중 하나다.

핵심 원칙: **모든 claim은 실험 결과가 뒷받침할 수 있는 범위 안에서만 서술한다.**

```
(X) "GPU stall을 원천적으로 제거한다."
    → "원천적으로 제거"하면 GPU-only 대비 1.0x가 되어야 하는데, 실제로 그렇지 않다면?
    → Reviewer: "Figure 3을 보면 stall이 완전히 제거되지 않았는데, 이 claim은 거짓이다."

(O) "GPU stall을 대부분 제거하여 near-optimal throughput을 달성한다."
    → 결과와 일치하는 정직한 표현
```

| 위험한 표현 | 왜 위험한가 | 대안 |
|------------|-----------|------|
| "원천적으로 제거" | 100% 제거가 아니면 거짓 | "대부분 제거", "최소화" |
| "완전히 해소" | 반례 하나로 무너짐 | "크게 완화", "효과적으로 해소" |
| "세계 최초" | 누락된 선행연구 하나로 무너짐 | "최초"를 쓰려면 범위를 한정: "A에 대해 B를 적용한 최초의 연구" |
| "significant improvement" | 구체적 수치 없으면 의미 없음 | "Up to 2.1x throughput 향상" |
| "크게/현저히 향상" | 구체적 수치를 써야 함 | 정확한 배수 또는 % |

**이런 표현은 reviewer의 "먹잇감"이 된다.** 과장된 claim은 reviewer가 reject 사유로 삼기 가장 쉬운 대상이다. Tone을 약간 낮추는 것이 논문의 신뢰도를 높인다.

> **자가 점검법**: 각 claim에 대해 "이 표현을 뒷받침하는 정확한 수치가 어디 있는가?"를 물어본다. 답할 수 없으면 표현을 수정한다.

### 논문 전체에서 설명 순서를 일관되게 유지할 것

**Introduction에서 Contribution A → B → C 순서로 소개했다면, Method에서도, Evaluation에서도, Conclusion에서도 동일한 순서를 따른다.**

```
(X) Introduction: ACE caching → Time-budgeted skipping
    Method: Time-budgeted skipping → ACE caching
    → 순서가 뒤바뀌면 reader가 혼란

(O) Introduction: ACE caching → Time-budgeted skipping
    Method: ACE caching → Time-budgeted skipping
    → 동일한 순서. Reader가 자연스럽게 따라감
```

이 원칙은 다음 단위에 모두 적용된다:

| 단위 | 적용 |
|------|------|
| Contribution 순서 | Introduction에서 나열한 순서 = Method subsection 순서 = Evaluation 순서 |
| Figure vs 본문 | 그림에서 A → B 순서라면, 본문에서도 A → B 순서로 설명 |
| Phase 순서 | Prefill → Decoding이 실제 실행 순서라면, 설명도 그 순서를 따름 |

> **자가 점검법**: Outline에서 각 section의 subsection 순서를 나란히 놓고, 순서가 일관되는지 확인한다.

---

## 1. 서론: Outline이 중요한 이유

### Outline이란 무엇인가

Outline은 논문의 뼈대다. 각 section에 어떤 내용이 들어가고, 어떤 순서로 논리가 전개되는지를 한눈에 보여주는 구조도다.

**핵심 원칙: 한 줄 = 하나의 아이디어 = 최종 논문의 한 문단**

Outline의 각 bullet은 나중에 하나의 paragraph로 확장된다. 따라서 outline을 쓸 때부터 "이 bullet은 어떤 하나의 메시지를 전달하는가?"를 스스로 물어야 한다.

### 왜 Outline부터 써야 하는가

- **구조 변경이 쉽다**: 완성된 글의 section 순서를 바꾸는 것은 고통스럽다. Outline에서는 bullet을 드래그하면 끝이다.
- **논리적 허점이 보인다**: Outline 단계에서 "이 주장의 근거가 어디에 있지?"를 확인할 수 있다.
- **공동 작업이 효율적이다**: 지도교수와 outline을 먼저 합의하면, 나중에 전체 rewrite하는 비극을 피할 수 있다.
- **분량 조절이 가능하다**: Bullet 수를 세면 예상 page 수를 추정할 수 있다 (대략 bullet 3~4개 = 1 column).

### Outline 작성 후 확장하는 법

Outline이 확정되면, 각 bullet 위에 LaTeX 주석으로 요약을 남기고 그 아래에 paragraph를 확장한다.

```latex
% TL;DR: 기존 MoE 라우팅은 load balancing만 고려하여 전문가 활용도가 낮다
기존 Mixture-of-Experts (MoE) 모델의 라우팅 알고리즘은 주로 load balancing에
초점을 맞추어 설계되어 왔다. 그러나 이러한 접근은 각 expert의 실제 전문성을
고려하지 않아, 결과적으로 expert utilization이 낮아지는 문제를 야기한다.
```

이렇게 하면 (a) 글을 쓰면서 원래 의도에서 벗어나지 않고, (b) 피드백을 줄 때 전체 흐름을 빠르게 파악할 수 있다.

---

## 2. 논문 작성 전 준비사항

### 2.1 Contribution 정리

**논문 쓰기 전에 contribution을 명확히 정리하고, 확실히 정리된 상태에서 작성을 시작한다.**

Contribution을 bullet point로 작성한다. 이 bullet point는 나중에 Introduction의 contribution list가 된다.

```
- (C1) CPU-GPU 이기종 시스템에서 MoE expert를 동적으로 배치하는
       adaptive placement 알고리즘 제안
- (C2) Expert 활용 패턴을 사전 분석하여 placement를 최적화하는
       profiling 기법 개발
- (C3) 다양한 MoE 모델에서 기존 대비 최대 2.1배 throughput 향상 검증
```

**주의: 반복 = 강조가 아니다!**

Contribution을 여러 section에서 강조해야 하지만, **동일한 문장을 복사-붙여넣기 하는 것은 강조가 아니라 게으름이다.** Introduction에서는 novelty 관점에서, Conclusion에서는 결과와 의의 관점에서 다른 방식으로 서술해야 한다.

### 2.2 타겟 Venue 분석

Outline을 쓰기 전에 반드시 확인할 것:

| 항목 | 확인 내용 |
|------|----------|
| Page limit | 본문 몇 페이지? Reference 포함/미포함? |
| Format | Single/double column? Template 종류? |
| 기대하는 결과 | 해당 venue의 최근 논문들이 어떤 실험을 포함하는지 확인 |
| Related work 위치 | Introduction 다음? Evaluation 다음? (venue마다 관행이 다름) |
| Supplementary | 허용 여부, 분량 제한 |

### 2.3 핵심 용어 정의 및 통일

- **논문 전체에서 핵심 용어는 하나로만 사용한다.** 예: "refresh interval"과 "refresh period"를 혼용하지 않는다.
- 독자가 헷갈릴 수 있는 용어는 첫 등장 시 정의하고, 필요하면 footnote로 명확히 한다.
- **새로운 용어를 만드는 것은 가급적 피한다.** Google Scholar에서 해당 분야 주요 논문들이 쓰는 표현을 확인한다.
- Acronym은 반드시 첫 등장 시 full name과 함께 정의하고, 정의한 후에만 사용한다.

### 2.4 표현 검증 습관

어떤 표현을 쓰기 전에 [Google Scholar](http://scholar.google.co.kr/)에서 검색한다.

- 검색 시 따옴표로 묶어서 exact match 검색: `"As we mentioned"`
- 검색도구에서 `"Transactions on"` 또는 `"IEEE International Conference on"`을 소스 필터로 넣으면, 학술 논문에서 실제로 쓰이는 표현인지 확인 가능
- 검색 결과가 거의 없는 표현은 사용하지 않는다

---

## 3. Section별 Outline 작성법

### 3.1 Abstract

#### 목적과 역할
- 논문 전체의 TL;DR. **모든 독자가 가장 먼저 읽는 부분**이다.
- Stand-alone이어야 한다: Abstract만 읽어도 논문의 핵심을 파악할 수 있어야 한다.
- 흥미를 유발하고, 연구의 필요성이 드러나야 한다.

#### 필수 포함 요소 (X-Y-Z 구조)

| 요소 | 내용 | Outline bullet 예시 |
|------|------|---------------------|
| **X** (What + Why) | 무엇을 하려 하고, 왜 중요한가 | "MoE 모델의 추론 효율성은 expert placement에 크게 좌우됨" |
| **Y** (Why hard) | 왜 어려운가 | "기존 static placement는 workload 변화에 대응 불가" |
| **Z** (Contribution) | 우리의 기여 | "adaptive expert placement 알고리즘 제안" |
| **검증** | 어떻게 검증했는가 | "3개 MoE 모델에서 최대 2.1배 throughput 향상" |

#### 흔한 실수와 주의사항
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

#### 예시

**나쁜 예:**
```
최근 대규모 언어 모델이 다양한 분야에서 활용되고 있다. Mixture-of-Experts는
이러한 모델의 효율성을 높이는 기법이다. 본 논문에서는 새로운 기법을 제안한다.
실험 결과 성능이 향상되었다.
```
문제점: X-Y-Z 중 Y(왜 어려운가)가 없고, 구체적 수치가 없으며, "새로운 기법"이 무엇인지 알 수 없다.

**좋은 예:**
```
Mixture-of-Experts (MoE) 모델의 추론 효율성은 expert의 물리적 배치에 크게
좌우된다 (X). 그러나 기존 static placement 방식은 입력 분포의 변화에 대응하지
못하여, 특정 expert에 부하가 집중되는 문제를 야기한다 (Y). 본 논문에서는
runtime에 expert 활용 패턴을 분석하고, CPU-GPU 간 expert를 동적으로
재배치하는 adaptive placement 알고리즘을 제안한다 (Z). LLaMA-MoE,
Mixtral, DeepSeek-MoE에서 평가한 결과, GPU-only baseline 대비 최대 2.1배의
throughput 향상을 달성하였다 (검증).
```

---

### 3.2 Introduction

#### 목적과 역할
- **논문 전체의 요약이다.** Abstract의 확장판이라고 생각하면 된다.
- 독자가 Introduction만 읽어도 이 논문이 무엇을, 왜, 어떻게 했는지 알 수 있어야 한다.
- Contribution을 bullet point로 명시적으로 나열한다.
- **첫 페이지에 Figure 1을 배치한다** (핵심 아이디어를 직관적으로 보여주는 그림).

#### 필수 포함 요소 (논리 흐름)

Introduction은 다음의 논리적 흐름을 따른다:

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

#### 흔한 실수와 주의사항
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

**Contribution 작성 원칙:**

- **각 contribution bullet은 그 자체로 완결적이어야 한다.** 부연 설명을 읽지 않아도 "무엇을, 왜, 어떻게"가 한 문장 안에서 전달되어야 한다. Reviewer는 contribution bullet만 읽고 novelty를 판단하는 경우가 많다.
- **Contribution은 3개 내외로 제한한다.** 5개 이상 나열하면 "unfocused" 또는 "incremental의 나열"이라는 인상을 준다.
- **Contribution이 장황해지면 focus를 잃는다.** 하위 bullet의 결과 설명은 최소화하고, 핵심 기여만 상위 레벨에서 명확히 드러낸다.

```
(X) "Expert의 중요도를 분석하여 효율적인 caching을 수행한다."
    → "효율적인 caching"이 무엇인지, "분석"이 어떤 방식인지 불명확.
    → Reviewer: "이게 contribution인가, background인가?"

(O) "Cross-layer global scoring을 통해 accuracy-critical expert를
     식별하고, 이를 GPU에 우선 배치하는 caching 정책을 제안한다."
    → 방법(cross-layer global scoring), 대상(accuracy-critical expert),
      행위(GPU 우선 배치)가 한 문장에 모두 담김
```

---

### 3.3 Background / Motivation

#### 목적과 역할
- 독자가 제안 기법을 이해하기 위해 필요한 사전 지식을 제공한다.
- **논문은 self-contained이어야 한다**: 다른 논문을 읽지 않아도 이해할 수 있어야 한다.
- Motivation section이 포함되는 경우, "왜 이 문제가 중요한가"를 데이터로 보여준다.

#### 필수 포함 요소
- 제안 기법이 의존하는 기반 기술 설명
- 문제 정의 (Problem definition): 입력, 출력, 가정, 표기법
- (Motivation 포함 시) 기존 접근의 한계를 보여주는 분석/데이터

#### 흔한 실수와 주의사항
- 교과서를 통째로 옮겨 적는 것: 제안 기법 이해에 **필요한 부분만** 설명한다.
- 반대로, 당연하다고 생각해서 중요한 배경을 생략하는 것: 본인에게 당연한 것이 reviewer에게는 아닐 수 있다.
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

---

### 3.4 Related Work

#### 목적과 역할
- 본 연구와 관련된 기존 연구를 **분류하고 비교/대조**하는 section이다.
- 단순히 기존 연구를 소개하는 것이 아니라, **본 연구의 novelty를 기존 연구와의 차이를 통해 부각**하는 것이 목적이다.

#### 필수 포함 요소
- 관련 연구의 카테고리별 분류 (2~4개 카테고리)
- 각 카테고리 내에서 주요 연구들의 핵심 아이디어 요약
- **본 연구와의 차별점** 명시
- (선택) 비교표 (comparison table with checkmarks)

#### 흔한 실수와 주의사항

**절대로 피해야 할 표현:**
```
(X) "Similar to their scheme, our proposed scheme~"
    → 이전 연구와 비슷하다면 novelty가 없어 보인다.

(O) "Different from their scheme, our proposed scheme~"
    → 차별성을 강조하여 novelty를 부각한다.
```

**추가 주의사항:**
- **각 문단의 첫 문장은 그 문단의 main idea여야 한다.** 혹은, 해당 카테고리를 소개하는 general한 문장으로 시작한다: "~를 위한 기법들이 제안되어 왔다."
- **비난하는 느낌이 들면 안 된다.** 해당 논문의 저자가 reviewer일 수 있다는 것을 항상 명심한다.
- 단순히 각 논문을 나열하는 것은 부족하다. 반드시 compare and contrast를 해야 한다.
- 예외적으로, 실험 parameter를 참고하는 경우 "동일한 parameter 설정을 사용하여 평가한다"는 표현은 허용된다.

**Related Work의 위치:**
- 본 논문의 실험에서 이전 연구의 기법과 직접 비교하는 경우: Introduction 바로 다음에 배치하여, 독자가 제안 기법을 읽기 전에 비교 대상을 인지하도록 한다.
- Conventional scheme과만 비교하고 이전 연구 기법과 직접 비교하지 않는 경우: Evaluation 다음에 배치한다.

#### Bullet Level 가이드

```
Related Work
├── [1문단] Category A: ~를 위한 기법들
│   ├── 연구 1의 핵심 아이디어 + 한계
│   ├── 연구 2의 핵심 아이디어 + 한계
│   └── "Different from these, our work~"
├── [1문단] Category B: ~를 활용한 기법들
│   ├── 연구 3의 핵심 아이디어 + 한계
│   ├── 연구 4의 핵심 아이디어 + 한계
│   └── "Our approach differs in that~"
├── [1문단] 특별히 가까운 연구 (주요 경쟁자)
│   └── 구체적 차이점 강조
└── (선택) [표] Comparison table
```

#### 작성 전략

1. 관련 논문을 모두 리스트업한다
2. 각 논문에 대해 1~2줄 요약을 쓴다
3. 카테고리로 분류한다 (2~4개)
4. 각 카테고리에 대해 paragraph를 작성한다

---

### 3.5 Proposed Method (제안 기법)

#### 목적과 역할
- 이전 연구들과 차별화된 새로운 기법을 설명하는 section이다.
- **Contribution이 무엇인지를 명확하게 인지하고** 작성한다.
- 독자가 다른 논문을 참조하지 않고도 기법을 이해할 수 있어야 한다 (self-contained).

#### 필수 포함 요소
- Overview: 전체 framework/pipeline의 big picture (overview figure 포함)
- 핵심 아이디어의 상세 설명
- 알고리즘/기법의 논리적 근거 (왜 이렇게 설계했는가)
- (선택) Pseudocode

#### 흔한 실수와 주의사항

**"Cooking recipe처럼 쓰지 말 것"**

Method section은 코드의 README가 아니다. **알고리즘을 서술하는 것이지, 코드를 설명하는 것이 아니다.** 독자가 알고 싶은 것은 핵심 아이디어이지, implementation 순서가 아니다.

```
(X) "먼저 입력을 받아 전처리한다. 그다음 모델에 넣는다. 그 출력을 후처리한다."
    → Cooking recipe

(O) "제안하는 알고리즘의 핵심은 expert 활용 패턴의 temporal locality를
     exploitation하는 것이다. 이를 위해..."
    → 아이디어 중심 서술
```

**Subsection 제목은 핵심 메시지를 담아야 한다:**

Section/subsection 제목은 내용의 핵심을 반영하는 informative title이어야 한다. 모호하거나 내용과 동떨어진 제목은 reader가 논문의 구조를 파악하기 어렵게 만든다.

```
(X) "3.2 Expert Skip이 필요해지게 되는 때"
    → 무엇이 문제인지, 어떤 방향인지 전혀 드러나지 않음

(O) "3.2 Limitation of Token-level Expert Selection"
    → 기존 방식(token-level)의 한계를 논의한다는 것이 제목에서 드러남
```

**Implementation detail은 novelty가 아니면 줄인다:**

기존 논문들에서 이미 다뤄진 저수준 구현 세부사항(e.g., fused kernel 사용, quantization scheme 선택)은 novelty가 없다면 간략히 언급하고 넘어간다. 단, **systems 분야(ISCA, MICRO 등)에서는 구현 자체가 contribution인 경우도 있으므로**, venue와 논문의 성격에 따라 판단한다.

```
(X) "단일 fused CUDA kernel을 구현하여 dequantization과 GEMM을
     하나의 kernel에서 수행하도록 하였다." (2문단에 걸쳐 상세 설명)
    → 기존 논문들이 이미 채택하는 standard practice라면 novelty 아님

(O) "Dequantization과 GEMM을 fused kernel로 처리하여 kernel launch
     overhead를 줄였다 [ref]." (한 문장으로 간결하게)
    → Novelty가 아닌 구현은 reference와 함께 간략히 언급
```

**추가 주의사항:**
- **이 section에서 overhead를 기술하지 않는다.** 논문의 목적은 제안 기법의 novelty를 알리는 것이 우선이다. Overhead는 Evaluation에서 다룬다.
- Pipeline 순서대로 설명할 필요가 없다. **핵심 component부터 먼저 설명**한다.
- Overview를 먼저 제시하고, 각 component의 상세 설명으로 들어간다.
- 한 문장에 4개 이상의 개념을 압축하지 않는다. 복잡한 내용은 여러 문장으로 나눈다.

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
│   ├── 구체적 설계
│   └── Design rationale
└── [Subsection] 통합 / 전체 알고리즘
    └── (선택) Pseudocode
```

---

### 3.6 Evaluation

#### 목적과 역할
- 제안 기법이 실제로 효과가 있음을 검증하는 section이다.
- **비슷한 논문들이 어떤 결과를 주로 사용하는지 파악하고**, 독자가 궁금해할 결과를 예상하여 작성한다.

#### 필수 포함 요소

| 항목 | 내용 |
|------|------|
| Experimental setup | HW/SW 환경, dataset, baseline, metric |
| 주요 결과 | Baseline 대비 성능 비교 (가장 인상적인 것 먼저) |
| Ablation study | 자명하지 않은 설계 결정의 효과 검증 |
| Sensitivity analysis | 주요 parameter에 대한 민감도 분석 |
| Overhead analysis | 제안 기법의 overhead (간략하게) |

#### 흔한 실수와 주의사항

- **Overhead 부분이 과하게 부각되지 않도록 한다.** 좋아지는 결과와 달리, overhead는 discussion 형식으로 기술하거나 간략한 표로 요약한다.
- **숫자를 무분별하게 나열하지 않는다.** 그림이 있으면 텍스트에서는 대표 사례만 언급한다. "Figure 5에서 보이는 바와 같이, 제안 기법은 평균 35%의 에너지 절감을 달성하며, 특히 memory-intensive workload에서 최대 52%까지 절감된다."
- **약한 결과는 분석적으로 다룬다.** 결과가 약한 케이스가 있을 때, 변명 톤("어쩔 수 없다")은 안 되지만 논리적 reasoning("이 조건에서는 ~이므로 이득이 제한적이다")은 괜찮다. 또한 workload grouping 등으로 전체 맥락 안에서 자연스럽게 제시하는 것도 효과적이다.
- **강한 주장에는 구체적 근거를 제시한다.** "GPU-only보다 빠르다" → 정확한 조건 (어떤 모델, 어떤 batch size, 어떤 hardware에서)을 명시한다.
- **Ablation은 필수이다.** 단, 모든 설계 요소에 대해 ablation이 필요한 것은 아니다. **그 요소를 포함/제외함에 따라 결과가 좋아질 수도, 나빠질 수도 있는 — 즉, 선택의 타당성이 자명하지 않은 — 설계 결정**에 대해 ablation을 수행한다.

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
│   ├── 결과 2: 특정 시나리오 분석
│   └── 핵심 takeaway
├── [Subsection] Ablation Study
│   ├── Component A의 효과
│   ├── Component B의 효과
│   └── 핵심 takeaway
├── [Subsection] Sensitivity Analysis
│   └── 주요 parameter 변화에 따른 영향
└── [Subsection] Overhead Discussion (간략하게)
    └── Area/power/latency overhead 요약
```

---

### 3.7 Conclusion

#### 목적과 역할
- 논문의 마무리. 해당 논문이 가지는 contribution의 impact를 강조한다.
- **Introduction과 다른 방식으로** contribution을 강조한다.

#### 필수 포함 요소
- Contribution 요약 (Introduction과 다른 관점에서)
- 가장 인상적인 결과 (전체가 아닌 대표 수치만)
- **Future work** (반드시 포함)

#### 흔한 실수와 주의사항

**Conclusion ≠ Introduction 복사**

```
(X) Introduction의 contribution 문장을 그대로 가져오는 것
(O) Introduction에서는 "무엇을 제안했는가 (novelty)"를 강조하고,
    Conclusion에서는 "그 결과가 가지는 의의 (impact)"를 강조한다
```

예시:
- **Introduction**: "본 논문에서는 expert 활용 패턴의 temporal locality를 활용한 adaptive placement 알고리즘을 제안한다."
- **Conclusion**: "실험 결과, adaptive placement를 통해 기존 GPU-only 대비 최대 2.1배의 throughput 향상을 달성하였으며, 이는 MoE 모델의 실용적 배포에 있어 이기종 시스템 활용의 가능성을 보여준다."

**Limitation 처리 전략:**

Conclusion에서 limitation을 직접적으로 나열하는 것은 위험하다. Reviewer에게 약점을 스스로 부각하는 꼴이 되기 때문이다. 대신, limitation을 **확장 가능성(extensibility)이나 future work로 자연스럽게 전환**하여 서술한다.

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

**핵심 원칙:** Limitation을 언급해야 할 때는, (1) 기존 기법과 orthogonal하게 적용 가능함을 강조하거나, (2) future work에서 자연스럽게 해결될 방향으로 서술하거나, (3) 현재 scope를 명확히 하되 확장 가능성을 함께 제시한다. 약점을 나열하는 것이 아니라, **남은 기회(opportunity)**로 프레이밍하는 것이 핵심이다.

**추가 주의사항:**
- 결과 section의 모든 숫자를 나열하지 않는다. 가장 인상적인 결과만 언급한다.

#### Bullet Level 가이드

```
Conclusion
├── [1~2문단] Contribution 요약 (결과/의의 관점)
│   └── 핵심 결과 수치 (대표 1~2개만)
└── [1문단] Future Work + 확장 가능성
    ├── 기존 기법과의 orthogonality / 호환성 언급
    └── 구체적인 확장 방향 제시 (limitation을 opportunity로 전환)
```

---

## 4. Outline 자기 검토 체크리스트

Outline 작성 후, 제출/확장 전에 다음을 검토한다.

### 4.1 Bullet Level 검증

- [ ] 핵심 결정/주장이 하위 bullet에 묻혀 있지 않은가?
- [ ] 각 bullet은 정확히 하나의 아이디어만 담고 있는가?
- [ ] Bullet의 계층 구조가 논리적인가? (상위 bullet이 하위 bullet을 포괄하는가?)
- [ ] 한 section 내에서 bullet 수가 적절한가? (너무 많으면 subsection으로 분리)
- [ ] 같은 section 내에서 동일한 내용을 중복 서술하고 있지 않은가? (특히 contribution의 상위/하위 bullet 간 중복에 주의)

### 4.2 Logic Flow 검증

- [ ] 각 section 간의 전환이 자연스러운가?
- [ ] Section A에서 제기한 문제가 Section B에서 해결되는가?
- [ ] "한편", "또한" 등의 약한 연결어에 과도하게 의존하고 있지 않은가?
- [ ] 각 paragraph의 첫 문장만 이어 읽었을 때 전체 논리가 통하는가?
- [ ] Introduction의 논리 흐름 (common tech → importance → weakness → prior work → gap → criticality → novelty)이 빠짐없이 있는가?

### 4.3 문장 복잡도 검증

- [ ] 한 bullet(= 한 paragraph)에 4개 이상의 개념이 압축되어 있지 않은가?
- [ ] 모호한 주장이 없는가? ("사전에 결정됨" → 구체적으로 어떤 시점에, 어떤 기준으로?)
- [ ] 모든 strong claim에 구체적 근거가 있는가?

### 4.4 그림/표 배치 검증

- [ ] Figure 1이 첫 페이지에 배치될 수 있는가?
- [ ] 각 주요 결과에 대응하는 figure/table이 계획되어 있는가?
- [ ] Overview figure가 있는가?
- [ ] 그림 없이 텍스트로만 설명하는 부분이 너무 길지 않은가?

### 4.5 주장-근거 검증

- [ ] 모든 주장에 근거(실험, 인용, 분석)가 outline에 포함되어 있는가?
- [ ] 근거 없는 주관적 주장 (보통 형용사가 red flag)이 없는가?
- [ ] 인용으로 뒷받침해야 할 claim이 빠져 있지 않은가?

### 4.6 글쓰기 기본 사항 (확장 시 확인)

- [ ] 동사 시제: 과거형은 Introduction/Related Work에서만 사용 (Evaluation 등에서는 현재형)
- [ ] 주어 일관성: 같은 paragraph 내에서 주어를 자주 바꾸지 않는가? (특히 "We"와 시스템명을 혼용하면 행위 주체가 불명확해질 수 있다)
- [ ] 핵심 용어가 전체에서 통일되어 있는가?
- [ ] Filler words (can, in order to, shall 등)를 제거했는가?
- [ ] 첫 draft 작성 후 약 1/3을 삭제할 수 있는가? (그만큼 불필요한 내용이 있다는 의미)

---

## 5. 그림 및 표 계획

### 5.1 언제 그림이 필요한가

다음 중 하나에 해당하면 그림이 필요하다:

- **시스템 구조/pipeline**: Overview figure는 거의 필수다
- **비교 결과**: 여러 기법의 성능 비교는 그래프가 텍스트보다 효과적이다
- **동작 과정**: 시간 순서가 있는 알고리즘은 다이어그램이 이해를 돕는다
- **데이터 분포/패턴**: Motivation에서 문제를 보여줄 때 효과적이다

### 5.2 그림 배치 원칙

| 그림 | 위치 | 목적 |
|------|------|------|
| **Figure 1** (Teaser) | 첫 페이지 | 핵심 아이디어의 직관적 전달. 가장 인상적인 결과를 teaser로 활용 가능 |
| Overview figure | Method section 시작 | 전체 framework의 big picture |
| Motivation figure | Background/Motivation | 문제의 심각성을 데이터로 보여줌 |
| 결과 figure | Evaluation | Baseline 대비 성능 비교 |
| Ablation figure | Evaluation | 각 component의 기여도 |

### 5.3 표 vs 그림 선택 기준

| 상황 | 추천 |
|------|------|
| 정확한 수치 비교가 중요할 때 | **표** |
| 전반적인 추세/경향을 보여줄 때 | **그림** (line chart, bar chart) |
| 기법 간 기능 비교 (있다/없다) | **표** (checkmark table) |
| 분포나 패턴을 보여줄 때 | **그림** (scatter, heatmap) |
| 실험 setup 요약 | **표** |

### 5.4 그래프는 데이터가 아니라 Story를 보여줘야 한다

그래프는 단순히 숫자를 시각화하는 것이 아니다. **Reader가 그래프를 보고 즉시 "이 논문의 기법이 왜 좋은가"를 파악할 수 있어야 한다.**

- **Baseline 기준선을 시각적으로 표시한다.** 예: 1.0x 위치에 점선을 긋고 "GPU-only baseline" 라벨을 붙인다. 기준선이 없으면 reader가 수치의 의미를 즉시 파악하지 못한다.
- **가장 중요한 takeaway를 그래프 안에 annotation으로 표시한다.** 예: "최대 2.1x" 화살표, 특정 구간에 대한 설명 라벨 등.
- **그래프에서 실제 결과보다 과장된 인상을 주지 않는다.** 예: 실제로 baseline과 비슷한 구간이 있다면 그래프에서도 비슷하게 그려야 한다. Reviewer는 그래프와 숫자를 대조한다.

### 5.5 Figure와 본문의 순서를 일치시킬 것

그림에서 A → B 순서로 보여주면, 본문에서도 A → B 순서로 설명한다. 그림은 Prefill → Decoding 순인데 본문이 Decoding → Prefill 순으로 설명하면 reader가 혼란스럽다.

이 원칙은 multi-panel figure (a), (b), (c)에도 적용된다. 본문에서 (b)를 먼저 설명하고 (a)를 나중에 설명하는 것은 피한다.

### 5.6 그림 분석 체크리스트 (Outline 단계)

Outline 단계에서 각 계획된 그림에 대해 다음을 판단한다:

- **추가해야 할 그림**: 텍스트만으로 설명이 어려운 부분이 있는가?
- **수정해야 할 그림**: 기존 그림이 메시지를 명확히 전달하는가?
- **이동해야 할 그림**: 이 그림이 현재 위치에서 가장 효과적인가?
- **삭제해야 할 그림**: 이 그림이 없어도 이해에 문제가 없는가? (공간이 부족할 때)

---

## 6. Reviewer를 고려한 Outline 검토

이 section은 가장 중요하다. 논문을 쓰는 것은 결국 **reviewer와 독자를 설득하는 행위**이다. Outline 단계에서부터 reviewer의 관점을 고려하면, 나중에 reject 사유가 될 문제를 미리 방지할 수 있다.

### 6.1 Best-Case Reviewer 가정

Best-case reviewer는 다음과 같은 이상적인 reviewer이다:

- 충분한 시간이 있다
- 모든 단어를 꼼꼼히 읽는다
- 인용된 논문을 찾아 맥락을 파악한다
- 해당 분야의 전문가이다
- 열린 마음으로, 저자에게 유리하게 해석해 준다
- 이해가 안 되는 부분은 다시 돌아가서 읽는다

이런 reviewer를 위해서도 다음은 필요하다:
- **완벽한 설명**: 전문가도 세부 사항은 놓칠 수 있으므로, 핵심 메커니즘은 명확히 서술한다
- **충분한 background**: 전문가라도 본인의 세부 분야가 아닐 수 있다
- **논리적 완결성**: 호의적인 reviewer도 논리적 허점은 지적한다

**그러나 best-case reviewer만을 위해 쓰면 안 된다.** 모든 reviewer가 이렇게 이상적이지는 않기 때문이다. 또한 논문이 accept된 후의 독자들은 reviewer보다 훨씬 빠르게 읽으며, 이해하기 어려우면 그냥 읽기를 멈춘다.

### 6.2 Worst-Case Reviewer 가정

**논문은 worst-case reviewer를 만족시키도록 써야 한다.** Worst-case reviewer를 만족시키면, (1) 논문의 명확성이 극대화되고, (2) accept 확률이 높아진다.

#### 6.2.1 모든 것을 아는 Reviewer

> "이 reviewer는 관련 연구를 모조리 읽었다."

한 명의 reviewer가 모든 것을 알지는 못하지만, **program committee 전체**는 관련 연구를 거의 다 알고 있다고 가정해야 한다.

**Outline 검토 질문:**
- [ ] 모든 직접 관련 연구 (related related work)를 인용했는가?
- [ ] 간접적으로라도 관련되어 보이는 연구 (unrelated related work)를 인용했는가? (누군가는 "왜 이 논문을 cite하지 않았지?"라고 생각할 수 있다)
- [ ] 인용한 논문의 결과를 실험에서 사용한다면, 해당 논문의 정확한 버전을 인용했는가? (arXiv vs. conference version)
- [ ] 기존 연구와의 차별점이 명확히 서술되어 있는가?

#### 6.2.2 아무것도 모르는 Reviewer

> "이 reviewer는 해당 분야의 배경 지식이 부족하지만, 본인은 안다고 생각한다."

전문가에게 당연한 것도 worst-case reviewer에게는 생소할 수 있다. 본인이 수개월간 몰두한 내용이 모든 사람에게 당연한 것은 아니다.

**Outline 검토 질문:**
- [ ] 최근 5년 이내의 새로운 기술에 의존하는가? → 설명이 있는가?
- [ ] 잘 알려진 기술의 세부 사항에 의존하는가? → 설명이 있는가?
- [ ] 기술을 색다른 방식으로 조합하여 사용하는가? → 설명이 있는가?
- [ ] Background section에 "너무 당연해서 쓰기 지루한" 내용이 있는가? → 그렇다면 올바른 방향이다. 그런 내용이 없다면 background가 부족할 가능성이 높다.
- [ ] 논문이 self-contained인가? 다른 논문을 읽지 않아도 이해할 수 있는가?

#### 6.2.3 피곤하고 짜증난 Reviewer

> "이 reviewer는 15편의 논문을 이틀 안에 읽어야 하며, reject 사유를 찾고 있다."

Worst-case reviewer의 상태:
- 성급하다. 빠르게 읽는다.
- 부정적인 결론으로 뛰어간다 (jumps to negative conclusions).
- 피곤하다. 집중력이 낮다.
- 명확하지 않은 부분은 건너뛰고, 최악의 해석을 한다.
- Related work를 읽지 않는다.
- **Reject할 이유를 찾고 있다.**

**Outline 검토 질문:**
- [ ] 논문의 구조가 명확하고 예측 가능한가? (표준적인 section 구성)
- [ ] 각 section의 핵심 메시지가 첫 문장에서 드러나는가?
- [ ] Figure와 table이 텍스트 없이도 대략적으로 이해 가능한가?
- [ ] 문법 오류와 typo가 없는가? (grammar check를 여러 번 실행)
- [ ] Contribution이 Introduction에서 bullet point로 명확히 제시되는가?
- [ ] 논문의 흐름이 자연스러워서, 건너뛰며 읽어도 전체 맥락을 잃지 않는가?

### 6.3 Reviewer 관점 Outline 검토 기준

각 section별로 reviewer가 물을 수 있는 질문을 미리 점검한다.

#### Abstract / Introduction에서 물을 수 있는 질문

| Reviewer 질문 | Outline에서 확인할 것 |
|-------------|---------------------|
| "이게 왜 중요한데?" | Motivation이 충분한가? 구체적 수치나 사례가 있는가? |
| "이전 연구와 뭐가 다른데?" | Novelty가 명확히 서술되어 있는가? |
| "기여가 정확히 뭔데?" | Contribution이 bullet point로 구체적으로 나열되어 있는가? |
| "이건 trivial한 거 아닌가?" | 왜 어려운지 (Y)가 설명되어 있는가? |

#### Method에서 물을 수 있는 질문

| Reviewer 질문 | Outline에서 확인할 것 |
|-------------|---------------------|
| "왜 이렇게 설계했는데?" | Design rationale이 있는가? |
| "다른 방법은 고려 안 했는가?" | Alternative approach를 논의했는가? |
| "이 가정은 현실적인가?" | 가정의 타당성을 근거와 함께 설명했는가? |
| "이건 이전 연구 X와 같은 거 아닌가?" | 차별점이 명확한가? |

#### Evaluation에서 물을 수 있는 질문

| Reviewer 질문 | Outline에서 확인할 것 |
|-------------|---------------------|
| "Baseline이 공정한가?" | 동일한 framework, 동일한 조건에서 비교하는가? |
| "Ablation은?" | 자명하지 않은 설계 결정에 대한 ablation이 있는가? |
| "왜 이 benchmark/workload만 사용했는가?" | Benchmark 선택의 근거가 있는가? |
| "Overhead는?" | Overhead 분석이 포함되어 있는가? |
| "이 숫자는 통계적으로 유의미한가?" | 여러 번 실행한 평균인가? Confidence interval이 있는가? |
| "Scalability는?" | 다양한 규모에서 테스트했는가? |

#### Strong Claim 체크리스트

논문에 강한 주장이 있을 때마다 다음을 확인한다:

- [ ] 주장: "X는 Y보다 빠르다" → **어떤 조건에서?** (모델, batch size, HW, ...)
- [ ] 주장: "최초로 ~를 제안한다" → **정말 최초인가?** (Google Scholar 재확인)
- [ ] 주장: "기존 연구는 ~를 고려하지 않았다" → **정말 없는가?** (관련 연구 재확인)
- [ ] 주장: "significant improvement" → **구체적 수치는?**
- [ ] 주장: "일반적으로 적용 가능하다" → **어디까지 테스트했는가?**

---

## 7. 분야별 특수 고려사항 (Computer Architecture / AI Systems)

Computer architecture 및 AI systems 분야에서는 일반적인 논문 작성 원칙 외에 추가로 주의해야 할 사항이 있다.

### 7.1 하드웨어 환경 명시 필수

실험 결과의 재현 가능성과 공정한 비교를 위해, 다음을 반드시 명시한다:

- GPU 모델, 수량, 메모리 용량
- CPU 모델, 코어 수, 메모리 용량
- 인터커넥트 (NVLink, PCIe 버전 등)
- CUDA/cuDNN/PyTorch 버전
- 기타 관련 하드웨어/소프트웨어 설정

### 7.2 Baseline Fairness

**같은 framework, 같은 조건에서 비교한다.**

```
(X) "우리의 PyTorch 구현이 논문에서 보고된 TensorFlow 결과보다 빠르다"
    → Framework 차이에 의한 것일 수 있음

(O) "동일한 PyTorch 환경에서 baseline을 재구현하여 비교하였다"
    → 공정한 비교
```

- Baseline을 직접 재구현했다면, 원 논문의 결과와 재구현 결과의 차이를 보고한다
- 가능하면 공식 코드를 사용하되, 동일한 하드웨어에서 실행한다

### 7.3 Latency/Throughput에 Context 필요

성능 수치를 보고할 때는 반드시 context를 함께 제공한다:

| 빠짐없이 명시할 것 | 예시 |
|------------------|------|
| Model 정보 | LLaMA-MoE 7B, 8 experts |
| Batch size | 32 |
| Sequence length | 2048 tokens |
| Precision | FP16 / BF16 / INT8 |
| 측정 방식 | Warm-up 후 100회 반복의 평균 |

### 7.4 Ablation Study

Architecture/AI systems 논문에서 ablation이 없으면 거의 확실히 reject 사유가 된다. 단, **모든 설계 요소에 대해 ablation이 필요한 것은 아니다.**

Ablation이 필요한 설계 결정은 **그 요소의 포함/제외에 따라 결과가 좋아질 수도, 나빠질 수도 있는 것** — 즉, reviewer가 "왜 이렇게 했는가?"라고 물을 수 있는 결정이다.

| Ablation 필요 | Ablation 불필요 |
|--------------|----------------|
| 제안하는 scoring 방식 (다른 scoring도 가능) | 이미 검증된 표준 기법의 채택 (e.g., Adam optimizer) |
| Global vs per-layer selection (둘 다 합리적) | Quantization bit-width (sensitivity로 충분) |
| 특정 threshold 값의 선택 | 자명하게 필요한 component |

**핵심 질문**: "이 설계 결정 대신 다른 선택을 했을 때 결과가 어떻게 달라지는가?"에 답해야 하는 경우 ablation이 필요하다.

### 7.5 Overhead는 별도 Discussion으로

제안 기법의 overhead (area, power, latency overhead 등)는:
- Method section에서 언급하지 않는다 (novelty에 집중)
- Evaluation section의 마지막 부분에서 별도 subsection으로 간략히 다룬다
- 가능하면 표 하나로 요약한다
- Overhead가 불가피하다면, 그로 인한 이득이 충분히 크다는 것을 보여준다

---

## 8. 흔한 실수 Top 12

실제 논문 리뷰 경험에 기반한, 학생들이 가장 자주 범하는 실수 10가지이다.

### 실수 1: 핵심 결정이 하위 bullet에 묻힘

논문의 가장 중요한 contribution이나 설계 결정이 subsection 깊숙이, 하위 bullet에 묻혀 있으면 reviewer가 놓친다. **가장 중요한 내용은 가장 눈에 띄는 위치에 배치한다.**

```
(X) 3.2.1절의 세 번째 문단에서 갑자기 핵심 알고리즘이 등장
(O) 3절 첫 문단에서 핵심 아이디어를 제시하고, 이후 상세 설명으로 전개
```

### 실수 2: 문단 간 전환에 "한편" 남용

"한편(On the other hand)"은 논리적으로 대조되는 내용을 연결할 때 사용한다. 그러나 학생들은 관련 없는 문단을 연결할 때도 습관적으로 사용하여, 논리의 흐름이 끊어진다. 문단 간 전환은 앞 문단의 내용을 자연스럽게 이어받아야 한다.

### 실수 3: 한 문장에 너무 많은 개념 압축 + 주어·목적어 실종

4개 이상의 개념을 한 문장에 넣으면 읽기 어렵다. 더 큰 문제는, 긴 문장일수록 주어와 목적어가 사라지기 쉽다는 것이다. **문장을 나누되, 나눈 각 문장에서 "누가 무엇을 어떻게 한다"가 명확해야 한다.**

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

"사전에 결정됨", "적절히 설정됨", "충분히 큰" 같은 모호한 표현은 reviewer의 불신을 산다.

```
(X) "Expert의 배치는 사전에 결정된다."
    → 언제? 어떤 기준으로? 누가?

(O) "Expert의 배치는 inference 시작 전 profiling 단계에서, 각 expert의
     평균 활용 빈도를 기준으로 결정된다."
```

> **Tip**: "적극적으로", "효율적으로", "효과적으로" 같은 수식어도 비슷한 문제를 일으킨다. 대부분의 경우 삭제하고 구체적 행위로 대체하면 문장이 더 강해진다. 단, "aggressive prefetching" 등 분야에서 통용되는 표현은 그대로 사용해도 된다.

### 실수 5: 반복을 강조로 착각

같은 문장을 Introduction, Method, Conclusion에 복사-붙여넣기하는 것은 강조가 아니라 게으름이다. 각 section의 맥락에 맞게 다른 관점에서 서술해야 한다.

### 실수 6: Related Work에서 비난

기존 연구의 한계를 지적하는 것과 비난하는 것은 다르다. **해당 논문의 저자가 reviewer일 수 있다.**

```
(X) "이전 연구 [5]는 이 중요한 문제를 완전히 간과하였다."
(O) "이전 연구 [5]는 A 문제를 효과적으로 해결하였으나, B 측면은
     고려하지 않았다. 본 논문에서는 이 B 측면까지 함께 해결한다."
```

### 실수 7: 숫자 나열식 결과 서술

Figure나 table이 있는데도 텍스트에서 모든 숫자를 나열하면, 읽기 지루하고 핵심이 묻힌다.

```
(X) "Workload A에서 15%, B에서 23%, C에서 31%, D에서 18%,
     E에서 27%, F에서 42%의 향상을 보인다."

(O) "Figure 5에서 보이는 바와 같이, 제안 기법은 평균 26%의 성능 향상을
     달성하며, 특히 memory-intensive workload (F)에서 최대 42%까지
     향상된다."
```

### 실수 8: Ablation Study 누락

Ablation이 아예 없으면 reviewer는 major weakness로 지적한다. **자명하지 않은 설계 결정** — 포함/제외에 따라 결과가 좋아질 수도 나빠질 수도 있는 요소 — 에 대해서는 반드시 ablation을 수행한다. 반면, 누구나 동의할 수 있는 표준적 설계까지 ablation할 필요는 없다.

### 실수 9: 약한 결과를 적절히 다루지 못함

결과가 안 좋은 부분이 있을 때, 이를 다루는 방법은 여러 가지가 있다. 핵심은 **변명이 아니라 분석**이 되어야 한다는 것이다.

**방법 1: 결과 서술 시 reasoning 제공**

약한 결과에 대해 논리적인 이유를 설명할 수 있다면, 결과 section에서 분석과 함께 서술하는 것이 자연스럽다. 이때 "변명"이 아니라 "분석"의 톤이 되어야 한다.

```
(X) "Workload G에서는 오히려 5% 성능이 떨어지는데,
     이는 어쩔 수 없는 한계이다."
    → 변명 톤. 분석이 없음.

(O) "Workload G에서는 이득이 제한적인데, 이는 해당 workload의
     expert 활용 패턴이 균일하여 재배치의 여지가 적기 때문이다.
     반면, 활용 패턴의 편차가 큰 나머지 workload에서는
     일관된 향상을 보인다."
    → 분석 톤. 약한 이유를 설명하되, 전체적 이득을 함께 보여줌.
```

**방법 2: 결과가 전체적으로 좋다면, 약한 케이스를 부각하지 않기**

Reasoning이 어려운 경우, 전체적인 gain이 좋다면 약한 케이스를 굳이 부각하지 않아도 된다. 예를 들어, workload grouping (memory-intensive vs compute-intensive 등)을 통해 카테고리별 평균으로 결과를 보여주면, 개별 workload의 약점이 자연스럽게 맥락 안에 들어간다.

```
(O) "Memory-intensive workload 그룹에서 평균 35%,
     compute-intensive 그룹에서 평균 12%의 향상을 보인다."
    → 개별 workload를 나열하지 않고 그룹으로 제시
```

**방법 3: Setup에서 적용 범위를 명시**

제안 기법의 적용 범위가 명확한 경우, Experimental Setup에서 미리 scope를 설명하는 것도 좋은 전략이다.

```
(O) [Setup] "제안 기법은 expert 간 활용 빈도 차이가 클수록
     효과적이다."
    [Results] "예상대로, 활용 패턴 편차가 큰 workload에서
     가장 큰 향상을 보인다."
```

요약하면: **약한 결과를 숨기라는 것이 아니라, 분석적으로 다루거나 전체 맥락 안에서 자연스럽게 제시하라는 것이다.** 변명 톤만 피하면 된다.

### 실수 10: Conclusion에서 Introduction 복사

Conclusion은 Introduction의 복사본이 아니다. Introduction에서는 "무엇을 제안하는가 (novelty)"를, Conclusion에서는 "그 결과가 가지는 의의 (impact)"를 중심으로 쓴다. Future work는 반드시 포함하되, limitation을 직접 나열하지 말고 **확장 가능성이나 기존 기법과의 orthogonality**로 자연스럽게 전환한다.

### 실수 11: 과장된 표현으로 reviewer에게 공격 빌미 제공

"원천적으로 제거", "완전히 해소", "획기적으로 향상" 같은 표현은 실제 결과와 조금이라도 괴리가 있으면 reviewer의 신뢰를 잃는다. **Claim의 강도는 evidence가 뒷받침할 수 있는 범위 내에서 설정한다.** 표현을 약간 낮추는 것이 논문의 신뢰도를 높인다.

```
(X) "GPU stall을 원천적으로 제거한다."
    → Reviewer: "Figure를 보면 stall이 남아있는데?"

(O) "GPU stall을 대부분 제거하여 near-optimal throughput을 달성한다."
    → 정직하고 방어 가능한 표현
```

### 실수 12: Section 간 설명 순서 불일치

Introduction에서 A → B → C 순서로 contribution을 소개해놓고, Method에서 B → A → C 순서로 설명하면 reader가 혼란스럽다. **논문 전체에서 핵심 개념의 등장 순서를 일관되게 유지한다.**

---

## 부록: Outline 작성 순서 권장 사항

논문의 각 section을 어떤 순서로 작성할지도 중요하다. 다음은 권장 순서이다:

```
1. Contribution bullet points 정리 (모든 작업의 출발점)
2. Method outline (핵심 기여를 먼저 구조화)
3. Evaluation outline (어떤 실험으로 검증할지)
4. Related Work outline (비교 대상 정리)
5. Background/Motivation outline
6. Introduction outline (전체가 잡힌 후 요약으로 작성)
7. Conclusion outline (모든 것이 확정된 후)
8. Abstract (가장 마지막. 단, 일찍 draft를 써두면 초점을 잡는 데 도움)
```

Abstract는 가장 마지막에 최종 확정하되, **가능한 일찍 draft를 써두는 것을 권장한다.** Abstract를 일찍 쓰면 논문의 story에 문제가 있는지 빨리 발견할 수 있다.

---

## 부록: 글쓰기 기본 규칙 체크리스트

Outline 확장 후, 제출 전 반드시 점검할 기본 규칙이다.

- [ ] 맞춤법/문법 검사를 여러 번 실행했는가 (Word grammar checker 포함)
- [ ] 3인칭 단수 동사에 -s를 빠뜨리지 않았는가
- [ ] 조동사 뒤에 동사원형을 사용했는가
- [ ] 과거형을 Intro/Related Work 이외에서 사용하지 않았는가
- [ ] 수동태를 과도하게 사용하지 않았는가 (능동태가 명확할 때는 능동태로)
- [ ] Filler words를 제거했는가 (can, in order to, shall, ...)
- [ ] 동일 paragraph 내에서 단어 반복이 없는가
- [ ] LaTeX 인용: 저자가 주어일 때 `\citet{}`, 아닐 때 `~\citep{}`
- [ ] LaTeX 따옴표: ` ``correct'' ` 형식 사용
- [ ] 수식 뒤에 마침표/쉼표를 빠뜨리지 않았는가
- [ ] Broken reference/citation (**??**)이 없는가
- [ ] Acronym을 정의 전에 사용하지 않았는가
- [ ] 인용 시 arXiv 버전이 아닌 conference/journal 버전을 인용했는가
- [ ] Page limit을 초과하지 않았는가 (초과 시 supplementary로 이동, 삭제하지 않음)

---

## 부록: 한글로 Outline 작성 시 주의사항

Outline을 한글로 먼저 작성하는 것은 자연스러운 과정이다. 그러나 한국어의 문법적 특성이 그대로 영어 논문에 옮겨지면 문제를 일으킬 수 있다. 다음은 **한글 outline 단계에서부터 의식해야 할 사항**이다.

### 한국어식 지시어/대명사를 기술 용어로 대체

한국어에서는 "바깥", "양쪽", "이것", "그쪽" 같은 지시어가 자연스럽지만, 이를 그대로 영어로 옮기면 모호해진다. Outline 단계에서부터 정확한 기술 용어를 사용하는 습관을 들인다.

| 한국어 습관 | 문제 | 수정 |
|------------|------|------|
| "Top-k **바깥**의 expert" | "바깥"이 영어로 어색 (outside top-k?) | "top-k+α 범위의 expert" 또는 "pre-routing stage의 expert score를 고려하여" 등 구체적으로 풀어쓴다 |
| "**양쪽** phase에서" | "양쪽"이 무엇을 가리키는지 불명확 | "Prefill phase와 decoding phase 모두에서" |
| "**이를** 기반으로 최적화" | "이"가 무엇인지 앞 문장을 다시 봐야 함 | "Profiling 결과를 기반으로 배치를 최적화" |
| "**그만큼** 성능이 올라간다" | 구체성 없음 | "Cache hit rate에 비례하여 throughput이 증가한다" |

### 한국어 문장 구조의 영어 전환 시 함정

한국어는 동사가 문장 끝에 오므로, 긴 수식절이 동사 앞에 쌓이기 쉽다. 이 구조를 그대로 영어로 옮기면 읽기 어려운 문장이 된다.

```
한국어 outline:
"Expert의 활용 빈도를 실시간으로 모니터링하여 temporal locality를
분석하고, 이를 기반으로 CPU-GPU 간 migration 비용을 고려한
ILP를 풀어 최적의 배치를 결정한다."
→ 동사("결정한다")가 끝에 와서 한국어로는 자연스럽지만,
  영어로 옮기면 한 문장에 4개 이상의 개념이 들어감

수정:
"제안 기법은 expert 활용 빈도의 temporal locality를 분석한다.
분석 결과를 기반으로, migration 비용을 고려한 ILP를 통해
최적의 CPU-GPU 배치를 결정한다."
→ 한 문장 = 한 행위. 영어로 옮겨도 자연스러움
```

> **Tip**: 한글 outline에서 한 bullet이 3줄 이상이면, 영어로 쓸 때 거의 확실히 한 문장이 너무 길어진다. Outline 단계에서 미리 나눈다.

---

> **마지막으로**: 좋은 논문은 좋은 연구에서 나오지만, 좋은 연구도 잘 쓰지 않으면 impact를 잃는다.
> Outline에 충분한 시간을 투자하라. 그것이 결국 전체 작성 시간을 줄여준다.
