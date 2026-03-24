# AICA Skills for Claude Code

논문 작성을 도와주는 Claude Code 스킬 2종입니다. 아웃라인 단계부터 최종 검토까지 체계적인 워크플로우를 제공합니다.

---

## 설치

### 1. Claude Code 설치

Claude Code가 설치되어 있지 않다면 먼저 설치합니다:

```bash
npm install -g @anthropic-ai/claude-code
```

### 2. 스킬 설치 (clone)

아래 두 명령을 실행하면 스킬이 자동으로 등록됩니다:

```bash
# Outline Assistant (아웃라인 리뷰)
git clone https://<토큰>@github.com/aica-ssu/aica-outline-assistant.git ~/.claude/skills/aica-outline-assistant

# Writing Assistant (논문 작성/편집)
git clone https://<토큰>@github.com/aica-ssu/aica-writing-assistant.git ~/.claude/skills/aica-writing-assistant
```

### 3. CLAUDE.md 설정 (선택)

DOCX 코멘트 기능을 사용하려면 `~/.claude/CLAUDE.md`에 저자 설정을 추가합니다:

```markdown
## User Preferences

### DOCX 편집 설정
- **Author name**: tracked changes와 comments에 `your_name` 사용
- **Comment language**: 메모(comments)는 **한국어**로 작성
```

### 4. 설치 확인

Claude Code에서 `/skills`를 입력하면 두 스킬이 목록에 나타나야 합니다.

### 5. 업데이트

```bash
cd ~/.claude/skills/aica-outline-assistant && git pull
cd ~/.claude/skills/aica-writing-assistant && git pull
```

---

## 스킬 개요

| 스킬 | 용도 | 사용 시점 |
|------|------|----------|
| **Outline Assistant** | 아웃라인 구조, 서사, 기여점, 논리 흐름 리뷰 | 논문 컨셉 및 실험결과가 있는 상태에서 **아웃라인 작성 시** |
| **Writing Assistant** | 초안 작성, 문장/문단 편집, Anti-AI 패턴 제거, 최종 검토 | **아웃라인 확정 후** |

### 권장 워크플로우

```
아웃라인 작성
  → Outline Assistant 리뷰 → 수정 → 재리뷰 → 아웃라인 확정
    → Writing Assistant로 초안 작성 → Full Draft Review → 수정
      → Finalization Review → 제출
```

---

## 스킬 상세

<details>
<summary><strong>📋 Outline Assistant — 아웃라인 리뷰 스킬</strong></summary>

### 전체 리뷰

```
이 아웃라인을 리뷰해줘. [파일.docx]
```

자동으로 3-pass 리뷰를 수행합니다:

| Pass | 초점 |
|------|------|
| **1. 구조와 서사** | 스토리 완성도, 기여점 명확성, Intro→Conclusion 아크 |
| **2. 계층·논리·리뷰어** | 불릿 계층, 논리 전환, 리뷰어 관점 약점 |
| **3. 시각자료·최종** | Figure/Table 완성도, 우선순위 정리 |

### 개별 관점 리뷰

| 관점 | 커맨드 |
|------|--------|
| 기여점 명확성 | `이 아웃라인의 contribution이 명확한지 검토해줘.` |
| Introduction 서사 | `Introduction의 narrative flow를 점검해줘.` |
| 불릿 계층 | `bullet hierarchy를 검토해줘. 핵심 결정이 sub-point에 묻혀있지 않은지.` |
| 논리 전환 | `섹션 간 논리적 연결을 검토해줘. 약한 연결 찾아줘.` |
| 리뷰어 관점 | `worst-case reviewer 관점에서 이 아웃라인을 공격해줘.` |
| Ablation 체크 | `논쟁 가능한 설계 결정에 ablation이 계획되어 있는지 대조해줘.` |
| Abstract 검토 | `Abstract가 독립적으로 읽히는지 검토해줘.` |
| Conclusion 검토 | `Conclusion이 Introduction의 복붙이 아닌지 검토해줘.` |
| Figure/Table | `빠진 figure/table을 추천해줘.` |
| Claim 검증 | `모든 주장에 근거가 있는지 검토해줘.` |

### 리뷰 결과 우선순위

| 우선순위 | 의미 | 대응 |
|---------|------|------|
| **[Critical]** | 리뷰어가 반드시 지적할 문제 | 즉시 수정 |
| **[Important]** | 논문 품질에 큰 영향 | 수정 권장 |
| **[Nice]** | 개선하면 좋지만 필수 아님 | 선택적 반영 |

### 핵심 원칙

1. **반복 ≠ 강조** — 같은 내용 반복은 구조 문제
2. **기여점을 먼저 정의** — 상세 섹션 전에 crystal clear
3. **Introduction ≠ Conclusion** — 다른 관점 필요
4. **논쟁 가능한 설계 결정에 검증 필요** — 표준적 선택은 불필요, 의심될 수 있는 선택에는 ablation 권장
5. **Method에서 overhead 논하지 않기** — overhead는 Evaluation에서
6. **약한 결과는 분석적으로** — 변명이 아닌 논리적 분석
7. **불릿 1개 = 아이디어 1개** — 각 불릿은 초안에서 1개 문단이 됨

### 흔한 아웃라인 문제

| 문제 | 신호 | 수정 |
|------|------|------|
| 기여점 매몰 | 핵심 insight가 ilvl>=2 | ilvl=0~1로 승격 |
| Motivation bridge 부재 | "한편"으로 method 전환 | "이 문제를 해결하기 위해" bridge 추가 |
| Conclusion 복붙 | Introduction과 거의 동일 | 결과 의의 + limitation + future work |
| 불릿 과밀 | 1개 불릿에 4+ 개념 | 별도 불릿으로 분리 |
| 모호한 주장 | "성능이 향상됨" | 정확한 조건과 수치 추가 |
| Ablation 누락 | 논쟁 가능한 설계 결정에 검증 없음 | 해당 결정에 ablation 또는 정당화 추가 |
| Method에 overhead | Proposed Method에서 비용 논의 | Evaluation으로 이동 |
| 계층 건너뛰기 | ilvl=0 → ilvl=2 | ilvl=1 그룹핑 추가 |
| Related Work 나열식 | 설명만, 비교 없음 | 제안 방법과의 대조 추가 |

### Related Work 원칙

| 원칙 | 나쁜 예 | 좋은 예 |
|------|---------|---------|
| 문단 첫 문장 = 주제 | "[논문A]는 X를 제안했다." | "메모리 최적화는 세 가지 접근으로 나뉜다." |
| 카테고리별 조직 | 시간순 나열 | 접근법별 분류 |
| 비교와 대조 | "A는 X를 했다. B는 Y를 했다." | "A는 X를 했으나, 우리의 Z 특성에는 적용 불가." |
| 파생 느낌 금지 | "Similar to [A], our~" | "Unlike [A] which targets X, our~" |
| 비판 금지 | "[A]의 방법은 비효율적이다." | "[A]는 다른 목표(Y)에 최적화되어 있다." |

### Evaluation 체크리스트 (CS Arch / AI Systems)

| 항목 | 필수 여부 |
|------|----------|
| HW 사양 (GPU, 메모리, interconnect) | 필수 |
| SW 버전 (Framework, CUDA, driver) | 필수 |
| Baseline 공정성 (동일 조건) | 필수 |
| 수치 context (batch, seq len, model, precision) | 필수 |
| 논쟁 가능한 설계 결정 Ablation | 권장 (표준 선택은 불필요) |
| 트렌드 (단일 포인트가 아닌 추세) | 권장 |
| 재현성 (다른 그룹 재현 가능?) | 권장 |

</details>

<details>
<summary><strong>✍️ Writing Assistant — 논문 작성/편집 스킬</strong></summary>

> **전제**: 아웃라인이 확정된 상태에서 사용합니다. 아웃라인 리뷰가 필요하면 Outline Assistant를 먼저 사용하세요.

### 전체 워크플로우 (7단계)

```
확정된 아웃라인
  → Step 1. 프로젝트 셋업 (아웃라인 + LaTeX 등록)
  → Step 2. 섹션별 초안 작성
  → Step 3. 문장/문단 편집
  → Step 4. Full Draft Review (3-Phase)
  → Step 5. 개별 이슈 수정
  → Step 6. Finalization Review (as-is / to-be)
  → Step 7. 수치 일관성 감사 → 제출
```

### Step 1. 프로젝트 셋업

논문 프로젝트 폴더에서 Claude Code를 실행하고 아웃라인을 등록합니다:

```
이 프로젝트는 [학회명] 제출용 논문이야.
아웃라인은 outline.docx에 있고, LaTeX 메인 파일은 main.tex야.
아웃라인을 읽고 전체 구조를 파악해줘.
```

### Step 2. 아웃라인 기반 섹션 초안 작성

**권장 순서**: Evaluation → Design → Motivation → Background → Introduction → Abstract → Conclusion (결과부터 작성하면 주장이 구체적이 됩니다.)

| 섹션 | 커맨드 |
|------|--------|
| Evaluation | `아웃라인의 Evaluation을 기반으로 초안을 작성해줘. Fig. X와 Table Y 데이터를 정량적으로.` |
| Design | `아웃라인의 Proposed Design을 기반으로 초안을 작성해줘. [기법 A], [기법 B] 순서로.` |
| Motivation | `아웃라인의 Motivation을 기반으로 초안을 작성해줘. Fig. X 기반으로 문제점을 보여줘.` |
| Background | `아웃라인의 Background를 기반으로 초안을 작성해줘. [개념 A], [개념 B] 정의 + 관련 연구.` |
| Introduction | `아웃라인의 Introduction을 기반으로 초안을 작성해줘. 핵심 주제는 [X], 한계는 [Y].` |
| Abstract | `본문 내용을 기반으로 Abstract을 작성해줘. 2문단 구조, 핵심 개념 + 주요 수치만.` |
| Conclusion | `Conclusion을 작성해줘. 핵심 기여 + 결과 수치, future work는 기회 중심으로.` |

섹션 작성 후 아웃라인 대조:

```
방금 작성한 [섹션] 초안이 아웃라인의 핵심 포인트를 모두 포함하는지 대조해줘.
```

### Step 3. 문장/문단 편집

| 상황 | 커맨드 |
|------|--------|
| 수동태 제거 | `이 섹션에서 수동태를 능동태로 바꿔줘.` |
| 문장 길이 조정 | `15-20 단어 기준으로 긴 문장을 분리해줘.` |
| 문단 병합 | `모델별 bold heading 마이크로 문단을 flowing prose로 병합해줘.` |
| 용어 통일 | `[개념]에 대해 사용된 모든 표현을 [통일 용어]로 맞춰줘.` |
| "This" 주어 수정 | `"This"로 시작하는 문장을 구체적인 주어로 교체해줘.` |
| Figure 참조 정리 | `Figure 참조 순서를 정리해줘.` |
| Em-dash 제거 | `모든 em-dash(---)를 세미콜론/마침표/각주로 교체해줘.` |
| Caption 수정 | `figure/table caption에서 콜론 제거하고 첫 문장 정리해줘.` |

### Step 4. 전체 리뷰 (3-Phase)

```
이 논문 전체를 Full Draft Review 워크플로우로 리뷰해줘.
Phase 1(학술 품질) → Phase 2(Anti-AI) → Phase 3(무결성 검증) 순서로.
```

| Phase | 체크 항목 |
|-------|----------|
| **Phase 1 (학술 품질)** | 주제 일관성, 문장 메트릭, 논리 흐름, 문단 파편화, 한계점 프레이밍, Bold 사용, 주장 검증 |
| **Phase 2 (Anti-AI)** | 콘텐츠 패턴, 언어 패턴, 스타일 패턴, 필러/헤징, Self-audit |
| **Phase 3 (무결성)** | 문장 메트릭 재검증, 용어 일관성 재검증, Figure 참조 재검증 |

#### 최종 검토 (Finalization Review)

```
이 논문을 Finalization Review 워크플로우로 검토해줘.
수정 사항은 직접 적용하지 말고 as-is / to-be 테이블로 정리해줘.
```

출력 형식:

| # | Location | Category | As-Is | To-Be |
|---|----------|----------|-------|-------|
| 1 | L.42 | (B) Numeric | "achieves 86%" | "achieves 86.5% for GPT-2 Medium" |
| 2 | L.107 | (D) AI-like | "highlighting the effectiveness" | (삭제) |

Categories: **(A)** Logic hole, **(B)** Numeric/grammar, **(C)** Tone/misleading, **(D)** AI-like

### Step 5. 개별 이슈 수정

| 상황 | 커맨드 |
|------|--------|
| Abstract 구조 개선 | `Abstract을 2문단 구조로 재구성해줘. 핵심 개념 + 주요 수치만.` |
| Reference 추가 | `이 문장 "[주장]"을 뒷받침하는 논문을 찾아줘. claim-driven으로.` |
| 수식 정리 | `불필요한 수식을 식별해줘. 자명한 수식은 제거 후보.` |
| 페이지 부족 | `논리 jump 사이에 bridging sentence 추가, 기술적 reasoning 보강해줘. 필러는 넣지 마.` |
| 아웃라인 대조 | `현재 초안이 아웃라인의 모든 핵심 포인트를 커버하는지 대조해줘.` |

### Step 6-7. 수치 일관성 감사

```
Abstract, Introduction, Conclusion의 모든 headline 수치를
Evaluation 테이블/그림의 원본 수치와 대조해줘. 불일치가 있으면 테이블로 정리해줘.
```

### 핵심 메트릭

| 메트릭 | 목표값 |
|--------|--------|
| 문장 길이 | 15-20 단어 |
| 문장당 쉼표 | 최대 1-2개 |
| 문단 길이 | 3-7 문장 (다양하게) |
| -ing 분사구 | 문장당 최대 1개 |
| Bold 사용 | 페이지당 3회 이하 (heading 제외) |
| Em-dash | 사용 금지 (세미콜론/마침표/각주로 대체) |
| Bullet list | Introduction contributions에서만 |

### Anti-AI 패턴 (절대 사용 금지)

| 패턴 | 예시 | 대체 |
|------|------|------|
| 과장 | "pivotal moment", "testament to" | 삭제, 사실만 기술 |
| 홍보성 | "groundbreaking", "showcasing" | 삭제 또는 중립 표현 |
| 모호 인용 | "experts argue" | 구체적 논문 인용 |
| 꼬리 -ing | "highlighting the effectiveness" | 삭제 (수치가 스스로 증명) |
| Copula 회피 | "serves as", "stands as" | "is" 사용 |
| 부정 병렬 | "Not only X, but also Y" | "X and Y" |
| 3의 법칙 남용 | "A, B, and C" 반복 | 실제 관련 있는 것만 |
| Em-dash | "---" | 세미콜론, 마침표, 각주 |
| AI 어휘 | "delve", "landscape", "crucial" | "analyze", "recent work", "important" |
| 필러 | "It is important to note that" | 삭제 |

### 허용되는 예외

| 패턴 | 조건 |
|------|------|
| "In this paper" | Introduction/Conclusion에서 범위 설정 목적 |
| 단일 -ing 분사구 | 문장당 1개는 허용 (2개부터 수정) |
| "First" 주장 | 적절한 프레이밍과 함께 사용 가능 |

### 41개 Core Principles 요약

| # | 원칙 | 핵심 |
|---|------|------|
| 1 | Active voice | "CPU updates" not "is updated" |
| 2 | Figure 참조 순서 | 첫 언급: figure as subject, 이후: parenthetical |
| 3 | 문장당 1 아이디어 | 쉼표 2+ 이면 분리 |
| 4 | 용어 일관성 | 동의어 금지, 표준 동사 반복 |
| 5 | Anti-AI 어휘 | 약간의 반복 > 부자연스러운 다양성 |
| 6-10 | 구조 원칙 | Design에서 설명, Evaluation에서 참조만 |
| 11-15 | 스타일 원칙 | Em-dash 금지, Caption 콜론 금지, 용어 통일 |
| 16-20 | 참조/평가 원칙 | Claim-driven reference, 정량 evaluation |
| 21-25 | 프레이밍 원칙 | 제안 기법이 주어, 겸손한 기여 주장 |
| 26-30 | 용어/문장 원칙 | General→Specific, "This" 주어 금지 |
| 31-35 | 일관성 원칙 | 동일 개념 = 동일 용어, Subject-verb 정합 |
| 36-41 | 최종 검토 원칙 | 수식 절약, -ing 제한, 수치 일관성, 표 축약 |

</details>

---

## Decision Log (반영 로그) 운영

> **AI의 제안을 무비판적으로 수용하지 마세요.** 모든 피드백에 대해 주체적으로 판단하고 기록을 남기세요.

### 로그 형식

| # | AI 제안 | 우선순위 | 수용 여부 | 사유 |
|---|--------|---------|----------|------|
| 1 | Contribution에 "왜 중요한지" 누락 | Critical | **수용** | 실제로 빠져있었음 |
| 2 | "한편" → 논리적 bridge | Important | **부분 수용** | 2곳만 교체 |
| 3 | Related Work에 [논문X] 추가 | Important | **거부** | 다른 도메인 |
| 4 | "achieves" → "attains" | Nice | **거부** | 표준 동사가 더 적합 |

### 판단 기준

| 질문 | 수용 | 거부 |
|------|------|------|
| 리뷰어가 실제로 지적할 가능성? | 높으면 수용 | 낮으면 재고 |
| 내 연구 서사를 강화하는가? | O | 흐려지면 거부 |
| 의도적으로 선택한 구조/표현인가? | - | 의도적이면 거부 |
| 수정 비용 대비 효과? | O | 대규모 재구성이면 재고 |

### 이전 거부 항목 전달

```
이전 리뷰에서 거부한 항목: [로그 테이블 붙여넣기]
이번 리뷰에서는 이 항목들을 다시 제안하지 마.
```

---

## 자주 하는 실수 Top 10

| # | 실수 | 수정 |
|---|------|------|
| 1 | AI 제안 전부 수용 | Decision Log 작성 |
| 2 | "This"로 문장 시작 | 구체적 주어로 교체 |
| 3 | 동의어 돌려쓰기 | 표준 동사 반복 사용 |
| 4 | Evaluation에서 정성적 요약 | 정량 데이터 제시 |
| 5 | Limitation 섹션 별도 작성 | Discussion에 긍정 프레이밍 통합 |
| 6 | Abstract에 구현 디테일 | 핵심 개념 + 주요 수치만 |
| 7 | Figure 전방 참조 | 첫 언급 위치에서 완전 설명 |
| 8 | 수치 범위 누락 | 항상 모델/설정 명시 |
| 9 | Em-dash 남용 | 세미콜론/마침표/각주 |
| 10 | Conclusion = Introduction 복붙 | 결과 의의 + limitation + future work |
