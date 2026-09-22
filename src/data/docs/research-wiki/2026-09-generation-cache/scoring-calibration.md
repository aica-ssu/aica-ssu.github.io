# Scoring 보정 — 원인, 적용, 해석

## 무엇이 빠져 있었나

- 초기 실행은 upstream 최신 `a008f1e`를 사용했으나, 이전 Codex 보정은 별도 `codex-conversion-local` fork에만 있어 적용되지 않았다.
- 이전 세션의 `docs/codex/ideation-calibration.md`와 배달·parser·gate 수정 사항을 현재 checkout에 이식했다. 원본 fork와 remote는 변경하지 않았다.
- **7.0은 논문 게재 예측이 아니라 검증할 가치가 있는 실행 가능한 연구 제안**의 기준이다. 평균7.0·최저축6.0과 기존 final gates는 유지했다.
- 미실험 상태만으로 모든 축을 낮게 묶지 않는다. `confirmed_defect`, `missing_evidence`, `out_of_scope`를 구분하고 각 축의 독립적 영향을 설명한다. 같은 결함의 중복 감점과 N/A=0 처리를 피한다.
- 모든 새 reviewer는 full calibration을 읽고 기존 점수·다른 reviewer 점수를 보지 않은 fresh context에서 평가했다. 총점은 명시한 sub-axis 가중합이다.

## 보정 전후

소수점은 계산 표시이며 객관적 신뢰도의 자릿수가 아니다. 연구 제안의 judgement이고 empirical performance가 아니다.

| 후보 | 초기 평균 | 재평가 평균 | 변화 | 비교 해석 |
|---|---:|---:|---:|---|
| S2 | 5.38 | 7.64 | +2.26 | rubric 재적용 + 구현·실험 설계 보완이 함께 반영됨 |
| A4 | 6.14 | 7.52 | +1.38 | rubric 재적용 + 구현·실험 설계 보완이 함께 반영됨 |
| A2 | 6.12 | 7.51 | +1.39 | mechanism 불변; rubric 재적용·논리표 명시·fresh review 변동 포함 |
| A1 | 5.92 | 7.20 | +1.28 | mechanism 불변; rubric 재적용·논리표 명시·fresh review 변동 포함 |
| S1 | 5.60 | 7.24 | +1.64 | mechanism 불변; rubric 재적용·논리표 명시·fresh review 변동 포함 |
| S4 | 5.34 | 7.26 | +1.92 | mechanism 불변; rubric 재적용·논리표 명시·fresh review 변동 포함 |
| B | 5.46 | 7.10 | +1.64 | rubric 재적용 + 구현·실험 설계 보완이 함께 반영됨 |
| A3 | 4.68 | 6.45 | +1.77 | mechanism 불변; rubric 재적용·논리표 명시·fresh review 변동 포함 |
| S3 | 4.32 | 6.30 | +1.98 | mechanism 불변; rubric 재적용·논리표 명시·fresh review 변동 포함 |
| S5 | 4.56 | 6.62 | +2.06 | mechanism 불변; rubric 재적용·논리표 명시·fresh review 변동 포함 |
| C | 4.52 | 6.50 | +1.98 | mechanism 불변; rubric 재적용·논리표 명시·fresh review 변동 포함 |
| D | 4.60 | 6.13 | +1.53 | mechanism 불변; rubric 재적용·논리표 명시·fresh review 변동 포함 |
| E | 4.76 | 6.53 | +1.77 | mechanism 불변; rubric 재적용·논리표 명시·fresh review 변동 포함 |
| F | 4.06 | 5.99 | +1.93 | mechanism 불변; rubric 재적용·논리표 명시·fresh review 변동 포함 |
| A5 | 4.08 | 5.73 | +1.65 | mechanism 불변; rubric 재적용·논리표 명시·fresh review 변동 포함 |

- 이 표는 Claude와 Codex를 통제 비교한 실험이 아니다. 상승분을 모델별 편향 크기나 일괄 가산점으로 사용할 수 없다.
- 새 점수가 높아도 실제 결함은 사라지지 않는다. A3의 observation-only 경로, S3의 batching 상태 계약 등은 구체적 구현 지적을 유지했다.
- B는 평균 기준을 충족해도 단일 family measurement의 LOCAL scope와 DPCM 중복 위험 때문에 이번 추천 포트폴리오에 넣지 않았다.
- S2/A4는 state 계약을 실제로 보완했으므로 점수 차이를 순수한 rubric 효과라고 해석하면 안 된다.

## 하네스 소프트웨어 검증

- `python3 -m pytest tests/unit -q`: **520 passed in 15.50s**.
- 규칙 무결성: **898/898**, `git diff --check` 통과.
- 독립 T2에 가짜 parent를 강제하던 parser 수정, 총점 범위·중복·라운드 혼합 검출, 5축/하한 검사, verdict-mode merge가 함께 포함된다.
- 이 검사는 하네스 소프트웨어의 회귀 테스트이며 연구 모델을 실행한 실험이 아니다.
- 개인 설치 symlink는 이전 작업의 별도-clone 범위를 존중해 변경하지 않았다. 현재 프로젝트에는 native `.agents/skills/aica-research-agent`와 새 calibration 배달 지침이 존재한다.

## 원본 보존

초기 리뷰와 점수는 `rounds/uncalibrated-r1/`, 새 평가의 입력 snapshot과 리뷰는 `rounds/calibrated-r2/`에 보존했다. 최종 문서의 국소 source integration 보완은 별도 기록하며 리뷰 당시 snapshot을 소급 수정하지 않는다.

최종 렌더링 점검에서는 독립 T2 설명과 scope-band dict 표시를 보완했다. 실제 dict 노출 회귀를 추가한 최종 전체 검사: **521 passed in 15.44s**. 연구 모델 실험과는 무관하다.
