# 생성 모델 캐시 연구 길잡이

**학부생용 · 단일 GPU 24–32GB · 2026-09-22**

[학생용 HTML 길잡이](/research-cache-guide-2026-09.html) · [기초](student-foundations.md) · [교수님용 판단](decision-report.md) · [Scoring 보정](scoring-calibration.md)

연구 진행 Meta: Mode1, harness a008f1e + 이전 Codex calibration 이식. 실험 미실행. 후보 채택·venue·외부 공개는 사용자 결정.

## 1. band 분포 · cross-lens idea 비율

- 전체15개 후보의 scope 분포: {'LOCAL': 2, 'WELL': 13, 'LARGE': 0}; cross-lens 비율 93%. Scope 분류는 과학적 성능·신규성 판정이 아니다.
- 원점수·보정후 점수 및 설계수정의 차이는 [Scoring 보고](scoring-calibration.md)에 있다.

## 2. Top-M (3–6)



- **추천 Build-up**: 캐시 저장량 감소 → history 의존성·복원 비용 확인(A4) → 같은 GPU cap에서 실제 비용을 고려한 보존 정책(S2). 평균 fidelity와 영상 실패가 같은 판단을 주는지는 A2로 함께 검증한다.
- **자원**: 사용자 확정 24–32GB 단일 GPU. 모델·실험은 실행하지 않았으며 모든 성능 목표는 사전 설계값이다.
- **평가 기준**: calibrated Accept는 실행할 가치가 있는 연구 제안이다. 성능 입증이나 논문 게재 예측이 아니다. S2/A4의 설계 보완과 rubric 보정은 구분했다.
- **구조**: 추천 카드 3개 = 두 연구 질문(S2·A2)과 S2 지원 진단(A4). 독립 방법 3개로 세지 않는다.

| 후보 | 전제·GAP | Overview | 기대 효과의 상태 | 5축 평균 | 최종 반대 검토 |
|---|---|---|---|---:|---|
| S2 | high-order state가 필요한 구간에서 history 삭제 후 재구성 비용이 남을 수 있음 | 연속 order prefix와 marginal rebuild cost를 함께 고려 | projected 채택 기준만; 실제 이득 unknown | 7.64 | WARN |
| A4 | 현재 입력의 full compute와 history/trajectory 회복은 다름 | 같은 schedule의 keep/clear와 같은 full-call 수의 placement를 분리 | S2 전제의 인과 진단; 효과 unknown | 7.52 | PASS — supporting only |
| A2 | 평균 fidelity·latency가 영상의 짧은 의미 실패를 대표하지 않을 수 있음 | matched operating point에서 cache policy 선택과 tail 진단 | measurement proposal; 새 가속 기법 아님 | 7.51 | WARN |

- **최신 선행과의 거리**: S2는 CG-Taylor/FreqCa/Xema, A4는 GP-Refiner/HiCache/BudCache, A2는 CachedSearch와 가깝다. Fresh 검색은 전부 concurrent로 분류했으며 신규성 확정으로 해석하지 않는다.
- **첫 중단 조건**: 고정 low-order/Lite가 충분하거나 모든 복원 작업이 예정 refresh에 흡수되면 S2의 독립 이득 가설을 지지하지 않는다. A2는 label 불일치에 민감한 rank reversal을 결론으로 사용하지 않는다.


- 5-reviewer mean은 연구 제안의 judgement이며, S2+A4는 한 흐름으로 구성한다.

## 3. 미선정 요약



| 묶음 | GAP·overview | 이번 주력에서 제외한 이유 |
|---|---|---|
| A1 | history order와 memory cap의 frontier | S2의 강한 static baseline·preliminary로 활용, 중복 기여 계산 방지 |
| S1/S4 | view ownership과 denoise/decode peak | common memory accounting으로 활용; 단순 clone/reset·기존 planner 대비 잔차 확인 우선 |
| B/E/F | codec·좌표·error correlation | B는 LOCAL measurement, DPCM 동치 위험; E/F는 후속 ablation 제안 |
| A3/C | CFG common/difference state | 같은 축의 중복과 실제 consumer/dataflow 보완 필요 |
| D | bounded coverage refresh | A2의 선택적 intervention; age bound가 quality bound는 아님 |
| A5 | conditioning 변경 invalidation | 좁은 engineering scope, 성능 연구의 잔여 질문 미정 |
| S3/S5 | batching/tile 실제 실행 절약 | 단일 GPU의 batching fit·수요, selected-query와 packing 경로 검증 우선 |


- [전체15개 비교](unselected.md). 모든 후보는 보존했고 자동 폐기하지 않았다.

## 4. 판단 필요

- Calibrated r2의 편차≥2.0은 0건이지만 실제 효과 확정이 아니다. S2의 high-order 필요성, A2의 label 신뢰도와 CachedSearch 대비 잔차를 먼저 확인한다.
- Decision Tree: source/hook smoke → A4 같은-call 진단 → 비용·품질 차이가 남으면 S2 정책 → A2 matched quality 평가. 저차/Lite로 충분하거나 복원이 예정 update에 흡수되면 확장을 보류한다.

## 5. scoop ⚠️

- Fresh 검색은 S2/A2/A4 모두 concurrent로 분류했다. 대표60%는 체크리스트 heuristic이며 확률이 아니다. [문헌 지도](literature-map.md).
- Essential Reading: [TaylorSeer](https://arxiv.org/abs/2503.06923), [CG-Taylor](https://arxiv.org/abs/2508.02240), [CachedSearch](https://arxiv.org/abs/2607.23159), [HiCache](https://arxiv.org/abs/2508.16984), [VBench](https://arxiv.org/abs/2311.17982). [읽을 부분·전체 지도](literature-map.md).

## 6. kill ⚙️

- S2 WARN: 저차에서는 rebuild charge가 사라질 수 있다. 고차가 실제로 필요한 regime을 먼저 입증한다.
- A2 WARN: candidate ranking 선행과의 차별화, semantic label 재현성 확인이 필요하다.
- A4 PASS: S2 지원 진단의 범위에 한정하며 독립 기여로 이중 계산하지 않는다.
- 별도 모델 gpt-5.6-sol 공격과 fresh 세션 모델 adjudication 뒤 계산기로 판정했다. 결과는 자동 채택·폐기가 아니다.

## 7. 파일 목록

- [학생용 HTML](https://aica-ssu.github.io/research-cache-guide-2026-09.html): 그림·개별 가설·평가·논문·기초·보정 보고를 한 파일에서 탐색.
- [기초](student-foundations.md), [S2](tier1/01-s2.md), [A4](tier2/01-a4.md), [A2](tier1/02-a2.md).
- [교수님용 판단](decision-report.md), [보정 전후](scoring-calibration.md), [문헌 지도](literature-map.md).
- 원문 증거와 실행 설계는 연구실 내부 bundle에 보관한다.

## 8. Mechanism Summary Table

| 문서 | 학생이 하는 일 | 처음 확인할 것 |
|---|---|---|
| A4 | 같은 계산 schedule에서 history만 개입 | clock/context·writer/reader 보존과 효과의 존재 |
| S2 | 보존량과 재구성 비용을 함께 고려 | 단순 저차·Lite보다 나은 이유 |
| A2 | 평균값과 실패 사례를 함께 비교 | blinded label 신뢰도·matched 조건의 성립 |

## 9. Glossary

모르는 용어는 CTRL+F 또는 Cmd+F로 찾는다. 상세 풀이와 코드 읽기 순서는 기초 파일에 있다.

- DiT — latent patch를 처리하는 생성 Transformer.
- VAE — latent를 영상으로 바꾸는 encoder/decoder 구성.
- History — 이후 feature 예측을 위해 남긴 과거 계산 상태.
- Prefix — 낮은 차수부터 빠짐없이 보존한 연속 상태.
- Refresh — 현재 입력에서 정확 계산을 다시 수행하는 것.
- Fidelity — 기준 결과와 닮은 정도이며 생성 품질 전체와 같지 않다.
- E2E — 입력부터 최종 영상까지 전체 실행 경로.
