# 지도교수용 판단 보고서

## 추천하는 Build-up

**주력 가설은 ‘제한된 GPU 메모리에서 predictor history를 버리고 복원하는 비용까지 고려한 캐시 관리’다.** 먼저 A4로 복원 비용의 의미를 분리하고, 유효한 차이가 있을 때 S2의 정책으로 이어간다. A2는 평균 품질 점수에 가려진 영상 실패를 확인하는 공통 평가 축으로 쓴다.

- 사용 가능 자원은 **24–32GB급 단일 GPU**다. Primary는 공개 Wan2.1-T2V-1.3B/CogVideoX-2b 계열의 작은 설정에서 시작하며, 실제 fit과 환경은 첫 smoke 단계에서 확인한다.
- 세 카드는 독립된 새로운 알고리즘 세 개를 뜻하지 않는다. **S2+A4는 한 연구 흐름**, A2는 응용 품질을 판별하는 연결 연구다.
- 제안 단계의 Accept는 ‘검증할 가치와 실행 가능한 경로가 있다’는 뜻이다. 실제 성능·품질 개선이나 논문 게재를 보증하지 않는다.

| 역할 | 후보 | 핵심 질문 | 첫 반증 | calibrated 평균 |
|---|---|---|---|---:|
| 주력 mechanism | S2 | 저장량만 보지 않고 history 복원에 드는 추가 계산까지 고려하면 같은 cap에서 유리한가 | 고정 낮은 order/Lite/CG-Taylor 또는 myopic policy가 같은 frontier를 얻는가 | 7.64 |
| prerequisite diagnosis | A4 | 같은 정확 계산 횟수에서 배치 방식과 history 보존/초기화의 효과를 분리할 수 있는가 | 좋은 결과가 추가 계산이나 schedule 변경만으로 설명되는가 | 7.52 |
| 응용 평가 | A2 | 평균 fidelity와 latency가 비슷한 cache policy가 motion·identity failure에서는 다르게 평가되는가 | 기존 평균 지표만으로 선택이 충분히 설명되는가 | 7.51 |

점수는 5개 fresh reviewer의 가중합 총점을 통합한 judgement다. 수치의 소수점은 계산 표시다. 새 score와 원점수, 보정 해석은 [scoring-calibration.md](scoring-calibration.md)에 있다.

## 왜 기존 ViT layer 탐색에서 이렇게 옮기는가

- 학생의 activation·layer 분석 경험은 재사용할 수 있다. 생성 모델에서는 여기에 denoising step과 history state의 시간적 의존성이 더해진다.
- ‘비슷한 feature를 cache한다’, ‘중요한 step을 찾는다’, ‘오차가 커지면 refresh한다’는 broad mechanism은 이미 많은 연구가 다룬다. [TeaCache](https://arxiv.org/abs/2411.19108), [TaylorSeer](https://arxiv.org/abs/2503.06923), [EpaCache](https://arxiv.org/abs/2608.29264), [GP-Refiner](https://arxiv.org/abs/2609.05981)가 중요한 비교군이다.
- 메모리도 빈 영역은 아니다. [CG-Taylor](https://arxiv.org/abs/2508.02240), [FreqCa](https://arxiv.org/abs/2510.08669), [LightCache](https://arxiv.org/abs/2510.05367), [Xema](https://arxiv.org/abs/2607.11136)를 이겨야 할 범위를 먼저 정한다.
- 이번 차별화 가설은 특정 layer 번호가 아니라 **predictor state가 의존성을 가진다는 점과 회수 이후의 재구성 비용**에 있다. 그 비용이 실제로 중요하지 않으면 더 큰 기법을 만들지 않는다.

## 코드에서 확인한 것과 아직 모르는 것

| 구분 | 내용 |
|---|---|
| source 확인 | TaylorSeer update는 이전 factor가 없으면 다음 높은 order 생성이 끊긴다. stock predict는 연속 order prefix와 tuple-output 간 같은 order 개수를 가정한다. |
| analytic | prefix0..r에서 최고차수p의 key 복구는 max(0,p−r)번 exact update가 필요하다. 새 관측만으로 구성한 history는 별개 문제다. |
| 금지할 비약 | exact update 수를 그대로 추가 full-model 호출 수나 E2E 시간으로 바꾸지 않는다. 원래 예정된 refresh와 공유 계산을 제외해야 한다. |
| 미측정 | 24–32GB 사용자 환경에서 history가 실제 peak를 지배하는지, 재구성 비용이 얼마나 큰지, 정책이 기존 방식보다 유리한지. |
| 첫 구현 위험 | default TaylorSeer regex가 대상 model의 attn1 이름을 놓칠 수 있다. source에서 확인해 명시 pattern 및 hook-count smoke 요구를 추가했다. |

Pinned source는 [TaylorSeer state와 hook](https://github.com/huggingface/diffusers/blob/fbf49e7f35857f76bc57b177e26f12b03687c668/src/diffusers/hooks/taylorseer_cache.py)다. 확인은 code inspection이며 model execution이 아니다.

## 학생에게 줄 첫 과제

1. 기초 자료를 읽고 denoising step, network block, video frame/chunk를 구분하는 그림을 그린다.
2. 모델 하나와 baseline 하나를 고정하고 hook이 실제로 붙는지 확인한다. 초기 latent, scheduler, guidance, precision, offload 조건을 기록한다.
3. A4의 작은 matched-budget 비교로 history key 복구와 실제 품질 회복을 구분한다. 동시에 A2 방식으로 motion·identity 실패를 확인한다.
4. 복원 비용이 무시할 수 없을 때만 S2 정책을 구현한다. 고정 낮은 order나 last-block-only가 충분하면 그 결과를 지도교수와 검토하고 범위를 재결정한다.

큰 grid를 먼저 돌리지 않는다. GPU 시간은 첫 baseline의 실제 소요시간으로 산정하며 이 문서가 특정 소요시간을 보장하지 않는다.

## 미선정 후보를 어떻게 활용할까

- A1의 history-order/cap frontier는 S2의 preliminary·강한 static baseline으로 포함하도록 권한다. 같은 효과를 별도 novelty로 두 번 세지 않는다.
- S1/S4의 ownership·phase peak 계측은 common preliminary에서 활용하되, always-clone/last-reader reset 또는 기존 memory planner만으로 충분한지 먼저 본다.
- B의 packed-history codec은 후속 선택지다. DPCM 구조 자체가 새롭지 않고 direct INT8가 더 작은 payload를 가질 수 있다. 범위를 좁힌 뒤 LOCAL 판정도 보존했다.
- A3/C는 같은 CFG 좌표 표현의 중복 위험, S3는 실제 batching 수요와 단일 GPU fit, S5는 selected-query 경로와 packing 비용 확인이 우선이다.
- 원래 15개 후보는 삭제하지 않았다. 미선정은 자동 폐기나 사용자 채택 결정이 아니다.

## 판단 편차와 남은 검증

Calibrated r2에서 축간 편차≥2.0인 후보는 없다. 이는 같은 proposal rubric 아래 점수 편차가 작다는 뜻이며 실험적 확실성이나 모델 간 독립 합의를 뜻하지 않는다. 원 r1의 큰 편차·낮은 점수는 별도 보존했다.

- 특히 단순 baseline 대비 잔여 이득, 실제 runtime fit, predictor-state 비용과 오류의 관계는 계속 검증할 가설이다.
- 연구 방향·제목·venue·최종 채택은 교수님 결정이다. 하네스의 포트폴리오는 선택을 돕는 권고다.

## Final Gates

| 후보 | Fresh literature screening | 독립 반대 검토 | 교수님이 먼저 확인할 것 |
|---|---|---|---|
| S2 | concurrent — CG-Taylor/FreqCa/Xema와 가까움 | WARN | 고차 history가 실제로 필요한가; 예정 refresh 차감 뒤에도 추가 복원 비용이 남는가 |
| A2 | concurrent — CachedSearch와 가까움 | WARN | candidate ranking과 policy ranking의 차이가 유의미한가; semantic label이 재현 가능한가 |
| A4 | concurrent — GP-Refiner/HiCache/BudCache와 가까움 | PASS, S2 지원 진단 범위에 한정 | 진단을 독립 새 알고리즘으로 이중 계산하지 않는가 |

- Attack은 다른 모델 `gpt-5.6-sol`, adjudication은 별도 fresh 세션 모델 context가 수행했다. 각 후보를 atomic point로 분해한 뒤 스크립트가 판정했다.
- Screening의 대표 유사도는 세 후보 모두 coarse checklist의 analytic 60%다. 확률이나 정밀한 novelty 점수가 아니며, 전체 논문 부재 증명도 아니다.
- S2의 기본 낮은 차수에서 debt=0인 경우는 유효한 반례다. 더 높은 차수를 켰다는 사실만으로 유용성을 입증하지 못하므로 low-order가 불충분한 workload가 먼저 필요하다.
- A2는 method-blinded 반복 또는 독립 annotation을 수행하고 disagreement/unknown을 보수적으로 처리해도 차이가 남는지 먼저 본다. 작은 pilot의 결과만으로 희귀 실패율을 확정하지 않는다.
- WARN은 자동 폐기나 허위 성공이 아니다. 남은 검증을 첫 과제로 드러낸다. 자세한 원문은 evidence/의 attack·kill·scoop 문서에 보존했다.

## 작업 범위

- 수행: 외부 문헌 실존·내용 확인, 공식 code clone/anchor 검증, 세 관점의 교차 분석, 15개 proposal, 독립 5축 평가 및 calibration 재평가, 문서·실험 설계.
- 미수행: 연구 모델 다운로드, GPU generation/benchmark, 기법 구현·품질/성능 실측, 외부 공개·push·송부.
- 하네스 소프트웨어 보정의 520개 regression test는 연구 모델 실험과 구분한다.
