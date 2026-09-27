# results

`experiments/`의 스크립트를 실제로 실행해 나온 값입니다. 로그 전체가 아니라 문서와 README에 적은 숫자를 다시 확인하는 데 필요한 최종 값, 실행 명령, 환경, seed, 소요 시간만 남겼습니다.

환경은 Python 3.14.7, NumPy 2.5.3, PyTorch 2.14.0, macOS arm64, CPU입니다. 실행일은 2026-09-28입니다. 말뭉치(`corpus.txt`)와 학습된 모델은 저장소에 넣지 않았고 `make_corpus.py`와 각 스크립트로 다시 만들 수 있습니다.

| 파일 | 명령 | 대응하는 숫자 | 쓰인 곳 |
|---|---|---|---|
| [param-count.txt](param-count.txt) | `python3 param_count.py` | PTB 설정 medium 19,775,200개, large 66,022,000개. 합성 말뭉치 모델 153,628 / 602,140 / 2,383,900개 | [01-why-scaling-rnn-overfits.md](../01-why-scaling-rnn-overfits.md) |
| [overfit.txt](overfit.txt) | `python3 overfit.py` | hidden 384의 학습 perplexity 1.154와 검증 2.664, 최저 검증 1.731(6 epoch) | [01-why-scaling-rnn-overfits.md](../01-why-scaling-rnn-overfits.md), [04-measuring-regularization.md](../04-measuring-regularization.md) |
| [mask-decay.txt](mask-decay.txt) | `python3 mask_decay.py` | $p^T$ 표와 표본 확인, 학습된 모델에서 마스크 종류별 닫는 이름 정답률 0.625 / 0.537 / 0.622 / 0.533 / 0.381 / 0.568 | [02-where-to-apply-dropout.md](../02-where-to-apply-dropout.md), [03-variational-dropout.md](../03-variational-dropout.md) |
| [dropout-placement.txt](dropout-placement.txt) | `python3 dropout_placement.py` | 검증 perplexity 없음 2.664, 비순환 1.850, 순환 per-step 2.227, 양쪽 1.747 | [02-where-to-apply-dropout.md](../02-where-to-apply-dropout.md) |
| [variational-dropout.txt](variational-dropout.txt) | `python3 variational_dropout.py` | per-sequence를 더하면 1.731, 양쪽 per-sequence 1.708, weight drop 1.753. 행 지우기 등식의 최대 절대차 0 | [03-variational-dropout.md](../03-variational-dropout.md) |
| [rate-sweep.txt](rate-sweep.txt) | `python3 rate_sweep.py` | hidden 384의 비율별 검증 perplexity 2.664 / 2.459 / 1.850 / 1.725 / 1.702, hidden 96의 1.982와 1.709 | [04-measuring-regularization.md](../04-measuring-regularization.md) |
| [ctc-forward.txt](ctc-forward.txt) | `python3 ctc_forward.py` | 프레임 6개 라벨 "cat"의 alpha 표와 P = 0.23127, 완전 탐색 경로 84개와 정확히 일치, torch ctc_loss와 차이 0, 수치 미분 상대 오차 최대 4.539e-08 | [05-ctc-alignment.md](../05-ctc-alignment.md), 루트 README |
| [ctc-decode.txt](ctc-decode.txt) | `python3 ctc_decode.py` | greedy CER 46.31%, 언어 모델 결합 beam 8에서 0.54%, beam 16 이상 0.36%, 언어 모델 없는 beam은 28%대에서 멈춤 | [06-ctc-decoding.md](../06-ctc-decoding.md), 루트 README |
| [ctc-stride.txt](ctc-stride.txt) | `python3 ctc_stride.py` | 라벨별 최소 스텝 수, stride 6에서 문자 단위 333/400, stride 8에서 144/400과 bigram 400/400, 계산량 1.97배 차이 | [07-stride-and-output-units.md](../07-stride-and-output-units.md) |
| [sequence-wise-batchnorm.txt](sequence-wise-batchnorm.txt) | `python3 sys_batchnorm.py` | 분산 추정의 상대 표준편차와 이론값 비교, 시점별 표본 수 24.00 / 12.13 / 3.53 / 1.76, frame error 0.2462 / 0.7749 / 0.2411 / 0.2378 | [08-sequence-wise-batchnorm.md](../08-sequence-wise-batchnorm.md) |
| [sortagrad-and-padding.txt](sortagrad-and-padding.txt) | `python3 sys_sortagrad.py` | 패딩 비율 0.4844와 0.0149, loss는 $T^{1.00}$, 기울기 노름은 $T^{0.97}$, 첫 epoch의 노름 최대 4.28e6과 128.2 | [09-sortagrad-and-padding.md](../09-sortagrad-and-padding.md) |
| [scaling-with-data.txt](scaling-with-data.txt) | `python3 sys_scaling.py` | 표 10의 구간별 감소율 52.79%와 38.70%, 회귀 지수 0.2729와 0.2137, 합성 실험의 0.4042에서 0.2251과 지수 0.1442 | [10-scaling-with-data.md](../10-scaling-with-data.md), 루트 README |

소요 시간은 다른 작업이 함께 돌던 상태에서 잰 값이라 환경에 따라 달라집니다. 학습이 들어간 스크립트는 `rate_sweep.py` 556초, `variational_dropout.py` 452초, `dropout_placement.py` 384초, `overfit.py` 215초입니다.
