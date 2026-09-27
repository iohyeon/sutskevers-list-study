# results

`experiments/`의 스크립트를 실제로 실행해 나온 값입니다. 로그 전체가 아니라 문서와 README에 적은 숫자를 다시 확인하는 데 필요한 최종 값, 실행 명령, 환경, seed, 소요 시간만 남겼습니다.

환경은 Python 3.14.7, NumPy 2.5.3, PyTorch 2.14.0, macOS arm64, CPU입니다. 실행일은 2026-09-21입니다. 말뭉치(`corpus.txt`)와 학습된 모델(`*.pt`)은 저장소에 넣지 않았고 `make_corpus.py`와 학습 스크립트로 다시 만들 수 있습니다.

| 파일 | 명령 | 대응하는 숫자 | 쓰인 곳 |
|---|---|---|---|
| [gradient-check.txt](gradient-check.txt) | `python3 gradient_check.py` | BPTT 기울기와 수치 미분의 상대 오차 최대 8.1 × 10⁻⁹ | [05-bptt.md](../05-bptt.md), 루트 README |
| [hand-calc.txt](hand-calc.txt) | `python3 hand_calc.py` | "hell" 네 시점의 h와 p, loss 1.425, temperature별 softmax, LSTM 한 시점, 야코비안 곱의 크기 | [02-softmax-and-perplexity.md](../02-softmax-and-perplexity.md), [04-rnn.md](../04-rnn.md), [05-bptt.md](../05-bptt.md), [06-lstm.md](../06-lstm.md) |
| [ngram.txt](ngram.txt) | `python3 ngram.py` | 문맥 4에서 perplexity 1.604, 환경 짝 35/69. 문맥 8과 12의 값 | [03-ngram-baseline.md](../03-ngram-baseline.md), [08-long-term-dependency.md](../08-long-term-dependency.md), 루트 README |
| [rnn-numpy.txt](rnn-numpy.txt) | `python3 char_rnn.py 96 64 12000` | NumPy vanilla RNN의 perplexity 1.561, 환경 짝 73/136 | [04-rnn.md](../04-rnn.md), [08-long-term-dependency.md](../08-long-term-dependency.md) |
| [rnn.txt](rnn.txt) | `python3 char_lstm.py --cell rnn --epochs 40` | vanilla RNN의 perplexity 1.542, 환경 짝 57/125, 파라미터 39,449 | [08-long-term-dependency.md](../08-long-term-dependency.md), 루트 README |
| [lstm.txt](lstm.txt) | `python3 char_lstm.py --cell lstm --epochs 40` | LSTM의 perplexity 1.511, 환경 짝 126/126, 파라미터 138,521 | [08-long-term-dependency.md](../08-long-term-dependency.md), 루트 README |
| [rnn-hidden256.txt](rnn-hidden256.txt) | `python3 char_lstm.py --cell rnn --hidden 256 --epochs 40` | 파라미터 수를 맞춘 vanilla RNN의 perplexity 1.556, 환경 짝 78/135 | [08-long-term-dependency.md](../08-long-term-dependency.md) |
| [forget-bias.txt](forget-bias.txt) | `python3 char_lstm.py --cell lstm`에 `--epochs 6`, `--epochs 6 --forget-bias 1.0`, `--epochs 12`, `--epochs 12 --forget-bias 1.0` | 기본 초기화 6 epoch 62/126, forget bias 1.0으로 6 epoch 125/125, 12 epoch에서는 127/127과 130/130 | [07-lstm-gradient-path.md](../07-lstm-gradient-path.md), [08-long-term-dependency.md](../08-long-term-dependency.md) |
| [probe.txt](probe.txt) | `python3 probe.py` | 환경을 가르는 cell state 차원 69, 88, 61과 분리도 3.52, 1.39, 1.32 | [09-interpreting-cell-state.md](../09-interpreting-cell-state.md) |
| [fgate.txt](fgate.txt) | `python3 fgate.py` | 그 세 차원의 forget gate 평균 0.999, 0.870, 0.999와 전체 평균 0.525 | [07-lstm-gradient-path.md](../07-lstm-gradient-path.md), [09-interpreting-cell-state.md](../09-interpreting-cell-state.md) |
| [intervene.txt](intervene.txt) | `python3 intervene.py` | 차원 1개를 덮어쓰면 1/200과 51/200, 3개를 덮어쓰면 200/200 | [09-interpreting-cell-state.md](../09-interpreting-cell-state.md) |

소요 시간은 다른 작업이 함께 돌던 상태에서 잰 값이라 환경에 따라 달라집니다. 환경 짝은 temperature 0.5로 6,000글자를 생성해 센 값입니다(n-gram은 분포 그대로 샘플링).
