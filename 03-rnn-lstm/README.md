# RNN과 LSTM

순서가 있는 데이터에서 다음 글자를 예측하는 모델인 n-gram, vanilla RNN, LSTM을 다룹니다. 문장의 확률을 어떻게 쓰는지에서 시작해 RNN과 LSTM의 한 시점을 손으로 계산하고, BPTT를 유도해 NumPy로 구현한 뒤 수치 미분과 맞췄습니다. 마지막으로 세 모델을 같은 합성 말뭉치에 학습시켜, 멀리 떨어진 정보를 기억해야 하는 과제에서 어떻게 갈리는지 쟀습니다.

## 문서

| 문서 | 질문 | 계산하거나 보여 주는 것 |
|---|---|---|
| [01-language-model.md](01-language-model.md) | 글자 단위 language model은 문장의 확률을 어떻게 계산하는가? | 연쇄법칙으로 문장의 확률을 조건부 확률의 곱으로 쪼개는 계산, one-hot에 행렬을 곱하면 열 하나를 고르는 것과 같다는 계산 |
| [02-softmax-and-perplexity.md](02-softmax-and-perplexity.md) | language model의 출력은 어떻게 확률이 되고, 그 확률은 어떻게 평가되는가? | softmax와 temperature 0.5, 2.0의 분포 손계산, cross-entropy와 perplexity의 관계, nat과 bit |
| [03-ngram-baseline.md](03-ngram-baseline.md) | 글자 n-gram은 무엇을 할 수 있고, 직전 n글자 밖의 정보는 왜 쓸 수 없는가? | 15글자 말뭉치로 만든 bigram, trigram 표, 문맥 길이별 perplexity와 환경 짝 정답률 |
| [04-rnn.md](04-rnn.md) | RNN은 길이가 정해지지 않은 입력을 고정된 수의 파라미터로 어떻게 처리하는가? | hidden 3차원 RNN에 "hell"을 넣은 네 시점의 상태와 출력 확률, NumPy 구현 |
| [05-bptt.md](05-bptt.md) | BPTT에서 기울기는 시간 방향으로 어떻게 전파되고, 왜 거리에 따라 줄어드는가? | 점화식 유도, 수치 미분과의 비교, 야코비안 곱의 크기 실측, gradient clipping, truncated BPTT |
| [06-lstm.md](06-lstm.md) | LSTM은 vanilla RNN의 상태 갱신을 어떻게 바꾸었는가? | gate와 cell state 수식, 2차원 예제의 한 시점 손계산, 파라미터 수 계산, GRU |
| [07-lstm-gradient-path.md](07-lstm-gradient-path.md) | LSTM의 cell state는 기울기가 지나는 경로를 어떻게 바꾸는가? | cell state 경로의 미분, ResNet의 skip connection과의 대응, forget gate 실측값, forget gate가 추가된 과정 |
| [08-long-term-dependency.md](08-long-term-dependency.md) | vanilla RNN과 LSTM은 먼 거리의 짝 맞추기에서 얼마나 다르게 동작하는가? | 합성 말뭉치에서 세 모델의 perplexity와 환경 짝 정답률, forget gate bias 초기값의 영향 |
| [09-interpreting-cell-state.md](09-interpreting-cell-state.md) | 학습된 LSTM은 환경 이름을 cell state의 어느 차원에 저장하는가? | 환경을 가르는 차원 찾기, 그 값을 덮어쓰는 개입 실험, temperature별 생성 결과 |
| [10-limits-of-recurrence.md](10-limits-of-recurrence.md) | 순환 구조에는 어떤 한계가 남았고, attention은 그것에 어떻게 답하는가? | 순차 계산 단계 수, 학습 시간 실측, 두 위치 사이의 경로 길이 비교 |

## 실험

| 스크립트 | 확인하려는 것 | 결과 파일 | 결과 |
|---|---|---|---|
| `gradient_check.py` | 직접 유도한 BPTT 기울기가 수치 미분과 일치하는가 | [gradient-check.txt](results/gradient-check.txt) | 상대 오차 최대 8.1 × 10⁻⁹ |
| `hand_calc.py` | 문서의 손계산 값이 코드 출력과 일치하는가 | [hand-calc.txt](results/hand-calc.txt) | "hell" 네 시점, LSTM 한 시점, softmax 값이 일치 |
| `ngram.py` | 직전 n글자만 보는 모델은 먼 거리의 짝을 맞출 수 있는가 | [ngram.txt](results/ngram.txt) | 문맥 4에서 perplexity 1.604, 환경 짝 35/69 |
| `char_rnn.py` | NumPy로 짠 vanilla RNN은 같은 과제를 푸는가 | [rnn-numpy.txt](results/rnn-numpy.txt) | perplexity 1.561, 환경 짝 73/136 |
| `char_lstm.py --cell rnn` | vanilla RNN은 평균 29글자 떨어진 이름을 기억하는가 | [rnn.txt](results/rnn.txt), [rnn-hidden256.txt](results/rnn-hidden256.txt) | perplexity 1.542, 환경 짝 57/125. 파라미터 수를 맞춰도 78/135 |
| `char_lstm.py --cell lstm` | LSTM은 같은 거리에서 기억하는가 | [lstm.txt](results/lstm.txt) | perplexity 1.511, 환경 짝 126/126 |
| `char_lstm.py --forget-bias 1.0` | forget gate bias 초기값이 학습 속도를 바꾸는가 | [forget-bias.txt](results/forget-bias.txt) | 6 epoch에서 기본값 62/126, bias 1.0은 125/125 |
| `probe.py`, `fgate.py` | 환경 이름은 cell state의 어느 차원에 있고 그 차원의 forget gate는 얼마인가 | [probe.txt](results/probe.txt), [fgate.txt](results/fgate.txt) | 차원 69, 88, 61. forget gate 평균 0.999, 0.870, 0.999 |
| `intervene.py` | 그 차원의 값을 바꾸면 닫는 이름이 바뀌는가 | [intervene.txt](results/intervene.txt) | 1개를 바꾸면 1/200과 51/200, 3개를 바꾸면 200/200 |

환경 짝은 생성한 텍스트에서 `\begin{X}` 로 연 줄이 `\end{X}` 로 닫힌 수와 전체 수입니다. 합성 말뭉치 하나, seed 하나의 결과이고 조건과 한계는 각 문서의 「한계와 주의할 점」에 적었습니다.

## 재현

Python 3, NumPy, PyTorch가 필요합니다. GPU 없이 CPU에서 실행됩니다.

```bash
cd 03-rnn-lstm/experiments
python3 -m venv .venv
.venv/bin/pip install numpy torch

.venv/bin/python make_corpus.py          # corpus.txt 생성 (148,458글자)
.venv/bin/python gradient_check.py
.venv/bin/python hand_calc.py
.venv/bin/python ngram.py
.venv/bin/python char_rnn.py 96 64 12000
.venv/bin/python char_lstm.py --cell rnn  --epochs 40
.venv/bin/python char_lstm.py --cell lstm --epochs 40   # model_lstm.pt 저장
.venv/bin/python probe.py                # 아래 세 개는 40 epoch LSTM 모델을 읽습니다
.venv/bin/python fgate.py
.venv/bin/python intervene.py
```

전체 실행에 CPU에서 3분 안쪽이 걸립니다. 실행한 환경과 출력은 [results/README.md](results/README.md)에 있습니다.

## 참고 자료

- [The Unreasonable Effectiveness of Recurrent Neural Networks](https://karpathy.github.io/2015/05/21/rnn-effectiveness/) (Karpathy, 2015): 글자 단위 RNN으로 텍스트를 생성한 실험과 뉴런 시각화를 소개한 글
- [karpathy/char-rnn](https://github.com/karpathy/char-rnn): 위 글에서 쓴 multi-layer RNN, LSTM, GRU 글자 단위 언어 모델 코드
- [Minimal character-level language model with a Vanilla Recurrent Neural Network](https://gist.github.com/karpathy/d4dee566867f8291f086) (Karpathy): numpy만으로 짠 최소 구현
- [Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/) (Olah, 2015): LSTM의 cell state와 게이트를 그림으로 설명한 글
- [The unreasonable effectiveness of Character-level Language Models](https://nbviewer.org/gist/yoavg/d76121dfde2618422139) (Goldberg, 2015): smoothing 없는 글자 n-gram으로 같은 종류의 텍스트를 만들어 비교한 글
- [Long Short-Term Memory](https://www.bioinf.jku.at/publications/older/2604.pdf) (Hochreiter, Schmidhuber, 1997): LSTM을 제안한 논문
- [On the difficulty of training Recurrent Neural Networks](https://arxiv.org/abs/1211.5063) (Pascanu, Mikolov, Bengio, 2013): RNN의 기울기 소실과 폭발 조건, gradient clipping
- [An Empirical Exploration of Recurrent Network Architectures](https://proceedings.mlr.press/v37/jozefowicz15.html) (Jozefowicz, Zaremba, Sutskever, 2015): RNN 구조 탐색과 forget gate bias 초기화
- [LSTM: A Search Space Odyssey](https://arxiv.org/abs/1503.04069) (Greff 등, 2015): LSTM 변형들의 비교 실험
- [Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation](https://arxiv.org/abs/1406.1078) (Cho 등, 2014): GRU를 제안한 논문
- [Recurrent neural network based language model](https://www.isca-archive.org/interspeech_2010/mikolov10_interspeech.html) (Mikolov 등, 2010): RNN 언어 모델과 n-gram의 perplexity, WER 비교
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (Vaswani 등, 2017): 순환 없이 attention만으로 구성한 Transformer
- [Recurrent Neural Network Regularization](https://arxiv.org/abs/1409.2329) (Zaremba, Sutskever, Vinyals, 2014): dropout을 layer 사이 연결에만 거는 방법
- [Highway Networks](https://arxiv.org/abs/1505.00387) (Srivastava, Greff, Schmidhuber, 2015): LSTM의 게이트를 layer 방향에 옮긴 구조
- [Sequence to Sequence Learning with Neural Networks](https://arxiv.org/abs/1409.3215) (Sutskever, Vinyals, Le, 2014): 입력 문장을 벡터 하나로 줄이는 encoder-decoder LSTM
- [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473) (Bahdanau, Cho, Bengio, 2014): RNN 위에 얹은 attention
- [Deep Speech 2](https://arxiv.org/abs/1512.02595) (Amodei 등, 2015): RNN 기반 음성 인식을 여러 GPU로 학습시킨 시스템
- [Mapping the Mind of a Large Language Model](https://www.anthropic.com/research/mapping-mind-language-model) (Anthropic, 2024): LLM 내부에서 개념 단위의 feature를 찾는 연구
