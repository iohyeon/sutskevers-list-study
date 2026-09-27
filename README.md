# Sutskever's List Study

AlexNet, ResNet, RNN, LSTM을 원 논문의 수식, 숫자를 넣은 손계산, NumPy 구현, PyTorch 실험으로 다시 확인한 기술 노트입니다.

논문을 요약하기보다 아래 질문을 직접 확인하는 데 초점을 둡니다.

- 논문의 수식에 실제 숫자를 넣으면 어떻게 동작하는가
- 그 구조가 왜 필요했는가
- 논문이 말한 현상을 작은 실험에서도 볼 수 있는가
- 비슷한 모델을 같은 조건에서 비교하면 무엇이 달라지는가

## Highlights

| 주제 | 확인한 것 | 방법 | 결과 |
|---|---|---|---|
| AlexNet | 파라미터 수 | layer별로 (커널 높이 × 너비 × 입력 채널 + 1) × 필터 수를 계산해 합산. 두 GPU 분할 반영 | [60,965,224개](01-alexnet/09-alexnet-parameters-and-gpus.md). 분할을 반영하지 않으면 62,378,344개 |
| Convolution | 커널 학습 | NumPy로 합성곱과 역전파를 구현하고 무작위 3×3 커널을 경사하강법으로 학습 | [Sobel 커널로 수렴](01-alexnet/04-backpropagation.md) |
| ResNet | 깊이에 따른 기울기 감소 | layer 하나가 기울기를 0.9배로 만든다고 두고 152번 곱함 | [0.9¹⁵² ≈ 1.1 × 10⁻⁷](02-resnet/02-vanishing-gradient.md) |
| Bottleneck | 파라미터 수 비교 | 256채널 기준으로 bottleneck 블록과 3×3 conv 두 장을 계산 | [69,632개 대 1,179,648개](02-resnet/06-bottleneck.md) |
| RNN | BPTT 구현 검증 | 직접 유도한 기울기와 수치 미분을 비교 | [상대 오차 최대 8.1 × 10⁻⁹](03-rnn-lstm/05-bptt.md) |
| RNN과 LSTM | 먼 거리의 짝 맞추기 | 같은 합성 말뭉치로 n-gram, vanilla RNN, LSTM을 학습 | [51%, 46%, 100%](03-rnn-lstm/08-long-term-dependency.md) |

## Approach

주제마다 가능한 범위에서 같은 순서로 확인합니다.

1. 구조: 데이터가 흐르는 순서와 tensor shape
2. 유도: 중심이 되는 수식을 직접 유도
3. 손계산: 작은 숫자를 넣어 계산 과정을 확인
4. NumPy: 프레임워크 없이 알고리즘을 구현
5. PyTorch: 모델을 구현하고 비교 실험
6. 측정: 파라미터 수, 기울기, loss, perplexity
7. 해석: 결과가 논문의 설명과 어떻게 이어지는지

## Topics

### AlexNet

Krizhevsky, Sutskever, Hinton. [ImageNet Classification with Deep Convolutional Neural Networks](https://papers.nips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html) (2012)

2012년의 AlexNet은 CNN을 크게 만든 것 이상으로 무엇을 했는가. 합성곱과 pooling의 출력 크기를 처음부터 다시 계산하고, 두 GPU 구조를 반영해 layer별 파라미터 수와 곱셈 횟수를 계산했습니다.

역전파, convolution, receptive field, ReLU, dropout, data augmentation, SGD momentum, 두 GPU 분할

→ [AlexNet 문서](01-alexnet/README.md)

### ResNet

He, Zhang, Ren, Sun. [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385) (2015), [Identity Mappings in Deep Residual Networks](https://arxiv.org/abs/1603.05027) (2016)

layer를 더 쌓으면 왜 학습 오차까지 나빠졌고, 덧셈 하나가 무엇을 바꿨는가. 기울기가 깊이에 따라 줄어드는 정도와 skip connection이 역전파 식에 더하는 항을 계산했습니다.

degradation, 잔차 학습, batch normalization, bottleneck, pre-activation, dilated convolution, ResNeXt

→ [ResNet 문서](02-resnet/README.md)

### RNN과 LSTM

Hochreiter, Schmidhuber. [Long Short-Term Memory](https://www.bioinf.jku.at/publications/older/2604.pdf) (1997), Pascanu, Mikolov, Bengio. [On the difficulty of training Recurrent Neural Networks](https://arxiv.org/abs/1211.5063) (2013)

RNN은 왜 멀리 있는 정보를 잊고, LSTM은 그것을 어떻게 고쳤는가. BPTT를 유도해 NumPy로 구현하고, 같은 데이터에서 n-gram, vanilla RNN, LSTM을 비교했습니다.

글자 단위 language model, n-gram, BPTT, 기울기 소실과 폭발, LSTM cell state, forget gate

→ [RNN과 LSTM 문서](03-rnn-lstm/README.md)

## Reproducing the experiments

Python 3, NumPy, PyTorch가 필요합니다. GPU 없이 CPU에서 실행됩니다. 설치 방법과 나머지 실험은 [RNN과 LSTM 문서](03-rnn-lstm/README.md)에 있고, 실행 결과 요약은 [results](03-rnn-lstm/results/README.md)에 있습니다.

```bash
cd 03-rnn-lstm/experiments

python3 make_corpus.py
python3 ngram.py
python3 char_lstm.py --cell rnn  --epochs 40
python3 char_lstm.py --cell lstm --epochs 40
```

| 모델 | 시험 perplexity | 환경 짝 정답률 | 파라미터 |
|---|---:|---:|---:|
| 글자 n-gram (문맥 4) | 1.60 | 51% (35/69) | |
| vanilla RNN | 1.542 | 46% (57/125) | 39,449 |
| LSTM | 1.511 | 100% (126/126) | 138,521 |

LaTeX 문서에서는 `\begin{proof}` 로 연 환경을 `\end{proof}` 로 닫아야 합니다. 여는 이름과 닫는 이름이 평균 29글자 떨어져 있는 합성 말뭉치(148,458글자)를 만들고, 글자를 하나씩 예측하는 모델이 닫는 이름을 맞게 쓰는지 봤습니다. 환경 짝 정답률은 생성한 텍스트에서 여는 이름과 닫는 이름이 같은 줄의 비율이고, 생성 temperature 0.5에서 잰 값입니다.

합성 말뭉치 하나, 시드 하나의 결과입니다. 조건을 통제한 시연이고 일반적인 벤치마크가 아닙니다. 위 명령은 CPU에서 약 1분이 걸립니다.

## References

문서에서 다룬 논문과 공개 자료는 각 주제 폴더의 README 끝에 모았습니다.

- Krizhevsky et al., 2012. [ImageNet Classification with Deep Convolutional Neural Networks](https://papers.nips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html)
- He et al., 2015. [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
- He et al., 2016. [Identity Mappings in Deep Residual Networks](https://arxiv.org/abs/1603.05027)
- Ioffe, Szegedy, 2015. [Batch Normalization](https://arxiv.org/abs/1502.03167)
- Hochreiter, Schmidhuber, 1997. [Long Short-Term Memory](https://www.bioinf.jku.at/publications/older/2604.pdf)
- Pascanu et al., 2013. [On the difficulty of training Recurrent Neural Networks](https://arxiv.org/abs/1211.5063)
- Karpathy, 2015. [The Unreasonable Effectiveness of Recurrent Neural Networks](https://karpathy.github.io/2015/05/21/rnn-effectiveness/)
- Olah, 2015. [Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)

## Notes

Richard Heimann의 [『Sutskever's List: Foundational Ideas of Modern AI』](https://www.manning.com/books/sutskevers-list)(Manning)를 읽으며 관심이 생긴 논문들을 직접 따라가 보려고 시작했습니다. 책의 내용을 옮기지 않고, 원 논문의 수식과 구현을 이해하고 재현하는 것을 목표로 합니다.

계산이나 구현에서 오류를 발견하면 Issue로 알려 주세요.

관련 글: https://develop-tracking.tistory.com/
