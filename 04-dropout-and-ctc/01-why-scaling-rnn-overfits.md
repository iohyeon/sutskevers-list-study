# 파라미터를 늘린 LSTM은 왜 작은 말뭉치에서 과적합하는가?

## 문제

Zaremba, Sutskever, Vinyals의 2014년 논문 *Recurrent Neural Network Regularization*은 Penn Treebank에서 정규화를 걸지 않은 LSTM의 perplexity를 validation 120.7, test 114.5로 보고합니다. 같은 데이터에 같은 종류의 LSTM을 쓰면서 dropout만 걸면 2 layer 650 unit 모델(dropout 50%)이 86.2 / 82.7, 2 layer 1500 unit 모델(dropout 65%)이 82.2 / 78.4까지 내려갑니다. 논문은 정규화 없이 모델을 키우면 과적합 때문에 쓸 수 없다고 적었습니다.

그러면 무엇이 얼마나 남는지를 먼저 세어야 합니다. 파라미터 수와 학습에 쓸 수 있는 토큰 수를 같은 단위로 놓고 비교하고, 학습 perplexity와 검증 perplexity가 실제로 어떻게 벌어지는지를 작은 말뭉치에서 재 봅니다.

## 수식으로 보기

LSTM layer 하나는 게이트 4개(input, forget, cell, output)의 선형 변환을 한꺼번에 계산합니다. 입력 차원이 $d$, hidden 차원이 $H$ 이면 layer 하나의 파라미터는 다음과 같습니다.

$$
\underbrace{4H d}_{W_x} + \underbrace{4H^2}_{W_h} + \underbrace{4H}_{b}
$$

$W_x$ 는 입력을, $W_h$ 는 직전 hidden state를 받습니다. 게이트가 4개라서 두 행렬 모두 행이 $4H$ 개입니다. 여기에 입력 쪽 embedding과 출력 쪽 Linear를 더하면 언어 모델 전체가 됩니다. 어휘 크기를 $V$, embedding 차원을 $H$ 로 같게 잡고 layer를 $L$ 개 쌓으면 다음과 같습니다.

$$
P = \underbrace{VH}_{\text{embedding}} + \sum_{l=1}^{L} \big(4H^2 + 4H^2 + 4H\big) + \underbrace{HV + V}_{\text{출력 Linear}}
$$

첫 layer의 입력 차원이 embedding 차원과 같으므로 $W_x$ 도 $4H^2$ 입니다. 파라미터가 $H^2$ 에 비례하므로 hidden을 2배로 하면 순환 부분은 4배가 됩니다.

학습에 쓸 수 있는 신호의 양은 다르게 늘어납니다. 학습 토큰이 $N$ 개면 손실 항도 $N$ 개입니다. 그래서 $P / N$, 즉 토큰 하나가 담당하는 파라미터 수가 얼마나 큰지를 보면 외울 여지가 얼마나 있는지 짐작할 수 있습니다.

## 작은 숫자로 직접 계산

논문이 쓴 PTB 설정을 그대로 넣습니다. 어휘 10,000, 학습 토큰 929,589개(표준 전처리판의 학습 split), 2 layer입니다.

| 부분 | medium ($H = 650$) | large ($H = 1500$) |
|---|---|---|
| embedding $VH$ | 6,500,000 | 15,000,000 |
| LSTM layer 1 ($W_x + W_h + b$) | 1,690,000 + 1,690,000 + 2,600 | 9,000,000 + 9,000,000 + 6,000 |
| LSTM layer 2 | 1,690,000 + 1,690,000 + 2,600 | 9,000,000 + 9,000,000 + 6,000 |
| 출력 Linear $HV + V$ | 6,510,000 | 15,010,000 |
| 합계 | 19,775,200 | 66,022,000 |
| 학습 토큰 수 | 929,589 | 929,589 |
| 토큰 하나당 파라미터 | 21.3개 | 71.0개 |

large에서는 토큰 하나에 파라미터 71개가 걸립니다. 파라미터 하나하나가 독립적인 자유도는 아니지만, 이 비율이면 학습 데이터를 그대로 재현하는 해가 많이 존재합니다.

두 번째로 볼 것은 어디에 파라미터가 있는지입니다. medium은 embedding과 출력 Linear가 13,010,000개로 전체의 66%이고, large는 30,010,000개로 45%입니다. 어휘가 10,000이라 단어 하나를 표현하는 자리 자체가 무겁습니다. 반대로 순환 행렬 $W_h$ 는 medium에서 3,380,000개, large에서 18,000,000개로 hidden의 제곱으로 늘어납니다. 모델을 키운다는 말은 주로 이 순환 행렬을 키운다는 뜻입니다.

## 코드로 확인

`experiments/param_count.py`가 위 계산을 그대로 옮기고, 같은 설정으로 만든 PyTorch 모듈의 파라미터 수와 맞춰 봅니다.

```python
def lstm_layer_params(n_in, n_hidden):
    wx = 4 * n_hidden * n_in
    wh = 4 * n_hidden * n_hidden
    b = 4 * n_hidden
    return wx, wh, b
```

이 폴더의 실험 모델(어휘 28, 2 layer)에 대해 손으로 센 값과 `sum(p.numel() for p in model.parameters())`가 hidden 96, 192, 384에서 모두 153,628 / 602,140 / 2,383,900으로 일치했습니다([results/param-count.txt](results/param-count.txt)).

## 실험 결과

### 말뭉치

PTB를 쓰지 않고 `experiments/make_corpus.py`로 합성 말뭉치를 만들었습니다. 환경 이름 6개 중 하나로 열고, 단어 20개에서 무작위로 뽑은 본문을 5개에서 12개까지 쓰고, 같은 이름으로 닫습니다.

```
\begin{claim} by x y all be holds some \end{claim}
\begin{proof} x y the the y we y all \end{proof}
```

블록 400개, 전체 23,546글자, 어휘 28글자입니다. 앞 80%를 학습(18,836글자), 다음 10%를 검증(2,355글자), 나머지를 시험으로 썼습니다. 본문 단어가 무작위라서 외울 수는 있어도 일반화할 여지는 제한됩니다. 여는 태그와 닫는 태그 사이의 거리는 최소 14, 평균 32.3, 최대 52글자입니다.

### 크기를 키우면 벌어지는 정도

정규화를 아무것도 걸지 않고 hidden만 바꿔 50 epoch씩 학습시켰습니다(2 layer, 조각 길이 50, 배치 20, Adam lr 3e-3, `torch.manual_seed(0)`). perplexity는 dropout을 끈 상태로 각 split 전체를 한 번 훑어 계산했습니다.

| hidden | 파라미터 | 학습 글자당 | 최저 검증 PP (epoch) | 50 epoch 학습 PP | 50 epoch 검증 PP | 차이 |
|---|---|---|---|---|---|---|
| 96 | 153,628 | 8.2 | 1.738 (11) | 1.494 | 1.982 | +0.488 |
| 192 | 602,140 | 32.0 | 1.732 (6) | 1.232 | 2.538 | +1.306 |
| 384 | 2,383,900 | 126.6 | 1.731 (6) | 1.154 | 2.664 | +1.510 |

hidden 384의 epoch별 값입니다([results/overfit.txt](results/overfit.txt)).

| epoch | 1 | 5 | 10 | 15 | 20 | 30 | 40 | 50 |
|---|---|---|---|---|---|---|---|---|
| 학습 PP | 2.016 | 1.724 | 1.682 | 1.651 | 1.568 | 1.322 | 1.191 | 1.154 |
| 검증 PP | 2.009 | 1.750 | 1.740 | 1.769 | 1.797 | 2.039 | 2.457 | 2.664 |

학습 PP는 끝까지 내려가고 검증 PP는 epoch 6에서 1.731로 가장 낮았다가 다시 올라갑니다. 50 epoch에서 시험 PP는 2.699로 검증 PP와 비슷했습니다. 검증 split이 시험 split을 대신 판단해 준다는 뜻입니다.

## 결과 해석

세 모델의 학습 곡선 모양이 다릅니다. hidden 96은 40 epoch까지도 차이가 +0.308이었고, hidden 384는 30 epoch에 이미 +0.717입니다. 파라미터가 많으면 같은 데이터를 더 빨리, 더 깊이 외웁니다.

그런데 최저 검증 perplexity는 1.738, 1.732, 1.731로 거의 같습니다. 파라미터를 15배 늘려도 일반화 성능이 사실상 그대로였습니다. 이 말뭉치의 본문이 무작위라 줄일 수 없는 엔트로피가 있고, 세 모델 모두 그 한계에 이미 닿아 있었기 때문입니다. 즉 모델을 키워서 얻은 표현력이 전부 학습 데이터를 외우는 쪽으로 갔습니다. 논문이 정규화 없이는 큰 모델이 작은 모델보다 낫지 않다고 적은 것과 같은 방향의 결과입니다.

여기서 선택지가 둘로 나뉩니다. 하나는 검증 perplexity가 가장 낮은 epoch에서 멈추는 것입니다. 위 실험에서는 epoch 6이고, 그렇게 하면 hidden 384 모델의 파라미터 230만 개 중 대부분을 쓰지 않는 셈입니다. 다른 하나는 외우는 경로를 막아 두고 계속 학습시키는 것입니다. dropout이 그것이고, 어디에 걸어야 하는지는 [dropout을 어디에 걸 것인가](02-where-to-apply-dropout.md)에서 다룹니다.

## 한계와 주의할 점

- 위 실험은 합성 말뭉치 하나, seed 하나, 글자 단위 모델의 결과입니다. 논문의 PTB 숫자(120.7 / 114.5 등)는 단어 단위이고 어휘가 10,000이라 perplexity의 크기 자체를 비교할 수 없습니다. 비교할 수 있는 것은 학습과 검증이 벌어지는 방향입니다.
- 이 말뭉치는 본문 단어가 균일 무작위라서 검증 perplexity에 바닥이 있습니다. 실제 말뭉치에서는 모델을 키우면 정규화 없이도 검증 성능이 조금은 좋아지는 구간이 있습니다. 위 결과에서 최저 검증 PP가 세 크기에서 같게 나온 것은 말뭉치의 성질이 섞인 결과이고, 파라미터 수만의 효과로 읽으면 안 됩니다.
- 파라미터 수 대 토큰 수 비율은 과적합 가능성을 짐작하는 눈금이고 기준선은 아닙니다. 같은 비율에서도 데이터의 반복 구조, 학습 길이, 최적화 방법에 따라 결과가 달라집니다.
- PTB 토큰 수 929,589는 널리 쓰이는 전처리판의 학습 split 기준입니다. 전처리가 다르면 이 값이 달라집니다.

## References

- [Recurrent Neural Network Regularization](https://arxiv.org/abs/1409.2329) (Zaremba, Sutskever, Vinyals, 2014)
- [Dropout: A Simple Way to Prevent Neural Networks from Overfitting](https://jmlr.org/papers/v15/srivastava14a.html) (Srivastava 등, 2014)
- [Long Short-Term Memory](https://www.bioinf.jku.at/publications/older/2604.pdf) (Hochreiter, Schmidhuber, 1997)
