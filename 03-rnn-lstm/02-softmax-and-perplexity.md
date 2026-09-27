# language model의 출력은 어떻게 확률이 되고, 그 확률은 어떻게 평가되는가?

## 문제

모델이 내놓는 값은 확률이 아니라 글자마다의 점수(logit)입니다. 이 점수를 확률로 바꾸는 softmax, 생성할 때 분포를 조절하는 temperature, 학습과 평가에 쓰는 cross-entropy와 perplexity를 손계산으로 확인합니다. 뒤의 실험 문서들이 비교에 쓰는 perplexity 값이 무엇을 뜻하는지가 여기서 정해집니다.

## 수식으로 보기

### softmax

모델의 마지막 layer는 어휘의 각 토큰에 실수 점수를 하나씩 냅니다. 이 점수를 **logit**이라고 부르고 $z_i$로 씁니다. logit은 음수일 수도 있고 합이 1도 아닙니다. softmax는 이것을 확률 분포로 바꿉니다.

$$
p_i = \frac{e^{z_i}}{\sum_{j=1}^{|V|} e^{z_j}}
$$

- $e^{z}$: 지수 함수. 어떤 실수든 양수로 바꿉니다.
- 분모: 모든 토큰의 $e^{z_j}$를 더한 값. 나누면 합이 1이 됩니다.

#### 손계산

어휘가 3개이고 logit이 $z = [2.0,\; 1.0,\; 0.1]$ 이라고 가정합니다.

| 토큰 | $z_i$ | $e^{z_i}$ | $p_i$ |
|---|---|---|---|
| A | 2.0 | 7.389 | 7.389 / 11.212 = **0.659** |
| B | 1.0 | 2.718 | 2.718 / 11.212 = **0.242** |
| C | 0.1 | 1.105 | 1.105 / 11.212 = **0.099** |
| 합 | | 11.212 | 1.000 |

logit의 차이가 1이면 확률의 **비율**이 $e^1 \approx 2.72$배입니다. softmax는 logit의 차이를 확률의 비율로 바꿉니다. 모든 logit에 같은 수를 더해도 결과는 같습니다(분자와 분모에 같은 수가 곱해지므로). 실습 코드에서 `y -= y.max()`를 하는 것은 이 성질을 이용해 $e^{z}$가 넘치는 것을 막는 것입니다.

### temperature

Karpathy의 char-rnn은 모델이 낸 확률 분포에서 글자를 하나씩 뽑아 텍스트를 만들고, 뽑을 때의 무작위성을 temperature라는 값으로 조절합니다. temperature가 낮으면 확률이 높은 글자만 나와서 반복적인 텍스트가 되고, 높으면 다양해지는 대신 철자와 구조가 흐트러집니다.

#### 수식

softmax에 넣기 전에 logit을 $T$로 나눕니다.

$$
p_i(T) = \frac{e^{z_i / T}}{\sum_j e^{z_j / T}}
$$

$T$는 학습되는 값이 아니고 **생성할 때 사람이 정하는 값**입니다. 모델의 가중치는 그대로입니다.

#### 손계산

같은 logit $[2.0, 1.0, 0.1]$에 $T$를 바꿔 적용합니다.

| $T$ | $z / T$ | $p_A$ | $p_B$ | $p_C$ | 읽는 법 |
|---|---|---|---|---|---|
| 0.5 | [4.0, 2.0, 0.2] | 0.864 | 0.117 | 0.019 | 1등에 몰립니다 |
| 1.0 | [2.0, 1.0, 0.1] | 0.659 | 0.242 | 0.099 | 모델이 학습한 분포 그대로 |
| 2.0 | [1.0, 0.5, 0.05] | 0.502 | 0.304 | 0.194 | 평평해집니다 |
| 10.0 | [0.2, 0.1, 0.01] | 0.366 | 0.331 | 0.303 | 거의 균등 |

$T = 0.5$ 한 줄만 직접 계산해 봅니다. $e^{4.0} = 54.60$, $e^{2.0} = 7.389$, $e^{0.2} = 1.221$, 합은 63.21 입니다. $54.60 / 63.21 = 0.864$ 가 맞습니다.

두 극한은 다음과 같습니다.

- $T \to 0$: 가장 큰 logit의 확률이 1이 됩니다. 항상 1등만 고릅니다(greedy). 같은 말을 반복하기 쉽습니다.
- $T \to \infty$: 모든 확률이 $1/|V|$ 가 됩니다. 모델을 무시하고 주사위를 던지는 것과 같습니다.

logit 차이가 확률 비율을 정한다는 성질로 보면 더 간단합니다. $T$로 나누면 **logit 차이가 $1/T$배**가 됩니다. $T = 0.5$면 차이가 2배로 벌어지고 $T = 2$면 절반으로 줄어듭니다.

### 샘플링

분포가 정해지면 그 분포에 따라 토큰 하나를 뽑습니다. numpy로는 한 줄입니다.

```python
ix = rng.choice(vocab_size, p=prob)
```

뽑기 방법에는 여러 변형이 있습니다. 이름만 적어 둡니다.

| 방법 | 하는 일 |
|---|---|
| greedy | 확률이 가장 높은 것만 고릅니다. $T \to 0$과 같습니다 |
| temperature 샘플링 | 위에서 설명한 것. char-rnn이 쓴 방법 |
| top-k | 확률 상위 $k$개만 남기고 나머지를 0으로 만든 뒤 샘플링합니다 |
| top-p (nucleus) | 확률을 큰 것부터 더해 $p$를 넘을 때까지의 토큰만 남깁니다 |

### 정답에 준 확률로 점수를 매긴다

모델은 매 위치에서 어휘 전체에 대한 확률 분포를 냅니다. 평가할 때 보는 것은 그중 **실제로 나온 글자에 준 확률** $p_t$ 하나입니다.

- $p_t = 1$: 완벽하게 맞혔습니다.
- $p_t = 0.5$: 둘 중 하나라고 봤습니다.
- $p_t \approx 0$: 거의 예상 못 했습니다.

이 확률에 $-\log$를 씌운 것이 그 위치의 loss입니다.

$$
\ell_t = -\log p_t
$$

| $p_t$ | $-\ln p_t$ |
|---|---|
| 1.0 | 0 |
| 0.5 | 0.693 |
| 0.1 | 2.303 |
| 0.01 | 4.605 |

확률이 낮을수록 loss가 빠르게 커집니다. 확신을 갖고 틀리면 크게 벌을 받습니다.

### cross-entropy

시퀀스 전체에 대해 평균을 낸 것이 **cross-entropy** loss입니다.

$$
\mathcal{L} = -\frac{1}{N} \sum_{t=1}^{N} \log p_t
$$

- $N$: 예측한 위치의 수.
- $\log$: 자연로그($\ln$)를 쓰면 단위가 **nat**, 밑이 2인 로그를 쓰면 **bit**입니다. PyTorch의 `CrossEntropyLoss`는 자연로그입니다.

[문장의 확률 계산](01-language-model.md)의 식과 나란히 놓으면 관계가 보입니다. $\sum \log p_t$ 는 모델이 시퀀스 전체에 준 로그 확률 $\log P(x_1, \dots, x_N)$ 입니다. 그러므로 cross-entropy를 줄이는 것은 학습 데이터 전체의 확률을 높이는 것과 같습니다. 이미지 분류의 loss도 같은 식이고, 클래스가 어휘로 바뀐 것뿐입니다.

#### 이름의 뜻

정답 분포를 $q$(정답 자리만 1인 one-hot), 모델 분포를 $p$라 하면 cross-entropy의 일반식은 $H(q, p) = -\sum_i q_i \log p_i$입니다. $q$가 one-hot이면 합에서 정답 항 하나만 남아 $-\log p_{\text{정답}}$ 이 됩니다. 위의 $\ell_t$ 와 같습니다.

### perplexity

cross-entropy에 지수 함수를 씌운 것이 **perplexity**입니다.

$$
PP = \exp\!\left(-\frac{1}{N} \sum_{t=1}^{N} \ln p_t\right) = e^{\mathcal{L}}
$$

로그의 성질로 풀면 다른 모습이 됩니다.

$$
PP = \left(\prod_{t=1}^{N} \frac{1}{p_t}\right)^{1/N}
$$

정답에 준 확률의 **역수의 기하평균**입니다. cross-entropy와 perplexity는 같은 정보이고 표기만 다릅니다.

#### 손계산

네 위치에서 정답에 준 확률이 $0.5,\; 0.25,\; 0.1,\; 0.8$ 이었습니다.

1. $\ln$ 값: $-0.693,\; -1.386,\; -2.303,\; -0.223$
2. 합: $-4.605$. 평균: $-1.151$
3. cross-entropy: $\mathcal{L} = 1.151$ nat
4. perplexity: $e^{1.151} = 3.162$

기하평균 쪽으로 검산합니다. 확률의 곱은 $0.5 \times 0.25 \times 0.1 \times 0.8 = 0.01$, 역수는 100, 네제곱근은 $100^{1/4} = 3.162$. 같습니다.

### perplexity를 읽는 법

perplexity는 모델이 다음 토큰을 고를 때 평균 몇 개의 선택지 사이에서 고르는 것과 같은지를 나타냅니다. perplexity가 100이면 매번 약 100개의 후보 중에서 균등하게 고르는 정도로 헷갈린다는 뜻입니다. 어휘가 50,000개인 모델이 무작위로 고르면 perplexity는 50,000이므로, 100은 그보다 훨씬 나은 값이지만 다음 토큰을 확신하는 수준은 아닙니다.

#### 왜 "선택지 수"로 읽을 수 있나

어휘 $k$개에 균등하게 $1/k$씩 주는 모델은 모든 위치에서 $p_t = 1/k$ 이므로 $PP = k$입니다. 그러므로 perplexity $k$인 모델은 "평균적으로, $k$개 중에 균등하게 찍는 것만큼 헷갈린다"고 읽을 수 있습니다.

| 상황 | perplexity |
|---|---|
| 항상 정답에 확률 1 | 1 (하한) |
| 글자 어휘 27개에서 균등하게 찍기 | 27 |
| 단어 어휘 50,000개에서 균등하게 찍기 | 50,000 |
| 정답에 확률 0을 한 번이라도 줌 | 무한대 |

마지막 줄이 [n-gram baseline](03-ngram-baseline.md)에서 smoothing 없는 n-gram의 perplexity가 `inf`로 나온 이유입니다.

## 실험 결과

### temperature를 바꿔 생성한 결과

학습한 LSTM(`experiments/char_lstm.py`, 40 epoch 학습. epoch은 학습 데이터 전체를 한 번 훑는 단위입니다)으로 6,000글자씩 생성해서 `\begin{X}` … `\end{X}` 짝이 맞는 줄을 세었습니다. 합성 말뭉치 하나, seed 하나의 결과이고 값은 [results/lstm.txt](results/lstm.txt)에 있습니다.

| $T$ | 맞게 닫음 | 틀리게 닫음 | 샘플 앞부분 |
|---|---|---|---|
| 0.2 | 133 | 0 | `\begin{proof} be and a the and \end{proof}` `\begin{proof} so so so so \end{proof}` |
| 0.5 | 126 | 0 | `\begin{proof} be let a and map \end{proof}` |
| 1.0 | 116 | 0 | `\begin{proof} so set holds so map y x and the \end{proof}` |
| 1.5 | 98 | 10 | `\begin{lemma} a sy a then map` (없는 단어 `sy`가 나옵니다) |

- $T = 0.2$: 구조는 완벽하지만 `so so so so`처럼 같은 단어를 반복합니다. 낮은 temperature에서 출력이 반복적이 된다는 것이 이 모습입니다.
- $T = 1.5$: 없는 단어가 생기고 짝도 틀리기 시작합니다. 모델은 `\end{` 다음에 올 환경 이름을 거의 확신하고 있었는데(그래서 $T \le 1$에서는 0번 틀립니다), 분포를 평평하게 만들자 낮은 확률의 오답이 뽑힌 것입니다.

temperature는 모델의 가중치를 바꾸지 않습니다. 모델이 낸 분포를 생성 단계에서 얼마나 그대로 따를지만 바꿉니다.

### 모델별 시험 perplexity

합성 말뭉치(글자 단위, 어휘 25개, 148,458글자 가운데 10%를 시험에 사용)의 시험 perplexity입니다. 값은 [results/ngram.txt](results/ngram.txt), [results/rnn-numpy.txt](results/rnn-numpy.txt), [results/rnn.txt](results/rnn.txt), [results/lstm.txt](results/lstm.txt)에 있습니다.

| 모델 | 시험 perplexity | BPC |
|---|---|---|
| 균등하게 찍기 | 25 | 4.64 |
| 글자 unigram ($\alpha = 0.01$) | 2.909 | 1.54 |
| 문맥 4 n-gram ($\alpha = 0.01$) | 1.604 | 0.68 |
| vanilla RNN (numpy, hidden 96) | 1.561 | 0.64 |
| vanilla RNN (PyTorch, hidden 128) | 1.542 | 0.62 |
| LSTM (PyTorch, hidden 128) | 1.511 | 0.60 |

## 결과 해석

모델의 출력은 softmax를 거쳐 확률이 되고, 그 확률은 정답 글자에 준 값의 로그 평균(cross-entropy)과 그 지수(perplexity)로 평가됩니다.

위 표에서 vanilla RNN과 LSTM의 perplexity 차이는 작습니다(1.542와 1.511). 같은 두 모델의 환경 짝 정답률은 46%와 100%입니다([먼 거리의 짝 맞추기](08-long-term-dependency.md)). 환경 이름을 고르는 결정은 한 줄에 한 번뿐이라 위치 전체의 평균인 perplexity에는 작게 반영됩니다. 이 말뭉치 하나에서 본 결과입니다.

## 한계와 주의할 점

### 비교할 수 있는 조건

perplexity는 같은 토큰 단위, 같은 어휘, 같은 시험 데이터일 때만 비교할 수 있습니다.

- 글자 단위 perplexity 1.5와 단어 단위 perplexity 100은 직접 비교할 수 없습니다. 단어 하나가 평균 6글자(공백 포함)라면 글자 perplexity 1.5는 단어 기준으로 대략 $1.5^6 \approx 11.4$ 에 해당합니다. 위치당 값이 작아 보여도 위치 수가 많습니다.
- 글자 단위 모델은 관례적으로 bits per character(BPC) 로 보고하는 경우가 많습니다. $\text{BPC} = \mathcal{L} / \ln 2 = \log_2 PP$. perplexity 1.56이면 BPC는 0.64입니다.

### perplexity와 다른 지표의 관계

Mikolov 등의 2010년 논문은 RNN 언어 모델이 backoff n-gram보다 perplexity는 약 50%, 음성 인식의 WER(word error rate)은 같은 양의 데이터에서 약 18% 낮다고 보고했습니다. 조건과 수치는 [n-gram baseline](03-ngram-baseline.md)에 정리했습니다.

perplexity 50% 감소를 선택지 수로 읽으면 매 단어에서 고르는 후보가 절반으로 줄었다는 뜻입니다. cross-entropy로는 $\ln 2 = 0.693$ nat, 단어당 1 bit입니다.

perplexity와 WER은 서로 다른 것을 잽니다. perplexity는 언어 모델 자체의 품질이고, WER은 그 언어 모델을 음성 인식 시스템에 넣었을 때의 최종 성능입니다. perplexity가 좋아진 비율만큼 WER이 좋아지지는 않습니다(50%와 18%).

## 확인 문제

1. logit $[1.0, 1.0, 1.0]$의 softmax는 얼마인가요? temperature를 바꾸면 달라지나요? (답: 모두 1/3. 차이가 0이라 달라지지 않습니다)
2. logit $[3.0, 0.0]$에 $T = 1$과 $T = 3$을 적용하면 1등의 확률은 각각 얼마인가요? (답: $e^3/(e^3+1) = 0.953$, $e^1/(e^1+1) = 0.731$)
3. 학습할 때 temperature는 얼마인가요? (답: 1. temperature는 생성 단계에서만 바꿉니다)
4. 모델이 모든 위치에서 정답에 0.25를 줬습니다. perplexity는 무엇인가요? (답: 4)
5. cross-entropy가 2.0 nat에서 1.5 nat으로 줄었습니다. perplexity는 몇 배가 됐나요? (답: $e^{-0.5} = 0.607$배)
6. 어휘 10,000개짜리 단어 모델의 perplexity가 100입니다. 균등 분포 대비 몇 bit를 아는 것인가요? (답: $\log_2 10000 - \log_2 100 = 13.29 - 6.64 = 6.64$ bit)

## References

- [The Unreasonable Effectiveness of Recurrent Neural Networks](https://karpathy.github.io/2015/05/21/rnn-effectiveness/) (Karpathy, 2015)
- [Recurrent neural network based language model](https://www.isca-archive.org/interspeech_2010/mikolov10_interspeech.html) (Mikolov 등, 2010)
