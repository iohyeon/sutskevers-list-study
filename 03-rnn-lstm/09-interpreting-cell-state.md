# 학습된 LSTM은 환경 이름을 cell state의 어느 차원에 저장하는가?

## 문제

Karpathy의 글은 따옴표 안에서만 켜지는 셀처럼 역할을 읽을 수 있는 셀을 보여 줍니다. 같은 방법으로, [먼 거리의 짝 맞추기 실험](08-long-term-dependency.md)에서 학습한 LSTM의 cell state에서 proof와 lemma를 가르는 차원을 찾고, 그 값을 강제로 바꿨을 때 닫는 이름이 바뀌는지 확인했습니다. vanilla RNN과 LSTM의 temperature별 생성 결과도 함께 봅니다.

## 아이디어

### Karpathy의 시각화

Karpathy는 글에서 학습된 RNN의 뉴런 활성화를 글자마다 색으로 칠해 보여 줬습니다. 따옴표 안에서만 켜지는 셀, URL 안에서 켜지는 셀, 줄의 위치에 따라 값이 변하는 셀, 코드의 중괄호 깊이를 따라가는 셀이 나왔습니다. 대부분의 셀은 해석이 안 됐고 일부만 이렇게 읽혔습니다.

### 찾는 방법

1. 학습된 모델에 텍스트를 한 글자씩 넣습니다.
2. 매 글자마다 hidden state(LSTM이면 cell state)의 값을 기록합니다. 글자 수 × hidden 크기의 표가 나옵니다.
3. 차원 하나를 골라 그 값으로 글자에 색을 입힙니다. 양수면 파랑, 음수면 빨강 같은 식입니다.
4. 사람이 읽을 수 있는 패턴이 보이는 차원을 찾습니다.

4번이 어렵습니다. 차원이 수백 개이고 대부분은 사람 눈에 의미 없어 보입니다. Karpathy도 원 글에서 해석 가능한 셀은 일부라고 썼습니다.

찾는 방법은 두 가지입니다.

- 눈으로 훑습니다. Karpathy가 한 방식. 가설 없이 시작할 수 있지만 운이 필요합니다.
- 가설을 세우고 통계로 찾습니다. "이런 정보를 추적하는 차원이 있을 것이다"를 정하고, 그 정보에 따라 값이 가장 크게 갈리는 차원을 고릅니다. 아래 실험이 이 방식입니다.

## 코드로 확인

### 생성

```python
@torch.no_grad()
def sample(model, chars, c2i, n, temperature, seed=0):
    ix = torch.tensor([[c2i["\n"]]]); state = None; out = []
    for _ in range(n):
        logits, state, _ = model(ix, state)                           # 한 글자씩 넣는다
        prob = torch.softmax(logits[0, -1] / temperature, dim=0)      # logit 을 T 로 나눈다
        ix = torch.multinomial(prob, 1).view(1, 1)                    # 분포에서 하나 뽑는다
        out.append(chars[ix.item()])
    return "".join(out)
```

학습 때는 64글자 조각을 한꺼번에 넣었지만 생성은 한 글자씩입니다. `state` 를 넘겨받아 이어 갑니다. 학습 때와 생성 때 모델을 부르는 방식이 다르다는 점이 RNN 구현에서 자주 헷갈리는 부분입니다.

### cell state 찍어 보기

`nn.LSTM` 은 시퀀스를 통째로 넣으면 마지막 시점의 cell state만 돌려줍니다. 시점마다 보려면 한 글자씩 넣어야 합니다.

```python
@torch.no_grad()
def trace(text):
    state = None; cs = []
    for ch in text:
        _, state, _ = model(torch.tensor([[c2i[ch]]]), state)
        cs.append(state[1][0, 0].clone())        # state = (h, C). C 의 layer 0, 배치 0 (이 모델은 layer 가 하나다)
    return torch.stack(cs)                        # (글자 수, hidden)
```

찾는 방법은 가설에서 출발합니다. "proof 본문과 lemma 본문에서 값이 가장 크게 갈리는 차원"을 고릅니다.

```python
gap = (C[P].mean(0) - C[L].mean(0)).abs() / (C[P].std(0) + C[L].std(0) + 1e-6)
top = gap.argsort(descending=True)[:3]
```

`P`, `L` 은 각 글자가 proof 본문인지 lemma 본문인지를 나타내는 불리언 마스크입니다.

결과는 아래 「실험 결과」에 있습니다.

### 개입 실험

분리도는 그 차원이 환경과 상관이 있다는 것만 보여 줍니다. 그 차원의 값이 닫는 이름을 정하는지는 값을 강제로 바꿔 봐야 알 수 있습니다.

```python
logits, state, _ = feed(f"\n\\begin{{{env}}} let y so ")   # 본문 중간까지 읽힌다
h, Cs = state; Cs = Cs.clone(); dims = order[:k]           # 분리도 상위 k 개 차원
Cs[0, 0, dims] = mean[other][dims]; state = (h, Cs)        # 반대 환경의 평균값으로 덮어쓴다
# 이어서 줄 끝까지 생성하고 \end{...} 의 이름을 센다
```

결과는 아래 「실험 결과」에 있습니다.

## 실험 결과

### temperature별 환경 짝

같은 40 epoch 모델로 6,000글자씩 생성해 센 값입니다([results/rnn.txt](results/rnn.txt), [results/lstm.txt](results/lstm.txt)). LSTM의 샘플과 temperature의 계산은 [softmax와 perplexity](02-softmax-and-perplexity.md)에 있습니다.

| $T$ | vanilla RNN 환경 짝 | LSTM 환경 짝 |
|---|---|---|
| 0.2 | 73 / 55 | 133 / 0 |
| 0.5 | 57 / 68 | 126 / 0 |
| 1.0 | 56 / 52 | 116 / 0 |

vanilla RNN은 temperature를 낮춰도 나아지지 않습니다. temperature는 모델이 낸 분포를 뾰족하게 하거나 평평하게 할 뿐이고, vanilla RNN은 환경 이름에 대해 50 대 50에 가까운 분포를 내기 때문입니다. LSTM 쪽에서 $T$ 가 낮을수록 줄 수가 많은 것(133, 126, 116)은 본문이 짧아지기 때문입니다. 단어를 더 쓸지 `\end` 로 갈지를 고르는 자리에서 낮은 temperature는 확률이 조금이라도 높은 쪽으로 쏠립니다.

### 환경 이름을 가르는 차원

[먼 거리의 짝 맞추기 실험](08-long-term-dependency.md)의 LSTM은 `\begin{proof}` 로 열면 `\end{proof}` 로 닫는 것을 100% 맞혔습니다. 그렇다면 "지금 proof 안인가 lemma 안인가"를 나타내는 차원이 cell state 어딘가에 있어야 합니다. 가설이 분명하므로 통계로 찾을 수 있습니다.

`experiments/probe.py` 가 하는 일은 다음과 같습니다.

1. 말뭉치 끝 6,000글자를 LSTM에 한 글자씩 넣고 매 시점의 cell state(128차원)를 기록합니다.
2. 각 글자에 "proof 본문", "lemma 본문", "그 외" 라벨을 붙입니다.
3. 차원마다 (proof 본문에서의 평균 − lemma 본문에서의 평균)의 절댓값을 두 표준편차의 합으로 나눈 값을 구합니다. 이 값이 클수록 두 경우를 잘 가릅니다.

결과입니다([results/probe.txt](results/probe.txt)).

```
가장 잘 가르는 차원: [69, 88, 61]   분리도: [3.52, 1.39, 1.32]
차원 69: proof 본문 평균 -4.80, lemma 본문 평균 +3.73
```

차원 69의 분리도(3.52)가 나머지 둘(1.39, 1.32)보다 큽니다. 이 차원의 값을 글자마다 찍었습니다. `+` 는 0.5 초과, `-` 는 $-0.5$ 미만, `.` 은 그 사이입니다.

```
\begin{lemma} a we set holds holds a let set let \end{lemma}
--.+--++++++++++++++++++++++++++++++++++++++++++++++++++++++

\begin{proof} let y so let holds the then by map \end{proof}
--.+--+.---------------------------------------------+------

\begin{proof} x the and map \end{proof}
--.+--+.------------------------+------

\begin{lemma} set be a we so \end{lemma}
--.+--+++++++++++++++++++++++++++..+++++
```

읽는 법은 다음과 같습니다.

- `\begin{` 까지는 네 줄이 똑같습니다(`--.+--+`). 아직 어떤 환경인지 모릅니다.
- 여덟 번째 글자가 `l` 이냐 `p` 냐에 따라 갈립니다. `lemma` 면 `+` 로, `proof` 면 `-` 로 갑니다.
- 본문 내내 그 부호를 유지합니다. 본문의 단어들은 이 차원을 건드리지 않습니다.
- `\end{` 근처에서 잠깐 흔들린 뒤 환경 이름을 쓰는 동안 다시 같은 부호입니다.

Karpathy의 글에 나온 scope 추적 셀과 같은 종류입니다. 설계한 것이 아니라 학습 결과로 생겼습니다. 이 동작은 [LSTM](06-lstm.md)에서 주인공을 기억하는 예로 설명한 동작 그대로입니다. 환경 이름을 읽을 때 input gate로 쓰고, 본문 동안 forget gate를 높게 두어 유지하고, 닫을 때 output gate로 꺼냅니다. 본문 구간에서 잰 forget gate 평균은 차원 69가 0.999입니다(`experiments/fgate.py`, [results/fgate.txt](results/fgate.txt)). 128개 차원 전체의 평균은 0.525입니다.

주의할 점이 있습니다.

- 차원 번호 69에는 의미가 없습니다. 초기값이 다르면 다른 번호에 생깁니다.
- 이 실험은 과제가 단순해서(기억할 것이 1비트) 한 차원이 눈에 띄게 도드라졌습니다. 실제 텍스트에서는 하나의 정보가 여러 차원에 나뉘어 담기고, 한 차원이 여러 정보에 관여합니다. 여기서도 분리도 2위와 3위(1.39, 1.32)가 0이 아닙니다. 정보가 퍼져 있다는 뜻이고, 아래 개입 실험에서 그것이 실제로 중요하다는 것이 드러납니다.
- 평균값이 $-4.80$, $+3.73$ 으로 $[-1, 1]$ 범위를 넘습니다. cell state에는 tanh가 걸려 있지 않아서 그렇습니다. $h_t = o_t \odot \tanh(C_t)$ 에서 비로소 눌립니다.
- "이 차원이 환경을 추적한다"는 상관관계입니다. 이 차원을 강제로 뒤집었을 때 모델이 반대 이름으로 닫는지까지 봐야 인과를 말할 수 있습니다. 아래 개입 실험에서 확인했습니다.

### 개입 실험

`\begin{proof} let y so ` 또는 `\begin{lemma} let y so ` 까지 읽힌 뒤, cell state에서 분리도 상위 $k$개 차원을 반대 환경의 평균값으로 덮어쓰고 이어서 생성하게 했습니다. 200번씩 반복했습니다(`experiments/intervene.py`, [results/intervene.txt](results/intervene.txt)).

| 덮어쓴 차원 수 | proof로 시작 → 닫는 이름 | lemma로 시작 → 닫는 이름 |
|---|---|---|
| 0 (개입 없음) | proof 200 | lemma 200 |
| 1 (차원 69만) | proof 199, lemma 1 | lemma 149, proof 51 |
| 3 (차원 69, 88, 61) | lemma 200 | proof 200 |
| 10 | lemma 200 | proof 200 |
| 128 (전부) | lemma 200 | proof 200 |

- 128개 중 3개 차원만 바꾸면 200번 모두 닫는 이름이 반대로 나옵니다. 본문은 그대로 이어 쓰고 닫는 이름만 반대로 씁니다. 이 모델에서는 이 세 차원의 값이 닫는 이름을 정합니다.
- 가장 눈에 띄던 차원 69 하나만으로는 부족합니다. proof 쪽은 거의 안 바뀌고(1/200) lemma 쪽은 4분의 1만 바뀝니다(51/200). 그림으로는 한 차원이 다 하는 것처럼 보였지만 정보는 몇 개 차원에 나뉘어 겹쳐서 저장되어 있습니다.

뉴런 하나를 단위로 보면 놓치는 것이 있고 여러 뉴런에 걸친 방향을 단위로 봐야 한다는 것은 뒤의 해석 가능성 연구가 다루는 문제이기도 합니다. 기억할 것이 1비트뿐인 이 작은 모델에서도 같은 일이 생겼습니다.

## 결과 해석

이 모델에서 환경 이름은 cell state의 차원 69, 88, 61에 나뉘어 저장돼 있었습니다. 차원 69는 환경 이름을 읽는 순간 부호가 갈려 줄 끝까지 유지됐고, 세 차원을 함께 덮어쓰면 닫는 이름이 200번 모두 바뀌었습니다. 차원 하나만 덮어써서는 1/200과 51/200만 바뀌었습니다.

### 해석 가능성 연구와의 관계

Karpathy의 시각화는 신경망 내부에서 사람이 읽을 수 있는 것을 찾아낸 초기 사례입니다. 같은 질문을 훨씬 큰 모델에 던지는 연구 분야가 mechanistic interpretability입니다.

- LSTM 설명 글을 쓴 Chris Olah가 이 분야를 이끄는 사람 중 하나입니다.
- Anthropic의 *Mapping the Mind of a Large Language Model* 은 LLM 내부에서 개념 단위의 feature를 찾는 연구입니다. 위의 개입 실험에서 부딪힌 문제(하나의 정보가 여러 차원에 퍼져 있습니다)를 풀기 위해, 뉴런 하나가 아니라 뉴런들의 조합 방향을 단위로 삼습니다.

2015년의 "따옴표 셀"과 지금의 feature 연구는 같은 질문을 다른 규모에서 합니다. 모델이 무엇을 계산하는지를 출력이 아니라 내부에서 확인할 수 있는가 하는 질문입니다.

## 한계와 주의할 점

- 학습한 모델 하나(40 epoch, seed 0)에서 본 결과입니다. 차원 번호 69, 88, 61은 이 모델에만 해당하고 seed를 바꾸면 다른 차원이 그 일을 맡습니다.
- 개입은 `let y so ` 뒤 한 위치에서 한 가지 방법(반대 환경의 평균값으로 덮어쓰기)으로만 했습니다.
- 기억할 것이 환경 이름 1비트뿐인 작은 모델입니다. 큰 모델의 해석에 그대로 옮길 수 있는 결과가 아닙니다.

실행 결과: [results/probe.txt](results/probe.txt), [results/intervene.txt](results/intervene.txt), [results/fgate.txt](results/fgate.txt)

## 더 해 볼 것

1. vanilla RNN에 같은 probe를 돌립니다. `probe.py` 는 LSTM의 `state[1]` 을 읽으므로 RNN용으로는 `state[0, 0]` (hidden state)을 읽게 고쳐야 합니다. 환경을 가르는 차원이 본문 초반에만 있다가 흐려지는지 봅니다.
2. input gate와 output gate를 찍습니다. forget gate는 `experiments/fgate.py`로 쟀습니다(본문 구간 평균이 차원 69와 61은 0.999, 차원 88은 0.870, 128개 차원 전체는 0.525). 같은 방법으로 환경 이름을 읽는 순간 input gate가 열리는지, 닫는 이름을 쓸 때 output gate가 열리는지 확인합니다.
3. 다른 차원 찾기. "지금 단어의 몇 번째 글자인가", "본문에 단어를 몇 개 썼는가"를 추적하는 차원이 있는지 같은 방법으로 찾습니다.
4. $T$ 를 0.05, 3.0 같은 극단으로. 0.05는 사실상 greedy입니다. 같은 줄이 무한 반복되는지 봅니다.

## 확인 문제

1. 위 실험에서 vanilla RNN에 같은 probe를 돌리면 어떤 결과가 예상되나요? (답: 짝 맞추기를 못 하므로 본문 끝까지 환경을 가르는 차원이 없거나, 있어도 본문 초반에만 갈리고 점점 흐려집니다)
2. cell state 값이 $-4.80$ 인 차원이 $h_t$ 에 기여하는 값의 범위는 얼마인가요? (답: $\tanh(-4.80) \approx -1$ 에 output gate(0~1)를 곱하므로 $-1$ 과 0 사이)
3. 위 개입 실험에서 차원 1개로는 안 되고 3개로는 된 것은 무엇을 뜻하나요? (답: 환경 정보가 한 차원이 아니라 여러 차원에 중복되어 담겨 있습니다. 하나를 바꿔도 나머지가 원래 정보를 들고 있습니다)

## References

- [The Unreasonable Effectiveness of Recurrent Neural Networks](https://karpathy.github.io/2015/05/21/rnn-effectiveness/) (Karpathy, 2015)
- [Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/) (Olah, 2015)
- [Mapping the Mind of a Large Language Model](https://www.anthropic.com/research/mapping-mind-language-model) (Anthropic, 2024)
