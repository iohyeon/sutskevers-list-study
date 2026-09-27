# 글자 n-gram은 무엇을 할 수 있고, 직전 n글자 밖의 정보는 왜 쓸 수 없는가?

## 문제

n-gram은 직전 몇 글자 뒤에 어떤 글자가 몇 번 나왔는지를 세어 다음 글자의 확률로 쓰는 모델입니다. RNN이 무엇을 더 하는지 말하려면 같은 데이터에서 n-gram이 어디까지 하는지를 먼저 재야 합니다. 15글자 말뭉치로 표를 손으로 만들고, 합성 말뭉치에서 perplexity와 환경 짝 정답률을 쟀습니다.

## 아이디어

### n-gram이 하는 일

n-gram은 직전 $n-1$개 토큰만 보고 다음 토큰을 예측하는 모델입니다. 학습은 횟수를 세는 것이 전부입니다. 글자 4-gram이라면 "hel" 다음 글자를 예측할 때 학습 데이터에서 "hel" 뒤에 어떤 글자가 몇 번 나왔는지를 봅니다. 보는 범위가 $n-1$개로 고정되어 있어서 그보다 먼 곳에 있는 정보는 쓰지 못합니다.

### 정의

n-gram 모델은 [문장의 확률 계산](01-language-model.md)의 조건부 확률을 횟수의 비율로 추정합니다.

$$
P(x_t \mid x_{t-n+1}, \dots, x_{t-1}) = \frac{\text{count}(x_{t-n+1}, \dots, x_{t-1}, x_t)}{\text{count}(x_{t-n+1}, \dots, x_{t-1})}
$$

- $n = 1$ (unigram): 문맥을 보지 않습니다. 글자별 빈도만.
- $n = 2$ (bigram): 직전 1개를 봅니다.
- $n = 3$ (trigram): 직전 2개를 봅니다.

용어가 헷갈리기 쉬운데, "n-gram"의 $n$은 문맥과 예측 대상을 합친 길이입니다. 문맥의 길이는 $n-1$입니다. 실험 코드(`experiments/ngram.py`)는 편의상 문맥 길이를 `n`이라는 변수로 씁니다.

### 배경: char-rnn과 n-gram의 비교

Karpathy의 글(2015)이 퍼진 직후 Yoav Goldberg가 후속 글 [The unreasonable effectiveness of Character-level Language Models](https://nbviewer.org/gist/yoavg/d76121dfde2618422139)를 썼습니다. smoothing도 없는 글자 단위 n-gram으로 셰익스피어 문체의 그럴듯한 텍스트를 만들어 보이고, 그럴듯한 영어가 나온다는 것은 n-gram으로도 되는 일이라고 지적했습니다. 그가 RNN에서 인상적이라고 꼽은 것은 Linux 소스 코드 예제입니다. 들여쓰기가 맞고 괄호가 바르게 짝지어져 있는데, 이것은 직전 $n$글자만 봐서는 할 수 없는 일이라고 썼습니다. 가장 단순한 baseline을 같은 데이터로 먼저 돌려 보면 새 모델이 실제로 더 하는 일이 무엇인지 드러난다는 이야기입니다.

RNN 언어 모델과 n-gram을 수치로 비교한 연구로는 Mikolov 등의 2010년 논문이 있습니다. RNN 언어 모델이 backoff n-gram보다 perplexity와 음성 인식 오류율(WER)에서 앞섰다는 결과입니다.

이 문서의 실험은 같은 비교를 합성 말뭉치에서 다시 해 본 것입니다. 그럴듯한 텍스트와 먼 거리의 짝 맞추기를 따로 잽니다.

## 작은 숫자로 직접 계산

### 손으로 세어 보기

말뭉치를 아주 작게 잡습니다. 공백은 `_`로 적습니다.

```
hello_help_hell
```

글자를 순서대로 늘어놓으면 `h e l l o _ h e l p _ h e l l` 로 15개입니다. 어휘는 $V = \{h, e, l, o, p, \_\}$, $|V| = 6$ 입니다.

#### bigram (직전 1글자 → 다음 글자)

인접한 두 글자 쌍을 전부 셉니다. 쌍은 14개입니다.

| 문맥 \ 다음 | h | e | l | o | p | _ | 합계 |
|---|---|---|---|---|---|---|---|
| h | 0 | 3 | 0 | 0 | 0 | 0 | 3 |
| e | 0 | 0 | 3 | 0 | 0 | 0 | 3 |
| l | 0 | 0 | 2 | 1 | 1 | 0 | 4 |
| o | 0 | 0 | 0 | 0 | 0 | 1 | 1 |
| p | 0 | 0 | 0 | 0 | 0 | 1 | 1 |
| _ | 2 | 0 | 0 | 0 | 0 | 0 | 2 |

`l` 행을 읽으면 `l` 다음에는 `l`이 2번, `o`가 1번, `p`가 1번 나왔습니다. 확률로 바꾸면 다음과 같습니다.

$$
P(\text{l} \mid \text{l}) = \tfrac{2}{4} = 0.5, \quad P(\text{o} \mid \text{l}) = 0.25, \quad P(\text{p} \mid \text{l}) = 0.25
$$

#### trigram (직전 2글자 → 다음 글자)

문맥 `el` 뒤에는 `l`, `p`, `l`이 나왔습니다. 문맥 `ll` 뒤에는 `o`가 한 번 나왔습니다(마지막 `ll`은 뒤가 없습니다).

$$
P(\text{l} \mid \text{el}) = \tfrac{2}{3}, \quad P(\text{p} \mid \text{el}) = \tfrac{1}{3}, \quad P(\text{o} \mid \text{ll}) = 1
$$

bigram은 `l` 다음에 `o`가 올 확률을 0.25로 보지만, trigram은 "`ll` 다음이면 `o`"를 확률 1로 압니다. 문맥을 한 글자 늘리자 "hello"의 두 번째 `l`과 첫 번째 `l`을 구분하게 됐습니다. 같은 글자 `l`인데 다음 글자가 다른 경우를 RNN은 hidden state로 구분하고, n-gram은 창을 넓혀서 해결합니다. 여기서 hidden state는 RNN이 지금까지 읽은 내용을 요약해 다음 시점으로 넘기는 벡터를 말합니다.

### 문제 1: 본 적 없는 문맥은 확률이 0이다

위 표에서 $P(\text{e} \mid \text{l}) = 0$ 입니다. "help me"라는 입력에는 `_` 다음 `m`이 필요한데 어휘에도 없습니다. 확률이 0인 글자가 하나라도 나오면 문장 전체의 확률이 0이 되고, 로그를 취하면 $-\infty$가 됩니다.

**smoothing**은 모든 횟수에 작은 값 $\alpha$를 더해서 0을 없애는 방법입니다. 가장 단순한 add-$\alpha$ smoothing은 다음과 같습니다.

$$
P_\alpha(x_t \mid c) = \frac{\text{count}(c, x_t) + \alpha}{\text{count}(c) + \alpha \lvert V \rvert}
$$

$c$는 문맥입니다. $\alpha = 1$로 두면 $P_1(\text{e} \mid \text{l}) = (0 + 1) / (4 + 6) = 0.1$, $P_1(\text{o} \mid \text{l}) = (1 + 1) / (4 + 6) = 0.2$ 가 됩니다. 본 적 없는 것에 확률을 나눠 준 만큼 본 것의 확률이 줄어듭니다(0.25 → 0.2).

smoothing 유무에 따라 이름이 나뉩니다.

- unsmoothed n-gram (Goldberg가 Karpathy의 글에 대한 후속 글에서 쓴 것): $\alpha = 0$. 위의 0 문제를 그대로 안고 있습니다.
- smoothed n-gram: 실제 음성 인식과 번역 시스템에서 쓰던 방식. add-$\alpha$보다 정교한 방법을 씁니다. Mikolov가 비교 대상으로 삼은 것은 Kneser-Ney smoothing을 쓴 backoff n-gram입니다.

perplexity는 모델이 다음 글자를 고를 때 평균 몇 개의 후보 사이에서 고르는 것과 같은지를 나타내는 값입니다. 낮을수록 좋고 계산 방법은 [softmax와 perplexity](02-softmax-and-perplexity.md)에 있습니다. 시험 데이터에 학습 때 못 본 (문맥, 글자) 쌍이 하나만 있어도 smoothing 없는 모델의 perplexity는 무한대가 됩니다.

### 문제 2: 표가 지수적으로 커진다

문맥 길이를 늘리면 좋아질 것 같지만, 문맥이 길어질수록 같은 문맥이 학습 데이터에 다시 나올 가능성이 줄어듭니다. 아래 「실험 결과」에서 문맥 8부터 perplexity가 다시 나빠지는 이유입니다.

가능한 문맥의 수는 $|V|^{n-1}$ 입니다.

| 단위 | $\lvert V \rvert$ | 문맥 길이 | 가능한 문맥 수 |
|---|---|---|---|
| 글자 | 100 | 2 | $10^4$ |
| 글자 | 100 | 5 | $10^{10}$ |
| 글자 | 100 | 10 | $10^{20}$ |
| 단어 | 50,000 | 2 | $2.5 \times 10^9$ |
| 단어 | 50,000 | 4 | $6.25 \times 10^{18}$ |

대부분의 문맥은 학습 데이터에 한 번도 나오지 않습니다. 이것을 **데이터 희소성** 문제라고 부릅니다. 데이터를 늘리면 완화되지만 문맥 길이를 하나 늘릴 때마다 필요한 데이터가 $|V|$배가 됩니다.

### 문제 3: 비슷한 문맥을 비슷하다고 보지 못한다

n-gram의 표에서 문맥은 문자열 키입니다. "the cat sat"과 "the dog sat"은 단어 하나만 다른데도 완전히 다른 키입니다. 한쪽에서 배운 것을 다른 쪽에 쓸 수 없습니다.

RNN은 빈도를 세는 대신 문맥을 실수 벡터로 나타냅니다. 이런 표현을 분산 표현이라고 합니다. 분산 표현은 하나의 개념을 벡터의 여러 차원에 나눠 담는 것입니다. 비슷한 문맥은 비슷한 벡터가 되고, 비슷한 벡터에 같은 가중치가 곱해지므로 비슷한 예측이 나옵니다. 표의 키는 같거나 다르거나 둘 중 하나지만, 벡터 사이에는 "가깝다"가 있습니다. one-hot과 임베딩은 [문장의 확률 계산](01-language-model.md)에 있습니다.

### 문제 4: 창 밖의 정보는 원리상 볼 수 없다

`\begin{proof}`로 연 환경을 `\end{proof}`로 닫으려면 닫는 시점에 "무엇으로 열었는지"를 알아야 합니다. 본문이 창보다 길면 n-gram은 알 방법이 없습니다. 합성 말뭉치에서 잰 결과는 아래 「실험 결과」에 있습니다.

## 코드로 확인

### 학습은 횟수 세기다

```python
def train(text, n):
    table = defaultdict(Counter)          # 키: 직전 n글자, 값: 다음 글자별 횟수
    pad = "~" * n
    data = pad + text
    for i in range(len(text)):
        table[data[i:i + n]][data[i + n]] += 1
    return table
```

이 코드의 `n` 은 **문맥 길이**입니다. 통상 용어로는 $(n+1)$-gram입니다. 말뭉치 앞에 `~` 를 채워서 첫 글자에도 문맥이 있게 했습니다.

기울기도 반복 학습도 없습니다. 말뭉치를 한 번 훑으면 끝입니다.

### 확률과 smoothing

```python
def prob(table, ctx, ch, vocab_size, alpha=0.0):
    c = table.get(ctx)
    total = sum(c.values()) if c else 0
    count = c[ch] if c else 0
    if total + alpha * vocab_size == 0:
        return 0.0
    return (count + alpha) / (total + alpha * vocab_size)
```

위의 add-$\alpha$ 식 그대로입니다. `alpha=0.0` 이 Goldberg가 쓴 "unsmoothed"입니다.

### 생성

```python
def generate(table, n, length, seed=0):
    out = "~" * n
    for _ in range(length):
        c = table.get(out[-n:])                 # 직전 n글자로 표를 조회한다
        if not c:
            break                               # 본 적 없는 문맥이면 멈춘다
        chars, weights = zip(*c.items())
        out += random.choices(chars, weights)[0]
    return out[n:]
```

RNN의 `sample` 과 구조가 같습니다. 분포를 얻고, 하나 뽑고, 뒤에 붙입니다. 분포를 얻는 방법만 다릅니다(표 조회 대 행렬 계산).

### 평가 지표 두 개

**perplexity**: 시험 데이터의 모든 위치에서 정답 글자에 준 확률로 계산합니다([softmax와 perplexity](02-softmax-and-perplexity.md)).

**환경 짝 정답률**: 생성한 텍스트에서 `\begin{X}` 로 시작하고 `\end{` 가 있는 줄을 골라, 여는 이름과 닫는 이름이 같은지 셉니다. 여는 쪽이 깨진 줄(`\begin{prof` 등)은 오답으로 셉니다.

```python
def closer_accuracy(text):
    ok = bad = 0
    for line in text.split("\n"):
        if line.startswith("\\begin{") and "\\end{" in line:
            head = line[7:line.index("\\end{")]
            opened = head[:head.index("}")] if "}" in head else None
            tail = line[line.index("\\end{") + 5:]
            closed = tail[:tail.index("}")] if "}" in tail else ""
            if opened is not None and opened == closed: ok += 1
            else: bad += 1
    return ok, bad
```

두 번째 지표를 따로 두는 이유가 있습니다. 새 구조만 풀 수 있을 것으로 예상되는 문제를 평균 지표와 따로 잽니다.

## 실험 결과

`experiments/ngram.py`를 합성 말뭉치(148,458글자, 어휘 25글자, 학습 90%와 시험 10%)에 실행한 출력입니다([results/ngram.txt](results/ngram.txt)). 첫 열 `n`은 문맥 길이입니다.

```
n= 1 contexts=    26 PP(no smoothing)=  inf PP(add-0.01)=   2.909 closer ok/bad=0/0
n= 2 contexts=    64 PP(no smoothing)=  inf PP(add-0.01)=   1.743 closer ok/bad=3/23
n= 4 contexts=   396 PP(no smoothing)=  inf PP(add-0.01)=   1.604 closer ok/bad=35/34
n= 8 contexts=  8240 PP(no smoothing)=  inf PP(add-0.01)=   1.782 closer ok/bad=64/57
n=12 contexts= 36180 PP(no smoothing)=  inf PP(add-0.01)=   3.177 closer ok/bad=44/57
```

출력에서 네 가지를 읽을 수 있습니다.

1. smoothing이 없으면 이 말뭉치의 시험 perplexity는 모든 문맥 길이에서 무한대입니다. Goldberg의 unsmoothed 모델은 텍스트 생성에는 쓸 수 있지만 시험 perplexity로는 비교할 수 없습니다.
2. 문맥 길이에는 최적점이 있습니다. 4에서 가장 좋고 그 뒤로 나빠집니다. 문맥이 길수록 표가 비어 있을 확률이 높습니다(`contexts` 열이 26 → 36,180으로 늘어납니다).
3. 짝 정답률은 문맥 길이와 무관하게 50% 근처입니다. 환경 이름이 평균 29글자 앞에 있어서 창 밖입니다.
4. 문맥 1, 2에서는 `\begin{…} … \end{…}` 형태의 줄 자체가 거의 만들어지지 않습니다(`0/0`, `3/23`).

문맥 8의 생성 결과입니다.

```
\begin{lemma} x we by the then holds the have \end{proof}
\begin{lemma} x a we have have set we and y map map then we holds \end{proof}
```

두 번째 줄은 본문이 단어 14개입니다. 말뭉치에서는 최대 9개인데 n-gram은 "지금까지 단어를 몇 개 썼는지"도 모릅니다. 이것도 창 밖의 정보입니다. 철자, 공백, 명령어, 중괄호는 전부 맞아서 언뜻 보면 RNN의 출력과 구분되지 않습니다.

### vanilla RNN, LSTM과의 비교

같은 말뭉치로 학습한 vanilla RNN과 LSTM의 값입니다. 학습 설정은 [먼 거리의 짝 맞추기](08-long-term-dependency.md)에 있고 값은 [results/rnn.txt](results/rnn.txt), [results/rnn-numpy.txt](results/rnn-numpy.txt), [results/lstm.txt](results/lstm.txt)에 있습니다.

| 모델 | 시험 perplexity |
|---|---|
| n-gram 문맥 4 ($\alpha = 0.01$) | 1.604 |
| vanilla RNN | 1.542 (PyTorch, hidden 128), 1.561 (numpy, hidden 96) |
| LSTM | 1.511 |

| 모델 | 환경 짝 (맞음 / 전체) | 정답률 |
|---|---|---|
| n-gram 문맥 4 | 35 / 69 | 51% |
| n-gram 문맥 8 | 64 / 121 | 53% |
| n-gram 문맥 12 | 44 / 101 | 44% |
| vanilla RNN (PyTorch, $T = 0.5$) | 57 / 125 | 46% |
| LSTM (PyTorch, $T = 0.5$) | 126 / 126 | 100% |

n-gram은 temperature 없이 분포 그대로 샘플링한 결과입니다.

perplexity는 세 모델이 비슷합니다. 환경 짝 정답률은 n-gram과 vanilla RNN이 50% 근처이고 LSTM만 100%입니다. 환경이 두 가지뿐이므로 50%는 무작위로 고른 것과 같은 수준입니다.

## 결과 해석

### 문맥을 더 늘리면 풀리는가

환경 이름까지의 거리가 최대 47글자이므로 문맥 50글자 n-gram이면 원리상 풀 수 있습니다. 그런데 문맥 12에서 이미 저장된 문맥이 36,180개이고 perplexity가 3.18로 나빠집니다. 본문이 무작위 단어라서 50글자 문맥은 학습 데이터에 같은 것이 다시 나오지 않습니다. 표를 조회하면 대부분 빈 칸입니다.

LSTM은 같은 문제를 128차원 상태 하나로 풉니다. 본문의 단어들을 무시하고 환경 이름 1비트만 유지하는 법을 배웠기 때문입니다([cell state 해석](09-interpreting-cell-state.md)). n-gram은 문맥의 일부만 골라 기억할 수 없습니다. 키는 통째로 일치해야 합니다.

| | n-gram | LSTM |
|---|---|---|
| 먼 정보를 쓰려면 | 창을 그만큼 넓혀야 합니다 | 그 정보만 상태에 남기면 됩니다 |
| 사이에 낀 무관한 입력 | 전부 키에 포함됩니다. 키가 폭발합니다 | 무시하는 법을 배웁니다 |
| 데이터를 늘리면 | 필요한 양이 거리에 대해 지수적입니다 | 이 실험의 거리(최대 47글자)에서는 같은 데이터로 배웠습니다. 거리가 더 길어지면 LSTM도 어려워집니다 |

### n-gram이 잘하는 것

- 학습이 횟수 세기라서 빠릅니다. 위 실험은 1초가 안 걸립니다.
- 결과를 해석하기 쉽습니다. 표를 열어 보면 됩니다.
- 데이터가 충분하고 필요한 문맥이 짧으면 성능이 좋습니다. 문맥 4의 perplexity 1.604는 같은 데이터에서 vanilla RNN이 낸 1.542, 1.561과 큰 차이가 없습니다.

그럴듯한 텍스트가 나온다는 것만으로는 RNN이 n-gram보다 낫다고 말할 수 없습니다. Goldberg의 글이 지적한 것이 이 점입니다.

### baseline과 비교할 때 필요한 것

새 구조를 들이기 전에 가장 단순한 모델을 같은 데이터로 돌려 봅니다. n-gram baseline(1.604)이 있어서 vanilla RNN의 perplexity 1.542가 큰 개선이 아니라는 것을 알 수 있었고, 환경 짝 정답률이라는 두 번째 지표가 있어서 LSTM이 더 하는 일이 무엇인지 볼 수 있었습니다.

baseline 비교에는 두 가지가 필요합니다.

1. 같은 데이터와 같은 평가 방법을 씁니다.
2. 평균 지표 하나로 끝내지 않고, 새 구조만 풀 수 있을 것으로 예상되는 문제를 따로 잽니다.

글자 n-gram은 직전 n글자 안에서 정해지는 철자와 명령어는 RNN과 비슷하게 맞힙니다. 직전 n글자 밖에 있는 환경 이름은 표의 키에 들어 있지 않아서 쓸 수 없고, 키를 그만큼 늘리면 같은 문맥이 학습 데이터에 다시 나오지 않습니다.

## 한계와 주의할 점

- vanilla RNN은 n-gram과 같은 곳에서 실패했습니다. 이 과제처럼 글자 수십 개를 건너는 정보가 필요한 경우에는 LSTM이어야 n-gram과 차이가 났습니다. Karpathy의 char-rnn도 LSTM이었습니다. 필요한 문맥이 짧은 단어 단위 언어 모델에서는 아래 Mikolov의 결과처럼 단순한 RNN으로도 n-gram을 앞섰습니다.
- 합성 데이터입니다. 실제 LaTeX나 소스 코드에서는 본문에 환경을 짐작할 단서가 있고(증명에는 "therefore"가 많습니다), 중첩이 있고, 거리가 훨씬 깁니다.
- 평균 perplexity로는 이 차이가 거의 보이지 않습니다(1.54 대 1.51). 평균 perplexity를 재면 두 모델이 비슷하고, 짝 맞추기 정답률을 재면 46%와 100%입니다.

### Mikolov 등(2010)이 보고한 수치

Mikolov 등의 2010년 Interspeech 논문(*Recurrent neural network based language model*) 초록이 보고한 것은 세 가지입니다. 여러 RNN 언어 모델을 섞었을 때 당시 최고 수준의 backoff 언어 모델보다 perplexity가 약 50% 줄었습니다. Wall Street Journal 음성 인식 과제에서 같은 양의 데이터로 학습한 모델끼리 비교해 WER이 약 18% 줄었습니다. 더 어려운 NIST RT05 과제에서는 backoff 모델을 훨씬 많은 데이터로 학습시켰는데도 WER이 약 5% 줄었습니다.

- perplexity 50% 감소는 단어당 1 bit의 정보를 더 아는 것입니다([softmax와 perplexity](02-softmax-and-perplexity.md)).
- WER 18%는 상대 감소입니다. WER이 예를 들어 17%에서 14%로 내려가면 절대 감소는 3%p, 상대 감소는 $3/17 = 18\%$입니다. 17과 14는 계산을 보이려고 넣은 숫자입니다.
- perplexity 개선(50%)에 비해 WER 개선(18%)이 작습니다. 음성 인식에서 언어 모델은 음향 모델이 낸 후보들의 순위를 다시 매기는 역할이라, 음향 쪽에서 이미 틀린 것은 고칠 수 없습니다.
- n-gram 쪽에 데이터를 훨씬 더 줬을 때도 차이가 남았지만, 그 크기는 약 5%로 같은 데이터양일 때의 18%보다 작습니다. 데이터를 늘리면 n-gram도 따라오지만 차이가 없어지지는 않았다는 뜻입니다.

이 수치는 2010~2011년의 것이고 LSTM이 아니라 단순한 RNN으로 낸 것입니다. 단어 단위에서는 필요한 문맥 거리가 글자 단위보다 5~6배 짧아서([문장의 확률 계산](01-language-model.md)) vanilla RNN으로도 n-gram을 넘을 수 있었습니다. 임베딩을 통한 일반화(「문제 3」)만으로도 이득이 컸다는 뜻입니다.

## 더 해 볼 것

1. 실제 텍스트(예: 공개된 Shakespeare 전집, 자기 프로젝트의 소스 코드)에 돌려 봅니다. 문맥 7~10에서 그럴듯한 영어가 나오는지, 소스 코드에서 괄호 짝이 맞는지 확인합니다.
2. smoothing을 add-$\alpha$ 에서 **backoff**(긴 문맥이 표에 없으면 한 글자 짧은 문맥으로 물러납니다)로 바꿉니다. 문맥 12의 perplexity가 얼마나 회복되는지 봅니다.
3. 말뭉치를 10배(`range(30000)`)로 늘려 다시 돌립니다. perplexity는 좋아지겠지만 짝 정답률은 그대로일 것이라는 예측을 확인합니다. "n-gram에 데이터를 더 줘도"에 해당하는 실험입니다.

## 확인 문제

1. 위 말뭉치에서 $P(\text{e} \mid \text{h})$는 얼마인가요? (답: 3/3 = 1)
2. $\alpha = 1$ smoothing을 쓰면 $P_1(\text{e} \mid \text{h})$는 얼마인가요? (답: (3+1)/(3+6) = 0.444)
3. 글자 어휘 100개, 문맥 길이 8이면 가능한 문맥은 몇 개인가요? 100만 글자짜리 말뭉치가 그중 몇 %를 채울 수 있나요? (답: $10^{16}$개. 최대 $10^6$개를 채우므로 $10^{-8}$%)
4. n-gram이 환경 짝 맞추기를 "원리상" 못 푸는 조건은 무엇인가요? (답: 환경 이름까지의 거리가 문맥 길이보다 길 때)
5. 문맥을 충분히 길게 잡으면 원리상은 풀 수 있는데 실제로 안 되는 이유는 무엇인가요? (답: 가능한 문맥 수가 지수적으로 늘어 같은 문맥이 학습 데이터에 다시 나오지 않습니다)
6. WER 20%가 16.4%로 줄면 상대 감소율은 얼마인가요? (답: $3.6 / 20 = 18\%$)

## References

- [The unreasonable effectiveness of Character-level Language Models](https://nbviewer.org/gist/yoavg/d76121dfde2618422139) (Goldberg, 2015)
- [Recurrent neural network based language model](https://www.isca-archive.org/interspeech_2010/mikolov10_interspeech.html) (Mikolov 등, 2010)
- [The Unreasonable Effectiveness of Recurrent Neural Networks](https://karpathy.github.io/2015/05/21/rnn-effectiveness/) (Karpathy, 2015)
