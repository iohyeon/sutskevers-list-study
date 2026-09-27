# BPTT에서 기울기는 시간 방향으로 어떻게 전파되고, 왜 거리에 따라 줄어드는가?

## 문제

RNN은 같은 가중치를 모든 시점에서 쓰기 때문에, 한 시점의 hidden state가 그 시점의 loss와 이후 모든 시점의 loss에 영향을 줍니다. BPTT(backpropagation through time)는 RNN을 시간 방향으로 펼친 뒤 역전파를 적용하는 것입니다. 점화식을 유도하고, 구현한 기울기를 수치 미분과 비교하고, 시점을 거슬러 갈 때 곱해지는 야코비안의 크기를 재서 기울기가 줄어드는 정도를 확인합니다.

## 수식으로 보기

### 설정

순전파 식을 다시 적습니다.

$$
a_t = W_{xh} x_t + W_{hh} h_{t-1} + b_h, \quad h_t = \tanh(a_t), \quad y_t = W_{hy} h_t + b_y, \quad p_t = \text{softmax}(y_t)
$$

loss는 시점별 loss의 합입니다. $k_t$는 시점 $t$의 정답 글자 번호입니다.

$$
L = \sum_{t=1}^{T} L_t, \qquad L_t = -\ln p_t[k_t]
$$

구하려는 것은 $\partial L / \partial W_{xh}$, $\partial L / \partial W_{hh}$, $\partial L / \partial W_{hy}$ 와 bias의 기울기입니다.

### 계산 그래프에서 $h_t$는 두 곳으로 나간다

```
        L_{t-1}          L_t            L_{t+1}
          ▲               ▲               ▲
         y_{t-1}         y_t            y_{t+1}
          ▲               ▲               ▲
 ... ──▶ h_{t-1} ──────▶ h_t ────────▶ h_{t+1} ──▶ ...
          ▲               ▲               ▲
         x_{t-1}         x_t            x_{t+1}
```

$h_t$는 (1) 같은 시점의 출력 $y_t$ 로, (2) 다음 시점의 $h_{t+1}$ 로 갑니다. 연쇄 법칙에서 한 변수가 여러 곳으로 나가면 각 경로로 돌아온 기울기를 더합니다. 그래서 $h_t$에 대한 기울기는 두 항의 합입니다.

$$
\frac{\partial L}{\partial h_t} = \underbrace{W_{hy}^\top \frac{\partial L_t}{\partial y_t}}_{\text{지금 시점의 출력에서}} + \underbrace{\frac{\partial L}{\partial h_{t+1}} \text{ 이 } h_t \text{ 로 넘어온 것}}_{\text{미래 시점 전체에서}}
$$

둘째 항이 "through time"입니다. 시점 $t$의 hidden state는 $t$ 이후 모든 시점의 loss에 책임이 있습니다.

### 조각별 미분

#### softmax와 cross-entropy

softmax 뒤에 cross-entropy가 붙으면 logit에 대한 기울기가 간단해집니다.

$$
\frac{\partial L_t}{\partial y_t} = p_t - \text{onehot}(k_t)
$$

모델이 낸 확률에서 정답 자리만 1을 뺀 것입니다. 정답에 0.2를 줬다면 그 자리는 $-0.8$(올려라), 오답에 0.5를 줬다면 그 자리는 $+0.5$(내려라)가 됩니다.

#### tanh

$\tanh'(a) = 1 - \tanh^2(a)$ 이고 $h_t = \tanh(a_t)$ 이므로 다음과 같습니다.

$$
\frac{\partial L}{\partial a_t} = (1 - h_t^2) \odot \frac{\partial L}{\partial h_t}
$$

$h_t$의 어떤 차원이 $\pm 1$에 가까우면(포화) 그 차원의 $1 - h_t^2$ 는 0에 가깝고 기울기가 거기서 끊깁니다.

#### $a_t$ 에서 가중치와 과거로

$a_t = W_{xh} x_t + W_{hh} h_{t-1} + b_h$ 는 선형입니다. 이 시점의 기울기를 $\delta_t = \partial L / \partial a_t$ 로 줄여 쓰면 다음과 같습니다.

$$
\frac{\partial L}{\partial W_{xh}} \mathrel{+}= \delta_t\, x_t^\top, \qquad
\frac{\partial L}{\partial W_{hh}} \mathrel{+}= \delta_t\, h_{t-1}^\top, \qquad
\frac{\partial L}{\partial b_h} \mathrel{+}= \delta_t
$$
$$
h_{t-1} \text{ 로 넘기는 기울기} = W_{hh}^\top \delta_t
$$

$\mathrel{+}=$ 인 이유는 가중치 공유입니다. 같은 $W_{hh}$ 가 모든 시점에서 쓰였으므로 각 시점의 기여를 전부 더합니다.

### 정리: 뒤에서 앞으로 도는 점화식

$$
\boxed{\;
\delta_t = (1 - h_t^2) \odot \Big( W_{hy}^\top (p_t - \text{onehot}(k_t)) + W_{hh}^\top \delta_{t+1} \Big), \qquad \delta_{T+1} = 0
\;}
$$

$$
\frac{\partial L}{\partial W_{hh}} = \sum_{t=1}^{T} \delta_t\, h_{t-1}^\top, \qquad
\frac{\partial L}{\partial W_{xh}} = \sum_{t=1}^{T} \delta_t\, x_t^\top, \qquad
\frac{\partial L}{\partial W_{hy}} = \sum_{t=1}^{T} (p_t - \text{onehot}(k_t))\, h_t^\top
$$

순전파가 $h_0 \to h_T$ 방향의 점화식이었다면 역전파는 $\delta_T \to \delta_1$ 방향의 점화식입니다. 구조가 대칭입니다.

### 야코비안

스칼라 함수의 미분은 숫자 하나입니다. 벡터를 받아 벡터를 내는 함수의 미분은 행렬이고 이것을 야코비안(Jacobian)이라고 부릅니다. $(i, j)$ 원소는 "입력의 $j$번째를 조금 바꾸면 출력의 $i$번째가 얼마나 바뀌나"입니다.

RNN의 한 시점은 $h_{t-1}$ 을 받아 $h_t$ 를 내는 함수입니다.

$$
h_t = \tanh(W_{hh} h_{t-1} + W_{xh} x_t + b_h)
$$

이 함수의 야코비안을 구합니다. tanh는 원소별 함수이므로 그 미분은 대각 행렬이고, 안쪽은 선형이므로 미분이 $W_{hh}$입니다.

$$
\frac{\partial h_t}{\partial h_{t-1}} = \text{diag}(1 - h_t^2)\; W_{hh}
$$

- $\text{diag}(v)$: 벡터 $v$를 대각선에 놓은 대각 행렬.
- $1 - h_t^2$: tanh의 미분. 각 원소는 0과 1 사이입니다.

### 여러 시점을 건너뛰면 곱이 된다

시점 $k$의 hidden state가 시점 $t$의 hidden state에 미치는 영향은 연쇄 법칙으로 중간 야코비안을 전부 곱한 것입니다.

$$
\frac{\partial h_t}{\partial h_k} = \prod_{j=k+1}^{t} \frac{\partial h_j}{\partial h_{j-1}} = \prod_{j=k+1}^{t} \text{diag}(1 - h_j^2)\; W_{hh}
$$

행렬이 $t - k$개 곱해집니다. 그리고 그 안의 $W_{hh}$ 는 전부 같은 행렬입니다. 위 점화식에서 $\delta$가 한 칸 거슬러 갈 때마다 $W_{hh}^\top$ 와 $(1 - h^2)$ 가 곱해지던 것이 이 식의 전치입니다.

### 곱의 크기를 묶는다

행렬의 "크기"를 재는 도구가 필요합니다.

- 특이값(singular value): 행렬이 벡터를 가장 많이 늘리는 배율부터 가장 적게 늘리는 배율까지를 나열한 값. 가장 큰 것을 $\sigma_{\max}$ 로 씁니다.
- 스펙트럼 노름 $\lVert A \rVert$: $A$의 가장 큰 특이값. "$A$를 곱하면 벡터 길이가 최대 몇 배가 되나"입니다.
- 성질: $\lVert AB \rVert \le \lVert A \rVert \, \lVert B \rVert$.

한 시점의 야코비안에 적용합니다. tanh 미분의 최댓값을 $\gamma$ 라 하면($\gamma \le 1$) 다음이 성립합니다.

$$
\left\lVert \frac{\partial h_j}{\partial h_{j-1}} \right\rVert \le \lVert \text{diag}(1 - h_j^2) \rVert \, \lVert W_{hh} \rVert \le \gamma\, \sigma_{\max}(W_{hh})
$$

$t - k$ 개를 곱하면 다음과 같습니다.

$$
\left\lVert \frac{\partial h_t}{\partial h_k} \right\rVert \le \big(\gamma\, \sigma_{\max}(W_{hh})\big)^{t-k}
$$

#### 읽는 법

| 조건 | 결과 |
|---|---|
| $\gamma\, \sigma_{\max} < 1$ | 상한이 0으로 갑니다. 기울기가 소실됩니다 (충분조건) |
| $\gamma\, \sigma_{\max} > 1$ | 상한이 커집니다. 폭발할 수 있습니다 (필요조건) |

두 번째 줄이 비대칭인 이유는 부등식이 상한만 주기 때문입니다. $\sigma_{\max} > 1$ 이어도 실제 곱은 작을 수 있습니다. tanh가 포화되면 $1 - h^2$ 이 0에 가까워져서 기울기를 끊습니다. 이 조건은 Pascanu, Mikolov, Bengio의 2013년 논문(*On the difficulty of training recurrent neural networks*)에서 정리된 것입니다.

#### 스칼라로 단순화하면

hidden이 1차원이면 $W_{hh}$ 는 숫자 $w$ 하나이고 식은 다음이 됩니다.

$$
\frac{\partial h_t}{\partial h_k} = \prod_{j=k+1}^{t} w\,(1 - h_j^2)
$$

$w = 0.9$, $1 - h^2 \approx 1$ 이면 $0.9^{t-k}$입니다. $0.9^{100} \approx 0.000027$ 입니다([먼 거리의 짝 맞추기](08-long-term-dependency.md)의 표).

## 코드로 확인

### 역전파

`experiments/char_rnn.py` 의 `backward`에서 반복문 부분입니다. 오른쪽 주석은 위에서 유도한 식과의 대응입니다. 반복문 앞에서 기울기 사전 `g`를 0으로 만들고 끝에서 `g`를 돌려줍니다.

```python
dh_next = np.zeros(self.H)                         # δ_{T+1} 에서 넘어온 것 = 0
for t in reversed(range(len(inputs))):             # 뒤에서 앞으로
    dy = ps[t].copy(); dy[targets[t]] -= 1.0       # p_t - onehot(k_t)
    g["Why"] += np.outer(dy, hs[t]); g["by"] += dy
    dh = p["Why"].T @ dy + dh_next                 # 지금 출력에서 온 것 + 미래에서 온 것
    da = (1.0 - hs[t] ** 2) * dh                   # δ_t. tanh 의 미분
    g["bh"] += da
    g["Wxh"] += np.outer(da, xs[t])                # δ_t x_t^T 를 누적
    g["Whh"] += np.outer(da, hs[t - 1])            # δ_t h_{t-1}^T 를 누적
    dh_next = p["Whh"].T @ da                      # 한 시점 과거로 넘긴다
```

### 기울기 검사

유도와 구현이 맞는지는 수치 미분과 비교해서 확인합니다. 가중치 하나를 아주 조금($\epsilon = 10^{-5}$) 올리고 내려서 loss의 차이를 재면 그 가중치의 기울기 근삿값이 나옵니다.

$$
\frac{\partial L}{\partial w} \approx \frac{L(w + \epsilon) - L(w - \epsilon)}{2\epsilon}
$$

`experiments/char_rnn.py`의 `grad_check()`가 모든 가중치에 대해 이것을 하고 BPTT 결과와 비교합니다. `experiments/gradient_check.py`는 이 검사만 따로 실행합니다. 결과는 아래 「실험 결과」에 있습니다.

프레임워크(PyTorch 등)를 쓰면 자동 미분이 해 주므로 역전파를 직접 짤 일은 드뭅니다. 직접 짰다면 이 검사 없이는 결과를 믿기 어렵습니다. 틀린 역전파도 loss는 어느 정도 줄어들기 때문에 버그가 눈에 잘 띄지 않습니다.

```python
def grad_check(seed=1):
    ...
    for k, W in m.p.items():
        for idx in (모든 원소):
            W[idx] = old + 1e-5; lp, _ = m.forward(inputs, targets, h0)
            W[idx] = old - 1e-5; lm, _ = m.forward(inputs, targets, h0)
            W[idx] = old
            num = (lp - lm) / 2e-5                                   # 수치 미분
            rel = abs(num - g[k][idx]) / (abs(num) + abs(g[k][idx])) # 상대 오차
```

## 실험 결과

### 기울기 검사

`python3 gradient_check.py`의 출력입니다.

```
grad check, worst relative error: 8.13299735387754e-09
```

모든 가중치 원소에서 BPTT 결과와 수치 미분의 상대 오차가 최대 $8.1 \times 10^{-9}$ 입니다([results/gradient-check.txt](results/gradient-check.txt)). seed를 2와 3으로 바꾸면 $5.1 \times 10^{-10}$, $1.0 \times 10^{-9}$ 입니다. 유도한 점화식과 구현이 이 정밀도에서 일치합니다.

직접 해 볼 것: `backward` 에서 `+ dh_next` 를 지우고 다시 돌립니다. 오차가 크게 뜁니다. 그 항이 "through time" 부분입니다.

### 야코비안 곱의 크기

hidden 50차원짜리 RNN에 무작위 입력을 60시점 넣으면서 $\partial h_t / \partial h_0$ 의 스펙트럼 노름을 쟀습니다(seed 하나, `experiments/hand_calc.py`, [results/hand-calc.txt](results/hand-calc.txt)). $W_{hh}$ 의 크기만 세 가지로 바꿨습니다.

| $\sigma_{\max}(W_{hh})$ | 1시점 뒤 | 10시점 뒤 | 30시점 뒤 | 60시점 뒤 |
|---|---|---|---|---|
| 0.94 | $7.4 \times 10^{-1}$ | $6.8 \times 10^{-5}$ | $4.1 \times 10^{-14}$ | $1.6 \times 10^{-28}$ |
| 1.94 | $1.5$ | $5.6 \times 10^{-2}$ | $7.0 \times 10^{-6}$ | $3.8 \times 10^{-12}$ |
| 3.83 | $3.1$ | $1.7 \times 10^{1}$ | $1.7 \times 10^{1}$ | $8.3 \times 10^{1}$ |

- 첫 줄: $\sigma_{\max} < 1$. 60시점 뒤에는 $10^{-28}$ 입니다. float32는 이 크기의 수를 표현할 수 있지만(정규 수의 최솟값이 약 $10^{-38}$), 유효숫자가 7자리 정도라서 크기 1 안팎인 다른 기울기와 더하면 흔적이 남지 않습니다.
- 둘째 줄: $\sigma_{\max} = 1.94 > 1$ 인데도 소실됩니다. tanh 미분이 평균적으로 기울기를 깎기 때문입니다. $\sigma_{\max} > 1$ 은 폭발의 필요조건이고 충분조건은 아닙니다.
- 셋째 줄: 충분히 크면 폭발합니다. 그런데 무한히 커지지는 않습니다. hidden이 포화되면서 tanh 미분이 작아져 증가가 멈춥니다.

표의 둘째 줄처럼 $\sigma_{\max}$ 가 1보다 큰데도 소실되는 범위가 넓습니다. 특별히 손쓰지 않으면 vanilla RNN의 기울기는 소실되는 쪽으로 갑니다.

## 결과 해석

BPTT에서 기울기는 한 시점을 거슬러 갈 때마다 $W_{hh}^\top$ 와 tanh의 미분이 곱해지면서 전파됩니다. 같은 행렬이 거리만큼 곱해지므로 크기는 거리에 대해 지수적으로 변하고, 위 측정에서는 $\sigma_{\max}$ 가 1.94여도 60시점 뒤에 $10^{-12}$ 까지 줄었습니다. 폭발과 소실은 대처 방법이 다릅니다.

### 폭발은 자르면 된다: gradient clipping

폭발은 눈에 띕니다. loss가 갑자기 `nan`이 되거나 튑니다. 해법도 단순합니다. 기울기 벡터 전체의 길이가 임계값 $\theta$ 를 넘으면 방향은 두고 길이만 $\theta$ 로 줄입니다.

$$
g \leftarrow \begin{cases} g \times \dfrac{\theta}{\lVert g \rVert} & \lVert g \rVert > \theta \\[2mm] g & \text{그 외} \end{cases}
$$

`experiments/char_lstm.py`의 한 줄이 이것입니다.

```python
nn.utils.clip_grad_norm_(model.parameters(), 1.0)
```

numpy 코드는 더 단순한 변형(원소별로 $[-5, 5]$ 범위로 자르기)을 씁니다. 방향이 조금 바뀌지만 구현이 한 줄입니다.

```python
np.clip(g[k], -clip, clip, out=g[k])
```

### 소실은 자를 수 없다

소실은 사정이 다릅니다.

- 눈에 띄지 않습니다. loss는 정상적으로 줄어듭니다. 가까운 패턴을 배우는 것만으로도 loss가 줄기 때문입니다. [먼 거리의 짝 맞추기](08-long-term-dependency.md) 실험에서 vanilla RNN의 perplexity(1.542)는 n-gram(1.604)보다 낮았습니다.
- 작아진 기울기만 골라 키울 수도 없습니다. 가중치의 기울기는 모든 시점에서 온 기울기의 합인데, 먼 시점에서 온 $10^{-28}$ 크기의 항은 가까운 시점에서 온 크기 1 안팎의 항과 더해지면서 float32의 유효숫자 아래로 사라집니다.
- 학습률을 올려도 안 됩니다. 가까운 시점의 기울기가 같이 커져서 발산합니다.

그래서 구조를 바꿔야 합니다. 기울기가 지나는 길에서 "같은 행렬의 반복 곱"을 없애는 것입니다.

| 해법 | 아이디어 | 어디서 다루나 |
|---|---|---|
| ReLU | 활성화 함수의 미분을 1로 (양수 구간) | $\gamma$ 를 키웁니다. $\sigma_{\max}$ 문제는 그대로입니다 |
| 잔차 연결 | 덧셈 경로를 따로 냅니다. 야코비안이 $I + \partial F/\partial x$ | ResNet |
| LSTM | cell state에 덧셈 경로를 내고 계수를 학습합니다 | [LSTM의 기울기 경로](07-lstm-gradient-path.md) |
| 초기화, 직교 행렬 | $\sigma_{\max} \approx 1$ 로 시작합니다 | 여기서는 다루지 않습니다 |
| attention | 먼 시점을 직접 잇습니다. 경로 길이가 1 | [순환 구조의 한계](10-limits-of-recurrence.md) |

#### 깊은 CNN과의 비교

RNN을 펼치면 100단어는 layer 100개짜리 네트워크가 됩니다. 깊은 CNN의 기울기도 layer별 야코비안의 곱이었습니다. 다른 점은 하나입니다. CNN은 layer마다 다른 가중치를 곱하고 RNN은 같은 가중치를 곱합니다. 서로 다른 행렬의 곱은 늘리는 방향이 layer마다 달라 상쇄될 여지가 있지만, 같은 행렬을 거듭 곱하면 절댓값이 가장 큰 고윳값(스펙트럼 반경)이 거듭제곱으로 작용해서, 그 값이 1보다 작으면 계속 줄고 1보다 크면 계속 커집니다. 그래서 같은 깊이라면 RNN 쪽이 더 불리합니다.

## 한계와 주의할 점

### truncated BPTT

전체 BPTT에는 비용 문제가 있습니다.

- 말뭉치가 100만 글자면 펼친 네트워크가 100만 layer입니다.
- 역전파를 하려면 모든 시점의 $h_t$를 메모리에 들고 있어야 합니다.
- 가중치를 한 번 갱신하려고 100만 시점을 앞뒤로 다 돌아야 합니다.

truncated BPTT는 시퀀스를 길이 $k$인 조각으로 끊고, 역전파를 조각 안에서만 합니다. hidden state의 값은 다음 조각으로 넘기되, 기울기는 조각 경계를 넘지 않습니다.

```
조각 1 (시점 1~64)          조각 2 (시점 65~128)         조각 3
h0 ─▶ ... ─▶ h64  ═══값만═══▶ h64 ─▶ ... ─▶ h128 ═══값만═══▶ ...
   ◀── 기울기 ──┘               ◀── 기울기 ──┘
                  ✕ 기울기는 넘지 않는다
```

numpy 코드에서는 `step()`이 조각 하나에 대해서만 `backward`를 돌리고 마지막 hidden state를 값으로만 돌려주는 것이 그것입니다. PyTorch 코드에서는 `detach()`입니다.

```python
logits, state, _ = model(x, detach(state))    # 값은 이어받고 계산 그래프는 끊는다
```

대가는 분명합니다. 조각 길이 $k$보다 먼 의존성은 기울기로 배울 수 없습니다. 순전파에서는 정보가 조각을 넘어 전달되지만, "그 정보를 남겨 두면 좋다"는 학습 신호는 $k$ 시점 안에서만 옵니다. 이 저장소의 실험은 $k = 64$ 이고 환경 이름까지의 거리는 최대 47이라 대부분 한 조각 안에 들어옵니다. 그래도 조각 경계에 걸친 줄은 신호가 끊깁니다.

Karpathy의 char-rnn은 조각 길이의 기본값이 50입니다.

## 확인 문제

1. $\delta_t$의 점화식에서 $W_{hh}^\top \delta_{t+1}$ 항을 지우면 무엇을 계산하게 되나요? (답: 각 시점을 독립된 분류 문제로 본 기울기. hidden state가 미래 loss에 미치는 영향을 무시합니다)
2. 길이 1,000의 시퀀스를 $k = 50$으로 끊으면 역전파를 위해 동시에 메모리에 들고 있어야 하는 hidden state는 몇 개인가요? (답: 50개)
3. 기울기 검사에서 $\epsilon$을 너무 작게($10^{-12}$) 잡으면 생기는 문제는 무엇인가요? (답: 두 loss의 차이가 부동소수점 오차에 묻힙니다)
4. $\gamma\, \sigma_{\max} = 0.95$ 이면 50시점 뒤 기울기 노름의 상한은 얼마인가요? (답: $0.95^{50} = 0.077$)
5. 위 실측표의 둘째 줄에서 $\sigma_{\max} > 1$ 인데도 소실된 이유는 무엇인가요? (답: tanh 미분이 1보다 작고 포화된 차원에서는 0에 가깝습니다. 부등식은 상한일 뿐입니다)
6. gradient clipping이 소실에는 도움이 안 되는 이유는 무엇인가요? (답: clipping은 큰 값을 줄이는 연산입니다. 작은 값을 키우지 않습니다)

## References

- [On the difficulty of training Recurrent Neural Networks](https://arxiv.org/abs/1211.5063) (Pascanu, Mikolov, Bengio, 2013)
- [Minimal character-level language model with a Vanilla Recurrent Neural Network](https://gist.github.com/karpathy/d4dee566867f8291f086) (Karpathy)
- [char-rnn](https://github.com/karpathy/char-rnn) (Karpathy)
