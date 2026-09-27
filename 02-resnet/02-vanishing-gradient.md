# 기울기는 왜 layer를 지날수록 작아지거나 커지고, 초기화는 그것을 어디까지 막는가?

## 문제

AlexNet이 ReLU로 기울기 소실을 해결했다고 배운 뒤에 ResNet이 같은 문제를 다시 다루는 것을 보면 앞뒤가 맞지 않아 보입니다. 연쇄법칙을 곱셈으로 풀어 ReLU가 고친 것과 남긴 것을 나누고, 가중치 초기화가 학습 시작 시점의 크기를 어떻게 맞추는지, 그리고 그것만으로는 왜 부족한지를 봅니다.

## 아이디어

순전파가 끝나면 출력과 정답의 차이인 손실을 계산하고, 손실을 뒤에서 앞으로 전파해서 각 layer의 가중치를 어떻게 고칠지 정합니다. 이 계산이 역전파(backpropagation)이고, 전파되는 값이 기울기(gradient)입니다. 기울기는 layer를 거꾸로 지날 때마다 그 layer의 가중치와 활성화 함수의 미분이 곱해집니다. 곱해지는 수가 1보다 조금만 작아도 입력 쪽 layer에 도달할 때는 거의 0이 되고, 1보다 조금만 커도 폭발합니다. 앞의 경우가 **기울기 소실**(vanishing gradient), 뒤의 경우가 **기울기 폭발**(exploding gradient)입니다.

같은 일이 순방향에서도 일어납니다. layer 하나를 지날 때마다 활성값의 분산에 어떤 수가 곱해지고, 그 수가 1이 아니면 깊이에 대해 지수적으로 사라지거나 커집니다. 가중치 초기화는 학습을 시작하는 시점에 이 수를 1로 맞추는 일입니다.

## 수식으로 보기

### 연쇄법칙은 곱셈이다

layer마다 뉴런이 하나뿐인 네트워크를 봅니다.

$$
a_0 = x,\qquad z_l = w_l\, a_{l-1} + b_l,\qquad a_l = \sigma(z_l),\qquad l = 1,\dots,L
$$

- $a_l$: $l$번째 layer의 출력(활성값)
- $w_l, b_l$: $l$번째 layer의 가중치와 편향
- $\sigma$: 활성화 함수 (sigmoid, tanh, ReLU 등)
- $L$: layer의 개수. 손실은 $\mathcal{L} = \text{loss}(a_L, \text{정답})$

첫 번째 layer의 가중치 $w_1$에 대한 기울기를 연쇄법칙으로 펼치면

$$
\frac{\partial \mathcal{L}}{\partial w_1}
= \frac{\partial \mathcal{L}}{\partial a_L}
\cdot \underbrace{\prod_{l=2}^{L} \frac{\partial a_l}{\partial a_{l-1}}}_{L-1 \text{개}}
\cdot \frac{\partial a_1}{\partial w_1},
\qquad
\frac{\partial a_l}{\partial a_{l-1}} = \sigma'(z_l)\, w_l
$$

layer 하나를 지날 때마다 $\sigma'(z_l)\,w_l$이 하나씩 곱해집니다.

뉴런이 여러 개면 스칼라 곱이 야코비안 행렬의 곱이 됩니다. $J_l = \partial a_l / \partial a_{l-1}$은 $l-1$번째 layer 출력의 각 원소가 $l$번째 layer 출력의 각 원소에 주는 영향을 모은 행렬입니다.

$$
\frac{\partial \mathcal{L}}{\partial a_0} = J_1^{\top} J_2^{\top} \cdots J_L^{\top}\, \frac{\partial \mathcal{L}}{\partial a_L},
\qquad J_l = \text{diag}\big(\sigma'(z_l)\big)\, W_l
$$

layer마다 다른 행렬이 곱해지므로, 각 행렬이 벡터의 길이를 최대 몇 배까지 늘리는가(가장 큰 특이값)의 곱이 전체 배율의 상한이 됩니다. 1보다 작은 쪽이 우세하면 소실, 큰 쪽이 우세하면 폭발입니다.

### ReLU가 고친 것

sigmoid의 도함수는 $\sigma'(z) = \sigma(z)(1-\sigma(z))$이고 최댓값이 $z=0$에서 0.25입니다. 가중치가 1 근처여도 layer마다 0.25 이하가 곱해집니다. ReLU는 $\max(0, z)$이고 도함수는 켜진 뉴런($z > 0$)에서 1, 꺼진 뉴런에서 0입니다.

| | sigmoid | ReLU |
|---|---|---|
| $\sigma'$의 범위 | 0 ~ 0.25 | 0 또는 1 |
| 8-layer에서 | 이미 어렵습니다 | 충분합니다 |
| 56-layer 이상에서 | 학습이 거의 불가능합니다 | $w_l$의 곱과 꺼진 뉴런 때문에 여전히 어렵습니다 |

ReLU는 곱해지는 수 $\sigma'(z_l)\,w_l$의 앞쪽 인수만 1로 만들었습니다. $w_l$은 그대로 남고, 꺼진 뉴런에서는 0이 곱해져 그 경로의 기울기가 끊깁니다.

### 초기화: 분산이 유지되는 조건

layer 하나의 출력 뉴런 하나를 봅니다.

$$
z = \sum_{i=1}^{n} w_i x_i
$$

- $n$: 이 뉴런으로 들어오는 입력의 개수. fan-in이라고 부릅니다. 합성곱이면 $k^2 \cdot C_{in}$입니다
- $w_i$: 평균 0, 분산 $\text{Var}(w)$인 분포에서 독립으로 뽑은 가중치
- $x_i$: 평균 0, 분산 $\text{Var}(x)$인 입력

독립인 확률변수의 곱과 합의 분산 규칙에 따라

$$
\text{Var}(z) = n \cdot \text{Var}(w) \cdot \text{Var}(x)
$$

이므로 layer 하나를 지날 때마다 분산이 $n \cdot \text{Var}(w)$배가 됩니다.

**Xavier 초기화**(Glorot, Bengio, 2010)는 순방향에서 $n_{in}\,\text{Var}(w) = 1$, 역방향에서 $n_{out}\,\text{Var}(w) = 1$이 되기를 원하는데 둘을 동시에 만족할 수 없으므로 절충합니다. $n_{out}$은 fan-out, 이 layer의 출력 하나가 다음 layer의 몇 개 뉴런으로 나가는가입니다.

$$
\text{Var}(w) = \frac{2}{n_{in} + n_{out}}
$$

이 유도는 활성화 함수가 0 근처에서 선형이라고 가정하므로 tanh에는 맞고 ReLU에는 맞지 않습니다.

**He 초기화**(He, Zhang, Ren, Sun, 2015)는 ReLU를 따로 다룹니다. 입력이 0을 중심으로 대칭이면 ReLU가 절반을 0으로 만들어 출력의 2차 모멘트가 절반이 됩니다.

$$
\mathbb{E}\big[\text{ReLU}(z)^2\big] = \tfrac{1}{2}\,\text{Var}(z)
\qquad\Longrightarrow\qquad
\text{Var}(w) = \frac{2}{n_{in}}
$$

Xavier보다 2배 크게 뽑아서 ReLU가 절반을 끄는 것을 보상합니다. ResNet의 저자들이 ResNet 직전에 낸 논문 "Delving Deep into Rectifiers"에 들어 있고, PyTorch에서는 `nn.init.kaiming_normal_(w, mode="fan_out", nonlinearity="relu")`로 씁니다. torchvision의 ResNet 구현이 이 방식입니다.

## 작은 숫자로 직접 계산

### 곱해지는 수 $c$가 $L$번 곱해지면

| $c$ | 8-layer (AlexNet) | 20-layer | 56-layer | 152-layer |
|---|---|---|---|---|
| 0.5 | $3.9 \times 10^{-3}$ | $9.5 \times 10^{-7}$ | $1.4 \times 10^{-17}$ | $1.8 \times 10^{-46}$ |
| 0.9 | 0.43 | 0.12 | $2.7 \times 10^{-3}$ | $1.1 \times 10^{-7}$ |
| 1.0 | 1 | 1 | 1 | 1 |
| 1.1 | 2.1 | 6.7 | 208 | $2.0 \times 10^{6}$ |

0.9는 layer가 8개일 때는 거의 문제가 없지만(0.43) 152-layer에서는 천만분의 1이 됩니다. 1.1은 152-layer에서 200만 배가 됩니다. 딱 1일 때만 깊이와 무관한데, 학습 중에 모든 layer가 1을 유지할 이유는 없습니다. sigmoid의 최댓값 0.25를 10번 곱하면 $9.5 \times 10^{-7}$입니다.

직렬 서비스 호출에 비유하면, 각 서비스의 성공률이 99%여도 50개를 직렬로 거치면 $0.99^{50} \approx 0.61$입니다. 개별 단계가 멀쩡해 보여도 직렬 길이가 길면 곱이 무너집니다.

### 초기화: fan-in 576인 layer를 50개 지나면

3x3 커널, 입력 64채널이면 fan-in은 $9 \times 64 = 576$입니다.

| $\text{Var}(w)$ | layer당 배율 $n\,\text{Var}(w)$ | layer 50개 뒤 분산 |
|---|---|---|
| $0.01^2$ (표준편차 0.01, AlexNet 방식) | 0.0576 | $0.0576^{50} \approx 10^{-62}$. 신호가 사라집니다 |
| $0.05^2$ | 1.44 | $1.44^{50} \approx 8 \times 10^{7}$. 폭발합니다 |
| $1/576$ | 1 | 1. 유지됩니다 |

표준편차 0.01은 8-layer인 AlexNet에서는 통했지만 50-layer에서는 첫 순전파에서 이미 출력이 0이 됩니다. He 초기화로 계산하면 $\text{Var}(w) = 2/576 \approx 0.00347$, 표준편차 약 0.0589입니다.

두 표의 숫자는 [results/hand-calc.txt](results/hand-calc.txt)와 같습니다.

## 코드로 확인

`experiments/grad_norms.py`의 실험 1은 폭 256인 50-layer MLP에 무작위 입력 64개를 넣고 역전파를 한 번 한 뒤, 가중치 텐서마다 기울기의 RMS(제곱 평균의 제곱근)를 입력 쪽부터 찍습니다. 학습은 하지 않습니다.

```python
def grad_norms(model, x, y):
    model.zero_grad(set_to_none=True)
    loss_fn(model(x), y).backward()
    out = []
    for name, p in model.named_parameters():
        if p.grad is not None and p.dim() > 1:          # 가중치만 (bias, BN 제외)
            # float32 그대로 제곱하면 아주 작은 값이 0이 되므로 float64로 계산한다
            out.append((name, p.grad.double().pow(2).mean().sqrt().item()))
    return out
```

세 가지 구성을 비교합니다. sigmoid, ReLU에 PyTorch `nn.Linear`의 기본 초기화, ReLU에 He 초기화입니다.

## 실험 결과

PyTorch 2.14, CPU, seed 0. 왼쪽 숫자는 가중치 텐서의 순서이고 0이 입력 쪽입니다([results/grad-norms.txt](results/grad-norms.txt)).

```
    sigmoid        ReLU (기본 초기화)   ReLU (He 초기화)
 0  0.000e+00      1.766e-22           3.746e-03
10  2.927e-37      4.999e-20           3.034e-03
20  8.254e-29      3.049e-16           3.231e-03
30  2.250e-20      1.754e-12           2.270e-03
40  6.473e-12      1.680e-08           1.532e-03
50  1.329e-02      6.744e-04           6.970e-03
```

- sigmoid: 출력 쪽에서 입력 쪽으로 layer 10개를 지날 때마다 기울기가 약 $10^{-9}$배가 됩니다. layer 하나당 약 7분의 1입니다. 입력 쪽 첫 layer의 값은 float32가 표현할 수 있는 가장 작은 수 아래로 내려가 0이 됐습니다.
- ReLU, 기본 초기화: sigmoid보다는 낫지만 layer 10개마다 약 $10^{-4}$배로 줄어듭니다. `nn.Linear`의 기본 초기화는 ReLU에 맞춘 분산보다 작습니다.
- ReLU, He 초기화: 입력 쪽 $3.7 \times 10^{-3}$, 출력 쪽 $7.0 \times 10^{-3}$으로 layer 50개에 걸쳐 거의 일정합니다. $\text{Var}(w) = 2/n_{in}$이 역방향에서도 크기를 유지합니다.

## 결과 해석

기울기가 layer를 지날수록 작아지거나 커지는 것은 연쇄법칙이 곱셈이기 때문입니다. layer마다 곱해지는 수가 1에서 조금만 벗어나도 깊이에 대해 지수적으로 벌어집니다. ReLU는 곱해지는 수 가운데 활성화 함수의 미분만 고쳤고, He 초기화는 학습 시작 시점에 순방향과 역방향의 배율을 1로 맞춥니다. 위 실험의 세 번째 열이 그 결과입니다.

초기화가 막는 것은 학습 시작 시점까지입니다. 가중치가 갱신되기 시작하면 독립성 가정이 깨지고 $n\,\text{Var}(w) = 1$이 유지된다는 보장이 없습니다. 학습 중에도 매 layer에서 크기를 다시 맞추는 장치가 batch normalization이고([05-batch-normalization.md](05-batch-normalization.md)), 곱셈 옆에 덧셈 경로를 두는 것이 residual connection입니다([04-skip-connection-gradient.md](04-skip-connection-gradient.md)).

| 해법 | 하는 일 | 곱해지는 수로 말하면 |
|---|---|---|
| 가중치 초기화 | 학습 시작 시점에 $c \approx 1$이 되게 합니다 | 학습이 진행되면 보장이 없습니다 |
| batch normalization | layer마다 값의 크기를 다시 맞춥니다 | $c$가 1에서 멀어지는 것을 매 layer에서 교정합니다 |
| gradient clipping | 기울기 norm이 상한을 넘으면 잘라 냅니다 | 폭발만 막고 소실에는 효과가 없습니다 |
| residual connection | 곱셈 옆에 덧셈 경로를 만듭니다 | $c$ 대신 $1 + c'$가 곱해집니다 |

"처음에는 아무것도 하지 않는 상태에서 시작한다"는 생각은 초기화에서도 쓰입니다.

| 기법 | 내용 |
|---|---|
| zero-init residual | 각 residual block의 마지막 BN의 $\gamma$를 0으로 초기화합니다. 시작 시점에 블록의 변환 경로 출력이 0이라 블록 전체가 항등 사상입니다. torchvision ResNet의 `zero_init_residual=True` 옵션입니다 |
| dilated convolution의 항등 초기화 | Yu와 Koltun(2016)은 dilated conv를 쌓은 모듈이 일반적인 무작위 초기화로는 잘 학습되지 않아, 각 필터가 대응하는 입력 채널을 그대로 복사하도록 초기화했습니다 |

## 한계와 주의할 점

- 실험은 학습 전 초기화 상태에서 역전파를 한 번 한 결과입니다. 학습이 진행되는 동안의 기울기 크기는 재지 않았습니다.
- MLP로 잰 값입니다. 합성곱과 BN이 있는 네트워크의 측정은 [04-skip-connection-gradient.md](04-skip-connection-gradient.md)에 있습니다.
- BN과 좋은 초기화를 쓰면 기울기 소실은 상당히 잡히는데도 깊은 plain 네트워크의 성능은 떨어졌습니다. 그 현상이 [01-why-depth-fails.md](01-why-depth-fails.md)의 degradation입니다.

## References

- [Understanding the difficulty of training deep feedforward neural networks](https://proceedings.mlr.press/v9/glorot10a.html) (Glorot, Bengio, 2010)
- [Delving Deep into Rectifiers](https://arxiv.org/abs/1502.01852) (He, Zhang, Ren, Sun, 2015)
- [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385) (He, Zhang, Ren, Sun, 2015)
- [Multi-Scale Context Aggregation by Dilated Convolutions](https://arxiv.org/abs/1511.07122) (Yu, Koltun, 2016)
