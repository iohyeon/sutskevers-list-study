# skip connection은 역전파 식에서 무엇을 바꾸고, 실제 네트워크에서 기울기는 어떻게 달라지는가?

## 문제

skip connection이 기울기 소실을 막는다고들 합니다. 덧셈 하나가 역전파 식에 무엇을 더하는지 유도하고, 곱셈이 $2^L$개 경로의 합으로 펼쳐지는 것을 계산한 뒤, BN이 없는 MLP와 BN이 있는 CNN에서 layer별 기울기를 실제로 잽니다.

## 아이디어

역전파에서 덧셈 노드는 들어온 기울기를 양쪽 입력에 그대로 복사합니다. $y = a + b$이면 $\partial y/\partial a = \partial y/\partial b = 1$이기 때문입니다. 그래서 $y = F(x) + x$에서 shortcut 쪽으로 돌아가는 기울기에는 가중치가 하나도 곱해지지 않습니다. 변환 경로의 기울기가 0으로 죽어도 shortcut으로 온 기울기는 남습니다.

## 수식으로 보기

### 덧셈의 기울기

블록 $y = F(x) + x$에서 손실 $\mathcal{L}$의 $x$에 대한 기울기는

$$
\frac{\partial \mathcal{L}}{\partial x}
= \frac{\partial \mathcal{L}}{\partial y}\left(1 + \frac{\partial F}{\partial x}\right)
= \underbrace{\frac{\partial \mathcal{L}}{\partial y}}_{\text{shortcut으로 온 것}} + \underbrace{\frac{\partial \mathcal{L}}{\partial y}\cdot\frac{\partial F}{\partial x}}_{\text{변환 경로로 온 것}}
$$

- $\partial \mathcal{L}/\partial y$: 뒤쪽에서 이 블록의 출력까지 도착한 기울기
- $\partial F/\partial x$: 변환 경로의 야코비안. conv 가중치와 활성화 도함수의 곱입니다
- $1$: shortcut 경로의 기울기. 벡터 표기로는 단위행렬 $I$

```
           순전파                          역전파
      x ──┬──> F ──┐                 g(1 + F') <──┬── g*F' <── F ──┐
          │        (+) ──> y                      │                (+) <── g
          └────────┘                              └────── g ───────┘
```

### 곱셈이 어떻게 바뀌는가

[02-vanishing-gradient.md](02-vanishing-gradient.md)에서 plain 네트워크의 기울기는 layer마다 곱해지는 수 $c_l$의 곱이었습니다.

$$
\text{plain}:\ \prod_{l=1}^{L} c_l
\qquad\qquad
\text{residual}:\ \prod_{l=1}^{L} (1 + c_l)
$$

블록이 3개일 때 펼치면

$$
(1+c_1)(1+c_2)(1+c_3) = 1 + (c_1 + c_2 + c_3) + (c_1c_2 + c_1c_3 + c_2c_3) + c_1c_2c_3
$$

항이 $2^3 = 8$개입니다. 각 항은 어느 블록에서 변환 경로를 탔고 어느 블록에서 shortcut을 탔는가의 한 가지 조합입니다. $1$은 세 블록 모두 shortcut을 탄 경로이고, $c_1c_2c_3$은 세 블록 모두 변환 경로를 탄 경로입니다. plain 네트워크에는 이 마지막 항 하나만 있습니다.

```
       ┌─ f1 ─┐     ┌─ f2 ─┐     ┌─ f3 ─┐
 x ──┬─┘      └─(+)─┬┘      └─(+)─┬┘      └─(+)──> y
     └──────────┘   └──────────┘   └──────────┘

 경로 8개: (지난 블록의 집합)
   {}  {1}  {2}  {3}  {1,2}  {1,3}  {2,3}  {1,2,3}
 길이:  0    1    1    1     2      2      2      3
```

블록 $n$개 중 $k$개를 지나는 경로의 수는 $\binom{n}{k}$이고, 경로 길이는 평균 $n/2$, 표준편차 $\sqrt{n}/2$인 이항분포를 따릅니다. 한편 경로 하나의 기울기는 길이에 대해 지수적으로 작아지므로, 길이 $k$인 경로들의 기울기 기여는 대략

$$
\binom{n}{k} \times c^{k}
$$

이고 최댓값은 $n/2$보다 짧은 쪽에 생깁니다.

### shortcut에 상수를 곱하면

$y = F(x) + \lambda x$처럼 shortcut에 상수를 곱하면 기울기의 첫 항이 1이 아니라 $\lambda$가 되고, $L$개 블록을 지나면 $\lambda^L$입니다. 덧셈 경로가 다시 곱셈 문제로 돌아갑니다.

## 작은 숫자로 직접 계산

모든 블록에서 변환 경로의 기울기 배율이 $c_l = 0.1$이고 블록이 50개라고 하겠습니다.

| | 계산 | 결과 |
|---|---|---|
| plain | $0.1^{50}$ | $10^{-50}$ |
| residual | $1.1^{50}$ | 약 117 |

shortcut에 상수 $\lambda$를 곱했을 때 50블록 뒤 직통 항은 $\lambda = 1$이면 1, $0.9$면 0.005, $0.5$면 $9 \times 10^{-16}$입니다.

$(1 + 0.1)(1 + 0.2)(1 + 0.3) = 1.716$이고, 항 8개를 따로 계산해 더해도 1.716입니다.

| 블록 수 $n$ | 경로 수 $2^n$ | 평균 길이 | 표준편차 |
|---|---|---|---|
| 3 | 8 | 1.5 | 0.87 |
| 16 (ResNet-34) | 65,536 | 8 | 2 |
| 54 (110-layer CIFAR ResNet) | 약 $1.8 \times 10^{16}$ | 27 | 3.7 |

숫자는 [results/hand-calc.txt](results/hand-calc.txt)와 같습니다. 1.1의 50제곱이 117이라는 것은 $c_l$이 크면 residual 쪽이 폭발할 수 있다는 뜻이기도 합니다. 실제로는 블록 안의 BN이 $F$의 출력 크기를 제한하고, $c_l$의 부호가 섞여 있으며, 학습이 진행되면 많은 블록의 $F$가 작아집니다.

## 코드로 확인

`experiments/grad_norms.py`의 실험 2와 3, `experiments/cosine_similarity.py`입니다. 모두 학습 없이 순전파와 역전파를 한 번 합니다.

- 실험 2: BN이 없는 폭 256, 50-layer MLP. 블록은 `x = x + relu(Wx + b)`이고 skip을 끄면 `x = relu(Wx + b)`입니다. 가중치를 표준편차 0.02로 일부러 작게 초기화해 변환 경로의 기울기가 죽는 상황을 만듭니다. 이때 layer당 배율은 $\tfrac{1}{2}\cdot 256 \cdot 0.02^2 \approx 0.05$입니다.
- 실험 3: BN과 He 초기화를 쓴 CIFAR용 CNN([09-plain-vs-residual-experiment.md](09-plain-vs-residual-experiment.md)의 `CifarNet`), 56-layer. skip의 유무만 다릅니다. BN이 배치 통계를 쓰도록 train 모드에서 잽니다.
- 코사인 유사도: 같은 56-layer 두 네트워크에 입력 16개와, 각 입력에 표준편차 0.01의 잡음을 더한 입력 16개를 넣고, 입력에 대한 기울기의 방향이 얼마나 비슷한지 잽니다.

```python
def input_grad(net, x, y):
    x = x.clone().requires_grad_(True)
    loss_fn(net(x), y).backward()
    return x.grad.flatten(1)

net = CifarNet(n=9, use_skip=use_skip).train()
base = torch.randn(16, 3, 32, 32)
noisy = base + 0.01 * torch.randn_like(base)
g1, g2 = input_grad(net, base, label), input_grad(net, noisy, label)
cos = F.cosine_similarity(g1, g2).mean()
```

`.eval()`로 재면 안 됩니다. 학습 전 BN의 running mean은 0, running var는 1이라 eval 모드의 BN은 아무 일도 하지 않고, BN이 없는 네트워크를 재는 셈이 됩니다.

## 실험 결과

PyTorch 2.14, CPU, seed 0. 결과 전체는 [results/grad-norms.txt](results/grad-norms.txt)와 [results/cosine-similarity.txt](results/cosine-similarity.txt)에 있습니다.

### BN 없는 MLP, 가중치 표준편차 0.02

```
      skip 없음          skip 있음
 0    7.041e-35         2.716e-01
10    7.190e-35         8.980e-01
20    7.819e-35         1.267e+00
30    5.473e-35         1.512e+00
40    4.046e-35         2.738e+00
50    1.047e-34 (head)  9.224e+01 (head)
```

- skip 없음: 모든 layer의 기울기가 $10^{-34}$ 안팎입니다. 가중치의 기울기는 그 layer로 들어온 순방향 값과 뒤에서 돌아온 역방향 값의 곱인데, 순방향 신호는 출력 쪽으로 갈수록 사라지고 역방향 신호는 입력 쪽으로 갈수록 사라져서 어느 layer에서나 곱이 비슷하게 작습니다.
- skip 있음: 입력 쪽 0.27에서 마지막 블록 4.5까지, layer 50개에 걸쳐 약 17배 차이로 유지됩니다.

가중치 표준편차를 0.1(layer당 배율 약 1.28)로 올리면 skip 없는 쪽은 모든 layer가 1.6~8.1 범위이고, skip 있는 쪽은 $2.9 \times 10^{9}$에서 $5.6 \times 10^{10}$까지 커집니다. BN이 없으면 블록마다 값이 더해져 커지기 때문입니다([05-batch-normalization.md](05-batch-normalization.md)).

### BN이 있는 CNN, 56-layer

```
      plain-56                          resnet-56
 0    conv1.weight           2.781e+01   conv1.weight           3.124e-01
12    blocks.5.conv2.weight  1.297e+00   blocks.5.conv2.weight  2.197e-02
24    blocks.11.conv2.weight 1.112e-01   blocks.11.conv1.weight 1.008e-02
36    blocks.17.conv2.weight 1.299e-02   blocks.17.conv1.weight 3.705e-03
48    blocks.23.conv2.weight 1.243e-03   blocks.22.conv2.weight 1.237e-03
54    blocks.26.conv2.weight 1.314e-03   blocks.25.conv2.weight 8.207e-04
```

두 네트워크 모두 기울기가 소실되지 않고, 입력 쪽으로 갈수록 오히려 커집니다. 첫 conv와 마지막 conv의 기울기 RMS 비율은 plain-56이 $1.8 \times 10^{4}$, resnet-56이 $3.9 \times 10^{2}$, resnet-20이 47입니다. 입력 쪽 layer는 feature map이 32x32로 넓어서 가중치 하나의 기울기에 더해지는 위치 수가 많다는 점도 이 비율에 들어 있습니다.

### 거의 같은 입력 두 개의 기울기 방향

| seed | plain-56 | resnet-56 |
|---|---|---|
| 0 | 0.001 | 0.430 |
| 1 | 0.013 | 0.411 |
| 2 | 0.023 | 0.435 |

입력을 1%만 바꿨는데 plain-56의 기울기는 방향이 거의 무관해지고, resnet-56은 0.4 정도의 유사도를 유지합니다.

## 결과 해석

skip connection이 역전파 식에서 바꾸는 것은 layer마다 곱해지는 수입니다. $c$ 대신 $1 + c$가 곱해지고, 곱을 펼치면 가중치가 하나도 곱해지지 않은 경로를 포함해 $2^L$개 경로의 합이 됩니다. BN이 없는 MLP 실험은 이 효과를 그대로 보여 줍니다. 변환 경로의 기울기가 $10^{-34}$로 죽은 상황에서도 skip이 있으면 모든 layer에 0.27 이상의 기울기가 도착했습니다.

BN이 있는 CNN에서는 그림이 다릅니다. plain-56의 기울기는 크기로는 소실되지 않았습니다. [01-why-depth-fails.md](01-why-depth-fails.md)에서 본 ResNet 논문의 관찰과 같습니다. 크기가 멀쩡하다면 남는 차이는 기울기의 질이라는 설명이 있습니다. Balduzzi 등(2017)은 깊은 plain 네트워크에서 입력이 조금만 달라도 기울기가 서로 상관없는 백색 잡음에 가까워진다는 것을 보였습니다. 크기는 있지만 방향이 제각각이면 미니배치에서 평균을 내도 쓸모가 없습니다. 위의 코사인 유사도 측정이 초기화 상태에서 이 현상을 보여 줍니다.

Veit 등(2016)은 학습된 110-layer ResNet에서 기울기의 대부분이 블록 5~17개(conv layer로 10~34개) 길이의 경로에서 온다고 측정했습니다. ResNet은 긴 경로의 기울기 소실을 없앤 것이 아니라, 짧은 경로를 충분히 많이 둬서 피해 갑니다. 같은 논문에서 학습된 ResNet의 블록 하나를 지워도 오차가 조금만 늘었다는 결과는 [07-preactivation.md](07-preactivation.md)에서 다룹니다.

## 한계와 주의할 점

- 세 측정 모두 학습 전 초기화 상태에서 무작위 입력으로 한 번 잰 값입니다. 학습 중에 어떻게 변하는지는 재지 않았습니다.
- 코사인 유사도는 seed 세 개의 결과입니다.
- 경로 길이별 기울기 기여의 식은 모든 블록의 $c$가 같다고 둔 단순화입니다. 실제 분포는 Veit 등의 측정을 따랐습니다.

## References

- [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385) (He, Zhang, Ren, Sun, 2015)
- [Identity Mappings in Deep Residual Networks](https://arxiv.org/abs/1603.05027) (He, Zhang, Ren, Sun, 2016)
- [Residual Networks Behave Like Ensembles of Relatively Shallow Networks](https://arxiv.org/abs/1605.06431) (Veit, Wilber, Belongie, 2016)
- [The Shattered Gradients Problem](https://arxiv.org/abs/1702.08591) (Balduzzi 등, 2017)
