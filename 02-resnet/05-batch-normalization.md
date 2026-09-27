# batch normalization은 무엇을 정규화하고, 왜 residual connection과 함께 필요한가?

## 문제

batch normalization(BN)을 두고 흔히 도는 설명이 셋 있습니다. 값을 0부터 1 사이로 압축한다는 설명, layer를 지나며 흔들리는 데이터 분포를 유지해 준다는 설명, $Wx + b$의 $b$가 BN 안으로 들어간다는 설명입니다. 첫째는 틀렸고, 둘째는 방향은 맞지만 입력 이미지의 분포를 유지한다는 뜻이 아니며, 셋째는 맞습니다. BN의 계산을 손으로 해 보고, BN이 왜 동작하는지에 대한 설명이 어떻게 바뀌어 왔는지, 그리고 BN과 shortcut이 서로 무엇을 보완하는지를 봅니다.

## 아이디어

BN은 ResNet보다 먼저 나왔고(Ioffe, Szegedy, 2015) ResNet 블록의 부품입니다. residual block의 변환 경로는 합성곱, BN, 비선형 활성화로 이뤄집니다. BN은 layer의 출력을 미니배치 통계로 평균 0, 분산 1에 맞춘 뒤, 학습되는 배율과 이동으로 다시 조정합니다. 학습 중에도 매 layer에서 값의 크기를 다시 맞추므로 초기화만으로는 유지되지 않던 크기를 계속 관리합니다([02-vanishing-gradient.md](02-vanishing-gradient.md)).

## 수식으로 보기

미니배치(한 번의 갱신에 함께 넣는 $m$개의 샘플)에서 특징 하나에 대한 값 $x_1, \dots, x_m$이 있습니다.

$$
\mu_B = \frac{1}{m}\sum_{i=1}^{m} x_i
\qquad
\sigma_B^2 = \frac{1}{m}\sum_{i=1}^{m} (x_i - \mu_B)^2
\qquad
\hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}
\qquad
y_i = \gamma\,\hat{x}_i + \beta
$$

| 기호 | 뜻 |
|---|---|
| $\mu_B, \sigma_B^2$ | 이 배치에서 이 특징의 평균과 분산. 학습되는 값이 아니라 그때그때 계산하는 통계입니다 |
| $\hat{x}_i$ | 평균을 빼고 표준편차로 나눈 값. 평균 0, 분산 1이 됩니다 |
| $\epsilon$ | 0으로 나누는 것을 막는 작은 수. 보통 $10^{-5}$ |
| $\gamma, \beta$ | 학습되는 배율(scale)과 이동(shift). 초기값은 1과 0 |

$\gamma = \sqrt{\sigma_B^2 + \epsilon},\ \beta = \mu_B$이면 $y = x$입니다. 정규화를 강제하되 필요하면 원래 스케일로 되돌릴 수 있게 한 장치입니다.

### conv layer에서는 채널 단위다

feature map의 모양이 $(N, C, H, W)$ = (배치, 채널, 세로, 가로)일 때 BN은 채널마다 통계를 하나씩 냅니다. 한 채널의 평균은 $N \times H \times W$개 값 전체로 계산합니다. 같은 채널은 같은 필터가 이미지의 모든 위치를 훑어 만든 결과이므로 위치가 달라도 같은 특징으로 취급하는 것입니다. 학습되는 파라미터는 채널당 $\gamma, \beta$ 두 개, 채널 64개짜리 BN이면 128개입니다.

### 학습과 추론에서 다르게 동작한다

| | 학습 시 | 추론 시 |
|---|---|---|
| $\mu, \sigma^2$ | 현재 미니배치에서 계산 | 학습 중 모아 둔 이동 평균(running mean, running var) |
| 배치의 다른 샘플 영향 | 받습니다 | 안 받습니다 |

이동 평균은 $\mu_{run} \leftarrow (1 - \alpha)\,\mu_{run} + \alpha\,\mu_B$로 쌓습니다($\alpha$는 PyTorch 기본값 0.1). 추론 시에는 $\mu, \sigma$가 상수라서 BN 전체가 $y = a x + c$ 꼴의 고정된 1차식이 되고, 배포할 때 앞의 conv 가중치에 합쳐 없앨 수 있습니다(conv-BN fusion).

모델을 서비스에 올릴 때 자주 실수하는 곳입니다. PyTorch에서 `model.eval()`을 호출하지 않고 추론하면 BN이 학습 모드로 동작해서 배치 구성에 따라 출력이 달라집니다. 배치 크기 1로 넣으면 conv 뒤의 BN은 그 이미지 한 장의 $H \times W$ 칸만으로 통계를 내므로 학습 때와 다른 값으로 정규화되고, FC layer 뒤의 BN은 값이 하나뿐이라 분산을 계산할 수 없습니다. `model.eval()`은 BN과 dropout을 추론용으로 바꾸고, `torch.no_grad()`는 기울기 기록을 꺼서 메모리와 시간을 아낍니다. 서로 다른 일을 하므로 둘 다 확인해야 합니다.

### conv 뒤에 BN이 오면 bias가 필요 없다

conv의 출력이 $z_i = w^{\top} x_i + b$일 때 $b$는 같은 채널의 모든 샘플에 똑같이 더해지는 상수입니다. BN이 평균을 빼면

$$
z_i - \mu_B = \big(w^{\top}x_i + b\big) - \big(\overline{w^{\top}x} + b\big) = w^{\top}x_i - \overline{w^{\top}x}
$$

$b$가 정확히 상쇄되고 분산에도 영향이 없습니다. $b$는 학습될 수 없는 파라미터이고 이동 역할은 BN의 $\beta$가 맡습니다. 그래서 ResNet 구현은 모든 conv에 `bias=False`를 줍니다.

### BN이 학습을 돕는 방식

- 가중치에 양수 $a$를 곱해도 BN 출력은 같습니다: $a > 0$일 때 $\text{BN}(aWx) = \text{BN}(Wx)$. 학습 중에 가중치가 커져도 BN을 지나면 출력 크기가 되돌아옵니다.
- ReLU에 들어가는 값이 0 근처에 퍼져 있게 되므로 뉴런이 전부 꺼지거나 전부 켜지는 상황이 줄어듭니다.
- 그래서 더 큰 학습률을 쓸 수 있고 초기화에 덜 민감해집니다.

### BN과 residual connection의 분산

BN이 없으면 변환 경로 $F$의 출력 분산이 입력 분산과 비슷해서, 블록마다 $x + F(x)$로 더할 때 분산이 약 두 배가 됩니다. 블록 $\ell$개를 지나면 $\text{Var}(x_\ell) \approx 2^\ell$로 깊이에 대해 지수적으로 커집니다. BN이 있으면 각 $F$의 출력이 분산 1 근처로 맞춰지므로 $\text{Var}(x_\ell) \approx \ell$로 선형으로만 커집니다. De와 Smith(2020)는 그 결과 깊은 쪽 블록에서 residual branch가 shortcut에 비해 깊이의 제곱근에 반비례하는 크기로 작아지고, 깊은 ResNet이 초기에 항등 함수에 가깝게 동작한다고 분석했습니다. 이 분석을 따라 BN 없이 ResNet을 학습시키는 방법들(SkipInit, Fixup, NFNet)은 residual branch 끝에 0으로 초기화한 스칼라를 곱합니다. zero-init residual과 같은 생각입니다.

## 작은 숫자로 직접 계산

배치 크기 4, 어떤 특징의 값이 $x = [2, 4, 6, 8]$입니다.

1. 평균: $\mu_B = (2+4+6+8)/4 = 5$
2. 분산: $\sigma_B^2 = \big((-3)^2 + (-1)^2 + 1^2 + 3^2\big)/4 = 5$
3. 표준편차: $\sqrt{5} \approx 2.236$
4. 정규화: $\hat{x} = [-3, -1, 1, 3]/2.236 = [-1.342,\ -0.447,\ 0.447,\ 1.342]$
5. $\gamma = 2,\ \beta = 1$이라면: $y = [-1.683,\ 0.106,\ 1.894,\ 3.683]$

- $\hat{x}$에는 음수도 있고 1보다 큰 값도 있습니다. BN은 값을 0~1 구간으로 누르지 않습니다. 0~1로 누르는 것은 sigmoid나 min-max 정규화입니다.
- 값들의 순서와 간격 비율은 그대로입니다. 평행이동하고 늘이거나 줄이는 아핀 변환이기 때문입니다.
- 같은 $x$에 bias $b = 100$을 더해 $[102, 104, 106, 108]$로 만들어도 평균은 105, 분산은 5이고 $\hat{x}$는 같습니다.

## 코드로 확인

`experiments/param_count.py`는 위 손계산을 `nn.BatchNorm1d`로 다시 하고, bias가 있는 conv와 없는 conv에 같은 가중치를 넣은 뒤 각각 `nn.BatchNorm2d`(train 모드)를 통과시켜 출력을 비교합니다.

```python
c1 = nn.Conv2d(64, 64, 3, padding=1, bias=False)
c2 = nn.Conv2d(64, 64, 3, padding=1, bias=True)
with torch.no_grad():
    c2.weight.copy_(c1.weight)
    nn.init.normal_(c2.bias, std=3.0)        # bias를 크게
bn = nn.BatchNorm2d(64).train()
x = torch.randn(8, 64, 16, 16)
print(torch.allclose(bn(c1(x)), bn(c2(x)), atol=1e-4))
```

## 실험 결과

[results/param-count.txt](results/param-count.txt)의 출력입니다.

- `nn.BatchNorm1d`($\gamma = 2, \beta = 1$)의 출력: $[-1.683,\ 0.106,\ 1.894,\ 3.683]$. 손계산과 같습니다.
- $x + 100$의 $\hat{x}$: $[-1.342,\ -0.447,\ 0.447,\ 1.342]$. bias를 더하기 전과 같습니다.
- bias 없는 conv + BN과 bias 있는 conv + BN의 출력: 같습니다(True).

BN이 없는 residual MLP에서 값이 블록마다 커지는 것은 [04-skip-connection-gradient.md](04-skip-connection-gradient.md)의 측정에서 봤습니다. 가중치 표준편차 0.1에서 skip이 없으면 기울기 RMS가 1.6~8.1인데, skip이 있으면 $2.9 \times 10^{9}$에서 $5.6 \times 10^{10}$까지 커졌습니다.

## 결과 해석

BN이 정규화하는 것은 입력 이미지가 아니라 각 layer의 각 채널에 들어오는 값의 미니배치 통계입니다. 고양이 이미지와 개 이미지는 같은 배치 안에서도 서로 다른 $\hat{x}$를 갖습니다. 손계산에서 네 값이 서로 다른 채로 남는 것과 같습니다. BN은 샘플 사이의 차이를 지우지 않고 배치 전체의 중심과 폭만 맞춥니다.

BN이 왜 동작하는지에 대한 설명은 바뀌어 왔습니다. BN 논문은 앞쪽 layer의 가중치가 갱신되면 뒤쪽 layer가 받는 입력의 분포가 계속 바뀌는 현상을 internal covariate shift(ICS)라고 부르고, BN이 이것을 줄여서 학습이 빨라진다고 설명했습니다. Santurkar 등(2018)은 두 실험으로 이 설명을 반박했습니다. BN 뒤에 매 단계 바뀌는 무작위 잡음을 넣어 ICS를 일부러 키워도 BN 네트워크는 BN이 없는 네트워크보다 훨씬 잘 학습됐고, ICS를 기울기의 변화로 정량화해 재 보니 BN이 ICS를 줄이지 않았습니다. 저자들의 설명은 BN이 손실과 기울기가 변하는 속도의 상한(립시츠 상수)을 줄여 손실 지형을 매끄럽게 만들고, 그래서 큰 학습률로도 발산하지 않는다는 것입니다. 이 밖에도 가중치의 크기와 방향을 분리한다는 설명, 미니배치마다 통계가 달라지는 잡음이 regularization(과적합을 누르는 효과)을 준다는 설명이 함께 쓰입니다. BN은 2015년에 나오자마자 거의 모든 CNN에 들어갔고, 그 설명이 반박된 것은 3년 뒤였습니다.

residual connection과 함께 필요한 이유는 둘이 하는 일이 다르기 때문입니다. ResNet 논문의 plain 네트워크는 BN을 썼는데도 degradation이 생겼으므로 BN만으로는 부족했습니다. 반대로 BN 없이 shortcut만 두면 덧셈이 쌓이면서 값의 분산이 깊이에 대해 지수적으로 커집니다. shortcut은 기울기가 지나갈 덧셈 경로를 주고, BN은 그 덧셈이 쌓여도 값이 지수적으로 커지지 않게 합니다.

## 한계와 주의할 점

| BN의 약점 | 설명 |
|---|---|
| 작은 배치에 약합니다 | 배치가 2~4개면 통계 추정이 부정확합니다. 이미지가 커서 배치를 작게 잡는 detection, segmentation에서 문제가 됩니다 |
| 샘플 사이의 의존 | 한 샘플의 출력이 같은 배치의 다른 샘플에 영향을 받습니다 |
| 시퀀스 모델에 맞지 않습니다 | 길이가 다른 시퀀스에서 시점별 통계를 내기 어렵습니다 |

그래서 Layer Normalization(샘플 하나 안에서 정규화)이 나왔고 Transformer는 이것을 씁니다. 평균을 내는 축만 다르고 식의 모양은 같습니다.

```
입력 모양 (N, C, H, W)에서 평균을 내는 범위
  BatchNorm : N, H, W 방향  (채널마다 통계 1개)
  LayerNorm : C, H, W 방향  (샘플마다 통계 1개)
  GroupNorm : 채널 그룹, H, W 방향
```

BN이 왜 동작하는지에 대한 합의된 단일 설명은 아직 없습니다. 위의 설명들은 이 저장소에서 실험으로 확인하지 않았고, 인용한 논문의 결과입니다.

## References

- [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/abs/1502.03167) (Ioffe, Szegedy, 2015)
- [How Does Batch Normalization Help Optimization?](https://arxiv.org/abs/1805.11604) (Santurkar, Tsipras, Ilyas, Madry, 2018)
- [Batch Normalization Biases Residual Blocks Towards the Identity Function in Deep Networks](https://arxiv.org/abs/2002.10444) (De, Smith, 2020)
- [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385) (He, Zhang, Ren, Sun, 2015)
