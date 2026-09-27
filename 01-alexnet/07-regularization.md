# 파라미터 약 6,100만 개짜리 모델이 학습 이미지 120만 장을 외우지 않도록 AlexNet은 무엇을 했는가?

## 문제

AlexNet은 파라미터 수가 학습 이미지 수의 약 50배입니다. 논문은 dropout 없이 학습하면 상당한 과적합이 생긴다고 적었습니다. 여기서는 dropout과 data augmentation이 각각 무엇을 제약하는지 식과 숫자로 정리하고, AlexNet을 32x32 입력에 맞게 줄인 CNN을 CIFAR-10 일부로 학습시켜 두 기법을 끄고 켰을 때의 차이를 봅니다.

## 아이디어

### 과적합(overfitting)

모델이 학습 데이터의 일반적인 규칙 대신 개별 사례를 외운 상태입니다. 학습 데이터에서는 잘 맞고 처음 보는 데이터에서는 틀립니다.

```
손실
 |\ \
 | \ \__________----''''   검증 손실: 어느 순간부터 다시 올라감
 |  \
 |   \____
 |        ''----.______    학습 손실: 계속 내려감
 +------------------------ epoch
             ^ 여기서부터 과적합
```

여기서 regularization(정규화)은 학습 중에 제약을 걸어 과적합을 막고 일반화 능력을 지키는 기법의 총칭입니다. AlexNet은 모델 쪽에 거는 dropout과 데이터 쪽에 거는 augmentation 두 가지를 썼습니다.

## 수식으로 보기

### dropout

#### 동작

학습의 매 iteration마다 뉴런의 일부를 무작위로 끕니다. AlexNet은 FC6과 FC7에서 50%를 껐습니다. 파라미터의 대부분이 FC layer에 있고 거기가 가장 과적합하기 쉽기 때문입니다.

식으로 쓰면, 활성화 벡터 $\mathbf{a}$ 에 0과 1로 된 무작위 마스크 $\mathbf{m}$ 을 곱합니다.

$$
\tilde{\mathbf{a}} = \mathbf{m} \odot \mathbf{a}, \qquad m_i \sim \text{Bernoulli}(1 - p)
$$

$p$ 는 끌 확률입니다. $m_i$ 는 확률 $1-p$ 로 1, 확률 $p$ 로 0입니다.

#### 왜 효과가 있는가

dropout은 뉴런들이 서로에게 의존하는 것(co-adaptation)을 막습니다. 어떤 뉴런 A가 뉴런 B의 출력에 맞춰 그 오차만 보정하도록 학습하면 A와 B는 학습 데이터에만 맞는 조합이 됩니다. B가 절반의 확률로 없어지면 A는 B 없이도 쓸모 있어야 합니다.

#### 학습과 추론의 크기 맞추기

추론할 때는 아무것도 끄지 않습니다. 그러면 다음 layer가 받는 값의 합이 학습 때의 약 2배가 됩니다. 이를 맞추는 방법이 둘입니다.

| 방식 | 학습 때 | 추론 때 |
|---|---|---|
| 초기 구현 (AlexNet) | 그냥 끕니다 | 출력에 $(1-p)$ 를 곱해 줄입니다 |
| inverted dropout (현재 표준) | 끄고, 남은 값을 $\dfrac{1}{1-p}$ 배 합니다 | 아무것도 안 합니다 |

$\mathbf{a} = (2, 4, 6, 8)$, $p = 0.5$, 마스크 $\mathbf{m} = (1, 0, 0, 1)$.
- 끈 결과: $(2, 0, 0, 8)$
- inverted: $\times 2$ 해서 $(4, 0, 0, 16)$
- 각 칸의 기댓값은 $0.5 \times 2a_i + 0.5 \times 0 = a_i$ 로 원래 값과 같습니다.

같은 계산을 [experiments/hand_calc.py](experiments/hand_calc.py)가 하고, 출력은 [results/hand-calc.txt](results/hand-calc.txt)에 있습니다.

두 방식 모두 평균 신호를 같게 유지합니다. inverted 방식은 추론 코드에 dropout 처리가 필요 없어서 지금의 표준이 됐습니다.

```python
import numpy as np

def dropout(a, p, training):
    if not training:
        return a
    mask = (np.random.rand(*a.shape) > p) / (1 - p)
    return a * mask
```

#### 비용

AlexNet 논문은 dropout을 넣으면 수렴에 필요한 iteration 수가 대략 2배가 된다고 적었습니다.

#### 앙상블로 보는 해석

마스크가 매번 다르므로 매 iteration마다 다른 부분 네트워크를 학습하는 셈입니다. 뉴런이 $n$개면 가능한 부분 네트워크는 $2^n$개이고 전부 가중치를 공유합니다. 추론 때 전체를 켜고 스케일을 맞추는 것은 이 많은 네트워크의 예측을 평균 내는 것의 근사입니다.

### data augmentation

학습 데이터에 라벨을 바꾸지 않는 변형을 가해 데이터의 실질적인 크기와 다양성을 늘립니다.

| 종류 | 방법 | 가르치는 것 |
|---|---|---|
| 기하 변환 | 무작위 crop, 좌우 반전 | 물체가 다른 위치나 방향에 있어도 같은 물체입니다 |
| 광도 왜곡 | PCA 기반 색 흔들기 | 조명이 달라도 같은 물체입니다 |

#### 구체적인 숫자

- 원본을 256x256으로 맞춘 뒤 224x224 조각을 무작위로 잘라 씁니다. 원 논문은 이것으로 학습 세트가 2,048배가 된다고 적습니다. $32 \times 32 \times 2$(가로 위치, 세로 위치, 좌우 반전)로 센 값입니다.
- 추론 때는 네 귀퉁이와 중앙의 5개 조각과 그 좌우 반전까지 10개 조각의 softmax 출력을 평균 냅니다.

#### PCA 색 흔들기

학습 세트 전체의 RGB 픽셀값에 주성분 분석(PCA)을 해서, 색이 주로 어느 방향으로 변하는지를 나타내는 세 축(고유벡터 $\mathbf{p}_1, \mathbf{p}_2, \mathbf{p}_3$)과 각 축의 분산(고유값 $\lambda_1, \lambda_2, \lambda_3$)을 구합니다. 이미지마다 모든 픽셀에 다음을 더합니다.

$$
\Delta = [\mathbf{p}_1, \mathbf{p}_2, \mathbf{p}_3]\,[\alpha_1\lambda_1,\ \alpha_2\lambda_2,\ \alpha_3\lambda_3]^{\top}, \qquad \alpha_i \sim \mathcal{N}(0,\ 0.1^2)
$$

$\alpha_i$ 는 평균 0, 표준편차 0.1의 가우시안에서 뽑은 난수입니다. 자연 이미지에서 색이 실제로 변하는 방향(주로 전체 밝기)으로만 흔듭니다. R, G, B를 각각 아무렇게나 흔들면 현실에 없는 색이 나옵니다. 원 논문은 이 기법으로 top-1 오류가 1%p 넘게 줄었다고 보고합니다.

#### 사전 지식을 넣는 위치

augmentation은 사람이 아는 불변성을 데이터의 형태로 주입하는 것입니다. [표현 학습](01-representation-learning.md)에서 SIFT는 크기와 회전 불변성을 feature 설계에 하드코딩했습니다. augmentation은 같은 지식을 "이렇게 변해도 정답은 같다"는 예제로 보여 주고, 불변성을 어떻게 구현할지는 네트워크가 정하게 둡니다. 사전 지식을 넣는 위치가 코드에서 데이터로 옮겨 갔습니다.

## 코드로 확인

AlexNet의 구성 요소를 하나씩 끄고 켜면서 학습 결과를 비교하는 실험입니다. 스크립트는 [experiments/small_cnn.py](experiments/small_cnn.py)입니다.

### 모델: AlexNet을 32x32 입력에 맞게 줄인 것

```python
import torch
import torch.nn as nn

class SmallAlexNet(nn.Module):
    def __init__(self, act="relu", p_drop=0.5):
        super().__init__()
        A = nn.ReLU if act == "relu" else nn.Tanh
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), A(), nn.MaxPool2d(2),      # 32 -> 16
            nn.Conv2d(64, 128, 3, padding=1), A(), nn.MaxPool2d(2),    # 16 -> 8
            nn.Conv2d(128, 256, 3, padding=1), A(),
            nn.Conv2d(256, 256, 3, padding=1), A(),
            nn.Conv2d(256, 128, 3, padding=1), A(), nn.MaxPool2d(2),   # 8 -> 4
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),                       # 128 * 4 * 4 = 2048
            nn.Dropout(p_drop), nn.Linear(2048, 512), A(),
            nn.Dropout(p_drop), nn.Linear(512, 512), A(),
            nn.Linear(512, 10),                 # softmax는 손실 함수가 처리합니다
        )

    def forward(self, x):
        return self.classifier(self.features(x))
```

conv layer 5개와 FC layer 3개, 채널이 늘었다 줄어드는 모양, ReLU, FC의 dropout 0.5는 AlexNet과 같습니다. 입력이 32x32라서 첫 layer의 큰 필터와 stride 4를 쓰지 않습니다. LRN과 두 GPU 분할은 뺐습니다. 파라미터는 2,572,810개입니다(`python3 small_cnn.py --params`).

### 데이터와 학습 설정

- 데이터: CIFAR-10(32x32 컬러 이미지, 10개 클래스)의 학습 이미지 앞 10,000장과 시험 이미지 앞 2,000장.
- augmentation: `RandomCrop(32, padding=4)` 와 `RandomHorizontalFlip()`. AlexNet이 쓴 기하 변환에 해당하고 PCA 색 흔들기는 뺐습니다. 학습 데이터에만 겁니다.
- 학습: SGD, 학습률 0.01, batch 128, weight decay 0.0005, 20 epoch, seed 0. 학습률 스케줄은 쓰지 않았습니다.
- 매 epoch 끝에 `model.eval()` 로 dropout을 끄고 시험 정확도를 잽니다. `model.eval()` 을 빼면 평가할 때도 뉴런이 무작위로 꺼져 정확도가 낮게 나오는데, 에러가 나지 않아 찾기 어렵습니다.
- 설정 7개: 기준(ReLU, 기본 초기화), ReLU + He 초기화, tanh, 그리고 He 초기화 위에서 dropout 없음, augmentation 없음, 둘 다 없음, momentum 0.

첫 epoch의 손실은 $\ln 10 \approx 2.30$ 근처에서 시작해야 합니다([신경망과 손실](02-neural-network-and-loss.md)). 크게 다르면 구현이 틀린 것입니다.

## 실험 결과

CIFAR-10 학습 이미지 10,000장, 시험 이미지 2,000장, 20 epoch, seed 0, CPU에서 돌린 축소판입니다. 모델과 설정은 [regularization](07-regularization.md)의 「코드로 확인」에 있습니다. 결과는 [results/small-cnn.txt](results/small-cnn.txt)에 있습니다. 네 설정 모두 ReLU와 He 초기화, momentum 0.9를 씁니다.

| 설정 | train loss | train 정확도 | 시험 정확도 |
|---|---:|---:|---:|
| dropout과 augmentation 둘 다 씀 | 1.423 | 48.1% | 52.9% |
| dropout 없음 | 0.936 | 66.7% | 65.5% |
| augmentation 없음 | 0.900 | 67.2% | 58.9% |
| 둘 다 없음 | 0.084 | 97.2% | 60.5% |

train 정확도는 dropout과 augmentation이 켜진 학습 중에 잰 값이고, 시험 정확도는 dropout을 끄고 잰 값입니다.

둘 다 없는 설정만 train 97.2%, 시험 60.5%로 과적합 모양이 뚜렷했습니다. 시험 정확도는 epoch 15의 60.6%에서 더 오르지 않았습니다. 나머지 세 설정은 train과 시험 정확도의 차이가 작았고, 20 epoch 뒤에도 train loss가 계속 내려가는 중이었습니다.


## 결과 해석

이 조건에서는 dropout이나 augmentation을 끈 쪽이 시험 정확도가 더 높았습니다. 20 epoch은 두 기법을 켠 모델이 학습 데이터를 다 맞추기에도 짧았고(train 48.1%), 그래서 두 기법이 과적합을 줄이기보다 학습을 늦추는 쪽으로 작용한 것으로 읽힙니다. AlexNet은 120만 장을 약 90 epoch 학습했고, 논문은 dropout이 없으면 과적합이 심했다고 보고했습니다. 이 실험의 조건은 그 상황과 다르므로 논문의 주장을 확인한 것도, 반박한 것도 아닙니다.

### 두 기법 비교

AlexNet은 파라미터가 데이터보다 훨씬 많은데도 두 기법을 같이 써서 처음 보는 이미지에 일반화했습니다.

| | dropout | augmentation |
|---|---|---|
| 제약을 거는 곳 | 모델 내부 | 입력 데이터 |
| 방식 | 뉴런을 무작위로 끕니다 | 입력을 무작위로 변형합니다 |
| 비용 | iteration 약 2배 | CPU에서 변형 계산 |
| 추론 때 | 쓰지 않습니다(모든 뉴런을 켭니다) | 10-crop 평균(선택) |

원 논문은 augmentation을 CPU의 Python 코드로 수행했고 GPU가 앞 batch를 학습하는 동안 다음 batch를 변형했으므로 계산 비용이 사실상 없었다고 적습니다. 생산자와 소비자를 나눈 파이프라인입니다.

## 한계와 주의할 점

- seed가 하나이고 시험 이미지가 2,000장이라 1~2%p 차이는 흔들림과 구분되지 않습니다.
- epoch을 늘리면 두 기법을 켠 설정이 앞설 수도 있습니다. 이 실험은 20 epoch에서 멈췄습니다.
- 학습률 스케줄을 쓰지 않았습니다.

## 더 해 볼 것

1. 학습이 끝난 모델의 첫 conv layer 가중치 `model.features[0].weight` (64 x 3 x 3 x 3)를 그림으로 그려, [합성곱](03-convolution.md)에서 사람이 정했던 에지 커널과 비슷한 것이 생겼는지 봅니다.
2. `nn.Conv2d(64, 128, 3, padding=1)` 을 `groups=2` 로 바꿔 파라미터 수가 어떻게 변하는지 봅니다. AlexNet의 두 GPU 분할이 이 옵션입니다([AlexNet의 파라미터와 두 GPU 분할](09-alexnet-parameters-and-gpus.md)).
3. 가중치를 전부 0.01 상수로 초기화하고 학습이 되는지 봅니다(대칭 깨기, [ReLU와 초기화](05-relu-and-initialization.md)).
4. 활성화를 sigmoid로 바꾸고 layer를 몇 개 더 쌓아, 첫 layer와 마지막 layer의 기울기 크기(`p.grad.norm()`)를 비교합니다. 기울기 소실을 숫자로 봅니다.
5. conv layer를 하나 뺀 모델과 비교합니다. AlexNet 논문은 layer 하나를 빼면 정확도가 떨어진다고 보고했습니다.
6. layer를 20개로 늘려 학습이 되는지 본 뒤 잔차 연결을 넣어 다시 봅니다. [ResNet](../02-resnet/README.md)에서 다루는 내용입니다.

## References

- [Dropout: A Simple Way to Prevent Neural Networks from Overfitting](https://jmlr.org/papers/v15/srivastava14a.html) (Srivastava 외, 2014)
- [ImageNet Classification with Deep Convolutional Neural Networks](https://papers.nips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html) (Krizhevsky, Sutskever, Hinton, 2012)
