# 얕은 네트워크와 깊은 네트워크에서 skip connection의 효과는 어떻게 다른가?

## 문제

ResNet을 배우고 나면 8-layer짜리 AlexNet 같은 네트워크에 skip connection을 붙여 보고 싶어집니다. 결과는 대개 붙이기 전과 거의 같습니다. CIFAR-10에서 plain 네트워크와 residual 네트워크를 깊이별로 비교하는 실험을 설계하고, CPU에서 돌릴 수 있는 크기로 줄여 실행한 뒤, 얕은 네트워크에서 효과가 없는 것이 왜 정상인지를 봅니다.

## 아이디어

residual connection은 깊어서 생기는 문제의 해법입니다. ResNet 논문의 ImageNet 비교에서 18-layer의 top-1 오차는 plain 27.94%, ResNet 27.88%로 차이가 0.06%p였고, 차이는 34-layer에서 벌어졌습니다(plain 28.54%, ResNet 25.03%). 논문은 18-layer에서 ResNet이 더 빨리 수렴한다는 이점만 보고했습니다.

실험 설계는 2x2입니다.

| | 20-layer | 56-layer |
|---|---|---|
| plain | A | B |
| residual | C | D |

1. B가 A보다 학습 오차가 높은가 (degradation)
2. D가 C보다 낮은가 (residual이면 깊이가 이득이 된다)
3. A와 C는 비슷한가 (얕으면 skip의 효과가 작다)

degradation은 학습 오차에서 드러나므로 테스트 오차가 아니라 학습 오차를 봅니다. 테스트 오차만 보면 과적합과 구분할 수 없습니다.

## 수식으로 보기

### CIFAR-10용 구조

CIFAR-10은 32x32 컬러 이미지 60,000장(학습 50,000장, 테스트 10,000장)을 10개로 분류하는 데이터셋입니다. ResNet 논문의 CIFAR용 구조는 ImageNet용과 달리 7x7 conv와 maxpool이 없습니다.

```
입력 3 x 32 x 32
  ├─ 3x3 conv, 16
  ├─ stage 1: 블록 n개, 16채널, 32x32
  ├─ stage 2: 블록 n개, 32채널, 16x16  (첫 블록 stride 2)
  ├─ stage 3: 블록 n개, 64채널, 8x8    (첫 블록 stride 2)
  ├─ global average pooling
  └─ FC 10
```

블록 하나에 conv 두 장, stage 세 개이므로 layer 수는 $6n + 2$입니다. $n = 1, 3, 9, 18$이면 8, 20, 56, 110-layer입니다.

### 작은 표본의 측정 오차

정확도 $p$인 모델이 테스트 이미지 $n$장에서 맞히는 개수의 표준편차는 $\sqrt{n\,p\,(1-p)}$입니다.

## 작은 숫자로 직접 계산

### 8-layer에서는 곱셈이 아직 무너지지 않는다

layer마다 기울기에 0.9가 곱해지면 8-layer 뒤에는 0.43, 56-layer 뒤에는 0.0027, 152-layer 뒤에는 $1.1 \times 10^{-7}$입니다([02-vanishing-gradient.md](02-vanishing-gradient.md)). 8-layer에서 기울기는 절반쯤 남고, 학습률이 그 정도는 흡수합니다.

### 12장으로 비교하면

테스트 이미지 12장을 뽑아 한쪽은 4장, 다른 쪽은 5장을 맞혔다고 하겠습니다. 정확도 33%인 모델이 12장에서 맞히는 개수의 표준편차는

$$
\sqrt{12 \times 0.33 \times 0.67} \approx 1.6
$$

장입니다. 같은 모델로 다른 12장을 뽑으면 2장에서 6장 사이가 쉽게 나오므로, 4장 대 5장은 차이가 있다는 증거도 없다는 증거도 아닙니다. 테스트셋 10,000장 전체로 재면 표준편차가 약 0.47%p로 줄어듭니다([results/hand-calc.txt](results/hand-calc.txt)).

## 코드로 확인

`experiments/resnet_cifar.py`는 `--n`으로 깊이를, `--plain`으로 skip의 유무를 정하고, 나머지 설정은 모두 같게 둡니다. 블록은 `use_skip` 하나만 다릅니다.

```python
class Block(nn.Module):
    """use_skip=False 이면 plain, True 이면 residual. 그 밖의 모든 것은 같다."""

    def __init__(self, in_planes, planes, stride, use_skip):
        super().__init__()
        self.use_skip = use_skip
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Identity()
        if use_skip and (stride != 1 or in_planes != planes):
            self.shortcut = nn.Sequential(nn.Conv2d(in_planes, planes, 1, stride, bias=False),
                                          nn.BatchNorm2d(planes))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.use_skip:
            out = out + self.shortcut(x)
        return F.relu(out)
```

학습 설정은 SGD(학습률 0.1, momentum 0.9, weight decay $10^{-4}$), 배치 128, 전체 epoch의 50%와 75% 지점에서 학습률을 10분의 1로 줄이는 스케줄, random crop과 좌우 반전 augmentation입니다. 모든 conv는 He 초기화이고 BN을 씁니다. epoch는 학습 데이터 전체를 한 번 본 것을 세는 단위이고, iteration은 미니배치 하나로 가중치를 한 번 갱신한 것입니다.

```bash
python3 resnet_cifar.py --n 3 --plain     # A: plain-20
python3 resnet_cifar.py --n 9 --plain     # B: plain-56
python3 resnet_cifar.py --n 3             # C: resnet-20
python3 resnet_cifar.py --n 9             # D: resnet-56
```

이 코드와 원 논문의 차이입니다.

| | 원 논문 | 이 코드 |
|---|---|---|
| 모양이 바뀌는 shortcut | 늘어난 채널을 0으로 채우기 (파라미터 없음) | 1x1 conv 사영 |
| 학습 길이 | 64,000 iteration. 배치 128, 50,000장이면 약 164 epoch | 기본 30 epoch |
| 검증 분할 | 45k/5k 분할은 학습을 끝낼 시점(64,000 iteration)을 정하는 데만 썼습니다 | 분할 없이 학습 |

## 실험 결과

전체 실험은 CPU로 현실적이지 않아 학습 이미지와 epoch를 줄여 코드가 끝까지 도는지 확인했습니다. PyTorch 2.14, CPU, seed 0이고, 테스트는 10,000장 전체로 쟀습니다([results/cifar-small.txt](results/cifar-small.txt)).

| 모델 | 학습 이미지, epoch | 파라미터 | 학습 오차 (epoch별) | 테스트 오차 (epoch별) |
|---|---|---|---|---|
| plain-8 | 5,000장, 3 | 75,290 | 79.36 → 71.74 → 67.68% | 83.07 → 68.92 → 69.19% |
| resnet-8 | 5,000장, 3 | 78,042 | 77.78 → 71.82 → 67.32% | 79.80 → 70.86 → 68.69% |
| plain-20 | 5,000장, 3 | 269,722 | 81.68 → 75.02 → 70.58% | 83.59 → 70.42 → 68.89% |
| resnet-20 | 5,000장, 3 | 272,474 | 81.46 → 68.80 → 63.64% | 84.34 → 66.21 → 63.12% |
| plain-56 | 2,000장, 2 | 853,018 | 90.60 → 89.15% | 89.52 → 89.44% |
| resnet-56 | 2,000장, 2 | 855,770 | 89.85 → 88.80% | 90.00 → 85.92% |

모델 하나에 CPU로 2~4분이 걸렸습니다. 56-layer는 더 오래 걸려서 학습 이미지 2,000장, 2 epoch로 더 줄였습니다.

파라미터 수는 plain-20이 269,722개, resnet-20이 272,474개입니다. 차이 2,752개는 projection shortcut 두 개의 1x1 conv(512 + 2,048)와 그 뒤 BN(64 + 128)입니다. 56-layer는 853,018개와 855,770개입니다.

## 결과 해석

축소 실행에서 8-layer는 plain과 residual의 마지막 학습 오차가 67.68%와 67.32%로 거의 같았습니다. 20-layer에서는 resnet-20이 3 epoch 뒤 63.64%, plain-20이 70.58%로 차이가 났습니다. 56-layer 두 모델은 2 epoch 뒤에도 학습 오차가 89~90%로 무작위 추측(90%)에 가까워서 비교할 수 없습니다. 이 방향은 논문과 같지만 seed 하나, 5,000장, 3 epoch의 결과라서 차이가 구조 때문인지 흔들림인지 이 실험만으로는 구분할 수 없습니다. 이 실험이 보여 주는 것은 두 가지입니다. 코드가 끝까지 돌고 오차가 내려간다는 것, 그리고 이 규모의 실행으로는 degradation을 보이지 못한다는 것입니다.

8-layer 네트워크에서 skip connection의 효과가 작은 것은 실험을 잘못해서가 아니라 정상입니다. 이유는 넷입니다.

1. 8-layer에서는 기울기의 곱이 아직 무너지지 않습니다. 풀어야 할 degradation이 없습니다.
2. ResNet 논문에서도 18-layer에서는 plain과 차이가 거의 없었습니다. residual connection은 layer를 더하는 것을 안전하게 만드는 장치라서, layer를 더하지 않으면 할 일이 없습니다.
3. AlexNet 같은 구조는 skip을 붙일 자리가 별로 없습니다. $F(x) + x$를 하려면 두 텐서의 모양이 같아야 하는데, AlexNet은 layer마다 모양이 달라서 항등 shortcut을 그대로 붙일 수 있는 곳이 conv3과 conv4 사이, fc6과 fc7 사이 정도입니다. 나머지는 1x1 conv 사영이 필요하고, 사영 shortcut은 순수한 항등이 아닙니다. ResNet이 효과를 보는 것은 같은 모양의 블록을 수십 개 반복하기 때문입니다.
4. AlexNet의 병목은 다른 곳에 있습니다. 파라미터의 대부분이 FC layer에 있고, CIFAR-10(32x32)에 그대로 쓰면 11x11, stride 4인 첫 layer가 이미지를 첫 단계에서 크게 줄입니다. BN을 넣거나 첫 layer를 3x3으로 바꾸는 쪽이 skip을 붙이는 것보다 영향이 클 것으로 예상하지만, 이 저장소에서 재지는 않았습니다.

layer를 더 올리면 계속 좋아지는지도 자주 묻습니다. ResNet 논문의 ImageNet 실험에서는 18, 34, 50, 101, 152-layer로 갈수록 오차가 줄었지만 CIFAR-10의 1,202-layer는 110-layer보다 나빴고(7.93% 대 6.43%), 저자들은 이것을 과적합으로 봤습니다. 1,202-layer의 파라미터는 1,940만 개로, 5만 장짜리 데이터에 비해 큽니다. 원래 설계를 ImageNet에서 200-layer로 늘렸을 때 나빠진 결과는 ResNet v2 논문에 있습니다([07-preactivation.md](07-preactivation.md)).

## 한계와 주의할 점

- 축소 실행의 결과는 코드가 동작하고 오차가 내려간다는 것만 보여 줍니다. 이 길이로는 plain과 residual의 차이나 깊이의 효과를 말할 수 없고, 논문의 수치를 재현한 것이 아닙니다.
- seed 하나의 결과입니다. CIFAR-10에서는 같은 설정이라도 seed에 따라 테스트 오차가 흔들리므로, 1%p 미만의 차이는 여러 번 돌려야 말할 수 있습니다.
- 제대로 비교하려면 테스트셋 10,000장 전체로 재고, 학습 오차와 테스트 오차를 둘 다 기록하고, seed를 바꿔 세 번 이상 돌리고, 깊이를 변수로 둬야 합니다. `--n 1, 3, 9`로 8, 20, 56-layer를 모두 돌리면 됩니다. GPU에서 30 epoch 이상을 권합니다.
- 원 논문의 CIFAR-10 테스트 오차는 ResNet 20, 32, 44, 56, 110-layer가 8.75%, 7.51%, 7.17%, 6.97%, 6.43%(다섯 번 중 가장 좋은 값, 평균 6.61%)이고 1,202-layer가 7.93%입니다. 이 저장소의 축소 실행과는 학습 길이와 데이터 양이 달라서 비교할 수 없습니다.

## References

- [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385) (He, Zhang, Ren, Sun, 2015)
- [Identity Mappings in Deep Residual Networks](https://arxiv.org/abs/1603.05027) (He, Zhang, Ren, Sun, 2016)
