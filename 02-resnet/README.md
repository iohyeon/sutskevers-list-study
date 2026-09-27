# ResNet

layer를 깊게 쌓으면 왜 학습 오차까지 커졌는지, $y = F(x) + x$라는 덧셈 하나가 순방향과 역방향에서 무엇을 바꿨는지, 그리고 bottleneck, batch normalization, pre-activation, ResNeXt가 그 위에서 무엇을 더했는지를 다룹니다. 기울기가 곱해지는 구조를 손으로 계산하고, 블록을 PyTorch로 구현해 파라미터 수를 손계산과 맞추고, 학습 없이 역전파를 한 번 해서 plain 네트워크와 residual 네트워크의 layer별 기울기를 쟀습니다. 『Sutskever's List』 3장을 읽으면서 공부한 내용입니다.

## 문서

| 문서 | 질문 | 계산하거나 보여 주는 것 |
|---|---|---|
| [01-why-depth-fails.md](01-why-depth-fails.md) | layer를 더 쌓았는데 왜 학습 오차까지 커지는가? | 수용 영역 공식과 손계산(3x3 세 장 = 7, dilation 1~16이면 63), 과적합, 기울기 소실, degradation의 구분, 포함 관계 논증, AlexNet부터 GoogLeNet까지의 비교 |
| [02-vanishing-gradient.md](02-vanishing-gradient.md) | 기울기는 왜 layer를 지날수록 작아지거나 커지고, 초기화는 그것을 어디까지 막는가? | $c^L$ 표($0.9^{152} \approx 1.1 \times 10^{-7}$), Xavier와 He 초기화의 유도, 50-layer MLP에서 sigmoid, ReLU, He 초기화의 layer별 기울기 |
| [03-residual-learning.md](03-residual-learning.md) | 잔차 학습 y = F(x) + x에서 항등 사상은 왜 배우기 쉬워지는가? | plain layer와 residual block이 항등이 되는 조건, zero-init 블록이 실제로 항등인지 확인 |
| [04-skip-connection-gradient.md](04-skip-connection-gradient.md) | skip connection은 역전파 식에서 무엇을 바꾸고, 실제 네트워크에서 기울기는 어떻게 달라지는가? | $1 + \partial F/\partial x$ 유도, $2^L$개 경로 전개, BN 없는 MLP와 BN 있는 56-layer CNN의 layer별 기울기, 기울기 방향의 코사인 유사도 |
| [05-batch-normalization.md](05-batch-normalization.md) | batch normalization은 무엇을 정규화하고, 왜 residual connection과 함께 필요한가? | $[2, 4, 6, 8]$ 손계산, conv bias가 상쇄되는 것, 학습과 추론의 차이, ICS 가설과 반론, 분산이 $2^\ell$과 $\ell$로 커지는 차이 |
| [06-bottleneck.md](06-bottleneck.md) | 1x1 convolution과 bottleneck은 어떻게 layer를 늘리면서 연산량을 유지하는가? | 69,632 대 1,179,648, ResNet-18~152 구성표와 layer 수 세는 법, projection shortcut, global average pooling과 VGG의 FC layer |
| [07-preactivation.md](07-preactivation.md) | ResNet v2는 왜 덧셈 뒤의 ReLU를 없앴고, 같은 구조는 Transformer에서 어떻게 쓰이는가? | $x_L = x_l + \sum F$ 전개, shortcut에 상수를 곱하면 무너지는 계산, pre-activation 블록의 항등 확인, Transformer의 residual stream과 LSTM의 cell state |
| [08-resnext.md](08-resnext.md) | ResNeXt의 cardinality는 무엇이고, 경로를 32개로 나눠도 비용이 같은 이유는 무엇인가? | 32x4d의 70,144, 경로 32개와 grouped convolution의 출력이 같은지 확인, DenseNet과 EfficientNet의 설계 |
| [09-plain-vs-residual-experiment.md](09-plain-vs-residual-experiment.md) | 얕은 네트워크와 깊은 네트워크에서 skip connection의 효과는 어떻게 다른가? | CIFAR-10 축소 실행(8, 20, 56-layer), 8-layer에서 효과가 없는 이유, 표본 크기와 측정 오차 |
| [10-common-misconceptions.md](10-common-misconceptions.md) | ResNet을 공부할 때 잘못 기억하기 쉬운 설명은 무엇이고, 바르게는 무엇인가? | 오해 18개와 바로잡는 설명 |

## 실험

| 스크립트 | 확인하려는 것 | 결과 파일 | 결과 |
|---|---|---|---|
| `hand_calc.py` | 문서의 손계산 숫자가 계산과 맞는가 | [hand-calc.txt](results/hand-calc.txt) | $c^L$ 표, 초기화 분산, 수용 영역, 파라미터와 FLOPs, 경로 수가 모두 일치 |
| `blocks.py` | 블록의 파라미터 수와 모양, zero-init 블록이 항등인가 | [blocks.txt](results/blocks.txt) | 73,728, 69,632, torchvision resnet50 25,557,032. post-activation은 음수 입력에서 항등이 아니고 pre-activation은 항등 |
| `param_count.py` | ResNeXt의 두 표현이 같은가, BN이 conv bias를 지우는가, dilation의 수용 영역 | [param-count.txt](results/param-count.txt) | 70,144와 출력 일치, bias 상쇄, 수용 영역 3, 7, 15, 31, 63 |
| `grad_norms.py` | 깊이와 skip에 따라 layer별 기울기가 어떻게 달라지는가 | [grad-norms.txt](results/grad-norms.txt) | BN 없는 MLP에서 skip이 없으면 $10^{-34}$, 있으면 0.27 이상. BN 있는 plain-56은 소실되지 않음 |
| `cosine_similarity.py` | 입력을 1% 바꿨을 때 기울기 방향이 유지되는가 | [cosine-similarity.txt](results/cosine-similarity.txt) | plain-56 0.001~0.023, resnet-56 0.41~0.44 |
| `resnet_cifar.py` | CIFAR-10을 줄여서 돌리면 plain과 residual이 어떻게 다른가 | [cifar-small.txt](results/cifar-small.txt) | 학습 이미지 5,000장, 3 epoch로는 차이를 말할 수 없음 |

실험은 모두 학습 전 초기화 상태이거나 CPU에서 몇 분 안에 끝나는 축소 실행입니다. 논문의 ImageNet, CIFAR-10 결과를 재현하지는 않았습니다. 실행 환경과 출력은 [results/README.md](results/README.md)에 있습니다.

## 재현

Python 3, NumPy, PyTorch, torchvision이 필요합니다. GPU 없이 CPU에서 실행됩니다.

```bash
cd 02-resnet/experiments
python3 -m venv .venv
.venv/bin/pip install numpy torch torchvision

.venv/bin/python hand_calc.py
.venv/bin/python blocks.py
.venv/bin/python param_count.py
.venv/bin/python grad_norms.py
.venv/bin/python cosine_similarity.py
# CIFAR-10 축소 실행 (처음 실행할 때 ./data 에 데이터를 내려받습니다)
.venv/bin/python resnet_cifar.py --n 3 --plain --limit 5000 --epochs 3 --seed 0 --device cpu
.venv/bin/python resnet_cifar.py --n 3         --limit 5000 --epochs 3 --seed 0 --device cpu
```

`resnet_cifar.py`를 뺀 다섯 스크립트는 합쳐서 30초 안쪽이고, CIFAR-10 축소 실행은 모델 하나에 CPU에서 2~4분입니다.

## Further Reading

- [Multi-Scale Context Aggregation by Dilated Convolutions](https://arxiv.org/abs/1511.07122) (Yu, Koltun, 2016): 커널 원소 사이에 간격을 두어 해상도를 줄이지 않고 수용 영역을 넓히는 dilated convolution. 논문은 VGG-16의 뒤쪽 pooling을 없애고 dilated conv로 바꿔 픽셀 단위 segmentation에 썼고, dilated conv를 쌓은 모듈을 항등 초기화로 학습시켰습니다. 수용 영역 계산은 [01-why-depth-fails.md](01-why-depth-fails.md)에 있습니다
- [Densely Connected Convolutional Networks](https://arxiv.org/abs/1608.06993) (Huang 등, 2017): 앞 layer의 출력을 더하지 않고 이어 붙이는 DenseNet
- [EfficientNet](https://arxiv.org/abs/1905.11946) (Tan, Le, 2019): 깊이, 너비, 해상도를 계수 하나로 함께 키우는 compound scaling
- [Residual Networks Behave Like Ensembles of Relatively Shallow Networks](https://arxiv.org/abs/1605.06431) (Veit, Wilber, Belongie, 2016): ResNet을 짧은 경로들의 모음으로 보는 해석

## 참고 자료

- [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385) (He, Zhang, Ren, Sun, 2015): 잔차 연결, degradation, 18-layer와 34-layer 비교
- [Identity Mappings in Deep Residual Networks](https://arxiv.org/abs/1603.05027) (He, Zhang, Ren, Sun, 2016): pre-activation, shortcut 변형 실험
- [Batch Normalization](https://arxiv.org/abs/1502.03167) (Ioffe, Szegedy, 2015): BN의 수식과 internal covariate shift 가설
- [How Does Batch Normalization Help Optimization?](https://arxiv.org/abs/1805.11604) (Santurkar 등, 2018): internal covariate shift 가설을 반박한 실험
- [Batch Normalization Biases Residual Blocks Towards the Identity Function in Deep Networks](https://arxiv.org/abs/2002.10444) (De, Smith, 2020): BN이 있을 때와 없을 때 residual 네트워크의 분산 증가
- [Understanding the difficulty of training deep feedforward neural networks](https://proceedings.mlr.press/v9/glorot10a.html) (Glorot, Bengio, 2010): Xavier 초기화
- [Delving Deep into Rectifiers](https://arxiv.org/abs/1502.01852) (He 등, 2015): ReLU에 맞춘 He 초기화
- [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556) (Simonyan, Zisserman, 2014): VGG
- [Going Deeper with Convolutions](https://arxiv.org/abs/1409.4842) (Szegedy 등, 2014): GoogLeNet과 Inception module
- [Visualizing and Understanding Convolutional Networks](https://arxiv.org/abs/1311.2901) (Zeiler, Fergus, 2013): ZFNet
- [Aggregated Residual Transformations for Deep Neural Networks](https://arxiv.org/abs/1611.05431) (Xie 등, 2017): ResNeXt와 cardinality
- [The Shattered Gradients Problem](https://arxiv.org/abs/1702.08591) (Balduzzi 등, 2017): 깊은 plain 네트워크에서 기울기의 상관이 무너지는 현상
- [Deep Networks with Stochastic Depth](https://arxiv.org/abs/1603.09382) (Huang 등, 2016): 학습 중에 블록을 무작위로 건너뛰는 방법
- [Neural Ordinary Differential Equations](https://arxiv.org/abs/1806.07366) (Chen 등, 2018): residual 구조를 미분방정식으로 보는 연구
- [Highway Networks](https://arxiv.org/abs/1505.00387) (Srivastava, Greff, Schmidhuber, 2015): 게이트로 두 경로의 비율을 학습하는 구조
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (Vaswani 등, 2017): residual connection과 LayerNorm을 쓰는 Transformer
- [CS231n Convolutional Neural Networks](https://cs231n.github.io/convolutional-networks/): 합성곱, 수용 영역, 파라미터 계산을 다루는 강의 노트
