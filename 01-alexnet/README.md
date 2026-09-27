# AlexNet

2012년의 AlexNet이 feature를 사람이 설계하던 방식을 어떻게 바꾸었고, 그 구조가 어떤 계산으로 이루어져 있는지 다룹니다. 신경망과 손실, 역전파, 합성곱을 숫자로 계산한 뒤, AlexNet의 layer별 출력 크기와 파라미터 수, 곱셈 횟수를 공식과 코드로 확인했습니다. 마지막으로 AlexNet을 줄인 CNN을 CIFAR-10 일부로 학습시켜 ReLU, 초기화, dropout, augmentation, momentum을 하나씩 바꿔 봤습니다.

## 문서

| 문서 | 질문 | 계산하거나 보여 주는 것 |
|---|---|---|
| [01-representation-learning.md](01-representation-learning.md) | AlexNet 이전의 이미지 인식은 무엇을 사람이 정했고, 표현 학습은 그중 무엇을 데이터에서 배우게 바꾸었는가? | SIFT, HOG, SVM 파이프라인과 end-to-end 학습의 비교, SVM이 볼록 최적화라는 근거, ILSVRC 연도별 top-5 오류 |
| [02-neural-network-and-loss.md](02-neural-network-and-loss.md) | 신경망은 입력에서 클래스 확률을 어떻게 계산하고, 예측이 틀린 정도는 어떤 숫자로 재는가? | 입력 3개, 뉴런 2개 layer의 손계산, softmax와 cross-entropy를 점수 (2, 1, 0.1)로 계산, 기울기가 $\mathbf{p} - \mathbf{y}$ 가 되는 유도 |
| [03-convolution.md](03-convolution.md) | 합성곱은 이미지에서 무엇을 계산하고, 같은 입력을 FC layer로 받을 때보다 파라미터가 왜 적은가? | 12x12 고양이 그림에 에지 커널을 적용한 10x10 feature map 두 장, ReLU와 max pooling, NumPy 구현과 im2col |
| [04-backpropagation.md](04-backpropagation.md) | 역전파는 수천만 개 가중치의 기울기를 어떻게 한 번의 역방향 계산으로 구하는가? | 변수 4개 네트워크의 한 걸음 학습(손실 0.125에서 0.0103), 행렬 형태의 역전파 식, conv layer의 역전파, 무작위 커널이 Sobel 커널로 수렴하는 실험 |
| [05-relu-and-initialization.md](05-relu-and-initialization.md) | ReLU와 가중치 초기화는 layer를 지날 때 신호와 기울기의 크기를 어떻게 바꾸는가? | sigmoid 미분 0.25를 8번 곱한 값, 초기화에 따른 분산 계산, 활성화 함수와 초기화만 바꾼 학습 비교 |
| [06-optimization.md](06-optimization.md) | AlexNet은 2차 최적화 없이 SGD와 momentum으로 좁은 골짜기 모양의 손실 곡면을 어떻게 내려갔는가? | 학습률별 경사하강법 손계산, 곡률이 100배 다른 골짜기에서의 40걸음, momentum의 등비급수, momentum 0과 0.9의 학습 비교 |
| [07-regularization.md](07-regularization.md) | 파라미터 약 6,100만 개짜리 모델이 학습 이미지 120만 장을 외우지 않도록 AlexNet은 무엇을 했는가? | inverted dropout 손계산, PCA 색 흔들기 식, dropout과 augmentation을 끈 학습 비교 |
| [08-alexnet-architecture.md](08-alexnet-architecture.md) | AlexNet의 각 layer는 입력을 어떤 크기로 바꾸고, 뒤쪽 layer의 뉴런은 입력 이미지의 어느 범위를 보는가? | layer별 출력 크기(227에서 6까지), receptive field(11픽셀에서 195픽셀까지), PyTorch로 모양 확인 |
| [09-alexnet-parameters-and-gpus.md](09-alexnet-parameters-and-gpus.md) | AlexNet의 파라미터와 곱셈은 어느 layer에 있고, 3GB GPU 두 장에는 어떻게 나뉘었는가? | 파라미터 60,965,224개와 곱셈 724,406,816회의 layer별 비율, 두 GPU 분할이 파라미터 수에 주는 영향 |

## 실험

| 스크립트 | 확인하려는 것 | 결과 파일 | 결과 |
|---|---|---|---|
| `alexnet_params.py` | layer별 공식으로 센 파라미터 수가 논문과 PyTorch 구현과 맞는가 | [alexnet-params.txt](results/alexnet-params.txt) | 60,965,224개. 분할을 반영하지 않으면 62,378,344개. 파라미터의 96.2%는 FC, 곱셈의 91.9%는 conv |
| `conv_numpy.py` | 반복문 합성곱과 im2col 행렬 곱이 같은 값을 내는가 | [conv-numpy.txt](results/conv-numpy.txt) | 고양이 feature map 두 장과 pooling 결과, 두 구현의 출력 일치 |
| `conv_backprop.py` | 유도한 conv layer의 기울기가 수치 미분과 일치하는가 | [conv-backprop.txt](results/conv-backprop.txt) | 커널과 입력의 기울기 모두 일치 |
| `learn_sobel.py` | 무작위 3x3 커널이 역전파만으로 Sobel 커널이 되는가 | [learn-sobel.txt](results/learn-sobel.txt) | 500걸음 뒤 Sobel 커널과의 최대 차이 $1.7 \times 10^{-14}$ |
| `hand_calc.py` | 문서의 손계산 값이 코드 출력과 일치하는가 | [hand-calc.txt](results/hand-calc.txt) | 역전파 한 걸음, softmax, 경사하강법, 좁은 골짜기, momentum, dropout 값이 일치 |
| `small_cnn.py` | 활성화 함수, 초기화, dropout, augmentation, momentum이 학습 결과를 어떻게 바꾸는가 | [small-cnn.txt](results/small-cnn.txt) | 시험 정확도: ReLU 기본 초기화 48.7%, He 초기화 52.9%, tanh 60.7%, momentum 0 43.1%. dropout과 augmentation을 끈 쪽이 더 높았고 둘 다 끈 설정만 train 97.2% 대 시험 60.5%로 과적합 |

`small_cnn.py` 는 CIFAR-10 학습 이미지 10,000장, 20 epoch, seed 하나로 CPU에서 돌린 축소판입니다. 조건과 한계는 [05](05-relu-and-initialization.md), [06](06-optimization.md), [07](07-regularization.md)의 「한계와 주의할 점」에 적었습니다.

## 재현

Python 3, NumPy, PyTorch가 필요합니다. `small_cnn.py` 는 torchvision도 쓰고, 처음 실행할 때 CIFAR-10을 `./data` 에 내려받습니다.

```bash
cd 01-alexnet/experiments
python3 -m venv .venv
.venv/bin/pip install numpy torch torchvision

.venv/bin/python alexnet_params.py
.venv/bin/python conv_numpy.py
.venv/bin/python conv_backprop.py
.venv/bin/python learn_sobel.py
.venv/bin/python hand_calc.py
.venv/bin/python small_cnn.py            # CPU에서 설정 7개에 약 1시간
```

`small_cnn.py` 를 뺀 나머지는 각각 몇 초 안에 끝납니다. 실행한 환경과 출력은 [results/README.md](results/README.md)에 있습니다.

## Further Reading

- ImageNet 데이터셋을 만든 과정(WordNet 기반 범주 체계, 크라우드소싱 라벨링)은 [ImageNet: A Large-Scale Hierarchical Image Database](https://www.image-net.org/static_files/papers/imagenet_cvpr09.pdf)에, 대회의 과제와 평가 방식, 사람 기준 오류율을 잰 실험은 [ImageNet Large Scale Visual Recognition Challenge](https://arxiv.org/abs/1409.0575)에 있습니다.
- 두 GPU 분할 이후 conv layer와 FC layer를 서로 다른 방식으로 병렬화하는 방법은 [One weird trick for parallelizing convolutional neural networks](https://arxiv.org/abs/1404.5997)에 있습니다.
- 합성곱과 pooling의 출력 크기, 가중치 공유를 다른 예로 설명한 자료로 [CS231n의 합성곱 노트](https://cs231n.github.io/convolutional-networks/)가 있습니다.

## 참고 자료

- [ImageNet Classification with Deep Convolutional Neural Networks](https://papers.nips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html) (Krizhevsky, Sutskever, Hinton, 2012): AlexNet 논문. 구조, ReLU, dropout, data augmentation, 두 GPU 분할
- [One weird trick for parallelizing convolutional neural networks](https://arxiv.org/abs/1404.5997) (Krizhevsky, 2014): conv layer와 FC layer를 서로 다른 방식으로 병렬화하는 이유
- [ImageNet: A Large-Scale Hierarchical Image Database](https://www.image-net.org/static_files/papers/imagenet_cvpr09.pdf) (Deng 외, 2009): ImageNet 데이터셋을 만든 방법
- [ImageNet Large Scale Visual Recognition Challenge](https://arxiv.org/abs/1409.0575) (Russakovsky 외, 2015): ILSVRC 대회의 과제, 평가 지표, 연도별 결과
- [Multi-column Deep Neural Networks for Image Classification](https://arxiv.org/abs/1202.2745) (Ciresan, Meier, Schmidhuber, 2012): AlexNet 이전에 GPU로 CNN을 학습한 DanNet
- [Learning representations by back-propagating errors](https://www.nature.com/articles/323533a0) (Rumelhart, Hinton, Williams, 1986): 역전파 논문
- [Understanding the difficulty of training deep feedforward neural networks](https://proceedings.mlr.press/v9/glorot10a.html) (Glorot, Bengio, 2010): 초기화와 활성화 함수가 깊은 네트워크의 학습에 주는 영향
- [Delving Deep into Rectifiers](https://arxiv.org/abs/1502.01852) (He, Zhang, Ren, Sun, 2015): ReLU 네트워크를 위한 초기화(He 초기화)
- [Deep learning via Hessian-free optimization](https://www.cs.toronto.edu/~jmartens/docs/Deep_HessianFree.pdf) (Martens, 2010): 곡률 차이가 큰 손실 곡면과 2차 최적화
- [On the importance of initialization and momentum in deep learning](https://proceedings.mlr.press/v28/sutskever13.html) (Sutskever 외, 2013): 초기화와 momentum만으로 깊은 네트워크를 학습한 결과
- [Greedy Layer-Wise Training of Deep Networks](https://proceedings.neurips.cc/paper/2006/hash/5da713a690c067105aeb2fae32403405-Abstract.html) (Bengio 외, 2006): layer별 비지도 사전학습
- [Dropout: A Simple Way to Prevent Neural Networks from Overfitting](https://jmlr.org/papers/v15/srivastava14a.html) (Srivastava 외, 2014): dropout의 방법과 실험
- [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556) (Simonyan, Zisserman, 2014): LRN을 넣어도 성능이 나아지지 않았다는 보고가 있는 VGG 논문
- [Identifying and attacking the saddle point problem in high-dimensional non-convex optimization](https://arxiv.org/abs/1406.2572) (Dauphin 외, 2014): 고차원에서는 local minima보다 saddle point가 문제라는 분석
- [The Loss Surfaces of Multilayer Networks](https://arxiv.org/abs/1412.0233) (Choromanska 외, 2015): 큰 네트워크의 local minima가 비슷한 손실값을 갖는다는 분석
- [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980) (Kingma, Ba, 2014): 파라미터별로 걸음 크기를 맞추는 최적화기
- [HOGgles: Visualizing Object Detection Features](http://www.cs.columbia.edu/~vondrick/ihog/iccv.pdf) (Vondrick 외, 2013): HOG feature를 이미지로 되돌려 보는 시각화
- [Unbiased Look at Dataset Bias](https://people.csail.mit.edu/torralba/publications/datasets_cvpr11.pdf) (Torralba, Efros, 2011): 데이터셋 사이의 일반화 문제
- [The Unreasonable Effectiveness of Data](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/35179.pdf) (Halevy, Norvig, Pereira, 2009): 단순한 모델과 많은 데이터
- [GPipe](https://arxiv.org/abs/1811.06965) (Huang 외, 2018): 파이프라인 병렬화
- [CS231n: Convolutional Neural Networks](https://cs231n.github.io/convolutional-networks/): 합성곱, pooling, 출력 크기 계산을 설명한 Stanford 강의 노트
