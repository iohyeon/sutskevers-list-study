# results

`experiments/`의 스크립트를 실제로 실행해 나온 값입니다. 로그 전체가 아니라 문서와 README에 적은 숫자를 다시 확인하는 데 필요한 최종 값, 실행 명령, 환경, seed, 소요 시간만 남겼습니다.

환경은 Python 3.14.7, NumPy 2.5.3, PyTorch 2.14.0, torchvision 0.29.0, macOS arm64, CPU입니다. 실행일은 2026-09-22입니다. CIFAR-10 데이터와 학습된 모델은 저장소에 넣지 않았고 `small_cnn.py`가 처음 실행될 때 데이터를 내려받습니다.

| 파일 | 명령 | 대응하는 숫자 | 쓰인 곳 |
|---|---|---|---|
| [alexnet-params.txt](alexnet-params.txt) | `python3 alexnet_params.py` | layer별 출력 크기, 파라미터 60,965,224개, 뉴런 659,272개, 곱셈 724,406,816회, FC의 파라미터 비율 96.17%, conv의 곱셈 비율 91.91%, 분할을 반영하지 않을 때 62,378,344개, receptive field 11에서 195, PyTorch 모델의 출력 모양과 파라미터 수 | [08-alexnet-architecture.md](../08-alexnet-architecture.md), [09-alexnet-parameters-and-gpus.md](../09-alexnet-parameters-and-gpus.md), 루트 README |
| [conv-numpy.txt](conv-numpy.txt) | `python3 conv_numpy.py` | 4x4 입력과 2x2 커널의 출력, 고양이 그림의 세로와 가로 에지 feature map, ReLU와 max pooling 결과, stride 2의 출력 (5, 5), conv1 모양 layer의 출력 (96, 15, 15)과 파라미터 34,944개, im2col 행렬 (225, 363)과 반복문 구현의 일치 | [03-convolution.md](../03-convolution.md) |
| [conv-backprop.txt](conv-backprop.txt) | `python3 conv_backprop.py` | 커널 기울기 3, 7, 0, 0(2x2), 입력 기울기, 두 기울기와 수치 미분의 일치 | [04-backpropagation.md](../04-backpropagation.md) |
| [learn-sobel.txt](learn-sobel.txt) | `python3 learn_sobel.py` | step 1, 10, 50, 100의 MSE 3.634803, 0.203471, 0.000011, 0.000000, 500 step 뒤 Sobel 커널과의 최대 차이 1.7 × 10⁻¹⁴ | [04-backpropagation.md](../04-backpropagation.md), 루트 README |
| [hand-calc.txt](hand-calc.txt) | `python3 hand_calc.py` | 행렬 곱과 ReLU, sigmoid 미분, 0.25⁸, softmax (0.659, 0.242, 0.099)와 loss 0.417, 역전파 한 걸음(손실 0.125에서 0.0103, 수치 미분 1.5), 경사하강법 세 걸음, 좁은 골짜기 40걸음과 momentum 0, 0.5, 0.9를 더한 40걸음, momentum 10배와 0.526배, 초기화 분산 0.10과 0.12, dropout, FC로 이미지를 받을 때의 가중치 수 | [02-neural-network-and-loss.md](../02-neural-network-and-loss.md), [03-convolution.md](../03-convolution.md), [04-backpropagation.md](../04-backpropagation.md), [05-relu-and-initialization.md](../05-relu-and-initialization.md), [06-optimization.md](../06-optimization.md), [07-regularization.md](../07-regularization.md) |
| [small-cnn.txt](small-cnn.txt) | `python3 small_cnn.py` | 설정 7개의 20 epoch 뒤 test acc: relu 기준 0.487, relu + He 0.529, tanh 0.607, dropout 없음 0.655, augmentation 없음 0.589, 둘 다 없음 0.605(train 0.972), momentum 0 0.431. epoch별 test acc와 파라미터 2,572,810개 | [05-relu-and-initialization.md](../05-relu-and-initialization.md), [06-optimization.md](../06-optimization.md), [07-regularization.md](../07-regularization.md) |

소요 시간은 다른 작업이 함께 돌던 상태에서 잰 값이라 환경에 따라 달라집니다. `small_cnn.py`는 CIFAR-10 학습 이미지 앞 10,000장과 시험 이미지 앞 2,000장, 20 epoch, seed 0 하나로 돌린 결과입니다. 이 실행은 torchvision이 내려받는 것과 같은 CIFAR-10을 미리 받아 둔 배열에서 읽었습니다.
