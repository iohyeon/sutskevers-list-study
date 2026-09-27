# results

`experiments/`의 스크립트를 실제로 실행해 나온 값입니다. 로그 전체가 아니라 문서와 README에 적은 숫자를 다시 확인하는 데 필요한 출력, 실행 명령, 환경, seed, 소요 시간만 남겼습니다.

환경은 Python 3.14.7, NumPy 2.5.3, PyTorch 2.14.0, torchvision 0.29.0, macOS arm64, CPU입니다. 실행일은 2026-09-22입니다. CIFAR-10 데이터(`data/`)와 학습 기록(`*.json`)은 저장소에 넣지 않았습니다.

| 파일 | 명령 | 대응하는 숫자 | 쓰인 곳 |
|---|---|---|---|
| [hand-calc.txt](hand-calc.txt) | `python3 hand_calc.py` | $c^L$ 표($0.9^{152} \approx 1.1 \times 10^{-7}$), 초기화 분산 배율, 수용 영역, bottleneck 69,632와 1,179,648, FLOPs 2.18억과 37억, ResNet layer 수, ResNeXt 70,144, 경로 수와 길이 분포, 12장 표본의 표준편차 1.6 | [01](../01-why-depth-fails.md), [02](../02-vanishing-gradient.md), [04](../04-skip-connection-gradient.md), [06](../06-bottleneck.md), [08](../08-resnext.md), [09](../09-plain-vs-residual-experiment.md), 루트 README |
| [blocks.txt](blocks.txt) | `python3 blocks.py` | BasicBlock 73,728(전체 73,984), Bottleneck 69,632(전체 70,400), projection 8,192, torchvision ResNet-18~152 파라미터 수, zero-init 블록과 pre-activation 블록의 항등 확인 | [03](../03-residual-learning.md), [06](../06-bottleneck.md), [07](../07-preactivation.md) |
| [param-count.txt](param-count.txt) | `python3 param_count.py` | ResNeXt 두 표현의 가중치 70,144와 출력 일치, BN 손계산 $[-1.683, 0.106, 1.894, 3.683]$, conv bias 상쇄, dilation별 수용 영역 3, 7, 15, 31, 63, VGG-16의 FC 비중 89.4% | [01](../01-why-depth-fails.md), [05](../05-batch-normalization.md), [06](../06-bottleneck.md), [08](../08-resnext.md) |
| [grad-norms.txt](grad-norms.txt) | `python3 grad_norms.py` | 50-layer MLP의 layer별 기울기(sigmoid, ReLU, He 초기화), BN 없는 MLP에서 skip 유무, BN 있는 plain-56과 resnet-56, 첫 conv와 마지막 conv의 기울기 비율 | [02](../02-vanishing-gradient.md), [04](../04-skip-connection-gradient.md), [05](../05-batch-normalization.md) |
| [cosine-similarity.txt](cosine-similarity.txt) | `python3 cosine_similarity.py` | 입력을 1% 바꿨을 때 기울기 방향의 코사인 유사도. plain-56 0.001, 0.013, 0.023, resnet-56 0.430, 0.411, 0.435 | [04](../04-skip-connection-gradient.md), [10](../10-common-misconceptions.md) |
| [cifar-small.txt](cifar-small.txt) | `python3 resnet_cifar.py`에 `--limit`, `--epochs`를 줄여서 | 8, 20, 56-layer plain과 residual의 축소 실행 결과, 파라미터 수 | [09](../09-plain-vs-residual-experiment.md) |

CIFAR-10 축소 실행은 학습 이미지 5,000장(56-layer는 2,000장)과 2~3 epoch로 코드가 끝까지 도는지 확인한 것입니다. 이 길이로는 plain과 residual, 깊이의 차이를 말할 수 없습니다. 실행 환경에서는 torchvision 다운로드 대신 같은 CIFAR-10 데이터의 로컬 사본을 읽었습니다. 소요 시간은 다른 작업과 함께 돌던 상태에서 잰 값이라 환경에 따라 달라집니다.
