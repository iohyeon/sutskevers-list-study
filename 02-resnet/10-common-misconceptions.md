# ResNet을 공부할 때 잘못 기억하기 쉬운 설명은 무엇이고, 바르게는 무엇인가?

## 문제

ResNet과 그 주변 논문을 공부할 때 잘못 기억하기 쉬운 설명을 모았습니다. 항목마다 오해를 한 줄로 적고, 바로잡는 내용을 두세 문장으로 적은 뒤, 자세한 설명이 있는 문서를 연결했습니다.

## skip connection

**1. skip connection은 가중치를 다음 layer로 넘긴다.**
넘어가는 것은 가중치가 아니라 그 블록의 입력 값 $x$입니다. 코드로는 입력 텐서를 변수에 잡아 뒀다가 변환 결과에 더하는 `out = out + identity` 한 줄이고, 더하기 연산에는 파라미터가 없습니다.
→ [03-residual-learning.md](03-residual-learning.md)

**2. 잔차 연결은 뒤쪽 fully connected layer에 관한 이야기다.**
residual block은 conv layer들로 이뤄져 있습니다. ResNet에서 FC layer는 맨 끝의 하나뿐입니다.
→ [06-bottleneck.md](06-bottleneck.md)

**3. skip connection을 붙이면 어떤 네트워크든 좋아진다.**
잔차 연결의 이득은 같은 깊이에서 정확도가 오르는 것이 아니라 더 깊게 쌓아도 학습이 된다는 것입니다. ResNet 논문에서도 18-layer에서는 plain 네트워크와 차이가 0.06%p였습니다.
→ [09-plain-vs-residual-experiment.md](09-plain-vs-residual-experiment.md)

**4. 채널을 줄이면서 잃은 정보는 layer가 늘어난 만큼 더 학습해서 메운다.**
채널을 줄이는 1x1 conv도 학습되는 가중치이므로 무엇을 남길지를 학습이 정합니다. 그리고 블록의 출력에는 입력 $x$가 그대로 더해지므로, 변환 경로가 맡는 일은 $x$를 다시 만드는 것이 아니라 $x$에 보탤 값을 계산하는 것입니다.
→ [06-bottleneck.md](06-bottleneck.md)

**5. 256채널을 64채널로 줄이는 1x1 conv는 깊이가 64인 필터다.**
64는 필터의 개수입니다. 필터 하나의 크기는 $1 \times 1 \times 256$이고, 출력 채널 수는 필터를 몇 개 두었는가로 정해집니다.
→ [06-bottleneck.md](06-bottleneck.md)

## 기울기

**6. 기울기 소실은 부동소수점 오차 때문에 생긴다.**
반올림 오차가 아니라 연쇄법칙의 곱셈 구조에서 나옵니다. layer마다 0.9가 곱해지면 layer 152개를 지난 뒤에는 $0.9^{152} \approx 1.1 \times 10^{-7}$이고, 정밀도를 높여도 같은 값이 나옵니다.
→ [02-vanishing-gradient.md](02-vanishing-gradient.md)

**7. ReLU가 기울기 소실을 해결했다.**
ReLU는 곱해지는 수 가운데 활성화 함수의 미분을 1로 만들었을 뿐입니다. 가중치의 곱과 꺼진 뉴런은 그대로 남습니다. skip connection은 곱해지는 값을 $c$에서 $1 + c$로 바꿨습니다.
→ [02-vanishing-gradient.md](02-vanishing-gradient.md), [04-skip-connection-gradient.md](04-skip-connection-gradient.md)

**8. degradation은 과적합이거나 기울기 소실이다.**
56-layer plain 네트워크는 20-layer보다 학습 오차도 높았습니다. 과적합이라면 학습 오차는 더 낮아야 합니다. ResNet 논문의 plain 네트워크는 BN을 썼고 기울기 크기도 정상이어서, 논문은 기울기 소실이 원인일 가능성을 낮게 봤습니다.
→ [01-why-depth-fails.md](01-why-depth-fails.md)

**9. BN이 있으면 plain 네트워크의 기울기는 크기부터 사라진다.**
BN과 He 초기화를 쓴 plain-56을 초기화 상태에서 재 보면 기울기는 소실되지 않고 입력 쪽으로 갈수록 커집니다. 차이는 크기보다 방향에서 보입니다. 입력을 1%만 바꿨을 때 plain-56의 기울기 방향은 코사인 유사도 0.001~0.023으로 거의 무관해졌고, resnet-56은 0.41~0.44를 유지했습니다.
→ [04-skip-connection-gradient.md](04-skip-connection-gradient.md)

## batch normalization

**10. BN은 값을 0과 1 사이로 넣는다.**
BN은 평균을 0, 분산을 1로 맞춘 뒤 학습되는 $\gamma$와 $\beta$로 배율과 위치를 다시 정합니다. 결과에는 음수도 있고 1보다 큰 값도 있습니다. 기준이 되는 통계는 입력 이미지의 분포가 아니라 그 layer에 들어온 값의 미니배치 통계입니다.
→ [05-batch-normalization.md](05-batch-normalization.md)

**11. BN은 internal covariate shift를 줄여서 잘 동작한다.**
BN 논문이 든 동기이지만, 뒤에 나온 실험은 BN 뒤에 분포를 일부러 흔들어도 학습이 잘 된다는 것을 보였습니다. 손실 지형을 매끄럽게 한다는 설명을 포함해 여러 설명이 함께 쓰입니다.
→ [05-batch-normalization.md](05-batch-normalization.md)

**12. `model.eval()`과 `torch.no_grad()`는 같은 일을 한다.**
`model.eval()`은 BN이 이동 평균을 쓰고 dropout이 꺼지도록 동작을 바꾸고, `torch.no_grad()`는 기울기 기록을 끕니다. 추론 코드에서 `eval()`을 빠뜨리면 BN이 배치 통계로 정규화해서 배치 구성에 따라 출력이 달라집니다.
→ [05-batch-normalization.md](05-batch-normalization.md)

## 주변 논문

**13. dilated convolution 논문은 ResNet을 detection에 써 본 것이다.**
Yu와 Koltun(2016)은 ResNet이 아니라 VGG-16을 고쳤고, 다룬 작업도 픽셀마다 라벨을 붙이는 segmentation입니다. dilation은 커널 하나가 입력의 어느 위치를 보는지를 정하고, residual connection은 블록의 입력과 출력을 어떻게 잇는지를 정합니다. 둘은 독립이고 함께 쓸 수 있습니다.
→ [01-why-depth-fails.md](01-why-depth-fails.md)의 dilation 계산, [README의 Further Reading](README.md#further-reading)

**14. cardinality는 서로 다른 모듈 종류의 수다.**
블록 안에 나란히 놓인 같은 모양의 경로가 몇 개인가입니다. ResNeXt의 표준 구성은 32개이고, 가중치는 경로마다 따로 두고 경로를 좁게 만들어 전체 비용을 맞춥니다. 구현은 grouped convolution과 같습니다.
→ [08-resnext.md](08-resnext.md)

**15. VGG는 11-layer다.**
VGG 논문에는 11-layer부터 19-layer까지 여러 구성이 있고, 널리 쓰이는 것은 VGG-16과 VGG-19입니다.
→ [01-why-depth-fails.md](01-why-depth-fails.md)

**16. GoogLeNet이 표준이 되지 못한 이유는 기울기 소실이다.**
주된 이유는 확장하기 어려웠다는 점입니다. Inception module은 병렬 경로의 필터 크기와 채널 수를 사람이 맞춰야 했고 깊이를 바꿀 때마다 다시 조정해야 했습니다. ResNet은 같은 블록의 반복 횟수만 바꾸면 됩니다.
→ [01-why-depth-fails.md](01-why-depth-fails.md)

**17. ResNet 논문은 152-layer가 최적이라고 밝혔다.**
논문의 ImageNet 실험에서 결과가 가장 좋았던 것이 152-layer였을 뿐, 최적의 깊이를 정하는 기준은 제시하지 않았습니다. 뒤에 나온 pre-activation 구조에서는 200-layer가 152-layer보다 나았습니다.
→ [07-preactivation.md](07-preactivation.md)

**18. pre-activation은 이유 없이 해 보니 좋아진 결과다.**
논문은 수식 분석을 먼저 제시합니다. 덧셈 뒤에 ReLU가 없어야 $x_L = x_l + \sum F$가 정확히 성립하고 기울기의 "+1"이 모든 블록에서 그대로 남습니다. 실험은 이 분석을 확인합니다.
→ [07-preactivation.md](07-preactivation.md)

## References

- [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385) (He, Zhang, Ren, Sun, 2015)
- [Identity Mappings in Deep Residual Networks](https://arxiv.org/abs/1603.05027) (He, Zhang, Ren, Sun, 2016)
- [How Does Batch Normalization Help Optimization?](https://arxiv.org/abs/1805.11604) (Santurkar 등, 2018)
- [Aggregated Residual Transformations for Deep Neural Networks](https://arxiv.org/abs/1611.05431) (Xie 등, 2017)
- [Multi-Scale Context Aggregation by Dilated Convolutions](https://arxiv.org/abs/1511.07122) (Yu, Koltun, 2016)
