# dropout의 자리와 CTC

LSTM을 키웠을 때 과적합을 막는 방법과, 정렬 라벨 없이 음성을 글자로 옮기는 방법을 다룹니다. dropout을 순환 연결에 걸 때와 layer 사이 연결에 걸 때 무엇이 달라지는지 마스크가 곱해지는 횟수로 계산하고 직접 학습시켜 비교했습니다. 이어서 CTC의 forward 점화식을 유도해 NumPy로 구현하고 완전 탐색, `torch.nn.functional.ctc_loss`, 수치 미분과 맞췄으며, beam search 디코딩과 stride, 출력 단위, 배치 정규화의 통계 범위, 길이순 정렬, 데이터 규모를 각각 실험으로 쟀습니다.

## 문서

| 문서 | 질문 | 계산하거나 보여 주는 것 |
|---|---|---|
| [01-why-scaling-rnn-overfits.md](01-why-scaling-rnn-overfits.md) | 파라미터를 늘린 LSTM은 왜 작은 말뭉치에서 과적합하는가? | PTB 설정의 파라미터 수 계산(medium 19,775,200개, large 66,022,000개)과 학습 토큰당 파라미터 수, hidden을 키웠을 때 학습과 검증 perplexity가 벌어지는 곡선 |
| [02-where-to-apply-dropout.md](02-where-to-apply-dropout.md) | dropout을 순환 연결에 걸면 무엇이 달라지는가? | 마스크가 $T$ 시점 곱해질 때 남는 비율 $p^T$ 의 계산과 표본 확인, dropout 자리 네 가지의 검증 perplexity, 학습 없이 마스크만 켜서 잰 기억 손상 |
| [03-variational-dropout.md](03-variational-dropout.md) | 마스크를 시퀀스 안에서 고정하면 순환 연결에도 dropout을 걸 수 있는가? | per-step과 per-sequence 마스크의 비교, per-sequence 마스크가 순환 가중치 행렬의 행을 지우는 것과 같다는 등식의 수치 확인, weight drop |
| [04-measuring-regularization.md](04-measuring-regularization.md) | 정규화가 되고 있는지 무엇으로 판단하는가? | 학습과 검증 perplexity의 차이, 최저 검증 지점이 뒤로 밀리는 정도, dropout 비율과 모델 크기의 조합별 결과 |
| [05-ctc-alignment.md](05-ctc-alignment.md) | 프레임과 글자의 정렬 라벨이 없을 때 어떻게 학습하는가? | blank와 접기 규칙, forward 점화식 유도, 프레임 6개와 라벨 "cat"의 alpha 표 손계산, 완전 탐색과의 일치, 기울기 검사, 로그 공간이 필요한 지점 |
| [06-ctc-decoding.md](06-ctc-decoding.md) | 프레임별 확률에서 문장을 어떻게 고르는가? | greedy와 beam search 구현, 언어 모델 결합 점수식의 $\alpha$, $\beta$ 고르기, beam 폭별 오류율과 계산량, greedy가 지는 발화의 실제 출력 |
| [07-stride-and-output-units.md](07-stride-and-output-units.md) | 시퀀스 길이를 줄이면서 출력할 글자 수를 지키려면 무엇을 바꿔야 하는가? | 라벨을 내는 데 필요한 최소 스텝 수, stride별로 표현 불가능해지는 발화 수, 문자 단위와 bigram 단위의 어휘 크기와 계산량 비교 |
| [08-sequence-wise-batchnorm.md](08-sequence-wise-batchnorm.md) | 순환 신경망에서 배치 정규화의 통계를 어느 범위로 잡아야 하는가? | 표본 수에 따른 분산 추정의 흔들림, 길이가 제각각인 배치에서 시점별 표본 수, 정규화 방식 네 가지의 학습 곡선 |
| [09-sortagrad-and-padding.md](09-sortagrad-and-padding.md) | 발화를 길이순으로 정렬해 배치를 만들면 무엇이 좋아지는가? | 패딩 비율의 식 유도와 실측, 시퀀스 길이에 따른 loss와 기울기 크기의 증가율, 첫 epoch의 기울기 폭발 횟수 |
| [10-scaling-with-data.md](10-scaling-with-data.md) | 학습 데이터를 10배 늘리면 오류율은 얼마나 줄어드는가? | 논문 표 10의 구간별 감소율 계산, 로그-로그 회귀로 추정한 멱법칙 지수, 합성 과제에서 데이터 양을 100배까지 늘린 측정 |

## 실험

| 스크립트 | 확인하려는 것 | 결과 파일 | 결과 |
|---|---|---|---|
| `param_count.py` | 손으로 센 파라미터 수가 구현과 맞는가 | [param-count.txt](results/param-count.txt) | hidden 96, 192, 384에서 153,628 / 602,140 / 2,383,900 일치. PTB 설정은 토큰당 21.3개와 71.0개 |
| `overfit.py` | 모델을 키우면 어디서 과적합이 드러나는가 | [overfit.txt](results/overfit.txt) | hidden 384의 50 epoch 학습 perplexity 1.154, 검증 2.664. 최저 검증은 세 크기 모두 1.73 근처 |
| `mask_decay.py` | 순환 연결의 마스크는 시점이 지나면 얼마나 남는가 | [mask-decay.txt](results/mask-decay.txt) | $0.5^{50}$ = 8.882e-16, 표본 200,000개로 확인. 학습된 모델에서는 per-step 0.25가 0.622로 dropout을 끈 0.625와 거의 같음 |
| `dropout_placement.py` | dropout을 어디에 걸어야 검증 perplexity가 낮은가 | [dropout-placement.txt](results/dropout-placement.txt) | 없음 2.664, 비순환만 1.850, 순환 per-step만 2.227, 양쪽 1.747 |
| `variational_dropout.py` | 마스크를 시퀀스 안에서 고정하면 달라지는가 | [variational-dropout.txt](results/variational-dropout.txt) | 비순환만 1.850, 순환 per-step 추가 1.747, per-sequence 추가 1.731, 양쪽 per-sequence 1.708. 행 지우기 등식의 최대 절대차 0 |
| `rate_sweep.py` | dropout 비율과 모델 크기는 어떻게 맞물리는가 | [rate-sweep.txt](results/rate-sweep.txt) | hidden 384는 0.5에서 1.850, 0.8에서 1.702. hidden 96은 0.5에서 1.709 |
| `ctc_forward.py` | forward 점화식이 모든 정렬의 확률 합과 같은가 | [ctc-forward.txt](results/ctc-forward.txt) | 프레임 6개에서 DP와 완전 탐색(경로 4,096개 중 84개)이 정확히 일치. torch ctc_loss와 차이 0, 수치 미분 상대 오차 최대 4.5e-08 |
| `ctc_decode.py` | beam search와 언어 모델은 무엇을 바꾸는가 | [ctc-decode.txt](results/ctc-decode.txt) | greedy CER 46.31%, beam 8과 언어 모델 결합 0.54%, beam 16 이상 0.36%. 언어 모델 없이 beam만 키우면 28% 아래로 내려가지 않음 |
| `ctc_stride.py` | stride로 스텝을 줄이면 무엇이 표현 불가능해지는가 | [ctc-stride.txt](results/ctc-stride.txt) | 문자 단위는 stride 6에서 333/400, stride 8에서 144/400. bigram 단위는 stride 8에서 400/400. 같은 표현 범위에서 계산량 1.97배 차이 |
| `sys_batchnorm.py` | 배치 정규화의 통계 범위가 학습을 바꾸는가 | [sequence-wise-batchnorm.txt](results/sequence-wise-batchnorm.txt) | frame error가 정규화 없음 0.2462, 시점별 0.7749, sequence-wise 0.2411, layer norm 0.2378. 시점별은 뒤쪽 시점의 표본이 1.76개까지 줄어듦 |
| `sys_sortagrad.py` | 길이순 정렬은 패딩과 학습 안정성을 어떻게 바꾸는가 | [sortagrad-and-padding.txt](results/sortagrad-and-padding.txt) | 패딩 비율 무작위 0.4844, 길이순 0.0149. 학습률 0.05의 첫 epoch에서 기울기 노름 $10^3$ 초과가 섞기 3~11회, 정렬 0회 |
| `sys_scaling.py` | 데이터를 늘리면 오류율이 멱법칙으로 줄어드는가 | [scaling-with-data.txt](results/scaling-with-data.txt) | 논문 표 10은 구간별 52.79%와 38.70% 감소, 5지점 회귀 지수 0.2729. 합성 과제는 지수 0.1442이고 양쪽 끝에서 눕는다 |

합성 말뭉치 하나, seed 하나로 잰 값이 많습니다. 조건과 한계는 각 문서의 「한계와 주의할 점」에 적었습니다.

## 재현

Python 3, NumPy, PyTorch가 필요합니다. GPU 없이 CPU에서 실행됩니다.

```bash
cd 04-dropout-and-ctc/experiments
python3 -m venv .venv
.venv/bin/pip install numpy torch

.venv/bin/python make_corpus.py       # corpus.txt 생성 (23,546글자)
.venv/bin/python param_count.py       # 아래 네 개는 학습을 포함합니다
.venv/bin/python overfit.py
.venv/bin/python dropout_placement.py
.venv/bin/python variational_dropout.py
.venv/bin/python rate_sweep.py
.venv/bin/python mask_decay.py
.venv/bin/python ctc_forward.py
.venv/bin/python ctc_decode.py
.venv/bin/python ctc_stride.py
.venv/bin/python sys_batchnorm.py
.venv/bin/python sys_sortagrad.py
.venv/bin/python sys_scaling.py
```

학습이 들어간 스크립트가 가장 오래 걸립니다(`rate_sweep.py` 556초, `variational_dropout.py` 452초, `dropout_placement.py` 384초). 나머지는 2분 안쪽입니다. 실행한 환경과 출력은 [results/README.md](results/README.md)에 있습니다.

## 참고 자료

- [Recurrent Neural Network Regularization](https://arxiv.org/abs/1409.2329) (Zaremba, Sutskever, Vinyals, 2014): dropout을 순환 연결이 아닌 layer 사이 연결에만 거는 방법
- [A Theoretically Grounded Application of Dropout in Recurrent Neural Networks](https://arxiv.org/abs/1512.05287) (Gal, Ghahramani, 2016): 시퀀스 안에서 마스크를 고정하는 variational dropout
- [Regularizing and Optimizing LSTM Language Models](https://arxiv.org/abs/1708.02182) (Merity, Keskar, Socher, 2017): 순환 가중치에 DropConnect를 거는 weight drop과 그 밖의 정규화
- [Dropout: A Simple Way to Prevent Neural Networks from Overfitting](https://jmlr.org/papers/v15/srivastava14a.html) (Srivastava 등, 2014): dropout의 원 논문
- [Connectionist Temporal Classification](https://www.cs.toronto.edu/~graves/icml_2006.pdf) (Graves, Fernández, Gomez, Schmidhuber, 2006): 정렬 라벨 없이 시퀀스를 학습시키는 방법
- [Deep Speech 2](https://arxiv.org/abs/1512.02595) (Amodei 등, 2015): 영어와 중국어 음성 인식을 여러 GPU로 학습시킨 시스템
- [Deep Speech](https://arxiv.org/abs/1412.5567) (Hannun 등, 2014): 그 전 세대 모델
- [Sequence Modeling With CTC](https://distill.pub/2017/ctc/) (Hannun, 2017): CTC의 점화식과 디코딩을 그림으로 설명한 글
- [Batch Normalization](https://arxiv.org/abs/1502.03167) (Ioffe, Szegedy, 2015): 배치 정규화의 원 논문
- [Layer Normalization](https://arxiv.org/abs/1607.06450) (Ba, Kiros, Hinton, 2016): 배치 대신 feature 방향으로 정규화하는 방법
- [Curriculum Learning](https://dl.acm.org/doi/10.1145/1553374.1553380) (Bengio 등, 2009): 쉬운 예제부터 학습시키는 방법
- [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361) (Kaplan 등, 2020): 데이터, 파라미터, 연산량과 loss의 멱법칙
