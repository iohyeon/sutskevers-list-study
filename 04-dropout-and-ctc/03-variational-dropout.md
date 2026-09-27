# 마스크를 시퀀스 안에서 고정하면 순환 연결에도 dropout을 걸 수 있는가?

## 문제

Zaremba, Sutskever, Vinyals는 순환 연결을 비워 두고 비순환 연결에만 dropout을 걸었습니다. [dropout을 어디에 걸 것인가](02-where-to-apply-dropout.md)에서 본 대로, 시점마다 독립 마스크를 곱하면 유지 확률이 $p^k$ 로 줄어들기 때문입니다.

Gal, Ghahramani의 2016년 논문 *A Theoretically Grounded Application of Dropout in Recurrent Neural Networks*는 문제가 순환 연결 자체가 아니라 마스크를 시점마다 새로 뽑는 데 있다고 봤습니다. 시퀀스 하나 동안 같은 마스크를 쓰면 순환 연결에도 dropout을 걸 수 있다는 주장입니다. 이 문서는 두 방식을 같은 조건에서 학습시켜 비교하고, 마스크를 고정하는 것이 가중치 행렬에 무엇을 하는 일인지 확인합니다.

## 아이디어

마스크를 뽑는 단위만 바꿉니다.

```
        per-step 마스크                     per-sequence 마스크
  t =   1     2     3     4           t =   1     2     3     4
  m =  m_1   m_2   m_3   m_4          m =   m     m     m     m
      (매 시점 새로 뽑는다)                 (시퀀스마다 한 번 뽑는다)
```

per-sequence 방식에서는 시퀀스 하나가 마스크를 하나만 만납니다. 어떤 유닛이 떨어지면 그 시퀀스 동안 계속 떨어지고, 살아 있으면 계속 살아 있습니다. 그래서 $p^k$ 로 줄어드는 항이 없습니다.

## 수식으로 보기

마스크의 효과를 가중치 쪽에서 보면 정리가 됩니다. $h_{t-1}$ 을 행 벡터로 두고 순환 가중치를 $U \in \mathbb{R}^{H \times 4H}$ 라 하면 순환 항은 $h_{t-1} U$ 입니다. 마스크 $m \in \{0, 1/p\}^H$ 를 $h_{t-1}$ 에 곱하면 다음이 성립합니다.

$$
(m \odot h_{t-1})\, U = h_{t-1} \big(\operatorname{diag}(m)\, U\big)
$$

$\operatorname{diag}(m) U$ 는 $m_j = 0$ 인 $j$ 에 대해 $U$ 의 $j$ 번째 행을 0으로 만든 행렬입니다. 즉 hidden의 한 차원을 떨어뜨리는 것은 순환 가중치 행렬에서 그 차원에 대응하는 행을 지우는 것과 같습니다. (열 벡터 규약으로 $W h_{t-1}$ 처럼 쓰면 같은 일이 열을 지우는 것으로 나타납니다. 지워지는 대상은 같고 이름만 바뀝니다.)

여기서 두 방식의 차이가 드러납니다.

$$
\text{per-step:}\quad h_t = f(x_t,\, h_{t-1};\, U_t), \quad U_t = \operatorname{diag}(m_t) U \ \ (t \text{ 마다 다르다})
$$
$$
\text{per-sequence:}\quad h_t = f(x_t,\, h_{t-1};\, \tilde{U}), \quad \tilde{U} = \operatorname{diag}(m) U \ \ (\text{시퀀스 내내 같다})
$$

per-step은 시퀀스 하나를 처리하는 동안 서로 다른 행렬 $T$ 개를 쓰는 것이고, per-sequence는 행렬 하나를 뽑아 끝까지 쓰는 것입니다. Gal, Ghahramani는 dropout을 가중치의 근사 사후분포에서 표본을 뽑는 일로 해석했습니다. 그 해석에서는 시퀀스 하나에 표본 하나가 대응해야 하고, 시점마다 새로 뽑는 것은 시퀀스를 처리하는 도중에 모델이 바뀌는 것이라 앞뒤가 맞지 않습니다.

## 작은 숫자로 직접 계산

위 등식을 $H = 6$ 짜리 예로 확인했습니다([results/variational-dropout.txt](results/variational-dropout.txt)).

```
H = 6, m = [0, 1, 1, 1, 1, 1], 0 이 된 자리 [0]
(h ⊙ m) U 와 h (diag(m) U) 의 최대 절대차 0.000e+00
```

$m_0 = 0$ 이므로 $U$ 의 0번 행이 지워진 것과 같습니다. 두 계산이 부동소수점 오차 없이 일치합니다.

행렬을 몇 개 쓰는지도 세어 둘 만합니다. 조각 길이 50, 배치 20이면 갱신 한 번에 per-step은 layer마다 마스크를 $50 \times 20 = 1{,}000$ 개 뽑고, per-sequence는 20개 뽑습니다. $H = 384$ 이면 마스크 하나가 $U \in \mathbb{R}^{384 \times 1536}$ 의 행 중 절반쯤을 지우는 선택입니다.

## 코드로 확인

`experiments/common.py`의 `LSTMLayer.forward`에서 두 방식이 갈립니다. `per_seq`는 반복문 앞에서 한 번 뽑고, `per_step`은 반복문 안에서 매번 뽑습니다.

```python
m = mask_like((B, self.n_hidden), rate) if on and mode == "per_seq" else None
for t in range(T):
    if on and mode == "per_step":
        m = mask_like((B, self.n_hidden), rate)
    hin = h if m is None else h * m
```

마스크의 모양이 `(B, H)` 이므로 배치 안의 시퀀스마다 다른 마스크가 붙습니다.

Merity, Keskar, Socher의 2017년 논문 *Regularizing and Optimizing LSTM Language Models*(AWD-LSTM)이 쓰는 weight drop은 마스크를 $h$ 가 아니라 $U$ 에 직접 겁니다. 행 단위가 아니라 원소 단위로 떨어뜨리고, 배치마다 한 번 뽑아 그 배치의 모든 시점과 모든 시퀀스에 같은 행렬을 씁니다.

```python
if self.weight_rate > 0.0:
    wh = F.dropout(cell.wh.weight, self.weight_rate, self.training)
```

세 방식을 자리와 단위로 놓고 비교하면 다음과 같습니다.

| 방식 | 마스크를 거는 대상 | 다시 뽑는 단위 | 지워지는 것 |
|---|---|---|---|
| per-step | $h_{t-1}$ | 시점마다, 시퀀스마다 | $U$ 의 행, 시점마다 바뀜 |
| per-sequence | $h_{t-1}$ | 시퀀스마다 | $U$ 의 행, 시퀀스 동안 고정 |
| weight drop | $U$ | 배치마다 | $U$ 의 원소, 배치 동안 고정 |

## 실험 결과

hidden 384, 2 layer로 다섯 조건을 같은 seed로 50 epoch씩 학습시켰습니다. 비순환 연결의 dropout 0.5는 모든 조건에 공통으로 걸었습니다. 태그 쌍은 temperature 0.5로 3,000글자를 생성해 여는 이름과 닫는 이름이 맞은 쌍의 수입니다([results/variational-dropout.txt](results/variational-dropout.txt)).

| 조건 | 최저 검증 PP (epoch) | 50 epoch 학습 PP | 검증 PP | 시험 PP | 차이 | 태그 쌍 |
|---|---|---|---|---|---|---|
| 비순환 0.5 (기준) | 1.728 (12) | 1.475 | 1.850 | 1.861 | +0.375 | 17/57 |
| 비순환 0.5 + 순환 per-step 0.5 | 1.722 (10) | 1.616 | 1.747 | 1.755 | +0.132 | 8/50 |
| 비순환 0.5 + 순환 per-seq 0.5 | 1.717 (19) | 1.630 | 1.731 | 1.742 | +0.101 | 16/55 |
| 비순환 per-seq 0.5 + 순환 per-seq 0.5 | 1.708 (50) | 1.625 | 1.708 | 1.729 | +0.084 | 25/53 |
| 비순환 0.5 + $W_h$ weight drop 0.5 | 1.722 (20) | 1.620 | 1.753 | 1.765 | +0.133 | 11/53 |

마지막 줄의 weight drop은 순환 가중치에만 걸었고 AWD-LSTM이 함께 쓰는 다른 장치는 넣지 않았습니다.

## 결과 해석

같은 비율 0.5를 순환 연결에 걸 때 마스크를 고정한 쪽이 낫습니다. 50 epoch 검증 PP가 per-step 1.747, per-seq 1.731이고 최저 검증 PP도 1.722와 1.717입니다. 차이는 크지 않지만 방향이 Gal, Ghahramani의 주장과 같습니다.

가장 좋았던 것은 비순환 연결의 마스크까지 시퀀스 안에서 고정한 조건입니다. 검증 PP 1.708이고, 최저값이 나온 곳이 epoch 50입니다. 50 epoch 안에서 검증 곡선이 한 번도 올라가지 않았다는 뜻입니다. 다른 네 조건은 모두 중간에 최저점을 지나 다시 올라갔습니다. 마스크를 뽑는 단위를 바꾸는 것만으로 이 차이가 났습니다.

weight drop은 1.753으로 per-step과 비슷했습니다. AWD-LSTM이 보고한 개선은 weight drop 하나가 아니라 NT-ASGD, embedding dropout, 가중치 묶기, 추가 미세 조정을 함께 쓴 결과입니다. 여기서는 weight drop만 떼어 넣었으므로 논문의 결과를 재현한 것이 아닙니다.

[dropout을 어디에 걸 것인가](02-where-to-apply-dropout.md)의 마스크 측정과 방향이 반대인 점을 적어 둘 필요가 있습니다. 거기서는 dropout 없이 학습한 모델에 마스크만 나중에 켰을 때 per-seq 0.5가 태그 이름을 맞히는 비율을 0.625에서 0.381까지 떨어뜨렸고, per-step 0.5는 0.533이었습니다. 고정 마스크가 더 많이 망가뜨렸습니다. 그런데 그 마스크로 처음부터 학습시키면 per-seq 쪽이 더 좋습니다. 두 측정이 재는 것이 다릅니다. 앞의 것은 이미 만들어진 표현이 노이즈에 얼마나 약한지이고, 뒤의 것은 그 노이즈를 견디도록 학습된 모델이 얼마나 일반화하는지입니다. 마스크가 고정되어 있으면 모델이 무엇을 대비해야 하는지가 일관되고, 그래서 적응할 여지가 생깁니다.

태그 쌍은 per-seq 마스크를 두 자리에 모두 쓴 조건이 25/53으로 dropout을 건 조건 중에서 가장 높았습니다. 다만 이 지표는 암기 정도에 크게 좌우되므로(dropout 없는 모델이 52/52) 순서만 참고할 값입니다.

## 한계와 주의할 점

- Gal, Ghahramani의 방법은 입력, 순환, 출력의 마스크를 시퀀스 단위로 고정하고 embedding에도 dropout을 겁니다. 위 실험은 마스크를 다시 뽑는 단위만 바꿨고 embedding dropout은 넣지 않았습니다. 논문의 구성을 그대로 재현한 것이 아닙니다.
- 여기서 말하는 "시퀀스"는 truncated BPTT로 끊은 50글자 조각입니다. 조각이 바뀌면 hidden state는 이어받지만 마스크는 새로 뽑습니다. 문장 하나를 통째로 넣는 설정과 다르므로, 마스크가 고정되는 길이가 조각 길이로 제한됩니다.
- 비율을 다섯 조건 모두 0.5로 맞췄습니다. 방식마다 최적 비율이 다를 수 있으므로 위 순서는 비율 0.5에서의 순서입니다.
- 합성 말뭉치 하나, seed 하나, 글자 단위 모델입니다. 검증 split이 2,355글자라 perplexity 소수 세 자리의 차이를 유의미하다고 읽으면 안 됩니다.
- weight drop은 AWD-LSTM의 여러 장치 중 하나만 떼어 넣은 것입니다. 이 결과로 weight drop이 per-seq 마스크보다 못하다고 결론짓기에는 조건이 부족합니다.

## References

- [A Theoretically Grounded Application of Dropout in Recurrent Neural Networks](https://arxiv.org/abs/1512.05287) (Gal, Ghahramani, 2016)
- [Regularizing and Optimizing LSTM Language Models](https://arxiv.org/abs/1708.02182) (Merity, Keskar, Socher, 2017)
- [Recurrent Neural Network Regularization](https://arxiv.org/abs/1409.2329) (Zaremba, Sutskever, Vinyals, 2014)
- [Regularization of Neural Networks using DropConnect](https://proceedings.mlr.press/v28/wan13.html) (Wan 등, 2013)
