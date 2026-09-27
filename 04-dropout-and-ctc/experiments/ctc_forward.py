"""CTC의 forward 변수 점화식을 NumPy로 구현하고 세 가지로 검증한다.

1. 작은 예(프레임 6개, 라벨 "cat")에서 모든 경로를 완전 탐색한 확률 합과
   동적 계획법(forward 변수)의 결과가 같은지 본다. Fraction 으로 정확히 비교한다.
2. torch.nn.functional.ctc_loss 의 값과 비교한다.
3. forward-backward 로 구한 logit 기울기를 중앙 차분과 비교한다.
4. 직접 곱 공간과 로그 공간의 수치 범위 차이를 잰다.

사용: python3 ctc_forward.py
"""
import itertools
import time
from fractions import Fraction

import numpy as np
import torch
import torch.nn.functional as F

BLANK = 0
SYMS = ["-", "c", "a", "t"]  # 0번이 blank

# 프레임 6개짜리 합성 확률 행렬. 손계산을 맞추기 위해 소수 한 자리로 골랐다.
P6 = [
    [0.1, 0.6, 0.2, 0.1],
    [0.2, 0.5, 0.2, 0.1],
    [0.1, 0.1, 0.7, 0.1],
    [0.2, 0.1, 0.6, 0.1],
    [0.1, 0.1, 0.2, 0.6],
    [0.2, 0.1, 0.2, 0.5],
]


def collapse(path):
    """CTC 접기 규칙: 연속 중복을 하나로 줄인 뒤 blank 를 지운다."""
    out = []
    prev = None
    for k in path:
        if k != prev:
            out.append(k)
        prev = k
    return tuple(k for k in out if k != BLANK)


def extend(labels):
    """라벨 사이와 양 끝에 blank 를 끼운 확장 라벨 z 를 만든다."""
    z = [BLANK]
    for k in labels:
        z += [k, BLANK]
    return z


def forward_table(probs, labels, num=float):
    """alpha[s][t]. num=Fraction 이면 정확한 유리수로 계산한다."""
    z = extend(labels)
    T, S = len(probs), len(z)
    zero = num(0)
    a = [[zero] * T for _ in range(S)]
    a[0][0] = num(probs[0][z[0]])
    if S > 1:
        a[1][0] = num(probs[0][z[1]])
    for t in range(1, T):
        for s in range(S):
            acc = a[s][t - 1]
            if s >= 1:
                acc = acc + a[s - 1][t - 1]
            if s >= 2 and z[s] != BLANK and z[s] != z[s - 2]:
                acc = acc + a[s - 2][t - 1]
            a[s][t] = num(probs[t][z[s]]) * acc
    return z, a


def backward_table(probs, labels):
    z = extend(labels)
    T, S = len(probs), len(z)
    b = np.zeros((S, T))
    b[S - 1][T - 1] = probs[T - 1][z[S - 1]]
    if S >= 2:
        b[S - 2][T - 1] = probs[T - 1][z[S - 2]]
    for t in range(T - 2, -1, -1):
        for s in range(S):
            acc = b[s][t + 1]
            if s + 1 < S:
                acc += b[s + 1][t + 1]
            if s + 2 < S and z[s] != BLANK and z[s] != z[s + 2]:
                acc += b[s + 2][t + 1]
            b[s][t] = probs[t][z[s]] * acc
    return z, b


def total_prob(probs, labels, num=float):
    z, a = forward_table(probs, labels, num)
    S, T = len(z), len(probs)
    if S == 1:
        return a[0][T - 1]
    return a[S - 1][T - 1] + a[S - 2][T - 1]


def logsumexp(xs):
    m = max(xs)
    if m == -np.inf:
        return -np.inf
    return m + np.log(sum(np.exp(x - m) for x in xs))


def log_total(logprobs, labels):
    """로그 공간 forward. logprobs[t][k] = log p_t(k)."""
    z = extend(labels)
    T, S = len(logprobs), len(z)
    a = [[-np.inf] * T for _ in range(S)]
    a[0][0] = logprobs[0][z[0]]
    if S > 1:
        a[1][0] = logprobs[0][z[1]]
    for t in range(1, T):
        for s in range(S):
            terms = [a[s][t - 1]]
            if s >= 1:
                terms.append(a[s - 1][t - 1])
            if s >= 2 and z[s] != BLANK and z[s] != z[s - 2]:
                terms.append(a[s - 2][t - 1])
            a[s][t] = logprobs[t][z[s]] + logsumexp(terms)
    if S == 1:
        return a[0][T - 1]
    return logsumexp([a[S - 1][T - 1], a[S - 2][T - 1]])


def enumerate_paths(probs, labels, vocab):
    """모든 경로를 완전 탐색해서 접은 결과가 labels 인 것의 확률을 더한다."""
    T = len(probs)
    target = tuple(labels)
    total = Fraction(0)
    count = 0
    by_label = {}
    for path in itertools.product(range(vocab), repeat=T):
        pr = Fraction(1)
        for t, k in enumerate(path):
            pr *= Fraction(probs[t][k]).limit_denominator(10)
        lab = collapse(path)
        by_label[lab] = by_label.get(lab, Fraction(0)) + pr
        if lab == target:
            total += pr
            count += 1
    return total, count, by_label


def ctc_grad_numpy(logits, labels):
    """forward-backward 로 -ln P 의 logit 기울기를 구한다. dL/dy_tk = p_tk - gamma_tk."""
    logits = np.asarray(logits, dtype=np.float64)
    T, K = logits.shape
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    p = e / e.sum(axis=1, keepdims=True)
    z, a = forward_table(p.tolist(), labels, float)
    _, b = backward_table(p.tolist(), labels)
    a = np.array(a)
    S = len(z)
    P = a[S - 1][T - 1] + (a[S - 2][T - 1] if S >= 2 else 0.0)
    g = p.copy()
    for t in range(T):
        for s in range(S):
            k = z[s]
            if p[t][k] > 0:
                g[t][k] -= a[s][t] * b[s][t] / (p[t][k] * P)
    return -np.log(P), g, p


def main():
    t0 = time.time()
    labels = [1, 2, 3]  # "cat"
    frac = [[Fraction(x).limit_denominator(10) for x in row] for row in P6]

    print("== 1. 프레임 6개, 라벨 \"cat\" ==")
    print("확장 라벨 z =", " ".join(SYMS[k] for k in extend(labels)), f"(길이 {2 * len(labels) + 1})")
    z, af = forward_table(frac, labels, Fraction)
    print("\nalpha 표 (행 = z 의 상태, 열 = 프레임)")
    head = "  s  z  " + "".join(f"{t + 1:>13}" for t in range(len(P6)))
    print(head)
    for s in range(len(z)):
        row = f"{s + 1:>3}  {SYMS[z[s]]}  "
        for t in range(len(P6)):
            row += f"{float(af[s][t]):>13.8f}"
        print(row)

    S, T = len(z), len(P6)
    dp = af[S - 1][T - 1] + af[S - 2][T - 1]
    print(f"\nDP:   P(cat|x) = alpha_6(6) + alpha_6(7) = {float(af[S - 2][T - 1]):.10f} + "
          f"{float(af[S - 1][T - 1]):.10f} = {float(dp):.10f}")
    print(f"      정확값 = {dp} ")

    brute, count, by_label = enumerate_paths(frac, labels, len(SYMS))
    print(f"완전탐색: 경로 {len(SYMS)}^{T} = {len(SYMS) ** T}개 중 접으면 \"cat\"이 되는 경로 {count}개")
    print(f"      P(cat|x) = {float(brute):.10f}")
    print(f"      정확값 = {brute}")
    print(f"두 값의 차이 = {dp - brute}  (Fraction 으로 정확히 0인지 비교: {dp == brute})")
    print(f"-ln P = {-np.log(float(dp)):.10f}")

    print("\n접은 결과별 확률 상위 8개 (전체 합 = 1)")
    tot = sum(by_label.values())
    for lab, pr in sorted(by_label.items(), key=lambda kv: -kv[1])[:8]:
        name = "".join(SYMS[k] for k in lab) or "(빈 문자열)"
        print(f"  {name:>8}  {float(pr):.6f}")
    print(f"  합계 = {float(tot):.10f}, 라벨 종류 {len(by_label)}개")

    print("\n== 2. torch.nn.functional.ctc_loss 와 비교 ==")
    lp = torch.log(torch.tensor(P6, dtype=torch.float64)).unsqueeze(1)
    tl = F.ctc_loss(lp, torch.tensor([labels]), torch.tensor([T]), torch.tensor([len(labels)]),
                    blank=BLANK, reduction="none")
    mine = -np.log(float(dp))
    print(f"직접 구현 -ln P = {mine:.12f}")
    print(f"torch ctc_loss  = {float(tl[0]):.12f}")
    print(f"차이 = {abs(mine - float(tl[0])):.3e}")

    print("\n== 3. 기울기와 중앙 차분 ==")
    rng = np.random.default_rng(0)
    for T2, K2, lab in [(12, 6, [1, 2, 2, 3]), (20, 8, [1, 3, 3, 5, 2, 7]), (6, 4, labels)]:
        logits = rng.normal(0.0, 1.5, size=(T2, K2))
        loss, g, _ = ctc_grad_numpy(logits, lab)
        eps = 1e-6
        worst = 0.0
        for t in range(T2):
            for k in range(K2):
                old = logits[t][k]
                logits[t][k] = old + eps
                lpos = ctc_grad_numpy(logits, lab)[0]
                logits[t][k] = old - eps
                lneg = ctc_grad_numpy(logits, lab)[0]
                logits[t][k] = old
                numg = (lpos - lneg) / (2 * eps)
                den = abs(numg) + abs(g[t][k])
                if den > 0:
                    worst = max(worst, abs(numg - g[t][k]) / den)
        tlp = torch.tensor(logits, dtype=torch.float64, requires_grad=True)
        tloss = F.ctc_loss(F.log_softmax(tlp, dim=1).unsqueeze(1), torch.tensor([lab]),
                           torch.tensor([T2]), torch.tensor([len(lab)]), blank=BLANK, reduction="none")
        tloss.backward()
        tg = tlp.grad.numpy()
        dg = np.abs(tg - g).max()
        print(f"T={T2:>2} K={K2} 라벨 {''.join(str(x) for x in lab):<8} "
              f"-lnP={loss:.6f}  torch={float(tloss[0].detach()):.6f}  "
              f"중앙차분 상대오차 최대 {worst:.3e}  torch 기울기와 절대차 최대 {dg:.3e}")

    print("\n== 4. 직접 곱 공간의 언더플로 ==")
    rng = np.random.default_rng(1)
    for T3 in (50, 100, 200, 400, 800):
        K3 = 30
        logits = rng.normal(0.0, 1.0, size=(T3, K3))
        e = np.exp(logits - logits.max(axis=1, keepdims=True))
        p = e / e.sum(axis=1, keepdims=True)
        lab = [1 + (i % 25) for i in range(T3 // 6)]
        direct = total_prob(p.tolist(), lab, float)
        logv = log_total(np.log(p).tolist(), lab)
        ok = "정상" if direct > 0 else "0으로 내려앉음"
        print(f"T={T3:>3} |y|={len(lab):>3}  직접 곱 P={direct:.3e} ({ok})  "
              f"로그 공간 lnP={logv:.4f} -> P={np.exp(logv):.3e}")

    print(f"\n소요 {time.time() - t0:.1f}초")


if __name__ == "__main__":
    main()
