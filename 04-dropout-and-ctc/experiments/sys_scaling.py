#!/usr/bin/env python3
"""학습 데이터를 늘릴 때 오류율이 줄어드는 모양을 잰다.

  1. Deep Speech 2 논문 표 10의 숫자로 구간별 상대 감소율 계산
  2. WER = c * D^(-alpha) 의 지수를 로그-로그 최소제곱으로 추정
  3. 합성 과제에서 학습 데이터 크기를 1, 3, 10, 30배로 바꿔 오류율 측정

합성 데이터 생성기는 sys_batchnorm.py 의 것을 그대로 쓴다.
"""
import math
import sys
import time

import numpy as np
import torch
import torch.nn as nn

import sys_batchnorm as base

SEED = 0
HIDDEN = 64
LAYERS = 2
N_STATE = base.N_STATE
D_IN = base.D_IN
BASE_SIZE = 32
FACTORS = (1, 3, 10, 30, 100)
STEPS = 1500
BATCH = 16
TEST_UTT = 512
SEEDS = (0, 1, 2)

# Deep Speech 2 (Amodei 등, 2015) 표 10. dev 세트 WER(%)
HOURS = np.array([120.0, 1200.0, 2400.0, 6000.0, 12000.0])
WER_REGULAR = np.array([29.23, 13.80, 11.65, 9.51, 8.46])
WER_NOISY = np.array([50.97, 22.99, 20.41, 15.90, 13.59])

ENV = "환경: Python 3.14.7, NumPy 2.5.3, PyTorch 2.14.0, macOS arm64, CPU"
RUNDATE = "실행일: 2026-09-28"
CMD = "명령: .venv/bin/python sys_scaling.py"

LINES = []


def log(s=""):
    LINES.append(s)
    print(s, file=sys.stderr)


# ------------------------------------------------- 1. 논문 표의 구간별 감소율
def exp_table10():
    log("[1] 논문 표 10의 구간별 상대 감소율 (dev 세트 WER)")
    log("")
    log("   구간(시간)          배수    regular WER        상대 감소   noisy WER        상대 감소")
    for i in range(len(HOURS) - 1):
        d0, d1 = HOURS[i], HOURS[i + 1]
        r0, r1 = WER_REGULAR[i], WER_REGULAR[i + 1]
        n0, n1 = WER_NOISY[i], WER_NOISY[i + 1]
        log(
            f"   {d0:6.0f} -> {d1:6.0f}   {d1/d0:5.2f}x   {r0:5.2f} -> {r1:5.2f}"
            f"   {(r0-r1)/r0*100:13.2f}%   {n0:5.2f} -> {n1:5.2f}   {(n0-n1)/n0*100:13.2f}%"
        )
    log("")
    log("   정확히 10배인 두 구간만 따로 본다")
    log("")
    for lo, hi in ((0, 1), (1, 4)):
        d0, d1 = HOURS[lo], HOURS[hi]
        r0, r1 = WER_REGULAR[lo], WER_REGULAR[hi]
        n0, n1 = WER_NOISY[lo], WER_NOISY[hi]
        log(
            f"   {d0:6.0f}시간 -> {d1:6.0f}시간 ({d1/d0:.0f}배):"
            f"  regular {(r0-r1)/r0*100:5.2f}% 감소,  noisy {(n0-n1)/n0*100:5.2f}% 감소"
        )
    log("")


# ------------------------------------------------- 2. 멱법칙 지수 추정
def fit_power(x, y):
    """log y = log c - alpha * log x 를 최소제곱으로 푼다."""
    lx, ly = np.log10(x), np.log10(y)
    slope, intercept = np.polyfit(lx, ly, 1)
    pred = 10 ** (intercept + slope * lx)
    ss_res = float(((ly - (intercept + slope * lx)) ** 2).sum())
    ss_tot = float(((ly - ly.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot
    return -slope, 10 ** intercept, r2, pred


def exp_power_law():
    log("[2] WER = c * D^(-alpha) 의 지수 추정 (로그-로그 최소제곱)")
    log("")
    log("   대상                        alpha      c        R^2    10배당 감소율 1-10^(-alpha)")
    cases = [
        ("regular, 5점 전부", HOURS, WER_REGULAR),
        ("noisy, 5점 전부", HOURS, WER_NOISY),
        ("regular, 1200시간 이상 4점", HOURS[1:], WER_REGULAR[1:]),
        ("noisy, 1200시간 이상 4점", HOURS[1:], WER_NOISY[1:]),
    ]
    fits = {}
    for name, x, y in cases:
        alpha, c, r2, pred = fit_power(x, y)
        fits[name] = (alpha, c, r2, x, y, pred)
        log(f"   {name:26s}  {alpha:7.4f}  {c:7.2f}  {r2:7.4f}  {(1-10**(-alpha))*100:22.2f}%")
    log("")
    log("   regular 5점 적합의 잔차")
    log("")
    log("      시간   실제 WER   적합값   차이")
    alpha, c, r2, x, y, pred = fits["regular, 5점 전부"]
    for xi, yi, pi in zip(x, y, pred):
        log(f"   {xi:7.0f}   {yi:8.2f}   {pi:6.2f}   {yi-pi:+5.2f}")
    log("")
    log(f"   지수 {alpha:.4f} 는 10배당 {(1-10**(-alpha))*100:.1f}% 감소, 2배당 {(1-2**(-alpha))*100:.1f}% 감소에 해당한다.")
    log(f"   논문이 적은 '10배당 상대 40%'를 지수로 되돌리면 alpha = {-math.log10(0.6):.4f} 이다.")
    log("")


# ------------------------------------------------- 3. 작은 규모의 데이터 스케일링
class RNNNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.RNN(D_IN, HIDDEN, num_layers=LAYERS, nonlinearity="tanh", batch_first=True)
        self.out = nn.Linear(HIDDEN, N_STATE)

    def forward(self, x):
        h, _ = self.rnn(x)
        return self.out(h)


def frame_stats(net, x, y, mask):
    with torch.no_grad():
        logits = net(x)
        ll = nn.functional.cross_entropy(
            logits.reshape(-1, N_STATE), y.reshape(-1), reduction="none"
        ).reshape(y.shape)
        loss = float((ll * mask).sum() / mask.sum())
        err = float(((logits.argmax(-1) != y).float() * mask).sum() / mask.sum())
    return loss, err


def make_pool(rng, means, n):
    items = []
    for length in base.make_lengths(rng, n):
        x, y, mask = base.make_batch(rng, means, np.array([length]))
        items.append((x[0], y[0], mask[0]))
    return items


def collate(items, ids):
    t = max(int(items[i][0].shape[0]) for i in ids)
    x = torch.zeros(len(ids), t, D_IN)
    y = torch.zeros(len(ids), t, dtype=torch.long)
    mask = torch.zeros(len(ids), t)
    for r, i in enumerate(ids):
        xi, yi, mi = items[i]
        length = xi.shape[0]
        x[r, :length] = xi
        y[r, :length] = yi
        mask[r, :length] = mi
    return x, y, mask


def train_size(items, n_use, seed, test):
    torch.manual_seed(seed)
    net = RNNNet()
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    rng = np.random.default_rng(seed + 900)
    sub = list(range(n_use))
    for _ in range(STEPS):
        ids = rng.choice(sub, size=min(BATCH, n_use), replace=n_use < BATCH)
        x, y, mask = collate(items, list(ids))
        logits = net(x)
        ll = nn.functional.cross_entropy(
            logits.reshape(-1, N_STATE), y.reshape(-1), reduction="none"
        ).reshape(y.shape)
        loss = (ll * mask).sum() / mask.sum()
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 5.0)
        opt.step()
    net.eval()
    tr = collate(items, sub[: min(len(sub), 256)])
    tr_loss, tr_err = frame_stats(net, *tr)
    te_loss, te_err = frame_stats(net, *test)
    return tr_loss, tr_err, te_loss, te_err


def exp_scaling():
    means = base.make_means(np.random.default_rng(SEED + 3))
    rng = np.random.default_rng(SEED + 21)
    n_max = BASE_SIZE * max(FACTORS)
    pool = make_pool(rng, means, n_max)
    frames = np.array([int(it[2].sum()) for it in pool])
    trng = np.random.default_rng(SEED + 55)
    test_items = make_pool(trng, means, TEST_UTT)
    test = collate(test_items, list(range(TEST_UTT)))

    log("[3] 합성 프레임 분류에서 학습 데이터 크기만 바꿔 측정한 오류율")
    log(f"    모델 고정(tanh RNN hidden {HIDDEN}, layer {LAYERS}), Adam 3e-3, {STEPS}스텝, 배치 {BATCH}, seed {len(SEEDS)}개")
    log(f"    시험 세트는 새 발화 {TEST_UTT}개({int(test[2].sum())}프레임), 모든 크기에서 같다")
    log("")
    log("    배수    학습 발화   학습 프레임   train frame error   test frame error   표준편차   test frame loss")
    rows = []
    for f in FACTORS:
        n_use = BASE_SIZE * f
        tr_errs, te_errs, te_losses = [], [], []
        for seed in SEEDS:
            tr_loss, tr_err, te_loss, te_err = train_size(pool, n_use, seed, test)
            tr_errs.append(tr_err)
            te_errs.append(te_err)
            te_losses.append(te_loss)
        rows.append((f, n_use, int(frames[:n_use].sum()), np.mean(tr_errs), np.mean(te_errs), np.std(te_errs), np.mean(te_losses)))
        log(
            f"    {f:4d}x   {n_use:9d}   {int(frames[:n_use].sum()):11d}   {np.mean(tr_errs):17.4f}"
            f"   {np.mean(te_errs):16.4f}   {np.std(te_errs):9.4f}   {np.mean(te_losses):15.4f}"
        )
    log("")

    d = np.array([r[2] for r in rows], dtype=float)
    e = np.array([r[4] for r in rows], dtype=float)
    alpha, c, r2, pred = fit_power(d, e)
    log("    로그-로그 최소제곱 적합 (x = 학습 프레임 수, y = test frame error)")
    log("")
    log("    적합 범위            alpha      c        R^2    10배당 감소율")
    log(f"    {'1x~100x 5점 전부':20s} {alpha:7.4f}  {c:6.4f}  {r2:7.4f}  {(1-10**(-alpha))*100:11.2f}%")
    a2, c2, r22, _ = fit_power(d[1:4], e[1:4])
    log(f"    {'3x~30x 3점':20s} {a2:7.4f}  {c2:6.4f}  {r22:7.4f}  {(1-10**(-a2))*100:11.2f}%")
    a3, c3, r23, _ = fit_power(d[2:], e[2:])
    log(f"    {'10x~100x 3점':20s} {a3:7.4f}  {c3:6.4f}  {r23:7.4f}  {(1-10**(-a3))*100:11.2f}%")
    log("")
    log("    구간별 상대 감소율")
    log("")
    log("    구간      배수    test frame error        상대 감소")
    for i in range(len(rows) - 1):
        e0, e1 = rows[i][4], rows[i + 1][4]
        log(f"    {rows[i][0]:2d}x -> {rows[i+1][0]:2d}x   {rows[i+1][0]/rows[i][0]:4.2f}x   {e0:.4f} -> {e1:.4f}   {(e0-e1)/e0*100:13.2f}%")
    log("")
    log("    잔차")
    log("")
    log("    학습 프레임   실제 error   적합값   차이")
    for di, ei, pi in zip(d, e, pred):
        log(f"    {di:11.0f}   {ei:10.4f}   {pi:6.4f}   {ei-pi:+7.4f}")
    log("")
    return rows, alpha


def main():
    t0 = time.time()
    torch.set_num_threads(4)
    exp_table10()
    exp_power_law()
    exp_scaling()
    elapsed = time.time() - t0
    cfg = (
        "설정: 논문 표 10은 인용한 숫자. 합성 실험은 sys_batchnorm.py 와 같은 생성기"
        f"(상태 {N_STATE}개, 특징 {D_IN}차원, 잡음 표준편차 {base.SIGMA}), seed {SEED} 고정"
    )
    print("\n".join([ENV, RUNDATE, CMD, cfg, f"소요: {elapsed:.0f}초", ""] + LINES))


if __name__ == "__main__":
    main()
