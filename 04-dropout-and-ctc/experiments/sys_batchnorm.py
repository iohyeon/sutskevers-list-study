#!/usr/bin/env python3
"""순환 신경망에서 배치 정규화의 통계 범위를 비교한다.

세 가지를 잰다.
  1. 표본 수에 따른 분산 추정의 흔들림 (몬테카를로)
  2. 길이가 제각각인 배치에서 시점별 표본 수와 그때의 분산 추정 흔들림
  3. 작은 순환 모델에서 정규화 방식별 학습 곡선
     none  : 정규화 없음
     step  : 시점마다 따로 통계 (배치 방향만)
     seq   : 배치의 모든 발화, 모든 시점에 걸쳐 통계 (sequence-wise)
     ln    : layer normalization (한 발화 한 시점 안에서 특징 방향)
"""
import math
import sys
import time

import numpy as np
import torch
import torch.nn as nn

SEED = 0
D_IN = 12
N_STATE = 8
HIDDEN = 64
LAYERS = 5
T_MAX = 120
BATCH = 24
STEPS = 120
MARKS = (1, 10, 20, 30, 40, 60, 80, 100, 120)
SIGMA = 2.0
EVAL_BATCHES = 8

ENV = "환경: Python 3.14.7, NumPy 2.5.3, PyTorch 2.14.0, macOS arm64, CPU"
RUNDATE = "실행일: 2026-09-28"
CMD = "명령: .venv/bin/python sys_batchnorm.py"

LINES = []


def log(s=""):
    LINES.append(s)
    print(s, file=sys.stderr)


# ---------------------------------------------------------------- 합성 데이터
def make_lengths(rng, n):
    """발화 길이. 중앙값 45프레임인 로그정규를 [20, 120]으로 자른다."""
    x = rng.lognormal(mean=math.log(45.0), sigma=0.55, size=n)
    return np.clip(np.rint(x), 20, T_MAX).astype(int)


def make_means(rng):
    return rng.standard_normal((N_STATE, D_IN)).astype(np.float32)


def make_batch(rng, means, lengths, sigma=SIGMA):
    """상태가 평균 4.6프레임 이어지는 열을 만들고 상태 평균에 잡음을 얹는다."""
    b = len(lengths)
    t = int(lengths.max())
    x = np.zeros((b, t, D_IN), dtype=np.float32)
    y = np.zeros((b, t), dtype=np.int64)
    mask = np.zeros((b, t), dtype=np.float32)
    for i, length in enumerate(lengths):
        states = []
        while len(states) < length:
            s = int(rng.integers(N_STATE))
            d = 1 + int(rng.geometric(0.28))
            states.extend([s] * d)
        seq = np.array(states[:length])
        y[i, :length] = seq
        x[i, :length] = means[seq] + sigma * rng.standard_normal((length, D_IN)).astype(np.float32)
        mask[i, :length] = 1.0
    return torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(mask)


# ---------------------------------------------------------------- 정규화 모듈
class SeqBN(nn.Module):
    """배치의 모든 발화, 모든 시점에 걸쳐 특징별 통계를 낸다."""

    def __init__(self, n, eps=1e-5, momentum=0.1):
        super().__init__()
        self.eps = eps
        self.momentum = momentum
        self.weight = nn.Parameter(torch.ones(n))
        self.bias = nn.Parameter(torch.zeros(n))
        self.register_buffer("run_mean", torch.zeros(n))
        self.register_buffer("run_var", torch.ones(n))

    def forward(self, x, mask):
        if self.training:
            m = mask.unsqueeze(-1)
            cnt = m.sum()
            mean = (x * m).sum((0, 1)) / cnt
            var = (((x - mean) ** 2) * m).sum((0, 1)) / cnt
            with torch.no_grad():
                self.run_mean.mul_(1 - self.momentum).add_(self.momentum * mean.detach())
                self.run_var.mul_(1 - self.momentum).add_(self.momentum * var.detach())
        else:
            mean, var = self.run_mean, self.run_var
        return (x - mean) / torch.sqrt(var + self.eps) * self.weight + self.bias


class StepBN(nn.Module):
    """시점마다 따로, 그 시점에 살아 있는 발화들만으로 통계를 낸다."""

    def __init__(self, n, t_max=T_MAX, eps=1e-5, momentum=0.1):
        super().__init__()
        self.eps = eps
        self.momentum = momentum
        self.weight = nn.Parameter(torch.ones(n))
        self.bias = nn.Parameter(torch.zeros(n))
        self.register_buffer("run_mean", torch.zeros(t_max, n))
        self.register_buffer("run_var", torch.ones(t_max, n))

    def forward(self, x, mask):
        t = x.shape[1]
        if self.training:
            m = mask.unsqueeze(-1)
            cnt = m.sum(0).clamp(min=1.0)
            mean = (x * m).sum(0) / cnt
            var = (((x - mean) ** 2) * m).sum(0) / cnt
            with torch.no_grad():
                upd = (mask.sum(0) > 0).float().unsqueeze(-1) * self.momentum
                self.run_mean[:t] = self.run_mean[:t] * (1 - upd) + upd * mean.detach()
                self.run_var[:t] = self.run_var[:t] * (1 - upd) + upd * var.detach()
        else:
            mean, var = self.run_mean[:t], self.run_var[:t]
        return (x - mean) / torch.sqrt(var + self.eps) * self.weight + self.bias


class Net(nn.Module):
    """비순환 연결에 정규화를 거는 단방향 tanh 순환 모델."""

    def __init__(self, norm="none"):
        super().__init__()
        self.norm_kind = norm
        self.wx = nn.ModuleList()
        self.wh = nn.ModuleList()
        self.norms = nn.ModuleList()
        d = D_IN
        for _ in range(LAYERS):
            self.wx.append(nn.Linear(d, HIDDEN, bias=norm in ("none", "ln")))
            self.wh.append(nn.Linear(HIDDEN, HIDDEN, bias=False))
            if norm == "seq":
                self.norms.append(SeqBN(HIDDEN))
            elif norm == "step":
                self.norms.append(StepBN(HIDDEN))
            elif norm == "ln":
                self.norms.append(nn.LayerNorm(HIDDEN))
            d = HIDDEN
        self.out = nn.Linear(HIDDEN, N_STATE)

    def forward(self, x, mask):
        b, t = x.shape[0], x.shape[1]
        cur = x
        for i in range(LAYERS):
            pre = self.wx[i](cur)
            if self.norm_kind in ("seq", "step"):
                pre = self.norms[i](pre, mask)
            h = torch.zeros(b, HIDDEN)
            outs = []
            for k in range(t):
                a = pre[:, k] + self.wh[i](h)
                if self.norm_kind == "ln":
                    a = self.norms[i](a)
                h = torch.tanh(a)
                outs.append(h)
            cur = torch.stack(outs, 1)
        return self.out(cur)


def masked_loss(logits, y, mask):
    ll = nn.functional.cross_entropy(
        logits.reshape(-1, N_STATE), y.reshape(-1), reduction="none"
    ).reshape(y.shape)
    return (ll * mask).sum() / mask.sum()


def frame_error(logits, y, mask):
    wrong = (logits.argmax(-1) != y).float()
    return ((wrong * mask).sum() / mask.sum()).item()


# ---------------------------------------------------------------- 실험 1
def exp_variance_spread():
    log("[1] 표본 n개로 분산을 추정할 때의 흔들림 (표준정규, 복제 40,000회, 불편추정량)")
    log("")
    log("      n   추정 분산 평균   추정 분산 표준편차   상대 표준편차   이론값 sqrt(2/(n-1))")
    rng = np.random.default_rng(SEED)
    reps = 40000
    for n in (2, 4, 8, 16, 32, 64, 128, 256, 1024):
        z = rng.standard_normal((reps, n))
        v = z.var(axis=1, ddof=1)
        rel = v.std() / v.mean()
        log(f"   {n:5d}   {v.mean():13.4f}   {v.std():18.4f}   {rel:13.4f}   {math.sqrt(2.0/(n-1)):20.4f}")
    log("")


# ---------------------------------------------------------------- 실험 2
def exp_step_counts():
    log("[2] 배치 24개 발화(길이 로그정규, 중앙값 45, [20,120] 절단)에서 시점별 표본 수")
    log("")
    rng = np.random.default_rng(SEED + 1)
    reps = 3000
    lens = [make_lengths(rng, BATCH) for _ in range(reps)]
    arr = np.zeros((reps, T_MAX), dtype=np.float64)
    for i, ln in enumerate(lens):
        for length in ln:
            arr[i, :length] += 1.0
    log("   시점 t   살아 있는 발화 수 평균   최소   최대   그 표본 수에서 분산 추정의 상대 표준편차")
    for t in (1, 10, 20, 30, 45, 60, 80, 100, 120):
        col = arr[:, t - 1]
        nbar = col.mean()
        rel = math.sqrt(2.0 / (nbar - 1)) if nbar > 1.5 else float("nan")
        log(f"   {t:6d}   {nbar:22.2f}   {col.min():4.0f}   {col.max():4.0f}   {rel:42.4f}")
    total = float(arr.sum(axis=1).mean())
    log("")
    log(f"   한 배치의 유효 프레임 수 평균: {total:.1f}")
    log(f"   sequence-wise 통계는 이 {total:.0f}개 전부로 특징별 통계 1벌을 만든다.")
    log(f"   시점별 통계는 같은 배치에서 최대 {int(arr.mean(axis=0).nonzero()[0].max())+1}벌을 만들고, 뒤쪽 벌은 표본이 몇 개뿐이다.")
    log("")

    log("   같은 설정에서 특징 하나(표준정규)의 분산 추정을 시점별로 실제로 재 본 결과")
    log("")
    log("   (분산 통계는 그 시점에 발화가 2개 이상 남은 배치만 모았다)\n   시점 t   표본 수 평균   추정 분산 평균   추정 분산 표준편차   1/sqrt(추정분산) 의 표준편차")
    g = np.random.default_rng(SEED + 2)
    for t in (1, 20, 45, 80, 110):
        vs = []
        for i in range(reps):
            n = int(arr[i, t - 1])
            if n < 2:
                continue
            z = g.standard_normal(n)
            vs.append(z.var(ddof=1))
        vs = np.array(vs)
        inv = 1.0 / np.sqrt(vs + 1e-5)
        log(f"   {t:6d}   {arr[:, t-1].mean():12.2f}   {vs.mean():13.4f}   {vs.std():18.4f}   {inv.std():27.4f}")
    log("")


# ---------------------------------------------------------------- 실험 3
def train_one(norm, lr, steps=STEPS):
    torch.manual_seed(SEED)
    net = Net(norm)
    rng = np.random.default_rng(SEED + 7)
    means = make_means(np.random.default_rng(SEED + 3))
    opt = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    curve = []
    diverged = False
    net.train()
    for s in range(1, steps + 1):
        lengths = make_lengths(rng, BATCH)
        x, y, mask = make_batch(rng, means, lengths)
        logits = net(x, mask)
        loss = masked_loss(logits, y, mask)
        opt.zero_grad()
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(net.parameters(), 1e9).item()
        if not math.isfinite(loss.item()):
            diverged = True
            curve.append((s, float("nan"), gnorm))
            break
        opt.step()
        if s in MARKS:
            curve.append((s, loss.item(), gnorm))
    net.eval()
    ev = np.random.default_rng(SEED + 99)
    errs, losses = [], []
    with torch.no_grad():
        for _ in range(EVAL_BATCHES):
            lengths = make_lengths(ev, BATCH)
            x, y, mask = make_batch(ev, means, lengths)
            logits = net(x, mask)
            losses.append(masked_loss(logits, y, mask).item())
            errs.append(frame_error(logits, y, mask))
    return curve, float(np.mean(losses)), float(np.mean(errs)), diverged


def exp_training(lr):
    log(f"[3] 정규화 방식별 학습 곡선 (lr={lr}, SGD momentum 0.9, {STEPS}스텝, 배치 {BATCH}, hidden {HIDDEN}, layer {LAYERS})")
    log("")
    header = "   방식    " + "".join(f"{s:>9d}" for s in MARKS)
    log("   학습 loss (스텝별)")
    log(header)
    res = {}
    for norm in ("none", "step", "seq", "ln"):
        t0 = time.time()
        curve, ev_loss, ev_err, diverged = train_one(norm, lr)
        res[norm] = (curve, ev_loss, ev_err, diverged, time.time() - t0)
        row = "   " + f"{norm:7s} " + "".join(
            ("      nan" if not math.isfinite(v) else f"{v:9.4f}") for _, v, _ in curve
        )
        log(row)
    log("")
    log("   평가(추론 모드, 새 발화 8배치)")
    log("   방식      frame loss   frame error rate   학습 중 발산   학습 시간(초)")
    for norm in ("none", "step", "seq", "ln"):
        curve, ev_loss, ev_err, diverged, el = res[norm]
        log(f"   {norm:7s}   {ev_loss:10.4f}   {ev_err:16.4f}   {'예' if diverged else '아니오':>12s}   {el:13.1f}")
    log("")
    return res


def main():
    t0 = time.time()
    torch.set_num_threads(4)
    exp_variance_spread()
    exp_step_counts()
    r1 = exp_training(0.1)
    r2 = exp_training(0.4)
    log("[4] 학습률 0.1과 0.4에서의 frame error rate 비교")
    log("")
    log("   방식      lr=0.1    lr=0.4")
    for norm in ("none", "step", "seq", "ln"):
        log(f"   {norm:7s}   {r1[norm][2]:6.4f}    {r2[norm][2]:6.4f}")
    elapsed = time.time() - t0

    cfg = (
        f"설정: 합성 프레임 분류. 상태 {N_STATE}개, 특징 {D_IN}차원, 상태 지속 평균 4.6프레임, "
        f"잡음 표준편차 2.0, 길이 로그정규(중앙값 45, [20,{T_MAX}] 절단), seed {SEED} 고정"
    )
    out = [ENV, RUNDATE, CMD, cfg, f"소요: {elapsed:.0f}초", ""] + LINES
    print("\n".join(out))


if __name__ == "__main__":
    main()
