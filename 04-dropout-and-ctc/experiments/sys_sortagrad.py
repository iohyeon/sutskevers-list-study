#!/usr/bin/env python3
"""길이순 배치가 무엇을 바꾸는지 잰다.

  1. 패딩 낭비: 무작위 배치와 길이순 배치의 패딩 비율 (계산식과 실측)
  2. 초기 시점의 loss와 기울기 크기가 발화 길이에 따라 어떻게 커지는지
  3. 첫 epoch만 정렬(SortaGrad), 계속 정렬, 계속 섞기의 비교

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

ENV = "환경: Python 3.14.7, NumPy 2.5.3, PyTorch 2.14.0, macOS arm64, CPU"
RUNDATE = "실행일: 2026-09-28"
CMD = "명령: .venv/bin/python sys_sortagrad.py"

LINES = []


def log(s=""):
    LINES.append(s)
    print(s, file=sys.stderr)


# ---------------------------------------------------------------- 1. 패딩 낭비
def pad_ratio(lengths, batch, order):
    n = len(lengths)
    idx = np.argsort(lengths, kind="stable") if order == "sorted" else np.arange(n)
    real = 0
    alloc = 0
    for s in range(0, n - n % batch, batch):
        chunk = lengths[idx[s:s + batch]]
        real += int(chunk.sum())
        alloc += int(chunk.max()) * batch
    return 1.0 - real / alloc


def exp_padding():
    rng = np.random.default_rng(SEED)
    n = 2048
    log("[1] 패딩 비율 = 1 - (실제 프레임 수) / (패딩까지 채운 칸 수)")
    log("")
    log("   (a) 길이가 1~2000 사이 균등분포인 발화 2,048개, 복제 200회 평균")
    log("")
    log("   배치 B   무작위 순서 실측   계산식 (B-1)/(2B)   길이순 실측   계산식 1/(G+1), G=2048/B")
    for b in (8, 16, 32, 64, 128):
        r, s = [], []
        for _ in range(200):
            lengths = rng.integers(1, 2001, size=n)
            r.append(pad_ratio(lengths, b, "random"))
            s.append(pad_ratio(lengths, b, "sorted"))
        g = n // b
        log(f"   {b:6d}   {np.mean(r):16.4f}   {(b-1)/(2*b):19.4f}   {np.mean(s):12.4f}   {1.0/(g+1):26.4f}")
    log("")

    log("   (b) 이 저장소 실험의 길이 분포(로그정규 중앙값 45, [20,120] 절단) 발화 2,048개, 복제 200회 평균")
    log("")
    log("   배치 B   무작위 순서   길이순   줄어든 패딩 비율")
    for b in (8, 16, 32, 64, 128):
        r, s = [], []
        for _ in range(200):
            lengths = base.make_lengths(rng, n)
            r.append(pad_ratio(lengths, b, "random"))
            s.append(pad_ratio(lengths, b, "sorted"))
        log(f"   {b:6d}   {np.mean(r):11.4f}   {np.mean(s):6.4f}   {np.mean(r)-np.mean(s):16.4f}")
    log("")

    log("   (c) 음성 말뭉치처럼 길이 폭이 넓을 때(로그정규 중앙값 500프레임, sigma 0.9, 절단 없음)")
    log("")
    log("   배치 B   무작위 순서   길이순   줄어든 패딩 비율")
    for b in (8, 16, 32, 64, 128):
        r, s = [], []
        for _ in range(200):
            lengths = np.rint(rng.lognormal(math.log(500.0), 0.9, size=n)).astype(int)
            r.append(pad_ratio(lengths, b, "random"))
            s.append(pad_ratio(lengths, b, "sorted"))
        log(f"   {b:6d}   {np.mean(r):11.4f}   {np.mean(s):6.4f}   {np.mean(r)-np.mean(s):16.4f}")
    log("")


# ---------------------------------------------------------------- 모델
class RNNNet(nn.Module):
    def __init__(self, hidden=HIDDEN, layers=LAYERS):
        super().__init__()
        self.rnn = nn.RNN(D_IN, hidden, num_layers=layers, nonlinearity="tanh", batch_first=True)
        self.out = nn.Linear(hidden, N_STATE)

    def forward(self, x):
        h, _ = self.rnn(x)
        return self.out(h)


def per_utt_loss(logits, y, mask):
    """발화 하나당 프레임 loss의 합. 길이에 비례해서 커진다."""
    ll = nn.functional.cross_entropy(
        logits.reshape(-1, N_STATE), y.reshape(-1), reduction="none"
    ).reshape(y.shape)
    return (ll * mask).sum() / y.shape[0]


def per_frame_loss(logits, y, mask):
    ll = nn.functional.cross_entropy(
        logits.reshape(-1, N_STATE), y.reshape(-1), reduction="none"
    ).reshape(y.shape)
    return (ll * mask).sum() / mask.sum()


def grad_norm(net):
    s = 0.0
    for p in net.parameters():
        if p.grad is not None:
            s += float(p.grad.detach().pow(2).sum())
    return math.sqrt(s)


# ------------------------------------------------- 2. 길이에 따른 loss와 기울기
def exp_length_scaling():
    log("[2] 초기 파라미터에서 배치의 발화 길이만 바꿔 가며 잰 loss와 기울기 노름")
    log("    (배치 32, 같은 초기값, 배치 안의 길이는 모두 같게 맞췄다. 복제 20회 평균)")
    log("")
    means = base.make_means(np.random.default_rng(SEED + 3))
    log("    길이 T   프레임당 loss   발화당 loss(합)   기울기 노름(발화당 loss)   기울기 노름(프레임당 loss)")
    rows = []
    for t in (10, 20, 40, 80, 160, 320):
        f_l, u_l, g_u, g_f = [], [], [], []
        for rep in range(20):
            torch.manual_seed(SEED)
            net = RNNNet()
            rng = np.random.default_rng(1000 + rep)
            x, y, mask = base.make_batch(rng, means, np.full(32, t))
            logits = net(x)
            lu = per_utt_loss(logits, y, mask)
            net.zero_grad()
            lu.backward()
            g_u.append(grad_norm(net))
            logits = net(x)
            lf = per_frame_loss(logits, y, mask)
            net.zero_grad()
            lf.backward()
            g_f.append(grad_norm(net))
            u_l.append(lu.item())
            f_l.append(lf.item())
        rows.append((t, np.mean(f_l), np.mean(u_l), np.mean(g_u), np.mean(g_f)))
        log(f"    {t:6d}   {np.mean(f_l):13.4f}   {np.mean(u_l):15.3f}   {np.mean(g_u):24.3f}   {np.mean(g_f):26.4f}")
    log("")
    t = np.array([r[0] for r in rows], dtype=float)
    for name, col in (("발화당 loss", 2), ("기울기 노름(발화당 loss)", 3), ("기울기 노름(프레임당 loss)", 4)):
        v = np.array([r[col] for r in rows])
        slope = np.polyfit(np.log(t), np.log(v), 1)[0]
        log(f"    {name} 의 로그-로그 기울기: {slope:.3f}  (T^{slope:.2f} 에 비례)")
    log("")


# ------------------------------------------------- 3. 교육과정 비교
def make_dataset(rng, means, n):
    lengths = base.make_lengths(rng, n)
    items = []
    for length in lengths:
        x, y, mask = base.make_batch(rng, means, np.array([length]))
        items.append((x[0], y[0], mask[0]))
    return items, lengths


def collate(items, ids):
    t = max(int(items[i][0].shape[0]) for i in ids)
    b = len(ids)
    x = torch.zeros(b, t, D_IN)
    y = torch.zeros(b, t, dtype=torch.long)
    mask = torch.zeros(b, t)
    for r, i in enumerate(ids):
        xi, yi, mi = items[i]
        length = xi.shape[0]
        x[r, :length] = xi
        y[r, :length] = yi
        mask[r, :length] = mi
    return x, y, mask


def batches(lengths, n, batch, mode, epoch, rng):
    idx = np.arange(n)
    if mode == "sorted" or (mode == "sortagrad" and epoch == 0):
        idx = idx[np.argsort(lengths[idx], kind="stable")]
    else:
        rng.shuffle(idx)
    return [idx[s:s + batch] for s in range(0, n - n % batch, batch)]


def run_curriculum(mode, lr, seed, items, lengths, val, epochs=3, batch=32):
    torch.manual_seed(seed)
    net = RNNNet()
    opt = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    rng = np.random.default_rng(seed + 500)
    n = len(items)
    hist = []
    gmax = 0.0
    spikes = 0
    blown = False
    for ep in range(epochs):
        losses = []
        for ids in batches(lengths, n, batch, mode, ep, rng):
            x, y, mask = collate(items, ids)
            logits = net(x)
            loss = per_utt_loss(logits, y, mask)
            opt.zero_grad()
            loss.backward()
            g = grad_norm(net)
            gmax = max(gmax, g)
            if g > 1e3:
                spikes += 1
            if not math.isfinite(loss.item()) or not math.isfinite(g):
                blown = True
                break
            opt.step()
            losses.append(per_frame_loss(logits, y, mask).item())
        vx, vy, vm = val
        with torch.no_grad():
            vl = per_frame_loss(net(vx), vy, vm).item()
            verr = float(((net(vx).argmax(-1) != vy).float() * vm).sum() / vm.sum())
        hist.append((float(np.mean(losses)) if losses else float("nan"), vl, verr))
        if blown:
            break
    return hist, gmax, spikes, blown


def exp_first_epoch(lr, seeds=8):
    means = base.make_means(np.random.default_rng(SEED + 3))
    rng = np.random.default_rng(SEED + 11)
    items, lengths = make_dataset(rng, means, 1536)
    vrng = np.random.default_rng(SEED + 77)
    val = base.make_batch(vrng, means, base.make_lengths(vrng, 96))
    log(f"[3] 첫 epoch만 놓고 본 안정성 (lr={lr}, clipping 없음, 발화당 loss 합으로 학습)")
    log(f"    학습 발화 1,536개, 배치 32(48스텝), 1 epoch, seed {seeds}개")
    log("")
    log("    방식         epoch1 val loss 중앙값   최소   최대   기울기 노름 최대 중앙값   노름>1e3 이던 스텝 수")
    out = {}
    for mode in ("shuffled", "sorted"):
        vls, gs, sps = [], [], []
        for seed in range(seeds):
            hist, gmax, spikes, _ = run_curriculum(mode, lr, seed, items, lengths, val, epochs=1)
            vls.append(hist[-1][1])
            gs.append(gmax)
            sps.append(spikes)
        out[mode] = (vls, gs, sps)
        log(
            f"    {mode:10s}   {np.median(vls):22.4f}   {min(vls):4.3f}   {max(vls):6.3f}"
            f"   {np.median(gs):23.1f}   {str(sps):>22s}"
        )
    log("")
    log("    sorted 는 SortaGrad 의 첫 epoch 과 같은 순서다. 짧은 발화부터 들어간다.")
    log("")
    return items, lengths, val, out


def exp_curriculum(lr, items, lengths, val, epochs=4, seeds=6):
    log(f"[4] 안정한 학습률에서 3가지 배치 구성 비교 (lr={lr}, {epochs} epoch, seed {seeds}개)")
    log("    shuffled  : 매 epoch 섞는다")
    log("    sorted    : 매 epoch 길이순으로 정렬한다")
    log("    sortagrad : 첫 epoch만 정렬하고 이후에는 섞는다")
    log("")
    log("    방식         epoch별 val frame loss (1, 2, 3, 4)                마지막 val frame error   seed 표준편차   기울기 노름 최대 중앙값")
    for mode in ("shuffled", "sorted", "sortagrad"):
        rows, errs, gs = [], [], []
        for seed in range(seeds):
            hist, gmax, spikes, _ = run_curriculum(mode, lr, seed, items, lengths, val, epochs=epochs)
            rows.append([h[1] for h in hist])
            errs.append(hist[-1][2])
            gs.append(gmax)
        a = np.array(rows)
        curve = "  ".join(f"{v:.4f}" for v in a.mean(0))
        log(
            f"    {mode:10s}   {curve}   {np.mean(errs):22.4f}   {np.std(errs):13.4f}   {np.median(gs):23.1f}"
        )
    log("")


def main():
    t0 = time.time()
    torch.set_num_threads(4)
    exp_padding()
    exp_length_scaling()
    items, lengths, val, _ = exp_first_epoch(0.05)
    exp_curriculum(0.01, items, lengths, val)
    elapsed = time.time() - t0
    cfg = (
        "설정: 합성 프레임 분류(sys_batchnorm.py 와 같은 생성기). 단방향 tanh RNN, "
        f"hidden {HIDDEN}, layer {LAYERS}, seed {SEED} 고정"
    )
    print("\n".join([ENV, RUNDATE, CMD, cfg, f"소요: {elapsed:.0f}초", ""] + LINES))


if __name__ == "__main__":
    main()
