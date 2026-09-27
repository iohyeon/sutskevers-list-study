"""04 폴더의 실험이 함께 쓰는 말뭉치 로딩, LSTM, 학습 루프.

LSTM 을 PyTorch 의 nn.LSTM 대신 직접 짠 이유는 dropout 을 거는 자리를 인자로 고르기
위해서다. nn.LSTM 의 dropout 인자는 layer 사이에만 걸리고 순환 연결에는 걸 수 없다.
"""
import math
import os
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

CORPUS = "corpus.txt"


def ensure_corpus(path=CORPUS):
    """말뭉치 파일이 없으면 make_corpus.py 와 같은 규칙으로 만든다."""
    if not os.path.exists(path):
        import make_corpus
        with open(path, "w") as f:
            f.write(make_corpus.build())
    return open(path).read()


def load(path=CORPUS):
    """(text, chars, c2i, train, valid, test). 앞 80% 학습, 다음 10% 검증, 나머지 시험."""
    text = ensure_corpus(path)
    chars = sorted(set(text))
    c2i = {c: i for i, c in enumerate(chars)}
    data = torch.tensor([c2i[c] for c in text])
    a, b = int(len(data) * 0.8), int(len(data) * 0.9)
    return text, chars, c2i, data[:a], data[a:b], data[b:]


def batches(data, batch, seq):
    n = (len(data) - 1) // (batch * seq)
    x = data[:n * batch * seq].view(batch, -1)
    y = data[1:n * batch * seq + 1].view(batch, -1)
    for i in range(0, x.size(1), seq):
        yield x[:, i:i + seq], y[:, i:i + seq]


def mask_like(shape, rate):
    """유지 확률 1-rate 인 Bernoulli 마스크. 학습 때 1/(1-rate) 로 키워 기댓값을 맞춘다."""
    keep = 1.0 - rate
    return torch.bernoulli(torch.full(shape, keep)) / keep


def drop_feed(x, rate, mode, training):
    """비순환 연결(embedding 출력, layer 사이, 출력 직전)에 거는 dropout.

    mode="per_step" 은 (batch, time, hidden) 전부에 독립 마스크,
    mode="per_seq" 는 시간 방향으로 같은 마스크를 쓴다.
    """
    if not training or rate <= 0.0:
        return x
    if mode == "per_seq":
        return x * mask_like((x.size(0), 1, x.size(2)), rate)
    return x * mask_like(x.shape, rate)


class LSTMLayer(nn.Module):
    """게이트 4개를 한 번에 계산하는 LSTM layer 하나. 순환 입력에 마스크를 곱할 수 있다."""

    def __init__(self, n_in, n_hidden):
        super().__init__()
        self.n_hidden = n_hidden
        self.wx = nn.Linear(n_in, 4 * n_hidden)
        self.wh = nn.Linear(n_hidden, 4 * n_hidden, bias=False)

    def forward(self, x, state, rate=0.0, mode="none", wh=None):
        B, T, _ = x.shape
        h, c = state
        W = self.wh.weight if wh is None else wh
        gx = self.wx(x)
        on = self.training and rate > 0.0 and mode in ("per_step", "per_seq")
        m = mask_like((B, self.n_hidden), rate) if on and mode == "per_seq" else None
        outs = []
        for t in range(T):
            if on and mode == "per_step":
                m = mask_like((B, self.n_hidden), rate)
            hin = h if m is None else h * m
            g = gx[:, t] + hin @ W.t()
            i, f, u, o = g.chunk(4, dim=1)
            c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(u)
            h = torch.sigmoid(o) * torch.tanh(c)
            outs.append(h)
        return torch.stack(outs, 1), (h, c)


class DropLSTM(nn.Module):
    """dropout 을 거는 자리를 인자로 고르는 글자 단위 LSTM 언어 모델.

    feed_rate  : 비순환 연결의 dropout 비율 (Zaremba 외 2014 가 쓴 자리)
    rec_rate   : 순환 입력 h_{t-1} 의 dropout 비율
    rec_mode   : "none" | "per_step" (시점마다 독립) | "per_seq" (시퀀스 안에서 고정)
    feed_mode  : "per_step" | "per_seq"
    weight_rate: 순환 가중치 행렬 W_hh 에 거는 DropConnect 비율 (배치마다 한 번)
    """

    def __init__(self, vocab, hidden, layers=2, feed_rate=0.0, rec_rate=0.0,
                 rec_mode="none", feed_mode="per_step", weight_rate=0.0):
        super().__init__()
        self.hidden, self.layers = hidden, layers
        self.feed_rate, self.rec_rate = feed_rate, rec_rate
        self.rec_mode, self.feed_mode = rec_mode, feed_mode
        self.weight_rate = weight_rate
        self.embed = nn.Embedding(vocab, hidden)
        self.cells = nn.ModuleList([LSTMLayer(hidden, hidden) for _ in range(layers)])
        self.out = nn.Linear(hidden, vocab)

    def init_state(self, batch):
        z = torch.zeros(batch, self.hidden)
        return [(z.clone(), z.clone()) for _ in range(self.layers)]

    def forward(self, x, state=None):
        if state is None:
            state = self.init_state(x.size(0))
        h = drop_feed(self.embed(x), self.feed_rate, self.feed_mode, self.training)
        new = []
        for k, cell in enumerate(self.cells):
            wh = None
            if self.weight_rate > 0.0:
                wh = F.dropout(cell.wh.weight, self.weight_rate, self.training)
            h, s = cell(h, state[k], self.rec_rate, self.rec_mode, wh)
            h = drop_feed(h, self.feed_rate, self.feed_mode, self.training)
            new.append(s)
        return self.out(h), new


def detach(state):
    return [(h.detach(), c.detach()) for h, c in state]


def count_params(model):
    return sum(p.numel() for p in model.parameters())


@torch.no_grad()
def perplexity(model, data, batch=1, seq=100):
    """dropout 을 끈 상태로 split 전체를 한 번 훑어 perplexity 를 낸다."""
    model.eval()
    lossf = nn.CrossEntropyLoss(reduction="sum")
    tot, n, state = 0.0, 0, None
    for x, y in batches(data, batch, seq):
        logits, state = model(x, state)
        tot += lossf(logits.reshape(-1, logits.size(-1)), y.reshape(-1)).item()
        n += y.numel()
    return math.exp(tot / n)


@torch.no_grad()
def sample(model, chars, c2i, n=3000, temperature=0.5, seed=0):
    torch.manual_seed(seed)
    model.eval()
    ix = torch.tensor([[c2i["\n"]]])
    state, out = None, []
    for _ in range(n):
        logits, state = model(ix, state)
        prob = torch.softmax(logits[0, -1] / temperature, dim=0)
        ix = torch.multinomial(prob, 1).view(1, 1)
        out.append(chars[ix.item()])
    return "".join(out)


def pair_accuracy(text):
    """여는 태그 바로 뒤에 오는 닫는 태그가 같은 이름인지 센다. (맞은 수, 전체 수)"""
    import re
    ok = tot = 0
    for m in re.finditer(r"\\begin\{([a-z]*)\}(.*?)\\end\{([a-z]*)\}", text):
        tot += 1
        ok += 1 if m.group(1) == m.group(3) else 0
    return ok, tot


def train(hidden=384, layers=2, feed_rate=0.0, rec_rate=0.0, rec_mode="none",
          feed_mode="per_step", weight_rate=0.0, epochs=50, seq=50, batch=20,
          lr=3e-3, seed=0, log=None, out=None, path=CORPUS):
    """한 설정을 학습시키고 epoch 별 train/valid perplexity 기록을 돌려준다."""
    torch.manual_seed(seed)
    random.seed(seed)
    text, chars, c2i, tr, va, te = load(path)
    model = DropLSTM(len(chars), hidden, layers, feed_rate, rec_rate,
                     rec_mode, feed_mode, weight_rate)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.CrossEntropyLoss()
    hist, t0 = [], time.time()
    for ep in range(1, epochs + 1):
        model.train()
        state = None
        for x, y in batches(tr, batch, seq):
            logits, state = model(x, detach(state) if state else None)
            loss = lossf(logits.reshape(-1, len(chars)), y.reshape(-1))
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        tr_pp, va_pp = perplexity(model, tr), perplexity(model, va)
        hist.append((ep, tr_pp, va_pp))
        if log and (ep in log or ep == epochs):
            (out or print)(f"  epoch {ep:3d} train PP {tr_pp:7.3f} valid PP {va_pp:7.3f}"
                           f" gap {va_pp - tr_pp:+7.3f} {time.time() - t0:.0f}s")
    res = {
        "params": count_params(model), "hist": hist,
        "train_pp": hist[-1][1], "valid_pp": hist[-1][2],
        "test_pp": perplexity(model, te), "seconds": time.time() - t0,
        "model": model, "chars": chars, "c2i": c2i,
        "n_train": len(tr), "n_valid": len(va), "n_test": len(te), "vocab": len(chars),
    }
    return res


def header(command, setting, seconds):
    """결과 파일 첫머리. 03 폴더의 results 형식을 따른다."""
    text, chars, _, tr, va, te = load()
    return "\n".join([
        "환경: Python 3.14.7, NumPy 2.5.3, PyTorch 2.14.0, macOS arm64, CPU",
        "실행일: 2026-09-28",
        f"말뭉치: make_corpus.py (seed 7), {len(text):,}글자, 어휘 {len(chars)}글자, "
        f"학습 {len(tr):,} / 검증 {len(va):,} / 시험 {len(te):,}글자",
        f"명령: python3 {command}",
        f"설정: {setting}",
        f"소요: {seconds:.0f}초",
    ]) + "\n"


class Log:
    """출력을 화면과 results 파일에 같이 남긴다."""

    def __init__(self):
        self.lines = []
        self.t0 = time.time()

    def __call__(self, line=""):
        print(line, flush=True)
        self.lines.append(line)

    def save(self, name, command, setting):
        path = os.path.join("..", "results", name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write(header(command, setting, time.time() - self.t0))
            f.write("\n" + "\n".join(self.lines) + "\n")
        print(f"\n[saved] results/{name}")
