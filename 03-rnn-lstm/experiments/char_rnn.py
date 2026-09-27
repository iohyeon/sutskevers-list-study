"""numpy 만으로 짠 최소 글자 단위 vanilla RNN. 순전파, BPTT, 샘플링, 기울기 검사."""
import sys, time
import numpy as np

class CharRNN:
    def __init__(self, vocab_size, hidden_size, seed=0):
        rng = np.random.default_rng(seed)
        V, H = vocab_size, hidden_size
        self.V, self.H = V, H
        self.p = {
            "Wxh": rng.normal(0, 0.01, (H, V)),   # 입력 -> hidden
            "Whh": rng.normal(0, 1.0 / np.sqrt(H), (H, H)),   # hidden -> hidden (모든 시점이 공유)
            "Why": rng.normal(0, 0.01, (V, H)),   # hidden -> 출력 logit
            "bh": np.zeros(H),
            "by": np.zeros(V),
        }
        self.mem = {k: np.zeros_like(v) for k, v in self.p.items()}  # Adagrad 누적값

    def forward(self, inputs, targets, h0):
        """inputs, targets: 정수 인덱스 리스트. 평균이 아닌 합계 loss 를 돌려준다."""
        p = self.p
        xs, hs, ps = {}, {-1: h0}, {}
        loss = 0.0
        for t, ix in enumerate(inputs):
            xs[t] = np.zeros(self.V); xs[t][ix] = 1.0
            hs[t] = np.tanh(p["Wxh"] @ xs[t] + p["Whh"] @ hs[t - 1] + p["bh"])
            y = p["Why"] @ hs[t] + p["by"]
            y -= y.max()                                   # 수치 안정화
            ps[t] = np.exp(y) / np.exp(y).sum()
            loss += -np.log(ps[t][targets[t]])
        return loss, (xs, hs, ps)

    def backward(self, inputs, targets, cache):
        p = self.p
        xs, hs, ps = cache
        g = {k: np.zeros_like(v) for k, v in p.items()}
        dh_next = np.zeros(self.H)                         # 미래 시점에서 넘어온 기울기
        for t in reversed(range(len(inputs))):
            dy = ps[t].copy(); dy[targets[t]] -= 1.0       # softmax + cross-entropy 의 기울기
            g["Why"] += np.outer(dy, hs[t]); g["by"] += dy
            dh = p["Why"].T @ dy + dh_next                 # 현재 출력에서 온 것 + 미래에서 온 것
            da = (1.0 - hs[t] ** 2) * dh                   # tanh 의 미분
            g["bh"] += da
            g["Wxh"] += np.outer(da, xs[t])
            g["Whh"] += np.outer(da, hs[t - 1])
            dh_next = p["Whh"].T @ da                      # 한 시점 과거로 전달
        return g

    def step(self, inputs, targets, h0, lr=0.1, clip=5.0):
        loss, cache = self.forward(inputs, targets, h0)
        g = self.backward(inputs, targets, cache)
        for k in self.p:
            np.clip(g[k], -clip, clip, out=g[k])           # 기울기 폭발 방지
            self.mem[k] += g[k] ** 2
            self.p[k] -= lr * g[k] / np.sqrt(self.mem[k] + 1e-8)
        return loss, cache[1][len(inputs) - 1]

    def sample(self, h, seed_ix, n, temperature=1.0, rng=None):
        rng = rng or np.random.default_rng()
        p = self.p
        ix, out = seed_ix, []
        for _ in range(n):
            x = np.zeros(self.V); x[ix] = 1.0
            h = np.tanh(p["Wxh"] @ x + p["Whh"] @ h + p["bh"])
            y = (p["Why"] @ h + p["by"]) / temperature
            y -= y.max()
            prob = np.exp(y) / np.exp(y).sum()
            ix = rng.choice(self.V, p=prob)
            out.append(ix)
        return out

def grad_check(seed=1):
    """해석적 기울기(BPTT)와 수치 미분을 비교한다."""
    rng = np.random.default_rng(seed)
    m = CharRNN(5, 4, seed)
    m.p["Wxh"] = rng.normal(0, 0.5, m.p["Wxh"].shape); m.p["Why"] = rng.normal(0, 0.5, m.p["Why"].shape)
    inputs = list(rng.integers(0, 5, 6)); targets = list(rng.integers(0, 5, 6))
    h0 = np.zeros(4)
    _, cache = m.forward(inputs, targets, h0)
    g = m.backward(inputs, targets, cache)
    worst = 0.0
    for k, W in m.p.items():
        it = np.nditer(W, flags=["multi_index"])
        for _ in it:
            idx = it.multi_index
            old = W[idx]
            W[idx] = old + 1e-5; lp, _ = m.forward(inputs, targets, h0)
            W[idx] = old - 1e-5; lm, _ = m.forward(inputs, targets, h0)
            W[idx] = old
            num = (lp - lm) / 2e-5
            rel = abs(num - g[k][idx]) / max(1e-12, abs(num) + abs(g[k][idx]))
            worst = max(worst, rel)
    return worst

if __name__ == "__main__":
    print("grad check, worst relative error:", grad_check())
    text = open("corpus.txt").read()
    chars = sorted(set(text)); c2i = {c: i for i, c in enumerate(chars)}
    data = [c2i[c] for c in text]
    split = int(len(data) * 0.9)
    train, test = data[:split], data[split:]
    H = int(sys.argv[1]) if len(sys.argv) > 1 else 64
    T = int(sys.argv[2]) if len(sys.argv) > 2 else 64
    iters = int(sys.argv[3]) if len(sys.argv) > 3 else 4000
    m = CharRNN(len(chars), H)
    h = np.zeros(H); pos = 0; smooth = None; t0 = time.time()
    for it in range(iters + 1):
        if pos + T + 1 >= len(train):
            pos = 0; h = np.zeros(H)
        loss, h = m.step(train[pos:pos + T], train[pos + 1:pos + T + 1], h)
        pos += T
        smooth = loss / T if smooth is None else 0.99 * smooth + 0.01 * loss / T
        if it % 1000 == 0:
            print(f"iter {it} loss/char {smooth:.3f} PP {np.exp(smooth):.3f} {time.time()-t0:.0f}s", flush=True)
    # test perplexity
    hh = np.zeros(H); tot = 0.0; n = 0
    for s in range(0, len(test) - T - 1, T):
        l, cache = m.forward(test[s:s + T], test[s + 1:s + T + 1], hh)
        hh = cache[1][T - 1]; tot += l; n += T
    print(f"test PP {np.exp(tot / n):.3f}")
    from ngram import closer_accuracy
    for temp in (0.5, 1.0):
        out = m.sample(np.zeros(H), c2i["\n"], 6000, temp, np.random.default_rng(0))
        s = "".join(chars[i] for i in out)
        print("T", temp, "closer ok/bad", closer_accuracy(s))
        print(s[:160].replace("\n", "\\n"))
