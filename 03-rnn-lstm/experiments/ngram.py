"""글자 단위 n-gram 언어 모델. smoothing 없이 빈도만 센다."""
import sys, math, random
from collections import defaultdict, Counter

def train(text, n):
    table = defaultdict(Counter)          # 키: 직전 n글자, 값: 다음 글자별 횟수
    pad = "~" * n
    data = pad + text
    for i in range(len(text)):
        table[data[i:i + n]][data[i + n]] += 1
    return table

def prob(table, ctx, ch, vocab_size, alpha=0.0):
    c = table.get(ctx)
    total = sum(c.values()) if c else 0
    count = c[ch] if c else 0
    if total + alpha * vocab_size == 0:
        return 0.0
    return (count + alpha) / (total + alpha * vocab_size)

def perplexity(table, text, n, vocab_size, alpha):
    data = "~" * n + text
    logsum = 0.0
    for i in range(len(text)):
        p = prob(table, data[i:i + n], data[i + n], vocab_size, alpha)
        if p == 0.0:
            return float("inf")
        logsum += math.log(p)
    return math.exp(-logsum / len(text))

def generate(table, n, length, seed=0):
    random.seed(seed)
    out = "~" * n
    for _ in range(length):
        c = table.get(out[-n:])
        if not c:
            break
        chars, weights = zip(*c.items())
        out += random.choices(chars, weights)[0]
    return out[n:]

def closer_accuracy(text):
    """\\begin{X} 로 연 환경을 \\end{X} 로 닫았는지 센다."""
    ok = bad = 0
    for line in text.split("\n"):
        if line.startswith("\\begin{") and "\\end{" in line:
            head = line[7:line.index("\\end{")]
            opened = head[:head.index("}")] if "}" in head else None   # 여는 쪽이 깨졌으면 오답으로 센다
            tail = line[line.index("\\end{") + 5:]
            closed = tail[:tail.index("}")] if "}" in tail else ""
            if opened is not None and opened == closed: ok += 1
            else: bad += 1
    return ok, bad

if __name__ == "__main__":
    text = open("corpus.txt").read()
    split = int(len(text) * 0.9)
    tr, te = text[:split], text[split:]
    V = len(set(text)) + 1
    for n in (1, 2, 4, 8, 12):
        t = train(tr, n)
        pp0 = perplexity(t, te, n, V, 0.0)
        pp1 = perplexity(t, te, n, V, 0.01)
        ok, bad = closer_accuracy(generate(t, n, 6000))
        print(f"n={n:2d} contexts={len(t):6d} PP(no smoothing)={pp0:8.3f} PP(add-0.01)={pp1:8.3f} closer ok/bad={ok}/{bad}")
    print(generate(train(tr, 8), 8, 200))
