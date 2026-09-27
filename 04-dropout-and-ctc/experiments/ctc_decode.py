"""프레임별 확률에서 문장을 고르는 두 방법(greedy, prefix beam search)을 구현해 비교한다.

음성은 내려받지 않고 합성 확률 행렬을 만든다. 목표 문장을 프레임으로 늘린 뒤
혼동 짝과 약한 발음을 섞어서, greedy 가 틀리고 beam search 가 맞히는 상황이 생기게 한다.
언어 모델은 같은 말뭉치로 학습한 글자 n-gram이고 점수식은
  log P_ctc(y|x) + alpha * log P_lm(y) + beta * |y|
이다.

사용: python3 ctc_decode.py
"""
import math
import random
import time
from collections import defaultdict

from ctc_forward import log_total, total_prob

CHARS = " abcdefghijklmnopqrstuvwxyz"
BLANK = 0
SYMS = ["-"] + list(CHARS)          # 0 = blank, 1 = space, 2.. = a..z
IDX = {c: i for i, c in enumerate(SYMS)}
V = len(SYMS)

# 합성 혼동 짝. 한쪽이 다른 쪽으로 새는 경우를 대칭으로 둔다.
CONFUSE_PAIRS = [("a", "e"), ("e", "i"), ("o", "u"), ("m", "n"), ("b", "d"),
                 ("s", "z"), ("t", "d"), ("c", "k"), ("f", "v"), ("p", "b"),
                 ("g", "k"), ("l", "r"), ("i", "y"), ("n", "l")]
CONFUSE = defaultdict(list)
for x, y in CONFUSE_PAIRS:
    CONFUSE[x].append(y)
    CONFUSE[y].append(x)

DET = ["the", "a", "this", "that"]
ADJ = ["cold", "warm", "quiet", "small", "little", "yellow", "narrow", "green"]
NOUN = ["cat", "dog", "river", "morning", "letter", "window", "garden", "bottle",
        "market", "signal", "summer", "mirror"]
VERB = ["ran", "slept", "waited", "opened", "carried", "followed", "listened", "settled"]
ADV = ["slowly", "again", "yesterday", "quietly", "outside", "nearby"]


def make_sentence(rng):
    parts = [rng.choice(DET), rng.choice(ADJ), rng.choice(NOUN), rng.choice(VERB)]
    if rng.random() < 0.7:
        parts.append(rng.choice(ADV))
    if rng.random() < 0.4:
        parts += ["near", rng.choice(DET), rng.choice(NOUN)]
    return " ".join(parts)


def make_corpus(n, seed):
    rng = random.Random(seed)
    return [make_sentence(rng) for _ in range(n)]


class CharLM:
    """글자 n-gram. 차수 1..N 을 고정 가중치로 섞는다(보간)."""

    def __init__(self, sentences, order=6, k=0.1):
        self.order, self.k = order, k
        self.counts = [defaultdict(lambda: defaultdict(int)) for _ in range(order)]
        for s in sentences:
            text = " " + s + " "
            for i in range(1, len(text)):
                for n in range(order):
                    if i - n < 0:
                        break
                    h = text[i - n:i]
                    self.counts[n][h][text[i]] += 1
        self.w = [0.0] * order
        tot = 0.0
        for n in range(order):
            self.w[n] = 4.0 ** n
            tot += self.w[n]
        self.w = [x / tot for x in self.w]
        self.cache = {}

    def logp(self, hist, ch):
        key = (hist[-(self.order - 1):], ch)
        got = self.cache.get(key)
        if got is not None:
            return got
        p = 0.0
        for n in range(self.order):
            if n > len(hist):
                break
            h = hist[len(hist) - n:] if n else ""
            d = self.counts[n].get(h)
            if not d:
                continue
            tot = sum(d.values())
            p += self.w[n] * (d.get(ch, 0) + self.k) / (tot + self.k * len(CHARS))
        if p <= 0:
            p = 1e-10
        out = math.log(p)
        self.cache[key] = out
        return out


def synth_probs(text, rng, weak_p=0.30, swap_p=0.30):
    """목표 문장을 프레임별 확률 행렬로 늘린다. 어느 프레임을 약하게 할지 기록해 둔다."""
    rows, notes = [], []
    prev = None
    for pos, ch in enumerate(text):
        if prev is not None:
            nb = rng.randint(1, 2) if ch == prev else rng.randint(0, 2)
            for _ in range(nb):
                rows.append(blank_row())
        prev = ch
        n = rng.randint(3, 5)
        weak = rng.random() < weak_p
        rival = None
        if not weak and CONFUSE[ch] and rng.random() < swap_p:
            rival = rng.choice(CONFUSE[ch])
        if weak:
            p_true, p_blank = 0.32, 0.36
        elif rival:
            p_true, p_blank = 0.33, 0.06
        else:
            p_true, p_blank = 0.80, 0.05
        for _ in range(n):
            row = [0.0] * V
            row[BLANK] = p_blank
            row[IDX[ch]] = p_true
            rest = 1.0 - p_blank - p_true
            if rival:
                row[IDX[rival]] = 0.50
                rest -= 0.50
            near = CONFUSE[ch] or ["e"]
            share = rest * 0.6 / len(near)
            for c in near:
                row[IDX[c]] += share
            spread = rest * 0.4 / (V - 2)
            for i in range(1, V):
                if i != IDX[ch]:
                    row[i] += spread
            s = sum(row)
            rows.append([x / s for x in row])
        if weak or rival:
            notes.append((pos, ch, "약한 발음" if weak else f"혼동: {rival} 가 더 셈", n))
    for _ in range(rng.randint(1, 3)):
        rows.append(blank_row())
    return rows, notes


def blank_row():
    row = [0.0] * V
    row[BLANK] = 0.90
    for i in range(1, V):
        row[i] = 0.10 / (V - 1)
    return row


def greedy(probs):
    best = [max(range(V), key=lambda k: row[k]) for row in probs]
    out, prev = [], None
    for k in best:
        if k != prev and k != BLANK:
            out.append(SYMS[k])
        prev = k
    return "".join(out)


def lse(a, b):
    if a == -math.inf:
        return b
    if b == -math.inf:
        return a
    m = a if a > b else b
    return m + math.log(math.exp(a - m) + math.exp(b - m))


def beam_decode(probs, beam=50, lm=None, alpha=0.0, beta=0.0, prune=5e-3):
    """CTC prefix beam search. 반환: (최선 문장, 점수, 확장 후보 수, 평균 접두사 수)."""
    NEG = -math.inf
    cur = {"": (0.0, NEG)}   # prefix -> (log p_blank, log p_nonblank)
    work = 0
    live = 0
    for row in probs:
        cands = [k for k in range(V) if row[k] > prune]
        lg = {k: math.log(row[k]) for k in cands}
        nxt = defaultdict(lambda: [NEG, NEG])
        for pref, (pb, pnb) in cur.items():
            ptot = lse(pb, pnb)
            last = pref[-1] if pref else None
            for k in cands:
                work += 1
                if k == BLANK:
                    e = nxt[pref]
                    e[0] = lse(e[0], ptot + lg[k])
                    continue
                ch = SYMS[k]
                if ch == last:
                    e = nxt[pref]
                    e[1] = lse(e[1], pnb + lg[k])
                    ext = pb
                else:
                    ext = ptot
                if ext == NEG:
                    continue
                bonus = 0.0
                if lm is not None and alpha:
                    bonus += alpha * lm.logp(pref, ch)
                bonus += beta
                e2 = nxt[pref + ch]
                e2[1] = lse(e2[1], ext + lg[k] + bonus)
        cur = dict(sorted(nxt.items(), key=lambda kv: -lse(kv[1][0], kv[1][1]))[:beam])
        live += len(cur)
    best = max(cur.items(), key=lambda kv: lse(kv[1][0], kv[1][1]))
    return best[0], lse(best[1][0], best[1][1]), work, live / len(probs)


def edit(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb))
        prev = cur
    return prev[-1]


def cer(refs, hyps):
    e = sum(edit(r, h) for r, h in zip(refs, hyps))
    n = sum(len(r) for r in refs)
    return 100.0 * e / n, e, n


def main():
    t0 = time.time()
    train = make_corpus(4000, 11)
    lm = CharLM(train, order=6, k=0.1)
    print(f"언어 모델: 글자 6-gram, 학습 문장 {len(train)}개 / "
          f"{sum(len(s) for s in train):,}글자, 어휘 {len(CHARS)}글자 + blank")

    rng = random.Random(7)
    dev_refs = [s for s in make_corpus(40, 99) if s not in set(train)][:8]
    test_refs = [s for s in make_corpus(60, 5) if s not in set(train)][:16]
    dev = [(r,) + synth_probs(r, rng) for r in dev_refs]
    test = [(r,) + synth_probs(r, rng) for r in test_refs]
    tf = sum(len(p) for _, p, _ in test)
    print(f"시험 발화 {len(test)}개, 목표 글자 {sum(len(r) for r in test_refs)}개, "
          f"프레임 {tf}개 (발화당 평균 {tf / len(test):.1f})")
    print(f"약한 발음 / 혼동을 심은 글자 {sum(len(n) for _, _, n in test)}개")

    print("\n== 1. greedy 디코딩 ==")
    g_hyp = [greedy(p) for _, p, _ in test]
    gc, ge, gn = cer(test_refs, g_hyp)
    print(f"greedy CER = {gc:.2f}% (편집거리 {ge} / 글자 {gn}), 완전히 맞은 발화 "
          f"{sum(1 for r, h in zip(test_refs, g_hyp) if r == h)}/{len(test)}")

    print("\n== 2. alpha, beta 고르기 (dev 발화 8개, beam 50) ==")
    best = None
    for alpha in (0.0, 0.3, 0.6, 1.0, 1.5):
        line = f"  alpha={alpha:<4}"
        for beta in (0.0, 0.5, 1.0, 2.0):
            hyp = [beam_decode(p, 50, lm, alpha, beta)[0] for _, p, _ in dev]
            c = cer(dev_refs, hyp)[0]
            line += f"  beta={beta}: {c:5.2f}%"
            if best is None or c < best[0]:
                best = (c, alpha, beta)
        print(line)
    _, A, B = best
    print(f"  고른 값: alpha={A}, beta={B} (dev CER {best[0]:.2f}%)")

    print("\n== 3. beam 폭별 정확도와 계산량 (시험 발화 16개) ==")
    print("  beam | 언어모델 없음                          | 언어모델 결합")
    print("       |   CER   맞은 발화    확장수 접두사   초 |   CER   맞은 발화    확장수 접두사   초")
    rows = {}
    for bw in (1, 2, 4, 8, 16, 50, 100, 200, 500):
        line = f"  {bw:>4} |"
        for alpha, beta in ((0.0, 0.0), (A, B)):
            s = time.time()
            res = [beam_decode(p, bw, lm, alpha, beta) for _, p, _ in test]
            dt = time.time() - s
            hyp = [r[0] for r in res]
            work = sum(r[2] for r in res)
            liv = sum(r[3] for r in res) / len(res)
            c = cer(test_refs, hyp)[0]
            exact = sum(1 for r, h in zip(test_refs, hyp) if r == h)
            line += f" {c:6.2f}%  {exact:>3}/{len(test)} {work:>9,} {liv:6.1f} {dt:5.1f} |"
            rows[(bw, alpha)] = (c, exact, work, dt, hyp, liv)
        print(line)

    print("\n== 4. greedy 가 지는 발화 ==")
    hyp_lm = rows[(500, A)][4]
    hyp_nolm = rows[(500, 0.0)][4]
    shown = 0
    for i, ref in enumerate(test_refs):
        if g_hyp[i] == ref or hyp_lm[i] != ref:
            continue
        notes = test[i][2]
        print(f"\n  목표          : {ref}")
        print(f"  greedy        : {g_hyp[i]}   (편집거리 {edit(ref, g_hyp[i])})")
        print(f"  beam 500      : {hyp_nolm[i]}   (편집거리 {edit(ref, hyp_nolm[i])})")
        print(f"  beam 500 + LM : {hyp_lm[i]}   (편집거리 {edit(ref, hyp_lm[i])})")
        print(f"  심어 둔 자리  : " + "; ".join(f"{p}번째 '{c}' {w} ({n}프레임)" for p, c, w, n in notes))
        shown += 1
        if shown == 3:
            break

    print("\n  -- 언어 모델을 결합하고도 남은 오류 (beam 50) --")
    left = 0
    for ref, h in zip(test_refs, rows[(50, A)][4]):
        if ref != h:
            left += 1
            print(f"  목표 : {ref}")
            print(f"  출력 : {h}")
            probs = test[test_refs.index(ref)][1]
            lp = [[math.log(max(x, 1e-300)) for x in row] for row in probs]
            print(f"  {'문장':<34} {'log P_ctc':>11} {'a*log P_lm':>11} {'합':>11}")
            for y in (ref, h):
                ac = log_total(lp, [IDX[c] for c in y])
                hist = " "
                lmv = 0.0
                for c in y:
                    lmv += lm.logp(hist, c)
                    hist += c
                print(f"  {y:<34} {ac:>11.3f} {A * lmv:>11.3f} {ac + A * lmv:>11.3f}")
    if left == 0:
        print("  없음")

    print("\n  -- beam 1 에서 언어 모델을 결합하면 무엇이 나오나 --")
    b1 = [beam_decode(p, 1, lm, A, B)[0] for _, p, _ in test[:3]]
    for ref, h in zip(test_refs[:3], b1):
        print(f"  목표 {ref!r} -> 출력 {h!r}")
    print(f"  글자를 하나 늘릴 때마다 alpha * log P_lm <= 0 이 더해지므로, 후보를 하나만 들고 가면")
    print(f"  아무 글자도 내지 않는 경로가 계속 1위를 차지한다.")

    print("\n== 5. greedy 가 글자를 빠뜨리는 이유 (프레임 4개, 기호 2개) ==")
    row = [0.0] * V
    row[BLANK], row[IDX["c"]] = 0.36, 0.32
    rest = (1.0 - 0.68) / (V - 2)
    for i in range(1, V):
        if i != IDX["c"]:
            row[i] = rest
    four = [row] * 4
    p_c = total_prob(four, [IDX["c"]])
    p_empty = 0.36 ** 4
    print(f"  프레임마다 p(blank)={row[BLANK]}, p(c)={row[IDX['c']]}. blank 가 매 프레임 argmax 이므로")
    print(f"  greedy 는 빈 문자열을 낸다.")
    print(f"  P(y=\"\"  | x) = 0.36^4                     = {p_empty:.6f}")
    print(f"  P(y=\"c\" | x) = 접으면 c 가 되는 경로 10개의 합 = {p_c:.6f}")
    print(f"  비 = {p_c / p_empty:.2f}배. 한 경로가 아니라 경로 묶음의 합을 비교해야 하므로")
    print(f"  프레임별 argmax 로는 이 판단을 할 수 없다.")

    print("\n== 6. 정확도와 계산량의 관계 ==")
    base = rows[(1, A)][2]
    for bw in (1, 8, 50, 500):
        c, exact, work, dt, _, liv = rows[(bw, A)]
        print(f"  beam {bw:>3}: CER {c:5.2f}%  확장수 {work:>9,} (beam 1의 {work / base:5.1f}배)  "
              f"살아 있는 접두사 평균 {liv:.1f}개  {dt:.1f}초")

    print("\n== 7. 프레임별 가지치기의 효과 (beam 50, 언어 모델 결합) ==")
    for pr in (0.0, 1e-4, 5e-3, 5e-2):
        s2 = time.time()
        res = [beam_decode(p, 50, lm, A, B, prune=pr) for _, p, _ in test]
        dt = time.time() - s2
        c = cer(test_refs, [r[0] for r in res])[0]
        work = sum(r[2] for r in res)
        print(f"  임계값 {pr:<7}: CER {c:5.2f}%  확장수 {work:>10,}  {dt:5.1f}초")

    print(f"\n소요 {time.time() - t0:.1f}초")


if __name__ == "__main__":
    main()
