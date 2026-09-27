"""CTC 의 구조적 제약을 세고, stride 로 프레임을 줄였을 때 어떤 발화가 표현 불가능해지는지 센다.

제약 두 가지를 그대로 계산한다.
  (1) 출력 라벨 길이는 입력 스텝 수를 넘을 수 없다.
  (2) 같은 라벨이 연속되면 사이에 blank 가 한 스텝 필요하다.
따라서 라벨 y 를 내려면 최소 |y| + (인접한 같은 라벨의 쌍 수) 스텝이 필요하다.

문자 단위 출력과 겹치지 않는 bigram 단위 출력을 표현 가능 범위, 어휘 크기, 계산량으로 비교한다.
음성 데이터는 쓰지 않고, 문법으로 만든 전사와 합성 발화 속도로 프레임 수를 정한다.

사용: python3 ctc_stride.py
"""
import itertools
import math
import random
import time

from ctc_decode import make_corpus
from ctc_forward import collapse, forward_table

H = 2048          # 순환 layer 하나의 hidden 크기 (가정)
NLAYER = 5        # 순방향 순환 layer 수 (가정)
FRAME_MS = 10     # 프레임 하나가 덮는 시간


def min_steps(units):
    """라벨 시퀀스를 내는 데 필요한 최소 입력 스텝 수."""
    n = len(units)
    rep = sum(1 for i in range(1, n) if units[i] == units[i - 1])
    return n + rep


def count_alignments(units, T):
    """길이 T 인 입력에서 이 라벨로 접히는 경로(정렬)의 수."""
    if T < min_steps(units):
        return 0
    uniq = {u: i + 1 for i, u in enumerate(dict.fromkeys(units))}
    lab = [uniq[u] for u in units]
    K = len(uniq) + 1
    ones = [[1] * K for _ in range(T)]
    z, a = forward_table(ones, lab, int)
    S = len(z)
    return a[S - 1][T - 1] + (a[S - 2][T - 1] if S >= 2 else 0)


def brute_alignments(units, T):
    """작은 경우에 대해 완전 탐색으로 정렬 수를 센다."""
    uniq = {u: i + 1 for i, u in enumerate(dict.fromkeys(units))}
    lab = tuple(uniq[u] for u in units)
    K = len(uniq) + 1
    return sum(1 for path in itertools.product(range(K), repeat=T) if collapse(path) == lab)


def to_bigrams(text):
    """겹치지 않는 bigram 으로 자른다. 길이가 홀수면 마지막은 글자 하나."""
    return [text[i:i + 2] for i in range(0, len(text), 2)]


def main():
    t0 = time.time()

    print("== 1. 라벨을 내는 데 필요한 최소 스텝 수 ==")
    print(f"  {'라벨':<14} {'|y|':>4} {'인접 중복':>9} {'최소 스텝':>10}")
    for w in ("cat", "the cat", "letter", "bottle", "bookkeeper", "aaaa", "mississippi"):
        need = min_steps(list(w))
        print(f"  {w:<14} {len(w):>4} {need - len(w):>9} {need:>10}")
    print("  bigram 단위로 자르면 인접 중복이 생기는 경우")
    for w in ("aaaa", "abab", "letter", "the cat"):
        b = to_bigrams(w)
        print(f"  {w:<14} -> {str(b):<28} |y|={len(b)} 인접 중복 {min_steps(b) - len(b)} "
              f"최소 스텝 {min_steps(b)}")

    print("\n== 2. 입력 스텝 수를 바꾸면 정렬이 몇 개인가 ==")
    print("  DP 로 센 값과 완전 탐색으로 센 값을 비교한다.")
    print(f"  {'라벨':<8} {'T':>3} {'DP':>10} {'완전탐색':>10} {'일치':>5}")
    for w, T in [("cat", 2), ("cat", 3), ("cat", 4), ("cat", 5), ("cat", 6), ("cat", 8),
                 ("aa", 2), ("aa", 3), ("aa", 4), ("aa", 6),
                 ("letter", 6), ("letter", 7), ("letter", 8)]:
        dp = count_alignments(list(w), T)
        bf = brute_alignments(list(w), T) if T <= 9 else None
        print(f"  {w:<8} {T:>3} {dp:>10,} {bf if bf is None else format(bf, ',') :>10} "
              f"{'예' if dp == bf else '-':>5}")
    print("  T 를 더 늘리면 (라벨 cat)")
    for T in (10, 20, 40, 80):
        print(f"    T={T:>3}: 정렬 {count_alignments(list('cat'), T):,}개")

    print("\n== 3. stride 로 스텝을 줄이면 무엇이 표현 불가능해지나 ==")
    corpus = make_corpus(400, 21)
    rng = random.Random(3)
    utt = []
    for text in corpus:
        cps = rng.uniform(9.0, 18.0)                       # 초당 글자 수
        T0 = max(1, int(round(len(text) / cps * 1000 / FRAME_MS)))
        utt.append((text, T0))
    lens = [t for _, t in utt]
    chars = [len(x) for x, _ in utt]
    print(f"  전사 {len(utt)}개, 글자 수 평균 {sum(chars) / len(chars):.1f} "
          f"(최소 {min(chars)}, 최대 {max(chars)})")
    print(f"  프레임 수 평균 {sum(lens) / len(lens):.1f} (최소 {min(lens)}, 최대 {max(lens)}), "
          f"프레임 {FRAME_MS}ms 기준")
    print(f"  글자당 프레임 평균 {sum(lens) / sum(chars):.2f}")

    print(f"\n  {'stride':>6} {'평균 스텝':>10} {'문자 단위':>22} {'bigram 단위':>22}")
    print(f"  {'':>6} {'':>10} {'표현 가능   여유 평균':>22} {'표현 가능   여유 평균':>22}")
    table = {}
    for s in (1, 2, 3, 4, 6, 8, 12):
        steps = [(T0 - 1) // s + 1 for _, T0 in utt]
        row = []
        for unit in ("char", "bigram"):
            ok, slack = 0, []
            for (text, _), T in zip(utt, steps):
                units = list(text) if unit == "char" else to_bigrams(text)
                need = min_steps(units)
                if T >= need:
                    ok += 1
                slack.append(T - need)
            row.append((ok, sum(slack) / len(slack)))
        table[s] = (sum(steps) / len(steps), row)
        print(f"  {s:>6} {sum(steps) / len(steps):>10.1f} "
              f"{row[0][0]:>8}/{len(utt)} {row[0][1]:>11.1f} "
              f"{row[1][0]:>8}/{len(utt)} {row[1][1]:>11.1f}")

    first = min(s for s in (1, 2, 3, 4, 6, 8, 12) if table[s][1][0][0] < len(utt))
    print(f"\n  문자 단위가 처음 깨지는 stride = {first} "
          f"({table[first][1][0][0]}/{len(utt)} 만 표현 가능)")
    print(f"  stride {first}, 문자 단위에서 표현 불가능해진 전사 (앞 5개)")
    shown = 0
    for (text, T0) in utt:
        T = (T0 - 1) // first + 1
        need_c = min_steps(list(text))
        need_b = min_steps(to_bigrams(text))
        if T < need_c:
            print(f"    글자 {len(text):>2}개 / 프레임 {T0:>3} -> 스텝 {T:>2}. "
                  f"문자 단위는 {need_c} 스텝 필요(부족 {need_c - T}), bigram 단위는 {need_b} 스텝 필요"
                  f"{' (가능)' if T >= need_b else f' (부족 {need_b - T})'}")
            print(f"      \"{text}\"")
            shown += 1
            if shown == 5:
                break

    print("\n== 4. 출력 단위의 어휘 크기 ==")
    alpha = sorted(set("".join(corpus)))
    bigs = sorted({b for text in corpus for b in to_bigrams(text)})
    print(f"  문자 단위  : 글자 {len(alpha)}종 + blank = {len(alpha) + 1}")
    print(f"  bigram 단위: 말뭉치에 실제로 나타난 겹치지 않는 조각 {len(bigs)}종 + blank = {len(bigs) + 1}")
    print(f"               (이론상 최대 {len(alpha)}^2 + {len(alpha)} = "
          f"{len(alpha) ** 2 + len(alpha)}, 실제로는 {len(bigs)}종만 쓰인다)")
    n_units_c = sum(len(t) for t in corpus)
    n_units_b = sum(len(to_bigrams(t)) for t in corpus)
    print(f"  출력 길이  : 문자 {n_units_c:,}개 vs bigram {n_units_b:,}개 "
          f"({n_units_b / n_units_c:.3f}배)")
    rep_c = sum(min_steps(list(t)) - len(t) for t in corpus)
    rep_b = sum(min_steps(to_bigrams(t)) - len(to_bigrams(t)) for t in corpus)
    print(f"  인접 중복  : 문자 {rep_c}쌍 vs bigram {rep_b}쌍 (blank 를 끼워야 하는 자리)")

    print("\n== 5. 계산량 ==")
    Vc, Vb = len(alpha) + 1, len(bigs) + 1
    print(f"  가정: 순방향 순환 layer {NLAYER}개, hidden {H}. 한 스텝당 순환부 MAC = "
          f"{NLAYER} x 2 x {H}^2 = {NLAYER * 2 * H * H / 1e6:.1f}M")
    print(f"  {'stride':>6} {'스텝':>6} {'순환부(G MAC)':>14} "
          f"{'출력층 문자(M MAC)':>20} {'출력층 bigram(M MAC)':>22} {'합 비율':>10}")
    for s in (1, 2, 3, 4, 8):
        steps = sum((T0 - 1) // s + 1 for _, T0 in utt)
        rec = steps * NLAYER * 2 * H * H
        oc, ob = steps * H * Vc, steps * H * Vb
        print(f"  {s:>6} {steps:>6,} {rec / 1e9:>14.1f} {oc / 1e6:>20.1f} {ob / 1e6:>22.1f} "
              f"{(rec + ob) / (rec + oc):>10.4f}")
    print(f"  출력층이 전체에서 차지하는 비율: 문자 {Vc * 100 / (NLAYER * 2 * H + Vc):.2f}%, "
          f"bigram {Vb * 100 / (NLAYER * 2 * H + Vb):.2f}%")

    print("\n== 6. bigram 단위가 늘리는 것과 줄이는 것 ==")
    for s in (2, 4, 8):
        steps = [(T0 - 1) // s + 1 for _, T0 in utt]
        okc = sum(1 for (t, _), T in zip(utt, steps) if T >= min_steps(list(t)))
        okb = sum(1 for (t, _), T in zip(utt, steps) if T >= min_steps(to_bigrams(t)))
        print(f"  stride {s}: 표현 가능 발화가 {okc}/{len(utt)} -> {okb}/{len(utt)} 로 "
              f"{okb - okc}개 늘고, 출력층 MAC 은 {Vb / Vc:.1f}배가 된다")

    print("\n== 6b. 계산량을 맞춰서 비교하면 ==")
    steps_c4 = sum((T0 - 1) // 4 + 1 for _, T0 in utt)
    steps_b8 = sum((T0 - 1) // 8 + 1 for _, T0 in utt)
    cost_c4 = steps_c4 * (NLAYER * 2 * H * H + H * Vc)
    cost_b8 = steps_b8 * (NLAYER * 2 * H * H + H * Vb)
    print(f"  문자 단위 stride 4  : 스텝 {steps_c4:,}, 전체 {cost_c4 / 1e9:.1f} G MAC, "
          f"표현 가능 {table[4][1][0][0]}/{len(utt)}")
    print(f"  bigram 단위 stride 8: 스텝 {steps_b8:,}, 전체 {cost_b8 / 1e9:.1f} G MAC, "
          f"표현 가능 {table[8][1][1][0]}/{len(utt)}")
    print(f"  같은 표현 범위를 지키면서 계산량이 {cost_c4 / cost_b8:.2f}배 줄어든다")

    print("\n== 7. 정렬 수는 여유에 따라 얼마나 늘어나나 (라벨 \"the cat\", 7글자) ==")
    for T in (7, 8, 10, 14, 20, 30):
        n = count_alignments(list("the cat"), T)
        print(f"  T={T:>3}: 정렬 {n:>22,}개  (log10 = {math.log10(n) if n else float('-inf'):.2f})")

    print(f"\n소요 {time.time() - t0:.1f}초")


if __name__ == "__main__":
    main()
