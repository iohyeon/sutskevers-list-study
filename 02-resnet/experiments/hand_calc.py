"""문서의 손계산 숫자를 다시 계산한다. 표준 라이브러리만 쓴다.

실행: python3 hand_calc.py
"""
import math


def main():
    print("== layer마다 c가 곱해질 때 L개를 지난 뒤의 기울기 배율 c^L")
    for c in (0.5, 0.9, 1.0, 1.1):
        row = "  ".join(f"L={L}: {c ** L:.2e}" for L in (8, 20, 56, 152))
        print(f"c={c}: {row}")
    print(f"sigmoid 도함수 최댓값 0.25를 10번: {0.25 ** 10:.2e}")
    print(f"서비스 50개 직렬, 성공률 0.99: {0.99 ** 50:.2f}")

    print("\n== plain 대 residual, 블록 50개")
    print(f"0.1^50 = {0.1 ** 50:.1e}   1.1^50 = {1.1 ** 50:.1f}")
    for lam in (1.0, 0.9, 0.5):
        print(f"shortcut에 {lam}를 곱하면 50블록 뒤 직통 항: {lam ** 50:.1e}")

    print("\n== 수용 영역 r_l = r_(l-1) + (k-1) * d * (앞 layer들의 stride 곱)")
    def rf(layers):
        r, jump, out = 1, 1, []
        for k, s, d in layers:
            r += (k - 1) * d * jump
            jump *= s
            out.append(r)
        return out
    print("3x3 세 장:", rf([(3, 1, 1)] * 3))
    print("ResNet 앞부분 (7x7 s2, 3x3 pool s2, 3x3, 3x3):", rf([(7, 2, 1), (3, 2, 1), (3, 1, 1), (3, 1, 1)]))
    print("dilation 1,2,4,8,16:", rf([(3, 1, d) for d in (1, 2, 4, 8, 16)]))
    print("dilation 없이 3x3 다섯 장:", rf([(3, 1, 1)] * 5))
    for d in (1, 2, 4, 8):
        print(f"3x3, dilation {d}: 한 변이 덮는 길이 {3 + 2 * (d - 1)}")
    C = 64
    print(f"C=64: 7x7 {49 * C * C:,}  3x3 세 장 {27 * C * C:,}  5x5 {25 * C * C:,}  3x3 두 장 {18 * C * C:,}")

    print("\n== 초기화: fan-in 576인 layer를 50개 지난 뒤의 분산 배율")
    n = 576
    for name, var in [("std 0.01", 0.01 ** 2), ("std 0.05", 0.05 ** 2), ("Var = 1/576", 1 / n)]:
        print(f"{name}: layer당 {n * var:.4f}, 50개 뒤 {(n * var) ** 50:.1e}")
    print(f"He 초기화 Var = 2/576 = {2 / n:.5f}, 표준편차 {math.sqrt(2 / n):.4f}")

    print("\n== bottleneck과 FLOPs (56x56 feature map)")
    bott = 256 * 64 + 9 * 64 * 64 + 64 * 256
    two = 2 * 9 * 256 * 256
    print(f"bottleneck {bott:,}  3x3 두 장 {two:,}  비율 {two / bott:.1f}")
    print(f"곱셈-덧셈: bottleneck {56 * 56 * bott / 1e8:.2f}억, 3x3 두 장 {56 * 56 * two / 1e8:.1f}억")
    print(f"projection 1x1 64->128: {64 * 128:,}   같은 블록의 3x3 두 장: {9 * 64 * 128 + 9 * 128 * 128:,}")

    print("\n== ResNet 이름의 layer 수")
    for name, blocks, per in [("18", [2, 2, 2, 2], 2), ("34", [3, 4, 6, 3], 2), ("50", [3, 4, 6, 3], 3),
                              ("101", [3, 4, 23, 3], 3), ("152", [3, 8, 36, 3], 3)]:
        print(f"ResNet-{name}: 1 + {sum(blocks)} x {per} + 1 = {1 + sum(blocks) * per + 1}")
    print("CIFAR용 6n+2:", {n: 6 * n + 2 for n in (1, 3, 5, 9, 18)})

    print("\n== ResNeXt 32x4d")
    path = 256 * 4 + 9 * 4 * 4 + 4 * 256
    print(f"경로 하나 {path:,}, 32개 {32 * path:,}, grouped 표현 {256 * 128 + 9 * 128 * 128 // 32 + 128 * 256:,}")

    print("\n== 경로 수와 경로 길이 분포 (블록 n개, 길이 = 지나는 변환 경로 수)")
    for nb in (3, 16, 54):
        print(f"n={nb}: 경로 {2 ** nb:.2e}개, 평균 길이 {nb / 2}, 표준편차 {math.sqrt(nb) / 2:.2f}")
    c = (1 + 0.1) * (1 + 0.2) * (1 + 0.3)
    expanded = 1 + (0.1 + 0.2 + 0.3) + (0.1 * 0.2 + 0.1 * 0.3 + 0.2 * 0.3) + 0.1 * 0.2 * 0.3
    print(f"(1+0.1)(1+0.2)(1+0.3) = {c:.4f}, 항 8개를 펼쳐 더한 값 = {expanded:.4f}")

    print("\n== 표본 12장, 정확도 0.33에서 맞힌 개수의 표준편차")
    print(f"{math.sqrt(12 * 0.33 * 0.67):.2f}장.  10,000장이면 {100 * math.sqrt(10000 * 0.33 * 0.67) / 10000:.2f}%p")


if __name__ == "__main__":
    main()
