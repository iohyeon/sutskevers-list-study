"""문서의 손계산 값을 코드로 다시 계산합니다.

python3 hand_calc.py
"""
import math

import numpy as np


def section(t):
    print(f"\n## {t}")


def softmax(z):
    z = np.asarray(z, dtype=float)
    e = np.exp(z - z.max())
    return e / e.sum()


if __name__ == "__main__":
    section("내적과 행렬 곱")
    W = np.array([[1, -2, 0.5], [0, 1, 1]]); x = np.array([4, 1, 2]); b = np.array([-1, -5])
    print("Wx =", W @ x, " Wx + b =", W @ x + b, " ReLU =", np.maximum(W @ x + b, 0))

    section("sigmoid와 미분")
    for z in (0, 2, 5, 10):
        s = 1 / (1 + math.exp(-z))
        print(f"z={z:>2}  sigmoid={s:.5f}  미분={s*(1-s):.6f}")
    print("0.25^8 =", 0.25 ** 8)

    section("항 하나의 크기를 20번 곱한 값")
    for a in (0.25, 0.9, 1.0, 1.1, 2.0):
        print(f"{a}^20 = {a**20:.3g}")

    section("softmax와 cross-entropy, 점수 (2, 1, 0.1), 정답은 첫 번째")
    p = softmax([2.0, 1.0, 0.1])
    print("p =", np.round(p, 3), " loss =", round(-math.log(p[0]), 3), " 기울기 p - y =", np.round(p - [1, 0, 0], 4))
    for q in (0.99, 0.659, 0.1, 0.001):
        print(f"-ln({q}) = {-math.log(q):.3f}")
    print("1,000개 클래스를 균등하게 찍을 때:", round(math.log(1000), 3), " 10개:", round(math.log(10), 3))

    section("역전파 손계산: x=2, y=1, w1=0.5, b1=0, w2=1.5, b2=0, 학습률 0.1")
    xv, yv, w1, b1, w2, b2, lr = 2.0, 1.0, 0.5, 0.0, 1.5, 0.0, 0.1
    z1 = w1 * xv + b1; h = max(0.0, z1); yh = w2 * h + b2; L = 0.5 * (yh - yv) ** 2
    dy = yh - yv; dw2 = dy * h; db2 = dy; dh = dy * w2; dz1 = dh * (1.0 if z1 > 0 else 0.0)
    dw1 = dz1 * xv; db1 = dz1
    print(f"순전파 z1={z1} h={h} y_hat={yh} L={L}")
    print(f"기울기 dL/dy_hat={dy} dw2={dw2} db2={db2} dL/dh={dh} dz1={dz1} dw1={dw1} db1={db1}")
    w1, b1, w2, b2 = w1 - lr * dw1, b1 - lr * db1, w2 - lr * dw2, b2 - lr * db2
    z1 = w1 * xv + b1; h = max(0.0, z1); yh = w2 * h + b2
    print(f"갱신 후 w1={w1:.3f} b1={b1:.3f} w2={w2:.2f} b2={b2:.2f} -> z1={z1:.3f} y_hat={yh:.5f} L={0.5*(yh-yv)**2:.4f}")

    def loss(w1, b1, w2, b2, x=2.0, y=1.0):
        return 0.5 * (w2 * max(0.0, w1 * x + b1) + b2 - y) ** 2
    eps = 1e-6
    print("수치 미분 dL/dw1 =", (loss(0.5 + eps, 0, 1.5, 0) - loss(0.5 - eps, 0, 1.5, 0)) / (2 * eps))

    section("경사하강법 L(w) = (w-3)^2, w0 = 0, 학습률 0.1")
    w = 0.0
    for step in range(4):
        g = 2 * (w - 3)
        print(f"걸음 {step}: w={w:.3f}  L'={g:.3f}  L={(w-3)**2:.2f}")
        w -= 0.1 * g
    for eta in (0.01, 0.1, 0.5, 0.9, 1.1):
        print(f"학습률 {eta}: 남은 거리에 곱해지는 값 {1 - 2*eta:+.2f}")

    section("좁은 골짜기 f = (x^2 + 100 y^2) / 2, (-10, 1)에서 40걸음")
    for eta in (0.019, 0.002):
        x_, y_ = -10.0, 1.0
        for _ in range(40):
            x_, y_ = x_ - eta * x_, y_ - eta * 100 * y_
        print(f"학습률 {eta}: x={x_:.2f}  y={y_:.4f}")

    section("같은 골짜기에서 momentum을 더한 40걸음 (학습률 0.019)")
    for beta in (0.0, 0.5, 0.9):
        x_, y_, vx, vy = -10.0, 1.0, 0.0, 0.0
        for _ in range(40):
            vx, vy = beta * vx - 0.019 * x_, beta * vy - 0.019 * 100 * y_
            x_, y_ = x_ + vx, y_ + vy
        print(f"momentum {beta}: x={x_:.2f}  y={y_:.4f}")

    section("momentum 0.9의 등비급수")
    print("같은 방향:", round(1 / (1 - 0.9), 2), " 부호가 번갈아:", round(1 / (1 + 0.9), 3))

    section("초기화 표준편차 0.01일 때 layer마다 분산에 곱해지는 값 n Var(w)")
    for n in (1000, 5 * 5 * 48):
        print(f"n={n}: {n * 0.01**2:.2f}")
    print("0.1을 8번 곱하면", 0.1 ** 8)

    section("dropout: a=(2,4,6,8), p=0.5, 마스크 (1,0,0,1)")
    a = np.array([2, 4, 6, 8]); m = np.array([1, 0, 0, 1])
    print("끈 결과", a * m, " inverted", a * m / 0.5)

    section("FC로 이미지를 받을 때의 가중치 수")
    n_in, n_c1 = 3 * 227 * 227, 96 * 55 * 55
    print(f"입력 {n_in:,d}개를 뉴런 4,096개에 연결: {n_in*4096:,d}")
    print(f"conv1 입출력을 FC로 연결: {n_in*n_c1:,d}, conv1 파라미터 34,944개의 {n_in*n_c1/34944:,.0f}배")
