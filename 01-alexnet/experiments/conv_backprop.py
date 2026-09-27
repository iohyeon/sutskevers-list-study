"""conv layer의 역전파를 3x3 입력, 2x2 커널로 계산하고 수치 미분과 비교합니다.

python3 conv_backprop.py
"""
import numpy as np

from conv_numpy import conv2d

X = np.array([[1, 2, 0], [0, 1, 3], [2, 1, 0]], dtype=float)
G = np.array([[1, 0], [-1, 2]], dtype=float)      # dL/dY 로 둡니다 (L = sum(G * Y))
K = np.array([[0.5, -1.0], [2.0, 0.0]])

if __name__ == "__main__":
    dK = conv2d(X, G)                                   # 커널의 기울기
    dX = conv2d(np.pad(G, 1), np.rot90(K, 2))           # 입력의 기울기
    eps = 1e-6
    num = np.zeros_like(K)
    for m in range(2):
        for n in range(2):
            Kp, Km = K.copy(), K.copy(); Kp[m, n] += eps; Km[m, n] -= eps
            num[m, n] = ((G * conv2d(X, Kp)).sum() - (G * conv2d(X, Km)).sum()) / (2 * eps)
    numX = np.zeros_like(X)
    for i in range(3):
        for j in range(3):
            Xp, Xm = X.copy(), X.copy(); Xp[i, j] += eps; Xm[i, j] -= eps
            numX[i, j] = ((G * conv2d(Xp, K)).sum() - (G * conv2d(Xm, K)).sum()) / (2 * eps)
    print("dK\n", dK)
    print("dK가 수치 미분과 일치:", np.allclose(num, dK))
    print("dX\n", dX)
    print("dX가 수치 미분과 일치:", np.allclose(numX, dX))
