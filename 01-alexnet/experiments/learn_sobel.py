"""무작위 3x3 커널을 경사하강법으로 학습시켜 세로 에지(Sobel) 커널의 출력을 흉내 내게 합니다.

python3 learn_sobel.py

손실은 두 feature map의 MSE이고, 커널의 기울기는 입력 위에서 dL/dY를 합성곱한 것입니다.
"""
import numpy as np

from conv_numpy import CAT, SOBEL_V, conv2d

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    target = conv2d(CAT, SOBEL_V)
    k = rng.standard_normal((3, 3)) * 0.1
    print("시작 커널\n", np.round(k, 2))
    for step in range(1, 501):
        diff = conv2d(CAT, k) - target            # dL/dY (상수배 생략)
        dk = conv2d(CAT, diff) / diff.size        # 커널의 기울기
        k -= 0.5 * dk
        if step in (1, 10, 50, 100, 500):
            print(f"step {step:>3}  MSE {np.mean(diff**2):.6f}")
    print("학습된 커널\n", np.round(k, 2))
    print("Sobel과의 최대 차이:", float(np.abs(k - SOBEL_V).max()))
