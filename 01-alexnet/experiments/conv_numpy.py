"""NumPy만으로 합성곱, ReLU, max pooling, 여러 채널 conv layer, im2col을 구현하고
12x12 고양이 그림에 적용합니다.

python3 conv_numpy.py
"""
import numpy as np

np.set_printoptions(linewidth=120)


def conv2d(x, k, stride=1, pad=0):
    """x: (H, W), k: (kh, kw). 채널이 하나인 경우."""
    if pad:
        x = np.pad(x, pad)
    H, W = x.shape
    kh, kw = k.shape
    oh = (H - kh) // stride + 1
    ow = (W - kw) // stride + 1
    out = np.zeros((oh, ow))
    for i in range(oh):
        for j in range(ow):
            window = x[i*stride:i*stride+kh, j*stride:j*stride+kw]
            out[i, j] = (window * k).sum()      # 창과 커널의 내적
    return out


def relu(x):
    return np.maximum(x, 0)


def maxpool2d(x, size=2, stride=2):
    H, W = x.shape
    oh = (H - size) // stride + 1
    ow = (W - size) // stride + 1
    out = np.zeros((oh, ow))
    for i in range(oh):
        for j in range(ow):
            out[i, j] = x[i*stride:i*stride+size, j*stride:j*stride+size].max()
    return out


def conv_layer(x, w, b, stride=1, pad=0):
    """x: (C_in, H, W), w: (C_out, C_in, kh, kw), b: (C_out,)"""
    if pad:
        x = np.pad(x, ((0, 0), (pad, pad), (pad, pad)))
    C_in, H, W = x.shape
    C_out, _, kh, kw = w.shape
    oh = (H - kh) // stride + 1
    ow = (W - kw) // stride + 1
    out = np.zeros((C_out, oh, ow))
    for o in range(C_out):                  # 필터마다
        for i in range(oh):
            for j in range(ow):
                window = x[:, i*stride:i*stride+kh, j*stride:j*stride+kw]
                out[o, i, j] = (window * w[o]).sum() + b[o]   # 모든 입력 채널에 걸친 내적
    return out


def im2col(x, kh, kw, stride=1):
    C, H, W = x.shape
    oh = (H - kh) // stride + 1
    ow = (W - kw) // stride + 1
    cols = np.zeros((oh * ow, C * kh * kw))
    for i in range(oh):
        for j in range(ow):
            cols[i*ow + j] = x[:, i*stride:i*stride+kh, j*stride:j*stride+kw].ravel()
    return cols, oh, ow


def conv_layer_fast(x, w, b, stride=1):
    C_out, C_in, kh, kw = w.shape
    cols, oh, ow = im2col(x, kh, kw, stride)          # (창의 수, C_in*kh*kw)
    out = cols @ w.reshape(C_out, -1).T + b           # 행렬 곱 한 번
    return out.T.reshape(C_out, oh, ow)


CAT = np.array([
    [0,1,0,0,0,0,0,0,0,0,1,0],
    [0,1,1,0,0,0,0,0,0,1,1,0],
    [0,1,0,1,1,1,1,1,1,0,1,0],
    [0,1,0,0,0,0,0,0,0,0,1,0],
    [0,1,0,1,1,0,0,1,1,0,1,0],
    [0,1,0,1,1,0,0,1,1,0,1,0],
    [0,1,0,0,0,0,0,0,0,0,1,0],
    [0,1,0,0,0,1,1,0,0,0,1,0],
    [1,1,1,0,0,1,1,0,0,1,1,1],
    [0,1,0,0,1,0,0,1,0,0,1,0],
    [0,0,1,0,0,0,0,0,0,1,0,0],
    [0,0,0,1,1,1,1,1,1,0,0,0]], dtype=float)
SOBEL_V = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=float)   # 세로 에지
SOBEL_H = SOBEL_V.T                                                     # 가로 에지

if __name__ == "__main__":
    X = np.array([[1, 2, 0, 1], [0, 1, 3, 1], [2, 1, 0, 0], [1, 0, 1, 2]], dtype=float)
    K = np.array([[1, 0], [0, -1]], dtype=float)
    print("## 4x4 입력, 2x2 커널\n", conv2d(X, K))
    P = np.array([[1, 3, 2, 0], [5, 2, 1, 1], [0, 1, 4, 2], [2, 0, 3, 6]], dtype=float)
    print("## 4x4 max pooling (2x2, stride 2)\n", maxpool2d(P))

    fv = conv2d(CAT, SOBEL_V)
    fh = conv2d(CAT, SOBEL_H)
    print("## 고양이 x 세로 에지 커널", fv.shape)
    print(fv.astype(int))
    print("## 고양이 x 가로 에지 커널")
    print(fh.astype(int))
    print("## 세로 에지 map에 ReLU, 2x2 max pooling")
    print(maxpool2d(relu(fv)).astype(int))
    print("## stride 2일 때 출력 크기:", conv2d(CAT, SOBEL_V, stride=2).shape)

    rng = np.random.default_rng(0)
    x = rng.standard_normal((3, 67, 67))
    w = rng.standard_normal((96, 3, 11, 11)) * 0.01     # AlexNet 초기화: 평균 0, 표준편차 0.01
    b = np.zeros(96)
    y = conv_layer(x, w, b, stride=4)
    print("## conv1과 같은 모양의 layer (입력 3x67x67)")
    print("출력 shape:", y.shape, " 파라미터 수:", w.size + b.size)
    print("im2col 결과가 반복문 결과와 같은가:", np.allclose(conv_layer_fast(x, w, b, stride=4), y))
    cols, oh, ow = im2col(x, 11, 11, 4)
    print("im2col 행렬 shape:", cols.shape)
