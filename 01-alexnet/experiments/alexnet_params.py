"""AlexNet의 layer별 출력 크기, 파라미터 수, 뉴런 수, 곱셈 횟수를 계산합니다.

python3 alexnet_params.py

입력은 227 x 227을 씁니다(224를 넣으면 conv1 출력이 55가 되지 않습니다).
두 GPU 분할 때문에 conv2, conv4, conv5는 같은 GPU에 있는 입력 채널 절반만 봅니다(groups=2).
"""


def out_size(i, k, s=1, p=0):
    return (i - k + 2 * p) // s + 1


# (이름, 커널, stride, pad, 입력 채널, 출력 채널, groups, 뒤따르는 max pool 여부)
CONV = [
    ("conv1", 11, 4, 0, 3, 96, 1, True),
    ("conv2", 5, 1, 2, 96, 256, 2, True),
    ("conv3", 3, 1, 1, 256, 384, 1, False),
    ("conv4", 3, 1, 1, 384, 384, 2, False),
    ("conv5", 3, 1, 1, 384, 256, 2, True),
]
FC = [("FC6", 9216, 4096), ("FC7", 4096, 4096), ("FC8", 4096, 1000)]


def count(split=True, only=None):
    """split=False 이면 모든 conv가 입력 채널 전체를 본다고 가정합니다.
    only 로 이름을 주면 그 layer만 분할을 반영하지 않습니다."""
    size, rows = 227, []
    for name, k, s, p, cin, cout, g, pool in CONV:
        if not split or name == only:
            g = 1
        size = out_size(size, k, s, p)
        per_out = k * k * (cin // g)             # 출력 칸 하나에 드는 곱셈 수
        params = per_out * cout + cout            # 가중치 + bias
        neurons = cout * size * size
        rows.append((name, f"{cout} x {size} x {size}", params, neurons, neurons * per_out))
        if pool:
            size = out_size(size, 3, 2)
    for name, nin, nout in FC:
        rows.append((name, f"{nout}", (nin + 1) * nout, nout, nin * nout))
    return rows


def show(rows):
    P = sum(r[2] for r in rows)
    M = sum(r[4] for r in rows)
    print(f"{'layer':<6}{'출력':>16}{'파라미터':>14}{'비율':>8}{'뉴런':>10}{'곱셈':>16}{'비율':>8}")
    for name, shape, p, n, m in rows:
        print(f"{name:<6}{shape:>16}{p:>14,d}{p/P:>8.2%}{n:>10,d}{m:>16,d}{m/M:>8.2%}")
    conv = [r for r in rows if r[0].startswith("conv")]
    fc = [r for r in rows if r[0].startswith("FC")]
    for label, part in (("conv 합", conv), ("FC 합", fc)):
        p, n, m = (sum(r[i] for r in part) for i in (2, 3, 4))
        print(f"{label:<6}{'':>16}{p:>14,d}{p/P:>8.2%}{n:>10,d}{m:>16,d}{m/M:>8.2%}")
    print(f"{'전체':<6}{'':>16}{P:>14,d}{'':>8}{sum(r[3] for r in rows):>10,d}{M:>16,d}")
    return P


if __name__ == "__main__":
    print("## 논문 구조 (두 GPU 분할 반영)")
    P = show(count())
    print(f"\n파라미터 메모리 (float32): {P*4/1e6:.0f} MB, 기울기와 momentum 포함 {P*4*3/1e6:.0f} MB")
    print("\n## 분할을 반영하지 않을 때의 합계")
    print(f"모든 conv가 입력 채널 전체를 볼 때: {sum(r[2] for r in count(split=False)):,d}")
    print(f"conv2만 반영하지 않을 때:           {sum(r[2] for r in count(only='conv2')):,d}")
    print("\n## receptive field (r: 한 변의 입력 픽셀 수, j: 앞선 stride의 곱)")
    r, j = 1, 1
    for name, k, st in [("conv1", 11, 4), ("pool1", 3, 2), ("conv2", 5, 1), ("pool2", 3, 2),
                        ("conv3", 3, 1), ("conv4", 3, 1), ("conv5", 3, 1), ("pool5", 3, 2)]:
        r, j = r + (k - 1) * j, j * st
        print(f"{name:<6} r = {r:>3}  j = {j:>2}")
    print(f"\n224 x 224를 넣으면 conv1 출력: (224 - 11) / 4 + 1 = {(224 - 11) / 4 + 1}")

    try:
        import torch
        import torch.nn as nn
    except ImportError:
        raise SystemExit
    features = nn.Sequential(
        nn.Conv2d(3, 96, 11, stride=4), nn.ReLU(), nn.MaxPool2d(3, 2),
        nn.Conv2d(96, 256, 5, padding=2, groups=2), nn.ReLU(), nn.MaxPool2d(3, 2),
        nn.Conv2d(256, 384, 3, padding=1), nn.ReLU(),
        nn.Conv2d(384, 384, 3, padding=1, groups=2), nn.ReLU(),
        nn.Conv2d(384, 256, 3, padding=1, groups=2), nn.ReLU(), nn.MaxPool2d(3, 2),
    )
    classifier = nn.Sequential(nn.Flatten(), nn.Linear(9216, 4096), nn.ReLU(),
                               nn.Linear(4096, 4096), nn.ReLU(), nn.Linear(4096, 1000))
    print("\n## PyTorch로 확인 (groups=2로 두 GPU 분할을 표현, LRN 생략)")
    x = torch.zeros(1, 3, 227, 227)
    for layer in features:
        x = layer(x)
        if not isinstance(layer, nn.ReLU):
            print(f"{layer.__class__.__name__:<10}{str(tuple(x.shape)):>22}")
    print(f"{'Linear 출력':<10}{str(tuple(classifier(x).shape)):>22}")
    n = sum(p.numel() for m in (features, classifier) for p in m.parameters())
    print(f"PyTorch 파라미터 수: {n:,d}")
