"""학습 없이 순전파와 역전파를 한 번 해서 layer별 기울기 크기를 찍는다.

실험 1: 50-layer MLP에서 sigmoid, ReLU(기본 초기화), ReLU(He 초기화)
실험 2: BN 없는 50-layer MLP에서 skip connection이 없을 때와 있을 때 (가중치 표준편차 0.02, 0.1)
실험 3: BN이 있는 CIFAR용 CNN(plain-56, resnet-56)
실행: python3 grad_norms.py
"""
import torch
import torch.nn as nn

from resnet_cifar import CifarNet

loss_fn = nn.CrossEntropyLoss()


def grad_norms(model, x, y):
    """역전파를 한 번 하고, 가중치 텐서마다 기울기의 RMS를 입력 쪽 layer부터 돌려준다."""
    model.zero_grad(set_to_none=True)
    loss_fn(model(x), y).backward()
    out = []
    for name, p in model.named_parameters():
        if p.grad is not None and p.dim() > 1:          # 가중치만 (bias, BN 제외)
            # 원소 수로 나눈 RMS. layer의 크기가 달라도 비교할 수 있다.
            # float32 그대로 제곱하면 아주 작은 값이 0이 되므로 float64로 바꿔서 계산한다
            out.append((name, p.grad.double().pow(2).mean().sqrt().item()))
    return out


def show(title, norms, every=5):
    print(f"\n== {title}")
    for i, (name, g) in enumerate(norms):
        if i % every == 0 or i == len(norms) - 1:
            print(f"{i:3d} {name:26s} {g:10.3e}")


def mlp(depth, width, act):
    layers = []
    for _ in range(depth):
        layers += [nn.Linear(width, width), act()]
    return nn.Sequential(*layers, nn.Linear(width, 10))


class ResMLP(nn.Module):
    """BN 없는 MLP. use_skip=True이면 x = x + relu(Wx + b)."""

    def __init__(self, depth, width, use_skip, std):
        super().__init__()
        self.use_skip = use_skip
        self.layers = nn.ModuleList([nn.Linear(width, width) for _ in range(depth)])
        self.head = nn.Linear(width, 10)
        for l in self.layers:
            nn.init.normal_(l.weight, std=std)
            nn.init.zeros_(l.bias)

    def forward(self, x):
        for l in self.layers:
            f = torch.relu(l(x))
            x = x + f if self.use_skip else f
        return self.head(x)


def first_last_ratio(model):
    """첫 conv와 마지막 conv의 기울기 RMS 비율. loss.backward() 뒤에 호출한다."""
    convs = [m for m in model.modules() if isinstance(m, nn.Conv2d)]
    first = convs[0].weight.grad.pow(2).mean().sqrt()
    last = convs[-1].weight.grad.pow(2).mean().sqrt()
    return (first / last).item()


def main():
    torch.manual_seed(0)
    x, y = torch.randn(64, 256), torch.randint(0, 10, (64,))

    # 실험 1
    show("sigmoid, 50-layer", grad_norms(mlp(50, 256, nn.Sigmoid), x, y))
    show("ReLU, 50-layer, 기본 초기화", grad_norms(mlp(50, 256, nn.ReLU), x, y))
    relu_he = mlp(50, 256, nn.ReLU)
    for m in relu_he.modules():
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            nn.init.zeros_(m.bias)
    show("ReLU, 50-layer, He 초기화", grad_norms(relu_he, x, y))

    # 실험 2
    torch.manual_seed(0)
    for use_skip in (False, True):
        m = ResMLP(depth=50, width=256, use_skip=use_skip, std=0.02)
        show(f"BN 없는 MLP, std=0.02, skip={use_skip}", grad_norms(m, x, y))
    print()
    for use_skip in (False, True):
        m = ResMLP(depth=50, width=256, use_skip=use_skip, std=0.1)
        ns = [g for _, g in grad_norms(m, x, y)][:-1]      # head 제외
        print(f"BN 없는 MLP, std=0.1, skip={use_skip}: 가중치 기울기 RMS 최소 {min(ns):.3e}, 최대 {max(ns):.3e}")

    # 실험 3
    torch.manual_seed(0)
    xi, yi = torch.randn(64, 3, 32, 32), torch.randint(0, 10, (64,))
    for use_skip in (False, True):
        net = CifarNet(n=9, use_skip=use_skip).train()      # BN이 배치 통계를 쓰도록
        show(f"CifarNet-56, skip={use_skip}", grad_norms(net, xi, yi), every=6)
    print()
    for n, use_skip in [(3, True), (9, False), (9, True)]:
        torch.manual_seed(0)
        net = CifarNet(n=n, use_skip=use_skip).train()
        loss_fn(net(xi), yi).backward()
        name = f"{'resnet' if use_skip else 'plain'}-{6 * n + 2}"
        print(f"{name}: 첫 conv / 마지막 conv 기울기 RMS 비율 = {first_last_ratio(net):.3e}")


if __name__ == "__main__":
    main()
