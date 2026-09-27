"""PyTorch 모듈로 파라미터 수와 몇 가지 성질을 직접 확인한다.

- bottleneck 블록과 3x3 두 장의 conv 가중치 수
- ResNeXt 32x4d: 경로 32개를 따로 만든 것과 grouped convolution의 가중치 수가 같은지, 출력도 같은지
- conv 뒤에 BN이 오면 conv의 bias가 출력에 영향을 주지 않는지
- BN 손계산 [2, 4, 6, 8]
- 수용 영역: dilation 1, 2, 4, 8, 16을 쌓았을 때 출력 한 칸에 영향을 주는 입력 범위(기울기로 측정)
- VGG-16의 파라미터와 그중 FC layer의 몫
실행: python3 param_count.py
"""
import torch
import torch.nn as nn
import torchvision


def conv_weights(module):
    return sum(m.weight.numel() for m in module.modules() if isinstance(m, nn.Conv2d))


def main():
    torch.manual_seed(0)

    print("== bottleneck 대 3x3 두 장 (256채널)")
    bottleneck = nn.Sequential(nn.Conv2d(256, 64, 1, bias=False),
                               nn.Conv2d(64, 64, 3, padding=1, bias=False),
                               nn.Conv2d(64, 256, 1, bias=False))
    two_3x3 = nn.Sequential(nn.Conv2d(256, 256, 3, padding=1, bias=False),
                            nn.Conv2d(256, 256, 3, padding=1, bias=False))
    b, t = conv_weights(bottleneck), conv_weights(two_3x3)
    print(f"bottleneck {b:,}   3x3 두 장 {t:,}   비율 {t / b:.1f}")
    basic64 = nn.Sequential(nn.Conv2d(64, 64, 3, padding=1, bias=False),
                            nn.Conv2d(64, 64, 3, padding=1, bias=False))
    print(f"64채널 basic block의 3x3 두 장 {conv_weights(basic64):,}")

    print("\n== ResNeXt 32x4d")
    C, d = 32, 4
    paths = nn.ModuleList([nn.Sequential(nn.Conv2d(256, d, 1, bias=False),
                                         nn.Conv2d(d, d, 3, padding=1, bias=False),
                                         nn.Conv2d(d, 256, 1, bias=False)) for _ in range(C)])
    grouped = nn.Sequential(nn.Conv2d(256, C * d, 1, bias=False),
                            nn.Conv2d(C * d, C * d, 3, padding=1, groups=C, bias=False),
                            nn.Conv2d(C * d, 256, 1, bias=False))
    print(f"경로 하나 {conv_weights(paths[0]):,}   경로 32개 {conv_weights(paths):,}   grouped 표현 {conv_weights(grouped):,}")
    print("grouped 3x3의 가중치 모양:", tuple(grouped[1].weight.shape), "개수", grouped[1].weight.numel())
    # 경로 32개의 가중치를 grouped 표현에 옮겨 담고 출력이 같은지 본다
    with torch.no_grad():
        for i, p in enumerate(paths):
            grouped[0].weight[i * d:(i + 1) * d] = p[0].weight
            grouped[1].weight[i * d:(i + 1) * d] = p[1].weight
            grouped[2].weight[:, i * d:(i + 1) * d] = p[2].weight
        x = torch.randn(2, 256, 8, 8)
        y_paths = sum(p(x) for p in paths)
        y_grouped = grouped(x)
    print("경로 32개를 더한 출력 == grouped 출력:", torch.allclose(y_paths, y_grouped, atol=1e-5))

    print("\n== conv 뒤 BN에서 bias가 상쇄되는가")
    c1 = nn.Conv2d(64, 64, 3, padding=1, bias=False)
    c2 = nn.Conv2d(64, 64, 3, padding=1, bias=True)
    with torch.no_grad():
        c2.weight.copy_(c1.weight)
        nn.init.normal_(c2.bias, std=3.0)
    bn = nn.BatchNorm2d(64).train()
    x = torch.randn(8, 64, 16, 16)
    print("bias 없는 conv + BN == bias 있는 conv + BN:", torch.allclose(bn(c1(x)), bn(c2(x)), atol=1e-4))

    print("\n== BN 손계산")
    bn1 = nn.BatchNorm1d(1, eps=1e-12).train()   # 손계산과 맞추려고 eps를 거의 0으로
    with torch.no_grad():
        bn1.weight.fill_(2.0)      # gamma
        bn1.bias.fill_(1.0)        # beta
    x = torch.tensor([[2.0], [4.0], [6.0], [8.0]])
    xhat = (x - x.mean()) / x.var(unbiased=False).sqrt()
    print("x_hat:", [round(v, 3) for v in xhat.flatten().tolist()])
    print("y (gamma=2, beta=1):", [round(v, 3) for v in bn1(x).flatten().tolist()])
    print("x + 100 의 x_hat:", [round(v, 3) for v in ((x + 100 - (x + 100).mean()) / (x + 100).var(unbiased=False).sqrt()).flatten().tolist()])

    print("\n== 수용 영역을 기울기로 재기 (3x3, stride 1)")
    for name, dilations in [("dilation 1,1,1,1,1", [1, 1, 1, 1, 1]),
                            ("dilation 1,2,4,8,16", [1, 2, 4, 8, 16])]:
        rf = []
        layers = []
        for dl in dilations:
            layers.append(nn.Conv2d(1, 1, 3, padding=dl, dilation=dl, bias=False))
            net = nn.Sequential(*layers)
            for m in net:
                nn.init.constant_(m.weight, 1.0)
            x = torch.zeros(1, 1, 129, 129, requires_grad=True)
            net(x)[0, 0, 64, 64].backward()
            cols = (x.grad[0, 0].abs().sum(0) > 0).nonzero().flatten()
            rf.append(int(cols.max() - cols.min() + 1))
        print(f"{name}: layer별 수용 영역 {rf}")

    print("\n== VGG-16")
    vgg = torchvision.models.vgg16(weights=None)
    total = sum(p.numel() for p in vgg.parameters())
    fc = sum(p.numel() for p in vgg.classifier.parameters())
    print(f"전체 {total:,}   FC {fc:,} ({100 * fc / total:.1f}%)")
    vgg19 = torchvision.models.vgg19(weights=None)
    print(f"VGG-19 전체 {sum(p.numel() for p in vgg19.parameters()):,}")
    r152 = torchvision.models.resnet152(weights=None)
    print(f"ResNet-152 마지막 FC {sum(p.numel() for p in r152.fc.parameters()):,}")


if __name__ == "__main__":
    main()
