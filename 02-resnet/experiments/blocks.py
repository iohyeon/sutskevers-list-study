"""ResNet의 블록 세 가지(BasicBlock, Bottleneck, pre-activation)와 파라미터 수, 모양, 항등 동작 확인.

실행: python3 blocks.py
다른 스크립트에서 import해서 블록 정의를 가져다 쓸 수 있다.
"""
import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    """3x3 conv 두 장. ResNet-18, 34에서 쓴다."""
    expansion = 1  # 블록 출력 채널 = planes * expansion

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        # BN이 뒤따르므로 bias=False (05-batch-normalization.md)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

        # 모양이 바뀌는 자리에서만 projection shortcut (ResNet 논문의 선택지 B)
        self.shortcut = nn.Identity()
        if stride != 1 or in_planes != planes * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes * self.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * self.expansion),
            )

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        out = out + identity          # skip connection
        return self.relu(out)


class Bottleneck(nn.Module):
    """1x1 -> 3x3 -> 1x1. ResNet-50, 101, 152에서 쓴다."""
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        out_planes = planes * self.expansion
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)   # 채널을 줄인다
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)           # 공간 방향 처리
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, out_planes, kernel_size=1, bias=False)  # 채널을 되돌린다
        self.bn3 = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

        self.shortcut = nn.Identity()
        if stride != 1 or in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_planes),
            )

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        out = out + identity
        return self.relu(out)


class SwitchableBasicBlock(BasicBlock):
    """같은 블록에서 덧셈만 켜고 끌 수 있게 한 것. plain과 residual 비교용."""

    def __init__(self, in_planes, planes, stride=1, use_skip=True):
        super().__init__(in_planes, planes, stride)
        self.use_skip = use_skip

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.use_skip:
            out = out + self.shortcut(x)
        return self.relu(out)


class PreActBasicBlock(nn.Module):
    """ResNet v2의 full pre-activation 블록. 덧셈 뒤에 아무 연산도 없다."""
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.shortcut = None
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Conv2d(in_planes, planes, 1, stride, bias=False)

    def forward(self, x):
        out = torch.relu(self.bn1(x))
        # 모양이 바뀌는 블록에서는 BN-ReLU를 지난 값에서 shortcut을 뽑는다
        identity = self.shortcut(out) if self.shortcut is not None else x
        out = self.conv1(out)
        out = self.conv2(torch.relu(self.bn2(out)))
        return out + identity          # 덧셈 뒤에 아무것도 없다


def count_conv_weights(module):
    """conv 가중치만 센다. BN의 gamma, beta는 뺀다."""
    return sum(m.weight.numel() for m in module.modules() if isinstance(m, nn.Conv2d))


def count_all(module):
    return sum(p.numel() for p in module.parameters())


def main():
    torch.manual_seed(0)
    basic = BasicBlock(64, 64)
    bottle = Bottleneck(256, 64)

    print("== 파라미터 수")
    print("BasicBlock(64, 64)    conv:", count_conv_weights(basic))    # 2 * 9*64*64 = 73,728
    print("Bottleneck(256, 64)   conv:", count_conv_weights(bottle))   # 16,384 + 36,864 + 16,384 = 69,632
    print("BasicBlock(64, 64)    all :", count_all(basic))             # 73,728 + BN 2개 * 64*2 = 73,984
    print("Bottleneck(256, 64)   all :", count_all(bottle))            # 69,632 + (64+64+256)*2 = 70,400

    print("\n== 모양")
    print("BasicBlock  (2,64,56,56)  ->", tuple(basic(torch.randn(2, 64, 56, 56)).shape))
    print("Bottleneck  (2,256,56,56) ->", tuple(bottle(torch.randn(2, 256, 56, 56)).shape))
    down = BasicBlock(64, 128, stride=2)
    print("BasicBlock(64,128,stride=2) (2,64,56,56) ->", tuple(down(torch.randn(2, 64, 56, 56)).shape))
    print("projection shortcut conv:", count_conv_weights(down.shortcut))   # 1*64*128 = 8,192

    print("\n== skip을 끄면")
    sw_on, sw_off = SwitchableBasicBlock(64, 64, use_skip=True), SwitchableBasicBlock(64, 64, use_skip=False)
    print("use_skip=True  params:", count_all(sw_on))
    print("use_skip=False params:", count_all(sw_off))

    print("\n== 초기 상태에서 항등으로 동작하는가")
    torch.manual_seed(0)
    blk = BasicBlock(64, 64)
    nn.init.zeros_(blk.bn2.weight)                  # zero-init residual: 마지막 BN의 gamma를 0으로
    blk.eval()
    x_pos = torch.relu(torch.randn(1, 64, 8, 8))    # 0 이상인 입력
    x_any = torch.randn(1, 64, 8, 8)                # 음수 포함
    print("post-activation, zero-init, 입력 >= 0     : y == x ->", torch.allclose(blk(x_pos), x_pos))
    print("post-activation, zero-init, 음수 포함 입력 : y == x ->", torch.allclose(blk(x_any), x_any))
    pre = PreActBasicBlock(64, 64)
    nn.init.zeros_(pre.conv2.weight)
    pre.eval()
    print("pre-activation,  F = 0,     음수 포함 입력 : y == x ->", torch.allclose(pre(x_any), x_any))

    print("\n== torchvision 구현")
    import torchvision
    m50 = torchvision.models.resnet50(weights=None)
    m18 = torchvision.models.resnet18(weights=None)
    m34 = torchvision.models.resnet34(weights=None)
    m101 = torchvision.models.resnet101(weights=None)
    m152 = torchvision.models.resnet152(weights=None)
    for name, m in [("resnet18", m18), ("resnet34", m34), ("resnet50", m50),
                    ("resnet101", m101), ("resnet152", m152)]:
        print(f"{name:10s} params: {count_all(m):,}")
    print("resnet50 layer1[0] downsample:", m50.layer1[0].downsample is not None,
          "| layer1[1] downsample:", m50.layer1[1].downsample is not None)


if __name__ == "__main__":
    main()
