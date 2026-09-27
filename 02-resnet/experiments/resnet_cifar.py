"""CIFAR-10용 plain 네트워크와 ResNet(6n+2 layer)을 같은 설정으로 학습한다.

실행 예: python3 resnet_cifar.py --n 3 --plain --limit 5000 --epochs 3
학습 오차와 테스트 오차를 epoch마다 출력하고 <이름>.json 으로 저장한다.
"""
import argparse
import json
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T

class Block(nn.Module):
    """use_skip=False 이면 plain, True 이면 residual. 그 밖의 모든 것은 같다."""

    def __init__(self, in_planes, planes, stride, use_skip):
        super().__init__()
        self.use_skip = use_skip
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Identity()
        if use_skip and (stride != 1 or in_planes != planes):
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.use_skip:
            out = out + self.shortcut(x)
        return F.relu(out)

class CifarNet(nn.Module):
    def __init__(self, n, use_skip, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        layers, in_planes = [], 16
        for planes, stride in [(16, 1), (32, 2), (64, 2)]:
            for i in range(n):
                layers.append(Block(in_planes, planes, stride if i == 0 else 1, use_skip))
                in_planes = planes
        self.blocks = nn.Sequential(*layers)
        self.fc = nn.Linear(64, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.blocks(x)
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)   # global average pooling
        return self.fc(x)

def get_loaders(batch_size, limit=0):
    mean, std = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
    train_tf = T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(),
                          T.ToTensor(), T.Normalize(mean, std)])
    test_tf = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
    train = torchvision.datasets.CIFAR10("./data", train=True, download=True, transform=train_tf)
    test = torchvision.datasets.CIFAR10("./data", train=False, download=True, transform=test_tf)
    if limit:                                        # 코드가 도는지 먼저 볼 때 학습 이미지 수를 줄인다
        train = torch.utils.data.Subset(train, range(limit))
    return (torch.utils.data.DataLoader(train, batch_size, shuffle=True, num_workers=2),
            torch.utils.data.DataLoader(test, 512, shuffle=False, num_workers=2))

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()   # BN이 running 통계를 쓰게 한다
    wrong = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        wrong += (model(x).argmax(1) != y).sum().item()
        total += y.numel()
    return 100.0 * wrong / total

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=3, help="stage당 블록 수. layer 수 = 6n + 2")
    p.add_argument("--plain", action="store_true", help="skip connection을 끈다")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--limit", type=int, default=0, help="학습 이미지 수. 0이면 50,000장 전체")
    p.add_argument("--device", default="auto", help="auto, cpu, mps, cuda")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = args.device
    if device == "auto":
        device = ("cuda" if torch.cuda.is_available()
                  else "mps" if torch.backends.mps.is_available() else "cpu")
    model = CifarNet(args.n, use_skip=not args.plain).to(device)
    train_loader, test_loader = get_loaders(args.batch_size, args.limit)

    opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    # 전체의 50%, 75% 지점에서 학습률을 10분의 1로 (원 논문의 32k, 48k / 64k 비율)
    sched = torch.optim.lr_scheduler.MultiStepLR(
        opt, milestones=[args.epochs // 2, args.epochs * 3 // 4], gamma=0.1)

    name = f"{'plain' if args.plain else 'resnet'}-{6 * args.n + 2}"
    n_params = sum(p.numel() for p in model.parameters())
    print(f"{name}  params={n_params:,}  device={device}")

    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, wrong, total = time.time(), 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            wrong += (logits.argmax(1) != y).sum().item()
            total += y.numel()
        sched.step()
        train_err = 100.0 * wrong / total
        test_err = evaluate(model, test_loader, device)
        history.append({"epoch": epoch, "train_err": train_err, "test_err": test_err})
        print(f"[{name}] epoch {epoch:3d}  train_err {train_err:5.2f}%  "
              f"test_err {test_err:5.2f}%  ({time.time() - t0:.0f}s)")

    with open(f"{name}.json", "w") as f:
        json.dump(history, f)

if __name__ == "__main__":
    main()
