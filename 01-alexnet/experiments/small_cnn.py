"""AlexNet을 32x32 입력에 맞게 줄인 CNN으로 CIFAR-10 학습 설정을 비교합니다.

python3 small_cnn.py                 # 비교 실험 전체 (CPU 축소판: 학습 10,000장, 20 epoch)
python3 small_cnn.py --params        # 파라미터 수만 출력

데이터는 torchvision이 ./data 에 내려받습니다. 학습 이미지 수와 epoch은 --train-size, --epochs 로 바꿉니다.
"""
import argparse
import time

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T

MEAN, STD = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)


class SmallAlexNet(nn.Module):
    def __init__(self, act="relu", p_drop=0.5):
        super().__init__()
        A = nn.ReLU if act == "relu" else nn.Tanh
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), A(), nn.MaxPool2d(2),      # 32 -> 16
            nn.Conv2d(64, 128, 3, padding=1), A(), nn.MaxPool2d(2),    # 16 -> 8
            nn.Conv2d(128, 256, 3, padding=1), A(),
            nn.Conv2d(256, 256, 3, padding=1), A(),
            nn.Conv2d(256, 128, 3, padding=1), A(), nn.MaxPool2d(2),   # 8 -> 4
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),                       # 128 * 4 * 4 = 2048
            nn.Dropout(p_drop), nn.Linear(2048, 512), A(),
            nn.Dropout(p_drop), nn.Linear(512, 512), A(),
            nn.Linear(512, 10),                 # softmax는 손실 함수가 처리합니다
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def he_init(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            nn.init.zeros_(m.bias)


def loaders(augment, train_size, test_size, batch_size=128, root="./data"):
    aug = [T.RandomCrop(32, padding=4), T.RandomHorizontalFlip()] if augment else []
    train_tf = T.Compose(aug + [T.ToTensor(), T.Normalize(MEAN, STD)])
    test_tf = T.Compose([T.ToTensor(), T.Normalize(MEAN, STD)])
    train = torchvision.datasets.CIFAR10(root, train=True, download=True, transform=train_tf)
    test = torchvision.datasets.CIFAR10(root, train=False, download=True, transform=test_tf)
    train = torch.utils.data.Subset(train, range(train_size))
    test = torch.utils.data.Subset(test, range(test_size))
    return (torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=0),
            torch.utils.data.DataLoader(test, batch_size=500, num_workers=0))


def evaluate(model, loader):
    model.eval()                                       # dropout을 끕니다
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            correct += (model(x).argmax(1) == y).sum().item()
            total += len(y)
    return correct / total


def run(name, act="relu", p_drop=0.5, augment=True, momentum=0.9, init="default",
        epochs=20, lr=0.01, train_size=10000, test_size=2000, seed=0):
    torch.manual_seed(seed)
    model = SmallAlexNet(act, p_drop)
    if init == "he":
        he_init(model)
    train_loader, test_loader = loaders(augment, train_size, test_size)
    loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=5e-4)
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.train()                                  # dropout을 켭니다
        total = correct = 0
        loss_sum = 0.0
        for x, y in train_loader:
            logits = model(x)
            loss = loss_fn(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            loss_sum += loss.item() * len(y)
            correct += (logits.argmax(1) == y).sum().item()
            total += len(y)
        test_acc = evaluate(model, test_loader)
        print(f"{name:<26s} epoch {epoch}  loss {loss_sum/total:.3f}  "
              f"train acc {correct/total:.3f}  test acc {test_acc:.3f}", flush=True)
    print(f"{name:<26s} ({time.time()-t0:.0f}s)", flush=True)


CONFIGS = [
    ("relu (기준)", dict()),
    ("relu + He 초기화", dict(init="he")),
    ("tanh", dict(act="tanh")),
    ("relu + He, dropout 없음", dict(init="he", p_drop=0.0)),
    ("relu + He, augmentation 없음", dict(init="he", augment=False)),
    ("relu + He, 둘 다 없음", dict(init="he", p_drop=0.0, augment=False)),
    ("relu + He, momentum 0", dict(init="he", momentum=0.0)),
]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", action="store_true")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--train-size", type=int, default=10000)
    ap.add_argument("--test-size", type=int, default=2000)
    a = ap.parse_args()
    if a.params:
        m = SmallAlexNet()
        for n, p in m.named_parameters():
            print(f"{n:<22s} {p.numel():>10,d}")
        print(f"{'total':<22s} {sum(p.numel() for p in m.parameters()):>10,d}")
    else:
        for name, kw in CONFIGS:
            run(name, epochs=a.epochs, train_size=a.train_size, test_size=a.test_size, **kw)
