"""입력을 1%만 바꿨을 때 입력에 대한 기울기의 방향이 얼마나 유지되는지 잰다.

plain-56과 resnet-56, 학습 전 초기화 상태, seed 0, 1, 2.
BN이 배치 통계를 쓰도록 train 모드에서 잰다. 학습 전의 BN은 running mean이 0, running var가 1이라
eval 모드에서는 아무 일도 하지 않고, BN이 없는 네트워크를 재는 셈이 된다.
실행: python3 cosine_similarity.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from resnet_cifar import CifarNet

loss_fn = nn.CrossEntropyLoss()


def input_grad(net, x, y):
    x = x.clone().requires_grad_(True)
    loss_fn(net(x), y).backward()
    return x.grad.flatten(1)


def main():
    for seed in (0, 1, 2):
        row = []
        for use_skip in (False, True):
            torch.manual_seed(seed)
            net = CifarNet(n=9, use_skip=use_skip).train()
            base = torch.randn(16, 3, 32, 32)
            noisy = base + 0.01 * torch.randn_like(base)      # 아주 조금 다른 입력
            label = torch.zeros(16, dtype=torch.long)
            g1, g2 = input_grad(net, base, label), input_grad(net, noisy, label)
            row.append(F.cosine_similarity(g1, g2).mean().item())
        print(f"seed {seed}: plain-56 {row[0]:.3f}   resnet-56 {row[1]:.3f}")


if __name__ == "__main__":
    main()
