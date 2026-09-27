"""문서의 손계산 숫자를 다시 찍어 보는 스크립트 (hello 네 시점, softmax temperature, perplexity, LSTM 한 시점, 야코비안 곱)."""
import numpy as np
np.set_printoptions(precision=3, suppress=True)

# vanilla RNN 으로 "hell" 네 시점 (02-핵심 개념/01 의 손계산)
Wxh = np.array([[0.5, -0.3, 0.8, 0.1], [-0.2, 0.9, 0.4, -0.6], [0.7, 0.2, -0.5, 0.3]])
Whh = np.array([[0.1, 0.4, -0.3], [-0.5, 0.2, 0.6], [0.3, -0.7, 0.1]])
Why = np.array([[0.2, -0.4, 0.1], [1.0, 0.3, -0.2], [-0.3, 0.8, 0.5], [0.4, -0.1, 0.9]])
vocab = "helo"; h = np.zeros(3); losses = []
for t, (ch, target) in enumerate(zip("hell", "ello"), 1):
    h = np.tanh(Wxh[:, vocab.index(ch)] + Whh @ h)
    y = Why @ h; p = np.exp(y) / np.exp(y).sum(); losses.append(-np.log(p[vocab.index(target)]))
    print("시점", t, "입력", ch, "h", h, "p", p, "정답", target, "%.3f" % p[vocab.index(target)])
print("hello loss %.3f perplexity %.2f" % (np.mean(losses), np.exp(np.mean(losses))))

z = np.array([2.0, 1.0, 0.1])
for T in (0.5, 1.0, 2.0, 10.0):
    p = np.exp(z / T) / np.exp(z / T).sum(); print("T", T, p)

ps = np.array([0.5, 0.25, 0.1, 0.8])
print("cross-entropy", -np.log(ps).mean(), "perplexity", np.exp(-np.log(ps).mean()))

sig = lambda v: 1 / (1 + np.exp(-v))
C_prev = np.array([0.8, -0.5])
f, i, c_tilde, o = sig(np.array([3.0, -2.0])), sig(np.array([-1.0, 2.0])), np.tanh(np.array([0.5, 1.5])), sig(np.array([0.0, 2.0]))
C = f * C_prev + i * c_tilde
print("f", f, "i", i, "C~", c_tilde, "C", C, "o", o, "h", o * np.tanh(C))

rng = np.random.default_rng(0)
for scale in (0.5, 1.0, 2.0):
    n = 50; W = rng.normal(0, scale / np.sqrt(n), (n, n)); h = np.zeros(n); J = np.eye(n); norms = []
    for t in range(60):
        h = np.tanh(W @ h + rng.normal(0, 1, n)); J = np.diag(1 - h ** 2) @ W @ J
        if t + 1 in (1, 10, 30, 60): norms.append(float(np.linalg.norm(J, 2)))
    print("sigma_max(W) = %.2f" % np.linalg.svd(W, compute_uv=False)[0], ["%.1e" % v for v in norms])
