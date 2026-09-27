"""학습된 LSTM 의 cell state 를 글자마다 찍어서, 어떤 차원이 '지금 proof 안인가 lemma 안인가' 를 들고 있는지 찾는다."""
import torch
from char_lstm import CharModel

ck = torch.load("model_lstm.pt"); chars = ck["chars"]; c2i = {c: i for i, c in enumerate(chars)}; a = ck["args"]
model = CharModel(len(chars), a["hidden"], a["layers"], "lstm", 0.0); model.load_state_dict(ck["model"]); model.eval()

@torch.no_grad()
def trace(text):
    state = None; cs = []
    for ch in text:
        _, state, _ = model(torch.tensor([[c2i[ch]]]), state)
        cs.append(state[1][0, 0].clone())        # state = (h, C). C 의 layer 0, 배치 0 (이 모델은 layer 가 하나다)
    return torch.stack(cs)                        # (글자 수, hidden)

text = open("corpus.txt").read()[-6000:]
text = text[text.index("\n") + 1:]
C = trace(text)
label = []; cur = None                            # 글자마다 'proof', 'lemma', None
pos = 0
for line in text.split("\n"):
    env = "proof" if line.startswith("\\begin{proof}") else "lemma" if line.startswith("\\begin{lemma}") else None
    start = len("\\begin{proof}")
    for j in range(len(line) + 1):
        label.append(env if start <= j < len(line) - len("\\end{proof}") else None)
label = label[:len(text)]
P = torch.tensor([l == "proof" for l in label]); L = torch.tensor([l == "lemma" for l in label])
gap = (C[P].mean(0) - C[L].mean(0)).abs() / (C[P].std(0) + C[L].std(0) + 1e-6)
top = gap.argsort(descending=True)[:3]
print("가장 잘 가르는 차원:", top.tolist(), "분리도:", [round(gap[i].item(), 2) for i in top])
d = top[0].item()
print(f"차원 {d}: proof 본문 평균 {C[P][:, d].mean():+.2f}, lemma 본문 평균 {C[L][:, d].mean():+.2f}")
for line in text.split("\n")[:4]:
    c = trace("\n" + line)[1:, d]
    print(line)
    print("".join("+" if v > 0.5 else "-" if v < -0.5 else "." for v in c))
