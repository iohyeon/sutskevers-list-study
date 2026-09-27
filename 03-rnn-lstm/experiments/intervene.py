"""개입 실험: 본문 중간에서 cell state 의 일부 차원을 반대 환경의 평균값으로 덮어쓰면 닫는 이름이 바뀌는가."""
import torch
from char_lstm import CharModel

ck = torch.load("model_lstm.pt"); chars = ck["chars"]; c2i = {c: i for i, c in enumerate(chars)}; a = ck["args"]
model = CharModel(len(chars), a["hidden"], a["layers"], "lstm", 0.0); model.load_state_dict(ck["model"]); model.eval()

@torch.no_grad()
def feed(text, state=None):
    cs = []
    for ch in text:
        logits, state, _ = model(torch.tensor([[c2i[ch]]]), state)
        cs.append(state[1][0, 0].clone())
    return logits, state, torch.stack(cs)

# 1) probe.py 와 같은 방법으로 환경별 평균과 분리도를 구한다
text = open("corpus.txt").read()[-6000:]; text = text[text.index("\n") + 1:]
_, _, C = feed(text)
label = []
for line in text.split("\n"):
    env = "proof" if line.startswith("\\begin{proof}") else "lemma" if line.startswith("\\begin{lemma}") else None
    for j in range(len(line) + 1):
        label.append(env if 13 <= j < len(line) - 11 else None)
label = label[:len(text)]
P = torch.tensor([l == "proof" for l in label]); L = torch.tensor([l == "lemma" for l in label])
mean = {"proof": C[P].mean(0), "lemma": C[L].mean(0)}
gap = (mean["proof"] - mean["lemma"]).abs() / (C[P].std(0) + C[L].std(0) + 1e-6)
order = gap.argsort(descending=True)

# 2) 본문 중간에서 상위 k 개 차원을 반대 환경의 평균으로 덮어쓰고 이어서 생성한다
@torch.no_grad()
def run(env, k, trials=200, temperature=0.5):
    other = "lemma" if env == "proof" else "proof"
    counts = {}
    for t in range(trials):
        torch.manual_seed(t)
        logits, state, _ = feed(f"\n\\begin{{{env}}} let y so ")
        if k > 0:
            h, Cs = state; Cs = Cs.clone(); dims = order[:k]
            Cs[0, 0, dims] = mean[other][dims]; state = (h, Cs)
        out = ""
        while not out.endswith("\n") and len(out) < 120:
            ix = torch.multinomial(torch.softmax(logits[0, -1] / temperature, dim=0), 1).view(1, 1)
            out += chars[ix.item()]
            logits, state, _ = model(ix, state)
        closed = out[out.index("\\end{") + 5:].split("}")[0] if "\\end{" in out else "?"
        counts[closed] = counts.get(closed, 0) + 1
    return counts

print("분리도 상위 5개 차원:", order[:5].tolist(), [round(gap[i].item(), 2) for i in order[:5]])
for env in ("proof", "lemma"):
    for k in (0, 1, 3, 10, 128):
        print(f"\\begin{{{env}}} 로 시작, 상위 {k:3d}개 차원을 반대 환경 평균으로 덮어씀 ->", run(env, k))
