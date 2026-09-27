"""학습된 LSTM 의 forget gate 값을 본문 구간에서 재서, 환경 이름을 담은 차원의 f 평균을 찍는다."""
import torch
from char_lstm import CharModel
ck = torch.load("model_lstm.pt"); chars = ck["chars"]; c2i = {c: i for i, c in enumerate(chars)}; a = ck["args"]
model = CharModel(len(chars), a["hidden"], a["layers"], "lstm", 0.0); model.load_state_dict(ck["model"]); model.eval()
H = a["hidden"]; W_ih, W_hh, b_ih, b_hh = (getattr(model.rnn, n + "_l0") for n in ("weight_ih", "weight_hh", "bias_ih", "bias_hh"))
text = open("corpus.txt").read()[-6000:]; text = text[text.index("\n") + 1:]
label = []
for line in text.split("\n"):
    start = len("\\begin{proof}")
    for j in range(len(line) + 1):
        label.append(start <= j < len(line) - len("\\end{proof}"))
label = torch.tensor(label[:len(text)])
fs = []; h = torch.zeros(H); c = torch.zeros(H)
with torch.no_grad():
    for ch in text:
        x = model.embed(torch.tensor(c2i[ch]))
        z = W_ih @ x + b_ih + W_hh @ h + b_hh
        i, f, g, o = torch.sigmoid(z[:H]), torch.sigmoid(z[H:2*H]), torch.tanh(z[2*H:3*H]), torch.sigmoid(z[3*H:])
        c = f * c + i * g; h = o * torch.tanh(c); fs.append(f)
F = torch.stack(fs)[label]
for d in (69, 88, 61):
    print(f"차원 {d}: 본문 구간 forget gate 평균 {F[:, d].mean():.3f}, 최소 {F[:, d].min():.3f}")
print(f"전체 128개 차원의 본문 구간 평균 {F.mean():.3f}")
print(f"29시점 직접 경로: 0.999^29 = {0.999**29:.3f}, 0.870^29 = {0.870**29:.4f}")
