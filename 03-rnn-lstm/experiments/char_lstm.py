"""PyTorch 글자 단위 생성 모델. --cell rnn 과 --cell lstm 을 같은 조건에서 비교한다."""
import argparse, math, time
import torch, torch.nn as nn
from ngram import closer_accuracy

class CharModel(nn.Module):
    def __init__(self, vocab, hidden, layers, cell, dropout):
        super().__init__()
        self.embed = nn.Embedding(vocab, hidden)
        rnn_cls = nn.LSTM if cell == "lstm" else nn.RNN
        self.rnn = rnn_cls(hidden, hidden, num_layers=layers, batch_first=True,
                           dropout=dropout if layers > 1 else 0.0)
        self.out = nn.Linear(hidden, vocab)

    def forward(self, x, state=None):
        h, state = self.rnn(self.embed(x), state)     # h: (batch, time, hidden)
        return self.out(h), state, h

def batches(data, batch, seq):
    n = (len(data) - 1) // (batch * seq)
    x = data[:n * batch * seq].view(batch, -1)
    y = data[1:n * batch * seq + 1].view(batch, -1)
    for i in range(0, x.size(1), seq):
        yield x[:, i:i + seq], y[:, i:i + seq]

def detach(state):
    if state is None: return None
    return tuple(s.detach() for s in state) if isinstance(state, tuple) else state.detach()

@torch.no_grad()
def sample(model, chars, c2i, n, temperature, seed=0):
    torch.manual_seed(seed)
    model.eval()
    ix = torch.tensor([[c2i["\n"]]]); state = None; out = []
    for _ in range(n):
        logits, state, _ = model(ix, state)
        prob = torch.softmax(logits[0, -1] / temperature, dim=0)
        ix = torch.multinomial(prob, 1).view(1, 1)
        out.append(chars[ix.item()])
    return "".join(out)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", default="lstm"); ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=1); ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--epochs", type=int, default=6); ap.add_argument("--seq", type=int, default=64)
    ap.add_argument("--batch", type=int, default=32); ap.add_argument("--corpus", default="corpus.txt")
    ap.add_argument("--forget-bias", type=float, default=None, help="LSTM forget gate bias 초기값 (두 bias 의 합)")
    a = ap.parse_args()
    torch.manual_seed(0)
    text = open(a.corpus).read()
    chars = sorted(set(text)); c2i = {c: i for i, c in enumerate(chars)}
    data = torch.tensor([c2i[c] for c in text])
    split = int(len(data) * 0.9); train, test = data[:split], data[split:]
    model = CharModel(len(chars), a.hidden, a.layers, a.cell, a.dropout)
    if a.cell == "lstm" and a.forget_bias is not None:
        H = a.hidden                                  # PyTorch 의 게이트 순서는 input, forget, cell, output
        for name in ("bias_ih_l0", "bias_hh_l0"):
            getattr(model.rnn, name).data[H:2 * H].fill_(a.forget_bias / 2)
    print("params", sum(p.numel() for p in model.parameters()))
    opt = torch.optim.Adam(model.parameters(), lr=3e-3); lossf = nn.CrossEntropyLoss()
    t0 = time.time()
    for ep in range(a.epochs):
        model.train(); state = None
        for x, y in batches(train, a.batch, a.seq):
            logits, state, _ = model(x, detach(state))          # detach: truncated BPTT
            loss = lossf(logits.reshape(-1, len(chars)), y.reshape(-1))
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # gradient clipping
            opt.step()
        model.eval(); tot = 0.0; n = 0; state = None
        with torch.no_grad():
            for x, y in batches(test, 1, a.seq):
                logits, state, _ = model(x, state)
                tot += lossf(logits.reshape(-1, len(chars)), y.reshape(-1)).item() * y.numel(); n += y.numel()
        print(f"epoch {ep+1} train loss {loss.item():.3f} test PP {math.exp(tot/n):.3f} {time.time()-t0:.0f}s", flush=True)
    for T in (0.2, 0.5, 1.0, 1.5):
        s = sample(model, chars, c2i, 6000, T)
        print(f"T={T} closer ok/bad={closer_accuracy(s)}  sample: {s[:110]!r}")
    torch.save({"model": model.state_dict(), "chars": chars, "args": vars(a)}, f"model_{a.cell}.pt")
