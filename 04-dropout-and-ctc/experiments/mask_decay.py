"""dropout 마스크를 순환 연결에 걸면 시점 수만큼 곱해진다는 것을 센다.

앞부분은 유지 확률의 거듭제곱을 그대로 계산하고 표본으로 확인한다. 뒷부분은 학습된
작은 모델에 같은 입력을 넣고, 마스크를 어디에 걸었을 때 마지막 hidden state 가
dropout 없는 값에서 얼마나 멀어지는지 코사인 유사도로 잰다.
"""
import torch

import common

KEEPS = (0.5, 0.65, 0.8, 0.9, 0.95)
STEPS = (1, 5, 10, 25, 50, 100)
TRIALS = 200_000
LAYERS = 2
DRAWS = 50
PROBE_EPOCHS = 50
PROBE_HIDDEN = 384
CTX = 80
N_PROBE = 100


def survival_table(log):
    log("=== 유지 확률 p 인 마스크를 T 시점 곱했을 때 한 경로가 살아남을 확률 p^T ===")
    log("p     | " + " | ".join(f"T={t:<10d}" for t in STEPS))
    for p in KEEPS:
        cells = []
        for t in STEPS:
            v = p ** t
            cells.append(f"{v:<12.3e}" if v < 1e-3 else f"{v:<12.6f}")
        log(f"{p:<5} | " + " | ".join(cells))
    log()


def monte_carlo(log):
    log(f"=== 표본 {TRIALS:,}개로 확인 (torch.manual_seed(0)) ===")
    torch.manual_seed(0)
    log("p     T    이론 p^T      표본 비율")
    for p in (0.5, 0.65, 0.8):
        for t in (10, 25, 50):
            m = torch.bernoulli(torch.full((TRIALS, t), p))
            hit = (m.sum(1) == t).float().mean().item()
            log(f"{p:<5} {t:<4d} {p ** t:<13.3e} {hit:.6f}")
    log()


def mask_count(log):
    log("=== 한 경로가 만나는 마스크의 수 ===")
    log(f"layer {LAYERS} 개짜리 모델에서 시점 t 의 입력이 시점 t+k 의 출력까지 가는 경로다.")
    log("k     | 비순환 연결에만 걸 때 | 순환 연결에도 걸 때 (layer 하나 기준)")
    for k in STEPS:
        log(f"{k:<5d} | {LAYERS + 1:<21d} | {k}")
    log("비순환 연결에만 걸면 마스크 수가 layer 수 + 1 로 고정되고 k 와 무관하다.")
    log()


# 길이가 같은 환경 이름끼리 바꾼다. 문맥의 글자 수가 그대로여야 비교가 된다.
SWAP = {"proof": "lemma", "lemma": "claim", "claim": "proof",
        "theorem": "example", "example": "theorem"}


def probe_sites(txt, c2i, limit):
    """닫는 태그 자리마다 (여는 이름을 바꾼 문맥, 바뀐 이름의 첫 글자)를 만든다."""
    ctx, tgt, plain, ptgt, pos = [], [], [], [], 0
    while len(ctx) < limit:
        pos = txt.find("\\end{", pos + 1)
        if pos < 0:
            break
        i = pos + 5                       # \end{ 다음 글자가 환경 이름의 첫 글자다
        if i < CTX:
            continue
        c = txt[i - CTX:i]
        j = c.rfind("\\begin{")
        if j < 0:
            continue
        k = c.index("}", j)
        name = c[j + 7:k]
        if name not in SWAP:
            continue
        new = SWAP[name]
        swapped = c[:j + 7] + new + c[k:]
        plain.append([c2i[ch] for ch in c])
        ptgt.append(c2i[txt[i]])
        ctx.append([c2i[ch] for ch in swapped])
        tgt.append(c2i[new[0]])
    return (torch.tensor(ctx), torch.tensor(tgt),
            torch.tensor(plain), torch.tensor(ptgt))


def closer_probe(log):
    """마스크를 어디에 걸면 여는 태그의 이름이 닫는 자리까지 남아 있는지 잰다."""
    log("=== 여는 태그의 이름이 닫는 자리까지 남아 있는지 ===")
    r = common.train(hidden=PROBE_HIDDEN, layers=2, epochs=PROBE_EPOCHS,
                     log={PROBE_EPOCHS}, out=log)
    model, chars, c2i = r["model"], r["chars"], r["c2i"]
    log(f"  모델은 hidden {PROBE_HIDDEN}, layer 2, dropout 없이 {PROBE_EPOCHS} epoch "
        "학습시킨 것을 그대로 쓴다.")

    text, _, _, tr, va, te = common.load()
    unseen = text[len(tr):]
    x, y, px, py = probe_sites(unseen, c2i, N_PROBE)
    log(f"  문맥은 학습에 쓰지 않은 {len(unseen):,}글자에서 \\end{{ 앞 {CTX}글자를 잘라 "
        f"쓴다. 자리 {len(y)}개")
    log("  그 문맥 안의 여는 태그 이름을 길이가 같은 다른 이름으로 바꿨다. 바꾼 문맥은 "
        "학습에 없으므로")
    log("  닫는 이름을 맞히려면 문맥에 있는 여는 이름을 읽는 수밖에 없다.")
    log(f"  맞혀야 하는 글자는 환경 이름의 첫 글자 {sorted(set(chars[k] for k in y.tolist()))}, "
        f"고르기만 하면 확률 {1 / len(set(chars[k] for k in y.tolist())):.3f}")
    log()

    def measure(inp, target, cfg, draws):
        for k, v in cfg.items():
            setattr(model, k, v)
        ps, accs = [], []
        with torch.no_grad():
            for d in range(draws):
                torch.manual_seed(2000 + d)
                logits, _ = model(inp)
                prob = torch.softmax(logits[:, -1], dim=1)
                ps.append(prob.gather(1, target.view(-1, 1)).mean().item())
                accs.append((prob.argmax(1) == target).float().mean().item())
        return sum(ps) / len(ps), sum(accs) / len(accs)

    off = dict(feed_rate=0.0, rec_rate=0.0, rec_mode="none", weight_rate=0.0)
    model.eval()
    pr, ac = measure(px, py, off, 1)
    log(f"이름을 바꾸지 않은 문맥, dropout 끈 상태: 정답 글자 확률 {pr:.3f}, "
        f"1위로 맞힌 비율 {ac:.3f}")
    log()
    log("이름을 바꾼 문맥")
    log("마스크 위치            | 정답 글자 확률 | 1위로 맞힌 비율")
    pr, ac = measure(x, y, off, 1)
    log(f"{'dropout 끈 상태 (eval)':<22} | {pr:14.3f} | {ac:15.3f}")
    model.train()
    conds = [
        ("비순환 연결만 0.5", dict(feed_rate=0.5)),
        ("순환 per-step 0.25", dict(rec_rate=0.25, rec_mode="per_step")),
        ("순환 per-step 0.5", dict(rec_rate=0.5, rec_mode="per_step")),
        ("순환 per-seq 0.5", dict(rec_rate=0.5, rec_mode="per_seq")),
        ("W_hh weight drop 0.5", dict(weight_rate=0.5)),
    ]
    for name, extra in conds:
        cfg = dict(off)
        cfg.update(extra)
        pr, ac = measure(x, y, cfg, DRAWS)
        log(f"{name:<22} | {pr:14.3f} | {ac:15.3f}")
    log()
    log(f"마스크 표본 {DRAWS}개의 평균이다. 학습이 끝난 모델에 마스크만 다시 켜서 쟀고 "
        "가중치는 고치지 않았다.")


if __name__ == "__main__":
    log = common.Log()
    survival_table(log)
    monte_carlo(log)
    mask_count(log)
    closer_probe(log)
    log.save("mask-decay.txt", "mask_decay.py",
             f"앞 세 절은 계산과 표본 추출({TRIALS:,}개, torch.manual_seed(0)). "
             f"마지막 절은 hidden {PROBE_HIDDEN} layer 2 모델을 dropout 없이 "
             f"{PROBE_EPOCHS} epoch 학습시킨 뒤, 학습에 쓰지 않은 split 의 닫는 태그 "
             f"자리에서 앞 {CTX}글자를 문맥으로 넣고 마스크 표본 {DRAWS}개로 잰 "
             f"정답 글자 확률")
