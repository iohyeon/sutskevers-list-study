"""마스크를 시퀀스 안에서 고정하면 순환 연결에도 dropout 을 걸 수 있는지 본다.

per-step 마스크와 per-sequence 마스크, 그리고 순환 가중치에 직접 거는 DropConnect
(AWD-LSTM 의 weight drop)를 같은 조건에서 비교한다. 앞에 per-sequence 마스크가
가중치 행렬의 행을 지우는 것과 같다는 것을 수치로 확인하는 절이 있다.
"""
import torch

import common

HIDDEN = 384
EPOCHS = 50
LOG_AT = {1, 5, 10, 20, 30, 40, 50}
RATE = 0.5

CONDS = [
    ("비순환 0.5 (기준)", dict(feed_rate=RATE)),
    ("비순환 0.5 + 순환 per-step 0.5",
     dict(feed_rate=RATE, rec_rate=RATE, rec_mode="per_step")),
    ("비순환 0.5 + 순환 per-seq 0.5",
     dict(feed_rate=RATE, rec_rate=RATE, rec_mode="per_seq")),
    ("비순환 per-seq 0.5 + 순환 per-seq 0.5",
     dict(feed_rate=RATE, feed_mode="per_seq", rec_rate=RATE, rec_mode="per_seq")),
    ("비순환 0.5 + W_hh weight drop 0.5",
     dict(feed_rate=RATE, weight_rate=RATE)),
]


def row_drop_identity(log):
    """(h ⊙ m) U 가 h 에 행이 지워진 U 를 곱한 것과 같은지 확인한다."""
    log("=== per-sequence 마스크는 순환 가중치의 행을 지우는 것과 같다 ===")
    torch.manual_seed(0)
    H = 6
    h = torch.randn(1, H)
    U = torch.randn(H, 4 * H)     # 행 벡터 규약: 순환 항이 h U 다
    m = torch.bernoulli(torch.full((1, H), 0.5))
    a = (h * m) @ U
    b = h @ (m.view(H, 1) * U)
    zero = [i for i, v in enumerate(m.view(-1).tolist()) if v == 0.0]
    log(f"H = {H}, m = {[int(v) for v in m.view(-1).tolist()]}, 0 이 된 자리 {zero}")
    log(f"(h ⊙ m) U 와 h (diag(m) U) 의 최대 절대차 {(a - b).abs().max().item():.3e}")
    log("두 값이 같으므로, h 의 j 번째를 떨어뜨리는 것은 U 의 j 번째 행을 지우는 것이다.")
    log("마스크가 시점마다 바뀌면 지워지는 행이 시점마다 바뀌고, 시퀀스 안에서 고정하면")
    log("그 시퀀스 동안은 행이 지워진 같은 행렬을 쓰는 것이 된다.")
    log()


if __name__ == "__main__":
    log = common.Log()
    row_drop_identity(log)
    rows = []
    for name, cfg in CONDS:
        log(f"=== {name} ===")
        r = common.train(hidden=HIDDEN, layers=2, epochs=EPOCHS, log=LOG_AT, out=log, **cfg)
        best = min(r["hist"], key=lambda t: t[2])
        s = common.sample(r["model"], r["chars"], r["c2i"], n=3000, temperature=0.5, seed=0)
        ok, tot = common.pair_accuracy(s)
        log(f"  최저 valid {best[2]:.3f} (epoch {best[0]})")
        log(f"  50 epoch: train {r['train_pp']:.3f} valid {r['valid_pp']:.3f} "
            f"test {r['test_pp']:.3f} 차이 {r['valid_pp'] - r['train_pp']:+.3f}")
        log(f"  생성 3,000글자에서 태그 이름이 맞은 쌍 {ok}/{tot}")
        log(f"  소요 {r['seconds']:.0f}초")
        log()
        rows.append((name, r, best, (ok, tot)))

    log("=== 정리 표 ===")
    log("조건 | 최저 valid (epoch) | 50 epoch train | valid | test | 차이 | 태그 쌍")
    for name, r, best, (ok, tot) in rows:
        log(f"{name} | {best[2]:.3f} ({best[0]}) | {r['train_pp']:.3f} | "
            f"{r['valid_pp']:.3f} | {r['test_pp']:.3f} | "
            f"{r['valid_pp'] - r['train_pp']:+.3f} | {ok}/{tot}")
    log.save("variational-dropout.txt", "variational_dropout.py",
             f"DropLSTM(hidden {HIDDEN}, layer 2), dropout 비율 {RATE}, epoch {EPOCHS}, "
             f"조각 길이 50, 배치 20, Adam lr 3e-3, torch.manual_seed(0), "
             f"weight drop 은 배치마다 W_hh 를 다시 뽑는다, 생성 temperature 0.5 seed 0")
