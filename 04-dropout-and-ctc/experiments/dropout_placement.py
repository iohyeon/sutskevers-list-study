"""dropout 을 비순환 연결에만 걸 때와 순환 연결에도 걸 때를 비교한다.

네 조건을 같은 말뭉치, 같은 seed 로 학습시킨다. perplexity 말고 긴 거리 기억도 함께 잰다.
여는 태그와 닫는 태그의 이름이 맞는 비율이 그것이다.
"""
import common

HIDDEN = 384
EPOCHS = 50
LOG_AT = {1, 5, 10, 20, 30, 40, 50}
RATE = 0.5

CONDS = [
    ("dropout 없음", dict()),
    ("비순환 연결에만 0.5", dict(feed_rate=RATE)),
    ("순환 연결에만 per-step 0.5", dict(rec_rate=RATE, rec_mode="per_step")),
    ("양쪽 모두 0.5 (순환은 per-step)",
     dict(feed_rate=RATE, rec_rate=RATE, rec_mode="per_step")),
]

if __name__ == "__main__":
    log = common.Log()
    rows = []
    for name, cfg in CONDS:
        log(f"=== {name} ===")
        r = common.train(hidden=HIDDEN, layers=2, epochs=EPOCHS, log=LOG_AT, out=log, **cfg)
        best = min(r["hist"], key=lambda t: t[2])
        s = common.sample(r["model"], r["chars"], r["c2i"], n=3000, temperature=0.5, seed=0)
        ok, tot = common.pair_accuracy(s)
        log(f"  파라미터 {r['params']:,}")
        log(f"  최저 valid {best[2]:.3f} (epoch {best[0]})")
        log(f"  50 epoch: train {r['train_pp']:.3f} valid {r['valid_pp']:.3f} "
            f"test {r['test_pp']:.3f} 차이 {r['valid_pp'] - r['train_pp']:+.3f}")
        log(f"  생성 3,000글자에서 태그 이름이 맞은 쌍 {ok}/{tot}")
        log(f"  생성 앞부분: {s[:100]!r}")
        log(f"  소요 {r['seconds']:.0f}초")
        log()
        rows.append((name, r, best, (ok, tot)))

    log("=== 정리 표 ===")
    log("조건 | 최저 valid (epoch) | 50 epoch train | valid | test | 차이 | 태그 쌍")
    for name, r, best, (ok, tot) in rows:
        log(f"{name} | {best[2]:.3f} ({best[0]}) | {r['train_pp']:.3f} | "
            f"{r['valid_pp']:.3f} | {r['test_pp']:.3f} | "
            f"{r['valid_pp'] - r['train_pp']:+.3f} | {ok}/{tot}")
    log.save("dropout-placement.txt", "dropout_placement.py",
             f"DropLSTM(hidden {HIDDEN}, layer 2), dropout 비율 {RATE}, epoch {EPOCHS}, "
             f"조각 길이 50, 배치 20, Adam lr 3e-3, torch.manual_seed(0), "
             f"생성 temperature 0.5 seed 0 으로 3,000글자")
