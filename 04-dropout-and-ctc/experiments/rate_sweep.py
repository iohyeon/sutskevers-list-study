"""dropout 비율과 모델 크기를 바꿔 가며 정규화가 되고 있는지 잰다.

큰 모델에서 비순환 dropout 비율을 0 부터 0.8 까지 올리고, 같은 조건을 작은 모델에도
적용해 모델을 키웠을 때 방향이 달라지는지 본다.
"""
import common

EPOCHS = 50
LOG_AT = {1, 5, 10, 15, 20, 30, 40, 50}
RUNS = [
    (384, 0.0), (384, 0.25), (384, 0.5), (384, 0.65), (384, 0.8),
    (96, 0.0), (96, 0.5),
]
CURVE_AT = (1, 5, 10, 15, 20, 30, 40, 50)

if __name__ == "__main__":
    log = common.Log()
    rows = []
    for hidden, rate in RUNS:
        log(f"=== hidden {hidden}, 비순환 dropout {rate} ===")
        r = common.train(hidden=hidden, layers=2, feed_rate=rate, epochs=EPOCHS,
                         log=LOG_AT, out=log)
        best = min(r["hist"], key=lambda t: t[2])
        log(f"  파라미터 {r['params']:,}, 최저 valid {best[2]:.3f} (epoch {best[0]}), "
            f"소요 {r['seconds']:.0f}초")
        log()
        rows.append((hidden, rate, r, best))

    log("=== 비율별 정리 (hidden 384) ===")
    log("dropout | 최저 valid (epoch) | 50 epoch train | valid | test | 차이")
    for hidden, rate, r, best in rows:
        if hidden != 384:
            continue
        log(f"{rate:7} | {best[2]:.3f} ({best[0]:2d}) | {r['train_pp']:.3f} | "
            f"{r['valid_pp']:.3f} | {r['test_pp']:.3f} | "
            f"{r['valid_pp'] - r['train_pp']:+.3f}")
    log()

    log("=== 크기와 dropout 의 조합 ===")
    log("hidden | dropout | 파라미터 | 최저 valid (epoch) | 50 epoch train | valid | 차이")
    for hidden, rate, r, best in rows:
        if rate not in (0.0, 0.5):
            continue
        log(f"{hidden:6d} | {rate:7} | {r['params']:8,} | {best[2]:.3f} ({best[0]:2d}) | "
            f"{r['train_pp']:.3f} | {r['valid_pp']:.3f} | "
            f"{r['valid_pp'] - r['train_pp']:+.3f}")
    log()

    log("=== 학습 곡선 (valid perplexity) ===")
    log("설정                     | " + " | ".join(f"ep{e:<3d}" for e in CURVE_AT))
    for hidden, rate, r, best in rows:
        d = {ep: va for ep, tr_, va in r["hist"]}
        log(f"hidden {hidden:3d} dropout {rate:<5} | " +
            " | ".join(f"{d[e]:<5.3f}" for e in CURVE_AT))
    log()
    log("=== 학습 곡선 (train perplexity) ===")
    log("설정                     | " + " | ".join(f"ep{e:<3d}" for e in CURVE_AT))
    for hidden, rate, r, best in rows:
        d = {ep: tr_ for ep, tr_, va in r["hist"]}
        log(f"hidden {hidden:3d} dropout {rate:<5} | " +
            " | ".join(f"{d[e]:<5.3f}" for e in CURVE_AT))

    log.save("rate-sweep.txt", "rate_sweep.py",
             f"DropLSTM(layer 2), 비순환 연결에만 dropout, hidden 과 비율 조합 {RUNS}, "
             f"epoch {EPOCHS}, 조각 길이 50, 배치 20, Adam lr 3e-3, torch.manual_seed(0)")
