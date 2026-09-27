"""모델을 키우면 학습 perplexity 와 검증 perplexity 가 어떻게 벌어지는지 잰다.

정규화를 아무것도 걸지 않고 hidden 만 96, 192, 384 로 바꿔 같은 말뭉치에 학습시킨다.
"""
import re

import common

SIZES = (96, 192, 384)
EPOCHS = 50
LOG_AT = {1, 3, 5, 10, 15, 20, 30, 40, 50}


def tag_distance(text):
    """여는 태그의 끝과 닫는 태그의 시작 사이의 거리. 기억해야 하는 길이."""
    d = [m.end(2) - m.start(2) for m in
         re.finditer(r"(\\begin\{[a-z]*\})(.*?)\\end\{[a-z]*\}", text)]
    return min(d), sum(d) / len(d), max(d), len(d)


if __name__ == "__main__":
    log = common.Log()
    text, chars, c2i, tr, va, te = common.load()
    lo, avg, hi, n = tag_distance(text)
    log(f"말뭉치의 태그 쌍 {n}개. 여는 태그와 닫는 태그 사이의 거리 최소 {lo}, "
        f"평균 {avg:.1f}, 최대 {hi}글자")
    log(f"조각 길이 50 이므로 평균 거리는 한 조각 안에 들어간다.")
    log()

    rows = []
    for hidden in SIZES:
        log(f"=== hidden {hidden}, layer 2, dropout 없음 ===")
        r = common.train(hidden=hidden, layers=2, epochs=EPOCHS, log=LOG_AT, out=log)
        best = min(r["hist"], key=lambda t: t[2])
        log(f"  파라미터 {r['params']:,}, 학습 글자당 {r['params'] / r['n_train']:.1f}개")
        log(f"  검증 perplexity 가 가장 낮았던 곳: epoch {best[0]}, "
            f"train {best[1]:.3f} valid {best[2]:.3f}")
        log(f"  마지막 epoch: train {r['train_pp']:.3f} valid {r['valid_pp']:.3f} "
            f"test {r['test_pp']:.3f}, {r['seconds']:.0f}초")
        log()
        rows.append((hidden, r, best))

    log("=== 정리 표 ===")
    log("hidden | 파라미터 | 글자당 | 최저 valid (epoch) | 50 epoch train | "
        "50 epoch valid | 차이")
    for hidden, r, best in rows:
        log(f"{hidden:6d} | {r['params']:8,} | {r['params'] / r['n_train']:6.1f} | "
            f"{best[2]:6.3f} ({best[0]:2d}) | {r['train_pp']:14.3f} | "
            f"{r['valid_pp']:14.3f} | {r['valid_pp'] - r['train_pp']:+.3f}")
    log.save("overfit.txt", "overfit.py",
             f"DropLSTM(layer 2, dropout 없음), hidden {SIZES}, epoch {EPOCHS}, "
             f"조각 길이 50, 배치 20, Adam lr 3e-3, torch.manual_seed(0), "
             f"perplexity 는 dropout 을 끈 상태로 split 전체를 훑어 계산")
