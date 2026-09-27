"""파라미터 수와 학습 토큰 수를 센다.

앞부분은 Zaremba 외(2014)가 PTB 에서 쓴 설정을 그대로 넣어 layer 별로 계산한 것이고,
뒷부분은 이 폴더의 합성 말뭉치와 실험 모델에 같은 계산을 한 것이다.
"""
import common

# PTB word-level 설정. 어휘 10,000, embedding 크기는 hidden 과 같다.
PTB_VOCAB = 10_000
PTB_TRAIN_TOKENS = 929_589          # Mikolov 전처리판 학습 split 의 토큰 수
PTB_VALID_TOKENS = 73_760
PTB_TEST_TOKENS = 82_430


def lstm_layer_params(n_in, n_hidden):
    """게이트 4개짜리 LSTM layer 하나의 파라미터 수. 입력 행렬, 순환 행렬, bias."""
    wx = 4 * n_hidden * n_in
    wh = 4 * n_hidden * n_hidden
    b = 4 * n_hidden
    return wx, wh, b


def model_params(vocab, hidden, layers):
    embed = vocab * hidden
    per = []
    for k in range(layers):
        per.append(lstm_layer_params(hidden, hidden))
    out = hidden * vocab + vocab
    total = embed + sum(sum(t) for t in per) + out
    return embed, per, out, total


def report(log, name, vocab, hidden, layers, tokens):
    embed, per, out, total = model_params(vocab, hidden, layers)
    log(f"[{name}] 어휘 {vocab:,}, hidden {hidden}, layer {layers}")
    log(f"  embedding        {vocab:,} x {hidden} = {embed:>12,}")
    for k, (wx, wh, b) in enumerate(per, 1):
        log(f"  LSTM layer {k}     W_x 4*{hidden}*{hidden} = {wx:>10,}  "
            f"W_h 4*{hidden}*{hidden} = {wh:>10,}  bias {b:,}")
    log(f"  출력 Linear      {hidden} x {vocab:,} + {vocab:,} = {out:>12,}")
    log(f"  합계             {total:>12,}")
    log(f"  학습 토큰 수     {tokens:>12,}")
    log(f"  토큰 하나당 파라미터 {total / tokens:8.1f}개")
    log()
    return total


if __name__ == "__main__":
    log = common.Log()
    log("=== 논문이 PTB 에서 쓴 설정 ===")
    log(f"PTB 토큰 수: 학습 {PTB_TRAIN_TOKENS:,} / 검증 {PTB_VALID_TOKENS:,} / "
        f"시험 {PTB_TEST_TOKENS:,}")
    log()
    report(log, "medium 2 layer 650", PTB_VOCAB, 650, 2, PTB_TRAIN_TOKENS)
    report(log, "large 2 layer 1500", PTB_VOCAB, 1500, 2, PTB_TRAIN_TOKENS)

    log("=== 이 폴더의 합성 말뭉치와 모델 ===")
    text, chars, c2i, tr, va, te = common.load()
    log(f"말뭉치 {len(text):,}글자, 어휘 {len(chars)}글자")
    log(f"학습 {len(tr):,} / 검증 {len(va):,} / 시험 {len(te):,}글자")
    for hidden in (96, 192, 384):
        total = report(log, f"hidden {hidden} 2 layer", len(chars), hidden, 2, len(tr))
        model = common.DropLSTM(len(chars), hidden, 2)
        log(f"  PyTorch 모듈이 실제로 만든 파라미터 수 {common.count_params(model):,}")
        log()

    log("=== 배치 하나가 보는 글자 수 ===")
    log("배치 20, 조각 길이 50 이면 갱신 한 번에 1,000글자를 본다.")
    log(f"학습 {len(tr):,}글자는 조각 {len(tr) // (20 * 50)}개, 즉 epoch 당 갱신 "
        f"{len(tr) // (20 * 50)}번이다.")
    log.save("param-count.txt", "param_count.py",
             "계산만 한다. PTB 설정은 논문 표 1 의 medium, large 구성, "
             "합성 말뭉치 모델은 hidden 96/192/384, layer 2")
