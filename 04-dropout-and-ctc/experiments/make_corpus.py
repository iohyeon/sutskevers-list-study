"""과적합을 보기 위한 작은 합성 말뭉치를 만든다.

환경 이름으로 열고, 무작위로 뽑은 본문을 쓰고, 같은 이름으로 닫는다. 본문이 무작위라
외울 수는 있어도 일반화할 수는 없고, 여는 이름과 닫는 이름 사이에는 수십 글자 떨어진
의존이 생긴다. 말뭉치를 작게 잡아 파라미터 수가 학습 글자 수를 크게 넘도록 했다.
"""
import random

SEED = 7
ENVS = ["proof", "lemma", "claim", "remark", "theorem", "example"]
WORDS = ["let", "x", "y", "be", "a", "set", "then", "we", "have", "so",
         "and", "holds", "by", "the", "map", "if", "for", "all", "some", "thus"]
BLOCKS = 400


def block(rng):
    env = rng.choice(ENVS)
    body = " ".join(rng.choice(WORDS) for _ in range(rng.randint(5, 12)))
    return f"\\begin{{{env}}} {body} \\end{{{env}}}\n"


def build(blocks=BLOCKS, seed=SEED):
    rng = random.Random(seed)
    return "".join(block(rng) for _ in range(blocks))


if __name__ == "__main__":
    text = build()
    with open("corpus.txt", "w") as f:
        f.write(text)
    chars = sorted(set(text))
    print(f"seed {SEED}, blocks {BLOCKS}")
    print(f"{len(text)} chars, vocab {len(chars)}: {''.join(chars)!r}")
    print(f"env names {len(ENVS)}, body words {len(WORDS)}")
    print(text[:180])
