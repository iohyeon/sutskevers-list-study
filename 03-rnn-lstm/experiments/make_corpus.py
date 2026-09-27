"""장기 의존성을 시험하는 합성 말뭉치. 환경을 열고, 본문을 쓰고, 같은 이름으로 닫는다."""
import random
random.seed(0)
ENVS = ["proof", "lemma"]
WORDS = ["let", "x", "be", "a", "set", "then", "we", "have", "so", "and", "y", "holds", "by", "the", "map"]

def block():
    env = random.choice(ENVS)
    body = " ".join(random.choice(WORDS) for _ in range(random.randint(4, 9)))
    return f"\\begin{{{env}}} {body} \\end{{{env}}}\n"

if __name__ == "__main__":
    text = "".join(block() for _ in range(3000))
    open("corpus.txt", "w").write(text)
    print(len(text), "chars")
    print(text[:200])
