"""BPTT로 구한 기울기와 수치 미분(중앙 차분)을 비교한다.

char_rnn.py 의 CharRNN 을 그대로 쓴다. 어휘 5, hidden 4, 길이 6인 무작위 입력에서
모든 파라미터 원소의 상대 오차를 구해 가장 큰 값을 출력한다.

사용: python3 gradient_check.py [seed ...]
"""
import sys

from char_rnn import grad_check

if __name__ == "__main__":
    seeds = [int(s) for s in sys.argv[1:]] or [1, 2, 3]
    for seed in seeds:
        print(f"seed {seed}: worst relative error = {grad_check(seed):.3e}")
