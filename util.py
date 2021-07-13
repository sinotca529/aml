import numpy as np
from numpy import ndarray

# xの各要素aをlog(1 + e^a)にする。
def log1p_exp(x: ndarray) -> ndarray:
    def log1p_exp_1d(a: float) -> float:
        # オーバーフローしない防止
        if a <= 0:
            return np.log1p(np.exp(a))
        else:
            return a + np.log1p(np.exp(-a))

    return np.vectorize(log1p_exp_1d)(x)

# matの各要素aを1/(1 + e^-a)にする。
def sigmoid(mat: ndarray) -> ndarray:
    def sigmoid_for1d(a: float) -> float:
        # オーバーフロー防止
        if a >= 0:
            return 1.0 / (1.0 + np.exp(-a))
        else:
            m = np.exp(a)
            return m / (1.0 + m)

    return np.vectorize(sigmoid_for1d)(mat)

# ベクトルを受け取り、各要素をexpしたものの和の、logをとる。
# Overflowが起きるなら、式変換した(まともな)実装に直す必要あり。
def log_sum_exp(x: ndarray) -> float:
    assert((x.shape == (x.shape[0],)) or (x.shape[1] == 1))
    lse = np.log(np.sum(np.exp(x)))
    return lse
