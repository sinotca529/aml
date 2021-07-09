import numpy as np

class DataSet():
    # x: 各行が1サンプルを表す行列
    # y: 各サンプルのラベルを表す行列(縦ベクトル)
    def __init__(self, x, y) -> None:
        (qty_sample, dim) = x.shape
        assert(qty_sample == y.shape[0])
        assert(y.shape[1] == 1)
        self.dim = dim
        self.qty_sample = qty_sample
        self.x = x
        self.y = y

def gen_dataset_iv(qty_sample = 200, dim = 4) -> DataSet:
    x = 3 * (np.random.rand(qty_sample, dim) - 0.5)
    y = (2 * x[:, 1:2] - 1 * x[:, 2:3] + 0.5 + 0.5 * np.random.rand(qty_sample, 1)) > 0
    y = 2 * y - 1

    x[:, 0] = 1
    assert(len(x) == qty_sample)
    assert(len(x) == len(y))
    return DataSet(x, y)

def gen_dataset_v(qty_sample = 200, dim = 4) -> DataSet:
    x = 3 * (np.random.rand(qty_sample, dim) - 0.5)
    W = np.array([
        [2, -1, 0.5],
        [-3, 2, 1],
        [1, 2, 3]
    ])

    (maxlogit, y) = max()

def test_data() -> DataSet:
    x = np.array([
        [1, 2, 3],
        [4, 5, 6]
    ])
    y = np.array([
        [1],
        [-1]
    ])
    return DataSet(x, y)