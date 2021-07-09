from abc import abstractmethod
import numpy as np
from numpy import ndarray

from dataset import *

EPS = 10e-7

# log(1 + e^x) を 計算する
# xがでかい場合でもオーバーフローしない
def log1p_exp(x: ndarray) -> ndarray:
    def for1d(a):
        if a <= 0:
            return np.log1p(np.exp(a))
        else:
            return a + np.log1p(np.exp(-a))

    return np.array([*map(lambda a: for1d(a), x)])


class ObjFunc():
    # データセットを用いたパラメタの設定などが必要な場合はこれを用いる
    @abstractmethod
    def setup_with_data_set(self, data_set: DataSet) -> None:
        pass

    @abstractmethod
    def apply(self, data_set: DataSet, w: ndarray) -> float:
        pass

    @abstractmethod
    def apply_grad(self, data_set: DataSet, w: ndarray) -> ndarray:
        pass

    @abstractmethod
    def apply_hessian(self, data_set: DataSet, w: ndarray) -> ndarray:
        pass

    # 損失関数の勾配のLipsitz連続性の係数
    @abstractmethod
    def get_grad_lipsitz(self, data_set: DataSet) -> float:
        pass

class LinearLogistic(ObjFunc):
    def __init__(self) -> None:
        super().__init__()

    # i番目の要素が p_i = p(y_i | x_i, w) となる列ベクトルを返す
    # p_i = 1/(1 + exp(-y w.T x))
    def __calc_posterior(self, data_set: DataSet, w: ndarray) -> ndarray:
        # overflow対策
        def for1d(a):
            if a <= 0:
                return 1.0 / (1.0 + np.exp(a))
            else:
                m = np.exp(-a)
                return m / (1.0 + m)

        assert(data_set.dim == w.shape[0])
        (x, y) = (data_set.x, data_set.y)
        # posterior = 1.0 / (1.0 + np.exp(-y * x.dot(w)))
        posterior = np.array([*map(lambda a: for1d(a), -y * x.dot(w))])
        assert(posterior.shape == (data_set.qty_sample, 1))
        return posterior

    def setup_with_data_set(self, data_set: DataSet) -> float:
        self.__lambda = 0.01 * data_set.qty_sample

    def apply(self, data_set: DataSet, w: ndarray) -> float:
        assert(data_set.dim == w.shape[0])
        (x, y) = (data_set.x, data_set.y)

        loss_without_reg = np.sum(log1p_exp(-y * x.dot(w)), axis=0)
        reg = self.__lambda * (w.T.dot(w))
        loss = loss_without_reg + reg
        assert(loss.shape == (1, 1))
        return loss

    def apply_grad(self, data_set: DataSet, w: ndarray) -> ndarray:
        assert(data_set.dim == w.shape[0])
        (x, y) = (data_set.x, data_set.y)

        posterior = self.__calc_posterior(data_set, w)
        grad_without_reg = np.sum(-y * x * (1.0 - posterior), axis=0)
        grad = grad_without_reg.reshape(w.shape) + 2 * self.__lambda * w
        assert(grad.shape == w.shape)
        return grad

    def apply_hessian(self, data_set: DataSet, w: ndarray) -> ndarray:
        assert(data_set.dim == w.shape[0])
        x = data_set.x

        p = self.__calc_posterior(data_set, w)
        acc = np.zeros((data_set.dim, data_set.dim))
        for i in range(0, data_set.qty_sample):
            xi = x[i,:].reshape((1, data_set.dim))
            acc += p[i] * (1 - p[i]) * xi.T.dot(xi)

        acc += 2 * self.__lambda

        return acc

    def get_grad_lipsitz(self, data_set: DataSet) -> float:
        x = data_set.x
        acc = 0.0
        for i in range(0, data_set.qty_sample):
            xi = x[i,:].reshape((1, data_set.dim))
            acc += xi.dot(xi.T)[0,0]

        lip = acc / 4.0 + 2 * self.__lambda
        # print("lip : ", str(lip))
        return lip

    # def test_apply(self):
    #     dset = test_data()
    #     w = np.array([
    #         [7],
    #         [8],
    #         [9]
    #     ])

    #     p1 = 1/(1 + np.exp(-50))
    #     p2 = 1/(1 + np.exp(122))
    #     print("priterior: (" + str(p1) + ", " + str(p2) + ")")
    #     print(self.__calc_posterior(dset, w))

    #     print("apply:")
    #     print(np.log1p(np.exp(-50)) + np.log1p(np.exp(122)) + self.__lambda * w.T.dot(w))
    #     print(self.apply(dset, w))
