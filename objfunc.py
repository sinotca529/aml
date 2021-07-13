from abc import abstractmethod
import collections
import numpy as np
from numpy import ndarray

from dataset import *
from util import *

EPS = 10e-7

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
        assert(data_set.dim == w.shape[0])
        (x, y) = (data_set.x, data_set.y)
        posterior = sigmoid(y * x.dot(w))
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
        return loss[0, 0]

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
        lip = np.trace(x.dot(x.T)) / 4.0 + 2.0 * self.__lambda
        return lip

# 重みwは、各行が各カテゴリの重みに対応。
# xは、各行が各サンプルに対応
# yは、各行が各サンプルのカテゴリに対応
class MulticlassLogisticRegression(ObjFunc):
    def setup_with_data_set(self, data_set: DataSet) -> float:
        self.__lambda = 0.01 * data_set.qty_sample

    def apply(self, data_set: DataSet, w: ndarray) -> float:
        (x, y) = (data_set.x, data_set.y)
        dim = data_set.dim
        qty_c = data_set.qty_category
        qty_sample = data_set.qty_sample
        assert(w.shape == (qty_c, dim))

        # lse[i] = ln(Σ[c=0..C] exp(<w_c, x_i>))
        lse = np.apply_along_axis(log_sum_exp, 1, x.dot(w.T)).reshape((qty_sample, 1))

        # slse = Σ[i=0..n] ln(Σ[c=0..C] exp(<w_c, x_i>))
        slse = np.sum(lse)

        # Σ[i=0..n] -<w_y(i) x_i>
        # を求める
        acc = 0.0
        for i in range(0, qty_sample):
            wyi = w[y[i]]
            xi = x[i]
            acc += -wyi.dot(xi)
        
        reg = np.linalg.norm(w, ord=2)
        loss = slse + acc + reg
        return loss

    # retval[r, i] = p(r | x_i)
    def __calc_posterior(self, data_set: DataSet, w: ndarray) -> ndarray:
        (x, y) = (data_set.x, data_set.y)
        dim = data_set.dim
        qty_c = data_set.qty_category
        qty_sample = data_set.qty_sample
        assert(w.shape == (qty_c, dim))

        # 分子: numerator[r, i] = exp(<w_r, x_i>)
        numerator = np.exp(w.dot(x.T))
        assert(numerator.shape == (qty_c, qty_sample))

        # 分母: denominator[i] = Σ_c exp(<w_c, x_i>)
        denominator = np.sum(numerator, axis=0)

        posterior =  numerator/denominator
        assert(posterior.shape == (qty_c, qty_sample))
        return posterior

    # 各行が各カテゴリの重みの勾配に相当
    def apply_grad(self, data_set: DataSet, w: ndarray) -> ndarray:
        (x, y) = (data_set.x, data_set.y)
        dim = data_set.dim
        qty_c = data_set.qty_category
        qty_sample = data_set.qty_sample
        assert(w.shape == (qty_c, dim))

        posterior = self.__calc_posterior(data_set, w)

        # yr[r, i] = [[ y_i == r ]] を作る
        yr = np.array([[int(i==y[j]) for j in range(0, y.shape[0])] for i in range(0, qty_c)])
        assert(yr.shape == (qty_c, qty_sample))

        grad = (posterior - yr).dot(x) + 2 * self.__lambda * w
        assert(grad.shape == w.shape)
        return grad

    # hessian[i] = w_r用のヘッシアン
    def apply_hessian(self, data_set: DataSet, w: ndarray) -> ndarray:
        (x, y) = (data_set.x, data_set.y)
        dim = data_set.dim
        qty_c = data_set.qty_category
        qty_sample = data_set.qty_sample
        assert(w.shape == (qty_c, dim))

        # xx[i] = (x_i.T)(x_i)な3次元配列を作る
        xx = np.array([x[i:i+1].T.dot(x[i:i+1]) for i in range(0, qty_sample)])

        posterior = self.__calc_posterior(data_set, w)

        # hessian[i] = w_r用のヘッシアン
        hessian = np.zeros((qty_c, dim, dim))
        # w_rに関する部分を個別に計算
        for r in range(0, qty_c):
            p1mp = posterior * (1 - posterior)
            hes_r = np.zeros((dim, dim))
            for i in range(0, qty_sample):
                hes_r += p1mp[r, i] * xx[i]
            hes_r += 2 * self.__lambda
            hessian[r] = hes_r

        assert(hessian.shape == (qty_c, dim, dim))
        return hessian

    def get_grad_lipsitz(self, data_set: DataSet) -> float:
        x = data_set.x
        lip = np.trace(x.dot(x.T)) / 4.0 + 2.0 * self.__lambda
        return lip
