import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from abc import abstractmethod
import numpy as np

EPS = 10e-7

def main():
    # solve problem
    data_set = gen_dataset_iv()
    obj_func = LinearLogistic()
    w0 = 3 * np.random.rand(data_set.dim, 1)

    solver = NewtonMethod(obj_func)
    (loss_hist_n, w_hist_n) = solver.batch_train(data_set, w0)

    solver = SteepestGradDescMethod(obj_func)
    (loss_hist_g, w_hist_g) = solver.batch_train(data_set, w0)

    # plot result
    mim_loss = loss_hist_n[-1]

    dif_loss_g = np.abs(loss_hist_g - mim_loss)
    ts_g = np.arange(0, len(dif_loss_g), 1)
    dif_loss_n = np.abs(loss_hist_n - mim_loss)
    ts_n = np.arange(0, len(dif_loss_n), 1)

    assert(dif_loss_n[0] == dif_loss_g[0])

    plt.plot(ts_n, dif_loss_n, 'bo-', linewidth=0.5, markersize=0.5, label='newton')
    plt.plot(ts_g, dif_loss_g, 'ro-', linewidth=0.5, markersize=0.5, label='steepest')
    plt.legend()
    plt.xlabel('t')
    plt.ylabel('|J(w(t)) - j(w^)|')
    plt.gca().set_yscale('log')
    plt.show()

# log(1 + e^x) を 計算する
# xがでかい場合でもオーバーフローしない
def log1p_exp(x):
    def for1d(a):
        if a <= 0:
            return np.log1p(np.exp(a))
        else:
            return a + np.log1p(np.exp(-a))

    return np.array([*map(lambda a: for1d(a), x)])


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


class ObjFunc():
    # データセットを用いたパラメタの設定などが必要な場合はこれを用いる
    @abstractmethod
    def setup_with_data_set(self, data_set: DataSet):
        pass

    @abstractmethod
    def apply(self, data_set: DataSet, w):
        pass

    @abstractmethod
    def apply_grad(self, data_set: DataSet, w):
        pass

    @abstractmethod
    def apply_hessian(self, data_set: DataSet, w):
        pass

    # 損失関数の勾配のLipsitz連続性の係数
    @abstractmethod
    def get_grad_lipsitz(self, data_set: DataSet):
        pass
        
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

class LinearLogistic(ObjFunc):
    def __init__(self) -> None:
        super().__init__()

    # i番目の要素が p_i = p(y_i | x_i, w) となる列ベクトルを返す
    # p_i = 1/(1 + exp(-y w.T x))
    def __calc_posterior(self, data_set: DataSet, w):
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

    def setup_with_data_set(self, data_set: DataSet):
        self.__lambda = 0.01 * data_set.qty_sample

    def apply(self, data_set: DataSet, w):
        assert(data_set.dim == w.shape[0])
        (x, y) = (data_set.x, data_set.y)

        loss_without_reg = np.sum(log1p_exp(-y * x.dot(w)), axis=0)
        reg = self.__lambda * (w.T.dot(w))
        loss = loss_without_reg + reg
        assert(loss.shape == (1, 1))
        return loss

    def apply_grad(self, data_set: DataSet, w):
        assert(data_set.dim == w.shape[0])
        (x, y) = (data_set.x, data_set.y)

        posterior = self.__calc_posterior(data_set, w)
        grad_without_reg = np.sum(-y * x * (1.0 - posterior), axis=0)
        grad = grad_without_reg.reshape(w.shape) + 2 * self.__lambda * w
        assert(grad.shape == w.shape)
        return grad

    def apply_hessian(self, data_set: DataSet, w):
        assert(data_set.dim == w.shape[0])
        x = data_set.x

        p = self.__calc_posterior(data_set, w)
        acc = np.zeros((data_set.dim, data_set.dim))
        for i in range(0, data_set.qty_sample):
            xi = x[i,:].reshape((1, data_set.dim))
            acc += p[i] * (1 - p[i]) * xi.T.dot(xi)

        acc += 2 * self.__lambda

        return acc

    def get_grad_lipsitz(self, data_set: DataSet):
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


class SteepestGradDescMethod():
    def __init__(self, obj_func):
        self.__obj_func: ObjFunc = obj_func

    def __calc_update_dir(self, data_set: DataSet, w):
        return -self.__obj_func.apply_grad(data_set, w)

    # argmin_α J(w + α * (-1)*grad(w)) に近いものを求める
    def __calc_step_length_by_back_tracking(self, data_set: DataSet, w):
        grad = self.__obj_func.apply_grad(data_set, w)
        d = self.__calc_update_dir(data_set, w)

        # initialize
        alpha = 10.0
        rho = 0.8
        c = 0.5

        # iterate (Armijo条件を満たすようになるまでalphaを小さくしていく)
        j = self.__obj_func.apply(data_set, w)
        g = lambda a : self.__obj_func.apply(data_set, w + a * d)
        

        while g(alpha) > j + c * alpha * (grad.T.dot(d)):
            alpha = rho * alpha

        # print("alpha: " + str(alpha))

        return alpha

    def batch_train(self, data_set: DataSet, w0):
        self.__obj_func.setup_with_data_set(data_set)
        # init w
        w = w0
        # init log
        w_hist = [w]
        loss_hist = [self.__obj_func.apply(data_set, w)[0, 0]]

        # do optimize
        # print("loss : ", str(loss_hist[-1]))
        lip = self.__obj_func.get_grad_lipsitz(data_set)

        max_step = 500
        t = 1
        while np.linalg.norm(self.__obj_func.apply_grad(data_set, w), ord=2) > EPS and t <= max_step:
            grad = self.__obj_func.apply_grad(data_set, w)

            # 良さげな更新幅を見るためのコード
            # lipは大きすぎて最適化が全く進まない
            # lx = np.arange(-200, 200, 1)
            # ly = np.array([*map(lambda a: self.__obj_func.apply(data_set, w - a*grad), lx)]).reshape((400,))
            # plt.plot(lx, ly)
            # plt.show()

            # ステップ幅を良さげに決める場合
            # step = self.__calc_step_length_by_back_tracking(data_set, w)
            # w = w - step * grad

            w = w - (1.0 / lip) * grad

            w_hist.append(w)
            loss_hist.append(self.__obj_func.apply(data_set, w)[0, 0])
            # print("loss : ", str(loss_hist[-1]))
            t += 1

        return (loss_hist, w_hist)

class NewtonMethod():
    def __init__(self, obj_func):
        self.__obj_func: ObjFunc = obj_func

    def __calc_update_dir(self, data_set: DataSet, w):
        neg_grad = -self.__obj_func.apply_grad(data_set, w)
        hessian = self.__obj_func.apply_hessian(data_set, w)
        return np.dot(np.linalg.inv(hessian), neg_grad)
    
    def batch_train(self, data_set: DataSet, w0):
        self.__obj_func.setup_with_data_set(data_set)

        w = w0
        w_hist = [w]
        loss_hist = [self.__obj_func.apply(data_set, w)[0, 0]]

        while np.linalg.norm(self.__obj_func.apply_grad(data_set, w), ord=2) > EPS:
            dir = self.__calc_update_dir(data_set, w)

            # 良さげな更新幅を見るためのコード
            # lx = np.arange(-3, 3, 0.1)
            # ly = np.array([*map(lambda a: self.__obj_func.apply(data_set, w + a*dir), lx)]).reshape((60,))
            # plt.plot(lx, ly)
            # plt.show()

            w = w + dir
            w_hist.append(w)
            loss_hist.append(self.__obj_func.apply(data_set, w)[0, 0])
            # print("loss : ", str(loss_hist[-1]))

        return (loss_hist, w_hist)



if __name__ == "__main__":
    np.random.seed(777) # 103 777 is OK
    # np.random.seed(777) # newton法が計算不能(ヘッシアンが正則でなくなる)
    main()

    # (x, y) = (data_set.x, data_set.y)
    # plt.plot(np.extract(y>0, x[:,1]), np.extract(y>0, x[:,2]), 'x')
    # plt.plot(np.extract(y<0, x[:,1]), np.extract(y<0, x[:,2]), 'o')
    # plt.xlabel('x[1]')
    # plt.ylabel('x[2]')

    # lx = np.arange(-2, 2, 0.1)
    # ly = -w[1]/w[2] * lx + w[0]
    # plt.plot(lx, ly)

    # plt.show()
