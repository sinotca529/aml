import numpy as np

from dataset import *
from objfunc import *

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


