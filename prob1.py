import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import numpy as np

from dataset import *
from objfunc import *
from solver import *


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
