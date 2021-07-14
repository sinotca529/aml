import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import numpy as np
from util import *

np.random.seed(777)
EPS = 10e-7

# prepare dataset iv
n = 200
dim = 4
# オフセット用に最後の要素を1にしておく
x = 3 * (np.random.rand(n, dim) - 0.5)
x[:,3] = 1

noize = 0.5 * np.random.rand(n, 1)
true_w = np.array([[2, -1, 0, 0.5]]).T
y = (x.dot(true_w) + noize) > 0
y = 2 * y - 1

assert(len(x) == n)
assert(len(x) == len(y))

# define objective function
lambda_ = 0.01 * n
w0 = 3 * np.random.rand(dim, 1)

## steepst gradient method
max_iter = 500
loss_hist_sgm = []
w_hist_sgm = []
lip = np.trace(x.dot(x.T)) / 4.0 + 2.0 * lambda_

iter = 0
w = w0
w_hist_sgm.append(w0)
loss_hist_sgm.append(np.sum(log1p_exp(-y * x.dot(w)), axis=0) + lambda_ * (w.T.dot(w))[0,0])
while iter < max_iter:
    posterior = sigmoid(y * x.dot(w))
    grad = np.sum(-y * x * (1.0 - posterior), axis=0).reshape((dim, 1)) + 2 * lambda_ * w

    # 終了判定
    if np.linalg.norm(grad, ord=2) < EPS:
        break

    # update weight and calc new loss
    w = w - (1.0 / lip) * grad
    loss = np.sum(log1p_exp(-y * x.dot(w)), axis=0) + lambda_ * (w.T.dot(w))[0,0]

    # update history
    w_hist_sgm.append(w)
    loss_hist_sgm.append(loss)

    iter += 1


## newton method
max_iter = 500
loss_hist_newton = []
w_hist_newton = []

iter = 0
w = w0
w_hist_newton.append(w0)
loss_hist_newton.append(np.sum(log1p_exp(-y * x.dot(w)), axis=0) + lambda_ * (w.T.dot(w))[0,0])
while iter < max_iter:
    posterior = sigmoid(y * x.dot(w))
    grad = np.sum(-y * x * (1.0 - posterior), axis=0).reshape((dim, 1)) + 2 * lambda_ * w
    # 終了判定
    if np.linalg.norm(grad, ord=2) < EPS:
        break

    # calc hessian
    p1mp = posterior * (1 - posterior)
    acc = np.zeros((dim, dim))
    for i in range(0, n):
        xi = x[i,:].reshape((1, dim))
        acc += p1mp[i] * xi.T.dot(xi)
    acc += 2 * lambda_ * np.identity(dim)
    hess = acc
    # hessianが正定値行列でないなら、うまく行かないので警告
    if not np.all(np.linalg.eigvals(hess) > 0):
        print(f"[WARNING@{iter}-th iter] ヘッシアンが正定値じゃないよ")

    # update weight and calc new loss
    w = w - np.linalg.inv(hess).dot(grad)
    loss = np.sum(log1p_exp(-y * x.dot(w)), axis=0) + lambda_ * (w.T.dot(w))[0,0]

    # update history
    w_hist_newton.append(w)
    loss_hist_newton.append(loss)

    iter += 1

print(f"sgm w :\n{w_hist_sgm[-1].reshape((4,))}")
print(f"newton w :\n{w_hist_newton[-1].reshape((4,))}")

# show graph
min_loss = loss_hist_newton[-1]

dif_loss_g = np.abs(loss_hist_sgm - min_loss)
ts_g = np.arange(0, len(dif_loss_g), 1)
dif_loss_n = np.abs(loss_hist_newton - min_loss)
ts_n = np.arange(0, len(dif_loss_n), 1)

plt.plot(ts_n, dif_loss_n, 'bo-', linewidth=0.5, markersize=0.5, label='newton')
plt.plot(ts_g, dif_loss_g, 'ro-', linewidth=0.5, markersize=0.5, label='steepest')
plt.legend()
plt.xlabel('#iteration')
plt.ylabel('|J(w(t)) - j(w^)|')
plt.gca().set_yscale('log')
plt.show()
