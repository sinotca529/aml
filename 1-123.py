import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import numpy as np
from util import *

np.random.seed(777)
EPS = 10e-7

# prepare dataset iv
N = 200
D = 4
# オフセット用に最後の要素を1にしておく
x = 3 * (np.random.rand(N, D) - 0.5)
x[:,3] = 1

noize = 0.5 * np.random.rand(N, 1)
true_w = np.array([[2, -1, 0, 0.5]]).T
y = 2 * ((x.dot(true_w) + noize) > 0) - 1
print("true w :")
print(true_w.reshape((4,)))
assert(len(x) == N)
assert(len(x) == len(y))

# define objective function
lambda_ = 0.01 * N
w0 = 3 * np.random.rand(D, 1)

## steepst gradient method
max_iter = 500
w = w0
loss_hist_sgm = [np.sum(log1p_exp(-y * x.dot(w)), axis=0) + lambda_ * (w.T.dot(w))[0,0]]
w_hist_sgm = [w]
lip = np.trace(x.dot(x.T)) / 4.0 + 2.0 * lambda_

for iter in range(max_iter):
    posterior = sigmoid(y * x.dot(w))
    grad = np.sum(-y * x * (1.0 - posterior), axis=0).reshape((D, 1)) + 2 * lambda_ * w

    # 終了判定
    if np.linalg.norm(grad, ord=2) < EPS:
        break

    # update weight
    w = w - (1.0 / lip) * grad

    # update history
    w_hist_sgm.append(w)
    loss_hist_sgm.append(np.sum(log1p_exp(-y * x.dot(w)), axis=0) + lambda_ * (w.T.dot(w))[0,0])

## newton method
max_iter = 500
w = w0
loss_hist_newton = [np.sum(log1p_exp(-y * x.dot(w)), axis=0) + lambda_ * (w.T.dot(w))[0,0]]
w_hist_newton = [w]

while iter in range(max_iter):
    posterior = sigmoid(y * x.dot(w))
    grad = np.sum(-y * x * (1.0 - posterior), axis=0).reshape((D, 1)) + 2 * lambda_ * w
    # 終了判定
    if np.linalg.norm(grad, ord=2) < EPS:
        break

    # calc hessian
    p1mp = posterior * (1 - posterior)
    pp = np.diag(p1mp.reshape((N,)))
    hess = x.T.dot(pp).dot(x) + 2 * lambda_ * np.identity(D)
    if not np.all(np.linalg.eigvals(hess) > 0):
        print(f"[WARNING@{iter}-th iter] ヘッシアンが正定値じゃないよ")

    # update weight
    w = w - np.linalg.inv(hess).dot(grad)

    # update history
    w_hist_newton.append(w)
    loss_hist_newton.append(np.sum(log1p_exp(-y * x.dot(w)), axis=0) + lambda_ * (w.T.dot(w))[0,0])

# 結果の観察
print(f"sgm w :\n{w_hist_sgm[-1].reshape((4,))}")
print(f"newton w :\n{w_hist_newton[-1].reshape((4,))}")

## 学習データセットの正解率
y_sgm = 2 * ((x.dot(w_hist_sgm[-1]) + noize) > 0) - 1
y_sgm_ok = np.count_nonzero(y_sgm == y)
print(f"train correct sgm    : {y_sgm_ok} / {N}")

y_new = 2 * ((x.dot(w_hist_newton[-1]) + noize) > 0) - 1
y_new_ok = np.count_nonzero(y_new == y)
print(f"train correct newton : {y_new_ok} / {N}")

## 新規データに対する正答率
x = 3 * (np.random.rand(N, D) - 0.5)
x[:,3] = 1
noize = 0.5 * np.random.rand(N, 1)
y = 2 * ((x.dot(true_w) + noize) > 0) - 1

y_sgm = 2 * ((x.dot(w_hist_sgm[-1]) + noize) > 0) - 1
y_sgm_ok = np.count_nonzero(y_sgm == y)
print(f"test  correct sgm    : {y_sgm_ok} / {N}")

y_new = 2 * ((x.dot(w_hist_newton[-1]) + noize) > 0) - 1
y_new_ok = np.count_nonzero(y_new == y)
print(f"test  correct newton : {y_new_ok} / {N}")

# show graph
min_loss = loss_hist_newton[-1]

dif_loss_g = np.abs(loss_hist_sgm - min_loss)
ts_g = np.arange(0, len(dif_loss_g), 1)
dif_loss_n = np.abs(loss_hist_newton - min_loss)
ts_n = np.arange(0, len(dif_loss_n), 1)

plt.plot(ts_n, loss_hist_newton, 'bo-', linewidth=0.5, markersize=0.5, label='newton')
plt.plot(ts_g, loss_hist_sgm, 'ro-', linewidth=0.5, markersize=0.5, label='steepest')
plt.legend()
plt.xlabel('#iteration')
plt.ylabel('|J(w(t))|')
plt.gca().set_yscale('log')
plt.show()


plt.plot(ts_n, dif_loss_n, 'bo-', linewidth=0.5, markersize=0.5, label='newton')
plt.plot(ts_g, dif_loss_g, 'ro-', linewidth=0.5, markersize=0.5, label='steepest')
plt.legend()
plt.xlabel('#iteration')
plt.ylabel('|J(w(t)) - j(w^)|')
plt.gca().set_yscale('log')
plt.show()
