import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import numpy as np
from util import *

np.random.seed(777)
EPS = 10e-7

# prepare dataset v
N = 200
D = 4
C = 3
# オフセット用に最後の要素を1にしておく
x = 3 * (np.random.rand(N, D) - 0.5)
x[:,2] = 1.0
W = np.array([
    [2, -1, 0.5, 0],
    [-3, 2, 1, 0],
    [1, 2, 3, 0]
])
print("true w:")
print(W)

noize = 0.5 * np.random.rand(N, 3)
wxs = x.dot(W.T)

wxs += noize
# maxlogit = np.max(wxs, axis=1)
y = np.argmax(wxs, axis=1)
y = y.reshape((len(y), 1))

# define objective function
lambda_ = 0.01 * N
w0 = 3 * np.random.rand(C, D)

# クロージャ。x,N,yが束縛される。
def calc_loss(w):
    ## lse[i] = ln(Σ[c=0..C] exp(<w_c, x_i>))
    lse = np.apply_along_axis(log_sum_exp, 1, x.dot(w.T)).reshape((N, 1))
    ## slse = Σ[i=0..n] ln(Σ[c=0..C] exp(<w_c, x_i>))
    slse = np.sum(lse)
    ## Σ[i=0..n] -<w_y(i) x_i>
    ## を求める
    acc = 0.0
    for i in range(0, N):
        wyi = w[y[i]]
        xi = x[i]
        acc += -wyi.dot(xi)
    reg = np.linalg.norm(w, ord=2)**2
    loss = slse + acc + reg
    return loss

# ## steepest gradient method
max_iter = 300
loss_hist_sgm = []
w_hist_sgm = []
lip = np.trace(x.dot(x.T)) / 4.0 + 2.0 * lambda_

iter = 0
w = w0
w_hist_sgm.append(w0)
loss_hist_sgm.append(calc_loss(w0))
for iter in range(max_iter):
    # calc posterior : posterior[r, i] = p(r | xi)
    posterior = np.apply_along_axis(softmax, 0, w.dot(x.T))

    # calc grad : grad[:, r] = grad of w_r
    ## yr[r, i] = [[ y_i == r ]]
    yr = np.identity(C)[y.reshape((N,))].T
    grad = (posterior - yr).dot(x) + 2 * lambda_ * w

    # 終了判定
    if np.linalg.norm(grad, ord=2) < EPS:
        break

    # update weight and calc new loss
    w = w - (1.0 / lip) * grad
    loss = calc_loss(w)

    # update history
    w_hist_sgm.append(w)
    loss_hist_sgm.append(loss)

# ts_g = np.arange(0, len(loss_hist_sgm), 1)
# plt.plot(ts_g, loss_hist_sgm, 'ro-', linewidth=0.5, markersize=0.5, label='steepest')
# plt.show()

# newton method
max_iter = 300
loss_hist_newton = []
w_hist_newton = []

iter = 0
w = w0
w_hist_newton.append(w0)
loss_hist_newton.append(calc_loss(w0))
for iter in range(max_iter):
    # calc posterior : posterior[r, i] = p(r | xi)
    posterior = np.apply_along_axis(softmax, 0, w.dot(x.T))

    # calc grad : grad[:, r] = grad of w_r
    ## yr[r, i] = [[ y_i == r ]]
    yr = np.identity(C)[y.reshape((N,))].T
    grad = (posterior - yr).dot(x) + 2 * lambda_ * w

    # 終了判定
    if np.linalg.norm(grad, ord=2) < EPS:
        break

    # calc hessian
    hess = np.zeros((D*C, D*C))
    ## xx[i] = (x_i.T)(x_i)な3次元配列を作る
    xx = np.array([x[i:i+1].T.dot(x[i:i+1]) for i in range(0, N)])
    for s in range(0, C):
        for r in range(0, C):
            # H_{sr} を計算
            ## pp[i, i] = p(r|x_i)( [s==r] - p(s|x_i) )
            pp = np.diag(posterior[r] * (int(s==r) - posterior[s]))
            hess[s*D:s*D + D, r*D:r*D + D] = x.T.dot(pp).dot(x)

    hess += 2 * lambda_ * np.identity(D*C)

    # hessianが正定値行列でないなら、うまく行かないので警告
    # if not np.all(np.linalg.eigvals(hess) > 0):
    #     print("[WARNING] ヘッシアンが正定値じゃないよ")

    # update weight and calc new loss

    # 良さげな更新幅を見るためのコード
    lx = np.arange(-2, 2, 0.1)
    ly = np.array([*map(lambda a: calc_loss(w - a*np.linalg.inv(hess).dot(grad.reshape((C*D,1))).reshape(w.shape)), lx)]).reshape((len(lx),))
    plt.plot(lx, ly)
    plt.show()

    w = w - np.linalg.inv(hess).dot(grad.reshape((C*D,1))).reshape(w.shape)

    # update history
    w_hist_newton.append(w)
    loss = calc_loss(w)
    loss_hist_newton.append(loss)


print(f"sgm w :\n{w_hist_sgm[-1]}")
print(f"newton w :\n{w_hist_newton[-1]}")

# show graph
min_loss = loss_hist_newton[-1]

dif_loss_g = np.abs(loss_hist_sgm - min_loss)
ts_g = np.arange(0, len(dif_loss_g), 1)
dif_loss_n = np.abs(loss_hist_newton - min_loss)
ts_n = np.arange(0, len(dif_loss_n), 1)

plt.plot(ts_n, loss_hist_newton, 'bo-', linewidth=0.5, markersize=0.5, label='newton')
plt.plot(ts_g, loss_hist_sgm, 'ro-', linewidth=0.5, markersize=0.5, label='steepest')
plt.xlabel('#iteration')
plt.ylabel('|J(w(t))|')
plt.legend()
plt.show()

plt.plot(ts_n, dif_loss_n, 'bo-', linewidth=0.5, markersize=0.5, label='newton')
plt.plot(ts_g, dif_loss_g, 'ro-', linewidth=0.5, markersize=0.5, label='steepest')
plt.legend()
plt.xlabel('#iteration')
plt.ylabel('|J(w(t)) - j(w^)|')
plt.gca().set_yscale('log')
plt.show()
