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
print("true w :")
print(true_w.reshape((4,)))
assert(len(x) == n)
assert(len(x) == len(y))

max_iter = 100
iter = 0
alpha = 3 * np.random.rand(n, 1)
lam = 0.01*n
lip = np.trace(x.dot(x.T)) / 4.0 + 2.0 * lam

a = (y * x).T
k = a.T.dot(a)

def weight():
    return 1/(2*lam) * a.dot(alpha)

def project(x: ndarray) -> ndarray:
    assert(x.shape == (n, 1))
    r = x
    for i in range(0, n):
        r[i] = min(max(x[i], 0), 1)
    return r

def dual_lag() -> float:
    return (-1/(4*lam) * alpha.T.dot(k).dot(alpha) + alpha.sum())[0,0]

def hinge_loss() -> float:
    acc = 0
    w = weight()
    for i in range(0, n):
        acc += max(0, 1 - y[i] * w.T.dot(x[i:i+1].T))
    acc += lam * np.linalg.norm(w, ord=2)**2
    return acc[0]

dual_lag_hist = [dual_lag()]
hinge_loss_hist = [hinge_loss()]

while iter < max_iter:
    grad = (1/(2*lam)) * k.dot(alpha) - 1
    # 終了判定
    if np.linalg.norm(grad, ord=2) < EPS:
        break
    alpha = project(alpha - (1/lip) * grad)
    dual_lag_hist.append(dual_lag())
    hinge_loss_hist.append(hinge_loss())
    iter += 1

# graph
print("inferred w :")
print(weight().reshape((4,)))

ts_dual = np.arange(0, len(dual_lag_hist), 1)
ts_hinge = np.arange(0, len(hinge_loss_hist), 1)

plt.plot(ts_dual, dual_lag_hist, 'bo-', linewidth=0.5, markersize=0.5, label='dual lag fn')
plt.plot(ts_hinge, hinge_loss_hist, 'ro-', linewidth=0.5, markersize=0.5, label='hinge')
plt.xlabel('#iteration')
plt.legend()
plt.show()