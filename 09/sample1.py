# coding:UTF-8
import numpy as np


# softmax
def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a - c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a

    return y


# 最小二乗法
def mean_squared_error(y, t):
    return 0.5 * np.sum((y - t)**2)


# 交差エントロピー
def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    batch_size = y.shape[0]
    return -np.sum(t*np.log(y)) / batch_size



# 微分(1変数の場合)
def numerical_diff(f, x):
    h = 1e-4
    return (f(x+h) - f(x-h)) / (2*h)


# 微分(偏微分もできるよ)
def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)

    for idx in range(x.size):
        tmp_val = x[idx]
        x[idx] = tmp_val + h
        fxh1 = f(x)

        x[idx] = tmp_val - h
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2*h)
        x[idx] = tmp_val

    return grad


# 勾配降下
def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x

    for i in range(step_num):
        grad = numerical_gradient(f, x)
        x -= lr * grad

    return x


class simpleNet:
    def __init__(self):
        self.W = np.random.randn(2, 3)

    def predict(self, x):
        return np.dot(x, self.W)

    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y, t)
        return loss



if __name__ == '__main__':
    t = np.array([0, 0, 1])
    net = simpleNet()
    print(net.W)

    x = np.array([0.6, 0.9])
    p = net.predict(x)
    print(p)

    loss = net.loss(x, t)
    print(loss)

    dW = numerical_gradient(net.loss, net.W)
    print(dW)

