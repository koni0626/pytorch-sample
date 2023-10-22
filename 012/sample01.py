# coding: UTF-8
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

"""
誤差伝搬法でx**2の最小値を求める
"""


def func(x):
    """
    xの2二乗を求める関数
    """
    return x ** 2


def gradient(x):
    """
    xの二乗を微分した時の値を求める関数
    """
    return 2 * x


def mse(y, t):
    """
    最小二乗法
    """
    return 0.5 * np.sum((y - t) ** 2)


def line(a, b):
    x = np.arange(-3, 3.1, 0.1)
    return a * x + b


def main():
    fig = plt.figure()

    x_record = np.arange(-3, 3.1, 0.1)
    y_record = func(x_record)
    g_record = gradient(x_record)
    l_record = []
    ims = []
    for x, y, g in zip(x_record, y_record, g_record):
        a = g
        b = y - (a * x)
        l = line(a, b)
        plt.xlim(-3, 3.1)
        plt.ylim(-0.5, 10.)
        plt.plot(x_record, y_record)
        im = plt.plot(x_record, l)
        ims.append(im)

    ani = animation.ArtistAnimation(fig, ims, interval=200)
    plt.show()


if __name__ == '__main__':
    main()
