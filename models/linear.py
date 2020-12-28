import numpy as np


def model(a, x):
    return a * x


def cost_function(a, x, y):
    return 0.5 * np.mean(np.square(model(a, x) - y))


def cost_function_gradient(a, x, y):
    return np.sum(2 * x * (model(a, x) - y)) / (2 * len(x))


def update_params(a, x, y, lr):
    dloss_da = cost_function_gradient(a, x, y)
    return a - lr * dloss_da


def train(dataset, epochs, lr, early_stopping_delta):
    x = dataset.x
    y = dataset.y
    a_0 = 1000.
    a_hist = [a_0]
    loss_hist = [cost_function(a_0, x, y)]
    for _ in range(epochs):
        a_i = update_params(a_hist[-1], x, y, lr)
        a_hist.append(a_i)
        loss_i = cost_function(a_i, x, y)
        loss_hist.append(loss_i)
        if abs(loss_hist[-1] - loss_hist[-2]) < early_stopping_delta:
            break
    return a_hist, loss_hist
