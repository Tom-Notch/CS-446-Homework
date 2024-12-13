#!/usr/bin/env python3
import hw1_utils as utils
import matplotlib.pyplot as plt
import torch

"""
    Important
    ========================================
    The autograder evaluates your code using FloatTensors for all computations.
    If you use DoubleTensors, your results will not match those of the autograder
    due to the higher precision.

    PyTorch constructs FloatTensors by default, so simply don't explicitly
    convert your tensors to DoubleTensors or change the default tensor.

    Be sure to modify your input matrix X in exactly the way specified. That is,
    make sure to prepend the column of ones to X and not put the column anywhere
    else, and make sure your feature-expanded matrix in Problem 4 is in the
    specified order (otherwise, your w will be ordered differently than the
    reference solution's in the autograder).
"""

# Problem 3


def linear_gd(X, Y, lrate=0.01, num_iter=1000):
    """
    Arguments:
        X (n x d FloatTensor): the feature matrix
        Y (n x 1 FloatTensor): the labels
        num_iter (int): iterations of gradient descent to perform

    Returns:
        (d + 1) x 1 FloatTensor: the parameters w
    """
    n = X.shape[0]
    d = X.shape[1]
    X1 = torch.cat((torch.ones(n, 1), X), 1)
    w = torch.zeros(d + 1)
    for _ in range(num_iter):
        w = w - lrate / n * torch.matmul(
            X1.t(), (torch.matmul(X1, w).reshape(n, 1) - Y)
        ).reshape(d + 1)
    return w


def linear_normal(X, Y):
    """
    Arguments:
        X (n x d FloatTensor): the feature matrix
        Y (n x 1 FloatTensor): the labels

    Returns:
        (d + 1) x 1 FloatTensor: the parameters w
    """
    n = X.shape[0]
    d = X.shape[1]
    X1 = torch.cat((torch.ones(n, 1), X), 1)
    w = torch.matmul(X1.pinverse(), Y).reshape(d + 1)
    return w


def plot_linear():
    """
    Returns:
        Figure: the figure plotted with matplotlib
    """
    (X, Y) = utils.load_reg_data()
    plt.scatter(X, Y)
    n = X.shape[0]
    w = linear_normal(X, Y)
    X1 = torch.cat((torch.ones(n, 1), X), 1)
    Y_hat = torch.matmul(X1, w)
    plt.plot(X, Y_hat)
    plt.show()
    return plt.gcf()


# Problem 4


def poly_gd(X, Y, lrate=0.01, num_iter=1000):
    """
    Arguments:
        X (n x d FloatTensor): the feature matrix
        Y (n x 1 FloatTensor): the labels
        lrate (float): the learning rate
        num_iter (int): number of iterations of gradient descent to perform

    Returns:
        (1 + d + d * (d + 1) / 2) x 1 FloatTensor: the parameters w
    """
    n = X.shape[0]
    d = X.shape[1]
    X1 = torch.cat((torch.ones(n, 1), X), 1)
    for i in range(d):
        for j in range(i, d):
            X1 = torch.cat((X1, torch.mul(X[:, i], X[:, j]).reshape(n, 1)), 1)
    d_poly = int(1 + d + d * (d + 1) / 2)
    w = torch.zeros(d_poly)
    for _ in range(num_iter):
        w = w - lrate / n * torch.matmul(
            X1.t(), (torch.matmul(X1, w).reshape(n, 1) - Y)
        ).reshape(d_poly)
    return w


def poly_normal(X, Y):
    """
    Arguments:
        X (n x d FloatTensor): the feature matrix
        Y (n x 1 FloatTensor): the labels

    Returns:
        (1 + d + d * (d + 1) / 2) x 1 FloatTensor: the parameters w
    """
    n = X.shape[0]
    d = X.shape[1]
    X1 = torch.cat((torch.ones(n, 1), X), 1)
    for i in range(d):
        for j in range(i, d):
            X1 = torch.cat((X1, torch.mul(X[:, i], X[:, j]).reshape(n, 1)), 1)
    d_poly = int(1 + d + d * (d + 1) / 2)
    w = torch.matmul(X1.pinverse(), Y).reshape(d_poly)
    return w


def plot_poly():
    """
    Returns:
        Figure: the figure plotted with matplotlib
    """
    (X, Y) = utils.load_reg_data()
    plt.scatter(X, Y)
    n = X.shape[0]
    w = poly_normal(X, Y)
    X1 = torch.cat((torch.ones(n, 1), X), 1)
    for i in range(d):
        for j in range(i, d):
            X1 = torch.cat((X1, torch.mul(X[:, i], X[:, j]).reshape(n, 1)), 1)
    Y_hat = torch.matmul(X1, w)
    plt.plot(X, Y_hat)
    plt.show()
    return plt.gcf()


def poly_xor():
    """
    Returns:
        n x 1 FloatTensor: the linear model's predictions on the XOR dataset
        n x 1 FloatTensor: the polynomial model's predictions on the XOR dataset
    """
    (X, Y) = utils.load_xor_data()
    w_linear = linear_normal(X, Y)
    w_poly = poly_normal(X, Y)

    def linear_predict(X):
        n = X.shape[0]
        # d = X.shape[1]
        X1 = torch.cat((torch.ones(n, 1), X), 1)
        Y_hat = torch.matmul(X1, w_linear)
        return Y_hat

    def poly_predict(X):
        n = X.shape[0]
        d = X.shape[1]
        X2 = torch.cat((torch.ones(n, 1), X), 1)
        for i in range(d):
            for j in range(i, d):
                X2 = torch.cat((X2, torch.mul(X[:, i], X[:, j]).reshape(n, 1)), 1)
        Y_hat = torch.matmul(X2, w_poly)
        return Y_hat

    utils.contour_plot(-1, 1, -1, 1, linear_predict)
    utils.contour_plot(-1, 1, -1, 1, poly_predict)

    Y_hat_linear = linear_predict(X)
    Y_hat_poly = poly_predict(X)

    return Y_hat_linear, Y_hat_poly


# Problem 5


def logistic(X, Y, lrate=0.01, num_iter=1000):
    """
    Arguments:
        X (n x d FloatTensor): the feature matrix
        Y (n x 1 FloatTensor): the labels

    Returns:
        (d + 1) x 1 FloatTensor: the parameters w
    """
    n = X.shape[0]
    d = X.shape[1]
    X1 = torch.cat((torch.ones(n, 1), X), 1)
    w = torch.zeros(d + 1)
    for _ in range(num_iter):
        logistic_gradient = torch.mul(Y, torch.matmul(X1, w).reshape(n, 1))
        logistic_gradient = torch.mul(
            Y, torch.add(logistic_gradient.exp(), 1).reciprocal()
        )
        w = w + lrate / n * torch.matmul(logistic_gradient.t(), X1).reshape(d + 1)
    return w


def logistic_vs_ols():
    """
    Returns:
        Figure: the figure plotted with matplotlib
    """
    (X, Y) = utils.load_logistic_data()
    n = X.shape[0]
    w_logistics = logistic(X, Y, num_iter=1000000)
    w_linear = linear_gd(X, Y)
    plt.scatter(X[:, 0], X[:, 1], c=Y)

    x_min = X[0][0]
    x_max = X[0][0]
    boundary_resolution = 100
    for i in range(n):
        if X[i][0] < x_min:
            x_min = X[i][0]
        if X[i][0] > x_max:
            x_max = X[i][0]
    Y_boundary_linear = torch.zeros(boundary_resolution)
    Y_boundary_logistics = torch.zeros(boundary_resolution)
    X_boundary = torch.linspace(x_min, x_max, steps=boundary_resolution)
    for i in range(boundary_resolution):
        Y_boundary_linear[i] = (
            -(w_linear[0] + w_linear[1] * X_boundary[i]) / w_linear[2]
        )
        Y_boundary_logistics[i] = (
            -(w_logistics[0] + w_logistics[1] * X_boundary[i]) / w_logistics[2]
        )

    plt.plot(X_boundary, Y_boundary_linear, "r")
    plt.plot(X_boundary, Y_boundary_logistics, "g")

    return plt.gcf()
