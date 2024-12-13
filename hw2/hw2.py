#!/usr/bin/env python3
import hw2_utils
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix


def svm_solver(
    x_train, y_train, lr, num_iters, kernel=hw2_utils.poly(degree=1), c=None
):
    """
    Computes an SVM given a training set, training labels, the number of
    iterations to perform projected gradient descent, a kernel, and a trade-off
    parameter for soft-margin SVM.

    Arguments:
        x_train: 2d tensor with shape (n, d).
        y_train: 1d tensor with shape (n,), whose elements are +1 or -1.
        lr: The learning rate.
        num_iters: The number of gradient descent steps.
        kernel: The kernel function.
           The default kernel function is 1 + <x, y>.
        c: The trade-off parameter in soft-margin SVM.
           The default value is None, referring to the basic, hard-margin SVM.

    Returns:
        alpha: a 1d tensor with shape (n,), denoting an optimal dual solution.
               Initialize alpha to be 0.
               Return alpha.detach() could possibly help you save some time
               when you try to use alpha in other places.

    Note that if you use something like alpha = alpha.clamp(...) with
    torch.no_grad(), you will have alpha.requires_grad=False after this step.
    You will then need to use alpha.requires_grad_().
    Alternatively, use in-place operations such as clamp_().
    """
    (n, d) = x_train.shape

    H = torch.zeros(n, n).detach()

    for i in range(n):
        for j in range(n):
            H[i, j] = y_train[i] * y_train[j] * kernel(x_train[i], x_train[j])

    # training preparation
    alpha = torch.zeros(n, 1, requires_grad=True)
    loss = torch.zeros(num_iters, 1)

    for i in range(num_iters):
        risk = 1 / 2 * alpha.T @ H @ alpha - alpha.sum()
        risk.backward()
        loss[i] = risk
        with torch.no_grad():
            alpha -= lr * alpha.grad
            alpha.clamp_(min=0, max=float("inf") if c is None else c)
            alpha.grad.zero_()

    return alpha


def svm_predictor(alpha, x_train, y_train, x_test, kernel=hw2_utils.poly(degree=1)):
    """
    Returns the kernel SVM's predictions for x_test using the SVM trained on
    x_train, y_train with computed dual variables alpha.

    Arguments:
        alpha: 1d tensor with shape (n,), denoting an optimal dual solution.
        x_train: 2d tensor with shape (n, d), denoting the training set.
        y_train: 1d tensor with shape (n,), whose elements are +1 or -1.
        x_test: 2d tensor with shape (m, d), denoting the test set.
        kernel: The kernel function.
           The default kernel function is 1 + <x, y>.

    Return:
        A 1d tensor with shape (m,), the outputs of SVM on the test set.
    """
    (n, d) = x_train.shape
    (m, d) = x_test.shape

    Y_hat = torch.zeros(m, 1)

    for i in range(m):
        for j in range(n):
            Y_hat[i] += alpha[j] * y_train[j] * kernel(x_test[i], x_train[j])

    return Y_hat


class CAFENet(nn.Module):
    def __init__(self):
        """
        Initialize the CAFENet by calling the superclass' constructor
        and initializing a linear layer to use in forward().

        Arguments:
            self: This object.
        """
        super(CAFENet, self).__init__()
        self.inputDim = hw2_utils.IMAGE_DIMS[0] * hw2_utils.IMAGE_DIMS[1]
        self.outputDim = len(hw2_utils.EXPRESSION_DICT)
        self.linear = nn.Linear(self.inputDim, self.outputDim, bias=True)

    def forward(self, x):
        """
        Computes the network's forward pass on the input tensor.
        Does not apply a softmax or other activation functions.

        Arguments:
            self: This object.
            x: The tensor to compute the forward pass on.
        """
        return self.linear(x)


def fit(net, X, y, n_epochs=201):
    """
    Trains the neural network with CrossEntropyLoss and an Adam optimizer on
    the training set X with training labels Y for n_epochs epochs.

    Arguments:
        net: The neural network to train
        X: n x d tensor
        y: n x 1 tensor
        n_epochs: The number of epochs to train with batch gradient descent.

    Returns:
        List of losses at every epoch, including before training
        (for use in plot_cafe_loss).
    """
    (n, d) = X.shape

    # y = y.view((n, 1)).float()

    optimizer = torch.optim.Adam(net.parameters())

    loss = torch.zeros(n_epochs)

    for i in range(n_epochs):
        optimizer.zero_grad()
        y_hat = net(X)
        risk = nn.CrossEntropyLoss()(y_hat, y)
        risk.backward()
        loss[i] = risk
        optimizer.step()

    return loss


def plot_cafe_loss():
    """
    Trains a CAFENet on the CAFE dataset and plots the zero'th through 200'th
    epoch's losses after training. Saves the trained network for use in
    visualize_weights.
    """
    net = CAFENet()
    X, y = hw2_utils.get_cafe_data()

    loss = fit(net, X, y).detach()

    epoch = torch.linspace(0, 200, 201)
    plt.plot(epoch, loss)
    torch.save(net, "./net.pt")


def visualize_weights():
    """
    Loads the CAFENet trained in plot_cafe_loss, maps the weights to the grayscale
    range, reshapes the weights into the original CAFE image dimensions, and
    plots the weights, displaying the six weight tensors corresponding to the six
    labels.
    """
    net = torch.load("./net.pt")
    for name, w in net.named_parameters():
        if name == "linear.weight":
            w = (w - w.min()) * 255 / (w.max() - w.min())
            gray_w = w.to(torch.uint8).reshape(6, 380, -1)
            for i in range(gray_w.shape[0]):
                plt.figure(i)
                plt.imshow(gray_w[i, ...], cmap="gray")


def print_confusion_matrix():
    """
    Loads the CAFENet trained in plot_cafe_loss, loads training and testing data
    from the CAFE dataset, computes the confusion matrices for both the
    training and testing data, and prints them out.
    """
    net = torch.load("./net.pt")
    X_train, y_train = hw2_utils.get_cafe_data()
    X_test, y_test = hw2_utils.get_cafe_data(set="test")

    y_train_pred = torch.argmax(net(X_train).detach(), axis=1)
    y_test_pred = torch.argmax(net(X_test).detach(), axis=1)

    confusion_train = confusion_matrix(y_train, y_train_pred)
    confusion_test = confusion_matrix(y_test, y_test_pred)

    print(confusion_train)
    print(confusion_test)
