#!/usr/bin/env python3
import re
from os import listdir

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

""" Start SVM helpers """


def svm_contour(pred_fxn, xmin=-5, xmax=5, ymin=-5, ymax=5, ngrid=33):
    """
    Produces a contour plot for the prediction function.

    Arguments:
        pred_fxn: Prediction function that takes an n x d tensor of test examples
        and returns your SVM's predictions.
        xmin: Minimum x-value to plot.
        xmax: Maximum x-value to plot.
        ymin: Minimum y-value to plot.
        ymax: Maximum y-value to plot.
        ngrid: Number of points to be plotted between max and min (granularity).
    """
    with torch.no_grad():
        xgrid = torch.linspace(xmin, xmax, ngrid)
        ygrid = torch.linspace(ymin, ymax, ngrid)
        (xx, yy) = torch.meshgrid((xgrid, ygrid))
        x_test = torch.cat(
            (xx.view(ngrid, ngrid, 1), yy.view(ngrid, ngrid, 1)), dim=2
        ).view(-1, 2)
        zz = pred_fxn(x_test)
        zz = zz.view(ngrid, ngrid)
        cs = plt.contour(
            xx.cpu().numpy(), yy.cpu().numpy(), zz.cpu().numpy(), cmap="coolwarm"
        )
        plt.clabel(cs)
        plt.show()


def poly_implementation(x, y, degree):
    assert x.size() == y.size(), "The dimensions of inputs do not match!"
    with torch.no_grad():
        return (1 + (x * y).sum()).pow(degree)


def poly(degree):
    return lambda x, y: poly_implementation(x, y, degree)


def rbf_implementation(x, y, sigma):
    assert x.size() == y.size(), "The dimensions of inputs do not match!"
    with torch.no_grad():
        return (-(x - y).norm().pow(2) / 2 / sigma / sigma).exp()


def rbf(sigma):
    return lambda x, y: rbf_implementation(x, y, sigma)


def xor_data():
    x = torch.tensor([[1, 1], [-1, 1], [-1, -1], [1, -1]], dtype=torch.float)
    y = torch.tensor([1, -1, 1, -1], dtype=torch.float)
    return x, y


""" End SVM Helpers """

""" Start CAFE Helpers """
IMAGE_DIR = "./CAFE Gamma/"  # Where the CAFE images are located
EXPRESSION_RE_TRAIN = r"\d[0-14-9]\d_([a-z])\d\.pgm"
EXPRESSION_RE_TEST = r"\d[2-3]\d_([a-z])\d\.pgm"
EXPRESSION_DICT = {  # Mapping from image labels to indices and meanings
    "a": {"expression": "anger", "index": 0},
    "d": {"expression": "disgusted", "index": 1},
    "h": {"expression": "happy", "index": 2},
    "m": {"expression": "maudlin", "index": 3},
    "f": {"expression": "fear", "index": 4},
    "s": {"expression": "surprise", "index": 5},
}
IMAGE_DIMS = (380, 240)  # Dimensions of a CAFE image

transform = transforms.Compose([transforms.ToTensor()])


def load_cafe(set="train"):
    files = listdir(IMAGE_DIR)

    expressions = [[] for i in range(6)]
    for file_name in files:
        if set == "train":
            expr = re.fullmatch(EXPRESSION_RE_TRAIN, file_name)
        else:
            expr = re.fullmatch(EXPRESSION_RE_TEST, file_name)
        if expr is not None:
            expressions[EXPRESSION_DICT[expr.group(1)]["index"]].append(file_name)

    for i in range(len(expressions)):
        expressions[i] = [
            Image.open(IMAGE_DIR + file_name) for file_name in expressions[i]
        ]

    return expressions


def get_cafe_data(set="train"):
    """
    Returns (data, labels) where data is an n x d tensor and labels is an
    n x 1 tensor.
    """
    images = load_cafe(set)

    # Build data and label tensors
    data = []
    labels = []

    for i, exprList in enumerate(images):
        data.extend([transform(image).reshape(-1) for image in exprList])
        labels.extend([i] * len(exprList))

    labels = torch.tensor(labels)
    data = torch.stack(data)

    return data, labels


""" End CAFE Helpers """
