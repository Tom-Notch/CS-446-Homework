#!/usr/bin/env python3
import hw4_utils
import torch


def k_means(X=None, init_c=None, n_iters=50):
    """K-Means.

    Argument:
        X: 2D data points, shape [2, N].
        init_c: initial centroids, shape [2, 2]. Each column is a centroid.

    Return:
        c: shape [2, 2]. Each column is a centroid.
    """

    if X is None:
        X, init_c = hw4_utils.load_data()

    r = torch.zeros(X.shape)

    for _ in range(n_iters):
        for i in range(X.shape[-1]):
            if torch.norm(X[:, i] - init_c[:, 0]) < torch.norm(X[:, i] - init_c[:, 1]):
                r[0, i] = 1
                r[1, i] = 0
            else:
                r[0, i] = 0
                r[1, i] = 1
        x1 = X[:, torch.nonzero(r[0, :]).T]
        x2 = X[:, torch.nonzero(r[1, :]).T]

        init_c[:, :1] = torch.mean(x1, dim=-1)
        init_c[:, 1:] = torch.mean(x2, dim=-1)

        hw4_utils.vis_cluster(init_c[:, :1], x1, init_c[:, 1:], x2)

    return init_c
