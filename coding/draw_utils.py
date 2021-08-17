import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm

from coding.utils import getLocation


def plot_centroids(centroids, weights=None, circle_color='w', cross_color='k'):
    if weights is not None:
        centroids = centroids[weights > weights.max() / 10]
    plt.scatter(centroids[:, 0], centroids[:, 1],
                marker='o', s=35, linewidths=8,
                color=circle_color, zorder=10, alpha=0.9)
    plt.scatter(centroids[:, 0], centroids[:, 1],
                marker='x', s=2, linewidths=12,
                color=cross_color, zorder=11, alpha=1)


def plot_gaussian_mixture(clusterer, X, resolution=1000, show_ylabels=True):
    mins = X.min(axis=0) - 0.1
    maxs = X.max(axis=0) + 0.1
    xx, yy = np.meshgrid(np.linspace(mins[0], maxs[0], resolution),
                         np.linspace(mins[1], maxs[1], resolution))
    Z = -clusterer.score_samples(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z,
                 norm=LogNorm(vmin=1.0, vmax=30.0),
                 levels=np.logspace(0, 2, 12))
    plt.contour(xx, yy, Z,
                norm=LogNorm(vmin=1.0, vmax=30.0),
                levels=np.logspace(0, 2, 12),
                linewidths=1, colors='k')

    Z = clusterer.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contour(xx, yy, Z,
                linewidths=2, colors='r', linestyles='dashed')

    plt.plot(X[:, 0], X[:, 1], 'k.', markersize=2)
    plot_centroids(clusterer.means_, clusterer.weights_)

    plt.xlabel("$x_1$", fontsize=14)
    if show_ylabels:
        plt.ylabel("$x_2$", fontsize=14, rotation=0)
    else:
        plt.tick_params(labelleft=False)




def PlotGraph():

    X_DataFrame = getLocation()
    # X_DataFrame
    X_train = X_DataFrame.values


    # y_pred = class_assign[:, 0]
    # print('y', y_pred)
    # print(center.shape)
    # y_pred = np.asarray(y_pred)
    # for i in range(k):
    #     y_pred = np._r[y_pred, class_assign[i,0]]

    plt.scatter(X_train[:, 0], X_train[:, 1],  cmap='coolwarm')


def PlotPath(path, label):
    # get location
    X_DataFrame = getLocation()
    # X_DataFrame
    paths = [0] * len(path)
    for i, points in enumerate(path):
        for point in points:
            print('point', point)
            for idx, index in enumerate(X_DataFrame.index):
                if index == point:
                    print(index, point)
                    # paths = np.r_[paths, [X_DataFrame.iloc[idx, :]]]

    print(paths)
    plt.plot(paths, label=label)
