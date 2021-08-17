# from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np

from coding.draw_utils import plot_centroids
from coding.utils import getLocation


def getFixedPoints():
    point_location = getLocation()
    boolArr = np.zeros(point_location.shape[0], dtype=bool)
    for idx, point in enumerate(point_location.index):
        if point.startswith("Z"):
            boolArr[idx] = 1
    # K = [point for point in point_location.index if point.startswith("Z") ]
    BuGiPoint = point_location[boolArr]
    centers = BuGiPoint.values
    return centers


from coding.Problem3.KMeansImpl import KMeans, biKmeans, PickNewPoint

plt.figure()
X_DataFrame = getLocation()
# X_DataFrame
X_train = X_DataFrame.values

k = 8
# clf = KMeans(n_clusters=k)
fixed_centroids = getFixedPoints()
clf = KMeans(k=k, itertimes=2000,fixed_centroids=fixed_centroids)

centroids, class_assign = clf.fit(dataSet=X_train)
# clf.fit(X_train)
# y_pred = clf.predict(X_train)
y_pred = class_assign[:, 0]
print('y', y_pred)
# print(centroids.shape)
y_pred = np.asarray(y_pred)
# for i in range(k):
#     y_pred = np._r[y_pred, class_assign[i,0]]

plt.scatter(X_train[:, 0], X_train[:, 1], c=y_pred + 1, cmap='coolwarm')


def plot_data(X):
    plt.plot(X[:, 0], X[:, 1], 'k.', markersize=2)


def plot_decision_boundaries(clusterer, X, resolution=1, show_centroids=True,
                             show_xlabels=True, show_ylabels=True):
    mins = X.min(axis=0) - 0.1
    maxs = X.max(axis=0) + 0.1
    xx, yy = np.meshgrid(np.linspace(mins[0], maxs[0], resolution),
                         np.linspace(mins[1], maxs[1], resolution))
    # np.c_[xx.ravel(), yy.ravel()]
    Zz = np.zeros((xx.shape[0],2))
    Zz[:,0] = xx
    Zz[:,1] = yy
    Z = clusterer.predict(Zz)
    print(Z)
    Z = Z.reshape(xx.shape[0])

    plt.contourf(Z, extent=(mins[0], maxs[0], mins[1], maxs[1]),
                 cmap="Pastel2")
    plt.contour(Z, extent=(mins[0], maxs[0], mins[1], maxs[1]),
                linewidths=1, colors='k')
    plot_data(X)
    if show_centroids:
        plot_centroids(clusterer.cluster_centers_)

    if show_xlabels:
        plt.xlabel("$x$", fontsize=14)
    else:
        plt.tick_params(labelbottom=False)
    if show_ylabels:
        plt.ylabel("$y$", fontsize=14, rotation=0)
    else:
        plt.tick_params(labelleft=False)


def plot_givenPoint():
    point_location = getLocation()
    boolArr = np.zeros(point_location.shape[0], dtype=bool)
    for idx, point in enumerate(point_location.index):
        if point.startswith("Z"):
            boolArr[idx] = 1
    # K = [point for point in point_location.index if point.startswith("Z") ]
    BuGiPoint = point_location[boolArr]

    def plot_data_with_text(X):
        plt.plot(X[:, 0], X[:, 1], 'k.', markersize=5)
        for idx, text in enumerate(BuGiPoint.index):
            plt.text(x=X[idx, 0], y=X[idx, 1], s=text, ha='center')

    plot_data_with_text(BuGiPoint.values)



new2_center_name = PickNewPoint(clf.centroids)

print(new2_center_name)
# # plot_decision_boundaries(clf, X_train)
plot_givenPoint()
centroids = np.asarray(centroids)
#
plot_centroids(centroids)
# plot_centroids(clf.cluster_centers_)
plt.show()
