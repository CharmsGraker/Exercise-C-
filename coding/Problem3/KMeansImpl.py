'''
Created on Feb 16, 2011
k Means Clustering for Ch10 of Machine Learning in Action
@author: Peter Harrington
'''
import copy

from numpy import *

import numpy as np

from coding.utils import getLocation, getJpointLocation


def loadDataSet(fileName):  # general function to parse tab -delimited floats
    dataMat = []  # assume last column is target value
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = map(float, curLine)  # map all elements to float()
        dataMat.append(fltLine)
    return dataMat


def distEclud(vecA, vecB):
    return sqrt(sum(power(vecA - vecB, 2)))  # la.norm(vecA-vecB)


def randCent(X, k):
    n = shape(X)[1]
    # print(X)
    centroids = np.mat(np.zeros((k, n)))  # create centroid mat
    for j in np.arange(n):  # create random cluster centers, within bounds of each dimension
        minJ = np.min(X[:, j])
        print(minJ)
        rangeJ = float(np.max(X[:, j]) - minJ)
        centroids[:, j] = np.mat(minJ + rangeJ * np.random.rand(k, 1))
    return centroids


class KMeans(object):
    def __init__(self, k, fixed_centroids, itertimes= 1000,distMeas=distEclud,createCent=randCent):
        self.k = k
        self.fixed_centroids = fixed_centroids
        self.distMeas = distMeas
        self.createCent = createCent
        self.itertimes = itertimes

    def fit(self, dataSet):
        m = shape(dataSet)[0]
        self.distMeas = distEclud

        clusterAssment = mat(zeros((m, 2)))  # create mat to assign data points
        # to a centroid, also holds SE of each point
        self.centroids = self.createCent(dataSet, self.k)
        self.clusterChanged = True
        i = 0
        print('--------start iter---------')
        while self.clusterChanged and i < self.itertimes:
            self.clusterChanged = False
            clusterAssment = self.predict(dataSet)
            # for i in range(m):  # for each data point assign it to the closest centroid
            #     minDist = inf
            #     minIndex = -1
            #     for j in range(k):
            #         distJI = distMeas(self.centroids[j, :], dataSet[i, :])
            #         if distJI < minDist:
            #             minDist = distJI
            #             minIndex = j
            #     if self.clusterAssment[i, 0] != minIndex:
            #         clusterChanged = True
            #     self.clusterAssment[i, :] = minIndex, minDist ** 2
            # print(self.centroids)
            for cent in range(self.k):  # recalculate centroids
                ptsInClust = dataSet[nonzero(clusterAssment[:, 0].A == cent)[0]]  # get all the point in this cluster
                # centroids[cent, :] = mean(ptsInClust, axis=0)  # assign centroid to mean

                self.centroids[cent, :] = np.mean(ptsInClust, axis=0)
                # print(self.fixed_centroids)
                if cent < 6:
                    scaler = 1
                    self.centroids[cent, :] = self.fixed_centroids[cent, :]
            i += 1
        print('--------end iter---------')
        return self.centroids, clusterAssment

    def predict(self, X):
        m = shape(X)[0]
        clusterAssment = mat(zeros((m, 2)))  # create mat to assign data points

        for i in range(m):  # for each data point assign it to the closest centroid
            minDist = inf
            minIndex = -1
            for j in range(self.k):
                distJI = self.distMeas(self.centroids[j, :], X[i, :])
                if distJI < minDist and self.centroids.shape[0] <= 8:
                    minDist = distJI
                    minIndex = j

            # print(clusterAssment.shape)
            # print(minIndex)
            if clusterAssment[i, 0] != minIndex:
                self.clusterChanged = True
            clusterAssment[i, :] = minIndex, minDist ** 2
        return clusterAssment


class biKmeans(KMeans):
    def __init__(self,*args, **kwargs):
        super().__init__(*args, **kwargs)

    def fit(self, dataSet):
        m = shape(dataSet)[0]
        clusterAssment = np.mat(zeros((m, 2)))
        centroid0 = np.mean(dataSet, axis=0).tolist()[0]
        self.centroids = [centroid0]  # create a list with one centroid
        for j in range(m):  # calc initial Error
            clusterAssment[j, 1] = self.distMeas(mat(centroid0), dataSet[j, :]) ** 2
        while (len(self.centroids) < self.k):
            lowestSSE = inf
            for i in range(len(self.centroids)):
                ptsInCurrCluster = dataSet[nonzero(clusterAssment[:, 0].A == i)[0],
                                   :]  # get the data points currently in cluster i
                # print(ptsInCurrCluster)
                # fixed = copy.deepcopy(self.fixed_centroids)
                print(self.fixed_centroids)
                model = KMeans(k=2,
                               fixed_centroids=self.fixed_centroids,
                               distMeas=self.distMeas,
                               itertimes=self.itertimes)
                centroidMat, splitClustAss = model.fit(dataSet=ptsInCurrCluster)
                sseSplit = sum(splitClustAss[:, 1])  # compare the SSE to the currrent minimum
                sseNotSplit = sum(clusterAssment[nonzero(clusterAssment[:, 0].A != i)[0], 1])
                print("sseSplit, and notSplit: ", sseSplit, sseNotSplit)
                if (sseSplit + sseNotSplit) < lowestSSE:
                    bestCentToSplit = i
                    bestNewCents = centroidMat
                    bestClustAss = splitClustAss.copy()
                    lowestSSE = sseSplit + sseNotSplit
            bestClustAss[nonzero(bestClustAss[:, 0].A == 1)[0], 0] = len(self.centroids)  # change 1 to 3,4, or whatever
            bestClustAss[nonzero(bestClustAss[:, 0].A == 0)[0], 0] = bestCentToSplit
            print('the bestCentToSplit is: ', bestCentToSplit)
            print('the len of bestClustAss is: ', len(bestClustAss))
            self.centroids[bestCentToSplit] = bestNewCents[0, :].tolist()[0]  # replace a centroid with two best centroids
            self.centroids.append(bestNewCents[1, :].tolist()[0])
            clusterAssment[nonzero(clusterAssment[:, 0].A == bestCentToSplit)[0],
            :] = bestClustAss  # reassign new clusters, and SSE
        return mat(self.centroids), clusterAssment


import urllib
import json


def geoGrab(stAddress, city):
    apiStem = 'http://where.yahooapis.com/geocode?'  # create a dict and constants for the goecoder
    params = {}
    params['flags'] = 'J'  # JSON return type
    params['appid'] = 'aaa0VN6k'
    params['location'] = '%s %s' % (stAddress, city)
    url_params = urllib.urlencode(params)
    yahooApi = apiStem + url_params  # print url_params
    print(yahooApi)
    c = urllib.urlopen(yahooApi)
    return json.loads(c.read())


from time import sleep


def massPlaceFind(fileName):
    fw = open('places.txt', 'w')
    for line in open(fileName).readlines():
        line = line.strip()
        lineArr = line.split('\t')
        retDict = geoGrab(lineArr[1], lineArr[2])
        if retDict['ResultSet']['Error'] == 0:
            lat = float(retDict['ResultSet']['Results'][0]['latitude'])
            lng = float(retDict['ResultSet']['Results'][0]['longitude'])
            print("%s\t%f\t%f") % (lineArr[0], lat, lng)
            fw.write('%s\t%f\t%f\n' % (line, lat, lng))
        else:
            print
        "error fetching"
        sleep(1)
    fw.close()


def distSLC(vecA, vecB):  # Spherical Law of Cosines
    a = sin(vecA[0, 1] * pi / 180) * sin(vecB[0, 1] * pi / 180)
    b = cos(vecA[0, 1] * pi / 180) * cos(vecB[0, 1] * pi / 180) * \
        cos(pi * (vecB[0, 0] - vecA[0, 0]) / 180)
    return arccos(a + b) * 6371.0  # pi is imported with numpy


import matplotlib
import matplotlib.pyplot as plt


def clusterClubs(numClust=5):
    datList = []
    for line in open('places.txt').readlines():
        lineArr = line.split('\t')
        datList.append([float(lineArr[4]), float(lineArr[3])])
    datMat = mat(datList)
    myCentroids, clustAssing = biKmeans(datMat, numClust, distMeas=distSLC)
    fig = plt.figure()
    rect = [0.1, 0.1, 0.8, 0.8]
    scatterMarkers = ['s', 'o', '^', '8', 'p', \
                      'd', 'v', 'h', '>', '<']
    axprops = dict(xticks=[], yticks=[])
    ax0 = fig.add_axes(rect, label='ax0', **axprops)
    imgP = plt.imread('Portland.png')
    ax0.imshow(imgP)
    ax1 = fig.add_axes(rect, label='ax1', frameon=False)
    for i in range(numClust):
        ptsInCurrCluster = datMat[nonzero(clustAssing[:, 0].A == i)[0], :]
        markerStyle = scatterMarkers[i % len(scatterMarkers)]
        ax1.scatter(ptsInCurrCluster[:, 0].flatten().A[0], ptsInCurrCluster[:, 1].flatten().A[0], marker=markerStyle,
                    s=90)
    ax1.scatter(myCentroids[:, 0].flatten().A[0], myCentroids[:, 1].flatten().A[0], marker='+', s=300)
    plt.show()


def PickNewPoint(centroids):
    get_2 = centroids[-2:, :]
    JpointLocations = getJpointLocation()
    # 对于新产生的两个点，选取最靠近他们的点为新增补给点
    names = []
    for Kmean_center in get_2:
        min_dist = inf
        dist = inf
        for JpointName in JpointLocations.keys():
            dist = distEclud(JpointLocations[JpointName],Kmean_center)
            if dist < min_dist:
                # 记录下标
                min_dist = dist
                min_name = JpointName
        names.append(min_name)
    return names


