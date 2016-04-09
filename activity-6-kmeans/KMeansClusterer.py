import pandas as pd
import numpy as np

class KMeansClusterer:

    def __assign_to_cluster__(self, point):
        # calculate distance (without sqrt) to each centroid
        distances = []
        for c in self.centroids:
            distances.append(((point - np.array(c)) ** 2).sum())
        # find index of closest cluster
        closest = np.argmin(distances)
        # assign point to that cluster
        self.clusters[closest].append(point.values)

    def train(self, data, k=3, n_iterations=100):
        """
        Train the K-Means clusterer
        :param data: a pandas DataFrame representing the data
        :param k: (optional) the number of clusters
        :param n_iterations: (optional) the number of training iterations
        :return: None
        """
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Data passed in must be a pandas DataFrame!")
        self.k = k
        self.centroids = []
        self.clusters = []
        self._data = data

        # random initialisation of centroids = pick K data points at random as centroids
        init_centroids = np.random.choice(range(len(data)), size=self.k, replace=False)

        for i in init_centroids:
            # get the data point at index i
            pt = self._data.iloc[i,:]
            # append it to the centroids list
            self.centroids.append(pt.values)

        for i in range(n_iterations):
            print('K-means iteration %d...' % i)
            # first, reset the clusters
            self.clusters = []
            for i in range(self.k):
                self.clusters.append([])
            # assign each data point to nearest cluster
            for i in range(len(self._data)):
                self.__assign_to_cluster__(self._data.iloc[i,:])

        # now, recalculate the centroids
        for i in range(k):
            self.centroids[i] = np.array(self.clusters[i]).mean(axis=0)