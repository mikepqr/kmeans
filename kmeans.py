import numpy as np
import matplotlib.pyplot as plt


def randomdata(K=2, m=100, ndim=2, sigma=0.2):
    """Generate random data for use in testing K-means algorithm.
    """

    # c will hold cluster assignment for each of m points
    c = np.zeros(m)
    # centroid will hold location of K centroids
    centroids = np.zeros([K, ndim])
    # x will hold location of m randomly generated data points, distributed
    # about the K centroids
    x = np.zeros([m, ndim])

    points_per_cluster = np.floor(m/K)
    # For each cluster
    for i in range(0, K):
        # Allocate m points to K clusters
        c[i*points_per_cluster:(i+1)*points_per_cluster] = i

    # For each dimension
    for i in range(0, ndim):
        centroids[:, i] = np.random.choice(range(K), K, replace=False)

    # For each data point
    for i in range(0, m):
        x[i, :] = centroids[c[i], :]
        x[i, :] += np.random.normal(0, sigma, ndim)

    np.random.shuffle(x)
    return x


def clusterassign(x, u):
    """
    For an array of centroids u in an n-dimensional space, return the index
    of the centroid in that array that is closest to the point x
    """
    return np.argmin(np.sum((x - u)**2, 1))


def choosestartingpoint(x, K):
    """Randomly choose K points from the array x of n-dimensional points
    """
    npoints = x.shape[0]
    i = np.random.choice(npoints, K, replace=False)
    return x[i, :]


def kmeans(x, K):
    """
    Divide data x into K clusters using K-means unsupervised learning
    """
    m, ndim = x.shape
    c = np.zeros(m)
    centroids = choosestartingpoint(x, K)
    newcentroids = choosestartingpoint(x, K)

    while True:
        print "Assigning"
        for i in range(0, m):
            c[i] = clusterassign(x[i, :], centroids)

        print "Moving"
        for i in range(0, K):
            newcentroids[i] = np.mean(x[c == i], 0)

        print "Testing for move"
        if np.mean(newcentroids - centroids) < 1e-12:
            break
        else:
            centroids = newcentroids

    markers = ['s', 'o', 'h', '+']
    colors = ['red', 'blue', 'green', 'cyan']
    for i in range(0, K):
        plt.scatter(x[c == i].T[0], x[c == i].T[1],
                    marker=markers[i], color=colors[i])

    return centroids, c
