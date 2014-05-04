import numpy as np
import matplotlib.pyplot as plt


def randomdata(K=4, m=1000, ndim=2, sigma=0.4):
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


def calcdistortion(x, c, u):
    """
    For a given set of data, cluster allocation and cluster centroids,
    calculate the distortion (i.e. cost function)
    """
    m = x.shape[0]
    distortion = 0
    for i in range(m):
        distortion += np.sum((x[i] - u[c[i]])**2)
    distortion /= m

    return distortion


def plotkmeans(x, c, centroids):
    """Scatter plot cluter centroids and data coded by cluster allocation"""
    markers = ['s', 'o', 'h', '+']
    colors = ['red', 'blue', 'green', 'cyan']
    for i in range(0, centroids.shape[0]):
        plt.scatter(x[c == i].T[0], x[c == i].T[1],
                    marker=markers[i], color=colors[i])
    plt.scatter(centroids.T[0], centroids.T[1], marker='*')


def kmeans(x, K, n=10):
    """Divide data x into K clusters using K-means unsupervised learning"""
    m, ndim = x.shape
    c = np.zeros(m)

    # Perform k-means algorithm n times, using different starting centroids
    # each time
    for j in range(n):
        # Initialize centroids
        centroids = choosestartingpoint(x, K)
        newcentroids = centroids * 0.

        # Repeat until centroids no longer moving
        while True:
            for i in range(0, m):
                c[i] = clusterassign(x[i], centroids)

            for i in range(0, K):
                newcentroids[i] = np.mean(x[c == i], 0)

            if np.mean(abs(newcentroids - centroids)) < 0.01:
                break
            else:
                centroids = newcentroids

        distortion = calcdistortion(x, c, centroids)

        try:
            lowestdistortion
        except NameError:
            lowestdistortion = distortion
            bestcentroids = centroids

        if distortion < lowestdistortion:
            lowestdistortion = distortion
            bestcentroids = centroids

    # Assign to final cluster locations
    for i in range(0, m):
        c[i] = clusterassign(x[i], bestcentroids)

    return bestcentroids, c


def demo(m=1000, K=4):
    x = randomdata(K=K, m=m)
    centroids, c = kmeans(x, K)
    plotkmeans(x, c, centroids)
