import numpy as np


def randomdata(K=2, m=100, ndim=2, sigma=0.5):
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
        # Generage K n-dimensional centroids
        centroids[i, :] = np.random.choice(range(K), ndim, replace=False)

    # For each data point
    for i in range(0, m):
        x[i, :] = centroids[c[i], :]
        x[i, :] += np.random.normal(0, sigma, ndim)

    return x


def clusterassign(x, u):
    """
    For an array of centroids u in an n-dimensional space, return the index
    of the centroid in that array that is closest to the point x
    """
    return np.argmin(np.sum((x - u)**2, 1))


def clustercentroid(x):
    """
    For a cluster of points u in an n-dimensional space, return the centroid
    location of the points
    """
    return np.mean(x, 1)


def choosestartingpoint(x, K):
    """Randomly choose K points from the array x of n-dimensional points
    """
    npoints = x.shape[0]
    i = np.random.choice(npoints, K, replace=False)
    return x[i, :]
