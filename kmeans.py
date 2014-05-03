import numpy as np


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
    """
    Randomly choose K points from the array x of n-dimensional points
    """
    npoints = x.shape[0]
    i = np.random.choice(npoints, K, replace=False)
    return x[i, :]
