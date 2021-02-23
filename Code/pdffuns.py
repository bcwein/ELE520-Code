"""
Python script with acollection of pdfs.

Functions:
    normal distribution 1D
    bivariate normal distribution (2D)
"""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import sys
import pdb
from functools import reduce


def norm1D(my, Sgm, x):
    """Calculate normal distribution in 1D."""
    d = np.shape(x)
    p = np.zeros(np.shape(x))
    for i in np.arange(0, d[0]):
        p[i] = 1 / (np.sqrt(2 * np.pi) * Sgm) * \
            np.exp(-1 / 2 * np.square((x[i] - my)) / (np.square(Sgm)))

    return p


def norm2D(my, Sgm, X, Y):
    """
    Calculate bivariate normal distribution pdf values for a grid.

    Based on answer by https://stackoverflow.com/users/692734/user692734 on
    question: https://stackoverflow.com/questions/11615664

    PDF is based on the non-degenerate case in:
    https://en.wikipedia.org/wiki/Multivariate_normal_distribution

    Args:
        my (Numpy array):   Mean vector
        Sgm (Numpy array):  Covariance matrix
        X (Numpy array):    X coordinate
        Y (Numpy array):    Y xoordinate

    Returns:
        p (Numpy array):    2D pdf values across grid.
        comp (Numpy array): Points of computation.
    """
    [i, d] = np.shape(X)
    p = np.zeros(np.shape(X))
    comp = np.ones(np.shape(X))*0
    det = np.linalg.det(Sgm)

    for i in np.arange(0, i):
        for j in np.arange(0, d):
            p[i, j] = 1 / (np.sqrt((np.pi*2)**2 * det)) * \
                np.exp(-0.5 * np.matmul(
                    ([X[i, j], Y[i, j]] - my).T,
                    np.matmul(
                        np.linalg.inv(Sgm),
                        ([X[i, j], Y[i, j]] - my)
                    )
                )
            )

    return comp, p


def parzen(data, h1, X, Y):
    """Use Parzen window density estimation to calculate decision boundary.

    Args:
        data (List of arrays): List of arrays for datasets.
        h1 (float): Window size
        X (X grid): values from meshgrid
        Y (Y grid): values from meshgrid

    Returns:
        classifier: Parzen window density estimation classification
        Z: Unscaled density.
    """
    N = len(data[0])
    d = len(data)
    hn = h1 / np.sqrt(N)
    hnI = np.eye(d) * hn**d
    Z = 0

    for i in range(N):
        mu = data[:, i]
        _, Ztmp = norm2D(mu, hnI, X, Y)
        Z += Ztmp

    classifier = 1 / N * Z
    return classifier, Z


def knn2D(data, kn, X, Y):
    """KNN for density estimation.

    Args:
        data: list of 2D matrixes of dataset.
        kn: No of neighbours to consider.
        X: X-values for meshgrid
        Y: Y-values for meshgrid

    Returns:
        p: calculated density.
    """
    [w, h] = np.shape(X)
    p = np.zeros(np.shape(X))
    N = len(data[0])
    coord = np.array([X, Y])
    distances = []

    for i in range(N):
        mu = data[:, i]
        distance = np.sqrt((mu[0] - coord[0])**2 + (mu[1] - coord[1])**2)
        distances.append(distance)

    distances = np.array(distances)

    for i in np.arange(0, w):
        for j in np.arange(0, h):
            distpoints = [distances[x][i][j] for x in range(len(distances))]
            idx = np.argsort(distpoints)
            kdistpoint = distpoints[idx[kn-1]]
            p[i, j] = (1/N) / (np.pi * kdistpoint**2)

    return p


def normND(mu, sigma, x):
    """Calculate pdf value for single point in multivariate gaussian."""
    dim = len(mu)
    det = np.linalg.det(sigma)
    p = 1 / (np.sqrt((np.pi*2)**dim * det)) * \
        np.exp(-0.5 * np.matmul(
            (x - mu).T,
            np.matmul(
                np.linalg.inv(sigma),
                (x - mu)
            )
        )
    )
    return p


def classifyML(training, test):
    """Classify data using bivariate normal distribution pdf values for a grid.

    This classifier is based on the Norm2D function.

    This functions expects a dataframe with this strict format:
        x1,x2,...,xn feature columns and a class column.
        Each row is one sampled datapoint.

    Args:
        training (dataframe): Training dataframe following above format.
        test (dataframe): Test dataframe following above format.

    Returns:
        predictions: predicted class values
        function values: density values from training set.
    """
    classes = set(training['class'])
    classpred = []
    mus = {}
    sigmas = {}

    # Estimate my and sigma
    for cl in classes:
        mus[cl] = np.array(training[
            training['class'] == cl
            ].loc[:, training.columns != 'class'].mean())

        sigmas[cl] = np.array(
                training[
                    training['class'] == cl
                ].loc[:, training.columns != 'class'].cov())

    # Classify training data from
    classpred = test.apply(
        lambda row: np.argmax(
            [normND(
                mus[x], sigmas[x], np.array(row.drop('class'))
                ) for x in classes]), axis=1
        )

    funcval = test.apply(
        lambda row: [normND(
                mus[x], sigmas[x], np.array(row.drop('class'))
                ) for x in classes], axis=1
        )

    return np.array(classpred.tolist()), np.array(funcval.tolist())


def densityPoint(density, mgrids, vec):
    """Find density value that are closest to a vector in meshgrid.

    Args:
        density : [description]
        X ([type]): [description]
        Y ([type]): [description]
        vec ([type]): [description]

    Returns:
        [type]: [description]
    """
    indexes = []

    for i, mgrid in enumerate(mgrids):
        indexes.append(
            list(np.unravel_index(
                np.abs(mgrid - vec[i]).argmin(), np.shape(mgrid)
            ))
        )

    for li in indexes:
        li.remove(0)
    indexes.reverse()
    flat_list = [item for sublist in indexes for item in sublist]

    return density[tuple(flat_list)]


def parzenND(data, h1, mgrids):
    """Use Parzen window density estimation to calculate decision boundary.

    Args:
        data (List of arrays): List of arrays for datasets.
        h1 (float): Window size
        X (X grid): values from meshgrid
        Y (Y grid): values from meshgrid

    Returns:
        classifier: Parzen window density estimation classification
        Z: Unscaled density.
    """
    N = len(data[0])
    d = len(data)
    hn = h1 / np.sqrt(N)
    hnI = np.eye(d) * hn**d
    Z = np.zeros(np.shape(mgrids[0]))
    mgrids = np.array(mgrids)

    dims = np.shape(mgrids)
    vectorized = mgrids.reshape((dims[0], np.prod(dims[1:])))
    for i in range(N):
        Z += np.array(
            list(
                map(lambda x: normND(data[:, i], hnI, x),
                    [vectorized[:, i] for i in range(np.prod(dims[1:]))])
            )
        ).reshape(dims[1:])

    return Z


def classifyParzen(train, test, window=0.5, delta=0.5):
    """Classify test dataset by estimating training sets density using Parzen.

    This classifier is based on the Parzen function.

    This functions expects a dataframe with this strict format:
        x1,x2,...,xn feature columns and a class column.
        Each row is one sampled datapoint.

    Args:
        train (DataFrame): Training dataset
        test (DataFrame): Test dataset
        window (float, optional): [h1 parameter in parzen]. Defaults to 0.5.
        delta (float, optional): [step in np.arange]. Defaults to 0.1.

    Returns:
        Classpred: Pandas Series of predicted class labels.
    """
    densities = {}
    classes = set(train['class'])
    ranges = []

    for feature in train.drop('class', axis=1):
        ranges.append(
            np.arange(
                train[feature].min(), train[feature].max(), delta
            ).reshape(-1, 1)
        )

    mgrid = np.meshgrid(*ranges)

    for cl in classes:
        data = np.array(train[
            train['class'] == cl
        ].loc[:, train.columns != 'class']).T
        densities[cl] = parzenND(data, window, mgrid)

    classpred = test.apply(
        lambda row: np.argmax(
            [densityPoint(
                densities[x], mgrid, vec=np.array(row.drop('class'))
            ) for x in classes]), axis=1
        )

    return classpred
