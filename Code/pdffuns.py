"""Python script with a collection of functions."""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import sys
import pdb
from functools import reduce
from matplotlib import colors


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
    """Find density value that are closest to a vector in meshgrid."""
    vectors = []
    for i in range(len(mgrids)):
        vectors.append(
            np.array(np.unravel_index(
                    np.abs(
                        np.array(mgrids[i]) - vec[i]
                    ).argmin(),
                    np.shape(mgrids[i])
                ))
            )

    return density[tuple(np.sum(vectors, axis=0))]


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


def knnCLassify(train, point, k=1):
    """Apply helper function. Used in knnCLassifier.

    Args:
        train (training dataset)
        point (point to classify)
        k (number of neighbors to compare with)

    Returns:
        integer: predicted class of point.
    """
    nearest = np.linalg.norm(
        train.drop('class', axis=1) - point.drop('class'),
        axis=1
        ).argsort()[:k]
    return (train.iloc[nearest]['class'].value_counts().index.tolist()[0])


def knnClassifier(train, test, k=1):
    """Classify dataset using knn method.

    This functions expects a dataframe with this strict format:
        x1,x2,...,xn feature columns and a class column.
        Each row is one sampled datapoint.

    Args:
        train (pandas): Training set.
        test (pandas): Test set.
        k (int, optional): [neigbors to visit]. Defaults to 1.

    Returns:
        [series]: Predicted classes.
    """
    classpred = test.apply(lambda row: knnCLassify(train, row, k=k), axis=1)
    return classpred


def plot_data(lda, X, y, y_pred):
    """Plot QuadraticDiscriminantAnalysis decision region.

    Stolen from sklearn:
    https://scikit-learn.org/stable/auto_examples/classification/plot_lda_qda.html.

    Args:
        lda (classifier):
        X (Features):
        y (Labels):
        y_pred (Predictions):
    Returns:
        splot: plot
    """
    splot = plt.subplot(1, 1, 1)

    tp = (y == y_pred)  # True Positive
    tp0, tp1 = tp[y == 0], tp[y == 1]
    X0, X1 = X[y == 0], X[y == 1]
    X0_tp, X0_fp = X0[tp0], X0[~tp0]
    X1_tp, X1_fp = X1[tp1], X1[~tp1]

    # class 0: dots
    plt.scatter(X0_tp[:, 0], X0_tp[:, 1], marker='.', color='red')
    plt.scatter(X0_fp[:, 0], X0_fp[:, 1], marker='x',
                s=20, color='#990000')  # dark red

    # class 1: dots
    plt.scatter(X1_tp[:, 0], X1_tp[:, 1], marker='.', color='blue')
    plt.scatter(X1_fp[:, 0], X1_fp[:, 1], marker='x',
                s=20, color='#000099')  # dark blue

    # class 0 and 1 : areas
    nx, ny = 200, 100
    x_min, x_max = plt.xlim()
    y_min, y_max = plt.ylim()
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, nx),
                         np.linspace(y_min, y_max, ny))
    Z = lda.predict_proba(np.c_[xx.ravel(), yy.ravel()])
    Z = Z[:, 1].reshape(xx.shape)
    plt.pcolormesh(xx, yy, Z, cmap='RdBu',
                   norm=colors.Normalize(0., 1.), zorder=0)
    plt.contour(xx, yy, Z, [0.5], linewidths=2., colors='white')

    # means
    try:
        plt.plot(lda.means_[0][0], lda.means_[0][1],
                 '*', color='yellow', markersize=15, markeredgecolor='grey')
        plt.plot(lda.means_[1][0], lda.means_[1][1],
                 '*', color='yellow', markersize=15, markeredgecolor='grey')
    except:
        return splot

    return splot
