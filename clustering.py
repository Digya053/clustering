import sys
import numpy as np


def euclidean_distance(data, centroid):
    return np.linalg.norm(centroid - data, 2, 1)


def KMeans(X, K, iteration, len_X):
    # initialization of centroid
    mu = X[np.random.choice(len_X, K, replace=False), :]
    for i in range(iteration):
        clusters = []
        for row in range(len_X):
            nearest_cluster = np.argmin(euclidean_distance(X[row], mu))
            clusters.append(nearest_cluster)

        for k in range(K):
            indices = [
                index for index,
                cluster_no in enumerate(clusters) if cluster_no == k]
            mu[k] = np.mean(X[indices], axis=0)

        filename = "centroids-" + str(i + 1) + ".csv"
        np.savetxt(filename, mu, delimiter=",")


def E_step(X, mu, sigma, phi, pi, K):
    for i in range(X.shape[0]):
        for k in range(K):
            dif = X[i] - mu[k]
            phi[i, k] = pi[k] * multivariate_normal_distribution(k, i, sigma, phi, pi, X.shape[1], dif)
        norm_constant = np.sum(phi[i])
        if norm_constant == 0:
            phi[i] = pi / K
        else:
            phi[i] = phi[i] / norm_constant

    return phi


def M_step(X, mu, sigma, phi, pi, iterr, K, len_X, len_Y):
    for k in range(K):
        nk = np.sum(phi, axis=0)
        pi = nk / float(len_X)

        if nk[k] == 0:
            mu[k] = X[np.random.choice(len_X, 1, replace=False), :]
            sigma[:, :, k] = np.eye(len_Y)
        else:
            mu[k] = np.dot(phi[:, k].T, X) / float(nk[k])
            sigma[:, :, k] = np.zeros((len_Y, len_Y))

            for i in range(len_X):
                dif = X[i] - mu[k]
                sigma[:, :, k] += phi[i, k] * np.outer(dif, dif)

            sigma[:, :, k] /= nk[k]

    filename = "pi-" + str(iterr + 1) + ".csv"
    np.savetxt(filename, pi, delimiter=",")
    filename = "mu-" + str(iterr + 1) + ".csv"
    np.savetxt(filename, mu, delimiter=",")

    for cl in range(K):
        filename = "Sigma-" + str(cl + 1) + "-" + str(iterr + 1) + ".csv"
        np.savetxt(filename, sigma[:, :, cl], delimiter=",")


def multivariate_normal_distribution(k, i, sigma, phi, pi, len_Y, dif):
    inv_sigma = np.linalg.inv(sigma[:, :, k])
    det_sigma = -0.5 * (np.linalg.det(sigma[:, :, k]))
    exponential_term = np.exp(-0.5 * np.dot(dif, np.dot(inv_sigma, dif.T)))
    return det_sigma * ((2 * np.pi) ** (-len_Y / 2)) * exponential_term


def EMGMM(X, K, iteration, len_X, len_Y):
    # initialization of centroid, covariance
    mu = X[np.random.choice(len_X, K, replace=False), :]
    sigma = np.dstack([np.eye(len_Y)] * K)
    phi = np.zeros((len_X, K))
    pi = np.ones(K) / K

    for i in range(iteration):
        phi = E_step(X, mu, sigma, phi, pi, K)
        M_step(X, mu, sigma, phi, pi, i, K, len_X, len_Y)


def main():
    X = np.genfromtxt(sys.argv[1], delimiter=",")
    len_X = X.shape[0]
    len_Y = X.shape[1]
    K = 5
    iteration = 10
    KMeans(X, K, iteration, len_X)
    EMGMM(X, K, iteration, len_X, len_Y)


if __name__ == "__main__":
    main()
