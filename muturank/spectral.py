"""
This is an extended version of scikit-learn's spectral clustering.
It adds fuzzy C-means to the assign_labels clustering algorithms.
"""

from sklearn.manifold import spectral_embedding
import sklearn.cluster.spectral as sp
import skfuzzy as fuzz
import numpy as np
import math


def spectral_clustering(affinity, n_clusters=8, n_components=None,
                        eigen_solver=None, random_state=None, n_init=10,
                        eigen_tol=0.0, assign_labels='kmeans',
                        fuzzy_m=2, fuzzy_error=0.0005, fuzzy_maxiter=10000,
                        fuzzy_label_threshold=None):
    if assign_labels not in ('kmeans', 'fuzzy_cmeans', 'discretize'):
        raise ValueError("The 'assign_labels' parameter should be "
                         "'kmeans', 'fuzzy_cmeans' or 'discretize', but '%s' was given"
                         % assign_labels)

    random_state_ = sp.check_random_state(random_state)
    n_components = n_clusters if n_components is None else n_components
    maps = spectral_embedding(affinity, n_components=n_components,
                              eigen_solver=eigen_solver,
                              random_state=random_state,
                              eigen_tol=eigen_tol, drop_first=False)

    if assign_labels == 'kmeans':
        _, labels, _ = sp.k_means(maps, n_clusters, random_state=random_state_,
                                  n_init=n_init)
    elif assign_labels == 'fuzzy_cmeans':
        if fuzzy_label_threshold is None:
            fuzzy_label_threshold = 1. / n_clusters

        _, u, _, _, _, _, _ = fuzz.cluster.cmeans(np.exp(maps.T), n_clusters, seed=random_state, m=fuzzy_m,
                                                  error=fuzzy_error, maxiter=fuzzy_maxiter)
        # from sklearn.mixture import GMM
        # gmm = GMM(n_components=n_clusters, covariance_type='full', random_state=random_state, n_init=n_init).fit(maps)
        # u = gmm.predict_proba(maps)
        # u = u.T
        assignments = np.argwhere(u.T >= fuzzy_label_threshold)
        labels = [[] for _ in range(u.shape[1])]
        for row in assignments:
            labels[row[0]].append(row[1])
    else:
        labels = sp.discretize(maps, random_state=random_state_)

    return labels


if __name__ == '__main__':
    import numpy as np
    import networkx as nx

    np.random.seed(1)
    adj_mat = np.random.rand(15, 15)

    # Cluster
    labels = spectral_clustering(adj_mat, 5, n_init=100)
    print(labels)

    flabels = spectral_clustering(adj_mat, 5, assign_labels='fuzzy_cmeans', random_state=42)
    print(flabels)
