import numpy as np

def knn_distance(X_train, X_test, k):
    """
    Compute pairwise distances and return k nearest neighbor indices.
    """
    # Write code here
    X_train = np.atleast_2d(X_train)
    X_test = np.atleast_2d(X_test)

    # Handle 1D inputs
    if X_train.shape[1] != X_test.shape[1]:
        X_train = X_train.reshape(-1, 1)
        X_test = X_test.reshape(-1, 1)

    # Compute pairwise distances
    diff = X_test[:, np.newaxis, :] - X_train[np.newaxis, :, :]
    distances = np.sqrt(np.sum(diff ** 2, axis=2))

    n_train = X_train.shape[0]

    # Sort indices
    sorted_indices = np.argsort(distances, axis=1)

    # Prepare output filled with -1
    result = -1 * np.ones((X_test.shape[0], k), dtype=int)

    # Fill available neighbors
    valid_k = min(k, n_train)
    result[:, :valid_k] = sorted_indices[:, :valid_k]

    return result