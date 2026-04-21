import numpy as np

def decision_tree_split(X, y):
    """
    Find the best feature and threshold to split the data.
    """
    # Write code here
    X, y = np.array(X), np.array(y)
    n, d = X.shape

    def gini(a):
        if len(a) == 0: return 0
        _, c = np.unique(a, return_counts=True)
        p = c / len(a)
        return 1 - np.sum(p**2)

    parent = gini(y)
    best_gain, best_f, best_t = -1, -1, float('inf')

    for f in range(d):
        vals = np.unique(X[:, f])
        for t in (vals[:-1] + vals[1:]) / 2:
            left = y[X[:, f] <= t]
            right = y[X[:, f] > t]

            g = parent - (len(left)/n)*gini(left) - (len(right)/n)*gini(right)

            if (g > best_gain or
               (np.isclose(g, best_gain) and (f < best_f or (f == best_f and t < best_t)))):
                best_gain, best_f, best_t = g, f, t

    return best_f, best_t