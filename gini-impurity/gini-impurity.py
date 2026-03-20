import numpy as np

def gini_impurity(y_left, y_right):
    """
    Compute weighted Gini impurity for a binary split.
    """
    # Write code here
    def gini(y):
        if len(y)==0:
            return 0.0
        p=np.unique(y,return_counts=True)[1]/len(y)
        return 1-np.sum(p**2)
    N=len(y_left)+len(y_right)
    if N==0:
        return 0.0
    return float(((len(y_left)/N)*gini(y_left))+((len(y_right)/N)*gini(y_right)))
    pass