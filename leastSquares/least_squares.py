# -*- coding: utf-8 -*-
"""implement a polynomial basis function."""
from linearRegression_gradientDescent import costs as cts
import numpy as np

def least_squares(y, tx):
    """calculate the least squares solution."""
    N, D = tx.shape    
    # svd decomposition of data matrix
    u, s, vh = np.linalg.svd(tx, full_matrices=True)
    # reshape s to its true shape (NxD)
    new_s = np.zeros((N, D))
    np.fill_diagonal(new_s, s)
    s = new_s
    # get pseudo-inverse
    s_ = np.linalg.pinv(s)
    # apply analytical solution
    w  = vh.T @ s_ @ u.T @ y
    # compute loss
    loss = cts.compute_mse(y, tx, w)
    
    return loss, w