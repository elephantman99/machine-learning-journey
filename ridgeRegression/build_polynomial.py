# -*- coding: utf-8 -*-
"""implement a polynomial basis function."""

import numpy as np


def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    # this function should return the matrix formed
    # by applying the polynomial basis to the input data
    tx = np.ones((len(x), 1))
    if x .ndim == 1:
        x = x[..., np.newaxis]
    for i in range(1, degree + 1):
        tx = np.concatenate((tx, x ** i), axis=1)
    return tx